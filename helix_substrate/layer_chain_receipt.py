"""
layer_chain_receipt.py — Load-time layer-chained integrity receipts for HXQ artifacts.

Uses SHAKE256 (Keccak sponge, XOF) for layer-resolved chain verification.

Architecture:
  At compression time → generate_receipt_manifest()
  At load time       → verify_receipt_manifest()  ← gates hxq_dequant()

Chain structure per layer:
  absorb(prev_state || layer_name || base_tensor_bytes || sidecar_bytes)
  squeeze(32) → layer_receipt

Final receipt = squeeze(all layer states, 64 bytes)

If ANY layer is tampered, that layer's receipt breaks and all subsequent
receipts are invalidated. Verifier reports exact break layer.

Scope / relationship to helix_substrate.receipt:
  This module covers model-load-time layer chain integrity.
  helix_substrate.receipt covers per-operation substrate execution proofs
  (OperationResult / ExecutionReceipt — SHA256 input→output hash chains).
  They are intentionally kept separate — do not merge.

Backwards compatibility warning:
  This is a NEW module. In the downloaded draft (hxq_receipt.py) the
  single-safetensors-file code path hashed
      name.encode() + WHOLE_DATA_REGION
  for every layer, producing O(layers × file_size) work — impractical for
  real models. This module fixes that bug by parsing the safetensors header
  ONCE and reading only the per-tensor slice via data_offsets. That changes
  the per-layer digest format for the single-file path. Any receipts
  generated against the draft with a single safetensors file will NOT
  verify against this module. Draft never shipped, so this is acceptable.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, asdict


SHAKE_CAPACITY = 256        # SHAKE256
LAYER_DIGEST_BYTES = 32     # per-layer receipt size
FINAL_DIGEST_BYTES = 64     # root receipt size
RECEIPT_VERSION = 1
RECEIPT_MAGIC = b"HXQR"    # 4-byte magic for receipt files


@dataclass
class LayerReceipt:
    layer_name: str
    layer_index: int
    base_digest: str       # hex: SHAKE of base tensor alone
    sidecar_digest: str    # hex: SHAKE of sidecar alone (or "none")
    chain_digest: str      # hex: SHAKE of prev_chain || base || sidecar
    base_bytes: int
    sidecar_bytes: int


@dataclass
class ReceiptManifest:
    version: int
    model_name: str
    root_receipt: str           # hex: final squeezed chain state
    layer_count: int
    layers: list                # list of LayerReceipt dicts
    build_metadata: dict        # timestamp, hxq version, etc.


# ---------------------------------------------------------------------------
# Core sponge helpers
# ---------------------------------------------------------------------------

def _shake(data: bytes, length: int = LAYER_DIGEST_BYTES) -> bytes:
    """Single SHAKE256 absorb+squeeze."""
    h = hashlib.shake_256()
    h.update(data)
    return h.digest(length)


def _shake_chain(prev_state: bytes, *absorb_items: bytes) -> bytes:
    """
    Chain absorb: prev_state || item0 || item1 || ...
    Returns new chain state (LAYER_DIGEST_BYTES).
    Domain-separates each item with its length prefix (4 bytes, big-endian).
    """
    h = hashlib.shake_256()
    h.update(prev_state)
    for item in absorb_items:
        # length prefix prevents extension attacks across item boundaries
        h.update(struct.pack(">I", len(item)))
        h.update(item)
    return h.digest(LAYER_DIGEST_BYTES)


# ---------------------------------------------------------------------------
# Tensor byte extraction
# ---------------------------------------------------------------------------

def _read_tensor_bytes(path: Path) -> bytes:
    """
    Read raw bytes from a tensor file (legacy whole-file helper).

    Supports: .safetensors, .bin, .pt, raw float dumps.
    For safetensors: reads the full data region (skips header JSON).

    NOTE: For single-safetensors-file inputs, per-layer generation uses
    _parse_safetensors_header() + _read_safetensor_slice() instead, so
    this helper is only called for per-layer sidecar files and directories.
    """
    if not path.exists():
        return b""

    with open(path, "rb") as f:
        raw = f.read()

    # safetensors: first 8 bytes = header length (little-endian u64)
    if path.suffix == ".safetensors" and len(raw) > 8:
        header_len = struct.unpack_from("<Q", raw, 0)[0]
        data_start = 8 + header_len
        if data_start < len(raw):
            return raw[data_start:]

    return raw


def _parse_safetensors_header(path: Path) -> Tuple[dict, int]:
    """
    Parse a safetensors file's JSON header.

    Returns (header_dict, data_region_offset) where header_dict maps
    tensor_name -> {'dtype', 'shape', 'data_offsets': [start, end]}
    and data_region_offset is the absolute file offset where tensor bytes begin.
    """
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"[HXQ] Not a safetensors file (too short): {path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json = f.read(header_len)
    header = json.loads(header_json.decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def _read_safetensor_slice(path: Path, data_region_offset: int, entry: dict) -> bytes:
    """
    Read one tensor's raw bytes using its safetensors header entry.
    Slice-only read — does not touch the rest of the file.
    """
    start, end = entry["data_offsets"]
    length = end - start
    with open(path, "rb") as f:
        f.seek(data_region_offset + start)
        return f.read(length)


def _tensor_bytes_from_dict(tensor_dict: dict, layer_name: str) -> bytes:
    """
    If tensors are already loaded in memory (numpy/torch),
    extract bytes for a specific layer key.
    Pass tensor_dict = {layer_name: np.ndarray or torch.Tensor}
    """
    if layer_name not in tensor_dict:
        return b""
    t = tensor_dict[layer_name]
    # numpy
    if hasattr(t, "tobytes"):
        return t.tobytes()
    # torch
    if hasattr(t, "numpy"):
        return t.detach().cpu().numpy().tobytes()
    return b""


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def _resolve_safetensors_entry(header: dict, name: str) -> Optional[dict]:
    """Header lookup with `.weight` fallback for naming drift resilience."""
    if name in header:
        return header[name]
    fallback = name + ".weight"
    if fallback in header:
        return header[fallback]
    return None


def generate_receipt_manifest(
    layer_names: list,
    base_tensor_source,          # Path to .safetensors OR Path to dir OR dict
    sidecar_source,              # Path to .safetensors OR Path to dir OR dict OR None
    model_name: str = "unnamed",
    build_metadata: Optional[dict] = None,
) -> ReceiptManifest:
    """
    Generate a receipt manifest by chaining SHAKE256 through all layers.

    layer_names: ordered list of layer keys (order matters — defines chain).
    base_tensor_source:
        - Path to a single `.safetensors` file: header parsed ONCE, per-layer
          slices read via data_offsets (O(total_tensor_bytes) total work).
        - Path to a directory: per-layer files matched by glob `{name}.*`.
        - dict {name: tensor}: in-memory numpy/torch tensors.
    sidecar_source:
        - Same three forms, or None (treated as empty sidecars).
    """
    if build_metadata is None:
        import datetime
        build_metadata = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "hxq_receipt_version": RECEIPT_VERSION,
        }

    # ------------------------------------------------------------------
    # Hoist header parsing out of the loop for single-safetensors inputs.
    # This is THE fix for O(layers × file_size) scaling.
    # ------------------------------------------------------------------
    base_header = None
    base_data_offset = 0
    base_path_is_file = False

    if isinstance(base_tensor_source, (str, Path)):
        base_path = Path(base_tensor_source)
        if base_path.is_file() and base_path.suffix == ".safetensors":
            base_header, base_data_offset = _parse_safetensors_header(base_path)
            base_path_is_file = True

    sidecar_header = None
    sidecar_data_offset = 0
    sidecar_path_is_file = False

    if isinstance(sidecar_source, (str, Path)):
        sidecar_path = Path(sidecar_source)
        if sidecar_path.is_file() and sidecar_path.suffix == ".safetensors":
            sidecar_header, sidecar_data_offset = _parse_safetensors_header(sidecar_path)
            sidecar_path_is_file = True

    chain_state = b"\x00" * LAYER_DIGEST_BYTES   # genesis state
    layer_receipts = []

    for idx, name in enumerate(layer_names):
        # --- get base tensor bytes ---
        if isinstance(base_tensor_source, (str, Path)):
            p = Path(base_tensor_source)
            if p.is_dir():
                candidates = list(p.glob(f"{name}.*"))
                base_bytes = _read_tensor_bytes(candidates[0]) if candidates else b""
            elif base_path_is_file:
                entry = _resolve_safetensors_entry(base_header, name)
                if entry is not None:
                    base_bytes = _read_safetensor_slice(p, base_data_offset, entry)
                else:
                    base_bytes = b""
            else:
                base_bytes = b""
        else:
            base_bytes = _tensor_bytes_from_dict(base_tensor_source, name)

        # --- get sidecar bytes ---
        if sidecar_source is None:
            sidecar_bytes = b""
        elif isinstance(sidecar_source, (str, Path)):
            p = Path(sidecar_source)
            if p.is_dir():
                candidates = list(p.glob(f"{name}_sidecar.*"))
                sidecar_bytes = _read_tensor_bytes(candidates[0]) if candidates else b""
            elif sidecar_path_is_file:
                # Look for "<name>_sidecar" then "<name>" then name + ".weight"
                entry = _resolve_safetensors_entry(sidecar_header, f"{name}_sidecar")
                if entry is None:
                    entry = _resolve_safetensors_entry(sidecar_header, name)
                if entry is not None:
                    sidecar_bytes = _read_safetensor_slice(
                        p, sidecar_data_offset, entry
                    )
                else:
                    sidecar_bytes = b""
            else:
                sidecar_bytes = b""
        else:
            sidecar_bytes = _tensor_bytes_from_dict(sidecar_source, f"{name}_sidecar")

        # --- individual tensor digests (for diagnostic reporting) ---
        base_digest = _shake(name.encode() + base_bytes).hex()
        sidecar_digest = (
            _shake(name.encode() + sidecar_bytes).hex() if sidecar_bytes else "none"
        )

        # --- chain absorb --- (unchanged from draft — manifest format stable)
        chain_state = _shake_chain(
            chain_state,
            name.encode(),
            base_bytes,
            sidecar_bytes,
        )

        layer_receipts.append(LayerReceipt(
            layer_name=name,
            layer_index=idx,
            base_digest=base_digest,
            sidecar_digest=sidecar_digest,
            chain_digest=chain_state.hex(),
            base_bytes=len(base_bytes),
            sidecar_bytes=len(sidecar_bytes),
        ))

    # final root: squeeze the accumulated chain state to FINAL_DIGEST_BYTES
    root_receipt = _shake(chain_state + b"ROOT", FINAL_DIGEST_BYTES).hex()

    return ReceiptManifest(
        version=RECEIPT_VERSION,
        model_name=model_name,
        root_receipt=root_receipt,
        layer_count=len(layer_names),
        layers=[asdict(r) for r in layer_receipts],
        build_metadata=build_metadata,
    )


# ---------------------------------------------------------------------------
# Manifest serialization
# ---------------------------------------------------------------------------

def save_manifest(manifest: ReceiptManifest, path: Path) -> None:
    """Write receipt manifest as JSON. Binary header for future binary format."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)


def load_manifest(path: Path) -> ReceiptManifest:
    path = Path(path)
    with open(path) as f:
        d = json.load(f)
    m = ReceiptManifest(
        **{k: v for k, v in d.items() if k != "layers"},
        layers=d["layers"],
    )
    return m


# ---------------------------------------------------------------------------
# Verification — the load-time gate
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    passed: bool
    root_match: bool
    first_break_layer: Optional[str]   # None if chain intact
    first_break_index: Optional[int]
    layers_verified: int
    layers_failed: int
    detail: list                        # per-layer pass/fail


def verify_receipt_manifest(
    manifest: ReceiptManifest,
    base_tensor_source,
    sidecar_source,
) -> VerificationResult:
    """
    Re-derive the receipt chain from current disk state.
    Returns VerificationResult. Caller should refuse to call hxq_dequant()
    if result.passed is False.
    """
    layer_names = [lr["layer_name"] for lr in manifest.layers]

    # re-generate manifest from current state
    live = generate_receipt_manifest(
        layer_names=layer_names,
        base_tensor_source=base_tensor_source,
        sidecar_source=sidecar_source,
        model_name=manifest.model_name,
        build_metadata=manifest.build_metadata,   # same metadata for reproducibility
    )

    detail = []
    first_break_layer = None
    first_break_index = None
    layers_failed = 0

    for stored, live_layer in zip(manifest.layers, live.layers):
        chain_ok = stored["chain_digest"] == live_layer["chain_digest"]
        base_ok = stored["base_digest"] == live_layer["base_digest"]
        scar_ok = stored["sidecar_digest"] == live_layer["sidecar_digest"]
        layer_ok = chain_ok and base_ok and scar_ok

        if not layer_ok and first_break_layer is None:
            first_break_layer = stored["layer_name"]
            first_break_index = stored["layer_index"]

        if not layer_ok:
            layers_failed += 1

        detail.append({
            "layer": stored["layer_name"],
            "index": stored["layer_index"],
            "passed": layer_ok,
            "base_ok": base_ok,
            "sidecar_ok": scar_ok,
            "chain_ok": chain_ok,
        })

    root_match = manifest.root_receipt == live.root_receipt

    return VerificationResult(
        passed=(root_match and layers_failed == 0),
        root_match=root_match,
        first_break_layer=first_break_layer,
        first_break_index=first_break_index,
        layers_verified=len(layer_names),
        layers_failed=layers_failed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Load gate — drop this in front of hxq_dequant()
# ---------------------------------------------------------------------------

def assert_receipt_valid(
    manifest_path: Path,
    base_tensor_source,
    sidecar_source,
    strict: bool = True,
) -> VerificationResult:
    """
    Call this before hxq_dequant(). Raises RuntimeError if chain broken.
    If strict=False, logs warning but doesn't raise (dev/debug mode only).
    """
    manifest = load_manifest(manifest_path)
    result = verify_receipt_manifest(manifest, base_tensor_source, sidecar_source)

    if not result.passed:
        msg_parts = [
            "[HXQ] RECEIPT CHAIN BROKEN — model load rejected.",
            f"  Root match:        {result.root_match}",
            f"  Layers verified:   {result.layers_verified}",
            f"  Layers failed:     {result.layers_failed}",
        ]
        if result.first_break_layer:
            msg_parts.append(
                f"  First break at:    layer[{result.first_break_index}] "
                f"'{result.first_break_layer}'"
            )
        msg = "\n".join(msg_parts)

        if strict:
            raise RuntimeError(msg)
        else:
            print(f"[HXQ WARNING] {msg}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Self-test (no disk side effects)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Smoke test: generate, verify, tamper, re-verify. Returns exit code."""
    try:
        import numpy as np
    except ImportError:
        print("FAIL: numpy required for selftest")
        return 1

    # 3 small in-memory tensors
    tensors = {
        "tensor_0": np.arange(16, dtype=np.float32),
        "tensor_1": np.arange(32, dtype=np.float32),
        "tensor_2": np.arange(8, dtype=np.float32),
    }
    layer_names = ["tensor_0", "tensor_1", "tensor_2"]

    # Clean generate + verify
    manifest = generate_receipt_manifest(
        layer_names=layer_names,
        base_tensor_source=tensors,
        sidecar_source=None,
        model_name="selftest",
    )
    result = verify_receipt_manifest(manifest, tensors, None)
    if not result.passed:
        print(f"FAIL: clean verify should pass (result={result})")
        return 1
    print(f"PASS: clean verify ({result.layers_verified} layers)")

    # Tamper: flip a byte in tensor_1
    tampered = {k: v.copy() for k, v in tensors.items()}
    tampered["tensor_1"][3] += 1.0
    result_bad = verify_receipt_manifest(manifest, tampered, None)
    if result_bad.passed:
        print("FAIL: tampered verify should fail")
        return 1
    if result_bad.first_break_layer != "tensor_1":
        print(
            f"FAIL: expected first_break_layer='tensor_1', "
            f"got '{result_bad.first_break_layer}'"
        )
        return 1
    print(
        f"PASS: tamper detected at layer '{result_bad.first_break_layer}' "
        f"(index {result_bad.first_break_index})"
    )

    # Also sanity-check chain propagation: because tensor_1 was tampered,
    # tensor_2's chain_digest should also be invalidated.
    t2_detail = next(d for d in result_bad.detail if d["layer"] == "tensor_2")
    if t2_detail["chain_ok"]:
        print("FAIL: tensor_2 chain should be invalid after tensor_1 tamper")
        return 1
    print("PASS: chain propagation — tensor_2 invalidated by tensor_1 tamper")

    print("PASS: all selftest checks green")
    return 0


# ---------------------------------------------------------------------------
# CLI for build-time receipt generation
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="HXQ Layer Chain Receipt Tool")
    sub = parser.add_subparsers(dest="cmd")

    # generate
    gen = sub.add_parser(
        "generate", help="Generate receipt manifest at compression time"
    )
    gen.add_argument(
        "--layers", required=True, help="Comma-separated ordered layer names"
    )
    gen.add_argument(
        "--base", required=True, help="Path to base safetensors file or dir"
    )
    gen.add_argument(
        "--sidecars", default=None, help="Path to sidecar dir or file (optional)"
    )
    gen.add_argument("--model-name", default="unnamed")
    gen.add_argument(
        "--out", required=True, help="Output path for receipt JSON"
    )

    # verify
    ver = sub.add_parser("verify", help="Verify receipt manifest at load time")
    ver.add_argument("--manifest", required=True, help="Path to receipt JSON")
    ver.add_argument(
        "--base", required=True, help="Path to base safetensors file or dir"
    )
    ver.add_argument("--sidecars", default=None)
    ver.add_argument("--no-strict", action="store_true")

    # selftest
    sub.add_parser("selftest", help="Run in-memory smoke test (no disk IO)")

    args = parser.parse_args()

    if args.cmd == "generate":
        layer_names = [l.strip() for l in args.layers.split(",")]
        manifest = generate_receipt_manifest(
            layer_names=layer_names,
            base_tensor_source=Path(args.base),
            sidecar_source=Path(args.sidecars) if args.sidecars else None,
            model_name=args.model_name,
        )
        save_manifest(manifest, Path(args.out))
        print(f"[HXQ] Receipt manifest written → {args.out}")
        print(f"      Root receipt: {manifest.root_receipt[:32]}...")
        print(f"      Layers:       {manifest.layer_count}")

    elif args.cmd == "verify":
        result = assert_receipt_valid(
            manifest_path=Path(args.manifest),
            base_tensor_source=Path(args.base),
            sidecar_source=Path(args.sidecars) if args.sidecars else None,
            strict=not args.no_strict,
        )
        if result.passed:
            print(
                f"[HXQ] ✓ Chain intact — "
                f"{result.layers_verified} layers verified."
            )
        else:
            print(
                f"[HXQ] ✗ Chain broken — first break: "
                f"layer[{result.first_break_index}] "
                f"'{result.first_break_layer}'"
            )
            sys.exit(1)

    elif args.cmd == "selftest":
        sys.exit(_selftest())

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
