#!/usr/bin/env python3
"""
Step 2: Convert CDNA v3 compressed model to HuggingFace-compatible safetensors.

Takes a model dir with cdnav3/ artifacts and produces a safetensors file with
HF-compatible key names for each HelixLinear component:
  - {module}.codebook         [256] float32
  - {module}.indices          [out, in] uint8
  - {module}.sidecar_indices  [N] int64  (flat positions of outliers)
  - {module}.sidecar_values   [N] float32 (exact values at those positions)
  - {module}.bias             [out] float32 (if present)

Exact-stored tensors (embeddings, norms, biases) are saved directly.

Also produces a config.json with quantization_config for HF auto-loading.

Usage:
    python3 tools/convert_to_hf.py \
        --model-dir ~/models/zamba2-1.2b \
        --output-dir ~/models/zamba2-1.2b-helix
"""

import argparse
import hashlib
import json
import platform
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert_cdnav3_to_hf(model_dir: Path, output_dir: Path):
    """Convert CDNA v3 directory to HF safetensors format."""
    cdna_dir = model_dir / "cdnav3"
    if not cdna_dir.exists():
        print(f"ERROR: {cdna_dir} not found", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    compressed_modules = []
    exact_tensors = []

    # Process .cdnav3 directories (compressed tensors)
    for tensor_path in sorted(cdna_dir.glob("*.cdnav3")):
        if not tensor_path.is_dir():
            continue

        meta_path = tensor_path / "meta.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        tensor_name = meta["tensor_name"]  # e.g. "model.layers.0.self_attn.q_proj.weight"

        # Strip .weight suffix to get module path
        if tensor_name.endswith(".weight"):
            module_path = tensor_name[:-len(".weight")]
        else:
            module_path = tensor_name

        rows, cols = meta["shape"]

        # Codebook: [256] float32
        codebook = np.load(tensor_path / "codebook.npy").astype(np.float32)
        tensors[f"{module_path}.codebook"] = torch.from_numpy(codebook)

        # Indices: [rows, cols] uint8
        raw_indices = np.fromfile(tensor_path / "indices.bin", dtype=np.uint8)
        indices = raw_indices.reshape(rows, cols)
        tensors[f"{module_path}.indices"] = torch.from_numpy(indices.copy())

        # Sidecar: optional outlier corrections
        sidecar_path = tensor_path / "sidecar.npz"
        if sidecar_path.exists():
            sidecar_data = np.load(sidecar_path)
            positions = sidecar_data["positions"].astype(np.int64)
            values = sidecar_data["values"].astype(np.float32)
            tensors[f"{module_path}.sidecar_positions"] = torch.from_numpy(positions.copy())
            tensors[f"{module_path}.sidecar_values"] = torch.from_numpy(values.copy())

        # SVD residual factors: DEAD (killed 2026-03-27, zero value at all scales).
        # Deliberately not packed — stripping dead artifacts reduces checkpoint size
        # and eliminates false signal during debugging.

        compressed_modules.append({
            "module": module_path,
            "shape": [rows, cols],
            "n_clusters": meta.get("n_clusters", 256),
            "sidecar": sidecar_path.exists(),
        })

    # Process .npy files (exact-stored tensors: embeddings, norms, biases)
    for meta_path in sorted(cdna_dir.glob("*.npy.meta.json")):
        meta = json.loads(meta_path.read_text())
        tensor_name = meta.get("tensor_name", "")
        if not tensor_name:
            continue

        npy_path = meta_path.parent / meta_path.name.replace(".meta.json", "")
        if not npy_path.exists():
            continue

        data = np.load(npy_path)
        # Preserve native dtype (FP16 for new compressions, FP32 for legacy)
        tensors[tensor_name] = torch.from_numpy(data.copy())
        exact_tensors.append(tensor_name)

    # --- Copy missing "skip" tensors from original model ---
    # compress.py skips 1D tensors (A_log, D, dt_bias, norms, biases) without
    # storing them in cdnav3/. We must copy them from the original safetensors.
    orig_sf_files = sorted(model_dir.glob("model*.safetensors"))
    if not orig_sf_files:
        pt_path = model_dir / "pytorch_model.bin"
        if pt_path.exists():
            orig_state = torch.load(pt_path, map_location="cpu", weights_only=True)
            orig_keys = set(orig_state.keys())
        else:
            orig_state = None
            orig_keys = set()
    else:
        from safetensors import safe_open
        orig_keys = set()
        for sf in orig_sf_files:
            with safe_open(str(sf), framework="pt") as f:
                orig_keys.update(f.keys())
        orig_state = None  # loaded on-demand below

    if orig_keys:
        # Keys covered by compressed modules (codebook/indices replace .weight)
        compressed_weight_keys = {m["module"] + ".weight" for m in compressed_modules}
        output_keys = set(tensors.keys())
        covered = set()
        for k in output_keys:
            covered.add(k)
            for suffix in (".codebook", ".indices", ".sidecar_positions", ".sidecar_values"):
                if k.endswith(suffix):
                    covered.add(k.rsplit(".", 1)[0] + ".weight")
        covered.update(compressed_weight_keys)

        missing = orig_keys - covered
        if missing:
            # Copy missing tensors from original model (skip tensors: 1D params)
            copied = 0
            if orig_sf_files:
                from safetensors import safe_open
                for sf in orig_sf_files:
                    with safe_open(str(sf), framework="pt") as f:
                        for k in list(missing):
                            if k in f.keys():
                                # Preserve native dtype (BF16/FP16) — don't inflate to FP32
                                tensors[k] = f.get_tensor(k)
                                exact_tensors.append(k)
                                missing.discard(k)
                                copied += 1
            elif orig_state is not None:
                for k in list(missing):
                    if k in orig_state:
                        tensors[k] = orig_state[k]
                        exact_tensors.append(k)
                        missing.discard(k)
                        copied += 1

            print(f"  Copied {copied} skip tensors from original model")
            if missing:
                print(f"\n  ERROR: {len(missing)} tensors still missing after copy!")
                for k in sorted(missing)[:20]:
                    print(f"    MISSING: {k}")
                sys.exit(1)
        else:
            print(f"  Validation: all {len(orig_keys)} original tensors accounted for")

    # Save safetensors (after all tensors collected: compressed + exact + skip)
    safetensors_path = output_dir / "model.safetensors"
    save_file(tensors, str(safetensors_path))
    print(f"  Saved {len(tensors)} tensors to {safetensors_path}")
    print(f"    {len(compressed_modules)} compressed modules")
    print(f"    {len(exact_tensors)} exact tensors (including skip)")

    # Copy original config.json and add quantization_config
    orig_config_path = model_dir / "config.json"
    if orig_config_path.exists():
        config = json.loads(orig_config_path.read_text())
    else:
        config = {}

    config["quantization_config"] = {
        "quant_method": "hxq",
        "codebook_size": 256,
        "sidecar_enabled": True,
        "exact_patterns": ["embed_tokens", "embed_positions", "wte", "wpe",
                           "lm_head", "layernorm", "layer_norm", "norm",
                           "backbone.embedding"],
        "compressed_modules": [m["module"] for m in compressed_modules],
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config with quantization_config to {config_path}")

    # Copy tokenizer files
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                     "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = model_dir / tok_file
        if src.exists():
            shutil.copy2(src, output_dir / tok_file)

    # Copy any custom modeling code (trust_remote_code models)
    for py_file in model_dir.glob("*.py"):
        shutil.copy2(py_file, output_dir / py_file.name)

    # Summary
    total_bytes = safetensors_path.stat().st_size
    print(f"\n  Output: {output_dir}")
    print(f"  Size: {total_bytes / 1024**2:.1f} MB")
    print(f"  Ready for: AutoModelForCausalLM.from_pretrained('{output_dir}')")

    # ── Gate 1: Validate the safetensors we just wrote ──
    receipt = validate_safetensors(safetensors_path, compressed_modules, exact_tensors)

    # Save receipt alongside model (travels with it)
    receipt_local = output_dir / "conversion_receipt.json"
    with open(receipt_local, "w") as f:
        json.dump(receipt, f, indent=2)

    if receipt["verdict"] == "PASS":
        print(f"\n  GATE 1 PASS — {receipt['validation']['total_keys']} keys validated, "
              f"SHA256={receipt['validation']['sha256'][:16]}...")
    else:
        print(f"\n  GATE 1 FAIL — conversion validation failed:", file=sys.stderr)
        for check, passed in receipt["validation"]["checks"].items():
            if not passed:
                print(f"    FAIL: {check}", file=sys.stderr)
        for detail in receipt["validation"].get("details", []):
            print(f"    {detail}", file=sys.stderr)

    return compressed_modules, exact_tensors, receipt


def validate_safetensors(safetensors_path: Path, compressed_modules: list,
                         exact_tensors: list) -> dict:
    """Reopen saved safetensors and validate structural integrity.

    Checks:
      1. Readable — safe_open() succeeds
      2. Compressed modules complete — codebook + indices exist, indices in range
      3. Exact tensors valid — no NaN/Inf
      4. Key count matches expected
      5. SHA256 of entire file

    Returns dict with verdict (PASS/FAIL), checks, details, sha256.
    """
    checks = {
        "readable": False,
        "compressed_complete": True,
        "indices_in_range": True,
        "exact_no_nan": True,
        "key_count_match": True,
    }
    details = []
    sha256_hex = ""

    # 1. Readable
    try:
        f = safe_open(str(safetensors_path), framework="numpy")
        all_keys = set(f.keys())
        checks["readable"] = True
    except Exception as e:
        checks["readable"] = False
        details.append(f"safe_open failed: {e}")
        return {
            "verdict": "FAIL",
            "validation": {"checks": checks, "details": details,
                           "total_keys": 0, "compressed_modules": len(compressed_modules),
                           "exact_tensors": len(exact_tensors), "sha256": ""},
        }

    # 2. Compressed modules complete + indices in range
    for mod in compressed_modules:
        module_path = mod["module"]
        k = mod.get("n_clusters", 256)

        cb_key = f"{module_path}.codebook"
        idx_key = f"{module_path}.indices"

        if cb_key not in all_keys:
            checks["compressed_complete"] = False
            details.append(f"MISSING codebook: {cb_key}")
            continue
        if idx_key not in all_keys:
            checks["compressed_complete"] = False
            details.append(f"MISSING indices: {idx_key}")
            continue

        # Validate codebook shape
        cb = f.get_tensor(cb_key)
        if cb.ndim != 1 or cb.shape[0] != k:
            checks["compressed_complete"] = False
            details.append(f"BAD codebook shape {cb_key}: {cb.shape}, expected [{k}]")
        del cb

        # Validate indices shape and range
        idx = f.get_tensor(idx_key)
        expected_shape = tuple(mod["shape"])
        if idx.shape != expected_shape:
            checks["compressed_complete"] = False
            details.append(f"BAD indices shape {idx_key}: {idx.shape}, expected {expected_shape}")
        max_idx = int(idx.max())
        if max_idx >= k:
            checks["indices_in_range"] = False
            details.append(f"INDEX OUT OF RANGE {idx_key}: max={max_idx}, codebook_size={k}")
        del idx

        # Sidecar keys (if module has sidecar)
        if mod.get("sidecar", False):
            pos_key = f"{module_path}.sidecar_positions"
            val_key = f"{module_path}.sidecar_values"
            if pos_key not in all_keys:
                checks["compressed_complete"] = False
                details.append(f"MISSING sidecar_positions: {pos_key}")
            if val_key not in all_keys:
                checks["compressed_complete"] = False
                details.append(f"MISSING sidecar_values: {val_key}")

    # 3. Exact tensors — no NaN/Inf
    for tensor_name in exact_tensors:
        if tensor_name not in all_keys:
            checks["exact_no_nan"] = False
            details.append(f"MISSING exact tensor: {tensor_name}")
            continue
        t = f.get_tensor(tensor_name)
        if not np.isfinite(t).all():
            checks["exact_no_nan"] = False
            n_bad = int((~np.isfinite(t)).sum())
            details.append(f"NaN/Inf in {tensor_name}: {n_bad} bad values")
        del t

    # 4. Key count
    # Expected: for each compressed module: codebook + indices + optional sidecar_positions + sidecar_values
    expected_keys = 0
    for mod in compressed_modules:
        expected_keys += 2  # codebook + indices
        if mod.get("sidecar", False):
            expected_keys += 2  # sidecar_positions + sidecar_values
    expected_keys += len(exact_tensors)

    if len(all_keys) != expected_keys:
        checks["key_count_match"] = False
        details.append(f"Key count mismatch: safetensors has {len(all_keys)}, expected {expected_keys}")

    # Close the safetensors handle (no explicit close needed, but release ref)
    del f

    # 5. SHA256 (chunked 8MB reads)
    h = hashlib.sha256()
    with open(safetensors_path, "rb") as fh:
        while True:
            chunk = fh.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    sha256_hex = h.hexdigest()

    verdict = "PASS" if all(checks.values()) else "FAIL"

    return {
        "verdict": verdict,
        "validation": {
            "checks": checks,
            "details": details,
            "total_keys": len(all_keys),
            "compressed_modules": len(compressed_modules),
            "exact_tensors": len(exact_tensors),
            "sha256": sha256_hex,
        },
    }


def emit_conversion_receipt(model_name: str, validation_result: dict,
                            output_dir: Path, cost: dict) -> Path:
    """Emit WO-RECEIPT-COST-01 compliant conversion receipt to archive dir."""
    receipt = {
        "work_order": f"convert-hf-{model_name}",
        "question": "Does the HF safetensors pass structural validation?",
        "verdict": validation_result["verdict"],
        "validation": validation_result["validation"],
        "cost": cost,
    }

    # Archive copy
    receipt_dir = Path(__file__).resolve().parent.parent / "receipts" / "convert"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"{model_name}_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    return receipt_path


def main():
    parser = argparse.ArgumentParser(description="Convert CDNA v3 to HF safetensors")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Model directory containing cdnav3/ subfolder")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for HF-compatible checkpoint")
    args = parser.parse_args()

    # Timing instrumentation (WO-RECEIPT-COST-01)
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    print(f"Converting {args.model_dir} → {args.output_dir}")
    compressed_modules, exact_tensors, receipt = convert_cdnav3_to_hf(
        args.model_dir, args.output_dir)

    # Build cost block
    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Emit archive receipt
    model_name = args.model_dir.name
    receipt_path = emit_conversion_receipt(model_name, receipt, args.output_dir, cost)
    print(f"  Receipt: {receipt_path}")

    if receipt["verdict"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
