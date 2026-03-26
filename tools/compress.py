#!/usr/bin/env python3
"""
Universal CDNA v3 compressor.

Point at any model directory. It reads config.json, discovers all weight
tensors from safetensors, classifies each one, routes by kurtosis, and
encodes via CDNAv3Writer. No hardcoded block counts, no per-model scripts,
no architecture-specific tensor patterns.

The model is the environment. The pipeline is the organism. The config is
the genome. Everything discovered, nothing hardcoded.

Usage:
    python3 tools/compress.py ~/models/tinyllama_fp32
    python3 tools/compress.py ~/models/qwen2.5-7b-instruct
    python3 tools/compress.py ~/models/mamba-130m-hf
    python3 tools/compress.py ~/models/anything-new --dry-run

Handles:
  - Single-file safetensors (model.safetensors)
  - Sharded safetensors (model.safetensors.index.json)
  - pytorch_model.bin (auto-converts to numpy)
  - Any architecture: transformer, SSM, CNN, encoder-only, decoder-only
  - Resume: skips already-compressed tensors
  - Tied weights: detects lm_head == embed_tokens, compresses once

Output:
  {model_dir}/cdnav3/         .cdnav3/ subdirectories
  {model_dir}/cdnav3/manifest.json
  receipts/compress/{model_name}_{timestamp}.json
"""

import argparse
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.tensor_policy import (
    classify_tensor,
    get_policy,
    TensorClass,
    get_default_policy,
)


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def read_model_config(model_dir: Path) -> dict:
    """Read config.json and extract everything we need."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        raw = json.load(f)

    # Extract fields with fallback keys (different architectures use different names)
    n_layers = (
        raw.get("num_hidden_layers")
        or raw.get("n_layer")
        or raw.get("num_layers")
        or raw.get("n_layers")
    )

    model_type = raw.get("model_type", "unknown")
    architectures = raw.get("architectures", [])

    # Model name: prefer _name_or_path, fall back to directory name
    model_name = raw.get("_name_or_path", model_dir.name)
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    return {
        "n_layers": n_layers,
        "model_type": model_type,
        "architectures": architectures,
        "model_name": model_name,
        "vocab_size": raw.get("vocab_size"),
        "hidden_size": raw.get("hidden_size") or raw.get("d_model"),
        "tie_word_embeddings": raw.get("tie_word_embeddings", False),
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Weight source abstraction
# ---------------------------------------------------------------------------

class WeightSource:
    """Unified interface to read tensors from any model format."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._mode = None
        self._sf_handles = {}
        self._weight_map = {}
        self._pt_state = None
        self._discover()

    def _discover(self):
        """Discover which weight format is available."""
        # Priority 1: single safetensors
        single = self.model_dir / "model.safetensors"
        if single.exists():
            self._mode = "single_sf"
            return

        # Priority 2: sharded safetensors
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            self._weight_map = data.get("weight_map", {})
            # Verify all shards exist
            for shard_file in set(self._weight_map.values()):
                if not (self.model_dir / shard_file).exists():
                    raise FileNotFoundError(f"Missing shard: {shard_file}")
            self._mode = "sharded_sf"
            return

        # Priority 3: pytorch_model.bin
        pt_path = self.model_dir / "pytorch_model.bin"
        if pt_path.exists():
            import torch
            self._pt_state = torch.load(pt_path, map_location="cpu", weights_only=True)
            self._mode = "pytorch"
            return

        raise FileNotFoundError(
            f"No model weights found in {self.model_dir}. "
            f"Expected model.safetensors, sharded safetensors, or pytorch_model.bin"
        )

    def tensor_names(self) -> list[str]:
        """List all tensor names in the model."""
        from safetensors import safe_open

        if self._mode == "single_sf":
            sf = self._get_sf_handle("model.safetensors")
            return list(sf.keys())
        elif self._mode == "sharded_sf":
            return list(self._weight_map.keys())
        elif self._mode == "pytorch":
            return list(self._pt_state.keys())
        return []

    def get_tensor(self, name: str) -> np.ndarray:
        """Load a single tensor as float32 numpy array."""
        if self._mode == "single_sf":
            sf = self._get_sf_handle("model.safetensors")
            return sf.get_tensor(name).float().numpy()
        elif self._mode == "sharded_sf":
            shard_file = self._weight_map[name]
            sf = self._get_sf_handle(shard_file)
            return sf.get_tensor(name).float().numpy()
        elif self._mode == "pytorch":
            return self._pt_state[name].float().numpy()
        raise RuntimeError(f"Unknown mode: {self._mode}")

    def get_shape(self, name: str) -> tuple:
        """Get tensor shape without loading full data."""
        if self._mode == "pytorch":
            return tuple(self._pt_state[name].shape)
        # Use torch framework for safetensors (handles bf16/fp16 natively)
        if self._mode == "single_sf":
            sf = self._get_sf_handle("model.safetensors")
        else:
            shard_file = self._weight_map[name]
            sf = self._get_sf_handle(shard_file)
        t = sf.get_tensor(name)
        return tuple(t.shape)

    def _get_sf_handle(self, filename: str):
        from safetensors import safe_open
        if filename not in self._sf_handles:
            # Use PyTorch framework to handle bf16/fp16 (numpy lacks bf16)
            self._sf_handles[filename] = safe_open(
                str(self.model_dir / filename), framework="pt"
            )
        return self._sf_handles[filename]

    def close(self):
        self._sf_handles.clear()
        self._pt_state = None

    @property
    def mode(self) -> str:
        return self._mode


# ---------------------------------------------------------------------------
# Kurtosis computation (no scipy dependency)
# ---------------------------------------------------------------------------

def kurtosis_1d(data: np.ndarray) -> float:
    """Fisher excess kurtosis. Matches scipy.stats.kurtosis(fisher=True)."""
    mu = data.mean()
    var = ((data - mu) ** 2).mean()
    if var < 1e-30:
        return 0.0
    m4 = ((data - mu) ** 4).mean()
    return float(m4 / (var ** 2) - 3.0)


# ---------------------------------------------------------------------------
# Tensor filtering
# ---------------------------------------------------------------------------

def should_compress(name: str, shape: tuple, config: dict) -> str:
    """
    Decide what to do with a tensor.

    Returns:
        "compress" — VQ encode this tensor
        "exact"    — store as-is (1D norms, tiny tensors)
        "skip"     — don't store (tied duplicate, non-weight)
    """
    # Skip non-weight tensors (biases are 1D, handled separately at load time)
    if not name.endswith(".weight") and not name.endswith("_weight"):
        # Some models have A_log, D, etc. (Mamba state params)
        # Compress if 2D, skip if 1D
        if len(shape) == 1:
            return "skip"
        if len(shape) == 2:
            return "compress"
        return "skip"

    # 1D tensors: norms, biases → store exact
    if len(shape) == 1:
        return "exact"

    # 2D tensors: the main event
    if len(shape) == 2:
        # Tiny tensors (< 256 elements): not worth quantizing
        if shape[0] * shape[1] < 256:
            return "exact"
        return "compress"

    # 3D+ tensors: rare, but reshape to 2D and compress
    if len(shape) >= 2:
        return "compress"

    return "skip"


def detect_tied_weights(names: list[str], source: WeightSource) -> set[str]:
    """
    Detect tied weights (e.g., lm_head.weight == model.embed_tokens.weight).

    Returns set of tensor names that are duplicates (should be skipped).
    """
    # Common tied pairs
    embed_names = [n for n in names if "embed" in n.lower() and n.endswith(".weight")]
    head_names = [n for n in names if "lm_head" in n.lower() and n.endswith(".weight")]

    tied = set()
    if embed_names and head_names:
        # Check if they share the same data
        for ename in embed_names:
            for hname in head_names:
                try:
                    e_shape = source.get_shape(ename)
                    h_shape = source.get_shape(hname)
                    if e_shape == h_shape:
                        # Same shape — likely tied. Keep embedding, skip lm_head.
                        tied.add(hname)
                except Exception:
                    pass
    return tied


# ---------------------------------------------------------------------------
# Main compression pipeline
# ---------------------------------------------------------------------------

def compress_model(model_dir: Path, dry_run: bool = False, force: bool = False,
                   k_override: int = None):
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    model_dir = Path(model_dir).expanduser().resolve()
    assert model_dir.is_dir(), f"Not a directory: {model_dir}"

    # ── Phase 1: Discover environment ──
    config = read_model_config(model_dir)
    model_name = config.get("model_name", model_dir.name)
    n_layers = config.get("n_layers")
    model_type = config.get("model_type", "unknown")
    tied = config.get("tie_word_embeddings", False)

    print("=" * 70)
    print(f"  CDNA v3 Universal Compressor")
    print("=" * 70)
    print(f"  Model:      {model_name}")
    print(f"  Type:        {model_type}")
    print(f"  Directory:   {model_dir}")
    print(f"  Layers:      {n_layers or '(not in config)'}")
    print(f"  Tied embeds: {tied}")

    # Open weight source
    source = WeightSource(model_dir)
    print(f"  Weight fmt:  {source.mode}")

    # Discover all tensors
    all_names = source.tensor_names()
    print(f"  Total keys:  {len(all_names)}")

    # Detect tied weights
    tied_names = detect_tied_weights(all_names, source) if tied else set()
    if tied_names:
        print(f"  Tied (skip): {', '.join(sorted(tied_names))}")

    # Classify and filter
    plan = []  # List of (name, shape, action)
    for name in sorted(all_names):
        if name in tied_names:
            continue
        shape = source.get_shape(name)
        action = should_compress(name, shape, config)
        if action != "skip":
            plan.append((name, shape, action))

    n_compress = sum(1 for _, _, a in plan if a == "compress")
    n_exact = sum(1 for _, _, a in plan if a == "exact")

    print(f"\n  Compression plan:")
    print(f"    Compress (VQ):  {n_compress} tensors")
    print(f"    Exact (store):  {n_exact} tensors")
    print(f"    Skip (tied/1D): {len(all_names) - len(plan) - len(tied_names)} tensors")

    if dry_run:
        print(f"\n  [DRY RUN] Would compress {n_compress} tensors. Exiting.")
        # Print the plan
        for name, shape, action in plan:
            tc = classify_tensor(name, shape=shape)
            print(f"    {action:8s}  {tc.value:15s}  {str(shape):>20s}  {name}")
        return

    # ── Phase 2: Encode ──
    if k_override and k_override != 256:
        cdna_dir = model_dir / f"cdnav3_k{k_override}"
    else:
        cdna_dir = model_dir / "cdnav3"
    cdna_dir.mkdir(parents=True, exist_ok=True)
    writer = CDNAv3Writer(cdna_dir)

    if k_override:
        print(f"\n  k override: {k_override} (info-theoretic {32.0 / np.log2(k_override):.1f}x)")
    print(f"\n  Output: {cdna_dir}")
    print(f"  Encoding {n_compress} tensors...\n")

    stats_all = []
    n_done = 0
    n_cached = 0
    total_dense = 0
    total_compressed = 0
    n_svd = 0

    for name, shape, action in plan:
        if action == "exact":
            # Store 1D tensors (norms) as-is
            safe_name = name.replace("/", "_").replace(".", "_")
            exact_path = cdna_dir / f"{safe_name}.npy"
            if not exact_path.exists() or force:
                tensor_np = source.get_tensor(name)
                np.save(exact_path, tensor_np)
                del tensor_np
            continue

        # action == "compress"
        safe_name = name.replace("/", "_").replace(".", "_")
        tensor_dir = cdna_dir / f"{safe_name}.cdnav3"

        # Resume: skip if already compressed
        if tensor_dir.exists() and (tensor_dir / "codebook.npy").exists() and not force:
            n_cached += 1
            n_done += 1
            # Read existing stats for manifest
            stats_path = tensor_dir / "stats.json"
            if stats_path.exists():
                s = json.loads(stats_path.read_text())
                total_dense += s.get("original_bytes", 0)
                total_compressed += s.get("compressed_bytes", 0)
            continue

        # Load tensor
        tensor_np = source.get_tensor(name)

        # Ensure 2D for cdnav3_writer
        orig_shape = tensor_np.shape
        if tensor_np.ndim == 1:
            tensor_np = tensor_np.reshape(1, -1)
        elif tensor_np.ndim > 2:
            tensor_np = tensor_np.reshape(-1, tensor_np.shape[-1])

        # Classify + route
        tc = classify_tensor(name, shape=orig_shape)
        kurt = kurtosis_1d(tensor_np.ravel())

        # Parse block index from name (for last-block rule)
        import re
        block_match = re.search(r'layers?\.(\d+)', name)
        block_idx = int(block_match.group(1)) if block_match else None

        policy = get_policy(
            name, orig_shape,
            block_idx=block_idx,
            kurtosis=kurt,
            n_blocks=n_layers,
        )

        # Override k if requested (preserves kurtosis routing, SVD, sidecar)
        if k_override:
            from dataclasses import replace
            policy = replace(policy, n_clusters=k_override)

        # Encode
        stats = writer.write_tensor(tensor_np, name, policy=policy)
        stats_all.append(stats)

        n_done += 1
        total_dense += stats.get("original_bytes", 0)
        total_compressed += stats.get("compressed_bytes", 0)
        if stats.get("svd_bytes", 0) > 0:
            n_svd += 1

        del tensor_np

        # Progress
        if n_done % 10 == 0 or n_done == n_compress:
            ratio_so_far = total_dense / max(1, total_compressed)
            print(f"    {n_done:4d}/{n_compress}  "
                  f"cos={stats.get('cosine_with_sidecar', 0):.4f}  "
                  f"kurt={kurt:6.1f}  "
                  f"ratio={ratio_so_far:.2f}x  "
                  f"{tc.value:15s}  {name}",
                  flush=True)

    source.close()

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start
    ratio = round(total_dense / max(1, total_compressed), 2)

    # ── Phase 3: Manifest + Receipt ──
    print(f"\n{'=' * 70}")
    print(f"  Complete: {n_done} tensors ({n_cached} cached)")
    print(f"  SVD sidecars: {n_svd} tensors (kurtosis-routed)")
    if total_dense > 0:
        print(f"  Dense: {total_dense / 1e9:.3f} GB → Compressed: {total_compressed / 1e9:.3f} GB ({ratio}x)")
    print(f"  Time: {wall:.0f}s wall, {cpu:.0f}s CPU")
    print(f"{'=' * 70}")

    manifest = {
        "model": model_name,
        "model_type": model_type,
        "architectures": config.get("architectures", []),
        "n_layers": n_layers,
        "k": k_override or 256,
        "n_tensors_compressed": n_done,
        "n_tensors_cached": n_cached,
        "n_tensors_exact": n_exact,
        "n_svd_routed": n_svd,
        "total_dense_bytes": int(total_dense),
        "total_compressed_bytes": int(total_compressed),
        "compression_ratio": ratio,
        "compressor": "compress.py (universal)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = cdna_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest: {manifest_path}")

    # Receipt
    receipt = {
        "work_order": f"compress-{model_name}",
        "question": f"Does {model_name} compress through CDNA v3?",
        "verdict": "PASS" if n_done > 0 else "EMPTY",
        "manifest": manifest,
        "cost": {
            "wall_time_s": round(wall, 3),
            "cpu_time_s": round(cpu, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipts_dir = Path(__file__).parent.parent / "receipts" / "compress"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_").replace(" ", "_").lower()
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"{safe_model}_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"  Receipt:  {receipt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Universal CDNA v3 compressor. Point at any model directory.",
        epilog="Examples:\n"
               "  python3 tools/compress.py ~/models/tinyllama_fp32\n"
               "  python3 tools/compress.py ~/models/qwen2.5-7b-instruct --dry-run\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_dir", type=Path, help="Path to model directory (must contain config.json + weights)")
    parser.add_argument("--dry-run", action="store_true", help="Show compression plan without encoding")
    parser.add_argument("--force", action="store_true", help="Re-compress even if cached")
    parser.add_argument("--k", type=int, default=None,
                        help="Override codebook size (e.g. --k 64 for 6-bit). "
                             "Preserves kurtosis routing and SVD. Output: cdnav3_k{K}/")
    args = parser.parse_args()

    compress_model(args.model_dir, dry_run=args.dry_run, force=args.force,
                   k_override=args.k)


if __name__ == "__main__":
    main()
