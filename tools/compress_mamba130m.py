#!/usr/bin/env python3
"""
Compress Mamba-130m to CDNA v3 / HelixLinear format.

First SSM (non-transformer) model through the HelixLinear pipeline.
Proves architecture-agnostic compression.

Mamba-130m has 4 linear layers per block (vs 7 for transformers):
  - in_proj:  (3072, 768)  — SSM input expansion
  - x_proj:   (80, 1536)   — state projection (tiny)
  - dt_proj:  (1536, 48)   — timestep projection (tiny)
  - out_proj: (768, 1536)  — SSM output contraction

Plus embedding: backbone.embeddings.weight (50280, 768)

Output:
  ~/models/mamba-130m-hf/cdnav3/   (24 blocks × 4 tensors + 1 embedding = 97)

Usage:
  python3 tools/compress_mamba130m.py
"""

import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.tensor_policy import get_policy

MODEL_DIR = Path.home() / "models" / "mamba-130m-hf"
N_BLOCKS = 24
TENSOR_TYPES = ["in_proj", "x_proj", "dt_proj", "out_proj"]

# Mamba uses backbone.layers.N.mixer.{proj}.weight
HF_PATTERNS = {
    "in_proj":  "backbone.layers.{i}.mixer.in_proj.weight",
    "x_proj":   "backbone.layers.{i}.mixer.x_proj.weight",
    "dt_proj":  "backbone.layers.{i}.mixer.dt_proj.weight",
    "out_proj": "backbone.layers.{i}.mixer.out_proj.weight",
}

# Embedding tensor (compressed separately)
EMBEDDING_NAME = "backbone.embeddings.weight"


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 70)
    print("  Compress Mamba-130m → CDNA v3")
    print("  First SSM through HelixLinear pipeline")
    print("=" * 70)

    if not MODEL_DIR.exists():
        print(f"  ERROR: {MODEL_DIR} not found")
        sys.exit(1)

    single_sf = MODEL_DIR / "model.safetensors"
    if not single_sf.exists():
        print(f"  ERROR: {single_sf} not found")
        sys.exit(1)

    from safetensors import safe_open
    sf = safe_open(str(single_sf), framework="pt")
    available_keys = sf.keys()
    print(f"  Source: {single_sf}")
    print(f"  Total safetensor keys: {len(available_keys)}")

    cdna_dir = MODEL_DIR / "cdnav3"
    cdna_dir.mkdir(parents=True, exist_ok=True)
    writer = CDNAv3Writer(cdna_dir)

    expected_total = N_BLOCKS * len(TENSOR_TYPES) + 1  # +1 for embedding
    print(f"  Target: {cdna_dir}")
    print(f"  Blocks: {N_BLOCKS}, Tensors/block: {len(TENSOR_TYPES)}")
    print(f"  Total: {expected_total} tensors (96 mixer + 1 embedding)")
    print()

    n_tensors = 0
    n_skipped = 0
    total_dense = 0
    total_compressed = 0
    per_tensor_stats = []

    # ── Compress embedding ──
    safe_name = EMBEDDING_NAME.replace("/", "_").replace(".", "_")
    tensor_dir = cdna_dir / f"{safe_name}.cdnav3"
    if tensor_dir.exists() and (tensor_dir / "codebook.npy").exists():
        print(f"  Embedding: (cached)")
        n_skipped += 1
        n_tensors += 1
    elif EMBEDDING_NAME in available_keys:
        tensor_np = sf.get_tensor(EMBEDDING_NAME).float().numpy()
        shape = tensor_np.shape
        kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
        policy = get_policy(EMBEDDING_NAME, shape, block_idx=None, kurtosis=kurt)

        stats = writer.write_tensor(tensor_np, EMBEDDING_NAME, policy=policy)
        n_tensors += 1
        total_dense += np.prod(shape) * 4
        total_compressed += stats.get("compressed_bytes", 0)
        per_tensor_stats.append({
            "name": EMBEDDING_NAME,
            "shape": list(shape),
            "kurtosis": round(kurt, 2),
            "ratio": round(stats.get("compression_ratio", 0), 2),
            "cosine": round(stats.get("cosine_with_sidecar", 0), 6),
        })
        print(f"  Embedding: {shape} → {stats.get('compression_ratio', 0):.2f}x, "
              f"cos={stats.get('cosine_with_sidecar', 0):.6f}")
        del tensor_np
    else:
        print(f"  WARNING: {EMBEDDING_NAME} not found in safetensors")

    # ── Compress mixer layers ──
    for block_idx in range(N_BLOCKS):
        block_t0 = time.time()
        block_tensors = 0

        for tensor_type in TENSOR_TYPES:
            hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)

            safe_name = hf_name.replace("/", "_").replace(".", "_")
            tensor_dir = cdna_dir / f"{safe_name}.cdnav3"
            if tensor_dir.exists() and (tensor_dir / "codebook.npy").exists():
                n_skipped += 1
                n_tensors += 1
                continue

            if hf_name not in available_keys:
                print(f"  WARNING: {hf_name} not found in safetensors — skipping")
                continue

            tensor_np = sf.get_tensor(hf_name).float().numpy()
            shape = tensor_np.shape
            kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
            policy = get_policy(hf_name, shape, block_idx=block_idx, kurtosis=kurt)

            stats = writer.write_tensor(tensor_np, hf_name, policy=policy)
            n_tensors += 1
            block_tensors += 1
            total_dense += np.prod(shape) * 4
            total_compressed += stats.get("compressed_bytes", 0)
            per_tensor_stats.append({
                "name": hf_name,
                "shape": list(shape),
                "kurtosis": round(kurt, 2),
                "ratio": round(stats.get("compression_ratio", 0), 2),
                "cosine": round(stats.get("cosine_with_sidecar", 0), 6),
            })

            del tensor_np

        block_elapsed = time.time() - block_t0
        if block_tensors > 0:
            print(f"  Block {block_idx:2d}/{N_BLOCKS} — "
                  f"{block_tensors} tensors, {block_elapsed:.1f}s", flush=True)

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start
    ratio = round(total_dense / total_compressed, 2) if total_compressed > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  Complete: {n_tensors} tensors ({n_skipped} cached)")
    if total_dense > 0:
        print(f"  Dense: {total_dense / 1e6:.1f} MB → Compressed: {total_compressed / 1e6:.1f} MB ({ratio}x)")
    print(f"  Time: {wall:.0f}s wall, {cpu:.0f}s CPU")
    print(f"{'=' * 70}")

    manifest = {
        "model": "Mamba-130m",
        "architecture": "MambaForCausalLM",
        "n_blocks": N_BLOCKS,
        "n_tensors": n_tensors,
        "n_newly_compressed": n_tensors - n_skipped,
        "total_dense_bytes": int(total_dense),
        "total_compressed_bytes": int(total_compressed),
        "compression_ratio": ratio,
        "note": "First SSM (non-transformer) through HelixLinear pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = cdna_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    receipt = {
        "work_order": "WO-MAMBA-COMPRESS-01",
        "question": "Does Mamba-130m (SSM) compress through CDNA v3?",
        "verdict": "PASS" if n_tensors >= N_BLOCKS * len(TENSOR_TYPES) else "PARTIAL",
        "manifest": manifest,
        "per_tensor_stats": per_tensor_stats,
        "cost": {
            "wall_time_s": round(wall, 3),
            "cpu_time_s": round(cpu, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipts_dir = Path(__file__).parent.parent / "receipts" / "mamba_compress"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"mamba_compress_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
