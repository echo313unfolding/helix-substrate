#!/usr/bin/env python3
"""
Compress Mamba-2 1.3B to CDNA v3 / HelixLinear format.

Second SSM through the pipeline. Validates architecture-agnostic compression
on a non-transformer (state space model) architecture.

Mamba-2 1.3B architecture (state-spaces/mamba2-1.3b):
  - 48 layers, d_model=2048
  - 2 linear layers per block (96 total):
    - in_proj:  (8512, 2048)  — SSM input expansion (packs B, C, dt, x, z)
    - out_proj: (2048, 4096)  — SSM output contraction
  - Plus: conv1d, A_log, D, dt_bias, norm per layer (not compressed)
  - Embedding: backbone.embedding.weight (50288, 2048)
  - lm_head: tied to embedding, stored separately in safetensors
  - Source model is FP16 (converted to FP32 for compression)

Output:
  ~/models/mamba2-1.3b/cdnav3/

Usage:
  python3 tools/compress_mamba2_1.3b.py
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

MODEL_DIR = Path.home() / "models" / "mamba2-1.3b"

# Embedding tensor (note: "embedding" not "embeddings")
EMBEDDING_NAME = "backbone.embedding.weight"
# lm_head is tied to embedding but stored separately in safetensors
LM_HEAD_NAME = "lm_head.weight"


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 70)
    print("  Compress Mamba-2 1.3B → CDNA v3")
    print("  Second SSM through HelixLinear pipeline")
    print("=" * 70)

    if not MODEL_DIR.exists():
        print(f"  ERROR: {MODEL_DIR} not found")
        sys.exit(1)

    # Find safetensors file(s), auto-convert from pytorch if needed
    single_sf = MODEL_DIR / "model.safetensors"
    if not single_sf.exists():
        pt_path = MODEL_DIR / "pytorch_model.bin"
        if pt_path.exists():
            print(f"  Converting pytorch_model.bin → model.safetensors...")
            import torch
            from safetensors.torch import save_file
            state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
            # Break shared memory (tied embeddings) for safetensors
            for k in list(state.keys()):
                state[k] = state[k].clone()
            save_file(state, str(single_sf))
            del state
            print(f"  Converted: {single_sf}")
        else:
            print(f"  ERROR: No model.safetensors or pytorch_model.bin in {MODEL_DIR}")
            sys.exit(1)

    from safetensors import safe_open
    sf = safe_open(str(single_sf), framework="pt")
    all_keys = list(sf.keys())
    print(f"  Source: {single_sf}")
    print(f"  Total safetensor keys: {len(all_keys)}")

    # Auto-detect architecture from tensor names
    # Find all mixer linear layers (2D weight tensors)
    mixer_keys = [k for k in all_keys if ".mixer." in k and k.endswith(".weight")]
    # Separate 2D (linear) from 1D (norm/bias)
    linear_keys = []
    for k in mixer_keys:
        t = sf.get_tensor(k)
        if t.ndim == 2:
            linear_keys.append(k)
        del t

    # Detect block count
    block_indices = set()
    for k in linear_keys:
        import re
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            block_indices.add(int(m.group(1)))

    n_blocks = max(block_indices) + 1 if block_indices else 0
    linear_per_block = len(linear_keys) // max(n_blocks, 1) if n_blocks > 0 else 0

    print(f"  Detected: {n_blocks} blocks, {linear_per_block} linear layers/block")
    print(f"  2D mixer tensors: {len(linear_keys)}")

    # List unique projection types
    proj_types = set()
    for k in linear_keys:
        parts = k.split(".")
        mixer_idx = parts.index("mixer")
        proj = parts[mixer_idx + 1]
        proj_types.add(proj)
    print(f"  Projection types: {sorted(proj_types)}")
    print()

    cdna_dir = MODEL_DIR / "cdnav3"
    cdna_dir.mkdir(parents=True, exist_ok=True)
    writer = CDNAv3Writer(cdna_dir)

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
    elif EMBEDDING_NAME in all_keys:
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

    # ── Compress lm_head (tied to embedding but separate in safetensors) ──
    safe_name_lm = LM_HEAD_NAME.replace("/", "_").replace(".", "_")
    tensor_dir_lm = cdna_dir / f"{safe_name_lm}.cdnav3"
    if tensor_dir_lm.exists() and (tensor_dir_lm / "codebook.npy").exists():
        print(f"  lm_head: (cached)")
        n_skipped += 1
        n_tensors += 1
    elif LM_HEAD_NAME in all_keys:
        tensor_np = sf.get_tensor(LM_HEAD_NAME).float().numpy()
        shape = tensor_np.shape
        kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
        policy = get_policy(LM_HEAD_NAME, shape, block_idx=None, kurtosis=kurt)

        stats = writer.write_tensor(tensor_np, LM_HEAD_NAME, policy=policy)
        n_tensors += 1
        total_dense += np.prod(shape) * 4
        total_compressed += stats.get("compressed_bytes", 0)
        per_tensor_stats.append({
            "name": LM_HEAD_NAME,
            "shape": list(shape),
            "kurtosis": round(kurt, 2),
            "ratio": round(stats.get("compression_ratio", 0), 2),
            "cosine": round(stats.get("cosine_with_sidecar", 0), 6),
        })
        print(f"  lm_head: {shape} → {stats.get('compression_ratio', 0):.2f}x, "
              f"cos={stats.get('cosine_with_sidecar', 0):.6f}")
        del tensor_np

    # ── Compress all 2D mixer layers ──
    for block_idx in range(n_blocks):
        block_t0 = time.time()
        block_tensors = 0

        # Get all linear keys for this block
        block_keys = [k for k in linear_keys
                      if f"layers.{block_idx}." in k]

        for hf_name in sorted(block_keys):
            safe_name = hf_name.replace("/", "_").replace(".", "_")
            tensor_dir = cdna_dir / f"{safe_name}.cdnav3"
            if tensor_dir.exists() and (tensor_dir / "codebook.npy").exists():
                n_skipped += 1
                n_tensors += 1
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
                "cosine": round(stats.get("cosine_with_svd",
                                          stats.get("cosine_with_sidecar", 0)), 6),
            })

            del tensor_np

        block_elapsed = time.time() - block_t0
        if block_tensors > 0:
            print(f"  Block {block_idx:2d}/{n_blocks} — "
                  f"{block_tensors} tensors, {block_elapsed:.1f}s", flush=True)

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start
    ratio = round(total_dense / total_compressed, 2) if total_compressed > 0 else 0

    total_expected = len(linear_keys) + \
        (1 if EMBEDDING_NAME in all_keys else 0) + \
        (1 if LM_HEAD_NAME in all_keys else 0)

    print(f"\n{'=' * 70}")
    print(f"  Complete: {n_tensors} tensors ({n_skipped} cached)")
    if total_dense > 0:
        print(f"  Dense: {total_dense / 1e6:.1f} MB → Compressed: {total_compressed / 1e6:.1f} MB ({ratio}x)")
    print(f"  Time: {wall:.0f}s wall, {cpu:.0f}s CPU")
    print(f"{'=' * 70}")

    manifest = {
        "model": "Mamba-2-1.3B",
        "architecture": "Mamba2ForCausalLM",
        "n_blocks": n_blocks,
        "n_tensors": n_tensors,
        "n_newly_compressed": n_tensors - n_skipped,
        "projection_types": sorted(proj_types),
        "total_dense_bytes": int(total_dense),
        "total_compressed_bytes": int(total_compressed),
        "compression_ratio": ratio,
        "note": "Second SSM (Mamba-2) through HelixLinear pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = cdna_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    receipt = {
        "work_order": "WO-MAMBA2-COMPRESS-01",
        "question": "Does Mamba-2 1.3B (SSM) compress through CDNA v3?",
        "verdict": "PASS" if n_tensors >= total_expected else "PARTIAL",
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

    receipts_dir = Path(__file__).parent.parent / "receipts" / "mamba2_compress"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"mamba2_compress_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
