#!/usr/bin/env python3
"""
Compress Qwen2.5-7B-Instruct to CDNA v3 / HelixLinear format.

Handles multi-shard safetensors (model-00001-of-00004.safetensors etc.)
Same pipeline as precompress_models.py but adapted for sharded models.

Output:
  ~/models/qwen2.5-7b-instruct/cdnav3/   (28 blocks × 7 tensors = 196 HelixLinear)

Usage:
  python3 tools/compress_qwen7b.py
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

MODEL_DIR = Path.home() / "models" / "qwen2.5-7b-instruct"
N_BLOCKS = 28
TENSOR_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

HF_PATTERNS = {
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}


def load_shard_index(model_dir: Path) -> dict:
    """Load the shard index mapping tensor names to shard files."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            data = json.load(f)
        return data.get("weight_map", {})
    return {}


def get_tensor_from_shards(tensor_name: str, weight_map: dict, model_dir: Path, shard_cache: dict):
    """Load a single tensor from the correct shard file."""
    from safetensors import safe_open

    shard_file = weight_map.get(tensor_name)
    if shard_file is None:
        raise KeyError(f"Tensor {tensor_name} not in weight_map")

    shard_path = model_dir / shard_file

    # Cache open shard handles to avoid re-opening
    if shard_file not in shard_cache:
        shard_cache[shard_file] = safe_open(str(shard_path), framework="pt")

    return shard_cache[shard_file].get_tensor(tensor_name).float().numpy()


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 70)
    print("  Compress Qwen2.5-7B-Instruct → CDNA v3")
    print("=" * 70)

    # Check model exists
    if not MODEL_DIR.exists():
        print(f"  ERROR: {MODEL_DIR} not found")
        sys.exit(1)

    # Check for single-file or sharded safetensors
    single_sf = MODEL_DIR / "model.safetensors"
    weight_map = load_shard_index(MODEL_DIR)

    if single_sf.exists():
        mode = "single"
        print(f"  Source: {single_sf}")
    elif weight_map:
        shard_files = sorted(set(weight_map.values()))
        mode = "sharded"
        print(f"  Source: {len(shard_files)} shards")
        # Verify all shards exist
        for sf in shard_files:
            if not (MODEL_DIR / sf).exists():
                print(f"  ERROR: Missing shard {sf}")
                sys.exit(1)
    else:
        print(f"  ERROR: No model.safetensors or shard index found in {MODEL_DIR}")
        sys.exit(1)

    cdna_dir = MODEL_DIR / "cdnav3"
    cdna_dir.mkdir(parents=True, exist_ok=True)
    writer = CDNAv3Writer(cdna_dir)
    print(f"  Target: {cdna_dir}")
    print(f"  Blocks: {N_BLOCKS}, Tensors/block: {len(TENSOR_TYPES)}")
    print(f"  Total: {N_BLOCKS * len(TENSOR_TYPES)} tensors")
    print()

    n_tensors = 0
    n_skipped = 0
    total_dense = 0
    total_compressed = 0
    shard_cache = {}

    for block_idx in range(N_BLOCKS):
        block_t0 = time.time()
        block_tensors = 0

        for tensor_type in TENSOR_TYPES:
            hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)

            # Check if already compressed
            safe_name = hf_name.replace("/", "_").replace(".", "_")
            tensor_dir = cdna_dir / f"{safe_name}.cdnav3"
            if tensor_dir.exists() and (tensor_dir / "codebook.npy").exists():
                n_skipped += 1
                n_tensors += 1
                continue

            # Load tensor
            if mode == "single":
                from safetensors import safe_open
                if not hasattr(main, '_sf_handle'):
                    main._sf_handle = safe_open(str(single_sf), framework="pt")
                tensor_np = main._sf_handle.get_tensor(hf_name).float().numpy()
            else:
                tensor_np = get_tensor_from_shards(hf_name, weight_map, MODEL_DIR, shard_cache)

            shape = tensor_np.shape
            kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
            policy = get_policy(hf_name, shape, block_idx=block_idx, kurtosis=kurt)

            stats = writer.write_tensor(tensor_np, hf_name, policy=policy)
            n_tensors += 1
            block_tensors += 1
            total_dense += np.prod(shape) * 4
            total_compressed += stats.get("compressed_bytes", 0)

            # Free memory
            del tensor_np

        block_elapsed = time.time() - block_t0
        if block_tensors > 0:
            print(f"  Block {block_idx:2d}/{N_BLOCKS} — "
                  f"{block_tensors} tensors, {block_elapsed:.1f}s", flush=True)
        elif (block_idx + 1) % 7 == 0:
            print(f"  Block {block_idx:2d}/{N_BLOCKS} — (cached)", flush=True)

    # Close shard handles
    shard_cache.clear()

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start
    ratio = round(total_dense / total_compressed, 2) if total_compressed > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  Complete: {n_tensors} tensors ({n_skipped} cached)")
    if total_dense > 0:
        print(f"  Dense: {total_dense / 1e9:.2f} GB → Compressed: {total_compressed / 1e9:.2f} GB ({ratio}x)")
    print(f"  Time: {wall:.0f}s wall, {cpu:.0f}s CPU")
    print(f"{'=' * 70}")

    # Write manifest
    manifest = {
        "model": "Qwen2.5-7B-Instruct",
        "n_blocks": N_BLOCKS,
        "n_tensors": n_tensors,
        "n_newly_compressed": n_tensors - n_skipped,
        "total_dense_bytes": int(total_dense),
        "total_compressed_bytes": int(total_compressed),
        "compression_ratio": ratio,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = cdna_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Write receipt
    receipt = {
        "work_order": "WO-QWEN7B-COMPRESS-01",
        "question": "Does Qwen2.5-7B-Instruct compress through CDNA v3?",
        "verdict": "PASS" if n_tensors == N_BLOCKS * len(TENSOR_TYPES) else "PARTIAL",
        "manifest": manifest,
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

    receipts_dir = Path(__file__).parent.parent / "receipts" / "qwen7b_compress"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"qwen7b_compress_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
