#!/usr/bin/env python3
"""
Pre-compress TinyLlama and Qwen to persistent CDNA v3 directories.

One-time operation. After this, model loading is just:
  model = load_from_cdnav3(cdna_dir)  → model.cuda()
instead of:
  model = load_safetensors() → compress() → swap_to_helix()

Output:
  ~/models/tinyllama_fp32/cdnav3/          (154 HelixLinear tensors)
  ~/models/qwen2.5-coder-1.5b-instruct/cdnav3/  (196 HelixLinear tensors)
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

MODELS = [
    {
        "name": "TinyLlama-1.1B",
        "model_dir": Path.home() / "models" / "tinyllama_fp32",
        "n_blocks": 22,
        "tensor_types": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    {
        "name": "Qwen2.5-Coder-1.5B",
        "model_dir": Path.home() / "models" / "qwen2.5-coder-1.5b-instruct",
        "n_blocks": 28,
        "tensor_types": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
]

HF_PATTERNS = {
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}


def compress_model(model_cfg):
    from safetensors import safe_open

    model_dir = model_cfg["model_dir"]
    cdna_dir = model_dir / "cdnav3"
    sf_path = model_dir / "model.safetensors"

    if not sf_path.exists():
        print(f"  SKIP: {sf_path} not found")
        return None

    cdna_dir.mkdir(parents=True, exist_ok=True)
    writer = CDNAv3Writer(cdna_dir)

    n_tensors = 0
    total_dense = 0
    total_compressed = 0
    t0 = time.time()

    with safe_open(str(sf_path), framework="pt") as sf:
        for block_idx in range(model_cfg["n_blocks"]):
            for tensor_type in model_cfg["tensor_types"]:
                hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)

                # Check if already compressed
                safe_name = hf_name.replace("/", "_").replace(".", "_")
                tensor_dir = cdna_dir / f"{safe_name}.cdnav3"
                if tensor_dir.exists() and (tensor_dir / "codebook.npy").exists():
                    n_tensors += 1
                    continue

                tensor_np = sf.get_tensor(hf_name).float().numpy()
                shape = tensor_np.shape
                kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
                policy = get_policy(hf_name, shape, block_idx=block_idx, kurtosis=kurt)

                stats = writer.write_tensor(tensor_np, hf_name, policy=policy)
                n_tensors += 1
                total_dense += np.prod(shape) * 4
                total_compressed += stats.get("compressed_bytes", 0)

            if (block_idx + 1) % 7 == 0 or block_idx == model_cfg["n_blocks"] - 1:
                print(f"  Block {block_idx + 1}/{model_cfg['n_blocks']} ({n_tensors} tensors)", flush=True)

    elapsed = time.time() - t0

    manifest = {
        "model": model_cfg["name"],
        "n_blocks": model_cfg["n_blocks"],
        "n_tensors": n_tensors,
        "total_dense_bytes": int(total_dense),
        "total_compressed_bytes": int(total_compressed),
        "compression_ratio": round(total_dense / total_compressed, 2) if total_compressed > 0 else 0,
        "compression_time_s": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = cdna_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest


def main():
    t_start = time.time()
    print("=" * 70)
    print("  Pre-compress Models to Persistent CDNA v3")
    print("=" * 70)

    results = []
    for model_cfg in MODELS:
        print(f"\n[{model_cfg['name']}]")
        print(f"  Source: {model_cfg['model_dir']}")
        print(f"  Target: {model_cfg['model_dir'] / 'cdnav3'}")

        manifest = compress_model(model_cfg)
        if manifest:
            print(f"  Done: {manifest['n_tensors']} tensors, "
                  f"{manifest['compression_ratio']}x, "
                  f"{manifest['compression_time_s']:.0f}s")
            results.append(manifest)
        else:
            print(f"  SKIPPED")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Total: {elapsed:.0f}s")
    for r in results:
        print(f"  {r['model']}: {r['n_tensors']} tensors, {r['compression_ratio']}x")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
