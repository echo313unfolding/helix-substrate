#!/usr/bin/env python3
"""Domain 3: Activation Checkpointing Compression.
Compress real TinyLlama activations already on disk."""

import numpy as np
from pathlib import Path
from _common import *

def main():
    t_start, cpu_start, start_iso = start_cost()
    print("=" * 72)
    print("DOMAIN 3: Activation Checkpointing Compression")
    print("=" * 72)

    act_dir = HELIX_ROOT / "calibration_activations"
    out_dir = HELIX_ROOT / "tensor_infra_scratch" / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all activation files
    act_files = sorted(act_dir.glob("layer_*_*_input.npy"))
    print(f"  Found {len(act_files)} activation tensors")

    policy = policy_vq(k=256, sidecar=True)
    policy_svd = policy_vq(k=256, sidecar=True, svd_rank=8)

    results = []
    total_orig_bytes = 0
    total_comp_bytes = 0

    for act_path in act_files:
        name = act_path.stem  # e.g., "layer_00_attn_input"
        tensor = np.load(act_path).astype(np.float32)

        # Ensure 2D
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        elif tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])

        kurt = kurtosis(tensor)
        total_orig_bytes += tensor.nbytes

        # Choose policy based on kurtosis (same routing as weights)
        p = policy_svd if kurt > 5.0 else policy

        stats, recon = compress_tensor(tensor, name, out_dir, p)
        cos = cosine_sim(tensor, recon)
        err = mse(tensor, recon)
        comp_bytes = stats.get("compressed_bytes", tensor.nbytes)
        total_comp_bytes += comp_bytes

        results.append({
            "tensor": name,
            "shape": list(tensor.shape),
            "kurtosis": round(kurt, 2),
            "cosine": round(cos, 6),
            "mse": round(err, 10),
            "ratio": round(tensor.nbytes / max(1, comp_bytes), 2),
            "policy": "VQ+SVD" if kurt > 5.0 else "VQ",
        })

    # Summary
    cosines = [r["cosine"] for r in results]
    ratios = [r["ratio"] for r in results]
    v, worst = verdict(cosines)

    print(f"\n  Total: {len(results)} tensors")
    print(f"  Original: {total_orig_bytes / 1e6:.1f} MB")
    print(f"  Compressed: {total_comp_bytes / 1e6:.1f} MB")
    print(f"  Overall ratio: {total_orig_bytes / max(1, total_comp_bytes):.2f}x")
    print(f"  Cosine: min={min(cosines):.6f}, mean={np.mean(cosines):.6f}")
    print(f"  Verdict: {v}")

    # Show worst 5
    results_sorted = sorted(results, key=lambda r: r["cosine"])
    print(f"\n  Worst 5 tensors:")
    for r in results_sorted[:5]:
        print(f"    {r['tensor']}: cos={r['cosine']:.6f}, kurt={r['kurtosis']:.1f}, policy={r['policy']}")

    cost = finish_cost(t_start, cpu_start, start_iso)
    write_receipt("tensor_infra_domain_3", "activation_checkpoint", {
        "n_tensors": len(results),
        "total_original_mb": round(total_orig_bytes / 1e6, 1),
        "total_compressed_mb": round(total_comp_bytes / 1e6, 1),
        "overall_ratio": round(total_orig_bytes / max(1, total_comp_bytes), 2),
        "cosine_min": round(min(cosines), 6),
        "cosine_mean": round(float(np.mean(cosines)), 6),
        "verdict": v,
        "per_tensor": results,
        "data_source": "REAL — TinyLlama calibration activations from WikiText-2",
    }, cost)

if __name__ == "__main__":
    main()
