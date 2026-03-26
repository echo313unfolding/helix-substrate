#!/usr/bin/env python3
"""Domain 8: Sparse MoE Tiered Compression.
Split TinyLlama FFN weights into simulated experts with different compression tiers."""

import numpy as np
from pathlib import Path
from _common import *

def main():
    t_start, cpu_start, start_iso = start_cost()
    print("=" * 72)
    print("DOMAIN 8: Sparse MoE Tiered Compression")
    print("=" * 72)

    weights = load_tinyllama_weights()
    out_dir = HELIX_ROOT / "tensor_infra_scratch" / "moe_tiered"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define tiers
    tiers = {
        "hot":  policy_vq(k=256, sidecar=True, svd_rank=8),   # High fidelity
        "warm": policy_vq(k=256, sidecar=True, svd_rank=0),   # Standard
        "cool": policy_vq(k=128, sidecar=True, svd_rank=0),   # Moderate
        "cold": policy_vq(k=64, sidecar=True, svd_rank=0),    # Aggressive
    }
    tier_names = ["hot", "warm", "cool", "cold"]
    n_experts = len(tier_names)

    # Uniform baseline for comparison
    uniform_policy = policy_vq(k=256, sidecar=True)

    # Process FFN tensors (gate_proj, up_proj, down_proj across 22 blocks)
    ffn_keys = [k for k in weights.keys()
                if any(x in k for x in ["gate_proj", "up_proj", "down_proj"])]
    print(f"  Found {len(ffn_keys)} FFN tensors")

    results = []
    total_tiered_bytes = 0
    total_uniform_bytes = 0
    total_original_bytes = 0

    for key in sorted(ffn_keys):
        tensor = weights[key].astype(np.float32)
        rows = tensor.shape[0]
        total_original_bytes += tensor.nbytes

        # Split into n_experts shards along rows
        shard_size = rows // n_experts
        shards = []
        for i in range(n_experts):
            start = i * shard_size
            end = start + shard_size if i < n_experts - 1 else rows
            shards.append(tensor[start:end])

        # Compress each shard with its tier policy
        tiered_cosines = []
        tiered_bytes = 0
        for i, (shard, tier) in enumerate(zip(shards, tier_names)):
            safe = f"{key.replace('.', '_')}_expert{i}"
            stats, recon = compress_tensor(shard, safe, out_dir / "tiered", tiers[tier])
            cos = cosine_sim(shard, recon)
            tiered_cosines.append(cos)
            tiered_bytes += stats.get("compressed_bytes", shard.nbytes)

        total_tiered_bytes += tiered_bytes

        # Uniform compression for comparison
        safe_uniform = f"{key.replace('.', '_')}_uniform"
        stats_u, recon_u = compress_tensor(tensor, safe_uniform, out_dir / "uniform", uniform_policy)
        uniform_cos = cosine_sim(tensor, recon_u)
        total_uniform_bytes += stats_u.get("compressed_bytes", tensor.nbytes)

        results.append({
            "tensor": key,
            "shape": list(tensor.shape),
            "hot_cosine": round(tiered_cosines[0], 6),
            "warm_cosine": round(tiered_cosines[1], 6),
            "cool_cosine": round(tiered_cosines[2], 6),
            "cold_cosine": round(tiered_cosines[3], 6),
            "uniform_cosine": round(uniform_cos, 6),
            "tiered_bytes": tiered_bytes,
            "uniform_bytes": stats_u.get("compressed_bytes", tensor.nbytes),
        })

    # Summary
    tiered_ratio = total_original_bytes / max(1, total_tiered_bytes)
    uniform_ratio = total_original_bytes / max(1, total_uniform_bytes)
    savings_pct = (1 - total_tiered_bytes / max(1, total_uniform_bytes)) * 100

    hot_min = min(r["hot_cosine"] for r in results)
    cold_min = min(r["cold_cosine"] for r in results)

    print(f"\n  Original: {total_original_bytes / 1e6:.1f} MB")
    print(f"  Uniform (k=256): {total_uniform_bytes / 1e6:.1f} MB ({uniform_ratio:.2f}x)")
    print(f"  Tiered: {total_tiered_bytes / 1e6:.1f} MB ({tiered_ratio:.2f}x)")
    print(f"  Tiered saves {savings_pct:.1f}% vs uniform")
    print(f"  Hot cosine min: {hot_min:.6f}")
    print(f"  Cold cosine min: {cold_min:.6f}")
    print(f"  FLAG: Simulated expert splitting on dense model")

    cost = finish_cost(t_start, cpu_start, start_iso)
    write_receipt("tensor_infra_domain_8", "moe_tiered", {
        "n_tensors": len(results),
        "n_experts": n_experts,
        "tiers": {t: {"k": tiers[t].n_clusters, "svd_rank": tiers[t].svd_residual_rank} for t in tier_names},
        "original_mb": round(total_original_bytes / 1e6, 1),
        "uniform_mb": round(total_uniform_bytes / 1e6, 1),
        "tiered_mb": round(total_tiered_bytes / 1e6, 1),
        "tiered_savings_pct": round(savings_pct, 1),
        "hot_cosine_min": round(hot_min, 6),
        "cold_cosine_min": round(cold_min, 6),
        "per_tensor": results,
        "data_source": "PARTIAL — real TinyLlama FFN weights, simulated expert splitting",
    }, cost)

if __name__ == "__main__":
    main()
