#!/usr/bin/env python3
"""
Profile k-means convergence as a function of k.

Tests k=16, 32, 64, 128, 256 on representative tensors from Qwen-1.5B.
Logs: iterations to convergence, per-iteration time, total time, reconstruction quality.

Output: receipts/kmeans_convergence/kmeans_convergence_{ts}.json
"""

import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR = Path.home() / "models" / "qwen2.5-coder-1.5b-instruct"

# Test tensors: one small (v_proj: 256×1536) and one large (gate_proj: 8960×1536)
TEST_TENSORS = [
    "model.layers.0.self_attn.v_proj.weight",    # small, early
    "model.layers.14.mlp.gate_proj.weight",       # large, middle
    "model.layers.27.self_attn.q_proj.weight",    # medium, last block
]

K_VALUES = [16, 32, 64, 128, 256]
MAX_ITERS = 30  # Higher than production (10) to see natural convergence
KMEANS_MAX_SAMPLES = 1_000_000


def profiled_kmeans(data, n_clusters, max_iters=MAX_ITERS):
    """K-means with per-iteration profiling. Returns (codebook, stats)."""
    n_unique = len(np.unique(data[:min(500_000, len(data))]))
    n_clusters = min(n_clusters, n_unique)

    # Subsample like production
    if len(data) > KMEANS_MAX_SAMPLES:
        rng = np.random.RandomState(42)
        sample = data[rng.choice(len(data), KMEANS_MAX_SAMPLES, replace=False)]
    else:
        sample = data.copy()

    # Initialize centroids using percentiles (matches production)
    centroids = np.percentile(sample, np.linspace(0, 100, n_clusters)).astype(np.float32)

    iter_times = []
    converged_at = max_iters  # default: hit max

    for iteration in range(max_iters):
        t0 = time.perf_counter()

        # Assign — chunked to avoid OOM
        assignments = np.empty(len(sample), dtype=np.int32)
        chunk = 2_000_000
        for s in range(0, len(sample), chunk):
            e = min(s + chunk, len(sample))
            dists = np.abs(sample[s:e, None] - centroids)
            assignments[s:e] = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.empty_like(centroids)
        for i in range(n_clusters):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = sample[mask].mean()
            else:
                new_centroids[i] = centroids[i]

        iter_time = time.perf_counter() - t0
        iter_times.append(iter_time)

        # Check convergence
        if np.allclose(centroids, new_centroids):
            converged_at = iteration + 1
            centroids = new_centroids
            break

        centroids = new_centroids

    # Final full assignment on ALL data (not just sample)
    indices = np.empty(len(data), dtype=np.int32)
    chunk = 2_000_000
    for s in range(0, len(data), chunk):
        e = min(s + chunk, len(data))
        dists = np.abs(data[s:e, None] - centroids)
        indices[s:e] = np.argmin(dists, axis=1)

    recon = centroids[indices]

    stats = {
        "k": n_clusters,
        "iterations": converged_at,
        "hit_max_iters": converged_at == max_iters,
        "iter_times_s": [round(t, 6) for t in iter_times],
        "mean_iter_time_s": round(float(np.mean(iter_times)), 6),
        "total_time_s": round(sum(iter_times), 6),
        "n_samples_kmeans": len(sample),
        "n_elements_total": len(data),
    }

    return centroids, recon, stats


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 70)
    print("  K-Means Convergence Profiling")
    print(f"  k values: {K_VALUES}")
    print(f"  max_iters: {MAX_ITERS}")
    print(f"  tensors: {len(TEST_TENSORS)}")
    print("=" * 70)

    assert MODEL_DIR.exists(), f"Model not found: {MODEL_DIR}"

    from safetensors import safe_open
    sf = safe_open(str(MODEL_DIR / "model.safetensors"), framework="pt")

    results = []

    for tensor_name in TEST_TENSORS:
        print(f"\n  === {tensor_name} ===")
        weight = sf.get_tensor(tensor_name).float().numpy()
        flat = weight.ravel().astype(np.float32)
        print(f"  Shape: {weight.shape}, elements: {len(flat):,}")

        for k in K_VALUES:
            print(f"    k={k:>3}: ", end="", flush=True)

            codebook, recon, stats = profiled_kmeans(flat, k)

            # Reconstruction quality
            cos = float(np.dot(flat, recon) /
                        (np.linalg.norm(flat) * np.linalg.norm(recon) + 1e-30))
            mse = float(np.mean((flat - recon) ** 2))

            stats["tensor_name"] = tensor_name
            stats["shape"] = list(weight.shape)
            stats["cosine"] = round(cos, 8)
            stats["mse"] = round(mse, 10)
            stats["info_theoretic_ratio"] = round(32.0 / np.log2(k), 2)

            results.append(stats)

            print(f"iters={stats['iterations']:>2}, "
                  f"total={stats['total_time_s']:.3f}s, "
                  f"per_iter={stats['mean_iter_time_s']:.4f}s, "
                  f"cos={cos:.6f}, "
                  f"mse={mse:.2e}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Convergence vs k")
    print(f"{'=' * 70}")
    print(f"  {'Tensor':<45} {'k':>4} {'Iters':>6} {'Per-iter':>10} {'Total':>8} {'Cosine':>10}")
    print(f"  {'-'*45} {'-'*4} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")

    for r in results:
        short_name = r["tensor_name"].split(".")[-2] + "." + r["tensor_name"].split(".")[-1]
        block = r["tensor_name"].split(".")[2]
        label = f"L{block}.{short_name}"
        print(f"  {label:<45} {r['k']:>4} {r['iterations']:>6} "
              f"{r['mean_iter_time_s']:>9.4f}s {r['total_time_s']:>7.3f}s "
              f"{r['cosine']:>10.6f}")

    # Aggregate by k
    print(f"\n  === Aggregate by k ===")
    print(f"  {'k':>4} {'Avg iters':>10} {'Avg per-iter':>13} {'Avg total':>10} {'Speedup vs 256':>15}")
    k256_total = None
    for k in K_VALUES:
        k_results = [r for r in results if r["k"] == k]
        avg_iters = np.mean([r["iterations"] for r in k_results])
        avg_per_iter = np.mean([r["mean_iter_time_s"] for r in k_results])
        avg_total = np.mean([r["total_time_s"] for r in k_results])
        if k == 256:
            k256_total = avg_total
        speedup = f"{k256_total / avg_total:.2f}x" if k256_total and avg_total > 0 else "-"
        print(f"  {k:>4} {avg_iters:>10.1f} {avg_per_iter:>12.4f}s {avg_total:>9.3f}s {speedup:>15}")

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    print(f"\n  Wall: {wall:.1f}s, CPU: {cpu:.1f}s")

    # Receipt
    receipt = {
        "work_order": "WO-KMEANS-CONVERGENCE-01",
        "question": "How does k-means convergence speed scale with k?",
        "model": "Qwen2.5-Coder-1.5B-Instruct",
        "k_values": K_VALUES,
        "max_iters": MAX_ITERS,
        "results": results,
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

    receipts_dir = Path(__file__).parent.parent / "receipts" / "kmeans_convergence"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"kmeans_convergence_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
