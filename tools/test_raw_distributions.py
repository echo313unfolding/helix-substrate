#!/usr/bin/env python3
"""Test HXQ affine6 on raw tensor distributions outside ML.

Three distributions:
  1. Mixed Gaussian — simulates generic weight-like data
  2. Uniform — simulates embedding/quantization-aware data
  3. Heavy-tailed (Cauchy) — hardest case, outlier-dominated

For each: generate 4096x4096 tensor, run HXQ affine g128 6-bit,
measure cosine similarity and RMS error.

Receipt: ~/receipts/hxq_raw_distribution_test_<timestamp>.json
"""

import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

def affine_quantize_g128_6bit(tensor, group_size=128, n_levels=64):
    """Per-group affine quantization: 6-bit, group_size=128."""
    flat = tensor.reshape(-1)
    n = len(flat)
    # Pad to group_size multiple
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])

    groups = flat.reshape(-1, group_size)
    n_groups = len(groups)

    # Per-group min/max
    g_min = groups.min(axis=1, keepdims=True)
    g_max = groups.max(axis=1, keepdims=True)

    # Avoid division by zero
    g_range = g_max - g_min
    g_range = np.where(g_range == 0, 1.0, g_range)

    # Quantize to [0, n_levels-1]
    normalized = (groups - g_min) / g_range
    indices = np.clip(np.round(normalized * (n_levels - 1)), 0, n_levels - 1).astype(np.uint8)

    # Dequantize
    decoded = g_min + indices.astype(np.float32) * g_range / (n_levels - 1)

    # Trim padding
    decoded_flat = decoded.reshape(-1)[:n]
    return decoded_flat.reshape(tensor.shape)


def cosine_sim(a, b):
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def rms_error(a, b):
    diff = (a.astype(np.float64) - b.astype(np.float64))
    return float(np.sqrt(np.mean(diff ** 2)))


def test_distribution(name, tensor):
    """Run affine g128 6-bit on a tensor, return metrics."""
    t0 = time.time()
    decoded = affine_quantize_g128_6bit(tensor)
    elapsed = time.time() - t0

    cos = cosine_sim(tensor, decoded)
    rms = rms_error(tensor, decoded)

    # Stats about the input
    kurtosis = float(np.mean(((tensor - tensor.mean()) / max(tensor.std(), 1e-10)) ** 4))

    return {
        "distribution": name,
        "shape": list(tensor.shape),
        "numel": int(tensor.size),
        "input_stats": {
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "kurtosis": round(kurtosis, 2),
        },
        "cosine_similarity": round(cos, 6),
        "rms_error": round(rms, 6),
        "max_abs_error": round(float(np.max(np.abs(tensor - decoded))), 6),
        "gate": "PASS" if cos >= 0.998 else "FAIL",
        "time_ms": round(elapsed * 1000, 1),
    }


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    np.random.seed(42)
    size = (4096, 4096)

    results = []

    # 1. Mixed Gaussian — two modes, different scales
    print("Testing Mixed Gaussian (4096x4096)...")
    g1 = np.random.normal(loc=-0.5, scale=0.3, size=size).astype(np.float32)
    g2 = np.random.normal(loc=0.5, scale=0.8, size=size).astype(np.float32)
    mask = np.random.random(size) > 0.5
    mixed_gaussian = np.where(mask, g1, g2)
    r = test_distribution("mixed_gaussian", mixed_gaussian)
    results.append(r)
    print(f"  cos={r['cosine_similarity']}, rms={r['rms_error']}, gate={r['gate']}")

    # 2. Uniform [-1, 1]
    print("Testing Uniform [-1, 1] (4096x4096)...")
    uniform = np.random.uniform(-1.0, 1.0, size=size).astype(np.float32)
    r = test_distribution("uniform", uniform)
    results.append(r)
    print(f"  cos={r['cosine_similarity']}, rms={r['rms_error']}, gate={r['gate']}")

    # 3. Heavy-tailed (Cauchy) — standard Cauchy, clipped to [-100, 100]
    print("Testing Heavy-tailed Cauchy (4096x4096)...")
    cauchy = np.random.standard_cauchy(size=size).astype(np.float32)
    cauchy = np.clip(cauchy, -100, 100)  # Clip extreme outliers for numerical stability
    r = test_distribution("heavy_tailed_cauchy", cauchy)
    results.append(r)
    print(f"  cos={r['cosine_similarity']}, rms={r['rms_error']}, gate={r['gate']}")

    # 4. Bonus: Log-normal (skewed, positive-only — like activation magnitudes)
    print("Testing Log-normal (4096x4096)...")
    lognormal = np.random.lognormal(mean=0, sigma=1.0, size=size).astype(np.float32)
    r = test_distribution("lognormal", lognormal)
    results.append(r)
    print(f"  cos={r['cosine_similarity']}, rms={r['rms_error']}, gate={r['gate']}")

    # 5. Bonus: Sparse (90% zeros, 10% Gaussian) — like pruned weights
    print("Testing Sparse 90% (4096x4096)...")
    sparse = np.random.normal(0, 1, size=size).astype(np.float32)
    sparse_mask = np.random.random(size) > 0.1  # 90% zeros
    sparse[sparse_mask] = 0.0
    r = test_distribution("sparse_90pct", sparse)
    results.append(r)
    print(f"  cos={r['cosine_similarity']}, rms={r['rms_error']}, gate={r['gate']}")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Distribution':<25} {'Cosine':>10} {'RMS':>10} {'Gate':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['distribution']:<25} {r['cosine_similarity']:>10.6f} {r['rms_error']:>10.6f} {r['gate']:>6}")
    print("=" * 60)

    all_pass = all(r['gate'] == 'PASS' for r in results)
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    # Receipt
    receipt = {
        "receipt_id": f"hxq_raw_distribution_test_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}",
        "experiment": "hxq_affine6_raw_distribution_test",
        "description": "HXQ affine g128 6-bit on raw tensor distributions (non-ML)",
        "method": "per-group affine quantization, group_size=128, 6-bit (64 levels)",
        "tensor_size": list(size),
        "seed": 42,
        "results": results,
        "summary": {
            "n_distributions": len(results),
            "n_pass": sum(1 for r in results if r['gate'] == 'PASS'),
            "n_fail": sum(1 for r in results if r['gate'] == 'FAIL'),
            "all_pass": all_pass,
            "gate_threshold": 0.998,
        },
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
        },
    }

    receipt_dir = Path.home() / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    receipt_path = receipt_dir / f"hxq_raw_distribution_test_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\nReceipt: {receipt_path}")


if __name__ == "__main__":
    main()
