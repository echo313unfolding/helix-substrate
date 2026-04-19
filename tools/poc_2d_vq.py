#!/usr/bin/env python3
"""
POC: 2D Vector Quantization vs Scalar VQ on TinyLlama.

Crystal vault → multi-dimensional VQ → GPTVQ's core technique.
Same k-means, same k=256, same uint8 indices.
Each index now covers TWO weights instead of one → 8x from FP32.

Usage:
    python3 tools/poc_2d_vq.py
"""

import time
import resource
import platform
import json
import numpy as np
from pathlib import Path

MODELS_DIR = Path.home() / "models"
TINYLLAMA_DIR = MODELS_DIR / "tinyllama-dense"
SAFETENSORS_PATH = TINYLLAMA_DIR / "model.safetensors"

# ── k-means (scalar, existing) ───────────────────────────────────────────

def scalar_kmeans(data_1d: np.ndarray, k: int = 256, max_iters: int = 15) -> tuple:
    """Standard scalar k-means (production path)."""
    n = min(500_000, len(data_1d))
    sample = data_1d[np.random.choice(len(data_1d), n, replace=False)] if len(data_1d) > n else data_1d

    percentiles = np.linspace(0, 100, k)
    centroids = np.percentile(sample, percentiles).astype(np.float32)

    for _ in range(max_iters):
        dists = np.abs(sample[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        new_c = np.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_c[i] = sample[mask].mean()
            else:
                new_c[i] = centroids[i]
        if np.max(np.abs(new_c - centroids)) < 1e-6:
            break
        centroids = new_c

    # Full assignment
    dists = np.abs(data_1d[:, None] - centroids[None, :])
    indices = np.argmin(dists, axis=1).astype(np.uint8)
    reconstructed = centroids[indices]
    return centroids, indices, reconstructed


# ── k-means (2D VQ) ──────────────────────────────────────────────────────

def vq2d_kmeans(data_2d: np.ndarray, k: int = 256, max_iters: int = 15) -> tuple:
    """2D vector quantization: k-means on pairs of weights.

    data_2d: shape (N/2, 2)
    Returns: (centroids [k,2], indices [N/2] uint8, reconstructed [N/2, 2])
    """
    n_sample = min(200_000, len(data_2d))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(data_2d), n_sample, replace=False) if len(data_2d) > n_sample else np.arange(len(data_2d))
    sample = data_2d[idx].astype(np.float32)

    # Initialize: grid on 2D percentiles (fast, deterministic)
    # 16x16 = 256 grid points covering the 2D distribution
    side = int(np.sqrt(k))  # 16 for k=256
    p1 = np.percentile(sample[:, 0], np.linspace(0, 100, side))
    p2 = np.percentile(sample[:, 1], np.linspace(0, 100, side))
    g1, g2 = np.meshgrid(p1, p2)
    centroids = np.stack([g1.ravel(), g2.ravel()], axis=1).astype(np.float32)  # (k, 2)

    # K-means iterations — chunked distance computation
    chunk_size = 50_000
    for it in range(max_iters):
        # Assign sample to nearest centroid (chunked)
        labels = np.zeros(len(sample), dtype=np.int32)
        for start in range(0, len(sample), chunk_size):
            end = min(start + chunk_size, len(sample))
            chunk = sample[start:end]
            # (chunk, 1, 2) - (1, k, 2) → (chunk, k, 2) → sum → (chunk, k)
            d0 = chunk[:, 0:1] - centroids[:, 0]  # (chunk, k)
            d1 = chunk[:, 1:2] - centroids[:, 1]  # (chunk, k)
            dists = d0 * d0 + d1 * d1              # (chunk, k) — no 3D intermediate
            labels[start:end] = np.argmin(dists, axis=1)

        # Update centroids
        new_c = np.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_c[i] = sample[mask].mean(axis=0)
            else:
                new_c[i] = centroids[i]

        max_shift = np.max(np.abs(new_c - centroids))
        centroids = new_c
        if max_shift < 1e-6:
            break

    # Full assignment (chunked)
    indices = np.zeros(len(data_2d), dtype=np.uint8)
    for start in range(0, len(data_2d), chunk_size):
        end = min(start + chunk_size, len(data_2d))
        chunk = data_2d[start:end].astype(np.float32)
        d0 = chunk[:, 0:1] - centroids[:, 0]
        d1 = chunk[:, 1:2] - centroids[:, 1]
        dists = d0 * d0 + d1 * d1
        indices[start:end] = np.argmin(dists, axis=1).astype(np.uint8)

    reconstructed = centroids[indices]
    return centroids, indices, reconstructed


# ── Metrics ───────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f, b_f = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    dot = np.dot(a_f, b_f)
    norm_a, norm_b = np.linalg.norm(a_f), np.linalg.norm(b_f)
    if norm_a < 1e-30 or norm_b < 1e-30:
        return 0.0
    return float(dot / (norm_a * norm_b))

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.flatten().astype(np.float64) - b.flatten().astype(np.float64)) ** 2))

def compression_ratio_scalar(shape, k=256):
    """Bytes: codebook (k*4) + indices (N*1) vs original (N*4)."""
    n = shape[0] * shape[1] if len(shape) > 1 else shape[0]
    compressed = k * 4 + n * 1  # codebook + uint8 indices
    original = n * 4  # FP32
    return original / compressed

def compression_ratio_2d(shape, k=256):
    """Bytes: codebook (k*2*4) + indices (N/2*1) vs original (N*4)."""
    n = shape[0] * shape[1] if len(shape) > 1 else shape[0]
    compressed = k * 2 * 4 + (n // 2) * 1  # 2D codebook + uint8 indices
    original = n * 4  # FP32
    return original / compressed


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    from safetensors.torch import load_file as torch_load_file
    import torch

    if not SAFETENSORS_PATH.exists():
        print(f"ERROR: {SAFETENSORS_PATH} not found")
        return

    print("=" * 72)
    print("POC: 2D VQ vs Scalar VQ on TinyLlama")
    print("=" * 72)

    # Load all tensors via torch (handles bfloat16)
    all_tensors = torch_load_file(str(SAFETENSORS_PATH), device="cpu")

    results = []
    keys = [k for k in all_tensors.keys() if any(p in k for p in ["q_proj.weight", "k_proj.weight", "v_proj.weight",
                                                          "o_proj.weight", "gate_proj.weight",
                                                          "up_proj.weight", "down_proj.weight"])]
    # Test on a representative subset (first 3 layers = 21 tensors)
    test_keys = [k for k in keys if any(f"layers.{i}." in k for i in range(3))]

    print(f"\nTesting {len(test_keys)} tensors from layers 0-2\n")
    print(f"{'Tensor':<55} {'Shape':<16} {'Scalar cos':<12} {'2D cos':<12} {'Scalar ratio':<14} {'2D ratio':<10}")
    print("-" * 119)

    for key in test_keys:
        tensor = all_tensors[key].float().numpy()
        if tensor.ndim != 2:
            continue

        flat = tensor.ravel()

        # Ensure even number of elements for 2D reshape
        if len(flat) % 2 != 0:
            flat = flat[:-1]  # drop last element

        pairs = flat.reshape(-1, 2)

        # Scalar VQ
        _, _, recon_s = scalar_kmeans(flat, k=256)
        cos_s = cosine_sim(flat, recon_s)
        ratio_s = compression_ratio_scalar(tensor.shape)

        # 2D VQ
        _, _, recon_2d = vq2d_kmeans(pairs, k=256)
        cos_2d = cosine_sim(pairs, recon_2d)
        ratio_2d = compression_ratio_2d(tensor.shape)

        short_name = key.replace("model.layers.", "L").replace(".self_attn.", ".attn.").replace(".mlp.", ".mlp.")
        results.append({
            "tensor": key,
            "shape": list(tensor.shape),
            "scalar_cos": round(cos_s, 6),
            "vq2d_cos": round(cos_2d, 6),
            "scalar_ratio": round(ratio_s, 2),
            "vq2d_ratio": round(ratio_2d, 2),
            "cos_delta": round(cos_2d - cos_s, 6),
        })

        print(f"{short_name:<55} {str(tensor.shape):<16} {cos_s:<12.6f} {cos_2d:<12.6f} {ratio_s:<14.2f} {ratio_2d:<10.2f}")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    cos_s_vals = [r["scalar_cos"] for r in results]
    cos_2d_vals = [r["vq2d_cos"] for r in results]
    deltas = [r["cos_delta"] for r in results]

    print(f"\nScalar VQ (k=256, 1D):")
    print(f"  Cosine: min={min(cos_s_vals):.6f}  mean={np.mean(cos_s_vals):.6f}  max={max(cos_s_vals):.6f}")
    print(f"  Ratio:  ~{results[0]['scalar_ratio']:.1f}x from FP32")

    print(f"\n2D VQ (k=256, pairs):")
    print(f"  Cosine: min={min(cos_2d_vals):.6f}  mean={np.mean(cos_2d_vals):.6f}  max={max(cos_2d_vals):.6f}")
    print(f"  Ratio:  ~{results[0]['vq2d_ratio']:.1f}x from FP32")

    print(f"\nDelta (2D - scalar):")
    print(f"  min={min(deltas):.6f}  mean={np.mean(deltas):.6f}  max={max(deltas):.6f}")

    # Verdict
    worst_2d = min(cos_2d_vals)
    if worst_2d >= 0.995:
        verdict = "PASS — 2D VQ quality sufficient for production"
    elif worst_2d >= 0.990:
        verdict = "MARGINAL — 2D VQ quality close, needs sidecar"
    else:
        verdict = "FAIL — 2D VQ quality too low without compensation"

    print(f"\nVerdict: {verdict}")

    # Cost block
    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    receipt = {
        "work_order": "WO-2D-VQ-01-POC",
        "question": "Does 2D VQ maintain quality at ~8x compression from FP32?",
        "verdict": verdict.split(" — ")[0],
        "results": results,
        "summary": {
            "n_tensors": len(results),
            "scalar_cos_mean": round(float(np.mean(cos_s_vals)), 6),
            "vq2d_cos_mean": round(float(np.mean(cos_2d_vals)), 6),
            "scalar_cos_min": round(float(min(cos_s_vals)), 6),
            "vq2d_cos_min": round(float(min(cos_2d_vals)), 6),
            "scalar_ratio": results[0]["scalar_ratio"],
            "vq2d_ratio": results[0]["vq2d_ratio"],
        },
        "cost": cost,
    }

    # Save receipt
    receipts_dir = Path("receipts/2d_vq_poc")
    receipts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%dT%H%M%S')
    receipt_path = receipts_dir / f"2d_vq_poc_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}")
    print(f"Cost: {cost['wall_time_s']}s wall, {cost['cpu_time_s']}s CPU, {cost['peak_memory_mb']} MB peak")


if __name__ == "__main__":
    main()
