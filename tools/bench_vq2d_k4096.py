#!/usr/bin/env python3
"""
WO-MULTIDIM-VQ Phase 1b: 2D VQ k=4096 focused test.

Tests 2D VQ with k=4096 (12 bits per 2 weights = 6 bits/weight = 2.67x from BF16).
This is the sweet spot hypothesis: bigger codebook captures pair correlations
while still beating scalar k=64 on compression ratio.

Uses matmul-based distance (||x-c||^2 = ||x||^2 + ||c||^2 - 2*x@c.T)
to avoid [N, 4096, d] broadcast intermediates.

Comparison configs:
  - scalar_256:  8 bits/weight, ~2x from BF16 (baseline)
  - vq2d_4096:  12 bits per 2 weights = 6 bits/weight, ~2.67x from BF16
  - scalar_64:   6 bits/weight, ~2.67x from BF16 (same ratio, no correlation)

Gate: 2D k=4096 must beat scalar k=64 at same bits/weight (correlation wins).

Usage:
    python3 tools/bench_vq2d_k4096.py
"""

import time
import resource
import platform
import json
import sys
import numpy as np
from pathlib import Path

TINYLLAMA_DIR = Path.home() / "models" / "tinyllama-dense"
SAFETENSORS_PATH = TINYLLAMA_DIR / "model.safetensors"

SKIP_PATTERNS = ["embed_tokens", "lm_head", "norm"]
_SUBSAMPLE = 50_000
_MAX_ITERS = 5


def _should_test(name, shape):
    if len(shape) != 2 or shape[0] * shape[1] < 256:
        return False
    return name.endswith(".weight") and not any(p in name for p in SKIP_PATTERNS)


def _cosine(a, b):
    a64, b64 = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.dot(a64, b64)
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    return float(d / (na * nb)) if na > 1e-30 and nb > 1e-30 else 0.0


def _ratio_bf16(n, d, k):
    idx_bytes = 1 if k <= 256 else 2
    return n * 2 / (k * d * 4 + (n // d) * idx_bytes)


def _sq_dists_matmul(X, C):
    """Squared L2 distances via matmul trick. X:[N,d], C:[k,d] -> [N,k].

    ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x@c.T
    Avoids [N, k, d] broadcast intermediate.
    """
    # [N, 1] and [1, k]
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # [N, 1]
    C_sq = np.sum(C ** 2, axis=1, keepdims=True).T  # [1, k]
    # [N, k] via BLAS matmul — fast
    XC = X @ C.T  # [N, k]
    return X_sq + C_sq - 2.0 * XC


def _chunked_sq_dists(X, C, chunk_size=10000):
    """Chunked squared distances for large N to limit memory."""
    N = len(X)
    k = len(C)
    C_sq = np.sum(C ** 2, axis=1, keepdims=True).T  # [1, k] — compute once
    labels = np.empty(N, dtype=np.int32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        X_chunk = X[start:end]
        X_sq = np.sum(X_chunk ** 2, axis=1, keepdims=True)
        dists = X_sq + C_sq - 2.0 * (X_chunk @ C.T)
        labels[start:end] = np.argmin(dists, axis=1)
    return labels


def _scalar_fit_eval(flat, k):
    """Scalar VQ: percentile init k-means."""
    n = len(flat)
    rng = np.random.RandomState(42)
    sub_n = min(n, _SUBSAMPLE)
    sample = flat[rng.choice(n, sub_n, replace=False)] if n > sub_n else flat.copy()

    pct = np.linspace(0, 100, k)
    centroids = np.percentile(sample, pct).astype(np.float32)
    cb_range = float(centroids[-1] - centroids[0])
    abs_tol = 0.001 * max(cb_range, 1e-30)

    for _ in range(_MAX_ITERS):
        dists = np.abs(sample[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        # Vectorized centroid update
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        sums = np.bincount(labels, weights=sample, minlength=k)
        new_c = np.where(counts > 0, sums / np.maximum(counts, 1), centroids)
        if np.max(np.abs(new_c - centroids)) < abs_tol:
            break
        centroids = new_c.astype(np.float32)

    dists = np.abs(sample[:, None] - centroids[None, :])
    indices = np.argmin(dists, axis=1)
    recon = centroids[indices]
    return _cosine(sample, recon)


def _vector_fit_eval(vectors, d, k):
    """Vector VQ with matmul-based distances for large k."""
    n = len(vectors)
    rng = np.random.RandomState(42)
    sub_n = min(n, _SUBSAMPLE)
    sample = vectors[rng.choice(n, sub_n, replace=False)].astype(np.float32) if n > sub_n else vectors.copy().astype(np.float32)

    # K-means++ init (first 64 full, rest random)
    centroids = np.zeros((k, d), dtype=np.float32)
    centroids[0] = sample[rng.randint(len(sample))]
    n_full = min(k, 64)
    for i in range(1, n_full):
        # Use matmul trick for init distances too
        dists = _sq_dists_matmul(sample, centroids[:i])  # [N, i]
        min_dists = dists.min(axis=1)
        probs = min_dists / max(min_dists.sum(), 1e-30)
        centroids[i] = sample[rng.choice(len(sample), p=probs)]
    if k > n_full:
        centroids[n_full:] = sample[rng.choice(len(sample), k - n_full, replace=False)]

    cb_range = float(centroids.max() - centroids.min())
    abs_tol = 0.001 * max(cb_range, 1e-30)

    # Iterate — use chunked distances + vectorized centroid update
    for it in range(_MAX_ITERS):
        labels = _chunked_sq_dists(sample, centroids)
        # Vectorized centroid update via bincount (avoids Python loop over k=4096)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        new_c = np.zeros_like(centroids)
        for dim in range(d):
            sums = np.bincount(labels, weights=sample[:, dim], minlength=k)
            new_c[:, dim] = np.where(counts > 0, sums / np.maximum(counts, 1), centroids[:, dim])
        if np.max(np.abs(new_c - centroids)) < abs_tol:
            break
        centroids = new_c

    # Final assignment and cosine
    labels = _chunked_sq_dists(sample, centroids)
    recon = centroids[labels]
    return _cosine(sample, recon)


def _fit_and_eval(data_flat, d, k):
    n = len(data_flat)
    n_use = n - (n % d) if n % d != 0 else n
    flat = data_flat[:n_use]
    if d == 1:
        return _scalar_fit_eval(flat, k)
    else:
        vectors = flat.reshape(-1, d)
        return _vector_fit_eval(vectors, d, k)


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    from safetensors.torch import load_file as torch_load_file

    if not SAFETENSORS_PATH.exists():
        print(f"ERROR: {SAFETENSORS_PATH} not found", file=sys.stderr)
        sys.exit(1)

    print("=" * 80, flush=True)
    print("WO-MULTIDIM-VQ Phase 1b: 2D VQ k=4096 Sweet Spot Test", flush=True)
    print("=" * 80, flush=True)

    print("Loading model...", flush=True)
    all_tensors = torch_load_file(str(SAFETENSORS_PATH), device="cpu")
    all_test_keys = sorted(k for k in all_tensors if _should_test(k, tuple(all_tensors[k].shape)))

    test_keys = all_test_keys
    print(f"\nTesting {len(test_keys)} compressible 2D tensors", flush=True)
    print(f"Quality measured on {_SUBSAMPLE:,} subsample per tensor", flush=True)

    # The key comparison: same bits/weight (6 bits) via different methods
    configs = [
        ("scalar_256",  1, 256),   # 8 bits/weight, 2x from BF16 (reference)
        ("scalar_64",   1, 64),    # 6 bits/weight, 2.67x from BF16
        ("vq2d_4096",   2, 4096),  # 6 bits/weight, 2.67x from BF16 (correlation capture)
    ]

    hdr = f"{'Tensor':<50} {'Shape':<16} {'Sc-256':<10} {'Sc-64':<10} {'2D-4096':<10}"
    print(f"\n{hdr}", flush=True)
    print("-" * len(hdr), flush=True)

    results = []
    for idx, key in enumerate(test_keys):
        tensor = all_tensors[key].float().numpy()
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        row = {"tensor": key, "shape": list(tensor.shape), "n_elements": n}

        t0 = time.time()
        cos_vals = []
        for cfg_name, d, k in configs:
            cos = _fit_and_eval(flat, d, k)
            cr = _ratio_bf16(n, d, k)
            row[f"{cfg_name}_cos"] = round(cos, 6)
            row[f"{cfg_name}_ratio_bf16"] = round(cr, 2)
            cos_vals.append(cos)

        dt = time.time() - t0
        short = key.replace("model.layers.", "L").replace(".self_attn.", ".attn.").replace(".weight", "")
        cos_str = " ".join(f"{c:<10.6f}" for c in cos_vals)
        print(f"{short:<50} {str(tensor.shape):<16} {cos_str} ({dt:.1f}s)", flush=True)
        results.append(row)

    # ── Summary ──
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 80, flush=True)

    for cfg_name, d, k in configs:
        cos_key = f"{cfg_name}_cos"
        all_cos = [r[cos_key] for r in results]
        ratio_key = f"{cfg_name}_ratio_bf16"
        ratios = [r[ratio_key] for r in results]
        print(f"\n  {cfg_name} (d={d}, k={k}):", flush=True)
        print(f"    Cosine: min={min(all_cos):.6f}  mean={np.mean(all_cos):.6f}  max={max(all_cos):.6f}", flush=True)
        print(f"    Ratio vs BF16: ~{np.mean(ratios):.1f}x", flush=True)

    # ── Gate: 2D k=4096 must beat scalar k=64 ──
    vq2d_cos = [r["vq2d_4096_cos"] for r in results]
    sc64_cos = [r["scalar_64_cos"] for r in results]
    mean_2d = float(np.mean(vq2d_cos))
    mean_sc64 = float(np.mean(sc64_cos))
    wins = sum(1 for v, s in zip(vq2d_cos, sc64_cos) if v > s)

    if mean_2d > mean_sc64:
        gate = "PASS"
        detail = f"2D k=4096 mean={mean_2d:.6f} > scalar k=64 mean={mean_sc64:.6f} ({wins}/{len(results)} tensor wins)"
    else:
        gate = "FAIL"
        detail = f"2D k=4096 mean={mean_2d:.6f} <= scalar k=64 mean={mean_sc64:.6f}"

    print(f"\n  GATE: 2D k=4096 beats scalar k=64 at same bits/weight", flush=True)
    print(f"  Result: {gate} — {detail}", flush=True)

    # Also check absolute quality
    if mean_2d >= 0.995:
        print(f"  BONUS: 2D k=4096 mean cosine {mean_2d:.6f} >= 0.995 (production quality)", flush=True)

    # ── Cost block ──
    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    receipt = {
        "work_order": "WO-MULTIDIM-VQ-01b-K4096",
        "question": "Does 2D VQ k=4096 beat scalar k=64 at same bits/weight (6 bits/weight, 2.67x from BF16)?",
        "gate": f"2D k=4096 > scalar k=64: {gate}",
        "verdict": gate,
        "summary": {
            "n_tensors": len(results),
            "subsample_size": _SUBSAMPLE,
            "scalar_256_cos_mean": round(float(np.mean([r["scalar_256_cos"] for r in results])), 6),
            "scalar_64_cos_mean": round(float(np.mean(sc64_cos)), 6),
            "scalar_64_cos_min": round(float(min(sc64_cos)), 6),
            "vq2d_4096_cos_mean": round(mean_2d, 6),
            "vq2d_4096_cos_min": round(float(min(vq2d_cos)), 6),
            "wins_2d_over_scalar": f"{wins}/{len(results)}",
        },
        "results": results,
        "cost": cost,
    }

    receipts_dir = Path(__file__).resolve().parent.parent / "receipts" / "multidim_vq_poc"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"vq2d_k4096_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}", flush=True)
    print(f"Cost: {cost['wall_time_s']}s wall, {cost['cpu_time_s']}s CPU, {cost['peak_memory_mb']} MB peak", flush=True)


if __name__ == "__main__":
    main()
