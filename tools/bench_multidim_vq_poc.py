#!/usr/bin/env python3
"""
WO-MULTIDIM-VQ Phase 1: Multi-Dimensional VQ POC on TinyLlama.

Tests scalar (1D), 2D, and 4D vector quantization with k=256 across
compressible TinyLlama tensors. Quality measured on 50K subsample per tensor
(representative; production uses same subsample for codebook fitting).

Gate: 2D VQ k=256 must achieve mean cosine >= 0.995.

Usage:
    python3 tools/bench_multidim_vq_poc.py
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
_SUBSAMPLE = 50_000  # Quality measured on this subsample
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


def _fit_and_eval(data_flat, d, k):
    """Fit k-means on subsample, measure cosine on subsample. Returns cosine."""
    n = len(data_flat)
    # Make divisible by d
    n_use = n - (n % d) if n % d != 0 else n
    flat = data_flat[:n_use]

    if d == 1:
        return _scalar_fit_eval(flat, k)
    else:
        vectors = flat.reshape(-1, d)
        return _vector_fit_eval(vectors, d, k)


def _scalar_fit_eval(flat, k):
    """Scalar VQ: percentile init k-means, eval on subsample."""
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
        new_c = np.zeros_like(centroids)
        for c in range(k):
            mask = labels == c
            new_c[c] = sample[mask].mean() if mask.any() else centroids[c]
        if np.max(np.abs(new_c - centroids)) < abs_tol:
            break
        centroids = new_c

    # Assign subsample and measure cosine
    dists = np.abs(sample[:, None] - centroids[None, :])
    indices = np.argmin(dists, axis=1).astype(np.uint8)
    recon = centroids[indices]
    return _cosine(sample, recon)


def _vector_fit_eval(vectors, d, k):
    """Vector VQ: k-means++ init, eval on subsample."""
    n = len(vectors)
    rng = np.random.RandomState(42)
    sub_n = min(n, _SUBSAMPLE)
    sample = vectors[rng.choice(n, sub_n, replace=False)] if n > sub_n else vectors.copy()

    # K-means++ init (first 32 full, rest random)
    centroids = np.zeros((k, d), dtype=np.float32)
    centroids[0] = sample[rng.randint(len(sample))]
    n_full = min(k, 32)
    for i in range(1, n_full):
        diffs = sample[:, None, :] - centroids[None, :i, :]
        min_dists = np.sum(diffs ** 2, axis=2).min(axis=1)
        probs = min_dists / max(min_dists.sum(), 1e-30)
        centroids[i] = sample[rng.choice(len(sample), p=probs)]
    if k > n_full:
        centroids[n_full:] = sample[rng.choice(len(sample), k - n_full, replace=False)]

    cb_range = float(centroids.max() - centroids.min())
    abs_tol = 0.001 * max(cb_range, 1e-30)

    # Iterate on subsample only
    for _ in range(_MAX_ITERS):
        diffs = sample[:, None, :] - centroids[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_c = np.zeros_like(centroids)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_c[c] = sample[mask].mean(axis=0)
            else:
                new_c[c] = centroids[c]
        if np.max(np.abs(new_c - centroids)) < abs_tol:
            break
        centroids = new_c

    # Assign subsample and measure cosine
    diffs = sample[:, None, :] - centroids[None, :, :]
    dists = np.sum(diffs ** 2, axis=2)
    indices = np.argmin(dists, axis=1)
    recon = centroids[indices]
    return _cosine(sample, recon)


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    from safetensors.torch import load_file as torch_load_file

    if not SAFETENSORS_PATH.exists():
        print(f"ERROR: {SAFETENSORS_PATH} not found", file=sys.stderr)
        sys.exit(1)

    print("=" * 80, flush=True)
    print("WO-MULTIDIM-VQ Phase 1: Multi-Dimensional VQ POC", flush=True)
    print("=" * 80, flush=True)

    print("Loading model...", flush=True)
    all_tensors = torch_load_file(str(SAFETENSORS_PATH), device="cpu")
    all_test_keys = sorted(k for k in all_tensors if _should_test(k, tuple(all_tensors[k].shape)))

    # Test all compressible tensors — feasible because we measure on subsample
    test_keys = all_test_keys
    print(f"\nTesting {len(test_keys)} compressible 2D tensors", flush=True)
    print(f"Quality measured on {_SUBSAMPLE:,} subsample per tensor", flush=True)

    configs = [
        ("scalar_256", 1, 256),
        ("vq2d_256",   2, 256),
        ("vq4d_256",   4, 256),
    ]

    print(f"\n{'Tensor':<50} {'Shape':<16} {'Scalar':<10} {'2D-256':<10} {'4D-256':<10}", flush=True)
    print("-" * 96, flush=True)

    results = []
    for idx, key in enumerate(test_keys):
        tensor = all_tensors[key].float().numpy()
        flat = tensor.ravel().astype(np.float32)
        n = len(flat)
        row = {"tensor": key, "shape": list(tensor.shape), "n_elements": n}

        cos_vals = []
        for cfg_name, d, k in configs:
            cos = _fit_and_eval(flat, d, k)
            cr = _ratio_bf16(n, d, k)
            row[f"{cfg_name}_cos"] = round(cos, 6)
            row[f"{cfg_name}_ratio_bf16"] = round(cr, 2)
            cos_vals.append(cos)

        short = key.replace("model.layers.", "L").replace(".self_attn.", ".attn.").replace(".weight", "")
        cos_str = " ".join(f"{c:<10.6f}" for c in cos_vals)
        print(f"{short:<50} {str(tensor.shape):<16} {cos_str}", flush=True)
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

    # ── Gate check ──
    vq2d_cos = [r["vq2d_256_cos"] for r in results]
    mean_2d = float(np.mean(vq2d_cos))
    min_2d = float(min(vq2d_cos))

    if mean_2d >= 0.995:
        gate = "PASS"
        detail = f"mean cosine {mean_2d:.6f} >= 0.995"
    elif min_2d >= 0.990:
        gate = "MARGINAL"
        detail = f"mean={mean_2d:.6f}, min={min_2d:.6f} — consider k=1024"
    else:
        gate = "FAIL"
        detail = f"mean={mean_2d:.6f}, min={min_2d:.6f} — quality too low"

    print(f"\n  GATE: 2D VQ k=256 mean cosine >= 0.995", flush=True)
    print(f"  Result: {gate} — {detail}", flush=True)

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
        "work_order": "WO-MULTIDIM-VQ-01-POC",
        "question": "Does multi-dim VQ maintain quality at higher compression than scalar VQ?",
        "gate": f"2D VQ k=256 mean cosine >= 0.995: {gate}",
        "verdict": gate,
        "summary": {
            "n_tensors": len(results),
            "subsample_size": _SUBSAMPLE,
            "scalar_256_cos_mean": round(float(np.mean([r["scalar_256_cos"] for r in results])), 6),
            "scalar_256_cos_min": round(float(min(r["scalar_256_cos"] for r in results)), 6),
            "vq2d_256_cos_mean": round(mean_2d, 6),
            "vq2d_256_cos_min": round(min_2d, 6),
            "vq4d_256_cos_mean": round(float(np.mean([r["vq4d_256_cos"] for r in results])), 6),
            "vq4d_256_cos_min": round(float(min(r["vq4d_256_cos"] for r in results)), 6),
        },
        "results": results,
        "cost": cost,
    }

    receipts_dir = Path(__file__).resolve().parent.parent / "receipts" / "multidim_vq_poc"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"multidim_vq_poc_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}", flush=True)
    print(f"Cost: {cost['wall_time_s']}s wall, {cost['cpu_time_s']}s CPU, {cost['peak_memory_mb']} MB peak", flush=True)


if __name__ == "__main__":
    main()
