#!/usr/bin/env python3
"""
bench_rvq_8x_poc.py — 4-bit VQ and Residual VQ proof-of-concept on real TinyLlama tensors.

Encodes ALL 154 linear tensors (22 blocks x 7 tensor types) with 4 strategies:
  a) 8-bit VQ (k=256) — current production baseline
  b) 4-bit VQ (k=16)  — coarse scalar quantization
  c) 4-bit Residual VQ (k=16+16) — coarse + residual, packed into 1 byte
  d) Mixed-rate — kurtosis > 5 → 8-bit, else → 4-bit residual

Reports per-strategy: mean/min/max cosine, compression ratio, blended ratio.
Writes receipt with cost block to receipts/rvq_8x_poc/.

Work Order: WO-RVQ-8X-01 Phase 1

Dependencies: numpy, safetensors (no sklearn, no scipy, no torch)

Usage:
    python tools/bench_rvq_8x_poc.py
"""

import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path.home() / "models" / "tinyllama_fp32"
MODEL_FILE = MODEL_DIR / "model.safetensors"
RECEIPT_DIR = Path(__file__).resolve().parent.parent / "receipts" / "rvq_8x_poc"

# ---------------------------------------------------------------------------
# HF tensor name patterns (22 layers x 7 types = 154 tensors)
# ---------------------------------------------------------------------------
HF_PATTERNS = {
    "q_proj":    "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj":    "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj":    "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj":    "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj":   "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}

N_LAYERS = 22
KURTOSIS_THRESHOLD = 5.0

# ---------------------------------------------------------------------------
# Manual k-means (1D, no sklearn)
# ---------------------------------------------------------------------------

def kmeans_1d(data: np.ndarray, k: int = 16, max_iters: int = 10,
              seed: int = 42) -> np.ndarray:
    """1D k-means with percentile initialization.

    Args:
        data: 1-D float32 array (sample, not full tensor).
        k: Number of centroids.
        max_iters: Iteration cap.
        seed: RNG seed (used only if data needs subsampling inside caller).

    Returns:
        centroids: float32 array of shape (k,).
    """
    centroids = np.percentile(data, np.linspace(0, 100, k)).astype(np.float32)
    for _ in range(max_iters):
        # Chunked distance to avoid huge temporaries
        assignments = assign_chunked(data, centroids, chunk_size=1_000_000)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                centroids[c] = data[mask].mean()
    return centroids


def assign_chunked(flat: np.ndarray, codebook: np.ndarray,
                   chunk_size: int = 1_000_000) -> np.ndarray:
    """Assign each element of flat to nearest codebook entry, in chunks.

    Avoids allocating (len(flat), k) all at once.
    """
    indices = np.empty(len(flat), dtype=np.uint8)
    for start in range(0, len(flat), chunk_size):
        end = min(start + chunk_size, len(flat))
        # (end-start, k) distance matrix
        dists = np.abs(flat[start:end, None] - codebook[None, :])
        indices[start:end] = np.argmin(dists, axis=1).astype(np.uint8)
    return indices


# ---------------------------------------------------------------------------
# Kurtosis (excess, Fisher definition — matches scipy kurtosis(fisher=True))
# ---------------------------------------------------------------------------

def excess_kurtosis(data: np.ndarray) -> float:
    """Compute excess kurtosis (Fisher) for a 1-D array."""
    n = len(data)
    if n < 4:
        return 0.0
    m = data.mean()
    s = data.std()
    if s < 1e-12:
        return 0.0
    z = (data - m) / s
    return float(np.mean(z ** 4) - 3.0)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat arrays."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))


# ---------------------------------------------------------------------------
# Sampling helper (for k-means fitting — subsample to keep fit fast)
# ---------------------------------------------------------------------------

def sample_flat(flat: np.ndarray, max_samples: int = 500_000,
                seed: int = 42) -> np.ndarray:
    """Subsample flat array for k-means fitting."""
    if len(flat) <= max_samples:
        return flat.copy()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(flat), max_samples, replace=False)
    return flat[idx]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def encode_8bit_vq(flat: np.ndarray) -> tuple:
    """8-bit VQ (k=256). Returns (recon, compressed_bytes, codebook_bytes)."""
    k = 256
    sample = sample_flat(flat)
    codebook = kmeans_1d(sample, k=k, max_iters=10)
    indices = assign_chunked(flat, codebook)
    recon = codebook[indices]
    # Storage: 1 byte per element + codebook
    compressed_bytes = len(flat) * 1 + k * 4
    return recon, compressed_bytes


def encode_4bit_vq(flat: np.ndarray) -> tuple:
    """4-bit VQ (k=16). Returns (recon, compressed_bytes)."""
    k = 16
    sample = sample_flat(flat)
    codebook = kmeans_1d(sample, k=k, max_iters=10)
    indices = assign_chunked(flat, codebook)
    recon = codebook[indices]
    # Pad to even length for nibble packing
    n = len(flat)
    packed_bytes = (n + 1) // 2  # ceil(n/2)
    compressed_bytes = packed_bytes + k * 4
    return recon, compressed_bytes


def encode_4bit_residual_vq(flat: np.ndarray) -> tuple:
    """4-bit Residual VQ (k=16+16). Returns (recon, compressed_bytes).

    Stage 1: coarse k=16 on original values.
    Stage 2: fine k=16 on residual (original - coarse_recon).
    Both indices packed into 1 byte (high nibble = coarse, low nibble = fine).
    """
    k = 16

    # --- Stage 1: coarse ---
    sample1 = sample_flat(flat)
    codebook1 = kmeans_1d(sample1, k=k, max_iters=10)
    indices1 = assign_chunked(flat, codebook1)
    coarse_recon = codebook1[indices1]

    # --- Stage 2: residual ---
    residual = flat - coarse_recon
    sample2 = sample_flat(residual)
    codebook2 = kmeans_1d(sample2, k=k, max_iters=10)
    indices2 = assign_chunked(residual, codebook2)
    fine_recon = codebook2[indices2]

    # --- Reconstruct ---
    recon = coarse_recon + fine_recon

    # --- Storage ---
    # Each element: 4 bits coarse + 4 bits fine = 1 byte per element.
    # Packed as: (coarse_idx << 4) | fine_idx
    # Same index storage as 8-bit VQ, but codebook is 2x16x4=128 bytes
    # vs 256x4=1024 bytes for 8-bit VQ. The point of this PoC is the
    # QUALITY comparison: does RVQ(16+16) match VQ(256) fidelity?

    n = len(flat)
    compressed_bytes = n * 1 + k * 4 * 2  # 1 byte/elem + 2 codebooks of 16 x f32
    return recon, compressed_bytes


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 70)
    print("  WO-RVQ-8X-01 Phase 1: 4-bit VQ / Residual VQ Proof-of-Concept")
    print(f"  Model: TinyLlama 1.1B ({MODEL_FILE})")
    print(f"  Tensors: {N_LAYERS} layers x {len(HF_PATTERNS)} types = "
          f"{N_LAYERS * len(HF_PATTERNS)} tensors")
    print(f"  Strategies: 8-bit VQ (k=256), 4-bit VQ (k=16), "
          f"4-bit RVQ (k=16+16), Mixed-rate")
    print(f"  Kurtosis threshold for mixed-rate: {KURTOSIS_THRESHOLD}")
    print(f"  Start: {start_iso}")
    print("=" * 70)

    # Verify model file exists
    if not MODEL_FILE.exists():
        print(f"\n  FATAL: model file not found: {MODEL_FILE}")
        sys.exit(1)

    sf = safe_open(str(MODEL_FILE), framework="numpy")

    # Strategy names
    STRATEGIES = ["vq8", "vq4", "rvq4", "mixed"]

    # Accumulators per strategy
    cosines = {s: [] for s in STRATEGIES}
    total_compressed = {s: 0 for s in STRATEGIES}
    total_original = 0

    # Per-tensor detail rows (for receipt)
    detail_rows = []

    n_high_kurtosis = 0
    n_total = 0

    for layer_idx in range(N_LAYERS):
        for ttype, pattern in HF_PATTERNS.items():
            hf_name = pattern.format(i=layer_idx)
            n_total += 1

            # Load tensor
            try:
                tensor = sf.get_tensor(hf_name).astype(np.float32)
            except Exception as e:
                print(f"  SKIP {hf_name}: {e}")
                continue

            rows, cols = tensor.shape
            flat = tensor.ravel()
            original_bytes = rows * cols * 4  # FP32
            total_original += original_bytes

            # Kurtosis
            kurt_sample = sample_flat(flat, max_samples=500_000)
            kurt = excess_kurtosis(kurt_sample)

            is_high_kurt = kurt > KURTOSIS_THRESHOLD
            if is_high_kurt:
                n_high_kurtosis += 1

            # --- Encode with all 4 strategies ---

            # (a) 8-bit VQ (k=256)
            recon_8, comp_8 = encode_8bit_vq(flat)
            cos_8 = cosine_sim(flat, recon_8)
            cosines["vq8"].append(cos_8)
            total_compressed["vq8"] += comp_8

            # (b) 4-bit VQ (k=16)
            recon_4, comp_4 = encode_4bit_vq(flat)
            cos_4 = cosine_sim(flat, recon_4)
            cosines["vq4"].append(cos_4)
            total_compressed["vq4"] += comp_4

            # (c) 4-bit Residual VQ (k=16+16)
            recon_r4, comp_r4 = encode_4bit_residual_vq(flat)
            cos_r4 = cosine_sim(flat, recon_r4)
            cosines["rvq4"].append(cos_r4)
            total_compressed["rvq4"] += comp_r4

            # (d) Mixed-rate: kurtosis > threshold → 8-bit, else → 4-bit RVQ
            if is_high_kurt:
                cosines["mixed"].append(cos_8)
                total_compressed["mixed"] += comp_8
            else:
                cosines["mixed"].append(cos_r4)
                total_compressed["mixed"] += comp_r4

            # Progress
            tag = "*" if is_high_kurt else " "
            print(f"  [{n_total:3d}/154] {tag} L{layer_idx:02d}.{ttype:<10s}  "
                  f"kurt={kurt:7.2f}  "
                  f"cos8={cos_8:.6f}  cos4={cos_4:.6f}  "
                  f"cosR4={cos_r4:.6f}  shape=({rows},{cols})")

            detail_rows.append({
                "layer": layer_idx,
                "type": ttype,
                "hf_name": hf_name,
                "shape": [rows, cols],
                "kurtosis": round(kurt, 4),
                "high_kurtosis": is_high_kurt,
                "cos_vq8": round(cos_8, 8),
                "cos_vq4": round(cos_4, 8),
                "cos_rvq4": round(cos_r4, 8),
            })

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    wall_time = time.time() - t_start
    cpu_time = time.process_time() - cpu_start
    end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    print("\n" + "=" * 70)
    print("  STRATEGY SUMMARY")
    print("=" * 70)

    strategy_labels = {
        "vq8":   "8-bit VQ (k=256)",
        "vq4":   "4-bit VQ (k=16)",
        "rvq4":  "4-bit RVQ (k=16+16)",
        "mixed": f"Mixed (kurt>{KURTOSIS_THRESHOLD}→8-bit, else→RVQ)",
    }

    summary = {}
    header = (f"  {'Strategy':<40s}  {'Mean cos':>10s}  {'Min cos':>10s}  "
              f"{'Max cos':>10s}  {'Ratio':>7s}")
    print(header)
    print("  " + "-" * 85)

    for s in STRATEGIES:
        arr = np.array(cosines[s])
        mean_cos = float(arr.mean())
        min_cos = float(arr.min())
        max_cos = float(arr.max())
        ratio = total_original / total_compressed[s] if total_compressed[s] > 0 else 0
        label = strategy_labels[s]
        print(f"  {label:<40s}  {mean_cos:>10.6f}  {min_cos:>10.6f}  "
              f"{max_cos:>10.6f}  {ratio:>7.2f}x")
        summary[s] = {
            "label": label,
            "mean_cosine": round(mean_cos, 8),
            "min_cosine": round(min_cos, 8),
            "max_cosine": round(max_cos, 8),
            "total_compressed_bytes": total_compressed[s],
            "compression_ratio": round(ratio, 4),
        }

    print(f"\n  Original FP32 total: {total_original / 1e9:.3f} GB")
    print(f"  High-kurtosis tensors (>{KURTOSIS_THRESHOLD}): "
          f"{n_high_kurtosis}/{n_total}")

    # -----------------------------------------------------------------------
    # Verdict: does 4-bit RVQ achieve >= 0.999 cosine on low-kurtosis tensors?
    # -----------------------------------------------------------------------
    low_kurt_rvq4_cos = [
        row["cos_rvq4"] for row in detail_rows if not row["high_kurtosis"]
    ]
    if low_kurt_rvq4_cos:
        low_kurt_min = min(low_kurt_rvq4_cos)
        low_kurt_mean = np.mean(low_kurt_rvq4_cos)
        all_above_999 = all(c >= 0.999 for c in low_kurt_rvq4_cos)
    else:
        low_kurt_min = 0.0
        low_kurt_mean = 0.0
        all_above_999 = False

    print(f"\n  VERDICT: 4-bit RVQ on low-kurtosis tensors "
          f"(n={len(low_kurt_rvq4_cos)}):")
    print(f"    Mean cosine: {low_kurt_mean:.6f}")
    print(f"    Min  cosine: {low_kurt_min:.6f}")
    if all_above_999:
        verdict = "PASS"
        print(f"    All >= 0.999: YES  → {verdict}")
    else:
        n_below = sum(1 for c in low_kurt_rvq4_cos if c < 0.999)
        verdict = "FAIL"
        print(f"    All >= 0.999: NO ({n_below} below threshold)  → {verdict}")

    # Also report high-kurtosis tensors separately
    high_kurt_rvq4_cos = [
        row["cos_rvq4"] for row in detail_rows if row["high_kurtosis"]
    ]
    if high_kurt_rvq4_cos:
        print(f"\n  High-kurtosis tensors (n={len(high_kurt_rvq4_cos)}):")
        print(f"    Mean RVQ4 cosine: {np.mean(high_kurt_rvq4_cos):.6f}")
        print(f"    Min  RVQ4 cosine: {min(high_kurt_rvq4_cos):.6f}")
        print(f"    These use 8-bit VQ in mixed-rate strategy.")

    # -----------------------------------------------------------------------
    # RVQ gain analysis: how much does the residual stage help?
    # -----------------------------------------------------------------------
    rvq4_gains = [
        row["cos_rvq4"] - row["cos_vq4"] for row in detail_rows
    ]
    print(f"\n  RVQ residual gain (cos_rvq4 - cos_vq4):")
    print(f"    Mean: {np.mean(rvq4_gains):.6f}")
    print(f"    Min:  {min(rvq4_gains):.6f}")
    print(f"    Max:  {max(rvq4_gains):.6f}")

    # -----------------------------------------------------------------------
    # Receipt
    # -----------------------------------------------------------------------
    peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    receipt = {
        "work_order": "WO-RVQ-8X-01",
        "phase": 1,
        "title": "4-bit VQ / Residual VQ Proof-of-Concept",
        "model": str(MODEL_FILE),
        "n_layers": N_LAYERS,
        "n_tensor_types": len(HF_PATTERNS),
        "n_tensors": n_total,
        "kurtosis_threshold": KURTOSIS_THRESHOLD,
        "n_high_kurtosis": n_high_kurtosis,
        "original_fp32_bytes": total_original,
        "summary": summary,
        "verdict": {
            "test": "4-bit RVQ cosine >= 0.999 on low-kurtosis tensors",
            "result": verdict,
            "n_low_kurtosis": len(low_kurt_rvq4_cos),
            "low_kurt_mean_cosine": round(float(low_kurt_mean), 8),
            "low_kurt_min_cosine": round(float(low_kurt_min), 8),
        },
        "rvq_gain": {
            "description": "cos_rvq4 - cos_vq4 (residual stage improvement)",
            "mean": round(float(np.mean(rvq4_gains)), 8),
            "min": round(float(min(rvq4_gains)), 8),
            "max": round(float(max(rvq4_gains)), 8),
        },
        "detail": detail_rows,
        "cost": {
            "wall_time_s": round(wall_time, 3),
            "cpu_time_s": round(cpu_time, 3),
            "peak_memory_mb": round(peak_mem_mb, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": end_iso,
        },
    }

    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"rvq_8x_poc_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time:.1f}s  CPU time: {cpu_time:.1f}s  "
          f"Peak mem: {peak_mem_mb:.0f} MB")
    print("=" * 70)

    return receipt


if __name__ == "__main__":
    main()
