#!/usr/bin/env python3
"""
bench_pergroup_vq_poc.py — Per-group k=16 VQ: can grouping break the BF16 ratio ceiling?

The problem:
  - Global k=256 (VQ-256): +0.10% PPL, 1 byte/weight → 2x from BF16. Ceiling.
  - Global k=16: +9.3% PPL. FAILED quality gate.
  - Question: does LOCAL k=16 (per-row-group) pass quality?

If per-group k=16 hits cosine >= 0.999, that's 0.5 bytes/weight + small codebook overhead
→ ~3.5-4x from BF16. Ceiling broken.

Strategies tested (all on 154 TinyLlama tensors):
  1. vq256_global   — production baseline (k=256, global)
  2. vq16_global    — known failure (k=16, global)
  3. vq16_group128  — per-128-row-group k=16
  4. vq16_group32   — per-32-row-group k=16
  5. vq16_perrow    — per-row k=16 (group_size=1, maximum locality)
  6. rvq16_perrow   — per-row RVQ k=16+16 (maximum quality)

Each "group" fits an independent 16-entry codebook on its subset of rows.
Compression ratio includes codebook overhead.

Work Order: WO-RVQ-8X-01 Phase 3 (per-group codec research)

Usage:
    python tools/bench_pergroup_vq_poc.py
"""

import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path.home() / "models" / "tinyllama-dense"
MODEL_FILE = MODEL_DIR / "model.safetensors"
RECEIPT_DIR = Path(__file__).resolve().parent.parent / "receipts" / "pergroup_vq_poc"

# ---------------------------------------------------------------------------
# Tensor names (22 layers × 7 types = 154 tensors)
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

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def kmeans_1d(data: np.ndarray, k: int = 16, max_iters: int = 10) -> np.ndarray:
    """1D k-means with percentile initialization."""
    centroids = np.percentile(data, np.linspace(0, 100, k)).astype(np.float32)
    for _ in range(max_iters):
        assignments = assign_chunked(data, centroids)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                centroids[c] = data[mask].mean()
    return centroids


def assign_chunked(flat: np.ndarray, codebook: np.ndarray,
                   chunk_size: int = 1_000_000) -> np.ndarray:
    """Assign nearest codebook entry, chunked to avoid OOM."""
    indices = np.empty(len(flat), dtype=np.uint8)
    for start in range(0, len(flat), chunk_size):
        end = min(start + chunk_size, len(flat))
        dists = np.abs(flat[start:end, None] - codebook[None, :])
        indices[start:end] = np.argmin(dists, axis=1).astype(np.uint8)
    return indices


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))


def sample_flat(flat: np.ndarray, max_samples: int = 500_000,
                seed: int = 42) -> np.ndarray:
    if len(flat) <= max_samples:
        return flat.copy()
    rng = np.random.RandomState(seed)
    return flat[rng.choice(len(flat), max_samples, replace=False)]


def excess_kurtosis(data: np.ndarray) -> float:
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
# Strategy implementations
# ---------------------------------------------------------------------------

def encode_global_vq(flat: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Global VQ with k centroids. Returns (recon, compressed_bytes)."""
    sample = sample_flat(flat)
    codebook = kmeans_1d(sample, k=k, max_iters=10)
    indices = assign_chunked(flat, codebook)
    recon = codebook[indices]
    if k <= 16:
        index_bytes = (len(flat) + 1) // 2  # nibble packed
    else:
        index_bytes = len(flat)  # 1 byte each
    compressed_bytes = index_bytes + k * 4
    return recon, compressed_bytes


def encode_pergroup_vq(tensor_2d: np.ndarray, k: int = 16,
                       group_size: int = 128) -> tuple[np.ndarray, int]:
    """Per-row-group VQ. Each group of `group_size` rows gets its own codebook.

    Args:
        tensor_2d: [rows, cols] float32
        k: centroids per group (16 for 4-bit)
        group_size: number of rows per group (1 = per-row)

    Returns:
        (recon_2d, compressed_bytes)
    """
    rows, cols = tensor_2d.shape
    recon = np.empty_like(tensor_2d)
    n_groups = 0

    for g_start in range(0, rows, group_size):
        g_end = min(g_start + group_size, rows)
        group = tensor_2d[g_start:g_end]
        flat = group.ravel()

        # Fit codebook on this group
        if len(flat) > 500_000:
            sample = sample_flat(flat)
        else:
            sample = flat
        codebook = kmeans_1d(sample, k=k, max_iters=10)
        indices = assign_chunked(flat, codebook)
        recon[g_start:g_end] = codebook[indices].reshape(g_end - g_start, cols)
        n_groups += 1

    total_elements = rows * cols
    index_bytes = (total_elements + 1) // 2  # nibble packed (k=16 → 4 bits)
    codebook_bytes = n_groups * k * 4  # one codebook per group
    compressed_bytes = index_bytes + codebook_bytes
    return recon, compressed_bytes


def encode_pergroup_rvq(tensor_2d: np.ndarray, k: int = 16,
                        group_size: int = 1) -> tuple[np.ndarray, int]:
    """Per-row-group Residual VQ (k+k). Two stages, each with per-group codebook.

    Returns:
        (recon_2d, compressed_bytes)
    """
    rows, cols = tensor_2d.shape
    recon = np.empty_like(tensor_2d)
    n_groups = 0

    for g_start in range(0, rows, group_size):
        g_end = min(g_start + group_size, rows)
        group = tensor_2d[g_start:g_end]
        flat = group.ravel()

        # Stage 1: coarse
        sample1 = sample_flat(flat) if len(flat) > 500_000 else flat
        codebook1 = kmeans_1d(sample1, k=k, max_iters=10)
        indices1 = assign_chunked(flat, codebook1)
        coarse = codebook1[indices1]

        # Stage 2: residual
        residual = flat - coarse
        sample2 = sample_flat(residual) if len(residual) > 500_000 else residual
        codebook2 = kmeans_1d(sample2, k=k, max_iters=10)
        indices2 = assign_chunked(residual, codebook2)
        fine = codebook2[indices2]

        recon[g_start:g_end] = (coarse + fine).reshape(g_end - g_start, cols)
        n_groups += 1

    total_elements = rows * cols
    # RVQ: 4 bits coarse + 4 bits residual = 1 byte per element
    index_bytes = total_elements
    codebook_bytes = n_groups * 2 * k * 4  # two codebooks per group
    compressed_bytes = index_bytes + codebook_bytes
    return recon, compressed_bytes


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

STRATEGIES = [
    ("vq256_global",   "k=256 global (production baseline)"),
    ("vq16_global",    "k=16 global (known failure)"),
    ("vq16_group128",  "k=16 per-128-row group"),
    ("vq16_group32",   "k=16 per-32-row group"),
    ("vq16_perrow",    "k=16 per-row (group=1)"),
    ("rvq16_perrow",   "RVQ k=16+16 per-row (max quality)"),
]


def run_strategy(name: str, tensor_2d: np.ndarray) -> tuple[float, int]:
    """Run one strategy on a tensor. Returns (cosine, compressed_bytes)."""
    flat = tensor_2d.ravel()

    if name == "vq256_global":
        recon, cb = encode_global_vq(flat, k=256)
    elif name == "vq16_global":
        recon, cb = encode_global_vq(flat, k=16)
    elif name == "vq16_group128":
        recon_2d, cb = encode_pergroup_vq(tensor_2d, k=16, group_size=128)
        recon = recon_2d.ravel()
    elif name == "vq16_group32":
        recon_2d, cb = encode_pergroup_vq(tensor_2d, k=16, group_size=32)
        recon = recon_2d.ravel()
    elif name == "vq16_perrow":
        recon_2d, cb = encode_pergroup_vq(tensor_2d, k=16, group_size=1)
        recon = recon_2d.ravel()
    elif name == "rvq16_perrow":
        recon_2d, cb = encode_pergroup_rvq(tensor_2d, k=16, group_size=1)
        recon = recon_2d.ravel()
    else:
        raise ValueError(f"Unknown strategy: {name}")

    cos = cosine_sim(flat, recon)
    return cos, cb


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 78)
    print("  Per-Group k=16 VQ: Breaking the BF16 Ratio Ceiling?")
    print(f"  Model: TinyLlama 1.1B ({MODEL_FILE})")
    print(f"  Tensors: {N_LAYERS} × {len(HF_PATTERNS)} = {N_LAYERS * len(HF_PATTERNS)}")
    print(f"  Strategies: {len(STRATEGIES)}")
    print(f"  Start: {start_iso}")
    print("=" * 78)

    if not MODEL_FILE.exists():
        print(f"\n  FATAL: model file not found: {MODEL_FILE}")
        sys.exit(1)

    # Use torch for loading (handles BF16 → float32 conversion)
    from safetensors.torch import load_file as _torch_load
    import torch
    _all_tensors = _torch_load(str(MODEL_FILE))

    # Accumulators
    cosines = {s[0]: [] for s in STRATEGIES}
    total_compressed = {s[0]: 0 for s in STRATEGIES}
    total_original_fp32 = 0
    total_original_bf16 = 0
    detail_rows = []
    n_total = 0

    for layer_idx in range(N_LAYERS):
        for ttype, pattern in HF_PATTERNS.items():
            hf_name = pattern.format(i=layer_idx)
            n_total += 1

            if hf_name not in _all_tensors:
                print(f"  SKIP {hf_name}: not found")
                continue
            tensor = _all_tensors[hf_name].float().numpy()

            rows, cols = tensor.shape
            fp32_bytes = rows * cols * 4
            bf16_bytes = rows * cols * 2
            total_original_fp32 += fp32_bytes
            total_original_bf16 += bf16_bytes

            kurt_sample = sample_flat(tensor.ravel(), max_samples=500_000)
            kurt = excess_kurtosis(kurt_sample)

            row_data = {
                "layer": layer_idx,
                "type": ttype,
                "hf_name": hf_name,
                "shape": [rows, cols],
                "kurtosis": round(kurt, 4),
            }

            # Run all strategies
            cos_strs = []
            for sname, _ in STRATEGIES:
                cos, cb = run_strategy(sname, tensor)
                cosines[sname].append(cos)
                total_compressed[sname] += cb
                row_data[f"cos_{sname}"] = round(cos, 8)
                cos_strs.append(f"{cos:.6f}")

            detail_rows.append(row_data)

            # Progress: show per-row vs global k=16 improvement
            cos_global16 = row_data["cos_vq16_global"]
            cos_perrow = row_data["cos_vq16_perrow"]
            gain = cos_perrow - cos_global16
            print(f"  [{n_total:3d}/154] L{layer_idx:02d}.{ttype:<10s}  "
                  f"k=16g={cos_global16:.6f}  "
                  f"k=16pr={cos_perrow:.6f}  "
                  f"gain={gain:+.6f}  "
                  f"k=256={row_data['cos_vq256_global']:.6f}  "
                  f"kurt={kurt:7.2f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    wall_time = time.time() - t_start
    cpu_time = time.process_time() - cpu_start
    end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    print("\n" + "=" * 78)
    print("  STRATEGY SUMMARY")
    print("=" * 78)

    header = (f"  {'Strategy':<35s}  {'Mean cos':>10s}  {'Min cos':>10s}  "
              f"{'FP32 ratio':>10s}  {'BF16 ratio':>10s}")
    print(header)
    print("  " + "-" * 80)

    summary = {}
    for sname, slabel in STRATEGIES:
        arr = np.array(cosines[sname])
        mean_cos = float(arr.mean())
        min_cos = float(arr.min())
        ratio_fp32 = total_original_fp32 / max(1, total_compressed[sname])
        ratio_bf16 = total_original_bf16 / max(1, total_compressed[sname])
        print(f"  {slabel:<35s}  {mean_cos:>10.6f}  {min_cos:>10.6f}  "
              f"{ratio_fp32:>9.2f}x  {ratio_bf16:>9.2f}x")
        summary[sname] = {
            "label": slabel,
            "mean_cosine": round(mean_cos, 8),
            "min_cosine": round(min_cos, 8),
            "total_compressed_bytes": total_compressed[sname],
            "ratio_from_fp32": round(ratio_fp32, 4),
            "ratio_from_bf16": round(ratio_bf16, 4),
        }

    print(f"\n  Original FP32: {total_original_fp32 / 1e9:.3f} GB")
    print(f"  Original BF16: {total_original_bf16 / 1e9:.3f} GB")

    # -----------------------------------------------------------------------
    # Verdict: does per-row k=16 pass the quality gate?
    # -----------------------------------------------------------------------
    for sname in ["vq16_perrow", "vq16_group32", "vq16_group128"]:
        arr = np.array(cosines[sname])
        n_below_999 = int((arr < 0.999).sum())
        n_below_9999 = int((arr < 0.9999).sum())
        label = dict(STRATEGIES)[sname]
        print(f"\n  VERDICT [{label}]:")
        print(f"    Mean cosine: {arr.mean():.6f}")
        print(f"    Min  cosine: {arr.min():.6f}")
        print(f"    Below 0.999:  {n_below_999}/154")
        print(f"    Below 0.9999: {n_below_9999}/154")
        if n_below_999 == 0:
            print(f"    → PASS (all >= 0.999)")
        else:
            print(f"    → {n_below_999} tensors below 0.999 threshold")

    # Compare: how much does per-row improve over global k=16?
    gains = np.array(cosines["vq16_perrow"]) - np.array(cosines["vq16_global"])
    print(f"\n  Per-row k=16 gain over global k=16:")
    print(f"    Mean gain: {gains.mean():.6f}")
    print(f"    Min  gain: {gains.min():.6f}")
    print(f"    Max  gain: {gains.max():.6f}")

    # Compare per-row k=16 vs production k=256
    vs256 = np.array(cosines["vq16_perrow"]) - np.array(cosines["vq256_global"])
    print(f"\n  Per-row k=16 vs production k=256:")
    print(f"    Mean delta: {vs256.mean():+.6f}")
    print(f"    Min  delta: {vs256.min():+.6f}")
    print(f"    Max  delta: {vs256.max():+.6f}")
    n_better = int((vs256 > 0).sum())
    print(f"    Per-row k=16 beats k=256: {n_better}/154 tensors")

    # -----------------------------------------------------------------------
    # Receipt
    # -----------------------------------------------------------------------
    peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    receipt = {
        "work_order": "WO-RVQ-8X-01",
        "phase": "3_pergroup_codec_research",
        "title": "Per-Group k=16 VQ: Breaking the BF16 Ratio Ceiling",
        "question": "Does per-row-group k=16 pass quality gate (cosine >= 0.999)?",
        "model": str(MODEL_FILE),
        "n_layers": N_LAYERS,
        "n_tensors": n_total,
        "original_fp32_bytes": total_original_fp32,
        "original_bf16_bytes": total_original_bf16,
        "summary": summary,
        "perrow_vs_global16_gain": {
            "mean": round(float(gains.mean()), 8),
            "min": round(float(gains.min()), 8),
            "max": round(float(gains.max()), 8),
        },
        "perrow_vs_production256": {
            "mean_delta": round(float(vs256.mean()), 8),
            "min_delta": round(float(vs256.min()), 8),
            "n_perrow_wins": n_better,
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
    receipt_path = RECEIPT_DIR / f"pergroup_vq_poc_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time:.1f}s  CPU: {cpu_time:.1f}s  Peak mem: {peak_mem_mb:.0f} MB")
    print("=" * 78)

    return receipt


if __name__ == "__main__":
    main()
