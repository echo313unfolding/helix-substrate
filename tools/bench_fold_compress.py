#!/usr/bin/env python3
"""
WO-37: Fold-and-compress proof for 1D/3D tensors.

Tests whether stacking same-type 1D parameters across layers into a 2D matrix
produces compressible structure for VQ-256. The biological metaphor: take a long
strand of small signals and fold it into a structure with spatial regularity.

This bench validates the TECHNIQUE on Zamba2-1.2B locally so we know it works
before spending cloud GPU money on Zamba2-7B (where 573 exact tensors drag
ratio from ~4x to 1.88x).

What it does:
    1. Load all exact (1D, 3D) tensors from a compressed model
    2. Group by parameter type across layers
    3. Stack each group into a 2D matrix (the "fold")
    4. Run VQ-256 + optional sidecar on each folded matrix
    5. Measure: cosine similarity, MSE, kurtosis, compression ratio
    6. Project savings at 7B scale (81 layers, hidden=3584)

Usage:
    python3 tools/bench_fold_compress.py
    python3 tools/bench_fold_compress.py --model-dir ~/models/zamba2-1.2b-helix
    python3 tools/bench_fold_compress.py --k 64    # test smaller codebook

Work Order: WO-FOLD-COMPRESS-PROOF-01
"""

import argparse
import json
import platform
import re
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from safetensors import safe_open

REPO_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# VQ-256 compression (same algorithm as compress.py, standalone)
# ---------------------------------------------------------------------------

def vq_compress(matrix: np.ndarray, k: int = 256, max_sidecar: int = 512,
                sidecar_enabled: bool = True) -> dict:
    """
    VQ compress a 2D matrix. Returns codebook, indices, sidecar, and metrics.

    Same algorithm as CDNA v3:
    - Flatten to 1D, k-means with k centroids on scalar values
    - Each position gets uint8 index (k<=256) or uint16 (k>256)
    - Sidecar: top-N outlier corrections (exact values at positions with highest error)
    """
    from sklearn.cluster import MiniBatchKMeans

    rows, cols = matrix.shape
    flat = matrix.flatten().astype(np.float32)

    # K-means on scalar values
    kmeans = MiniBatchKMeans(
        n_clusters=k, random_state=42, batch_size=min(10000, len(flat)),
        max_iter=100, n_init=3,
    )
    kmeans.fit(flat.reshape(-1, 1))
    codebook = kmeans.cluster_centers_.flatten().astype(np.float32)
    indices = kmeans.predict(flat.reshape(-1, 1)).astype(np.uint8 if k <= 256 else np.uint16)

    # Reconstruct
    reconstructed = codebook[indices].reshape(rows, cols)

    # Sidecar: fix top-N worst errors
    sidecar_positions = np.array([], dtype=np.int64)
    sidecar_values = np.array([], dtype=np.float32)
    if sidecar_enabled and max_sidecar > 0:
        errors = np.abs(flat - codebook[indices])
        n_fix = min(max_sidecar, len(flat))
        worst_idx = np.argpartition(errors, -n_fix)[-n_fix:]
        # Only keep positions where error is significant
        threshold = np.percentile(errors, 99)
        worst_idx = worst_idx[errors[worst_idx] >= threshold]
        if len(worst_idx) > max_sidecar:
            worst_idx = worst_idx[np.argpartition(errors[worst_idx], -max_sidecar)[-max_sidecar:]]

        sidecar_positions = worst_idx.astype(np.int64)
        sidecar_values = flat[worst_idx].astype(np.float32)

        # Apply sidecar to reconstruction
        reconstructed_flat = reconstructed.flatten()
        reconstructed_flat[sidecar_positions] = sidecar_values
        reconstructed = reconstructed_flat.reshape(rows, cols)

    # Metrics
    cosine = float(np.dot(flat, reconstructed.flatten()) /
                   (np.linalg.norm(flat) * np.linalg.norm(reconstructed.flatten()) + 1e-12))
    mse = float(np.mean((matrix - reconstructed) ** 2))
    max_err = float(np.max(np.abs(matrix - reconstructed)))

    # Size calculation
    dense_bytes = rows * cols * 4  # FP32
    index_bytes = rows * cols * (1 if k <= 256 else 2)
    codebook_bytes = k * 4
    sidecar_bytes = len(sidecar_positions) * (8 + 4)  # int64 pos + float32 val
    compressed_bytes = index_bytes + codebook_bytes + sidecar_bytes
    ratio = dense_bytes / compressed_bytes if compressed_bytes > 0 else float('inf')

    return {
        "codebook": codebook,
        "indices": indices.reshape(rows, cols),
        "reconstructed": reconstructed,
        "sidecar_positions": sidecar_positions,
        "sidecar_values": sidecar_values,
        "cosine": round(cosine, 6),
        "mse": round(mse, 8),
        "max_error": round(max_err, 6),
        "dense_bytes": dense_bytes,
        "compressed_bytes": compressed_bytes,
        "ratio": round(ratio, 2),
        "n_sidecar": len(sidecar_positions),
    }


# ---------------------------------------------------------------------------
# Folding logic
# ---------------------------------------------------------------------------

def extract_foldable_groups(model_path: Path) -> dict:
    """
    Extract all non-artifact tensors from safetensors, group by parameter type.

    Returns dict of {param_type: [(key, tensor_np), ...]}.
    """
    f = safe_open(str(model_path), framework='numpy')
    artifact_suffixes = ('.codebook', '.indices', '.sidecar_positions',
                         '.sidecar_values', '.svd_U', '.svd_s', '.svd_Vt')

    groups = defaultdict(list)
    for key in sorted(f.keys()):
        if any(key.endswith(s) for s in artifact_suffixes):
            continue

        tensor = f.get_tensor(key)

        # Only 1D and 3D are foldable (2D are already compressed or exact embeddings)
        if tensor.ndim == 2:
            continue  # embeddings, already handled

        if tensor.ndim == 1 or tensor.ndim == 3:
            # Normalize to type key: strip layer index
            param_type = re.sub(r'model\.layers\.\d+\.', 'LAYER.', key)
            param_type = re.sub(r'^model\.', '', param_type)
            groups[param_type].append((key, tensor.astype(np.float32)))

    return dict(groups)


def fold_group(items: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    """
    Stack tensors of the same type into a 2D matrix.

    For 1D tensors: stack rows → (n_layers, dim)
    For 3D tensors: reshape each to 2D first, then stack
    """
    names = [name for name, _ in items]
    tensors = [t for _, t in items]

    if tensors[0].ndim == 1:
        # Simple stack: each tensor is one row
        matrix = np.stack(tensors, axis=0)  # (n_layers, dim)
    elif tensors[0].ndim == 3:
        # Reshape each 3D to 2D: (out, 1, kernel) → (out, kernel)
        reshaped = [t.reshape(t.shape[0], -1) for t in tensors]
        matrix = np.stack(reshaped, axis=0)  # (n_layers, out, kernel)
        # Fold to 2D: (n_layers, out * kernel)
        matrix = matrix.reshape(len(tensors), -1)
    else:
        raise ValueError(f"Unexpected ndim={tensors[0].ndim}")

    return matrix, names


# ---------------------------------------------------------------------------
# Projection to 7B scale
# ---------------------------------------------------------------------------

def project_7b_savings(results: list[dict], scale_factor: float = 2.13) -> dict:
    """
    Project savings to Zamba2-7B scale.

    Zamba2-7B: 81 layers, hidden=3584 (vs 1.2B: 38 layers, hidden=2048)
    Layer ratio: 81/38 = 2.13
    Dimension ratio: 3584/2048 = 1.75
    Combined scale: ~3.7x more 1D params
    """
    total_dense_1d = 0
    total_compressed_1d = 0

    for r in results:
        # Scale: more layers AND wider dimensions
        layer_scale = 81.0 / max(r["n_layers"], 1)
        # Dimension scales with hidden_size for norms/biases
        dim_name = r["group"]
        if "norm" in dim_name or "layernorm" in dim_name:
            dim_scale = 3584.0 / 2048.0
        elif "A_log" in dim_name or ".D" in dim_name or "dt_bias" in dim_name:
            dim_scale = 128.0 / 64.0  # Mamba state dim scales
        elif "conv1d" in dim_name:
            dim_scale = 7168.0 / 4352.0  # expanded state dim
        else:
            dim_scale = 3584.0 / 2048.0  # default

        projected_dense = r["dense_bytes"] * layer_scale * dim_scale
        projected_compressed = r["compressed_bytes"] * layer_scale * dim_scale
        total_dense_1d += projected_dense
        total_compressed_1d += projected_compressed

    savings_mb = (total_dense_1d - total_compressed_1d) / 1024 / 1024
    return {
        "projected_dense_mb": round(total_dense_1d / 1024 / 1024, 2),
        "projected_compressed_mb": round(total_compressed_1d / 1024 / 1024, 2),
        "projected_savings_mb": round(savings_mb, 2),
        "note": "81 layers, hidden=3584, mamba_state=128, expanded=7168",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fold-and-compress proof: VQ-256 on stacked 1D tensors across layers")
    parser.add_argument("--model-dir", type=Path,
                        default=Path.home() / "models" / "zamba2-1.2b-helix",
                        help="Model directory with model.safetensors")
    parser.add_argument("--k", type=int, default=256,
                        help="Codebook size (default: 256)")
    parser.add_argument("--max-sidecar", type=int, default=512,
                        help="Max sidecar corrections per group (default: 512)")
    parser.add_argument("--no-sidecar", action="store_true",
                        help="Disable sidecar corrections")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    model_path = args.model_dir / "model.safetensors"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  FOLD-AND-COMPRESS PROOF — WO-37")
    print(f"  Model: {args.model_dir}")
    print(f"  k={args.k}, sidecar={'off' if args.no_sidecar else f'max={args.max_sidecar}'}")
    print(f"{'='*70}\n")

    # ── Extract and group ──
    print("Extracting foldable tensor groups...", flush=True)
    groups = extract_foldable_groups(model_path)

    print(f"Found {len(groups)} groups:\n")
    for ptype, items in sorted(groups.items()):
        shapes = set(tuple(t.shape) for _, t in items)
        total_kb = sum(t.nbytes for _, t in items) / 1024
        print(f"  {ptype:55s} {len(items):3d} tensors, shapes={shapes}, {total_kb:.1f} KB")

    # ── Fold and compress each group ──
    print(f"\n{'─'*70}")
    print(f"  {'Group':45s} {'Shape':15s} {'Cosine':>8s} {'MSE':>12s} {'Ratio':>6s} {'Kurt':>8s}")
    print(f"{'─'*70}")

    results = []
    for ptype in sorted(groups):
        items = groups[ptype]
        if len(items) < 2:
            # Can't fold a single tensor — skip (e.g., final_layernorm)
            print(f"  {ptype:45s} {'skip (n=1)':>15s}")
            continue

        # Check all same shape
        shapes = set(tuple(t.shape) for _, t in items)
        if len(shapes) > 1:
            print(f"  {ptype:45s} {'skip (mixed shapes)':>15s}")
            continue

        # Fold
        matrix, names = fold_group(items)
        shape_str = f"{matrix.shape[0]}×{matrix.shape[1]}"

        # Kurtosis of the folded matrix (Fisher's, excess)
        flat_vals = matrix.flatten()
        kurtosis = float(scipy_stats.kurtosis(flat_vals, fisher=True))

        # VQ compress
        result = vq_compress(
            matrix, k=args.k,
            max_sidecar=args.max_sidecar if not args.no_sidecar else 0,
            sidecar_enabled=not args.no_sidecar,
        )

        print(f"  {ptype:45s} {shape_str:15s} {result['cosine']:8.6f} "
              f"{result['mse']:12.8f} {result['ratio']:5.1f}x {kurtosis:8.2f}")

        results.append({
            "group": ptype,
            "n_layers": matrix.shape[0],
            "dim": matrix.shape[1],
            "shape": list(matrix.shape),
            "kurtosis": round(kurtosis, 3),
            "cosine": result["cosine"],
            "mse": result["mse"],
            "max_error": result["max_error"],
            "ratio": result["ratio"],
            "n_sidecar": result["n_sidecar"],
            "dense_bytes": result["dense_bytes"],
            "compressed_bytes": result["compressed_bytes"],
            "tensor_names": names,
        })

    # ── Summary ──
    total_dense = sum(r["dense_bytes"] for r in results)
    total_compressed = sum(r["compressed_bytes"] for r in results)
    overall_ratio = total_dense / total_compressed if total_compressed > 0 else 0
    savings_kb = (total_dense - total_compressed) / 1024
    min_cosine = min(r["cosine"] for r in results) if results else 0
    mean_cosine = np.mean([r["cosine"] for r in results]) if results else 0

    print(f"\n{'='*70}")
    print(f"  FOLD-COMPRESS SUMMARY")
    print(f"{'='*70}")
    print(f"  Groups compressed:   {len(results)}")
    print(f"  Total dense (1D):    {total_dense/1024:.1f} KB")
    print(f"  Total compressed:    {total_compressed/1024:.1f} KB")
    print(f"  Overall ratio:       {overall_ratio:.2f}x")
    print(f"  Savings:             {savings_kb:.1f} KB")
    print(f"  Min cosine:          {min_cosine:.6f}")
    print(f"  Mean cosine:         {mean_cosine:.6f}")

    # ── Project to 7B ──
    projection = project_7b_savings(results)
    print(f"\n  --- Projected to Zamba2-7B ({projection['note']}) ---")
    print(f"  Dense 1D params:     {projection['projected_dense_mb']:.2f} MB")
    print(f"  Compressed:          {projection['projected_compressed_mb']:.2f} MB")
    print(f"  Savings:             {projection['projected_savings_mb']:.2f} MB")

    # ── How does this affect overall ratio? ──
    # Zamba2-7B: ~14 GB total, 213 compressed at 4x, rest exact
    zamba7b_total_gb = 14.0
    zamba7b_compressed_2d_orig = 8.43  # GB (estimated from 1.88x ratio)
    zamba7b_exact_orig = zamba7b_total_gb - zamba7b_compressed_2d_orig  # ~5.57 GB
    zamba7b_size_before_fold = zamba7b_compressed_2d_orig / 4 + zamba7b_exact_orig
    zamba7b_size_after_fold = zamba7b_size_before_fold - projection["projected_savings_mb"] / 1024

    ratio_before = zamba7b_total_gb / zamba7b_size_before_fold
    ratio_after = zamba7b_total_gb / zamba7b_size_after_fold

    print(f"\n  --- Impact on Zamba2-7B overall ratio ---")
    print(f"  Before fold:         {ratio_before:.2f}x ({zamba7b_size_before_fold*1024:.0f} MB)")
    print(f"  After fold:          {ratio_after:.2f}x ({zamba7b_size_after_fold*1024:.0f} MB)")
    print(f"  Ratio improvement:   {ratio_before:.2f}x → {ratio_after:.2f}x")
    print(f"{'='*70}\n")

    # ── Receipt ──
    wall_time = round(time.time() - t_start, 3)
    cpu_time = round(time.process_time() - cpu_start, 3)
    peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

    receipt = {
        "work_order": "WO-FOLD-COMPRESS-PROOF-01",
        "model": str(args.model_dir),
        "settings": {"k": args.k, "max_sidecar": args.max_sidecar,
                      "sidecar_enabled": not args.no_sidecar},
        "groups": [{k: v for k, v in r.items() if k != "tensor_names"}
                   for r in results],
        "summary": {
            "n_groups": len(results),
            "total_dense_bytes": total_dense,
            "total_compressed_bytes": total_compressed,
            "overall_ratio": round(overall_ratio, 2),
            "savings_kb": round(savings_kb, 1),
            "min_cosine": round(min_cosine, 6),
            "mean_cosine": round(float(mean_cosine), 6),
        },
        "projection_7b": projection,
        "cost": {
            "wall_time_s": wall_time,
            "cpu_time_s": cpu_time,
            "peak_memory_mb": peak_mb,
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipt_dir = REPO_DIR / "receipts" / "fold_compress"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"fold_proof_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"Receipt: {receipt_path}")

    # JSON to stdout
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
