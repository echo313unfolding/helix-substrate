#!/usr/bin/env python3
"""
Group VQ Experiment — Can per-group codebooks match global VQ quality at 2x compression?

Current approach: 1 codebook of 256 entries per tensor (8-bit indices, ~4x from FP32)
Group VQ:         1 codebook of 16 entries per GROUP of 128 columns (4-bit indices, ~8x from FP32)

AQLM and GPTVQ use this technique to reach 2-bit. We just need 4-bit (8x) with
architecture-agnostic k-means. No calibration, no Hessian, no activation data.

Test: Compare per-tensor cosine similarity of both approaches across all tensors
in a model, then compute reconstruction error.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Group VQ parameters
GROUP_SIZE = 128       # columns per group
K_PER_GROUP = 16       # codebook entries per group (4-bit)
K_GLOBAL = 256         # codebook entries for global VQ (8-bit, current)


def simple_kmeans_1d(data, k, max_iters=15, rtol=0.001):
    """1D k-means with GPU acceleration when available."""
    k = min(k, len(np.unique(data)))
    if k < 2:
        return np.array([np.mean(data)], dtype=np.float32), np.zeros(len(data), dtype=np.uint8)

    # Subsample for centroid initialization if data is large
    if len(data) > 1_000_000:
        rng = np.random.RandomState(42)
        sample = data[rng.choice(len(data), 500_000, replace=False)]
    else:
        sample = data

    percentiles = np.linspace(0, 100, k)
    centroids = np.percentile(sample, percentiles).astype(np.float32)

    cb_range = float(centroids[-1] - centroids[0])
    if cb_range < 1e-30:
        cb_range = 1.0
    abs_tol = rtol * cb_range

    # GPU path
    try:
        import torch
        if torch.cuda.is_available():
            return _gpu_kmeans_1d(data, centroids, k, max_iters, abs_tol)
    except ImportError:
        pass

    # CPU path with chunked assignment
    CHUNK = 500_000
    for _ in range(max_iters):
        assignments = np.empty(len(data), dtype=np.uint8)
        for start in range(0, len(data), CHUNK):
            end = min(start + CHUNK, len(data))
            dists = np.abs(data[start:end, np.newaxis] - centroids)
            assignments[start:end] = np.argmin(dists, axis=1).astype(np.uint8)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = assignments == i
            if np.any(mask):
                new_centroids[i] = np.mean(data[mask])
            else:
                new_centroids[i] = centroids[i]

        if float(np.max(np.abs(new_centroids - centroids))) < abs_tol:
            break
        centroids = new_centroids

    return centroids, assignments


def _gpu_kmeans_1d(data_np, centroids_np, k, max_iters, abs_tol):
    """GPU k-means — much faster on large tensors."""
    import torch
    CHUNK = 2_000_000  # GPU chunk size
    data = torch.from_numpy(data_np).cuda()
    centroids = torch.from_numpy(centroids_np).cuda()

    for _ in range(max_iters):
        # Chunked GPU assignment
        assignments = torch.empty(len(data), dtype=torch.uint8, device='cuda')
        for start in range(0, len(data), CHUNK):
            end = min(start + CHUNK, len(data))
            dists = torch.abs(data[start:end].unsqueeze(1) - centroids.unsqueeze(0))
            assignments[start:end] = torch.argmin(dists, dim=1).to(torch.uint8)

        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = data[mask].mean()
            else:
                new_centroids[i] = centroids[i]

        max_delta = float((new_centroids - centroids).abs().max())
        if max_delta < abs_tol:
            break
        centroids = new_centroids

    return centroids.cpu().numpy(), assignments.cpu().numpy().astype(np.uint8)


def global_vq(tensor, k=K_GLOBAL):
    """Standard global VQ: one codebook per tensor."""
    flat = tensor.flatten().astype(np.float32)
    codebook, indices = simple_kmeans_1d(flat, k)
    reconstructed = codebook[indices].reshape(tensor.shape)

    # Storage cost
    codebook_bytes = k * 4  # float32
    indices_bytes = len(indices) * 1  # uint8
    total_bytes = codebook_bytes + indices_bytes
    original_bytes = tensor.size * 4  # float32

    return reconstructed, {
        "method": f"global_vq_k{k}",
        "codebook_entries": k,
        "bits_per_weight": 8,
        "compressed_bytes": total_bytes,
        "original_bytes": original_bytes,
        "ratio": round(original_bytes / total_bytes, 2),
    }


def group_vq(tensor, group_size=GROUP_SIZE, k=K_PER_GROUP):
    """Group VQ: one codebook per group of columns."""
    rows, cols = tensor.shape
    # Pad cols to multiple of group_size if needed
    n_groups = (cols + group_size - 1) // group_size
    padded_cols = n_groups * group_size

    # Pad tensor if needed
    if padded_cols > cols:
        padded = np.zeros((rows, padded_cols), dtype=np.float32)
        padded[:, :cols] = tensor
    else:
        padded = tensor.astype(np.float32)

    # Reshape into groups: [rows, n_groups, group_size]
    grouped = padded.reshape(rows, n_groups, group_size)

    codebooks = np.zeros((n_groups, k), dtype=np.float32)
    all_indices = np.zeros((rows, n_groups, group_size), dtype=np.uint8)

    for g in range(n_groups):
        group_data = grouped[:, g, :].flatten()  # [rows * group_size]
        cb, idx = simple_kmeans_1d(group_data, k)
        codebooks[g, :len(cb)] = cb
        all_indices[:, g, :] = idx.reshape(rows, group_size)

    # Reconstruct
    reconstructed = np.zeros_like(grouped)
    for g in range(n_groups):
        reconstructed[:, g, :] = codebooks[g][all_indices[:, g, :]]
    reconstructed = reconstructed.reshape(rows, padded_cols)[:, :cols]

    # Storage cost
    codebook_bytes = n_groups * k * 4  # float32 per codebook
    # 4-bit indices: 2 per byte
    n_elements = rows * cols
    indices_bytes = (n_elements + 1) // 2  # packed 4-bit
    total_bytes = codebook_bytes + indices_bytes
    original_bytes = tensor.size * 4

    return reconstructed, {
        "method": f"group_vq_g{group_size}_k{k}",
        "n_groups": n_groups,
        "codebook_entries_per_group": k,
        "bits_per_weight": 4,
        "compressed_bytes": total_bytes,
        "original_bytes": original_bytes,
        "ratio": round(original_bytes / total_bytes, 2),
        "codebook_overhead_bytes": codebook_bytes,
    }


def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def mse(a, b):
    """Mean squared error."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def test_tensor(tensor_np, name, shape, existing_cos=None):
    """Test group VQ on a single tensor. Use existing global VQ cosine if available."""
    if tensor_np.ndim == 1:
        tensor_np = tensor_np.reshape(1, -1)
    elif tensor_np.ndim > 2:
        tensor_np = tensor_np.reshape(-1, tensor_np.shape[-1])
    tensor_np = tensor_np.astype(np.float32)

    # Global VQ (current, 4x) — use existing stats if available, else compute
    if existing_cos is not None:
        cos_global = existing_cos
        t_global = 0
    else:
        t0 = time.time()
        recon_global, stats_global = global_vq(tensor_np, k=K_GLOBAL)
        t_global = time.time() - t0
        cos_global = cosine_sim(tensor_np, recon_global)
        del recon_global

    # Group VQ k=16 (proposed, 4-bit, ~8x)
    t0 = time.time()
    recon_group, stats_group = group_vq(tensor_np, group_size=GROUP_SIZE, k=K_PER_GROUP)
    t_group = time.time() - t0
    cos_group = cosine_sim(tensor_np, recon_group)
    del recon_group

    # Group VQ k=32 (5-bit, ~6.4x)
    t0 = time.time()
    recon_g32, stats_g32 = group_vq(tensor_np, group_size=GROUP_SIZE, k=32)
    t_g32 = time.time() - t0
    cos_g32 = cosine_sim(tensor_np, recon_g32)
    del recon_g32

    return {
        "name": name,
        "shape": list(shape),
        "elements": int(np.prod(shape)),
        "global_vq_k256": {
            "cosine": round(cos_global, 6),
            "ratio": 4.0,
            "time_s": round(t_global, 2),
            "from_cache": existing_cos is not None,
        },
        "group_vq_g128_k16": {
            "cosine": round(cos_group, 6),
            "ratio": stats_group["ratio"],
            "n_groups": stats_group["n_groups"],
            "time_s": round(t_group, 2),
        },
        "group_vq_g128_k32": {
            "cosine": round(cos_g32, 6),
            "ratio": stats_g32["ratio"],
            "time_s": round(t_g32, 2),
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-tensors", type=int, default=None,
                        help="Max tensors to test (default: all)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # WeightSource is defined in compress.py, not a separate module
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compress import WeightSource

    t_start = time.time()

    source = WeightSource(args.model_dir)
    all_names = source.tensor_names()

    # Filter to 2D weight tensors only
    tensors_to_test = []
    for name in all_names:
        shape = source.get_shape(name)
        if len(shape) != 2:
            continue
        if shape[0] * shape[1] < 256:
            continue
        if name.endswith(".bias"):
            continue
        # Skip embeddings and lm_head (exact in production)
        if "embed_tokens" in name or "lm_head" in name:
            continue
        tensors_to_test.append((name, shape))

    if args.max_tensors:
        tensors_to_test = tensors_to_test[:args.max_tensors]

    print(f"{'='*90}")
    print(f"  GROUP VQ EXPERIMENT — {args.model_dir.name}")
    print(f"  Testing {len(tensors_to_test)} tensors")
    print(f"  Global VQ: k=256, 8-bit, ~4x | Group VQ: g=128, k=16, 4-bit, ~8x")
    print(f"{'='*90}")

    # Load existing global VQ cosines from cdnav3 stats.json (avoid recomputing)
    cdna_dir = args.model_dir / "cdnav3"
    existing_cosines = {}
    if cdna_dir.exists():
        for td in cdna_dir.iterdir():
            if not td.is_dir() or not td.name.endswith(".cdnav3"):
                continue
            sp = td / "stats.json"
            if sp.exists():
                s = json.loads(sp.read_text())
                tname = s.get("tensor_name", "")
                cos = s.get("cosine_no_sidecar")
                if tname and cos:
                    existing_cosines[tname] = cos
        print(f"  Loaded {len(existing_cosines)} existing global VQ cosines from disk")

    results = []
    for i, (name, shape) in enumerate(tensors_to_test):
        tensor_np = source.get_tensor(name)
        existing_cos = existing_cosines.get(name)
        r = test_tensor(tensor_np, name, shape, existing_cos=existing_cos)
        results.append(r)

        # Progress
        g_cos = r["global_vq_k256"]["cosine"]
        grp_cos = r["group_vq_g128_k16"]["cosine"]
        g32_cos = r["group_vq_g128_k32"]["cosine"]
        delta = grp_cos - g_cos
        print(f"  [{i+1:3d}/{len(tensors_to_test)}] {name:50s} "
              f"global={g_cos:.5f}  grp16={grp_cos:.5f} ({delta:+.5f})  "
              f"grp32={g32_cos:.5f}")
        del tensor_np

    # Summary statistics
    global_cosines = [r["global_vq_k256"]["cosine"] for r in results]
    group_cosines = [r["group_vq_g128_k16"]["cosine"] for r in results]
    g32_cosines = [r["group_vq_g128_k32"]["cosine"] for r in results]
    deltas = [g - gl for g, gl in zip(group_cosines, global_cosines)]

    # Count wins
    group_wins = sum(1 for d in deltas if d > 0.0001)
    global_wins = sum(1 for d in deltas if d < -0.0001)
    ties = len(deltas) - group_wins - global_wins

    print(f"\n{'='*90}")
    print(f"  SUMMARY — {args.model_dir.name}")
    print(f"{'='*90}")
    print(f"  {'Metric':<30} {'Global VQ k=256':>15} {'Group g128 k=16':>15} {'Group g128 k=32':>15}")
    print(f"  {'─'*30} {'─'*15} {'─'*15} {'─'*15}")
    print(f"  {'Compression ratio':<30} {'~4x':>15} {'~8x':>15} {'~6.4x':>15}")
    print(f"  {'Bits per weight':<30} {'8':>15} {'4':>15} {'5':>15}")
    print(f"  {'Avg cosine':<30} {np.mean(global_cosines):>15.6f} {np.mean(group_cosines):>15.6f} {np.mean(g32_cosines):>15.6f}")
    print(f"  {'Min cosine':<30} {np.min(global_cosines):>15.6f} {np.min(group_cosines):>15.6f} {np.min(g32_cosines):>15.6f}")
    print(f"  {'Median cosine':<30} {np.median(global_cosines):>15.6f} {np.median(group_cosines):>15.6f} {np.median(g32_cosines):>15.6f}")
    print(f"  {'Std cosine':<30} {np.std(global_cosines):>15.6f} {np.std(group_cosines):>15.6f} {np.std(g32_cosines):>15.6f}")
    print(f"  {'Tensors below 0.999':<30} {sum(1 for c in global_cosines if c < 0.999):>15} {sum(1 for c in group_cosines if c < 0.999):>15} {sum(1 for c in g32_cosines if c < 0.999):>15}")
    print(f"  {'Tensors below 0.995':<30} {sum(1 for c in global_cosines if c < 0.995):>15} {sum(1 for c in group_cosines if c < 0.995):>15} {sum(1 for c in g32_cosines if c < 0.995):>15}")
    print(f"\n  Group k=16 vs Global k=256:")
    print(f"    Group wins (>0.0001 better):  {group_wins}")
    print(f"    Global wins:                  {global_wins}")
    print(f"    Ties:                         {ties}")
    print(f"    Avg delta:                    {np.mean(deltas):+.6f}")

    wall = round(time.time() - t_start, 1)

    # Save receipt
    receipt = {
        "work_order": "WO-GROUP-VQ-01",
        "question": "Does group VQ (g=128, k=16, 4-bit, 8x) match global VQ (k=256, 8-bit, 4x)?",
        "model": args.model_dir.name,
        "n_tensors": len(results),
        "config": {
            "global": {"k": K_GLOBAL, "bits": 8, "ratio": "~4x"},
            "group_k16": {"group_size": GROUP_SIZE, "k": K_PER_GROUP, "bits": 4, "ratio": "~8x"},
            "group_k32": {"group_size": GROUP_SIZE, "k": 32, "bits": 5, "ratio": "~6.4x"},
        },
        "summary": {
            "global_avg_cos": round(float(np.mean(global_cosines)), 6),
            "group_k16_avg_cos": round(float(np.mean(group_cosines)), 6),
            "group_k32_avg_cos": round(float(np.mean(g32_cosines)), 6),
            "global_min_cos": round(float(np.min(global_cosines)), 6),
            "group_k16_min_cos": round(float(np.min(group_cosines)), 6),
            "group_k32_min_cos": round(float(np.min(g32_cosines)), 6),
            "group_wins": group_wins,
            "global_wins": global_wins,
            "ties": ties,
        },
        "per_tensor": results,
        "cost": {
            "wall_time_s": wall,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    out_dir = Path("receipts/group_vq")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = out_dir / f"group_vq_{args.model_dir.name}_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Wall time: {wall}s")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
