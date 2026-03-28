#!/usr/bin/env python3
"""
RVQ Benchmark — Does ResidualVQ beat flat k-means on LLM weights?

Uses Lucidrain's vector-quantize-pytorch ResidualVQ.
Tests on both Qwen (transformer) and Mamba (SSM) weights.

Question 1: Does 2-stage RVQ k=16 beat flat k=256 at same bits/weight?
Question 2: Does it work identically on Mamba?
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress import WeightSource


def cosine_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def flat_vq_torch(tensor, k=256):
    """Flat 1D VQ using k-means (our current approach)."""
    flat = tensor.flatten()
    # Subsample for init
    n = len(flat)
    if n > 500_000:
        idx = torch.randperm(n, device="cpu")[:500_000]
        sample = flat[idx.to(flat.device)]
    else:
        sample = flat

    percentiles = torch.linspace(0, 1, k, device=flat.device)
    sorted_sample, _ = torch.sort(sample)
    indices = (percentiles * (len(sorted_sample) - 1)).long()
    centroids = sorted_sample[indices].float()

    # K-means iterations
    CHUNK = 500_000
    for _ in range(15):
        assignments = torch.empty(n, dtype=torch.long, device=flat.device)
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            dists = torch.abs(flat[start:end].unsqueeze(1) - centroids.unsqueeze(0))
            assignments[start:end] = torch.argmin(dists, dim=1)

        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = flat[mask].mean()
            else:
                new_centroids[i] = centroids[i]

        if (new_centroids - centroids).abs().max() < 1e-5:
            break
        centroids = new_centroids

    recon = centroids[assignments].reshape(tensor.shape)
    return recon


def rvq_encode(tensor_2d, n_quantizers=2, codebook_size=16, n_train_iters=100):
    """
    Use Lucidrain's ResidualVQ on weight tensor.

    ResidualVQ expects [batch, seq, dim] input.
    For a [out, in] weight matrix, treat each row as a vector to quantize.

    Runs multiple forward passes to let EMA codebook updates converge.
    """
    from vector_quantize_pytorch import ResidualVQ

    rows, cols = tensor_2d.shape
    device = tensor_2d.device

    rvq = ResidualVQ(
        dim=cols,
        num_quantizers=n_quantizers,
        codebook_size=codebook_size,
        decay=0.99,
        commitment_weight=0.0,
        kmeans_init=True,
        kmeans_iters=20,
    ).to(device)

    # ResidualVQ expects [batch, seq_len, dim]
    x = tensor_2d.unsqueeze(0)  # [1, rows, cols]

    # Train the codebooks with multiple forward passes (EMA needs iterations)
    rvq.train()
    with torch.no_grad():
        for _ in range(n_train_iters):
            quantized, indices, commit_loss = rvq(x)

    # Final eval pass
    rvq.eval()
    with torch.no_grad():
        quantized, indices, commit_loss = rvq(x)

    recon = quantized.squeeze(0)  # [rows, cols]
    return recon, indices


def test_tensor(name, tensor_np, device="cuda"):
    """Test flat VQ vs RVQ on a single tensor."""
    tensor = torch.from_numpy(tensor_np.astype(np.float32)).to(device)

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])

    rows, cols = tensor.shape
    original = tensor_np.astype(np.float32)
    if original.ndim == 1:
        original = original.reshape(1, -1)
    elif original.ndim > 2:
        original = original.reshape(-1, original.shape[-1])

    results = {"name": name, "shape": [rows, cols], "elements": rows * cols}

    # --- Flat VQ k=256 (8-bit, ~4x) — our current approach ---
    t0 = time.time()
    recon_flat = flat_vq_torch(tensor, k=256)
    t_flat = time.time() - t0
    cos_flat = cosine_sim(original, recon_flat.cpu().numpy())
    results["flat_k256"] = {
        "cosine": round(cos_flat, 6),
        "bits_per_weight": 8,
        "ratio": "~4x",
        "time_s": round(t_flat, 2),
    }

    # --- RVQ: 2 quantizers × k=16 (2×4=8 bits, ~4x) ---
    t0 = time.time()
    recon_rvq2x16, _ = rvq_encode(tensor, n_quantizers=2, codebook_size=16)
    t_rvq = time.time() - t0
    cos_rvq2x16 = cosine_sim(original, recon_rvq2x16.cpu().numpy())
    results["rvq_2x16"] = {
        "cosine": round(cos_rvq2x16, 6),
        "bits_per_weight": 8,
        "ratio": "~4x (2×4-bit)",
        "time_s": round(t_rvq, 2),
    }

    # --- RVQ: 2 quantizers × k=256 (2×8=16 bits, ~2x) ---
    t0 = time.time()
    recon_rvq2x256, _ = rvq_encode(tensor, n_quantizers=2, codebook_size=256)
    t_rvq2 = time.time() - t0
    cos_rvq2x256 = cosine_sim(original, recon_rvq2x256.cpu().numpy())
    results["rvq_2x256"] = {
        "cosine": round(cos_rvq2x256, 6),
        "bits_per_weight": 16,
        "ratio": "~2x (2×8-bit)",
        "time_s": round(t_rvq2, 2),
    }

    # --- RVQ: 4 quantizers × k=16 (4×4=16 bits, ~2x but finer residual) ---
    t0 = time.time()
    recon_rvq4x16, _ = rvq_encode(tensor, n_quantizers=4, codebook_size=16)
    t_rvq4 = time.time() - t0
    cos_rvq4x16 = cosine_sim(original, recon_rvq4x16.cpu().numpy())
    results["rvq_4x16"] = {
        "cosine": round(cos_rvq4x16, 6),
        "bits_per_weight": 16,
        "ratio": "~2x (4×4-bit)",
        "time_s": round(t_rvq4, 2),
    }

    # Clean up GPU
    del tensor, recon_flat, recon_rvq2x16, recon_rvq2x256, recon_rvq4x16
    torch.cuda.empty_cache()

    return results


def run_model(model_dir, label, max_tensors=10):
    """Run benchmark on a model."""
    source = WeightSource(model_dir)
    all_names = source.tensor_names()

    # Filter to 2D weight tensors
    tensors = []
    for name in all_names:
        shape = source.get_shape(name)
        if len(shape) != 2 or shape[0] * shape[1] < 256:
            continue
        if name.endswith(".bias") or "embed_tokens" in name or "lm_head" in name:
            continue
        tensors.append((name, shape))

    if max_tensors:
        # Sample from different parts of the model
        step = max(1, len(tensors) // max_tensors)
        tensors = tensors[::step][:max_tensors]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*100}")
    print(f"  {label} — {len(tensors)} tensors on {device}")
    print(f"{'='*100}")
    print(f"  {'Tensor':<45} {'Flat k256':>9} {'RVQ 2×16':>9} {'RVQ 2×256':>10} {'RVQ 4×16':>9}")
    print(f"  {'-'*45} {'-'*9} {'-'*9} {'-'*10} {'-'*9}")

    results = []
    for name, shape in tensors:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        tensor_np = source.get_tensor(name)
        # Skip tensors too large for 4GB GPU (>8M elements)
        if shape[0] * shape[1] > 8_000_000 and device == "cuda":
            print(f"  {name:<45} SKIPPED (too large for GPU)")
            del tensor_np
            continue
        r = test_tensor(name, tensor_np, device=device)
        results.append(r)

        f = r["flat_k256"]["cosine"]
        r2x16 = r["rvq_2x16"]["cosine"]
        r2x256 = r["rvq_2x256"]["cosine"]
        r4x16 = r["rvq_4x16"]["cosine"]
        print(f"  {name:<45} {f:>9.6f} {r2x16:>9.6f} {r2x256:>10.6f} {r4x16:>9.6f}")
        del tensor_np

    # Summary
    if not results:
        print(f"\n  ALL TENSORS SKIPPED (too large for GPU)")
        return results

    flat_cos = [r["flat_k256"]["cosine"] for r in results]
    rvq2x16_cos = [r["rvq_2x16"]["cosine"] for r in results]
    rvq2x256_cos = [r["rvq_2x256"]["cosine"] for r in results]
    rvq4x16_cos = [r["rvq_4x16"]["cosine"] for r in results]

    print(f"\n  {'AVERAGES':<45} {np.mean(flat_cos):>9.6f} {np.mean(rvq2x16_cos):>9.6f} "
          f"{np.mean(rvq2x256_cos):>10.6f} {np.mean(rvq4x16_cos):>9.6f}")
    print(f"  {'MINIMUMS':<45} {np.min(flat_cos):>9.6f} {np.min(rvq2x16_cos):>9.6f} "
          f"{np.min(rvq2x256_cos):>10.6f} {np.min(rvq4x16_cos):>9.6f}")

    # Count RVQ 2x16 wins vs flat k=256 (same bit budget)
    wins = sum(1 for f, r in zip(flat_cos, rvq2x16_cos) if r > f + 0.0001)
    losses = sum(1 for f, r in zip(flat_cos, rvq2x16_cos) if r < f - 0.0001)
    print(f"\n  RVQ 2×16 vs Flat k=256 (both 8 bits/weight):")
    print(f"    RVQ wins:  {wins}/{len(results)}")
    print(f"    Flat wins: {losses}/{len(results)}")
    print(f"    Avg delta: {np.mean(np.array(rvq2x16_cos) - np.array(flat_cos)):+.6f}")

    return results


def main():
    t_start = time.time()

    # Test Qwen 3B (transformer)
    qwen_dir = Path.home() / "models/qwen2.5-3b-instruct"
    qwen_results = run_model(qwen_dir, "QWEN 3B (Transformer)", max_tensors=10)

    # Test Mamba 130M (SSM)
    mamba_dir = Path.home() / "models/mamba-130m-hf"
    mamba_results = run_model(mamba_dir, "MAMBA 130M (SSM)", max_tensors=10)

    # Test Mamba2 1.3B (SSM v2)
    mamba2_dir = Path.home() / "models/mamba2-1.3b"
    if mamba2_dir.exists():
        mamba2_results = run_model(mamba2_dir, "MAMBA2 1.3B (SSM v2)", max_tensors=10)
    else:
        mamba2_results = []

    wall = round(time.time() - t_start, 1)

    print(f"\n{'='*100}")
    print(f"  VERDICT")
    print(f"{'='*100}")
    print(f"  Wall time: {wall}s")
    print(f"  Key question: Does ResidualVQ work universally across architectures?")
    print(f"{'='*100}")

    # Save receipt
    receipt = {
        "work_order": "WO-RVQ-BENCHMARK-01",
        "question": "Does ResidualVQ beat flat VQ and work on both transformers and SSMs?",
        "library": "vector-quantize-pytorch==1.28.0 (Lucidrain)",
        "models": {
            "qwen_3b": qwen_results,
            "mamba_130m": mamba_results,
            "mamba2_1.3b": mamba2_results,
        },
        "cost": {
            "wall_time_s": wall,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    out_dir = Path("receipts/rvq_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"rvq_benchmark_{ts}.json"
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"  Receipt: {path}")


if __name__ == "__main__":
    main()
