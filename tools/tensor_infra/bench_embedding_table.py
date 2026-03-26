#!/usr/bin/env python3
"""Domain 2: Embedding Table Compression for Recommendation Systems.
Compress TinyLlama embedding tables, measure retrieval fidelity."""

import numpy as np
from pathlib import Path
from _common import *

def recall_at_k(original, compressed, n_queries=100, k=10):
    """Measure nearest-neighbor recall@k between original and compressed embedding tables."""
    rng = np.random.RandomState(42)
    n_rows = original.shape[0]
    query_indices = rng.choice(n_rows, size=min(n_queries, n_rows), replace=False)

    recalls = []
    for qi in query_indices:
        query = original[qi]
        # Distances in original space
        orig_dists = np.linalg.norm(original - query, axis=1)
        orig_dists[qi] = np.inf  # exclude self
        orig_topk = set(np.argsort(orig_dists)[:k])

        # Distances in compressed space
        comp_dists = np.linalg.norm(compressed - query, axis=1)
        comp_dists[qi] = np.inf
        comp_topk = set(np.argsort(comp_dists)[:k])

        recalls.append(len(orig_topk & comp_topk) / k)
    return recalls

def main():
    t_start, cpu_start, start_iso = start_cost()
    print("=" * 72)
    print("DOMAIN 2: Embedding Table Compression")
    print("=" * 72)

    weights = load_tinyllama_weights()
    out_dir = HELIX_ROOT / "tensor_infra_scratch" / "embedding_table"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Two embedding tensors
    targets = {
        "embed_tokens": "model.embed_tokens.weight",
        "lm_head": "lm_head.weight",
    }

    results = []
    for label, key in targets.items():
        if key not in weights:
            print(f"  SKIP {key} (not found, may be tied)")
            continue
        tensor = weights[key].astype(np.float32)
        print(f"\n  {label}: shape={tensor.shape}, kurtosis={kurtosis(tensor):.2f}")

        for k_val in [256, 64]:
            policy = policy_vq(k=k_val, sidecar=True)
            stats, recon = compress_tensor(tensor, f"{label}_k{k_val}", out_dir, policy)

            cos = cosine_sim(tensor, recon)

            # Per-row cosine (each row = one token's embedding)
            # Filter out near-zero rows (rare/unused tokens) where cosine is meaningless
            row_norms = np.linalg.norm(tensor, axis=1)
            norm_threshold = np.percentile(row_norms[row_norms > 0], 1)  # bottom 1% cutoff
            active_mask = row_norms > norm_threshold

            row_cosines = []
            for i in range(tensor.shape[0]):
                rc = cosine_sim(tensor[i], recon[i])
                row_cosines.append(rc)
            row_cosines = np.array(row_cosines)
            active_cosines = row_cosines[active_mask]

            # Per-row MSE (norm-independent metric)
            row_mses = np.array([mse(tensor[i], recon[i]) for i in range(tensor.shape[0])])

            # Recall@10 — only query from active (non-zero) rows
            active_indices = np.where(active_mask)[0]
            recalls = recall_at_k(tensor, recon,
                                  n_queries=min(100, len(active_indices)), k=10)

            result = {
                "tensor": label,
                "shape": list(tensor.shape),
                "k": k_val,
                "full_cosine": round(cos, 6),
                "active_row_cosine_min": round(float(np.min(active_cosines)), 6),
                "active_row_cosine_mean": round(float(np.mean(active_cosines)), 6),
                "active_row_cosine_p5": round(float(np.percentile(active_cosines, 5)), 6),
                "n_active_rows": int(active_mask.sum()),
                "n_total_rows": tensor.shape[0],
                "row_mse_mean": round(float(np.mean(row_mses)), 10),
                "row_mse_max": round(float(np.max(row_mses)), 10),
                "recall_at_10_mean": round(float(np.mean(recalls)), 4),
                "recall_at_10_min": round(float(np.min(recalls)), 4),
                "compression_ratio": stats.get("compression_ratio", 1.0),
            }
            results.append(result)
            print(f"    k={k_val}: cos={cos:.6f}, active_row_cos_min={np.min(active_cosines):.6f}, "
                  f"recall@10={np.mean(recalls):.4f}, ratio={result['compression_ratio']:.2f}x")

    cost = finish_cost(t_start, cpu_start, start_iso)
    k256_results = [r for r in results if r["k"] == 256]
    if k256_results:
        v, worst = verdict([r["active_row_cosine_min"] for r in k256_results])
        print(f"\n  Verdict (k=256): {v} (worst active row cosine={worst:.6f})")

    write_receipt("tensor_infra_domain_2", "embedding_table", {
        "per_tensor": results,
        "data_source": "REAL — TinyLlama embed_tokens + lm_head",
    }, cost)

if __name__ == "__main__":
    main()
