#!/usr/bin/env python3
"""Domain 6: RAG Index Compression.
Compress document embedding vectors (the index itself, not the model)."""

import numpy as np
from pathlib import Path
from _common import *

def main():
    t_start, cpu_start, start_iso = start_cost()
    print("=" * 72)
    print("DOMAIN 6: RAG Index Compression")
    print("=" * 72)

    # Generate real embeddings using MiniLM
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Use real text from WikiText-2 as document corpus
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    docs = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50][:500]

    print(f"  Encoding {len(docs)} documents...")
    embeddings = model.encode(docs, show_progress_bar=False, batch_size=32)
    embeddings = embeddings.astype(np.float32)  # (500, 384)
    print(f"  Embedding matrix: {embeddings.shape}, {embeddings.nbytes / 1e6:.2f} MB")

    out_dir = HELIX_ROOT / "tensor_infra_scratch" / "rag_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for k_val in [256, 64]:
        policy = policy_vq(k=k_val, sidecar=True)
        stats, recon = compress_tensor(embeddings, f"rag_index_k{k_val}", out_dir, policy)

        cos = cosine_sim(embeddings, recon)

        # Retrieval test: 50 random queries, find top-5 in original vs compressed
        rng = np.random.RandomState(42)
        query_ids = rng.choice(len(docs), size=50, replace=False)
        top1_matches = 0
        top5_overlaps = []

        for qi in query_ids:
            query = embeddings[qi]
            # Original similarities
            orig_sims = embeddings @ query
            orig_sims[qi] = -np.inf
            orig_top5 = set(np.argsort(orig_sims)[-5:])
            orig_top1 = np.argsort(orig_sims)[-1]
            # Compressed similarities (query from original, index from compressed)
            comp_sims = recon @ query
            comp_sims[qi] = -np.inf
            comp_top5 = set(np.argsort(comp_sims)[-5:])
            comp_top1 = np.argsort(comp_sims)[-1]

            if orig_top1 == comp_top1:
                top1_matches += 1
            top5_overlaps.append(len(orig_top5 & comp_top5))

        top1_rate = top1_matches / 50
        avg_top5 = np.mean(top5_overlaps)

        result = {
            "k": k_val,
            "full_cosine": round(cos, 6),
            "top1_match_rate": round(top1_rate, 4),
            "avg_top5_overlap": round(avg_top5, 2),
            "compression_ratio": stats.get("compression_ratio", 1.0),
            "original_mb": round(embeddings.nbytes / 1e6, 2),
        }
        results.append(result)
        print(f"\n  k={k_val}: cos={cos:.6f}, top1={top1_rate:.2%}, "
              f"top5={avg_top5:.1f}/5, ratio={result['compression_ratio']:.2f}x")

    cost = finish_cost(t_start, cpu_start, start_iso)
    write_receipt("tensor_infra_domain_6", "rag_index", {
        "n_documents": len(docs),
        "embedding_dim": embeddings.shape[1],
        "per_variant": results,
        "data_source": "REAL — WikiText-2 documents encoded by MiniLM",
    }, cost)

if __name__ == "__main__":
    main()
