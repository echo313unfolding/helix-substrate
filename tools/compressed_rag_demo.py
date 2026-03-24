#!/usr/bin/env python3
"""
Compressed MiniLM RAG Demo — Prove HelixLinear works for real retrieval.

Indexes all receipts in receipts/ using both dense and compressed MiniLM.
Runs the same queries against both. Compares:
  - Retrieval ranking (should be identical or near-identical)
  - Memory footprint (compressed ~4x smaller)
  - Latency

This is a functional proof: compressed embeddings give the SAME search results.
"""

import gc
import json
import os
import platform
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_receipts(receipts_dir):
    """Load all JSON receipts into indexable documents."""
    docs = []
    for json_path in sorted(Path(receipts_dir).rglob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue

        # Build a text representation for embedding
        parts = [f"File: {json_path.name}"]
        parts.append(f"Directory: {json_path.parent.name}")

        for key in ("experiment", "schema", "work_order", "model", "verdict"):
            if key in data and isinstance(data[key], str):
                parts.append(f"{key}: {data[key]}")

        if "cost" in data and isinstance(data["cost"], dict):
            cost = data["cost"]
            if "wall_time_s" in cost:
                parts.append(f"wall_time: {cost['wall_time_s']}s")
            if "peak_memory_mb" in cost:
                parts.append(f"peak_memory: {cost['peak_memory_mb']} MB")

        # Add numeric highlights
        for key in ("compression_ratio", "perplexity", "cosine", "avg_cosine",
                     "vram_peak_eval_mb", "vram_model_mb", "eval_tokens",
                     "gpu_vram_mb", "n_tensors", "n_compressed"):
            if key in data:
                parts.append(f"{key}: {data[key]}")

        # Check nested results
        if "results" in data and isinstance(data["results"], dict):
            for rname, rval in data["results"].items():
                if isinstance(rval, dict):
                    for k, v in rval.items():
                        if isinstance(v, (int, float)) and k not in ("compress_time_s",):
                            parts.append(f"{rname}.{k}: {v}")
                        elif isinstance(v, str) and len(v) < 100:
                            parts.append(f"{rname}.{k}: {v}")

        # Helix-specific fields
        if "helix" in data and isinstance(data["helix"], dict):
            for k, v in data["helix"].items():
                if isinstance(v, (int, float)):
                    parts.append(f"helix.{k}: {v}")

        text = "\n".join(parts)
        docs.append({
            "path": str(json_path),
            "name": json_path.name,
            "dir": json_path.parent.name,
            "text": text,
        })

    return docs


def compress_model_linears(model):
    """Compress all nn.Linear in the sentence-transformer's auto_model."""
    from helix_substrate.cdnav3_writer import CDNAv3Writer
    from helix_substrate.tensor_policy import TensorPolicy, TensorClass
    from helix_substrate.helix_linear import load_helix_linear_from_cdnav3

    policy = TensorPolicy(
        tensor_class=TensorClass.UNKNOWN,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        percentile=99.9,
        use_kmeans=True,
        sidecar_enabled=True,
        block_rows=32,
        max_corrections=256,
    )

    transformer = model[0].auto_model

    with tempfile.TemporaryDirectory() as tmp_dir:
        cdna_dir = Path(tmp_dir) / "minilm_cdnav3"
        cdna_dir.mkdir()
        writer = CDNAv3Writer(cdna_dir)

        n_replaced = 0
        for name, module in list(transformer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if module.weight.numel() < 64:
                continue

            weight = module.weight.data.float().cpu().numpy()
            if weight.ndim == 1:
                weight = weight.reshape(1, -1)

            tensor_name = f"{name}.weight"
            safe_name = tensor_name.replace("/", "_").replace(".", "_")

            try:
                writer.write_tensor(weight, tensor_name, policy=policy)
            except Exception:
                continue

            # Find the written directory
            tensor_dir = None
            for d in cdna_dir.glob("*.cdnav3"):
                meta_path = d / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if meta.get("tensor_name") == tensor_name:
                        tensor_dir = d
                        break

            if tensor_dir is None or not (tensor_dir / "codebook.npy").exists():
                continue

            bias = module.bias.data.cpu().clone() if module.bias is not None else None
            helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)

            if helix_mod.in_features != module.in_features:
                continue
            if helix_mod.out_features != module.out_features:
                continue

            # Replace
            parts = name.split(".")
            parent = transformer
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            if parts[-1].isdigit():
                parent[int(parts[-1])] = helix_mod
            else:
                setattr(parent, parts[-1], helix_mod)
            n_replaced += 1

    return n_replaced


def measure_model_memory(model):
    """Measure actual tensor memory of the transformer."""
    transformer = model[0].auto_model
    dense_bytes = 0
    compressed_bytes = 0
    from helix_substrate.helix_linear import HelixLinear

    for name, module in transformer.named_modules():
        if isinstance(module, HelixLinear):
            s = module.memory_savings()
            compressed_bytes += s["compressed_bytes"]
            dense_bytes += s["dense_bytes"]
        elif isinstance(module, nn.Linear):
            dense_bytes += module.weight.numel() * module.weight.element_size()

    return dense_bytes, compressed_bytes


def run_queries(model, doc_embeddings, docs, queries):
    """Run queries and return ranked results."""
    results = []
    for query in queries:
        t0 = time.time()
        q_embed = model.encode([query], convert_to_tensor=True)
        # Cosine similarity (embeddings are normalized by sentence-transformers)
        sims = torch.nn.functional.cosine_similarity(q_embed, doc_embeddings, dim=1)
        latency = time.time() - t0

        ranked_idx = sims.argsort(descending=True).cpu().tolist()
        ranked_scores = sims[ranked_idx].cpu().tolist()

        results.append({
            "query": query,
            "latency_ms": round(latency * 1000, 1),
            "top5_idx": ranked_idx[:5],
            "top5_scores": [round(s, 6) for s in ranked_scores[:5]],
            "top5_docs": [docs[i]["name"] for i in ranked_idx[:5]],
        })
    return results


def main():
    t_global = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("  Compressed MiniLM RAG Demo")
    print(f"  {start_iso}")
    print("=" * 70)

    receipts_dir = Path(__file__).parent.parent / "receipts"
    docs = load_receipts(receipts_dir)
    print(f"\n  Indexed {len(docs)} receipt documents")

    queries = [
        "What was the VRAM peak in the GPU viability test?",
        "Which model had the best compression ratio?",
        "What is the perplexity of the compressed TinyLlama?",
        "How many tensors were compressed for Qwen?",
        "CLIP vision encoder compression results",
        "sentence transformer embedding cosine similarity",
        "ResNet prediction accuracy after compression",
        "Triton kernel memory usage",
    ]

    # =========================================================
    # Phase 1: Dense baseline
    # =========================================================
    print("\n--- Phase 1: Dense MiniLM ---")
    from sentence_transformers import SentenceTransformer

    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    dense_mem, _ = measure_model_memory(dense_model)
    print(f"  Model memory: {dense_mem / 1e6:.1f} MB (dense)")

    t_index = time.time()
    doc_texts = [d["text"] for d in docs]
    dense_embeddings = dense_model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=False)
    index_time_dense = time.time() - t_index
    print(f"  Indexed {len(docs)} docs in {index_time_dense:.2f}s")

    dense_results = run_queries(dense_model, dense_embeddings, docs, queries)
    for r in dense_results:
        print(f"  Q: {r['query'][:50]}...")
        print(f"    Top: {r['top5_docs'][0]} (score={r['top5_scores'][0]:.4f}, {r['latency_ms']:.0f}ms)")

    # Store embeddings on CPU, free GPU
    dense_embeddings_cpu = dense_embeddings.cpu().clone()
    del dense_model, dense_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # Phase 2: Compressed
    # =========================================================
    print("\n--- Phase 2: Compressed MiniLM (HelixLinear) ---")

    comp_model = SentenceTransformer("all-MiniLM-L6-v2")
    t_compress = time.time()
    n_replaced = compress_model_linears(comp_model)
    compress_time = time.time() - t_compress
    _, comp_mem = measure_model_memory(comp_model)
    dense_equiv, _ = measure_model_memory(comp_model)
    # Re-measure properly
    total_dense = 0
    total_comp = 0
    from helix_substrate.helix_linear import HelixLinear
    for name, module in comp_model[0].auto_model.named_modules():
        if isinstance(module, HelixLinear):
            s = module.memory_savings()
            total_dense += s["dense_bytes"]
            total_comp += s["compressed_bytes"]
        elif isinstance(module, nn.Linear):
            total_dense += module.weight.numel() * 4

    ratio = round(total_dense / max(1, total_comp), 2)
    print(f"  Replaced {n_replaced} layers in {compress_time:.1f}s")
    print(f"  Linear memory: {total_dense / 1e6:.1f} MB dense → {total_comp / 1e6:.1f} MB compressed ({ratio}x)")

    t_index = time.time()
    comp_embeddings = comp_model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=False)
    index_time_comp = time.time() - t_index
    print(f"  Indexed {len(docs)} docs in {index_time_comp:.2f}s")

    comp_results = run_queries(comp_model, comp_embeddings, docs, queries)
    for r in comp_results:
        print(f"  Q: {r['query'][:50]}...")
        print(f"    Top: {r['top5_docs'][0]} (score={r['top5_scores'][0]:.4f}, {r['latency_ms']:.0f}ms)")

    # =========================================================
    # Phase 3: Comparison
    # =========================================================
    print("\n" + "=" * 70)
    print("  COMPARISON: Dense vs Compressed Retrieval")
    print("=" * 70)

    # Embedding-level comparison
    comp_embeddings_cpu = comp_embeddings.cpu()
    embed_cosines = []
    for i in range(len(docs)):
        cos = float(torch.nn.functional.cosine_similarity(
            dense_embeddings_cpu[i:i+1], comp_embeddings_cpu[i:i+1]
        ))
        embed_cosines.append(cos)

    avg_embed_cos = sum(embed_cosines) / len(embed_cosines)
    min_embed_cos = min(embed_cosines)
    print(f"\n  Document embedding cosine (avg): {avg_embed_cos:.6f}")
    print(f"  Document embedding cosine (min): {min_embed_cos:.6f}")

    # Ranking comparison
    rank_matches = 0
    top5_overlaps = []
    total_queries = len(queries)

    print(f"\n  {'Query':<52} {'Top-1 Match':>11} {'Top-5 Overlap':>13}")
    print(f"  {'-'*52} {'-'*11} {'-'*13}")

    for dr, cr in zip(dense_results, comp_results):
        top1_match = dr["top5_idx"][0] == cr["top5_idx"][0]
        overlap = len(set(dr["top5_idx"]) & set(cr["top5_idx"]))
        top5_overlaps.append(overlap)
        if top1_match:
            rank_matches += 1

        q_short = dr["query"][:50]
        print(f"  {q_short:<52} {'YES' if top1_match else 'NO':>11} {overlap}/5{' ':>9}")

    top1_accuracy = rank_matches / total_queries
    avg_top5_overlap = sum(top5_overlaps) / len(top5_overlaps)

    print(f"\n  Top-1 match rate: {rank_matches}/{total_queries} ({top1_accuracy*100:.0f}%)")
    print(f"  Avg top-5 overlap: {avg_top5_overlap:.1f}/5")
    print(f"  Memory: {total_dense/1e6:.1f} MB → {total_comp/1e6:.1f} MB ({ratio}x savings)")
    print(f"  Index time: {index_time_dense:.2f}s (dense) vs {index_time_comp:.2f}s (compressed)")

    # =========================================================
    # Receipt
    # =========================================================
    wall_time = time.time() - t_global

    receipt = {
        "experiment": "compressed_rag_demo",
        "timestamp": start_iso,
        "model": "all-MiniLM-L6-v2",
        "n_documents": len(docs),
        "n_queries": len(queries),
        "n_layers_compressed": n_replaced,
        "compression_ratio": ratio,
        "dense_linear_mb": round(total_dense / 1e6, 1),
        "compressed_linear_mb": round(total_comp / 1e6, 1),
        "avg_embedding_cosine": round(avg_embed_cos, 6),
        "min_embedding_cosine": round(min_embed_cos, 6),
        "top1_match_rate": round(top1_accuracy, 4),
        "avg_top5_overlap": round(avg_top5_overlap, 2),
        "index_time_dense_s": round(index_time_dense, 3),
        "index_time_compressed_s": round(index_time_comp, 3),
        "compress_time_s": round(compress_time, 1),
        "queries": [
            {
                "query": dr["query"],
                "dense_top1": dr["top5_docs"][0],
                "compressed_top1": cr["top5_docs"][0],
                "top1_match": dr["top5_idx"][0] == cr["top5_idx"][0],
                "top5_overlap": len(set(dr["top5_idx"]) & set(cr["top5_idx"])),
                "dense_score": dr["top5_scores"][0],
                "compressed_score": cr["top5_scores"][0],
            }
            for dr, cr in zip(dense_results, comp_results)
        ],
        "cost": {
            "wall_time_s": round(wall_time, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
        },
    }

    # Verdict
    if top1_accuracy >= 0.875 and avg_embed_cos >= 0.995:
        receipt["verdict"] = "PROVEN"
    elif top1_accuracy >= 0.75 and avg_embed_cos >= 0.99:
        receipt["verdict"] = "STRONG"
    else:
        receipt["verdict"] = "WEAK"

    receipt_dir = Path(__file__).parent.parent / "receipts" / "compressed_rag_demo"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"rag_demo_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))

    print(f"\n  Verdict: {receipt['verdict']}")
    print(f"  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
