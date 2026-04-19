#!/usr/bin/env python3
"""WO-DONOR-SURVEY-01 Step 4: Architecture Pattern Evaluation

Quick forward-pass evaluation of block placement patterns WITHOUT full training.
Measures:
  1. Initial compression quality (cosine after first compress) for each pattern
  2. Forward-pass loss on init (before any gradient steps)
  3. Cross-references with Zamba2's proven 5:1 mamba:hybrid ratio

Tests several patterns at the CURRENT d=768 scale (Phase 1 architecture).
Also evaluates what patterns would look like at Mamba2 donor scale.

Receipt includes cost block per WO-RECEIPT-COST-01.
"""

import json
import math
import os
import platform
import resource
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

t_start = time.time()
cpu_start = time.process_time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

# Add parent to path for echo_hybrid imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from echo_hybrid.config import EchoHybridConfig, EchoHybridModel
from echo_hybrid.train_phase1 import compress_all_linears, compress_linear

RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "donor_survey"
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


def compute_compress_quality(model: EchoHybridModel, n_clusters: int = 256, vector_dim: int = 1) -> dict:
    """Compress all linears and measure reconstruction quality."""
    compressed = compress_all_linears(model, n_clusters=n_clusters, vector_dim=vector_dim)

    layer_stats = []
    total_cos = 0.0
    total_mse = 0.0
    n_layers = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name == "lm_head":
            continue
        if name not in compressed:
            continue

        c = compressed[name]
        original = mod.weight.data.float().numpy()

        # Reconstruct
        codebook = c["codebook"].numpy()
        indices = c["indices"].numpy()
        reconstructed = codebook[indices.astype(np.int32)]

        # Apply sidecars
        if c["sidecar_positions"].numel() > 0:
            flat = reconstructed.ravel()
            pos = c["sidecar_positions"].numpy()
            val = c["sidecar_values"].numpy()
            valid = pos < flat.size
            flat[pos[valid]] = val[valid]
            reconstructed = flat.reshape(reconstructed.shape)

        # Metrics
        orig_flat = original.ravel().astype(np.float64)
        recon_flat = reconstructed.ravel().astype(np.float64)
        cos = float(np.dot(orig_flat, recon_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat) + 1e-10))
        mse = float(np.mean((orig_flat - recon_flat) ** 2))

        # Determine block type
        block_type = "unknown"
        for i, bt in enumerate(model.cfg.block_pattern):
            if f"blocks.{i}." in name:
                block_type = bt
                break

        layer_stats.append({
            "name": name,
            "block_type": block_type,
            "cosine": round(cos, 6),
            "mse": round(mse, 8),
            "utilization": round(c.get("codebook_utilization", c.get("n_active_centroids", 256)) / 256, 4)
                          if isinstance(c.get("codebook_utilization", c.get("n_active_centroids", 256)), (int, float))
                          else round(c["n_active_centroids"] / 256, 4),
        })
        total_cos += cos
        total_mse += mse
        n_layers += 1

    # Per-block-type aggregates
    ssm_cos = [s["cosine"] for s in layer_stats if s["block_type"] == "ssm"]
    attn_cos = [s["cosine"] for s in layer_stats if s["block_type"] == "attn"]

    return {
        "n_layers": n_layers,
        "mean_cosine": round(total_cos / max(n_layers, 1), 6),
        "mean_mse": round(total_mse / max(n_layers, 1), 8),
        "ssm_mean_cosine": round(float(np.mean(ssm_cos)), 6) if ssm_cos else None,
        "attn_mean_cosine": round(float(np.mean(attn_cos)), 6) if attn_cos else None,
        "per_layer": layer_stats,
    }


def eval_init_loss(model: EchoHybridModel, n_tokens: int = 1024, batch_size: int = 4) -> float:
    """Evaluate forward-pass loss on random tokens (measures init quality)."""
    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, model.cfg.vocab_size, (batch_size, n_tokens // batch_size))
        out = model(input_ids=input_ids, labels=input_ids)
        return out["loss"].item()


def main():
    print("WO-DONOR-SURVEY-01 Step 4: Architecture Pattern Evaluation")
    print("=" * 70)

    # Reference from Zamba2:
    # Zamba2-1.2B: 38 layers, 5.3:1 mamba:hybrid, hybrid every 6 layers
    # Zamba2-2.7B: 54 layers, 5.0:1 mamba:hybrid, hybrid every 6 layers
    # For our 9-block model: 5:1 ratio → ~1.5 attn blocks → test both 1 and 2 attn

    # Patterns to evaluate (name, block_pattern)
    patterns = [
        # === 2 attention blocks (current: 7:2 = 3.5:1 ratio) ===
        ("SSASSASSS", ["ssm","ssm","attn","ssm","ssm","attn","ssm","ssm","ssm"]),  # reference
        ("SSSASSSAS", ["ssm","ssm","ssm","attn","ssm","ssm","ssm","attn","ssm"]),  # Zamba2-like spacing
        ("ASSSASSSS", ["attn","ssm","ssm","ssm","attn","ssm","ssm","ssm","ssm"]),  # front-loaded
        ("SSSSSSSAA", ["ssm","ssm","ssm","ssm","ssm","ssm","ssm","attn","attn"]),  # back cluster

        # === 1 attention block (8:1 = 8:1 ratio, closer to Zamba2's 5:1) ===
        ("SSSSASSSS", ["ssm","ssm","ssm","ssm","attn","ssm","ssm","ssm","ssm"]),  # center
        ("SSSSSASSS", ["ssm","ssm","ssm","ssm","ssm","attn","ssm","ssm","ssm"]),  # slightly off-center

        # === 3 attention blocks (6:3 = 2:1 ratio, more attention-heavy) ===
        ("SASSASSAS", ["ssm","attn","ssm","ssm","attn","ssm","ssm","attn","ssm"]),  # even spacing
    ]

    print(f"\nTesting {len(patterns)} patterns")
    print(f"Reference: Zamba2 uses 5:1 mamba:hybrid ratio, evenly spaced\n")

    results = {}
    for i, (name, pattern) in enumerate(patterns):
        n_ssm = sum(1 for b in pattern if b == "ssm")
        n_attn = sum(1 for b in pattern if b == "attn")
        ratio = f"{n_ssm}:{n_attn}"
        attn_pos = [j for j, b in enumerate(pattern) if b == "attn"]

        print(f"[{i+1}/{len(patterns)}] {name} (ratio {ratio}, attn at {attn_pos})")

        cfg = EchoHybridConfig(block_pattern=pattern)
        model = EchoHybridModel(cfg)

        # 1. Init loss (random tokens, measures parameter initialization quality)
        init_loss = eval_init_loss(model)

        # 2. Compression quality at init
        compress_stats = compute_compress_quality(model)

        # 3. Param count
        n_params = model.n_params()
        n_params_nonemb = model.n_params(exclude_embeddings=True)

        results[name] = {
            "pattern": pattern,
            "n_ssm": n_ssm,
            "n_attn": n_attn,
            "ratio": ratio,
            "attn_positions": attn_pos,
            "init_loss": round(init_loss, 4),
            "n_params": n_params,
            "n_params_non_embedding": n_params_nonemb,
            "compress_quality": compress_stats,
        }

        print(f"  init_loss={init_loss:.4f}, compress_cos={compress_stats['mean_cosine']:.6f}, "
              f"ssm_cos={compress_stats.get('ssm_mean_cosine', 'N/A')}, "
              f"attn_cos={compress_stats.get('attn_mean_cosine', 'N/A')}")

        del model  # Free memory

    # Rank by init compression quality (mean cosine)
    ranked_compress = sorted(results.items(), key=lambda x: -x[1]["compress_quality"]["mean_cosine"])
    ranked_loss = sorted(results.items(), key=lambda x: x[1]["init_loss"])

    # Analysis: does attention count/placement affect compression?
    by_n_attn = {}
    for name, r in results.items():
        n = r["n_attn"]
        if n not in by_n_attn:
            by_n_attn[n] = []
        by_n_attn[n].append({
            "pattern": name,
            "compress_cos": r["compress_quality"]["mean_cosine"],
            "init_loss": r["init_loss"],
        })

    # Zamba2-closest pattern analysis
    # Zamba2 uses evenly-spaced hybrid blocks. In 9 blocks with 2 attn,
    # the Zamba2-like pattern would be SSSASSSAS (attn at 3, 7 — every 4th)
    zamba2_closest = "SSSASSSAS"

    receipt = {
        "work_order": "WO-DONOR-SURVEY-01",
        "step": "4_pattern_evaluation",
        "question": "Which block pattern gives the best initial compression quality?",
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "n_patterns_tested": len(patterns),
        "zamba2_reference": {
            "1.2b_ratio": "5.3:1 (32 mamba : 6 hybrid)",
            "1.2b_spacing": "every 6 layers",
            "2.7b_ratio": "5.0:1 (45 mamba : 9 hybrid)",
            "2.7b_spacing": "every 6 layers",
            "closest_9block": zamba2_closest,
        },
        "ranking_by_compress_quality": [
            {"rank": i+1, "pattern": name, "compress_cos": r["compress_quality"]["mean_cosine"],
             "init_loss": r["init_loss"], "ratio": r["ratio"]}
            for i, (name, r) in enumerate(ranked_compress)
        ],
        "ranking_by_init_loss": [
            {"rank": i+1, "pattern": name, "init_loss": r["init_loss"],
             "compress_cos": r["compress_quality"]["mean_cosine"], "ratio": r["ratio"]}
            for i, (name, r) in enumerate(ranked_loss)
        ],
        "by_attention_count": by_n_attn,
        "per_pattern": results,
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"pattern_eval_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\nReceipt written: {receipt_path}")

    # Summary
    print(f"\n{'='*80}")
    print("PATTERN RANKING (by compression quality)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Pattern':<12} {'Ratio':<8} {'Cos':<10} {'Init Loss':<10}")
    print(f"{'-'*5} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
    for i, (name, r) in enumerate(ranked_compress):
        marker = " ← Zamba2-like" if name == zamba2_closest else ""
        marker = " ← reference" if name == "SSASSASSS" and not marker else marker
        print(f"{i+1:<5} {name:<12} {r['ratio']:<8} {r['compress_quality']['mean_cosine']:<10.6f} "
              f"{r['init_loss']:<10.4f}{marker}")

    return receipt_path


if __name__ == "__main__":
    main()
