#!/usr/bin/env python3
"""
Step 4: Per-block sensitivity mapping.

Compresses each of TinyLlama's 22 blocks individually, measures perplexity
delta vs uncompressed baseline on WikiText-2 validation. Builds a
compression_budget_map[block_idx] = sensitivity_tier.

Uses the routed policy (get_policy with kurtosis) from Step 3.

Usage:
    python tools/step4_sensitivity_map.py
    python tools/step4_sensitivity_map.py --tokens 8192   # more tokens, slower
"""

import argparse
import hashlib
import json
import platform
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from safetensors import safe_open
from scipy.stats import kurtosis as scipy_kurtosis
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.cdnav3_reader import CDNAv3Reader
from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.tensor_policy import get_policy

MODEL_DIR = Path.home() / "models" / "tinyllama_fp32"
MODEL_PATH = MODEL_DIR / "model.safetensors"
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "step4_sensitivity"

# All 2D weight tensors per block
BLOCK_TENSORS = [
    ("q_proj", "model.layers.{i}.self_attn.q_proj.weight"),
    ("k_proj", "model.layers.{i}.self_attn.k_proj.weight"),
    ("v_proj", "model.layers.{i}.self_attn.v_proj.weight"),
    ("o_proj", "model.layers.{i}.self_attn.o_proj.weight"),
    ("gate_proj", "model.layers.{i}.mlp.gate_proj.weight"),
    ("up_proj", "model.layers.{i}.mlp.up_proj.weight"),
    ("down_proj", "model.layers.{i}.mlp.down_proj.weight"),
]

GGUF_NAMES = {
    "q_proj": "blk.{i}.attn_q.weight",
    "k_proj": "blk.{i}.attn_k.weight",
    "v_proj": "blk.{i}.attn_v.weight",
    "o_proj": "blk.{i}.attn_output.weight",
    "gate_proj": "blk.{i}.ffn_gate.weight",
    "up_proj": "blk.{i}.ffn_up.weight",
    "down_proj": "blk.{i}.ffn_down.weight",
}

N_BLOCKS = 22


def get_module(model, block_idx, tensor_type):
    """Get the nn.Linear module for a tensor in a block."""
    layer = model.model.layers[block_idx]
    if tensor_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, tensor_type)
    return getattr(layer.mlp, tensor_type)


def compute_perplexity(model, eval_tokens, seq_len=2048):
    """Compute perplexity on pre-tokenized input IDs. Returns (ppl, nll, n_tokens)."""
    model.eval()
    nlls = []
    n_tokens = 0

    for i in range(0, len(eval_tokens) - 1, seq_len):
        end = min(i + seq_len, len(eval_tokens))
        input_ids = torch.tensor(
            eval_tokens[i:end], dtype=torch.long
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        chunk_tokens = input_ids.shape[1] - 1
        nlls.append(outputs.loss.item() * chunk_tokens)
        n_tokens += chunk_tokens

        if end >= len(eval_tokens):
            break

    mean_nll = sum(nlls) / n_tokens
    return float(np.exp(mean_nll)), mean_nll, n_tokens


def compress_block(block_idx):
    """Compress all 2D tensors in a block. Returns list of (tensor_type, W_hat, info)."""
    results = []
    with safe_open(str(MODEL_PATH), framework="numpy") as f:
        for tensor_type, hf_pattern in BLOCK_TENSORS:
            hf_key = hf_pattern.format(i=block_idx)
            W = f.get_tensor(hf_key)
            cdna_name = GGUF_NAMES[tensor_type].format(i=block_idx)

            kurt = float(scipy_kurtosis(W.ravel(), fisher=True))
            policy = get_policy(hf_key, W.shape, kurtosis=kurt)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                writer = CDNAv3Writer(tmpdir)
                stats = writer.write_tensor(W, cdna_name, policy=policy)

                safe = cdna_name.replace("/", "_").replace(".", "_")
                tensor_dir = tmpdir / f"{safe}.cdnav3"

                if stats.get("storage_mode") == "exact":
                    W_hat = W.copy()
                else:
                    reader = CDNAv3Reader(tensor_dir)
                    W_hat = reader.reconstruct()

            cos = float(np.dot(W.ravel(), W_hat.ravel()) / (
                np.linalg.norm(W.ravel()) * np.linalg.norm(W_hat.ravel()) + 1e-30
            ))

            results.append((tensor_type, W_hat, {
                "kurtosis": round(kurt, 2),
                "svd_rank": policy.svd_residual_rank,
                "compressed_bytes": stats.get("compressed_bytes", W.nbytes),
                "original_bytes": int(W.nbytes),
                "weight_cosine": round(cos, 6),
            }))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096,
                        help="Number of eval tokens (default: 4096, ~2 chunks)")
    args = parser.parse_args()

    t_wall_start = time.time()
    t_cpu_start = time.process_time()
    ts_start = datetime.now(timezone.utc).isoformat()

    print("Step 4: Per-Block Sensitivity Mapping")
    print("=" * 90)

    # Load tokenizer + eval data
    print("Loading tokenizer and WikiText-2 validation...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    all_tokens = tokenizer.encode(text)
    eval_tokens = all_tokens[:args.tokens]
    n_chunks = (len(eval_tokens) - 1) // 2048 + 1
    print(f"  Eval tokens: {len(eval_tokens)} ({n_chunks} chunks from {len(all_tokens)} total)")

    # Load model
    print("Loading TinyLlama FP32...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.float32)
    model.eval()

    # Baseline perplexity
    print("Computing baseline perplexity...", flush=True)
    t0 = time.time()
    baseline_ppl, baseline_nll, n_eval = compute_perplexity(model, eval_tokens)
    baseline_s = time.time() - t0
    print(f"  Baseline: ppl={baseline_ppl:.4f}  nll={baseline_nll:.6f}  "
          f"({n_eval} tokens, {baseline_s:.1f}s)\n")

    # Per-block sweep
    all_results = []
    for block_idx in range(N_BLOCKS):
        print(f"  [{block_idx + 1}/{N_BLOCKS}] Block {block_idx}...", end=" ", flush=True)

        # Compress
        t0 = time.time()
        compressed = compress_block(block_idx)
        t_compress = time.time() - t0

        # Swap weights
        originals = {}
        with torch.no_grad():
            for tensor_type, W_hat, _ in compressed:
                mod = get_module(model, block_idx, tensor_type)
                originals[tensor_type] = mod.weight.data.clone()
                mod.weight.data = torch.from_numpy(W_hat).float()

        # Eval
        t0 = time.time()
        block_ppl, block_nll, _ = compute_perplexity(model, eval_tokens)
        t_eval = time.time() - t0

        # Restore
        with torch.no_grad():
            for tensor_type, orig in originals.items():
                mod = get_module(model, block_idx, tensor_type)
                mod.weight.data = orig

        delta_ppl = block_ppl - baseline_ppl
        delta_pct = (delta_ppl / baseline_ppl) * 100

        total_orig = sum(info["original_bytes"] for _, _, info in compressed)
        total_comp = sum(info["compressed_bytes"] for _, _, info in compressed)
        n_svd = sum(1 for _, _, info in compressed if info["svd_rank"] > 0)
        mean_cos = np.mean([info["weight_cosine"] for _, _, info in compressed])

        result = {
            "block_idx": block_idx,
            "block_ppl": round(block_ppl, 4),
            "delta_ppl": round(delta_ppl, 4),
            "delta_pct": round(delta_pct, 4),
            "original_bytes": total_orig,
            "compressed_bytes": total_comp,
            "compression_ratio": round(total_orig / total_comp, 2),
            "mean_weight_cosine": round(float(mean_cos), 6),
            "n_tensors": len(compressed),
            "n_svd_upgraded": n_svd,
            "compress_s": round(t_compress, 1),
            "eval_s": round(t_eval, 1),
            "tensors": {
                ttype: info for ttype, _, info in compressed
            },
        }

        marker = ""
        if abs(delta_ppl) < 0.05:
            marker = " (negligible)"
        elif delta_ppl < 0:
            marker = " (IMPROVED)"
        elif delta_pct > 2.0:
            marker = " (CRITICAL)"
        elif delta_pct > 1.0:
            marker = " (SENSITIVE)"

        print(f"ppl={block_ppl:.4f}  delta={delta_ppl:+.4f} ({delta_pct:+.2f}%)  "
              f"cos={mean_cos:.5f}  ratio={total_orig / total_comp:.2f}x  "
              f"svd={n_svd}  ({t_compress:.0f}s+{t_eval:.0f}s){marker}")

        all_results.append(result)

    # Summary table
    print(f"\n{'=' * 100}")
    print(f"  PER-BLOCK SENSITIVITY MAP (baseline ppl={baseline_ppl:.4f})")
    print(f"{'=' * 100}")
    print(f"  {'blk':>3}  {'ppl':>10}  {'delta':>10}  {'delta%':>8}  "
          f"{'w_cos':>8}  {'ratio':>6}  {'svd':>3}  {'tier':>12}")
    print(f"  {'-' * 3}  {'-' * 10}  {'-' * 10}  {'-' * 8}  "
          f"{'-' * 8}  {'-' * 6}  {'-' * 3}  {'-' * 12}")

    for r in all_results:
        if abs(r["delta_ppl"]) < 0.05:
            tier = "negligible"
        elif r["delta_ppl"] < 0:
            tier = "IMPROVED"
        elif r["delta_pct"] < 0.5:
            tier = "low"
        elif r["delta_pct"] < 1.0:
            tier = "moderate"
        elif r["delta_pct"] < 2.0:
            tier = "HIGH"
        else:
            tier = "CRITICAL"

        print(f"  {r['block_idx']:>3}  {r['block_ppl']:>10.4f}  "
              f"{r['delta_ppl']:>+10.4f}  {r['delta_pct']:>+7.2f}%  "
              f"{r['mean_weight_cosine']:>8.5f}  {r['compression_ratio']:>6.2f}  "
              f"{r['n_svd_upgraded']:>3}  {tier:>12}")

    # Aggregate stats
    deltas = [r["delta_ppl"] for r in all_results]
    deltas_pct = [r["delta_pct"] for r in all_results]

    print(f"\n  Summary:")
    print(f"    Baseline perplexity: {baseline_ppl:.4f}")
    print(f"    Mean delta: {np.mean(deltas):+.4f} ({np.mean(deltas_pct):+.2f}%)")
    print(f"    Max delta:  block {np.argmax(deltas)} = {np.max(deltas):+.4f} "
          f"({np.max(deltas_pct):+.2f}%)")
    print(f"    Min delta:  block {np.argmin(deltas)} = {np.min(deltas):+.4f} "
          f"({np.min(deltas_pct):+.2f}%)")

    # Build budget map
    budget_map = {}
    for r in all_results:
        if r["delta_pct"] < 0 or abs(r["delta_ppl"]) < 0.05:
            budget_map[str(r["block_idx"])] = "aggressive"
        elif r["delta_pct"] < 0.5:
            budget_map[str(r["block_idx"])] = "standard"
        elif r["delta_pct"] < 1.0:
            budget_map[str(r["block_idx"])] = "conservative"
        else:
            budget_map[str(r["block_idx"])] = "minimal"

    tiers = {}
    for idx, tier in budget_map.items():
        tiers.setdefault(tier, []).append(int(idx))
    print(f"\n  Compression budget tiers:")
    for tier in ["aggressive", "standard", "conservative", "minimal"]:
        blocks = sorted(tiers.get(tier, []))
        if blocks:
            print(f"    {tier:>12}: blocks {blocks}")

    # Cost block
    cost = {
        "wall_time_s": round(time.time() - t_wall_start, 3),
        "cpu_time_s": round(time.process_time() - t_cpu_start, 3),
        "peak_memory_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": ts_start,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
    }

    # Receipt
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "step4_sensitivity_map:v1",
        "model": str(MODEL_PATH),
        "eval_dataset": "wikitext-2-raw-v1/validation",
        "eval_tokens": len(eval_tokens),
        "n_chunks": n_chunks,
        "baseline_perplexity": round(baseline_ppl, 4),
        "baseline_nll": round(baseline_nll, 6),
        "n_blocks": N_BLOCKS,
        "results": all_results,
        "budget_map": budget_map,
        "tiers": {k: sorted(v) for k, v in tiers.items()},
        "cost": cost,
    }

    ts_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"sensitivity_map_{ts_tag}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    receipt["receipt_sha256"] = hashlib.sha256(
        receipt_path.read_text().encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2))

    print(f"\nReceipt: {receipt_path}")
    print(f"Cost: {cost['wall_time_s']:.1f}s wall, {cost['peak_memory_mb']:.0f}MB peak")


if __name__ == "__main__":
    main()
