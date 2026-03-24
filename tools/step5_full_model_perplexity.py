#!/usr/bin/env python3
"""
Step 5: Full-model perplexity validation.

Compresses ALL blocks simultaneously using the routed policy, then measures
perplexity vs uncompressed baseline on WikiText-2 validation. This is the
end-to-end proof: do per-block negligible errors remain negligible when
all 22 blocks are compressed together?

Usage:
    python tools/step5_full_model_perplexity.py
    python tools/step5_full_model_perplexity.py --tokens 8192
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
import torch.nn.functional as F
from datasets import load_dataset
from safetensors import safe_open
from scipy.stats import kurtosis as scipy_kurtosis
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.cdnav3_reader import CDNAv3Reader
from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.tensor_policy import classify_tensor, get_policy

MODEL_DIR = Path.home() / "models" / "tinyllama_fp32"
MODEL_PATH = MODEL_DIR / "model.safetensors"
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "step5_full_model"

# All 2D weight tensors per block
BLOCK_TENSOR_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

HF_PATTERNS = {
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}

GGUF_NAMES = {
    "q_proj": "blk.{i}.attn_q.weight",
    "k_proj": "blk.{i}.attn_k.weight",
    "v_proj": "blk.{i}.attn_v.weight",
    "o_proj": "blk.{i}.attn_output.weight",
    "gate_proj": "blk.{i}.ffn_gate.weight",
    "up_proj": "blk.{i}.ffn_up.weight",
    "down_proj": "blk.{i}.ffn_down.weight",
}

# Non-block tensors
SPECIAL_TENSORS = {
    "embed_tokens": ("model.embed_tokens.weight", "token_embd.weight"),
    "lm_head": ("lm_head.weight", "output.weight"),
}

N_BLOCKS = 22


def get_module(model, block_idx, tensor_type):
    """Get the nn.Linear module for a tensor in a block."""
    layer = model.model.layers[block_idx]
    if tensor_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, tensor_type)
    return getattr(layer.mlp, tensor_type)


def compute_perplexity(model, eval_tokens, seq_len=2048):
    """Compute perplexity. Returns (ppl, nll, n_tokens)."""
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


def compute_logit_kl(model_a_logits, model_b_logits):
    """Compute KL(A || B) per position, return stats."""
    # model_a_logits, model_b_logits: list of [seq_len, vocab] tensors
    all_kl = []
    for logits_a, logits_b in zip(model_a_logits, model_b_logits):
        log_p = F.log_softmax(logits_a, dim=-1)
        log_q = F.log_softmax(logits_b, dim=-1)
        # KL(P || Q) = sum(P * (log_P - log_Q))
        kl = F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1)
        all_kl.append(kl)
    all_kl = torch.cat(all_kl)
    return {
        "mean": float(all_kl.mean()),
        "max": float(all_kl.max()),
        "median": float(all_kl.median()),
        "std": float(all_kl.std()),
        "p95": float(all_kl.quantile(0.95)),
        "p99": float(all_kl.quantile(0.99)),
        "n_tokens": int(all_kl.shape[0]),
    }


def collect_logits(model, eval_tokens, seq_len=2048):
    """Run forward pass and collect per-token logits."""
    model.eval()
    all_logits = []

    for i in range(0, len(eval_tokens) - 1, seq_len):
        end = min(i + seq_len, len(eval_tokens))
        input_ids = torch.tensor(
            eval_tokens[i:end], dtype=torch.long
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids)
        # logits: [1, seq_len, vocab] → [seq_len, vocab]
        all_logits.append(outputs.logits[0].float())

        if end >= len(eval_tokens):
            break

    return all_logits


def compress_tensor_cdna(W, hf_key, cdna_name):
    """Compress one tensor via CDNA pipeline. Returns (W_hat, stats, policy_info)."""
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

    return W_hat, {
        "hf_key": hf_key,
        "shape": list(W.shape),
        "kurtosis": round(kurt, 2),
        "svd_rank": policy.svd_residual_rank,
        "storage_mode": stats.get("storage_mode", "codebook"),
        "compressed_bytes": stats.get("compressed_bytes", W.nbytes),
        "original_bytes": int(W.nbytes),
        "weight_cosine": round(cos, 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096,
                        help="Number of eval tokens (default: 4096)")
    args = parser.parse_args()

    t_wall_start = time.time()
    t_cpu_start = time.process_time()
    ts_start = datetime.now(timezone.utc).isoformat()

    print("Step 5: Full-Model Perplexity Validation")
    print("=" * 90)

    # Load tokenizer + eval data
    print("Loading tokenizer and WikiText-2 validation...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    all_tokens = tokenizer.encode(text)
    eval_tokens = all_tokens[:args.tokens]
    n_chunks = (len(eval_tokens) - 1) // 2048 + 1
    print(f"  Eval tokens: {len(eval_tokens)} ({n_chunks} chunks)")

    # Load model
    print("Loading TinyLlama FP32...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.float32)
    model.eval()

    # Phase 1: Baseline
    print("\nPhase 1: Baseline perplexity...", flush=True)
    t0 = time.time()
    baseline_ppl, baseline_nll, n_eval = compute_perplexity(model, eval_tokens)
    baseline_s = time.time() - t0
    print(f"  Baseline: ppl={baseline_ppl:.4f}  nll={baseline_nll:.6f}  ({baseline_s:.1f}s)")

    print("  Collecting baseline logits...", flush=True)
    t0 = time.time()
    baseline_logits = collect_logits(model, eval_tokens)
    logit_s = time.time() - t0
    print(f"  Logits collected ({logit_s:.1f}s)")

    # Phase 2: Compress ALL tensors
    print(f"\nPhase 2: Compressing all tensors ({N_BLOCKS} blocks × 7 + specials)...", flush=True)
    t_compress_start = time.time()

    all_tensor_stats = []
    total_original = 0
    total_compressed = 0
    n_svd = 0
    n_exact = 0
    n_compressed = 0

    # Compress block tensors and swap into model
    with safe_open(str(MODEL_PATH), framework="numpy") as f:
        for block_idx in range(N_BLOCKS):
            block_stats = []
            for tensor_type in BLOCK_TENSOR_TYPES:
                hf_key = HF_PATTERNS[tensor_type].format(i=block_idx)
                cdna_name = GGUF_NAMES[tensor_type].format(i=block_idx)
                W = f.get_tensor(hf_key)

                W_hat, info = compress_tensor_cdna(W, hf_key, cdna_name)

                # Swap weight
                with torch.no_grad():
                    mod = get_module(model, block_idx, tensor_type)
                    mod.weight.data = torch.from_numpy(W_hat).float()

                total_original += info["original_bytes"]
                total_compressed += info["compressed_bytes"]
                if info["svd_rank"] > 0:
                    n_svd += 1
                if info["storage_mode"] == "exact":
                    n_exact += 1
                else:
                    n_compressed += 1

                block_stats.append(info)
                all_tensor_stats.append(info)

            # Progress
            n_done = (block_idx + 1) * 7
            elapsed = time.time() - t_compress_start
            svd_tag = f" (svd={n_svd})" if block_idx == 0 else ""
            print(f"  Block {block_idx:>2}: {n_done} tensors compressed "
                  f"({elapsed:.0f}s){svd_tag}", flush=True)

        # Compress special tensors (embed_tokens, lm_head)
        for name, (hf_key, cdna_name) in SPECIAL_TENSORS.items():
            if hf_key in f.keys():
                W = f.get_tensor(hf_key)
                W_hat, info = compress_tensor_cdna(W, hf_key, cdna_name)

                # Swap weight
                with torch.no_grad():
                    if name == "embed_tokens":
                        model.model.embed_tokens.weight.data = torch.from_numpy(W_hat).float()
                    elif name == "lm_head":
                        model.lm_head.weight.data = torch.from_numpy(W_hat).float()

                total_original += info["original_bytes"]
                total_compressed += info["compressed_bytes"]
                n_compressed += 1
                all_tensor_stats.append(info)
                print(f"  {name}: {info['original_bytes']:,} -> "
                      f"{info['compressed_bytes']:,} bytes "
                      f"(cos={info['weight_cosine']:.6f})", flush=True)

    # Add norm/bias tensors (exact, no compression — just count bytes)
    norm_bytes = 0
    with safe_open(str(MODEL_PATH), framework="numpy") as f:
        for key in f.keys():
            already_handled = any(
                key == HF_PATTERNS[t].format(i=i)
                for i in range(N_BLOCKS) for t in BLOCK_TENSOR_TYPES
            ) or any(key == hf for hf, _ in SPECIAL_TENSORS.values())

            if not already_handled:
                W = f.get_tensor(key)
                norm_bytes += W.nbytes
                total_original += W.nbytes
                total_compressed += W.nbytes  # exact = no compression
                n_exact += 1

    t_compress = time.time() - t_compress_start
    print(f"\n  Compression complete: {n_compressed} compressed, {n_exact} exact, "
          f"{n_svd} SVD-upgraded ({t_compress:.0f}s)")
    print(f"  Original:   {total_original:>12,} bytes ({total_original/1e9:.2f} GB)")
    print(f"  Compressed: {total_compressed:>12,} bytes ({total_compressed/1e9:.2f} GB)")
    print(f"  Ratio:      {total_original/total_compressed:.2f}x")
    print(f"  Norm/exact: {norm_bytes:>12,} bytes (passed through)")

    # Phase 3: Compressed model perplexity
    print(f"\nPhase 3: Compressed model perplexity...", flush=True)
    t0 = time.time()
    compressed_ppl, compressed_nll, _ = compute_perplexity(model, eval_tokens)
    compressed_s = time.time() - t0
    print(f"  Compressed: ppl={compressed_ppl:.4f}  nll={compressed_nll:.6f}  ({compressed_s:.1f}s)")

    delta_ppl = compressed_ppl - baseline_ppl
    delta_pct = (delta_ppl / baseline_ppl) * 100
    print(f"  Delta:      {delta_ppl:+.4f} ({delta_pct:+.3f}%)")

    # Phase 4: Logit KL divergence
    print(f"\nPhase 4: Logit KL divergence...", flush=True)
    t0 = time.time()
    compressed_logits = collect_logits(model, eval_tokens)
    kl_stats = compute_logit_kl(baseline_logits, compressed_logits)
    kl_s = time.time() - t0
    print(f"  KL(baseline || compressed):")
    print(f"    mean={kl_stats['mean']:.6f}  median={kl_stats['median']:.6f}  "
          f"max={kl_stats['max']:.6f}")
    print(f"    p95={kl_stats['p95']:.6f}  p99={kl_stats['p99']:.6f}  ({kl_s:.1f}s)")

    # Summary
    print(f"\n{'=' * 90}")
    print(f"  STEP 5 RESULTS: FULL-MODEL COMPRESSION")
    print(f"{'=' * 90}")
    print(f"  Baseline perplexity:     {baseline_ppl:.4f}")
    print(f"  Compressed perplexity:   {compressed_ppl:.4f}")
    print(f"  Delta:                   {delta_ppl:+.4f} ({delta_pct:+.3f}%)")
    print(f"  Compression ratio:       {total_original / total_compressed:.2f}x")
    print(f"  Original size:           {total_original / 1e9:.2f} GB")
    print(f"  Compressed size:         {total_compressed / 1e9:.2f} GB")
    print(f"  Savings:                 {(total_original - total_compressed) / 1e9:.2f} GB "
          f"({(1 - total_compressed / total_original) * 100:.1f}%)")
    print(f"  SVD-upgraded tensors:    {n_svd}")
    print(f"  KL divergence (mean):    {kl_stats['mean']:.6f}")
    print(f"  KL divergence (p99):     {kl_stats['p99']:.6f}")

    verdict = "UNKNOWN"
    if delta_pct < 0.5:
        verdict = "EXCELLENT — errors do NOT compound"
    elif delta_pct < 1.0:
        verdict = "GOOD — minor compounding, still safe"
    elif delta_pct < 2.0:
        verdict = "ACCEPTABLE — moderate compounding, may want stricter routing on final layers"
    elif delta_pct < 5.0:
        verdict = "MARGINAL — significant compounding, needs investigation"
    else:
        verdict = "FAIL — errors compound unacceptably"

    print(f"\n  VERDICT: {verdict}")
    print(f"{'=' * 90}")

    # Step 4 comparison
    step4_sum = sum([
        0.0000, -0.0020, 0.0024, -0.0028, -0.0009, -0.0012,
        0.0001, 0.0025, 0.0054, 0.0001, 0.0006, -0.0013,
        0.0026, 0.0055, 0.0002, -0.0004, 0.0026, -0.0027,
        -0.0005, -0.0001, 0.0068, 0.0290
    ])
    print(f"\n  Step 4 reference (sum of individual deltas): {step4_sum:+.4f}")
    print(f"  Step 5 actual delta:                          {delta_ppl:+.4f}")
    if abs(delta_ppl) > 0:
        compounding = delta_ppl / step4_sum if step4_sum != 0 else float('inf')
        print(f"  Compounding factor:                           {compounding:.2f}x")

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
        "schema": "step5_full_model_perplexity:v1",
        "model": str(MODEL_PATH),
        "eval_dataset": "wikitext-2-raw-v1/validation",
        "eval_tokens": len(eval_tokens),
        "n_chunks": n_chunks,
        "baseline_perplexity": round(baseline_ppl, 4),
        "baseline_nll": round(baseline_nll, 6),
        "compressed_perplexity": round(compressed_ppl, 4),
        "compressed_nll": round(compressed_nll, 6),
        "delta_ppl": round(delta_ppl, 4),
        "delta_pct": round(delta_pct, 4),
        "total_original_bytes": total_original,
        "total_compressed_bytes": total_compressed,
        "compression_ratio": round(total_original / total_compressed, 2),
        "savings_bytes": total_original - total_compressed,
        "savings_pct": round((1 - total_compressed / total_original) * 100, 2),
        "n_tensors_compressed": n_compressed,
        "n_tensors_exact": n_exact,
        "n_svd_upgraded": n_svd,
        "norm_bytes": norm_bytes,
        "logit_kl": kl_stats,
        "verdict": verdict,
        "step4_sum_of_deltas": round(step4_sum, 4),
        "compounding_factor": round(delta_ppl / step4_sum, 2) if step4_sum != 0 else None,
        "tensor_stats": all_tensor_stats,
        "cost": cost,
    }

    ts_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"full_model_{ts_tag}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    receipt["receipt_sha256"] = hashlib.sha256(
        receipt_path.read_text().encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2))

    print(f"\nReceipt: {receipt_path}")
    print(f"Cost: {cost['wall_time_s']:.1f}s wall, {cost['peak_memory_mb']:.0f}MB peak")


if __name__ == "__main__":
    main()
