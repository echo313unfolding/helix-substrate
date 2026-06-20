#!/usr/bin/env python3
"""
CDNA v3 Compression Benchmark — Standalone Reproducible Demo

Compresses TinyLlama 1.1B FP32 with the routed CDNA v3 pipeline,
measures perplexity on WikiText-2 validation, and reports all key metrics.

Usage:
    python tools/benchmark_cdnav3.py
    python tools/benchmark_cdnav3.py --tokens 8192      # more eval tokens
    python tools/benchmark_cdnav3.py --decode-bench      # include decode latency
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
from helix_substrate.tensor_policy import classify_tensor, get_policy

MODEL_DIR = Path.home() / "models" / "tinyllama_fp32"
MODEL_PATH = MODEL_DIR / "model.safetensors"
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "benchmarks"

N_BLOCKS = 22

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

SPECIAL_TENSORS = {
    "embed_tokens": ("model.embed_tokens.weight", "token_embd.weight"),
    "lm_head": ("lm_head.weight", "output.weight"),
}


def get_module(model, block_idx, tensor_type):
    layer = model.model.layers[block_idx]
    if tensor_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, tensor_type)
    return getattr(layer.mlp, tensor_type)


def compute_perplexity(model, eval_tokens, seq_len=2048):
    model.eval()
    nlls = []
    n_tokens = 0
    for i in range(0, len(eval_tokens) - 1, seq_len):
        end = min(i + seq_len, len(eval_tokens))
        input_ids = torch.tensor(eval_tokens[i:end], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        chunk_tokens = input_ids.shape[1] - 1
        nlls.append(outputs.loss.item() * chunk_tokens)
        n_tokens += chunk_tokens
        if end >= len(eval_tokens):
            break
    mean_nll = sum(nlls) / n_tokens
    return float(np.exp(mean_nll)), mean_nll, n_tokens


def compress_tensor(W, hf_key, cdna_name, tmpdir):
    """Compress one tensor via CDNA pipeline. Returns (W_hat, info, tensor_dir_or_None)."""
    kurt = float(scipy_kurtosis(W.ravel(), fisher=True))
    policy = get_policy(hf_key, W.shape, kurtosis=kurt)

    writer = CDNAv3Writer(tmpdir)
    stats = writer.write_tensor(W, cdna_name, policy=policy)

    safe = cdna_name.replace("/", "_").replace(".", "_")
    tensor_dir = tmpdir / f"{safe}.cdnav3"

    if stats.get("storage_mode") == "exact":
        W_hat = W.copy()
        td = None
    else:
        reader = CDNAv3Reader(tensor_dir)
        W_hat = reader.reconstruct()
        td = tensor_dir

    cos = float(np.dot(W.ravel(), W_hat.ravel()) / (
        np.linalg.norm(W.ravel()) * np.linalg.norm(W_hat.ravel()) + 1e-30
    ))

    return W_hat, {
        "compressed_bytes": stats.get("compressed_bytes", W.nbytes),
        "original_bytes": int(W.nbytes),
        "weight_cosine": round(cos, 6),
        "svd_rank": policy.svd_residual_rank,
        "storage_mode": stats.get("storage_mode", "codebook"),
    }, td


def benchmark_decode(tensor_dirs, n_iters=5):
    """Benchmark decode latency across a sample of compressed tensors."""
    if not tensor_dirs:
        return None

    # Sample up to 10 tensors
    sample = tensor_dirs[:10] if len(tensor_dirs) > 10 else tensor_dirs
    times_per_tensor = []

    for td in sample:
        reader = CDNAv3Reader(td)
        iter_times = []
        for _ in range(n_iters):
            r = CDNAv3Reader(td)  # fresh reader each time
            t0 = time.perf_counter()
            _ = r.reconstruct()
            iter_times.append(time.perf_counter() - t0)
        times_per_tensor.append(np.median(iter_times))

    return {
        "n_tensors_sampled": len(sample),
        "median_per_tensor_ms": round(float(np.median(times_per_tensor)) * 1000, 2),
        "mean_per_tensor_ms": round(float(np.mean(times_per_tensor)) * 1000, 2),
        "max_per_tensor_ms": round(float(np.max(times_per_tensor)) * 1000, 2),
        "total_model_estimate_ms": round(float(np.mean(times_per_tensor)) * 156 * 1000, 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="CDNA v3 Compression Benchmark — TinyLlama 1.1B"
    )
    parser.add_argument("--tokens", type=int, default=4096,
                        help="Eval tokens from WikiText-2 validation (default: 4096)")
    parser.add_argument("--decode-bench", action="store_true",
                        help="Include decode latency benchmark (slower)")
    args = parser.parse_args()

    t_wall_start = time.time()
    t_cpu_start = time.process_time()
    ts_start = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("  CDNA v3 Compression Benchmark")
    print("  Model: TinyLlama 1.1B FP32")
    print("  Eval:  WikiText-2 validation")
    print("=" * 70)

    # Load eval data
    print("\nLoading eval data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:args.tokens]
    print(f"  Tokens: {len(eval_tokens)}")

    # Load model
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.float32)
    model.eval()

    # Baseline
    print("Computing baseline perplexity...", flush=True)
    t0 = time.time()
    baseline_ppl, baseline_nll, n_eval = compute_perplexity(model, eval_tokens)
    baseline_s = time.time() - t0
    print(f"  Baseline ppl: {baseline_ppl:.4f} ({baseline_s:.1f}s)")

    # Compress all tensors
    print("Compressing model...", flush=True)
    t_compress_start = time.time()

    total_orig = 0
    total_comp = 0
    n_svd = 0
    n_compressed = 0
    n_exact = 0
    tensor_dirs = []  # for decode bench

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)

        with safe_open(str(MODEL_PATH), framework="numpy") as f:
            # Block tensors
            for block_idx in range(N_BLOCKS):
                for tensor_type in BLOCK_TENSOR_TYPES:
                    hf_key = HF_PATTERNS[tensor_type].format(i=block_idx)
                    cdna_name = GGUF_NAMES[tensor_type].format(i=block_idx)
                    W = f.get_tensor(hf_key)

                    W_hat, info, td = compress_tensor(W, hf_key, cdna_name, tmpdir)

                    with torch.no_grad():
                        mod = get_module(model, block_idx, tensor_type)
                        mod.weight.data = torch.from_numpy(W_hat).float()

                    total_orig += info["original_bytes"]
                    total_comp += info["compressed_bytes"]
                    if info["svd_rank"] > 0:
                        n_svd += 1
                    if info["storage_mode"] == "exact":
                        n_exact += 1
                    else:
                        n_compressed += 1
                    if td is not None:
                        tensor_dirs.append(td)

                if (block_idx + 1) % 5 == 0 or block_idx == 0:
                    print(f"  Block {block_idx:>2} done "
                          f"({time.time() - t_compress_start:.0f}s)", flush=True)

            # Special tensors
            for name, (hf_key, cdna_name) in SPECIAL_TENSORS.items():
                if hf_key in f.keys():
                    W = f.get_tensor(hf_key)
                    W_hat, info, td = compress_tensor(W, hf_key, cdna_name, tmpdir)

                    with torch.no_grad():
                        if name == "embed_tokens":
                            model.model.embed_tokens.weight.data = torch.from_numpy(W_hat).float()
                        elif name == "lm_head":
                            model.lm_head.weight.data = torch.from_numpy(W_hat).float()

                    total_orig += info["original_bytes"]
                    total_comp += info["compressed_bytes"]
                    n_compressed += 1
                    if td is not None:
                        tensor_dirs.append(td)

            # Norm/exact tensors (pass-through)
            norm_bytes = 0
            for key in f.keys():
                already = any(
                    key == HF_PATTERNS[t].format(i=i)
                    for i in range(N_BLOCKS) for t in BLOCK_TENSOR_TYPES
                ) or any(key == hf for hf, _ in SPECIAL_TENSORS.values())
                if not already:
                    W = f.get_tensor(key)
                    norm_bytes += W.nbytes
                    total_orig += W.nbytes
                    total_comp += W.nbytes
                    n_exact += 1

        t_compress = time.time() - t_compress_start
        print(f"  Compression: {t_compress:.0f}s")

        # Compressed perplexity
        print("Computing compressed perplexity...", flush=True)
        t0 = time.time()
        comp_ppl, comp_nll, _ = compute_perplexity(model, eval_tokens)
        comp_s = time.time() - t0

        delta_ppl = comp_ppl - baseline_ppl
        delta_pct = (delta_ppl / baseline_ppl) * 100

        # Decode bench
        decode_stats = None
        if args.decode_bench:
            print("Benchmarking decode latency...", flush=True)
            decode_stats = benchmark_decode(tensor_dirs)

    # Report
    ratio = total_orig / total_comp
    savings_gb = (total_orig - total_comp) / 1e9
    savings_pct = (1 - total_comp / total_orig) * 100

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Model size (FP32):       {total_orig / 1e9:.2f} GB")
    print(f"  Compressed size:         {total_comp / 1e9:.2f} GB")
    print(f"  Compression ratio:       {ratio:.2f}x")
    print(f"  Savings:                 {savings_gb:.2f} GB ({savings_pct:.1f}%)")
    print(f"  Tensors compressed:      {n_compressed}")
    print(f"  Tensors exact:           {n_exact}")
    print(f"  SVD-upgraded:            {n_svd}")
    print(f"  ---")
    print(f"  Baseline perplexity:     {baseline_ppl:.4f}")
    print(f"  Compressed perplexity:   {comp_ppl:.4f}")
    print(f"  Delta:                   {delta_ppl:+.4f} ({delta_pct:+.2f}%)")
    print(f"  ---")
    print(f"  Compression time:        {t_compress:.0f}s")
    print(f"  Baseline eval time:      {baseline_s:.0f}s")
    print(f"  Compressed eval time:    {comp_s:.0f}s")
    print(f"  Eval overhead:           {comp_s - baseline_s:+.1f}s ({(comp_s/baseline_s - 1)*100:+.1f}%)")
    print(f"  Peak memory:             {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")

    if decode_stats:
        print(f"  ---")
        print(f"  Decode latency (median): {decode_stats['median_per_tensor_ms']:.1f} ms/tensor")
        print(f"  Decode latency (max):    {decode_stats['max_per_tensor_ms']:.1f} ms/tensor")
        print(f"  Full model estimate:     {decode_stats['total_model_estimate_ms']:.0f} ms")

    print(f"{'=' * 70}")

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
        "schema": "benchmark_cdnav3:v1",
        "model": str(MODEL_PATH),
        "model_size_bytes": total_orig,
        "compressed_size_bytes": total_comp,
        "compression_ratio": round(ratio, 2),
        "savings_pct": round(savings_pct, 2),
        "eval_dataset": "wikitext-2-raw-v1/validation",
        "eval_tokens": len(eval_tokens),
        "baseline_perplexity": round(baseline_ppl, 4),
        "compressed_perplexity": round(comp_ppl, 4),
        "delta_ppl": round(delta_ppl, 4),
        "delta_pct": round(delta_pct, 4),
        "n_tensors_compressed": n_compressed,
        "n_tensors_exact": n_exact,
        "n_svd_upgraded": n_svd,
        "compression_time_s": round(t_compress, 1),
        "baseline_eval_time_s": round(baseline_s, 1),
        "compressed_eval_time_s": round(comp_s, 1),
        "decode_stats": decode_stats,
        "cost": cost,
    }

    ts_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"benchmark_{ts_tag}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    receipt["receipt_sha256"] = hashlib.sha256(
        receipt_path.read_text().encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2))

    print(f"\nReceipt: {receipt_path}")


if __name__ == "__main__":
    main()
