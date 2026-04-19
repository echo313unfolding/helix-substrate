#!/usr/bin/env python3
"""
PPL evaluation for HXQ compressed models via from_pretrained().

Replaces eval_ppl_cpu.py for models in HF format (safetensors + quantization_config).
Supports GPU, BF16, and mixed-codec models automatically.

Usage:
    # GPU (recommended — fixes WO-CPU-FORWARD-BUG)
    python3 tools/eval_ppl.py --model ~/models/tinyllama-1.1b-helix --device cuda

    # CPU fallback
    python3 tools/eval_ppl.py --model ~/models/tinyllama-1.1b-helix --device cpu

    # HuggingFace Hub model
    python3 tools/eval_ppl.py --model EchoLabs33/zamba2-1.2b-helix --device cuda

    # With dense baseline comparison
    python3 tools/eval_ppl.py --model ~/models/tinyllama-1.1b-helix \
        --dense ~/models/tinyllama_fp32 --device cuda
"""

import argparse
import gc
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def eval_ppl(model, tokenizer, device="cpu", n_tokens=8192, seq_len=2048):
    """Evaluate perplexity on WikiText-2 test set."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens + 1)
    ids = enc.input_ids[:, :n_tokens + 1].to(device)

    nlls, n_eval = [], 0
    model.eval()
    n_chunks = (ids.shape[1] - 1 + seq_len - 1) // seq_len
    chunk_i = 0

    with torch.no_grad():
        for i in range(0, ids.shape[1] - 1, seq_len):
            end = min(i + seq_len + 1, ids.shape[1])
            chunk = ids[:, i:end]
            if chunk.shape[1] < 2:
                break
            chunk_i += 1
            t0 = time.time()
            out = model(input_ids=chunk[:, :-1])
            logits = out.logits.float()
            labels = chunk[:, 1:]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            n = labels.numel()
            nlls.append(loss.item() * n)
            n_eval += n
            elapsed = time.time() - t0
            ppl_so_far = round(float(np.exp(sum(nlls) / n_eval)), 4)
            print(f"  Chunk {chunk_i}/{n_chunks}: {n} tokens, "
                  f"loss={loss.item():.4f}, PPL={ppl_so_far}, {elapsed:.1f}s",
                  file=sys.stderr, flush=True)
            if n_eval >= n_tokens:
                break

    return round(float(np.exp(sum(nlls) / n_eval)), 4), n_eval


def count_helix_modules(model) -> int:
    """Count HelixLinear modules in a model."""
    count = 0
    for m in model.modules():
        if type(m).__name__ == "HelixLinear":
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="PPL evaluation for HXQ compressed models (from_pretrained path)"
    )
    parser.add_argument("--model", required=True,
                        help="Path to HXQ model dir or HuggingFace Hub ID")
    parser.add_argument("--dense", default=None,
                        help="Path to dense baseline model (for delta comparison)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device (default: auto = cuda if available)")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "bfloat16", "float16"],
                        help="Compute dtype (default: auto = bf16 on CUDA, fp32 on CPU)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--output", type=str, default=None,
                        help="Write receipt JSON to this path")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Resolve dtype
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}[args.dtype]

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    results = {}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── HXQ compressed model ──
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  HXQ PPL EVAL (from_pretrained path)", file=sys.stderr)
    print(f"  Model:  {args.model}", file=sys.stderr)
    print(f"  Device: {device}, Dtype: {dtype}", file=sys.stderr)
    print(f"  Tokens: {args.n_tokens}, Seq len: {args.seq_len}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    # Register HXQ quantizer
    import helix_substrate.hf_quantizer  # noqa: F401

    print(f"  Loading compressed model...", file=sys.stderr, flush=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    n_helix = count_helix_modules(model)
    load_time = round(time.time() - t_load, 1)
    print(f"  Loaded: {n_helix} HelixLinear modules, {load_time}s", file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"\n  Evaluating HXQ PPL...", file=sys.stderr, flush=True)
    t_ppl = time.time()
    helix_ppl, n_tok = eval_ppl(model, tokenizer, device=device,
                                 n_tokens=args.n_tokens, seq_len=args.seq_len)
    ppl_time = round(time.time() - t_ppl, 1)
    print(f"\n  HXQ PPL = {helix_ppl} ({n_tok} tokens, {ppl_time}s)",
          file=sys.stderr, flush=True)

    results["helix"] = {
        "ppl": helix_ppl,
        "n_tokens": n_tok,
        "helix_modules": n_helix,
        "device": device,
        "dtype": str(dtype),
        "load_time_s": load_time,
        "eval_time_s": ppl_time,
    }

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Dense baseline ──
    if args.dense:
        print(f"\n  Loading dense baseline: {args.dense}", file=sys.stderr, flush=True)
        t_load = time.time()
        dense_model = AutoModelForCausalLM.from_pretrained(
            args.dense, torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if device == "cuda":
            dense_model = dense_model.cuda()
        dense_model.eval()
        load_time = round(time.time() - t_load, 1)

        dense_tokenizer = AutoTokenizer.from_pretrained(args.dense, trust_remote_code=True)

        print(f"\n  Evaluating Dense PPL...", file=sys.stderr, flush=True)
        t_ppl = time.time()
        dense_ppl, n_tok = eval_ppl(dense_model, dense_tokenizer, device=device,
                                     n_tokens=args.n_tokens, seq_len=args.seq_len)
        ppl_time = round(time.time() - t_ppl, 1)
        print(f"\n  Dense PPL = {dense_ppl} ({n_tok} tokens, {ppl_time}s)",
              file=sys.stderr, flush=True)

        results["dense"] = {
            "ppl": dense_ppl,
            "n_tokens": n_tok,
            "device": device,
            "dtype": str(dtype),
            "load_time_s": load_time,
            "eval_time_s": ppl_time,
        }

        delta = (helix_ppl - dense_ppl) / dense_ppl * 100
        results["delta_pct"] = round(delta, 2)
        print(f"\n  Delta: {delta:+.2f}%", file=sys.stderr, flush=True)

        del dense_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Cost block
    wall = round(time.time() - t_start, 3)
    cost = {
        "wall_time_s": wall,
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    receipt = {
        "tool": "eval_ppl",
        "model": args.model,
        "eval_setup": {
            "seq_len": args.seq_len,
            "n_tokens": args.n_tokens,
            "device": device,
            "dtype": str(dtype),
            "dataset": "wikitext-2-raw-v1",
            "path": "from_pretrained (HXQ quantizer)",
        },
        "results": results,
        "cost": cost,
    }

    # Save receipt
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("receipts/ppl_eval")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_name = Path(args.model).name if "/" not in args.model else args.model.split("/")[-1]
        out_path = out_dir / f"ppl_{model_name}_{time.strftime('%Y%m%dT%H%M%S')}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\n  Receipt: {out_path}", file=sys.stderr)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
