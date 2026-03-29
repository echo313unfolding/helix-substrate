#!/usr/bin/env python3
"""
CPU PPL evaluation for HelixLinear models.

Usage:
    python3 tools/eval_ppl_cpu.py --model-dir ~/models/qwen2.5-7b-instruct \
        --seq-len 2048 --n-tokens 8192

Also evaluates FP16 dense baseline for comparison.
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from helix_substrate.rapl_meter import RaplMeter


def eval_ppl(model, tokenizer, device="cpu", n_tokens=8192, seq_len=2048):
    """Evaluate perplexity on WikiText-2."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--skip-dense", action="store_true",
                        help="Skip FP16 dense baseline (saves RAM and time)")
    args = parser.parse_args()

    device = "cpu"
    results = {}
    t_start = time.time()
    rapl = RaplMeter()
    rapl.__enter__()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # ── Helix (compressed) ──
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  HELIX PPL EVAL — {args.model_dir.name}", file=sys.stderr)
    print(f"  seq_len={args.seq_len}, n_tokens={args.n_tokens}, device={device}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helix_substrate.helix_linear import load_cdna_factors, swap_to_helix, swap_summary

    cdna_dir = args.model_dir / "cdnav3"
    if not cdna_dir.exists():
        print(f"ERROR: {cdna_dir} not found", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading model (FP32, CPU)...", file=sys.stderr, flush=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"  Loading factors from {cdna_dir}...", file=sys.stderr, flush=True)
    factors = load_cdna_factors(cdna_dir, model=model)
    print(f"  Loaded {len(factors)} factors", file=sys.stderr, flush=True)

    model = swap_to_helix(model, factors)
    del factors
    gc.collect()
    summary = swap_summary(model)
    model.eval()
    load_time = round(time.time() - t_load, 1)
    print(f"  Model loaded: {summary['helix_modules']} HelixLinear, {load_time}s",
          file=sys.stderr, flush=True)

    print(f"\n  Evaluating Helix PPL...", file=sys.stderr, flush=True)
    t_ppl = time.time()
    helix_ppl, n_tok = eval_ppl(model, tokenizer, device=device,
                                 n_tokens=args.n_tokens, seq_len=args.seq_len)
    ppl_time = round(time.time() - t_ppl, 1)
    print(f"\n  Helix PPL = {helix_ppl} ({n_tok} tokens, {ppl_time}s)",
          file=sys.stderr, flush=True)

    results["helix"] = {
        "ppl": helix_ppl,
        "n_tokens": n_tok,
        "helix_modules": summary["helix_modules"],
        "load_time_s": load_time,
        "eval_time_s": ppl_time,
    }

    # Free helix model
    del model
    gc.collect()

    # ── FP16 Dense baseline ──
    if not args.skip_dense:
        print(f"\n  Loading FP32 dense baseline...", file=sys.stderr, flush=True)
        t_load = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.float32, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        load_time = round(time.time() - t_load, 1)
        print(f"  Dense loaded: {load_time}s", file=sys.stderr, flush=True)

        print(f"\n  Evaluating Dense PPL...", file=sys.stderr, flush=True)
        t_ppl = time.time()
        dense_ppl, n_tok = eval_ppl(model, tokenizer, device=device,
                                     n_tokens=args.n_tokens, seq_len=args.seq_len)
        ppl_time = round(time.time() - t_ppl, 1)
        print(f"\n  Dense PPL = {dense_ppl} ({n_tok} tokens, {ppl_time}s)",
              file=sys.stderr, flush=True)

        results["dense"] = {
            "ppl": dense_ppl,
            "n_tokens": n_tok,
            "load_time_s": load_time,
            "eval_time_s": ppl_time,
        }

        delta = (helix_ppl - dense_ppl) / dense_ppl * 100
        results["delta_pct"] = round(delta, 2)
        print(f"\n  Delta: {delta:+.2f}%", file=sys.stderr, flush=True)

        del model
        gc.collect()

    # Receipt
    wall = round(time.time() - t_start, 1)
    rapl.__exit__(None, None, None)
    receipt = {
        "work_order": "WO-SCALING-PPL-01",
        "question": "7B PPL with embed_tokens=exact, lm_head=exact",
        "model": args.model_dir.name,
        "eval_setup": {
            "seq_len": args.seq_len,
            "n_tokens": args.n_tokens,
            "device": device,
            "dataset": "wikitext-2-raw-v1",
        },
        "results": results,
        "cost": {
            "wall_time_s": wall,
            "timestamp_start": datetime.utcnow().isoformat(),
        },
    }
    if rapl.available and rapl.joules is not None:
        receipt["cost"]["energy_joules"] = round(rapl.joules, 3)

    out_dir = Path("receipts/scaling_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = out_dir / f"ppl_eval_{args.model_dir.name}_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\n  Receipt: {receipt_path}", file=sys.stderr)
    print(f"  Wall time: {wall}s", file=sys.stderr)

    # Print JSON result on stdout for scripting
    print(json.dumps(results))


if __name__ == "__main__":
    main()
