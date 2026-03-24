#!/usr/bin/env python3
"""
Dense baseline PPL measurement for Qwen2.5-3B-Instruct.

CPU-only, FP32. WikiText-2 validation set, 8192 tokens, seq_len=2048.
Emits WO-RECEIPT-COST-01 compliant receipt.

Usage:
    python3 tools/bench_qwen3b_dense_ppl.py
"""

import gc
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = Path(os.path.expanduser("~/models/qwen2.5-3b-instruct"))
RECEIPT_DIR = Path(__file__).resolve().parent.parent / "receipts" / "qwen3b_instruct"
N_TOKENS = 8192
SEQ_LEN = 2048
MODEL_NAME = "Qwen2.5-3B-Instruct"


def eval_perplexity_cpu(model, tokenizer, n_tokens=8192, seq_len=2048):
    """Compute perplexity on WikiText-2 validation set, CPU only."""
    from datasets import load_dataset

    print("  Loading WikiText-2 validation set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])

    print(f"  Tokenizing (requesting {n_tokens + 1} tokens)...")
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens + 1)
    input_ids = encodings.input_ids[:, : n_tokens + 1]
    actual_tokens = input_ids.shape[1] - 1
    print(f"  Got {actual_tokens} tokens for evaluation.")

    nlls = []
    n_evaluated = 0

    print(f"  Evaluating PPL (seq_len={seq_len}, n_tokens={n_tokens})...")
    with torch.no_grad():
        for i in range(0, input_ids.shape[1] - 1, seq_len):
            end = min(i + seq_len + 1, input_ids.shape[1])
            chunk = input_ids[:, i:end]
            if chunk.shape[1] < 2:
                break

            t0 = time.time()
            outputs = model(input_ids=chunk[:, :-1])
            logits = outputs.logits
            fwd_time = time.time() - t0

            # Loss on CPU (already on CPU, but .float() to ensure FP32)
            shift_logits = logits.float().cpu()
            shift_labels = chunk[:, 1:].cpu()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            n_tok = shift_labels.numel()
            nlls.append(loss.item() * n_tok)
            n_evaluated += n_tok

            chunk_ppl = np.exp(loss.item())
            elapsed_total = sum(nlls) / n_evaluated
            running_ppl = np.exp(elapsed_total)
            print(
                f"    Chunk {i//seq_len + 1}: {n_tok} tok, "
                f"chunk_ppl={chunk_ppl:.4f}, running_ppl={running_ppl:.4f}, "
                f"fwd={fwd_time:.1f}s"
            )

            # Free memory
            del outputs, logits, shift_logits, shift_labels
            gc.collect()

            if n_evaluated >= n_tokens:
                break

    avg_nll = sum(nlls) / n_evaluated
    ppl = np.exp(avg_nll)
    print(f"\n  FINAL PPL = {ppl:.4f} (nll={avg_nll:.6f}, {n_evaluated} tokens)")
    return {
        "ppl": round(float(ppl), 4),
        "nll": round(float(avg_nll), 6),
        "n_tokens": n_evaluated,
        "seq_len": seq_len,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 60)
    print(f"Dense Baseline PPL — {MODEL_NAME}")
    print(f"Device: CPU, dtype: float32")
    print(f"Tokens: {N_TOKENS}, seq_len: {SEQ_LEN}")
    print("=" * 60)

    # Load model
    print("\n[1/2] Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"  Parameters: {n_params:,} ({size_mb:.1f} MB FP32)")

    # Smoke test
    print("  Smoke test...")
    test_ids = tokenizer("Hello", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(test_ids)
    assert out.logits.shape[-1] == 151936, f"Unexpected vocab size: {out.logits.shape[-1]}"
    assert not torch.isnan(out.logits).any(), "NaN in smoke test output"
    print("  Smoke test passed.")
    del out
    gc.collect()

    # Evaluate PPL
    print(f"\n[2/2] Evaluating perplexity...")
    ppl_result = eval_perplexity_cpu(model, tokenizer, n_tokens=N_TOKENS, seq_len=SEQ_LEN)

    # Build receipt
    end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB

    receipt = {
        "work_order": "WO-QWEN3B-DENSE-BASELINE-01",
        "description": "Dense FP32 baseline PPL for Qwen2.5-3B-Instruct on WikiText-2 validation",
        "model": MODEL_NAME,
        "model_path": str(MODEL_DIR),
        "dtype": "float32",
        "device": "cpu",
        "n_params": n_params,
        "model_size_mb": round(size_mb, 1),
        "eval_dataset": "wikitext-2-raw-v1 (validation)",
        "dense_ppl": ppl_result["ppl"],
        "dense_nll": ppl_result["nll"],
        "n_tokens": ppl_result["n_tokens"],
        "seq_len": ppl_result["seq_len"],
        "load_time_s": round(load_time, 2),
        "compressed_ppl_reference": 5.5331,
        "ppl_delta_pct": round(
            (5.5331 - ppl_result["ppl"]) / ppl_result["ppl"] * 100, 4
        ),
        "cost": {
            "wall_time_s": round(time.time() - t_total, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(peak_mem, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": end_iso,
        },
    }

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"dense_baseline_ppl_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULT")
    print("=" * 60)
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Dense PPL:        {ppl_result['ppl']:.4f}")
    print(f"  Dense NLL:        {ppl_result['nll']:.6f}")
    print(f"  Tokens:           {ppl_result['n_tokens']}")
    print(f"  Compressed PPL:   5.5331 (reference)")
    print(f"  PPL delta:        {receipt['ppl_delta_pct']:+.4f}%")
    print(f"  Wall time:        {receipt['cost']['wall_time_s']:.0f}s")
    print(f"  Peak memory:      {receipt['cost']['peak_memory_mb']:.0f} MB")
    print(f"  Receipt:          {receipt_path}")


if __name__ == "__main__":
    main()
