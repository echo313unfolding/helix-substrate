#!/usr/bin/env python3
"""
Cloud GPU Benchmark — HelixLinear at scale.

Runs on a rented GPU (RTX 4090 / A100 / etc). Tests:
  Phase A: Load compressed model via HelixLinear, verify tensor count
  Phase B: Perplexity on WikiText-2 (compressed vs HF baseline if available)
  Phase C: Inference speed (prefill + decode tok/s)
  Phase D: Generation quality (3 coding prompts, 3 general prompts)
  Phase E: Memory profile (VRAM peak, host RAM)

Emits WO-RECEIPT-COST-01 compliant receipt.

Usage:
    python3 tools/cloud_bench_run.py --model-dir /workspace/models/qwen2.5-7b-instruct
"""

import argparse
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

# ── Project path ──
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from helix_substrate.helix_linear import (
    HelixLinear,
    load_cdna_factors,
    swap_to_helix,
    swap_summary,
)


def jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def get_gpu_info():
    """Collect GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024),
        "capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
        "driver": torch.version.cuda,
    }


def load_model_compressed(model_dir: Path, device="cuda"):
    """Load HF model shell + swap in HelixLinear from CDNA v3."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print("  Loading model shell...")
    t0 = time.time()
    # Try loading normally first; if no weights exist (cloud deploy ships CDNA only),
    # initialize empty shell from config.
    has_weights = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "model.safetensors.index.json").exists()
        or (model_dir / "pytorch_model.bin").exists()
    )
    if has_weights:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        # No dense weights — create empty shell from config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        # Materialize on CPU with empty tensors (HelixLinear swap will fill them)
        model = model.to_empty(device="cpu")
        model = model.to(torch.float32)
    shell_time = time.time() - t0
    print(f"  Shell loaded: {shell_time:.1f}s (weights={'found' if has_weights else 'empty/CDNA-only'})")

    # Load CDNA factors and swap
    cdna_dir = model_dir / "cdnav3"
    print(f"  Loading CDNA v3 factors from {cdna_dir}...")
    t0 = time.time()
    factors = load_cdna_factors(cdna_dir, model=model)
    load_time = time.time() - t0
    print(f"  Factors loaded: {len(factors)} tensors, {load_time:.1f}s")

    print("  Swapping to HelixLinear...")
    t0 = time.time()
    model = swap_to_helix(model, factors)
    swap_time = time.time() - t0

    summary = swap_summary(model)
    print(f"  Swap complete: {summary['helix_modules']} HelixLinear, "
          f"{summary['linear_modules']} nn.Linear remaining, {swap_time:.1f}s")

    # Cast remaining non-HelixLinear params to float32 (avoids dtype mismatch
    # when HelixLinear outputs float32 but embeddings/norms/lm_head are bfloat16)
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()

    # Move to GPU
    print(f"  Moving to {device}...")
    t0 = time.time()
    model = model.to(device).eval()
    move_time = time.time() - t0

    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print(f"  On GPU: {vram_mb:.0f} MB VRAM, {move_time:.1f}s")

    return model, tokenizer, {
        "shell_time_s": round(shell_time, 2),
        "factor_load_time_s": round(load_time, 2),
        "swap_time_s": round(swap_time, 2),
        "move_time_s": round(move_time, 2),
        "helix_modules": summary["helix_modules"],
        "linear_modules": summary["linear_modules"],
        "vram_mb": round(vram_mb, 1),
        "compressed_mb": round(summary.get("compressed_bytes", 0) / 1024 / 1024, 1),
        "dense_equivalent_mb": round(summary.get("dense_equivalent_bytes", 0) / 1024 / 1024, 1),
    }


def load_model_dense(model_dir: Path, device="cuda", dtype=torch.bfloat16):
    """Load dense HF model for baseline comparison."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading dense baseline ({dtype})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device).eval()
    load_time = time.time() - t0

    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print(f"  Dense loaded: {vram_mb:.0f} MB VRAM, {load_time:.1f}s")
    return model, tokenizer, {"vram_mb": round(vram_mb, 1), "load_time_s": round(load_time, 2), "dtype": str(dtype)}


# ── Phase B: Perplexity ──

def eval_perplexity(model, tokenizer, device="cuda", n_tokens=8192, seq_len=2048):
    """Compute perplexity on WikiText-2 test set."""
    from datasets import load_dataset

    print("  Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens + 1)
    input_ids = encodings.input_ids[:, :n_tokens + 1].to(device)

    nlls = []
    n_evaluated = 0

    print(f"  Evaluating PPL (seq_len={seq_len}, n_tokens={n_tokens})...")
    with torch.no_grad():
        for i in range(0, input_ids.shape[1] - 1, seq_len):
            end = min(i + seq_len + 1, input_ids.shape[1])
            chunk = input_ids[:, i:end]
            if chunk.shape[1] < 2:
                break

            outputs = model(input_ids=chunk[:, :-1])
            logits = outputs.logits

            # Compute loss on CPU to avoid OOM with large vocab
            shift_logits = logits.float().cpu()
            shift_labels = chunk[:, 1:].cpu()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            n_tok = shift_labels.numel()
            nlls.append(loss.item() * n_tok)
            n_evaluated += n_tok

            if n_evaluated >= n_tokens:
                break

    ppl = np.exp(sum(nlls) / n_evaluated)
    print(f"  PPL = {ppl:.4f} ({n_evaluated} tokens)")
    return {"ppl": round(float(ppl), 4), "n_tokens": n_evaluated, "seq_len": seq_len}


# ── Phase C: Inference Speed ──

def bench_inference_speed(model, tokenizer, device="cuda", n_warmup=2, n_runs=5):
    """Benchmark prefill and decode speeds."""
    # Short prompt for decode speed
    short_prompt = "def fibonacci(n):\n    "
    # Long prompt for prefill speed
    long_prompt = "Explain the theory of general relativity in detail, covering spacetime curvature, " * 20

    results = {}

    for label, prompt, max_tokens in [
        ("decode_short", short_prompt, 128),
        ("decode_long", short_prompt, 256),
        ("prefill", long_prompt, 1),
    ]:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[1]

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=4, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        gen_tokens = []
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_gen = out.shape[1] - prompt_len
            times.append(elapsed)
            gen_tokens.append(n_gen)

        avg_time = np.mean(times)
        avg_tokens = np.mean(gen_tokens)
        tok_s = avg_tokens / avg_time if avg_time > 0 else 0

        results[label] = {
            "prompt_tokens": prompt_len,
            "avg_gen_tokens": round(float(avg_tokens), 1),
            "avg_time_s": round(float(avg_time), 3),
            "tok_s": round(float(tok_s), 1),
            "n_runs": n_runs,
        }
        print(f"  {label}: {tok_s:.1f} tok/s ({avg_tokens:.0f} tokens in {avg_time:.2f}s)")

    # VRAM peak
    vram_peak = 0
    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    results["vram_peak_mb"] = round(vram_peak, 1)
    print(f"  VRAM peak: {vram_peak:.0f} MB")

    return results


# ── Phase D: Generation Quality ──

PROMPTS = [
    {"name": "sieve", "text": "Write a Python function for the Sieve of Eratosthenes with type hints.", "max_tokens": 256},
    {"name": "binary_search_fix", "text": "Fix this buggy binary search:\ndef binary_search(arr, target):\n    low, high = 0, len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid\n        else: high = mid\n    return -1", "max_tokens": 256},
    {"name": "explain_attention", "text": "Explain multi-head attention in transformers in 3 sentences.", "max_tokens": 128},
    {"name": "sql_query", "text": "Write a SQL query to find the top 5 customers by total order value, including their name and email.", "max_tokens": 128},
    {"name": "refactor", "text": "Refactor this to use list comprehension:\ndef get_squares(nums):\n    result = []\n    for n in nums:\n        if n > 0:\n            result.append(n**2)\n    return result", "max_tokens": 128},
    {"name": "explain_compression", "text": "What is vector quantization in the context of neural network compression? Keep it under 100 words.", "max_tokens": 128},
]


def bench_generation_quality(model, tokenizer, device="cuda"):
    """Run qualitative prompts and capture outputs."""
    results = []

    for p in PROMPTS:
        input_ids = tokenizer(p["text"], return_tensors="pt").input_ids.to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=p["max_tokens"],
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.perf_counter() - t0
        gen_ids = out[0, input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_tok = len(gen_ids)

        results.append({
            "name": p["name"],
            "n_tokens": n_tok,
            "time_s": round(elapsed, 2),
            "tok_s": round(n_tok / elapsed, 1) if elapsed > 0 else 0,
            "output_preview": text[:500],
        })
        print(f"  {p['name']}: {n_tok} tok, {n_tok/elapsed:.1f} tok/s")

    return results


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Cloud GPU HelixLinear Benchmark")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--model-name", type=str, default="qwen2.5-7b-instruct", help="Model name for receipt")
    parser.add_argument("--output-dir", type=str, default="receipts/cloud_bench", help="Output directory")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense baseline (saves time + VRAM)")
    parser.add_argument("--ppl-tokens", type=int, default=8192, help="Number of tokens for PPL eval")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    gpu_info = get_gpu_info()

    print("=" * 60)
    print(f"Helix Cloud Benchmark — {args.model_name}")
    print(f"GPU: {gpu_info.get('name', 'N/A')} ({gpu_info.get('total_mb', '?')} MB)")
    print("=" * 60)

    receipt = {
        "benchmark": "cloud_helix_linear",
        "model_name": args.model_name,
        "gpu": gpu_info,
        "system": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        },
    }

    # ── Phase A: Load Compressed ──
    print(f"\n{'='*60}")
    print("[Phase A] Load HelixLinear model")
    print("=" * 60)
    model, tokenizer, load_info = load_model_compressed(model_dir, device=args.device)
    receipt["phase_a_load"] = load_info

    # Smoke test
    print("  Smoke test...")
    test_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(args.device)
    with torch.no_grad():
        out = model(test_ids)
    assert out.logits.shape[-1] > 0, "Smoke test failed"
    assert not torch.isnan(out.logits).any(), "NaN in output"
    print("  Smoke test passed.")

    # ── Phase B: Perplexity ──
    print(f"\n{'='*60}")
    print("[Phase B] Perplexity (WikiText-2)")
    print("=" * 60)
    ppl_compressed = eval_perplexity(model, tokenizer, device=args.device, n_tokens=args.ppl_tokens)
    receipt["phase_b_ppl_compressed"] = ppl_compressed

    # Dense baseline (optional)
    if not args.skip_dense:
        print("\n  --- Dense baseline ---")
        # Free compressed model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            dense_model, _, dense_load = load_model_dense(model_dir, device=args.device)
            ppl_dense = eval_perplexity(dense_model, tokenizer, device=args.device, n_tokens=args.ppl_tokens)
            receipt["phase_b_ppl_dense"] = ppl_dense
            receipt["phase_b_dense_load"] = dense_load

            ppl_delta = (ppl_compressed["ppl"] - ppl_dense["ppl"]) / ppl_dense["ppl"] * 100
            receipt["phase_b_ppl_delta_pct"] = round(ppl_delta, 2)
            print(f"\n  PPL delta: {ppl_delta:+.2f}% (compressed {ppl_compressed['ppl']:.4f} vs dense {ppl_dense['ppl']:.4f})")

            del dense_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"  Dense baseline failed: {e}")
            receipt["phase_b_dense_error"] = str(e)

        # Reload compressed for remaining phases
        print("\n  Reloading compressed model...")
        model, tokenizer, _ = load_model_compressed(model_dir, device=args.device)

    # ── Phase C: Inference Speed ──
    print(f"\n{'='*60}")
    print("[Phase C] Inference Speed")
    print("=" * 60)
    speed_results = bench_inference_speed(model, tokenizer, device=args.device)
    receipt["phase_c_speed"] = speed_results

    # ── Phase D: Generation Quality ──
    print(f"\n{'='*60}")
    print("[Phase D] Generation Quality")
    print("=" * 60)
    quality_results = bench_generation_quality(model, tokenizer, device=args.device)
    receipt["phase_d_quality"] = quality_results

    # ── Phase E: Memory Profile ──
    print(f"\n{'='*60}")
    print("[Phase E] Memory Profile")
    print("=" * 60)
    mem = {
        "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1) if torch.cuda.is_available() else 0,
        "vram_reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 1) if torch.cuda.is_available() else 0,
        "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1) if torch.cuda.is_available() else 0,
        "host_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    receipt["phase_e_memory"] = mem
    print(f"  VRAM allocated: {mem['vram_allocated_mb']} MB")
    print(f"  VRAM peak:      {mem['vram_peak_mb']} MB")
    print(f"  Host RSS:       {mem['host_rss_mb']} MB")

    # ── Cost block ──
    receipt["cost"] = {
        "wall_time_s": round(time.time() - t_total, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": mem["host_rss_mb"],
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
    }

    # ── Save receipt ──
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = output_dir / f"cloud_bench_{args.model_name}_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, default=jsonable)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model:          {args.model_name}")
    print(f"  GPU:            {gpu_info.get('name', 'N/A')}")
    print(f"  HelixLinear:    {load_info['helix_modules']} modules")
    print(f"  VRAM:           {load_info['vram_mb']} MB loaded, {mem['vram_peak_mb']} MB peak")
    print(f"  Compression:    {load_info['dense_equivalent_mb']:.0f} → {load_info['compressed_mb']:.0f} MB "
          f"({load_info['dense_equivalent_mb']/max(load_info['compressed_mb'],1):.2f}x)")
    print(f"  PPL:            {ppl_compressed['ppl']:.4f}")
    if "phase_b_ppl_delta_pct" in receipt:
        print(f"  PPL delta:      {receipt['phase_b_ppl_delta_pct']:+.2f}%")
    print(f"  Decode:         {speed_results.get('decode_short', {}).get('tok_s', '?')} tok/s (short)")
    print(f"  Wall time:      {receipt['cost']['wall_time_s']:.0f}s")
    print(f"  Receipt:        {receipt_path}")


if __name__ == "__main__":
    main()
