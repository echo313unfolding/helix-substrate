#!/usr/bin/env python3
"""
Cloud 5-Config Comparison Benchmark — RTX 4090
===============================================
Head-to-head comparison of 5 compression configs on Qwen2.5-7B-Instruct:

  1. FP16 Dense (baseline)
  2. HelixLinear k=256 (4.0x)
  3. HelixLinear k=64 (5.3x information-theoretic)
  4. GPTQ Int4 (~8x)
  5. AWQ Int4 (~8x)

Per config: PPL (WikiText-2, 8192 tokens), decode tok/s (128 tokens × 5 runs),
prefill tok/s (512 tokens × 5 runs), VRAM, load time.

Emits WO-RECEIPT-COST-01 compliant comparison receipt.

Usage:
    python3 tools/cloud_comparison_bench.py \\
        --model-dir /workspace/models/qwen2.5-7b-instruct \\
        --output-dir /workspace/receipts
"""

import argparse
import gc
import json
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from helix_substrate.helix_linear import (
    load_cdna_factors,
    swap_to_helix,
    swap_summary,
)


# ── Utilities ──

def jsonable(obj):
    """JSON serializer for numpy/torch types."""
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


def make_cost_block(t_start, cpu_start, start_iso):
    """Build a WO-RECEIPT-COST-01 cost block."""
    return {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
    }


def write_receipt(receipt, output_dir, filename):
    """Write a JSON receipt to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, default=jsonable)
    print(f"  Receipt written: {path}")
    return path


def gpu_cleanup():
    """Free GPU memory between configs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_vram_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
    return 0


def get_vram_peak_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    return 0


# ── Shared evaluation ──

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
    model.eval()
    with torch.no_grad():
        for i in range(0, input_ids.shape[1] - 1, seq_len):
            end = min(i + seq_len + 1, input_ids.shape[1])
            chunk = input_ids[:, i:end]
            if chunk.shape[1] < 2:
                break

            outputs = model(input_ids=chunk[:, :-1])
            logits = outputs.logits

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
    return {"ppl": round(float(ppl), 4), "n_tokens": n_evaluated}


def bench_decode_speed(model, tokenizer, device="cuda", n_warmup=2, n_runs=5, max_tokens=128):
    """Benchmark autoregressive decode speed."""
    prompt = "def fibonacci(n):\n    "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

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
            out = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_gen = out.shape[1] - prompt_len
        times.append(elapsed)
        gen_tokens.append(n_gen)

    avg_time = float(np.mean(times))
    avg_tokens = float(np.mean(gen_tokens))
    tok_s = avg_tokens / avg_time if avg_time > 0 else 0

    return {
        "prompt_tokens": prompt_len,
        "avg_gen_tokens": round(avg_tokens, 1),
        "avg_time_s": round(avg_time, 3),
        "decode_tok_s": round(tok_s, 1),
        "n_runs": n_runs,
    }


def bench_prefill_speed(model, tokenizer, device="cuda", n_warmup=2, n_runs=5, prefill_tokens=512):
    """Benchmark prefill (prompt processing) speed."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=prefill_tokens)
    input_ids = encodings.input_ids.to(device)
    actual_len = input_ids.shape[1]

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(input_ids[:, :min(32, actual_len)])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_time = float(np.mean(times))
    tok_s = actual_len / avg_time if avg_time > 0 else 0

    return {
        "prefill_tokens": actual_len,
        "avg_time_s": round(avg_time, 4),
        "prefill_tok_s": round(tok_s, 1),
        "n_runs": n_runs,
    }


# ── Config runners ──

def run_config_dense_fp16(model_dir, device="cuda"):
    """Config 1: FP16 Dense baseline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n  --- Config 1: FP16 Dense ---")
    gpu_cleanup()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    load_time = round(time.time() - t0, 2)

    vram = get_vram_mb()
    print(f"  Loaded: {vram:.0f} MB VRAM, {load_time}s")

    ppl = eval_perplexity(model, tokenizer, device=device)
    decode = bench_decode_speed(model, tokenizer, device=device)
    prefill = bench_prefill_speed(model, tokenizer, device=device)
    vram_peak = get_vram_peak_mb()

    del model
    gpu_cleanup()

    return {
        "config": "FP16 Dense",
        "ratio": "1x",
        "load_time_s": load_time,
        "vram_mb": vram,
        "vram_peak_mb": vram_peak,
        **ppl,
        **decode,
        **prefill,
    }, tokenizer


def run_config_helix(model_dir, cdna_subdir, config_name, ratio_label, device="cuda"):
    """Config 2/3: HelixLinear compressed (k=256 or k=64)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  --- {config_name} ---")
    gpu_cleanup()

    cdna_dir = model_dir / cdna_subdir

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    # Load model shell
    has_weights = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "model.safetensors.index.json").exists()
    )
    if has_weights:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to_empty(device="cpu").to(torch.float32)

    # Load + swap CDNA
    factors = load_cdna_factors(cdna_dir, model=model)
    model = swap_to_helix(model, factors)
    summary = swap_summary(model)

    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()

    model = model.to(device).eval()
    load_time = round(time.time() - t0, 2)

    vram = get_vram_mb()
    print(f"  Loaded: {vram:.0f} MB VRAM, {load_time}s, "
          f"{summary['helix_modules']} HelixLinear")

    ppl = eval_perplexity(model, tokenizer, device=device)
    decode = bench_decode_speed(model, tokenizer, device=device)
    prefill = bench_prefill_speed(model, tokenizer, device=device)
    vram_peak = get_vram_peak_mb()

    del model
    gpu_cleanup()

    return {
        "config": config_name,
        "ratio": ratio_label,
        "load_time_s": load_time,
        "vram_mb": vram,
        "vram_peak_mb": vram_peak,
        "helix_modules": summary["helix_modules"],
        "linear_modules": summary["linear_modules"],
        **ppl,
        **decode,
        **prefill,
    }, tokenizer


def run_config_quantized(model_dir, quant_model_id, config_name, ratio_label, device="cuda"):
    """Config 4/5: GPTQ or AWQ from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  --- {config_name} ---")
    gpu_cleanup()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    # Try loading the quantized model
    # GPTQ and AWQ are auto-detected by transformers when quantization_config is in config.json
    quant_path = Path.home() / "models" / quant_model_id.replace("/", "_")
    if quant_path.exists():
        load_path = str(quant_path)
    else:
        load_path = quant_model_id

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    load_time = round(time.time() - t0, 2)

    vram = get_vram_mb()
    print(f"  Loaded: {vram:.0f} MB VRAM, {load_time}s")

    ppl = eval_perplexity(model, tokenizer, device=device)
    decode = bench_decode_speed(model, tokenizer, device=device)
    prefill = bench_prefill_speed(model, tokenizer, device=device)
    vram_peak = get_vram_peak_mb()

    del model
    gpu_cleanup()

    return {
        "config": config_name,
        "ratio": ratio_label,
        "load_time_s": load_time,
        "vram_mb": vram,
        "vram_peak_mb": vram_peak,
        **ppl,
        **decode,
        **prefill,
    }, tokenizer


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="5-Config Cloud Comparison Benchmark")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Path to model directory (e.g. Qwen2.5-14B-Instruct)")
    parser.add_argument("--output-dir", type=Path, default=Path("receipts"),
                        help="Where to write receipts")
    parser.add_argument("--gptq-model", type=str, default=None,
                        help="GPTQ model ID or local path (auto-detected if not set)")
    parser.add_argument("--awq-model", type=str, default=None,
                        help="AWQ model ID or local path (auto-detected if not set)")
    parser.add_argument("--skip-gptq", action="store_true")
    parser.add_argument("--skip-awq", action="store_true")
    args = parser.parse_args()

    # Auto-detect model name from config.json for GPTQ/AWQ lookup
    config_path = args.model_dir / "config.json"
    model_name = args.model_dir.name  # e.g. "qwen2.5-14b-instruct"
    if config_path.exists():
        _cfg = json.load(open(config_path))
        n_layers = _cfg.get("num_hidden_layers") or _cfg.get("n_layer", "?")
    else:
        n_layers = "?"

    # Auto-detect GPTQ/AWQ model IDs from model dir name
    # Maps common dir names to HF repo IDs
    _QUANT_MAP = {
        "qwen2.5-7b-instruct": ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
        "qwen2.5-14b-instruct": ("Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", "Qwen/Qwen2.5-14B-Instruct-AWQ"),
    }
    default_gptq, default_awq = _QUANT_MAP.get(model_name, (None, None))
    if args.gptq_model is None:
        args.gptq_model = default_gptq
    if args.awq_model is None:
        args.awq_model = default_awq

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print(f"  CLOUD 5-CONFIG COMPARISON BENCHMARK — {model_name} ({n_layers} layers)")
    print("=" * 70)
    print(f"  Model dir: {args.model_dir}")
    print(f"  GPU: {get_gpu_info()}")
    print()

    results = []

    # Config 1: FP16 Dense
    try:
        r1, tokenizer = run_config_dense_fp16(args.model_dir)
        results.append(r1)
    except Exception as e:
        print(f"  ERROR Config 1: {e}")
        traceback.print_exc()
        results.append({"config": "FP16 Dense", "error": str(e)})

    # Config 2: HelixLinear k=256
    try:
        r2, _ = run_config_helix(args.model_dir, "cdnav3",
                                  "HelixLinear k=256", "4.0x")
        results.append(r2)
    except Exception as e:
        print(f"  ERROR Config 2: {e}")
        traceback.print_exc()
        results.append({"config": "HelixLinear k=256", "error": str(e)})

    # Config 3: HelixLinear k=64
    cdna_k64 = args.model_dir / "cdnav3_k64"
    if cdna_k64.exists():
        try:
            r3, _ = run_config_helix(args.model_dir, "cdnav3_k64",
                                      "HelixLinear k=64", "5.3x*")
            results.append(r3)
        except Exception as e:
            print(f"  ERROR Config 3: {e}")
            traceback.print_exc()
            results.append({"config": "HelixLinear k=64", "error": str(e)})
    else:
        print(f"\n  SKIP Config 3: {cdna_k64} not found")
        results.append({"config": "HelixLinear k=64", "error": "cdnav3_k64/ not found"})

    # Config 4: GPTQ Int4
    if not args.skip_gptq and args.gptq_model is not None:
        try:
            r4, _ = run_config_quantized(args.model_dir, args.gptq_model,
                                          "GPTQ Int4", "~8x")
            results.append(r4)
        except Exception as e:
            print(f"  ERROR Config 4: {e}")
            traceback.print_exc()
            results.append({"config": "GPTQ Int4", "error": str(e)})
    else:
        reason = "--skip-gptq" if args.skip_gptq else "no GPTQ model ID"
        print(f"\n  SKIP Config 4: {reason}")
        results.append({"config": "GPTQ Int4", "skipped": True})

    # Config 5: AWQ Int4
    if not args.skip_awq and args.awq_model is not None:
        try:
            r5, _ = run_config_quantized(args.model_dir, args.awq_model,
                                          "AWQ Int4", "~8x")
            results.append(r5)
        except Exception as e:
            print(f"  ERROR Config 5: {e}")
            traceback.print_exc()
            results.append({"config": "AWQ Int4", "error": str(e)})
    else:
        reason = "--skip-awq" if args.skip_awq else "no AWQ model ID"
        print(f"\n  SKIP Config 5: {reason}")
        results.append({"config": "AWQ Int4", "skipped": True})

    # ── Comparison Table ──
    print(f"\n{'=' * 90}")
    print(f"  5-CONFIG COMPARISON TABLE — {model_name} ({n_layers} layers)")
    print(f"{'=' * 90}")

    header = f"  {'Config':<22} {'Ratio':>6} {'PPL':>8} {'Decode':>8} {'Prefill':>9} {'VRAM':>7} {'Load':>6}"
    print(header)
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*7} {'-'*6}")

    baseline_ppl = None
    for r in results:
        if "error" in r or "skipped" in r:
            print(f"  {r['config']:<22} {'—':>6} {'ERROR' if 'error' in r else 'SKIP':>8}")
            continue
        if baseline_ppl is None:
            baseline_ppl = r.get("ppl", 0)
        ppl = r.get("ppl", 0)
        decode = r.get("decode_tok_s", 0)
        prefill = r.get("prefill_tok_s", 0)
        vram = r.get("vram_mb", 0)
        load_t = r.get("load_time_s", 0)
        print(f"  {r['config']:<22} {r.get('ratio', '?'):>6} {ppl:>8.2f} "
              f"{decode:>7.1f} {prefill:>8.1f} {vram:>6.0f} {load_t:>5.1f}s")

    if baseline_ppl and baseline_ppl > 0:
        print(f"\n  PPL deltas vs FP16 Dense ({baseline_ppl:.2f}):")
        for r in results:
            if "error" in r or "skipped" in r:
                continue
            ppl = r.get("ppl", 0)
            if ppl > 0:
                delta = (ppl - baseline_ppl) / baseline_ppl * 100
                print(f"    {r['config']:<22} {delta:+.2f}%")

    print(f"\n  * 5.3x is information-theoretic (6 bits vs 32 bits).")
    print(f"    On-disk same as k=256 with uint8 indices.")

    # ── Receipt ──
    receipt = {
        "work_order": "WO-CLOUD-COMPARISON-01",
        "question": "How does HelixLinear (k=256, k=64) compare to GPTQ/AWQ on RTX 4090?",
        "model": model_name,
        "n_layers": n_layers,
        "gpu": get_gpu_info(),
        "configs": results,
        "cost": make_cost_block(t_start, cpu_start, start_iso),
    }

    ts = time.strftime("%Y%m%dT%H%M%S")
    write_receipt(receipt, args.output_dir / "cloud_comparison",
                  f"cloud_comparison_{ts}.json")

    total_wall = time.time() - t_start
    print(f"\n  Total wall time: {total_wall:.0f}s ({total_wall/60:.1f} min)")
    print("=" * 90)


if __name__ == "__main__":
    main()
