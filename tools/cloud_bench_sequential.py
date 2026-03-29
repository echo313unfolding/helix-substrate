#!/usr/bin/env python3
"""
Sequential Cloud Benchmark — runs each config in a separate subprocess
to avoid OOM from VRAM not being freed between configs.

Usage:
    python3 tools/cloud_bench_sequential.py --model-dir /home/user/models/qwen2.5-7b-instruct
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def run_config(model_dir, config_id, extra_args=None):
    """Run a single config in a subprocess and return JSON result."""
    cmd = [
        sys.executable, "-c", SINGLE_CONFIG_SCRIPT,
        str(model_dir), config_id,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*70}")
    print(f"  RUNNING CONFIG: {config_id}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    print(result.stdout)
    if result.stderr:
        # Filter out progress bars and warnings
        for line in result.stderr.split('\n'):
            if any(skip in line for skip in ['Loading weights:', 'Generating', 'examples/s', '|']):
                continue
            if line.strip():
                print(f"  STDERR: {line}", file=sys.stderr)

    # Extract JSON result from last line of stdout
    for line in reversed(result.stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            try:
                data = json.loads(line)
                data['subprocess_wall_s'] = round(time.time() - t0, 1)
                return data
            except json.JSONDecodeError:
                pass

    return {"config": config_id, "error": f"exit code {result.returncode}", "stderr_tail": result.stderr[-500:] if result.stderr else ""}


SINGLE_CONFIG_SCRIPT = r'''
import gc, json, sys, time
import numpy as np
import torch
import torch.nn.functional as F

model_dir = sys.argv[1]
config_id = sys.argv[2]

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_vram():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024**2, 1)
    return 0

def get_vram_peak():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return 0

def eval_ppl(model, tokenizer, device="cuda", n_tokens=8192, seq_len=512):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens+1)
    ids = enc.input_ids[:, :n_tokens+1].to(device)
    nlls, n_eval = [], 0
    model.eval()
    with torch.no_grad():
        for i in range(0, ids.shape[1]-1, seq_len):
            end = min(i+seq_len+1, ids.shape[1])
            chunk = ids[:, i:end]
            if chunk.shape[1] < 2: break
            out = model(input_ids=chunk[:, :-1])
            logits = out.logits.float().cpu()
            labels = chunk[:, 1:].cpu()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            n = labels.numel()
            nlls.append(loss.item() * n)
            n_eval += n
            if n_eval >= n_tokens: break
    return round(float(np.exp(sum(nlls)/n_eval)), 4), n_eval

def bench_decode(model, tokenizer, device="cuda", n_runs=5, max_tokens=128):
    prompt = "def fibonacci(n):\n    "
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    plen = ids.shape[1]
    for _ in range(2):
        with torch.no_grad():
            model.generate(input_ids=ids, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()
    times, toks = [], []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter()-t0)
        toks.append(out.shape[1]-plen)
    avg_t = float(np.mean(times))
    avg_tok = float(np.mean(toks))
    return round(avg_tok/avg_t, 1) if avg_t > 0 else 0

def bench_prefill(model, tokenizer, device="cuda", n_runs=5, prefill_tokens=512):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=prefill_tokens)
    ids = enc.input_ids.to(device)
    alen = ids.shape[1]
    for _ in range(2):
        with torch.no_grad():
            model(ids[:, :min(32, alen)])
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(ids)
        torch.cuda.synchronize()
        times.append(time.perf_counter()-t0)
    avg_t = float(np.mean(times))
    return round(alen/avg_t, 1) if avg_t > 0 else 0

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.reset_peak_memory_stats()

result = {"config": config_id}

try:
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if config_id == "fp16_dense":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, trust_remote_code=True,
            low_cpu_mem_usage=True, device_map="auto",
        ).eval()
        result["ratio"] = "1x"

    elif config_id.startswith("helix_"):
        sys.path.insert(0, str(Path(__file__).parent if '__file__' in dir() else Path(model_dir).parent))
        # Find helix-substrate
        for p in [Path("/home/user/helix-substrate"), Path.home() / "helix-substrate"]:
            if p.exists():
                sys.path.insert(0, str(p))
                break
        from helix_substrate.helix_linear import load_cdna_factors, swap_to_helix, swap_summary

        cdna_subdir = "cdnav3" if "k256" in config_id else "cdnav3_k64"
        cdna_dir = Path(model_dir) / cdna_subdir

        if not cdna_dir.exists():
            result["error"] = f"{cdna_dir} not found"
            print(json.dumps(result))
            sys.exit(0)

        # Load factors standalone (biases from .npy.meta.json, no model needed)
        factors = load_cdna_factors(cdna_dir)
        result["helix_factors_loaded"] = len(factors)

        # Load model with device_map="auto" for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, trust_remote_code=True,
            low_cpu_mem_usage=True, device_map="auto",
        )

        model = swap_to_helix(model, factors)
        del factors
        gc.collect()
        summary = swap_summary(model)

        # Remove accelerate hooks and move to single GPU
        try:
            from accelerate.hooks import remove_hook_from_module
            for _, mod in model.named_modules():
                remove_hook_from_module(mod, recurse=False)
        except ImportError:
            pass
        if hasattr(model, 'hf_device_map'):
            delattr(model, 'hf_device_map')

        # Move each module individually, skipping meta tensors (they are unused placeholders)
        for name, param in list(model.named_parameters()):
            if param.device.type != "meta" and param.device.type != device:
                parts = name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                setattr(mod, parts[-1], torch.nn.Parameter(
                    param.data.to(device), requires_grad=False))
        for name, buf in list(model.named_buffers()):
            if buf is not None and buf.device.type != "meta" and buf.device.type != device:
                parts = name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                mod._buffers[parts[-1]] = buf.to(device)
        model.eval()

        result["helix_modules"] = summary["helix_modules"]
        result["ratio"] = "4.0x" if "k256" in config_id else "5.3x*"

    elif config_id == "gptq_int4":
        # Auto-detect model size from dir name, prefer local paths
        model_name = Path(model_dir).name.lower()
        if "14b" in model_name:
            gptq_id = str(Path.home() / "models/qwen14b-gptq")
            if not Path(gptq_id).exists():
                gptq_id = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
        else:
            gptq_id = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
        model = AutoModelForCausalLM.from_pretrained(
            gptq_id, torch_dtype=torch.float16, trust_remote_code=True,
            device_map="auto",
        ).eval()
        result["ratio"] = "~8x"
        result["source"] = gptq_id

    elif config_id == "awq_int4":
        model_name = Path(model_dir).name.lower()
        if "14b" in model_name:
            awq_id = str(Path.home() / "models/qwen14b-awq")
            if not Path(awq_id).exists():
                awq_id = "Qwen/Qwen2.5-14B-Instruct-AWQ"
        else:
            awq_id = "Qwen/Qwen2.5-7B-Instruct-AWQ"
        model = AutoModelForCausalLM.from_pretrained(
            awq_id, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto",
        ).eval()
        result["ratio"] = "~8x"
        result["source"] = awq_id

    load_time = round(time.time() - t0, 2)
    result["load_time_s"] = load_time
    result["vram_mb"] = get_vram()

    print(f"  Loaded: {result['vram_mb']:.0f} MB VRAM, {load_time}s", file=sys.stderr)

    # PPL
    ppl, n_tok = eval_ppl(model, tokenizer, device=device)
    result["ppl"] = ppl
    result["n_tokens"] = n_tok
    print(f"  PPL = {ppl} ({n_tok} tokens)", file=sys.stderr)

    # Decode speed
    decode_toks = bench_decode(model, tokenizer, device=device)
    result["decode_tok_s"] = decode_toks
    print(f"  Decode: {decode_toks} tok/s", file=sys.stderr)

    # Prefill speed
    prefill_toks = bench_prefill(model, tokenizer, device=device)
    result["prefill_tok_s"] = prefill_toks
    print(f"  Prefill: {prefill_toks} tok/s", file=sys.stderr)

    result["vram_peak_mb"] = get_vram_peak()

except Exception as e:
    import traceback
    result["error"] = str(e)
    traceback.print_exc(file=sys.stderr)

print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("receipts"))
    parser.add_argument("--skip-gptq", action="store_true")
    parser.add_argument("--skip-awq", action="store_true")
    parser.add_argument("--skip-k64", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    start_iso = datetime.now(timezone.utc).isoformat()
    t_start = time.time()

    configs = []
    if not args.skip_dense:
        configs.append("fp16_dense")
    configs.append("helix_k256")
    if not args.skip_k64:
        configs.append("helix_k64")
    if not args.skip_gptq:
        configs.append("gptq_int4")
    if not args.skip_awq:
        configs.append("awq_int4")

    print("=" * 70)
    print(f"  SEQUENTIAL CLOUD BENCHMARK — {args.model_dir.name}")
    print(f"  Configs: {', '.join(configs)}")
    print(f"  Each config runs in a separate process (no OOM between configs)")
    print("=" * 70)

    results = []
    for cfg in configs:
        r = run_config(args.model_dir, cfg)
        results.append(r)
        print(f"  -> {cfg}: {'OK' if 'error' not in r else 'ERROR: ' + r.get('error', '')[:80]}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"  COMPARISON TABLE — {args.model_dir.name}")
    print(f"{'='*90}")
    header = f"  {'Config':<22} {'Ratio':>6} {'PPL':>8} {'Decode':>8} {'Prefill':>9} {'VRAM':>7} {'Load':>6}"
    print(header)
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*7} {'-'*6}")

    baseline_ppl = None
    for r in results:
        if "error" in r:
            print(f"  {r['config']:<22} {'—':>6} {'ERROR':>8}")
            continue
        if baseline_ppl is None:
            baseline_ppl = r.get("ppl", 0)
        print(f"  {r['config']:<22} {r.get('ratio','?'):>6} {r.get('ppl',0):>8.2f} "
              f"{r.get('decode_tok_s',0):>7.1f} {r.get('prefill_tok_s',0):>8.1f} "
              f"{r.get('vram_mb',0):>6.0f} {r.get('load_time_s',0):>5.1f}s")

    if baseline_ppl and baseline_ppl > 0:
        print(f"\n  PPL deltas vs baseline ({baseline_ppl:.2f}):")
        for r in results:
            if "error" in r:
                continue
            ppl = r.get("ppl", 0)
            if ppl > 0:
                delta = (ppl - baseline_ppl) / baseline_ppl * 100
                verdict = "PASS" if abs(delta) < 2.0 else "FAIL"
                print(f"    {r['config']:<22} {delta:+.2f}% -> {verdict}")

    # Receipt
    receipt = {
        "work_order": "WO-CLOUD-COMPARISON-01",
        "question": "How does HelixLinear compare to GPTQ/AWQ on RTX 4090?",
        "model": args.model_dir.name,
        "gpu": "RTX 4090 24GB",
        "configs": results,
        "cost": {
            "wall_time_s": round(time.time() - t_start, 1),
            "timestamp_start": start_iso,
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
        },
    }

    out_dir = args.output_dir / "cloud_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = out_dir / f"cloud_comparison_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\n  Receipt: {receipt_path}")
    print(f"  Total wall: {time.time()-t_start:.0f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
