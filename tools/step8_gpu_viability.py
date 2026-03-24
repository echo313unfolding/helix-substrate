#!/usr/bin/env python3
"""
Step 8: GPU Viability Benchmark — Quadro T2000 (4 GB VRAM)

Tests whether TinyLlama 1.1B with HelixLinear fits and runs on a 4 GB GPU.
Measures VRAM usage, latency, tokens/sec for both dense and HelixLinear paths.

Usage:
    python tools/step8_gpu_viability.py
    python tools/step8_gpu_viability.py --tokens 2048
    python tools/step8_gpu_viability.py --dense-comparison   # also test dense (may OOM)

Work Order: WO-GPU-VIABILITY-01
"""

import argparse
import gc
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

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR = Path.home() / "models" / "tinyllama_fp32"
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "step8_gpu_viability"

BLOCK_TENSOR_TYPES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

HF_PATTERNS = {
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}

N_BLOCKS = 22


def get_module(model, block_idx, tensor_type):
    layer = model.model.layers[block_idx]
    if tensor_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, tensor_type)
    return getattr(layer.mlp, tensor_type)


def gpu_mem_mb():
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def gpu_mem_peak_mb():
    """Peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def gpu_mem_reserved_mb():
    """GPU memory reserved by allocator in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0.0


def compute_perplexity(model, eval_tokens, seq_len=2048):
    """Compute perplexity on GPU. Returns (ppl, nll, n_tokens, actual_seq_len).
    Automatically reduces seq_len on OOM."""
    model.eval()
    device = next(model.parameters()).device

    # Try progressively shorter sequences if OOM
    for attempt_seq_len in [seq_len, seq_len // 2, seq_len // 4, 128, 64]:
        try:
            nlls = []
            n_tokens = 0

            for i in range(0, len(eval_tokens) - 1, attempt_seq_len):
                end = min(i + attempt_seq_len, len(eval_tokens))
                input_ids = torch.tensor(
                    eval_tokens[i:end], dtype=torch.long
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)

                chunk_tokens = input_ids.shape[1] - 1
                nlls.append(outputs.loss.item() * chunk_tokens)
                n_tokens += chunk_tokens

                # Free intermediates
                del outputs, input_ids
                torch.cuda.empty_cache()

                if end >= len(eval_tokens):
                    break

            mean_nll = sum(nlls) / n_tokens
            return float(np.exp(mean_nll)), mean_nll, n_tokens, attempt_seq_len

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at seq_len={attempt_seq_len}, trying shorter...", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            continue

    raise RuntimeError("OOM even at seq_len=64")


def measure_prompt_latency(model, eval_tokens, n_warmup=1, n_measure=3):
    """Measure prompt processing latency (prefill)."""
    device = next(model.parameters()).device
    model.eval()

    # Try progressively shorter sequences
    for seq_len in [min(len(eval_tokens), 512), 256, 128, 64]:
        try:
            input_ids = torch.tensor(
                eval_tokens[:seq_len], dtype=torch.long
            ).unsqueeze(0).to(device)

            # Warmup
            for _ in range(n_warmup):
                with torch.no_grad():
                    _ = model(input_ids)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Measure
            times = []
            for _ in range(n_measure):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(input_ids)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                del out
                torch.cuda.empty_cache()

            del input_ids
            torch.cuda.empty_cache()

            return {
                "seq_len": seq_len,
                "median_ms": round(float(np.median(times)) * 1000, 1),
                "mean_ms": round(float(np.mean(times)) * 1000, 1),
                "tokens_per_sec": round(seq_len / float(np.median(times)), 1),
            }
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at prompt seq_len={seq_len}, trying shorter...", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            continue

    return {"seq_len": 0, "median_ms": 0, "mean_ms": 0, "tokens_per_sec": 0, "error": "OOM"}


def measure_decode_latency(model, eval_tokens, n_tokens_gen=16, n_warmup=1, n_measure=3):
    """Measure autoregressive decode latency (token-by-token)."""
    device = next(model.parameters()).device
    model.eval()

    for prompt_len in [32, 16, 8]:
        for gen_tokens in [n_tokens_gen, 8, 4]:
            try:
                input_ids = torch.tensor(
                    eval_tokens[:prompt_len], dtype=torch.long
                ).unsqueeze(0).to(device)

                # Warmup
                for _ in range(n_warmup):
                    with torch.no_grad():
                        out = model.generate(input_ids, max_new_tokens=2, do_sample=False)
                    del out
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                # Measure
                times = []
                for _ in range(n_measure):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model.generate(
                            input_ids, max_new_tokens=gen_tokens, do_sample=False
                        )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    actual_new = out.shape[1] - input_ids.shape[1]
                    times.append((elapsed, actual_new))
                    del out
                    torch.cuda.empty_cache()

                del input_ids
                torch.cuda.empty_cache()

                median_time = float(np.median([t for t, _ in times]))
                median_tokens = int(np.median([n for _, n in times]))

                return {
                    "prompt_tokens": prompt_len,
                    "generated_tokens": median_tokens,
                    "total_ms": round(median_time * 1000, 1),
                    "ms_per_token": round(median_time / max(1, median_tokens) * 1000, 1),
                    "tokens_per_sec": round(median_tokens / median_time, 1),
                }
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM at prompt={prompt_len}, gen={gen_tokens}, trying smaller...",
                      flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                continue

    return {"prompt_tokens": 0, "generated_tokens": 0, "total_ms": 0,
            "ms_per_token": 0, "tokens_per_sec": 0, "error": "OOM"}


def main():
    parser = argparse.ArgumentParser(
        description="Step 8: GPU Viability Benchmark (Quadro T2000)"
    )
    parser.add_argument("--tokens", type=int, default=4096,
                        help="Eval tokens for perplexity (default: 4096)")
    parser.add_argument("--dense-comparison", action="store_true",
                        help="Also test dense model on GPU (may OOM)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.")
        sys.exit(1)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    print("=" * 70)
    print("  Step 8: GPU Viability Benchmark")
    print(f"  GPU: {gpu_name} ({gpu_vram:.0f} MB VRAM)")
    print(f"  Model: TinyLlama 1.1B")
    print("=" * 70)

    # --- Load eval tokens ---
    print("\n[1/6] Loading eval tokens...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:args.tokens]
    print(f"  Tokens: {len(eval_tokens)}")

    results = {
        "gpu": gpu_name,
        "gpu_vram_mb": round(gpu_vram),
    }

    # --- Dense comparison (optional) ---
    dense_result = None
    if args.dense_comparison:
        print("\n[2/6] Dense model GPU test...", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        try:
            model_dense = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR), dtype=torch.float32
            )
            model_dense.eval()

            mem_before = gpu_mem_mb()
            model_dense = model_dense.cuda()
            mem_after = gpu_mem_mb()
            mem_peak = gpu_mem_peak_mb()

            print(f"  Dense model loaded on GPU: {mem_after:.0f} MB "
                  f"(delta: {mem_after - mem_before:.0f} MB, peak: {mem_peak:.0f} MB)")

            # Perplexity
            print("  Computing dense perplexity on GPU...", flush=True)
            t0 = time.perf_counter()
            dense_ppl, dense_nll, _, _ = compute_perplexity(model_dense, eval_tokens)
            dense_ppl_time = time.perf_counter() - t0
            print(f"  Dense PPL: {dense_ppl:.4f} ({dense_ppl_time:.1f}s)")

            # Prompt latency
            print("  Measuring dense prompt latency...", flush=True)
            dense_prompt = measure_prompt_latency(model_dense, eval_tokens)
            print(f"  Dense prompt: {dense_prompt['tokens_per_sec']:.0f} tok/s "
                  f"({dense_prompt['median_ms']:.0f} ms for {dense_prompt['seq_len']} tokens)")

            # Decode latency
            print("  Measuring dense decode latency...", flush=True)
            dense_decode = measure_decode_latency(model_dense, eval_tokens)
            print(f"  Dense decode: {dense_decode['tokens_per_sec']:.1f} tok/s "
                  f"({dense_decode['ms_per_token']:.0f} ms/token)")

            dense_result = {
                "vram_model_mb": round(mem_after, 1),
                "vram_peak_mb": round(gpu_mem_peak_mb(), 1),
                "perplexity": round(dense_ppl, 4),
                "ppl_eval_time_s": round(dense_ppl_time, 1),
                "prompt_latency": dense_prompt,
                "decode_latency": dense_decode,
                "status": "OK",
            }

            # Cleanup
            del model_dense
            torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError:
            print("  Dense model: OUT OF MEMORY (as expected for 4 GB GPU)")
            dense_result = {"status": "OOM"}
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print("\n[2/6] Skipping dense comparison (use --dense-comparison to enable)")

    # --- Load model on CPU first ---
    print("\n[3/6] Loading TinyLlama on CPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32
    )
    model.eval()

    # --- Compress and build HelixLinear modules ---
    print("[4/6] Compressing blocks → HelixLinear...", flush=True)
    from safetensors import safe_open
    from scipy.stats import kurtosis as scipy_kurtosis
    from helix_substrate.cdnav3_writer import CDNAv3Writer
    from helix_substrate.helix_linear import (
        HelixLinear, load_helix_linear_from_cdnav3,
        swap_to_helix, swap_summary,
    )
    from helix_substrate.tensor_policy import get_policy

    t_compress = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        writer = CDNAv3Writer(tmpdir)
        sf_path = MODEL_DIR / "model.safetensors"
        helix_modules = {}

        with safe_open(str(sf_path), framework="numpy") as sf:
            for block_idx in range(N_BLOCKS):
                for tensor_type in BLOCK_TENSOR_TYPES:
                    hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                    tensor_np = sf.get_tensor(hf_name).astype(np.float32)

                    kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
                    policy = get_policy(
                        hf_name, tensor_np.shape,
                        block_idx=block_idx, kurtosis=kurt
                    )

                    stats = writer.write_tensor(tensor_np, hf_name, policy=policy)

                    safe_name = hf_name.replace("/", "_").replace(".", "_")
                    tensor_dir = tmpdir / f"{safe_name}.cdnav3"

                    mod = get_module(model, block_idx, tensor_type)
                    bias = mod.bias.data.clone() if mod.bias is not None else None

                    helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
                    module_path = hf_name.replace(".weight", "")
                    helix_modules[module_path] = helix_mod

                if (block_idx + 1) % 11 == 0 or block_idx == N_BLOCKS - 1:
                    print(f"  Block {block_idx + 1}/{N_BLOCKS} done", flush=True)

    compression_time = time.time() - t_compress
    print(f"  Compression: {compression_time:.0f}s")

    # --- Swap to HelixLinear ---
    print(f"\n  Swapping {len(helix_modules)} modules...", flush=True)
    model = swap_to_helix(model, helix_modules)
    summary = swap_summary(model)
    print(f"  {summary['helix_modules']} HelixLinear, "
          f"{summary['linear_modules']} nn.Linear remaining")

    # --- Move to GPU ---
    print("\n[5/6] Moving HelixLinear model to GPU...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    mem_before = gpu_mem_mb()
    try:
        model = model.cuda()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print("  ERROR: HelixLinear model doesn't fit on GPU!")
        print(f"  VRAM before: {mem_before:.0f} MB")
        print(f"  GPU: {gpu_name} ({gpu_vram:.0f} MB)")
        results["helix"] = {"status": "OOM"}
        # Write receipt and exit
        _write_receipt(results, args, eval_tokens, t_start, cpu_start, start_iso,
                       dense_result, summary)
        sys.exit(1)

    mem_after = gpu_mem_mb()
    mem_peak = gpu_mem_peak_mb()
    mem_reserved = gpu_mem_reserved_mb()

    print(f"  VRAM allocated: {mem_after:.0f} MB (delta: {mem_after - mem_before:.0f} MB)")
    print(f"  VRAM peak:      {mem_peak:.0f} MB")
    print(f"  VRAM reserved:  {mem_reserved:.0f} MB")
    print(f"  VRAM free:      {gpu_vram - mem_reserved:.0f} MB")

    # --- Benchmark ---
    print("\n[6/6] Benchmarking HelixLinear on GPU...", flush=True)

    # Check if Triton fused path is active
    from helix_substrate.triton_vq_matmul import is_available as triton_available
    fused_active = triton_available()
    print(f"  Triton fused kernel: {'ACTIVE' if fused_active else 'INACTIVE (naive path)'}")

    # Perplexity
    print("  Computing HelixLinear perplexity on GPU...", flush=True)
    t0 = time.perf_counter()
    helix_ppl, helix_nll, n_tokens, actual_seq_len = compute_perplexity(model, eval_tokens)
    helix_ppl_time = time.perf_counter() - t0
    helix_ppl_peak = gpu_mem_peak_mb()
    print(f"  HelixLinear PPL: {helix_ppl:.4f} (seq_len={actual_seq_len}, {helix_ppl_time:.1f}s)")
    print(f"  VRAM peak after eval: {helix_ppl_peak:.0f} MB")

    # Prompt latency
    print("  Measuring prompt latency...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    prompt_stats = measure_prompt_latency(model, eval_tokens)
    prompt_peak = gpu_mem_peak_mb()
    print(f"  Prompt: {prompt_stats['tokens_per_sec']:.0f} tok/s "
          f"({prompt_stats['median_ms']:.0f} ms for {prompt_stats['seq_len']} tokens)")
    print(f"  VRAM peak during prompt: {prompt_peak:.0f} MB")

    # Decode latency
    print("  Measuring decode latency...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    decode_stats = measure_decode_latency(model, eval_tokens)
    decode_peak = gpu_mem_peak_mb()
    print(f"  Decode: {decode_stats['tokens_per_sec']:.1f} tok/s "
          f"({decode_stats['ms_per_token']:.0f} ms/token)")
    print(f"  VRAM peak during decode: {decode_peak:.0f} MB")

    # --- Results ---
    helix_result = {
        "vram_model_mb": round(mem_after, 1),
        "vram_peak_eval_mb": round(helix_ppl_peak, 1),
        "vram_peak_prompt_mb": round(prompt_peak, 1),
        "vram_peak_decode_mb": round(decode_peak, 1),
        "vram_reserved_mb": round(mem_reserved, 1),
        "perplexity": round(helix_ppl, 4),
        "ppl_seq_len": actual_seq_len,
        "ppl_eval_time_s": round(helix_ppl_time, 1),
        "prompt_latency": prompt_stats,
        "decode_latency": decode_stats,
        "triton_fused": fused_active,
        "compression_ratio": summary["overall_ratio"],
        "helix_modules": summary["helix_modules"],
        "linear_modules": summary["linear_modules"],
        "compressed_bytes": summary["compressed_bytes"],
        "dense_equivalent_bytes": summary["dense_equivalent_bytes"],
        "status": "OK",
    }

    results["helix"] = helix_result
    results["dense"] = dense_result

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"  GPU VIABILITY RESULTS — {gpu_name}")
    print(f"{'=' * 70}")
    print(f"  VRAM total:           {gpu_vram:.0f} MB")
    print(f"  VRAM model (helix):   {mem_after:.0f} MB ({mem_after/gpu_vram*100:.0f}%)")
    print(f"  VRAM peak (eval):     {helix_ppl_peak:.0f} MB ({helix_ppl_peak/gpu_vram*100:.0f}%)")
    print(f"  VRAM peak (decode):   {decode_peak:.0f} MB ({decode_peak/gpu_vram*100:.0f}%)")
    print(f"  VRAM headroom:        {gpu_vram - decode_peak:.0f} MB")
    print(f"  ---")
    print(f"  Perplexity:           {helix_ppl:.4f}")
    print(f"  Prompt throughput:    {prompt_stats['tokens_per_sec']:.0f} tok/s")
    print(f"  Decode throughput:    {decode_stats['tokens_per_sec']:.1f} tok/s")
    print(f"  Decode latency:       {decode_stats['ms_per_token']:.0f} ms/token")
    print(f"  Triton fused:         {'YES' if fused_active else 'NO (naive path)'}")
    print(f"  ---")
    print(f"  Compression ratio:    {summary['overall_ratio']}x")
    print(f"  Modules:              {summary['helix_modules']} helix + {summary['linear_modules']} dense")

    if dense_result and dense_result.get("status") == "OK":
        print(f"  ---")
        print(f"  DENSE COMPARISON:")
        print(f"  Dense VRAM:           {dense_result['vram_model_mb']:.0f} MB")
        print(f"  Dense PPL:            {dense_result['perplexity']:.4f}")
        print(f"  Dense prompt tok/s:   {dense_result['prompt_latency']['tokens_per_sec']:.0f}")
        print(f"  Dense decode tok/s:   {dense_result['decode_latency']['tokens_per_sec']:.1f}")
        speedup = dense_result['decode_latency']['ms_per_token'] / max(1, decode_stats['ms_per_token'])
        print(f"  Decode speedup:       {speedup:.2f}x (dense/helix)")
    elif dense_result and dense_result.get("status") == "OOM":
        print(f"  ---")
        print(f"  Dense model: OUT OF MEMORY (does not fit on {gpu_name})")

    fits = decode_peak < gpu_vram * 0.95
    print(f"\n  VERDICT: {'FITS' if fits else 'DOES NOT FIT'} on {gpu_name}")
    print(f"{'=' * 70}")

    results["verdict"] = "FITS" if fits else "DOES_NOT_FIT"

    # --- Receipt ---
    _write_receipt(results, args, eval_tokens, t_start, cpu_start, start_iso,
                   dense_result, summary)


def _write_receipt(results, args, eval_tokens, t_start, cpu_start, start_iso,
                   dense_result, summary):
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

    receipt = {
        "schema": "gpu_viability_v1",
        "work_order": "WO-GPU-VIABILITY-01",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "eval_tokens": len(eval_tokens),
        **results,
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
        },
    }

    ts_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"gpu_viability_{ts_tag}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
