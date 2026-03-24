#!/usr/bin/env python3
"""
Step 8: Speed profiling — HelixLinear vs dense FP32.

Measures:
  1. Dense FP32 on CPU (baseline — doesn't fit on 4GB GPU)
  2. HelixLinear on CPU (naive path)
  3. HelixLinear on GPU (Triton fused path)
  4. HelixLinear on GPU with torch.compile (if available)
  5. Qualitative generation comparison (same prompts, side-by-side)

Key question: is HelixLinear *usable* on low-end hardware (Quadro T2000, 4GB VRAM)?

Usage:
    python3 tools/step8_speed_profile.py
    python3 tools/step8_speed_profile.py --max-new-tokens 200

Work Order: WO-HELIX-LINEAR-01
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
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "step8_speed_profile"

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


def measure_prefill(model, input_ids, device, n_warmup=2, n_runs=5):
    """Measure prefill (forward pass) latency."""
    ids = input_ids.to(device)
    model.eval()

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(ids)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "median_ms": round(np.median(times) * 1000, 2),
        "mean_ms": round(np.mean(times) * 1000, 2),
        "min_ms": round(min(times) * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
        "tokens": ids.shape[1],
        "tokens_per_sec": round(ids.shape[1] / np.median(times), 1),
    }


def measure_generation(model, tokenizer, prompt, device, max_new_tokens=64,
                        n_warmup=1, n_runs=3):
    """Measure autoregressive generation speed."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    model.eval()

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(
                input_ids, max_new_tokens=min(8, max_new_tokens),
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times = []
    last_output = None
    for _ in range(n_runs):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last_output = output

    new_tokens = last_output.shape[1] - prompt_len
    median_time = np.median(times)

    result = {
        "prompt_tokens": prompt_len,
        "new_tokens": int(new_tokens),
        "median_s": round(median_time, 3),
        "tokens_per_sec": round(new_tokens / median_time, 2),
        "ms_per_token": round(median_time * 1000 / max(new_tokens, 1), 1),
    }

    if device.type == "cuda":
        result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    # Decode text
    text = tokenizer.decode(last_output[0], skip_special_tokens=True)
    return result, text


def build_helix_model(model, tmpdir):
    """Compress all blocks → HelixLinear, return swapped model + stats."""
    from scipy.stats import kurtosis as scipy_kurtosis
    from safetensors import safe_open

    from helix_substrate.cdnav3_writer import CDNAv3Writer
    from helix_substrate.helix_linear import (
        load_helix_linear_from_cdnav3,
        swap_to_helix,
        swap_summary,
    )
    from helix_substrate.tensor_policy import get_policy

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
                    block_idx=block_idx, kurtosis=kurt,
                )
                writer.write_tensor(tensor_np, hf_name, policy=policy)

                safe_name = hf_name.replace("/", "_").replace(".", "_")
                tensor_dir = tmpdir / f"{safe_name}.cdnav3"
                mod = get_module(model, block_idx, tensor_type)
                bias = mod.bias.data.clone() if mod.bias is not None else None
                helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
                module_path = hf_name.replace(".weight", "")
                helix_modules[module_path] = helix_mod

            if (block_idx + 1) % 11 == 0 or block_idx == N_BLOCKS - 1:
                print(f"    Block {block_idx + 1}/{N_BLOCKS}")

    model = swap_to_helix(model, helix_modules)
    summary = swap_summary(model)
    return model, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        print(f"ERROR: TinyLlama not found at {MODEL_DIR}")
        sys.exit(1)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("Step 8: Speed Profile — HelixLinear vs Dense FP32")
    print("=" * 70)

    # --- Load model + tokenizer ---
    print("\n[1/6] Loading TinyLlama FP32...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model_dense = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32,
    )
    model_dense.eval()

    # --- Test prompts ---
    prompts = [
        "The future of AI compression is",
        "Explain quantum entanglement in simple terms:",
        "Once upon a time in a small village,",
    ]
    test_prompt = prompts[0]

    # --- Eval input for prefill ---
    eval_input = tokenizer(
        "The transformer architecture revolutionizes deep learning by introducing "
        "self-attention mechanisms that allow models to process sequences in parallel",
        return_tensors="pt",
    )["input_ids"]

    results = {}
    gen_texts = {}

    # ================================================================
    # Benchmark 1: Dense FP32 on CPU
    # ================================================================
    print("\n[2/6] Benchmarking: Dense FP32 on CPU...")
    device_cpu = torch.device("cpu")
    model_dense = model_dense.to(device_cpu)

    prefill_dense_cpu = measure_prefill(model_dense, eval_input, device_cpu, n_warmup=1, n_runs=3)
    print(f"  Prefill: {prefill_dense_cpu['median_ms']:.0f} ms "
          f"({prefill_dense_cpu['tokens_per_sec']:.0f} tok/s)")

    gen_dense_cpu, text_dense = measure_generation(
        model_dense, tokenizer, test_prompt, device_cpu,
        max_new_tokens=args.max_new_tokens, n_warmup=1, n_runs=2,
    )
    print(f"  Generation: {gen_dense_cpu['tokens_per_sec']:.1f} tok/s "
          f"({gen_dense_cpu['ms_per_token']:.0f} ms/tok)")
    results["dense_fp32_cpu"] = {"prefill": prefill_dense_cpu, "generation": gen_dense_cpu}
    gen_texts["dense_fp32_cpu"] = text_dense

    # ================================================================
    # Benchmark 2: Build HelixLinear model
    # ================================================================
    print("\n[3/6] Compressing → HelixLinear...")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_helix, summary = build_helix_model(model_dense, tmpdir)

    print(f"  {summary['helix_modules']} HelixLinear, "
          f"{summary['overall_ratio']}x compression")

    # Free dense model to save RAM
    del model_dense
    gc.collect()

    # ================================================================
    # Benchmark 3: HelixLinear on CPU (naive path)
    # ================================================================
    print("\n[4/6] Benchmarking: HelixLinear on CPU (naive path)...")
    model_helix = model_helix.to(device_cpu)

    prefill_helix_cpu = measure_prefill(model_helix, eval_input, device_cpu, n_warmup=1, n_runs=3)
    print(f"  Prefill: {prefill_helix_cpu['median_ms']:.0f} ms "
          f"({prefill_helix_cpu['tokens_per_sec']:.0f} tok/s)")

    gen_helix_cpu, text_helix_cpu = measure_generation(
        model_helix, tokenizer, test_prompt, device_cpu,
        max_new_tokens=args.max_new_tokens, n_warmup=1, n_runs=2,
    )
    print(f"  Generation: {gen_helix_cpu['tokens_per_sec']:.1f} tok/s "
          f"({gen_helix_cpu['ms_per_token']:.0f} ms/tok)")
    results["helix_cpu_naive"] = {"prefill": prefill_helix_cpu, "generation": gen_helix_cpu}
    gen_texts["helix_cpu_naive"] = text_helix_cpu

    # ================================================================
    # Benchmark 4: HelixLinear on GPU (Triton fused path)
    # ================================================================
    gpu_results = {}
    if torch.cuda.is_available():
        print("\n[5/6] Benchmarking: HelixLinear on GPU (Triton fused)...")
        device_gpu = torch.device("cuda")

        try:
            torch.cuda.reset_peak_memory_stats()
            mem_before_load = torch.cuda.memory_allocated()
            model_helix = model_helix.to(device_gpu)
            torch.cuda.synchronize()
            model_vram = (torch.cuda.memory_allocated() - mem_before_load) / 1e6
            print(f"  Model VRAM: {model_vram:.0f} MB")

            prefill_helix_gpu = measure_prefill(
                model_helix, eval_input, device_gpu, n_warmup=2, n_runs=5,
            )
            print(f"  Prefill: {prefill_helix_gpu['median_ms']:.1f} ms "
                  f"({prefill_helix_gpu['tokens_per_sec']:.0f} tok/s)")

            gen_helix_gpu, text_helix_gpu = measure_generation(
                model_helix, tokenizer, test_prompt, device_gpu,
                max_new_tokens=args.max_new_tokens, n_warmup=1, n_runs=3,
            )
            print(f"  Generation: {gen_helix_gpu['tokens_per_sec']:.1f} tok/s "
                  f"({gen_helix_gpu['ms_per_token']:.0f} ms/tok)")
            print(f"  Peak VRAM: {gen_helix_gpu.get('peak_vram_mb', 0):.0f} MB")

            results["helix_gpu_fused"] = {
                "prefill": prefill_helix_gpu,
                "generation": gen_helix_gpu,
                "model_vram_mb": round(model_vram, 1),
            }
            gen_texts["helix_gpu_fused"] = text_helix_gpu
            gpu_results = results["helix_gpu_fused"]

        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            results["helix_gpu_fused"] = {"error": "OOM"}
            model_helix = model_helix.to(device_cpu)

    else:
        print("\n[5/6] SKIP: No CUDA available")

    # ================================================================
    # Benchmark 5: torch.compile on HelixLinear (optional)
    # ================================================================
    if not args.skip_compile and hasattr(torch, "compile"):
        print("\n[6/6] Benchmarking: HelixLinear + torch.compile...")
        from helix_substrate.helix_linear import HelixLinear

        # Move to CPU for compile test (GPU may be tight)
        model_helix = model_helix.to(device_cpu)
        compiled_count = 0
        try:
            for name, module in model_helix.named_modules():
                if isinstance(module, HelixLinear):
                    module.forward = torch.compile(
                        module.forward,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    compiled_count += 1
            print(f"  Compiled {compiled_count} HelixLinear modules")

            # Warmup compile (first run triggers compilation)
            print("  Warming up torch.compile (this takes a while)...")
            with torch.no_grad():
                model_helix(eval_input.to(device_cpu))

            prefill_compiled = measure_prefill(
                model_helix, eval_input, device_cpu, n_warmup=1, n_runs=3,
            )
            print(f"  Prefill: {prefill_compiled['median_ms']:.0f} ms "
                  f"({prefill_compiled['tokens_per_sec']:.0f} tok/s)")

            gen_compiled, text_compiled = measure_generation(
                model_helix, tokenizer, test_prompt, device_cpu,
                max_new_tokens=args.max_new_tokens, n_warmup=1, n_runs=2,
            )
            print(f"  Generation: {gen_compiled['tokens_per_sec']:.1f} tok/s "
                  f"({gen_compiled['ms_per_token']:.0f} ms/tok)")

            results["helix_cpu_compiled"] = {
                "prefill": prefill_compiled,
                "generation": gen_compiled,
            }
            gen_texts["helix_cpu_compiled"] = text_compiled

        except Exception as e:
            print(f"  torch.compile failed: {e}")
            results["helix_cpu_compiled"] = {"error": str(e)}
    else:
        print("\n[6/6] SKIP: torch.compile")

    # ================================================================
    # Qualitative generation comparison
    # ================================================================
    print(f"\n{'=' * 70}")
    print("QUALITATIVE GENERATION COMPARISON")
    print(f"{'=' * 70}")
    for label, text in gen_texts.items():
        print(f"\n--- {label} ---")
        print(text[:500])

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'=' * 70}")
    print("SPEED COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"{'Config':<30} {'Prefill ms':>12} {'Gen tok/s':>12} {'ms/tok':>10}")
    print("-" * 70)
    for label, data in results.items():
        if "error" in data:
            print(f"{label:<30} {'ERROR':>12} {data['error']:>22}")
            continue
        pf = data.get("prefill", {})
        gen = data.get("generation", {})
        print(f"{label:<30} {pf.get('median_ms', 'N/A'):>12} "
              f"{gen.get('tokens_per_sec', 'N/A'):>12} "
              f"{gen.get('ms_per_token', 'N/A'):>10}")

    # Speedup ratios
    if "dense_fp32_cpu" in results and "helix_gpu_fused" in results:
        dense_ms = results["dense_fp32_cpu"]["generation"].get("ms_per_token", 0)
        helix_ms = results["helix_gpu_fused"]["generation"].get("ms_per_token", 0)
        if dense_ms > 0 and helix_ms > 0:
            print(f"\n  GPU HelixLinear vs CPU Dense: {dense_ms/helix_ms:.1f}x speedup")

    if "dense_fp32_cpu" in results and "helix_cpu_naive" in results:
        dense_tps = results["dense_fp32_cpu"]["generation"].get("tokens_per_sec", 0)
        helix_tps = results["helix_cpu_naive"]["generation"].get("tokens_per_sec", 0)
        if dense_tps > 0 and helix_tps > 0:
            ratio = helix_tps / dense_tps
            print(f"  CPU HelixLinear vs CPU Dense: {ratio:.2f}x "
                  f"({'faster' if ratio > 1 else 'slower'})")

    # Key insight
    if gpu_results:
        vram = gpu_results.get("model_vram_mb", 0)
        print(f"\n  KEY: Dense FP32 (4200 MB) does NOT fit on 4GB GPU")
        print(f"       HelixLinear ({vram:.0f} MB) DOES fit → enables GPU inference")

    # ================================================================
    # Receipt
    # ================================================================
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "speed_profile_v1",
        "work_order": "WO-HELIX-LINEAR-01",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "max_new_tokens": args.max_new_tokens,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_vram_gb": round(torch.cuda.mem_get_info()[1] / 1e9, 2) if torch.cuda.is_available() else 0,
        "results": results,
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

    receipt_path = RECEIPT_DIR / f"speed_profile_{time.strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
