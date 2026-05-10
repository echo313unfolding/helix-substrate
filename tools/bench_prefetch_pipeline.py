"""
Benchmark: prefetch pipeline vs standard buffered forward.

Measures per-token latency with and without inter-layer double-buffering.
Runs on T2000 (or any CUDA device) with an HXQ-compressed model.

Usage:
    python3 tools/bench_prefetch_pipeline.py --model EchoLabs33/zamba2-1.2b-helix

Produces a WO-RECEIPT-COST-01 receipt.

Work Order: WO-PREFETCH-PIPELINE-01
"""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import torch


def bench_decode_latency(model, tokenizer, prompt: str, n_tokens: int = 32,
                         warmup: int = 3, trials: int = 5) -> dict:
    """Measure decode latency (tok/s) for a given model configuration."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if next(model.parameters()).is_cuda:
        input_ids = input_ids.cuda()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize()

    # Timed trials
    latencies = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=n_tokens, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # Subtract prompt tokens
        generated = out.shape[1] - input_ids.shape[1]
        latencies.append((t1 - t0, generated))

    total_tokens = sum(g for _, g in latencies)
    total_time = sum(t for t, _ in latencies)
    tok_per_sec = total_tokens / total_time

    per_trial = []
    for t, g in latencies:
        per_trial.append({"wall_s": round(t, 4), "tokens": g, "tok_s": round(g / t, 2)})

    return {
        "mean_tok_s": round(tok_per_sec, 2),
        "trials": per_trial,
        "n_tokens_requested": n_tokens,
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 3),
    }


def count_helix_linear(model) -> int:
    """Count HelixLinear modules."""
    from helix_substrate.helix_linear import HelixLinear
    return sum(1 for m in model.modules() if isinstance(m, HelixLinear))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--output", default="receipts/prefetch_pipeline_bench.json")
    args = parser.parse_args()

    t_start = time.time()
    ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import helix_substrate.hf_quantizer  # noqa: F401 — registers HXQ quantizer
    import helix_substrate.mamba_scan_patch  # noqa: F401 — memory-efficient Mamba scan for T2000

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    n_helix = count_helix_linear(model)
    print(f"Found {n_helix} HelixLinear layers")

    # Phase 1: Baseline (standard buffered)
    print(f"\n--- Baseline (buffered, no pipeline) ---")
    baseline = bench_decode_latency(model, tokenizer, args.prompt, args.n_tokens, trials=args.trials)
    print(f"  {baseline['mean_tok_s']} tok/s ({args.trials} trials, {args.n_tokens} tokens each)")

    # Phase 2: Prefetch pipeline
    print(f"\n--- Prefetch pipeline (double-buffered, 2 streams) ---")
    from helix_substrate.prefetch_pipeline import enable_prefetch, disable_prefetch

    state = enable_prefetch(model)
    pipelined = bench_decode_latency(model, tokenizer, args.prompt, args.n_tokens, trials=args.trials)
    print(f"  {pipelined['mean_tok_s']} tok/s ({args.trials} trials, {args.n_tokens} tokens each)")

    disable_prefetch(model)

    # Compute delta
    speedup = pipelined["mean_tok_s"] / baseline["mean_tok_s"] if baseline["mean_tok_s"] > 0 else 0
    delta_pct = (speedup - 1.0) * 100

    print(f"\n{'=' * 50}")
    print(f"Baseline:  {baseline['mean_tok_s']} tok/s")
    print(f"Pipelined: {pipelined['mean_tok_s']} tok/s")
    print(f"Speedup:   {speedup:.3f}x ({delta_pct:+.1f}%)")
    print(f"{'=' * 50}")

    # VRAM snapshot
    vram_allocated = torch.cuda.max_memory_allocated() / 1024**2
    vram_reserved = torch.cuda.max_memory_reserved() / 1024**2

    receipt = {
        "experiment": "prefetch_pipeline_bench",
        "model": args.model,
        "n_helix_layers": n_helix,
        "prompt": args.prompt,
        "n_tokens": args.n_tokens,
        "trials": args.trials,
        "baseline": baseline,
        "pipelined": pipelined,
        "speedup": round(speedup, 4),
        "delta_pct": round(delta_pct, 2),
        "vram_peak_allocated_mb": round(vram_allocated, 1),
        "vram_peak_reserved_mb": round(vram_reserved, 1),
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time(), 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": ts_start,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "device": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vram_total_mb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 0
            ) if torch.cuda.is_available() else 0,
            "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}" if torch.cuda.is_available() else "n/a",
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt written to {out_path}")


if __name__ == "__main__":
    main()
