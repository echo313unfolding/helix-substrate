"""
Benchmark: sequential vs chunked Mamba2 scan.

Tests correctness (outputs match within tolerance) and speed
(kernel launch overhead reduction) at various sequence lengths.

Usage:
    python3 tools/bench_mamba_scan.py                    # GPU benchmark
    python3 tools/bench_mamba_scan.py --seq-lens 4 32 128 256 512
    python3 tools/bench_mamba_scan.py --full-model EchoLabs33/zamba2-1.2b-helix

Work Order: WO-PREFETCH-PIPELINE-01
"""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import torch


def make_random_inputs(batch=1, seq_len=128, heads=16, head_dim=64,
                       state_size=64, device='cuda', dtype=torch.float32):
    """Create random SSM inputs matching Zamba2 dimensions."""
    h = torch.randn(batch, heads, head_dim, state_size, device=device, dtype=dtype)
    dt = torch.rand(batch, seq_len, heads, device=device, dtype=dtype) * 0.1 + 0.01
    A = -torch.rand(heads, device=device, dtype=dtype) * 5  # negative decay
    x = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
    B = torch.randn(batch, seq_len, heads, state_size, device=device, dtype=dtype)
    C = torch.randn(batch, seq_len, heads, state_size, device=device, dtype=dtype)
    D = torch.randn(heads, 1, device=device, dtype=dtype)
    return h, dt, A, x, B, C, D


def verify_correctness(seq_lens=(4, 16, 32, 64, 128, 256), chunk_sizes=(8, 16, 32)):
    """Verify chunked scan matches sequential scan for various configs."""
    from helix_substrate.mamba_scan_chunked import sequential_scan, chunked_scan

    print("=== Correctness Verification ===\n")
    all_pass = True

    for sl in seq_lens:
        for cs in chunk_sizes:
            h, dt, A, x, B, C, D = make_random_inputs(seq_len=sl)

            y_seq, h_seq = sequential_scan(h.clone(), dt, A, x, B, C, D)
            y_chk, h_chk = chunked_scan(h.clone(), dt, A, x, B, C, D, chunk_size=cs)

            y_close = torch.allclose(y_seq, y_chk, atol=1e-3, rtol=1e-3)
            h_close = torch.allclose(h_seq, h_chk, atol=1e-3, rtol=1e-3)

            if y_close and h_close:
                max_y_err = (y_seq - y_chk).abs().max().item()
                max_h_err = (h_seq - h_chk).abs().max().item()
                status = "PASS"
            else:
                max_y_err = (y_seq - y_chk).abs().max().item()
                max_h_err = (h_seq - h_chk).abs().max().item()
                status = "FAIL"
                all_pass = False

            print(f"  seq_len={sl:>4d}, chunk={cs:>3d}: {status}  "
                  f"(max_y_err={max_y_err:.2e}, max_h_err={max_h_err:.2e})")

    print(f"\n{'All tests passed.' if all_pass else 'SOME TESTS FAILED.'}\n")
    return all_pass


def bench_scan(seq_lens=(4, 16, 32, 64, 128, 256, 512),
               chunk_size=32, warmup=5, trials=20):
    """Benchmark sequential vs chunked scan at various sequence lengths."""
    from helix_substrate.mamba_scan_chunked import sequential_scan, chunked_scan

    print(f"=== Scan Benchmark (chunk_size={chunk_size}, "
          f"{warmup} warmup, {trials} trials) ===\n")
    print(f"  {'seq_len':>8s}  {'sequential':>12s}  {'chunked':>12s}  "
          f"{'speedup':>8s}  {'launches_seq':>12s}  {'launches_chk':>12s}")
    print(f"  {'':>8s}  {'(ms)':>12s}  {'(ms)':>12s}  {'':>8s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*12}")

    results = []

    for sl in seq_lens:
        h, dt, A, x, B, C, D = make_random_inputs(seq_len=sl)

        # Warmup sequential
        for _ in range(warmup):
            sequential_scan(h.clone(), dt, A, x, B, C, D)
            torch.cuda.synchronize()

        # Time sequential
        seq_times = []
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sequential_scan(h.clone(), dt, A, x, B, C, D)
            torch.cuda.synchronize()
            seq_times.append(time.perf_counter() - t0)

        # Warmup chunked
        for _ in range(warmup):
            chunked_scan(h.clone(), dt, A, x, B, C, D, chunk_size=chunk_size)
            torch.cuda.synchronize()

        # Time chunked
        chk_times = []
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunked_scan(h.clone(), dt, A, x, B, C, D, chunk_size=chunk_size)
            torch.cuda.synchronize()
            chk_times.append(time.perf_counter() - t0)

        seq_ms = sum(seq_times) / trials * 1000
        chk_ms = sum(chk_times) / trials * 1000
        speedup = seq_ms / chk_ms if chk_ms > 0 else 0

        # Kernel launch counts (approximate)
        launches_seq = sl * 6
        n_chunks = (sl + chunk_size - 1) // chunk_size
        launches_chk = n_chunks * 8 + 2  # +2 for precompute

        print(f"  {sl:>8d}  {seq_ms:>12.3f}  {chk_ms:>12.3f}  "
              f"{speedup:>7.2f}x  {launches_seq:>12d}  {launches_chk:>12d}")

        results.append({
            "seq_len": sl,
            "sequential_ms": round(seq_ms, 3),
            "chunked_ms": round(chk_ms, 3),
            "speedup": round(speedup, 3),
            "kernel_launches_sequential": launches_seq,
            "kernel_launches_chunked": launches_chk,
        })

    return results


def bench_full_model(model_id, n_tokens=32, trials=3):
    """Benchmark full model with sequential vs chunked scan."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import helix_substrate.hf_quantizer  # noqa: F401

    print(f"\n=== Full Model Benchmark: {model_id} ===\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    prompt = "The quick brown fox"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    results = {}

    for scan_type in ["sequential", "chunked"]:
        # Explicitly call apply_patch (import caches modules, so re-import is a no-op)
        if scan_type == "sequential":
            from helix_substrate.mamba_scan_patch import apply_patch as apply_seq
            apply_seq()
        else:
            from helix_substrate.mamba_scan_chunked import apply_patch as apply_chk
            apply_chk()

        # Load model fresh each time to avoid state contamination
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        ids = input_ids.to(next(model.parameters()).device)

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                model.generate(ids, max_new_tokens=4, do_sample=False)
            torch.cuda.synchronize()

        # Timed
        times = []
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=n_tokens, do_sample=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            gen = out.shape[1] - ids.shape[1]
            times.append((t1 - t0, gen))

        total_tok = sum(g for _, g in times)
        total_time = sum(t for t, _ in times)
        tok_s = total_tok / total_time

        print(f"  {scan_type:>12s}: {tok_s:.2f} tok/s ({trials} trials, {n_tokens} tokens)")
        results[scan_type] = {"tok_s": round(tok_s, 2), "total_time_s": round(total_time, 3)}

        del model
        torch.cuda.empty_cache()

    if results.get("sequential") and results.get("chunked"):
        seq_tps = results["sequential"]["tok_s"]
        chk_tps = results["chunked"]["tok_s"]
        speedup = chk_tps / seq_tps if seq_tps > 0 else 0
        print(f"\n  Speedup: {speedup:.3f}x ({(speedup-1)*100:+.1f}%)")
        results["speedup"] = round(speedup, 4)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[4, 16, 32, 64, 128, 256])
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--full-model", type=str, default=None,
                        help="Run full model benchmark (e.g. EchoLabs33/zamba2-1.2b-helix)")
    parser.add_argument("--output", default="receipts/mamba_scan_bench.json")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")

    if not torch.cuda.is_available():
        print("CUDA required for this benchmark")
        return

    # Phase 1: Correctness
    if not args.skip_correctness:
        correct = verify_correctness(seq_lens=args.seq_lens,
                                     chunk_sizes=[8, 16, args.chunk_size])
        if not correct:
            print("ABORTING: correctness check failed")
            return
    else:
        correct = "skipped"

    # Phase 2: Isolated scan benchmark
    scan_results = bench_scan(seq_lens=args.seq_lens,
                              chunk_size=args.chunk_size,
                              trials=args.trials)

    # Phase 3: Full model (optional)
    model_results = None
    if args.full_model:
        model_results = bench_full_model(args.full_model)

    # Receipt
    receipt = {
        "experiment": "mamba_scan_chunked_bench",
        "chunk_size": args.chunk_size,
        "correctness": "all_pass" if correct is True else correct,
        "scan_results": scan_results,
        "model_results": model_results,
        "device": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": (
                f"{torch.cuda.get_device_properties(0).major}"
                f".{torch.cuda.get_device_properties(0).minor}"
            ),
        },
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time(), 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": ts_start,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt written to {out_path}")


if __name__ == "__main__":
    main()
