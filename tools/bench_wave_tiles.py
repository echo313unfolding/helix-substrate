#!/usr/bin/env python3
"""
Benchmark: Wave tile dispatch vs sequential naive path.

Compares _forward_naive (sequential, OMP=8) against _forward_wave
(parallel tiles, FibPi3D core pinning, OMP=1 per worker).

Baseline to beat: 32ms (from OMP_NUM_THREADS=8 benchmark, 2026-04-04).

Usage:
    # Sequential baseline
    python3 tools/bench_wave_tiles.py

    # Wave dispatch
    HELIX_WAVE_TILES=1 python3 tools/bench_wave_tiles.py

    # Compare both in one run
    python3 tools/bench_wave_tiles.py --compare
"""

import argparse
import json
import os
import platform
import sys
import time
import resource

import torch
import numpy as np


def make_helix_linear(out_features=2048, in_features=2048, k=256):
    """Create a HelixLinear module with random VQ factors for benchmarking."""
    from helix_substrate.helix_linear import HelixLinear

    codebook = torch.randn(k, dtype=torch.float32)
    indices = torch.randint(0, k, (out_features, in_features), dtype=torch.uint8)

    return HelixLinear(
        in_features=in_features,
        out_features=out_features,
        codebook=codebook,
        indices=indices,
        tensor_name="bench_layer",
    )


def bench_path(layer, x, n_warmup=3, n_iters=20):
    """Time a forward path over n_iters, return median ms."""
    for _ in range(n_warmup):
        layer(x)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        layer(x)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def run_benchmark(out_features=2048, in_features=2048, batch=1, n_iters=20, compare=False):
    """Run benchmark and print results."""
    from helix_substrate.tile_scheduler import tile_schedule, get_cpu_count, fibonacci_firing_order

    n_cores = get_cpu_count()
    print(f"CPU cores: {n_cores}")
    print(f"torch threads: {torch.get_num_threads()}")
    print(f"Matrix: [{batch}, {in_features}] @ [{out_features}, {in_features}].T")
    print(f"Tiles: {(out_features + 255) // 256} × 256 rows")
    print()

    x = torch.randn(batch, in_features)

    # Show firing order
    firing = fibonacci_firing_order(n_cores)
    print(f"FibPi3D firing order: {firing}")

    schedule = tile_schedule(out_features, 256, n_cores)
    print(f"Tile schedule: {len(schedule)} tiles → {n_cores} cores")
    for s, e, c in schedule:
        print(f"  rows [{s:4d}:{e:4d}] → core {c}")
    print()

    results = {}
    t_start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    wall_start = time.time()
    cpu_start = time.process_time()

    if compare:
        # --- Naive path ---
        layer_naive = make_helix_linear(out_features, in_features)
        layer_naive._wave_enabled = False
        naive_ms = bench_path(layer_naive, x, n_iters=n_iters)
        print(f"naive  (sequential, OMP={torch.get_num_threads()}): {naive_ms:.1f} ms")
        results["naive_ms"] = round(naive_ms, 2)

        # --- Wave path ---
        layer_wave = make_helix_linear(out_features, in_features)
        layer_wave._wave_enabled = True
        wave_ms = bench_path(layer_wave, x, n_iters=n_iters)
        print(f"wave   (parallel, {n_cores} workers):  {wave_ms:.1f} ms")
        results["wave_ms"] = round(wave_ms, 2)

        speedup = naive_ms / wave_ms if wave_ms > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x")
        results["speedup"] = round(speedup, 2)

        # Correctness check
        layer_naive.eval()
        layer_wave.eval()
        # Copy weights so both compute same thing
        layer_wave.codebook.copy_(layer_naive.codebook)
        layer_wave.indices.copy_(layer_naive.indices)
        with torch.no_grad():
            out_naive = layer_naive(x)
            out_wave = layer_wave(x)
        max_diff = (out_naive - out_wave).abs().max().item()
        print(f"Max output diff: {max_diff:.2e} (should be 0.0)")
        results["max_diff"] = max_diff
    else:
        # Single path based on env var
        layer = make_helix_linear(out_features, in_features)
        wave_on = os.environ.get("HELIX_WAVE_TILES", "0") == "1"
        path_name = "wave" if wave_on else "naive"
        ms = bench_path(layer, x, n_iters=n_iters)
        print(f"{path_name}: {ms:.1f} ms")
        results[f"{path_name}_ms"] = round(ms, 2)

    # Cost block
    wall_time = time.time() - wall_start
    cpu_time = time.process_time() - cpu_start
    cost = {
        "wall_time_s": round(wall_time, 3),
        "cpu_time_s": round(cpu_time, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": t_start_iso,
        "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    receipt = {
        "benchmark": "wave_tile_dispatch",
        "matrix": {"out": out_features, "in": in_features, "batch": batch},
        "n_cores": n_cores,
        "n_iters": n_iters,
        "torch_threads": torch.get_num_threads(),
        "results": results,
        "cost": cost,
    }

    # Save receipt
    receipt_dir = os.path.join(os.path.dirname(__file__), "..", "receipts", "wave_tiles")
    os.makedirs(receipt_dir, exist_ok=True)
    receipt_path = os.path.join(receipt_dir, f"bench_{time.strftime('%Y%m%dT%H%M%S')}.json")
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}")

    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark wave tile dispatch vs naive")
    parser.add_argument("--out", type=int, default=2048, help="Output features")
    parser.add_argument("--in-features", type=int, default=2048, dest="in_feat", help="Input features")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--compare", action="store_true", help="Run both paths and compare")
    args = parser.parse_args()

    run_benchmark(
        out_features=args.out,
        in_features=args.in_feat,
        batch=args.batch,
        n_iters=args.iters,
        compare=args.compare,
    )
