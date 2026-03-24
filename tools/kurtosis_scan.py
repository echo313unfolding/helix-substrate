#!/usr/bin/env python3
"""
Kurtosis preflight scanner for neural network weight compression.

Scans all 2D weight tensors in a model, computes Fisher kurtosis, and
predicts which tensors need SVD correction for VQ compression.

This is a zero-cost diagnostic — it reads weights and computes statistics
without actually compressing anything. Use it before compression to:

  1. Identify high-kurtosis tensors that will need SVD correction
  2. Predict expected compression difficulty per tensor
  3. Compare kurtosis profiles across architectures
  4. Estimate total SVD budget (how many tensors get expensive treatment)

Evidence (cross-architecture):
  TinyLlama  (transformer):     rho=0.7835, p=3.2e-33, n=154
  Mamba-130m (SSM):             rho=0.8534, p=1.2e-28, n=97
  Qwen2.5-7B (transformer+GQA): rho=0.5334, p=8.3e-16, n=196

Usage:
  python3 tools/kurtosis_scan.py --model ~/models/mamba-130m-hf
  python3 tools/kurtosis_scan.py --model ~/models/qwen2.5-7b-instruct --top 20
  python3 tools/kurtosis_scan.py --model ~/models/tinyllama_fp32 --json
"""

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.tensor_policy import classify_tensor, TensorClass, get_policy


def load_tensors_from_safetensors(model_dir: Path):
    """Yield (name, numpy_array) for all 2D weight tensors in a model."""
    from safetensors import safe_open

    model_dir = Path(model_dir)

    # Check for shard index
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"

    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        shard_cache = {}

        for tensor_name, shard_file in sorted(weight_map.items()):
            if not tensor_name.endswith(".weight"):
                continue
            shard_path = model_dir / shard_file
            if shard_file not in shard_cache:
                shard_cache[shard_file] = safe_open(str(shard_path), framework="pt")
            tensor = shard_cache[shard_file].get_tensor(tensor_name).float().numpy()
            if tensor.ndim == 2:
                yield tensor_name, tensor
            del tensor

        shard_cache.clear()

    elif single_path.exists():
        sf = safe_open(str(single_path), framework="pt")
        for tensor_name in sorted(sf.keys()):
            if not tensor_name.endswith(".weight"):
                continue
            tensor = sf.get_tensor(tensor_name).float().numpy()
            if tensor.ndim == 2:
                yield tensor_name, tensor
            del tensor
    else:
        print(f"ERROR: No safetensors found in {model_dir}", file=sys.stderr)
        sys.exit(1)


def scan_model(model_dir: Path):
    """Scan all 2D tensors and return kurtosis stats."""
    results = []

    for name, tensor in load_tensors_from_safetensors(model_dir):
        shape = tensor.shape
        kurt = float(scipy_kurtosis(tensor.ravel(), fisher=True))
        tc = classify_tensor(name, shape=shape)
        policy = get_policy(name, shape, kurtosis=kurt)
        needs_svd = policy.svd_residual_rank > 0

        results.append({
            "name": name,
            "shape": list(shape),
            "params": int(np.prod(shape)),
            "kurtosis": round(kurt, 2),
            "tensor_class": tc.value,
            "needs_svd": needs_svd,
            "svd_rank": policy.svd_residual_rank,
        })

    return results


def print_report(results, top_n=None, model_dir=None):
    """Print human-readable kurtosis report."""
    if not results:
        print("No 2D tensors found.")
        return

    # Sort by kurtosis descending
    ranked = sorted(results, key=lambda r: -r["kurtosis"])
    n_total = len(ranked)
    n_svd = sum(1 for r in ranked if r["needs_svd"])
    n_vq_only = n_total - n_svd

    total_params = sum(r["params"] for r in ranked)
    svd_params = sum(r["params"] for r in ranked if r["needs_svd"])

    kurts = [r["kurtosis"] for r in ranked]
    mean_kurt = np.mean(kurts)
    median_kurt = np.median(kurts)

    # Header
    print("=" * 78)
    print(f"  Kurtosis Preflight Scan")
    if model_dir:
        print(f"  Model: {model_dir}")
    print(f"  Tensors: {n_total} (2D weights only)")
    print(f"  Total params: {total_params:,}")
    print("=" * 78)

    # Summary
    print(f"\n  Kurtosis distribution:")
    print(f"    Mean:   {mean_kurt:.2f}")
    print(f"    Median: {median_kurt:.2f}")
    print(f"    Min:    {min(kurts):.2f}")
    print(f"    Max:    {max(kurts):.2f}")

    print(f"\n  Routing decision:")
    print(f"    VQ+SVD (kurtosis > 5):  {n_svd:3d} tensors ({svd_params:,} params)")
    print(f"    VQ only (kurtosis ≤ 5): {n_vq_only:3d} tensors ({total_params - svd_params:,} params)")
    print(f"    SVD budget: {n_svd}/{n_total} = {100*n_svd/n_total:.1f}% of tensors")

    # Kurtosis buckets
    buckets = [(0, 2, "low"), (2, 5, "moderate"), (5, 20, "elevated"),
               (20, 50, "high"), (50, float("inf"), "extreme")]
    print(f"\n  Kurtosis buckets:")
    for lo, hi, label in buckets:
        count = sum(1 for k in kurts if lo <= k < hi)
        if count > 0:
            hi_str = f"{hi}" if hi != float("inf") else "∞"
            print(f"    [{lo:5.0f}, {hi_str:>5s})  {label:10s}  {count:3d} tensors")

    # Top tensors
    display_n = top_n or 15
    print(f"\n  Top {min(display_n, n_total)} by kurtosis:")
    print(f"  {'Tensor':<55s} {'Shape':>14s} {'Kurt':>8s} {'Route':>7s}")
    print(f"  {'-'*55} {'-'*14} {'-'*8} {'-'*7}")
    for r in ranked[:display_n]:
        shape_str = f"{r['shape'][0]}×{r['shape'][1]}"
        route = "SVD" if r["needs_svd"] else "VQ"
        name_short = r["name"]
        if len(name_short) > 55:
            name_short = "..." + name_short[-52:]
        print(f"  {name_short:<55s} {shape_str:>14s} {r['kurtosis']:8.1f} {route:>7s}")

    # Bottom 5 (lowest kurtosis — easiest to compress)
    print(f"\n  Bottom 5 (easiest to compress):")
    for r in ranked[-5:]:
        shape_str = f"{r['shape'][0]}×{r['shape'][1]}"
        print(f"  {r['name']:<55s} {shape_str:>14s} {r['kurtosis']:8.1f}")

    # Architecture profile
    class_stats = {}
    for r in ranked:
        tc = r["tensor_class"]
        if tc not in class_stats:
            class_stats[tc] = []
        class_stats[tc].append(r["kurtosis"])

    print(f"\n  By tensor class:")
    for tc, ks in sorted(class_stats.items()):
        print(f"    {tc:15s}  n={len(ks):3d}  "
              f"mean={np.mean(ks):6.1f}  max={max(ks):7.1f}  "
              f"svd={sum(1 for k in ks if k > 5):d}/{len(ks)}")

    print(f"\n{'=' * 78}")


def main():
    parser = argparse.ArgumentParser(
        description="Kurtosis preflight scanner for neural network compression"
    )
    parser.add_argument("--model", required=True, help="Path to model directory with safetensors")
    parser.add_argument("--top", type=int, default=15, help="Show top N tensors (default: 15)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    parser.add_argument("--output", type=str, default=None, help="Write receipt JSON to file")
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser()
    if not model_dir.exists():
        print(f"ERROR: {model_dir} not found", file=sys.stderr)
        sys.exit(1)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"Scanning {model_dir}...", file=sys.stderr, flush=True)
    results = scan_model(model_dir)

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    if args.json:
        output = {
            "model_dir": str(model_dir),
            "n_tensors": len(results),
            "n_svd": sum(1 for r in results if r["needs_svd"]),
            "kurtosis_mean": round(float(np.mean([r["kurtosis"] for r in results])), 2) if results else 0,
            "kurtosis_max": round(float(max(r["kurtosis"] for r in results)), 2) if results else 0,
            "tensors": results,
            "cost": {
                "wall_time_s": round(wall, 3),
                "cpu_time_s": round(cpu, 3),
                "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
                "timestamp_start": start_iso,
                "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(results, top_n=args.top, model_dir=model_dir)
        print(f"  Scan time: {wall:.1f}s wall, {cpu:.1f}s CPU")

    # Write receipt if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        receipt = {
            "tool": "kurtosis_scan",
            "model_dir": str(model_dir),
            "n_tensors": len(results),
            "n_svd": sum(1 for r in results if r["needs_svd"]),
            "kurtosis_mean": round(float(np.mean([r["kurtosis"] for r in results])), 2) if results else 0,
            "kurtosis_max": round(float(max(r["kurtosis"] for r in results)), 2) if results else 0,
            "tensors": results,
            "cost": {
                "wall_time_s": round(wall, 3),
                "cpu_time_s": round(cpu, 3),
                "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
                "timestamp_start": start_iso,
                "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        output_path.write_text(json.dumps(receipt, indent=2))
        print(f"  Receipt: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
