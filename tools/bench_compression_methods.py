#!/usr/bin/env python3
"""
R&D bench: Compare compression methods beyond k=256 VQ.

Tests three approaches to break the 2x BF16 ceiling:
  1. Per-group k=16 (group_size=128/256/512) — local codebooks, packed 4-bit
  2. Packed 6-bit k=64 — global codebook, 2.67x from BF16
  3. Multi-stage RVQ — residual VQ, compound codebooks

Each is tested on representative tensors from a real model (Zamba2-7B by default).

Usage:
    python3 tools/bench_compression_methods.py \
        --model-dir ~/models/zamba2-7b-instruct \
        --tensors 4 \
        [--output receipts/compression_rd/methods_bench.json]
"""

import argparse
import json
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Compression methods
# ---------------------------------------------------------------------------

def vq_compress(data_flat: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard scalar VQ. Returns (codebook[k], indices[N])."""
    km = MiniBatchKMeans(n_clusters=k, n_init=1, max_iter=50,
                         batch_size=min(10000, len(data_flat)),
                         random_state=42)
    km.fit(data_flat.reshape(-1, 1))
    codebook = km.cluster_centers_.ravel().astype(np.float32)
    indices = km.predict(data_flat.reshape(-1, 1)).astype(np.int32)
    return codebook, indices


def vq_decode(codebook: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Reconstruct from VQ."""
    return codebook[indices]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat arrays."""
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    dot = np.dot(a_f, b_f)
    na = np.linalg.norm(a_f)
    nb = np.linalg.norm(b_f)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


# --- Method 1: Global VQ (baseline) ---

def method_global_vq(data: np.ndarray, k: int) -> dict:
    """Standard global VQ at given k."""
    flat = data.ravel().astype(np.float32)
    t0 = time.time()
    codebook, indices = vq_compress(flat, k)
    recon = vq_decode(codebook, indices).reshape(data.shape)
    elapsed = time.time() - t0

    bits_per_elem = np.log2(k)
    # Storage: indices (bits_per_elem bits each) + codebook (k * 4 bytes)
    index_bytes = len(flat) * bits_per_elem / 8
    codebook_bytes = k * 4
    total_bytes = index_bytes + codebook_bytes
    orig_bf16 = len(flat) * 2
    ratio_bf16 = orig_bf16 / total_bytes

    return {
        "method": f"global_vq_k{k}",
        "k": k,
        "cosine": cosine_sim(data, recon),
        "mse": mse(data, recon),
        "bits_per_elem": round(bits_per_elem, 2),
        "ratio_from_bf16": round(ratio_bf16, 3),
        "ratio_from_fp32": round(len(flat) * 4 / total_bytes, 3),
        "compressed_bytes": int(total_bytes),
        "time_s": round(elapsed, 2),
    }


# --- Method 2: Per-group uniform quantization ---
# This is what GPTQ/RTN/AWQ do: divide weights into groups, quantize each
# group to k uniform levels between its local min/max. Vectorized — O(n).

def method_pergroup_uniform(data: np.ndarray, k: int, group_size: int) -> dict:
    """Per-group uniform quantization: vectorized min/max → k levels per group."""
    flat = data.ravel().astype(np.float32)
    n = len(flat)
    t0 = time.time()

    # Pad to multiple of group_size
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    else:
        padded = flat.copy()

    n_groups = len(padded) // group_size
    groups = padded.reshape(n_groups, group_size)

    # Per-group min/max (vectorized)
    g_min = groups.min(axis=1, keepdims=True)  # [n_groups, 1]
    g_max = groups.max(axis=1, keepdims=True)
    g_range = g_max - g_min
    g_range = np.maximum(g_range, 1e-10)  # avoid div by zero

    # Quantize: map to [0, k-1], round, dequantize
    normalized = (groups - g_min) / g_range  # [0, 1]
    indices = np.clip(np.round(normalized * (k - 1)), 0, k - 1).astype(np.int32)
    recon_groups = g_min + indices.astype(np.float32) / (k - 1) * g_range

    recon = recon_groups.ravel()[:n].reshape(data.shape)
    elapsed = time.time() - t0

    # Storage: per group need min(FP16=2B) + scale(FP16=2B) + indices at log2(k) bits
    bits_per_idx = np.log2(k)
    index_bytes = n_groups * group_size * bits_per_idx / 8
    # Per-group overhead: min + scale = 4 bytes (FP16 each)
    group_overhead_bytes = n_groups * 4
    total_bytes = index_bytes + group_overhead_bytes
    orig_bf16 = n * 2
    bits_per_elem = (total_bytes * 8) / n

    return {
        "method": f"pergroup_k{k}_g{group_size}",
        "k": k,
        "group_size": group_size,
        "n_groups": n_groups,
        "cosine": cosine_sim(data, recon),
        "mse": mse(data, recon),
        "bits_per_elem": round(bits_per_elem, 2),
        "ratio_from_bf16": round(orig_bf16 / total_bytes, 3),
        "ratio_from_fp32": round(n * 4 / total_bytes, 3),
        "compressed_bytes": int(total_bytes),
        "group_overhead_pct": round(group_overhead_bytes / total_bytes * 100, 1),
        "time_s": round(elapsed, 2),
    }


# --- Method 2b: Per-group VQ (k-means per group) ---
# More accurate than uniform but MUCH slower. Only run on small tensors.

def method_pergroup_vq(data: np.ndarray, k: int, group_size: int) -> dict:
    """Per-group VQ with k-means. Only feasible for small tensors (<5M elements)."""
    flat = data.ravel().astype(np.float32)
    n = len(flat)

    if n > 5_000_000:
        # Too slow for large tensors — fall back to uniform
        result = method_pergroup_uniform(data, k, group_size)
        result["method"] = f"pergroup_vq_k{k}_g{group_size}_UNIFORM_FALLBACK"
        result["note"] = f"Fell back to uniform (tensor has {n} elements, limit 5M)"
        return result

    t0 = time.time()

    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    else:
        padded = flat.copy()

    n_groups = len(padded) // group_size
    recon_padded = np.zeros_like(padded)

    total_codebook_bytes = 0
    total_index_bits = 0

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = padded[start:end]

        km = MiniBatchKMeans(n_clusters=k, n_init=1, max_iter=30,
                             batch_size=min(group_size, 1000),
                             random_state=42)
        km.fit(group.reshape(-1, 1))
        cb = km.cluster_centers_.ravel().astype(np.float32)
        idx = km.predict(group.reshape(-1, 1))
        recon_padded[start:end] = cb[idx]

        total_codebook_bytes += k * 4
        total_index_bits += group_size * np.log2(k)

    recon = recon_padded[:n].reshape(data.shape)
    elapsed = time.time() - t0

    total_bytes = total_codebook_bytes + total_index_bits / 8
    orig_bf16 = n * 2
    bits_per_elem = (total_bytes * 8) / n

    return {
        "method": f"pergroup_vq_k{k}_g{group_size}",
        "k": k,
        "group_size": group_size,
        "n_groups": n_groups,
        "cosine": cosine_sim(data, recon),
        "mse": mse(data, recon),
        "bits_per_elem": round(bits_per_elem, 2),
        "ratio_from_bf16": round(orig_bf16 / total_bytes, 3),
        "ratio_from_fp32": round(n * 4 / total_bytes, 3),
        "compressed_bytes": int(total_bytes),
        "codebook_overhead_pct": round(total_codebook_bytes / total_bytes * 100, 1),
        "time_s": round(elapsed, 2),
    }


# --- Method 3: Residual VQ (RVQ) ---

def method_rvq(data: np.ndarray, k: int, stages: int) -> dict:
    """Multi-stage Residual VQ. Each stage quantizes the residual of the previous."""
    flat = data.ravel().astype(np.float32)
    n = len(flat)
    t0 = time.time()

    residual = flat.copy()
    stage_codebooks = []
    stage_indices = []

    for s in range(stages):
        codebook, indices = vq_compress(residual, k)
        recon_stage = vq_decode(codebook, indices)
        residual = residual - recon_stage
        stage_codebooks.append(codebook)
        stage_indices.append(indices)

    # Full reconstruction: sum of all stages
    recon = np.zeros(n, dtype=np.float32)
    for cb, idx in zip(stage_codebooks, stage_indices):
        recon += cb[idx]
    recon = recon.reshape(data.shape)
    elapsed = time.time() - t0

    bits_per_stage = np.log2(k)
    total_bits_per_elem = bits_per_stage * stages
    total_bytes = n * total_bits_per_elem / 8 + stages * k * 4  # indices + codebooks
    orig_bf16 = n * 2

    return {
        "method": f"rvq_{stages}stage_k{k}",
        "k": k,
        "stages": stages,
        "cosine": cosine_sim(data, recon),
        "mse": mse(data, recon),
        "bits_per_elem": round(total_bits_per_elem, 2),
        "ratio_from_bf16": round(orig_bf16 / total_bytes, 3),
        "ratio_from_fp32": round(n * 4 / total_bytes, 3),
        "compressed_bytes": int(total_bytes),
        "residual_norms": [round(float(np.linalg.norm(flat)), 4)]
                          + [round(float(np.linalg.norm(
                              flat - sum(cb[idx] for cb, idx in
                                         zip(stage_codebooks[:i+1],
                                             stage_indices[:i+1]))
                          )), 4) for i in range(stages)],
        "time_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Tensor loading
# ---------------------------------------------------------------------------

def load_test_tensors(model_dir: Path, n_tensors: int = 4) -> list[tuple[str, np.ndarray]]:
    """Load representative tensors from the model for testing."""
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single safetensors
        sf = safe_open(str(model_dir / "model.safetensors"), framework="pt")
        all_keys = list(sf.keys())
    else:
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        all_keys = list(weight_map.keys())

    # Pick diverse representative tensors
    # Priority: 1 large mamba in_proj, 1 mamba out_proj, 1 attention weight, 1 LoRA adapter
    candidates = {
        "mamba_in_proj": [],
        "mamba_out_proj": [],
        "attention": [],
        "lora_adapter": [],
        "ffn": [],
    }

    for name in all_keys:
        if "embed" in name or "norm" in name or "conv1d" in name:
            continue
        if not name.endswith(".weight"):
            continue
        if "in_proj" in name and "mamba" in name:
            candidates["mamba_in_proj"].append(name)
        elif "out_proj" in name and "mamba" in name:
            candidates["mamba_out_proj"].append(name)
        elif "self_attn" in name or "q_proj" in name or "k_proj" in name:
            candidates["attention"].append(name)
        elif "adapter" in name:
            candidates["lora_adapter"].append(name)
        elif "feed_forward" in name or "mlp" in name:
            candidates["ffn"].append(name)

    # Select one from each category (middle of model for representativeness)
    selected = []
    for cat in ["mamba_in_proj", "mamba_out_proj", "attention", "lora_adapter", "ffn"]:
        if candidates[cat]:
            mid = len(candidates[cat]) // 2
            selected.append(candidates[cat][mid])
        if len(selected) >= n_tensors:
            break

    # Fill remaining with any 2D weights
    if len(selected) < n_tensors:
        for name in all_keys:
            if name not in selected and name.endswith(".weight"):
                selected.append(name)
            if len(selected) >= n_tensors:
                break

    # Load tensors
    result = []
    for name in selected:
        if index_path.exists():
            shard = weight_map[name]
            sf = safe_open(str(model_dir / shard), framework="pt")
        else:
            sf = safe_open(str(model_dir / "model.safetensors"), framework="pt")

        tensor = sf.get_tensor(name).float().numpy()
        if tensor.ndim >= 2:
            result.append((name, tensor))
            print(f"  Loaded: {name} {tensor.shape} ({tensor.nbytes/1024/1024:.1f} MB)",
                  file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tensors", type=int, default=4,
                        help="Number of test tensors to use")
    parser.add_argument("--output", type=Path,
                        default=Path("receipts/compression_rd/methods_bench.json"))
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  Compression Methods R&D Bench", file=sys.stderr)
    print(f"  Model: {args.model_dir}", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)

    # Load test tensors
    tensors = load_test_tensors(args.model_dir, args.tensors)
    if not tensors:
        print("ERROR: No tensors loaded", file=sys.stderr)
        sys.exit(1)

    # Define test matrix
    methods = [
        # Baselines
        ("global_vq_k256", lambda d: method_global_vq(d, k=256)),
        ("global_vq_k64", lambda d: method_global_vq(d, k=64)),
        ("global_vq_k16", lambda d: method_global_vq(d, k=16)),

        # Per-group k=16 (uniform quantization — vectorized, fast)
        ("pergroup_k16_g128", lambda d: method_pergroup_uniform(d, k=16, group_size=128)),
        ("pergroup_k16_g256", lambda d: method_pergroup_uniform(d, k=16, group_size=256)),
        ("pergroup_k16_g512", lambda d: method_pergroup_uniform(d, k=16, group_size=512)),
        ("pergroup_k16_g1024", lambda d: method_pergroup_uniform(d, k=16, group_size=1024)),

        # Per-group k=64 (comparison)
        ("pergroup_k64_g128", lambda d: method_pergroup_uniform(d, k=64, group_size=128)),
        ("pergroup_k64_g256", lambda d: method_pergroup_uniform(d, k=64, group_size=256)),

        # Per-group VQ (k-means) on small tensor only — quality upper bound
        ("pergroup_vq_k16_g128", lambda d: method_pergroup_vq(d, k=16, group_size=128)),

        # RVQ
        ("rvq_2stage_k16", lambda d: method_rvq(d, k=16, stages=2)),
        ("rvq_3stage_k16", lambda d: method_rvq(d, k=16, stages=3)),
        ("rvq_2stage_k64", lambda d: method_rvq(d, k=64, stages=2)),
    ]

    all_results = []

    for tensor_name, tensor_data in tensors:
        print(f"\n  ── {tensor_name} {tensor_data.shape} ──", file=sys.stderr)
        kurt = float(((tensor_data - tensor_data.mean()) ** 4).mean() /
                     (((tensor_data - tensor_data.mean()) ** 2).mean() ** 2) - 3.0)
        print(f"  Kurtosis: {kurt:.2f}", file=sys.stderr)

        tensor_results = {
            "tensor": tensor_name,
            "shape": list(tensor_data.shape),
            "numel": int(tensor_data.size),
            "kurtosis": round(kurt, 2),
            "methods": [],
        }

        for method_name, method_fn in methods:
            print(f"    {method_name:30s} ", end="", file=sys.stderr, flush=True)
            try:
                result = method_fn(tensor_data)
                tensor_results["methods"].append(result)
                cos = result["cosine"]
                ratio = result["ratio_from_bf16"]
                bpe = result["bits_per_elem"]
                t = result["time_s"]
                # Color-code quality
                q_mark = "OK" if cos >= 0.998 else ("WARN" if cos >= 0.995 else "FAIL")
                print(f"cos={cos:.6f}  ratio_bf16={ratio:.2f}x  "
                      f"bpe={bpe:.1f}  {t:.1f}s  [{q_mark}]",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr, flush=True)
                tensor_results["methods"].append({
                    "method": method_name,
                    "error": str(e),
                })

        all_results.append(tensor_results)

    # Summary table
    print(f"\n\n{'='*70}", file=sys.stderr)
    print(f"  SUMMARY: Mean cosine across all test tensors", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"  {'Method':35s} {'Cos':>8s} {'Ratio':>8s} {'BPE':>6s} {'Verdict':>8s}",
          file=sys.stderr)
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*6} {'-'*8}", file=sys.stderr)

    method_names_seen = []
    for method_name, _ in methods:
        cosines = []
        ratios = []
        bpes = []
        for tr in all_results:
            for mr in tr["methods"]:
                if mr.get("method", "").replace(f"global_vq_k", "global_vq_k") == method_name or \
                   mr.get("method", "") == method_name:
                    if "cosine" in mr:
                        cosines.append(mr["cosine"])
                        ratios.append(mr["ratio_from_bf16"])
                        bpes.append(mr["bits_per_elem"])

        # Try matching by the actual method field
        if not cosines:
            for tr in all_results:
                for mr in tr["methods"]:
                    if "method" in mr and method_name in mr["method"]:
                        if "cosine" in mr:
                            cosines.append(mr["cosine"])
                            ratios.append(mr["ratio_from_bf16"])
                            bpes.append(mr["bits_per_elem"])

        if cosines:
            avg_cos = np.mean(cosines)
            avg_ratio = np.mean(ratios)
            avg_bpe = np.mean(bpes)
            verdict = "PASS" if avg_cos >= 0.998 else ("MARGINAL" if avg_cos >= 0.995 else "FAIL")
            print(f"  {method_name:35s} {avg_cos:8.6f} {avg_ratio:7.2f}x {avg_bpe:6.1f} {verdict:>8s}",
                  file=sys.stderr)

    # Cost block
    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    receipt = {
        "work_order": "WO-COMPRESSION-RD-01",
        "question": "Which method breaks the 2x BF16 ceiling while maintaining cos>=0.998?",
        "model": args.model_dir.name,
        "n_tensors_tested": len(tensors),
        "results": all_results,
        "cost": cost,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\n  Receipt: {args.output}", file=sys.stderr)
    print(f"  Wall time: {cost['wall_time_s']}s", file=sys.stderr)

    # Also print JSON to stdout for scripting
    print(json.dumps({"summary": {
        tr["tensor"]: {
            mr["method"]: {"cos": mr["cosine"], "ratio_bf16": mr["ratio_from_bf16"]}
            for mr in tr["methods"] if "cosine" in mr
        }
        for tr in all_results
    }}, indent=2))


if __name__ == "__main__":
    main()
