#!/usr/bin/env python3
"""
Per-class codebook vs global codebook sidecar norm test.

Hypothesis: Fitting separate VQ codebooks per tensor class (attention, SSM, MLP)
produces lower reconstruction residuals than one global codebook, because each
class occupies a distinct region in (kurtosis, magnitude, sensitivity) space.

Method:
  1. Load a compressed HXQ model (Zamba2-2.7B or any model with stats.json)
  2. For each HelixLinear tensor, read the original weight, record its class
  3. Global baseline: fit one k-means codebook across ALL tensors' flattened weights
  4. Per-class: fit one k-means codebook per TensorClass
  5. Compare: sidecar L2 norms (residual after VQ reconstruction)

Gate: per-class sidecar norm drops >15% averaged across classes → per-class wins.

Receipt: receipts/per_class_codebook/per_class_vs_global.json

Usage:
  python3 tools/bench_per_class_codebook.py --model EchoLabs33/mamba-130m-hxq
  python3 tools/bench_per_class_codebook.py --model EchoLabs33/Zamba2-2.7B-instruct-hxq
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_substrate.tensor_policy import classify_tensor, TensorClass
from helix_substrate.cdna_encoder import _simple_kmeans


# ============================================================================
# Suffix-based fallback for architectures classify_tensor doesn't cover
# (e.g. Zamba2: shared_transformer.self_attn.q_proj, mamba.in_proj)
# ============================================================================

_SUFFIX_CLASS = {
    "q_proj": "attention_qk",
    "k_proj": "attention_qk",
    "v_proj": "attention_vo",
    "o_proj": "attention_vo",
    "gate_proj": "ffn",
    "up_proj": "ffn",
    "down_proj": "ffn",
    "gate_up_proj": "ffn",
}


def classify_with_fallback(name: str, shape: tuple) -> TensorClass:
    """classify_tensor, then fall back to suffix matching + mamba detection."""
    tc = classify_tensor(name, shape=shape)
    if tc != TensorClass.UNKNOWN:
        return tc

    # Strip ".weight" suffix before suffix matching
    stem = name.removesuffix(".weight")

    # Suffix match (covers Zamba2 shared_transformer paths & adapter lists)
    leaf = stem.rsplit(".", 1)[-1]
    if leaf in _SUFFIX_CLASS:
        return TensorClass(_SUFFIX_CLASS[leaf])

    # Mamba SSM paths: *.mamba.in_proj, *.mamba.out_proj, *.mamba_decoder.*
    if ".mamba." in stem or ".mamba_decoder." in stem:
        return TensorClass("ffn")  # Same policy as tensor_policy.py lines 103-104

    # Adapter list entries (e.g. gate_up_proj_adapter_list.1.0) — FFN adapters
    if "adapter_list" in stem:
        return TensorClass("ffn")

    return TensorClass.UNKNOWN


# ============================================================================
# Config
# ============================================================================

N_CLUSTERS = 256          # Same k for both global and per-class
KMEANS_ITERS = 10
KMEANS_SAMPLE = 500_000   # Max samples for k-means fitting
GATE_THRESHOLD = 0.15     # 15% reduction in sidecar norm = PASS

RECEIPT_DIR = Path.home() / "helix-substrate" / "receipts" / "per_class_codebook"


# ============================================================================
# Core functions
# ============================================================================

def load_model_weights(model_id: str) -> dict:
    """Load model and return {name: (weight_numpy, tensor_class)} for all 2D tensors.

    For HelixLinear modules, calls decode_weight() to reconstruct the full
    weight matrix from codebook + indices.  Falls back to named_parameters()
    for any remaining plain 2D tensors (e.g. A_log).
    """
    import helix_substrate.hf_quantizer  # noqa — registers HXQ
    from helix_substrate.helix_linear import HelixLinear
    from transformers import AutoModelForCausalLM
    import torch

    print(f"Loading {model_id}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )

    tensors = {}
    helix_names = set()

    # Pass 1: HelixLinear modules — decode compressed weights
    for name, module in model.named_modules():
        if isinstance(module, HelixLinear):
            w = module.decode_weight().detach().cpu().float().numpy()
            # classify_tensor expects ".weight" suffix for pattern matching
            full_name = name + ".weight"
            tc = classify_with_fallback(full_name, shape=w.shape)
            if tc in (TensorClass.NORM, TensorClass.EMBEDDING, TensorClass.LM_HEAD):
                continue
            tensors[full_name] = (w, tc)
            helix_names.add(name)

    # Pass 2: remaining plain 2D parameters (A_log, D, etc.)
    for name, param in model.named_parameters():
        # Skip anything under a HelixLinear we already decoded
        module_prefix = name.rsplit(".", 1)[0] if "." in name else name
        if module_prefix in helix_names:
            continue
        w = param.detach().cpu().float().numpy()
        if w.ndim != 2:
            continue
        tc = classify_with_fallback(name, shape=w.shape)
        if tc in (TensorClass.NORM, TensorClass.EMBEDDING, TensorClass.LM_HEAD):
            continue
        tensors[name] = (w, tc)

    print(f"  Loaded {len(tensors)} tensors for VQ comparison "
          f"({len(helix_names)} HelixLinear decoded)", file=sys.stderr)
    return tensors


def fit_codebook(flat_data: np.ndarray, n_clusters: int = N_CLUSTERS) -> np.ndarray:
    """Fit a k-means codebook on (possibly subsampled) data."""
    if len(flat_data) > KMEANS_SAMPLE:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(flat_data), KMEANS_SAMPLE, replace=False)
        sample = flat_data[idx]
    else:
        sample = flat_data
    codebook, _ = _simple_kmeans(sample, n_clusters, max_iters=KMEANS_ITERS)
    return codebook


def assign_and_residual(flat: np.ndarray, codebook: np.ndarray) -> dict:
    """Assign flat weights to nearest codebook entry, compute residual stats."""
    # Chunked assignment to avoid OOM
    chunk_size = 1_000_000
    indices = np.empty(len(flat), dtype=np.uint8 if len(codebook) <= 256 else np.uint16)

    for i in range(0, len(flat), chunk_size):
        chunk = flat[i:i + chunk_size].astype(np.float32)
        cb = codebook.astype(np.float32)
        # Distance: |x - c|^2 = x^2 - 2xc + c^2
        x_sq = chunk ** 2
        c_sq = cb ** 2
        dists = x_sq[:, None] - 2 * chunk[:, None] * cb[None, :] + c_sq[None, :]
        indices[i:i + chunk_size] = np.argmin(dists, axis=1)

    reconstructed = codebook[indices]
    residual = flat - reconstructed

    l2_norm = float(np.linalg.norm(residual))
    cosine = float(np.dot(flat, reconstructed) / (np.linalg.norm(flat) * np.linalg.norm(reconstructed) + 1e-30))
    mse = float(np.mean(residual ** 2))

    return {
        "l2_norm": round(l2_norm, 6),
        "cosine": round(cosine, 6),
        "mse": round(mse, 10),
        "max_abs_error": round(float(np.max(np.abs(residual))), 6),
        "n_elements": len(flat),
    }


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(model_id: str) -> dict:
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    tensors = load_model_weights(model_id)

    # Group tensors by class
    by_class = defaultdict(list)
    for name, (w, tc) in tensors.items():
        by_class[tc.value].append((name, w))

    print(f"\nTensor classes:", file=sys.stderr)
    for cls, items in sorted(by_class.items()):
        total_params = sum(w.size for _, w in items)
        print(f"  {cls}: {len(items)} tensors, {total_params:,} params", file=sys.stderr)

    # ---- Step 1: Global codebook (all flattened weights pooled) ----
    print(f"\nFitting GLOBAL codebook (k={N_CLUSTERS})...", file=sys.stderr)
    all_flat = np.concatenate([w.ravel() for _, (w, _) in tensors.items()])
    global_codebook = fit_codebook(all_flat)
    print(f"  Global codebook range: [{global_codebook.min():.6f}, {global_codebook.max():.6f}]",
          file=sys.stderr)

    # ---- Step 2: Per-class codebooks ----
    print(f"\nFitting PER-CLASS codebooks (k={N_CLUSTERS} each)...", file=sys.stderr)
    class_codebooks = {}
    for cls, items in by_class.items():
        cls_flat = np.concatenate([w.ravel() for _, w in items])
        class_codebooks[cls] = fit_codebook(cls_flat)
        print(f"  {cls}: range [{class_codebooks[cls].min():.6f}, {class_codebooks[cls].max():.6f}]",
              file=sys.stderr)

    # ---- Step 3: Evaluate both on every tensor ----
    print(f"\nEvaluating residuals...", file=sys.stderr)
    results_per_tensor = []
    class_global_norms = defaultdict(list)
    class_perclass_norms = defaultdict(list)

    for name, (w, tc) in tensors.items():
        flat = w.ravel().astype(np.float32)
        cls = tc.value

        # Global
        global_res = assign_and_residual(flat, global_codebook)
        # Per-class
        perclass_res = assign_and_residual(flat, class_codebooks[cls])

        class_global_norms[cls].append(global_res["l2_norm"])
        class_perclass_norms[cls].append(perclass_res["l2_norm"])

        results_per_tensor.append({
            "name": name,
            "class": cls,
            "n_params": w.size,
            "global_l2": global_res["l2_norm"],
            "global_cosine": global_res["cosine"],
            "global_mse": global_res["mse"],
            "perclass_l2": perclass_res["l2_norm"],
            "perclass_cosine": perclass_res["cosine"],
            "perclass_mse": perclass_res["mse"],
            "l2_reduction_pct": round(
                100 * (1 - perclass_res["l2_norm"] / (global_res["l2_norm"] + 1e-30)), 2
            ),
        })

    # ---- Step 4: Aggregate by class ----
    class_summary = {}
    for cls in by_class.keys():
        g_mean = np.mean(class_global_norms[cls])
        p_mean = np.mean(class_perclass_norms[cls])
        reduction = 1 - p_mean / (g_mean + 1e-30)
        class_summary[cls] = {
            "n_tensors": len(by_class[cls]),
            "global_l2_mean": round(float(g_mean), 4),
            "perclass_l2_mean": round(float(p_mean), 4),
            "l2_reduction_pct": round(float(reduction * 100), 2),
        }

    # Overall weighted average
    total_global = sum(np.sum(class_global_norms[c]) for c in by_class)
    total_perclass = sum(np.sum(class_perclass_norms[c]) for c in by_class)
    overall_reduction = 1 - total_perclass / (total_global + 1e-30)

    # ---- Gate decision ----
    decision = "PASS" if overall_reduction >= GATE_THRESHOLD else "FAIL"

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  PER-CLASS CODEBOOK TEST", file=sys.stderr)
    print(f"  Model: {model_id}", file=sys.stderr)
    print(f"  k={N_CLUSTERS}, {len(tensors)} tensors, {len(by_class)} classes", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"\nPer-class results:", file=sys.stderr)
    for cls, summary in sorted(class_summary.items()):
        print(f"  {cls}: global_L2={summary['global_l2_mean']:.4f} → "
              f"perclass_L2={summary['perclass_l2_mean']:.4f} "
              f"({summary['l2_reduction_pct']:+.1f}%)", file=sys.stderr)
    print(f"\n  Overall L2 reduction: {overall_reduction*100:.1f}%", file=sys.stderr)
    print(f"  Gate threshold: {GATE_THRESHOLD*100:.0f}%", file=sys.stderr)
    print(f"  Decision: {decision}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

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
        "test": "per_class_vs_global_codebook",
        "model": model_id,
        "n_clusters": N_CLUSTERS,
        "kmeans_iters": KMEANS_ITERS,
        "n_tensors": len(tensors),
        "n_classes": len(by_class),
        "class_summary": class_summary,
        "overall_l2_reduction_pct": round(float(overall_reduction * 100), 2),
        "gate_threshold_pct": GATE_THRESHOLD * 100,
        "decision": decision,
        "per_tensor_results": results_per_tensor,
        "cost": cost,
    }

    return receipt


def main():
    parser = argparse.ArgumentParser(description="Per-class vs global codebook sidecar test")
    parser.add_argument("--model", type=str, default="EchoLabs33/mamba-130m-hxq",
                        help="HuggingFace model ID to test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model, print class distribution, exit")
    args = parser.parse_args()

    if args.dry_run:
        tensors = load_model_weights(args.model)
        by_class = defaultdict(int)
        for _, (w, tc) in tensors.items():
            by_class[tc.value] += 1
        print(f"Classes: {dict(by_class)}")
        return 0

    receipt = run_experiment(args.model)

    # Save receipt
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace(".", "_")
    receipt_path = RECEIPT_DIR / f"per_class_vs_global_{model_tag}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(json.dumps(receipt, indent=2))
    print(f"\nReceipt: {receipt_path}", file=sys.stderr)

    return 0 if receipt["decision"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
