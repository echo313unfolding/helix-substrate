#!/usr/bin/env python3
"""
Residual Structure Gate — cheapest falsifier for executable sidecar.

Question: Do VQ residuals (weight - codebook_reconstruction) have exploitable
sequential structure, or are they IID?

If IID → executable sidecar idea dies. No walk format helps.
If structured → green light for walk-format spike.

Method per tensor class:
  1. Compute per-class VQ residuals (same as bench_per_class_codebook.py)
  2. Flatten three ways: row-major, column-major, SVD-principal
  3. Autocorrelation function (ACF) — decaying = structure, flat = IID
  4. Power spectral density — concentrated bands = structure, flat = IID
  5. SVD rank deficiency — top-k explains >90% variance = low-rank = structure

Kill condition (ALL must hold for idea to die):
  - All 3 axes × all classes: ACF drops to <0.05 by lag 5
  - All power spectra flat (max/mean < 3.0)
  - All classes need >90% of SVD components for 99% explained variance

Green light (ANY one = idea lives):
  - Any class shows ACF > 0.1 at lag 10 on any axis
  - Any power spectrum has spectral ratio > 5.0
  - Any class has 99% variance from <50% of components

Receipt: receipts/residual_structure_gate/

Usage:
  python3 tools/residual_structure_gate.py --model EchoLabs33/mamba-130m-hxq
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
# Reuse from bench_per_class_codebook.py
# ============================================================================

_SUFFIX_CLASS = {
    "q_proj": "attention_qk", "k_proj": "attention_qk",
    "v_proj": "attention_vo", "o_proj": "attention_vo",
    "gate_proj": "ffn", "up_proj": "ffn", "down_proj": "ffn",
    "gate_up_proj": "ffn",
}


def classify_with_fallback(name: str, shape: tuple) -> TensorClass:
    tc = classify_tensor(name, shape=shape)
    if tc != TensorClass.UNKNOWN:
        return tc
    stem = name.removesuffix(".weight")
    leaf = stem.rsplit(".", 1)[-1]
    if leaf in _SUFFIX_CLASS:
        return TensorClass(_SUFFIX_CLASS[leaf])
    if ".mamba." in stem or ".mamba_decoder." in stem:
        return TensorClass("ffn")
    if "adapter_list" in stem:
        return TensorClass("ffn")
    return TensorClass.UNKNOWN


def load_model_weights(model_id: str) -> dict:
    import helix_substrate.hf_quantizer  # noqa
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

    for name, module in model.named_modules():
        if isinstance(module, HelixLinear):
            w = module.decode_weight().detach().cpu().float().numpy()
            full_name = name + ".weight"
            tc = classify_with_fallback(full_name, shape=w.shape)
            if tc in (TensorClass.NORM, TensorClass.EMBEDDING, TensorClass.LM_HEAD):
                continue
            tensors[full_name] = (w, tc)
            helix_names.add(name)

    for name, param in model.named_parameters():
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

    print(f"  Loaded {len(tensors)} tensors ({len(helix_names)} HelixLinear)", file=sys.stderr)
    return tensors


def fit_codebook(flat_data: np.ndarray, n_clusters: int = 256) -> np.ndarray:
    if len(flat_data) > 500_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(flat_data), 500_000, replace=False)
        sample = flat_data[idx]
    else:
        sample = flat_data
    codebook, _ = _simple_kmeans(sample, n_clusters, max_iters=10)
    return codebook


def compute_residual_2d(weight: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Compute 2D residual matrix: weight - VQ_reconstruction."""
    flat = weight.ravel().astype(np.float32)
    cb = codebook.astype(np.float32)

    # Chunked assignment
    chunk_size = 1_000_000
    indices = np.empty(len(flat), dtype=np.uint8)
    for i in range(0, len(flat), chunk_size):
        chunk = flat[i:i + chunk_size]
        x_sq = chunk ** 2
        c_sq = cb ** 2
        dists = x_sq[:, None] - 2 * chunk[:, None] * cb[None, :] + c_sq[None, :]
        indices[i:i + chunk_size] = np.argmin(dists, axis=1)

    reconstructed = codebook[indices].reshape(weight.shape)
    return weight.astype(np.float32) - reconstructed


# ============================================================================
# Structure tests
# ============================================================================

def autocorrelation(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Normalized autocorrelation function for 1D signal."""
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-30:
        return np.zeros(max_lag + 1)
    n = len(x)
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        acf[lag] = np.dot(x[:n - lag], x[lag:]) / (var * n)
    return acf


def spectral_ratio(x: np.ndarray) -> float:
    """Ratio of max to mean power spectral density. Flat spectrum → ~1.0."""
    x = x - x.mean()
    if np.var(x) < 1e-30:
        return 1.0
    # Use rfft for real signal
    n = min(len(x), 1_000_000)  # Cap for speed
    psd = np.abs(np.fft.rfft(x[:n])) ** 2
    psd = psd[1:]  # Drop DC
    if len(psd) == 0 or psd.mean() < 1e-30:
        return 1.0
    return float(psd.max() / psd.mean())


def svd_rank_deficiency(residual_2d: np.ndarray) -> dict:
    """Check how many SVD components needed for 99% explained variance."""
    # Subsample if huge
    r, c = residual_2d.shape
    if r > 1000:
        idx = np.random.RandomState(42).choice(r, 1000, replace=False)
        residual_2d = residual_2d[idx, :]
    if c > 1000:
        idx = np.random.RandomState(43).choice(c, 1000, replace=False)
        residual_2d = residual_2d[:, idx]

    try:
        _, s, _ = np.linalg.svd(residual_2d, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"n_components_99pct": residual_2d.shape[0],
                "total_components": residual_2d.shape[0],
                "ratio_99pct": 1.0,
                "top10_explained": 0.0}

    total_var = np.sum(s ** 2)
    if total_var < 1e-30:
        return {"n_components_99pct": 0, "total_components": len(s),
                "ratio_99pct": 0.0, "top10_explained": 0.0}

    cumvar = np.cumsum(s ** 2) / total_var
    n99 = int(np.searchsorted(cumvar, 0.99) + 1)
    top10_exp = float(cumvar[min(9, len(cumvar) - 1)])

    return {
        "n_components_99pct": n99,
        "total_components": len(s),
        "ratio_99pct": round(n99 / len(s), 4),
        "top10_explained": round(top10_exp, 4),
    }


def svd_principal_flatten(residual_2d: np.ndarray) -> np.ndarray:
    """Flatten residual along top SVD direction."""
    r, c = residual_2d.shape
    # Subsample rows for SVD computation, then project all
    if r > 1000:
        sub_idx = np.random.RandomState(42).choice(r, 1000, replace=False)
        sub = residual_2d[sub_idx, :]
    else:
        sub = residual_2d

    try:
        _, _, Vt = np.linalg.svd(sub, full_matrices=False)
        # Project all rows onto top right-singular vector
        return residual_2d @ Vt[0]
    except np.linalg.LinAlgError:
        return residual_2d.ravel()


def analyze_tensor_residual(residual_2d: np.ndarray, max_lag: int = 50) -> dict:
    """Full structure analysis on one tensor's 2D residual."""
    # Three flatten orders
    row_flat = residual_2d.ravel()             # row-major (C order)
    col_flat = residual_2d.T.ravel()           # column-major
    svd_flat = svd_principal_flatten(residual_2d)  # SVD principal direction

    results = {}
    for name, flat in [("row", row_flat), ("col", col_flat), ("svd", svd_flat)]:
        acf = autocorrelation(flat, max_lag=max_lag)
        sr = spectral_ratio(flat)
        results[name] = {
            "acf_lag1": round(float(acf[1]), 6),
            "acf_lag5": round(float(acf[5]) if len(acf) > 5 else 0.0, 6),
            "acf_lag10": round(float(acf[10]) if len(acf) > 10 else 0.0, 6),
            "acf_lag20": round(float(acf[20]) if len(acf) > 20 else 0.0, 6),
            "acf_lag50": round(float(acf[50]) if len(acf) > 50 else 0.0, 6),
            "spectral_ratio": round(sr, 2),
        }

    # SVD rank deficiency on the 2D residual directly
    results["svd_rank"] = svd_rank_deficiency(residual_2d)

    return results


# ============================================================================
# Main
# ============================================================================

def run_gate(model_id: str) -> dict:
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    tensors = load_model_weights(model_id)

    # Group by class
    by_class = defaultdict(list)
    for name, (w, tc) in tensors.items():
        by_class[tc.value].append((name, w))

    # Fit per-class codebooks
    print("Fitting per-class codebooks...", file=sys.stderr)
    class_codebooks = {}
    for cls, items in by_class.items():
        cls_flat = np.concatenate([w.ravel() for _, w in items])
        class_codebooks[cls] = fit_codebook(cls_flat)
        print(f"  {cls}: {len(items)} tensors", file=sys.stderr)

    # Analyze residuals per tensor
    print("Computing residuals and structure analysis...", file=sys.stderr)
    per_tensor = []
    class_acf_summaries = defaultdict(lambda: {"row": [], "col": [], "svd": []})

    for name, (w, tc) in tensors.items():
        cls = tc.value
        residual_2d = compute_residual_2d(w, class_codebooks[cls])
        analysis = analyze_tensor_residual(residual_2d, max_lag=50)

        entry = {"name": name, "class": cls, "shape": list(w.shape)}
        entry.update(analysis)
        per_tensor.append(entry)

        for axis in ["row", "col", "svd"]:
            class_acf_summaries[cls][axis].append({
                "acf_lag1": analysis[axis]["acf_lag1"],
                "acf_lag5": analysis[axis]["acf_lag5"],
                "acf_lag10": analysis[axis]["acf_lag10"],
                "spectral_ratio": analysis[axis]["spectral_ratio"],
            })

    # Aggregate by class
    print("\n" + "=" * 60, file=sys.stderr)
    print("  RESIDUAL STRUCTURE GATE", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    class_summary = {}
    any_green = False

    for cls in sorted(by_class.keys()):
        summary = {}
        for axis in ["row", "col", "svd"]:
            acf1_vals = [x["acf_lag1"] for x in class_acf_summaries[cls][axis]]
            acf5_vals = [x["acf_lag5"] for x in class_acf_summaries[cls][axis]]
            acf10_vals = [x["acf_lag10"] for x in class_acf_summaries[cls][axis]]
            sr_vals = [x["spectral_ratio"] for x in class_acf_summaries[cls][axis]]

            mean_acf1 = float(np.mean(acf1_vals))
            mean_acf5 = float(np.mean(acf5_vals))
            mean_acf10 = float(np.mean(acf10_vals))
            max_acf10 = float(np.max(acf10_vals))
            mean_sr = float(np.mean(sr_vals))
            max_sr = float(np.max(sr_vals))

            summary[axis] = {
                "mean_acf_lag1": round(mean_acf1, 6),
                "mean_acf_lag5": round(mean_acf5, 6),
                "mean_acf_lag10": round(mean_acf10, 6),
                "max_acf_lag10": round(max_acf10, 6),
                "mean_spectral_ratio": round(mean_sr, 2),
                "max_spectral_ratio": round(max_sr, 2),
            }

            # Green light checks
            if max_acf10 > 0.1:
                any_green = True
            if max_sr > 5.0:
                any_green = True

        # SVD rank deficiency (pooled across tensors in class)
        rank_ratios = [t["svd_rank"]["ratio_99pct"] for t in per_tensor if t["class"] == cls]
        top10s = [t["svd_rank"]["top10_explained"] for t in per_tensor if t["class"] == cls]
        mean_rank_ratio = float(np.mean(rank_ratios))
        mean_top10 = float(np.mean(top10s))

        summary["svd_rank"] = {
            "mean_ratio_99pct": round(mean_rank_ratio, 4),
            "mean_top10_explained": round(mean_top10, 4),
            "min_ratio_99pct": round(float(np.min(rank_ratios)), 4),
        }

        # Green: <50% components for 99% variance
        if float(np.min(rank_ratios)) < 0.5:
            any_green = True

        class_summary[cls] = summary

        print(f"\n  {cls} ({len(by_class[cls])} tensors):", file=sys.stderr)
        for axis in ["row", "col", "svd"]:
            s = summary[axis]
            print(f"    {axis:4s}: ACF@1={s['mean_acf_lag1']:+.4f}  "
                  f"ACF@10={s['mean_acf_lag10']:+.4f} (max={s['max_acf_lag10']:+.4f})  "
                  f"SR={s['mean_spectral_ratio']:.1f} (max={s['max_spectral_ratio']:.1f})",
                  file=sys.stderr)
        rs = summary["svd_rank"]
        print(f"    rank: 99%var needs {rs['mean_ratio_99pct']*100:.1f}% components "
              f"(min={rs['min_ratio_99pct']*100:.1f}%)  "
              f"top10 explains {rs['mean_top10_explained']*100:.1f}%",
              file=sys.stderr)

    # Gate decision
    decision = "GREEN_LIGHT" if any_green else "KILL"

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  GATE DECISION: {decision}", file=sys.stderr)
    if decision == "GREEN_LIGHT":
        print("  Structure detected — executable sidecar spike is justified.", file=sys.stderr)
    else:
        print("  Residuals are IID — executable sidecar idea is DEAD.", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

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
        "test": "residual_structure_gate",
        "model": model_id,
        "question": "Do VQ residuals have exploitable sequential structure (non-IID)?",
        "kill_condition": "All axes × all classes: ACF<0.05@lag5, spectral_ratio<3, rank_ratio>0.9",
        "green_condition": "Any class: ACF>0.1@lag10 OR spectral_ratio>5 OR rank_ratio<0.5",
        "decision": decision,
        "n_tensors": len(tensors),
        "n_classes": len(by_class),
        "class_summary": class_summary,
        "per_tensor_results": per_tensor,
        "cost": cost,
    }

    return receipt


def main():
    parser = argparse.ArgumentParser(description="Residual structure gate for executable sidecar")
    parser.add_argument("--model", type=str, default="EchoLabs33/mamba-130m-hxq")
    args = parser.parse_args()

    receipt = run_gate(args.model)

    receipt_dir = Path.home() / "helix-substrate" / "receipts" / "residual_structure_gate"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1].replace(".", "_")
    receipt_path = receipt_dir / f"structure_gate_{model_tag}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(json.dumps(receipt, indent=2))
    print(f"\nReceipt: {receipt_path}", file=sys.stderr)

    return 0 if receipt["decision"] == "GREEN_LIGHT" else 1


if __name__ == "__main__":
    sys.exit(main())
