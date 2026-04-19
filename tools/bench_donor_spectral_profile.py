#!/usr/bin/env python3
"""WO-DONOR-SURVEY-01 Step 2: Kurtosis + Effective Rank Profile of Donor Candidates

For top donor candidates (Mamba2-1.3B, Qwen2.5-Coder-1.5B, and others from Step 1):
  - Compute kurtosis per weight tensor
  - Compute effective rank via SVD for key layers (in_proj, out_proj, q_proj, v_proj)
  - Cross-reference with existing kurtosis routing receipt

Decision gate: High kurtosis (outlier-heavy, SVD-routable) vs low kurtosis (smooth, VQ-optimal).

Receipt includes cost block per WO-RECEIPT-COST-01.
"""

import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

t_start = time.time()
cpu_start = time.process_time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

MODELS_DIR = Path("/home/voidstr3m33/models")
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "donor_survey"
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

# Primary donor candidates
PRIMARY_DONORS = [
    "mamba2-1.3b-helix",
    "qwen2.5-coder-1.5b-helix",
    "qwen2.5-coder-1.5b-instruct-helix",
]

# Secondary donors for comparison
SECONDARY_DONORS = [
    "mamba-130m-helix",
    "zamba2-1.2b-helix",
    "qwen2.5-3b-instruct-helix",
    "tinyllama-1.1b-helix",
]


def compute_kurtosis(arr: np.ndarray) -> float:
    """Fisher excess kurtosis: E[((x-mu)/sigma)^4] - 3."""
    flat = arr.ravel().astype(np.float64)
    if flat.size < 4:
        return 0.0
    m = np.mean(flat)
    s = np.std(flat)
    if s < 1e-8:
        return 0.0
    return float(np.mean(((flat - m) / s) ** 4) - 3.0)


def effective_rank(matrix: np.ndarray, max_dim: int = 2048) -> float:
    """Effective rank via exp(Shannon entropy of normalized singular values).

    Caps matrix dimension to max_dim for tractability on large layers.
    """
    if matrix.ndim != 2:
        return 0.0
    m, n = matrix.shape
    # Subsample if too large
    if m > max_dim:
        indices = np.linspace(0, m - 1, max_dim, dtype=int)
        matrix = matrix[indices]
    if n > max_dim:
        indices = np.linspace(0, n - 1, max_dim, dtype=int)
        matrix = matrix[:, indices]

    try:
        s = np.linalg.svd(matrix.astype(np.float32), compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0

    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0

    # Normalize to probability distribution
    s_norm = s / s.sum()
    entropy = -np.sum(s_norm * np.log(s_norm))
    return float(np.exp(entropy))


def stable_rank(matrix: np.ndarray) -> float:
    """Stable rank: ||A||_F^2 / ||A||_2^2. Cheaper than full SVD."""
    if matrix.ndim != 2:
        return 0.0
    fro_sq = float(np.sum(matrix.astype(np.float64) ** 2))
    try:
        s_max = np.linalg.svd(matrix.astype(np.float32), compute_uv=False)[0]
    except np.linalg.LinAlgError:
        return 0.0
    if s_max < 1e-10:
        return 0.0
    return fro_sq / (float(s_max) ** 2)


def classify_role(name: str) -> str:
    """Extract role from tensor name."""
    for role in ["in_proj", "out_proj", "dt_proj", "x_proj",
                 "q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]:
        if role in name:
            return role
    return "other"


def profile_model(model_dir: Path, compute_eff_rank: bool = True) -> dict:
    """Compute kurtosis + eff_rank profile for a model's compressed layers."""
    from safetensors import safe_open

    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if not shards:
            return {"model": model_dir.name, "error": "no safetensors found"}
        safetensors_files = shards
    else:
        safetensors_files = [safetensors_path]

    # Collect keys
    all_keys = []
    key_to_file = {}
    for sf_path in safetensors_files:
        sf = safe_open(str(sf_path), framework="numpy")
        for k in sf.keys():
            all_keys.append(k)
            key_to_file[k] = sf_path

    codebook_keys = sorted([k for k in all_keys if k.endswith(".codebook")])
    base_names = [k.rsplit(".codebook", 1)[0] for k in codebook_keys]

    layer_results = []
    for base_name in base_names:
        cb_key = f"{base_name}.codebook"
        idx_key = f"{base_name}.indices"
        sp_key = f"{base_name}.sidecar_positions"
        sv_key = f"{base_name}.sidecar_values"

        if not all(k in key_to_file for k in [cb_key, idx_key, sp_key, sv_key]):
            continue

        sf = safe_open(str(key_to_file[cb_key]), framework="numpy")
        codebook = sf.get_tensor(cb_key).astype(np.float32)
        indices = sf.get_tensor(idx_key)
        sidecar_pos = sf.get_tensor(sp_key)
        sidecar_val = sf.get_tensor(sv_key).astype(np.float32)

        # Reconstruct weight matrix
        weight = codebook[indices.astype(np.int32)]
        # Apply sidecars
        if sidecar_pos.size > 0:
            flat = weight.ravel()
            valid_mask = sidecar_pos < flat.size
            flat[sidecar_pos[valid_mask].astype(np.int64)] = sidecar_val[valid_mask]
            weight = flat.reshape(weight.shape)

        role = classify_role(base_name)
        shape = list(indices.shape)

        # Kurtosis (fast, O(n))
        kurt = compute_kurtosis(weight)

        # Weight RMS
        weight_rms = float(np.sqrt(np.mean(weight ** 2)))

        # Codebook kurtosis (distribution of centroids)
        cb_kurt = compute_kurtosis(codebook)

        result = {
            "tensor_name": base_name,
            "role": role,
            "shape": shape,
            "n_params": int(np.prod(shape)),
            "kurtosis": round(kurt, 4),
            "codebook_kurtosis": round(cb_kurt, 4),
            "weight_rms": round(weight_rms, 6),
        }

        # Effective rank (expensive, only for key layers)
        key_roles = {"in_proj", "out_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj"}
        if compute_eff_rank and role in key_roles and weight.ndim == 2:
            t_svd = time.time()
            er = effective_rank(weight)
            sr = stable_rank(weight)
            svd_time = time.time() - t_svd
            result["eff_rank"] = round(er, 2)
            result["stable_rank"] = round(sr, 2)
            result["svd_time_s"] = round(svd_time, 3)
        elif weight.ndim == 2:
            # At least compute stable rank (cheap)
            sr = stable_rank(weight)
            result["stable_rank"] = round(sr, 2)

        layer_results.append(result)

    if not layer_results:
        return {"model": model_dir.name, "error": "no layers profiled"}

    # Aggregate by role
    role_agg = {}
    for r in layer_results:
        role = r["role"]
        if role not in role_agg:
            role_agg[role] = {"kurtosis": [], "weight_rms": [], "eff_rank": [], "stable_rank": []}
        role_agg[role]["kurtosis"].append(r["kurtosis"])
        role_agg[role]["weight_rms"].append(r["weight_rms"])
        if "eff_rank" in r:
            role_agg[role]["eff_rank"].append(r["eff_rank"])
        if "stable_rank" in r:
            role_agg[role]["stable_rank"].append(r["stable_rank"])

    role_summary = {}
    for role, vals in role_agg.items():
        role_summary[role] = {
            "count": len(vals["kurtosis"]),
            "kurtosis_mean": round(float(np.mean(vals["kurtosis"])), 4),
            "kurtosis_std": round(float(np.std(vals["kurtosis"])), 4),
            "weight_rms_mean": round(float(np.mean(vals["weight_rms"])), 6),
        }
        if vals["eff_rank"]:
            role_summary[role]["eff_rank_mean"] = round(float(np.mean(vals["eff_rank"])), 2)
        if vals["stable_rank"]:
            role_summary[role]["stable_rank_mean"] = round(float(np.mean(vals["stable_rank"])), 2)

    all_kurtosis = [r["kurtosis"] for r in layer_results]
    return {
        "model": model_dir.name,
        "n_layers_profiled": len(layer_results),
        "aggregate": {
            "kurtosis_mean": round(float(np.mean(all_kurtosis)), 4),
            "kurtosis_std": round(float(np.std(all_kurtosis)), 4),
            "kurtosis_min": round(float(np.min(all_kurtosis)), 4),
            "kurtosis_max": round(float(np.max(all_kurtosis)), 4),
        },
        "per_role": role_summary,
        "per_layer": layer_results,
    }


def main():
    all_donors = PRIMARY_DONORS + SECONDARY_DONORS
    available = []
    for name in all_donors:
        d = MODELS_DIR / name
        if d.exists():
            available.append(d)
        else:
            print(f"SKIP: {name} not found")

    print(f"Profiling {len(available)} donor candidates")

    all_results = []
    for i, model_dir in enumerate(available):
        is_primary = model_dir.name in PRIMARY_DONORS
        tag = "PRIMARY" if is_primary else "secondary"
        print(f"\n[{i+1}/{len(available)}] {tag}: {model_dir.name}", flush=True)
        t0 = time.time()
        # Full eff_rank for primary donors, stable_rank only for secondary
        result = profile_model(model_dir, compute_eff_rank=is_primary)
        elapsed = time.time() - t0
        result["survey_time_s"] = round(elapsed, 2)
        result["is_primary"] = is_primary
        all_results.append(result)
        if "error" not in result:
            print(f"  → {result['n_layers_profiled']} layers, "
                  f"kurtosis_mean={result['aggregate']['kurtosis_mean']:.4f} "
                  f"({elapsed:.1f}s)")

    # Cross-donor comparison for key roles
    comparison = {}
    for r in all_results:
        if "error" in r:
            continue
        for role, stats in r.get("per_role", {}).items():
            if role not in comparison:
                comparison[role] = []
            comparison[role].append({
                "model": r["model"],
                "is_primary": r.get("is_primary", False),
                "kurtosis_mean": stats["kurtosis_mean"],
                "weight_rms_mean": stats["weight_rms_mean"],
                "eff_rank_mean": stats.get("eff_rank_mean"),
                "stable_rank_mean": stats.get("stable_rank_mean"),
            })

    receipt = {
        "work_order": "WO-DONOR-SURVEY-01",
        "step": "2_spectral_profile",
        "question": "Which donor layers have the best spectral profile for born-compressed training?",
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "primary_donors": PRIMARY_DONORS,
        "secondary_donors": SECONDARY_DONORS,
        "n_models_profiled": len([r for r in all_results if "error" not in r]),
        "cross_donor_comparison": comparison,
        "per_model": all_results,
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"spectral_profile_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\nReceipt written: {receipt_path}")

    # Summary
    print(f"\n{'='*80}")
    print("CROSS-DONOR COMPARISON BY ROLE")
    print(f"{'='*80}")
    for role in sorted(comparison.keys()):
        entries = comparison[role]
        print(f"\n  {role}:")
        for e in sorted(entries, key=lambda x: x["kurtosis_mean"]):
            tag = " ★" if e["is_primary"] else ""
            er_str = f"eff_rank={e['eff_rank_mean']:.1f}" if e.get("eff_rank_mean") else f"stable_rank={e.get('stable_rank_mean', 'N/A')}"
            print(f"    {e['model']:<40} kurtosis={e['kurtosis_mean']:>8.4f}  {er_str}{tag}")

    return receipt_path


if __name__ == "__main__":
    main()
