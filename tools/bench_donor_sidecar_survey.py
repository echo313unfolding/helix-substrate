#!/usr/bin/env python3
"""WO-DONOR-SURVEY-01 Step 1: Sidecar Norm Survey Across All Compressed Models

Ranks every compressed model's layers by sidecar norm distribution.
Lower mean sidecar norm = better codebook fit = better donor candidate.

For each model:
  - Load HXQ compressed tensors from safetensors
  - Extract per-layer: sidecar norm mean, std, count, density
  - Codebook spread: std, range, utilization (index entropy)
  - Classify layer type: ssm_in_proj, ssm_out_proj, attn_q, attn_k, attn_v, attn_o, mlp_gate, mlp_up, mlp_down, etc.
  - Output: JSON receipt with per-layer rankings and model-level aggregates

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

# Cost tracking
t_start = time.time()
cpu_start = time.process_time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

MODELS_DIR = Path("/home/voidstr3m33/models")
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "donor_survey"
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


def shannon_entropy(indices: np.ndarray, n_bins: int = 256) -> float:
    """Shannon entropy of index distribution in bits. Max = log2(n_bins)."""
    counts = np.bincount(indices.ravel().astype(np.int32), minlength=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def classify_layer(tensor_name: str) -> dict:
    """Classify a tensor by architecture role."""
    name_lower = tensor_name.lower()
    info = {"tensor_name": tensor_name, "arch_type": "unknown", "role": "unknown", "block_idx": -1}

    # Extract block index
    parts = tensor_name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                info["block_idx"] = int(parts[i + 1])
            except ValueError:
                pass

    # Architecture type
    if "mixer." in name_lower:
        info["arch_type"] = "mamba_v1"
    elif "mamba." in name_lower:
        # Could be Mamba2 (pure) or Zamba2 (hybrid)
        info["arch_type"] = "mamba_v2"
    elif "self_attn." in name_lower:
        info["arch_type"] = "transformer"
    elif "mlp." in name_lower:
        info["arch_type"] = "transformer_mlp"
    elif "expert" in name_lower:
        info["arch_type"] = "moe"

    # Role
    if "in_proj" in name_lower:
        info["role"] = "in_proj"
    elif "out_proj" in name_lower:
        info["role"] = "out_proj"
    elif "dt_proj" in name_lower:
        info["role"] = "dt_proj"
    elif "x_proj" in name_lower:
        info["role"] = "x_proj"
    elif "q_proj" in name_lower:
        info["role"] = "q_proj"
    elif "k_proj" in name_lower:
        info["role"] = "k_proj"
    elif "v_proj" in name_lower:
        info["role"] = "v_proj"
    elif "o_proj" in name_lower:
        info["role"] = "o_proj"
    elif "gate_proj" in name_lower:
        info["role"] = "gate_proj"
    elif "up_proj" in name_lower:
        info["role"] = "up_proj"
    elif "down_proj" in name_lower:
        info["role"] = "down_proj"

    return info


def detect_model_arch(tensor_names: list) -> str:
    """Detect model architecture from tensor naming."""
    has_mixer = any("mixer." in n for n in tensor_names)
    has_mamba = any(".mamba." in n for n in tensor_names)
    has_attn = any("self_attn." in n for n in tensor_names)
    has_expert = any("expert" in n.lower() for n in tensor_names)

    if has_mixer:
        return "mamba_v1"
    elif has_mamba and has_attn:
        return "hybrid_zamba2"
    elif has_mamba:
        return "mamba_v2"
    elif has_expert:
        return "moe_transformer"
    elif has_attn:
        return "transformer"
    return "unknown"


def survey_model(model_dir: Path) -> dict:
    """Survey a single compressed model's sidecar and codebook quality."""
    from safetensors import safe_open

    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        # Check for sharded safetensors
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if not shards:
            return {"model": model_dir.name, "error": "no safetensors found"}
        safetensors_files = shards
    else:
        safetensors_files = [safetensors_path]

    # Collect all tensor keys across shards
    all_keys = []
    key_to_file = {}
    for sf_path in safetensors_files:
        sf = safe_open(str(sf_path), framework="numpy")
        for k in sf.keys():
            all_keys.append(k)
            key_to_file[k] = sf_path

    # Find compressed layers (those with .codebook suffix)
    codebook_keys = sorted([k for k in all_keys if k.endswith(".codebook")])
    base_names = [k.rsplit(".codebook", 1)[0] for k in codebook_keys]

    model_arch = detect_model_arch(all_keys)

    layer_results = []
    for base_name in base_names:
        cb_key = f"{base_name}.codebook"
        idx_key = f"{base_name}.indices"
        sp_key = f"{base_name}.sidecar_positions"
        sv_key = f"{base_name}.sidecar_values"

        # Check all required keys exist
        if not all(k in key_to_file for k in [cb_key, idx_key, sp_key, sv_key]):
            continue

        sf = safe_open(str(key_to_file[cb_key]), framework="numpy")
        codebook = sf.get_tensor(cb_key)
        indices = sf.get_tensor(idx_key)
        sidecar_pos = sf.get_tensor(sp_key)
        sidecar_val = sf.get_tensor(sv_key)

        # Layer classification
        layer_info = classify_layer(base_name)

        # Sidecar metrics
        n_sidecar = int(sidecar_pos.shape[0])
        total_weights = int(np.prod(indices.shape))
        sidecar_density = n_sidecar / total_weights if total_weights > 0 else 0.0

        sv_float = sidecar_val.astype(np.float32)
        sidecar_norm_l2 = float(np.linalg.norm(sv_float))
        sidecar_abs_mean = float(np.mean(np.abs(sv_float))) if n_sidecar > 0 else 0.0
        sidecar_abs_std = float(np.std(np.abs(sv_float))) if n_sidecar > 0 else 0.0
        sidecar_max = float(np.max(np.abs(sv_float))) if n_sidecar > 0 else 0.0

        # Codebook metrics
        cb_float = codebook.astype(np.float32)
        cb_std = float(np.std(cb_float))
        cb_range = float(np.max(cb_float) - np.min(cb_float))
        cb_mean = float(np.mean(cb_float))

        # Index entropy (codebook utilization)
        idx_entropy = shannon_entropy(indices, n_bins=int(codebook.shape[0]))
        max_entropy = math.log2(codebook.shape[0])
        codebook_utilization = idx_entropy / max_entropy if max_entropy > 0 else 0.0

        # Unique centroids actually used
        n_unique_used = int(len(np.unique(indices)))

        # Normalized sidecar norm (sidecar energy / total weight energy estimate)
        # Reconstruct weight to get total energy
        weight_reconstructed = cb_float[indices.astype(np.int32)]
        weight_norm_l2 = float(np.linalg.norm(weight_reconstructed))
        sidecar_energy_ratio = (sidecar_norm_l2 / weight_norm_l2) if weight_norm_l2 > 0 else 0.0

        layer_results.append({
            **layer_info,
            "shape": list(indices.shape),
            "n_params": total_weights,
            "n_sidecar": n_sidecar,
            "sidecar_density": round(sidecar_density, 6),
            "sidecar_norm_l2": round(sidecar_norm_l2, 4),
            "sidecar_abs_mean": round(sidecar_abs_mean, 6),
            "sidecar_abs_std": round(sidecar_abs_std, 6),
            "sidecar_max": round(sidecar_max, 6),
            "sidecar_energy_ratio": round(sidecar_energy_ratio, 6),
            "codebook_std": round(cb_std, 6),
            "codebook_range": round(cb_range, 6),
            "codebook_mean": round(cb_mean, 6),
            "index_entropy_bits": round(idx_entropy, 4),
            "codebook_utilization": round(codebook_utilization, 4),
            "n_unique_centroids": n_unique_used,
        })

    if not layer_results:
        return {"model": model_dir.name, "error": "no compressed layers found"}

    # Aggregate statistics
    sidecar_norms = [r["sidecar_abs_mean"] for r in layer_results]
    energy_ratios = [r["sidecar_energy_ratio"] for r in layer_results]
    utilizations = [r["codebook_utilization"] for r in layer_results]

    # Per-role aggregates
    role_agg = {}
    for r in layer_results:
        role = r["role"]
        if role not in role_agg:
            role_agg[role] = {"sidecar_abs_mean": [], "sidecar_energy_ratio": [], "codebook_utilization": []}
        role_agg[role]["sidecar_abs_mean"].append(r["sidecar_abs_mean"])
        role_agg[role]["sidecar_energy_ratio"].append(r["sidecar_energy_ratio"])
        role_agg[role]["codebook_utilization"].append(r["codebook_utilization"])

    role_summary = {}
    for role, vals in role_agg.items():
        role_summary[role] = {
            "count": len(vals["sidecar_abs_mean"]),
            "sidecar_abs_mean": round(float(np.mean(vals["sidecar_abs_mean"])), 6),
            "sidecar_energy_ratio_mean": round(float(np.mean(vals["sidecar_energy_ratio"])), 6),
            "codebook_utilization_mean": round(float(np.mean(vals["codebook_utilization"])), 4),
        }

    return {
        "model": model_dir.name,
        "model_arch": model_arch,
        "n_compressed_layers": len(layer_results),
        "aggregate": {
            "sidecar_abs_mean": round(float(np.mean(sidecar_norms)), 6),
            "sidecar_abs_std": round(float(np.std(sidecar_norms)), 6),
            "sidecar_energy_ratio_mean": round(float(np.mean(energy_ratios)), 6),
            "sidecar_energy_ratio_std": round(float(np.std(energy_ratios)), 6),
            "codebook_utilization_mean": round(float(np.mean(utilizations)), 4),
            "codebook_utilization_min": round(float(np.min(utilizations)), 4),
        },
        "per_role": role_summary,
        "per_layer": layer_results,
    }


def main():
    helix_dirs = sorted(MODELS_DIR.glob("*-helix"))
    print(f"Found {len(helix_dirs)} helix models")

    all_results = []
    for i, model_dir in enumerate(helix_dirs):
        print(f"[{i+1}/{len(helix_dirs)}] Surveying {model_dir.name}...", flush=True)
        t0 = time.time()
        result = survey_model(model_dir)
        elapsed = time.time() - t0
        result["survey_time_s"] = round(elapsed, 2)
        all_results.append(result)
        if "error" not in result:
            print(f"  → {result['n_compressed_layers']} layers, "
                  f"sidecar_energy_ratio={result['aggregate']['sidecar_energy_ratio_mean']:.4f}, "
                  f"utilization={result['aggregate']['codebook_utilization_mean']:.4f} "
                  f"({elapsed:.1f}s)")
        else:
            print(f"  → ERROR: {result['error']}")

    # Rank models by sidecar energy ratio (lower = better codebook fit)
    valid = [r for r in all_results if "error" not in r]
    ranked = sorted(valid, key=lambda r: r["aggregate"]["sidecar_energy_ratio_mean"])

    ranking = []
    for i, r in enumerate(ranked):
        ranking.append({
            "rank": i + 1,
            "model": r["model"],
            "arch": r["model_arch"],
            "sidecar_energy_ratio": r["aggregate"]["sidecar_energy_ratio_mean"],
            "codebook_utilization": r["aggregate"]["codebook_utilization_mean"],
            "n_layers": r["n_compressed_layers"],
        })

    # SSM-specific ranking
    ssm_models = [r for r in valid if r["model_arch"] in ("mamba_v1", "mamba_v2", "hybrid_zamba2")]
    ssm_ranked = sorted(ssm_models, key=lambda r: r["aggregate"]["sidecar_energy_ratio_mean"])

    # Attention-specific ranking (from transformer and hybrid models)
    attn_models = [r for r in valid if r["model_arch"] in ("transformer", "hybrid_zamba2", "moe_transformer")]
    attn_ranked = sorted(attn_models, key=lambda r: r["aggregate"]["sidecar_energy_ratio_mean"])

    # Build receipt
    receipt = {
        "work_order": "WO-DONOR-SURVEY-01",
        "step": "1_sidecar_norm_survey",
        "question": "Which compressed models have the best codebook fit (lowest sidecar energy ratio)?",
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "n_models_surveyed": len(helix_dirs),
        "n_models_valid": len(valid),
        "overall_ranking": ranking,
        "ssm_donor_ranking": [
            {"rank": i+1, "model": r["model"], "arch": r["model_arch"],
             "sidecar_energy_ratio": r["aggregate"]["sidecar_energy_ratio_mean"]}
            for i, r in enumerate(ssm_ranked)
        ],
        "attention_donor_ranking": [
            {"rank": i+1, "model": r["model"], "arch": r["model_arch"],
             "sidecar_energy_ratio": r["aggregate"]["sidecar_energy_ratio_mean"]}
            for i, r in enumerate(attn_ranked)
        ],
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

    # Write receipt
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"sidecar_survey_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\nReceipt written: {receipt_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"OVERALL RANKING (by sidecar energy ratio, lower = better codebook fit)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Model':<40} {'Arch':<18} {'Energy Ratio':<14} {'Util':<8} {'Layers':<6}")
    print(f"{'-'*5} {'-'*40} {'-'*18} {'-'*14} {'-'*8} {'-'*6}")
    for r in ranking:
        print(f"{r['rank']:<5} {r['model']:<40} {r['arch']:<18} {r['sidecar_energy_ratio']:<14.6f} "
              f"{r['codebook_utilization']:<8.4f} {r['n_layers']:<6}")

    print(f"\nSSM DONOR RANKING:")
    for r in receipt["ssm_donor_ranking"]:
        print(f"  #{r['rank']}: {r['model']} ({r['arch']}) — energy_ratio={r['sidecar_energy_ratio']:.6f}")

    print(f"\nATTENTION DONOR RANKING:")
    for r in receipt["attention_donor_ranking"]:
        print(f"  #{r['rank']}: {r['model']} ({r['arch']}) — energy_ratio={r['sidecar_energy_ratio']:.6f}")

    return receipt_path


if __name__ == "__main__":
    main()
