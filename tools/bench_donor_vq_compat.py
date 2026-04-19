#!/usr/bin/env python3
"""WO-DONOR-SURVEY-01 Step 3: Grouped VQ Compatibility Test on Donor Weights

For primary donors (Mamba2-1.3B, Qwen2.5-Coder-1.5B):
  - Reconstruct raw weights from HXQ compressed form
  - Re-compress at d=1, d=2, d=4 with k=256
  - Measure: cosine similarity, MSE, residual norm distribution
  - Predict whether grouped VQ will help or hurt during born-compressed training

Decision gate: cosine > 0.998 at d=2 means grouped VQ is safe for training.

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

# Only test primary donors — the ones we'll actually use
DONORS = [
    "mamba2-1.3b-helix",
    "qwen2.5-coder-1.5b-helix",
]

# Key roles to test (most impactful for training)
KEY_ROLES = {"in_proj", "out_proj", "q_proj", "v_proj", "gate_proj", "down_proj"}

# Test dimensions
VQ_DIMS = [1, 2, 4]
N_CLUSTERS = 256
MAX_ITERS = 20


def vector_kmeans(data: np.ndarray, n_clusters: int, vector_dim: int, max_iters: int = 20) -> tuple:
    """K-means with vector_dim grouping. Returns (centroids, indices, codebook)."""
    original_shape = data.shape
    flat = data.ravel().astype(np.float32)

    # Pad if needed
    remainder = flat.size % vector_dim
    if remainder > 0:
        flat = np.concatenate([flat, np.zeros(vector_dim - remainder, dtype=np.float32)])

    # Reshape to vectors
    vectors = flat.reshape(-1, vector_dim)
    n_vectors = vectors.shape[0]

    # K-means++ init
    indices_init = [np.random.randint(n_vectors)]
    for _ in range(n_clusters - 1):
        dists = np.min(
            np.sum((vectors[:, None] - vectors[indices_init][None, :]) ** 2, axis=2),
            axis=1,
        )
        probs = dists / (dists.sum() + 1e-10)
        indices_init.append(np.random.choice(n_vectors, p=probs))

    centroids = vectors[indices_init].copy()

    # Lloyd's iterations
    for iteration in range(max_iters):
        # Assignment (chunked for memory)
        chunk_size = 100000
        assignments = np.empty(n_vectors, dtype=np.int32)
        for start in range(0, n_vectors, chunk_size):
            end = min(start + chunk_size, n_vectors)
            chunk = vectors[start:end]
            dists = np.sum((chunk[:, None] - centroids[None, :]) ** 2, axis=2)
            assignments[start:end] = np.argmin(dists, axis=1)

        # Update
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(n_clusters, dtype=np.int32)
        for c in range(n_clusters):
            mask = assignments == c
            if np.any(mask):
                new_centroids[c] = vectors[mask].mean(axis=0)
                counts[c] = mask.sum()
            else:
                new_centroids[c] = centroids[c]

        # Convergence check
        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if shift < 1e-5:
            break

    # Final assignment
    chunk_size = 100000
    assignments = np.empty(n_vectors, dtype=np.int32)
    for start in range(0, n_vectors, chunk_size):
        end = min(start + chunk_size, n_vectors)
        chunk = vectors[start:end]
        dists = np.sum((chunk[:, None] - centroids[None, :]) ** 2, axis=2)
        assignments[start:end] = np.argmin(dists, axis=1)

    # Reconstruct
    reconstructed_vectors = centroids[assignments]
    reconstructed = reconstructed_vectors.ravel()[:np.prod(original_shape)]
    reconstructed = reconstructed.reshape(original_shape)

    n_used = len(np.unique(assignments))
    return reconstructed, assignments, centroids, n_used


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute reconstruction quality metrics."""
    orig_flat = original.ravel().astype(np.float64)
    recon_flat = reconstructed.ravel().astype(np.float64)

    # Cosine similarity
    dot = np.dot(orig_flat, recon_flat)
    norm_orig = np.linalg.norm(orig_flat)
    norm_recon = np.linalg.norm(recon_flat)
    cosine = dot / (norm_orig * norm_recon + 1e-10)

    # MSE
    mse = float(np.mean((orig_flat - recon_flat) ** 2))

    # Relative error
    rel_error = np.linalg.norm(orig_flat - recon_flat) / (norm_orig + 1e-10)

    # Residual norm distribution
    residuals = np.abs(orig_flat - recon_flat)

    return {
        "cosine": round(float(cosine), 6),
        "mse": round(mse, 8),
        "relative_error": round(float(rel_error), 6),
        "residual_mean": round(float(np.mean(residuals)), 6),
        "residual_std": round(float(np.std(residuals)), 6),
        "residual_max": round(float(np.max(residuals)), 6),
        "residual_p99": round(float(np.percentile(residuals, 99)), 6),
    }


def classify_role(name: str) -> str:
    for role in ["in_proj", "out_proj", "dt_proj", "x_proj",
                 "q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]:
        if role in name:
            return role
    return "other"


def test_model(model_dir: Path) -> dict:
    """Test VQ compatibility at d=1, d=2, d=4 for a model's key layers."""
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

    # Filter to key roles only
    target_layers = [(bn, classify_role(bn)) for bn in base_names if classify_role(bn) in KEY_ROLES]

    # Sample: first and last block of each role type, plus middle
    role_indices = {}
    for bn, role in target_layers:
        if role not in role_indices:
            role_indices[role] = []
        role_indices[role].append(bn)

    sampled = []
    for role, names in role_indices.items():
        if len(names) <= 3:
            sampled.extend((n, role) for n in names)
        else:
            # First, middle, last
            sampled.append((names[0], role))
            sampled.append((names[len(names) // 2], role))
            sampled.append((names[-1], role))

    print(f"  Testing {len(sampled)} layers across {len(role_indices)} roles")

    layer_results = []
    for base_name, role in sampled:
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

        # Reconstruct original weight
        original = codebook[indices.astype(np.int32)]
        if sidecar_pos.size > 0:
            flat = original.ravel()
            valid = sidecar_pos < flat.size
            flat[sidecar_pos[valid].astype(np.int64)] = sidecar_val[valid]
            original = flat.reshape(original.shape)

        # Test each VQ dimension
        dim_results = {}
        for vq_dim in VQ_DIMS:
            np.random.seed(42)  # Reproducibility
            t0 = time.time()
            reconstructed, assignments, centroids, n_used = vector_kmeans(
                original, N_CLUSTERS, vq_dim, MAX_ITERS
            )
            compress_time = time.time() - t0

            metrics = compute_metrics(original, reconstructed)
            metrics["vq_dim"] = vq_dim
            metrics["n_used_centroids"] = n_used
            metrics["compress_time_s"] = round(compress_time, 3)
            dim_results[f"d={vq_dim}"] = metrics

        # Verdict: does d=2 maintain > 0.998 cosine?
        d2_cos = dim_results.get("d=2", {}).get("cosine", 0.0)
        d1_cos = dim_results.get("d=1", {}).get("cosine", 0.0)
        d4_cos = dim_results.get("d=4", {}).get("cosine", 0.0)

        layer_results.append({
            "tensor_name": base_name,
            "role": role,
            "shape": list(original.shape),
            "n_params": int(np.prod(original.shape)),
            "vq_results": dim_results,
            "d2_safe": d2_cos > 0.998,
            "best_dim": min(dim_results.keys(), key=lambda d: dim_results[d]["mse"]),
            "cosine_progression": {
                "d1": d1_cos,
                "d2": d2_cos,
                "d4": d4_cos,
            },
        })

        print(f"    {base_name}: d1={d1_cos:.4f} d2={d2_cos:.4f} d4={d4_cos:.4f} "
              f"{'✓' if d2_cos > 0.998 else '✗'}")

    if not layer_results:
        return {"model": model_dir.name, "error": "no layers tested"}

    # Aggregate by role
    role_verdicts = {}
    for r in layer_results:
        role = r["role"]
        if role not in role_verdicts:
            role_verdicts[role] = {"d1_cos": [], "d2_cos": [], "d4_cos": [], "d2_safe_count": 0, "total": 0}
        role_verdicts[role]["d1_cos"].append(r["cosine_progression"]["d1"])
        role_verdicts[role]["d2_cos"].append(r["cosine_progression"]["d2"])
        role_verdicts[role]["d4_cos"].append(r["cosine_progression"]["d4"])
        role_verdicts[role]["total"] += 1
        if r["d2_safe"]:
            role_verdicts[role]["d2_safe_count"] += 1

    role_summary = {}
    for role, v in role_verdicts.items():
        role_summary[role] = {
            "n_tested": v["total"],
            "d2_safe_pct": round(v["d2_safe_count"] / v["total"] * 100, 1),
            "d1_cos_mean": round(float(np.mean(v["d1_cos"])), 4),
            "d2_cos_mean": round(float(np.mean(v["d2_cos"])), 4),
            "d4_cos_mean": round(float(np.mean(v["d4_cos"])), 4),
        }

    n_d2_safe = sum(1 for r in layer_results if r["d2_safe"])
    verdict = "PASS" if n_d2_safe / len(layer_results) > 0.8 else "FAIL"

    return {
        "model": model_dir.name,
        "n_layers_tested": len(layer_results),
        "n_d2_safe": n_d2_safe,
        "d2_safe_pct": round(n_d2_safe / len(layer_results) * 100, 1),
        "verdict": verdict,
        "per_role": role_summary,
        "per_layer": layer_results,
    }


def main():
    available = []
    for name in DONORS:
        d = MODELS_DIR / name
        if d.exists():
            available.append(d)
        else:
            print(f"SKIP: {name} not found")

    print(f"Testing VQ compatibility for {len(available)} donors at d={VQ_DIMS}")

    all_results = []
    for i, model_dir in enumerate(available):
        print(f"\n[{i+1}/{len(available)}] {model_dir.name}", flush=True)
        t0 = time.time()
        result = test_model(model_dir)
        elapsed = time.time() - t0
        result["test_time_s"] = round(elapsed, 2)
        all_results.append(result)
        if "error" not in result:
            print(f"  → {result['verdict']}: {result['d2_safe_pct']}% of layers safe at d=2 ({elapsed:.1f}s)")

    # Recommended VQ dim per model per role
    recommendations = {}
    for r in all_results:
        if "error" in r:
            continue
        model = r["model"]
        recommendations[model] = {}
        for role, stats in r.get("per_role", {}).items():
            if stats["d2_cos_mean"] > 0.998:
                rec = "d=2"
            elif stats["d4_cos_mean"] > 0.998:
                rec = "d=4"
            else:
                rec = "d=1"
            recommendations[model][role] = {
                "recommended_dim": rec,
                "d1_cos": stats["d1_cos_mean"],
                "d2_cos": stats["d2_cos_mean"],
                "d4_cos": stats["d4_cos_mean"],
            }

    receipt = {
        "work_order": "WO-DONOR-SURVEY-01",
        "step": "3_vq_compatibility",
        "question": "Is grouped VQ safe for born-compressed training with donor weights?",
        "gate": "d=2 cosine > 0.998 for > 80% of key layers",
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "donors_tested": DONORS,
        "vq_dims_tested": VQ_DIMS,
        "n_clusters": N_CLUSTERS,
        "recommendations": recommendations,
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
    receipt_path = RECEIPT_DIR / f"vq_compat_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\nReceipt written: {receipt_path}")

    # Summary
    print(f"\n{'='*80}")
    print("VQ DIMENSION RECOMMENDATIONS")
    print(f"{'='*80}")
    for model, roles in recommendations.items():
        print(f"\n  {model}:")
        for role, rec in sorted(roles.items()):
            print(f"    {role:<15} → {rec['recommended_dim']:<5} "
                  f"(d1={rec['d1_cos']:.4f} d2={rec['d2_cos']:.4f} d4={rec['d4_cos']:.4f})")

    return receipt_path


if __name__ == "__main__":
    main()
