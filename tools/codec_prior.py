#!/usr/bin/env python3
"""
Codec Structural Prior: Can the compression codec become a meta-model?

This experiment mines codebook patterns across all compressed models on the box
and tests whether the codec has learned transferable structural knowledge about
neural network architectures.

Three experiments:
  1. FINGERPRINT — Extract structural signatures from codebooks. Can we identify
     architecture family (Transformer/SSM/Hybrid) from codebooks alone?
  2. TRANSFER — Take codebooks from Model A, apply to Model B's weights.
     If the codec captures structure, cross-model codebooks should partially work.
  3. PRIOR — Given only (architecture, layer_type, shape), predict the codebook
     distribution. If this works, the codec becomes a generative structural prior.

This is Stage 3 of: compression → execution → structural prior → generative.

Usage:
  python3 tools/codec_prior.py                    # Run all experiments
  python3 tools/codec_prior.py --fingerprint      # Experiment 1 only
  python3 tools/codec_prior.py --transfer         # Experiment 2 only
  python3 tools/codec_prior.py --prior            # Experiment 3 only
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors required. pip install safetensors", file=sys.stderr)
    sys.exit(1)

try:
    import resource as _resource
    def _peak_memory_mb():
        return round(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
except ImportError:
    def _peak_memory_mb():
        return 0.0

# ──────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("/home/voidstr3m33/models")

# Architecture families
ARCH_FAMILY = {
    "tinyllama-1.1b-helix": "transformer",
    "qwen2.5-coder-1.5b-helix": "transformer",
    "qwen2.5-coder-1.5b-instruct-helix": "transformer",
    "qwen2.5-coder-3b-helix": "transformer",
    "qwen2.5-3b-instruct-helix": "transformer",
    "qwen2.5-7b-helix": "transformer",
    "qwen2.5-7b-instruct-helix": "transformer",
    "qwen2.5-14b-helix": "transformer",
    "mamba-130m-helix": "ssm",
    "mamba2-1.3b-helix": "ssm",
    "zamba2-1.2b-helix": "hybrid",
    "zamba2-2.7b-instruct-helix": "hybrid",
    "zamba2-7b-instruct-helix": "hybrid",
}


@dataclass
class CodebookRecord:
    """Structural features extracted from a single codebook."""
    model_name: str
    arch_family: str       # transformer / ssm / hybrid
    tensor_name: str
    layer_type: str        # attention_qk / attention_vo / ffn / mixer / lm_head / unknown
    layer_idx: int         # -1 for non-layer tensors
    shape: tuple           # shape of the weight tensor (from indices)
    n_centroids: int       # codebook size (k)
    # Distribution features of the 256 centroids
    cb_mean: float
    cb_std: float
    cb_range: float
    cb_skew: float
    cb_kurtosis: float
    cb_min: float
    cb_max: float
    # Percentile features
    cb_q25: float
    cb_q75: float
    cb_iqr: float
    # Concentration features
    cb_entropy: float      # How uniformly spread are centroids?
    cb_gap_ratio: float    # max_gap / median_gap between sorted centroids


def classify_layer_type(tensor_name: str) -> str:
    """Classify layer type from tensor name."""
    name = tensor_name.lower()
    if any(p in name for p in ['q_proj', 'k_proj', 'attn_q', 'attn_k']):
        return 'attention_qk'
    if any(p in name for p in ['v_proj', 'o_proj', 'attn_v', 'attn_output']):
        return 'attention_vo'
    if any(p in name for p in ['mlp', 'ffn', 'gate_proj', 'up_proj', 'down_proj',
                                'feed_forward', 'fc1', 'fc2']):
        return 'ffn'
    if any(p in name for p in ['mixer', 'in_proj', 'out_proj', 'x_proj', 'dt_proj',
                                'mamba', 'b_proj', 'c_proj']):
        return 'mixer'
    if any(p in name for p in ['lm_head', 'output.codebook']):
        return 'lm_head'
    return 'unknown'


def extract_layer_idx(tensor_name: str) -> int:
    """Extract layer index from tensor name."""
    import re
    m = re.search(r'layers?\.(\d+)', tensor_name)
    if m:
        return int(m.group(1))
    m = re.search(r'blk\.(\d+)', tensor_name)
    if m:
        return int(m.group(1))
    return -1


def compute_entropy(values: np.ndarray) -> float:
    """Compute entropy of the centroid distribution (binned)."""
    # Bin the centroids into 32 bins and compute entropy
    hist, _ = np.histogram(values, bins=32, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log2(hist + 1e-12)))


def compute_gap_ratio(values: np.ndarray) -> float:
    """Ratio of largest gap to median gap in sorted centroids."""
    sorted_vals = np.sort(values)
    gaps = np.diff(sorted_vals)
    if len(gaps) == 0 or np.median(gaps) == 0:
        return 0.0
    return float(np.max(gaps) / np.median(gaps))


def extract_codebook_record(
    model_name: str,
    arch_family: str,
    cb_key: str,
    sf,
) -> CodebookRecord:
    """Extract structural features from a single codebook in safetensors."""
    cb = sf.get_tensor(cb_key).ravel().astype(np.float64)
    n_centroids = len(cb)

    # Get weight shape from indices
    idx_key = cb_key.replace('.codebook', '.indices')
    try:
        idx = sf.get_tensor(idx_key)
        weight_shape = tuple(idx.shape)
    except Exception:
        weight_shape = (0, 0)

    tensor_name = cb_key.replace('.codebook', '')
    layer_type = classify_layer_type(tensor_name)
    layer_idx = extract_layer_idx(tensor_name)

    # Distribution features
    cb_mean = float(np.mean(cb))
    cb_std = float(np.std(cb))
    cb_range = float(np.max(cb) - np.min(cb))
    # Skewness
    if cb_std > 0:
        cb_skew = float(np.mean(((cb - cb_mean) / cb_std) ** 3))
        cb_kurtosis = float(np.mean(((cb - cb_mean) / cb_std) ** 4) - 3.0)
    else:
        cb_skew = 0.0
        cb_kurtosis = 0.0

    q25, q75 = np.percentile(cb, [25, 75])

    return CodebookRecord(
        model_name=model_name,
        arch_family=arch_family,
        tensor_name=tensor_name,
        layer_type=layer_type,
        layer_idx=layer_idx,
        shape=weight_shape,
        n_centroids=n_centroids,
        cb_mean=cb_mean,
        cb_std=cb_std,
        cb_range=cb_range,
        cb_skew=cb_skew,
        cb_kurtosis=cb_kurtosis,
        cb_min=float(np.min(cb)),
        cb_max=float(np.max(cb)),
        cb_q25=float(q25),
        cb_q75=float(q75),
        cb_iqr=float(q75 - q25),
        cb_entropy=compute_entropy(cb),
        cb_gap_ratio=compute_gap_ratio(cb),
    )


def collect_all_codebooks() -> list[CodebookRecord]:
    """Scan all compressed models and extract codebook records."""
    records = []
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.name.endswith('-helix'):
            continue
        if model_dir.name not in ARCH_FAMILY:
            continue

        sf_path = model_dir / 'model.safetensors'
        if not sf_path.exists():
            # Check for sharded model
            index_path = model_dir / 'model.safetensors.index.json'
            if index_path.exists():
                print(f"  {model_dir.name}: sharded (skipping for now)", file=sys.stderr)
                continue
            continue

        arch = ARCH_FAMILY[model_dir.name]
        print(f"  {model_dir.name} ({arch})...", file=sys.stderr, end=" ", flush=True)

        try:
            sf = safe_open(str(sf_path), framework='numpy')
            cb_keys = [k for k in sf.keys() if k.endswith('.codebook')]
            for k in cb_keys:
                rec = extract_codebook_record(model_dir.name, arch, k, sf)
                records.append(rec)
            print(f"{len(cb_keys)} codebooks", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)

    return records


# ──────────────────────────────────────────────────────────────────────
# Experiment 1: FINGERPRINT
# Can we identify architecture family from codebook statistics alone?
# ──────────────────────────────────────────────────────────────────────

def experiment_fingerprint(records: list[CodebookRecord]) -> dict:
    """Test whether codebook distributions are architecture-specific."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Architecture Fingerprinting from Codebooks")
    print("=" * 70)

    # Group by architecture family
    by_arch = {}
    for r in records:
        by_arch.setdefault(r.arch_family, []).append(r)

    print(f"\n  Records: {len(records)} codebooks from {len(by_arch)} architecture families")

    # Compute per-architecture aggregate stats
    arch_profiles = {}
    for arch, recs in sorted(by_arch.items()):
        stds = [r.cb_std for r in recs]
        ranges = [r.cb_range for r in recs]
        kurts = [r.cb_kurtosis for r in recs]
        skews = [r.cb_skew for r in recs]
        entropies = [r.cb_entropy for r in recs]
        gap_ratios = [r.cb_gap_ratio for r in recs]

        profile = {
            "n_codebooks": len(recs),
            "std_mean": round(np.mean(stds), 6),
            "std_std": round(np.std(stds), 6),
            "range_mean": round(np.mean(ranges), 4),
            "range_std": round(np.std(ranges), 4),
            "kurtosis_mean": round(np.mean(kurts), 4),
            "kurtosis_std": round(np.std(kurts), 4),
            "skew_mean": round(np.mean(skews), 4),
            "entropy_mean": round(np.mean(entropies), 4),
            "gap_ratio_mean": round(np.mean(gap_ratios), 4),
        }
        arch_profiles[arch] = profile

        print(f"\n  {arch.upper()} ({len(recs)} codebooks):")
        print(f"    Centroid std:     {profile['std_mean']:.6f} +/- {profile['std_std']:.6f}")
        print(f"    Centroid range:   {profile['range_mean']:.4f} +/- {profile['range_std']:.4f}")
        print(f"    Centroid kurtosis:{profile['kurtosis_mean']:8.4f} +/- {profile['kurtosis_std']:.4f}")
        print(f"    Centroid skew:    {profile['skew_mean']:8.4f}")
        print(f"    Entropy:          {profile['entropy_mean']:.4f}")
        print(f"    Gap ratio:        {profile['gap_ratio_mean']:.4f}")

    # Classification test: leave-one-out using centroid std as a single feature
    # Simple nearest-centroid classifier on the feature vector
    print("\n  --- Classification Test ---")
    print("  Can we identify architecture from a single codebook?")

    # Build feature vectors: [std, range, kurtosis, skew, entropy, gap_ratio]
    features = []
    labels = []
    for r in records:
        features.append([r.cb_std, r.cb_range, r.cb_kurtosis, r.cb_skew,
                         r.cb_entropy, r.cb_gap_ratio])
        labels.append(r.arch_family)

    features = np.array(features)
    labels = np.array(labels)

    # Normalize features
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    features_norm = (features - feat_mean) / feat_std

    # Compute per-class centroids
    unique_labels = sorted(set(labels))
    class_centroids = {}
    for lbl in unique_labels:
        mask = labels == lbl
        class_centroids[lbl] = features_norm[mask].mean(axis=0)

    # Leave-one-out nearest centroid
    correct = 0
    confusion = {a: {b: 0 for b in unique_labels} for a in unique_labels}
    for i in range(len(records)):
        # Exclude this sample from class centroid
        true_label = labels[i]
        dists = {}
        for lbl in unique_labels:
            mask = (labels == lbl)
            mask[i] = False
            if mask.sum() == 0:
                dists[lbl] = float('inf')
                continue
            centroid = features_norm[mask].mean(axis=0)
            dists[lbl] = np.linalg.norm(features_norm[i] - centroid)

        pred_label = min(dists, key=dists.get)
        if pred_label == true_label:
            correct += 1
        confusion[true_label][pred_label] += 1

    accuracy = correct / len(records)
    print(f"\n  Leave-one-out accuracy: {correct}/{len(records)} = {accuracy:.1%}")
    print(f"\n  Confusion matrix:")
    header = f"  {'':>12s}"
    for lbl in unique_labels:
        header += f" {lbl:>12s}"
    print(header)
    for true_lbl in unique_labels:
        row = f"  {true_lbl:>12s}"
        for pred_lbl in unique_labels:
            row += f" {confusion[true_lbl][pred_lbl]:12d}"
        print(row)

    # Per-layer-type analysis
    print(f"\n  --- Per Layer Type ---")
    by_type = {}
    for r in records:
        by_type.setdefault(r.layer_type, []).append(r)

    for lt, recs in sorted(by_type.items()):
        by_arch_lt = {}
        for r in recs:
            by_arch_lt.setdefault(r.arch_family, []).append(r.cb_std)
        parts = []
        for arch in sorted(by_arch_lt):
            vals = by_arch_lt[arch]
            parts.append(f"{arch}={np.mean(vals):.5f}")
        print(f"  {lt:>14s} (n={len(recs):3d}):  std by arch: {', '.join(parts)}")

    return {
        "n_records": len(records),
        "arch_profiles": arch_profiles,
        "classification_accuracy": round(accuracy, 4),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


# ──────────────────────────────────────────────────────────────────────
# Experiment 2: TRANSFER
# Take codebooks from Model A, use them on Model B's weights.
# If codec captures structure, cross-model assignment should partially work.
# ──────────────────────────────────────────────────────────────────────

def experiment_transfer(records: list[CodebookRecord]) -> dict:
    """Test codebook transferability across models."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Cross-Model Codebook Transfer")
    print("=" * 70)

    # Pick pairs: same layer type, different models
    # We need access to the actual weight tensors from the DENSE models
    # and codebooks from the HELIX models

    # Available dense models
    dense_models = {}
    for d in sorted(MODELS_DIR.iterdir()):
        if d.name.endswith('-helix') or d.name == 'model':
            continue
        sf_path = d / 'model.safetensors'
        if sf_path.exists():
            dense_models[d.name] = sf_path

    if not dense_models:
        print("  No dense models available for transfer test.", file=sys.stderr)
        return {"error": "no dense models"}

    print(f"\n  Dense models available: {list(dense_models.keys())}")

    # Strategy: for each dense model, find a helix model of DIFFERENT architecture,
    # take its codebook for a matching layer type, and measure assignment quality
    results = []

    # Use TinyLlama dense + Mamba helix codebooks (transformer←SSM transfer)
    # and Mamba dense + TinyLlama helix codebooks (SSM←transformer transfer)
    transfer_pairs = [
        # Cross-architecture transfers
        ("mamba-130m-hf-dense", "tinyllama-1.1b-helix", "ssm<-transformer"),
        ("tinyllama-dense", "mamba-130m-helix", "transformer<-ssm"),
        ("tinyllama-dense", "zamba2-1.2b-helix", "transformer<-hybrid"),
        ("zamba2-1.2b", "tinyllama-1.1b-helix", "hybrid<-transformer"),
        ("zamba2-1.2b", "mamba-130m-helix", "hybrid<-ssm"),
        # Same-architecture transfers (should work better if codec captures structure)
        ("tinyllama-dense", "qwen2.5-coder-3b-helix", "transformer<-transformer"),
        ("tinyllama-dense", "qwen2.5-3b-instruct-helix", "transformer<-transformer2"),
        ("mamba-130m-hf-dense", "mamba2-1.3b-helix", "ssm<-ssm"),
    ]

    for dense_name, helix_name, desc in transfer_pairs:
        dense_path = dense_models.get(dense_name)
        helix_path = MODELS_DIR / helix_name / 'model.safetensors'

        if dense_path is None or not helix_path.exists():
            continue

        print(f"\n  Transfer: {desc}")
        print(f"    Dense weights from: {dense_name}")
        print(f"    Codebook from:      {helix_name}")

        try:
            sf_dense = safe_open(str(dense_path), framework='numpy')
            sf_helix = safe_open(str(helix_path), framework='numpy')

            helix_cb_keys = [k for k in sf_helix.keys() if k.endswith('.codebook')]
            dense_weight_keys = [k for k in sf_dense.keys() if k.endswith('.weight')]

            # Find matching FFN layers (most comparable across architectures)
            # Take first FFN-like weight from dense, first FFN codebook from helix
            dense_ffn = None
            for k in dense_weight_keys:
                if any(p in k for p in ['mlp.down_proj', 'mlp.gate_proj', 'ffn_down',
                                         'mixer.in_proj', 'mixer.out_proj',
                                         'feed_forward', 'mamba.in_proj']):
                    try:
                        w = sf_dense.get_tensor(k).astype(np.float32)
                    except (TypeError, ValueError):
                        # bfloat16 not understood by numpy — load single tensor via torch
                        try:
                            import torch
                            sf_torch = safe_open(str(dense_path), framework='pt')
                            w = sf_torch.get_tensor(k).float().numpy()
                            del sf_torch
                        except Exception:
                            continue
                    if w.ndim == 2:
                        dense_ffn = (k, w)
                        break

            helix_ffn_cb = None
            for k in helix_cb_keys:
                if any(p in k for p in ['mlp.down_proj', 'mlp.gate_proj', 'ffn_down',
                                         'mixer.in_proj', 'mixer.out_proj']):
                    cb = sf_helix.get_tensor(k).astype(np.float32).ravel()
                    helix_ffn_cb = (k, cb)
                    break

            if dense_ffn is None or helix_ffn_cb is None:
                print("    Could not find matching FFN layers")
                continue

            w_name, w = dense_ffn
            cb_name, cb = helix_ffn_cb

            print(f"    Dense tensor: {w_name} shape={w.shape}")
            print(f"    Foreign codebook: {cb_name} k={len(cb)}")

            # Self-codebook: compress this weight with its OWN best codebook
            from helix_substrate.cdna_encoder import _simple_kmeans
            own_cb, own_indices = _simple_kmeans(
                w.ravel(), n_clusters=256, max_iters=20
            )
            own_cb = own_cb.ravel()
            own_recon = own_cb[own_indices].reshape(w.shape)
            own_cos = np.dot(w.ravel(), own_recon.ravel()) / (
                np.linalg.norm(w.ravel()) * np.linalg.norm(own_recon.ravel()) + 1e-12)

            # Foreign codebook: assign weights to nearest centroid from other model
            flat = w.ravel()
            # Chunked nearest-centroid assignment (avoid OOM on large tensors)
            CHUNK = 500_000
            foreign_indices = np.empty(len(flat), dtype=np.uint8)
            for ci in range(0, len(flat), CHUNK):
                chunk = flat[ci:ci + CHUNK]
                dists = np.abs(chunk[:, None] - cb[None, :])
                foreign_indices[ci:ci + CHUNK] = np.argmin(dists, axis=1).astype(np.uint8)
                del dists
            foreign_recon = cb[foreign_indices]
            foreign_cos = np.dot(flat, foreign_recon) / (
                np.linalg.norm(flat) * np.linalg.norm(foreign_recon) + 1e-12)
            del foreign_recon

            # Random codebook: assign to random centroids (baseline)
            rng = np.random.RandomState(42)
            random_cb = rng.uniform(cb.min(), cb.max(), size=256).astype(np.float32)
            random_indices = np.empty(len(flat), dtype=np.uint8)
            for ci in range(0, len(flat), CHUNK):
                chunk = flat[ci:ci + CHUNK]
                dists_r = np.abs(chunk[:, None] - random_cb[None, :])
                random_indices[ci:ci + CHUNK] = np.argmin(dists_r, axis=1).astype(np.uint8)
                del dists_r
            random_recon = random_cb[random_indices]
            random_cos = np.dot(flat, random_recon) / (
                np.linalg.norm(flat) * np.linalg.norm(random_recon) + 1e-12)
            del random_recon

            print(f"\n    Cosine fidelity:")
            print(f"      Own codebook (optimal):     {own_cos:.6f}")
            print(f"      Foreign codebook (transfer): {foreign_cos:.6f}")
            print(f"      Random codebook (baseline):  {random_cos:.6f}")
            print(f"      Transfer retention:           {foreign_cos/own_cos:.4f} of optimal")
            print(f"      Above random by:              {(foreign_cos - random_cos):.6f}")

            results.append({
                "transfer": desc,
                "dense_tensor": w_name,
                "dense_shape": list(w.shape),
                "foreign_codebook": cb_name,
                "cosine_own": round(float(own_cos), 6),
                "cosine_foreign": round(float(foreign_cos), 6),
                "cosine_random": round(float(random_cos), 6),
                "retention": round(float(foreign_cos / own_cos), 4),
            })

            del w

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    return {"transfers": results}


# ──────────────────────────────────────────────────────────────────────
# Experiment 3: PRIOR
# Given (architecture, layer_type, layer_depth, shape), can we predict
# the codebook distribution? If yes, the codec is a generative prior.
# ──────────────────────────────────────────────────────────────────────

def experiment_prior(records: list[CodebookRecord]) -> dict:
    """Test whether codebook distributions are predictable from metadata."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Structural Prior — Codebook Prediction")
    print("=" * 70)

    # Leave-one-model-out: predict all codebooks of a held-out model
    # using the mean codebook statistics from the same (arch, layer_type) group
    # in all OTHER models.

    models = sorted(set(r.model_name for r in records))
    print(f"\n  Models: {len(models)}")
    print(f"  Strategy: leave-one-model-out prediction of codebook std and range")

    all_results = []

    for held_out in models:
        train = [r for r in records if r.model_name != held_out]
        test = [r for r in records if r.model_name == held_out]

        if not test:
            continue

        held_arch = test[0].arch_family

        # Build prediction model: mean codebook stats per (arch_family, layer_type)
        prior = {}
        for r in train:
            key = (r.arch_family, r.layer_type)
            prior.setdefault(key, []).append(r)

        # Fallback: just layer_type (ignore arch)
        fallback = {}
        for r in train:
            fallback.setdefault(r.layer_type, []).append(r)

        errors_std = []
        errors_range = []
        for r in test:
            key = (r.arch_family, r.layer_type)
            if key in prior and len(prior[key]) >= 2:
                pred_std = np.mean([p.cb_std for p in prior[key]])
                pred_range = np.mean([p.cb_range for p in prior[key]])
            elif r.layer_type in fallback:
                pred_std = np.mean([p.cb_std for p in fallback[r.layer_type]])
                pred_range = np.mean([p.cb_range for p in fallback[r.layer_type]])
            else:
                continue

            errors_std.append(abs(r.cb_std - pred_std) / (r.cb_std + 1e-12))
            errors_range.append(abs(r.cb_range - pred_range) / (r.cb_range + 1e-12))

        if errors_std:
            mape_std = np.mean(errors_std) * 100
            mape_range = np.mean(errors_range) * 100
            all_results.append({
                "held_out": held_out,
                "arch": held_arch,
                "n_test": len(test),
                "mape_std_pct": round(mape_std, 2),
                "mape_range_pct": round(mape_range, 2),
            })
            status = "GOOD" if mape_std < 30 else "WEAK" if mape_std < 60 else "FAIL"
            print(f"  {held_out:>35s} ({held_arch:>11s}): "
                  f"std MAPE={mape_std:5.1f}%  range MAPE={mape_range:5.1f}%  [{status}]")

    # Can we generate a synthetic codebook that's usable?
    print(f"\n  --- Synthetic Codebook Generation ---")
    print(f"  Can the prior generate a codebook that works on unseen weights?")

    # Take a real weight tensor, generate a predicted codebook from the prior,
    # and measure how well it compresses vs the real codebook
    dense_path = MODELS_DIR / "tinyllama-1.1b-chat-v1.0" / "model.safetensors"
    helix_path = MODELS_DIR / "tinyllama-1.1b-helix" / "model.safetensors"

    if dense_path.exists() and helix_path.exists():
        sf_dense = safe_open(str(dense_path), framework='numpy')
        sf_helix = safe_open(str(helix_path), framework='numpy')

        # Find an FFN weight
        target_key = None
        for k in sf_dense.keys():
            if 'layers.10.mlp.down_proj.weight' in k:
                target_key = k
                break

        if target_key:
            w = sf_dense.get_tensor(target_key).astype(np.float32)
            real_cb = sf_helix.get_tensor(
                target_key.replace('.weight', '') + '.codebook'
            ).astype(np.float32).ravel()

            # Generate synthetic codebook from prior statistics
            # Use mean std and range of FFN codebooks from OTHER transformer models
            ffn_transformer = [r for r in records
                               if r.arch_family == 'transformer'
                               and r.layer_type == 'ffn'
                               and r.model_name != 'tinyllama-1.1b-helix']

            if ffn_transformer:
                pred_std = np.mean([r.cb_std for r in ffn_transformer])
                pred_mean = np.mean([r.cb_mean for r in ffn_transformer])
                pred_kurtosis = np.mean([r.cb_kurtosis for r in ffn_transformer])

                # Generate centroids: uniform spacing over predicted range
                # (simplest possible synthetic codebook)
                pred_range_low = pred_mean - 3 * pred_std
                pred_range_high = pred_mean + 3 * pred_std
                synthetic_cb = np.linspace(pred_range_low, pred_range_high, 256).astype(np.float32)

                # Measure cosine: real codebook vs synthetic codebook
                flat = w.ravel()

                # Chunked assignment (avoid OOM)
                CHUNK = 500_000
                def chunked_assign(data, codebook):
                    indices = np.empty(len(data), dtype=np.uint8)
                    for ci in range(0, len(data), CHUNK):
                        chunk = data[ci:ci + CHUNK]
                        d = np.abs(chunk[:, None] - codebook[None, :])
                        indices[ci:ci + CHUNK] = np.argmin(d, axis=1).astype(np.uint8)
                        del d
                    return indices

                # Real codebook assignment
                real_idx = chunked_assign(flat, real_cb)
                real_recon = real_cb[real_idx]
                cos_real = np.dot(flat, real_recon) / (
                    np.linalg.norm(flat) * np.linalg.norm(real_recon) + 1e-12)
                del real_recon

                # Synthetic codebook assignment
                syn_idx = chunked_assign(flat, synthetic_cb)
                syn_recon = synthetic_cb[syn_idx]
                cos_syn = np.dot(flat, syn_recon) / (
                    np.linalg.norm(flat) * np.linalg.norm(syn_recon) + 1e-12)
                del syn_recon

                print(f"\n  Target: {target_key} {w.shape}")
                print(f"  Prior source: {len(ffn_transformer)} FFN codebooks from other transformers")
                print(f"  Predicted std={pred_std:.5f}  mean={pred_mean:.6f}")
                print(f"\n  Cosine fidelity:")
                print(f"    Real codebook (k-means):     {cos_real:.6f}")
                print(f"    Synthetic codebook (prior):   {cos_syn:.6f}")
                print(f"    Retention:                    {cos_syn/cos_real:.4f} of optimal")

                all_results.append({
                    "synthetic_test": True,
                    "target": target_key,
                    "cosine_real": round(float(cos_real), 6),
                    "cosine_synthetic": round(float(cos_syn), 6),
                    "retention": round(float(cos_syn / cos_real), 4),
                    "prior_source_count": len(ffn_transformer),
                })

    return {"predictions": all_results}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Codec Structural Prior experiments")
    parser.add_argument("--fingerprint", action="store_true", help="Run Experiment 1 only")
    parser.add_argument("--transfer", action="store_true", help="Run Experiment 2 only")
    parser.add_argument("--prior", action="store_true", help="Run Experiment 3 only")
    parser.add_argument("--json", action="store_true", help="Output JSON receipt")
    parser.add_argument("--output", type=str, help="Save receipt to file")
    args = parser.parse_args()

    run_all = not (args.fingerprint or args.transfer or args.prior)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("Collecting codebooks from all compressed models...", file=sys.stderr)
    records = collect_all_codebooks()
    print(f"Collected {len(records)} codebook records.\n", file=sys.stderr)

    results = {}

    if args.fingerprint or run_all:
        results["fingerprint"] = experiment_fingerprint(records)

    if args.transfer or run_all:
        results["transfer"] = experiment_transfer(records)

    if args.prior or run_all:
        results["prior"] = experiment_prior(records)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Is the codec a structural prior?")
    print("=" * 70)

    if "fingerprint" in results:
        acc = results["fingerprint"]["classification_accuracy"]
        verdict = "YES" if acc > 0.7 else "PARTIAL" if acc > 0.5 else "NO"
        print(f"\n  1. Architecture fingerprinting: {acc:.1%} accuracy → {verdict}")
        print(f"     (Can we ID architecture from codebook stats alone?)")

    if "transfer" in results:
        transfers = results["transfer"].get("transfers", [])
        if transfers:
            avg_retention = np.mean([t["retention"] for t in transfers])
            verdict = "YES" if avg_retention > 0.95 else "PARTIAL" if avg_retention > 0.85 else "NO"
            print(f"\n  2. Cross-model codebook transfer: {avg_retention:.4f} retention → {verdict}")
            print(f"     (Do codebooks from one model work on another?)")

    if "prior" in results:
        preds = [p for p in results["prior"]["predictions"] if "mape_std_pct" in p]
        syn = [p for p in results["prior"]["predictions"] if p.get("synthetic_test")]
        if preds:
            avg_mape = np.mean([p["mape_std_pct"] for p in preds])
            verdict = "YES" if avg_mape < 30 else "PARTIAL" if avg_mape < 50 else "NO"
            print(f"\n  3. Codebook prediction from metadata: {avg_mape:.1f}% MAPE → {verdict}")
            print(f"     (Can we predict codebook shape without seeing weights?)")
        if syn:
            ret = syn[0]["retention"]
            verdict = "YES" if ret > 0.95 else "PARTIAL" if ret > 0.85 else "NO"
            print(f"\n  4. Synthetic codebook generation: {ret:.4f} retention → {verdict}")
            print(f"     (Can a generated codebook compress unseen weights?)")

    print(f"\n{'=' * 70}")

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    cost = {
        "wall_time_s": round(wall, 3),
        "cpu_time_s": round(cpu, 3),
        "peak_memory_mb": _peak_memory_mb(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    receipt = {
        "experiment": "codec_structural_prior",
        "question": "Can the compression codec become a structural meta-model?",
        "n_codebooks": len(records),
        "n_models": len(set(r.model_name for r in records)),
        "results": results,
        "cost": cost,
    }

    if args.json:
        print(json.dumps(receipt, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(f"\n  Receipt: {out_path}", file=sys.stderr)

    # Always save to receipts
    receipts_dir = Path(__file__).resolve().parent.parent / "receipts" / "codec_prior"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"codec_prior_{time.strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"  Receipt saved: {receipt_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
