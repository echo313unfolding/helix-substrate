"""Cross-model compression-signal sweep.

Tests whether compression-induced routing signals are:
1. LLM-specific?
2. Neural-weight-specific?
3. Codec-specific?
4. A general property of lossy encoding under heterogeneous structure?

Hypothesis: lossy compression creates measurable failure modes whose geometry
can be used as routing signals. This is especially true when:
  - the input has structure
  - the codec has an inductive bias
  - the compression is strong enough to reveal mismatch
  - the residual is not pure noise
  - the routing system can act conservatively

The sweep runs multiple codec families across multiple model families and
measures whether residual geometry predicts codec suitability better than
scalar reconstruction metrics alone.

Usage:
    python tools/sweep_compression_routing_signal.py [--models-dir PATH] [--output PATH]
    python tools/sweep_compression_routing_signal.py --synthetic-only

Requires: numpy. Optional: safetensors, torch (for loading real models).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_substrate.residual_contract import (
    DamageType,
    ResidualProfile,
    profile_residual,
    compare_codecs,
    residual_routing_signal,
)
from helix_substrate.ghost_bridge import ghost_features_from_bytes


# ═══════════════════════════════════════════════════════════════════════════
# Codec implementations (numpy-only, self-contained)
# ═══════════════════════════════════════════════════════════════════════════

def _codec_exact(W: np.ndarray) -> tuple[np.ndarray, dict]:
    """Exact passthrough — zero error baseline."""
    return W.copy(), {"codec": "exact", "bpw": 32.0}


def _codec_affine(W: np.ndarray, block_size: int = 128) -> tuple[np.ndarray, dict]:
    """Block-affine quantization (6-bit equivalent, g128)."""
    flat = W.ravel().astype(np.float32)
    n = len(flat)
    bs = min(block_size, n)
    if bs < 1:
        return W.copy(), {"codec": "affine", "bpw": 32.0}

    # Pad to block boundary
    pad_n = (bs - n % bs) % bs
    if pad_n > 0:
        flat = np.concatenate([flat, np.zeros(pad_n, dtype=np.float32)])

    blocks = flat.reshape(-1, bs)
    bias = blocks.mean(axis=1, keepdims=True)
    residual = blocks - bias
    scale = np.abs(residual).max(axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-8)

    # 6-bit: 64 levels in [-1, 1]
    n_levels = 64
    quantized = np.round(residual / scale * (n_levels // 2 - 1))
    quantized = np.clip(quantized, -(n_levels // 2), n_levels // 2 - 1)
    recon = (quantized / (n_levels // 2 - 1)) * scale + bias
    recon_flat = recon.ravel()[:n]

    # Encoded bytes: quantized indices as uint8
    indices = (quantized + n_levels // 2).astype(np.uint8).ravel()[:n]

    bpw = 6.0 + 32.0 / bs  # 6 bits per weight + scale/bias overhead
    return recon_flat.reshape(W.shape), {
        "codec": "affine_g128",
        "bpw": round(bpw, 2),
        "encoded_bytes": indices.tobytes(),
    }


def _codec_vq(W: np.ndarray, k: int = 256, max_iters: int = 10) -> tuple[np.ndarray, dict]:
    """Scalar VQ (k-means, k=256 = 8-bit)."""
    flat = W.ravel().astype(np.float32)
    n = len(flat)

    # Subsample for k-means fitting
    rng = np.random.RandomState(42)
    sample_n = min(500_000, n)
    sample = flat[rng.choice(n, sample_n, replace=False)] if n > sample_n else flat.copy()

    centroids = np.percentile(sample, np.linspace(0, 100, k)).astype(np.float32)

    for _ in range(max_iters):
        # Assign in chunks
        chunk = 500_000
        assignments = np.empty(len(sample), dtype=np.int32)
        for s in range(0, len(sample), chunk):
            e = min(s + chunk, len(sample))
            dists = np.abs(sample[s:e, None] - centroids[None, :])
            assignments[s:e] = np.argmin(dists, axis=1)

        new_centroids = np.empty_like(centroids)
        for i in range(k):
            mask = assignments == i
            new_centroids[i] = sample[mask].mean() if mask.any() else centroids[i]

        if np.allclose(centroids, new_centroids, atol=1e-7):
            break
        centroids = new_centroids

    # Full assignment
    indices = np.empty(n, dtype=np.uint8)
    for s in range(0, n, 500_000):
        e = min(s + 500_000, n)
        dists = np.abs(flat[s:e, None] - centroids[None, :])
        indices[s:e] = np.argmin(dists, axis=1).astype(np.uint8)

    recon = centroids[indices].reshape(W.shape)
    bpw = 8.0 + (k * 32.0) / n  # 8-bit indices + codebook overhead
    return recon, {
        "codec": "vq_k256",
        "bpw": round(bpw, 2),
        "encoded_bytes": indices.tobytes(),
    }


def _codec_rvq(W: np.ndarray, k1: int = 16, k2: int = 16) -> tuple[np.ndarray, dict]:
    """Two-stage residual VQ (4-bit + 4-bit = 8-bit total)."""
    from helix_substrate.rvq_codec import encode_rvq_tensor

    if W.ndim == 1:
        W2d = W.reshape(1, -1)
    elif W.ndim > 2:
        W2d = W.reshape(-1, W.shape[-1])
    else:
        W2d = W

    cb1, cb2, packed, recon = encode_rvq_tensor(W2d.astype(np.float32), k1=k1, k2=k2)
    recon = recon.reshape(W.shape)
    n = W.size
    bpw = 8.0 + (k1 + k2) * 32.0 / n
    return recon, {
        "codec": "rvq_16x16",
        "bpw": round(bpw, 2),
        "encoded_bytes": packed.tobytes(),
    }


def _codec_svd_lowrank(W: np.ndarray, rank: int = 8) -> tuple[np.ndarray, dict]:
    """Low-rank SVD approximation (keep top-k singular values)."""
    if W.ndim == 1:
        W2d = W.reshape(1, -1)
    elif W.ndim > 2:
        W2d = W.reshape(-1, W.shape[-1])
    else:
        W2d = W

    try:
        U, s, Vt = np.linalg.svd(W2d.astype(np.float32), full_matrices=False)
    except np.linalg.LinAlgError:
        return W.copy(), {"codec": "svd_lowrank", "bpw": 32.0, "error": "svd_failed"}

    r = min(rank, len(s))
    recon = (U[:, :r] * s[:r]) @ Vt[:r, :]
    recon = recon.reshape(W.shape)

    # bpw: store U[:,:r], s[:r], Vt[:r,:] in float32
    rows, cols = W2d.shape
    stored_params = rows * r + r + r * cols
    bpw = (stored_params * 32.0) / W.size
    return recon, {"codec": f"svd_rank{r}", "bpw": round(bpw, 2)}


CODEC_REGISTRY = {
    "exact": _codec_exact,
    "affine_g128": lambda W: _codec_affine(W, block_size=128),
    "vq_k256": lambda W: _codec_vq(W, k=256),
    "rvq_16x16": _codec_rvq,
    "svd_rank8": lambda W: _codec_svd_lowrank(W, rank=8),
}


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic tensor generators (for testing without model downloads)
# ═══════════════════════════════════════════════════════════════════════════

def _make_transformer_mlp(rng, rows=256, cols=512):
    """Dense Transformer MLP-like: near-Gaussian with mild outliers."""
    W = rng.randn(rows, cols).astype(np.float32) * 0.02
    # Add a few outlier channels
    W[0, :] *= 10.0
    W[1, :] *= 8.0
    return W, {"family": "transformer", "role": "mlp", "synthetic": True}


def _make_attention_qkv(rng, rows=256, cols=256):
    """Attention QKV-like: structured low-rank pattern."""
    U = rng.randn(rows, 4).astype(np.float32) * 0.1
    V = rng.randn(4, cols).astype(np.float32) * 0.1
    W = U @ V + rng.randn(rows, cols).astype(np.float32) * 0.005
    return W, {"family": "transformer", "role": "attention", "synthetic": True}


def _make_ssm_state(rng, rows=128, cols=256):
    """SSM/Mamba state-like: smooth, low-frequency structure."""
    base = rng.randn(rows, cols).astype(np.float32) * 0.01
    # Add smooth spatial structure
    kernel = np.ones(16) / 16.0
    for i in range(rows):
        base[i] = np.convolve(base[i], kernel, mode='same')
    return base, {"family": "ssm", "role": "state_proj", "synthetic": True}


def _make_moe_expert(rng, rows=256, cols=512):
    """MoE expert-like: sparse with expert-specific distribution."""
    W = rng.randn(rows, cols).astype(np.float32) * 0.015
    # Make it sparse (many near-zero)
    mask = rng.rand(rows, cols) < 0.3
    W[mask] *= 0.01
    return W, {"family": "moe", "role": "expert_ffn", "synthetic": True}


def _make_cnn_conv(rng, rows=64, cols=576):
    """CNN conv-like: spatial/frequency structure (flattened 3x3x64 kernel)."""
    W = rng.randn(rows, cols).astype(np.float32) * 0.05
    # Add spatial frequency pattern
    for i in range(rows):
        freq = rng.uniform(0.1, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        pattern = np.sin(np.linspace(0, freq * np.pi, cols) + phase) * 0.02
        W[i] += pattern
    return W, {"family": "cnn", "role": "conv_weight", "synthetic": True}


def _make_embedding(rng, rows=1000, cols=128):
    """Embedding table-like: clustered rows, high inter-row variance."""
    n_clusters = 5
    centers = rng.randn(n_clusters, cols).astype(np.float32) * 0.5
    labels = rng.randint(0, n_clusters, rows)
    W = centers[labels] + rng.randn(rows, cols).astype(np.float32) * 0.05
    return W, {"family": "embedding", "role": "embed_tokens", "synthetic": True}


SYNTHETIC_GENERATORS = {
    "transformer_mlp": _make_transformer_mlp,
    "attention_qkv": _make_attention_qkv,
    "ssm_state": _make_ssm_state,
    "moe_expert": _make_moe_expert,
    "cnn_conv": _make_cnn_conv,
    "embedding": _make_embedding,
}


# ═══════════════════════════════════════════════════════════════════════════
# Tensor loading from real models
# ═══════════════════════════════════════════════════════════════════════════

def _classify_tensor_role(name: str) -> str:
    """Classify tensor role from parameter name."""
    name_lower = name.lower()
    for keyword in ["embed", "lm_head", "wte", "wpe"]:
        if keyword in name_lower:
            return "embedding"
    for keyword in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn", "attention"]:
        if keyword in name_lower:
            return "attention"
    for keyword in ["gate_proj", "up_proj", "down_proj", "mlp", "fc1", "fc2", "ffn"]:
        if keyword in name_lower:
            return "mlp"
    for keyword in ["in_proj", "out_proj", "x_proj", "dt_proj", "ssm", "mixer"]:
        if keyword in name_lower:
            return "state_proj"
    for keyword in ["experts", "gate"]:
        if keyword in name_lower:
            return "moe"
    for keyword in ["conv", "dwconv"]:
        if keyword in name_lower:
            return "conv_weight"
    for keyword in ["norm", "layernorm", "rmsnorm"]:
        if keyword in name_lower:
            return "norm"
    return "unknown"


def _classify_model_family(model_path: str) -> str:
    """Guess model family from path/name."""
    name_lower = model_path.lower()
    if "zamba" in name_lower:
        return "hybrid"
    if "mamba" in name_lower and "hybrid" in name_lower:
        return "hybrid"
    if "mamba" in name_lower:
        return "ssm"
    if "olmoe" in name_lower or "mixtral" in name_lower or "moe" in name_lower:
        return "moe"
    if "clip" in name_lower or "vit" in name_lower or "resnet" in name_lower:
        return "cnn"
    if "bert" in name_lower:
        return "encoder_transformer"
    return "transformer"


def load_tensors_from_safetensors(path: Path, max_tensors: int = 20,
                                  max_elements: int = 512 * 1024) -> list[tuple[str, np.ndarray, dict]]:
    """Load tensors from a safetensors file, filtering by size."""
    results = []
    try:
        from safetensors import safe_open
    except ImportError:
        return results

    model_family = _classify_model_family(str(path))

    if not Path(path).exists():
        return results

    with safe_open(str(path), framework="numpy") as f:
        keys = list(f.keys())
        for key in keys:
            if len(results) >= max_tensors:
                break
            tensor = f.get_tensor(key)
            if tensor.size > max_elements or tensor.ndim < 1:
                continue
            if tensor.size < 64:
                continue
            role = _classify_tensor_role(key)
            if role == "norm":
                continue  # Skip 1D norms, not interesting for compression
            results.append((key, tensor.astype(np.float32), {
                "family": model_family,
                "role": role,
                "synthetic": False,
                "source": str(path),
            }))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Sweep engine
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SweepRecord:
    """One tensor × one codec result."""
    tensor_id: str
    model_family: str
    tensor_role: str
    synthetic: bool
    codec_name: str
    bpw: float
    # Scalar metrics
    cosine: float
    rms_error: float
    max_abs_error: float
    # Residual profile
    residual_profile: dict
    # Ghost features (if encoded bytes available)
    ghost_features: Optional[dict]
    # Routing signal
    route_signal: dict
    # Receipt
    receipt_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_sweep(
    tensors: list[tuple[str, np.ndarray, dict]],
    codecs: dict = None,
    output_path: Path = None,
) -> list[SweepRecord]:
    """Run the cross-model compression-signal sweep.

    Args:
        tensors: list of (name, array, metadata) tuples
        codecs: codec registry override (default: CODEC_REGISTRY)
        output_path: if set, write JSONL receipts here

    Returns:
        list of SweepRecord
    """
    if codecs is None:
        codecs = CODEC_REGISTRY

    records = []
    t_start = time.time()

    for tensor_name, W, meta in tensors:
        family = meta.get("family", "unknown")
        role = meta.get("role", "unknown")
        synthetic = meta.get("synthetic", True)

        for codec_name, codec_fn in codecs.items():
            try:
                result = codec_fn(W)
                W_hat, codec_meta = result
            except Exception as e:
                # Skip codecs that fail on this tensor
                continue

            # Residual profile
            rp = profile_residual(W, W_hat)

            # Ghost features from encoded bytes if available
            ghost = None
            encoded_bytes = codec_meta.get("encoded_bytes")
            if encoded_bytes and len(encoded_bytes) >= 64:
                shape = W.shape if W.ndim == 2 else ()
                ghost = ghost_features_from_bytes(encoded_bytes, shape=shape)

            # Routing signal
            signal = residual_routing_signal(rp)

            # Receipt hash
            content = f"{tensor_name}:{codec_name}:{rp.cosine}:{rp.structure_score}"
            receipt_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            record = SweepRecord(
                tensor_id=tensor_name,
                model_family=family,
                tensor_role=role,
                synthetic=synthetic,
                codec_name=codec_name,
                bpw=codec_meta.get("bpw", 0.0),
                cosine=rp.cosine,
                rms_error=rp.rms_error,
                max_abs_error=rp.max_abs_error,
                residual_profile=rp.to_dict(),
                ghost_features=ghost,
                route_signal=signal,
                receipt_hash=receipt_hash,
            )
            records.append(record)

    wall_time = time.time() - t_start

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")

        # Write summary
        summary = analyze_sweep(records)
        summary["cost"] = {
            "wall_time_s": round(wall_time, 3),
            "cpu_time_s": round(time.process_time(), 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "n_records": len(records),
        }
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2))

    return records


# ═══════════════════════════════════════════════════════════════════════════
# Quality-gated codec ranking
# ═══════════════════════════════════════════════════════════════════════════

def quality_gated_best_codec(
    lossy_records: list[SweepRecord],
    cosine_window: float = 0.01,
    min_cosine_floor: float = 0.90,
) -> dict:
    """Choose best codec using quality gate first, then residual structure.

    Two-stage ranking:
      1. Quality gate: reject codecs whose cosine is too bad.
      2. Among eligible codecs, pick lowest structure_score (most benign residual).

    Args:
        lossy_records: SweepRecords for one tensor, exact excluded
        cosine_window: eligible if cosine >= max_cosine - window
        min_cosine_floor: absolute minimum cosine to be eligible

    Returns:
        dict with best_codec, reason, eligible_codecs, rejected_for_quality, etc.
    """
    if not lossy_records:
        return {
            "best_codec": None,
            "reason": "no_candidates",
            "max_cosine": None,
            "eligible_count": 0,
            "eligible_codecs": [],
            "rejected_for_quality": [],
            "residual_changed_choice": False,
        }

    max_cosine = max(r.cosine for r in lossy_records)
    best_by_cosine = max(lossy_records, key=lambda r: r.cosine)

    # Quality gate: cosine >= floor AND within window of best
    eligible = [
        r for r in lossy_records
        if r.cosine >= min_cosine_floor and r.cosine >= max_cosine - cosine_window
    ]
    rejected = [r.codec_name for r in lossy_records if r not in eligible]

    if not eligible:
        # No codec meets quality gate — fall back to best cosine
        return {
            "best_codec": best_by_cosine.codec_name,
            "reason": "best_cosine_no_quality_peer",
            "max_cosine": round(max_cosine, 6),
            "eligible_count": 0,
            "eligible_codecs": [],
            "rejected_for_quality": rejected,
            "residual_changed_choice": False,
        }

    # Among eligible, pick lowest structure_score, break ties by bpw then cosine
    eligible_sorted = sorted(
        eligible,
        key=lambda r: (
            r.residual_profile["structure_score"],
            r.bpw,
            -r.cosine,
        ),
    )
    best = eligible_sorted[0]
    residual_changed = best.codec_name != best_by_cosine.codec_name

    return {
        "best_codec": best.codec_name,
        "reason": (
            "residual_tiebreak" if residual_changed
            else "cosine_and_residual_agree"
        ),
        "max_cosine": round(max_cosine, 6),
        "eligible_count": len(eligible),
        "eligible_codecs": [r.codec_name for r in eligible_sorted],
        "rejected_for_quality": rejected,
        "residual_changed_choice": residual_changed,
        "best_structure_score": round(best.residual_profile["structure_score"], 4),
        "best_cosine": round(best.cosine, 6),
        "best_bpw": best.bpw,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_sweep(records: list[SweepRecord]) -> dict:
    """Analyze sweep results against the six research questions.

    Q1. Can residual features predict tensor role across model families?
    Q2. Do residual features improve routing among quality-eligible codecs?
    Q3. Do features transfer from one model family to another?
    Q4. Which codecs create useful routing signals?
    Q5. Are signals stronger in lossy vs lossless?
    Q6. Do residual features add information beyond cosine/error magnitude?
    """
    if not records:
        return {"error": "no_records"}

    # Group by tensor
    tensor_groups = {}
    for r in records:
        tensor_groups.setdefault(r.tensor_id, []).append(r)

    # Q2: Do residual features improve routing among quality-eligible codecs?
    # Two-stage: quality gate first, then residual structure among eligible.
    q2_results = []
    n_residual_changed = 0
    for tid, group in tensor_groups.items():
        lossy = [r for r in group if r.codec_name != "exact"]
        if len(lossy) < 2:
            continue
        best_by_cosine = max(lossy, key=lambda r: r.cosine)
        gated = quality_gated_best_codec(lossy)
        q2_results.append({
            "tensor": tid,
            "best_cosine_codec": best_by_cosine.codec_name,
            "best_gated_codec": gated["best_codec"],
            "agree": best_by_cosine.codec_name == gated["best_codec"],
            "reason": gated["reason"],
            "eligible_count": gated["eligible_count"],
            "rejected_for_quality": gated["rejected_for_quality"],
            "residual_changed_choice": gated["residual_changed_choice"],
        })
        if gated["residual_changed_choice"]:
            n_residual_changed += 1

    q2_agreement = (
        sum(1 for r in q2_results if r["agree"]) / len(q2_results)
        if q2_results else 0.0
    )

    # Q4: Which codecs create structured residuals?
    codec_structure = {}
    for r in records:
        if r.codec_name == "exact":
            continue
        codec_structure.setdefault(r.codec_name, []).append(
            r.residual_profile["structure_score"]
        )

    q4_results = {
        codec: {
            "mean_structure_score": round(float(np.mean(scores)), 4),
            "std_structure_score": round(float(np.std(scores)), 4),
            "n_structured": sum(1 for s in scores if s > 0.3),
            "n_total": len(scores),
            "fraction_structured": round(
                sum(1 for s in scores if s > 0.3) / len(scores), 3
            ),
        }
        for codec, scores in codec_structure.items()
    }

    # Q5: exact baseline should have zero/near-zero structure
    exact_records = [r for r in records if r.codec_name == "exact"]
    q5_exact_structure = (
        [r.residual_profile["structure_score"] for r in exact_records]
        if exact_records else []
    )

    # Q6: Do residual features add info beyond cosine?
    # Compare: rank codecs by cosine vs rank by structure_score
    # If rankings differ, structure_score adds information
    q6_results = []
    for tid, group in tensor_groups.items():
        lossy = [r for r in group if r.codec_name != "exact"]
        if len(lossy) < 2:
            continue
        rank_cosine = sorted(lossy, key=lambda r: -r.cosine)
        rank_structure = sorted(lossy, key=lambda r: r.residual_profile["structure_score"])
        cosine_order = [r.codec_name for r in rank_cosine]
        structure_order = [r.codec_name for r in rank_structure]
        q6_results.append({
            "tensor": tid,
            "cosine_ranking": cosine_order,
            "structure_ranking": structure_order,
            "rankings_identical": cosine_order == structure_order,
        })

    q6_ranking_divergence = (
        sum(1 for r in q6_results if not r["rankings_identical"]) / len(q6_results)
        if q6_results else 0.0
    )

    # Damage type distribution per family
    family_damage = {}
    for r in records:
        if r.codec_name == "exact":
            continue
        dt = r.residual_profile.get("damage_type", "unknown")
        family_damage.setdefault(r.model_family, {}).setdefault(dt, 0)
        family_damage[r.model_family][dt] += 1

    # Ghost feature availability
    ghost_available = sum(1 for r in records if r.ghost_features is not None)

    return {
        "n_tensors": len(tensor_groups),
        "n_records": len(records),
        "n_families": len(set(r.model_family for r in records)),
        "families": sorted(set(r.model_family for r in records)),
        "codecs_tested": sorted(set(r.codec_name for r in records)),
        "Q2_quality_gated_routing": {
            "agreement_rate": round(q2_agreement, 3),
            "n_tensors_compared": len(q2_results),
            "n_residual_changed_choice": n_residual_changed,
            "interpretation": (
                "residual geometry does not replace reconstruction quality; "
                "it improves routing among quality-eligible codecs"
            ),
            "detail": q2_results[:10],
        },
        "Q4_codec_structure_signals": q4_results,
        "Q5_exact_baseline": {
            "n_exact": len(q5_exact_structure),
            "max_structure_score": round(max(q5_exact_structure), 6) if q5_exact_structure else None,
            "all_near_zero": all(s < 0.01 for s in q5_exact_structure) if q5_exact_structure else None,
        },
        "Q6_ranking_divergence": {
            "fraction_divergent": round(q6_ranking_divergence, 3),
            "n_compared": len(q6_results),
            "interpretation": (
                "structure_score adds information beyond cosine"
                if q6_ranking_divergence > 0.2
                else "structure_score largely agrees with cosine"
            ),
        },
        "damage_type_by_family": family_damage,
        "ghost_features_available": ghost_available,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-model compression-signal sweep")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Directory containing .safetensors files")
    parser.add_argument("--output", type=Path,
                        default=Path("receipts/compression_signal_sweep.jsonl"))
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Only run synthetic tensors (no model loading)")
    parser.add_argument("--max-tensors-per-model", type=int, default=10)
    args = parser.parse_args()

    tensors = []

    # Always include synthetic tensors
    rng = np.random.RandomState(42)
    for gen_name, gen_fn in SYNTHETIC_GENERATORS.items():
        W, meta = gen_fn(rng)
        tensors.append((f"synthetic/{gen_name}", W, meta))

    # Load real models if available and not synthetic-only
    if not args.synthetic_only and args.models_dir:
        model_dir = Path(args.models_dir)
        if model_dir.exists():
            for sf_path in sorted(model_dir.glob("**/*.safetensors")):
                loaded = load_tensors_from_safetensors(
                    sf_path, max_tensors=args.max_tensors_per_model
                )
                if loaded:
                    print(f"Loaded {len(loaded)} tensors from {sf_path.name}")
                    tensors.extend(loaded)
                else:
                    print(f"Skipped {sf_path.name} (no suitable tensors or missing safetensors)")
        else:
            print(f"Models dir {model_dir} not found, using synthetic only")

    if not tensors:
        print("No tensors to sweep. Use --synthetic-only or provide --models-dir.")
        return

    print(f"\nSweeping {len(tensors)} tensors × {len(CODEC_REGISTRY)} codecs...")
    records = run_sweep(tensors, output_path=args.output)

    summary = analyze_sweep(records)
    print(f"\nResults: {summary['n_records']} records across {summary['n_families']} families")
    q2 = summary["Q2_quality_gated_routing"]
    print(f"Q2 quality-gated agreement: {q2['agreement_rate']:.1%} "
          f"(residual changed choice: {q2['n_residual_changed_choice']})")
    print(f"Q4 codecs with structured residuals:")
    for codec, info in summary["Q4_codec_structure_signals"].items():
        print(f"  {codec}: {info['fraction_structured']:.0%} structured "
              f"(mean score {info['mean_structure_score']:.3f})")
    print(f"Q5 exact baseline all near-zero: {summary['Q5_exact_baseline']['all_near_zero']}")
    print(f"Q6 ranking divergence: {summary['Q6_ranking_divergence']['fraction_divergent']:.1%}")
    print(f"\nReceipts: {args.output}")


if __name__ == "__main__":
    main()
