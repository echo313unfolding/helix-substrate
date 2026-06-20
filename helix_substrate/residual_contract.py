"""Residual Contract — structured damage profiling for codec comparison.

Every codec produces damage: E = W - W_hat. Today codecs are compared only
by output cosine and error magnitude. This module profiles the residual's
STRUCTURE — its kurtosis, rank, autocorrelation, spectral concentration, and
channel distribution — giving heterogeneous codecs a common language.

The residual structure gate (tools/residual_structure_gate.py) proved residuals
ARE structured:
  - spectral ratio 201x (frequency concentration)
  - ACF@10 0.157 (spatial autocorrelation)
  - SVD rank 41% (low-rank structure)

This module takes those findings and makes them actionable for routing:
  - ResidualProfile: structured fingerprint of codec damage
  - profile_residual(): compute from any codec's W and W_hat
  - DamageType: classify damage pattern for routing decisions
  - compare_codecs(): rank codecs by residual quality

Integration point: Hydra Router can use damage_type and structure_score to:
  1. Detect when a codec is suboptimal for a tensor (high structure_score)
  2. Choose between correction strategies (concentrated → sidecar, low-rank → SVD)
  3. Validate codec selection post-hoc (good codec → random residual)

Lineage:
  residual_structure_gate.py: GREEN_LIGHT (structure exists)
  sidecar probe: DEAD for affine (distributed error, not sparse outliers)
  This module: makes damage pattern a routing input, not just a validation check
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Damage classification
# ═══════════════════════════════════════════════════════════════════════════

class DamageType(str, Enum):
    DISTRIBUTED = "distributed"    # error spread evenly (affine-typical)
    CONCENTRATED = "concentrated"  # error in few positions (outlier-friendly)
    LOW_RANK = "low_rank"          # error has low-rank structure
    STRUCTURED = "structured"      # significant spatial/spectral pattern
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Residual profile
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResidualProfile:
    """Structured fingerprint of codec damage (E = W - W_hat).

    All fields are deterministic and reproducible from the same inputs.
    """
    # Overall magnitude
    rms_error: float = 0.0         # sqrt(mean(E^2))
    cosine: float = 1.0            # cos(W, W_hat)
    max_abs_error: float = 0.0     # max(|E|)
    mean_abs_error: float = 0.0    # mean(|E|)

    # Distribution shape
    kurtosis: float = 0.0          # excess kurtosis of E (normal=0)
    sparsity: float = 0.0          # fraction of |E| < 0.1 * rms (0=dense, 1=sparse)

    # Spatial structure (from residual_structure_gate proven features)
    acf_lag1: float = 0.0          # autocorrelation at lag 1 (row-major)
    acf_lag10: float = 0.0         # autocorrelation at lag 10
    spectral_ratio: float = 1.0    # max/mean PSD (1.0 = flat/random)

    # Rank structure
    svd_rank_ratio: float = 1.0    # components for 99% var / total (1.0 = full rank)
    top10_explained: float = 0.0   # fraction of variance in top 10 SVD components

    # Channel concentration
    channel_concentration: float = 1.0  # max per-row rms / mean per-row rms

    # Composite
    structure_score: float = 0.0   # 0-1 composite (higher = more exploitable structure)
    damage_type: DamageType = DamageType.UNKNOWN

    def to_dict(self) -> dict:
        d = asdict(self)
        d["damage_type"] = self.damage_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ResidualProfile":
        d = dict(d)
        if "damage_type" in d:
            d["damage_type"] = DamageType(d["damage_type"])
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════
# Profile computation
# ═══════════════════════════════════════════════════════════════════════════

def profile_residual(W: np.ndarray, W_hat: np.ndarray) -> ResidualProfile:
    """Compute ResidualProfile from original and reconstructed tensors.

    Args:
        W: original weight tensor (any shape, will use as-is for 2D analysis)
        W_hat: reconstructed tensor (same shape as W)

    Returns:
        ResidualProfile with all fields populated
    """
    W = np.asarray(W, dtype=np.float32)
    W_hat = np.asarray(W_hat, dtype=np.float32)

    E = W - W_hat
    flat = E.ravel()
    n = len(flat)

    if n == 0:
        return ResidualProfile()

    # ── Overall magnitude ──
    rms = float(np.sqrt(np.mean(flat ** 2)))
    max_abs = float(np.max(np.abs(flat)))
    mean_abs = float(np.mean(np.abs(flat)))

    # Cosine similarity
    w_flat = W.ravel()
    w_norm = np.linalg.norm(w_flat)
    wh_norm = np.linalg.norm(W_hat.ravel())
    if w_norm > 1e-12 and wh_norm > 1e-12:
        cosine = float(np.dot(w_flat, W_hat.ravel()) / (w_norm * wh_norm))
    else:
        cosine = 1.0 if rms < 1e-12 else 0.0

    # ── Distribution shape ──
    var = float(np.var(flat))
    if var > 1e-30:
        m4 = float(np.mean((flat - flat.mean()) ** 4))
        kurtosis = m4 / (var ** 2) - 3.0  # excess kurtosis
    else:
        kurtosis = 0.0

    # Sparsity: fraction of elements with |E| < 0.1 * rms
    if rms > 1e-12:
        sparsity = float(np.mean(np.abs(flat) < 0.1 * rms))
    else:
        sparsity = 1.0

    # ── Spatial structure ──
    acf1, acf10 = _autocorrelation_fast(flat)
    sr = _spectral_ratio_fast(flat)

    # ── Rank structure (only for 2D) ──
    svd_rank_ratio = 1.0
    top10_explained = 0.0
    if E.ndim == 2 and min(E.shape) > 1:
        svd_rank_ratio, top10_explained = _svd_rank_fast(E)

    # ── Channel concentration (only for 2D) ──
    channel_conc = 1.0
    if E.ndim == 2 and E.shape[0] > 1:
        row_rms = np.sqrt(np.mean(E ** 2, axis=1))
        mean_row_rms = float(row_rms.mean())
        if mean_row_rms > 1e-12:
            channel_conc = float(row_rms.max() / mean_row_rms)

    # ── Composite score and classification ──
    structure_score = _compute_structure_score(
        acf1, acf10, sr, svd_rank_ratio, kurtosis, sparsity, channel_conc
    )
    damage_type = _classify_damage(
        kurtosis, sparsity, svd_rank_ratio, sr, acf10, channel_conc
    )

    return ResidualProfile(
        rms_error=round(rms, 8),
        cosine=round(cosine, 6),
        max_abs_error=round(max_abs, 8),
        mean_abs_error=round(mean_abs, 8),
        kurtosis=round(kurtosis, 4),
        sparsity=round(sparsity, 4),
        acf_lag1=round(acf1, 6),
        acf_lag10=round(acf10, 6),
        spectral_ratio=round(sr, 2),
        svd_rank_ratio=round(svd_rank_ratio, 4),
        top10_explained=round(top10_explained, 4),
        channel_concentration=round(channel_conc, 4),
        structure_score=round(structure_score, 4),
        damage_type=damage_type,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fast feature extractors (extracted from residual_structure_gate.py)
# ═══════════════════════════════════════════════════════════════════════════

def _autocorrelation_fast(flat: np.ndarray) -> tuple[float, float]:
    """Compute ACF at lag 1 and lag 10 for a 1D signal."""
    x = flat - flat.mean()
    var = float(np.var(flat))
    if var < 1e-30 or len(x) < 12:
        return 0.0, 0.0

    n = len(x)
    acf1 = float(np.dot(x[:n - 1], x[1:]) / (var * n))
    acf10 = float(np.dot(x[:n - 10], x[10:]) / (var * n)) if n > 10 else 0.0
    return acf1, acf10


def _spectral_ratio_fast(flat: np.ndarray) -> float:
    """Max/mean PSD ratio. Flat spectrum ≈ 1.0, structured >> 1."""
    x = flat - flat.mean()
    if np.var(x) < 1e-30:
        return 1.0
    n = min(len(x), 500_000)
    psd = np.abs(np.fft.rfft(x[:n])) ** 2
    psd = psd[1:]  # drop DC
    if len(psd) == 0 or psd.mean() < 1e-30:
        return 1.0
    return float(psd.max() / psd.mean())


def _svd_rank_fast(E_2d: np.ndarray) -> tuple[float, float]:
    """SVD rank ratio and top-10 explained variance for 2D residual."""
    r, c = E_2d.shape
    # Subsample for speed
    if r > 500:
        idx = np.random.RandomState(42).choice(r, 500, replace=False)
        E_2d = E_2d[idx, :]
    if c > 500:
        idx = np.random.RandomState(43).choice(c, 500, replace=False)
        E_2d = E_2d[:, idx]

    try:
        _, s, _ = np.linalg.svd(E_2d, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0, 0.0

    total_var = np.sum(s ** 2)
    if total_var < 1e-30:
        return 1.0, 0.0

    cumvar = np.cumsum(s ** 2) / total_var
    n99 = int(np.searchsorted(cumvar, 0.99) + 1)
    rank_ratio = n99 / len(s)
    top10_exp = float(cumvar[min(9, len(cumvar) - 1)])

    return round(rank_ratio, 4), round(top10_exp, 4)


# ═══════════════════════════════════════════════════════════════════════════
# Classification logic
# ═══════════════════════════════════════════════════════════════════════════

def _compute_structure_score(
    acf1: float, acf10: float, spectral_ratio: float,
    svd_rank_ratio: float, kurtosis: float,
    sparsity: float, channel_conc: float,
) -> float:
    """Composite structure score: 0 = random residual, 1 = highly structured.

    Weights chosen to reflect proven importance from residual_structure_gate:
    spectral_ratio was 201x (dominant), ACF@10 was 0.157 (moderate),
    SVD rank was 41% (moderate). Kurtosis and channel concentration are
    supplementary signals.
    """
    # Normalize each feature to [0, 1]
    # spectral_ratio: 1 = random, 200+ = highly structured
    sr_norm = min(1.0, max(0.0, (spectral_ratio - 1.0) / 50.0))

    # ACF@10: 0 = no correlation, 0.15+ = structured
    acf_norm = min(1.0, max(0.0, abs(acf10) / 0.2))

    # SVD rank: 1.0 = full rank (random), 0.4 = low rank (structured)
    rank_norm = max(0.0, 1.0 - svd_rank_ratio)

    # Kurtosis: 0 = normal, high = heavy tails (concentrated)
    kurt_norm = min(1.0, max(0.0, abs(kurtosis) / 10.0))

    # Channel concentration: 1.0 = uniform, 5+ = concentrated
    conc_norm = min(1.0, max(0.0, (channel_conc - 1.0) / 4.0))

    # Weighted combination (spectral dominates per proven results)
    score = (
        0.35 * sr_norm +
        0.25 * acf_norm +
        0.20 * rank_norm +
        0.10 * kurt_norm +
        0.10 * conc_norm
    )

    return min(1.0, max(0.0, score))


def _classify_damage(
    kurtosis: float, sparsity: float, svd_rank_ratio: float,
    spectral_ratio: float, acf10: float, channel_conc: float,
) -> DamageType:
    """Classify damage pattern from residual features.

    Rules (in priority order):
    1. Low-rank (SVD rank ratio < 0.5) → LOW_RANK
    2. Concentrated (kurtosis > 6 AND channel_conc > 3) → CONCENTRATED
    3. Structured (spectral_ratio > 10 OR acf10 > 0.1) → STRUCTURED
    4. Otherwise → DISTRIBUTED
    """
    if svd_rank_ratio < 0.5:
        return DamageType.LOW_RANK

    if kurtosis > 6.0 and channel_conc > 3.0:
        return DamageType.CONCENTRATED

    if spectral_ratio > 20.0 or abs(acf10) > 0.1:
        return DamageType.STRUCTURED

    return DamageType.DISTRIBUTED


# ═══════════════════════════════════════════════════════════════════════════
# Codec comparison
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CodecComparison:
    """Result of comparing multiple codecs on the same tensor."""
    tensor_name: str
    profiles: dict[str, ResidualProfile]  # codec_name → ResidualProfile
    best_codec: str                        # lowest structure_score
    ranking: list[str]                     # codecs sorted by quality

    def to_dict(self) -> dict:
        return {
            "tensor_name": self.tensor_name,
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "best_codec": self.best_codec,
            "ranking": self.ranking,
        }


def compare_codecs(
    tensor_name: str,
    codec_results: dict[str, tuple[np.ndarray, np.ndarray]],
) -> CodecComparison:
    """Compare multiple codecs on the same tensor by residual quality.

    Args:
        tensor_name: name of the tensor
        codec_results: {codec_name: (W, W_hat)} — all W should be identical

    Returns:
        CodecComparison with profiles and ranking

    The best codec is the one with the lowest structure_score (most random
    residual = closest to optimal compression). Ties broken by cosine.
    """
    profiles = {}
    for codec_name, (W, W_hat) in codec_results.items():
        profiles[codec_name] = profile_residual(W, W_hat)

    # Rank: lowest structure_score first, cosine as tiebreaker (higher better)
    ranking = sorted(
        profiles.keys(),
        key=lambda k: (profiles[k].structure_score, -profiles[k].cosine),
    )

    return CodecComparison(
        tensor_name=tensor_name,
        profiles=profiles,
        best_codec=ranking[0] if ranking else "",
        ranking=ranking,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Routing integration
# ═══════════════════════════════════════════════════════════════════════════

def residual_routing_signal(profile: ResidualProfile) -> dict:
    """Extract routing-relevant signals from a residual profile.

    Returns dict with:
        codec_optimal: bool — True if residual is near-random (codec is good fit)
        try_correction: bool — True if structured residual suggests correction
        correction_hint: str — what kind of correction might help
        confidence: float — how confident the signal is
    """
    codec_optimal = profile.structure_score < 0.2
    try_correction = profile.structure_score > 0.5

    if profile.damage_type == DamageType.CONCENTRATED:
        hint = "outlier_correction"
    elif profile.damage_type == DamageType.LOW_RANK:
        hint = "low_rank_correction"
    elif profile.damage_type == DamageType.STRUCTURED:
        hint = "spatial_correction"
    else:
        hint = "none"

    # Confidence: high when structure_score is clearly above/below threshold
    if profile.structure_score < 0.1 or profile.structure_score > 0.7:
        confidence = 0.9
    elif profile.structure_score < 0.2 or profile.structure_score > 0.5:
        confidence = 0.7
    else:
        confidence = 0.4  # ambiguous zone

    return {
        "codec_optimal": codec_optimal,
        "try_correction": try_correction,
        "correction_hint": hint,
        "damage_type": profile.damage_type.value,
        "structure_score": profile.structure_score,
        "confidence": confidence,
    }
