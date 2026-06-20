"""Residual Router — damage-aware codec routing for Hydra.

Sits after codec candidate reconstruction. Uses ResidualProfile to decide
whether the chosen codec's damage pattern is acceptable, needs correction,
or requires fallback to a safer head.

The routing loop becomes:
  1. Ghost pre-route (encoded-domain, skip/probe)
  2. Hydra route (probe-based head selection)
  3. Codec candidate reconstruction
  4. Residual Router checks damage shape → accept / correct / fallback

This is the "post-hoc verification" half of the routing contract.
Ghost Bridge is the "pre-screening" half. Together they close the loop.

Lineage:
  residual_contract.py: ResidualProfile, DamageType, residual_routing_signal
  sidecar probe: DEAD for affine (distributed error), alive for VQ outliers
  residual_structure_gate.py: GREEN_LIGHT (spectral 201x, ACF 0.157, rank 41%)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional

from helix_substrate.residual_contract import (
    DamageType,
    ResidualProfile,
    residual_routing_signal,
)
from helix_substrate.hydra_router import Head


# ═══════════════════════════════════════════════════════════════════════════
# Correction types
# ═══════════════════════════════════════════════════════════════════════════

class CorrectionType(str, Enum):
    NONE = "none"                          # residual is acceptable
    OUTLIER_SIDECAR = "outlier_sidecar"    # sparse outlier repair
    LOW_RANK_SIDECAR = "low_rank_sidecar"  # low-rank residual correction
    STRUCTURE_FALLBACK = "structure_fallback"  # fall back to safer head
    PROBE_REQUIRED = "probe_required"      # need more information


# ═══════════════════════════════════════════════════════════════════════════
# Routing decision
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResidualRouteDecision:
    """Post-reconstruction routing decision from residual analysis."""
    tensor_name: str
    selected_codec: str              # codec name that was evaluated
    selected_head: Head              # Hydra head for this codec
    damage_type: DamageType
    correction_type: CorrectionType
    fallback_required: bool          # True if selected codec is rejected
    fallback_head: Optional[Head]    # head to fall back to (if fallback_required)
    confidence: float                # 0-1
    reason: list[str]
    residual_summary: dict           # subset of ResidualProfile for receipt

    def to_dict(self) -> dict:
        d = asdict(self)
        d["selected_head"] = self.selected_head.value
        d["damage_type"] = self.damage_type.value
        d["correction_type"] = self.correction_type.value
        d["fallback_head"] = self.fallback_head.value if self.fallback_head else None
        return d


# ═══════════════════════════════════════════════════════════════════════════
# Decision logic
# ═══════════════════════════════════════════════════════════════════════════

# Minimum confidence to act on residual signal (below this → accept codec as-is)
_MIN_CONFIDENCE = 0.6

# Structure score above which fallback is triggered
_FALLBACK_THRESHOLD = 0.6

# Head safety ordering (safer = higher quality, more bits)
_HEAD_SAFETY_ORDER = [
    Head.EXACT, Head.AFFINE6, Head.AFFINE5, Head.AFFINE4, Head.AFFINE3,
]


def _safer_head(current: Head) -> Optional[Head]:
    """Return the next safer (higher quality) head, or None if already safest."""
    try:
        idx = _HEAD_SAFETY_ORDER.index(current)
    except ValueError:
        return Head.AFFINE6  # unknown head → fall back to affine6
    if idx == 0:
        return None  # already EXACT
    return _HEAD_SAFETY_ORDER[idx - 1]


def decide_from_residual(
    tensor_name: str,
    codec_name: str,
    head: Head,
    residual_profile: ResidualProfile,
) -> ResidualRouteDecision:
    """Make a routing decision from a single codec's residual profile.

    Args:
        tensor_name: name of the tensor
        codec_name: codec that produced the reconstruction
        head: Hydra head that was used
        residual_profile: profiled E = W - W_hat

    Returns:
        ResidualRouteDecision
    """
    signal = residual_routing_signal(residual_profile)
    damage = residual_profile.damage_type
    confidence = signal["confidence"]
    reasons = []

    # Low confidence → accept whatever Hydra chose
    if confidence < _MIN_CONFIDENCE:
        return ResidualRouteDecision(
            tensor_name=tensor_name,
            selected_codec=codec_name,
            selected_head=head,
            damage_type=damage,
            correction_type=CorrectionType.NONE,
            fallback_required=False,
            fallback_head=None,
            confidence=confidence,
            reason=["low_residual_confidence", f"confidence={confidence:.2f}"],
            residual_summary=_summary(residual_profile),
        )

    # Codec is optimal → accept
    if signal["codec_optimal"]:
        return ResidualRouteDecision(
            tensor_name=tensor_name,
            selected_codec=codec_name,
            selected_head=head,
            damage_type=damage,
            correction_type=CorrectionType.NONE,
            fallback_required=False,
            fallback_head=None,
            confidence=confidence,
            reason=["codec_optimal", f"structure_score={residual_profile.structure_score:.3f}"],
            residual_summary=_summary(residual_profile),
        )

    # Damage-type-specific routing
    correction = CorrectionType.NONE
    fallback_required = False
    fallback_head = None

    if damage == DamageType.CONCENTRATED:
        correction = CorrectionType.OUTLIER_SIDECAR
        reasons.append("concentrated_damage")
        reasons.append("suggest_outlier_sidecar")

    elif damage == DamageType.LOW_RANK:
        correction = CorrectionType.LOW_RANK_SIDECAR
        reasons.append("low_rank_damage")
        reasons.append("suggest_low_rank_correction")

    elif damage == DamageType.STRUCTURED:
        if residual_profile.structure_score >= _FALLBACK_THRESHOLD:
            correction = CorrectionType.STRUCTURE_FALLBACK
            fallback_required = True
            fallback_head = _safer_head(head)
            reasons.append("high_structure_score")
            reasons.append(f"structure_score={residual_profile.structure_score:.3f}")
            reasons.append(f"fallback_to={fallback_head.value if fallback_head else 'exact'}")
        else:
            correction = CorrectionType.PROBE_REQUIRED
            reasons.append("moderate_structure")
            reasons.append("probe_for_better_codec")

    elif damage == DamageType.DISTRIBUTED:
        # Distributed but not optimal (score between 0.2 and threshold)
        reasons.append("distributed_acceptable")

    if not reasons:
        reasons.append("accepted")

    return ResidualRouteDecision(
        tensor_name=tensor_name,
        selected_codec=codec_name,
        selected_head=head,
        damage_type=damage,
        correction_type=correction,
        fallback_required=fallback_required,
        fallback_head=fallback_head,
        confidence=confidence,
        reason=reasons,
        residual_summary=_summary(residual_profile),
    )


def decide_from_candidates(
    tensor_name: str,
    candidates: dict[str, tuple[Head, ResidualProfile]],
    policy: str = "edge_balanced",
) -> ResidualRouteDecision:
    """Choose the best codec from multiple candidates using residual quality.

    Args:
        tensor_name: tensor being routed
        candidates: {codec_name: (head, residual_profile)}
        policy: routing policy (affects tiebreaking)

    Returns:
        ResidualRouteDecision for the best candidate
    """
    if not candidates:
        return ResidualRouteDecision(
            tensor_name=tensor_name,
            selected_codec="",
            selected_head=Head.EXACT,
            damage_type=DamageType.UNKNOWN,
            correction_type=CorrectionType.PROBE_REQUIRED,
            fallback_required=True,
            fallback_head=Head.EXACT,
            confidence=0.0,
            reason=["no_candidates"],
            residual_summary={},
        )

    # Rank candidates by: structure_score (lower = better), then cosine (higher = better)
    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            item[1][1].structure_score,
            -item[1][1].cosine,
        ),
    )

    best_name, (best_head, best_profile) = ranked[0]
    decision = decide_from_residual(tensor_name, best_name, best_head, best_profile)

    if len(ranked) > 1:
        decision.reason.append(f"best_of_{len(ranked)}_candidates")

    return decision


def _summary(p: ResidualProfile) -> dict:
    """Compact summary for receipts."""
    return {
        "cosine": p.cosine,
        "rms_error": p.rms_error,
        "structure_score": p.structure_score,
        "damage_type": p.damage_type.value,
        "kurtosis": p.kurtosis,
        "spectral_ratio": p.spectral_ratio,
        "svd_rank_ratio": p.svd_rank_ratio,
    }
