"""Gauge-Only Routing ablation.

Tests whether compression-induced routing signals work without semantic labels,
tensor names, architecture names, or human-readable roles.

The gauge-only router operates on numeric features ONLY:
  - Ghost: te, tr, mo, ac
  - Residual: rms_error, cosine, kurtosis, sparsity, acf_lag1, acf_lag10,
              spectral_ratio, svd_rank_ratio, top10_explained,
              channel_concentration, structure_score
  - confidence scores

It may NOT use: tensor_name, model_name, model_family, layer_index,
tensor_role, architecture_label, or words from parameter names.

"The routing layer is intentionally non-semantic. It operates on gauges, not words."

Usage:
    python tools/sweep_gauge_only_routing.py [--output PATH]

Requires: numpy. Optional: safetensors (for real model tensors).
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
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_substrate.residual_contract import (
    DamageType,
    ResidualProfile,
    profile_residual,
    residual_routing_signal,
)
from helix_substrate.ghost_bridge import ghost_features_from_bytes
from helix_substrate.residual_router import (
    CorrectionType,
    ResidualRouteDecision,
    decide_from_residual,
)
from helix_substrate.hydra_router import Head

# Import sweep codecs and generators
from tools.sweep_compression_routing_signal import (
    CODEC_REGISTRY,
    SYNTHETIC_GENERATORS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Gauge vector: the ONLY input to the blind router
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GaugeVector:
    """Numeric-only feature vector for blind routing.

    Contains NO semantic information: no names, no labels, no words.
    Only measured gauge readings from compression artifacts.
    """
    # Residual gauges (from E = W - W_hat)
    rms_error: float = 0.0
    cosine: float = 1.0
    kurtosis: float = 0.0
    sparsity: float = 0.0
    acf_lag1: float = 0.0
    acf_lag10: float = 0.0
    spectral_ratio: float = 1.0
    svd_rank_ratio: float = 1.0
    top10_explained: float = 0.0
    channel_concentration: float = 1.0
    structure_score: float = 0.0

    # Ghost gauges (from encoded bytes, pre-decompression)
    ghost_te: float = 0.0   # transition entropy
    ghost_tr: float = 0.0   # transition rank
    ghost_mo: float = 0.0   # markov order
    ghost_ac: float = 0.0   # index autocorrelation

    # Derived
    confidence: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numeric array for computation."""
        return np.array([
            self.rms_error, self.cosine, self.kurtosis, self.sparsity,
            self.acf_lag1, self.acf_lag10, self.spectral_ratio,
            self.svd_rank_ratio, self.top10_explained,
            self.channel_concentration, self.structure_score,
            self.ghost_te, self.ghost_tr, self.ghost_mo, self.ghost_ac,
            self.confidence,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_residual_profile(cls, rp: ResidualProfile,
                              ghost: dict = None) -> "GaugeVector":
        """Build gauge vector from residual profile + optional ghost."""
        signal = residual_routing_signal(rp)
        g = cls(
            rms_error=rp.rms_error,
            cosine=rp.cosine,
            kurtosis=rp.kurtosis,
            sparsity=rp.sparsity,
            acf_lag1=rp.acf_lag1,
            acf_lag10=rp.acf_lag10,
            spectral_ratio=rp.spectral_ratio,
            svd_rank_ratio=rp.svd_rank_ratio,
            top10_explained=rp.top10_explained,
            channel_concentration=rp.channel_concentration,
            structure_score=rp.structure_score,
            confidence=signal["confidence"],
        )
        if ghost:
            g.ghost_te = ghost.get("te", 0.0)
            g.ghost_tr = ghost.get("tr", 0.0)
            g.ghost_mo = ghost.get("mo", 0.0)
            g.ghost_ac = ghost.get("ac", 0.0)
        return g


# ═══════════════════════════════════════════════════════════════════════════
# Gauge-only router: operates ONLY on GaugeVector, no metadata
# ═══════════════════════════════════════════════════════════════════════════

class GaugeAction(str, Enum):
    ACCEPT = "accept"                    # codec is good
    CORRECTION_OUTLIER = "correction_outlier"    # suggest outlier sidecar
    CORRECTION_LOWRANK = "correction_lowrank"    # suggest low-rank sidecar
    FALLBACK = "fallback"                # fall back to safer codec
    PROBE = "probe"                      # need more information


@dataclass
class GaugeRouteDecision:
    """Route decision from gauge readings alone. No metadata."""
    blinded_id: str          # opaque hash, not a name
    gauge_vector: dict       # the gauges that drove the decision
    action: GaugeAction
    confidence: float
    fallback_required: bool
    reason: list[str]        # numeric reasons only
    receipt_hash: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


# Gauge thresholds — learned from proven Phase 0.17b + residual router
_GAUGE_CONFIDENCE_MIN = 0.6
_GAUGE_STRUCTURE_ACCEPT = 0.2       # below → accept
_GAUGE_STRUCTURE_CORRECT = 0.5      # above → try correction
_GAUGE_STRUCTURE_FALLBACK = 0.6     # above → fallback
_GAUGE_KURTOSIS_OUTLIER = 6.0       # above → outlier pattern
_GAUGE_CONC_OUTLIER = 3.0           # above → concentrated damage
_GAUGE_SVD_LOWRANK = 0.5            # below → low-rank pattern
_GAUGE_SPECTRAL_STRUCTURED = 20.0   # above → spectral structure
_GAUGE_ACF10_STRUCTURED = 0.1       # above → autocorrelation structure


def gauge_route(gauge: GaugeVector) -> GaugeRouteDecision:
    """Route from gauge readings alone. No names, no labels, no words.

    This is the blind router. It watches pressure, heat, vibration, flow.
    Then it opens the right valve.

    Args:
        gauge: GaugeVector with numeric features only

    Returns:
        GaugeRouteDecision
    """
    reasons = []

    # Generate blinded ID from gauge values (deterministic, not from names)
    gauge_bytes = gauge.to_array().tobytes()
    blinded_id = hashlib.sha256(gauge_bytes).hexdigest()[:12]

    # Low confidence → accept (conservative)
    if gauge.confidence < _GAUGE_CONFIDENCE_MIN:
        return GaugeRouteDecision(
            blinded_id=blinded_id,
            gauge_vector=gauge.to_dict(),
            action=GaugeAction.ACCEPT,
            confidence=gauge.confidence,
            fallback_required=False,
            reason=["low_confidence", f"conf={gauge.confidence:.2f}"],
            receipt_hash=hashlib.sha256(
                f"gauge:{blinded_id}:accept".encode()
            ).hexdigest()[:16],
        )

    # Codec is good → accept
    if gauge.structure_score < _GAUGE_STRUCTURE_ACCEPT:
        return GaugeRouteDecision(
            blinded_id=blinded_id,
            gauge_vector=gauge.to_dict(),
            action=GaugeAction.ACCEPT,
            confidence=gauge.confidence,
            fallback_required=False,
            reason=["gauges_nominal", f"structure={gauge.structure_score:.3f}"],
            receipt_hash=hashlib.sha256(
                f"gauge:{blinded_id}:accept_nominal".encode()
            ).hexdigest()[:16],
        )

    # Damage pattern detection from gauges
    action = GaugeAction.ACCEPT
    fallback = False

    # Check outlier pattern: high kurtosis + concentrated channels
    if (gauge.kurtosis > _GAUGE_KURTOSIS_OUTLIER and
            gauge.channel_concentration > _GAUGE_CONC_OUTLIER):
        action = GaugeAction.CORRECTION_OUTLIER
        reasons.extend([
            "gauge_kurtosis_spike",
            f"kurtosis={gauge.kurtosis:.1f}",
            f"concentration={gauge.channel_concentration:.1f}",
        ])

    # Check low-rank pattern: low SVD rank ratio
    elif gauge.svd_rank_ratio < _GAUGE_SVD_LOWRANK:
        action = GaugeAction.CORRECTION_LOWRANK
        reasons.extend([
            "gauge_rank_deficit",
            f"svd_rank={gauge.svd_rank_ratio:.3f}",
        ])

    # Check structured pattern: spectral or autocorrelation
    elif (gauge.spectral_ratio > _GAUGE_SPECTRAL_STRUCTURED or
          abs(gauge.acf_lag10) > _GAUGE_ACF10_STRUCTURED):
        if gauge.structure_score >= _GAUGE_STRUCTURE_FALLBACK:
            action = GaugeAction.FALLBACK
            fallback = True
            reasons.extend([
                "gauge_structure_alarm",
                f"spectral={gauge.spectral_ratio:.1f}",
                f"acf10={gauge.acf_lag10:.4f}",
                f"structure={gauge.structure_score:.3f}",
            ])
        elif gauge.structure_score >= _GAUGE_STRUCTURE_CORRECT:
            action = GaugeAction.PROBE
            reasons.extend([
                "gauge_structure_elevated",
                f"structure={gauge.structure_score:.3f}",
            ])

    # Moderate structure without clear pattern
    elif gauge.structure_score >= _GAUGE_STRUCTURE_CORRECT:
        action = GaugeAction.PROBE
        reasons.extend([
            "gauge_structure_moderate",
            f"structure={gauge.structure_score:.3f}",
        ])

    if not reasons:
        reasons.append("gauges_within_tolerance")

    receipt_hash = hashlib.sha256(
        f"gauge:{blinded_id}:{action.value}".encode()
    ).hexdigest()[:16]

    return GaugeRouteDecision(
        blinded_id=blinded_id,
        gauge_vector=gauge.to_dict(),
        action=action,
        confidence=gauge.confidence,
        fallback_required=fallback,
        reason=reasons,
        receipt_hash=receipt_hash,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Comparison engine: gauge-only vs full router
# ═══════════════════════════════════════════════════════════════════════════

def _full_router_action(rp: ResidualProfile) -> str:
    """Get the full router's action for comparison."""
    decision = decide_from_residual("_blind_", "_blind_", Head.AFFINE5, rp)
    return decision.correction_type.value


def _gauge_action_to_correction(action: GaugeAction) -> str:
    """Map gauge action to CorrectionType string for comparison."""
    mapping = {
        GaugeAction.ACCEPT: "none",
        GaugeAction.CORRECTION_OUTLIER: "outlier_sidecar",
        GaugeAction.CORRECTION_LOWRANK: "low_rank_sidecar",
        GaugeAction.FALLBACK: "structure_fallback",
        GaugeAction.PROBE: "probe_required",
    }
    return mapping[action]


@dataclass
class ComparisonRecord:
    """Comparison between gauge-only and full router."""
    blinded_id: str
    model_family: str         # for analysis only (not used in routing)
    tensor_role: str          # for analysis only
    codec_name: str
    gauge_action: str
    full_action: str
    agree: bool
    gauge_fallback: bool
    full_fallback: bool
    missed_fallback: bool     # full says fallback, gauge says accept
    false_fallback: bool      # gauge says fallback, full says accept

    def to_dict(self) -> dict:
        return asdict(self)


def run_comparison(
    tensors: list[tuple[str, np.ndarray, dict]],
    codecs: dict = None,
    output_path: Path = None,
) -> tuple[list[ComparisonRecord], dict]:
    """Compare gauge-only routing vs full routing.

    Returns:
        (records, summary)
    """
    if codecs is None:
        codecs = CODEC_REGISTRY

    records = []
    t_start = time.time()

    for tensor_name, W, meta in tensors:
        for codec_name, codec_fn in codecs.items():
            if codec_name == "exact":
                continue  # skip exact (zero error, not interesting)

            try:
                W_hat, codec_meta = codec_fn(W)
            except Exception:
                continue

            rp = profile_residual(W, W_hat)

            # Ghost features
            ghost = None
            encoded_bytes = codec_meta.get("encoded_bytes")
            if encoded_bytes and len(encoded_bytes) >= 64:
                ghost = ghost_features_from_bytes(
                    encoded_bytes, shape=W.shape if W.ndim == 2 else ()
                )

            # Gauge-only route (blind)
            gauge = GaugeVector.from_residual_profile(rp, ghost)
            gauge_decision = gauge_route(gauge)

            # Full route (has access to names, but we use dummy names)
            full_action = _full_router_action(rp)

            gauge_act = _gauge_action_to_correction(gauge_decision.action)
            agree = gauge_act == full_action

            full_decision = decide_from_residual("_", "_", Head.AFFINE5, rp)

            records.append(ComparisonRecord(
                blinded_id=gauge_decision.blinded_id,
                model_family=meta.get("family", "unknown"),
                tensor_role=meta.get("role", "unknown"),
                codec_name=codec_name,
                gauge_action=gauge_act,
                full_action=full_action,
                agree=agree,
                gauge_fallback=gauge_decision.fallback_required,
                full_fallback=full_decision.fallback_required,
                missed_fallback=(full_decision.fallback_required and
                                 not gauge_decision.fallback_required),
                false_fallback=(gauge_decision.fallback_required and
                                not full_decision.fallback_required),
            ))

    wall_time = time.time() - t_start

    summary = analyze_comparison(records)
    summary["cost"] = {
        "wall_time_s": round(wall_time, 3),
        "n_comparisons": len(records),
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2))

    return records, summary


def analyze_comparison(records: list[ComparisonRecord]) -> dict:
    """Analyze gauge-only vs full router agreement."""
    if not records:
        return {"error": "no_records"}

    n = len(records)
    n_agree = sum(1 for r in records if r.agree)
    n_missed = sum(1 for r in records if r.missed_fallback)
    n_false = sum(1 for r in records if r.false_fallback)

    # Per-family agreement
    family_agreement = {}
    for r in records:
        family_agreement.setdefault(r.model_family, {"agree": 0, "total": 0})
        family_agreement[r.model_family]["total"] += 1
        if r.agree:
            family_agreement[r.model_family]["agree"] += 1

    for fam in family_agreement:
        info = family_agreement[fam]
        info["rate"] = round(info["agree"] / info["total"], 3) if info["total"] > 0 else 0

    # Per-codec agreement
    codec_agreement = {}
    for r in records:
        codec_agreement.setdefault(r.codec_name, {"agree": 0, "total": 0})
        codec_agreement[r.codec_name]["total"] += 1
        if r.agree:
            codec_agreement[r.codec_name]["agree"] += 1

    for codec in codec_agreement:
        info = codec_agreement[codec]
        info["rate"] = round(info["agree"] / info["total"], 3) if info["total"] > 0 else 0

    return {
        "n_comparisons": n,
        "agreement_rate": round(n_agree / n, 3) if n > 0 else 0,
        "missed_fallback_rate": round(n_missed / n, 3) if n > 0 else 0,
        "false_fallback_rate": round(n_false / n, 3) if n > 0 else 0,
        "agreement_by_family": family_agreement,
        "agreement_by_codec": codec_agreement,
        "verdict": (
            "gauge_only_sufficient"
            if n_agree / n >= 0.85 and n_missed == 0
            else "gauge_only_partial" if n_agree / n >= 0.7
            else "gauge_only_insufficient"
        ) if n > 0 else "no_data",
    }


# ═══════════════════════════════════════════════════════════════════════════
# JSONL adapter: consume real sweep receipts without forbidden metadata
# ═══════════════════════════════════════════════════════════════════════════

# Fields that the gauge-only router is FORBIDDEN from using
_FORBIDDEN_FIELDS = frozenset([
    "tensor_id", "tensor_name", "model_name", "model_family",
    "tensor_role", "layer_index", "architecture",
])


def gauge_vector_from_sweep_record(record: dict) -> GaugeVector:
    """Build GaugeVector from a sweep JSONL record, stripping forbidden metadata.

    The gauge-only router may use ONLY numeric residual/ghost fields.
    tensor_id, model_family, tensor_role are stripped before routing.
    """
    rp = record.get("residual_profile", {})
    ghost = record.get("ghost_features")
    route = record.get("route_signal", {})

    return GaugeVector(
        rms_error=rp.get("rms_error", 0.0),
        cosine=rp.get("cosine", 1.0),
        kurtosis=rp.get("kurtosis", 0.0),
        sparsity=rp.get("sparsity", 0.0),
        acf_lag1=rp.get("acf_lag1", 0.0),
        acf_lag10=rp.get("acf_lag10", 0.0),
        spectral_ratio=rp.get("spectral_ratio", 1.0),
        svd_rank_ratio=rp.get("svd_rank_ratio", 1.0),
        top10_explained=rp.get("top10_explained", 0.0),
        channel_concentration=rp.get("channel_concentration", 1.0),
        structure_score=rp.get("structure_score", 0.0),
        ghost_te=ghost.get("te", 0.0) if ghost else 0.0,
        ghost_tr=ghost.get("tr", 0.0) if ghost else 0.0,
        ghost_mo=ghost.get("mo", 0.0) if ghost else 0.0,
        ghost_ac=ghost.get("ac", 0.0) if ghost else 0.0,
        confidence=route.get("confidence", 0.0),
    )


def run_comparison_from_jsonl(
    jsonl_path: Path,
    output_path: Path = None,
) -> tuple[list[ComparisonRecord], dict]:
    """Run gauge-only vs full-router comparison on sweep JSONL receipts.

    Loads records from a compression signal sweep JSONL file, strips
    forbidden metadata, builds GaugeVectors from numeric fields only,
    and compares gauge-only routing against the full router's decision.

    Args:
        jsonl_path: Path to compression_signal_sweep.*.jsonl
        output_path: If set, write comparison JSONL + summary

    Returns:
        (records, summary)
    """
    with open(jsonl_path) as f:
        sweep_records = [json.loads(line) for line in f if line.strip()]

    records = []
    t_start = time.time()

    for sr in sweep_records:
        # Skip exact codec (zero error, not useful for routing comparison)
        if sr.get("codec_name") == "exact":
            continue

        # Build gauge vector from numeric fields ONLY (no forbidden metadata)
        gauge = gauge_vector_from_sweep_record(sr)
        gauge_decision = gauge_route(gauge)

        # Build ResidualProfile for full router comparison
        rp_dict = sr.get("residual_profile", {})
        rp = ResidualProfile(
            rms_error=rp_dict.get("rms_error", 0.0),
            cosine=rp_dict.get("cosine", 1.0),
            max_abs_error=rp_dict.get("max_abs_error", 0.0),
            mean_abs_error=rp_dict.get("mean_abs_error", 0.0),
            kurtosis=rp_dict.get("kurtosis", 0.0),
            sparsity=rp_dict.get("sparsity", 0.0),
            acf_lag1=rp_dict.get("acf_lag1", 0.0),
            acf_lag10=rp_dict.get("acf_lag10", 0.0),
            spectral_ratio=rp_dict.get("spectral_ratio", 1.0),
            svd_rank_ratio=rp_dict.get("svd_rank_ratio", 1.0),
            top10_explained=rp_dict.get("top10_explained", 0.0),
            channel_concentration=rp_dict.get("channel_concentration", 1.0),
            structure_score=rp_dict.get("structure_score", 0.0),
            damage_type=DamageType(rp_dict.get("damage_type", "distributed")),
        )
        full_action = _full_router_action(rp)
        full_decision = decide_from_residual("_", "_", Head.AFFINE5, rp)

        gauge_act = _gauge_action_to_correction(gauge_decision.action)
        agree = gauge_act == full_action

        # Preserve family/role for analysis only (NOT used in routing)
        records.append(ComparisonRecord(
            blinded_id=gauge_decision.blinded_id,
            model_family=sr.get("model_family", "unknown"),
            tensor_role=sr.get("tensor_role", "unknown"),
            codec_name=sr.get("codec_name", "unknown"),
            gauge_action=gauge_act,
            full_action=full_action,
            agree=agree,
            gauge_fallback=gauge_decision.fallback_required,
            full_fallback=full_decision.fallback_required,
            missed_fallback=(full_decision.fallback_required and
                             not gauge_decision.fallback_required),
            false_fallback=(gauge_decision.fallback_required and
                            not full_decision.fallback_required),
        ))

    wall_time = time.time() - t_start

    summary = analyze_comparison(records)
    summary["cost"] = {
        "wall_time_s": round(wall_time, 3),
        "n_comparisons": len(records),
        "source": str(jsonl_path),
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2))

    return records, summary


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gauge-only routing ablation")
    parser.add_argument("--output", type=Path,
                        default=Path("receipts/gauge_only_routing.jsonl"))
    parser.add_argument("--from-jsonl", type=Path, default=None,
                        help="Run on sweep JSONL receipts instead of synthetic tensors")
    args = parser.parse_args()

    if args.from_jsonl:
        print(f"Loading sweep receipts from {args.from_jsonl}...")
        records, summary = run_comparison_from_jsonl(
            args.from_jsonl, output_path=args.output
        )
    else:
        # Build synthetic tensors
        rng = np.random.RandomState(42)
        tensors = []
        for gen_name, gen_fn in SYNTHETIC_GENERATORS.items():
            W, meta = gen_fn(rng)
            tensors.append((f"synthetic/{gen_name}", W, meta))

        print(f"Comparing gauge-only vs full routing on {len(tensors)} tensors "
              f"x {len(CODEC_REGISTRY) - 1} lossy codecs...")

        records, summary = run_comparison(tensors, output_path=args.output)

    print(f"\nResults:")
    print(f"  Agreement rate: {summary['agreement_rate']:.1%}")
    print(f"  Missed fallback rate: {summary['missed_fallback_rate']:.1%}")
    print(f"  False fallback rate: {summary['false_fallback_rate']:.1%}")
    print(f"  Verdict: {summary['verdict']}")
    print(f"\n  Per-family:")
    for fam, info in summary.get("agreement_by_family", {}).items():
        print(f"    {fam}: {info['rate']:.1%} ({info['agree']}/{info['total']})")
    print(f"\nReceipts: {args.output}")


if __name__ == "__main__":
    main()
