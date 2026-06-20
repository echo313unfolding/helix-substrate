"""Tests for Residual Router — damage-aware codec routing."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_substrate.residual_contract import (
    DamageType,
    ResidualProfile,
    profile_residual,
)
from helix_substrate.residual_router import (
    CorrectionType,
    ResidualRouteDecision,
    decide_from_residual,
    decide_from_candidates,
)
from helix_substrate.hydra_router import (
    Head,
    HydraRouter,
    TensorProfile,
)


# ═══════════════════════════════════════════════════════════════════════════
# Core decision logic
# ═══════════════════════════════════════════════════════════════════════════

class TestDecideFromResidual:
    def test_distributed_residual_accepted(self):
        """Distributed damage → NONE correction, no fallback."""
        rp = ResidualProfile(
            structure_score=0.05,
            damage_type=DamageType.DISTRIBUTED,
            cosine=0.999,
            rms_error=0.01,
        )
        d = decide_from_residual("t1", "affine5", Head.AFFINE5, rp)
        assert d.correction_type == CorrectionType.NONE
        assert d.fallback_required is False
        assert "codec_optimal" in d.reason

    def test_concentrated_residual_suggests_sidecar(self):
        """Concentrated damage → OUTLIER_SIDECAR correction."""
        rp = ResidualProfile(
            structure_score=0.6,
            damage_type=DamageType.CONCENTRATED,
            cosine=0.998,
            rms_error=0.02,
            kurtosis=8.0,
            channel_concentration=4.0,
        )
        d = decide_from_residual("t2", "affine5", Head.AFFINE5, rp)
        assert d.correction_type == CorrectionType.OUTLIER_SIDECAR
        assert "concentrated_damage" in d.reason

    def test_low_rank_residual_suggests_correction(self):
        """Low-rank damage → LOW_RANK_SIDECAR correction."""
        rp = ResidualProfile(
            structure_score=0.7,
            damage_type=DamageType.LOW_RANK,
            cosine=0.997,
            rms_error=0.03,
            svd_rank_ratio=0.1,
        )
        d = decide_from_residual("t3", "affine5", Head.AFFINE5, rp)
        assert d.correction_type == CorrectionType.LOW_RANK_SIDECAR
        assert "low_rank_damage" in d.reason

    def test_structured_high_score_triggers_fallback(self):
        """Structured damage with high score → STRUCTURE_FALLBACK + fallback head."""
        rp = ResidualProfile(
            structure_score=0.75,
            damage_type=DamageType.STRUCTURED,
            cosine=0.996,
            rms_error=0.04,
            spectral_ratio=100.0,
        )
        d = decide_from_residual("t4", "affine5", Head.AFFINE5, rp)
        assert d.correction_type == CorrectionType.STRUCTURE_FALLBACK
        assert d.fallback_required is True
        assert d.fallback_head == Head.AFFINE6  # one step safer than AFFINE5

    def test_structured_moderate_score_probes(self):
        """Structured damage with moderate score (above confidence threshold) → PROBE_REQUIRED."""
        rp = ResidualProfile(
            structure_score=0.55,  # above 0.5 → confidence=0.7 → above MIN_CONFIDENCE
            damage_type=DamageType.STRUCTURED,
            cosine=0.998,
            rms_error=0.02,
            spectral_ratio=30.0,
        )
        d = decide_from_residual("t5", "affine5", Head.AFFINE5, rp)
        assert d.correction_type == CorrectionType.PROBE_REQUIRED
        assert "moderate_structure" in d.reason

    def test_low_confidence_accepts_codec(self):
        """Low residual confidence → accept codec as-is regardless of damage type."""
        rp = ResidualProfile(
            structure_score=0.35,  # ambiguous zone → low confidence
            damage_type=DamageType.DISTRIBUTED,
            cosine=0.998,
        )
        d = decide_from_residual("t6", "affine5", Head.AFFINE5, rp)
        assert d.correction_type == CorrectionType.NONE
        assert "low_residual_confidence" in d.reason

    def test_exact_head_no_safer_fallback(self):
        """EXACT head with structured residual → fallback_head is None (already safest)."""
        rp = ResidualProfile(
            structure_score=0.8,
            damage_type=DamageType.STRUCTURED,
            cosine=0.990,
        )
        d = decide_from_residual("t7", "exact", Head.EXACT, rp)
        assert d.fallback_head is None

    def test_affine4_falls_back_to_affine5(self):
        """AFFINE4 with structured residual → fallback to AFFINE5."""
        rp = ResidualProfile(
            structure_score=0.7,
            damage_type=DamageType.STRUCTURED,
            cosine=0.995,
        )
        d = decide_from_residual("t8", "affine4", Head.AFFINE4, rp)
        assert d.fallback_required is True
        assert d.fallback_head == Head.AFFINE5

    def test_deterministic(self):
        """Same inputs produce same decision."""
        rp = ResidualProfile(
            structure_score=0.6, damage_type=DamageType.CONCENTRATED,
            cosine=0.998, kurtosis=8.0, channel_concentration=4.0,
        )
        d1 = decide_from_residual("t", "a5", Head.AFFINE5, rp)
        d2 = decide_from_residual("t", "a5", Head.AFFINE5, rp)
        assert d1.to_dict() == d2.to_dict()


# ═══════════════════════════════════════════════════════════════════════════
# Multi-candidate decision
# ═══════════════════════════════════════════════════════════════════════════

class TestDecideFromCandidates:
    def test_chooses_lowest_structure_score(self):
        """Best candidate = lowest structure_score."""
        candidates = {
            "codec_bad": (Head.AFFINE5, ResidualProfile(structure_score=0.6, damage_type=DamageType.STRUCTURED)),
            "codec_good": (Head.AFFINE5, ResidualProfile(structure_score=0.1, damage_type=DamageType.DISTRIBUTED)),
        }
        d = decide_from_candidates("t1", candidates)
        assert d.selected_codec == "codec_good"

    def test_empty_candidates(self):
        """No candidates → PROBE_REQUIRED fallback."""
        d = decide_from_candidates("t1", {})
        assert d.correction_type == CorrectionType.PROBE_REQUIRED
        assert d.fallback_required is True

    def test_single_candidate(self):
        """Single candidate is evaluated normally."""
        candidates = {
            "only": (Head.AFFINE6, ResidualProfile(structure_score=0.05, damage_type=DamageType.DISTRIBUTED)),
        }
        d = decide_from_candidates("t1", candidates)
        assert d.selected_codec == "only"
        assert d.correction_type == CorrectionType.NONE


# ═══════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_decision_to_dict(self):
        """ResidualRouteDecision serializes to JSON."""
        d = decide_from_residual(
            "t1", "a5", Head.AFFINE5,
            ResidualProfile(structure_score=0.1, damage_type=DamageType.DISTRIBUTED),
        )
        dct = d.to_dict()
        text = json.dumps(dct)
        assert "selected_head" in text
        assert "affine5" in text

    def test_fallback_decision_serializable(self):
        """Fallback decision with fallback_head serializes."""
        d = decide_from_residual(
            "t1", "a5", Head.AFFINE5,
            ResidualProfile(structure_score=0.8, damage_type=DamageType.STRUCTURED),
        )
        dct = d.to_dict()
        text = json.dumps(dct)
        assert "affine6" in text  # fallback head

    def test_none_fallback_serializable(self):
        """Decision with no fallback head serializes."""
        d = decide_from_residual(
            "t1", "exact", Head.EXACT,
            ResidualProfile(structure_score=0.8, damage_type=DamageType.STRUCTURED),
        )
        dct = d.to_dict()
        assert dct["fallback_head"] is None
        text = json.dumps(dct)
        assert "null" in text


# ═══════════════════════════════════════════════════════════════════════════
# Hydra integration: route_with_residuals
# ═══════════════════════════════════════════════════════════════════════════

class TestHydraResidualIntegration:
    def _make_profile(self, name, ttype="gate_proj", cosine=0.999):
        return TensorProfile(
            tensor_name=name, shape=(256, 512), layer_index=5,
            tensor_type=ttype, n_params=256 * 512,
            affine5_cosine=cosine, affine6_cosine=cosine + 0.0005,
        )

    def test_exact_tensors_unmodified(self):
        """Exact tensors pass through regardless of residual data."""
        router = HydraRouter(policy="edge_balanced")
        profiles = [
            TensorProfile(
                tensor_name="model.embed_tokens.weight", shape=(32000, 768),
                layer_index=0, tensor_type="embed_tokens", n_params=32000 * 768,
            ),
        ]
        rp_map = {
            "model.embed_tokens.weight": ResidualProfile(
                structure_score=0.9, damage_type=DamageType.STRUCTURED,
            ),
        }
        plan, decisions = router.route_with_residuals(profiles, rp_map)
        assert plan.tensors[0].head == Head.EXACT
        assert len(decisions) == 0  # exact tensors skip residual check

    def test_distributed_residual_keeps_plan(self):
        """Distributed residual doesn't change Hydra's original routing."""
        router = HydraRouter(policy="edge_balanced")
        profiles = [self._make_profile("layer.5.gate_proj.weight")]
        rp_map = {
            "layer.5.gate_proj.weight": ResidualProfile(
                structure_score=0.05, damage_type=DamageType.DISTRIBUTED,
            ),
        }
        plan, decisions = router.route_with_residuals(profiles, rp_map)
        assert plan.tensors[0].head == Head.AFFINE5
        assert len(decisions) == 1
        assert decisions[0].correction_type == CorrectionType.NONE

    def test_structured_residual_triggers_fallback(self):
        """Structured high-score residual upgrades head."""
        router = HydraRouter(policy="edge_balanced")
        profiles = [self._make_profile("layer.5.gate_proj.weight")]
        rp_map = {
            "layer.5.gate_proj.weight": ResidualProfile(
                structure_score=0.8, damage_type=DamageType.STRUCTURED,
            ),
        }
        plan, decisions = router.route_with_residuals(profiles, rp_map)
        # Original route would be AFFINE5; structured residual should upgrade to AFFINE6
        assert plan.tensors[0].head == Head.AFFINE6
        assert "residual_fallback" in plan.tensors[0].reason

    def test_concentrated_residual_annotates_sidecar(self):
        """Concentrated residual annotates sidecar correction."""
        router = HydraRouter(policy="edge_balanced")
        profiles = [self._make_profile("layer.5.gate_proj.weight")]
        rp_map = {
            "layer.5.gate_proj.weight": ResidualProfile(
                structure_score=0.6, damage_type=DamageType.CONCENTRATED,
                kurtosis=8.0, channel_concentration=4.0,
            ),
        }
        plan, decisions = router.route_with_residuals(profiles, rp_map)
        # Head stays AFFINE5 but annotated with sidecar suggestion
        assert plan.tensors[0].head == Head.AFFINE5
        assert any("outlier_sidecar" in r for r in plan.tensors[0].reason)

    def test_missing_residual_keeps_plan(self):
        """Tensor without residual data passes through unchanged."""
        router = HydraRouter(policy="edge_balanced")
        profiles = [self._make_profile("layer.5.gate_proj.weight")]
        plan, decisions = router.route_with_residuals(profiles, {})
        assert plan.tensors[0].head == Head.AFFINE5
        assert len(decisions) == 0

    def test_multiple_tensors_mixed(self):
        """Multiple tensors with different residual types route correctly."""
        router = HydraRouter(policy="edge_balanced")
        profiles = [
            self._make_profile("layer.5.gate_proj.weight"),
            self._make_profile("layer.5.up_proj.weight"),
            self._make_profile("layer.5.down_proj.weight"),
        ]
        rp_map = {
            "layer.5.gate_proj.weight": ResidualProfile(
                structure_score=0.05, damage_type=DamageType.DISTRIBUTED,
            ),
            "layer.5.up_proj.weight": ResidualProfile(
                structure_score=0.8, damage_type=DamageType.STRUCTURED,
            ),
            # down_proj has no residual data
        }
        plan, decisions = router.route_with_residuals(profiles, rp_map)

        # gate_proj: distributed → stays AFFINE5
        assert plan.tensors[0].head == Head.AFFINE5
        # up_proj: structured → falls back to AFFINE6
        assert plan.tensors[1].head == Head.AFFINE6
        # down_proj: no residual → stays AFFINE5
        assert plan.tensors[2].head == Head.AFFINE5
        assert len(decisions) == 2

    def test_quality_first_policy(self):
        """Quality-first policy with structured residual falls back to EXACT."""
        router = HydraRouter(policy="quality_first")
        profiles = [self._make_profile("layer.5.gate_proj.weight", cosine=0.9995)]
        rp_map = {
            "layer.5.gate_proj.weight": ResidualProfile(
                structure_score=0.8, damage_type=DamageType.STRUCTURED,
            ),
        }
        plan, decisions = router.route_with_residuals(profiles, rp_map)
        # quality_first routes to AFFINE6; structured → fallback to EXACT
        assert plan.tensors[0].head == Head.EXACT


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end with real residual profiling
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_real_random_residual_accepted(self):
        """Synthetic random error → distributed → accepted."""
        rng = np.random.RandomState(42)
        W = rng.randn(128, 256).astype(np.float32)
        W_hat = W + rng.randn(128, 256).astype(np.float32) * 0.001

        rp = profile_residual(W, W_hat)
        d = decide_from_residual("t", "affine5", Head.AFFINE5, rp)
        assert d.fallback_required is False

    def test_real_structured_residual_detected(self):
        """Synthetic correlated error → structured → correction or fallback."""
        rng = np.random.RandomState(42)
        W = rng.randn(128, 256).astype(np.float32)
        raw = rng.randn(128, 256).astype(np.float32) * 0.01
        kernel = np.ones(30) / 30.0
        structured = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'), 1, raw
        )
        W_hat = W + structured

        rp = profile_residual(W, W_hat)
        d = decide_from_residual("t", "affine5", Head.AFFINE5, rp)
        # Should detect structure and suggest correction or fallback
        assert d.correction_type != CorrectionType.NONE

    def test_real_low_rank_residual_detected(self):
        """Synthetic rank-2 error → low-rank damage type correctly identified.

        Note: structure_score ~0.44 puts confidence at 0.4 (ambiguous zone),
        so the router conservatively accepts the codec. This is correct:
        rank structure IS detected, but the router won't act on low confidence.
        """
        rng = np.random.RandomState(42)
        W = rng.randn(128, 256).astype(np.float32)
        u = rng.randn(128, 2).astype(np.float32)
        v = rng.randn(2, 256).astype(np.float32)
        W_hat = W + (u @ v) * 0.01

        rp = profile_residual(W, W_hat)
        assert rp.damage_type == DamageType.LOW_RANK
        assert rp.svd_rank_ratio < 0.1  # correctly identified as low-rank

        d = decide_from_residual("t", "affine5", Head.AFFINE5, rp)
        assert d.damage_type == DamageType.LOW_RANK
        # Conservative: low confidence → accept as-is (correct behavior)
        assert d.correction_type == CorrectionType.NONE
        assert "low_residual_confidence" in d.reason
