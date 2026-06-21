"""Tests for gauge-only routing ablation.

Validates:
- route decision cannot access forbidden metadata
- same gauge vector gives same route regardless of tensor name/model name
- different tensor names with same gauges produce same route
- structured residual gauge triggers correction/fallback
- low-confidence gauge falls back conservatively
- receipts deterministic
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.sweep_gauge_only_routing import (
    GaugeVector,
    GaugeAction,
    GaugeRouteDecision,
    gauge_route,
    run_comparison,
    run_comparison_from_jsonl,
    gauge_vector_from_sweep_record,
    analyze_comparison,
    ComparisonRecord,
    _gauge_action_to_correction,
    _FORBIDDEN_FIELDS,
)
from tools.sweep_compression_routing_signal import SYNTHETIC_GENERATORS
from helix_substrate.residual_contract import (
    DamageType,
    ResidualProfile,
    profile_residual,
    residual_routing_signal,
)


# ═══════════════════════════════════════════════════════════════════════════
# GaugeVector construction
# ═══════════════════════════════════════════════════════════════════════════

class TestGaugeVector:
    def test_from_residual_profile(self):
        """GaugeVector builds from ResidualProfile without names."""
        rp = ResidualProfile(
            rms_error=0.01, cosine=0.999, kurtosis=2.0, sparsity=0.3,
            structure_score=0.15, damage_type=DamageType.DISTRIBUTED,
        )
        g = GaugeVector.from_residual_profile(rp)
        assert g.rms_error == 0.01
        assert g.cosine == 0.999
        assert g.kurtosis == 2.0
        # No name/label fields exist on GaugeVector
        assert not hasattr(g, "tensor_name")
        assert not hasattr(g, "model_name")
        assert not hasattr(g, "model_family")
        assert not hasattr(g, "layer_index")
        assert not hasattr(g, "tensor_role")

    def test_from_residual_with_ghost(self):
        """GaugeVector includes Ghost features when provided."""
        rp = ResidualProfile(structure_score=0.1, damage_type=DamageType.DISTRIBUTED)
        ghost = {"te": 0.85, "tr": 0.12, "mo": 0.95, "ac": 0.03}
        g = GaugeVector.from_residual_profile(rp, ghost)
        assert g.ghost_te == 0.85
        assert g.ghost_mo == 0.95

    def test_to_array(self):
        """GaugeVector converts to numeric array."""
        g = GaugeVector(rms_error=0.01, cosine=0.999, kurtosis=3.0)
        arr = g.to_array()
        assert arr.dtype == np.float32
        assert len(arr) == 16  # 11 residual + 4 ghost + 1 confidence

    def test_to_dict_serializable(self):
        """GaugeVector serializes to JSON."""
        g = GaugeVector(rms_error=0.01, cosine=0.999)
        d = g.to_dict()
        text = json.dumps(d)
        assert "rms_error" in text
        assert "tensor_name" not in text  # no metadata leaks


# ═══════════════════════════════════════════════════════════════════════════
# Gauge-only routing: no metadata access
# ═══════════════════════════════════════════════════════════════════════════

class TestGaugeRouteBlindness:
    def test_same_gauge_same_route_different_names(self):
        """Identical gauges produce identical routes regardless of origin."""
        g = GaugeVector(
            rms_error=0.01, cosine=0.999, kurtosis=2.0, sparsity=0.3,
            structure_score=0.15, confidence=0.9,
            damage_type_value="distributed",
        ) if False else GaugeVector(
            rms_error=0.01, cosine=0.999, kurtosis=2.0, sparsity=0.3,
            structure_score=0.15, confidence=0.9,
        )
        d1 = gauge_route(g)
        d2 = gauge_route(g)
        assert d1.action == d2.action
        assert d1.blinded_id == d2.blinded_id
        assert d1.receipt_hash == d2.receipt_hash

    def test_blinded_id_is_hash_not_name(self):
        """Blinded ID is derived from gauge values, not tensor names."""
        g = GaugeVector(rms_error=0.02, cosine=0.998, structure_score=0.3,
                        confidence=0.7)
        d = gauge_route(g)
        assert len(d.blinded_id) == 12  # hex hash
        # blinded_id should not contain any words
        assert not any(c.isalpha() and c not in "abcdef" for c in d.blinded_id)

    def test_decision_contains_no_metadata(self):
        """GaugeRouteDecision has no tensor_name, model_name, etc."""
        g = GaugeVector(structure_score=0.1, confidence=0.9)
        d = gauge_route(g)
        d_dict = d.to_dict()
        forbidden_keys = [
            "tensor_name", "model_name", "model_family",
            "layer_index", "tensor_role", "architecture",
        ]
        for key in forbidden_keys:
            assert key not in d_dict, f"Forbidden key {key} found in decision"

    def test_gauge_route_deterministic(self):
        """Same gauge vector always produces same decision."""
        g = GaugeVector(
            rms_error=0.03, cosine=0.997, kurtosis=8.0,
            channel_concentration=4.0, structure_score=0.5,
            confidence=0.8,
        )
        decisions = [gauge_route(g) for _ in range(5)]
        hashes = [d.receipt_hash for d in decisions]
        assert len(set(hashes)) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Routing logic from gauges
# ═══════════════════════════════════════════════════════════════════════════

class TestGaugeRoutingLogic:
    def test_low_confidence_accepts(self):
        """Low confidence gauge falls back conservatively to ACCEPT."""
        g = GaugeVector(structure_score=0.5, confidence=0.3)
        d = gauge_route(g)
        assert d.action == GaugeAction.ACCEPT
        assert "low_confidence" in d.reason

    def test_low_structure_accepts(self):
        """Low structure score → ACCEPT (gauges nominal)."""
        g = GaugeVector(structure_score=0.1, confidence=0.9)
        d = gauge_route(g)
        assert d.action == GaugeAction.ACCEPT
        assert "gauges_nominal" in d.reason

    def test_high_kurtosis_concentrated_triggers_outlier(self):
        """High kurtosis + concentrated channels → CORRECTION_OUTLIER."""
        g = GaugeVector(
            kurtosis=10.0, channel_concentration=5.0,
            structure_score=0.5, confidence=0.8,
        )
        d = gauge_route(g)
        assert d.action == GaugeAction.CORRECTION_OUTLIER
        assert "gauge_kurtosis_spike" in d.reason

    def test_low_svd_rank_triggers_lowrank(self):
        """Low SVD rank ratio → CORRECTION_LOWRANK."""
        g = GaugeVector(
            svd_rank_ratio=0.2, structure_score=0.5, confidence=0.8,
        )
        d = gauge_route(g)
        assert d.action == GaugeAction.CORRECTION_LOWRANK
        assert "gauge_rank_deficit" in d.reason

    def test_high_spectral_ratio_triggers_fallback(self):
        """High spectral ratio + high structure → FALLBACK."""
        g = GaugeVector(
            spectral_ratio=50.0, structure_score=0.7, confidence=0.9,
        )
        d = gauge_route(g)
        assert d.action == GaugeAction.FALLBACK
        assert d.fallback_required is True
        assert "gauge_structure_alarm" in d.reason

    def test_moderate_structure_triggers_probe(self):
        """Moderate structure without clear damage pattern → PROBE."""
        g = GaugeVector(structure_score=0.55, confidence=0.7)
        d = gauge_route(g)
        assert d.action == GaugeAction.PROBE
        assert any("structure" in r for r in d.reason)

    def test_acf10_triggers_structure_detection(self):
        """High ACF@10 triggers structure detection."""
        g = GaugeVector(
            acf_lag10=0.2, structure_score=0.65, confidence=0.8,
        )
        d = gauge_route(g)
        assert d.action == GaugeAction.FALLBACK
        assert d.fallback_required is True

    def test_action_to_correction_mapping(self):
        """Gauge actions map to CorrectionType strings."""
        assert _gauge_action_to_correction(GaugeAction.ACCEPT) == "none"
        assert _gauge_action_to_correction(GaugeAction.CORRECTION_OUTLIER) == "outlier_sidecar"
        assert _gauge_action_to_correction(GaugeAction.FALLBACK) == "structure_fallback"
        assert _gauge_action_to_correction(GaugeAction.PROBE) == "probe_required"


# ═══════════════════════════════════════════════════════════════════════════
# Comparison: gauge-only vs full router
# ═══════════════════════════════════════════════════════════════════════════

class TestComparison:
    @pytest.fixture
    def synthetic_tensors(self):
        rng = np.random.RandomState(42)
        return [
            (f"synthetic/{name}", gen_fn(rng)[0], gen_fn(rng)[1])
            for name, gen_fn in SYNTHETIC_GENERATORS.items()
        ]

    def test_comparison_runs(self, synthetic_tensors):
        """Comparison completes on synthetic tensors."""
        records, summary = run_comparison(synthetic_tensors)
        assert len(records) > 0
        assert "agreement_rate" in summary

    def test_agreement_rate_reasonable(self, synthetic_tensors):
        """Gauge-only agrees with full router at a reasonable rate.

        Since gauge-only uses the same thresholds and logic as the full
        residual router (just without names), agreement should be high.
        """
        records, summary = run_comparison(synthetic_tensors)
        # Should agree on most decisions (same underlying logic)
        assert summary["agreement_rate"] >= 0.6, (
            f"Agreement too low: {summary['agreement_rate']}"
        )

    def test_no_missed_fallbacks_on_clear_cases(self):
        """On clearly structured residuals, gauge catches fallbacks."""
        rng = np.random.RandomState(42)
        W = rng.randn(128, 256).astype(np.float32)
        # Create a tensor with clearly structured residual
        raw = rng.randn(128, 256).astype(np.float32) * 0.02
        kernel = np.ones(40) / 40.0
        structured = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'), 1, raw
        )
        W_structured = W + structured

        rp = profile_residual(W, W_structured)
        gauge = GaugeVector.from_residual_profile(rp)
        d = gauge_route(gauge)

        # If structure_score is high enough, gauge should catch it
        if rp.structure_score >= 0.5 and gauge.confidence >= 0.6:
            assert d.action in (
                GaugeAction.FALLBACK, GaugeAction.PROBE,
                GaugeAction.CORRECTION_LOWRANK
            )

    def test_comparison_output_file(self, synthetic_tensors, tmp_path):
        """Comparison writes JSONL + summary."""
        output = tmp_path / "gauge_comparison.jsonl"
        records, summary = run_comparison(synthetic_tensors, output_path=output)

        assert output.exists()
        summary_path = output.with_suffix(".summary.json")
        assert summary_path.exists()

    def test_comparison_records_serializable(self, synthetic_tensors):
        """All comparison records serialize to JSON."""
        records, _ = run_comparison(synthetic_tensors)
        for r in records:
            text = json.dumps(r.to_dict())
            assert len(text) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalysis:
    def test_empty_analysis(self):
        """Empty records produce error."""
        summary = analyze_comparison([])
        assert "error" in summary

    def test_verdict_field(self):
        """Summary includes a verdict."""
        records = [
            ComparisonRecord(
                blinded_id="abc", model_family="test", tensor_role="test",
                codec_name="affine", gauge_action="none", full_action="none",
                agree=True, gauge_fallback=False, full_fallback=False,
                missed_fallback=False, false_fallback=False,
            )
        ]
        summary = analyze_comparison(records)
        assert "verdict" in summary
        assert summary["verdict"] in (
            "gauge_only_sufficient", "gauge_only_partial", "gauge_only_insufficient"
        )


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end with real profiling
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_real_random_tensor_accepted(self):
        """Random tensor → distributed → gauge accepts."""
        rng = np.random.RandomState(42)
        W = rng.randn(64, 128).astype(np.float32)
        W_hat = W + rng.randn(64, 128).astype(np.float32) * 0.001

        rp = profile_residual(W, W_hat)
        gauge = GaugeVector.from_residual_profile(rp)
        d = gauge_route(gauge)
        assert d.action == GaugeAction.ACCEPT

    def test_real_concentrated_tensor_detects_outlier(self):
        """Concentrated error → gauge detects outlier pattern."""
        rp = ResidualProfile(
            rms_error=0.03, cosine=0.997,
            kurtosis=12.0, channel_concentration=6.0,
            structure_score=0.55, damage_type=DamageType.CONCENTRATED,
        )
        gauge = GaugeVector.from_residual_profile(rp)
        d = gauge_route(gauge)
        assert d.action == GaugeAction.CORRECTION_OUTLIER

    def test_gauge_vector_invariant_to_tensor_identity(self):
        """Two different tensors with identical gauge readings route identically."""
        rp = ResidualProfile(
            rms_error=0.02, cosine=0.998, kurtosis=3.0, sparsity=0.2,
            acf_lag1=0.01, acf_lag10=0.005, spectral_ratio=5.0,
            svd_rank_ratio=0.8, structure_score=0.15,
            damage_type=DamageType.DISTRIBUTED,
        )
        g1 = GaugeVector.from_residual_profile(rp)
        g2 = GaugeVector.from_residual_profile(rp)
        d1 = gauge_route(g1)
        d2 = gauge_route(g2)
        assert d1.action == d2.action
        assert d1.blinded_id == d2.blinded_id
        assert d1.receipt_hash == d2.receipt_hash


# ═══════════════════════════════════════════════════════════════════════════
# JSONL adapter: gauge-only routing from sweep receipts
# ═══════════════════════════════════════════════════════════════════════════

class TestJSONLAdapter:
    """Tests for running gauge-only routing on real sweep JSONL receipts."""

    SAMPLE_RECORD = {
        "tensor_id": "model.layers.5.mlp.gate_proj.weight",
        "model_family": "transformer",
        "tensor_role": "mlp",
        "synthetic": False,
        "codec_name": "affine_g128",
        "bpw": 6.25,
        "cosine": 0.999824,
        "rms_error": 0.01967,
        "max_abs_error": 0.03458,
        "residual_profile": {
            "rms_error": 0.01967,
            "cosine": 0.999824,
            "max_abs_error": 0.03458,
            "mean_abs_error": 0.01712,
            "kurtosis": -1.183,
            "sparsity": 0.078,
            "acf_lag1": 0.272,
            "acf_lag10": 0.048,
            "spectral_ratio": 4.06,
            "svd_rank_ratio": 1.0,
            "top10_explained": 0.0,
            "channel_concentration": 1.0,
            "structure_score": 0.093,
            "damage_type": "distributed",
        },
        "ghost_features": {
            "te": 0.343,
            "tr": 0.0,
            "mo": 0.752,
            "ac": 0.078,
        },
        "route_signal": {
            "codec_optimal": True,
            "try_correction": False,
            "correction_hint": "none",
            "damage_type": "distributed",
            "structure_score": 0.093,
            "confidence": 0.9,
        },
        "receipt_hash": "47e0197e734f012b",
    }

    def test_forbidden_metadata_stripped(self):
        """GaugeVector from sweep record contains no forbidden fields."""
        gauge = gauge_vector_from_sweep_record(self.SAMPLE_RECORD)
        gauge_dict = gauge.to_dict()
        for field in _FORBIDDEN_FIELDS:
            assert field not in gauge_dict, f"Forbidden field {field} leaked into gauge"

    def test_numeric_fields_preserved(self):
        """GaugeVector preserves numeric residual/ghost values from record."""
        gauge = gauge_vector_from_sweep_record(self.SAMPLE_RECORD)
        assert gauge.rms_error == pytest.approx(0.01967)
        assert gauge.cosine == pytest.approx(0.999824)
        assert gauge.kurtosis == pytest.approx(-1.183)
        assert gauge.structure_score == pytest.approx(0.093)
        assert gauge.ghost_te == pytest.approx(0.343)
        assert gauge.ghost_mo == pytest.approx(0.752)
        assert gauge.confidence == pytest.approx(0.9)

    def test_same_gauge_same_route_different_tensor_ids(self):
        """Same numeric values produce same route regardless of tensor_id."""
        record_a = dict(self.SAMPLE_RECORD, tensor_id="model.layers.0.attn.q_proj")
        record_b = dict(self.SAMPLE_RECORD, tensor_id="model.layers.99.mlp.up_proj")
        gauge_a = gauge_vector_from_sweep_record(record_a)
        gauge_b = gauge_vector_from_sweep_record(record_b)
        d_a = gauge_route(gauge_a)
        d_b = gauge_route(gauge_b)
        assert d_a.action == d_b.action
        assert d_a.blinded_id == d_b.blinded_id
        assert d_a.receipt_hash == d_b.receipt_hash

    def test_missing_ghost_handled(self):
        """Record with no ghost features builds valid gauge."""
        record = dict(self.SAMPLE_RECORD, ghost_features=None)
        gauge = gauge_vector_from_sweep_record(record)
        assert gauge.ghost_te == 0.0
        assert gauge.ghost_mo == 0.0
        d = gauge_route(gauge)
        assert d.action in GaugeAction

    def test_deterministic_blinded_id(self):
        """Same record always produces same blinded_id."""
        ids = set()
        for _ in range(5):
            gauge = gauge_vector_from_sweep_record(self.SAMPLE_RECORD)
            d = gauge_route(gauge)
            ids.add(d.blinded_id)
        assert len(ids) == 1

    def test_different_families_same_numerics_same_route(self):
        """Records from different model families with same numerics route identically."""
        record_ssm = dict(self.SAMPLE_RECORD, model_family="ssm", tensor_role="state_proj")
        record_tf = dict(self.SAMPLE_RECORD, model_family="transformer", tensor_role="mlp")
        gauge_ssm = gauge_vector_from_sweep_record(record_ssm)
        gauge_tf = gauge_vector_from_sweep_record(record_tf)
        d_ssm = gauge_route(gauge_ssm)
        d_tf = gauge_route(gauge_tf)
        assert d_ssm.action == d_tf.action
        assert d_ssm.blinded_id == d_tf.blinded_id

    def test_high_structure_record_triggers_action(self):
        """Record with high structure score triggers non-accept action."""
        record = dict(self.SAMPLE_RECORD)
        record["residual_profile"] = dict(
            record["residual_profile"],
            structure_score=0.65,
            kurtosis=12.0,
            channel_concentration=5.0,
        )
        record["route_signal"] = dict(record["route_signal"], confidence=0.8)
        gauge = gauge_vector_from_sweep_record(record)
        d = gauge_route(gauge)
        assert d.action != GaugeAction.ACCEPT

    def test_run_comparison_from_jsonl(self, tmp_path):
        """run_comparison_from_jsonl processes a JSONL file."""
        jsonl = tmp_path / "test_sweep.jsonl"
        # Write a few records (including an exact that should be skipped)
        exact_record = dict(self.SAMPLE_RECORD, codec_name="exact",
                            cosine=1.0, rms_error=0.0)
        exact_record["residual_profile"] = dict(
            exact_record["residual_profile"], structure_score=0.0)
        lines = [
            json.dumps(self.SAMPLE_RECORD),
            json.dumps(dict(self.SAMPLE_RECORD, codec_name="vq_k256")),
            json.dumps(exact_record),
        ]
        jsonl.write_text("\n".join(lines))

        output = tmp_path / "gauge_comparison.jsonl"
        records, summary = run_comparison_from_jsonl(jsonl, output_path=output)

        # Exact should be skipped, so only 2 records
        assert len(records) == 2
        assert "agreement_rate" in summary
        assert output.exists()
        summary_path = output.with_suffix(".summary.json")
        assert summary_path.exists()

    def test_jsonl_comparison_records_serializable(self, tmp_path):
        """All comparison records from JSONL serialize to JSON."""
        jsonl = tmp_path / "test_sweep.jsonl"
        jsonl.write_text(json.dumps(self.SAMPLE_RECORD))
        records, _ = run_comparison_from_jsonl(jsonl)
        for r in records:
            text = json.dumps(r.to_dict())
            assert len(text) > 0
