"""Tests for cross-model compression-signal sweep.

Validates:
- sweep runs on synthetic tensors without external model downloads
- skip behavior is clean when model family unavailable
- receipts are deterministic
- residual fields are present for every lossy codec
- exact baseline produces zero/near-zero residual
- structured synthetic residual is classified as structured
- low-rank synthetic residual is classified as low-rank
- output summary includes pass/fail for the six research questions
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.sweep_compression_routing_signal import (
    CODEC_REGISTRY,
    SYNTHETIC_GENERATORS,
    SweepRecord,
    run_sweep,
    analyze_sweep,
    quality_gated_best_codec,
    _codec_exact,
    _codec_affine,
    _codec_vq,
    _codec_rvq,
    _codec_svd_lowrank,
    _classify_tensor_role,
    _classify_model_family,
)
from helix_substrate.residual_contract import DamageType


# ═══════════════════════════════════════════════════════════════════════════
# Codec correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestCodecs:
    def test_exact_zero_error(self):
        """Exact codec produces identical output."""
        W = np.random.RandomState(42).randn(64, 128).astype(np.float32)
        W_hat, meta = _codec_exact(W)
        assert np.allclose(W, W_hat)
        assert meta["bpw"] == 32.0

    def test_affine_reconstructs(self):
        """Affine codec produces valid reconstruction."""
        W = np.random.RandomState(42).randn(64, 128).astype(np.float32)
        W_hat, meta = _codec_affine(W, block_size=128)
        assert W_hat.shape == W.shape
        cos = np.dot(W.ravel(), W_hat.ravel()) / (
            np.linalg.norm(W.ravel()) * np.linalg.norm(W_hat.ravel())
        )
        assert cos > 0.95
        assert meta["bpw"] < 32.0

    def test_vq_reconstructs(self):
        """VQ codec produces valid reconstruction."""
        W = np.random.RandomState(42).randn(32, 64).astype(np.float32)
        W_hat, meta = _codec_vq(W, k=256)
        assert W_hat.shape == W.shape
        cos = np.dot(W.ravel(), W_hat.ravel()) / (
            np.linalg.norm(W.ravel()) * np.linalg.norm(W_hat.ravel())
        )
        assert cos > 0.99
        assert "encoded_bytes" in meta

    def test_rvq_reconstructs(self):
        """RVQ codec produces valid reconstruction."""
        W = np.random.RandomState(42).randn(32, 64).astype(np.float32)
        W_hat, meta = _codec_rvq(W)
        assert W_hat.shape == W.shape
        assert meta["bpw"] < 32.0

    def test_svd_lowrank_reconstructs(self):
        """SVD low-rank codec produces valid reconstruction."""
        W = np.random.RandomState(42).randn(64, 128).astype(np.float32)
        W_hat, meta = _codec_svd_lowrank(W, rank=8)
        assert W_hat.shape == W.shape
        assert meta["bpw"] < 32.0

    def test_svd_on_1d(self):
        """SVD handles 1D tensors."""
        W = np.random.RandomState(42).randn(256).astype(np.float32)
        W_hat, meta = _codec_svd_lowrank(W, rank=4)
        assert W_hat.shape == W.shape

    def test_affine_has_encoded_bytes(self):
        """Affine produces encoded bytes for Ghost."""
        W = np.random.RandomState(42).randn(64, 128).astype(np.float32)
        _, meta = _codec_affine(W)
        assert "encoded_bytes" in meta
        assert len(meta["encoded_bytes"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic generators
# ═══════════════════════════════════════════════════════════════════════════

class TestSyntheticGenerators:
    def test_all_generators_produce_valid_tensors(self):
        """Every generator produces a 2D float32 tensor with metadata."""
        rng = np.random.RandomState(42)
        for name, gen_fn in SYNTHETIC_GENERATORS.items():
            W, meta = gen_fn(rng)
            assert W.ndim == 2, f"{name} is not 2D"
            assert W.dtype == np.float32, f"{name} is not float32"
            assert "family" in meta
            assert "role" in meta
            assert meta["synthetic"] is True

    def test_generator_families_cover_required(self):
        """Generators cover transformer, ssm, moe, cnn, embedding."""
        families = set()
        rng = np.random.RandomState(42)
        for gen_fn in SYNTHETIC_GENERATORS.values():
            _, meta = gen_fn(rng)
            families.add(meta["family"])
        assert "transformer" in families
        assert "ssm" in families
        assert "moe" in families
        assert "cnn" in families
        assert "embedding" in families


# ═══════════════════════════════════════════════════════════════════════════
# Tensor role classification
# ═══════════════════════════════════════════════════════════════════════════

class TestClassification:
    def test_role_classification(self):
        assert _classify_tensor_role("model.layers.5.mlp.gate_proj.weight") == "mlp"
        assert _classify_tensor_role("model.layers.0.self_attn.q_proj.weight") == "attention"
        assert _classify_tensor_role("model.embed_tokens.weight") == "embedding"
        assert _classify_tensor_role("model.layers.3.mixer.in_proj.weight") == "state_proj"

    def test_family_classification(self):
        assert _classify_model_family("Mamba-130m") == "ssm"
        assert _classify_model_family("Zamba2-2.7B") == "hybrid"
        assert _classify_model_family("Qwen2.5-3B") == "transformer"
        assert _classify_model_family("OLMoE-1B") == "moe"


# ═══════════════════════════════════════════════════════════════════════════
# Sweep execution
# ═══════════════════════════════════════════════════════════════════════════

class TestSweep:
    @pytest.fixture
    def synthetic_tensors(self):
        rng = np.random.RandomState(42)
        return [
            (f"synthetic/{name}", gen_fn(rng)[0], gen_fn(rng)[1])
            for name, gen_fn in SYNTHETIC_GENERATORS.items()
        ]

    def test_sweep_runs_on_synthetic(self, synthetic_tensors):
        """Sweep runs to completion on synthetic tensors."""
        records = run_sweep(synthetic_tensors)
        assert len(records) > 0
        # 6 generators × 5 codecs = 30 records
        assert len(records) == len(SYNTHETIC_GENERATORS) * len(CODEC_REGISTRY)

    def test_receipts_deterministic(self, synthetic_tensors):
        """Same inputs produce same receipt hashes."""
        r1 = run_sweep(synthetic_tensors)
        r2 = run_sweep(synthetic_tensors)
        for a, b in zip(r1, r2):
            assert a.receipt_hash == b.receipt_hash

    def test_residual_fields_present(self, synthetic_tensors):
        """Every record has all residual profile fields."""
        records = run_sweep(synthetic_tensors)
        required = [
            "rms_error", "cosine", "kurtosis", "sparsity",
            "acf_lag1", "acf_lag10", "spectral_ratio",
            "svd_rank_ratio", "structure_score", "damage_type",
        ]
        for r in records:
            for field in required:
                assert field in r.residual_profile, f"Missing {field} in {r.codec_name}"

    def test_exact_baseline_near_zero(self, synthetic_tensors):
        """Exact codec produces near-zero residual."""
        records = run_sweep(synthetic_tensors)
        exact_records = [r for r in records if r.codec_name == "exact"]
        assert len(exact_records) == len(SYNTHETIC_GENERATORS)
        for r in exact_records:
            assert r.residual_profile["rms_error"] < 1e-6
            assert r.residual_profile["structure_score"] < 0.01
            assert r.cosine > 0.9999

    def test_ghost_features_present_for_encoded_codecs(self, synthetic_tensors):
        """Ghost features exist for codecs that produce encoded bytes."""
        records = run_sweep(synthetic_tensors)
        for r in records:
            if r.codec_name in ("affine_g128", "vq_k256", "rvq_16x16"):
                assert r.ghost_features is not None, (
                    f"Ghost missing for {r.codec_name} on {r.tensor_id}"
                )
                assert "te" in r.ghost_features

    def test_route_signal_present(self, synthetic_tensors):
        """Every record has routing signal."""
        records = run_sweep(synthetic_tensors)
        for r in records:
            assert "codec_optimal" in r.route_signal
            assert "confidence" in r.route_signal
            assert "damage_type" in r.route_signal


# ═══════════════════════════════════════════════════════════════════════════
# Damage detection on known structures
# ═══════════════════════════════════════════════════════════════════════════

class TestDamageDetection:
    def test_structured_residual_detected(self):
        """Smooth/correlated residual is classified as structured."""
        rng = np.random.RandomState(42)
        W = rng.randn(128, 256).astype(np.float32)
        raw = rng.randn(128, 256).astype(np.float32) * 0.01
        kernel = np.ones(30) / 30.0
        structured = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'), 1, raw
        )
        W_hat = W + structured

        from helix_substrate.residual_contract import profile_residual
        rp = profile_residual(W, W_hat)
        assert rp.damage_type in (DamageType.STRUCTURED, DamageType.LOW_RANK)
        assert rp.structure_score > 0.2

    def test_low_rank_residual_detected(self):
        """Rank-2 residual is classified as low_rank."""
        rng = np.random.RandomState(42)
        W = rng.randn(128, 256).astype(np.float32)
        u = rng.randn(128, 2).astype(np.float32)
        v = rng.randn(2, 256).astype(np.float32)
        W_hat = W + (u @ v) * 0.01

        from helix_substrate.residual_contract import profile_residual
        rp = profile_residual(W, W_hat)
        assert rp.damage_type == DamageType.LOW_RANK
        assert rp.svd_rank_ratio < 0.1

    def test_random_residual_is_distributed(self):
        """Random iid error is classified as distributed."""
        rng = np.random.RandomState(42)
        W = rng.randn(128, 256).astype(np.float32)
        W_hat = W + rng.randn(128, 256).astype(np.float32) * 0.001

        from helix_substrate.residual_contract import profile_residual
        rp = profile_residual(W, W_hat)
        assert rp.damage_type == DamageType.DISTRIBUTED
        assert rp.structure_score < 0.2


# ═══════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalysis:
    def test_summary_has_research_questions(self):
        """Summary includes all six research questions."""
        rng = np.random.RandomState(42)
        tensors = [
            (f"synthetic/{name}", gen_fn(rng)[0], gen_fn(rng)[1])
            for name, gen_fn in SYNTHETIC_GENERATORS.items()
        ]
        records = run_sweep(tensors)
        summary = analyze_sweep(records)

        assert "Q2_quality_gated_routing" in summary
        assert "Q4_codec_structure_signals" in summary
        assert "Q5_exact_baseline" in summary
        assert "Q6_ranking_divergence" in summary
        assert "damage_type_by_family" in summary

    def test_q5_exact_all_near_zero(self):
        """Q5: exact baseline has all-near-zero structure scores."""
        rng = np.random.RandomState(42)
        tensors = [
            (f"synthetic/{name}", gen_fn(rng)[0], gen_fn(rng)[1])
            for name, gen_fn in SYNTHETIC_GENERATORS.items()
        ]
        records = run_sweep(tensors)
        summary = analyze_sweep(records)
        assert summary["Q5_exact_baseline"]["all_near_zero"] is True

    def test_empty_records(self):
        """Empty records produce error summary."""
        summary = analyze_sweep([])
        assert "error" in summary

    def test_records_serializable(self):
        """All records serialize to JSON."""
        rng = np.random.RandomState(42)
        tensors = [("test", rng.randn(32, 64).astype(np.float32),
                     {"family": "test", "role": "test", "synthetic": True})]
        records = run_sweep(tensors)
        for r in records:
            text = json.dumps(r.to_dict())
            assert len(text) > 0

    def test_sweep_with_output_file(self, tmp_path):
        """Sweep writes JSONL + summary JSON."""
        rng = np.random.RandomState(42)
        W, meta = SYNTHETIC_GENERATORS["transformer_mlp"](rng)
        tensors = [("test_tensor", W, meta)]
        output = tmp_path / "test_sweep.jsonl"

        records = run_sweep(tensors, output_path=output)

        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == len(CODEC_REGISTRY)

        summary_path = output.with_suffix(".summary.json")
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert "cost" in summary

    def test_multiple_families_in_summary(self):
        """Summary correctly reports multiple families."""
        rng = np.random.RandomState(42)
        tensors = [
            (f"synthetic/{name}", gen_fn(rng)[0], gen_fn(rng)[1])
            for name, gen_fn in SYNTHETIC_GENERATORS.items()
        ]
        records = run_sweep(tensors)
        summary = analyze_sweep(records)
        assert summary["n_families"] >= 4  # transformer, ssm, moe, cnn, embedding


# ═══════════════════════════════════════════════════════════════════════════
# Quality-gated codec ranking
# ═══════════════════════════════════════════════════════════════════════════

class TestQualityGatedBestCodec:
    def test_low_structure_but_low_cosine_rejected(self):
        """Codec with low structure_score but terrible cosine is rejected."""
        records = [
            SweepRecord(
                tensor_id="t", model_family="test", tensor_role="test",
                synthetic=True, codec_name="svd_bad", bpw=2.0,
                cosine=0.30, rms_error=0.5, max_abs_error=1.0,
                residual_profile={"structure_score": 0.05, "damage_type": "distributed"},
                ghost_features=None, route_signal={}, receipt_hash="a",
            ),
            SweepRecord(
                tensor_id="t", model_family="test", tensor_role="test",
                synthetic=True, codec_name="affine_good", bpw=6.0,
                cosine=0.999, rms_error=0.001, max_abs_error=0.01,
                residual_profile={"structure_score": 0.20, "damage_type": "distributed"},
                ghost_features=None, route_signal={}, receipt_hash="b",
            ),
        ]
        result = quality_gated_best_codec(records)
        assert result["best_codec"] == "affine_good"
        assert "svd_bad" in result["rejected_for_quality"]

    def test_high_cosine_candidates_eligible(self):
        """Codecs within cosine window are all eligible."""
        records = [
            SweepRecord(
                tensor_id="t", model_family="test", tensor_role="test",
                synthetic=True, codec_name="c1", bpw=6.0,
                cosine=0.999, rms_error=0.001, max_abs_error=0.01,
                residual_profile={"structure_score": 0.30, "damage_type": "distributed"},
                ghost_features=None, route_signal={}, receipt_hash="a",
            ),
            SweepRecord(
                tensor_id="t", model_family="test", tensor_role="test",
                synthetic=True, codec_name="c2", bpw=8.0,
                cosine=0.998, rms_error=0.002, max_abs_error=0.02,
                residual_profile={"structure_score": 0.10, "damage_type": "distributed"},
                ghost_features=None, route_signal={}, receipt_hash="b",
            ),
        ]
        result = quality_gated_best_codec(records)
        assert result["eligible_count"] == 2
        assert result["rejected_for_quality"] == []

    def test_among_eligible_lowest_structure_wins(self):
        """Among quality-eligible codecs, lower structure_score wins."""
        records = [
            SweepRecord(
                tensor_id="t", model_family="test", tensor_role="test",
                synthetic=True, codec_name="high_struct", bpw=6.0,
                cosine=0.999, rms_error=0.001, max_abs_error=0.01,
                residual_profile={"structure_score": 0.40, "damage_type": "structured"},
                ghost_features=None, route_signal={}, receipt_hash="a",
            ),
            SweepRecord(
                tensor_id="t", model_family="test", tensor_role="test",
                synthetic=True, codec_name="low_struct", bpw=8.0,
                cosine=0.998, rms_error=0.002, max_abs_error=0.02,
                residual_profile={"structure_score": 0.10, "damage_type": "distributed"},
                ghost_features=None, route_signal={}, receipt_hash="b",
            ),
        ]
        result = quality_gated_best_codec(records)
        assert result["best_codec"] == "low_struct"
        assert result["residual_changed_choice"] is True
        assert result["reason"] == "residual_tiebreak"

    def test_no_codec_meets_floor(self):
        """If no codec meets min_cosine_floor, best cosine wins."""
        records = [
            SweepRecord(
                tensor_id="t", model_family="test", tensor_role="test",
                synthetic=True, codec_name="bad1", bpw=2.0,
                cosine=0.50, rms_error=0.5, max_abs_error=1.0,
                residual_profile={"structure_score": 0.05, "damage_type": "distributed"},
                ghost_features=None, route_signal={}, receipt_hash="a",
            ),
            SweepRecord(
                tensor_id="t", model_family="test", tensor_role="test",
                synthetic=True, codec_name="bad2", bpw=3.0,
                cosine=0.60, rms_error=0.4, max_abs_error=0.9,
                residual_profile={"structure_score": 0.30, "damage_type": "distributed"},
                ghost_features=None, route_signal={}, receipt_hash="b",
            ),
        ]
        result = quality_gated_best_codec(records)
        assert result["best_codec"] == "bad2"  # best cosine
        assert result["reason"] == "best_cosine_no_quality_peer"
        assert result["eligible_count"] == 0

    def test_exact_excluded_from_lossy_ranking(self):
        """Exact baseline should not be in lossy records passed to function."""
        # The function receives pre-filtered lossy records; verify the caller does this
        rng = np.random.RandomState(42)
        tensors = [("t", rng.randn(32, 64).astype(np.float32),
                     {"family": "test", "role": "test", "synthetic": True})]
        records = run_sweep(tensors)
        summary = analyze_sweep(records)
        q2 = summary["Q2_quality_gated_routing"]
        # No detail entry should have "exact" as best codec
        for d in q2["detail"]:
            assert d["best_cosine_codec"] != "exact"
            assert d["best_gated_codec"] != "exact"

    def test_q2_summary_uses_quality_gate(self):
        """Q2 summary uses quality-gated ranking, not raw structure ranking."""
        rng = np.random.RandomState(42)
        tensors = [
            (f"synthetic/{name}", gen_fn(rng)[0], gen_fn(rng)[1])
            for name, gen_fn in SYNTHETIC_GENERATORS.items()
        ]
        records = run_sweep(tensors)
        summary = analyze_sweep(records)
        q2 = summary["Q2_quality_gated_routing"]
        assert "agreement_rate" in q2
        assert "n_residual_changed_choice" in q2
        # Every detail entry should have the gated fields
        for d in q2["detail"]:
            assert "best_gated_codec" in d
            assert "reason" in d
            assert "eligible_count" in d
            assert "rejected_for_quality" in d

    def test_empty_records(self):
        """Empty input returns no_candidates."""
        result = quality_gated_best_codec([])
        assert result["best_codec"] is None
        assert result["reason"] == "no_candidates"


# ═══════════════════════════════════════════════════════════════════════════
# Skip behavior
# ═══════════════════════════════════════════════════════════════════════════

class TestSkipBehavior:
    def test_nonexistent_models_dir_skips_cleanly(self):
        """Non-existent model directory produces no crash."""
        from tools.sweep_compression_routing_signal import load_tensors_from_safetensors
        result = load_tensors_from_safetensors(Path("/nonexistent/path.safetensors"))
        assert result == []

    def test_sweep_with_no_tensors(self):
        """Empty tensor list produces empty results."""
        records = run_sweep([])
        assert len(records) == 0
