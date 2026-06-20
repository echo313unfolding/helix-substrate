"""Tests for Residual Contract — structured damage profiling."""

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
    compare_codecs,
    residual_routing_signal,
)


# ═══════════════════════════════════════════════════════════════════════════
# ResidualProfile basics
# ═══════════════════════════════════════════════════════════════════════════

class TestResidualProfileBasics:
    def test_default_profile(self):
        p = ResidualProfile()
        assert p.damage_type == DamageType.UNKNOWN
        assert p.structure_score == 0.0
        assert p.cosine == 1.0

    def test_to_dict_roundtrip(self):
        p = ResidualProfile(
            rms_error=0.01, cosine=0.999, kurtosis=2.5,
            structure_score=0.3, damage_type=DamageType.STRUCTURED,
        )
        d = p.to_dict()
        assert d["damage_type"] == "structured"
        p2 = ResidualProfile.from_dict(d)
        assert p2.damage_type == DamageType.STRUCTURED
        assert p2.kurtosis == 2.5
        assert p2.structure_score == 0.3

    def test_json_serializable(self):
        p = ResidualProfile(damage_type=DamageType.DISTRIBUTED)
        text = json.dumps(p.to_dict())
        assert "distributed" in text


# ═══════════════════════════════════════════════════════════════════════════
# Profile computation
# ═══════════════════════════════════════════════════════════════════════════

class TestProfileResidual:
    def test_zero_error(self):
        """Perfect reconstruction → zero error, cosine 1.0."""
        W = np.random.randn(64, 128).astype(np.float32)
        p = profile_residual(W, W.copy())
        assert p.rms_error < 1e-6
        assert p.cosine > 0.9999
        assert p.max_abs_error < 1e-6

    def test_random_error(self):
        """Random error → distributed damage, low structure score."""
        rng = np.random.RandomState(42)
        W = rng.randn(256, 512).astype(np.float32)
        noise = rng.randn(256, 512).astype(np.float32) * 0.01
        W_hat = W + noise

        p = profile_residual(W, W_hat)
        assert p.rms_error > 0
        assert abs(p.kurtosis) < 2.0  # near-Gaussian noise
        # Random noise spectral ratio ~10-15x at large FFT sizes (extreme value stats)
        # Genuine structure gives 200x+. Threshold for STRUCTURED is 20x.
        assert p.spectral_ratio < 20.0
        assert p.damage_type == DamageType.DISTRIBUTED

    def test_structured_error(self):
        """Correlated error → structured damage, high structure score."""
        rng = np.random.RandomState(42)
        W = rng.randn(256, 512).astype(np.float32)
        # Create strongly correlated error (low-pass filtered)
        raw_noise = rng.randn(256, 512).astype(np.float32) * 0.01
        # Smooth along rows to create autocorrelation
        kernel = np.ones(20) / 20.0
        structured_noise = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'), 1, raw_noise
        )
        W_hat = W + structured_noise

        p = profile_residual(W, W_hat)
        assert p.acf_lag1 > 0.5, f"Expected high ACF@1 for smoothed error, got {p.acf_lag1}"
        assert p.spectral_ratio > 5.0, f"Expected high SR for smoothed error, got {p.spectral_ratio}"

    def test_concentrated_error(self):
        """Error concentrated in few rows → concentrated damage."""
        rng = np.random.RandomState(42)
        W = rng.randn(256, 512).astype(np.float32)
        W_hat = W.copy()
        # Inject large error in only 5 rows
        W_hat[:5, :] += rng.randn(5, 512).astype(np.float32) * 1.0

        p = profile_residual(W, W_hat)
        assert p.channel_concentration > 3.0, (
            f"Expected high channel concentration, got {p.channel_concentration}"
        )

    def test_low_rank_error(self):
        """Low-rank error → LOW_RANK damage type."""
        rng = np.random.RandomState(42)
        W = rng.randn(256, 512).astype(np.float32)
        # Rank-2 error
        u = rng.randn(256, 2).astype(np.float32)
        v = rng.randn(2, 512).astype(np.float32)
        low_rank_error = (u @ v) * 0.01
        W_hat = W + low_rank_error

        p = profile_residual(W, W_hat)
        assert p.svd_rank_ratio < 0.1, (
            f"Expected very low SVD rank ratio for rank-2 error, got {p.svd_rank_ratio}"
        )
        assert p.damage_type == DamageType.LOW_RANK

    def test_1d_tensor(self):
        """1D tensor (bias) works without SVD/channel features."""
        W = np.random.randn(512).astype(np.float32)
        W_hat = W + np.random.randn(512).astype(np.float32) * 0.001
        p = profile_residual(W, W_hat)
        assert p.svd_rank_ratio == 1.0  # no SVD for 1D
        assert p.channel_concentration == 1.0  # no channels for 1D

    def test_empty_tensor(self):
        """Empty tensor returns default profile."""
        W = np.array([], dtype=np.float32)
        p = profile_residual(W, W.copy())
        assert p.rms_error == 0.0

    def test_deterministic(self):
        """Same inputs produce same profile."""
        rng = np.random.RandomState(42)
        W = rng.randn(128, 256).astype(np.float32)
        W_hat = W + rng.randn(128, 256).astype(np.float32) * 0.01

        p1 = profile_residual(W, W_hat)
        p2 = profile_residual(W, W_hat)
        assert p1.to_dict() == p2.to_dict()

    def test_all_fields_populated(self):
        """All fields are non-None after profiling."""
        rng = np.random.RandomState(42)
        W = rng.randn(64, 128).astype(np.float32)
        W_hat = W + rng.randn(64, 128).astype(np.float32) * 0.01

        p = profile_residual(W, W_hat)
        d = p.to_dict()
        for key, val in d.items():
            assert val is not None, f"Field {key} is None"

    def test_cosine_matches_manual(self):
        """Cosine field matches manual computation."""
        rng = np.random.RandomState(42)
        W = rng.randn(64, 128).astype(np.float32)
        W_hat = W + rng.randn(64, 128).astype(np.float32) * 0.01

        p = profile_residual(W, W_hat)
        manual_cos = float(
            np.dot(W.ravel(), W_hat.ravel()) /
            (np.linalg.norm(W.ravel()) * np.linalg.norm(W_hat.ravel()))
        )
        assert abs(p.cosine - manual_cos) < 1e-4


# ═══════════════════════════════════════════════════════════════════════════
# Codec comparison
# ═══════════════════════════════════════════════════════════════════════════

class TestCodecComparison:
    def test_compare_two_codecs(self):
        """Better codec (less error) ranks higher."""
        rng = np.random.RandomState(42)
        W = rng.randn(128, 256).astype(np.float32)

        # Good codec: small random error
        W_hat_good = W + rng.randn(128, 256).astype(np.float32) * 0.001

        # Bad codec: larger structured error
        raw = rng.randn(128, 256).astype(np.float32) * 0.01
        kernel = np.ones(20) / 20.0
        structured = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'), 1, raw
        )
        W_hat_bad = W + structured

        result = compare_codecs("test_tensor", {
            "good_codec": (W, W_hat_good),
            "bad_codec": (W, W_hat_bad),
        })

        assert result.best_codec == "good_codec"
        assert result.ranking[0] == "good_codec"
        assert result.ranking[1] == "bad_codec"

    def test_compare_serializable(self):
        """Comparison result is JSON-serializable."""
        rng = np.random.RandomState(42)
        W = rng.randn(32, 64).astype(np.float32)
        result = compare_codecs("t", {
            "a": (W, W + rng.randn(32, 64).astype(np.float32) * 0.01),
        })
        text = json.dumps(result.to_dict())
        assert "a" in text

    def test_single_codec(self):
        """Single codec comparison works."""
        W = np.random.randn(32, 64).astype(np.float32)
        result = compare_codecs("t", {"only": (W, W.copy())})
        assert result.best_codec == "only"
        assert len(result.ranking) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Routing integration
# ═══════════════════════════════════════════════════════════════════════════

class TestRoutingSignal:
    def test_optimal_codec(self):
        """Low structure score → codec_optimal=True."""
        p = ResidualProfile(structure_score=0.1, damage_type=DamageType.DISTRIBUTED)
        sig = residual_routing_signal(p)
        assert sig["codec_optimal"] is True
        assert sig["try_correction"] is False
        assert sig["correction_hint"] == "none"

    def test_suboptimal_codec(self):
        """High structure score → try_correction=True."""
        p = ResidualProfile(structure_score=0.7, damage_type=DamageType.STRUCTURED)
        sig = residual_routing_signal(p)
        assert sig["codec_optimal"] is False
        assert sig["try_correction"] is True
        assert sig["correction_hint"] == "spatial_correction"

    def test_concentrated_hint(self):
        """Concentrated damage → outlier correction hint."""
        p = ResidualProfile(structure_score=0.6, damage_type=DamageType.CONCENTRATED)
        sig = residual_routing_signal(p)
        assert sig["correction_hint"] == "outlier_correction"

    def test_low_rank_hint(self):
        """Low-rank damage → low_rank correction hint."""
        p = ResidualProfile(structure_score=0.6, damage_type=DamageType.LOW_RANK)
        sig = residual_routing_signal(p)
        assert sig["correction_hint"] == "low_rank_correction"

    def test_confidence_clear(self):
        """Clear signals get high confidence."""
        p_clear = ResidualProfile(structure_score=0.05)
        sig = residual_routing_signal(p_clear)
        assert sig["confidence"] >= 0.9

    def test_confidence_ambiguous(self):
        """Ambiguous signals get low confidence."""
        p_ambig = ResidualProfile(structure_score=0.35, damage_type=DamageType.DISTRIBUTED)
        sig = residual_routing_signal(p_ambig)
        assert sig["confidence"] < 0.5

    def test_signal_json_serializable(self):
        """Routing signal is JSON-serializable."""
        p = ResidualProfile(structure_score=0.3, damage_type=DamageType.STRUCTURED)
        sig = residual_routing_signal(p)
        text = json.dumps(sig)
        assert "structure_score" in text


# ═══════════════════════════════════════════════════════════════════════════
# Real-data test (if HXQ model available)
# ═══════════════════════════════════════════════════════════════════════════

class TestRealData:
    @pytest.fixture
    def real_tensor(self):
        """Load a real 2D tensor from cached safetensors if available."""
        try:
            import safetensors.numpy as sfnp
        except ImportError:
            pytest.skip("safetensors not installed")

        model_dir = Path.home() / ".cache/huggingface/hub"
        # Look for any cached safetensors
        shard_paths = sorted(model_dir.glob("**/model*.safetensors"))
        if not shard_paths:
            pytest.skip("No cached safetensors found")

        # Try first file — load only metadata to find a small 2D tensor
        for sp in shard_paths[:3]:
            try:
                tensors = sfnp.load_file(str(sp))
            except Exception:
                continue
            for name, arr in tensors.items():
                if arr.ndim == 2 and 32 < min(arr.shape) and max(arr.shape) <= 2048:
                    return name, arr

        pytest.skip("No suitable 2D tensor found")

    def test_real_vq_residual(self, real_tensor):
        """Profile residual from real VQ on a real tensor."""
        name, W = real_tensor
        W = W.astype(np.float32)

        # Simple VQ: k-means with k=256, small sample
        flat = W.ravel()
        sample = flat[:min(len(flat), 50000)]
        from helix_substrate.hxq_encoder import _simple_kmeans
        codebook, _ = _simple_kmeans(sample, 256, max_iters=5)

        # Assign and reconstruct (chunked for speed)
        chunk_size = 100000
        indices = np.empty(len(flat), dtype=np.int32)
        for i in range(0, len(flat), chunk_size):
            chunk = flat[i:i + chunk_size]
            d = np.abs(chunk[:, None] - codebook[None, :])
            indices[i:i + chunk_size] = np.argmin(d, axis=1)
        W_hat = codebook[indices].reshape(W.shape)

        p = profile_residual(W, W_hat)
        assert p.rms_error > 0
        assert 0.0 <= p.structure_score <= 1.0
        assert p.damage_type != DamageType.UNKNOWN

    def test_real_affine_residual(self, real_tensor):
        """Profile residual from real affine quantization on a real tensor."""
        name, W = real_tensor
        W = W.astype(np.float32)

        # Simple per-group affine (g=128)
        g = 128
        flat = W.ravel()
        n = len(flat)
        padded_n = ((n + g - 1) // g) * g
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat

        groups = padded.reshape(-1, g)
        mins = groups.min(axis=1, keepdims=True)
        maxs = groups.max(axis=1, keepdims=True)
        scales = (maxs - mins) / 63.0
        scales = np.maximum(scales, 1e-10)
        quantized = np.round((groups - mins) / scales)
        dequantized = quantized * scales + mins

        W_hat = dequantized.ravel()[:n].reshape(W.shape)

        p = profile_residual(W, W_hat)
        assert p.rms_error > 0
        assert 0.0 <= p.structure_score <= 1.0

    def test_real_vq_vs_affine(self, real_tensor):
        """Compare VQ and affine on same real tensor."""
        name, W = real_tensor
        W = W.astype(np.float32)
        flat = W.ravel()

        # VQ reconstruction
        sample = flat[:min(len(flat), 50000)]
        from helix_substrate.hxq_encoder import _simple_kmeans
        codebook, _ = _simple_kmeans(sample, 256, max_iters=5)
        chunk_size = 100000
        indices = np.empty(len(flat), dtype=np.int32)
        for i in range(0, len(flat), chunk_size):
            chunk = flat[i:i + chunk_size]
            d = np.abs(chunk[:, None] - codebook[None, :])
            indices[i:i + chunk_size] = np.argmin(d, axis=1)
        W_hat_vq = codebook[indices].reshape(W.shape)

        # Affine reconstruction
        g = 128
        n = len(flat)
        padded_n = ((n + g - 1) // g) * g
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        groups = padded.reshape(-1, g)
        mins = groups.min(axis=1, keepdims=True)
        maxs = groups.max(axis=1, keepdims=True)
        scales = (maxs - mins) / 63.0
        scales = np.maximum(scales, 1e-10)
        quantized = np.round((groups - mins) / scales)
        dequantized = quantized * scales + mins
        W_hat_affine = dequantized.ravel()[:n].reshape(W.shape)

        result = compare_codecs(name, {
            "vq_k256": (W, W_hat_vq),
            "affine6_g128": (W, W_hat_affine),
        })

        assert len(result.ranking) == 2
        assert result.best_codec in ("vq_k256", "affine6_g128")
        for prof in result.profiles.values():
            assert prof.rms_error > 0
            assert 0.0 <= prof.structure_score <= 1.0
