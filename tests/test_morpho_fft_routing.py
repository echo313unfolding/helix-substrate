"""
Tests for MORPHO_FFT_POLICY hybrid routing in CDNAv3Writer.

Covers:
- FFT routing for easy norms (encode + roundtrip decode)
- Quality gate fallback to exact for hard norms
- Adaptive coefficient count for broadband signals
- Policy field validation
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from helix_substrate.tensor_policy import (
    TensorClass,
    TensorPolicy,
    MORPHO_FFT_POLICY,
    MORPHO_FIT_POLICY,
)
from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.morpho_codec import (
    morpho_decode,
    morpho_encode,
    fit_to_target_fft,
    grow_from_fft_result,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="test_fft_")
    yield Path(d)
    shutil.rmtree(d)


class TestFFTPolicyFields:
    """Verify MORPHO_FFT_POLICY has expected settings."""

    def test_storage_mode_is_morpho(self):
        assert MORPHO_FFT_POLICY.storage_mode == "morpho"

    def test_spectral_enabled(self):
        assert MORPHO_FFT_POLICY.morpho_spectral is True

    def test_min_cosine_set(self):
        assert MORPHO_FFT_POLICY.morpho_min_cosine == 0.90

    def test_fit_enabled(self):
        assert MORPHO_FFT_POLICY.morpho_fit is True

    def test_tensor_class_is_norm(self):
        assert MORPHO_FFT_POLICY.tensor_class == TensorClass.NORM

    def test_old_policy_unchanged(self):
        """MORPHO_FIT_POLICY should still exist and not have spectral."""
        assert MORPHO_FIT_POLICY.morpho_spectral is False
        assert MORPHO_FIT_POLICY.morpho_min_cosine == 0.0


class TestFFTDirectProjection:
    """Test fit_to_target_fft and grow_from_fft_result."""

    def test_roundtrip_constant(self):
        """A constant signal should perfectly roundtrip with 1 coefficient."""
        target = np.ones(128, dtype=np.float64) * 3.14
        result = fit_to_target_fft(target, n_coeffs=2)
        assert result["cosine"] > 0.99
        decoded = grow_from_fft_result(result, (128,))
        cos = np.dot(target, decoded.ravel().astype(np.float64)) / (
            np.linalg.norm(target) * np.linalg.norm(decoded)
        )
        assert cos > 0.99

    def test_roundtrip_sine(self):
        """A pure sine should roundtrip perfectly with few coefficients."""
        x = np.linspace(0, 2 * np.pi, 256)
        target = np.sin(x) * 0.5
        result = fit_to_target_fft(target, n_coeffs=4)
        assert result["cosine"] > 0.99

    def test_deterministic(self):
        """Same input → same output."""
        rng = np.random.RandomState(42)
        target = rng.randn(512)
        r1 = fit_to_target_fft(target, n_coeffs=16, seed=b"test")
        r2 = fit_to_target_fft(target, n_coeffs=16, seed=b"test")
        assert r1["cosine"] == r2["cosine"]
        assert len(r1["coefficients"]) == len(r2["coefficients"])

    def test_more_coeffs_better(self):
        """More coefficients should give better or equal cosine."""
        rng = np.random.RandomState(123)
        target = rng.randn(1024)
        r16 = fit_to_target_fft(target, n_coeffs=16)
        r64 = fit_to_target_fft(target, n_coeffs=64)
        assert r64["cosine"] >= r16["cosine"] - 1e-10


class TestHybridRouting:
    """Test CDNAv3Writer with MORPHO_FFT_POLICY."""

    def test_easy_norm_uses_fft(self, tmp_dir):
        """High-structure norm should encode as FFT (no fallback)."""
        # Simulate an easy norm: dominated by DC component
        tensor = np.ones(2048, dtype=np.float32) * 1.5 + np.random.RandomState(0).randn(2048).astype(np.float32) * 0.01
        writer = CDNAv3Writer(tmp_dir)
        stats = writer.write_tensor(tensor, "test_easy_norm.weight", policy=MORPHO_FFT_POLICY)
        assert stats.get("morpho_fallback") is not True
        assert stats.get("cosine", 0) > 0.90

    def test_hard_norm_falls_back(self, tmp_dir):
        """Broadband noise norm should fall back to exact."""
        rng = np.random.RandomState(42)
        tensor = rng.randn(2048).astype(np.float32) * 0.05
        writer = CDNAv3Writer(tmp_dir)
        stats = writer.write_tensor(tensor, "test_hard_norm.weight", policy=MORPHO_FFT_POLICY)
        assert stats.get("morpho_fallback") is True
        assert stats["storage_mode"] == "exact"

    def test_fallback_roundtrip_exact(self, tmp_dir):
        """Fallback norms should roundtrip exactly via .npy."""
        rng = np.random.RandomState(42)
        tensor = rng.randn(2048).astype(np.float32) * 0.05
        writer = CDNAv3Writer(tmp_dir)
        writer.write_tensor(tensor, "test_fallback.weight", policy=MORPHO_FFT_POLICY)
        npy_path = tmp_dir / "test_fallback_weight.npy"
        assert npy_path.exists()
        decoded = np.load(npy_path)
        np.testing.assert_array_equal(tensor, decoded)

    def test_fft_roundtrip_decode(self, tmp_dir):
        """FFT-encoded norms should roundtrip via morpho_decode."""
        tensor = np.ones(512, dtype=np.float32) * 2.0 + np.random.RandomState(7).randn(512).astype(np.float32) * 0.1
        writer = CDNAv3Writer(tmp_dir)
        stats = writer.write_tensor(tensor, "test_decode.weight", policy=MORPHO_FFT_POLICY)
        assert stats.get("morpho_fallback") is not True

        morpho_dir = tmp_dir / "test_decode_weight.cdnav3"
        decoded = morpho_decode(morpho_dir)
        cos = float(np.dot(tensor.ravel(), decoded.ravel()) / (
            np.linalg.norm(tensor) * np.linalg.norm(decoded)))
        assert cos > 0.90


class TestAdaptiveCoefficients:
    """Test that broadband signals get more FFT coefficients."""

    def test_broadband_gets_more_coeffs(self, tmp_dir):
        """Broadband 1D tensor should use adaptive (128+) coefficients."""
        rng = np.random.RandomState(99)
        # Broadband: energy spread across all frequencies
        tensor = rng.randn(2048).astype(np.float32) * 0.05
        out_dir = tmp_dir / "broadband.cdnav3"
        stats = morpho_encode(
            tensor, "broadband.weight", out_dir,
            fit=True, spectral=True, n_codons=32,
        )
        # Should have adaptively bumped to 128 coefficients
        import json
        config = json.loads((out_dir / "morpho_config.json").read_text())
        assert config.get("n_coeffs", 32) > 32

    def test_narrowband_keeps_default(self, tmp_dir):
        """Narrowband 1D tensor should keep default coefficient count."""
        # Narrowband: almost all energy in DC
        tensor = np.ones(2048, dtype=np.float32) * 5.0
        out_dir = tmp_dir / "narrowband.cdnav3"
        stats = morpho_encode(
            tensor, "narrowband.weight", out_dir,
            fit=True, spectral=True, n_codons=32,
        )
        import json
        config = json.loads((out_dir / "morpho_config.json").read_text())
        assert config.get("n_coeffs", 32) == 32
