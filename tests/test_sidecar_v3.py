"""Tests for CDNA v3 sidecar generation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from helix_substrate.generate_sidecars_v3 import (
    find_outliers_contribution,
    find_outliers_percentile,
    read_sidecar_v3,
    write_sidecar_npz,
)


class TestPercentileOutliers:
    def test_roughly_correct_count(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(1000, 1000).astype(np.float32)
        positions, values = find_outliers_percentile(tensor, percentile=99.9)
        # Expect ~0.2% of 1M = ~2000 outliers (both tails)
        assert 1000 < len(positions) < 4000

    def test_values_are_extreme(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(500, 500).astype(np.float32)
        positions, values = find_outliers_percentile(tensor, percentile=99.0)
        flat = tensor.ravel()
        lo = np.percentile(flat, 1.0)
        hi = np.percentile(flat, 99.0)
        # All returned values should be outside the [lo, hi] range
        assert np.all((values < lo) | (values > hi))

    def test_positions_are_valid(self):
        tensor = np.arange(10000, dtype=np.float32).reshape(100, 100)
        positions, values = find_outliers_percentile(tensor, percentile=99.0)
        assert np.all(positions >= 0)
        assert np.all(positions < 10000)
        # Values at positions match tensor
        np.testing.assert_array_equal(values, tensor.ravel()[positions])

    def test_empty_for_uniform(self):
        tensor = np.ones((100, 100), dtype=np.float32)
        positions, values = find_outliers_percentile(tensor, percentile=99.9)
        assert len(positions) == 0


class TestContributionOutliers:
    def test_picks_high_impact(self):
        rng = np.random.RandomState(42)
        original = rng.randn(64, 64).astype(np.float32)
        quantized = original.copy()
        # Inject errors: one in a high-activation column, one in a low-activation column
        quantized[0, 0] += 1.0  # col 0 will have high activation
        quantized[1, 1] += 1.0  # col 1 will have low activation

        activations = np.zeros((10, 64), dtype=np.float32)
        activations[:, 0] = 10.0  # High activation on col 0
        activations[:, 1] = 0.01  # Low activation on col 1

        positions, values = find_outliers_contribution(
            original, quantized, activations, top_k=1
        )
        # Should pick col 0 (high impact) over col 1 (low impact)
        row, col = np.unravel_index(positions[0], (64, 64))
        assert row == 0 and col == 0

    def test_default_top_k(self):
        rng = np.random.RandomState(42)
        original = rng.randn(100, 100).astype(np.float32)
        quantized = original + rng.randn(100, 100).astype(np.float32) * 0.01
        activations = np.abs(rng.randn(10, 100)).astype(np.float32)
        positions, values = find_outliers_contribution(original, quantized, activations)
        # Default: 0.1% of 10000 = 10
        assert len(positions) == 10


class TestSidecarNpzRoundtrip:
    def test_roundtrip(self):
        positions = np.array([0, 5, 100, 999], dtype=np.int64)
        values = np.array([1.5, -2.3, 0.01, 100.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sidecar.npz"
            receipt = write_sidecar_npz(positions, values, path)

            assert receipt["num_corrections"] == 4
            assert receipt["size_bytes"] > 0

            pos_out, val_out = read_sidecar_v3(path)
            np.testing.assert_array_equal(pos_out, positions)
            # fp16 precision loss is expected
            np.testing.assert_allclose(val_out, values, atol=0.1)


class TestSidecarPatching:
    def test_exact_recovery_at_outlier_positions(self):
        rng = np.random.RandomState(42)
        original = rng.randn(64, 128).astype(np.float32)

        # Simulate quantization (uniform)
        vmin, vmax = original.min(), original.max()
        codebook = np.linspace(vmin, vmax, 256).astype(np.float32)
        flat = original.ravel()
        indices = np.clip(
            ((flat - vmin) / (vmax - vmin + 1e-10) * 255).astype(np.int32), 0, 255
        ).astype(np.uint8)
        quantized = codebook[indices].reshape(original.shape)

        # Find outliers
        positions, values = find_outliers_percentile(original, percentile=99.0)
        assert len(positions) > 0

        # Patch
        patched = quantized.copy().ravel()
        patched[positions] = values
        patched = patched.reshape(original.shape)

        # Outlier positions should now match original exactly
        for pos in positions:
            row, col = np.unravel_index(pos, original.shape)
            assert patched[row, col] == original.ravel()[pos]
