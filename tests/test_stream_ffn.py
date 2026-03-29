"""Tests for stream_ffn.py — streaming FFN layer components."""

import numpy as np
from helix_substrate.stream_ffn import silu, FFNLayerReceipt


class TestSilu:
    def test_zero(self):
        x = np.array([0.0], dtype=np.float32)
        y = silu(x)
        assert abs(y[0]) < 1e-6  # silu(0) = 0 * sigmoid(0) = 0

    def test_large_positive(self):
        x = np.array([10.0], dtype=np.float32)
        y = silu(x)
        # silu(10) ≈ 10 * 1.0 ≈ 10
        assert abs(y[0] - 10.0) < 0.01

    def test_large_negative(self):
        x = np.array([-10.0], dtype=np.float32)
        y = silu(x)
        # silu(-10) ≈ -10 * sigmoid(-10) ≈ -10 * ~0 ≈ 0
        assert abs(y[0]) < 0.01

    def test_positive_values_are_positive(self):
        x = np.array([1.0, 2.0, 5.0], dtype=np.float32)
        y = silu(x)
        assert all(y > 0)

    def test_negative_values_are_negative(self):
        x = np.array([-1.0, -2.0, -5.0], dtype=np.float32)
        y = silu(x)
        assert all(y < 0)

    def test_monotonic_for_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = silu(x)
        assert all(y[i] < y[i + 1] for i in range(len(y) - 1))

    def test_shape_preserved(self):
        x = np.random.randn(3, 4).astype(np.float32)
        y = silu(x)
        assert y.shape == (3, 4)

    def test_numerically_stable_extreme(self):
        # Should not overflow/NaN on extreme values
        x = np.array([80.0, -80.0, 100.0, -100.0], dtype=np.float32)
        y = silu(x)
        assert not np.any(np.isnan(y))
        assert not np.any(np.isinf(y))


class TestFFNLayerReceipt:
    def test_default_state(self):
        r = FFNLayerReceipt()
        assert r.status == "PENDING"
        assert r.block_index == 0
        assert r.gate_receipt is None

    def test_to_dict(self):
        r = FFNLayerReceipt(block_index=5, status="PASS")
        d = r.to_dict()
        assert d["block_index"] == 5
        assert d["status"] == "PASS"
        assert "claim_hygiene" in d
        assert "projections" in d

    def test_claim_hygiene_all_true_streaming(self):
        r = FFNLayerReceipt(
            ffn_config={"streaming_modes_used": ["true_block_streaming"],
                        "codec_versions_used": ["cdna_v2"]},
        )
        d = r.to_dict()
        assert d["claim_hygiene"]["all_true_block_streaming"] is True

    def test_claim_hygiene_mixed_modes(self):
        r = FFNLayerReceipt(
            ffn_config={"streaming_modes_used": ["true_block_streaming", "cached"],
                        "codec_versions_used": ["cdna_v2"]},
        )
        d = r.to_dict()
        assert d["claim_hygiene"]["all_true_block_streaming"] is False
