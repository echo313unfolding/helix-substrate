"""Tests for kurtosis_gate.py — runtime SVD correction gate."""

import torch
from helix_substrate.kurtosis_gate import compute_kurtosis, KurtosisGate


class TestComputeKurtosis:
    def test_normal_distribution(self):
        # Normal distribution has excess kurtosis ~0
        torch.manual_seed(42)
        x = torch.randn(10000)
        k = compute_kurtosis(x)
        assert abs(k) < 0.5  # should be near 0

    def test_uniform_distribution(self):
        # Uniform has negative excess kurtosis (~-1.2)
        x = torch.linspace(-1, 1, 10000)
        k = compute_kurtosis(x)
        assert k < 0  # platykurtic

    def test_heavy_tails(self):
        # Heavy-tailed distribution has positive excess kurtosis
        torch.manual_seed(42)
        x = torch.randn(10000)
        x[0] = 100.0  # add outliers
        x[1] = -100.0
        k = compute_kurtosis(x)
        assert k > 1.0  # leptokurtic

    def test_constant_tensor(self):
        x = torch.ones(100)
        k = compute_kurtosis(x)
        assert k == 0.0  # std ~0 -> returns 0

    def test_too_small_tensor(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        k = compute_kurtosis(x)
        assert k == 0.0  # < 4 elements

    def test_multidimensional(self):
        torch.manual_seed(42)
        x = torch.randn(32, 64)
        k = compute_kurtosis(x)
        # Should work on flattened tensor
        assert isinstance(k, float)


class TestKurtosisGate:
    def test_init_state(self):
        gate = KurtosisGate(threshold=5.0)
        assert gate._enable_svd is False
        assert gate._total_forwards == 0

    def test_low_kurtosis_stays_off(self):
        gate = KurtosisGate(threshold=5.0, switch_to_svd_after=2)
        # Feed low kurtosis values
        gate.update(1.0)
        gate.update(2.0)
        gate.update(3.0)
        assert gate._enable_svd is False

    def test_high_kurtosis_enables_svd(self):
        gate = KurtosisGate(threshold=5.0, switch_to_svd_after=2, ema_decay=0.0)
        # ema_decay=0 means no smoothing — raw value used directly
        gate.update(10.0)
        gate.update(10.0)  # 2 consecutive high -> switch
        assert gate._enable_svd is True

    def test_hysteresis_prevents_oscillation(self):
        gate = KurtosisGate(threshold=5.0, switch_to_svd_after=2,
                           recover_to_skip_after=2, ema_decay=0.0)
        gate.update(10.0)
        gate.update(10.0)  # -> SVD on
        assert gate._enable_svd is True

        # One low value shouldn't turn it off (need 2 consecutive)
        gate.update(1.0)
        assert gate._enable_svd is True

        # Two consecutive low -> SVD off
        gate.update(1.0)
        assert gate._enable_svd is False

    def test_ema_smoothing(self):
        gate = KurtosisGate(threshold=5.0, ema_decay=0.5)
        gate.update(10.0)  # ema = 10.0 (initial)
        gate.update(0.0)   # ema = 0.5*10 + 0.5*0 = 5.0
        assert abs(gate._ema_kurtosis - 5.0) < 0.1

    def test_step_with_tensor(self):
        gate = KurtosisGate(threshold=5.0, check_interval=1)
        torch.manual_seed(42)
        x = torch.randn(100)
        result = gate.step(x)
        assert isinstance(result, bool)
        assert gate._total_forwards == 1

    def test_check_interval_amortization(self):
        gate = KurtosisGate(threshold=5.0, check_interval=3)
        torch.manual_seed(42)
        x = torch.randn(100)
        # Forward 1: computes (cold start, not initialized)
        gate.step(x)
        assert gate._total_updates == 1
        # Forward 2: cached (2%3!=0)
        gate.step(x)
        assert gate._total_updates == 1
        # Forward 3: computes (3%3==0)
        gate.step(x)
        assert gate._total_updates == 2
        # Forward 4: cached (4%3!=0)
        gate.step(x)
        assert gate._total_updates == 2
        # Forward 5: cached (5%3!=0)
        gate.step(x)
        assert gate._total_updates == 2
        # Forward 6: computes (6%3==0)
        gate.step(x)
        assert gate._total_updates == 3

    def test_summary(self):
        gate = KurtosisGate(threshold=5.0)
        gate.update(10.0)
        s = gate.summary()
        assert s["threshold"] == 5.0
        assert s["total_updates"] == 1
        assert "svd_rate" in s

    def test_reset(self):
        gate = KurtosisGate(threshold=5.0, ema_decay=0.0)
        gate.update(10.0)
        gate.update(10.0)
        assert gate._enable_svd is True
        gate.reset()
        assert gate._enable_svd is False
        assert gate._total_forwards == 0

    def test_trace_bounded(self):
        gate = KurtosisGate(threshold=5.0, _trace_max=5)
        for i in range(20):
            gate.update(float(i))
        assert len(gate.trace()) == 5  # bounded
