"""Tests for ported modules that belong in helix-substrate."""

import pytest


# ===== RAPL Meter =====

class TestRaplMeter:
    def test_import(self):
        from helix_substrate.rapl_meter import RaplMeter
        meter = RaplMeter()
        assert isinstance(meter.available, bool)

    def test_context_manager(self):
        from helix_substrate.rapl_meter import RaplMeter
        with RaplMeter() as meter:
            _ = sum(range(10000))
        if meter.available:
            assert isinstance(meter.joules, float)
            assert meter.joules >= 0
        else:
            assert meter.joules is None

    def test_manual_enter_exit(self):
        """Matches the integration pattern used in compress.py / eval_ppl_cpu.py."""
        from helix_substrate.rapl_meter import RaplMeter
        rapl = RaplMeter()
        rapl.__enter__()
        _ = sum(range(10000))
        rapl.__exit__(None, None, None)
        if rapl.available:
            assert isinstance(rapl.joules, float)


# ===== Hybrid Scheduler =====

class TestHybridScheduler:
    def test_init(self):
        from helix_substrate.hybrid_scheduler import HybridScheduler
        s = HybridScheduler()
        assert s.promote_threshold == 1.5
        assert s.active_gpu_jobs == 0

    def test_r_star_axes(self):
        from helix_substrate.hybrid_scheduler import HybridScheduler
        s = HybridScheduler()
        r = s.get_r_star_axes({"temp": 42.5, "power": 125, "mem_used": 1024, "mem_total": 4096})
        assert 0 <= r["thermal"] <= 1
        assert 0 <= r["power"] <= 1
        assert 0 <= r["memory"] <= 1
        assert r["queue"] == 1.0

    def test_promote_high_gain(self):
        from helix_substrate.hybrid_scheduler import HybridScheduler
        s = HybridScheduler(promote_threshold=1.0, cooldown_seconds=0)
        promote, info = s.should_promote(
            predicted_gain=5.0,
            telemetry={"temp": 40, "power": 50, "mem_used": 500, "mem_total": 4096},
        )
        assert promote is True
        assert info["decision"] == "promote_to_gpu"

    def test_no_promote_low_gain(self):
        from helix_substrate.hybrid_scheduler import HybridScheduler
        s = HybridScheduler(promote_threshold=2.0, cooldown_seconds=0)
        promote, info = s.should_promote(
            predicted_gain=0.5,
            telemetry={"temp": 40, "power": 50, "mem_used": 500, "mem_total": 4096},
        )
        assert promote is False

    def test_thermal_clamp(self):
        from helix_substrate.hybrid_scheduler import HybridScheduler
        s = HybridScheduler()
        demote, reason = s.should_demote(temp=82.0, marginal_improvement=0.5)
        assert demote is True
        assert reason == "thermal_clamp"

    def test_low_marginal_return(self):
        from helix_substrate.hybrid_scheduler import HybridScheduler
        s = HybridScheduler()
        demote, reason = s.should_demote(temp=50.0, marginal_improvement=0.005)
        assert demote is True
        assert reason == "low_marginal_return"

    def test_allocate_release(self):
        from helix_substrate.hybrid_scheduler import HybridScheduler
        s = HybridScheduler(max_concurrent=2)
        assert s.allocate() is True
        assert s.allocate() is True
        assert s.allocate() is False
        s.release()
        assert s.allocate() is True

    def test_hysteresis(self):
        """Once promoted, demote threshold is lower than promote threshold."""
        from helix_substrate.hybrid_scheduler import HybridScheduler
        s = HybridScheduler(promote_threshold=1.5, demote_threshold=0.8, cooldown_seconds=0)
        tel = {"temp": 40, "power": 50, "mem_used": 500, "mem_total": 4096}

        # First: promote (need gain ~5 to exceed threshold with R* ~0.37)
        promote, info = s.should_promote(predicted_gain=5.0, telemetry=tel)
        assert promote is True

        # Now at score ~1.1 (below promote 1.5 but above demote 0.8): stay on GPU
        promote2, info2 = s.should_promote(predicted_gain=3.0, telemetry=tel)
        assert info2["decision"] == "stay_on_gpu"
