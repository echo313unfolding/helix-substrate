"""
Tests for WO-N: Market Realism Layer

Tests:
  DataSnapshot — build, hash, match, detect database change
  ExecutionModel — fill price, trade cost, presets, dynamic slippage
  RobustnessTest — Monte Carlo perturbation, fragility flags, sensitivity
  Integration — cost summary, price noise
"""

import math
import os
import sqlite3
import tempfile

import pytest

from helix_substrate.market_realism import (
    SCHEMA,
    DataSnapshot,
    ExecutionModel,
    PerturbationConfig,
    RobustnessReport,
    TrialResult,
    apply_price_noise,
    build_data_snapshot,
    build_execution_cost_summary,
    run_robustness_test,
)


# ---------------------------------------------------------------------------
# Test database helper
# ---------------------------------------------------------------------------

def _make_test_db(extra_rows=False):
    """Create a temporary test database with price_history table."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name)

    conn.execute("""
        CREATE TABLE price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, date TEXT,
            open REAL, high REAL, low REAL, close REAL, volume INTEGER
        )
    """)

    import hashlib
    from datetime import datetime as dt, timedelta

    for symbol in ["SMR", "NUE"]:
        base = 12.0 if symbol == "SMR" else 140.0
        day = 0
        current = dt(2025, 1, 2)
        end = dt(2025, 2, 28)
        while current <= end:
            if current.weekday() < 5:
                date = current.strftime("%Y-%m-%d")
                seed = int(hashlib.sha256(
                    f"{symbol}:{date}".encode()).hexdigest()[:4], 16)
                noise = (seed % 100 - 50) / 500.0
                price = base * (1 + noise + day * 0.002)
                conn.execute("""
                    INSERT INTO price_history
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, date,
                      round(price * 0.99, 2), round(price * 1.02, 2),
                      round(price * 0.98, 2), round(price, 2),
                      500000 + seed * 10))
                day += 1
            current += timedelta(days=1)

    if extra_rows:
        # Add one more row for SMR to simulate data change
        conn.execute("""
            INSERT INTO price_history
            (symbol, date, open, high, low, close, volume)
            VALUES ('SMR', '2025-03-03', 13.0, 13.5, 12.5, 13.2, 600000)
        """)

    conn.commit()
    return tmp.name, conn


# =========================================================================
# DataSnapshot tests
# =========================================================================

class TestDataSnapshot:

    def test_build_snapshot(self):
        db_path, conn = _make_test_db()
        try:
            snap = build_data_snapshot(
                conn, ["SMR", "NUE"], "2025-01-01", "2025-02-28",
                source=db_path)
            assert snap.symbols == ["NUE", "SMR"]  # sorted
            assert snap.total_rows > 0
            assert len(snap.price_hash) == 64
            assert len(snap.snapshot_hash) == 64
            assert snap.row_counts["SMR"] > 0
            assert snap.row_counts["NUE"] > 0
        finally:
            conn.close(); os.unlink(db_path)

    def test_hash_deterministic(self):
        db_path, conn = _make_test_db()
        try:
            s1 = build_data_snapshot(
                conn, ["SMR", "NUE"], "2025-01-01", "2025-02-28")
            s2 = build_data_snapshot(
                conn, ["SMR", "NUE"], "2025-01-01", "2025-02-28")
            assert s1.price_hash == s2.price_hash
            assert s1.snapshot_hash == s2.snapshot_hash
        finally:
            conn.close(); os.unlink(db_path)

    def test_hash_changes_with_data(self):
        db_path1, conn1 = _make_test_db()
        db_path2, conn2 = _make_test_db(extra_rows=True)
        try:
            s1 = build_data_snapshot(
                conn1, ["SMR"], "2025-01-01", "2025-03-31")
            s2 = build_data_snapshot(
                conn2, ["SMR"], "2025-01-01", "2025-03-31")
            assert s1.price_hash != s2.price_hash
            assert not s1.matches(s2)
        finally:
            conn1.close(); os.unlink(db_path1)
            conn2.close(); os.unlink(db_path2)

    def test_matches_same_data(self):
        db_path, conn = _make_test_db()
        try:
            s1 = build_data_snapshot(
                conn, ["SMR"], "2025-01-01", "2025-02-28")
            s2 = build_data_snapshot(
                conn, ["SMR"], "2025-01-01", "2025-02-28")
            assert s1.matches(s2)
        finally:
            conn.close(); os.unlink(db_path)

    def test_to_dict(self):
        db_path, conn = _make_test_db()
        try:
            snap = build_data_snapshot(
                conn, ["SMR"], "2025-01-01", "2025-02-28", source="test.db")
            d = snap.to_dict()
            assert d["source"] == "test.db"
            assert "price_hash" in d
            assert "snapshot_hash" in d
            assert "row_counts" in d
        finally:
            conn.close(); os.unlink(db_path)

    def test_subset_different_hash(self):
        db_path, conn = _make_test_db()
        try:
            s_full = build_data_snapshot(
                conn, ["SMR", "NUE"], "2025-01-01", "2025-02-28")
            s_partial = build_data_snapshot(
                conn, ["SMR"], "2025-01-01", "2025-02-28")
            assert s_full.price_hash != s_partial.price_hash
        finally:
            conn.close(); os.unlink(db_path)

    def test_empty_symbols(self):
        db_path, conn = _make_test_db()
        try:
            snap = build_data_snapshot(
                conn, ["FAKEXYZ"], "2025-01-01", "2025-02-28")
            assert snap.total_rows == 0
            assert snap.row_counts["FAKEXYZ"] == 0
        finally:
            conn.close(); os.unlink(db_path)


# =========================================================================
# ExecutionModel tests
# =========================================================================

class TestExecutionModel:

    def test_default_model(self):
        m = ExecutionModel()
        assert m.slippage_bps == 10.0
        assert m.commission_per_trade == 0.0
        assert m.fill_probability == 1.0
        assert m.stop_loss_pct == 0.15

    def test_fill_price_buy(self):
        m = ExecutionModel(slippage_bps=10.0)
        fill = m.compute_fill_price("BUY", 100.0)
        assert fill > 100.0
        assert abs(fill - 100.10) < 0.01  # 10 bps

    def test_fill_price_sell(self):
        m = ExecutionModel(slippage_bps=10.0)
        fill = m.compute_fill_price("SELL", 100.0)
        assert fill < 100.0
        assert abs(fill - 99.90) < 0.01  # 10 bps

    def test_dynamic_slippage(self):
        m = ExecutionModel(slippage_bps=10.0, market_impact_bps_per_pct=5.0)
        # 2% of ADV → 10 + (0.02 * 5.0) = 10.1 bps
        eff = m.effective_slippage_bps(0.02)
        assert abs(eff - 10.1) < 0.01

    def test_trade_cost(self):
        m = ExecutionModel(slippage_bps=10.0, commission_per_trade=5.0)
        cost = m.compute_trade_cost(100, 50.0, "BUY")
        assert cost["mid_price"] == 50.0
        assert cost["fill_price"] > 50.0
        assert cost["commission"] == 5.0
        assert cost["total_cost"] > 5.0

    def test_conservative_preset(self):
        m = ExecutionModel.conservative()
        assert m.slippage_bps == 25.0
        assert m.commission_per_trade == 5.0
        assert m.fill_probability == 0.9
        assert m.max_portfolio_drawdown == 0.20
        assert m.model_name == "conservative"

    def test_optimistic_preset(self):
        m = ExecutionModel.optimistic()
        assert m.slippage_bps == 5.0
        assert m.commission_per_trade == 0.0
        assert m.fill_probability == 1.0
        assert m.model_name == "optimistic"

    def test_to_dict(self):
        m = ExecutionModel(slippage_bps=15.0, take_profit_pct=0.25)
        d = m.to_dict()
        assert d["slippage_bps"] == 15.0
        assert d["take_profit_pct"] == 0.25
        assert "model_name" in d

    def test_zero_slippage(self):
        m = ExecutionModel(slippage_bps=0.0)
        fill_buy = m.compute_fill_price("BUY", 100.0)
        fill_sell = m.compute_fill_price("SELL", 100.0)
        assert fill_buy == 100.0
        assert fill_sell == 100.0


# =========================================================================
# RobustnessTest tests
# =========================================================================

def _simple_backtest(**kwargs):
    """Deterministic mock backtest for testing."""
    slippage = kwargs.get("slippage_bps", 10.0)
    commission = kwargs.get("commission_per_trade", 0.0)
    base_sharpe = 1.5
    base_return = 0.12

    # Degrade performance with higher costs
    cost_drag = (slippage / 10000) * 50 + (commission / 100)
    sharpe = base_sharpe - cost_drag
    total_return = base_return - cost_drag * 2

    return {
        "sharpe_ratio": round(sharpe, 4),
        "total_return": round(total_return, 4),
        "max_drawdown": round(-0.05 - cost_drag, 4),
        "win_rate": round(0.55 - cost_drag * 0.1, 4),
    }


class TestRobustnessTest:

    def test_basic_perturbation(self):
        config = PerturbationConfig(
            slippage_range=(5.0, 50.0),
            n_trials=10,
            base_seed=42,
        )
        report = run_robustness_test(
            backtest_fn=_simple_backtest,
            base_params={"slippage_bps": 10.0},
            perturbation=config,
        )
        assert report.n_trials == 10
        assert report.n_passed + report.n_failed == 10
        assert len(report.trials) == 10

    def test_pass_rate(self):
        def strict_integrity(metrics):
            return metrics.get("sharpe_ratio", 0) > 1.0

        config = PerturbationConfig(
            slippage_range=(5.0, 100.0),  # wide range
            n_trials=20,
            base_seed=42,
        )
        report = run_robustness_test(
            backtest_fn=_simple_backtest,
            base_params={"slippage_bps": 10.0},
            perturbation=config,
            integrity_fn=strict_integrity,
        )
        assert 0 <= report.pass_rate <= 1.0
        assert report.n_passed <= report.n_trials

    def test_metric_distributions(self):
        config = PerturbationConfig(
            slippage_range=(5.0, 30.0),
            n_trials=15,
            base_seed=42,
        )
        report = run_robustness_test(
            backtest_fn=_simple_backtest,
            base_params={"slippage_bps": 10.0},
            perturbation=config,
        )
        assert "sharpe_ratio" in report.metric_distributions
        dist = report.metric_distributions["sharpe_ratio"]
        assert "mean" in dist
        assert "std" in dist
        assert "p50" in dist
        assert dist["min"] <= dist["mean"] <= dist["max"]

    def test_sensitivity_analysis(self):
        config = PerturbationConfig(
            slippage_range=(5.0, 50.0),
            n_trials=20,
            base_seed=42,
        )
        report = run_robustness_test(
            backtest_fn=_simple_backtest,
            base_params={"slippage_bps": 10.0},
            perturbation=config,
        )
        # Higher slippage should hurt Sharpe → negative sensitivity
        assert "slippage_bps" in report.sensitivity
        assert report.sensitivity["slippage_bps"] < 0

    def test_fragility_flags(self):
        def fragile_backtest(**kwargs):
            slippage = kwargs.get("slippage_bps", 10.0)
            # Very sensitive to slippage
            sharpe = 1.5 - slippage * 0.05
            return {
                "sharpe_ratio": sharpe,
                "total_return": sharpe * 0.05,
                "max_drawdown": -0.05 - slippage * 0.01,
                "win_rate": 0.5,
            }

        config = PerturbationConfig(
            slippage_range=(5.0, 100.0),
            n_trials=30,
            base_seed=42,
        )
        report = run_robustness_test(
            backtest_fn=fragile_backtest,
            base_params={"slippage_bps": 10.0},
            perturbation=config,
        )
        # Should detect negative p5 Sharpe at high slippage
        assert len(report.fragility_flags) > 0

    def test_reproducible_with_seed(self):
        config = PerturbationConfig(
            slippage_range=(5.0, 30.0),
            n_trials=10,
            base_seed=42,
        )
        r1 = run_robustness_test(
            backtest_fn=_simple_backtest,
            base_params={"slippage_bps": 10.0},
            perturbation=config,
        )
        r2 = run_robustness_test(
            backtest_fn=_simple_backtest,
            base_params={"slippage_bps": 10.0},
            perturbation=config,
        )
        assert r1.metric_distributions == r2.metric_distributions
        for t1, t2 in zip(r1.trials, r2.trials):
            assert t1.metrics == t2.metrics

    def test_commission_perturbation(self):
        config = PerturbationConfig(
            commission_range=(0.0, 20.0),
            n_trials=10,
            base_seed=42,
        )
        report = run_robustness_test(
            backtest_fn=_simple_backtest,
            base_params={"commission_per_trade": 0.0},
            perturbation=config,
        )
        assert report.n_trials == 10
        # Different commissions used across trials
        commissions = [t.params.get("commission_per_trade", 0)
                       for t in report.trials]
        assert len(set(commissions)) > 1

    def test_to_dict(self):
        config = PerturbationConfig(
            slippage_range=(5.0, 30.0),
            n_trials=5,
            base_seed=42,
        )
        report = run_robustness_test(
            backtest_fn=_simple_backtest,
            base_params={"slippage_bps": 10.0},
            perturbation=config,
        )
        d = report.to_dict()
        assert d["schema"] == SCHEMA
        assert d["n_trials"] == 5
        assert "metric_distributions" in d
        assert "fragility_flags" in d
        assert "trials" in d

    def test_handles_backtest_error(self):
        def failing_backtest(**kwargs):
            if kwargs.get("slippage_bps", 0) > 30:
                raise ValueError("Slippage too high")
            return {"sharpe_ratio": 1.0, "total_return": 0.05,
                    "max_drawdown": -0.05, "win_rate": 0.5}

        config = PerturbationConfig(
            slippage_range=(5.0, 50.0),
            n_trials=10,
            base_seed=42,
        )
        report = run_robustness_test(
            backtest_fn=failing_backtest,
            base_params={"slippage_bps": 10.0},
            perturbation=config,
        )
        # Should not crash; errors treated as failed trials
        assert report.n_trials == 10


# =========================================================================
# Integration tests
# =========================================================================

class TestIntegration:

    def test_execution_cost_summary(self):
        trades = [
            {"price": 50.0, "shares": 100, "action": "BUY"},
            {"price": 55.0, "shares": 100, "action": "SELL"},
        ]
        model = ExecutionModel(slippage_bps=10.0, commission_per_trade=5.0)
        summary = build_execution_cost_summary(trades, model)
        assert summary["n_trades"] == 2
        assert summary["total_commission"] == 10.0
        assert summary["total_slippage"] > 0
        assert summary["total_cost"] > 10.0
        assert summary["cost_pct_of_turnover"] > 0

    def test_price_noise_preserves_length(self):
        prices = [100.0, 101.0, 102.0, 101.5, 103.0]
        noisy = apply_price_noise(prices, 0.1, seed=42)
        assert len(noisy) == len(prices)

    def test_price_noise_deterministic(self):
        prices = [100.0, 101.0, 102.0, 101.5, 103.0]
        n1 = apply_price_noise(prices, 0.1, seed=42)
        n2 = apply_price_noise(prices, 0.1, seed=42)
        assert n1 == n2

    def test_price_noise_different_seeds(self):
        prices = [100.0, 101.0, 102.0, 101.5, 103.0]
        n1 = apply_price_noise(prices, 0.1, seed=42)
        n2 = apply_price_noise(prices, 0.1, seed=99)
        assert n1 != n2

    def test_zero_noise_returns_original(self):
        prices = [100.0, 101.0, 102.0]
        noisy = apply_price_noise(prices, 0.0, seed=42)
        assert noisy == prices

    def test_cost_summary_zero_trades(self):
        model = ExecutionModel()
        summary = build_execution_cost_summary([], model)
        assert summary["n_trades"] == 0
        assert summary["total_cost"] == 0
        assert summary["avg_cost_per_trade"] == 0

    def test_data_snapshot_in_trace_context(self):
        """DataSnapshot can fingerprint data for trace chain."""
        db_path, conn = _make_test_db()
        try:
            snap = build_data_snapshot(
                conn, ["SMR"], "2025-01-01", "2025-02-28", source=db_path)
            d = snap.to_dict()

            # Can be included in a trace chain dict
            assert isinstance(d["price_hash"], str)
            assert isinstance(d["snapshot_hash"], str)
            assert d["total_rows"] > 0
        finally:
            conn.close(); os.unlink(db_path)
