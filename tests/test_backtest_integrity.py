"""Tests for WO-L: Backtest Integrity and Research Receipts.

Tests LeakageChecker, DataValidator, ResultAuditor, BenchmarkComparison,
ResearchReceipt, and the full integrity pipeline.
"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.backtest_integrity import (
    SCHEMA,
    AuditReport,
    BenchmarkComparison,
    DataQualityReport,
    DataValidator,
    LeakageChecker,
    LeakageReport,
    ResearchReceipt,
    ResultAuditor,
    build_benchmark_comparison,
    compute_buy_hold_return,
    run_integrity_pipeline,
)


def _make_test_db():
    """Create temporary database with price data for testing."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    conn = sqlite3.connect(tmp.name)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume INTEGER
        );
    """)

    # Insert contiguous price data for 2 symbols across all trading days
    import hashlib
    from datetime import datetime as dt, timedelta
    for symbol in ["SMR", "NUE"]:
        base = 12.0 if symbol == "SMR" else 140.0
        day = 0
        current = dt(2025, 1, 2)
        end = dt(2025, 2, 28)
        while current <= end:
            if current.weekday() < 5:  # Mon-Fri
                date = current.strftime("%Y-%m-%d")
                seed = int(hashlib.sha256(
                    f"{symbol}:{date}".encode()).hexdigest()[:4], 16)
                noise = (seed % 100 - 50) / 500.0  # ±10%
                price = base * (1 + noise + day * 0.002)
                conn.execute("""
                    INSERT INTO price_history
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, date,
                    round(price * 0.99, 2),   # open
                    round(price * 1.02, 2),   # high
                    round(price * 0.98, 2),   # low
                    round(price, 2),          # close
                    500000 + seed * 10,       # volume
                ))
                day += 1
            current += timedelta(days=1)

    conn.commit()
    return tmp.name, conn


def _make_trades(n=5, start_date="2025-01-06", symbol="SMR", conn=None):
    """Generate test trades with prices from actual DB data."""
    trades = []
    from datetime import datetime, timedelta
    d = datetime.fromisoformat(start_date)
    for i in range(n):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        date_str = d.strftime("%Y-%m-%d")

        # Use actual close price from DB if available
        price = 12.0 + i * 0.1  # fallback
        if conn is not None:
            row = conn.execute(
                "SELECT close FROM price_history WHERE UPPER(symbol) = UPPER(?) AND date = ?",
                (symbol, date_str)).fetchone()
            if row:
                price = float(row[0])

        trades.append({
            "date": date_str,
            "symbol": symbol,
            "action": "BUY" if i % 2 == 0 else "SELL",
            "price": price,
            "shares": 100,
            "signal_date": (d - timedelta(days=1)).strftime("%Y-%m-%d"),
        })
        d += timedelta(days=3)
    return trades


def _make_metrics(sharpe=1.5, win_rate=0.55, total_return=0.12,
                  max_drawdown=-0.08, profit_factor=1.8, **extra):
    """Generate test performance metrics."""
    m = {
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
    }
    m.update(extra)
    return m


# ===================================================================
# LeakageChecker
# ===================================================================

class TestLeakageChecker(unittest.TestCase):

    def setUp(self):
        self.db_path, self.conn = _make_test_db()
        self.checker = LeakageChecker(self.conn)

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_clean_trades_pass(self):
        trades = _make_trades(3, start_date="2025-01-06", conn=self.conn)
        report = self.checker.check_all(
            trades, "2025-01-01", "2025-02-28")
        self.assertTrue(report.passed)
        self.assertEqual(report.n_fatal_failures, 0)

    def test_trade_before_data_start(self):
        trades = [{"date": "2024-12-01", "symbol": "SMR",
                   "action": "BUY", "price": 12.0}]
        report = self.checker.check_all(
            trades, "2025-01-01", "2025-02-28")
        self.assertFalse(report.passed)
        self.assertGreater(report.n_fatal_failures, 0)

    def test_trade_after_data_end(self):
        trades = [{"date": "2025-06-01", "symbol": "SMR",
                   "action": "BUY", "price": 12.0}]
        report = self.checker.check_all(
            trades, "2025-01-01", "2025-02-28")
        self.assertFalse(report.passed)

    def test_future_signal_detected(self):
        trades = [{
            "date": "2025-01-06",
            "symbol": "SMR",
            "action": "BUY",
            "price": 12.0,
            "signal_date": "2025-01-10",  # signal AFTER trade!
        }]
        report = self.checker.check_all(
            trades, "2025-01-01", "2025-02-28")
        self.assertFalse(report.passed)

    def test_train_test_split(self):
        trades = _make_trades(6, start_date="2025-01-06", conn=self.conn)
        report = self.checker.check_all(
            trades, "2025-01-01", "2025-02-28", train_end="2025-01-15")
        # Should pass (just verifies split exists)
        split_check = [c for c in report.checks
                       if c.check_name == "train_test_split"]
        self.assertEqual(len(split_check), 1)

    def test_receipt_format(self):
        trades = _make_trades(2, conn=self.conn)
        report = self.checker.check_all(
            trades, "2025-01-01", "2025-02-28")
        receipt = report.to_receipt()
        self.assertIn("passed", receipt)
        self.assertIn("checks", receipt)
        self.assertIsInstance(receipt["checks"], list)


class TestLeakagePriceCheck(unittest.TestCase):
    """Test that price leakage is detected."""

    def setUp(self):
        self.db_path, self.conn = _make_test_db()
        self.checker = LeakageChecker(self.conn)

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_price_outside_range_detected(self):
        """Trade price wildly outside day's OHLC should be flagged."""
        # Get an actual date with data
        row = self.conn.execute(
            "SELECT date, high FROM price_history WHERE symbol='SMR' LIMIT 1"
        ).fetchone()
        date = row[0]
        high = float(row[1])

        trades = [{
            "date": date,
            "symbol": "SMR",
            "action": "BUY",
            "price": high * 2.0,  # way above high
            "shares": 100,
        }]
        report = self.checker.check_all(
            trades, "2025-01-01", "2025-02-28")
        future_price = [c for c in report.checks
                        if c.check_name == "no_future_price"]
        self.assertEqual(len(future_price), 1)
        self.assertFalse(future_price[0].passed)


# ===================================================================
# DataValidator
# ===================================================================

class TestDataValidator(unittest.TestCase):

    def setUp(self):
        self.db_path, self.conn = _make_test_db()
        self.validator = DataValidator(self.conn)

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_valid_data_passes(self):
        report = self.validator.validate(
            ["SMR", "NUE"], "2025-01-01", "2025-02-28",
            min_coverage_pct=60.0)  # test DB has ~72% coverage
        self.assertTrue(report.passed)
        self.assertGreater(report.coverage_pct, 0)

    def test_missing_symbol_fails(self):
        report = self.validator.validate(
            ["SMR", "FAKE_TICKER"], "2025-01-01", "2025-02-28")
        # FAKE_TICKER has no data → should fail
        self.assertFalse(report.passed)

    def test_coverage_in_receipt(self):
        report = self.validator.validate(
            ["SMR"], "2025-01-01", "2025-02-28")
        receipt = report.to_receipt()
        self.assertIn("coverage_pct", receipt)
        self.assertIn("symbols", receipt)
        self.assertIn("passed", receipt)

    def test_gap_detection(self):
        """Large gap in data should be detected."""
        # Delete some data to create a gap
        self.conn.execute("""
            DELETE FROM price_history
            WHERE symbol = 'SMR' AND date BETWEEN '2025-01-10' AND '2025-01-20'
        """)
        self.conn.commit()

        report = self.validator.validate(
            ["SMR"], "2025-01-01", "2025-02-28", max_gap_days=3)
        # Should detect the gap
        gap_checks = [c for c in report.checks if "gaps" in c.check_name]
        has_gap_failure = any(not c.passed for c in gap_checks)
        self.assertTrue(has_gap_failure)


# ===================================================================
# ResultAuditor
# ===================================================================

class TestResultAuditor(unittest.TestCase):

    def test_clean_results_pass(self):
        auditor = ResultAuditor()
        metrics = _make_metrics()
        report = auditor.audit(metrics, n_trades=20)
        self.assertFalse(report.rejected)

    def test_too_few_trades_rejected(self):
        auditor = ResultAuditor(min_trades=10)
        report = auditor.audit(_make_metrics(), n_trades=3)
        self.assertTrue(report.rejected)
        self.assertTrue(any("trades" in r for r in report.rejection_reasons))

    def test_suspicious_sharpe_rejected(self):
        auditor = ResultAuditor(max_sharpe=3.0)
        metrics = _make_metrics(sharpe=5.0)
        report = auditor.audit(metrics, n_trades=50)
        self.assertTrue(report.rejected)
        self.assertTrue(any("Sharpe" in r for r in report.rejection_reasons))

    def test_suspicious_win_rate_rejected(self):
        auditor = ResultAuditor(max_win_rate=0.80)
        metrics = _make_metrics(win_rate=0.95)
        report = auditor.audit(metrics, n_trades=30)
        self.assertTrue(report.rejected)

    def test_impossible_drawdown_rejected(self):
        metrics = _make_metrics(total_return=0.20, max_drawdown=0.0)
        auditor = ResultAuditor()
        report = auditor.audit(metrics, n_trades=20)
        self.assertTrue(report.rejected)

    def test_leakage_failure_rejected(self):
        auditor = ResultAuditor()
        report = auditor.audit(
            _make_metrics(), n_trades=20, leakage_passed=False)
        self.assertTrue(report.rejected)
        self.assertTrue(any("leakage" in r.lower()
                            for r in report.rejection_reasons))

    def test_low_coverage_rejected(self):
        auditor = ResultAuditor(min_data_coverage=90.0)
        report = auditor.audit(
            _make_metrics(), n_trades=20, data_coverage_pct=50.0)
        self.assertTrue(report.rejected)

    def test_receipt_format(self):
        auditor = ResultAuditor()
        report = auditor.audit(_make_metrics(), n_trades=20)
        receipt = report.to_receipt()
        self.assertIn("rejected", receipt)
        self.assertIn("flags", receipt)
        self.assertIsInstance(receipt["flags"], list)

    def test_normal_results_not_flagged(self):
        """Reasonable metrics should produce no rejections."""
        auditor = ResultAuditor()
        metrics = _make_metrics(
            sharpe=1.2, win_rate=0.52, total_return=0.08,
            max_drawdown=-0.12, profit_factor=1.4)
        report = auditor.audit(metrics, n_trades=50,
                               data_coverage_pct=97.0)
        self.assertFalse(report.rejected)
        self.assertEqual(len(report.rejection_reasons), 0)


# ===================================================================
# BenchmarkComparison
# ===================================================================

class TestBenchmarkComparison(unittest.TestCase):

    def setUp(self):
        self.db_path, self.conn = _make_test_db()

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_buy_hold_return(self):
        ret, sharpe = compute_buy_hold_return(
            self.conn, ["SMR", "NUE"], "2025-01-01", "2025-02-28")
        # Should be non-zero (price data exists)
        self.assertIsInstance(ret, float)

    def test_comparison_structure(self):
        bc = build_benchmark_comparison(
            self.conn, ["SMR"], "2025-01-01", "2025-02-28",
            strategy_return=0.15, strategy_sharpe=1.5)
        receipt = bc.to_receipt()
        self.assertIn("strategy_return", receipt)
        self.assertIn("buy_hold_return", receipt)
        self.assertIn("strategy_beats_buy_hold", receipt)

    def test_missing_symbol_returns_zero(self):
        ret, _ = compute_buy_hold_return(
            self.conn, ["FAKE"], "2025-01-01", "2025-02-28")
        self.assertEqual(ret, 0.0)


# ===================================================================
# ResearchReceipt
# ===================================================================

class TestResearchReceipt(unittest.TestCase):

    def test_receipt_schema(self):
        receipt = ResearchReceipt(
            receipt_id="test-001",
            thesis_id="nuclear-smr-thesis",
            generated_at="2026-03-15T00:00:00Z",
            symbols=["SMR"],
            data_start="2025-01-01",
            data_end="2025-12-31",
            train_end="2025-06-30",
            signal_rules="conviction >= 3",
            position_sizing="conviction_based",
            slippage_bps=10.0,
            commission_per_trade=0.0,
            n_trades=25,
            metrics=_make_metrics(),
            leakage={"passed": True, "checks": []},
            data_quality={"passed": True, "checks": []},
            audit={"rejected": False, "flags": []},
            benchmark={"strategy_return": 0.12, "buy_hold_return": 0.08},
            integrity_passed=True,
            rejection_reasons=[],
            cost={"wall_time_s": 0.5, "cpu_time_s": 0.3},
        )
        d = receipt.to_dict()
        self.assertEqual(d["schema"], SCHEMA)
        self.assertEqual(d["thesis_id"], "nuclear-smr-thesis")
        self.assertIn("universe", d)
        self.assertIn("rules", d)
        self.assertIn("performance", d)
        self.assertIn("integrity", d)
        self.assertIn("benchmark", d)
        self.assertIn("verdict", d)
        self.assertIn("cost", d)

    def test_save_receipt(self):
        receipt = ResearchReceipt(
            receipt_id="test-002",
            thesis_id="test",
            generated_at="2026-03-15T00:00:00Z",
            symbols=["SMR"],
            data_start="2025-01-01",
            data_end="2025-12-31",
            train_end=None,
            signal_rules="test",
            position_sizing="equal",
            slippage_bps=10.0,
            commission_per_trade=0.0,
            n_trades=10,
            metrics=_make_metrics(),
            leakage={"passed": True},
            data_quality={"passed": True},
            audit={"rejected": False},
            benchmark={},
            integrity_passed=True,
            rejection_reasons=[],
            cost={"wall_time_s": 0.1},
        )
        tmp_dir = tempfile.mkdtemp()
        path = receipt.save(tmp_dir)
        self.assertTrue(os.path.exists(path))

        with open(path) as f:
            data = json.load(f)
        self.assertIn("sha256", data)
        self.assertEqual(data["schema"], SCHEMA)

        # Cleanup
        os.unlink(path)
        os.rmdir(tmp_dir)


# ===================================================================
# Full Pipeline
# ===================================================================

class TestFullPipeline(unittest.TestCase):
    """Test run_integrity_pipeline end-to-end."""

    def setUp(self):
        self.db_path, self.conn = _make_test_db()

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_clean_backtest_passes(self):
        trades = _make_trades(12, start_date="2025-01-06", conn=self.conn)
        metrics = _make_metrics()

        receipt = run_integrity_pipeline(
            conn=self.conn,
            thesis_id="nuclear-smr-thesis",
            symbols=["SMR"],
            trades=trades,
            metrics=metrics,
            data_start="2025-01-01",
            data_end="2025-02-28",
        )

        self.assertTrue(receipt.integrity_passed)
        self.assertEqual(len(receipt.rejection_reasons), 0)
        self.assertEqual(receipt.thesis_id, "nuclear-smr-thesis")
        self.assertIn("wall_time_s", receipt.cost)

    def test_too_few_trades_rejected(self):
        trades = _make_trades(2, start_date="2025-01-06", conn=self.conn)
        metrics = _make_metrics()

        receipt = run_integrity_pipeline(
            conn=self.conn,
            thesis_id="test",
            symbols=["SMR"],
            trades=trades,
            metrics=metrics,
            data_start="2025-01-01",
            data_end="2025-02-28",
        )

        self.assertFalse(receipt.integrity_passed)
        self.assertGreater(len(receipt.rejection_reasons), 0)

    def test_leakage_rejected(self):
        """Future signal date should cause rejection."""
        from datetime import datetime, timedelta
        trades = [{
            "date": "2025-01-06",
            "symbol": "SMR",
            "action": "BUY",
            "price": 12.0,
            "shares": 100,
            "signal_date": "2025-01-10",  # FUTURE signal
        }] * 12  # enough trades to pass min_trades

        receipt = run_integrity_pipeline(
            conn=self.conn,
            thesis_id="test",
            symbols=["SMR"],
            trades=trades,
            metrics=_make_metrics(),
            data_start="2025-01-01",
            data_end="2025-02-28",
        )

        self.assertFalse(receipt.integrity_passed)
        self.assertIn("leakage_detected", receipt.rejection_reasons)

    def test_suspicious_sharpe_rejected(self):
        trades = _make_trades(15, start_date="2025-01-06", conn=self.conn)
        metrics = _make_metrics(sharpe=6.0)  # impossibly high

        receipt = run_integrity_pipeline(
            conn=self.conn,
            thesis_id="test",
            symbols=["SMR"],
            trades=trades,
            metrics=metrics,
            data_start="2025-01-01",
            data_end="2025-02-28",
        )

        self.assertFalse(receipt.integrity_passed)

    def test_receipt_saveable(self):
        trades = _make_trades(5, conn=self.conn)
        receipt = run_integrity_pipeline(
            conn=self.conn,
            thesis_id="save-test",
            symbols=["SMR"],
            trades=trades,
            metrics=_make_metrics(),
            data_start="2025-01-01",
            data_end="2025-02-28",
        )

        tmp_dir = tempfile.mkdtemp()
        path = receipt.save(tmp_dir)
        self.assertTrue(os.path.exists(path))

        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["schema"], SCHEMA)
        self.assertIn("integrity", data)

        os.unlink(path)
        os.rmdir(tmp_dir)

    def test_with_train_test_split(self):
        trades = _make_trades(12, start_date="2025-01-06", conn=self.conn)
        receipt = run_integrity_pipeline(
            conn=self.conn,
            thesis_id="split-test",
            symbols=["SMR"],
            trades=trades,
            metrics=_make_metrics(),
            data_start="2025-01-01",
            data_end="2025-02-28",
            train_end="2025-01-31",
        )

        self.assertEqual(receipt.train_end, "2025-01-31")
        # Leakage report should include train/test split check
        self.assertIn("checks", receipt.leakage)

    def test_benchmark_included(self):
        trades = _make_trades(5, conn=self.conn)
        receipt = run_integrity_pipeline(
            conn=self.conn,
            thesis_id="bench-test",
            symbols=["SMR", "NUE"],
            trades=trades,
            metrics=_make_metrics(),
            data_start="2025-01-01",
            data_end="2025-02-28",
        )

        self.assertIn("strategy_return", receipt.benchmark)
        self.assertIn("buy_hold_return", receipt.benchmark)


if __name__ == "__main__":
    unittest.main()
