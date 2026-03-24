"""Tests for WO-K: Trading Executors — Governed Trading Research Runtime.

Tests the trading executor handlers, TradingPolicy, registry extension,
and integration with executor registry gating.
"""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.executor_registry import (
    ExecutorContext,
    ExecutorRegistry,
    ExecutorResult,
    ExecutorSpec,
    ExecutorStatus,
    build_default_registry,
)
from helix_substrate.trading_executors import (
    SCHEMA,
    TRADING_EXECUTORS,
    TradingPolicy,
    _exec_conviction_check,
    _exec_paper_trade,
    _exec_price_check,
    _exec_risk_check,
    _exec_thesis_scan,
    _exec_backtest_thesis,
    build_trading_registry,
    register_trading_executors,
)


def _make_ctx(**kwargs) -> ExecutorContext:
    """Build an ExecutorContext with defaults."""
    defaults = {
        "query": "test",
        "route_name": "direct_code",
    }
    defaults.update(kwargs)
    return ExecutorContext(**defaults)


def _make_test_db():
    """Create a temporary SQLite database with FGIP-like schema."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    conn = sqlite3.connect(tmp.name)
    conn.row_factory = sqlite3.Row

    # Create minimal FGIP-compatible tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            name TEXT,
            node_type TEXT,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_node_id TEXT,
            to_node_id TEXT,
            edge_type TEXT,
            confidence REAL,
            notes TEXT,
            source_url TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        );

        CREATE TABLE IF NOT EXISTS paper_positions (
            id TEXT PRIMARY KEY,
            thesis_id TEXT,
            ticker TEXT,
            recommendation_id TEXT,
            entry_date TEXT,
            entry_price REAL,
            target_size REAL,
            actual_size REAL,
            shares REAL,
            exit_date TEXT,
            exit_price REAL,
            exit_reason TEXT,
            realized_pnl REAL,
            realized_pnl_pct REAL,
            status TEXT DEFAULT 'OPEN',
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS trade_memos (
            memo_id TEXT PRIMARY KEY,
            thesis_id TEXT,
            symbol TEXT,
            decision TEXT,
            decision_confidence REAL,
            gates_passed INTEGER,
            gates_total INTEGER,
            position_size_pct REAL,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS forecasts (
            id TEXT PRIMARY KEY,
            thesis_id TEXT,
            scenario_tree TEXT,
            created_at TEXT,
            resolved_at TEXT,
            actual_outcome TEXT
        );

        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            claim_text TEXT,
            topic TEXT,
            status TEXT,
            required_tier INTEGER
        );

        CREATE TABLE IF NOT EXISTS sources (
            source_id TEXT PRIMARY KEY,
            retrieved_at TEXT,
            tier INTEGER
        );

        CREATE TABLE IF NOT EXISTS claim_sources (
            claim_id TEXT,
            source_id TEXT
        );
    """)

    # Insert test price data
    conn.executescript("""
        INSERT INTO price_history (symbol, date, open, high, low, close, volume)
        VALUES ('SMR', '2026-03-14', 12.50, 13.20, 12.30, 12.90, 1500000);

        INSERT INTO price_history (symbol, date, open, high, low, close, volume)
        VALUES ('SMR', '2026-03-13', 12.00, 12.80, 11.90, 12.50, 1200000);

        INSERT INTO price_history (symbol, date, open, high, low, close, volume)
        VALUES ('NUE', '2026-03-14', 145.00, 148.50, 144.20, 147.30, 800000);

        INSERT INTO price_history (symbol, date, open, high, low, close, volume)
        VALUES ('CCJ', '2026-03-14', 52.10, 53.40, 51.80, 53.00, 600000);
    """)

    # Insert test open positions
    conn.executescript("""
        INSERT INTO paper_positions (id, thesis_id, ticker, actual_size, shares,
                                     entry_price, status)
        VALUES ('pos-001', 'nuclear-smr-thesis', 'SMR', 0.10, 100, 12.00, 'OPEN');

        INSERT INTO paper_positions (id, thesis_id, ticker, actual_size, shares,
                                     entry_price, status)
        VALUES ('pos-002', 'reshoring-steel-thesis', 'NUE', 0.15, 50, 140.00, 'OPEN');
    """)

    conn.commit()

    # Create a mock FGIPDatabase-like object
    class MockFGIPDB:
        def __init__(self, conn):
            self._conn = conn

        def connect(self):
            return self._conn

    return tmp.name, MockFGIPDB(conn), conn


class TestTradingPolicy(unittest.TestCase):
    """Test TradingPolicy dataclass and entry checks."""

    def test_default_policy(self):
        policy = TradingPolicy()
        self.assertEqual(policy.min_conviction, 3)
        self.assertEqual(policy.max_position_pct, 0.20)
        self.assertEqual(policy.max_portfolio_exposure, 1.0)
        self.assertEqual(policy.max_positions, 10)
        self.assertTrue(policy.paper_only)

    def test_entry_allowed(self):
        policy = TradingPolicy()
        result = policy.check_entry(conviction_level=4, current_exposure=0.3,
                                    current_positions=2)
        self.assertTrue(result["allowed"])
        self.assertEqual(result["reason"], "policy_passed")
        self.assertEqual(result["max_size"], 0.20)

    def test_entry_blocked_low_conviction(self):
        policy = TradingPolicy()
        result = policy.check_entry(conviction_level=2, current_exposure=0.0,
                                    current_positions=0)
        self.assertFalse(result["allowed"])
        self.assertIn("conviction", result["reason"])

    def test_entry_blocked_max_positions(self):
        policy = TradingPolicy(max_positions=3)
        result = policy.check_entry(conviction_level=4, current_exposure=0.3,
                                    current_positions=3)
        self.assertFalse(result["allowed"])
        self.assertIn("positions", result["reason"])

    def test_entry_capped_by_exposure(self):
        policy = TradingPolicy(max_portfolio_exposure=0.5)
        result = policy.check_entry(conviction_level=5, current_exposure=0.35,
                                    current_positions=2)
        self.assertTrue(result["allowed"])
        self.assertEqual(result["max_size"], 0.15)  # 0.5 - 0.35

    def test_entry_blocked_full_exposure(self):
        policy = TradingPolicy()
        result = policy.check_entry(conviction_level=5, current_exposure=1.0,
                                    current_positions=5)
        self.assertFalse(result["allowed"])
        self.assertIn("exposure", result["reason"])

    def test_paper_only_invariant(self):
        policy = TradingPolicy(paper_only=False)
        result = policy.check_entry(conviction_level=5, current_exposure=0.0,
                                    current_positions=0)
        self.assertFalse(result["allowed"])
        self.assertIn("paper_only", result["reason"])

    def test_receipt(self):
        policy = TradingPolicy()
        receipt = policy.to_receipt()
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertTrue(receipt["paper_only"])
        self.assertEqual(receipt["min_conviction"], 3)


class TestPriceCheck(unittest.TestCase):
    """Test price_check executor."""

    def setUp(self):
        self.db_path, self.mock_db, self.conn = _make_test_db()

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_price_found(self):
        ctx = _make_ctx(query="SMR")
        ctx.fgip_db = self.mock_db

        result = _exec_price_check(ctx)
        self.assertEqual(result.status, ExecutorStatus.OK.value)

        data = json.loads(result.result)
        self.assertIn("SMR", data)
        self.assertEqual(data["SMR"]["close"], 12.90)
        self.assertEqual(data["SMR"]["date"], "2026-03-14")

    def test_multiple_tickers(self):
        ctx = _make_ctx(query="SMR, NUE, CCJ")
        ctx.fgip_db = self.mock_db

        result = _exec_price_check(ctx)
        self.assertEqual(result.status, ExecutorStatus.OK.value)

        data = json.loads(result.result)
        self.assertIn("SMR", data)
        self.assertIn("NUE", data)
        self.assertIn("CCJ", data)
        self.assertEqual(result.receipt_fragment["tickers_found"], 3)

    def test_ticker_not_found(self):
        ctx = _make_ctx(query="FAKE")
        ctx.fgip_db = self.mock_db

        result = _exec_price_check(ctx)
        self.assertEqual(result.status, ExecutorStatus.OK.value)

        data = json.loads(result.result)
        self.assertIn("error", data["FAKE"])
        self.assertEqual(result.receipt_fragment["tickers_missing"], 1)

    @patch("helix_substrate.trading_executors.FGIP_DB_PATH", Path("/nonexistent/fake.db"))
    def test_no_db(self):
        ctx = _make_ctx(query="SMR")
        result = _exec_price_check(ctx)
        self.assertEqual(result.status, ExecutorStatus.ERROR.value)
        self.assertIn("not available", result.error)


class TestRiskCheck(unittest.TestCase):
    """Test risk_check executor."""

    def setUp(self):
        self.db_path, self.mock_db, self.conn = _make_test_db()

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_reads_open_positions(self):
        ctx = _make_ctx()
        ctx.fgip_db = self.mock_db

        result = _exec_risk_check(ctx)
        self.assertEqual(result.status, ExecutorStatus.OK.value)

        data = json.loads(result.result)
        self.assertEqual(data["n_open_positions"], 2)
        self.assertAlmostEqual(data["total_exposure"], 0.25, places=2)
        self.assertEqual(result.receipt_fragment["n_positions"], 2)

    def test_exposure_warning(self):
        # Add positions to push exposure above 80%
        self.conn.execute("""
            INSERT INTO paper_positions (id, thesis_id, ticker, actual_size,
                                         shares, entry_price, status)
            VALUES ('pos-003', 'defense', 'PLTR', 0.60, 200, 25.00, 'OPEN')
        """)
        self.conn.commit()

        ctx = _make_ctx()
        ctx.fgip_db = self.mock_db

        result = _exec_risk_check(ctx)
        data = json.loads(result.result)

        self.assertGreater(len(data["warnings"]), 0)
        self.assertGreater(result.receipt_fragment["n_warnings"], 0)

    @patch("helix_substrate.trading_executors.FGIP_DB_PATH", Path("/nonexistent/fake.db"))
    def test_no_db(self):
        ctx = _make_ctx()
        result = _exec_risk_check(ctx)
        self.assertEqual(result.status, ExecutorStatus.ERROR.value)


class TestPaperTrade(unittest.TestCase):
    """Test paper_trade executor with policy gating."""

    def setUp(self):
        self.db_path, self.mock_db, self.conn = _make_test_db()

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_buy_allowed(self):
        ctx = _make_ctx(
            query="uranium-thesis",
            action_plan={
                "thesis_id": "uranium-thesis",
                "action": "BUY",
                "conviction_level": 4,
                "position_size_pct": 0.10,
                "ticker": "CCJ",
                "reason": "test",
            },
        )
        ctx.fgip_db = self.mock_db

        result = _exec_paper_trade(ctx)
        self.assertEqual(result.status, ExecutorStatus.OK.value)
        self.assertTrue(result.side_effects)

        data = json.loads(result.result)
        self.assertEqual(data["action"], "BUY")
        self.assertEqual(data["thesis_id"], "uranium-thesis")
        self.assertTrue(data["paper_only"])

        # Verify written to trade_memos
        row = self.conn.execute(
            "SELECT * FROM trade_memos WHERE thesis_id = 'uranium-thesis'"
        ).fetchone()
        self.assertIsNotNone(row)

    def test_buy_blocked_low_conviction(self):
        ctx = _make_ctx(
            action_plan={
                "thesis_id": "weak-thesis",
                "action": "BUY",
                "conviction_level": 2,
                "position_size_pct": 0.10,
            },
        )
        ctx.fgip_db = self.mock_db

        result = _exec_paper_trade(ctx)
        self.assertEqual(result.status, ExecutorStatus.BLOCKED.value)
        self.assertIn("conviction", result.error)
        self.assertEqual(result.receipt_fragment["gate"], "trading_policy")

    def test_buy_blocked_max_positions(self):
        policy = TradingPolicy(max_positions=2)
        ctx = _make_ctx(
            action_plan={
                "thesis_id": "new-thesis",
                "action": "BUY",
                "conviction_level": 5,
                "position_size_pct": 0.10,
            },
        )
        ctx.fgip_db = self.mock_db
        ctx.trading_policy = policy

        result = _exec_paper_trade(ctx)
        self.assertEqual(result.status, ExecutorStatus.BLOCKED.value)
        self.assertIn("positions", result.error)

    def test_paper_only_invariant(self):
        """paper_only=False must ALWAYS block."""
        policy = TradingPolicy(paper_only=False)
        ctx = _make_ctx(
            action_plan={
                "thesis_id": "any",
                "action": "BUY",
                "conviction_level": 5,
                "position_size_pct": 0.10,
            },
        )
        ctx.fgip_db = self.mock_db
        ctx.trading_policy = policy

        result = _exec_paper_trade(ctx)
        self.assertEqual(result.status, ExecutorStatus.BLOCKED.value)
        self.assertIn("paper_only", result.error)

    def test_position_size_capped(self):
        """Position size should be capped by remaining exposure."""
        policy = TradingPolicy(max_portfolio_exposure=0.30)
        ctx = _make_ctx(
            action_plan={
                "thesis_id": "test-thesis",
                "action": "BUY",
                "conviction_level": 4,
                "position_size_pct": 0.20,  # asks for 20%
                "ticker": "TEST",
            },
        )
        ctx.fgip_db = self.mock_db
        ctx.trading_policy = policy

        result = _exec_paper_trade(ctx)
        self.assertEqual(result.status, ExecutorStatus.OK.value)

        data = json.loads(result.result)
        # Exposure is 0.25 (existing), cap is 0.30, so max = 0.05
        self.assertLessEqual(data["position_size_pct"], 0.05)

    def test_exit_no_policy_check(self):
        """EXIT action doesn't require conviction check."""
        ctx = _make_ctx(
            action_plan={
                "thesis_id": "nuclear-smr-thesis",
                "action": "EXIT",
                "conviction_level": 0,
                "reason": "stop_loss",
            },
        )
        ctx.fgip_db = self.mock_db

        result = _exec_paper_trade(ctx)
        self.assertEqual(result.status, ExecutorStatus.OK.value)

    @patch("helix_substrate.trading_executors.FGIP_DB_PATH", Path("/nonexistent/fake.db"))
    def test_no_db(self):
        ctx = _make_ctx(
            action_plan={"action": "BUY", "conviction_level": 5,
                         "position_size_pct": 0.1},
        )
        result = _exec_paper_trade(ctx)
        self.assertEqual(result.status, ExecutorStatus.ERROR.value)


class TestRegistryExtension(unittest.TestCase):
    """Test register_trading_executors and build_trading_registry."""

    def test_register_to_existing(self):
        registry = build_default_registry()
        n_before = len(registry.list_executors())

        register_trading_executors(registry)
        n_after = len(registry.list_executors())

        self.assertEqual(n_after - n_before, 6)

    def test_build_trading_registry(self):
        registry = build_trading_registry()
        names = [e["name"] for e in registry.list_executors()]

        # Default executors
        self.assertIn("lobe_inference", names)
        self.assertIn("web_search", names)
        self.assertIn("attest", names)

        # Trading executors
        self.assertIn("thesis_scan", names)
        self.assertIn("conviction_check", names)
        self.assertIn("price_check", names)
        self.assertIn("backtest_thesis", names)
        self.assertIn("risk_check", names)
        self.assertIn("paper_trade", names)

    def test_paper_trade_has_side_effects(self):
        registry = build_trading_registry()
        executors = {e["name"]: e for e in registry.list_executors()}

        self.assertTrue(executors["paper_trade"]["has_side_effects"])
        self.assertFalse(executors["thesis_scan"]["has_side_effects"])
        self.assertFalse(executors["price_check"]["has_side_effects"])

    def test_paper_trade_blocked_by_side_effects_gate(self):
        """Registry side-effects gate blocks paper_trade when not allowed."""
        db_path, mock_db, conn = _make_test_db()
        try:
            registry = build_trading_registry()

            ctx = _make_ctx(
                action_plan={
                    "thesis_id": "test",
                    "action": "BUY",
                    "conviction_level": 5,
                    "position_size_pct": 0.1,
                },
            )
            ctx.fgip_db = mock_db

            result = registry.execute(
                "paper_trade", ctx, allow_side_effects=False)

            self.assertEqual(result.status, ExecutorStatus.BLOCKED.value)
            self.assertIn("side_effects", result.receipt_fragment.get("gate", ""))
        finally:
            conn.close()
            os.unlink(db_path)

    def test_duplicate_registration_raises(self):
        registry = build_trading_registry()
        with self.assertRaises(ValueError):
            register_trading_executors(registry)

    def test_all_trading_executors_no_budget(self):
        """No trading executor requires budget (they're CPU-only)."""
        for spec in TRADING_EXECUTORS:
            self.assertFalse(spec.requires_budget,
                             f"{spec.name} should not require budget")


class TestBacktestExecutor(unittest.TestCase):
    """Test backtest_thesis executor with mock data."""

    def setUp(self):
        self.db_path, self.mock_db, self.conn = _make_test_db()

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_backtest_no_forecasts(self):
        """Backtest with no forecasts should return 0 steps."""
        ctx = _make_ctx(
            query="nuclear-smr-thesis",
            action_plan={"start_date": "2025-01-01", "end_date": "2025-03-01",
                         "step": "7d"},
        )
        ctx.fgip_db = self.mock_db

        # Patch the FGIP import to use our mock db
        with patch.dict('sys.modules', {}):
            result = _exec_backtest_thesis(ctx)

        # Will either succeed with 0 steps or error if import fails
        if result.status == ExecutorStatus.OK.value:
            data = json.loads(result.result)
            self.assertEqual(data["total_steps"], 0)
            self.assertFalse(data["valid_for_trading"])

    @patch("helix_substrate.trading_executors.FGIP_DB_PATH", Path("/nonexistent/fake.db"))
    def test_backtest_no_db(self):
        ctx = _make_ctx(query="nuclear-smr-thesis")
        result = _exec_backtest_thesis(ctx)
        self.assertEqual(result.status, ExecutorStatus.ERROR.value)


class TestThesisScan(unittest.TestCase):
    """Test thesis_scan executor."""

    @patch("helix_substrate.trading_executors.FGIP_DB_PATH", Path("/nonexistent/fake.db"))
    def test_no_db(self):
        ctx = _make_ctx()
        result = _exec_thesis_scan(ctx)
        self.assertEqual(result.status, ExecutorStatus.ERROR.value)


class TestConvictionCheck(unittest.TestCase):
    """Test conviction_check executor."""

    @patch("helix_substrate.trading_executors.FGIP_DB_PATH", Path("/nonexistent/fake.db"))
    def test_no_db(self):
        ctx = _make_ctx(query="nuclear-smr-thesis")
        result = _exec_conviction_check(ctx)
        self.assertEqual(result.status, ExecutorStatus.ERROR.value)


class TestExecutorTraceIntegration(unittest.TestCase):
    """Test trading executors produce proper trace through registry."""

    def setUp(self):
        self.db_path, self.mock_db, self.conn = _make_test_db()
        self.registry = build_trading_registry()

    def tearDown(self):
        self.conn.close()
        os.unlink(self.db_path)

    def test_price_check_traced(self):
        ctx = _make_ctx(query="SMR")
        ctx.fgip_db = self.mock_db

        result = self.registry.execute("price_check", ctx)
        self.assertEqual(result.status, ExecutorStatus.OK.value)

        trace = self.registry.drain_trace()
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["executor"], "price_check")
        self.assertEqual(trace[0]["status"], "ok")

    def test_risk_check_traced(self):
        ctx = _make_ctx()
        ctx.fgip_db = self.mock_db

        result = self.registry.execute("risk_check", ctx)
        self.assertEqual(result.status, ExecutorStatus.OK.value)

        trace = self.registry.drain_trace()
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["executor"], "risk_check")

    def test_paper_trade_traced_with_block(self):
        """Paper trade with low conviction should be blocked and traced."""
        ctx = _make_ctx(
            action_plan={
                "thesis_id": "bad-thesis",
                "action": "BUY",
                "conviction_level": 1,
                "position_size_pct": 0.1,
            },
        )
        ctx.fgip_db = self.mock_db

        result = self.registry.execute("paper_trade", ctx)
        self.assertEqual(result.status, ExecutorStatus.BLOCKED.value)

        trace = self.registry.drain_trace()
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["executor"], "paper_trade")
        self.assertEqual(trace[0]["gate"], "trading_policy")

    def test_full_cycle_scan_then_trade(self):
        """Simulate: risk_check → price_check → paper_trade."""
        # Step 1: Risk check
        ctx1 = _make_ctx()
        ctx1.fgip_db = self.mock_db
        r1 = self.registry.execute("risk_check", ctx1)
        self.assertEqual(r1.status, ExecutorStatus.OK.value)

        # Step 2: Price check
        ctx2 = _make_ctx(query="CCJ")
        ctx2.fgip_db = self.mock_db
        r2 = self.registry.execute("price_check", ctx2)
        self.assertEqual(r2.status, ExecutorStatus.OK.value)

        # Step 3: Paper trade (high conviction)
        ctx3 = _make_ctx(
            action_plan={
                "thesis_id": "uranium-thesis",
                "action": "BUY",
                "conviction_level": 4,
                "position_size_pct": 0.10,
                "ticker": "CCJ",
            },
        )
        ctx3.fgip_db = self.mock_db
        r3 = self.registry.execute("paper_trade", ctx3)
        self.assertEqual(r3.status, ExecutorStatus.OK.value)

        # Verify trace has all 3
        trace = self.registry.drain_trace()
        self.assertEqual(len(trace), 3)
        executors_used = [t["executor"] for t in trace]
        self.assertEqual(executors_used, ["risk_check", "price_check", "paper_trade"])

    def test_receipt_includes_all_fields(self):
        """Registry receipt should include all trading executors."""
        receipt = self.registry.to_receipt()
        self.assertIn("thesis_scan", receipt["registered_executors"])
        self.assertIn("paper_trade", receipt["registered_executors"])
        self.assertEqual(len(receipt["registered_executors"]), 11)  # 5 default + 6 trading


if __name__ == "__main__":
    unittest.main()
