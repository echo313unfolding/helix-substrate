"""
Trading Executors — WO-K

Governed trading research runtime. Bridges the executor registry to the
existing FGIP analytical engine for thesis evaluation, backtesting, and
paper trading.

Every trading action goes through the executor registry's 3-gate chain
(registration → side-effects → budget), plus a trading-specific policy
layer that enforces:
    - No live trades (paper only)
    - Conviction threshold for position entry
    - Position size limits
    - Portfolio exposure caps

Built-in trading executors:
    thesis_scan      — scan all theses for conviction levels
    conviction_check — evaluate single thesis via ConvictionEngine
    price_check      — get latest OHLCV from price_history table
    backtest_thesis  — run walk-forward backtest
    risk_check       — check position limits and portfolio exposure
    paper_trade      — record a paper trade decision (side effects!)

Work Order: WO-K (Governed Trading Research Runtime)
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .executor_registry import (
    ExecutorContext,
    ExecutorRegistry,
    ExecutorResult,
    ExecutorSpec,
    ExecutorStatus,
)

SCHEMA = "trading_executors:v1"

# FGIP database default path
FGIP_DB_PATH = Path("/home/voidstr3m33/fgip-engine/fgip.db")


# ---------------------------------------------------------------------------
# Trading Policy
# ---------------------------------------------------------------------------

@dataclass
class TradingPolicy:
    """Governs trading research decisions. Paper-only, never live."""

    # Position entry
    min_conviction: int = 3              # Minimum ConvictionLevel to enter
    max_position_pct: float = 0.20       # Max single position (20%)
    max_portfolio_exposure: float = 1.0  # Max total exposure (100%)
    max_positions: int = 10              # Max concurrent positions

    # Backtest
    min_backtest_steps: int = 4          # Minimum steps for valid backtest

    # Hard invariant: NEVER live trade
    paper_only: bool = True

    def check_entry(self, conviction_level: int, current_exposure: float,
                    current_positions: int) -> Dict[str, Any]:
        """Check if a paper trade entry is allowed by policy.

        Returns dict with 'allowed', 'reason', and 'max_size'.
        """
        if not self.paper_only:
            return {"allowed": False, "reason": "paper_only invariant violated",
                    "max_size": 0.0}

        if conviction_level < self.min_conviction:
            return {"allowed": False,
                    "reason": f"conviction {conviction_level} < min {self.min_conviction}",
                    "max_size": 0.0}

        if current_positions >= self.max_positions:
            return {"allowed": False,
                    "reason": f"positions {current_positions} >= max {self.max_positions}",
                    "max_size": 0.0}

        remaining_exposure = max(0.0, self.max_portfolio_exposure - current_exposure)
        max_size = min(self.max_position_pct, remaining_exposure)

        if max_size < 0.01:
            return {"allowed": False, "reason": "no remaining exposure",
                    "max_size": 0.0}

        return {"allowed": True, "reason": "policy_passed",
                "max_size": round(max_size, 4)}

    def to_receipt(self) -> dict:
        return {
            "schema": SCHEMA,
            "min_conviction": self.min_conviction,
            "max_position_pct": self.max_position_pct,
            "max_portfolio_exposure": self.max_portfolio_exposure,
            "max_positions": self.max_positions,
            "min_backtest_steps": self.min_backtest_steps,
            "paper_only": self.paper_only,
        }


# ---------------------------------------------------------------------------
# FGIP bridge — thin wrappers around FGIP engine functions
# ---------------------------------------------------------------------------

def _get_fgip_db(ctx: ExecutorContext) -> Optional[Any]:
    """Get FGIP database from context or default path."""
    # Check if fgip_db is injected via context
    fgip_db = getattr(ctx, 'fgip_db', None)
    if fgip_db is not None:
        return fgip_db

    # Fall back to default path
    if not FGIP_DB_PATH.exists():
        return None

    # Lazy import to avoid hard dependency
    try:
        import sys
        fgip_root = str(FGIP_DB_PATH.parent)
        if fgip_root not in sys.path:
            sys.path.insert(0, fgip_root)
        from fgip.db import FGIPDatabase
        db = FGIPDatabase(str(FGIP_DB_PATH))
        db.connect()
        return db
    except Exception:
        return None


def _get_trading_policy(ctx: ExecutorContext) -> TradingPolicy:
    """Get trading policy from context or default."""
    policy = getattr(ctx, 'trading_policy', None)
    if isinstance(policy, TradingPolicy):
        return policy
    return TradingPolicy()


# ---------------------------------------------------------------------------
# Executor handlers
# ---------------------------------------------------------------------------

def _exec_thesis_scan(ctx: ExecutorContext) -> ExecutorResult:
    """Scan all theses for conviction levels.

    Returns summary of all thesis evaluations with conviction scores.
    Uses ConvictionEngine.evaluate_all_theses() from FGIP.
    """
    db = _get_fgip_db(ctx)
    if db is None:
        return ExecutorResult(
            executor_name="thesis_scan",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed"},
            error="FGIP database not available",
        )

    try:
        import sys
        fgip_root = str(FGIP_DB_PATH.parent)
        if fgip_root not in sys.path:
            sys.path.insert(0, fgip_root)
        from fgip.agents.conviction_engine import ConvictionEngine, INVESTMENT_THESES

        engine = ConvictionEngine(db)
        reports = engine.evaluate_all_theses()

        summaries = []
        for r in reports:
            summaries.append({
                "thesis_id": r.thesis_id,
                "conviction_level": r.conviction_level,
                "conviction_score": r.conviction_score,
                "position_size_pct": r.position_size_pct,
                "recommendation": r.recommendation,
                "triangulation_met": r.triangulation_met,
                "counter_thesis_severity": r.counter_thesis_severity,
                "confirming_signals": len(r.confirming_signals),
                "refuting_signals": len(r.refuting_signals),
            })

        result_text = json.dumps(summaries, indent=2)

        return ExecutorResult(
            executor_name="thesis_scan",
            status=ExecutorStatus.OK.value,
            result=result_text,
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={
                "gate": "passed",
                "n_theses": len(summaries),
                "theses_above_min": sum(
                    1 for s in summaries
                    if s["conviction_level"] >= _get_trading_policy(ctx).min_conviction
                ),
                "total_theses": len(INVESTMENT_THESES),
            },
        )
    except Exception as e:
        return ExecutorResult(
            executor_name="thesis_scan",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed", "error_type": type(e).__name__},
            error=str(e),
        )


def _exec_conviction_check(ctx: ExecutorContext) -> ExecutorResult:
    """Evaluate a single thesis via ConvictionEngine.

    Expects ctx.query to contain the thesis_id.
    Returns full ConvictionReport as JSON.
    """
    thesis_id = ctx.query.strip()

    db = _get_fgip_db(ctx)
    if db is None:
        return ExecutorResult(
            executor_name="conviction_check",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed"},
            error="FGIP database not available",
        )

    try:
        import sys
        fgip_root = str(FGIP_DB_PATH.parent)
        if fgip_root not in sys.path:
            sys.path.insert(0, fgip_root)
        from fgip.agents.conviction_engine import ConvictionEngine

        engine = ConvictionEngine(db)
        report = engine.evaluate_thesis(thesis_id)

        result_text = json.dumps(report.to_dict(), indent=2, default=str)

        return ExecutorResult(
            executor_name="conviction_check",
            status=ExecutorStatus.OK.value,
            result=result_text,
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={
                "gate": "passed",
                "thesis_id": thesis_id,
                "conviction_level": report.conviction_level,
                "conviction_score": report.conviction_score,
                "position_size_pct": report.position_size_pct,
                "recommendation": report.recommendation,
                "triangulation_met": report.triangulation_met,
                "confirming_signals": len(report.confirming_signals),
                "refuting_signals": len(report.refuting_signals),
            },
        )
    except Exception as e:
        return ExecutorResult(
            executor_name="conviction_check",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed", "thesis_id": thesis_id,
                              "error_type": type(e).__name__},
            error=str(e),
        )


def _exec_price_check(ctx: ExecutorContext) -> ExecutorResult:
    """Get latest price data for tickers from FGIP price_history table.

    Expects ctx.query to contain comma-separated ticker symbols.
    Returns OHLCV data for each ticker.
    """
    tickers = [t.strip().upper() for t in ctx.query.split(",") if t.strip()]

    db = _get_fgip_db(ctx)
    if db is None:
        return ExecutorResult(
            executor_name="price_check",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed"},
            error="FGIP database not available",
        )

    try:
        conn = db.connect()
        prices = {}

        for ticker in tickers:
            row = conn.execute("""
                SELECT symbol, date, open, high, low, close, volume
                FROM price_history
                WHERE UPPER(symbol) = ?
                ORDER BY date DESC LIMIT 1
            """, (ticker,)).fetchone()

            if row:
                prices[ticker] = {
                    "symbol": row["symbol"] if hasattr(row, "keys") else row[0],
                    "date": row["date"] if hasattr(row, "keys") else row[1],
                    "open": row["open"] if hasattr(row, "keys") else row[2],
                    "high": row["high"] if hasattr(row, "keys") else row[3],
                    "low": row["low"] if hasattr(row, "keys") else row[4],
                    "close": row["close"] if hasattr(row, "keys") else row[5],
                    "volume": row["volume"] if hasattr(row, "keys") else row[6],
                }
            else:
                prices[ticker] = {"error": "no price data"}

        result_text = json.dumps(prices, indent=2)

        return ExecutorResult(
            executor_name="price_check",
            status=ExecutorStatus.OK.value,
            result=result_text,
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={
                "gate": "passed",
                "tickers_requested": tickers,
                "tickers_found": sum(1 for v in prices.values() if "error" not in v),
                "tickers_missing": sum(1 for v in prices.values() if "error" in v),
            },
        )
    except Exception as e:
        return ExecutorResult(
            executor_name="price_check",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed", "error_type": type(e).__name__},
            error=str(e),
        )


def _exec_backtest_thesis(ctx: ExecutorContext) -> ExecutorResult:
    """Run walk-forward backtest on a thesis.

    Expects ctx.query to contain thesis_id.
    Optional action_plan keys: start_date, end_date, step.
    Returns BacktestResult summary.
    """
    thesis_id = ctx.query.strip()

    db = _get_fgip_db(ctx)
    if db is None:
        return ExecutorResult(
            executor_name="backtest_thesis",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed"},
            error="FGIP database not available",
        )

    try:
        import sys
        fgip_root = str(FGIP_DB_PATH.parent)
        if fgip_root not in sys.path:
            sys.path.insert(0, fgip_root)
        from fgip.calibration.backtest import WalkForwardBacktest

        plan = ctx.action_plan or {}
        start_date = plan.get("start_date", "2024-01-01")
        end_date = plan.get("end_date", "2025-12-31")
        step = plan.get("step", "7d")

        backtest = WalkForwardBacktest(db)
        result = backtest.run(start_date, end_date, step=step,
                              thesis_ids=[thesis_id])

        policy = _get_trading_policy(ctx)
        valid = result.total_steps >= policy.min_backtest_steps

        result_dict = result.to_dict()
        result_dict["valid_for_trading"] = valid
        result_text = json.dumps(result_dict, indent=2)

        return ExecutorResult(
            executor_name="backtest_thesis",
            status=ExecutorStatus.OK.value,
            result=result_text,
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={
                "gate": "passed",
                "thesis_id": thesis_id,
                "start_date": start_date,
                "end_date": end_date,
                "step": step,
                "total_steps": result.total_steps,
                "avg_brier_score": result.avg_brier_score,
                "lookahead_violations": result.lookahead_violations,
                "valid_for_trading": valid,
            },
        )
    except Exception as e:
        return ExecutorResult(
            executor_name="backtest_thesis",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed", "thesis_id": thesis_id,
                              "error_type": type(e).__name__},
            error=str(e),
        )


def _exec_risk_check(ctx: ExecutorContext) -> ExecutorResult:
    """Check position limits and portfolio exposure.

    Reads current paper_positions from FGIP database and evaluates
    against TradingPolicy limits.
    """
    db = _get_fgip_db(ctx)
    if db is None:
        return ExecutorResult(
            executor_name="risk_check",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed"},
            error="FGIP database not available",
        )

    try:
        conn = db.connect()
        policy = _get_trading_policy(ctx)

        # Get open positions
        try:
            rows = conn.execute("""
                SELECT thesis_id, ticker, actual_size, shares, entry_price, status
                FROM paper_positions WHERE status = 'OPEN'
            """).fetchall()
        except sqlite3.OperationalError:
            # Table might not exist yet
            rows = []

        positions = []
        total_exposure = 0.0
        for row in rows:
            pos = {
                "thesis_id": row[0] if not hasattr(row, "keys") else row["thesis_id"],
                "ticker": row[1] if not hasattr(row, "keys") else row["ticker"],
                "actual_size": float(row[2] if not hasattr(row, "keys") else row["actual_size"]),
                "shares": float(row[3] if not hasattr(row, "keys") else row["shares"]),
                "entry_price": float(row[4] if not hasattr(row, "keys") else row["entry_price"]),
            }
            total_exposure += pos["actual_size"]
            positions.append(pos)

        n_positions = len(positions)

        risk_report = {
            "n_open_positions": n_positions,
            "total_exposure": round(total_exposure, 4),
            "remaining_exposure": round(
                max(0.0, policy.max_portfolio_exposure - total_exposure), 4),
            "positions_headroom": max(0, policy.max_positions - n_positions),
            "policy": policy.to_receipt(),
            "positions": positions,
            "warnings": [],
        }

        # Check warnings
        if total_exposure > policy.max_portfolio_exposure * 0.8:
            risk_report["warnings"].append(
                f"exposure {total_exposure:.1%} approaching limit "
                f"{policy.max_portfolio_exposure:.0%}")

        for pos in positions:
            if pos["actual_size"] > policy.max_position_pct:
                risk_report["warnings"].append(
                    f"{pos['thesis_id']} position {pos['actual_size']:.1%} "
                    f"exceeds limit {policy.max_position_pct:.0%}")

        result_text = json.dumps(risk_report, indent=2)

        return ExecutorResult(
            executor_name="risk_check",
            status=ExecutorStatus.OK.value,
            result=result_text,
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={
                "gate": "passed",
                "n_positions": n_positions,
                "total_exposure": round(total_exposure, 4),
                "n_warnings": len(risk_report["warnings"]),
            },
        )
    except Exception as e:
        return ExecutorResult(
            executor_name="risk_check",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed", "error_type": type(e).__name__},
            error=str(e),
        )


def _exec_paper_trade(ctx: ExecutorContext) -> ExecutorResult:
    """Record a paper trade decision.

    This executor HAS SIDE EFFECTS — it writes to the paper_positions table.

    Expects ctx.action_plan with:
        - thesis_id: str
        - action: "BUY" | "REDUCE" | "EXIT"
        - conviction_level: int (required for BUY)
        - position_size_pct: float (for BUY)
        - ticker: str (optional, resolved from thesis)
        - reason: str

    Policy gates:
        - paper_only must be True
        - conviction >= min_conviction for BUY
        - position within limits
    """
    plan = ctx.action_plan or {}
    thesis_id = plan.get("thesis_id", ctx.query.strip())
    action = plan.get("action", "BUY").upper()
    conviction_level = int(plan.get("conviction_level", 0))
    position_size_pct = float(plan.get("position_size_pct", 0.0))
    ticker = plan.get("ticker", "")
    reason = plan.get("reason", "governed_controller_decision")

    policy = _get_trading_policy(ctx)

    # Hard invariant: paper only
    if not policy.paper_only:
        return ExecutorResult(
            executor_name="paper_trade",
            status=ExecutorStatus.BLOCKED.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "policy", "reason": "paper_only_invariant"},
            error="CRITICAL: paper_only invariant is False — refusing to execute",
        )

    db = _get_fgip_db(ctx)
    if db is None:
        return ExecutorResult(
            executor_name="paper_trade",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed"},
            error="FGIP database not available",
        )

    try:
        conn = db.connect()

        # Get current portfolio state for policy check
        try:
            rows = conn.execute("""
                SELECT actual_size FROM paper_positions WHERE status = 'OPEN'
            """).fetchall()
            current_exposure = sum(
                float(r[0] if not hasattr(r, "keys") else r["actual_size"])
                for r in rows
            )
            current_positions = len(rows)
        except sqlite3.OperationalError:
            current_exposure = 0.0
            current_positions = 0

        # Policy check for BUY
        if action == "BUY":
            entry_check = policy.check_entry(
                conviction_level, current_exposure, current_positions)

            if not entry_check["allowed"]:
                return ExecutorResult(
                    executor_name="paper_trade",
                    status=ExecutorStatus.BLOCKED.value,
                    result="",
                    timing_ms=0.0,
                    side_effects=False,
                    receipt_fragment={
                        "gate": "trading_policy",
                        "action": action,
                        "thesis_id": thesis_id,
                        "conviction_level": conviction_level,
                        "reason": entry_check["reason"],
                    },
                    error=f"Trading policy blocked: {entry_check['reason']}",
                )

            # Cap position size
            position_size_pct = min(position_size_pct, entry_check["max_size"])

        # Record the decision (paper trade memo)
        import uuid
        trade_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"

        trade_record = {
            "trade_id": trade_id,
            "thesis_id": thesis_id,
            "action": action,
            "ticker": ticker,
            "conviction_level": conviction_level,
            "position_size_pct": round(position_size_pct, 4),
            "reason": reason,
            "timestamp": timestamp,
            "policy_check": "passed",
            "paper_only": True,
        }

        # Write to trade_memos table
        try:
            conn.execute("""
                INSERT INTO trade_memos
                (memo_id, thesis_id, symbol, decision, decision_confidence,
                 gates_passed, gates_total, position_size_pct, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, thesis_id, ticker, action,
                conviction_level / 5.0,  # normalize to 0-1
                conviction_level,  # gates_passed = conviction level
                5,  # gates_total = max conviction
                position_size_pct,
                timestamp,
            ))
            conn.commit()
        except sqlite3.OperationalError:
            # trade_memos table might not exist — record in receipt only
            trade_record["db_write"] = "skipped_no_table"

        result_text = json.dumps(trade_record, indent=2)

        return ExecutorResult(
            executor_name="paper_trade",
            status=ExecutorStatus.OK.value,
            result=result_text,
            timing_ms=0.0,
            side_effects=True,
            receipt_fragment={
                "gate": "passed",
                "trade_id": trade_id,
                "thesis_id": thesis_id,
                "action": action,
                "conviction_level": conviction_level,
                "position_size_pct": round(position_size_pct, 4),
                "paper_only": True,
            },
        )
    except Exception as e:
        return ExecutorResult(
            executor_name="paper_trade",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed", "thesis_id": thesis_id,
                              "error_type": type(e).__name__},
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Registry extension
# ---------------------------------------------------------------------------

TRADING_EXECUTORS = [
    ExecutorSpec(
        name="thesis_scan",
        handler=_exec_thesis_scan,
        has_side_effects=False,
        requires_budget=False,
        description="Scan all FGIP theses for conviction levels",
    ),
    ExecutorSpec(
        name="conviction_check",
        handler=_exec_conviction_check,
        has_side_effects=False,
        requires_budget=False,
        description="Evaluate single thesis via ConvictionEngine",
    ),
    ExecutorSpec(
        name="price_check",
        handler=_exec_price_check,
        has_side_effects=False,
        requires_budget=False,
        description="Get latest OHLCV from FGIP price_history",
    ),
    ExecutorSpec(
        name="backtest_thesis",
        handler=_exec_backtest_thesis,
        has_side_effects=False,
        requires_budget=False,
        description="Run walk-forward backtest on thesis",
    ),
    ExecutorSpec(
        name="risk_check",
        handler=_exec_risk_check,
        has_side_effects=False,
        requires_budget=False,
        description="Check position limits and portfolio exposure",
    ),
    ExecutorSpec(
        name="paper_trade",
        handler=_exec_paper_trade,
        has_side_effects=True,      # Writes to database!
        requires_budget=False,
        description="Record paper trade decision (side effects!)",
    ),
]


def register_trading_executors(registry: ExecutorRegistry) -> None:
    """Register all trading executors with an existing registry."""
    for spec in TRADING_EXECUTORS:
        registry.register(spec)


def build_trading_registry() -> ExecutorRegistry:
    """Build a registry with both default and trading executors."""
    from .executor_registry import build_default_registry
    registry = build_default_registry()
    register_trading_executors(registry)
    return registry
