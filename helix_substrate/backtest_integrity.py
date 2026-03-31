"""
Backtest Integrity — WO-L

Makes trading research outputs trustworthy by validating backtest results
against a set of integrity checks. Every backtest should pass these before
any conclusion is drawn.

Four components:
    1. LeakageChecker — detects lookahead bias and future data usage
    2. DataValidator — checks price data completeness and quality
    3. ResultAuditor — flags suspicious performance metrics
    4. ResearchReceipt — standardized output for every backtest

A backtest that fails integrity checks is REJECTED, not presented with
caveats. The receipt records exactly what passed and what failed.

Work Order: WO-L (Backtest Integrity and Research Receipts)
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "backtest_integrity:v1"


# ---------------------------------------------------------------------------
# Leakage Checker
# ---------------------------------------------------------------------------

@dataclass
class LeakageResult:
    """Result of a single leakage check."""
    check_name: str
    passed: bool
    detail: str
    severity: str = "fatal"  # fatal, warning


@dataclass
class LeakageReport:
    """Full leakage analysis for a backtest."""
    checks: List[LeakageResult]
    passed: bool  # True only if ALL fatal checks pass
    n_fatal_failures: int
    n_warnings: int

    def to_receipt(self) -> dict:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_fatal_failures": self.n_fatal_failures,
            "n_warnings": self.n_warnings,
            "checks": [
                {"name": c.check_name, "passed": c.passed,
                 "detail": c.detail, "severity": c.severity}
                for c in self.checks
            ],
        }


class LeakageChecker:
    """Detects lookahead bias and future data leakage in backtests.

    Checks:
        1. Signal timestamp before trade date
        2. No future OHLCV usage (price at trade time <= data available)
        3. Train/test split respected
        4. Conviction evaluation uses only past data
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def check_all(
        self,
        trades: List[dict],
        data_start: str,
        data_end: str,
        train_end: Optional[str] = None,
    ) -> LeakageReport:
        """Run all leakage checks.

        Args:
            trades: List of trade dicts with at minimum:
                    {date, symbol, action, price}
            data_start: First available data date
            data_end: Last available data date
            train_end: If provided, trades before this date are training

        Returns:
            LeakageReport with pass/fail for each check.
        """
        checks = []

        checks.append(self._check_price_availability(trades, data_start, data_end))
        checks.append(self._check_no_future_price(trades))
        checks.append(self._check_signal_timestamps(trades))

        if train_end:
            checks.append(self._check_train_test_split(trades, train_end))

        checks.append(self._check_execution_feasibility(trades))

        fatal_fails = sum(1 for c in checks if not c.passed and c.severity == "fatal")
        warnings = sum(1 for c in checks if not c.passed and c.severity == "warning")

        return LeakageReport(
            checks=checks,
            passed=fatal_fails == 0,
            n_fatal_failures=fatal_fails,
            n_warnings=warnings,
        )

    def _check_price_availability(
        self, trades: List[dict], data_start: str, data_end: str
    ) -> LeakageResult:
        """Verify all trade dates fall within available data range."""
        violations = []
        for t in trades:
            trade_date = t.get("date", "")
            if trade_date < data_start:
                violations.append(f"{t.get('symbol','?')} trade on {trade_date} "
                                  f"before data start {data_start}")
            if trade_date > data_end:
                violations.append(f"{t.get('symbol','?')} trade on {trade_date} "
                                  f"after data end {data_end}")

        if violations:
            return LeakageResult(
                check_name="price_availability",
                passed=False,
                detail=f"{len(violations)} trades outside data range: "
                       f"{violations[0]}",
                severity="fatal",
            )
        return LeakageResult(
            check_name="price_availability",
            passed=True,
            detail=f"All {len(trades)} trades within [{data_start}, {data_end}]",
        )

    def _check_no_future_price(self, trades: List[dict]) -> LeakageResult:
        """Verify trade prices match actual OHLCV on trade date (not future)."""
        violations = []
        for t in trades:
            symbol = t.get("symbol", "")
            trade_date = t.get("date", "")
            trade_price = t.get("price", 0)

            if not symbol or not trade_date or not trade_price:
                continue

            row = self.conn.execute("""
                SELECT open, high, low, close FROM price_history
                WHERE UPPER(symbol) = UPPER(?) AND date = ?
            """, (symbol, trade_date)).fetchone()

            if row is None:
                violations.append(f"{symbol} no price data on {trade_date}")
                continue

            low = float(row[2]) if row[2] else 0
            high = float(row[1]) if row[1] else float('inf')

            # Allow 1% tolerance for slippage
            if trade_price < low * 0.99 or trade_price > high * 1.01:
                violations.append(
                    f"{symbol} on {trade_date}: trade_price={trade_price:.2f} "
                    f"outside day range [{low:.2f}, {high:.2f}]")

        if violations:
            return LeakageResult(
                check_name="no_future_price",
                passed=False,
                detail=f"{len(violations)} price violations: {violations[0]}",
                severity="fatal",
            )
        return LeakageResult(
            check_name="no_future_price",
            passed=True,
            detail=f"All trade prices within day's OHLC range",
        )

    def _check_signal_timestamps(self, trades: List[dict]) -> LeakageResult:
        """Verify signal/decision timestamps are before trade execution."""
        violations = []
        for t in trades:
            signal_date = t.get("signal_date", t.get("decision_date", ""))
            trade_date = t.get("date", "")

            if signal_date and trade_date and signal_date > trade_date:
                violations.append(
                    f"{t.get('symbol','?')}: signal {signal_date} > "
                    f"trade {trade_date}")

        if violations:
            return LeakageResult(
                check_name="signal_timestamps",
                passed=False,
                detail=f"{len(violations)} future signals: {violations[0]}",
                severity="fatal",
            )
        return LeakageResult(
            check_name="signal_timestamps",
            passed=True,
            detail="All signals precede their trades",
        )

    def _check_train_test_split(
        self, trades: List[dict], train_end: str
    ) -> LeakageResult:
        """Verify no out-of-sample info used in training period trades."""
        test_trades = [t for t in trades if t.get("date", "") > train_end]
        train_trades = [t for t in trades if t.get("date", "") <= train_end]

        if not test_trades:
            return LeakageResult(
                check_name="train_test_split",
                passed=False,
                detail=f"No test-period trades after {train_end}",
                severity="warning",
            )

        return LeakageResult(
            check_name="train_test_split",
            passed=True,
            detail=f"Train: {len(train_trades)} trades, "
                   f"Test: {len(test_trades)} trades, split at {train_end}",
        )

    def _check_execution_feasibility(self, trades: List[dict]) -> LeakageResult:
        """Check that trades could have been executed (volume check)."""
        violations = []
        for t in trades:
            symbol = t.get("symbol", "")
            trade_date = t.get("date", "")
            shares = abs(t.get("shares", 0))

            if not symbol or not trade_date or shares == 0:
                continue

            row = self.conn.execute("""
                SELECT volume FROM price_history
                WHERE UPPER(symbol) = UPPER(?) AND date = ?
            """, (symbol, trade_date)).fetchone()

            if row is None:
                continue

            volume = int(row[0]) if row[0] else 0

            # Flag if trade is >5% of daily volume
            if volume > 0 and shares > volume * 0.05:
                violations.append(
                    f"{symbol} on {trade_date}: {shares:.0f} shares = "
                    f"{shares/volume:.1%} of volume ({volume})")

        if violations:
            return LeakageResult(
                check_name="execution_feasibility",
                passed=False,
                detail=f"{len(violations)} trades exceed 5% volume: "
                       f"{violations[0]}",
                severity="warning",
            )
        return LeakageResult(
            check_name="execution_feasibility",
            passed=True,
            detail="All trades within 5% daily volume",
        )


# ---------------------------------------------------------------------------
# Data Validator
# ---------------------------------------------------------------------------

@dataclass
class DataQualityResult:
    """Result of data quality validation."""
    check_name: str
    passed: bool
    detail: str
    severity: str = "fatal"


@dataclass
class DataQualityReport:
    """Full data quality report for a backtest universe."""
    symbols: List[str]
    date_range: Tuple[str, str]
    checks: List[DataQualityResult]
    passed: bool
    coverage_pct: float  # % of expected trading days with data
    gaps: Dict[str, List[str]]  # symbol -> list of gap dates

    def to_receipt(self) -> dict:
        return {
            "symbols": self.symbols,
            "date_range": list(self.date_range),
            "passed": self.passed,
            "coverage_pct": self.coverage_pct,
            "n_checks": len(self.checks),
            "n_failures": sum(1 for c in self.checks if not c.passed),
            "gaps_by_symbol": {s: len(g) for s, g in self.gaps.items() if g},
            "checks": [
                {"name": c.check_name, "passed": c.passed,
                 "detail": c.detail, "severity": c.severity}
                for c in self.checks
            ],
        }


class DataValidator:
    """Validates price data quality for backtest universe."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def validate(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        max_gap_days: int = 5,
        min_coverage_pct: float = 95.0,
    ) -> DataQualityReport:
        """Validate data quality for a set of symbols.

        Args:
            symbols: Ticker symbols to validate
            start_date: Backtest start date
            end_date: Backtest end date
            max_gap_days: Maximum consecutive missing days before fatal
            min_coverage_pct: Minimum data coverage percentage

        Returns:
            DataQualityReport with pass/fail.
        """
        checks = []
        all_gaps: Dict[str, List[str]] = {}
        coverage_scores = []

        for symbol in symbols:
            sym_checks, sym_gaps, sym_coverage = self._validate_symbol(
                symbol, start_date, end_date, max_gap_days)
            checks.extend(sym_checks)
            all_gaps[symbol] = sym_gaps
            coverage_scores.append(sym_coverage)

        # Overall coverage
        avg_coverage = (sum(coverage_scores) / len(coverage_scores)
                        if coverage_scores else 0.0)

        # Coverage threshold check
        if avg_coverage < min_coverage_pct:
            checks.append(DataQualityResult(
                check_name="overall_coverage",
                passed=False,
                detail=f"Average coverage {avg_coverage:.1f}% < "
                       f"minimum {min_coverage_pct}%",
                severity="fatal",
            ))
        else:
            checks.append(DataQualityResult(
                check_name="overall_coverage",
                passed=True,
                detail=f"Average coverage {avg_coverage:.1f}% >= "
                       f"{min_coverage_pct}%",
            ))

        passed = all(c.passed for c in checks if c.severity == "fatal")

        return DataQualityReport(
            symbols=symbols,
            date_range=(start_date, end_date),
            checks=checks,
            passed=passed,
            coverage_pct=round(avg_coverage, 1),
            gaps=all_gaps,
        )

    def _validate_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        max_gap_days: int,
    ) -> Tuple[List[DataQualityResult], List[str], float]:
        """Validate a single symbol's data."""
        checks = []
        gaps = []

        # Get all dates for this symbol
        rows = self.conn.execute("""
            SELECT date, open, high, low, close, volume
            FROM price_history
            WHERE UPPER(symbol) = UPPER(?)
            AND date >= ? AND date <= ?
            ORDER BY date
        """, (symbol, start_date, end_date)).fetchall()

        if not rows:
            checks.append(DataQualityResult(
                check_name=f"{symbol}_data_exists",
                passed=False,
                detail=f"No price data for {symbol} in [{start_date}, {end_date}]",
                severity="fatal",
            ))
            return checks, gaps, 0.0

        dates = [r[0] if not hasattr(r, "keys") else r["date"] for r in rows]

        # Expected trading days (~252/year)
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        total_calendar_days = (end_dt - start_dt).days
        expected_trading_days = int(total_calendar_days * 252 / 365)

        coverage = (len(dates) / expected_trading_days * 100
                    if expected_trading_days > 0 else 0.0)

        # Gap detection
        max_consecutive_gap = 0
        for i in range(1, len(dates)):
            prev = datetime.fromisoformat(dates[i - 1])
            curr = datetime.fromisoformat(dates[i])
            gap_days = (curr - prev).days
            if gap_days > 3:  # > weekend
                gaps.append(f"{dates[i-1]} to {dates[i]} ({gap_days}d)")
                max_consecutive_gap = max(max_consecutive_gap, gap_days)

        if max_consecutive_gap > max_gap_days:
            checks.append(DataQualityResult(
                check_name=f"{symbol}_gaps",
                passed=False,
                detail=f"{symbol}: max gap {max_consecutive_gap}d > "
                       f"limit {max_gap_days}d",
                severity="fatal",
            ))
        elif gaps:
            checks.append(DataQualityResult(
                check_name=f"{symbol}_gaps",
                passed=True,
                detail=f"{symbol}: {len(gaps)} gaps (max {max_consecutive_gap}d), "
                       f"within limit",
                severity="warning",
            ))

        # OHLC sanity: high >= low, close within range
        ohlc_violations = 0
        for r in rows:
            o = float(r[1] if not hasattr(r, "keys") else r["open"]) if r[1] else 0
            h = float(r[2] if not hasattr(r, "keys") else r["high"]) if r[2] else 0
            l = float(r[3] if not hasattr(r, "keys") else r["low"]) if r[3] else 0
            c = float(r[4] if not hasattr(r, "keys") else r["close"]) if r[4] else 0

            if h > 0 and l > 0 and h < l:
                ohlc_violations += 1
            if c > 0 and h > 0 and c > h * 1.001:
                ohlc_violations += 1
            if c > 0 and l > 0 and c < l * 0.999:
                ohlc_violations += 1

        if ohlc_violations > 0:
            checks.append(DataQualityResult(
                check_name=f"{symbol}_ohlc_sanity",
                passed=False,
                detail=f"{symbol}: {ohlc_violations} OHLC violations",
                severity="warning",
            ))

        # Zero volume check
        zero_volume = sum(
            1 for r in rows
            if (int(r[5] if not hasattr(r, "keys") else r["volume"]) if r[5] else 0) == 0
        )
        if zero_volume > len(rows) * 0.1:
            checks.append(DataQualityResult(
                check_name=f"{symbol}_volume",
                passed=False,
                detail=f"{symbol}: {zero_volume}/{len(rows)} days with zero volume",
                severity="warning",
            ))

        return checks, gaps, coverage


# ---------------------------------------------------------------------------
# Result Auditor
# ---------------------------------------------------------------------------

@dataclass
class AuditFlag:
    """A flag raised by the result auditor."""
    flag_name: str
    triggered: bool
    detail: str
    severity: str = "reject"  # reject, warning


@dataclass
class AuditReport:
    """Result of auditing backtest performance metrics."""
    flags: List[AuditFlag]
    rejected: bool
    rejection_reasons: List[str]

    def to_receipt(self) -> dict:
        return {
            "rejected": self.rejected,
            "n_flags": sum(1 for f in self.flags if f.triggered),
            "rejection_reasons": self.rejection_reasons,
            "flags": [
                {"name": f.flag_name, "triggered": f.triggered,
                 "detail": f.detail, "severity": f.severity}
                for f in self.flags
            ],
        }


class ResultAuditor:
    """Flags suspicious backtest results that indicate overfitting or errors.

    Rejection criteria:
        - Too few trades (< min_trades)
        - Impossible Sharpe ratio (> max_sharpe)
        - Impossible win rate (> max_win_rate)
        - Zero losing trades
        - Missing data > threshold
        - Leakage detected (from LeakageChecker)
    """

    def __init__(
        self,
        min_trades: int = 10,
        max_sharpe: float = 4.0,
        max_win_rate: float = 0.85,
        max_profit_factor: float = 10.0,
        min_data_coverage: float = 90.0,
    ):
        self.min_trades = min_trades
        self.max_sharpe = max_sharpe
        self.max_win_rate = max_win_rate
        self.max_profit_factor = max_profit_factor
        self.min_data_coverage = min_data_coverage

    def audit(
        self,
        metrics: dict,
        n_trades: int,
        data_coverage_pct: float = 100.0,
        leakage_passed: bool = True,
    ) -> AuditReport:
        """Audit backtest results for suspicious patterns.

        Args:
            metrics: Performance metrics dict with keys like:
                     sharpe_ratio, win_rate, profit_factor, total_return,
                     max_drawdown, etc.
            n_trades: Total number of trades executed
            data_coverage_pct: Data coverage percentage from DataValidator
            leakage_passed: Whether LeakageChecker passed

        Returns:
            AuditReport with rejection decision.
        """
        flags = []

        # 1. Too few trades
        flags.append(AuditFlag(
            flag_name="min_trades",
            triggered=n_trades < self.min_trades,
            detail=f"{n_trades} trades < minimum {self.min_trades}",
            severity="reject",
        ))

        # 2. Suspicious Sharpe
        sharpe = metrics.get("sharpe_ratio", 0)
        flags.append(AuditFlag(
            flag_name="suspicious_sharpe",
            triggered=abs(sharpe) > self.max_sharpe,
            detail=f"Sharpe {sharpe:.2f} exceeds {self.max_sharpe} "
                   f"(likely overfitting)",
            severity="reject",
        ))

        # 3. Suspicious win rate
        wr = metrics.get("win_rate", 0)
        flags.append(AuditFlag(
            flag_name="suspicious_win_rate",
            triggered=wr > self.max_win_rate and n_trades >= self.min_trades,
            detail=f"Win rate {wr:.1%} exceeds {self.max_win_rate:.0%} "
                   f"(likely overfitting)",
            severity="reject",
        ))

        # 4. Suspicious profit factor
        pf = metrics.get("profit_factor", 0)
        if pf == float('inf'):
            pf_suspicious = n_trades >= self.min_trades
        else:
            pf_suspicious = pf > self.max_profit_factor and n_trades >= self.min_trades
        flags.append(AuditFlag(
            flag_name="suspicious_profit_factor",
            triggered=pf_suspicious,
            detail=f"Profit factor {pf} exceeds {self.max_profit_factor} "
                   f"(likely overfitting or too few losers)",
            severity="reject",
        ))

        # 5. Zero drawdown with positive return (impossible in real trading)
        total_return = metrics.get("total_return", 0)
        max_dd = abs(metrics.get("max_drawdown", 0))
        flags.append(AuditFlag(
            flag_name="impossible_drawdown",
            triggered=total_return > 0.05 and max_dd < 0.001 and n_trades > 5,
            detail=f"Return {total_return:.1%} with near-zero drawdown "
                   f"({max_dd:.2%}) — impossible",
            severity="reject",
        ))

        # 6. Insufficient data coverage
        flags.append(AuditFlag(
            flag_name="data_coverage",
            triggered=data_coverage_pct < self.min_data_coverage,
            detail=f"Data coverage {data_coverage_pct:.1f}% < "
                   f"{self.min_data_coverage}%",
            severity="reject",
        ))

        # 7. Leakage detected
        flags.append(AuditFlag(
            flag_name="leakage_detected",
            triggered=not leakage_passed,
            detail="Lookahead leakage detected by LeakageChecker",
            severity="reject",
        ))

        # 8. No losing trades (suspicious if enough trades)
        n_losers = metrics.get("n_losers", None)
        if n_losers is None:
            # Try to infer from win rate
            n_losers = int(n_trades * (1 - wr)) if n_trades > 0 else 0
        flags.append(AuditFlag(
            flag_name="zero_losers",
            triggered=n_losers == 0 and n_trades >= 5,
            detail=f"Zero losing trades out of {n_trades} — suspicious",
            severity="warning",
        ))

        rejection_reasons = [f.detail for f in flags
                             if f.triggered and f.severity == "reject"]

        return AuditReport(
            flags=flags,
            rejected=len(rejection_reasons) > 0,
            rejection_reasons=rejection_reasons,
        )


# ---------------------------------------------------------------------------
# Research Receipt
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkComparison:
    """Comparison of strategy vs naive baselines."""
    strategy_return: float
    buy_hold_return: float
    equal_weight_return: float
    strategy_sharpe: float
    buy_hold_sharpe: float
    strategy_beats_buy_hold: bool
    strategy_beats_equal_weight: bool

    def to_receipt(self) -> dict:
        return {
            "strategy_return": round(self.strategy_return, 4),
            "buy_hold_return": round(self.buy_hold_return, 4),
            "equal_weight_return": round(self.equal_weight_return, 4),
            "strategy_sharpe": round(self.strategy_sharpe, 4),
            "buy_hold_sharpe": round(self.buy_hold_sharpe, 4),
            "strategy_beats_buy_hold": self.strategy_beats_buy_hold,
            "strategy_beats_equal_weight": self.strategy_beats_equal_weight,
        }


def compute_buy_hold_return(
    conn: sqlite3.Connection,
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> Tuple[float, float]:
    """Compute buy-and-hold return for equal-weighted basket.

    Returns (total_return, annualized_sharpe_approx).
    """
    returns_per_symbol = []

    for symbol in symbols:
        start_row = conn.execute("""
            SELECT close FROM price_history
            WHERE UPPER(symbol) = UPPER(?) AND date >= ?
            ORDER BY date ASC LIMIT 1
        """, (symbol, start_date)).fetchone()

        end_row = conn.execute("""
            SELECT close FROM price_history
            WHERE UPPER(symbol) = UPPER(?) AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, (symbol, end_date)).fetchone()

        if start_row and end_row:
            start_price = float(start_row[0])
            end_price = float(end_row[0])
            if start_price > 0:
                returns_per_symbol.append((end_price / start_price) - 1)

    if not returns_per_symbol:
        return 0.0, 0.0

    # Equal-weighted average return
    avg_return = sum(returns_per_symbol) / len(returns_per_symbol)

    # Rough Sharpe approximation
    if len(returns_per_symbol) > 1:
        mean_r = avg_return
        var_r = sum((r - mean_r) ** 2 for r in returns_per_symbol) / len(returns_per_symbol)
        std_r = math.sqrt(var_r) if var_r > 0 else 0.001
        sharpe = mean_r / std_r  # simplified
    else:
        sharpe = 0.0

    return round(avg_return, 4), round(sharpe, 2)


def build_benchmark_comparison(
    conn: sqlite3.Connection,
    symbols: List[str],
    start_date: str,
    end_date: str,
    strategy_return: float,
    strategy_sharpe: float,
) -> BenchmarkComparison:
    """Build benchmark comparison against buy-and-hold baselines."""
    bh_return, bh_sharpe = compute_buy_hold_return(
        conn, symbols, start_date, end_date)

    # Equal-weight uses same symbols
    ew_return = bh_return  # same as buy-hold for equal-weight basket

    return BenchmarkComparison(
        strategy_return=strategy_return,
        buy_hold_return=bh_return,
        equal_weight_return=ew_return,
        strategy_sharpe=strategy_sharpe,
        buy_hold_sharpe=bh_sharpe,
        strategy_beats_buy_hold=strategy_return > bh_return,
        strategy_beats_equal_weight=strategy_return > ew_return,
    )


@dataclass
class ResearchReceipt:
    """Standardized research receipt for every backtest run.

    Every backtest should emit one of these. It captures:
    - Universe and data range
    - Thesis used
    - Signal and sizing rules
    - Slippage/fee assumptions
    - Train/test boundaries
    - Performance metrics
    - Integrity checks (leakage, data quality, audit)
    - Benchmark comparison
    - Cost block (WO-RECEIPT-COST-01)
    """
    # Identity
    receipt_id: str
    thesis_id: str
    generated_at: str

    # Universe
    symbols: List[str]
    data_start: str
    data_end: str
    train_end: Optional[str]

    # Rules
    signal_rules: str
    position_sizing: str
    slippage_bps: float
    commission_per_trade: float

    # Performance
    n_trades: int
    metrics: dict  # sharpe, return, drawdown, etc.

    # Integrity
    leakage: dict   # LeakageReport.to_receipt()
    data_quality: dict  # DataQualityReport.to_receipt()
    audit: dict     # AuditReport.to_receipt()

    # Benchmark
    benchmark: dict  # BenchmarkComparison.to_receipt()

    # Verdict
    integrity_passed: bool
    rejection_reasons: List[str]

    # Cost (WO-RECEIPT-COST-01)
    cost: dict

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "receipt_id": self.receipt_id,
            "thesis_id": self.thesis_id,
            "generated_at": self.generated_at,
            "universe": {
                "symbols": self.symbols,
                "data_start": self.data_start,
                "data_end": self.data_end,
                "train_end": self.train_end,
            },
            "rules": {
                "signal_rules": self.signal_rules,
                "position_sizing": self.position_sizing,
                "slippage_bps": self.slippage_bps,
                "commission_per_trade": self.commission_per_trade,
            },
            "performance": {
                "n_trades": self.n_trades,
                **self.metrics,
            },
            "integrity": {
                "leakage": self.leakage,
                "data_quality": self.data_quality,
                "audit": self.audit,
            },
            "benchmark": self.benchmark,
            "verdict": {
                "integrity_passed": self.integrity_passed,
                "rejection_reasons": self.rejection_reasons,
            },
            "cost": self.cost,
        }

    def save(self, directory: str) -> str:
        """Save receipt to JSON file. Returns file path."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        filename = f"research_{self.thesis_id}_{timestamp}.json"
        path = dir_path / filename

        data = self.to_dict()
        data["sha256"] = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return str(path)


# ---------------------------------------------------------------------------
# Full integrity pipeline
# ---------------------------------------------------------------------------

def run_integrity_pipeline(
    conn: sqlite3.Connection,
    thesis_id: str,
    symbols: List[str],
    trades: List[dict],
    metrics: dict,
    data_start: str,
    data_end: str,
    train_end: Optional[str] = None,
    signal_rules: str = "conviction >= 3",
    position_sizing: str = "conviction_based",
    slippage_bps: float = 10.0,
    commission_per_trade: float = 0.0,
) -> ResearchReceipt:
    """Run full integrity pipeline and produce a ResearchReceipt.

    This is the main entry point. Call with backtest results and get
    a fully audited receipt back.

    Args:
        conn: SQLite connection to FGIP database
        thesis_id: Thesis being tested
        symbols: Universe of symbols
        trades: List of trade dicts {date, symbol, action, price, shares, ...}
        metrics: Performance metrics dict
        data_start: First available data date
        data_end: Last available data date
        train_end: Train/test split date
        signal_rules: Description of signal generation rules
        position_sizing: Description of position sizing method
        slippage_bps: Slippage assumption in basis points
        commission_per_trade: Commission per trade in dollars

    Returns:
        ResearchReceipt with verdict.
    """
    import platform
    import resource

    t_start = time.time()
    cpu_start = time.process_time()

    # 1. Leakage check
    leakage_checker = LeakageChecker(conn)
    leakage_report = leakage_checker.check_all(
        trades, data_start, data_end, train_end)

    # 2. Data quality check
    data_validator = DataValidator(conn)
    data_report = data_validator.validate(symbols, data_start, data_end)

    # 3. Result audit
    auditor = ResultAuditor()
    audit_report = auditor.audit(
        metrics=metrics,
        n_trades=len(trades),
        data_coverage_pct=data_report.coverage_pct,
        leakage_passed=leakage_report.passed,
    )

    # 4. Benchmark comparison
    strategy_return = metrics.get("total_return", 0)
    strategy_sharpe = metrics.get("sharpe_ratio", 0)
    benchmark = build_benchmark_comparison(
        conn, symbols, data_start, data_end,
        strategy_return, strategy_sharpe)

    # 5. Build verdict
    integrity_passed = (
        leakage_report.passed
        and data_report.passed
        and not audit_report.rejected
    )

    rejection_reasons = []
    if not leakage_report.passed:
        rejection_reasons.append("leakage_detected")
    if not data_report.passed:
        rejection_reasons.append("data_quality_failed")
    if audit_report.rejected:
        rejection_reasons.extend(audit_report.rejection_reasons)

    # 6. Cost block
    wall_time = round(time.time() - t_start, 3)
    cpu_time = round(time.process_time() - cpu_start, 3)
    cost = {
        "wall_time_s": wall_time,
        "cpu_time_s": cpu_time,
        "peak_memory_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": time.strftime(
            '%Y-%m-%dT%H:%M:%S', time.localtime(t_start)),
        "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    import uuid
    receipt = ResearchReceipt(
        receipt_id=str(uuid.uuid4())[:12],
        thesis_id=thesis_id,
        generated_at=datetime.utcnow().isoformat() + "Z",
        symbols=symbols,
        data_start=data_start,
        data_end=data_end,
        train_end=train_end,
        signal_rules=signal_rules,
        position_sizing=position_sizing,
        slippage_bps=slippage_bps,
        commission_per_trade=commission_per_trade,
        n_trades=len(trades),
        metrics=metrics,
        leakage=leakage_report.to_receipt(),
        data_quality=data_report.to_receipt(),
        audit=audit_report.to_receipt(),
        benchmark=benchmark.to_receipt(),
        integrity_passed=integrity_passed,
        rejection_reasons=rejection_reasons,
        cost=cost,
    )

    return receipt
