"""
Market Realism Layer — WO-N

Makes research results closer to real trading conditions by adding:

1. DataSnapshot  — fingerprint of the dataset used (replay determinism)
2. ExecutionModel — structured execution assumptions (slippage, costs, fills)
3. RobustnessTest — Monte Carlo perturbation to detect fragile strategies

These integrate with the existing trace chain (WO-M) and integrity pipeline
(WO-L) to ensure results are not only traceable and integrity-checked, but
also grounded in realistic execution assumptions.

Work Order: WO-N (Market Realism Layer)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

SCHEMA = "market_realism:v1"


# ---------------------------------------------------------------------------
# Hash utility
# ---------------------------------------------------------------------------

def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _dict_hash(d: dict) -> str:
    return _sha256(json.dumps(d, sort_keys=True, default=str))


# ===========================================================================
# 1. DataSnapshot — dataset fingerprinting for replay determinism
# ===========================================================================

@dataclass
class DataSnapshot:
    """Fingerprint of the dataset used for a backtest or trace chain.

    Captures enough metadata to detect if the data has changed since
    the original analysis, without storing the full dataset.

    Fields:
        symbols: tickers included
        date_range: (start, end) of price data
        row_counts: {symbol: n_rows}
        price_hash: SHA256 of concatenated (symbol, date, close) tuples
        total_rows: sum of all row counts
        snapshot_date: when this snapshot was taken
        source: database path or identifier
    """
    symbols: List[str]
    date_start: str
    date_end: str
    row_counts: Dict[str, int]
    price_hash: str
    total_rows: int
    snapshot_date: str
    source: str
    snapshot_hash: str = ""

    def __post_init__(self):
        if not self.snapshot_hash:
            self.snapshot_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = {
            "symbols": sorted(self.symbols),
            "date_start": self.date_start,
            "date_end": self.date_end,
            "row_counts": {k: self.row_counts[k] for k in sorted(self.row_counts)},
            "price_hash": self.price_hash,
            "total_rows": self.total_rows,
        }
        return _dict_hash(payload)

    def to_dict(self) -> dict:
        return {
            "symbols": self.symbols,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "row_counts": self.row_counts,
            "price_hash": self.price_hash,
            "total_rows": self.total_rows,
            "snapshot_date": self.snapshot_date,
            "source": self.source,
            "snapshot_hash": self.snapshot_hash,
        }

    def matches(self, other: "DataSnapshot") -> bool:
        """Check if two snapshots represent the same dataset."""
        return self.price_hash == other.price_hash


def build_data_snapshot(
    conn: sqlite3.Connection,
    symbols: List[str],
    date_start: str,
    date_end: str,
    source: str = "",
) -> DataSnapshot:
    """Build a DataSnapshot from a live database connection.

    Queries price_history table to compute:
    - Row count per symbol
    - SHA256 of all (symbol, date, close) tuples sorted by (symbol, date)

    Args:
        conn: SQLite connection with price_history table
        symbols: Tickers to snapshot
        date_start: Start of date range
        date_end: End of date range
        source: Database path or identifier

    Returns:
        DataSnapshot with fingerprint
    """
    row_counts: Dict[str, int] = {}
    hash_data_parts: List[str] = []

    for symbol in sorted(symbols):
        rows = conn.execute("""
            SELECT symbol, date, close FROM price_history
            WHERE UPPER(symbol) = UPPER(?)
            AND date >= ? AND date <= ?
            ORDER BY date ASC
        """, (symbol, date_start, date_end)).fetchall()

        row_counts[symbol] = len(rows)

        for row in rows:
            sym = row[0] if not hasattr(row, "keys") else row["symbol"]
            date = row[1] if not hasattr(row, "keys") else row["date"]
            close = row[2] if not hasattr(row, "keys") else row["close"]
            hash_data_parts.append(f"{sym}:{date}:{close}")

    price_hash = _sha256("|".join(hash_data_parts)) if hash_data_parts else _sha256("")
    total_rows = sum(row_counts.values())

    return DataSnapshot(
        symbols=sorted(symbols),
        date_start=date_start,
        date_end=date_end,
        row_counts=row_counts,
        price_hash=price_hash,
        total_rows=total_rows,
        snapshot_date=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        source=source,
    )


# ===========================================================================
# 2. ExecutionModel — structured execution assumptions
# ===========================================================================

@dataclass
class ExecutionModel:
    """Captures execution assumptions for a backtest or trade decision.

    All costs that affect realized P&L should be declared here. The model
    is descriptive (captures assumptions), not prescriptive (doesn't apply them).
    The actual application happens in the backtest engine.

    Supports both fixed and dynamic slippage.
    """
    # Fixed costs
    slippage_bps: float = 10.0           # Base slippage in basis points
    commission_per_trade: float = 0.0    # Per-trade commission ($)

    # Dynamic slippage (volume-based)
    # Effective slippage = slippage_bps + (order_pct_of_adv * market_impact_bps_per_pct)
    market_impact_bps_per_pct: float = 0.0  # Additional bps per 1% of ADV
    max_order_pct_of_adv: float = 0.02      # Max order size as % of ADV (2%)

    # Fill modeling
    fill_probability: float = 1.0        # Probability of full fill (0-1)
    partial_fill_min_pct: float = 0.5    # Minimum fill if partial (50%)

    # Stop/target levels
    stop_loss_pct: float = 0.15          # 15% stop loss
    trailing_stop_pct: Optional[float] = None   # Trailing stop (None = disabled)
    take_profit_pct: Optional[float] = None     # Take profit target (None = disabled)

    # Holding constraints
    min_hold_days: int = 0               # Minimum holding period
    max_hold_days: Optional[int] = None  # Maximum holding period (None = indefinite)

    # Portfolio-level
    max_portfolio_drawdown: Optional[float] = None  # Portfolio DD halt (None = disabled)

    # Label
    model_name: str = "default"

    def effective_slippage_bps(self, order_pct_of_adv: float = 0.0) -> float:
        """Compute effective slippage including market impact."""
        impact = order_pct_of_adv * self.market_impact_bps_per_pct
        return self.slippage_bps + impact

    def compute_fill_price(
        self, side: str, mid_price: float, order_pct_of_adv: float = 0.0
    ) -> float:
        """Compute fill price given side and market impact.

        Args:
            side: "BUY" or "SELL"
            mid_price: Mid or close price
            order_pct_of_adv: Order size as fraction of ADV

        Returns:
            Estimated fill price (worse than mid for both sides)
        """
        bps = self.effective_slippage_bps(order_pct_of_adv)
        slip_fraction = bps / 10_000

        if side.upper() == "BUY":
            return mid_price * (1 + slip_fraction)
        else:
            return mid_price * (1 - slip_fraction)

    def compute_trade_cost(
        self, shares: float, price: float, side: str,
        order_pct_of_adv: float = 0.0,
    ) -> Dict[str, float]:
        """Compute total execution cost for a trade.

        Returns:
            Dict with fill_price, slippage_cost, commission, total_cost,
            effective_bps.
        """
        fill_price = self.compute_fill_price(side, price, order_pct_of_adv)
        slippage_cost = abs(fill_price - price) * shares
        commission = self.commission_per_trade

        return {
            "mid_price": round(price, 4),
            "fill_price": round(fill_price, 4),
            "slippage_cost": round(slippage_cost, 4),
            "commission": round(commission, 4),
            "total_cost": round(slippage_cost + commission, 4),
            "effective_bps": round(self.effective_slippage_bps(order_pct_of_adv), 2),
        }

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "slippage_bps": self.slippage_bps,
            "commission_per_trade": self.commission_per_trade,
            "market_impact_bps_per_pct": self.market_impact_bps_per_pct,
            "max_order_pct_of_adv": self.max_order_pct_of_adv,
            "fill_probability": self.fill_probability,
            "partial_fill_min_pct": self.partial_fill_min_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "take_profit_pct": self.take_profit_pct,
            "min_hold_days": self.min_hold_days,
            "max_hold_days": self.max_hold_days,
            "max_portfolio_drawdown": self.max_portfolio_drawdown,
        }

    @classmethod
    def conservative(cls) -> "ExecutionModel":
        """Conservative execution model: high slippage, tight stops."""
        return cls(
            slippage_bps=25.0,
            commission_per_trade=5.0,
            market_impact_bps_per_pct=10.0,
            fill_probability=0.9,
            stop_loss_pct=0.10,
            max_portfolio_drawdown=0.20,
            model_name="conservative",
        )

    @classmethod
    def optimistic(cls) -> "ExecutionModel":
        """Optimistic execution model: low friction."""
        return cls(
            slippage_bps=5.0,
            commission_per_trade=0.0,
            market_impact_bps_per_pct=0.0,
            fill_probability=1.0,
            stop_loss_pct=0.20,
            model_name="optimistic",
        )


# ===========================================================================
# 3. RobustnessTest — Monte Carlo perturbation
# ===========================================================================

@dataclass
class TrialResult:
    """Result of a single Monte Carlo trial."""
    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    passed_integrity: bool
    seed: int


@dataclass
class RobustnessReport:
    """Aggregated robustness test results across all trials."""
    n_trials: int
    n_passed: int
    n_failed: int
    pass_rate: float

    # Distribution of key metrics across trials
    metric_distributions: Dict[str, Dict[str, float]]
    # e.g. {"sharpe_ratio": {"mean": 1.2, "std": 0.3, "min": 0.5, "max": 2.1,
    #                        "p5": 0.6, "p25": 1.0, "p50": 1.2, "p75": 1.4, "p95": 1.8}}

    # Sensitivity analysis
    sensitivity: Dict[str, float]
    # e.g. {"slippage_bps": -0.15}  means +1 unit slippage → -0.15 Sharpe

    # Fragility indicators
    fragility_flags: List[str]

    # Individual trials
    trials: List[TrialResult]

    # Config
    base_params: Dict[str, Any]
    perturbation_config: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "n_trials": self.n_trials,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "pass_rate": round(self.pass_rate, 4),
            "metric_distributions": self.metric_distributions,
            "sensitivity": self.sensitivity,
            "fragility_flags": self.fragility_flags,
            "base_params": self.base_params,
            "perturbation_config": self.perturbation_config,
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "params": t.params,
                    "metrics": t.metrics,
                    "passed_integrity": t.passed_integrity,
                    "seed": t.seed,
                }
                for t in self.trials
            ],
        }


def _percentile(values: List[float], pct: float) -> float:
    """Simple percentile without numpy."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = pct / 100.0 * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _distribution_stats(values: List[float]) -> Dict[str, float]:
    """Compute distribution statistics."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0,
                "p5": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0}

    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0
    std = math.sqrt(var)

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "p5": round(_percentile(values, 5), 4),
        "p25": round(_percentile(values, 25), 4),
        "p50": round(_percentile(values, 50), 4),
        "p75": round(_percentile(values, 75), 4),
        "p95": round(_percentile(values, 95), 4),
    }


@dataclass
class PerturbationConfig:
    """Configuration for Monte Carlo perturbation testing.

    Each field controls how the corresponding parameter is varied
    across trials. Set to None to hold constant.
    """
    # Slippage perturbation: (min_bps, max_bps)
    slippage_range: Optional[Tuple[float, float]] = None

    # Commission perturbation: (min, max)
    commission_range: Optional[Tuple[float, float]] = None

    # Price noise: fraction of daily return std to inject
    price_noise_fraction: float = 0.0

    # Start date offset: random shift by ±N trading days
    start_date_offset_days: int = 0

    # End date offset: random shift by ±N trading days
    end_date_offset_days: int = 0

    # Conviction threshold perturbation: {level_to_try: [2, 3, 4]}
    conviction_thresholds: Optional[List[int]] = None

    # Number of trials
    n_trials: int = 20

    # Random seed for reproducibility
    base_seed: int = 42

    def to_dict(self) -> dict:
        return {
            "slippage_range": self.slippage_range,
            "commission_range": self.commission_range,
            "price_noise_fraction": self.price_noise_fraction,
            "start_date_offset_days": self.start_date_offset_days,
            "end_date_offset_days": self.end_date_offset_days,
            "conviction_thresholds": self.conviction_thresholds,
            "n_trials": self.n_trials,
            "base_seed": self.base_seed,
        }


def run_robustness_test(
    backtest_fn: Callable[..., Dict[str, Any]],
    base_params: Dict[str, Any],
    perturbation: PerturbationConfig,
    integrity_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> RobustnessReport:
    """Run Monte Carlo robustness test over a backtest function.

    The backtest_fn should accept keyword arguments matching the keys
    in base_params and return a dict with at least 'sharpe_ratio',
    'total_return', 'max_drawdown', 'win_rate'.

    Args:
        backtest_fn: Callable(**params) -> metrics dict
        base_params: Default parameters for the backtest
        perturbation: How to perturb parameters across trials
        integrity_fn: Optional function(metrics) -> bool for pass/fail

    Returns:
        RobustnessReport with distribution stats and fragility flags
    """
    trials: List[TrialResult] = []
    rng = random.Random(perturbation.base_seed)

    for i in range(perturbation.n_trials):
        trial_seed = perturbation.base_seed + i
        trial_rng = random.Random(trial_seed)

        # Build perturbed params
        params = dict(base_params)

        if perturbation.slippage_range:
            lo, hi = perturbation.slippage_range
            params["slippage_bps"] = round(
                trial_rng.uniform(lo, hi), 1)

        if perturbation.commission_range:
            lo, hi = perturbation.commission_range
            params["commission_per_trade"] = round(
                trial_rng.uniform(lo, hi), 2)

        if perturbation.price_noise_fraction > 0:
            params["price_noise_seed"] = trial_seed
            params["price_noise_fraction"] = perturbation.price_noise_fraction

        if perturbation.start_date_offset_days > 0:
            offset = trial_rng.randint(
                -perturbation.start_date_offset_days,
                perturbation.start_date_offset_days)
            params["start_date_offset"] = offset

        if perturbation.end_date_offset_days > 0:
            offset = trial_rng.randint(
                -perturbation.end_date_offset_days,
                perturbation.end_date_offset_days)
            params["end_date_offset"] = offset

        if perturbation.conviction_thresholds:
            params["min_conviction"] = trial_rng.choice(
                perturbation.conviction_thresholds)

        # Run backtest
        try:
            metrics = backtest_fn(**params)
        except Exception as e:
            metrics = {
                "sharpe_ratio": 0.0, "total_return": 0.0,
                "max_drawdown": -1.0, "win_rate": 0.0,
                "error": str(e),
            }

        # Check integrity
        passed = True
        if integrity_fn:
            try:
                passed = integrity_fn(metrics)
            except Exception:
                passed = False

        trials.append(TrialResult(
            trial_id=i,
            params={k: v for k, v in params.items() if k != base_params.get(k)},
            metrics=metrics,
            passed_integrity=passed,
            seed=trial_seed,
        ))

    # Aggregate
    n_passed = sum(1 for t in trials if t.passed_integrity)
    n_failed = len(trials) - n_passed
    pass_rate = n_passed / len(trials) if trials else 0.0

    # Metric distributions
    metric_keys = ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]
    metric_distributions: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        values = [t.metrics.get(key, 0.0) for t in trials
                  if "error" not in t.metrics]
        if values:
            metric_distributions[key] = _distribution_stats(values)

    # Sensitivity analysis: correlation of slippage with sharpe
    sensitivity: Dict[str, float] = {}
    if perturbation.slippage_range:
        slippages = [t.params.get("slippage_bps", base_params.get("slippage_bps", 10))
                     for t in trials if "error" not in t.metrics]
        sharpes = [t.metrics.get("sharpe_ratio", 0) for t in trials
                   if "error" not in t.metrics]
        if len(slippages) >= 3 and len(set(slippages)) > 1:
            sensitivity["slippage_bps"] = round(
                _simple_sensitivity(slippages, sharpes), 4)

    if perturbation.commission_range:
        commissions = [t.params.get("commission_per_trade",
                                    base_params.get("commission_per_trade", 0))
                       for t in trials if "error" not in t.metrics]
        sharpes = [t.metrics.get("sharpe_ratio", 0) for t in trials
                   if "error" not in t.metrics]
        if len(commissions) >= 3 and len(set(commissions)) > 1:
            sensitivity["commission_per_trade"] = round(
                _simple_sensitivity(commissions, sharpes), 4)

    # Fragility flags
    fragility_flags = []
    sharpe_dist = metric_distributions.get("sharpe_ratio", {})

    if sharpe_dist.get("std", 0) > abs(sharpe_dist.get("mean", 0)) * 0.5:
        fragility_flags.append(
            "high_sharpe_variance: std > 50% of mean — strategy is parameter-sensitive")

    if sharpe_dist.get("p5", 0) < 0:
        fragility_flags.append(
            "negative_p5_sharpe: bottom 5% of trials have negative Sharpe")

    if pass_rate < 0.8:
        fragility_flags.append(
            f"low_pass_rate: only {pass_rate:.0%} of trials pass integrity")

    dd_dist = metric_distributions.get("max_drawdown", {})
    if dd_dist.get("p95", 0) < -0.30:
        fragility_flags.append(
            "severe_tail_drawdown: p95 drawdown exceeds -30%")

    return_dist = metric_distributions.get("total_return", {})
    if return_dist and return_dist.get("p25", 0) < 0:
        fragility_flags.append(
            "return_fragility: bottom 25% of trials produce negative returns")

    return RobustnessReport(
        n_trials=len(trials),
        n_passed=n_passed,
        n_failed=n_failed,
        pass_rate=pass_rate,
        metric_distributions=metric_distributions,
        sensitivity=sensitivity,
        fragility_flags=fragility_flags,
        trials=trials,
        base_params=base_params,
        perturbation_config=perturbation.to_dict(),
    )


def _simple_sensitivity(x_values: List[float], y_values: List[float]) -> float:
    """Simple linear sensitivity: dy/dx (OLS slope)."""
    n = len(x_values)
    if n < 2:
        return 0.0
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x_values, y_values))
    den = sum((xi - x_mean) ** 2 for xi in x_values)
    if abs(den) < 1e-12:
        return 0.0
    return num / den


# ===========================================================================
# Integration helpers
# ===========================================================================

def apply_price_noise(
    prices: List[float],
    noise_fraction: float,
    seed: int,
) -> List[float]:
    """Apply random noise to a price series.

    Perturbs each price by ±noise_fraction of the standard deviation
    of daily returns. Preserves the general trend.

    Args:
        prices: List of close prices
        noise_fraction: Fraction of return std to use as noise scale
        seed: Random seed

    Returns:
        New price list with noise applied
    """
    if len(prices) < 2 or noise_fraction <= 0:
        return list(prices)

    # Compute daily returns std
    returns = [(prices[i] / prices[i-1]) - 1 for i in range(1, len(prices))
               if prices[i-1] > 0]
    if not returns:
        return list(prices)

    ret_std = math.sqrt(sum(r**2 for r in returns) / len(returns))
    noise_scale = ret_std * noise_fraction

    rng = random.Random(seed)
    noisy = [prices[0]]
    for i in range(1, len(prices)):
        noise = rng.gauss(0, noise_scale)
        factor = 1 + noise
        noisy.append(round(prices[i] * max(0.5, factor), 4))

    return noisy


def build_execution_cost_summary(
    trades: List[dict],
    execution_model: ExecutionModel,
) -> Dict[str, Any]:
    """Compute execution cost summary for a list of trades.

    Args:
        trades: List of trade dicts with 'price', 'shares', 'action'
        execution_model: Execution assumptions

    Returns:
        Summary dict with total costs, avg cost per trade, cost as % of turnover
    """
    total_slippage = 0.0
    total_commission = 0.0
    total_turnover = 0.0

    for trade in trades:
        price = float(trade.get("price", 0))
        shares = float(trade.get("shares", 0))
        action = trade.get("action", "BUY")

        cost = execution_model.compute_trade_cost(shares, price, action)
        total_slippage += cost["slippage_cost"]
        total_commission += cost["commission"]
        total_turnover += price * shares

    total_cost = total_slippage + total_commission
    n_trades = len(trades)

    return {
        "n_trades": n_trades,
        "total_slippage": round(total_slippage, 2),
        "total_commission": round(total_commission, 2),
        "total_cost": round(total_cost, 2),
        "avg_cost_per_trade": round(total_cost / n_trades, 2) if n_trades else 0,
        "cost_pct_of_turnover": round(
            (total_cost / total_turnover * 100) if total_turnover > 0 else 0, 4),
        "execution_model": execution_model.to_dict(),
    }
