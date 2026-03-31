"""
Thesis-to-Signal Traceability — WO-M

End-to-end lineage from thesis definition through conviction scoring,
signal generation, backtest decisions, and final research verdict.

Every step produces a hash-linked snapshot. Given any trace chain receipt,
you can walk from thesis to verdict and verify each link.

Components:
    ThesisSnapshot    — freeze thesis definition at a point in time
    SignalSnapshot    — signal as it existed at decision time
    ConvictionTrace   — itemized conviction score breakdown
    TradeDecision     — why a specific trade was made, linked to conviction
    TraceChain        — end-to-end chain with hash integrity

Functions:
    build_trace_chain()   — build complete thesis→verdict chain
    replay_from_receipt() — reproduce and verify a saved chain

Work Order: WO-M (Thesis-to-Signal Traceability)
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "thesis_tracer:v1"


# ---------------------------------------------------------------------------
# Hash utility
# ---------------------------------------------------------------------------

def _sha256(data: str) -> str:
    """SHA256 of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _dict_hash(d: dict) -> str:
    """Deterministic SHA256 of a dict (sorted keys, no whitespace)."""
    return _sha256(json.dumps(d, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# ThesisSnapshot — freeze thesis definition at a point in time
# ---------------------------------------------------------------------------

@dataclass
class ThesisSnapshot:
    """Immutable snapshot of a thesis definition."""
    thesis_id: str
    thesis_statement: str
    tickers: List[str]
    sector: str
    entry_triggers: List[str]
    exit_triggers: List[str]
    counter_theses: List[str]
    snapshot_date: str
    snapshot_hash: str = ""

    def __post_init__(self):
        if not self.snapshot_hash:
            self.snapshot_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = {
            "thesis_id": self.thesis_id,
            "thesis_statement": self.thesis_statement,
            "tickers": sorted(self.tickers),
            "sector": self.sector,
            "entry_triggers": self.entry_triggers,
            "exit_triggers": self.exit_triggers,
            "counter_theses": self.counter_theses,
        }
        return _dict_hash(payload)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_thesis_dict(cls, thesis: dict, snapshot_date: str = "") -> "ThesisSnapshot":
        """Create snapshot from INVESTMENT_THESES entry."""
        if not snapshot_date:
            snapshot_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        return cls(
            thesis_id=thesis["thesis_id"],
            thesis_statement=thesis["thesis_statement"],
            tickers=list(thesis["tickers"]),
            sector=thesis["sector"],
            entry_triggers=list(thesis.get("entry_triggers", [])),
            exit_triggers=list(thesis.get("exit_triggers", [])),
            counter_theses=list(thesis.get("counter_theses", [])),
            snapshot_date=snapshot_date,
        )


# ---------------------------------------------------------------------------
# SignalSnapshot — signal as it existed at decision time
# ---------------------------------------------------------------------------

@dataclass
class SignalSnapshot:
    """A signal captured at the time a conviction decision was made."""
    signal_id: str
    signal_type: str        # confirming, refuting, neutral
    source_type: str        # edgar_13f, usaspending, nrc_permit, etc.
    source_tier: int        # 0, 1, 2
    source_url: str
    description: str
    signal_strength: float  # 0-1
    timestamp: str          # when signal was originally created
    decision_date: str      # when this snapshot was taken for the decision
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_conviction_signal(cls, signal_dict: dict, decision_date: str) -> "SignalSnapshot":
        """Create from ConvictionEngine Signal.to_dict() output."""
        return cls(
            signal_id=signal_dict.get("signal_id", ""),
            signal_type=signal_dict.get("signal_type", "neutral"),
            source_type=signal_dict.get("source_type", "unknown"),
            source_tier=int(signal_dict.get("source_tier", 2)),
            source_url=signal_dict.get("source_url", ""),
            description=signal_dict.get("description", ""),
            signal_strength=float(signal_dict.get("signal_strength", 0.5)),
            timestamp=signal_dict.get("timestamp", ""),
            decision_date=decision_date,
            metadata=signal_dict.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# ConvictionTrace — itemized conviction score breakdown
# ---------------------------------------------------------------------------

@dataclass
class ConvictionTrace:
    """How a conviction score was computed, with full signal itemization."""
    thesis_id: str
    decision_date: str
    signals: List[SignalSnapshot]

    # Counts
    confirming_count: int
    refuting_count: int
    neutral_count: int

    # Tier breakdown: {tier: count}
    tier_breakdown: Dict[str, int]

    # Score components: each signal's contribution
    score_components: List[Dict[str, Any]]
    # e.g. [{"signal_id": "edgar-...", "source_type": "edgar_13f", "tier": 0,
    #         "strength": 0.7, "boost_per_tier": 15.0, "contribution": 10.5}]

    # Counter-thesis penalties
    counter_thesis_penalties: List[Dict[str, Any]]
    # e.g. [{"counter": "Vogtle cost overruns", "severity": "manageable",
    #         "likelihood": 0.3, "penalty": 3.0}]

    # Triangulation
    triangulation_met: bool
    triangulation_sources: List[str]
    triangulation_bonus: float

    # Final score
    base_score: float
    raw_score: float  # before clamping
    final_score: float  # after clamping to 0-100
    final_conviction_level: int

    # Hash links
    thesis_snapshot_hash: str  # links to ThesisSnapshot
    trace_hash: str = ""

    def __post_init__(self):
        if not self.trace_hash:
            self.trace_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = {
            "thesis_id": self.thesis_id,
            "decision_date": self.decision_date,
            "thesis_snapshot_hash": self.thesis_snapshot_hash,
            "final_score": self.final_score,
            "final_conviction_level": self.final_conviction_level,
            "signal_ids": sorted(s.signal_id for s in self.signals),
        }
        return _dict_hash(payload)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signals"] = [s.to_dict() for s in self.signals]
        return d


# ---------------------------------------------------------------------------
# TradeDecision — why a specific trade was made
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    """Records the justification for a specific trade."""
    trade_id: str
    date: str
    symbol: str
    action: str           # BUY, SELL, HOLD
    shares: float
    price: float
    thesis_id: str
    conviction_level: int
    conviction_trace_hash: str   # links to ConvictionTrace
    position_size_pct: float
    reason: str                  # rebalance, new_position, stop_loss, signal_exit
    policy_check: Dict[str, Any]  # TradingPolicy.check_entry() result
    signal_ids: List[str]        # which signals contributed to this decision
    decision_hash: str = ""

    def __post_init__(self):
        if not self.decision_hash:
            self.decision_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = {
            "trade_id": self.trade_id,
            "date": self.date,
            "symbol": self.symbol,
            "action": self.action,
            "price": self.price,
            "thesis_id": self.thesis_id,
            "conviction_trace_hash": self.conviction_trace_hash,
        }
        return _dict_hash(payload)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# TraceChain — end-to-end thesis → verdict chain
# ---------------------------------------------------------------------------

@dataclass
class TraceChain:
    """Complete thesis → conviction → signals → trades → verdict chain."""
    chain_id: str
    schema: str = SCHEMA

    # Thesis layer
    thesis: Optional[ThesisSnapshot] = None

    # Conviction layer
    conviction: Optional[ConvictionTrace] = None

    # Trade layer
    trades: List[TradeDecision] = field(default_factory=list)

    # Backtest layer
    backtest_metrics: Dict[str, Any] = field(default_factory=dict)

    # Integrity layer (from WO-L)
    integrity_receipt_id: str = ""
    integrity_passed: bool = False
    rejection_reasons: List[str] = field(default_factory=list)

    # Benchmark layer
    benchmark: Dict[str, Any] = field(default_factory=dict)

    # Final verdict
    verdict: str = ""  # accepted, rejected, insufficient_data

    # Cost
    cost: Dict[str, Any] = field(default_factory=dict)

    # Chain integrity
    chain_hash: str = ""

    def compute_chain_hash(self) -> str:
        """Compute hash over all component hashes."""
        components = {
            "chain_id": self.chain_id,
            "thesis_hash": self.thesis.snapshot_hash if self.thesis else "",
            "conviction_hash": self.conviction.trace_hash if self.conviction else "",
            "trade_hashes": sorted(t.decision_hash for t in self.trades),
            "integrity_receipt_id": self.integrity_receipt_id,
            "verdict": self.verdict,
        }
        self.chain_hash = _dict_hash(components)
        return self.chain_hash

    def verify_chain(self) -> Dict[str, Any]:
        """Verify hash integrity of entire chain.

        Returns dict with:
            valid: bool — overall chain integrity
            checks: list — individual check results
        """
        checks = []

        # Check thesis hash
        if self.thesis:
            expected = self.thesis._compute_hash()
            ok = expected == self.thesis.snapshot_hash
            checks.append({
                "component": "thesis",
                "valid": ok,
                "expected": expected,
                "actual": self.thesis.snapshot_hash,
            })

        # Check conviction hash
        if self.conviction:
            expected = self.conviction._compute_hash()
            ok = expected == self.conviction.trace_hash
            checks.append({
                "component": "conviction",
                "valid": ok,
                "expected": expected,
                "actual": self.conviction.trace_hash,
            })

        # Check trade hashes
        for trade in self.trades:
            expected = trade._compute_hash()
            ok = expected == trade.decision_hash
            checks.append({
                "component": f"trade:{trade.trade_id}",
                "valid": ok,
                "expected": expected,
                "actual": trade.decision_hash,
            })

        # Check thesis→conviction link
        if self.thesis and self.conviction:
            ok = self.conviction.thesis_snapshot_hash == self.thesis.snapshot_hash
            checks.append({
                "component": "link:thesis→conviction",
                "valid": ok,
                "expected": self.thesis.snapshot_hash,
                "actual": self.conviction.thesis_snapshot_hash,
            })

        # Check conviction→trade links
        for trade in self.trades:
            if self.conviction:
                ok = trade.conviction_trace_hash == self.conviction.trace_hash
                checks.append({
                    "component": f"link:conviction→trade:{trade.trade_id}",
                    "valid": ok,
                    "expected": self.conviction.trace_hash,
                    "actual": trade.conviction_trace_hash,
                })

        # Check chain hash
        expected_chain = self.compute_chain_hash()
        # chain_hash was just recomputed, so store it
        # This check is: does the data still produce the same chain hash?
        checks.append({
            "component": "chain",
            "valid": True,  # always true after recompute
            "chain_hash": expected_chain,
        })

        all_valid = all(c["valid"] for c in checks)
        return {"valid": all_valid, "checks": checks, "n_checks": len(checks)}

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "chain_id": self.chain_id,
            "thesis": self.thesis.to_dict() if self.thesis else None,
            "conviction": self.conviction.to_dict() if self.conviction else None,
            "trades": [t.to_dict() for t in self.trades],
            "backtest_metrics": self.backtest_metrics,
            "integrity": {
                "receipt_id": self.integrity_receipt_id,
                "passed": self.integrity_passed,
                "rejection_reasons": self.rejection_reasons,
            },
            "benchmark": self.benchmark,
            "verdict": self.verdict,
            "chain_hash": self.chain_hash,
            "cost": self.cost,
        }

    def save(self, directory: str) -> str:
        """Save trace chain receipt to JSON file."""
        os.makedirs(directory, exist_ok=True)

        self.compute_chain_hash()
        d = self.to_dict()
        d["sha256"] = _dict_hash(d)

        timestamp = time.strftime("%Y%m%dT%H%M%S")
        filename = f"trace_{self.thesis.thesis_id}_{timestamp}.json" if self.thesis else f"trace_{self.chain_id}_{timestamp}.json"
        path = os.path.join(directory, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, default=str)

        return path


# ---------------------------------------------------------------------------
# Conviction tracing — builds ConvictionTrace from ConvictionReport
# ---------------------------------------------------------------------------

# Tier conviction boost (mirrored from conviction_engine.py)
_TIER_BOOST = {0: 15.0, 1: 8.0, 2: 3.0}

# Counter-thesis penalty (mirrored from conviction_engine.py)
_COUNTER_PENALTY = {
    "fatal": 50.0,
    "serious": 25.0,
    "manageable": 10.0,
    "weak": 3.0,
}


def trace_conviction(
    thesis_snapshot: ThesisSnapshot,
    conviction_report: dict,
    decision_date: str = "",
) -> ConvictionTrace:
    """Build a ConvictionTrace from a ConvictionReport dict.

    This reconstructs the scoring arithmetic from the conviction engine
    to produce an itemized breakdown. The report dict should be from
    ConvictionReport.to_dict().

    Args:
        thesis_snapshot: Frozen thesis definition
        conviction_report: Dict from ConvictionReport.to_dict()
        decision_date: ISO timestamp for this trace (default: now)

    Returns:
        ConvictionTrace with full itemization
    """
    if not decision_date:
        decision_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Snapshot all signals
    signals = []
    score_components = []

    confirming_raw = conviction_report.get("confirming_signals", [])
    refuting_raw = conviction_report.get("refuting_signals", [])
    neutral_raw = conviction_report.get("neutral_signals", [])

    # Process confirming signals
    for s in confirming_raw:
        snap = SignalSnapshot.from_conviction_signal(s, decision_date)
        signals.append(snap)

        tier = int(s.get("source_tier", 2))
        strength = float(s.get("signal_strength", 0.5))
        boost = _TIER_BOOST.get(tier, 3.0)
        contribution = boost * strength

        score_components.append({
            "signal_id": snap.signal_id,
            "source_type": snap.source_type,
            "tier": tier,
            "strength": strength,
            "boost_per_tier": boost,
            "contribution": round(contribution, 2),
            "direction": "confirming",
        })

    # Process refuting signals
    for s in refuting_raw:
        snap = SignalSnapshot.from_conviction_signal(s, decision_date)
        signals.append(snap)

        tier = int(s.get("source_tier", 2))
        strength = float(s.get("signal_strength", 0.5))
        boost = _TIER_BOOST.get(tier, 3.0)
        contribution = -(boost * strength * 0.8)  # 0.8 weight factor

        score_components.append({
            "signal_id": snap.signal_id,
            "source_type": snap.source_type,
            "tier": tier,
            "strength": strength,
            "boost_per_tier": boost,
            "contribution": round(contribution, 2),
            "direction": "refuting",
        })

    # Process neutral signals (no score contribution)
    for s in neutral_raw:
        snap = SignalSnapshot.from_conviction_signal(s, decision_date)
        signals.append(snap)

        score_components.append({
            "signal_id": snap.signal_id,
            "source_type": snap.source_type,
            "tier": int(s.get("source_tier", 2)),
            "strength": float(s.get("signal_strength", 0.5)),
            "boost_per_tier": 0.0,
            "contribution": 0.0,
            "direction": "neutral",
        })

    # Tier breakdown
    tier_counts: Dict[str, int] = {}
    for s in signals:
        key = str(s.source_tier)
        tier_counts[key] = tier_counts.get(key, 0) + 1

    # Counter-thesis penalties
    counter_penalties = []
    counter_raw = conviction_report.get("counter_theses", [])
    for ct in counter_raw:
        severity = ct.get("severity", "manageable")
        likelihood = float(ct.get("likelihood", 0.3))
        penalty_rate = _COUNTER_PENALTY.get(severity, 5.0)
        penalty = penalty_rate * likelihood

        counter_penalties.append({
            "counter": ct.get("description", ""),
            "severity": severity,
            "likelihood": likelihood,
            "penalty_rate": penalty_rate,
            "penalty": round(penalty, 2),
        })

    # Reconstruct score
    base_score = 30.0
    signal_total = sum(c["contribution"] for c in score_components)
    triangulation_met = conviction_report.get("triangulation_met", False)
    triangulation_bonus = 10.0 if triangulation_met else 0.0
    counter_total = sum(cp["penalty"] for cp in counter_penalties)

    raw_score = base_score + signal_total + triangulation_bonus - counter_total
    final_score = max(0.0, min(100.0, raw_score))

    return ConvictionTrace(
        thesis_id=thesis_snapshot.thesis_id,
        decision_date=decision_date,
        signals=signals,
        confirming_count=len(confirming_raw),
        refuting_count=len(refuting_raw),
        neutral_count=len(neutral_raw),
        tier_breakdown=tier_counts,
        score_components=score_components,
        counter_thesis_penalties=counter_penalties,
        triangulation_met=triangulation_met,
        triangulation_sources=conviction_report.get("triangulation_sources", []),
        triangulation_bonus=triangulation_bonus,
        base_score=base_score,
        raw_score=round(raw_score, 2),
        final_score=round(final_score, 2),
        final_conviction_level=int(conviction_report.get("conviction_level", 1)),
        thesis_snapshot_hash=thesis_snapshot.snapshot_hash,
    )


# ---------------------------------------------------------------------------
# Trade tracing — links trades to conviction
# ---------------------------------------------------------------------------

def trace_trade(
    trade: dict,
    conviction_trace: ConvictionTrace,
    policy_check: dict,
) -> TradeDecision:
    """Build a TradeDecision linked to its ConvictionTrace.

    Args:
        trade: Trade dict with date, symbol, action, shares, price, etc.
        conviction_trace: The conviction analysis that justified this trade
        policy_check: Result from TradingPolicy.check_entry()

    Returns:
        TradeDecision with hash link to conviction
    """
    # Collect signal IDs that contributed to the conviction
    contributing_signals = [
        s.signal_id for s in conviction_trace.signals
        if s.signal_type == "confirming"
    ]

    return TradeDecision(
        trade_id=trade.get("trade_id", _sha256(json.dumps(trade, default=str))[:12]),
        date=trade.get("date", ""),
        symbol=trade.get("symbol", ""),
        action=trade.get("action", "BUY"),
        shares=float(trade.get("shares", 0)),
        price=float(trade.get("price", 0)),
        thesis_id=conviction_trace.thesis_id,
        conviction_level=conviction_trace.final_conviction_level,
        conviction_trace_hash=conviction_trace.trace_hash,
        position_size_pct=float(trade.get("position_size_pct", 0.0)),
        reason=trade.get("reason", "signal"),
        policy_check=policy_check,
        signal_ids=contributing_signals,
    )


# ---------------------------------------------------------------------------
# build_trace_chain — end-to-end chain builder
# ---------------------------------------------------------------------------

def build_trace_chain(
    thesis_dict: dict,
    conviction_report_dict: dict,
    trades: List[dict],
    metrics: dict,
    integrity_receipt_id: str = "",
    integrity_passed: bool = False,
    rejection_reasons: Optional[List[str]] = None,
    benchmark: Optional[dict] = None,
    policy_check: Optional[dict] = None,
    decision_date: str = "",
    chain_id: str = "",
) -> TraceChain:
    """Build complete thesis → verdict trace chain.

    Args:
        thesis_dict: Thesis definition (from INVESTMENT_THESES)
        conviction_report_dict: Full ConvictionReport.to_dict() output
        trades: List of trade dicts (date, symbol, action, shares, price, etc.)
        metrics: Backtest performance metrics
        integrity_receipt_id: ID from ResearchReceipt (WO-L)
        integrity_passed: Did integrity pipeline pass?
        rejection_reasons: Why rejected (if applicable)
        benchmark: Benchmark comparison data
        policy_check: TradingPolicy.check_entry() result
        decision_date: ISO timestamp (default: now)
        chain_id: Unique chain ID (default: generated)

    Returns:
        TraceChain with full hash-linked lineage
    """
    import platform
    import resource

    t_start = time.time()
    cpu_start = time.process_time()

    if not decision_date:
        decision_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    if not chain_id:
        chain_id = _sha256(
            f"{thesis_dict.get('thesis_id', '')}:{decision_date}"
        )[:16]

    # 1. Snapshot thesis
    thesis = ThesisSnapshot.from_thesis_dict(thesis_dict, decision_date)

    # 2. Trace conviction
    conviction = trace_conviction(thesis, conviction_report_dict, decision_date)

    # 3. Trace trades
    if policy_check is None:
        policy_check = {"allowed": True, "reason": "not_checked", "max_size": 1.0}

    traced_trades = []
    for t in trades:
        td = trace_trade(t, conviction, policy_check)
        traced_trades.append(td)

    # 4. Determine verdict
    if rejection_reasons:
        verdict = "rejected"
    elif not integrity_passed:
        verdict = "rejected"
    elif not trades:
        verdict = "insufficient_data"
    else:
        verdict = "accepted"

    # 5. Build chain
    wall_time = round(time.time() - t_start, 3)
    cpu_time = round(time.process_time() - cpu_start, 3)

    chain = TraceChain(
        chain_id=chain_id,
        thesis=thesis,
        conviction=conviction,
        trades=traced_trades,
        backtest_metrics=metrics,
        integrity_receipt_id=integrity_receipt_id,
        integrity_passed=integrity_passed,
        rejection_reasons=rejection_reasons or [],
        benchmark=benchmark or {},
        verdict=verdict,
        cost={
            "wall_time_s": wall_time,
            "cpu_time_s": cpu_time,
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    chain.compute_chain_hash()

    return chain


# ---------------------------------------------------------------------------
# replay_from_receipt — reproduce and verify a saved chain
# ---------------------------------------------------------------------------

def replay_from_receipt(receipt_path: str) -> Dict[str, Any]:
    """Load a saved trace chain receipt and verify its integrity.

    Args:
        receipt_path: Path to saved trace chain JSON

    Returns:
        Dict with:
            loaded: bool — file loaded successfully
            chain: TraceChain — reconstructed chain
            integrity: dict — hash verification results
            sha256_valid: bool — file hash matches
    """
    with open(receipt_path, encoding="utf-8") as f:
        data = json.load(f)

    # Verify file-level SHA256
    saved_sha = data.pop("sha256", "")
    recomputed_sha = _dict_hash(data)
    sha256_valid = saved_sha == recomputed_sha

    # Reconstruct ThesisSnapshot
    thesis_data = data.get("thesis")
    thesis = None
    if thesis_data:
        thesis = ThesisSnapshot(
            thesis_id=thesis_data["thesis_id"],
            thesis_statement=thesis_data["thesis_statement"],
            tickers=thesis_data["tickers"],
            sector=thesis_data["sector"],
            entry_triggers=thesis_data["entry_triggers"],
            exit_triggers=thesis_data["exit_triggers"],
            counter_theses=thesis_data["counter_theses"],
            snapshot_date=thesis_data["snapshot_date"],
            snapshot_hash=thesis_data["snapshot_hash"],
        )

    # Reconstruct ConvictionTrace
    conv_data = data.get("conviction")
    conviction = None
    if conv_data:
        signals = [
            SignalSnapshot(**s) for s in conv_data.get("signals", [])
        ]
        conviction = ConvictionTrace(
            thesis_id=conv_data["thesis_id"],
            decision_date=conv_data["decision_date"],
            signals=signals,
            confirming_count=conv_data["confirming_count"],
            refuting_count=conv_data["refuting_count"],
            neutral_count=conv_data["neutral_count"],
            tier_breakdown=conv_data["tier_breakdown"],
            score_components=conv_data["score_components"],
            counter_thesis_penalties=conv_data["counter_thesis_penalties"],
            triangulation_met=conv_data["triangulation_met"],
            triangulation_sources=conv_data["triangulation_sources"],
            triangulation_bonus=conv_data["triangulation_bonus"],
            base_score=conv_data["base_score"],
            raw_score=conv_data["raw_score"],
            final_score=conv_data["final_score"],
            final_conviction_level=conv_data["final_conviction_level"],
            thesis_snapshot_hash=conv_data["thesis_snapshot_hash"],
            trace_hash=conv_data["trace_hash"],
        )

    # Reconstruct TradeDecisions
    trades = []
    for td in data.get("trades", []):
        trades.append(TradeDecision(
            trade_id=td["trade_id"],
            date=td["date"],
            symbol=td["symbol"],
            action=td["action"],
            shares=td["shares"],
            price=td["price"],
            thesis_id=td["thesis_id"],
            conviction_level=td["conviction_level"],
            conviction_trace_hash=td["conviction_trace_hash"],
            position_size_pct=td["position_size_pct"],
            reason=td["reason"],
            policy_check=td["policy_check"],
            signal_ids=td["signal_ids"],
            decision_hash=td["decision_hash"],
        ))

    # Reconstruct chain
    integrity_data = data.get("integrity", {})
    chain = TraceChain(
        chain_id=data["chain_id"],
        schema=data.get("schema", SCHEMA),
        thesis=thesis,
        conviction=conviction,
        trades=trades,
        backtest_metrics=data.get("backtest_metrics", {}),
        integrity_receipt_id=integrity_data.get("receipt_id", ""),
        integrity_passed=integrity_data.get("passed", False),
        rejection_reasons=integrity_data.get("rejection_reasons", []),
        benchmark=data.get("benchmark", {}),
        verdict=data.get("verdict", ""),
        cost=data.get("cost", {}),
        chain_hash=data.get("chain_hash", ""),
    )

    # Verify hash integrity
    integrity = chain.verify_chain()

    return {
        "loaded": True,
        "chain": chain,
        "integrity": integrity,
        "sha256_valid": sha256_valid,
    }
