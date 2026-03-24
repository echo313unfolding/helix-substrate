"""
Tests for WO-M: Thesis-to-Signal Traceability

Tests:
  ThesisSnapshot — creation, hashing, from_thesis_dict
  SignalSnapshot — from_conviction_signal
  ConvictionTrace — trace_conviction, score itemization, hash links
  TradeDecision — trace_trade, hash links to conviction
  TraceChain — build_trace_chain, verify_chain, save/replay
"""

import json
import os
import tempfile

import pytest

from helix_substrate.thesis_tracer import (
    SCHEMA,
    ConvictionTrace,
    SignalSnapshot,
    ThesisSnapshot,
    TraceChain,
    TradeDecision,
    build_trace_chain,
    replay_from_receipt,
    trace_conviction,
    trace_trade,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SAMPLE_THESIS = {
    "thesis_id": "nuclear-smr-thesis",
    "thesis_statement": "Nuclear SMR companies will outperform as grid baseload demand exceeds supply.",
    "tickers": ["SMR", "OKLO", "CEG"],
    "sector": "nuclear",
    "entry_triggers": ["NRC permit approval", "DoE grant announcement"],
    "exit_triggers": ["Construction delays >6mo", "Cost overrun >30%"],
    "counter_theses": [
        "Vogtle cost overruns repeat",
        "Natural gas stays cheap",
    ],
}

SAMPLE_CONVICTION_REPORT = {
    "thesis_id": "nuclear-smr-thesis",
    "thesis_statement": "Nuclear SMR companies will outperform...",
    "tickers": ["SMR", "OKLO", "CEG"],
    "sector": "nuclear",
    "conviction_level": 4,
    "conviction_score": 82.5,
    "position_size_pct": 0.15,
    "confirming_signals": [
        {
            "signal_id": "edgar-blackrock-smr",
            "signal_type": "confirming",
            "source_type": "edgar_13f",
            "source_tier": 0,
            "source_url": "https://sec.gov/cgi-bin/browse-edgar",
            "description": "BlackRock increased SMR position by 15%",
            "signal_strength": 0.85,
            "timestamp": "2025-12-15T00:00:00Z",
            "metadata": {},
        },
        {
            "signal_id": "grant-nuscale-doe",
            "signal_type": "confirming",
            "source_type": "usaspending",
            "source_tier": 0,
            "source_url": "https://usaspending.gov",
            "description": "NuScale funded by DOE ARDP",
            "signal_strength": 0.9,
            "timestamp": "2025-11-01T00:00:00Z",
            "metadata": {},
        },
        {
            "signal_id": "nrc-nuscale-permit",
            "signal_type": "confirming",
            "source_type": "nrc_permit",
            "source_tier": 0,
            "source_url": "https://nrc.gov",
            "description": "NuScale licensed by NRC",
            "signal_strength": 0.95,
            "timestamp": "2025-10-01T00:00:00Z",
            "metadata": {},
        },
        {
            "signal_id": "news-smr-buildout",
            "signal_type": "confirming",
            "source_type": "news_article",
            "source_tier": 2,
            "source_url": "https://reuters.com/article/smr-expansion",
            "description": "Reuters: SMR expansion plans announced",
            "signal_strength": 0.6,
            "timestamp": "2026-01-05T00:00:00Z",
            "metadata": {},
        },
    ],
    "refuting_signals": [
        {
            "signal_id": "news-cost-concern",
            "signal_type": "refuting",
            "source_type": "news_article",
            "source_tier": 2,
            "source_url": "https://wsj.com/article/cost-concerns",
            "description": "WSJ: SMR cost concerns raised",
            "signal_strength": 0.5,
            "timestamp": "2026-01-10T00:00:00Z",
            "metadata": {},
        },
    ],
    "neutral_signals": [
        {
            "signal_id": "youtube-nuclear-talk",
            "signal_type": "neutral",
            "source_type": "youtube_signal",
            "source_tier": 2,
            "source_url": "https://youtube.com/watch?v=abc123",
            "description": "YouTube: nuclear energy discussion",
            "signal_strength": 0.4,
            "timestamp": "2026-01-12T00:00:00Z",
            "metadata": {},
        },
    ],
    "triangulation_count": 3,
    "triangulation_sources": ["edgar_13f", "usaspending", "nrc_permit"],
    "triangulation_met": True,
    "counter_theses": [
        {
            "counter_id": "counter-nuclear-0",
            "description": "Vogtle cost overruns repeat",
            "severity": "manageable",
            "likelihood": 0.3,
            "evidence": [],
            "mitigation": "Position sizing, stop-loss on -30%",
        },
        {
            "counter_id": "counter-nuclear-1",
            "description": "Natural gas stays cheap",
            "severity": "weak",
            "likelihood": 0.2,
            "evidence": [],
            "mitigation": "Monitor commodity prices",
        },
    ],
    "strongest_counter": {
        "counter_id": "counter-nuclear-0",
        "description": "Vogtle cost overruns repeat",
        "severity": "manageable",
        "likelihood": 0.3,
        "evidence": [],
        "mitigation": "Position sizing, stop-loss on -30%",
    },
    "counter_thesis_severity": "manageable",
    "recommendation": "BUY",
    "entry_timing": "NOW",
}

SAMPLE_TRADES = [
    {
        "trade_id": "t001",
        "date": "2026-01-15",
        "symbol": "SMR",
        "action": "BUY",
        "shares": 100,
        "price": 15.50,
        "position_size_pct": 0.10,
        "reason": "new_position",
    },
    {
        "trade_id": "t002",
        "date": "2026-01-20",
        "symbol": "CEG",
        "action": "BUY",
        "shares": 50,
        "price": 280.00,
        "position_size_pct": 0.05,
        "reason": "new_position",
    },
]

SAMPLE_METRICS = {
    "sharpe_ratio": 1.4,
    "win_rate": 0.58,
    "total_return": 0.12,
    "max_drawdown": -0.07,
    "profit_factor": 1.7,
}

SAMPLE_POLICY_CHECK = {
    "allowed": True,
    "reason": "policy_passed",
    "max_size": 0.20,
}


# =========================================================================
# ThesisSnapshot tests
# =========================================================================

class TestThesisSnapshot:

    def test_from_thesis_dict(self):
        snap = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS, "2026-03-15T12:00:00Z")
        assert snap.thesis_id == "nuclear-smr-thesis"
        assert snap.tickers == ["SMR", "OKLO", "CEG"]
        assert snap.sector == "nuclear"
        assert snap.snapshot_date == "2026-03-15T12:00:00Z"
        assert len(snap.snapshot_hash) == 64

    def test_hash_deterministic(self):
        s1 = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS, "2026-03-15T12:00:00Z")
        s2 = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS, "2026-03-15T12:00:00Z")
        assert s1.snapshot_hash == s2.snapshot_hash

    def test_hash_changes_with_content(self):
        s1 = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        modified = {**SAMPLE_THESIS, "tickers": ["SMR", "OKLO"]}
        s2 = ThesisSnapshot.from_thesis_dict(modified)
        assert s1.snapshot_hash != s2.snapshot_hash

    def test_hash_independent_of_date(self):
        """Hash is content-only, not date-dependent."""
        s1 = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS, "2026-01-01T00:00:00Z")
        s2 = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS, "2026-12-31T23:59:59Z")
        assert s1.snapshot_hash == s2.snapshot_hash

    def test_to_dict(self):
        snap = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        d = snap.to_dict()
        assert d["thesis_id"] == "nuclear-smr-thesis"
        assert "snapshot_hash" in d
        assert len(d["counter_theses"]) == 2


# =========================================================================
# SignalSnapshot tests
# =========================================================================

class TestSignalSnapshot:

    def test_from_conviction_signal(self):
        raw = SAMPLE_CONVICTION_REPORT["confirming_signals"][0]
        snap = SignalSnapshot.from_conviction_signal(raw, "2026-03-15T12:00:00Z")
        assert snap.signal_id == "edgar-blackrock-smr"
        assert snap.source_type == "edgar_13f"
        assert snap.source_tier == 0
        assert snap.signal_strength == 0.85
        assert snap.decision_date == "2026-03-15T12:00:00Z"

    def test_preserves_metadata(self):
        raw = {
            "signal_id": "test",
            "signal_type": "confirming",
            "source_type": "edgar_13f",
            "source_tier": 0,
            "source_url": "https://sec.gov",
            "description": "test signal",
            "signal_strength": 0.8,
            "timestamp": "2026-01-01T00:00:00Z",
            "metadata": {"agent": "edgar", "filing_id": "12345"},
        }
        snap = SignalSnapshot.from_conviction_signal(raw, "2026-03-15")
        assert snap.metadata["agent"] == "edgar"
        assert snap.metadata["filing_id"] == "12345"

    def test_defaults_for_missing_fields(self):
        raw = {"signal_id": "minimal"}
        snap = SignalSnapshot.from_conviction_signal(raw, "2026-03-15")
        assert snap.signal_type == "neutral"
        assert snap.source_tier == 2
        assert snap.signal_strength == 0.5


# =========================================================================
# ConvictionTrace tests
# =========================================================================

class TestConvictionTrace:

    def test_trace_conviction_basic(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        trace = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT, "2026-03-15")
        assert trace.thesis_id == "nuclear-smr-thesis"
        assert trace.confirming_count == 4
        assert trace.refuting_count == 1
        assert trace.neutral_count == 1
        assert len(trace.signals) == 6
        assert trace.triangulation_met is True

    def test_score_components_itemized(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        trace = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT)
        assert len(trace.score_components) == 6
        confirming_components = [c for c in trace.score_components if c["direction"] == "confirming"]
        assert len(confirming_components) == 4
        refuting_components = [c for c in trace.score_components if c["direction"] == "refuting"]
        assert len(refuting_components) == 1

    def test_score_reconstruction(self):
        """Verify score is reconstructed correctly from components."""
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        trace = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT)

        # Manual calculation:
        # Base: 30.0
        # Confirming: 15*0.85 + 15*0.9 + 15*0.95 + 3*0.6 = 12.75+13.5+14.25+1.8 = 42.3
        # Refuting: -(3*0.5*0.8) = -1.2
        # Triangulation: +10.0
        # Counter penalties: 10.0*0.3 + 3.0*0.2 = 3.0+0.6 = 3.6
        # Raw: 30.0 + 42.3 - 1.2 + 10.0 - 3.6 = 77.5
        signal_total = sum(c["contribution"] for c in trace.score_components)
        expected_raw = 30.0 + signal_total + trace.triangulation_bonus - sum(
            cp["penalty"] for cp in trace.counter_thesis_penalties
        )

        assert abs(trace.raw_score - expected_raw) < 0.1

    def test_tier_breakdown(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        trace = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT)
        # 3 tier-0 confirming + 2 tier-2 (1 refuting + 1 neutral + 1 confirming tier-2)
        assert int(trace.tier_breakdown.get("0", 0)) == 3
        assert int(trace.tier_breakdown.get("2", 0)) == 3

    def test_counter_thesis_penalties(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        trace = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT)
        assert len(trace.counter_thesis_penalties) == 2
        manageable = trace.counter_thesis_penalties[0]
        assert manageable["severity"] == "manageable"
        assert manageable["penalty_rate"] == 10.0
        assert manageable["penalty"] == 3.0  # 10.0 * 0.3

    def test_hash_links_to_thesis(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        trace = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT)
        assert trace.thesis_snapshot_hash == thesis.snapshot_hash
        assert len(trace.trace_hash) == 64

    def test_hash_deterministic(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        t1 = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT, "2026-03-15")
        t2 = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT, "2026-03-15")
        assert t1.trace_hash == t2.trace_hash


# =========================================================================
# TradeDecision tests
# =========================================================================

class TestTradeDecision:

    def test_trace_trade(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        conviction = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT)
        trade = trace_trade(SAMPLE_TRADES[0], conviction, SAMPLE_POLICY_CHECK)

        assert trade.trade_id == "t001"
        assert trade.symbol == "SMR"
        assert trade.action == "BUY"
        assert trade.price == 15.50
        assert trade.conviction_level == conviction.final_conviction_level
        assert trade.conviction_trace_hash == conviction.trace_hash

    def test_signal_ids_captured(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        conviction = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT)
        trade = trace_trade(SAMPLE_TRADES[0], conviction, SAMPLE_POLICY_CHECK)
        # Should have confirming signal IDs
        assert len(trade.signal_ids) == 4
        assert "edgar-blackrock-smr" in trade.signal_ids

    def test_policy_check_recorded(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        conviction = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT)
        trade = trace_trade(SAMPLE_TRADES[0], conviction, SAMPLE_POLICY_CHECK)
        assert trade.policy_check["allowed"] is True
        assert trade.policy_check["max_size"] == 0.20

    def test_hash_deterministic(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        conviction = trace_conviction(thesis, SAMPLE_CONVICTION_REPORT, "2026-03-15")
        t1 = trace_trade(SAMPLE_TRADES[0], conviction, SAMPLE_POLICY_CHECK)
        t2 = trace_trade(SAMPLE_TRADES[0], conviction, SAMPLE_POLICY_CHECK)
        assert t1.decision_hash == t2.decision_hash


# =========================================================================
# TraceChain tests
# =========================================================================

class TestTraceChain:

    def test_build_trace_chain(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_receipt_id="receipt-001",
            integrity_passed=True,
            benchmark={"buy_hold_return": 0.08, "strategy_beats_buy_hold": True},
            policy_check=SAMPLE_POLICY_CHECK,
            decision_date="2026-03-15T12:00:00Z",
        )
        assert chain.thesis.thesis_id == "nuclear-smr-thesis"
        assert chain.conviction.confirming_count == 4
        assert len(chain.trades) == 2
        assert chain.verdict == "accepted"
        assert len(chain.chain_hash) == 64

    def test_verdict_rejected_on_integrity_fail(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_passed=False,
        )
        assert chain.verdict == "rejected"

    def test_verdict_rejected_on_rejection_reasons(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_passed=True,
            rejection_reasons=["suspicious_sharpe"],
        )
        assert chain.verdict == "rejected"

    def test_verdict_insufficient_data_no_trades(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=[],
            metrics={},
            integrity_passed=True,
        )
        assert chain.verdict == "insufficient_data"

    def test_verify_chain_valid(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_passed=True,
            policy_check=SAMPLE_POLICY_CHECK,
            decision_date="2026-03-15T12:00:00Z",
        )
        result = chain.verify_chain()
        assert result["valid"] is True
        assert result["n_checks"] >= 5  # thesis + conviction + 2 trades + links + chain

    def test_verify_chain_detects_tamper(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_passed=True,
            policy_check=SAMPLE_POLICY_CHECK,
            decision_date="2026-03-15T12:00:00Z",
        )
        # Tamper with thesis
        chain.thesis.thesis_statement = "TAMPERED"
        result = chain.verify_chain()
        assert result["valid"] is False
        thesis_check = next(c for c in result["checks"] if c["component"] == "thesis")
        assert thesis_check["valid"] is False

    def test_hash_links_consistent(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_passed=True,
            policy_check=SAMPLE_POLICY_CHECK,
            decision_date="2026-03-15T12:00:00Z",
        )
        # thesis → conviction link
        assert chain.conviction.thesis_snapshot_hash == chain.thesis.snapshot_hash
        # conviction → trade links
        for trade in chain.trades:
            assert trade.conviction_trace_hash == chain.conviction.trace_hash

    def test_to_dict_schema(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_passed=True,
        )
        d = chain.to_dict()
        assert d["schema"] == SCHEMA
        assert "thesis" in d
        assert "conviction" in d
        assert "trades" in d
        assert "integrity" in d
        assert "benchmark" in d
        assert "verdict" in d
        assert "chain_hash" in d
        assert "cost" in d

    def test_cost_block(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_passed=True,
        )
        cost = chain.cost
        assert "wall_time_s" in cost
        assert "cpu_time_s" in cost
        assert "peak_memory_mb" in cost
        assert "python_version" in cost
        assert "hostname" in cost

    def test_save_and_replay(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_receipt_id="receipt-test-001",
            integrity_passed=True,
            benchmark={"buy_hold_return": 0.08},
            policy_check=SAMPLE_POLICY_CHECK,
            decision_date="2026-03-15T12:00:00Z",
        )

        tmp_dir = tempfile.mkdtemp()
        try:
            path = chain.save(tmp_dir)
            assert os.path.exists(path)

            # Replay
            result = replay_from_receipt(path)
            assert result["loaded"] is True
            assert result["sha256_valid"] is True
            assert result["integrity"]["valid"] is True

            # Verify reconstructed chain matches
            replayed = result["chain"]
            assert replayed.chain_id == chain.chain_id
            assert replayed.thesis.thesis_id == chain.thesis.thesis_id
            assert replayed.conviction.final_score == chain.conviction.final_score
            assert len(replayed.trades) == len(chain.trades)
            assert replayed.verdict == chain.verdict
        finally:
            # Clean up
            for f in os.listdir(tmp_dir):
                os.unlink(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)

    def test_replay_detects_file_tamper(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_passed=True,
            decision_date="2026-03-15T12:00:00Z",
        )

        tmp_dir = tempfile.mkdtemp()
        try:
            path = chain.save(tmp_dir)

            # Tamper with the file
            with open(path) as f:
                data = json.load(f)
            data["verdict"] = "TAMPERED"
            with open(path, "w") as f:
                json.dump(data, f)

            result = replay_from_receipt(path)
            assert result["sha256_valid"] is False
        finally:
            for f in os.listdir(tmp_dir):
                os.unlink(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:

    def test_empty_signals(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        empty_report = {
            "thesis_id": "nuclear-smr-thesis",
            "conviction_level": 1,
            "conviction_score": 30.0,
            "confirming_signals": [],
            "refuting_signals": [],
            "neutral_signals": [],
            "triangulation_count": 0,
            "triangulation_sources": [],
            "triangulation_met": False,
            "counter_theses": [],
        }
        trace = trace_conviction(thesis, empty_report)
        assert trace.confirming_count == 0
        assert trace.final_score == 30.0  # base score only
        assert len(trace.signals) == 0

    def test_no_trades_chain(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=[],
            metrics={},
            integrity_passed=True,
        )
        assert chain.verdict == "insufficient_data"
        assert len(chain.trades) == 0
        result = chain.verify_chain()
        assert result["valid"] is True

    def test_single_signal(self):
        thesis = ThesisSnapshot.from_thesis_dict(SAMPLE_THESIS)
        single_report = {
            "thesis_id": "nuclear-smr-thesis",
            "conviction_level": 2,
            "conviction_score": 45.0,
            "confirming_signals": [{
                "signal_id": "one-signal",
                "signal_type": "confirming",
                "source_type": "edgar_13f",
                "source_tier": 0,
                "source_url": "https://sec.gov",
                "description": "single signal",
                "signal_strength": 1.0,
                "timestamp": "2026-01-01",
            }],
            "refuting_signals": [],
            "neutral_signals": [],
            "triangulation_met": False,
            "triangulation_sources": ["edgar_13f"],
            "counter_theses": [],
        }
        trace = trace_conviction(thesis, single_report)
        assert trace.confirming_count == 1
        assert len(trace.score_components) == 1
        assert trace.score_components[0]["contribution"] == 15.0  # tier 0 * strength 1.0

    def test_chain_with_all_rejections(self):
        chain = build_trace_chain(
            thesis_dict=SAMPLE_THESIS,
            conviction_report_dict=SAMPLE_CONVICTION_REPORT,
            trades=SAMPLE_TRADES,
            metrics=SAMPLE_METRICS,
            integrity_passed=False,
            rejection_reasons=["leakage", "suspicious_sharpe", "too_few_trades"],
        )
        assert chain.verdict == "rejected"
        assert len(chain.rejection_reasons) == 3
