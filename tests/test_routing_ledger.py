"""
Tests for WO-ECON-01: Routing economy ledger.

Verifies:
    - LedgerEntry net calculation (profit - cost)
    - RoutingLedger aggregation by mechanism
    - Edge cases (empty, single query, all-cache, all-failure)
    - Serialization (as_dict, summary)
"""

import time
import pytest

from helix_substrate.routing_ledger import LedgerEntry, RoutingLedger


# ── LedgerEntry ──

class TestLedgerEntry:

    def test_net_ms_zero_by_default(self):
        e = LedgerEntry(query_idx=0, timestamp=time.time())
        assert e.net_ms == 0.0

    def test_net_ms_cache_profit(self):
        e = LedgerEntry(query_idx=0, timestamp=time.time(),
                        cache_hit=True, cache_saved_ms=500.0)
        assert e.net_ms == 500.0

    def test_net_ms_kv_profit(self):
        e = LedgerEntry(query_idx=0, timestamp=time.time(),
                        kv_event="reused", kv_saved_ms=120.0)
        assert e.net_ms == 120.0

    def test_net_ms_swap_cost(self):
        e = LedgerEntry(query_idx=0, timestamp=time.time(),
                        did_swap=True, swap_time_ms=900.0)
        assert e.net_ms == -900.0

    def test_net_ms_quality_cost(self):
        e = LedgerEntry(query_idx=0, timestamp=time.time(),
                        quality_passed=False, quality_cost_ms=300.0)
        assert e.net_ms == -300.0

    def test_net_ms_mixed(self):
        """Cache profit + KV profit - swap cost - quality cost."""
        e = LedgerEntry(
            query_idx=0, timestamp=time.time(),
            cache_hit=True, cache_saved_ms=500.0,
            kv_event="reused", kv_saved_ms=100.0,
            did_swap=True, swap_time_ms=200.0,
            quality_passed=False, quality_cost_ms=50.0,
        )
        # 500 + 100 - 200 - 50 = 350
        assert abs(e.net_ms - 350.0) < 0.01

    def test_as_dict_has_all_sections(self):
        e = LedgerEntry(query_idx=0, timestamp=time.time())
        d = e.as_dict()
        assert "preflight" in d
        assert "inheritance" in d
        assert "cache" in d
        assert "kv" in d
        assert "model_swap" in d
        assert "quality" in d
        assert "route" in d
        assert "gen" in d
        assert "net_ms" in d


# ── RoutingLedger ──

class TestRoutingLedger:

    def test_empty_summary(self):
        ledger = RoutingLedger()
        s = ledger.summary()
        assert s["n_queries"] == 0
        assert s["net_profit_ms"] == 0.0

    def test_single_query_no_savings(self):
        ledger = RoutingLedger()
        ledger.record(LedgerEntry(
            query_idx=0, timestamp=time.time(),
            wall_time_ms=1500.0, gen_time_ms=1200.0, n_tokens=50,
            route_model="tinyllama", route_retrieval="cache_first",
            route_budget="factual_64",
        ))
        s = ledger.summary()
        assert s["n_queries"] == 1
        assert s["net_profit_ms"] == 0.0
        assert s["by_mechanism"]["response_cache"]["hits"] == 0
        assert s["by_mechanism"]["kv_prefix_reuse"]["reuses"] == 0

    def test_cache_hit_profit(self):
        ledger = RoutingLedger()
        ledger.record(LedgerEntry(
            query_idx=0, timestamp=time.time(),
            cache_hit=True, cache_saved_ms=800.0,
            wall_time_ms=50.0,
        ))
        s = ledger.summary()
        assert s["net_profit_ms"] == 800.0
        assert s["by_mechanism"]["response_cache"]["hits"] == 1
        assert s["by_mechanism"]["response_cache"]["hit_rate"] == 1.0

    def test_kv_reuse_profit(self):
        ledger = RoutingLedger()
        ledger.record(LedgerEntry(
            query_idx=0, timestamp=time.time(),
            kv_event="reused", kv_saved_ms=150.0,
            kv_prefix_tokens_saved=200,
            wall_time_ms=300.0,
        ))
        s = ledger.summary()
        assert s["net_profit_ms"] == 150.0
        assert s["by_mechanism"]["kv_prefix_reuse"]["reuses"] == 1

    def test_swap_cost(self):
        ledger = RoutingLedger()
        ledger.record(LedgerEntry(
            query_idx=0, timestamp=time.time(),
            did_swap=True, swap_time_ms=4000.0,
            wall_time_ms=5000.0,
        ))
        s = ledger.summary()
        assert s["net_profit_ms"] == -4000.0
        assert s["by_mechanism"]["model_swap"]["count"] == 1

    def test_quality_failure_cost(self):
        ledger = RoutingLedger()
        ledger.record(LedgerEntry(
            query_idx=0, timestamp=time.time(),
            quality_passed=False, quality_verdict="repetition",
            quality_cost_ms=1200.0,
            wall_time_ms=1500.0,
        ))
        s = ledger.summary()
        assert s["net_profit_ms"] == -1200.0
        assert s["by_mechanism"]["quality_gate"]["failures"] == 1

    def test_mixed_session(self):
        """5-query session: 2 cache hits, 1 KV reuse, 1 swap, 1 quality fail."""
        ledger = RoutingLedger()
        now = time.time()

        # Q0: normal generation
        ledger.record(LedgerEntry(
            query_idx=0, timestamp=now,
            wall_time_ms=1500.0, gen_time_ms=1200.0, n_tokens=50,
            preflight_model_match=True, preflight_retrieval_match=True,
            preflight_budget_match=True,
            route_model="tinyllama",
        ))

        # Q1: cache hit (saves 1200ms gen time)
        ledger.record(LedgerEntry(
            query_idx=1, timestamp=now + 1,
            wall_time_ms=50.0,
            cache_hit=True, cache_saved_ms=1200.0,
            preflight_model_match=True, preflight_retrieval_match=True,
            preflight_budget_match=True,
        ))

        # Q2: KV reuse (saves 80ms prefill)
        ledger.record(LedgerEntry(
            query_idx=2, timestamp=now + 2,
            wall_time_ms=1000.0, gen_time_ms=800.0, n_tokens=40,
            kv_event="reused", kv_saved_ms=80.0, kv_prefix_tokens_saved=150,
            was_inherited=True, inheritance_state="inherited",
            preflight_model_match=True, preflight_retrieval_match=True,
            preflight_budget_match=True,
        ))

        # Q3: model swap to Qwen (costs 4000ms)
        ledger.record(LedgerEntry(
            query_idx=3, timestamp=now + 3,
            wall_time_ms=5500.0, gen_time_ms=1200.0, n_tokens=30,
            did_swap=True, swap_time_ms=4000.0,
            preflight_model_match=False,
            route_model="qwen_coder",
        ))

        # Q4: quality failure (wastes 1000ms gen)
        ledger.record(LedgerEntry(
            query_idx=4, timestamp=now + 4,
            wall_time_ms=1200.0, gen_time_ms=1000.0, n_tokens=20,
            quality_passed=False, quality_verdict="repetition",
            quality_cost_ms=1000.0,
            route_model="qwen_coder",
        ))

        s = ledger.summary()
        assert s["n_queries"] == 5

        # Net: 1200(cache) + 80(kv) - 4000(swap) - 1000(quality) = -3720
        assert abs(s["net_profit_ms"] - (-3720.0)) < 0.01

        assert s["by_mechanism"]["response_cache"]["hits"] == 1
        assert s["by_mechanism"]["kv_prefix_reuse"]["reuses"] == 1
        assert s["by_mechanism"]["inheritance"]["inherited"] == 1
        assert s["by_mechanism"]["model_swap"]["count"] == 1
        assert s["by_mechanism"]["quality_gate"]["failures"] == 1

        # Preflight: 3/5 fully correct
        assert abs(s["by_mechanism"]["preflight"]["accuracy"] - 0.6) < 0.01

    def test_all_cache_hits(self):
        """Session where every query hits cache = maximum profit."""
        ledger = RoutingLedger()
        for i in range(10):
            ledger.record(LedgerEntry(
                query_idx=i, timestamp=time.time(),
                cache_hit=True, cache_saved_ms=1000.0,
                wall_time_ms=20.0,
            ))
        s = ledger.summary()
        assert s["net_profit_ms"] == 10000.0
        assert s["by_mechanism"]["response_cache"]["hit_rate"] == 1.0

    def test_inheritance_tracking(self):
        ledger = RoutingLedger()
        for i in range(5):
            ledger.record(LedgerEntry(
                query_idx=i, timestamp=time.time(),
                was_inherited=(i >= 3),
                inheritance_state="inherited" if i >= 3 else "warm_candidate",
                wall_time_ms=500.0,
            ))
        s = ledger.summary()
        assert s["by_mechanism"]["inheritance"]["inherited"] == 2
        assert abs(s["by_mechanism"]["inheritance"]["inheritance_rate"] - 0.4) < 0.01
        # Currently no profit from inheritance
        assert s["by_mechanism"]["inheritance"]["profit_ms"] == 0.0

    def test_entries_as_dicts(self):
        ledger = RoutingLedger()
        ledger.record(LedgerEntry(query_idx=0, timestamp=time.time()))
        ledger.record(LedgerEntry(query_idx=1, timestamp=time.time()))
        dicts = ledger.entries_as_dicts()
        assert len(dicts) == 2
        assert dicts[0]["query_idx"] == 0
        assert dicts[1]["query_idx"] == 1

    def test_recent(self):
        ledger = RoutingLedger()
        for i in range(10):
            ledger.record(LedgerEntry(query_idx=i, timestamp=time.time()))
        recent = ledger.recent(3)
        assert len(recent) == 3
        assert recent[0].query_idx == 7

    def test_n_queries(self):
        ledger = RoutingLedger()
        assert ledger.n_queries == 0
        ledger.record(LedgerEntry(query_idx=0, timestamp=time.time()))
        assert ledger.n_queries == 1


class TestPreflight:
    """Test preflight accuracy tracking in the ledger."""

    def test_perfect_preflight(self):
        ledger = RoutingLedger()
        for i in range(5):
            ledger.record(LedgerEntry(
                query_idx=i, timestamp=time.time(),
                preflight_model_match=True,
                preflight_retrieval_match=True,
                preflight_budget_match=True,
                preflight_warming_useful=True,
                preflight_warming_action="preload_qwen",
                wall_time_ms=500.0,
            ))
        s = ledger.summary()
        assert s["by_mechanism"]["preflight"]["accuracy"] == 1.0
        assert s["by_mechanism"]["preflight"]["warming_useful"] == 5
        assert s["by_mechanism"]["preflight"]["warming_rate"] == 1.0

    def test_partial_preflight(self):
        ledger = RoutingLedger()
        # Model matches but retrieval doesn't
        ledger.record(LedgerEntry(
            query_idx=0, timestamp=time.time(),
            preflight_model_match=True,
            preflight_retrieval_match=False,
            preflight_budget_match=True,
            wall_time_ms=500.0,
        ))
        s = ledger.summary()
        # Not all 3 matched → accuracy = 0
        assert s["by_mechanism"]["preflight"]["accuracy"] == 0.0
