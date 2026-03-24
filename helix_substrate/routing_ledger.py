"""
Routing economy ledger — per-query profit/loss accounting for the AI OS.

Tracks whether inheritance + preflight + response cache + KV reuse create
net runtime value. Each query produces a LedgerEntry with per-mechanism
cost/benefit. The cumulative ledger produces an honest P/L receipt.

Profit sources (time saved):
    - response_cache_hit: skip generation entirely
    - kv_prefix_reuse: skip prefix prefill
    - inheritance: (future) skip classification
    - preflight_warming: early model/cache/web warming when prediction is correct

Cost sources (time spent or wasted):
    - model_swap: GPU swap overhead
    - quality_failure: wasted generation + reset
    - preflight_miss: wasted prefetch (model preload that wasn't needed)

Work Order: WO-ECON-01
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LedgerEntry:
    """Per-query economic accounting.

    Every field that is 'time saved' is positive.
    Every field that is 'time spent/wasted' is negative.
    Net = sum of all values.
    """
    query_idx: int
    timestamp: float

    # Wall time
    wall_time_ms: float = 0.0

    # ── Preflight ──
    preflight_model_match: bool = False
    preflight_retrieval_match: bool = False
    preflight_budget_match: bool = False
    preflight_warming_action: str = "none"
    preflight_warming_useful: bool = False

    # ── Inheritance ──
    was_inherited: bool = False
    inheritance_state: str = "cold"
    inheritance_confidence: float = 0.0

    # ── Response cache ──
    cache_hit: bool = False
    cache_saved_ms: float = 0.0   # gen_time skipped (positive = profit)

    # ── KV prefix reuse ──
    kv_event: str = "n/a"          # "reused", "rebuilt", "n/a"
    kv_prefix_tokens_saved: int = 0
    kv_prefill_ms: float = 0.0
    kv_saved_ms: float = 0.0      # estimated prefill time saved (positive = profit)

    # ── Model swap ──
    did_swap: bool = False
    swap_time_ms: float = 0.0     # negative cost

    # ── Quality ──
    quality_passed: bool = True
    quality_verdict: str = "pass"
    quality_cost_ms: float = 0.0  # wasted gen_time on failure (negative cost)

    # ── Route decision ──
    route_model: str = ""
    route_retrieval: str = ""
    route_budget: str = ""
    route_inherited_flag: bool = False

    # ── Reset ──
    reset_reason: str = ""        # "" = no reset

    # ── Generation ──
    gen_time_ms: float = 0.0
    n_tokens: int = 0

    @property
    def net_ms(self) -> float:
        """Net profit/loss in milliseconds.

        Positive = time saved, negative = time wasted.
        """
        profit = self.cache_saved_ms + self.kv_saved_ms
        cost = self.swap_time_ms + self.quality_cost_ms
        return profit - cost

    def as_dict(self) -> dict:
        return {
            "query_idx": self.query_idx,
            "wall_time_ms": round(self.wall_time_ms, 2),
            "net_ms": round(self.net_ms, 2),
            "preflight": {
                "model_match": self.preflight_model_match,
                "retrieval_match": self.preflight_retrieval_match,
                "budget_match": self.preflight_budget_match,
                "warming_action": self.preflight_warming_action,
                "warming_useful": self.preflight_warming_useful,
            },
            "inheritance": {
                "was_inherited": self.was_inherited,
                "state": self.inheritance_state,
                "confidence": round(self.inheritance_confidence, 3),
            },
            "cache": {
                "hit": self.cache_hit,
                "saved_ms": round(self.cache_saved_ms, 2),
            },
            "kv": {
                "event": self.kv_event,
                "prefix_tokens_saved": self.kv_prefix_tokens_saved,
                "saved_ms": round(self.kv_saved_ms, 2),
            },
            "model_swap": {
                "did_swap": self.did_swap,
                "time_ms": round(self.swap_time_ms, 2),
            },
            "quality": {
                "passed": self.quality_passed,
                "verdict": self.quality_verdict,
                "cost_ms": round(self.quality_cost_ms, 2),
            },
            "route": {
                "model": self.route_model,
                "retrieval": self.route_retrieval,
                "budget": self.route_budget,
                "inherited": self.route_inherited_flag,
            },
            "gen": {
                "time_ms": round(self.gen_time_ms, 2),
                "n_tokens": self.n_tokens,
            },
            "reset_reason": self.reset_reason,
        }


class RoutingLedger:
    """Session-level routing economy tracker.

    Records a LedgerEntry per query and computes cumulative profit/loss
    broken down by mechanism: cache, KV reuse, inheritance, preflight,
    model swaps, quality failures.

    Usage::

        ledger = RoutingLedger()

        # In process_query():
        entry = LedgerEntry(query_idx=ledger.n_queries, timestamp=time.time())
        # ... fill entry fields ...
        ledger.record(entry)

        # At end of session:
        summary = ledger.summary()
        # summary["net_profit_ms"], summary["by_mechanism"], etc.
    """

    def __init__(self):
        self._entries: List[LedgerEntry] = []

    @property
    def n_queries(self) -> int:
        return len(self._entries)

    def record(self, entry: LedgerEntry) -> None:
        """Record a completed query's ledger entry."""
        self._entries.append(entry)

    def summary(self) -> Dict[str, Any]:
        """Cumulative profit/loss broken down by mechanism.

        Returns a dict suitable for receipt embedding.
        """
        n = len(self._entries)
        if n == 0:
            return {
                "n_queries": 0,
                "net_profit_ms": 0.0,
                "by_mechanism": {},
                "total_wall_ms": 0.0,
            }

        # ── Per-mechanism aggregation ──
        cache_profit = 0.0
        cache_hits = 0

        kv_profit = 0.0
        kv_reuses = 0

        inherit_count = 0

        preflight_correct = 0
        preflight_warming = 0

        swap_cost = 0.0
        swap_count = 0

        quality_cost = 0.0
        quality_failures = 0

        total_wall = 0.0
        total_gen = 0.0
        total_tokens = 0

        for e in self._entries:
            total_wall += e.wall_time_ms
            total_gen += e.gen_time_ms
            total_tokens += e.n_tokens

            # Cache
            if e.cache_hit:
                cache_hits += 1
                cache_profit += e.cache_saved_ms

            # KV
            if e.kv_event == "reused":
                kv_reuses += 1
                kv_profit += e.kv_saved_ms

            # Inheritance
            if e.was_inherited:
                inherit_count += 1

            # Preflight
            if e.preflight_model_match and e.preflight_retrieval_match and e.preflight_budget_match:
                preflight_correct += 1
            if e.preflight_warming_useful:
                preflight_warming += 1

            # Model swap
            if e.did_swap:
                swap_count += 1
                swap_cost += e.swap_time_ms

            # Quality failure
            if not e.quality_passed:
                quality_failures += 1
                quality_cost += e.quality_cost_ms

        net_profit = cache_profit + kv_profit - swap_cost - quality_cost

        return {
            "n_queries": n,
            "total_wall_ms": round(total_wall, 2),
            "total_gen_ms": round(total_gen, 2),
            "total_tokens": total_tokens,
            "net_profit_ms": round(net_profit, 2),
            "by_mechanism": {
                "response_cache": {
                    "hits": cache_hits,
                    "hit_rate": round(cache_hits / n, 3),
                    "profit_ms": round(cache_profit, 2),
                },
                "kv_prefix_reuse": {
                    "reuses": kv_reuses,
                    "reuse_rate": round(kv_reuses / n, 3),
                    "profit_ms": round(kv_profit, 2),
                },
                "inheritance": {
                    "inherited": inherit_count,
                    "inheritance_rate": round(inherit_count / n, 3),
                    "profit_ms": 0.0,  # currently "first do no harm" — no skip savings yet
                    "note": "inheritance confirms route but does not yet skip classification",
                },
                "preflight": {
                    "accuracy": round(preflight_correct / n, 3),
                    "warming_useful": preflight_warming,
                    "warming_rate": round(preflight_warming / n, 3),
                },
                "model_swap": {
                    "count": swap_count,
                    "cost_ms": round(swap_cost, 2),
                },
                "quality_gate": {
                    "failures": quality_failures,
                    "failure_rate": round(quality_failures / n, 3),
                    "cost_ms": round(quality_cost, 2),
                },
            },
        }

    def entries_as_dicts(self) -> List[dict]:
        """All entries as dicts (for receipt detail)."""
        return [e.as_dict() for e in self._entries]

    def recent(self, n: int = 5) -> List[LedgerEntry]:
        """Last n entries."""
        return self._entries[-n:]
