"""
Session Advisor — WO-J

Tracks per-session cycle outcomes and produces deterministic adaptation
hints for route selection, shaping policy, and executor preference.

Not learned routing. Deterministic rules based on observed patterns
within the current session. All decisions are receipted.

Adaptation rules:
    1. Route bias — if >50% of recent cycles needed shaping, prefer lighter routes
    2. Memory trim bias — if memory keeps getting trimmed, go aggressive early
    3. Verifier lock — if any recent cycle was blocked/failed, lock verifier
    4. Route continuity — if last 3 successful cycles used same route, prefer it
    5. Executor preference — track success/error rate per executor

Session preferences never override hard policy invariants (risky query locks,
symbolic route locks). They only bias the *starting point* for each cycle.

Work Order: WO-J (Session-Aware Adaptation)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


SCHEMA = "session_advisor:v1"

# Route downgrade chain (same as route_shaper.DOWNGRADE_MAP)
_LIGHTER_ROUTE = {
    "plan_code_verify": "code_verify",
    "code_verify": "direct_code",
    "plan_verify": "direct_plan",
}


# ---------------------------------------------------------------------------
# Cycle outcome
# ---------------------------------------------------------------------------

@dataclass
class CycleOutcome:
    """Compact metrics from one completed cycle."""
    cycle_number: int
    route_name: str
    verdict: str               # resident_ok, swap_required, deny_budget, error
    blocked: bool
    executor: Optional[str]    # lobe_inference, web_search, route_only, etc.
    wall_time_s: float
    tokens_generated: int
    shaped: bool
    shaping_actions: List[str]  # ["cap_output", "downgrade_route", ...]
    verifier_passed: bool
    memory_helped: bool         # did memory recall find matches?
    memory_trimmed: bool        # was memory trimmed by shaper?

    @staticmethod
    def from_cycle_data(
        cycle_number: int,
        route_name: str,
        blocked: bool,
        exec_result: dict,
        budget_section: Optional[dict],
        memory_result: dict,
        cost: dict,
    ) -> CycleOutcome:
        """Extract CycleOutcome from the data already collected by controller.

        This avoids re-parsing receipts — it consumes what the controller
        already has in-memory at the end of each cycle.
        """
        # Budget/shaping info
        verdict = "unknown"
        shaped = False
        shaping_actions: List[str] = []
        memory_trimmed = False

        if budget_section:
            verdict = budget_section.get("verdict", "unknown")
            shaped = budget_section.get("shaped", False)
            for a in budget_section.get("actions", []):
                rule = a.get("rule", "") if isinstance(a, dict) else ""
                if rule:
                    shaping_actions.append(rule)
                if rule and "trim_memory" in rule and a.get("policy_allowed", True):
                    memory_trimmed = True

        # Executor info
        executor = exec_result.get("executor")

        # Verifier info — check route_receipt steps for verification
        verifier_passed = True
        route_receipt = exec_result.get("route_receipt")
        if route_receipt and isinstance(route_receipt, dict):
            for step in route_receipt.get("steps", []):
                v = step.get("verification", {})
                if v and v.get("verification_result") == "fail":
                    verifier_passed = False

        # Memory
        memory_helped = memory_result.get("match_count", 0) > 0

        return CycleOutcome(
            cycle_number=cycle_number,
            route_name=route_name,
            verdict=verdict,
            blocked=blocked,
            executor=executor,
            wall_time_s=cost.get("wall_time_s", 0.0),
            tokens_generated=exec_result.get("tokens", 0),
            shaped=shaped,
            shaping_actions=shaping_actions,
            verifier_passed=verifier_passed,
            memory_helped=memory_helped,
            memory_trimmed=memory_trimmed,
        )


# ---------------------------------------------------------------------------
# Session preferences
# ---------------------------------------------------------------------------

@dataclass
class SessionPreferences:
    """Current session-level preferences derived from outcome history."""
    prefer_lighter_routes: bool = False
    aggressive_memory_trim: bool = False
    verifier_locked_by_session: bool = False
    route_bias: Optional[str] = None
    executor_preference: Optional[str] = None
    low_latency_mode: bool = False
    adaptation_reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------

class SessionAdvisor:
    """
    Tracks per-session cycle outcomes and produces adaptation hints.

    Rules are deterministic and receipted. No learned routing.
    Preferences are advisory — hard policy invariants from ShapingPolicy
    always take precedence.
    """

    def __init__(self, max_history: int = 20, window: int = 5):
        """
        Args:
            max_history: Maximum outcomes to retain.
            window: Number of recent outcomes for trend detection.
        """
        self._outcomes: List[CycleOutcome] = []
        self._max_history = max_history
        self._window = window
        self._executor_stats: Dict[str, Dict[str, int]] = {}  # name → {ok, error, blocked}

    @property
    def cycle_count(self) -> int:
        return len(self._outcomes)

    def record_outcome(self, outcome: CycleOutcome) -> None:
        """Record a cycle outcome for future bias computation."""
        self._outcomes.append(outcome)
        if len(self._outcomes) > self._max_history:
            self._outcomes = self._outcomes[-self._max_history:]

        # Track executor stats
        if outcome.executor:
            stats = self._executor_stats.setdefault(
                outcome.executor, {"ok": 0, "error": 0, "blocked": 0})
            if outcome.blocked:
                stats["blocked"] += 1
            elif outcome.verdict == "deny_budget":
                stats["error"] += 1
            else:
                stats["ok"] += 1

    def get_preferences(self) -> SessionPreferences:
        """Compute current session preferences from outcome history.

        Rules (all deterministic):
            1. >50% of recent cycles shaped → prefer lighter routes
            2. >50% of recent cycles trimmed memory → aggressive memory trim
            3. Any recent block or verifier fail → lock verifier for session
            4. Last 3 successful cycles same route → route continuity bias
            5. High average wall time → low latency mode
            6. Executor with best success rate → executor preference
        """
        if not self._outcomes:
            return SessionPreferences()

        recent = self._outcomes[-self._window:]
        n = len(recent)
        prefs = SessionPreferences()

        # Rule 1: Shaping frequency → prefer lighter routes
        shaped_rate = sum(1 for o in recent if o.shaped) / n
        if shaped_rate > 0.5:
            prefs.prefer_lighter_routes = True
            prefs.adaptation_reasons.append(
                f"shaped_rate={shaped_rate:.0%} in last {n} cycles")

        # Rule 2: Memory trim frequency → aggressive trim
        trim_rate = sum(1 for o in recent if o.memory_trimmed) / n
        if trim_rate > 0.5:
            prefs.aggressive_memory_trim = True
            prefs.adaptation_reasons.append(
                f"memory_trim_rate={trim_rate:.0%} in last {n} cycles")

        # Rule 3: Block or verifier fail → session verifier lock
        has_block = any(o.blocked for o in recent)
        has_verify_fail = any(not o.verifier_passed for o in recent)
        if has_block or has_verify_fail:
            prefs.verifier_locked_by_session = True
            reasons = []
            if has_block:
                reasons.append("blocked_cycle")
            if has_verify_fail:
                reasons.append("verifier_fail")
            prefs.adaptation_reasons.append(
                f"verifier_locked: {'+'.join(reasons)} in recent window")

        # Rule 4: Route continuity — last 3 successful cycles same route
        successful = [o for o in recent
                      if not o.blocked and o.verdict != "deny_budget"]
        if len(successful) >= 3:
            last_3 = [o.route_name for o in successful[-3:]]
            if len(set(last_3)) == 1:
                prefs.route_bias = last_3[0]
                prefs.adaptation_reasons.append(
                    f"route_continuity: last 3 successes all {last_3[0]}")

        # Rule 5: High wall time → low latency mode
        avg_time = sum(o.wall_time_s for o in recent) / n
        if avg_time > 10.0:
            prefs.low_latency_mode = True
            prefs.adaptation_reasons.append(
                f"avg_wall_time={avg_time:.1f}s > 10s threshold")

        # Rule 6: Executor preference — best success rate
        if self._executor_stats:
            best_name = None
            best_rate = -1.0
            for name, stats in self._executor_stats.items():
                total = stats["ok"] + stats["error"] + stats["blocked"]
                if total >= 2:  # need at least 2 attempts
                    rate = stats["ok"] / total
                    if rate > best_rate:
                        best_rate = rate
                        best_name = name
            if best_name and best_rate > 0.5:
                prefs.executor_preference = best_name

        return prefs

    def get_policy_overrides(self) -> dict:
        """Get ShapingPolicy field overrides based on session history.

        These are merged into the query-level policy as additional
        constraints. They can only *tighten* policy, never loosen it.

        Returns:
            Dict of ShapingPolicy field names → values.
        """
        prefs = self.get_preferences()
        overrides: dict = {}

        if prefs.verifier_locked_by_session:
            overrides["allow_verifier_drop"] = False

        return overrides

    def suggest_route(self, proposed_route: str) -> Tuple[str, str]:
        """Suggest route adjustment based on session bias.

        Returns (suggested_route, reason). If no change, reason is empty.
        This is advisory — the controller can ignore it.
        """
        prefs = self.get_preferences()

        # Route continuity: if session has bias and it matches query type,
        # suggest it (but don't force)
        if prefs.route_bias and prefs.route_bias != proposed_route:
            return (prefs.route_bias,
                    f"session_continuity: last 3 successes on {prefs.route_bias}")

        # Lighter route if budget pressure is common
        if prefs.prefer_lighter_routes and proposed_route in _LIGHTER_ROUTE:
            lighter = _LIGHTER_ROUTE[proposed_route]
            return (lighter,
                    f"session_pressure: shaped_rate>50%, suggesting {lighter}")

        # Low latency: prefer direct routes
        if prefs.low_latency_mode and proposed_route in _LIGHTER_ROUTE:
            lighter = _LIGHTER_ROUTE[proposed_route]
            return (lighter,
                    f"low_latency: avg_time>10s, suggesting {lighter}")

        return (proposed_route, "")

    def suggest_executor(self, proposed: str) -> Tuple[str, str]:
        """Suggest executor based on session success rates.

        Returns (suggested_executor, reason). Advisory only.
        """
        prefs = self.get_preferences()
        if (prefs.executor_preference
                and prefs.executor_preference != proposed
                and proposed in self._executor_stats):
            stats = self._executor_stats[proposed]
            total = stats["ok"] + stats["error"] + stats["blocked"]
            if total >= 2:
                fail_rate = (stats["error"] + stats["blocked"]) / total
                if fail_rate > 0.5:
                    return (prefs.executor_preference,
                            f"executor_{proposed}_fail_rate={fail_rate:.0%}")
        return (proposed, "")

    def reset(self) -> None:
        """Reset all session state. For testing or session boundary."""
        self._outcomes.clear()
        self._executor_stats.clear()

    def to_receipt(self) -> dict:
        """Receipt of current adaptation state."""
        prefs = self.get_preferences()
        return {
            "schema": SCHEMA,
            "n_outcomes": len(self._outcomes),
            "preferences": {
                "prefer_lighter_routes": prefs.prefer_lighter_routes,
                "aggressive_memory_trim": prefs.aggressive_memory_trim,
                "verifier_locked_by_session": prefs.verifier_locked_by_session,
                "route_bias": prefs.route_bias,
                "executor_preference": prefs.executor_preference,
                "low_latency_mode": prefs.low_latency_mode,
            },
            "adaptation_reasons": prefs.adaptation_reasons,
            "policy_overrides": self.get_policy_overrides(),
            "executor_stats": dict(self._executor_stats),
        }
