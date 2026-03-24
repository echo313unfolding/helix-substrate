"""
Tests for SessionAdvisor (WO-J).

Tests cover:
1. Empty advisor returns neutral preferences
2. Recording outcomes and history limits
3. Route bias from shaping pressure
4. Memory trim bias from trim frequency
5. Verifier lock from blocked/failed cycles
6. Route continuity from repeated success
7. Low latency mode from high wall time
8. Executor preference from success rates
9. Policy overrides only tighten
10. Route suggestion logic
11. Executor suggestion logic
12. CycleOutcome.from_cycle_data factory
13. Receipt serialization
14. Reset clears all state
15. Integration: multi-cycle scenario
"""
import json
import pytest

from helix_substrate.session_advisor import (
    SessionAdvisor, SessionPreferences, CycleOutcome, SCHEMA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outcome(
    cycle=1, route="direct_plan", verdict="resident_ok",
    blocked=False, executor="lobe_inference", wall_time=1.0,
    tokens=42, shaped=False, shaping_actions=None,
    verifier_passed=True, memory_helped=False, memory_trimmed=False,
):
    return CycleOutcome(
        cycle_number=cycle,
        route_name=route,
        verdict=verdict,
        blocked=blocked,
        executor=executor,
        wall_time_s=wall_time,
        tokens_generated=tokens,
        shaped=shaped,
        shaping_actions=shaping_actions or [],
        verifier_passed=verifier_passed,
        memory_helped=memory_helped,
        memory_trimmed=memory_trimmed,
    )


# ---------------------------------------------------------------------------
# Empty advisor
# ---------------------------------------------------------------------------

class TestEmptyAdvisor:
    def test_neutral_preferences(self):
        advisor = SessionAdvisor()
        prefs = advisor.get_preferences()
        assert not prefs.prefer_lighter_routes
        assert not prefs.aggressive_memory_trim
        assert not prefs.verifier_locked_by_session
        assert prefs.route_bias is None
        assert prefs.executor_preference is None
        assert not prefs.low_latency_mode
        assert len(prefs.adaptation_reasons) == 0

    def test_no_overrides(self):
        advisor = SessionAdvisor()
        assert advisor.get_policy_overrides() == {}

    def test_no_route_suggestion(self):
        advisor = SessionAdvisor()
        route, reason = advisor.suggest_route("code_verify")
        assert route == "code_verify"
        assert reason == ""

    def test_cycle_count_zero(self):
        advisor = SessionAdvisor()
        assert advisor.cycle_count == 0


# ---------------------------------------------------------------------------
# Recording and limits
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_increments(self):
        advisor = SessionAdvisor()
        advisor.record_outcome(_make_outcome(cycle=1))
        assert advisor.cycle_count == 1
        advisor.record_outcome(_make_outcome(cycle=2))
        assert advisor.cycle_count == 2

    def test_max_history(self):
        advisor = SessionAdvisor(max_history=5)
        for i in range(10):
            advisor.record_outcome(_make_outcome(cycle=i))
        assert advisor.cycle_count == 5

    def test_executor_stats_tracked(self):
        advisor = SessionAdvisor()
        advisor.record_outcome(_make_outcome(executor="lobe_inference"))
        advisor.record_outcome(_make_outcome(
            executor="lobe_inference", blocked=True))
        receipt = advisor.to_receipt()
        stats = receipt["executor_stats"]["lobe_inference"]
        assert stats["ok"] == 1
        assert stats["blocked"] == 1


# ---------------------------------------------------------------------------
# Rule 1: Shaping pressure → lighter routes
# ---------------------------------------------------------------------------

class TestShapingBias:
    def test_no_bias_when_no_shaping(self):
        advisor = SessionAdvisor(window=5)
        for i in range(5):
            advisor.record_outcome(_make_outcome(cycle=i, shaped=False))
        prefs = advisor.get_preferences()
        assert not prefs.prefer_lighter_routes

    def test_bias_when_mostly_shaped(self):
        advisor = SessionAdvisor(window=5)
        for i in range(3):
            advisor.record_outcome(_make_outcome(cycle=i, shaped=True))
        for i in range(2):
            advisor.record_outcome(_make_outcome(cycle=i + 3, shaped=False))
        prefs = advisor.get_preferences()
        assert prefs.prefer_lighter_routes
        assert any("shaped_rate" in r for r in prefs.adaptation_reasons)


# ---------------------------------------------------------------------------
# Rule 2: Memory trim frequency
# ---------------------------------------------------------------------------

class TestMemoryTrimBias:
    def test_no_aggressive_when_no_trims(self):
        advisor = SessionAdvisor(window=5)
        for i in range(5):
            advisor.record_outcome(_make_outcome(cycle=i, memory_trimmed=False))
        prefs = advisor.get_preferences()
        assert not prefs.aggressive_memory_trim

    def test_aggressive_when_mostly_trimmed(self):
        advisor = SessionAdvisor(window=4)
        for i in range(3):
            advisor.record_outcome(_make_outcome(cycle=i, memory_trimmed=True))
        advisor.record_outcome(_make_outcome(cycle=3, memory_trimmed=False))
        prefs = advisor.get_preferences()
        assert prefs.aggressive_memory_trim


# ---------------------------------------------------------------------------
# Rule 3: Verifier lock
# ---------------------------------------------------------------------------

class TestVerifierLock:
    def test_locked_on_blocked_cycle(self):
        advisor = SessionAdvisor(window=5)
        advisor.record_outcome(_make_outcome(cycle=1, blocked=True))
        prefs = advisor.get_preferences()
        assert prefs.verifier_locked_by_session

    def test_locked_on_verifier_fail(self):
        advisor = SessionAdvisor(window=5)
        advisor.record_outcome(_make_outcome(cycle=1, verifier_passed=False))
        prefs = advisor.get_preferences()
        assert prefs.verifier_locked_by_session

    def test_not_locked_when_clean(self):
        advisor = SessionAdvisor(window=5)
        for i in range(5):
            advisor.record_outcome(_make_outcome(
                cycle=i, blocked=False, verifier_passed=True))
        prefs = advisor.get_preferences()
        assert not prefs.verifier_locked_by_session

    def test_policy_override_tightens(self):
        advisor = SessionAdvisor(window=5)
        advisor.record_outcome(_make_outcome(cycle=1, blocked=True))
        overrides = advisor.get_policy_overrides()
        assert overrides.get("allow_verifier_drop") is False


# ---------------------------------------------------------------------------
# Rule 4: Route continuity
# ---------------------------------------------------------------------------

class TestRouteContinuity:
    def test_bias_after_3_same_route(self):
        advisor = SessionAdvisor(window=5)
        for i in range(3):
            advisor.record_outcome(_make_outcome(
                cycle=i, route="code_verify"))
        prefs = advisor.get_preferences()
        assert prefs.route_bias == "code_verify"

    def test_no_bias_with_mixed_routes(self):
        advisor = SessionAdvisor(window=5)
        advisor.record_outcome(_make_outcome(cycle=1, route="code_verify"))
        advisor.record_outcome(_make_outcome(cycle=2, route="direct_plan"))
        advisor.record_outcome(_make_outcome(cycle=3, route="code_verify"))
        prefs = advisor.get_preferences()
        assert prefs.route_bias is None

    def test_no_bias_from_blocked_cycles(self):
        advisor = SessionAdvisor(window=5)
        for i in range(3):
            advisor.record_outcome(_make_outcome(
                cycle=i, route="code_verify", blocked=True))
        prefs = advisor.get_preferences()
        # Blocked cycles don't count for route continuity
        assert prefs.route_bias is None


# ---------------------------------------------------------------------------
# Rule 5: Low latency mode
# ---------------------------------------------------------------------------

class TestLowLatency:
    def test_triggered_by_high_wall_time(self):
        advisor = SessionAdvisor(window=3)
        for i in range(3):
            advisor.record_outcome(_make_outcome(cycle=i, wall_time=15.0))
        prefs = advisor.get_preferences()
        assert prefs.low_latency_mode

    def test_not_triggered_by_fast_cycles(self):
        advisor = SessionAdvisor(window=3)
        for i in range(3):
            advisor.record_outcome(_make_outcome(cycle=i, wall_time=1.0))
        prefs = advisor.get_preferences()
        assert not prefs.low_latency_mode


# ---------------------------------------------------------------------------
# Rule 6: Executor preference
# ---------------------------------------------------------------------------

class TestExecutorPreference:
    def test_prefers_successful_executor(self):
        advisor = SessionAdvisor(window=10)
        # lobe_inference: 3 ok, 0 error
        for i in range(3):
            advisor.record_outcome(_make_outcome(
                cycle=i, executor="lobe_inference"))
        # web_search: 1 ok, 2 error (deny_budget counts as error)
        advisor.record_outcome(_make_outcome(
            cycle=4, executor="web_search"))
        advisor.record_outcome(_make_outcome(
            cycle=5, executor="web_search", verdict="deny_budget"))
        advisor.record_outcome(_make_outcome(
            cycle=6, executor="web_search", verdict="deny_budget"))

        prefs = advisor.get_preferences()
        assert prefs.executor_preference == "lobe_inference"

    def test_no_preference_with_insufficient_data(self):
        advisor = SessionAdvisor()
        advisor.record_outcome(_make_outcome(cycle=1, executor="lobe_inference"))
        prefs = advisor.get_preferences()
        # Only 1 attempt — not enough
        assert prefs.executor_preference is None


# ---------------------------------------------------------------------------
# Route suggestion
# ---------------------------------------------------------------------------

class TestRouteSuggestion:
    def test_no_change_when_neutral(self):
        advisor = SessionAdvisor()
        route, reason = advisor.suggest_route("code_verify")
        assert route == "code_verify"
        assert reason == ""

    def test_continuity_suggestion(self):
        advisor = SessionAdvisor(window=5)
        for i in range(3):
            advisor.record_outcome(_make_outcome(
                cycle=i, route="direct_plan"))
        route, reason = advisor.suggest_route("code_verify")
        assert route == "direct_plan"
        assert "continuity" in reason

    def test_lighter_suggestion_under_pressure(self):
        advisor = SessionAdvisor(window=5)
        # Use varied routes so continuity bias doesn't trigger
        routes = ["code_verify", "direct_plan", "code_verify", "direct_plan"]
        for i, r in enumerate(routes):
            advisor.record_outcome(_make_outcome(
                cycle=i, route=r, shaped=True))
        route, reason = advisor.suggest_route("plan_code_verify")
        assert route == "code_verify"
        assert "pressure" in reason

    def test_no_change_for_already_lightest(self):
        advisor = SessionAdvisor(window=5)
        routes = ["code_verify", "direct_plan", "code_verify", "direct_plan"]
        for i, r in enumerate(routes):
            advisor.record_outcome(_make_outcome(
                cycle=i, route=r, shaped=True))
        route, reason = advisor.suggest_route("direct_code")
        # direct_code is not in _LIGHTER_ROUTE, so no suggestion
        assert route == "direct_code"


# ---------------------------------------------------------------------------
# Executor suggestion
# ---------------------------------------------------------------------------

class TestExecutorSuggestion:
    def test_no_change_when_no_data(self):
        advisor = SessionAdvisor()
        exec_name, reason = advisor.suggest_executor("lobe_inference")
        assert exec_name == "lobe_inference"
        assert reason == ""

    def test_suggests_better_executor(self):
        advisor = SessionAdvisor()
        # lobe_inference: 5 ok
        for i in range(5):
            advisor.record_outcome(_make_outcome(
                cycle=i, executor="lobe_inference"))
        # web_search: 1 ok, 2 error
        advisor.record_outcome(_make_outcome(
            cycle=6, executor="web_search"))
        advisor.record_outcome(_make_outcome(
            cycle=7, executor="web_search", verdict="deny_budget"))
        advisor.record_outcome(_make_outcome(
            cycle=8, executor="web_search", verdict="deny_budget"))

        exec_name, reason = advisor.suggest_executor("web_search")
        assert exec_name == "lobe_inference"
        assert "fail_rate" in reason


# ---------------------------------------------------------------------------
# CycleOutcome.from_cycle_data
# ---------------------------------------------------------------------------

class TestCycleOutcomeFactory:
    def test_basic_extraction(self):
        outcome = CycleOutcome.from_cycle_data(
            cycle_number=1,
            route_name="code_verify",
            blocked=False,
            exec_result={
                "executor": "lobe_inference",
                "tokens": 42,
                "route_receipt": {
                    "steps": [
                        {"verification": {"verification_result": "pass"}},
                    ],
                },
            },
            budget_section={
                "verdict": "swap_required",
                "shaped": True,
                "actions": [{"rule": "cap_output", "policy_allowed": True}],
            },
            memory_result={"match_count": 2},
            cost={"wall_time_s": 1.5},
        )
        assert outcome.route_name == "code_verify"
        assert outcome.verdict == "swap_required"
        assert outcome.shaped is True
        assert outcome.shaping_actions == ["cap_output"]
        assert outcome.memory_helped is True
        assert outcome.verifier_passed is True
        assert outcome.executor == "lobe_inference"
        assert outcome.tokens_generated == 42

    def test_verifier_fail_detected(self):
        outcome = CycleOutcome.from_cycle_data(
            cycle_number=1,
            route_name="code_verify",
            blocked=False,
            exec_result={
                "route_receipt": {
                    "steps": [
                        {"verification": {"verification_result": "fail"}},
                    ],
                },
            },
            budget_section=None,
            memory_result={"match_count": 0},
            cost={"wall_time_s": 1.0},
        )
        assert outcome.verifier_passed is False

    def test_memory_trimmed_detected(self):
        outcome = CycleOutcome.from_cycle_data(
            cycle_number=1,
            route_name="direct_plan",
            blocked=False,
            exec_result={},
            budget_section={
                "verdict": "swap_required",
                "shaped": True,
                "actions": [
                    {"rule": "trim_memory_top1", "policy_allowed": True},
                ],
            },
            memory_result={"match_count": 0},
            cost={"wall_time_s": 1.0},
        )
        assert outcome.memory_trimmed is True

    def test_minimal_data(self):
        outcome = CycleOutcome.from_cycle_data(
            cycle_number=0,
            route_name="blocked",
            blocked=True,
            exec_result={},
            budget_section=None,
            memory_result={},
            cost={},
        )
        assert outcome.blocked is True
        assert outcome.verdict == "unknown"
        assert outcome.tokens_generated == 0


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

class TestReceipt:
    def test_receipt_has_schema(self):
        advisor = SessionAdvisor()
        advisor.record_outcome(_make_outcome(cycle=1))
        receipt = advisor.to_receipt()
        assert receipt["schema"] == SCHEMA
        assert receipt["n_outcomes"] == 1
        assert "preferences" in receipt
        assert "adaptation_reasons" in receipt
        assert "policy_overrides" in receipt
        assert "executor_stats" in receipt

    def test_receipt_json_serializable(self):
        advisor = SessionAdvisor()
        for i in range(5):
            advisor.record_outcome(_make_outcome(
                cycle=i, shaped=True, blocked=(i == 2)))
        receipt = advisor.to_receipt()
        json_str = json.dumps(receipt)
        roundtrip = json.loads(json_str)
        assert roundtrip["schema"] == SCHEMA


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all(self):
        advisor = SessionAdvisor()
        for i in range(5):
            advisor.record_outcome(_make_outcome(cycle=i))
        advisor.reset()
        assert advisor.cycle_count == 0
        assert advisor.get_preferences().route_bias is None
        assert advisor.to_receipt()["executor_stats"] == {}


# ---------------------------------------------------------------------------
# Integration: multi-cycle scenario
# ---------------------------------------------------------------------------

class TestIntegrationScenario:
    def test_coding_session_adapts(self):
        """Simulate a coding session that hits budget pressure."""
        advisor = SessionAdvisor(window=5)

        # First 3 cycles: plan_code_verify works fine
        for i in range(3):
            advisor.record_outcome(_make_outcome(
                cycle=i, route="plan_code_verify", shaped=False))

        prefs = advisor.get_preferences()
        assert prefs.route_bias == "plan_code_verify"
        assert not prefs.prefer_lighter_routes

        # Next 3 cycles: budget pressure forces shaping
        for i in range(3):
            advisor.record_outcome(_make_outcome(
                cycle=i + 3, route="plan_code_verify", shaped=True,
                shaping_actions=["cap_output", "downgrade_route"],
                verdict="swap_required"))

        prefs = advisor.get_preferences()
        assert prefs.prefer_lighter_routes
        # Route bias should be gone — last 3 successful were shaped
        # but still plan_code_verify
        route, reason = advisor.suggest_route("plan_code_verify")
        assert route == "code_verify"  # lighter suggestion
        assert "pressure" in reason

    def test_risky_session_locks_verifier(self):
        """Session with blocked cycle locks verifier."""
        advisor = SessionAdvisor(window=5)

        # Normal cycles
        advisor.record_outcome(_make_outcome(cycle=1))
        advisor.record_outcome(_make_outcome(cycle=2))

        # Blocked by verification
        advisor.record_outcome(_make_outcome(cycle=3, blocked=True))

        prefs = advisor.get_preferences()
        assert prefs.verifier_locked_by_session

        overrides = advisor.get_policy_overrides()
        assert overrides["allow_verifier_drop"] is False

    def test_receipt_captures_adaptation(self):
        """Receipt shows why adaptation happened."""
        advisor = SessionAdvisor(window=3)

        for i in range(3):
            advisor.record_outcome(_make_outcome(
                cycle=i, shaped=True, memory_trimmed=True, wall_time=15.0))

        receipt = advisor.to_receipt()
        reasons = receipt["adaptation_reasons"]
        assert any("shaped_rate" in r for r in reasons)
        assert any("memory_trim_rate" in r for r in reasons)
        assert any("avg_wall_time" in r for r in reasons)
        assert receipt["preferences"]["prefer_lighter_routes"] is True
        assert receipt["preferences"]["aggressive_memory_trim"] is True
        assert receipt["preferences"]["low_latency_mode"] is True
