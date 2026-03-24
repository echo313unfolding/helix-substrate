"""
Tests for WO-TRANSFORM-01: Transform law — system state transitions.

Tests every coupling rule, every illegal state combination, every degradation
path, and every carry rule defined in the transform law.
"""

import pytest

from helix_substrate.transform_law import (
    RouteState, CacheEvent, QualityState, BudgetState,
    ServiceLevel, SystemState,
    COUPLING_RULES, CARRY_RULES, DEGRADATION_SEQUENCE,
    QUALITY_FLOOR, DEFAULT_QUALITY_FLOOR,
    check_illegal_states, required_transitions,
    get_quality_floor, can_degrade_to, degradation_path,
    get_carry_rule, law_summary,
)


# ── Degradation ladder ordering ──

class TestServiceLevelOrdering:

    def test_full_is_highest(self):
        assert ServiceLevel.FULL > ServiceLevel.INHERITED
        assert ServiceLevel.FULL > ServiceLevel.DENIED

    def test_denied_is_lowest(self):
        assert ServiceLevel.DENIED < ServiceLevel.ADMIT_UNKNOWN
        assert ServiceLevel.DENIED < ServiceLevel.FULL

    def test_ladder_is_monotonically_decreasing(self):
        for i in range(len(DEGRADATION_SEQUENCE) - 1):
            assert DEGRADATION_SEQUENCE[i] > DEGRADATION_SEQUENCE[i + 1], \
                f"{DEGRADATION_SEQUENCE[i].name} should be > {DEGRADATION_SEQUENCE[i+1].name}"

    def test_all_levels_in_sequence(self):
        assert set(DEGRADATION_SEQUENCE) == set(ServiceLevel)


# ── Quality floors ──

class TestQualityFloors:

    def test_code_cannot_degrade_below_capped(self):
        assert can_degrade_to("code", ServiceLevel.CAPPED)
        assert not can_degrade_to("code", ServiceLevel.FALLBACK)
        assert not can_degrade_to("code", ServiceLevel.ADMIT_UNKNOWN)

    def test_factual_can_degrade_to_web(self):
        assert can_degrade_to("factual", ServiceLevel.WEB_PASSTHROUGH)
        assert can_degrade_to("factual", ServiceLevel.CACHE_REPLAY)
        assert not can_degrade_to("factual", ServiceLevel.FALLBACK)

    def test_neutral_can_fall_back(self):
        assert can_degrade_to("neutral", ServiceLevel.FALLBACK)
        assert not can_degrade_to("neutral", ServiceLevel.ADMIT_UNKNOWN)

    def test_unknown_type_uses_default(self):
        floor = get_quality_floor("unknown_type")
        assert floor == DEFAULT_QUALITY_FLOOR

    def test_every_type_has_path_ending_in_denied(self):
        for qt in QUALITY_FLOOR:
            path = degradation_path(qt)
            assert path[-1] == ServiceLevel.DENIED, \
                f"{qt} path doesn't end in DENIED"
            assert ServiceLevel.ADMIT_UNKNOWN in path, \
                f"{qt} path missing ADMIT_UNKNOWN"

    def test_degradation_path_respects_floor(self):
        for qt, floor in QUALITY_FLOOR.items():
            path = degradation_path(qt)
            # All levels above floor should be present
            for level in ServiceLevel:
                if level >= floor:
                    assert level in path, \
                        f"{qt}: {level.name} >= floor {floor.name} but not in path"


# ── Coupling rules ──

class TestCouplingRules:

    def test_quality_fail_forces_route_reset(self):
        effects = required_transitions("quality_fail")
        assert len(effects) == 1
        assert effects[0].forced_axis == "route"
        assert effects[0].forced_to == "RESET"

    def test_model_swap_forces_route_reset_and_cache_clear(self):
        effects = required_transitions("model_swap")
        assert len(effects) == 2
        axes = {e.forced_axis: e.forced_to for e in effects}
        assert axes["route"] == "RESET"
        assert axes["cache"] == "CLEAR"

    def test_hard_drift_forces_route_reset(self):
        effects = required_transitions("hard_drift")
        assert len(effects) == 1
        assert effects[0].forced_to == "RESET"

    def test_budget_denied_forces_denied_service(self):
        effects = required_transitions("budget_denied")
        assert len(effects) == 1
        assert effects[0].forced_to == "DENIED"

    def test_unknown_event_returns_empty(self):
        assert required_transitions("nonexistent_event") == []


# ── Illegal states ──

class TestIllegalStates:

    def test_legal_state_no_violations(self):
        state = SystemState(
            route=RouteState.INHERITED,
            cache=CacheEvent.MISS,
            quality=QualityState.PASS,
            budget=BudgetState.NORMAL,
            service_level=ServiceLevel.INHERITED,
        )
        assert check_illegal_states(state) == []

    def test_budget_denied_with_generation(self):
        state = SystemState(
            route=RouteState.COLD,
            cache=CacheEvent.MISS,
            quality=QualityState.UNCHECKED,
            budget=BudgetState.DENIED,
            service_level=ServiceLevel.FULL,
        )
        violations = check_illegal_states(state)
        assert len(violations) >= 1
        assert "DENIED" in violations[0]

    def test_budget_denied_with_cache_replay_is_legal(self):
        """Serving from cache when budget is denied is OK — no generation cost."""
        state = SystemState(
            route=RouteState.WARM_CANDIDATE,
            cache=CacheEvent.RESPONSE_HIT,
            quality=QualityState.UNCHECKED,
            budget=BudgetState.DENIED,
            service_level=ServiceLevel.CACHE_REPLAY,
        )
        assert check_illegal_states(state) == []

    def test_cache_hit_with_quality_fail(self):
        state = SystemState(
            route=RouteState.INHERITED,
            cache=CacheEvent.RESPONSE_HIT,
            quality=QualityState.FAIL_REPETITION,
            budget=BudgetState.NORMAL,
            service_level=ServiceLevel.CACHE_REPLAY,
        )
        violations = check_illegal_states(state)
        assert any("cache" in v.lower() or "Cache" in v for v in violations)

    def test_kv_reuse_with_cold_route(self):
        state = SystemState(
            route=RouteState.COLD,
            cache=CacheEvent.KV_REUSE,
            quality=QualityState.PASS,
            budget=BudgetState.NORMAL,
            service_level=ServiceLevel.KV_PREFIXED,
        )
        violations = check_illegal_states(state)
        assert any("KV" in v for v in violations)

    def test_inherited_service_with_cold_route(self):
        state = SystemState(
            route=RouteState.COLD,
            cache=CacheEvent.MISS,
            quality=QualityState.PASS,
            budget=BudgetState.NORMAL,
            service_level=ServiceLevel.INHERITED,
        )
        violations = check_illegal_states(state)
        assert any("INHERITED" in v for v in violations)

    def test_quality_fail_with_high_service_level(self):
        state = SystemState(
            route=RouteState.UNSTABLE,
            cache=CacheEvent.MISS,
            quality=QualityState.FAIL_EMPTY,
            budget=BudgetState.NORMAL,
            service_level=ServiceLevel.FULL,
        )
        violations = check_illegal_states(state)
        assert any("Quality" in v or "quality" in v for v in violations)


# ── Carry rules ──

class TestCarryRules:

    def test_model_swap_invalidates_kv_and_route(self):
        rule = get_carry_rule("model_swap")
        assert rule is not None
        assert rule.response_cache_safe is True
        assert rule.kv_cache_safe is False
        assert rule.route_safe is False

    def test_quality_failure_preserves_kv(self):
        rule = get_carry_rule("quality_failure")
        assert rule is not None
        assert rule.kv_cache_safe is True
        assert rule.route_safe is False

    def test_same_model_carries_everything(self):
        rule = get_carry_rule("same_model_sequential")
        assert rule is not None
        assert rule.response_cache_safe is True
        assert rule.kv_cache_safe is True
        assert rule.route_safe is True

    def test_hard_drift_invalidates_kv_and_route(self):
        rule = get_carry_rule("hard_drift")
        assert rule is not None
        assert rule.kv_cache_safe is False
        assert rule.route_safe is False
        assert rule.response_cache_safe is True

    def test_budget_exhaustion_preserves_everything(self):
        rule = get_carry_rule("budget_exhausted")
        assert rule is not None
        assert rule.response_cache_safe is True
        assert rule.kv_cache_safe is True
        assert rule.route_safe is True

    def test_session_reset_clears_everything(self):
        rule = get_carry_rule("session_reset")
        assert rule is not None
        assert rule.response_cache_safe is False
        assert rule.kv_cache_safe is False
        assert rule.route_safe is False

    def test_unknown_transition_returns_none(self):
        assert get_carry_rule("alien_invasion") is None


# ── Cross-axis consistency ──

class TestCrossAxisConsistency:

    def test_coupling_rules_reference_valid_axes(self):
        valid_axes = {"route", "cache", "quality", "budget", "service_level"}
        for rule in COUPLING_RULES:
            assert rule.source_axis in valid_axes, \
                f"Unknown source axis: {rule.source_axis}"
            assert rule.forced_axis in valid_axes, \
                f"Unknown forced axis: {rule.forced_axis}"

    def test_all_carry_rules_have_reason(self):
        for rule in CARRY_RULES:
            assert rule.reason, f"Carry rule {rule.transition} has no reason"

    def test_all_coupling_rules_have_reason(self):
        for rule in COUPLING_RULES:
            assert rule.reason, f"Coupling rule {rule.trigger} has no reason"


# ── Law summary (serialization) ──

class TestLawSummary:

    def test_summary_is_serializable(self):
        import json
        summary = law_summary()
        # Should not raise
        serialized = json.dumps(summary)
        assert len(serialized) > 0

    def test_summary_has_all_sections(self):
        summary = law_summary()
        assert "axes" in summary
        assert "service_levels" in summary
        assert "quality_floors" in summary
        assert "coupling_rules" in summary
        assert "carry_rules" in summary
        assert "degradation_paths" in summary

    def test_summary_counts_match(self):
        summary = law_summary()
        assert summary["n_coupling_rules"] == len(COUPLING_RULES)
        assert summary["n_carry_rules"] == len(CARRY_RULES)
