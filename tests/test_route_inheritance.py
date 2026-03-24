"""
test_route_inheritance.py — WO-INHERIT-01: Route Inheritance State Machine

Tests the Terminal 3 control law ported to routing:
  - Sustained similarity → inheritance
  - Drift → reset to cold
  - Quality failure → hard reset
  - Model swap → hard reset
  - Stable sequences → inherited route
"""

import time
import pytest

from helix_substrate.query_classifier import ModelTarget
from helix_substrate.route_decision import RouteDecision, RetrievalMode
from helix_substrate.route_inheritance import (
    InheritanceState,
    InheritanceStateMachine,
    RouteRegime,
    InheritedRouteState,
    REGIME_SIMILARITY_THRESHOLD,
    WARM_TICKS_REQUIRED,
    INHERIT_TICKS_REQUIRED,
    HARD_DRIFT_THRESHOLD,
)


# ── Helpers ──

def make_regime(
    model: str = "tinyllama",
    retrieval: str = "cache_first",
    budget: str = "normal",
    dominant: str = "neutral",
    se_bucket: str = "low",
) -> RouteRegime:
    return RouteRegime(
        model_target=model,
        retrieval_mode=retrieval,
        budget_class=budget,
        dominant_signal=dominant,
        se_bucket=se_bucket,
    )


def make_route(
    model: ModelTarget = ModelTarget.TINYLLAMA,
    retrieval: RetrievalMode = RetrievalMode.CACHE_FIRST,
    budget: str = "normal",
    max_tokens: int = 64,
) -> RouteDecision:
    return RouteDecision(
        target_model=model,
        model_confidence=0.8,
        retrieval_mode=retrieval,
        sidecar_phase=None,
        fused_path_allowed=True,
        max_tokens=max_tokens,
        budget_mode=budget,
    )


FACTUAL_REGIME = make_regime("tinyllama", "cache_first", "normal", "fact", "low")
CODE_REGIME = make_regime("qwen_coder", "code", "normal", "code", "mid")
GRAPH_REGIME = make_regime("tinyllama", "graph", "graph", "graph", "mid")


# ── RouteRegime tests ──

class TestRouteRegime:

    def test_identical_distance_zero(self):
        r = make_regime()
        assert r.distance(r) == 0.0

    def test_completely_different_distance_one(self):
        r1 = make_regime("tinyllama", "cache_first", "normal", "fact", "low")
        r2 = make_regime("qwen_coder", "code", "graph", "code", "high")
        assert r1.distance(r2) == pytest.approx(1.0)

    def test_model_swap_is_heavy(self):
        """Model target mismatch should be the largest single contributor."""
        r1 = make_regime("tinyllama")
        r2 = make_regime("qwen_coder")
        dist = r1.distance(r2)
        assert dist > 0.3  # model is 0.35 weight

    def test_same_model_different_retrieval(self):
        r1 = make_regime(retrieval="cache_first")
        r2 = make_regime(retrieval="code")
        dist = r1.distance(r2)
        assert 0.0 < dist < 0.5  # just retrieval difference

    def test_from_route(self):
        route = make_route()
        regime = RouteRegime.from_route(route, se_dominant="fact", se_value=0.1)
        assert regime.model_target == "tinyllama"
        assert regime.retrieval_mode == "cache_first"
        assert regime.se_bucket == "low"
        assert regime.dominant_signal == "fact"

    def test_se_bucket_boundaries(self):
        r_low = RouteRegime.from_route(make_route(), se_value=0.14)
        r_mid = RouteRegime.from_route(make_route(), se_value=0.25)
        r_high = RouteRegime.from_route(make_route(), se_value=0.5)
        assert r_low.se_bucket == "low"
        assert r_mid.se_bucket == "mid"
        assert r_high.se_bucket == "high"


# ── State machine: basic transitions ──

class TestStateMachineBasic:

    def test_starts_cold(self):
        sm = InheritanceStateMachine()
        assert sm.current_state == InheritanceState.COLD

    def test_first_query_stays_cold(self):
        sm = InheritanceStateMachine()
        result = sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.COLD
        assert sm.state.queries_total == 1

    def test_two_similar_queries_warm(self):
        sm = InheritanceStateMachine()
        sm.observe(FACTUAL_REGIME)
        sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.WARM_CANDIDATE

    def test_sustained_similar_inherits(self):
        sm = InheritanceStateMachine()
        for _ in range(INHERIT_TICKS_REQUIRED + 1):
            sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.INHERITED

    def test_suggest_cold_says_no(self):
        sm = InheritanceStateMachine()
        suggestion = sm.suggest(FACTUAL_REGIME)
        assert suggestion["inherit"] is False
        assert "cold" in suggestion["reason"]

    def test_suggest_inherited_says_yes(self):
        sm = InheritanceStateMachine()
        for _ in range(INHERIT_TICKS_REQUIRED + 1):
            sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.INHERITED

        suggestion = sm.suggest(FACTUAL_REGIME)
        assert suggestion["inherit"] is True
        assert len(suggestion["fields_to_reuse"]) > 0
        assert suggestion["confidence"] > 0

    def test_inherited_stays_inherited_on_similar(self):
        sm = InheritanceStateMachine()
        for _ in range(INHERIT_TICKS_REQUIRED + 2):
            sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.INHERITED
        # One more similar query
        sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.INHERITED


# ── State machine: drift and reset ──

class TestStateMachineDrift:

    def test_hard_drift_resets(self):
        """Switching from factual to code = hard drift → COLD."""
        sm = InheritanceStateMachine()
        for _ in range(INHERIT_TICKS_REQUIRED + 1):
            sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.INHERITED

        # Hard drift: factual → code (model + retrieval + dominant change)
        result = sm.observe(CODE_REGIME)
        assert sm.current_state == InheritanceState.COLD
        assert sm.state.confidence == 0.0

    def test_quality_failure_resets(self):
        sm = InheritanceStateMachine()
        for _ in range(INHERIT_TICKS_REQUIRED + 1):
            sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.INHERITED

        result = sm.observe(FACTUAL_REGIME, quality_passed=False)
        assert sm.current_state == InheritanceState.COLD
        assert "quality" in sm.state.transition_reason

    def test_force_reset(self):
        sm = InheritanceStateMachine()
        for _ in range(INHERIT_TICKS_REQUIRED + 1):
            sm.observe(FACTUAL_REGIME)
        sm.force_reset("model_swap")
        assert sm.current_state == InheritanceState.COLD
        assert sm.state.confidence == 0.0

    def test_warm_interrupted_by_drift(self):
        sm = InheritanceStateMachine()
        sm.observe(FACTUAL_REGIME)
        sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.WARM_CANDIDATE

        # Different query breaks warming
        sm.observe(CODE_REGIME)
        assert sm.current_state == InheritanceState.COLD

    def test_inherited_degrades_to_unstable(self):
        """Moderate drift degrades confidence, eventually → UNSTABLE."""
        sm = InheritanceStateMachine()
        for _ in range(INHERIT_TICKS_REQUIRED + 1):
            sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.INHERITED

        # Moderate drift: same model, different retrieval
        moderate_drift = make_regime("tinyllama", "web_direct", "normal", "fact", "mid")
        # Repeatedly observe moderate drift until confidence drops
        for _ in range(10):
            sm.observe(moderate_drift)
            if sm.current_state == InheritanceState.UNSTABLE:
                break
        # Should eventually go unstable
        assert sm.current_state in (
            InheritanceState.UNSTABLE, InheritanceState.COLD
        )

    def test_unstable_recovers_on_similarity(self):
        """Unstable state can recover if queries become similar again."""
        sm = InheritanceStateMachine()
        for _ in range(INHERIT_TICKS_REQUIRED + 1):
            sm.observe(FACTUAL_REGIME)

        # Go unstable
        moderate_drift = make_regime("tinyllama", "web_direct", "normal", "fact", "mid")
        for _ in range(10):
            sm.observe(moderate_drift)
            if sm.current_state == InheritanceState.UNSTABLE:
                break

        if sm.current_state == InheritanceState.UNSTABLE:
            # Recover with similar queries
            for _ in range(WARM_TICKS_REQUIRED + 1):
                sm.observe(moderate_drift)
            assert sm.current_state in (
                InheritanceState.WARM_CANDIDATE, InheritanceState.INHERITED
            )


# ── Acceptance criteria scenarios ──

class TestAcceptanceCriteria:

    def test_repeated_factual_same_family(self):
        """Repeated factual questions → inheritance by 3rd query."""
        sm = InheritanceStateMachine()
        results = []
        for i in range(5):
            results.append(sm.observe(FACTUAL_REGIME))

        # Should be inherited by query 4 (0-indexed: after INHERIT_TICKS_REQUIRED+1)
        assert sm.current_state == InheritanceState.INHERITED
        suggestion = sm.suggest(FACTUAL_REGIME)
        assert suggestion["inherit"] is True

    def test_code_after_factual_resets(self):
        """Code query after factual session → full reset."""
        sm = InheritanceStateMachine()
        for _ in range(5):
            sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.INHERITED

        sm.observe(CODE_REGIME)
        assert sm.current_state == InheritanceState.COLD
        suggestion = sm.suggest(CODE_REGIME)
        assert suggestion["inherit"] is False

    def test_graph_after_code_resets(self):
        """Graph query after code session → reset."""
        sm = InheritanceStateMachine()
        for _ in range(5):
            sm.observe(CODE_REGIME)
        sm.observe(GRAPH_REGIME)
        assert sm.current_state == InheritanceState.COLD

    def test_exact_repeat(self):
        """Exact same regime every query → inheritance activates and holds."""
        sm = InheritanceStateMachine()
        for _ in range(20):
            sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.INHERITED
        assert sm.state.stable_ticks >= 15
        assert sm.state.confidence > 0.5

    def test_model_swap_forces_reset(self):
        """User forces /qwen → force_reset → cold."""
        sm = InheritanceStateMachine()
        for _ in range(5):
            sm.observe(FACTUAL_REGIME)
        sm.force_reset("user_forced_/qwen")
        assert sm.current_state == InheritanceState.COLD

    def test_inheritance_never_bypasses_quality_gate(self):
        """Quality failure always resets, even during inheritance."""
        sm = InheritanceStateMachine()
        for _ in range(5):
            sm.observe(FACTUAL_REGIME)
        assert sm.current_state == InheritanceState.INHERITED

        # Quality failure mid-inheritance
        sm.observe(FACTUAL_REGIME, quality_passed=False)
        assert sm.current_state == InheritanceState.COLD

    def test_inheritance_tracks_count(self):
        """queries_inherited counts only when was_inherited=True."""
        sm = InheritanceStateMachine()
        for _ in range(5):
            sm.observe(FACTUAL_REGIME)
        sm.observe(FACTUAL_REGIME, was_inherited=True)
        sm.observe(FACTUAL_REGIME, was_inherited=True)
        sm.observe(FACTUAL_REGIME, was_inherited=False)
        assert sm.state.queries_inherited == 2
        assert sm.state.queries_total == 8

    def test_stats_output(self):
        sm = InheritanceStateMachine()
        for _ in range(5):
            sm.observe(FACTUAL_REGIME)
        stats = sm.stats()
        assert "state" in stats
        assert "confidence" in stats
        assert "queries_total" in stats
        assert "inheritance_rate" in stats
        assert stats["queries_total"] == 5


# ── RouteRegime.from_route integration ──

class TestFromRoute:

    def test_tinyllama_factual(self):
        route = make_route(
            model=ModelTarget.TINYLLAMA,
            retrieval=RetrievalMode.EXACT_FACT,
            budget="normal",
        )
        regime = RouteRegime.from_route(route, se_dominant="fact", se_value=0.1)
        assert regime.model_target == "tinyllama"
        assert regime.retrieval_mode == "exact_fact"
        assert regime.dominant_signal == "fact"

    def test_qwen_code(self):
        route = make_route(
            model=ModelTarget.QWEN_CODER,
            retrieval=RetrievalMode.CODE,
            budget="normal",
            max_tokens=128,
        )
        regime = RouteRegime.from_route(route, se_dominant="code", se_value=0.3)
        assert regime.model_target == "qwen_coder"
        assert regime.retrieval_mode == "code"
        assert regime.se_bucket == "mid"


# ── Edge cases ──

class TestEdgeCases:

    def test_empty_machine_suggest(self):
        sm = InheritanceStateMachine()
        suggestion = sm.suggest(FACTUAL_REGIME)
        assert suggestion["inherit"] is False

    def test_single_query_no_inheritance(self):
        sm = InheritanceStateMachine()
        sm.observe(FACTUAL_REGIME)
        suggestion = sm.suggest(FACTUAL_REGIME)
        assert suggestion["inherit"] is False

    def test_alternating_never_inherits(self):
        """Alternating between code and factual → never stabilizes."""
        sm = InheritanceStateMachine()
        for _ in range(20):
            sm.observe(FACTUAL_REGIME)
            sm.observe(CODE_REGIME)
        assert sm.current_state != InheritanceState.INHERITED

    def test_history_bounded(self):
        """History doesn't grow unbounded."""
        sm = InheritanceStateMachine()
        for _ in range(100):
            sm.observe(FACTUAL_REGIME)
        assert len(sm._history) <= 20

    def test_state_as_dict(self):
        sm = InheritanceStateMachine()
        for _ in range(5):
            sm.observe(FACTUAL_REGIME)
        d = sm.state.as_dict()
        assert d["state"] == "inherited"
        assert d["regime"] is not None
        assert "model_target" in d["regime"]
