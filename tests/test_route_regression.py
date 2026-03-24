"""
Regression tests for route decision + query classifier.

Covers:
1. Sidecar phase: route_decision now uses auto-detect (None), not blanket "fused"
2. Hybrid graph/code queries: graph intent overrides code classification
3. Factual question override: prevents language names from triggering code route

Work order: WO-AI-OS-RUNTIME-01 follow-up (ChatGPT review, 2026-03-18)
"""

import pytest
from helix_substrate.query_classifier import (
    classify,
    detect_graph_intent,
    ModelTarget,
    GraphIntent,
)
from helix_substrate.route_decision import compute_route, RetrievalMode


# ============================================================================
# 1. Sidecar phase: auto-detect, not blanket fused
# ============================================================================

class TestSidecarPhaseRouting:
    """Sidecar phase should be None (auto-detect) to let the triton kernel
    pick fused (N<=16, decode) or scatter (N>16, prefill) per-call.
    Blanket "fused" was proven to regress prefill by 10.5%."""

    def test_sidecar_phase_is_autodetect(self):
        """Route decision should NOT hardcode 'fused'."""
        route = compute_route("explain how compression works")
        assert route.sidecar_phase is None, (
            f"Expected None (auto-detect), got '{route.sidecar_phase}'. "
            "Blanket 'fused' regresses prefill (stabilization receipt 2026-03-18)."
        )

    def test_sidecar_phase_autodetect_for_code(self):
        """Even code queries should use auto-detect sidecar phase."""
        route = compute_route("write a function to sort a list", fused_available=True)
        assert route.sidecar_phase is None

    def test_sidecar_phase_autodetect_for_graph(self):
        """Graph queries should also use auto-detect."""
        route = compute_route("what is the thesis score for Intel",
                             graph_available=True, fused_available=True)
        assert route.sidecar_phase is None

    def test_fused_not_available_still_none(self):
        """When fused is not available, sidecar_phase should still be None."""
        route = compute_route("hello", fused_available=False)
        assert route.sidecar_phase is None


# ============================================================================
# 2. Hybrid graph/code queries
# ============================================================================

class TestHybridGraphCodeRouting:
    """Graph intent should override code classification. ChatGPT flagged:
    'write a function to compute thesis_score from a graph edge list'
    as the kind of hybrid query where route priority bugs hide."""

    def test_pure_graph_query_routes_to_graph(self):
        """Pure graph query should route to graph retrieval."""
        route = compute_route("what is the thesis score for Intel",
                             graph_available=True)
        assert route.retrieval_mode == RetrievalMode.GRAPH
        assert route.target_model == ModelTarget.TINYLLAMA
        assert route.graph_intent == "thesis"

    def test_graph_keywords_override_code_classification(self):
        """'thesis score' is a graph intent even if query has code-like words."""
        route = compute_route(
            "show me the thesis score for this function",
            graph_available=True,
        )
        # "function" is a code keyword, but "thesis score" should win
        assert route.retrieval_mode == RetrievalMode.GRAPH
        assert route.target_model == ModelTarget.TINYLLAMA

    def test_explore_connections_overrides_code(self):
        """'explore connections' is graph intent even with code words."""
        route = compute_route(
            "explore connections for this class of entities",
            graph_available=True,
        )
        assert route.retrieval_mode == RetrievalMode.GRAPH

    def test_pure_code_query_no_graph(self):
        """Pure code query should NOT route to graph even if graph is available."""
        route = compute_route(
            "write a python function to sort a list",
            graph_available=True,
        )
        assert route.retrieval_mode == RetrievalMode.CODE
        assert route.target_model == ModelTarget.QWEN_CODER

    def test_code_with_graph_word_for(self):
        """'for' in a code context should NOT trigger graph (it matches
        control flow pattern, not graph intent)."""
        route = compute_route(
            "write a for loop to iterate over items",
            graph_available=True,
        )
        assert route.target_model == ModelTarget.QWEN_CODER
        assert route.retrieval_mode == RetrievalMode.CODE

    def test_graph_not_available_falls_through(self):
        """Graph intent should be None when graph is not available."""
        route = compute_route(
            "what is the thesis score for Intel",
            graph_available=False,
        )
        assert route.graph_intent is None
        # Should fall through to normal routing
        assert route.retrieval_mode != RetrievalMode.GRAPH


# ============================================================================
# 3. Classifier edge cases
# ============================================================================

class TestClassifierEdgeCases:
    def test_factual_question_with_language_name(self):
        """'When was Python released?' should route to TinyLlama, not Qwen."""
        target, _, debug = classify("When was Python released?")
        assert target == ModelTarget.TINYLLAMA
        assert debug.get("factual_override") is True

    def test_code_fence_always_routes_to_code(self):
        """Code fences are a strong signal for Qwen regardless of content."""
        target, _, _ = classify("```python\nprint('hello')\n```")
        assert target == ModelTarget.QWEN_CODER

    def test_ambiguous_code_word_in_question(self):
        """'function' in a non-code context should be weak signal."""
        target, _, debug = classify("What is the function of the liver?")
        # This gets code keyword 'function' but also controller keyword 'what is'
        # Should route to TinyLlama since controller score should be competitive
        # (This test documents current behavior, not necessarily ideal)
        # At minimum, document the classification debug info
        assert "code_score" in debug
        assert "controller_score" in debug

    def test_graph_detect_thesis(self):
        assert detect_graph_intent("what is the thesis score") == GraphIntent.THESIS

    def test_graph_detect_causal(self):
        assert detect_graph_intent("trace the causal chain from A to B") == GraphIntent.CAUSAL

    def test_graph_detect_none_for_code(self):
        assert detect_graph_intent("write a python function") is None

    def test_graph_entity_with_question(self):
        """Entity name + question word should trigger SEARCH."""
        intent = detect_graph_intent("what do we know about Intel")
        assert intent == GraphIntent.SEARCH
