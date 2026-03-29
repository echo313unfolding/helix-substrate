"""Tests for preflight_router.py — Se-based preflight route prediction."""

from helix_substrate.preflight_router import (
    compute_query_se, SeSignal, PreflightRouter, PreflightGuess,
)


class TestComputeQuerySe:
    def test_empty_input(self):
        sig = compute_query_se("")
        assert sig.se == 0
        assert sig.h == 0

    def test_code_query(self):
        sig = compute_query_se("write a python function to sort a list")
        assert sig.c_code > 0
        assert sig.dominant_signal() == "code"

    def test_code_fence_strong(self):
        sig = compute_query_se("```python\ndef foo(): pass\n```")
        assert sig.c_code > 0.5

    def test_factual_query(self):
        sig = compute_query_se("when was Python released")
        assert sig.factual_override is True
        assert sig.c_code == 0.0  # factual override suppresses code

    def test_exact_fact_pattern(self):
        sig = compute_query_se("how many planets are in the solar system")
        assert sig.c_fact > 0

    def test_creative_query(self):
        sig = compute_query_se("imagine a dream about infinite beauty")
        assert sig.c_creative > 0
        assert sig.dominant_signal() == "creative"

    def test_drift_first_query(self):
        sig = compute_query_se("hello world")
        assert sig.d == 1.0  # first query = max novelty

    def test_drift_repeated(self):
        import hashlib
        text = "hello world"
        h = hashlib.sha256(text.encode()).hexdigest()[:12]
        sig = compute_query_se("hello world", drift_history=[h, h, h])
        assert sig.d < 1.0  # repeated -> lower drift

    def test_se_is_nonnegative(self):
        sig = compute_query_se("any random query text here")
        assert sig.se >= 0

    def test_as_dict(self):
        sig = compute_query_se("hello")
        d = sig.as_dict()
        assert "se" in d
        assert "dominant" in d
        assert "factual_override" in d


class TestPreflightRouter:
    def test_empty_input(self):
        router = PreflightRouter()
        g = router.guess("")
        assert g.se_model_guess == "tinyllama"
        assert g.se_confidence == 0.0

    def test_code_routes_to_qwen(self):
        router = PreflightRouter()
        g = router.guess("write a python function to sort a list")
        assert g.se_model_guess == "qwen_coder"
        assert g.se_retrieval_guess == "code"
        assert g.se_budget_guess == "code_128"

    def test_factual_routes_to_tinyllama(self):
        router = PreflightRouter()
        g = router.guess("when was the Eiffel Tower built")
        assert g.se_model_guess == "tinyllama"
        assert g.se_retrieval_guess == "exact_fact"

    def test_generic_routes_to_tinyllama(self):
        router = PreflightRouter()
        g = router.guess("hello there")
        assert g.se_model_guess == "tinyllama"
        assert g.se_budget_guess == "factual_64"

    def test_confidence_increases_with_length(self):
        router = PreflightRouter()
        g_short = router.guess("hi")
        g_long = router.guess("write a comprehensive python function to sort a list of integers")
        assert g_long.se_confidence >= g_short.se_confidence

    def test_record_query(self):
        router = PreflightRouter()
        router.record_query("test query")
        assert len(router._drift_history) == 1

    def test_drift_history_bounded(self):
        router = PreflightRouter()
        for i in range(60):
            router.record_query(f"query {i}")
        assert len(router._drift_history) <= 50

    def test_should_update_debounce(self):
        router = PreflightRouter(debounce_ms=1000)
        assert router.should_update() is True  # no previous guess
        router.guess("test")
        assert router.should_update() is False  # just guessed

    def test_kv_reuse_impossible_on_model_swap(self):
        router = PreflightRouter()
        g = router.guess("write python code", current_model="tinyllama")
        # Code query → qwen_coder predicted, but current is tinyllama
        if g.se_model_guess != "tinyllama":
            assert g.se_kv_guess == "impossible"

    def test_guess_as_dict(self):
        router = PreflightRouter()
        g = router.guess("test query")
        d = g.as_dict()
        assert "se_model_guess" in d
        assert "se" in d  # nested Se signal

    def test_compatibility_aliases(self):
        router = PreflightRouter()
        g = router.guess("test")
        assert g.model_target == g.se_model_guess
        assert g.retrieval_mode == g.se_retrieval_guess
        assert g.confidence == g.se_confidence
