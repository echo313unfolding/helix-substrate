"""
Tests for TokenAccountant (WO-E).

Tests cover:
1. Heuristic fallback when no tokenizer available
2. Real tokenizer counting (requires model dirs)
3. Per-step accounting correctness
4. Per-route accounting correctness
5. Edge cases (empty strings, non-model lobes)
6. Graceful degradation
"""
import pytest
from unittest.mock import MagicMock, patch

from helix_substrate.token_accountant import (
    TokenAccountant,
    TokenAccount,
    RouteTokenAccount,
    SCHEMA,
    _ZERO_ACCOUNT,
)
from helix_substrate.query_classifier import ModelTarget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Deterministic tokenizer for testing."""

    def encode(self, text, add_special_tokens=False):
        if not text:
            return []
        # Simple: split on whitespace, each word = 1 token
        return list(range(len(text.split())))


class FakeModelManager:
    def _get_tokenizer(self, target):
        return FakeTokenizer()


@pytest.fixture
def accountant():
    return TokenAccountant(model_manager=FakeModelManager())


@pytest.fixture
def accountant_no_tokenizer():
    """Accountant with no model manager and no model dirs."""
    a = TokenAccountant(model_manager=None)
    # Patch out the direct-load fallback
    a._get_tokenizer = lambda target: None
    return a


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string(self, accountant):
        count, src = accountant.count_tokens("", ModelTarget.TINYLLAMA)
        assert count == 0
        assert src == "tokenizer"

    def test_real_count(self, accountant):
        count, src = accountant.count_tokens("hello world foo", ModelTarget.TINYLLAMA)
        assert count == 3  # FakeTokenizer: 3 words
        assert src == "tokenizer"

    def test_heuristic_fallback(self, accountant_no_tokenizer):
        # 20 chars => 20 // 4 = 5
        count, src = accountant_no_tokenizer.count_tokens("a" * 20, ModelTarget.TINYLLAMA)
        assert count == 5
        assert src == "heuristic"
        assert accountant_no_tokenizer._fallback_count == 1

    def test_heuristic_minimum_one(self, accountant_no_tokenizer):
        # 3 chars => 3 // 4 = 0 => clamp to 1
        count, src = accountant_no_tokenizer.count_tokens("abc", ModelTarget.TINYLLAMA)
        assert count == 1
        assert src == "heuristic"


# ---------------------------------------------------------------------------
# account_step
# ---------------------------------------------------------------------------

class TestAccountStep:
    def test_nonmodel_lobe(self, accountant):
        """Non-model lobes (parser, compiler, memory) return zero account."""
        account = accountant.account_step("parser", "hello world")
        assert account == _ZERO_ACCOUNT
        assert account.source == "none"

    def test_planner_step(self, accountant):
        account = accountant.account_step("planner", "write a hello world")
        assert account.source == "tokenizer"
        assert account.query_tokens > 0
        assert account.prompt_tokens > 0
        assert account.expected_output_tokens == 256  # planner default
        assert account.total_input_tokens > 0
        assert account.total_budget_tokens == account.total_input_tokens + 256
        assert account.memory_tokens == 0
        assert account.context_tokens == 0

    def test_coder_step(self, accountant):
        account = accountant.account_step("coder", "write a hello world")
        assert account.expected_output_tokens == 256  # coder default
        assert account.total_budget_tokens > 256

    def test_with_memory(self, accountant):
        a1 = accountant.account_step("planner", "hello")
        a2 = accountant.account_step("planner", "hello", memory_context="some memory context here")
        assert a2.memory_tokens > 0
        assert a2.total_input_tokens > a1.total_input_tokens

    def test_with_prev_step(self, accountant):
        a1 = accountant.account_step("verifier", "hello")
        a2 = accountant.account_step("verifier", "hello", prev_step_output="previous output text")
        assert a2.context_tokens > 0
        assert a2.total_input_tokens > a1.total_input_tokens

    def test_max_tokens_override(self, accountant):
        a1 = accountant.account_step("planner", "hello")
        a2 = accountant.account_step("planner", "hello", max_tokens_override=64)
        assert a1.expected_output_tokens == 256
        assert a2.expected_output_tokens == 64
        assert a2.total_budget_tokens < a1.total_budget_tokens

    def test_receipt_serializable(self, accountant):
        account = accountant.account_step("planner", "test query")
        receipt = account.to_receipt()
        assert isinstance(receipt, dict)
        assert "query_tokens" in receipt
        assert "total_budget_tokens" in receipt
        assert "source" in receipt


# ---------------------------------------------------------------------------
# account_route
# ---------------------------------------------------------------------------

class TestAccountRoute:
    def test_direct_plan(self, accountant):
        from helix_substrate.lobe_scheduler import ROUTE_DIRECT_PLAN
        ra = accountant.account_route(ROUTE_DIRECT_PLAN, "simple question")
        assert len(ra.steps) == 1
        assert ra.peak_step_tokens > 0
        assert ra.total_output_tokens == 256

    def test_plan_code_verify(self, accountant):
        from helix_substrate.lobe_scheduler import ROUTE_PLAN_CODE_VERIFY
        ra = accountant.account_route(ROUTE_PLAN_CODE_VERIFY, "complex task")
        assert len(ra.steps) == 3
        # Peak should be the step with most context (likely coder or verifier)
        assert ra.peak_step_tokens >= ra.steps[0].total_budget_tokens
        assert ra.total_output_tokens == 256 + 256 + 128  # planner + coder + verifier

    def test_receipt_serializable(self, accountant):
        from helix_substrate.lobe_scheduler import ROUTE_CODE_VERIFY
        ra = accountant.account_route(ROUTE_CODE_VERIFY, "fix the bug")
        receipt = ra.to_receipt()
        assert isinstance(receipt, dict)
        assert "steps" in receipt
        assert "peak_step_tokens" in receipt
        assert len(receipt["steps"]) == 2


# ---------------------------------------------------------------------------
# estimate_context
# ---------------------------------------------------------------------------

class TestEstimateContext:
    def test_replaces_heuristic(self, accountant):
        """estimate_context returns real token count, not len//4."""
        query = "write a function that sorts a list"
        result = accountant.estimate_context(query, ModelTarget.QWEN_CODER, "direct_code")
        # Should include prompt tokens + query tokens + output budget
        # With FakeTokenizer, prompt+query word count + 256
        assert result > 256  # At minimum, output budget
        assert isinstance(result, int)

    def test_with_account(self, accountant):
        tokens, account = accountant.estimate_context_with_account(
            "test query", ModelTarget.TINYLLAMA, "direct_plan",
        )
        assert tokens == account.total_budget_tokens
        assert account.source == "tokenizer"


# ---------------------------------------------------------------------------
# lobe_for_route
# ---------------------------------------------------------------------------

class TestLobeForRoute:
    def test_known_routes(self):
        assert TokenAccountant._lobe_for_route("direct_code", ModelTarget.QWEN_CODER) == "coder"
        assert TokenAccountant._lobe_for_route("direct_plan", ModelTarget.TINYLLAMA) == "planner"
        assert TokenAccountant._lobe_for_route("code_verify", ModelTarget.QWEN_CODER) == "coder"
        assert TokenAccountant._lobe_for_route("plan_code_verify", ModelTarget.QWEN_CODER) == "coder"

    def test_unknown_route_qwen(self):
        assert TokenAccountant._lobe_for_route("unknown_route", ModelTarget.QWEN_CODER) == "coder"

    def test_unknown_route_tinyllama(self):
        assert TokenAccountant._lobe_for_route("unknown_route", ModelTarget.TINYLLAMA) == "planner"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status(self, accountant):
        # Warm up tokenizer
        accountant.count_tokens("hello", ModelTarget.TINYLLAMA)
        s = accountant.status()
        assert s["schema"] == SCHEMA
        assert s["fallback_count"] == 0
        assert len(s["loaded_tokenizers"]) >= 1

    def test_status_with_fallbacks(self, accountant_no_tokenizer):
        accountant_no_tokenizer.count_tokens("hello world", ModelTarget.TINYLLAMA)
        s = accountant_no_tokenizer.status()
        assert s["fallback_count"] == 1
