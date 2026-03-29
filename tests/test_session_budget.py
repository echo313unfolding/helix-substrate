"""Tests for session_budget.py — session-level token budget governor."""

from helix_substrate.session_budget import SessionBudget, BudgetVerdict


class TestBudgetBasics:
    def test_initial_state(self):
        b = SessionBudget(max_tokens=1000)
        assert b.total_tokens == 0
        assert b.remaining == 1000
        assert b.usage_pct == 0.0

    def test_record_query(self):
        b = SessionBudget(max_tokens=1000)
        b.record_query(prompt_tokens=50, generated_tokens=30)
        assert b.total_tokens == 80
        assert b.remaining == 920
        assert b.queries_served == 1

    def test_multiple_queries(self):
        b = SessionBudget(max_tokens=1000)
        b.record_query(prompt_tokens=100, generated_tokens=50)
        b.record_query(prompt_tokens=200, generated_tokens=100)
        assert b.total_tokens == 450
        assert b.queries_served == 2


class TestBudgetVerdicts:
    def test_normal(self):
        b = SessionBudget(max_tokens=1000)
        assert b.check() == BudgetVerdict.NORMAL

    def test_capped(self):
        b = SessionBudget(max_tokens=100, cap_pct=0.90)
        b.record_query(prompt_tokens=91, generated_tokens=0)
        assert b.check() == BudgetVerdict.CAPPED

    def test_denied(self):
        b = SessionBudget(max_tokens=100)
        b.record_query(prompt_tokens=100, generated_tokens=0)
        assert b.check() == BudgetVerdict.DENIED

    def test_capped_max_tokens_full_budget(self):
        b = SessionBudget(max_tokens=1000)
        assert b.capped_max_tokens(128) == 128

    def test_capped_max_tokens_tight_budget(self):
        b = SessionBudget(max_tokens=100)
        b.record_query(prompt_tokens=80, generated_tokens=0)
        # 20 remaining, requesting 128
        assert b.capped_max_tokens(128) == 20

    def test_capped_max_tokens_minimum(self):
        b = SessionBudget(max_tokens=100)
        b.record_query(prompt_tokens=95, generated_tokens=0)
        # Only 5 remaining, but minimum is 16
        assert b.capped_max_tokens(128) == 16


class TestBudgetWarning:
    def test_no_warning_early(self):
        b = SessionBudget(max_tokens=1000, warn_pct=0.80)
        b.record_query(prompt_tokens=100, generated_tokens=0)
        assert not b.should_warn()

    def test_warning_at_threshold(self):
        b = SessionBudget(max_tokens=1000, warn_pct=0.80)
        b.record_query(prompt_tokens=800, generated_tokens=0)
        assert b.should_warn()


class TestBudgetReset:
    def test_reset(self):
        b = SessionBudget(max_tokens=1000)
        b.record_query(prompt_tokens=500, generated_tokens=200)
        b.reset()
        assert b.total_tokens == 0
        assert b.remaining == 1000
        assert b.queries_served == 0


class TestBudgetReceipt:
    def test_as_dict(self):
        b = SessionBudget(max_tokens=1000)
        b.record_query(prompt_tokens=100, generated_tokens=50, was_capped=True)
        d = b.as_dict()
        assert d["max_tokens"] == 1000
        assert d["total_tokens"] == 150
        assert d["remaining"] == 850
        assert d["queries_capped"] == 1
        assert d["usage_pct"] == 15.0
