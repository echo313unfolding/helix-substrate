"""
Tests for ShapingPolicy (WO-H).

Tests cover:
1. Default policy is fully permissive
2. Risky queries lock verifier
3. Symbolic routes lock all drops
4. Output cap checks respect min_output_tokens
5. Downgrade checks respect verifier/planner locks
6. Policy receipt is serializable
7. Integration with route shaper — risky query blocks verifier drop
"""
import pytest

from helix_substrate.shaping_policy import (
    ShapingPolicy, is_risky_query, SCHEMA,
    _RISKY_KEYWORDS, _HAS_VERIFIER, _HAS_PLANNER, _SYMBOLIC_ROUTES,
)


class TestDefaultPolicy:
    def test_all_allowed(self):
        p = ShapingPolicy()
        assert p.allow_output_cap
        assert p.allow_memory_trim
        assert p.allow_verifier_drop
        assert p.allow_planner_drop
        assert p.prefer_swap

    def test_constraints_summary_permissive(self):
        p = ShapingPolicy()
        assert p.constraints_summary == "prefer_swap"


class TestForQuery:
    def test_normal_query(self):
        p = ShapingPolicy.for_query("write a hello world function", "direct_code")
        assert p.allow_verifier_drop
        assert p.allow_planner_drop

    def test_risky_query_delete(self):
        p = ShapingPolicy.for_query("delete all user data from production", "code_verify")
        assert not p.allow_verifier_drop
        assert p.allow_planner_drop  # planner can still be dropped

    def test_risky_query_deploy(self):
        p = ShapingPolicy.for_query("deploy this to production", "plan_verify")
        assert not p.allow_verifier_drop

    def test_risky_query_destroy(self):
        p = ShapingPolicy.for_query("destroy the old database", "code_verify")
        assert not p.allow_verifier_drop

    def test_risky_query_rm(self):
        p = ShapingPolicy.for_query("rm -rf /tmp/build", "direct_code")
        assert not p.allow_verifier_drop

    def test_symbolic_route(self):
        p = ShapingPolicy.for_query("bloom crystal", "symbolic_parse_verify", is_symbolic=True)
        assert not p.allow_verifier_drop
        assert not p.allow_planner_drop

    def test_symbolic_flag_overrides(self):
        p = ShapingPolicy.for_query("hello world", "direct_plan", is_symbolic=True)
        assert not p.allow_verifier_drop
        assert not p.allow_planner_drop

    def test_constraints_summary_risky(self):
        p = ShapingPolicy.for_query("delete everything", "code_verify")
        assert "verifier_locked" in p.constraints_summary


class TestCheckOutputCap:
    def test_allowed(self):
        p = ShapingPolicy()
        ok, reason = p.check_output_cap(128)
        assert ok

    def test_below_minimum(self):
        p = ShapingPolicy(min_output_tokens=64)
        ok, reason = p.check_output_cap(32)
        assert not ok
        assert "min_output_tokens" in reason

    def test_disabled(self):
        p = ShapingPolicy(allow_output_cap=False)
        ok, reason = p.check_output_cap(128)
        assert not ok
        assert "disabled by policy" in reason


class TestCheckMemoryTrim:
    def test_allowed(self):
        p = ShapingPolicy()
        ok, _ = p.check_memory_trim()
        assert ok

    def test_disabled(self):
        p = ShapingPolicy(allow_memory_trim=False)
        ok, reason = p.check_memory_trim()
        assert not ok
        assert "disabled" in reason


class TestCheckDowngrade:
    def test_allowed_normal(self):
        p = ShapingPolicy()
        ok, _ = p.check_downgrade("plan_code_verify", "code_verify")
        assert ok

    def test_verifier_locked(self):
        p = ShapingPolicy(allow_verifier_drop=False)
        # code_verify → direct_code loses verifier
        ok, reason = p.check_downgrade("code_verify", "direct_code")
        assert not ok
        assert "verifier" in reason

    def test_verifier_preserved(self):
        p = ShapingPolicy(allow_verifier_drop=False)
        # plan_code_verify → code_verify keeps verifier
        ok, _ = p.check_downgrade("plan_code_verify", "code_verify")
        assert ok

    def test_planner_locked(self):
        p = ShapingPolicy(allow_planner_drop=False)
        # plan_code_verify → code_verify loses planner
        ok, reason = p.check_downgrade("plan_code_verify", "code_verify")
        assert not ok
        assert "planner" in reason

    def test_both_locked(self):
        p = ShapingPolicy(allow_verifier_drop=False, allow_planner_drop=False)
        # plan_code_verify → code_verify: loses planner
        ok, reason = p.check_downgrade("plan_code_verify", "code_verify")
        assert not ok
        assert "planner" in reason


class TestIsRiskyQuery:
    def test_safe(self):
        assert not is_risky_query("write a hello world function")

    def test_delete(self):
        assert is_risky_query("delete all files")

    def test_case_insensitive(self):
        assert is_risky_query("DELETE the database")

    def test_production(self):
        assert is_risky_query("push to production")


class TestReceipt:
    def test_receipt_has_schema(self):
        p = ShapingPolicy()
        r = p.to_receipt()
        assert r["schema"] == SCHEMA
        assert "allow_verifier_drop" in r
        assert "min_output_tokens" in r


class TestIntegrationWithShaper:
    """Integration: risky query + route shaper → verifier cannot be dropped."""

    def test_risky_query_blocks_verifier_drop(self):
        from helix_substrate.route_shaper import shape_route
        from helix_substrate.token_accountant import TokenAccountant
        from helix_substrate.budget_gate import BudgetGate
        from helix_substrate.lobe_scheduler import ROUTE_CODE_VERIFY

        class FakeTok:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text.split()))) if text else []

        class FakeMM:
            def _get_tokenizer(self, target):
                return FakeTok()

        accountant = TokenAccountant(model_manager=FakeMM())
        gate = BudgetGate()

        # Risky query: should lock verifier
        policy = ShapingPolicy.for_query(
            "delete all user records from production db",
            "code_verify",
        )
        assert not policy.allow_verifier_drop

        result = shape_route(
            route=ROUTE_CODE_VERIFY,
            route_name="code_verify",
            query="delete all user records from production db",
            memory_context="",
            max_tokens=None,
            accountant=accountant,
            gate=gate,
            resident_models=["tinyllama", "qwen_coder"],
            policy=policy,
        )

        # The shaper should NOT have downgraded to direct_code
        # (which would drop the verifier)
        assert result.shaped_route_name != "direct_code" or not result.shaped, \
            "Risky query should not allow verifier drop"

        # Check that if downgrade was attempted, it was blocked
        blocked = [a for a in result.actions if not a.policy_allowed]
        if any(a.rule == "downgrade_route" for a in result.actions):
            # At least one downgrade should be blocked
            assert len(blocked) > 0, \
                "Downgrade that drops verifier should be policy-blocked"

        receipt = result.to_receipt()
        assert receipt["policy_constraints"] is not None
        assert "verifier_locked" in receipt["policy_constraints"]
