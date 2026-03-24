"""
Tests for lobe_scheduler: route selection, receipt validation, lobe registry,
and (when GPU available) end-to-end multi-lobe execution.

Work Order: WO-LOBE-SCHEDULER-V0
"""

import pytest

from helix_substrate.lobe_scheduler import (
    LOBE_REGISTRY,
    ROUTE_DIRECT_CODE,
    ROUTE_DIRECT_PLAN,
    ROUTE_PLAN_CODE_VERIFY,
    ROUTE_CODE_VERIFY,
    ROUTE_PLAN_VERIFY,
    ROUTE_SYMBOLIC_PARSE_VERIFY,
    ROUTE_SYMBOLIC_FULL,
    SCHEMA_ROUTE_RECEIPT,
    Capability,
    Lobe,
    Policy,
    Route,
    RouteStep,
    StepResult,
    TaskResult,
    get_lobe,
    list_lobes,
    validate_route_receipt,
)
from helix_substrate.query_classifier import ModelTarget


# ---------------------------------------------------------------------------
# Lobe Registry
# ---------------------------------------------------------------------------

class TestLobeRegistry:

    def test_six_lobes_registered(self):
        """Registry has exactly planner, coder, verifier, memory, parser, compiler."""
        names = list_lobes()
        assert set(names) == {"planner", "coder", "verifier", "memory", "parser", "compiler"}

    def test_get_lobe(self):
        """get_lobe returns correct lobe."""
        coder = get_lobe("coder")
        assert coder.name == "coder"
        assert coder.capability.model_target == ModelTarget.QWEN_CODER

    def test_get_lobe_missing(self):
        """Unknown lobe raises KeyError."""
        with pytest.raises(KeyError):
            get_lobe("nonexistent")

    def test_planner_uses_tinyllama(self):
        planner = get_lobe("planner")
        assert planner.capability.model_target == ModelTarget.TINYLLAMA

    def test_verifier_uses_tinyllama(self):
        verifier = get_lobe("verifier")
        assert verifier.capability.model_target == ModelTarget.TINYLLAMA

    def test_memory_has_no_model(self):
        memory = get_lobe("memory")
        assert memory.capability.model_target is None

    def test_parser_has_no_model(self):
        parser = get_lobe("parser")
        assert parser.capability.model_target is None
        assert "symbolic_parsing" in parser.capability.strengths

    def test_compiler_has_no_model(self):
        compiler = get_lobe("compiler")
        assert compiler.capability.model_target is None
        assert "symbolic_compilation" in compiler.capability.strengths

    def test_prompt_template_has_task(self):
        """All model-backed lobes have {task} in prompt template."""
        for name, lobe in LOBE_REGISTRY.items():
            if lobe.capability.model_target is not None:
                assert "{task}" in lobe.capability.prompt_template, (
                    f"{name} prompt template missing {{task}}"
                )

    def test_format_prompt(self):
        coder = get_lobe("coder")
        prompt = coder.format_prompt(task="Write hello world", context="Plan: print it\n\n")
        assert "Write hello world" in prompt
        assert "Plan: print it" in prompt


# ---------------------------------------------------------------------------
# Route Structure
# ---------------------------------------------------------------------------

class TestRouteStructure:

    def test_direct_code_single_step(self):
        assert len(ROUTE_DIRECT_CODE.steps) == 1
        assert ROUTE_DIRECT_CODE.steps[0].lobe_name == "coder"
        assert ROUTE_DIRECT_CODE.verification_required is False

    def test_direct_plan_single_step(self):
        assert len(ROUTE_DIRECT_PLAN.steps) == 1
        assert ROUTE_DIRECT_PLAN.steps[0].lobe_name == "planner"

    def test_plan_code_verify_three_steps(self):
        assert len(ROUTE_PLAN_CODE_VERIFY.steps) == 3
        assert ROUTE_PLAN_CODE_VERIFY.selected_lobes == ["planner", "coder", "verifier"]
        assert ROUTE_PLAN_CODE_VERIFY.verification_required is True

    def test_code_verify_two_steps(self):
        assert len(ROUTE_CODE_VERIFY.steps) == 2
        assert ROUTE_CODE_VERIFY.selected_lobes == ["coder", "verifier"]
        assert ROUTE_CODE_VERIFY.verification_required is True

    def test_route_data_flow(self):
        """plan_code_verify: planner→coder→verifier input chain."""
        steps = ROUTE_PLAN_CODE_VERIFY.steps
        assert steps[0].input_from is None  # planner starts fresh
        assert steps[1].input_from == "planner"  # coder gets planner output
        assert steps[2].input_from == "coder"  # verifier gets coder output

    def test_plan_verify_two_steps(self):
        assert len(ROUTE_PLAN_VERIFY.steps) == 2
        assert ROUTE_PLAN_VERIFY.selected_lobes == ["planner", "verifier"]
        assert ROUTE_PLAN_VERIFY.verification_required is True

    def test_verification_type_on_verify_steps(self):
        """Verify steps have verification_type set."""
        for route in [ROUTE_CODE_VERIFY, ROUTE_PLAN_CODE_VERIFY, ROUTE_PLAN_VERIFY]:
            verify_steps = [s for s in route.steps if s.role == "verify"]
            assert len(verify_steps) >= 1
            for step in verify_steps:
                assert step.verification_type == "code_sanity"

    def test_symbolic_parse_verify_route(self):
        assert len(ROUTE_SYMBOLIC_PARSE_VERIFY.steps) == 2
        assert ROUTE_SYMBOLIC_PARSE_VERIFY.selected_lobes == ["parser", "verifier"]
        assert ROUTE_SYMBOLIC_PARSE_VERIFY.verification_required is True
        assert ROUTE_SYMBOLIC_PARSE_VERIFY.steps[1].verification_type == "symbolic_ir"

    def test_symbolic_full_route(self):
        assert len(ROUTE_SYMBOLIC_FULL.steps) == 3
        assert ROUTE_SYMBOLIC_FULL.selected_lobes == ["parser", "verifier", "compiler"]
        assert ROUTE_SYMBOLIC_FULL.verification_required is True
        assert ROUTE_SYMBOLIC_FULL.steps[2].role == "compile"

    def test_selected_lobes_matches_steps(self):
        for route in [ROUTE_DIRECT_CODE, ROUTE_DIRECT_PLAN,
                      ROUTE_PLAN_CODE_VERIFY, ROUTE_CODE_VERIFY,
                      ROUTE_PLAN_VERIFY, ROUTE_SYMBOLIC_PARSE_VERIFY,
                      ROUTE_SYMBOLIC_FULL]:
            assert route.selected_lobes == [s.lobe_name for s in route.steps]


# ---------------------------------------------------------------------------
# Policy — Route Selection
# ---------------------------------------------------------------------------

class TestPolicy:

    def test_simple_code_request(self):
        """Simple code query → code_verify (v1: all code routes verify by default)."""
        route = Policy.select_route("Write a Python function to add two numbers")
        assert route.name == "code_verify"

    def test_planning_request(self):
        """Non-code query → direct_plan."""
        route = Policy.select_route("What is the capital of France?")
        assert route.name == "direct_plan"

    def test_complex_code_request(self):
        """Code + complexity keywords → plan_code_verify."""
        route = Policy.select_route(
            "Design and implement a class for a binary search tree"
        )
        assert route.name == "plan_code_verify"

    def test_verify_code_request(self):
        """Code + verify keywords → code_verify."""
        route = Policy.select_route(
            "Write a sort function and verify it handles edge cases"
        )
        assert route.name == "code_verify"

    def test_build_keyword_triggers_complex(self):
        route = Policy.select_route("Build a REST API endpoint for user auth")
        assert route.name == "plan_code_verify"

    def test_refactor_keyword_triggers_complex(self):
        route = Policy.select_route("Refactor the database connection pool code")
        assert route.name == "plan_code_verify"

    def test_escalation_delete_triggers_plan_verify(self):
        """Non-code + 'delete' → plan_verify (escalation)."""
        route = Policy.select_route("Delete all records from the production database")
        assert route.name == "plan_verify"

    def test_escalation_deploy_triggers_plan_verify(self):
        """Non-code + 'deploy' → plan_verify (escalation)."""
        route = Policy.select_route("Deploy the new configuration to production")
        assert route.name == "plan_verify"

    def test_route_has_reason(self):
        """Every route has a non-empty reason."""
        for query in [
            "hello",
            "write code",
            "implement a system",
            "check this code works",
        ]:
            route = Policy.select_route(query)
            assert route.reason, f"Empty reason for query: {query}"

    def test_all_routes_have_valid_lobes(self):
        """All lobe names in routes exist in registry."""
        for query in [
            "hello",
            "write code",
            "implement a system",
            "verify my function",
        ]:
            route = Policy.select_route(query)
            for lobe_name in route.selected_lobes:
                assert lobe_name in LOBE_REGISTRY, (
                    f"Route {route.name} references unknown lobe {lobe_name}"
                )


# ---------------------------------------------------------------------------
# Route Receipt Validation
# ---------------------------------------------------------------------------

def _make_valid_route_receipt():
    """Build a minimal valid route receipt."""
    return {
        "schema": SCHEMA_ROUTE_RECEIPT,
        "query_sha256": "abc123",
        "query_length": 42,
        "route": {
            "name": "plan_code_verify",
            "selected_lobes": ["planner", "coder", "verifier"],
            "route_reason": "Complex code request",
            "verification_required": True,
            "n_steps": 3,
        },
        "steps": [
            {
                "lobe": "planner",
                "role": "plan",
                "model_used": "tinyllama",
                "dispatch_path": "fused",
                "tokens_generated": 50,
                "timing_ms": 500.0,
            },
            {
                "lobe": "coder",
                "role": "generate",
                "model_used": "qwen_coder",
                "dispatch_path": "fused",
                "tokens_generated": 100,
                "timing_ms": 1200.0,
            },
            {
                "lobe": "verifier",
                "role": "verify",
                "model_used": "tinyllama",
                "dispatch_path": "fused",
                "tokens_generated": 30,
                "timing_ms": 300.0,
            },
        ],
        "summary": {
            "total_tokens_generated": 180,
            "total_timing_ms": 2000.0,
            "models_used": ["qwen_coder", "tinyllama"],
            "n_model_swaps": 1,
            "all_fused": True,
        },
        "cost": {
            "wall_time_s": 12.5,
            "cpu_time_s": 11.0,
            "peak_memory_mb": 3000.0,
            "python_version": "3.10.12",
            "hostname": "Echo",
            "timestamp_start": "2026-03-13T20:00:00+00:00",
            "timestamp_end": "2026-03-13T20:00:12+00:00",
        },
    }


class TestValidateRouteReceipt:

    def test_valid_receipt(self):
        issues = validate_route_receipt(_make_valid_route_receipt())
        assert issues == [], f"Unexpected issues: {issues}"

    def test_missing_schema(self):
        r = _make_valid_route_receipt()
        del r["schema"]
        issues = validate_route_receipt(r)
        assert any("schema" in i for i in issues)

    def test_wrong_schema(self):
        r = _make_valid_route_receipt()
        r["schema"] = "wrong:v1"
        issues = validate_route_receipt(r)
        assert any("schema mismatch" in i for i in issues)

    def test_missing_route_field(self):
        r = _make_valid_route_receipt()
        del r["route"]["selected_lobes"]
        issues = validate_route_receipt(r)
        assert any("selected_lobes" in i for i in issues)

    def test_missing_step_field(self):
        r = _make_valid_route_receipt()
        del r["steps"][0]["lobe"]
        issues = validate_route_receipt(r)
        assert any("lobe" in i for i in issues)

    def test_missing_summary_field(self):
        r = _make_valid_route_receipt()
        del r["summary"]["all_fused"]
        issues = validate_route_receipt(r)
        assert any("all_fused" in i for i in issues)

    def test_missing_cost_field(self):
        r = _make_valid_route_receipt()
        del r["cost"]["wall_time_s"]
        issues = validate_route_receipt(r)
        assert any("wall_time_s" in i for i in issues)

    def test_unknown_lobe_in_route(self):
        r = _make_valid_route_receipt()
        r["route"]["selected_lobes"] = ["planner", "mystery_lobe"]
        issues = validate_route_receipt(r)
        assert any("unknown lobe" in i for i in issues)

    def test_n_steps_mismatch(self):
        r = _make_valid_route_receipt()
        r["route"]["n_steps"] = 5
        issues = validate_route_receipt(r)
        assert any("n_steps" in i for i in issues)

    def test_verification_required_but_no_verify_step(self):
        r = _make_valid_route_receipt()
        r["route"]["verification_required"] = True
        # Remove verifier step
        r["steps"] = [s for s in r["steps"] if s["role"] != "verify"]
        r["route"]["n_steps"] = len(r["steps"])
        r["route"]["selected_lobes"] = ["planner", "coder"]
        issues = validate_route_receipt(r)
        assert any("verification_required" in i for i in issues)

    def test_not_a_dict(self):
        issues = validate_route_receipt("not a dict")
        assert issues == ["receipt is not a dict"]


# ---------------------------------------------------------------------------
# Integration: LobeScheduler (GPU required, real models)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA required for e2e scheduler test",
)
class TestLobeSchedulerE2E:
    """End-to-end scheduler tests on real models.

    These load actual models on GPU. Slow but proves the full pipeline.
    Skipped in CI without GPU.
    """

    @pytest.fixture(autouse=True)
    def setup_manager(self):
        import gc
        gc.collect()
        if __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()
        from helix_substrate.model_manager import ModelManager
        from helix_substrate.lobe_scheduler import LobeScheduler
        self.mgr = ModelManager(device="cuda:0")
        self.scheduler = LobeScheduler(self.mgr)
        yield
        # Cleanup: unload model to free VRAM for next test
        self.mgr._unload()
        gc.collect()
        if __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()

    def test_direct_code_route(self):
        """Direct code route: single coder step, produces code."""
        result = self.scheduler.execute(
            "Write a Python function to compute fibonacci numbers",
            max_tokens=64,
            route_override="direct_code",
        )
        assert result.route.name == "direct_code"
        assert len(result.steps) == 1
        assert result.steps[0].lobe_name == "coder"
        assert result.steps[0].tokens_generated > 0
        assert result.final_output != ""

        # Receipt validation
        issues = validate_route_receipt(result.route_receipt)
        assert issues == [], f"Receipt issues: {issues}"

        # Receipt content
        r = result.route_receipt
        assert r["route"]["selected_lobes"] == ["coder"]
        assert r["route"]["verification_required"] is False
        assert r["summary"]["models_used"] == ["qwen_coder"]

    def test_plan_code_verify_route(self):
        """Full 3-step route: planner → coder → verifier."""
        result = self.scheduler.execute(
            "Design and implement a class for a stack data structure",
            max_tokens=64,
            route_override="plan_code_verify",
        )
        assert result.route.name == "plan_code_verify"
        assert len(result.steps) == 3
        assert [s.lobe_name for s in result.steps] == ["planner", "coder", "verifier"]

        # All steps produced output
        for step in result.steps:
            assert step.tokens_generated > 0, f"{step.lobe_name} produced 0 tokens"

        # Receipt validation
        issues = validate_route_receipt(result.route_receipt)
        assert issues == [], f"Receipt issues: {issues}"

        # Receipt content
        r = result.route_receipt
        assert r["route"]["selected_lobes"] == ["planner", "coder", "verifier"]
        assert r["route"]["verification_required"] is True
        assert r["summary"]["n_model_swaps"] >= 1  # at least one swap
        assert r["summary"]["total_tokens_generated"] > 0
        assert "cost" in r
        assert r["cost"]["wall_time_s"] > 0

    def test_route_receipt_has_all_step_details(self):
        """Each step in receipt has model, dispatch, timing."""
        result = self.scheduler.execute(
            "Write a hello world function",
            max_tokens=32,
            route_override="direct_code",
        )
        for step_receipt in result.route_receipt["steps"]:
            assert step_receipt["model_used"] is not None
            assert step_receipt["timing_ms"] > 0
            assert step_receipt["tokens_generated"] >= 0
