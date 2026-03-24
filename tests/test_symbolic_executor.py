"""
Tests for symbolic executor v1: ActionPlan compilation, executor policy,
action-type dispatch, receipt fields, strict mode, and scheduler integration.

Test categories:
    1. ActionPlan schema — field presence, known types, hashes
    2. Compilation per action type — create, transform, query, structure,
       flow, emit, remember, evolve
    3. Executor policy — blocked IR, unknown actions, fail closed
    4. Validation — required fields, semantic consistency
    5. Scheduler integration — symbolic_full route, receipt fields
    6. Strict mode — blocked IR cannot compile to executable plan

Work Order: WO-SYMBOLIC-EXECUTOR-V1
"""

import json

import pytest

from helix_substrate.symbolic_ir import (
    SCHEMA_SYMBOLIC_IR,
    ActionType,
    CompilerTarget,
    parse_symbolic,
    validate_symbolic_ir,
)
from helix_substrate.symbolic_executor import (
    COMPILE_BLOCKED,
    COMPILE_COMPILED,
    COMPILE_ERROR,
    COMPILE_STUBBED,
    SCHEMA_ACTION_PLAN,
    ExecutionMode,
    compile_symbolic_ir,
    validate_action_plan,
)
from helix_substrate.verifier import (
    StrictVerificationMode,
    verify_symbolic_ir,
)
from helix_substrate.lobe_scheduler import (
    ROUTE_SYMBOLIC_FULL,
    Policy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verified_ir(source_text: str) -> dict:
    """Parse + simulate verifier approval → IR with execution_allowed=True."""
    ir = parse_symbolic(source_text)
    # Simulate verifier approval (in real flow, verifier sets this)
    ir["execution_allowed"] = True
    return ir


def _unverified_ir(source_text: str) -> dict:
    """Parse without verification → execution_allowed=False."""
    return parse_symbolic(source_text)


# ---------------------------------------------------------------------------
# ActionPlan Schema
# ---------------------------------------------------------------------------

class TestActionPlanSchema:

    def test_schema_constant(self):
        assert SCHEMA_ACTION_PLAN == "action_plan:v1"

    def test_execution_modes_complete(self):
        expected = {"lobe_route", "tool_action", "output_only",
                    "memory_write", "symbolic_mutation", "blocked", "stub"}
        assert {m.value for m in ExecutionMode} == expected

    def test_compiled_plan_has_all_fields(self):
        ir = _verified_ir("bloom the crystal seed")
        plan = compile_symbolic_ir(ir)
        required = [
            "plan_schema", "ir_schema", "ir_hash", "action_type",
            "execution_mode", "target", "steps", "tool_calls",
            "lobe_route", "execution_allowed", "fallback_reason",
            "compile_result", "source_symbolic_text", "compiler_target",
            "action_plan_hash",
        ]
        for f in required:
            assert f in plan, f"Missing field: {f}"

    def test_plan_schema_correct(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        assert plan["plan_schema"] == SCHEMA_ACTION_PLAN

    def test_ir_hash_is_sha256(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        assert len(plan["ir_hash"]) == 64  # SHA256 hex

    def test_action_plan_hash_is_sha256(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        assert len(plan["action_plan_hash"]) == 64

    def test_different_inputs_different_hashes(self):
        plan_a = compile_symbolic_ir(_verified_ir("bloom x"))
        plan_b = compile_symbolic_ir(_verified_ir("whisper y"))
        assert plan_a["ir_hash"] != plan_b["ir_hash"]
        assert plan_a["action_plan_hash"] != plan_b["action_plan_hash"]

    def test_source_text_preserved(self):
        ir = _verified_ir("bloom the crystal seed")
        plan = compile_symbolic_ir(ir)
        assert plan["source_symbolic_text"] == "bloom the crystal seed"


# ---------------------------------------------------------------------------
# Compilation per action type
# ---------------------------------------------------------------------------

class TestCompileCreate:

    def test_create_compiles(self):
        plan = compile_symbolic_ir(_verified_ir("bloom the crystal seed"))
        assert plan["compile_result"] == COMPILE_COMPILED
        assert plan["action_type"] == "create"
        assert plan["execution_mode"] == "lobe_route"
        assert plan["lobe_route"] == "plan_code_verify"
        assert plan["execution_allowed"] is True
        assert plan["fallback_reason"] is None

    def test_create_has_steps(self):
        plan = compile_symbolic_ir(_verified_ir("seed new pattern"))
        assert len(plan["steps"]) > 0
        step_actions = [s["action"] for s in plan["steps"]]
        assert "decompose" in step_actions

    def test_grow_is_create(self):
        plan = compile_symbolic_ir(_verified_ir("grow the tree"))
        assert plan["action_type"] == "create"


class TestCompileTransform:

    def test_transform_compiles(self):
        plan = compile_symbolic_ir(_verified_ir("spiral the data inward"))
        assert plan["action_type"] == "transform"
        assert plan["execution_mode"] == "lobe_route"
        assert plan["lobe_route"] == "plan_code_verify"
        assert plan["compile_result"] == COMPILE_COMPILED

    def test_fold_is_transform(self):
        plan = compile_symbolic_ir(_verified_ir("fold the tensor"))
        assert plan["action_type"] == "transform"


class TestCompileQuery:

    def test_query_compiles(self):
        plan = compile_symbolic_ir(_verified_ir("whisper to the void"))
        assert plan["action_type"] == "query"
        assert plan["execution_mode"] == "lobe_route"
        assert plan["lobe_route"] == "direct_plan"
        assert plan["compile_result"] == COMPILE_COMPILED

    def test_echo_is_query(self):
        plan = compile_symbolic_ir(_verified_ir("echo hello world"))
        assert plan["action_type"] == "query"


class TestCompileStructure:

    def test_structure_compiles(self):
        plan = compile_symbolic_ir(_verified_ir("crystallize the lattice"))
        assert plan["action_type"] == "structure"
        assert plan["execution_mode"] == "lobe_route"
        assert plan["lobe_route"] == "plan_code_verify"

    def test_harmonize_is_structure(self):
        plan = compile_symbolic_ir(_verified_ir("harmonize the nodes"))
        assert plan["action_type"] == "structure"


class TestCompileFlow:

    def test_flow_compiles(self):
        plan = compile_symbolic_ir(_verified_ir("flow through the pipeline"))
        assert plan["action_type"] == "flow"
        assert plan["execution_mode"] == "lobe_route"
        assert plan["lobe_route"] == "direct_plan"

    def test_drift_is_flow(self):
        plan = compile_symbolic_ir(_verified_ir("drift along the stream"))
        assert plan["action_type"] == "flow"


class TestCompileEmit:

    def test_emit_compiles(self):
        plan = compile_symbolic_ir(_verified_ir("emit the compressed output"))
        assert plan["action_type"] == "emit"
        assert plan["execution_mode"] == "output_only"
        assert plan["lobe_route"] is None
        assert plan["compile_result"] == COMPILE_COMPILED

    def test_celebrate_is_emit(self):
        plan = compile_symbolic_ir(_verified_ir("celebrate the result"))
        assert plan["action_type"] == "emit"


class TestCompileRemember:

    def test_remember_compiles(self):
        plan = compile_symbolic_ir(_verified_ir("remember the last state"))
        assert plan["action_type"] == "remember"
        assert plan["execution_mode"] == "memory_write"
        assert plan["lobe_route"] is None
        assert plan["compile_result"] == COMPILE_COMPILED

    def test_remember_has_write_step(self):
        plan = compile_symbolic_ir(_verified_ir("remember this"))
        step_actions = [s["action"] for s in plan["steps"]]
        assert "store" in step_actions


class TestCompileEvolve:

    def test_evolve_is_stubbed(self):
        """Evolve actions are stubbed — KRISPER runtime not implemented."""
        plan = compile_symbolic_ir(_verified_ir("evolve the population"))
        assert plan["action_type"] == "evolve"
        assert plan["execution_mode"] == "stub"
        assert plan["execution_allowed"] is False
        assert plan["compile_result"] == COMPILE_STUBBED
        assert "not yet implemented" in plan["fallback_reason"]

    def test_mutate_is_stubbed(self):
        plan = compile_symbolic_ir(_verified_ir("mutate pattern_a"))
        assert plan["compile_result"] == COMPILE_STUBBED

    def test_select_is_stubbed(self):
        plan = compile_symbolic_ir(_verified_ir("select best candidates"))
        assert plan["compile_result"] == COMPILE_STUBBED


# ---------------------------------------------------------------------------
# Executor policy: blocked IR, unknown actions, fail closed
# ---------------------------------------------------------------------------

class TestExecutorPolicy:

    def test_unverified_ir_blocked(self):
        """IR without execution_allowed=True produces blocked plan."""
        ir = _unverified_ir("bloom the crystal seed")
        assert ir["execution_allowed"] is False
        plan = compile_symbolic_ir(ir)
        assert plan["compile_result"] == COMPILE_BLOCKED
        assert plan["execution_allowed"] is False
        assert plan["execution_mode"] == "blocked"
        assert "not allowed by verifier" in plan["fallback_reason"]

    def test_ambiguous_ir_blocked(self):
        """Ambiguous parse → blocked."""
        ir = _unverified_ir("random text no verbs")
        plan = compile_symbolic_ir(ir)
        assert plan["compile_result"] == COMPILE_BLOCKED
        assert plan["execution_allowed"] is False

    def test_empty_ir_blocked(self):
        """Empty parse → blocked."""
        ir = _unverified_ir("")
        plan = compile_symbolic_ir(ir)
        assert plan["compile_result"] == COMPILE_BLOCKED

    def test_unknown_action_type_error(self):
        """Unknown action_type → error plan (fail closed)."""
        ir = _verified_ir("bloom x")
        ir["action_type"] = "destroy_world"
        plan = compile_symbolic_ir(ir)
        assert plan["compile_result"] == COMPILE_ERROR
        assert plan["execution_allowed"] is False
        assert "unknown action_type" in plan["fallback_reason"]

    def test_not_a_dict_error(self):
        plan = compile_symbolic_ir("not a dict")
        assert plan["compile_result"] == COMPILE_ERROR
        assert plan["execution_allowed"] is False

    def test_blocked_plan_has_no_steps(self):
        """Blocked/error plans have empty steps and tool_calls."""
        ir = _unverified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        assert plan["steps"] == []
        assert plan["tool_calls"] == []

    def test_blocked_plan_has_no_lobe_route(self):
        ir = _unverified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        assert plan["lobe_route"] is None

    def test_parse_issues_in_fallback_reason(self):
        """Parse issues propagate to fallback_reason."""
        ir = _unverified_ir("random no verbs")
        plan = compile_symbolic_ir(ir)
        assert "no recognized" in plan["fallback_reason"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_plan_passes(self):
        ir = _verified_ir("bloom the seed")
        plan = compile_symbolic_ir(ir)
        issues = validate_action_plan(plan)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_blocked_plan_passes(self):
        """Even blocked plans should validate (they have all fields)."""
        ir = _unverified_ir("bloom the seed")
        plan = compile_symbolic_ir(ir)
        issues = validate_action_plan(plan)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_stubbed_plan_passes(self):
        ir = _verified_ir("evolve pop")
        plan = compile_symbolic_ir(ir)
        issues = validate_action_plan(plan)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_not_a_dict(self):
        issues = validate_action_plan("not a dict")
        assert issues == ["plan is not a dict"]

    def test_missing_field(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        del plan["ir_hash"]
        issues = validate_action_plan(plan)
        assert any("ir_hash" in i for i in issues)

    def test_wrong_schema(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        plan["plan_schema"] = "wrong:v0"
        issues = validate_action_plan(plan)
        assert any("schema mismatch" in i for i in issues)

    def test_unknown_execution_mode(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        plan["execution_mode"] = "teleport"
        issues = validate_action_plan(plan)
        assert any("unknown execution_mode" in i for i in issues)

    def test_blocked_but_allowed_inconsistent(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        plan["execution_mode"] = "blocked"
        plan["execution_allowed"] = True
        issues = validate_action_plan(plan)
        assert any("inconsistent" in i.lower() or "blocked" in i for i in issues)

    def test_not_allowed_needs_reason(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        plan["execution_allowed"] = False
        plan["fallback_reason"] = None
        issues = validate_action_plan(plan)
        assert any("fallback_reason" in i for i in issues)

    def test_lobe_route_mode_needs_route(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        plan["lobe_route"] = None
        issues = validate_action_plan(plan)
        assert any("lobe_route" in i for i in issues)


# ---------------------------------------------------------------------------
# Full pipeline: parse → verify → compile
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_bio_poetica_full_pipeline(self):
        """Bio-Poetica text → IR → verify → compile → valid plan."""
        source = "bloom the crystal seed"
        ir = parse_symbolic(source)
        assert ir["parse_status"] == "ok"

        # Verify
        vr = verify_symbolic_ir(json.dumps(ir))
        assert vr.action_allowed is True

        # Approve and compile
        ir["execution_allowed"] = True
        plan = compile_symbolic_ir(ir)
        assert plan["compile_result"] == COMPILE_COMPILED
        assert plan["execution_allowed"] is True
        assert validate_action_plan(plan) == []

    def test_krisper_full_pipeline_stubbed(self):
        """KRISPER text → IR → verify → compile → stubbed plan."""
        source = "evolve population fitness"
        ir = parse_symbolic(source)
        ir["execution_allowed"] = True
        plan = compile_symbolic_ir(ir)
        assert plan["compile_result"] == COMPILE_STUBBED
        assert plan["execution_allowed"] is False
        assert validate_action_plan(plan) == []

    def test_ambiguous_blocks_at_verify(self):
        """Ambiguous input blocks at verification, compiler sees blocked IR."""
        source = "random text with no verbs"
        ir = parse_symbolic(source)
        vr = verify_symbolic_ir(json.dumps(ir))
        assert vr.action_allowed is False
        # IR stays execution_allowed=False
        plan = compile_symbolic_ir(ir)
        assert plan["compile_result"] == COMPILE_BLOCKED

    def test_empty_blocks_at_parse(self):
        """Empty input blocks at parse, never reaches compiler."""
        ir = parse_symbolic("")
        plan = compile_symbolic_ir(ir)
        assert plan["compile_result"] == COMPILE_BLOCKED


# ---------------------------------------------------------------------------
# Strict mode integration
# ---------------------------------------------------------------------------

class TestStrictModeIntegration:

    def test_valid_plan_passes_strict(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        # Plan is compiled — should not raise
        vr = verify_symbolic_ir(json.dumps(ir))
        with StrictVerificationMode():
            StrictVerificationMode.enforce(vr)

    def test_invalid_ir_blocks_under_strict(self):
        """Invalid IR blocked by verifier → cannot compile under strict mode."""
        ir = parse_symbolic("random text")
        vr = verify_symbolic_ir(json.dumps(ir))
        with pytest.raises(RuntimeError, match="StrictVerificationMode"):
            with StrictVerificationMode():
                StrictVerificationMode.enforce(vr)


# ---------------------------------------------------------------------------
# Policy routing
# ---------------------------------------------------------------------------

class TestPolicyRouting:

    def test_symbolic_routes_to_full(self):
        """Symbolic input → symbolic_full route (parse → verify → compile)."""
        route = Policy.select_route("bloom the crystal seed")
        assert route.name == "symbolic_full"
        assert route.selected_lobes == ["parser", "verifier", "compiler"]

    def test_route_has_compile_step(self):
        route = Policy.select_route("whisper to the void")
        compile_steps = [s for s in route.steps if s.role == "compile"]
        assert len(compile_steps) == 1


# ---------------------------------------------------------------------------
# Receipt fields
# ---------------------------------------------------------------------------

class TestReceiptFields:

    def test_plan_has_receipt_fields(self):
        """ActionPlan includes all WO-specified receipt fields."""
        ir = _verified_ir("bloom the seed")
        plan = compile_symbolic_ir(ir)
        assert plan["source_symbolic_text"] == "bloom the seed"
        assert len(plan["ir_hash"]) == 64
        assert len(plan["action_plan_hash"]) == 64
        assert plan["compile_result"] == COMPILE_COMPILED
        assert plan["execution_mode"] == "lobe_route"
        assert plan["fallback_reason"] is None

    def test_blocked_plan_has_blocked_reason(self):
        ir = _unverified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        assert plan["fallback_reason"] is not None
        assert "not allowed" in plan["fallback_reason"]

    def test_compiler_target_preserved(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        assert plan["compiler_target"] == "helix_capsule"

        ir_k = _verified_ir("evolve pop")
        plan_k = compile_symbolic_ir(ir_k)
        assert plan_k["compiler_target"] == "krisper_engine"

    def test_ir_schema_in_plan(self):
        ir = _verified_ir("bloom x")
        plan = compile_symbolic_ir(ir)
        assert plan["ir_schema"] == SCHEMA_SYMBOLIC_IR
