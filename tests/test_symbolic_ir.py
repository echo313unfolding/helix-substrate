"""
Tests for symbolic IR v1: schema, parser, validation, verifier integration,
strict mode enforcement, and scheduler routing.

Test categories:
    1. SymbolicIR schema — field presence, known types
    2. Parser — Bio-Poetica verb detection, KRISPER function detection,
       ambiguous input handling, empty input
    3. Validation — required fields, action_type, compiler_target,
       constraint types, parse_status
    4. Verifier integration — verify_symbolic_ir with JSON IR dict,
       raw text fallback, receipt fields
    5. Strict mode — invalid IR blocks execution
    6. Policy routing — symbolic input → symbolic_parse_verify
    7. Receipt fields — ir_schema, ir_valid, ir_issues, compiler_target,
       execution_allowed

Work Order: WO-SYMBOLIC-IR-V1
"""

import json

import pytest

from helix_substrate.symbolic_ir import (
    KRISPER_TO_ACTION,
    KNOWN_CONSTRAINT_TYPES,
    SCHEMA_SYMBOLIC_IR,
    VERB_TO_ACTION,
    ActionType,
    CompilerTarget,
    is_symbolic_input,
    parse_symbolic,
    validate_symbolic_ir,
)
from helix_substrate.verifier import (
    StrictVerificationMode,
    VerificationType,
    verify_symbolic_ir,
)
from helix_substrate.lobe_scheduler import (
    ROUTE_SYMBOLIC_FULL,
    ROUTE_SYMBOLIC_PARSE_VERIFY,
    Policy,
)


# ---------------------------------------------------------------------------
# SymbolicIR Schema
# ---------------------------------------------------------------------------

class TestSchema:

    def test_schema_constant(self):
        assert SCHEMA_SYMBOLIC_IR == "symbolic_ir:v1"

    def test_action_types_complete(self):
        expected = {"create", "transform", "query", "structure",
                    "flow", "emit", "remember", "evolve"}
        assert {a.value for a in ActionType} == expected

    def test_compiler_targets_complete(self):
        expected = {"helix_capsule", "krisper_engine", "lobe_scheduler", "none"}
        assert {c.value for c in CompilerTarget} == expected

    def test_canonical_bio_verbs_mapped(self):
        """All 8 canonical Bio-Poetica verbs are in VERB_TO_ACTION."""
        canonical = {"whisper", "bloom", "flow", "spiral",
                     "grow", "emit", "crystallize", "harmonize"}
        assert canonical.issubset(set(VERB_TO_ACTION.keys()))

    def test_krisper_functions_mapped(self):
        """All 5 KRISPER functions are mapped."""
        expected = {"select", "mutate", "crossover", "evaluate", "evolve"}
        assert set(KRISPER_TO_ACTION.keys()) == expected

    def test_known_constraint_types(self):
        assert "requires_model" in KNOWN_CONSTRAINT_TYPES
        assert "max_tokens" in KNOWN_CONSTRAINT_TYPES
        assert "allow_side_effects" in KNOWN_CONSTRAINT_TYPES


# ---------------------------------------------------------------------------
# Detection: is_symbolic_input
# ---------------------------------------------------------------------------

class TestIsSymbolicInput:

    def test_bio_poetica_verb(self):
        assert is_symbolic_input("bloom the crystal seed") is True

    def test_krisper_function(self):
        assert is_symbolic_input("evolve population over 10 generations") is True

    def test_krisper_function_syntax(self):
        assert is_symbolic_input("mutate(pattern_a, rate=0.1)") is True

    def test_normal_code_query(self):
        assert is_symbolic_input("Write a Python function to sort a list") is False

    def test_normal_plan_query(self):
        assert is_symbolic_input("What is the capital of France?") is False

    def test_empty(self):
        assert is_symbolic_input("") is False

    def test_comment_only(self):
        assert is_symbolic_input("# just a comment") is False

    def test_whisper_at_start(self):
        assert is_symbolic_input("whisper to the void") is True

    def test_crystallize_at_start(self):
        assert is_symbolic_input("crystallize the lattice structure") is True


# ---------------------------------------------------------------------------
# Parser: Bio-Poetica
# ---------------------------------------------------------------------------

class TestParserBioPoetica:

    def test_single_verb(self):
        ir = parse_symbolic("bloom the crystal seed")
        assert ir["ir_schema"] == SCHEMA_SYMBOLIC_IR
        assert ir["action_type"] == "create"
        assert ir["compiler_target"] == "helix_capsule"
        assert ir["parse_status"] == "ok"
        assert ir["parse_issues"] == []
        assert ir["verification_required"] is True
        assert ir["execution_allowed"] is False  # verifier must approve

    def test_multi_verb(self):
        ir = parse_symbolic("whisper hello\nbloom new world\nemit result")
        assert ir["parse_status"] == "ok"
        assert ir["action_type"] == "query"  # first verb: whisper
        assert ir["arguments"]["n_statements"] == 3
        assert ir["arguments"]["verbs"] == ["whisper", "bloom", "emit"]

    def test_intent_from_verb(self):
        ir = parse_symbolic("crystallize the data")
        assert ir["intent"] == "crystallize the data"

    def test_target_from_rest(self):
        ir = parse_symbolic("seed the pattern engine")
        assert ir["target"] == "the pattern engine"

    def test_spiral_is_transform(self):
        ir = parse_symbolic("spiral the data inward")
        assert ir["action_type"] == "transform"

    def test_harmonize_is_structure(self):
        ir = parse_symbolic("harmonize the lattice nodes")
        assert ir["action_type"] == "structure"

    def test_emit_is_emit(self):
        ir = parse_symbolic("emit the compressed output")
        assert ir["action_type"] == "emit"

    def test_remember_is_remember(self):
        ir = parse_symbolic("remember the last state")
        assert ir["action_type"] == "remember"

    def test_comments_stripped(self):
        ir = parse_symbolic("# this is a comment\nbloom the seed")
        assert ir["parse_status"] == "ok"
        assert ir["action_type"] == "create"

    def test_constraints_present(self):
        ir = parse_symbolic("bloom the seed")
        assert isinstance(ir["constraints"], list)
        assert len(ir["constraints"]) > 0
        for c in ir["constraints"]:
            assert "type" in c
            assert "value" in c

    def test_source_text_preserved(self):
        src = "whisper to the void\nbloom the crystal"
        ir = parse_symbolic(src)
        assert ir["source_text"] == src


# ---------------------------------------------------------------------------
# Parser: KRISPER
# ---------------------------------------------------------------------------

class TestParserKRISPER:

    def test_krisper_function(self):
        ir = parse_symbolic("evolve population over 10 generations")
        assert ir["action_type"] == "evolve"
        assert ir["compiler_target"] == "krisper_engine"
        assert ir["parse_status"] == "ok"

    def test_krisper_function_syntax(self):
        ir = parse_symbolic("mutate(pattern_a, rate=0.1)")
        assert ir["action_type"] == "evolve"
        assert ir["compiler_target"] == "krisper_engine"

    def test_krisper_multi_function(self):
        ir = parse_symbolic("select best\nevaluate fitness\nmutate winners")
        assert ir["arguments"]["functions"] == ["select", "evaluate", "mutate"]
        assert ir["arguments"]["n_statements"] == 3

    def test_krisper_intent(self):
        ir = parse_symbolic("crossover parents")
        assert ir["intent"] == "KRISPER crossover parents"


# ---------------------------------------------------------------------------
# Parser: Edge cases
# ---------------------------------------------------------------------------

class TestParserEdgeCases:

    def test_empty_input(self):
        ir = parse_symbolic("")
        assert ir["parse_status"] == "error"
        assert ir["execution_allowed"] is False
        assert any("empty" in i for i in ir["parse_issues"])

    def test_whitespace_only(self):
        ir = parse_symbolic("   \n  \n  ")
        assert ir["parse_status"] == "error"

    def test_ambiguous_input(self):
        ir = parse_symbolic("hello world this is normal text")
        assert ir["parse_status"] == "ambiguous"
        assert ir["execution_allowed"] is False
        assert ir["compiler_target"] == "none"

    def test_verb_no_target(self):
        ir = parse_symbolic("bloom")
        assert ir["parse_status"] == "ok"
        assert ir["target"] == "default"  # fallback

    def test_all_ir_fields_present(self):
        """Every parsed IR has all required fields."""
        for text in ["bloom seed", "evolve pop", "", "random text"]:
            ir = parse_symbolic(text) if text else parse_symbolic("  ")
            for field in ["ir_schema", "intent", "target", "action_type",
                          "arguments", "constraints", "verification_required",
                          "execution_allowed", "source_text", "compiler_target",
                          "parse_status", "parse_issues"]:
                assert field in ir, f"Missing {field} for input '{text}'"


# ---------------------------------------------------------------------------
# Validation: validate_symbolic_ir
# ---------------------------------------------------------------------------

class TestValidation:

    def _make_valid_ir(self):
        return parse_symbolic("bloom the crystal seed")

    def test_valid_ir_passes(self):
        ir = self._make_valid_ir()
        issues = validate_symbolic_ir(ir)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_not_a_dict(self):
        issues = validate_symbolic_ir("not a dict")
        assert issues == ["IR is not a dict"]

    def test_missing_required_field(self):
        ir = self._make_valid_ir()
        del ir["action_type"]
        issues = validate_symbolic_ir(ir)
        assert any("action_type" in i for i in issues)

    def test_wrong_schema(self):
        ir = self._make_valid_ir()
        ir["ir_schema"] = "wrong:v0"
        issues = validate_symbolic_ir(ir)
        assert any("schema mismatch" in i for i in issues)

    def test_unknown_action_type(self):
        ir = self._make_valid_ir()
        ir["action_type"] = "destroy_everything"
        issues = validate_symbolic_ir(ir)
        assert any("unknown action_type" in i for i in issues)

    def test_unknown_compiler_target(self):
        ir = self._make_valid_ir()
        ir["compiler_target"] = "magic_runtime"
        issues = validate_symbolic_ir(ir)
        assert any("unknown compiler_target" in i for i in issues)

    def test_empty_target_non_query(self):
        ir = self._make_valid_ir()
        ir["target"] = ""
        # bloom → create, which is non-query
        issues = validate_symbolic_ir(ir)
        assert any("target is empty" in i for i in issues)

    def test_empty_target_query_ok(self):
        ir = self._make_valid_ir()
        ir["action_type"] = "query"
        ir["target"] = ""
        issues = validate_symbolic_ir(ir)
        assert not any("target is empty" in i for i in issues)

    def test_bad_constraints_type(self):
        ir = self._make_valid_ir()
        ir["constraints"] = "not a list"
        issues = validate_symbolic_ir(ir)
        assert any("constraints is not a list" in i for i in issues)

    def test_unknown_constraint_type(self):
        ir = self._make_valid_ir()
        ir["constraints"] = [{"type": "magic_constraint", "value": True}]
        issues = validate_symbolic_ir(ir)
        assert any("unknown type" in i for i in issues)

    def test_verification_required_false(self):
        ir = self._make_valid_ir()
        ir["verification_required"] = False
        issues = validate_symbolic_ir(ir)
        assert any("verification_required must be True" in i for i in issues)

    def test_ambiguous_parse_flagged(self):
        ir = parse_symbolic("random unrecognized text")
        issues = validate_symbolic_ir(ir)
        assert any("ambiguous" in i for i in issues)

    def test_error_parse_flagged(self):
        ir = parse_symbolic("")
        issues = validate_symbolic_ir(ir)
        assert any("parse error" in i for i in issues)

    def test_empty_source_text(self):
        ir = self._make_valid_ir()
        ir["source_text"] = ""
        issues = validate_symbolic_ir(ir)
        assert any("source_text is empty" in i for i in issues)


# ---------------------------------------------------------------------------
# Verifier integration: verify_symbolic_ir
# ---------------------------------------------------------------------------

class TestVerifierIntegration:

    def test_json_ir_valid(self):
        """Valid SymbolicIR JSON → pass with high confidence."""
        ir = parse_symbolic("bloom the crystal seed")
        result = verify_symbolic_ir(json.dumps(ir))
        assert result.result == "pass"
        assert result.action_allowed is True
        assert result.confidence > 0.9
        assert result.verification_type == VerificationType.SYMBOLIC_IR

    def test_json_ir_invalid(self):
        """Invalid SymbolicIR JSON → fail."""
        ir = parse_symbolic("bloom the crystal seed")
        ir["action_type"] = "nonexistent"
        result = verify_symbolic_ir(json.dumps(ir))
        assert result.result == "fail"
        assert result.action_allowed is False

    def test_json_ir_ambiguous_parse(self):
        """Ambiguous parse → fail, execution blocked."""
        ir = parse_symbolic("random text with no verbs")
        result = verify_symbolic_ir(json.dumps(ir))
        assert result.result == "fail"
        assert result.action_allowed is False

    def test_raw_text_balanced(self):
        """Raw text with balanced delimiters → pass (text fallback)."""
        result = verify_symbolic_ir("(define (add a b) (+ a b))")
        assert result.result == "pass"
        assert result.confidence < 0.8  # lower confidence for text check

    def test_raw_text_unbalanced(self):
        """Raw text with unbalanced delimiters → fail."""
        result = verify_symbolic_ir("(define (add a b)")
        assert result.result == "fail"

    def test_receipt_has_ir_fields(self):
        """Verification receipt includes IR-specific fields."""
        ir = parse_symbolic("bloom the crystal seed")
        result = verify_symbolic_ir(json.dumps(ir))
        receipt = result.to_receipt()

        assert "ir_schema" in receipt
        assert receipt["ir_schema"] == SCHEMA_SYMBOLIC_IR
        assert "ir_valid" in receipt
        assert receipt["ir_valid"] is True
        assert "ir_issues" in receipt
        assert receipt["ir_issues"] == []
        assert "compiler_target" in receipt
        assert receipt["compiler_target"] == "helix_capsule"
        assert "execution_allowed" in receipt
        assert receipt["execution_allowed"] is True

    def test_receipt_ir_fields_on_failure(self):
        """Failed IR receipt still has all IR fields."""
        ir = parse_symbolic("random text")
        result = verify_symbolic_ir(json.dumps(ir))
        receipt = result.to_receipt()

        assert receipt["ir_valid"] is False
        assert len(receipt["ir_issues"]) > 0
        assert receipt["execution_allowed"] is False

    def test_empty_text_fails(self):
        result = verify_symbolic_ir("")
        assert result.result == "fail"
        assert result.action_allowed is False


# ---------------------------------------------------------------------------
# Strict mode: invalid IR blocks execution
# ---------------------------------------------------------------------------

class TestStrictMode:

    def test_valid_ir_passes_strict(self):
        """Valid IR under strict mode should NOT raise."""
        ir = parse_symbolic("bloom the crystal seed")
        result = verify_symbolic_ir(json.dumps(ir))
        with StrictVerificationMode():
            StrictVerificationMode.enforce(result)  # no raise

    def test_invalid_ir_blocked_strict(self):
        """Invalid IR under strict mode MUST raise RuntimeError."""
        ir = parse_symbolic("random unrecognized text")
        result = verify_symbolic_ir(json.dumps(ir))
        assert result.action_allowed is False
        with pytest.raises(RuntimeError, match="StrictVerificationMode"):
            with StrictVerificationMode():
                StrictVerificationMode.enforce(result)

    def test_ambiguous_parse_blocked_strict(self):
        """Ambiguous parse under strict mode blocks."""
        ir = parse_symbolic("hello world no verbs here")
        result = verify_symbolic_ir(json.dumps(ir))
        with pytest.raises(RuntimeError, match="StrictVerificationMode"):
            with StrictVerificationMode():
                StrictVerificationMode.enforce(result)

    def test_empty_input_blocked_strict(self):
        """Empty IR under strict mode blocks."""
        ir = parse_symbolic("")
        result = verify_symbolic_ir(json.dumps(ir))
        with pytest.raises(RuntimeError, match="StrictVerificationMode"):
            with StrictVerificationMode():
                StrictVerificationMode.enforce(result)


# ---------------------------------------------------------------------------
# Policy routing
# ---------------------------------------------------------------------------

class TestPolicyRouting:

    def test_bio_poetica_routes_to_symbolic(self):
        route = Policy.select_route("bloom the crystal seed")
        assert route.name == "symbolic_full"

    def test_krisper_routes_to_symbolic(self):
        route = Policy.select_route("evolve population fitness")
        assert route.name == "symbolic_full"

    def test_normal_code_not_symbolic(self):
        route = Policy.select_route("Write a Python function to sort a list")
        assert route.name != "symbolic_parse_verify"

    def test_normal_plan_not_symbolic(self):
        route = Policy.select_route("What is the capital of France?")
        assert route.name != "symbolic_parse_verify"

    def test_whisper_routes_symbolic(self):
        route = Policy.select_route("whisper to the void")
        assert route.name == "symbolic_full"

    def test_crystallize_routes_symbolic(self):
        route = Policy.select_route("crystallize the lattice")
        assert route.name == "symbolic_full"

    def test_mutate_routes_symbolic(self):
        route = Policy.select_route("mutate(pattern, 0.1)")
        assert route.name == "symbolic_full"


# ---------------------------------------------------------------------------
# Route structure
# ---------------------------------------------------------------------------

class TestRouteStructure:

    def test_parse_verify_route(self):
        """2-step route: parse + verify (no compilation)."""
        route = ROUTE_SYMBOLIC_PARSE_VERIFY
        assert route.selected_lobes == ["parser", "verifier"]
        assert route.steps[0].role == "parse"
        assert route.steps[1].role == "verify"
        assert route.steps[1].verification_type == "symbolic_ir"
        assert route.verification_required is True

    def test_full_route_steps(self):
        """3-step route: parse + verify + compile."""
        route = ROUTE_SYMBOLIC_FULL
        assert route.selected_lobes == ["parser", "verifier", "compiler"]
        assert route.steps[0].role == "parse"
        assert route.steps[1].role == "verify"
        assert route.steps[2].role == "compile"
        assert route.verification_required is True

    def test_compiler_reads_parser_output(self):
        """Compiler's input_from is parser (needs IR, not verifier text)."""
        route = ROUTE_SYMBOLIC_FULL
        assert route.steps[2].input_from == "parser"

    def test_parser_feeds_verifier(self):
        route = ROUTE_SYMBOLIC_FULL
        assert route.steps[0].input_from is None
        assert route.steps[1].input_from == "parser"
