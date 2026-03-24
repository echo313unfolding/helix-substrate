"""
Symbolic IR v1 — strict intermediate representation for KRISPER/BioPoetica input.

All symbolic input MUST become a SymbolicIR dict before tool/model execution.
No freeform symbolic execution.

Schema fields:
    ir_schema            — "symbolic_ir:v1"
    intent               — what the user wants (human-readable)
    target               — what entity/object is affected
    action_type          — ActionType value
    arguments            — key-value parameters
    constraints          — list of constraint dicts
    verification_required — always True (strict IR)
    execution_allowed    — starts False, set True by verifier
    source_text          — original input
    compiler_target      — which compiler/runtime handles this
    parse_status         — "ok", "ambiguous", "error"
    parse_issues         — list of parse issues (empty if ok)

Parser:
    parse_symbolic(text) → SymbolicIR dict
    Detects Bio-Poetica verbs and KRISPER functions, maps to ActionType,
    sets execution_allowed=False until verifier approves.

Validation:
    validate_symbolic_ir(ir_dict) → list of issues (empty = valid)

Work Order: WO-SYMBOLIC-IR-V1
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SYMBOLIC_IR = "symbolic_ir:v1"


# ---------------------------------------------------------------------------
# Action types — semantic categories for symbolic operations
# ---------------------------------------------------------------------------

class ActionType(Enum):
    CREATE = "create"           # bloom, seed, grow, ignite
    TRANSFORM = "transform"     # transmute, mutate, crossover, fold, unfold, spiral
    QUERY = "query"             # whisper, echo, feel, pulse, breathe
    STRUCTURE = "structure"     # crystallize, harmonize, weave, align, anchor, bridge
    FLOW = "flow"               # flow, drift, resonate, resilience
    EMIT = "emit"               # emit, celebrate, flare
    REMEMBER = "remember"       # remember, mirror
    EVOLVE = "evolve"           # evolve, evaluate, select (KRISPER)


_ACTION_TYPE_VALUES: Set[str] = {a.value for a in ActionType}


# ---------------------------------------------------------------------------
# Compiler targets
# ---------------------------------------------------------------------------

class CompilerTarget(Enum):
    HELIX_CAPSULE = "helix_capsule"       # Bio-Poetica canonical substrate
    KRISPER_ENGINE = "krisper_engine"      # KRISPER evolutionary runtime
    LOBE_SCHEDULER = "lobe_scheduler"      # Route to lobe scheduler for LLM
    NONE = "none"                          # No execution target


_COMPILER_TARGET_VALUES: Set[str] = {c.value for c in CompilerTarget}


# ---------------------------------------------------------------------------
# Verb / function → ActionType mappings
# ---------------------------------------------------------------------------

# Bio-Poetica verbs (canonical 8 + extended from prototype + symbol registry)
VERB_TO_ACTION: Dict[str, ActionType] = {
    # Canonical 8
    "whisper": ActionType.QUERY,
    "bloom": ActionType.CREATE,
    "flow": ActionType.FLOW,
    "spiral": ActionType.TRANSFORM,
    "grow": ActionType.CREATE,
    "emit": ActionType.EMIT,
    "crystallize": ActionType.STRUCTURE,
    "harmonize": ActionType.STRUCTURE,
    # Extended (prototype compiler)
    "breathe": ActionType.QUERY,
    "celebrate": ActionType.EMIT,
    "weave": ActionType.STRUCTURE,
    "align": ActionType.STRUCTURE,
    "feel": ActionType.QUERY,
    "pulse": ActionType.QUERY,
    "fold": ActionType.TRANSFORM,
    "unfold": ActionType.TRANSFORM,
    "resonate": ActionType.FLOW,
    "transmute": ActionType.TRANSFORM,
    "seed": ActionType.CREATE,
    "drift": ActionType.FLOW,
    "anchor": ActionType.STRUCTURE,
    "bridge": ActionType.STRUCTURE,
    "echo": ActionType.QUERY,
    "remember": ActionType.REMEMBER,
    # Symbol registry extras
    "ignite": ActionType.CREATE,
    "flare": ActionType.EMIT,
    "resilience": ActionType.FLOW,
    "mirror": ActionType.REMEMBER,
}

# KRISPER evolutionary functions
KRISPER_TO_ACTION: Dict[str, ActionType] = {
    "select": ActionType.EVOLVE,
    "mutate": ActionType.EVOLVE,
    "crossover": ActionType.EVOLVE,
    "evaluate": ActionType.EVOLVE,
    "evolve": ActionType.EVOLVE,
}

# All known symbolic tokens (for detection)
ALL_SYMBOLIC_TOKENS: Set[str] = set(VERB_TO_ACTION) | set(KRISPER_TO_ACTION)


# ---------------------------------------------------------------------------
# Known constraint types
# ---------------------------------------------------------------------------

KNOWN_CONSTRAINT_TYPES: Set[str] = {
    "requires_model",
    "max_tokens",
    "target_type",
    "allow_side_effects",
    "idempotent",
    "requires_gpu",
    "memory_limit_mb",
}


# ---------------------------------------------------------------------------
# Default constraints per action type
# ---------------------------------------------------------------------------

_DEFAULT_CONSTRAINTS: Dict[str, List[dict]] = {
    "create": [
        {"type": "allow_side_effects", "value": True},
        {"type": "requires_model", "value": False},
    ],
    "transform": [
        {"type": "allow_side_effects", "value": True},
        {"type": "requires_model", "value": False},
    ],
    "query": [
        {"type": "allow_side_effects", "value": False},
        {"type": "requires_model", "value": False},
    ],
    "structure": [
        {"type": "allow_side_effects", "value": True},
        {"type": "requires_model", "value": False},
    ],
    "flow": [
        {"type": "allow_side_effects", "value": False},
        {"type": "requires_model", "value": False},
    ],
    "emit": [
        {"type": "allow_side_effects", "value": True},
        {"type": "requires_model", "value": False},
    ],
    "remember": [
        {"type": "allow_side_effects", "value": True},
        {"type": "requires_model", "value": False},
    ],
    "evolve": [
        {"type": "allow_side_effects", "value": True},
        {"type": "requires_model", "value": False},
        {"type": "idempotent", "value": False},
    ],
}


# ---------------------------------------------------------------------------
# IR builder
# ---------------------------------------------------------------------------

def _make_ir(
    *,
    intent: str,
    target: str,
    action_type: str,
    arguments: dict,
    constraints: List[dict],
    verification_required: bool,
    execution_allowed: bool,
    source_text: str,
    compiler_target: str,
    parse_status: str,
    parse_issues: List[str],
) -> dict:
    """Build a SymbolicIR dict. All fields explicit — no defaults."""
    return {
        "ir_schema": SCHEMA_SYMBOLIC_IR,
        "intent": intent,
        "target": target,
        "action_type": action_type,
        "arguments": arguments,
        "constraints": constraints,
        "verification_required": verification_required,
        "execution_allowed": execution_allowed,
        "source_text": source_text,
        "compiler_target": compiler_target,
        "parse_status": parse_status,
        "parse_issues": parse_issues,
    }


# ---------------------------------------------------------------------------
# Detection: is this symbolic input?
# ---------------------------------------------------------------------------

def is_symbolic_input(query: str) -> bool:
    """Check if query is symbolic (Bio-Poetica or KRISPER) input.

    Checks first token of first non-comment line against known verbs/functions.
    Also checks for KRISPER function syntax: func(...).
    """
    lines = [l.strip() for l in query.strip().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return False

    first_line = lines[0]
    tokens = first_line.split()
    if not tokens:
        return False

    first_token = tokens[0].lower()

    # Direct verb/function match
    if first_token in ALL_SYMBOLIC_TOKENS:
        return True

    # KRISPER function syntax: func(...)
    m = re.match(r"(\w+)\s*\(", first_line)
    if m and m.group(1).lower() in KRISPER_TO_ACTION:
        return True

    return False


# ---------------------------------------------------------------------------
# Parser: text → SymbolicIR dict
# ---------------------------------------------------------------------------

def parse_symbolic(source_text: str) -> dict:
    """Parse KRISPER/BioPoetica text into a SymbolicIR dict.

    Rules:
        - Detects Bio-Poetica verbs and KRISPER functions
        - Maps to ActionType and CompilerTarget
        - Sets execution_allowed=False (verifier must approve)
        - Ambiguous input gets parse_status="ambiguous", execution blocked

    Returns:
        SymbolicIR dict (always valid schema, even on parse failure)
    """
    lines = [l.strip() for l in source_text.strip().splitlines()
             if l.strip() and not l.strip().startswith("#")]

    if not lines:
        return _make_ir(
            intent="empty",
            target="",
            action_type=ActionType.QUERY.value,
            arguments={},
            constraints=[],
            verification_required=True,
            execution_allowed=False,
            source_text=source_text,
            compiler_target=CompilerTarget.NONE.value,
            parse_status="error",
            parse_issues=["empty source text"],
        )

    # Scan for recognized tokens
    verbs_found: List[Tuple[str, str]] = []    # (verb, rest_of_line)
    krisper_found: List[Tuple[str, str]] = []   # (func, args_text)

    for line in lines:
        tokens = line.split()
        first = tokens[0].lower()
        rest = " ".join(tokens[1:]) if len(tokens) > 1 else ""

        if first in VERB_TO_ACTION:
            verbs_found.append((first, rest))
            continue

        if first in KRISPER_TO_ACTION:
            krisper_found.append((first, rest))
            continue

        # Check for KRISPER function syntax: func(args)
        m = re.match(r"(\w+)\s*\((.*)\)\s*$", line)
        if m and m.group(1).lower() in KRISPER_TO_ACTION:
            krisper_found.append((m.group(1).lower(), m.group(2).strip()))
            continue

    # Determine primary action and compiler target
    if verbs_found:
        primary_verb, primary_rest = verbs_found[0]
        action_type = VERB_TO_ACTION[primary_verb]
        compiler_target = CompilerTarget.HELIX_CAPSULE
        intent = f"{primary_verb} {primary_rest}".strip() if primary_rest else f"{primary_verb} operation"
        target = primary_rest or "default"
        arguments = {
            "verbs": [v[0] for v in verbs_found],
            "targets": [v[1] for v in verbs_found if v[1]],
            "n_statements": len(verbs_found),
        }
        constraints = _DEFAULT_CONSTRAINTS.get(action_type.value, [])

        return _make_ir(
            intent=intent,
            target=target,
            action_type=action_type.value,
            arguments=arguments,
            constraints=constraints,
            verification_required=True,
            execution_allowed=False,
            source_text=source_text,
            compiler_target=compiler_target.value,
            parse_status="ok",
            parse_issues=[],
        )

    if krisper_found:
        primary_func, primary_args = krisper_found[0]
        action_type = KRISPER_TO_ACTION[primary_func]
        compiler_target = CompilerTarget.KRISPER_ENGINE
        intent = f"KRISPER {primary_func} {primary_args}".strip() if primary_args else f"KRISPER {primary_func} operation"
        target = primary_args or "population"
        arguments = {
            "functions": [k[0] for k in krisper_found],
            "params": [k[1] for k in krisper_found if k[1]],
            "n_statements": len(krisper_found),
        }
        constraints = _DEFAULT_CONSTRAINTS.get(action_type.value, [])

        return _make_ir(
            intent=intent,
            target=target,
            action_type=action_type.value,
            arguments=arguments,
            constraints=constraints,
            verification_required=True,
            execution_allowed=False,
            source_text=source_text,
            compiler_target=compiler_target.value,
            parse_status="ok",
            parse_issues=[],
        )

    # Ambiguous — no recognized verbs or functions
    return _make_ir(
        intent="ambiguous",
        target="",
        action_type=ActionType.QUERY.value,
        arguments={"raw_lines": lines},
        constraints=[],
        verification_required=True,
        execution_allowed=False,
        source_text=source_text,
        compiler_target=CompilerTarget.NONE.value,
        parse_status="ambiguous",
        parse_issues=["no recognized Bio-Poetica verbs or KRISPER functions"],
    )


# ---------------------------------------------------------------------------
# Validation: SymbolicIR dict → issues list
# ---------------------------------------------------------------------------

_REQUIRED_IR_FIELDS = (
    "ir_schema", "intent", "target", "action_type", "arguments",
    "constraints", "verification_required", "execution_allowed",
    "source_text", "compiler_target",
)


def validate_symbolic_ir(ir_dict: dict) -> List[str]:
    """Validate a SymbolicIR dict for structural and semantic correctness.

    Returns list of issues (empty = valid). Never raises.
    """
    issues: List[str] = []

    if not isinstance(ir_dict, dict):
        return ["IR is not a dict"]

    # Required fields
    for f in _REQUIRED_IR_FIELDS:
        if f not in ir_dict:
            issues.append(f"missing required field: {f}")

    if issues:
        return issues  # can't validate further without required fields

    # Schema check
    if ir_dict["ir_schema"] != SCHEMA_SYMBOLIC_IR:
        issues.append(
            f"schema mismatch: got {ir_dict['ir_schema']!r}, "
            f"expected {SCHEMA_SYMBOLIC_IR!r}"
        )

    # Action type check
    if ir_dict["action_type"] not in _ACTION_TYPE_VALUES:
        issues.append(
            f"unknown action_type: {ir_dict['action_type']!r}, "
            f"known: {sorted(_ACTION_TYPE_VALUES)}"
        )

    # Compiler target check
    if ir_dict["compiler_target"] not in _COMPILER_TARGET_VALUES:
        issues.append(
            f"unknown compiler_target: {ir_dict['compiler_target']!r}, "
            f"known: {sorted(_COMPILER_TARGET_VALUES)}"
        )

    # Target must not be empty for non-query actions
    if ir_dict["action_type"] != "query" and not ir_dict.get("target"):
        issues.append("target is empty for non-query action")

    # Arguments must be a dict
    if not isinstance(ir_dict.get("arguments"), dict):
        issues.append("arguments is not a dict")

    # Constraints must be a list of dicts with known types
    constraints = ir_dict.get("constraints")
    if not isinstance(constraints, list):
        issues.append("constraints is not a list")
    elif constraints:
        for i, c in enumerate(constraints):
            if not isinstance(c, dict):
                issues.append(f"constraints[{i}] is not a dict")
            elif "type" not in c:
                issues.append(f"constraints[{i}] missing 'type'")
            elif c["type"] not in KNOWN_CONSTRAINT_TYPES:
                issues.append(
                    f"constraints[{i}] unknown type: {c['type']!r}"
                )

    # verification_required must be True (strict IR)
    if ir_dict.get("verification_required") is not True:
        issues.append("verification_required must be True (strict IR)")

    # Parse status check
    parse_status = ir_dict.get("parse_status")
    if parse_status == "error":
        parse_issues = ir_dict.get("parse_issues", [])
        for pi in parse_issues:
            issues.append(f"parse error: {pi}")
    elif parse_status == "ambiguous":
        issues.append("parse is ambiguous — execution not allowed")

    # Source text must be present
    if not ir_dict.get("source_text", "").strip():
        issues.append("source_text is empty")

    return issues
