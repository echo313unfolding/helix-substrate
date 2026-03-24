"""
Symbolic Executor v1 — compiles validated SymbolicIR into explicit ActionPlans.

No freeform symbolic execution. All symbolic input must go through IR and
verifier first. The executor is a pure compiler: it produces action plans,
not side effects.

ActionPlan schema (action_plan:v1):
    plan_schema, ir_schema, ir_hash, action_type, execution_mode, target,
    steps, tool_calls, lobe_route, execution_allowed, fallback_reason,
    compile_result, source_symbolic_text, compiler_target, action_plan_hash

compile_symbolic_ir(ir_dict) → ActionPlan dict:
    - Only accepts IR with execution_allowed=True
    - Blocked IR produces explicit non-execution plan
    - Unknown/unsafe actions fail closed
    - Each ActionType maps to a specific execution mode and lobe route

validate_action_plan(plan_dict) → list of issues

Work Order: WO-SYMBOLIC-EXECUTOR-V1
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_ACTION_PLAN = "action_plan:v1"


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------

class ExecutionMode(Enum):
    LOBE_ROUTE = "lobe_route"               # delegate to lobe scheduler route
    TOOL_ACTION = "tool_action"             # direct tool invocation
    OUTPUT_ONLY = "output_only"             # no side effects, produce output
    MEMORY_WRITE = "memory_write"           # write to memory/state
    SYMBOLIC_MUTATION = "symbolic_mutation"  # KRISPER evolutionary operation
    BLOCKED = "blocked"                     # execution not allowed
    STUB = "stub"                           # not yet implemented


_EXECUTION_MODE_VALUES = {m.value for m in ExecutionMode}

# Compile result values
COMPILE_COMPILED = "compiled"
COMPILE_BLOCKED = "blocked"
COMPILE_ERROR = "error"
COMPILE_STUBBED = "stubbed"

_COMPILE_RESULT_VALUES = {COMPILE_COMPILED, COMPILE_BLOCKED, COMPILE_ERROR, COMPILE_STUBBED}


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

def _compute_hash(data: dict) -> str:
    """SHA256 of deterministic JSON representation."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# ActionPlan builder
# ---------------------------------------------------------------------------

def _make_plan(
    ir_dict: dict,
    *,
    execution_mode: str,
    steps: List[dict],
    tool_calls: List[dict],
    lobe_route: Optional[str],
    execution_allowed: bool,
    fallback_reason: Optional[str],
    compile_result: str,
) -> dict:
    """Build an ActionPlan dict with hash computation."""
    from helix_substrate.symbolic_ir import SCHEMA_SYMBOLIC_IR

    ir_hash = _compute_hash(ir_dict) if ir_dict else ""

    plan = {
        "plan_schema": SCHEMA_ACTION_PLAN,
        "ir_schema": ir_dict.get("ir_schema", SCHEMA_SYMBOLIC_IR),
        "ir_hash": ir_hash,
        "action_type": ir_dict.get("action_type", ""),
        "execution_mode": execution_mode,
        "target": ir_dict.get("target", ""),
        "steps": steps,
        "tool_calls": tool_calls,
        "lobe_route": lobe_route,
        "execution_allowed": execution_allowed,
        "fallback_reason": fallback_reason,
        "compile_result": compile_result,
        "source_symbolic_text": ir_dict.get("source_text", ""),
        "compiler_target": ir_dict.get("compiler_target", "none"),
    }

    # Compute action_plan_hash AFTER building all other fields
    plan["action_plan_hash"] = _compute_hash(plan)
    return plan


def _make_blocked_plan(ir_dict: dict, reason: str) -> dict:
    """Build a blocked/non-execution ActionPlan."""
    return _make_plan(
        ir_dict,
        execution_mode=ExecutionMode.BLOCKED.value,
        steps=[],
        tool_calls=[],
        lobe_route=None,
        execution_allowed=False,
        fallback_reason=reason,
        compile_result=COMPILE_BLOCKED,
    )


def _make_error_plan(ir_dict: dict, reason: str) -> dict:
    """Build an error ActionPlan."""
    return _make_plan(
        ir_dict,
        execution_mode=ExecutionMode.BLOCKED.value,
        steps=[],
        tool_calls=[],
        lobe_route=None,
        execution_allowed=False,
        fallback_reason=reason,
        compile_result=COMPILE_ERROR,
    )


# ---------------------------------------------------------------------------
# Action-type compilers
# ---------------------------------------------------------------------------

def _compile_create(ir: dict) -> dict:
    """create → plan + code + verify via lobe scheduler."""
    target = ir.get("target", "")
    return _make_plan(
        ir,
        execution_mode=ExecutionMode.LOBE_ROUTE.value,
        steps=[
            {"step": "plan", "action": "decompose",
             "description": f"Break down creation of: {target}"},
            {"step": "generate", "action": "code",
             "description": "Generate code or content"},
            {"step": "verify", "action": "review",
             "description": "Verify output correctness"},
        ],
        tool_calls=[],
        lobe_route="plan_code_verify",
        execution_allowed=True,
        fallback_reason=None,
        compile_result=COMPILE_COMPILED,
    )


def _compile_transform(ir: dict) -> dict:
    """transform → code + verify via lobe scheduler."""
    target = ir.get("target", "")
    return _make_plan(
        ir,
        execution_mode=ExecutionMode.LOBE_ROUTE.value,
        steps=[
            {"step": "analyze", "action": "identify_source",
             "description": f"Identify transform target: {target}"},
            {"step": "generate", "action": "transform",
             "description": "Generate transformation code"},
            {"step": "verify", "action": "review",
             "description": "Verify transform correctness"},
        ],
        tool_calls=[],
        lobe_route="plan_code_verify",
        execution_allowed=True,
        fallback_reason=None,
        compile_result=COMPILE_COMPILED,
    )


def _compile_query(ir: dict) -> dict:
    """query → planner route (reasoning/observation)."""
    target = ir.get("target", "")
    return _make_plan(
        ir,
        execution_mode=ExecutionMode.LOBE_ROUTE.value,
        steps=[
            {"step": "query", "action": "reason",
             "description": f"Process query: {target}"},
        ],
        tool_calls=[],
        lobe_route="direct_plan",
        execution_allowed=True,
        fallback_reason=None,
        compile_result=COMPILE_COMPILED,
    )


def _compile_structure(ir: dict) -> dict:
    """structure → plan + code + verify (organizational operations)."""
    target = ir.get("target", "")
    return _make_plan(
        ir,
        execution_mode=ExecutionMode.LOBE_ROUTE.value,
        steps=[
            {"step": "analyze", "action": "map_structure",
             "description": f"Analyze structure of: {target}"},
            {"step": "plan", "action": "reorganize",
             "description": "Plan structural changes"},
            {"step": "generate", "action": "apply",
             "description": "Apply structural changes"},
        ],
        tool_calls=[],
        lobe_route="plan_code_verify",
        execution_allowed=True,
        fallback_reason=None,
        compile_result=COMPILE_COMPILED,
    )


def _compile_flow(ir: dict) -> dict:
    """flow → planner route (control/data flow orchestration)."""
    target = ir.get("target", "")
    return _make_plan(
        ir,
        execution_mode=ExecutionMode.LOBE_ROUTE.value,
        steps=[
            {"step": "map", "action": "trace_flow",
             "description": f"Map flow for: {target}"},
            {"step": "configure", "action": "route",
             "description": "Configure routing/orchestration"},
        ],
        tool_calls=[],
        lobe_route="direct_plan",
        execution_allowed=True,
        fallback_reason=None,
        compile_result=COMPILE_COMPILED,
    )


def _compile_emit(ir: dict) -> dict:
    """emit → output-only path (no side effects)."""
    target = ir.get("target", "")
    return _make_plan(
        ir,
        execution_mode=ExecutionMode.OUTPUT_ONLY.value,
        steps=[
            {"step": "format", "action": "emit",
             "description": f"Format and emit: {target}"},
        ],
        tool_calls=[],
        lobe_route=None,
        execution_allowed=True,
        fallback_reason=None,
        compile_result=COMPILE_COMPILED,
    )


def _compile_remember(ir: dict) -> dict:
    """remember → memory write plan."""
    target = ir.get("target", "")
    return _make_plan(
        ir,
        execution_mode=ExecutionMode.MEMORY_WRITE.value,
        steps=[
            {"step": "identify", "action": "parse_key",
             "description": f"Identify memory key from: {target}"},
            {"step": "serialize", "action": "encode_value",
             "description": "Serialize value for storage"},
            {"step": "write", "action": "store",
             "description": "Write to memory"},
        ],
        tool_calls=[],
        lobe_route=None,
        execution_allowed=True,
        fallback_reason=None,
        compile_result=COMPILE_COMPILED,
    )


def _compile_evolve(ir: dict) -> dict:
    """evolve → symbolic mutation plan (stubbed — KRISPER runtime not implemented)."""
    return _make_plan(
        ir,
        execution_mode=ExecutionMode.STUB.value,
        steps=[],
        tool_calls=[],
        lobe_route=None,
        execution_allowed=False,
        fallback_reason="KRISPER evolutionary runtime not yet implemented",
        compile_result=COMPILE_STUBBED,
    )


# Dispatch table
_ACTION_COMPILERS: Dict[str, callable] = {
    "create": _compile_create,
    "transform": _compile_transform,
    "query": _compile_query,
    "structure": _compile_structure,
    "flow": _compile_flow,
    "emit": _compile_emit,
    "remember": _compile_remember,
    "evolve": _compile_evolve,
}


# ---------------------------------------------------------------------------
# Main compiler entry point
# ---------------------------------------------------------------------------

def compile_symbolic_ir(ir_dict: dict) -> dict:
    """Compile validated SymbolicIR into an ActionPlan.

    Rules:
        - Only accepts IR with execution_allowed=True
        - Blocked IR produces explicit non-execution plan
        - Unknown action types produce error plan (fail closed)
        - evolve actions are stubbed (KRISPER runtime not implemented)

    Args:
        ir_dict: SymbolicIR dict (from parse_symbolic + verifier approval)

    Returns:
        ActionPlan dict (always valid schema, even on failure)
    """
    if not isinstance(ir_dict, dict):
        return _make_error_plan({}, "IR is not a dict")

    # Execution gate: only verified IR proceeds
    if not ir_dict.get("execution_allowed"):
        reason = "IR execution not allowed by verifier"
        parse_issues = ir_dict.get("parse_issues", [])
        if parse_issues:
            reason += f": {'; '.join(parse_issues)}"
        return _make_blocked_plan(ir_dict, reason)

    action_type = ir_dict.get("action_type", "")

    # Unknown action type: fail closed
    compiler = _ACTION_COMPILERS.get(action_type)
    if compiler is None:
        return _make_error_plan(
            ir_dict, f"unknown action_type: {action_type!r}"
        )

    return compiler(ir_dict)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_PLAN_FIELDS = (
    "plan_schema", "ir_schema", "ir_hash", "action_type", "execution_mode",
    "target", "steps", "tool_calls", "lobe_route", "execution_allowed",
    "fallback_reason", "compile_result", "source_symbolic_text",
    "compiler_target", "action_plan_hash",
)


def validate_action_plan(plan: dict) -> List[str]:
    """Validate an ActionPlan dict for structural and semantic correctness.

    Returns list of issues (empty = valid). Never raises.
    """
    issues: List[str] = []

    if not isinstance(plan, dict):
        return ["plan is not a dict"]

    # Required fields
    for f in _REQUIRED_PLAN_FIELDS:
        if f not in plan:
            issues.append(f"missing required field: {f}")

    if issues:
        return issues

    # Schema check
    if plan["plan_schema"] != SCHEMA_ACTION_PLAN:
        issues.append(
            f"schema mismatch: got {plan['plan_schema']!r}, "
            f"expected {SCHEMA_ACTION_PLAN!r}"
        )

    # Execution mode check
    if plan["execution_mode"] not in _EXECUTION_MODE_VALUES:
        issues.append(f"unknown execution_mode: {plan['execution_mode']!r}")

    # Compile result check
    if plan["compile_result"] not in _COMPILE_RESULT_VALUES:
        issues.append(f"unknown compile_result: {plan['compile_result']!r}")

    # Steps must be a list
    if not isinstance(plan.get("steps"), list):
        issues.append("steps is not a list")

    # Tool calls must be a list
    if not isinstance(plan.get("tool_calls"), list):
        issues.append("tool_calls is not a list")

    # Semantic: blocked but execution_allowed=True is inconsistent
    if plan["execution_mode"] == "blocked" and plan["execution_allowed"] is True:
        issues.append(
            "semantic: execution_mode is 'blocked' but execution_allowed is True"
        )

    # Semantic: not allowed but no reason given
    if not plan["execution_allowed"] and not plan.get("fallback_reason"):
        issues.append(
            "semantic: execution not allowed but no fallback_reason"
        )

    # Semantic: lobe_route should be set for lobe_route execution mode
    if plan["execution_mode"] == "lobe_route" and not plan.get("lobe_route"):
        issues.append(
            "semantic: execution_mode is 'lobe_route' but lobe_route is empty"
        )

    # ir_hash should be non-empty
    if not plan.get("ir_hash"):
        issues.append("ir_hash is empty")

    # action_plan_hash should be non-empty
    if not plan.get("action_plan_hash"):
        issues.append("action_plan_hash is empty")

    return issues
