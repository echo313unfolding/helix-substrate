"""
Executor Registry — WO-I

Registered executors behind the governed controller. Every execution action
must go through an explicitly registered executor that is policy-gated,
budget-aware, and receipt-emitting.

No freeform execution. Each executor:
    - Is registered explicitly by name
    - Declares side_effects and budget requirements
    - Is policy-gated before dispatch
    - Returns a receipt fragment with timing
    - Can be blocked without crash (fail-closed)

Built-in executors:
    lobe_inference  — LLM inference via lobe scheduler route
    web_search      — factual web search via WebFactTool
    query_memory    — echo memory retrieval
    attest          — SHA256 attestation of content/plan
    route_only      — dry-run planning (no execution)

Work Order: WO-I (Real Executor Wiring)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


SCHEMA = "executor_registry:v1"


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class ExecutorStatus(Enum):
    OK = "ok"
    ERROR = "error"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExecutorContext:
    """Context passed to each executor. Dependencies injected by registry."""
    query: str
    route_name: str
    memory_context: str = ""
    max_tokens: Optional[int] = None
    action_plan: Optional[dict] = None
    is_symbolic: bool = False
    cycle_id: str = ""
    session_id: str = ""
    # Injected dependencies — set before dispatch
    lobe_scheduler: Any = None
    memory_module: Any = None
    web_tool: Any = None
    model_manager: Any = None


@dataclass
class ExecutorResult:
    """Result from a single executor invocation."""
    executor_name: str
    status: str          # ExecutorStatus value
    result: str          # output text/data
    timing_ms: float
    side_effects: bool   # did this executor cause side effects?
    receipt_fragment: dict
    error: Optional[str] = None

    def to_receipt(self) -> dict:
        return {
            "executor": self.executor_name,
            "status": self.status,
            "timing_ms": self.timing_ms,
            "side_effects": self.side_effects,
            "error": self.error,
            **self.receipt_fragment,
        }


@dataclass(frozen=True)
class ExecutorSpec:
    """Specification for a registered executor."""
    name: str
    handler: Callable[[ExecutorContext], ExecutorResult]
    has_side_effects: bool = False
    requires_budget: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ExecutorRegistry:
    """
    Explicit registry of allowed executors.

    No freeform execution — only registered executors can run.
    Each call is gated (registration, side-effects, budget) and receipted.
    """

    def __init__(self):
        self._executors: Dict[str, ExecutorSpec] = {}
        self._trace: List[dict] = []

    def register(self, spec: ExecutorSpec) -> None:
        """Register an executor. Raises if name already taken."""
        if spec.name in self._executors:
            raise ValueError(f"Executor already registered: {spec.name}")
        self._executors[spec.name] = spec

    def list_executors(self) -> List[dict]:
        """List all registered executors with metadata."""
        return [
            {
                "name": s.name,
                "has_side_effects": s.has_side_effects,
                "requires_budget": s.requires_budget,
                "description": s.description,
            }
            for s in self._executors.values()
        ]

    def is_registered(self, name: str) -> bool:
        return name in self._executors

    def execute(
        self,
        executor_name: str,
        context: ExecutorContext,
        *,
        allow_side_effects: bool = True,
        budget_ok: bool = True,
    ) -> ExecutorResult:
        """Execute a registered executor with policy gating.

        Gates (checked in order, fail-closed):
            1. Registration — executor must be explicitly registered
            2. Side effects — blocked if executor has side effects and not allowed
            3. Budget — blocked if executor requires budget and budget denied

        Args:
            executor_name: Name of the registered executor.
            context: Execution context with query, deps, etc.
            allow_side_effects: If False, block executors that have side effects.
            budget_ok: If False, block executors that require budget.

        Returns:
            ExecutorResult with receipt fragment. Never raises.
        """
        # Gate 1: Registration (fail closed)
        if executor_name not in self._executors:
            result = ExecutorResult(
                executor_name=executor_name,
                status=ExecutorStatus.BLOCKED.value,
                result="",
                timing_ms=0.0,
                side_effects=False,
                receipt_fragment={"gate": "unregistered"},
                error=f"Executor not registered: {executor_name}",
            )
            self._trace.append(result.to_receipt())
            return result

        spec = self._executors[executor_name]

        # Gate 2: Side effects
        if spec.has_side_effects and not allow_side_effects:
            result = ExecutorResult(
                executor_name=executor_name,
                status=ExecutorStatus.BLOCKED.value,
                result="",
                timing_ms=0.0,
                side_effects=False,
                receipt_fragment={"gate": "side_effects_blocked"},
                error="Side effects not allowed by policy",
            )
            self._trace.append(result.to_receipt())
            return result

        # Gate 3: Budget
        if spec.requires_budget and not budget_ok:
            result = ExecutorResult(
                executor_name=executor_name,
                status=ExecutorStatus.BLOCKED.value,
                result="",
                timing_ms=0.0,
                side_effects=False,
                receipt_fragment={"gate": "budget_denied"},
                error="Budget denied for this executor",
            )
            self._trace.append(result.to_receipt())
            return result

        # All gates passed — execute with timing
        t0 = time.time()
        try:
            result = spec.handler(context)
            result.timing_ms = round((time.time() - t0) * 1000, 3)
        except Exception as e:
            result = ExecutorResult(
                executor_name=executor_name,
                status=ExecutorStatus.ERROR.value,
                result="",
                timing_ms=round((time.time() - t0) * 1000, 3),
                side_effects=False,
                receipt_fragment={"gate": "passed", "error_type": type(e).__name__},
                error=str(e),
            )

        self._trace.append(result.to_receipt())
        return result

    def drain_trace(self) -> List[dict]:
        """Drain and return the execution trace. Clears for next cycle."""
        trace = list(self._trace)
        self._trace.clear()
        return trace

    def to_receipt(self) -> dict:
        """Full registry receipt with current trace."""
        return {
            "schema": SCHEMA,
            "registered_executors": [s.name for s in self._executors.values()],
            "trace": list(self._trace),
            "n_executions": len(self._trace),
        }


# ---------------------------------------------------------------------------
# Built-in executor handlers
# ---------------------------------------------------------------------------

def _exec_lobe_inference(ctx: ExecutorContext) -> ExecutorResult:
    """Primary LLM inference via lobe scheduler."""
    if ctx.lobe_scheduler is None:
        return ExecutorResult(
            executor_name="lobe_inference",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed"},
            error="No lobe_scheduler in context",
        )

    task_result = ctx.lobe_scheduler.execute(
        ctx.query,
        max_tokens=ctx.max_tokens,
        route_override=ctx.route_name,
        memory_context=ctx.memory_context,
    )

    lobe = task_result.steps[-1].lobe_name if task_result.steps else None
    model = task_result.steps[-1].model_used if task_result.steps else None
    tokens = sum(s.tokens_generated for s in task_result.steps)

    return ExecutorResult(
        executor_name="lobe_inference",
        status=ExecutorStatus.OK.value,
        result=task_result.final_output,
        timing_ms=0.0,  # filled by registry wrapper
        side_effects=False,
        receipt_fragment={
            "gate": "passed",
            "lobe": lobe,
            "model": model,
            "tokens_generated": tokens,
            "n_steps": len(task_result.steps),
            "route_receipt": task_result.route_receipt,
        },
    )


def _exec_web_search(ctx: ExecutorContext) -> ExecutorResult:
    """Factual web search via WebFactTool."""
    if ctx.web_tool is None:
        return ExecutorResult(
            executor_name="web_search",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed"},
            error="No web_tool in context",
        )

    search_result = ctx.web_tool.search(ctx.query)
    best = search_result.best_fact_passthrough()
    result_text = best if best else search_result.as_context()

    return ExecutorResult(
        executor_name="web_search",
        status=ExecutorStatus.OK.value,
        result=result_text or "",
        timing_ms=search_result.total_latency_ms,
        side_effects=False,
        receipt_fragment={
            "gate": "passed",
            "n_facts": len(search_result.facts),
            "backend": search_result.backend,
            "latency_ms": search_result.total_latency_ms,
            "passthrough": best is not None,
        },
    )


def _exec_query_memory(ctx: ExecutorContext) -> ExecutorResult:
    """Echo memory retrieval."""
    if ctx.memory_module is None:
        return ExecutorResult(
            executor_name="query_memory",
            status=ExecutorStatus.SKIPPED.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed", "reason": "memory_unavailable"},
        )

    try:
        results = ctx.memory_module.search(ctx.query, top_k=3, hybrid=True)
        texts = []
        for r in results:
            t = r.get("text", "").strip()
            if t:
                texts.append(t[:200])

        return ExecutorResult(
            executor_name="query_memory",
            status=ExecutorStatus.OK.value,
            result="\n".join(texts) if texts else "",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={
                "gate": "passed",
                "n_results": len(results),
                "n_nonempty": len(texts),
            },
        )
    except Exception as e:
        return ExecutorResult(
            executor_name="query_memory",
            status=ExecutorStatus.ERROR.value,
            result="",
            timing_ms=0.0,
            side_effects=False,
            receipt_fragment={"gate": "passed"},
            error=str(e),
        )


def _exec_attest(ctx: ExecutorContext) -> ExecutorResult:
    """SHA256 attestation of content and plan."""
    content = ctx.query
    sha256 = hashlib.sha256(content.encode()).hexdigest()

    receipt_frag: dict = {
        "gate": "passed",
        "content_sha256": sha256,
        "content_length": len(content),
    }

    if ctx.action_plan:
        receipt_frag["action_plan_hash"] = ctx.action_plan.get(
            "action_plan_hash", "")

    if ctx.memory_context:
        receipt_frag["memory_sha256"] = hashlib.sha256(
            ctx.memory_context.encode()).hexdigest()

    return ExecutorResult(
        executor_name="attest",
        status=ExecutorStatus.OK.value,
        result=f"SHA256: {sha256}",
        timing_ms=0.0,
        side_effects=False,
        receipt_fragment=receipt_frag,
    )


def _exec_route_only(ctx: ExecutorContext) -> ExecutorResult:
    """Dry-run: return routing plan without execution."""
    plan_info: dict = {
        "route_name": ctx.route_name,
        "is_symbolic": ctx.is_symbolic,
        "memory_available": ctx.memory_context != "",
        "max_tokens": ctx.max_tokens,
    }
    if ctx.action_plan:
        plan_info["action_type"] = ctx.action_plan.get("action_type", "")
        plan_info["execution_mode"] = ctx.action_plan.get("execution_mode", "")
        plan_info["lobe_route"] = ctx.action_plan.get("lobe_route", "")

    return ExecutorResult(
        executor_name="route_only",
        status=ExecutorStatus.OK.value,
        result=json.dumps(plan_info, indent=2),
        timing_ms=0.0,
        side_effects=False,
        receipt_fragment={
            "gate": "passed",
            "plan": plan_info,
        },
    )


# ---------------------------------------------------------------------------
# Executor selection logic
# ---------------------------------------------------------------------------

# Maps symbolic ExecutionMode values to executor names
_MODE_TO_EXECUTOR = {
    "lobe_route": "lobe_inference",
    "tool_action": "web_search",     # default tool action is web search
    "output_only": "route_only",     # output-only = dry run
    "memory_write": "query_memory",  # memory ops
    # "symbolic_mutation", "blocked", "stub" → no executor
}


def select_executor(
    route_name: str,
    action_plan: Optional[dict] = None,
    is_symbolic: bool = False,
    mode: str = "full",
) -> Optional[str]:
    """Select the appropriate executor for a given route/plan.

    Returns executor name or None if no execution should happen.

    Selection logic:
        1. trace_only mode → route_only
        2. Symbolic with action_plan → map execution_mode to executor
        3. NL route → lobe_inference
    """
    if mode == "trace_only":
        return "route_only"

    if is_symbolic and action_plan:
        exec_mode = action_plan.get("execution_mode", "")

        # Blocked/stubbed → no execution
        if exec_mode in ("blocked", "stub"):
            return None

        return _MODE_TO_EXECUTOR.get(exec_mode)

    # NL route → lobe inference
    return "lobe_inference"


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def build_default_registry() -> ExecutorRegistry:
    """Build the default executor registry with all built-in executors."""
    registry = ExecutorRegistry()

    registry.register(ExecutorSpec(
        name="lobe_inference",
        handler=_exec_lobe_inference,
        has_side_effects=False,
        requires_budget=True,
        description="LLM inference via lobe scheduler route",
    ))

    registry.register(ExecutorSpec(
        name="web_search",
        handler=_exec_web_search,
        has_side_effects=False,
        requires_budget=False,
        description="Factual web search via DuckDuckGo",
    ))

    registry.register(ExecutorSpec(
        name="query_memory",
        handler=_exec_query_memory,
        has_side_effects=False,
        requires_budget=False,
        description="Echo memory retrieval",
    ))

    registry.register(ExecutorSpec(
        name="attest",
        handler=_exec_attest,
        has_side_effects=False,
        requires_budget=False,
        description="SHA256 attestation of content/plan",
    ))

    registry.register(ExecutorSpec(
        name="route_only",
        handler=_exec_route_only,
        has_side_effects=False,
        requires_budget=False,
        description="Dry-run planning without execution",
    ))

    return registry
