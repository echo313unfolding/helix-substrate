"""
Lobe Scheduler v1 — multi-role orchestration for HelixLinear inference.

Routes a task across named lobes (planner, coder, verifier, memory) instead
of assuming one model per request. Each lobe is a role backed by a model +
prompt template. The scheduler classifies the request, builds a route
(ordered lobe sequence), executes each step, and produces a route receipt.

v1 changes (WO-VERIFIER-LOBE-V1):
    - Rule-based verification: code_sanity, shell_safety, receipt_consistency,
      symbolic_ir (stub). Each returns structured VerificationResult.
    - Policy upgrade: code routes default to coder→verifier. High-risk
      non-code escalates to plan→verify.
    - Verification receipt fields on verify steps: verification_target,
      verification_result, issues_found, confidence, action_allowed.
    - StrictVerificationMode blocks pipeline on verification failure.
    - Symbolic IR / parser routes deferred — no parser infrastructure yet.

Sits ON TOP of ModelManager + basin_runtime. No kernel changes.

Core objects:
    Capability  — what a lobe can do (model, strengths, prompt template)
    Lobe        — a named role (planner, coder, verifier, memory)
    Route       — ordered list of lobe invocations for a task
    StepResult  — output + timing + dispatch info from one lobe
    TaskResult  — final output + full route receipt
    Policy      — rules for choosing routes

Usage:
    from helix_substrate.lobe_scheduler import LobeScheduler
    from helix_substrate.model_manager import ModelManager

    mgr = ModelManager(device="cuda:0")
    scheduler = LobeScheduler(mgr)
    result = scheduler.execute("Write a function to validate IPv4 addresses")
    print(result.final_output)
    print(json.dumps(result.route_receipt, indent=2))

Work Orders: WO-LOBE-SCHEDULER-V0, WO-VERIFIER-LOBE-V1
"""

from __future__ import annotations

import hashlib
import platform
import resource
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch

from helix_substrate.query_classifier import ModelTarget, classify

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_ROUTE_RECEIPT = "lobe_route_receipt:v1"
SCHEMA_VERIFICATION_RECEIPT = "lobe_verification:v1"


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Capability:
    """What a lobe can do."""
    model_target: Optional[ModelTarget]  # None for non-model lobes (memory)
    strengths: Tuple[str, ...]
    max_tokens: int
    prompt_template: str  # must contain {task} placeholder


# ---------------------------------------------------------------------------
# Lobe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lobe:
    """A named role backed by a model + prompt template."""
    name: str
    capability: Capability

    def format_prompt(self, task: str, context: str = "") -> str:
        """Build the prompt for this lobe."""
        return self.capability.prompt_template.format(
            task=task, context=context,
        )


# ---------------------------------------------------------------------------
# Lobe Registry
# ---------------------------------------------------------------------------

LOBE_REGISTRY: Dict[str, Lobe] = {
    "planner": Lobe(
        name="planner",
        capability=Capability(
            model_target=ModelTarget.TINYLLAMA,
            strengths=("planning", "decomposition", "reasoning"),
            max_tokens=256,
            prompt_template=(
                "You are a task planner. Break down the following request into "
                "clear, actionable steps. Be concise.\n\n"
                "Request: {task}\n\n"
                "Steps:"
            ),
        ),
    ),
    "coder": Lobe(
        name="coder",
        capability=Capability(
            model_target=ModelTarget.QWEN_CODER,
            strengths=("code_generation", "bug_fixing", "structured_output"),
            max_tokens=256,
            prompt_template=(
                "<|im_start|>user\n{context}{task}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        ),
    ),
    "verifier": Lobe(
        name="verifier",
        capability=Capability(
            model_target=ModelTarget.TINYLLAMA,
            strengths=("verification", "review", "error_detection"),
            max_tokens=128,
            prompt_template=(
                "Review the following code for correctness. "
                "List any bugs or issues. If correct, say 'PASS'.\n\n"
                "Code:\n{context}\n\n"
                "Original request: {task}\n\n"
                "Review:"
            ),
        ),
    ),
    "memory": Lobe(
        name="memory",
        capability=Capability(
            model_target=None,  # no model — local retrieval only
            strengths=("retrieval", "context_lookup"),
            max_tokens=0,
            prompt_template="",  # not used
        ),
    ),
    "parser": Lobe(
        name="parser",
        capability=Capability(
            model_target=None,  # no model — rule-based parsing
            strengths=("symbolic_parsing", "ir_generation"),
            max_tokens=0,
            prompt_template="",  # not used — parser calls parse_symbolic()
        ),
    ),
    "compiler": Lobe(
        name="compiler",
        capability=Capability(
            model_target=None,  # no model — rule-based IR compilation
            strengths=("symbolic_compilation", "action_planning"),
            max_tokens=0,
            prompt_template="",  # not used — compiler calls compile_symbolic_ir()
        ),
    ),
}


def get_lobe(name: str) -> Lobe:
    """Get a lobe by name. Raises KeyError if not found."""
    return LOBE_REGISTRY[name]


def list_lobes() -> List[str]:
    """Return all registered lobe names."""
    return list(LOBE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteStep:
    """One step in a route."""
    lobe_name: str
    input_from: Optional[str]  # previous step's lobe_name, or None for first step
    role: str  # what this step does: "plan", "generate", "verify", "retrieve"
    verification_type: Optional[str] = None  # VerificationType.value for verify steps


@dataclass(frozen=True)
class Route:
    """An ordered sequence of lobe invocations."""
    name: str
    steps: Tuple[RouteStep, ...]
    reason: str
    verification_required: bool

    @property
    def selected_lobes(self) -> List[str]:
        return [s.lobe_name for s in self.steps]


# ---------------------------------------------------------------------------
# Pre-defined routes
# ---------------------------------------------------------------------------

ROUTE_DIRECT_CODE = Route(
    name="direct_code",
    steps=(
        RouteStep(lobe_name="coder", input_from=None, role="generate"),
    ),
    reason="Simple code request — direct to coder",
    verification_required=False,
)

ROUTE_DIRECT_PLAN = Route(
    name="direct_plan",
    steps=(
        RouteStep(lobe_name="planner", input_from=None, role="plan"),
    ),
    reason="Planning/reasoning request — direct to planner",
    verification_required=False,
)

ROUTE_PLAN_CODE_VERIFY = Route(
    name="plan_code_verify",
    steps=(
        RouteStep(lobe_name="planner", input_from=None, role="plan"),
        RouteStep(lobe_name="coder", input_from="planner", role="generate"),
        RouteStep(lobe_name="verifier", input_from="coder", role="verify",
                  verification_type="code_sanity"),
    ),
    reason="Complex code request — plan, generate, then verify",
    verification_required=True,
)

ROUTE_CODE_VERIFY = Route(
    name="code_verify",
    steps=(
        RouteStep(lobe_name="coder", input_from=None, role="generate"),
        RouteStep(lobe_name="verifier", input_from="coder", role="verify",
                  verification_type="code_sanity"),
    ),
    reason="Code request with verification — generate then verify",
    verification_required=True,
)

ROUTE_PLAN_VERIFY = Route(
    name="plan_verify",
    steps=(
        RouteStep(lobe_name="planner", input_from=None, role="plan"),
        RouteStep(lobe_name="verifier", input_from="planner", role="verify",
                  verification_type="code_sanity"),
    ),
    reason="High-risk request — plan then verify before action",
    verification_required=True,
)

ROUTE_SYMBOLIC_PARSE_VERIFY = Route(
    name="symbolic_parse_verify",
    steps=(
        RouteStep(lobe_name="parser", input_from=None, role="parse"),
        RouteStep(lobe_name="verifier", input_from="parser", role="verify",
                  verification_type="symbolic_ir"),
    ),
    reason="Symbolic input — parse to strict IR then verify before execution",
    verification_required=True,
)

ROUTE_SYMBOLIC_FULL = Route(
    name="symbolic_full",
    steps=(
        RouteStep(lobe_name="parser", input_from=None, role="parse"),
        RouteStep(lobe_name="verifier", input_from="parser", role="verify",
                  verification_type="symbolic_ir"),
        RouteStep(lobe_name="compiler", input_from="parser", role="compile"),
    ),
    reason="Symbolic input — parse, verify, compile to action plan",
    verification_required=True,
)

_ALL_ROUTES = {
    "direct_code": ROUTE_DIRECT_CODE,
    "direct_plan": ROUTE_DIRECT_PLAN,
    "plan_code_verify": ROUTE_PLAN_CODE_VERIFY,
    "code_verify": ROUTE_CODE_VERIFY,
    "plan_verify": ROUTE_PLAN_VERIFY,
    "symbolic_parse_verify": ROUTE_SYMBOLIC_PARSE_VERIFY,
    "symbolic_full": ROUTE_SYMBOLIC_FULL,
}


# ---------------------------------------------------------------------------
# Policy — route selection rules
# ---------------------------------------------------------------------------

# Complexity indicators that upgrade from direct to multi-step
_COMPLEXITY_KEYWORDS = (
    "implement", "build", "design", "create a class", "refactor",
    "multi-step", "pipeline", "architecture", "system",
)

# Verification indicators (kept for reference — all code routes now verify by default)
_VERIFY_KEYWORDS = (
    "correct", "verify", "check", "review", "test", "validate", "ensure",
)

# Escalation indicators — non-code queries with these go to plan_verify
_ESCALATION_KEYWORDS = (
    "delete", "remove", "drop", "destroy", "production", "deploy",
    "migrate", "dangerous", "critical", "irreversible",
)


class Policy:
    """Rules for choosing routes based on query classification.

    v1 policy (WO-VERIFIER-LOBE-V1):
        - Code routes default to coder→verifier (code_verify).
        - Complex code still gets full plan→code→verify.
        - Non-code with high-risk keywords escalates to plan→verify.
        - Symbolic/parser routes deferred (no parser infrastructure yet).

    Wire 3 (WO-HOMEOSTATIC-WIRE-01):
        - When se_override is provided, Se value biases route selection
          toward more cautious routes as system entropy increases.
    """

    @staticmethod
    def select_route(query: str, se_override: Optional[float] = None) -> Route:
        """Classify query and select the best route.

        Decision tree (v1 + symbolic IR + Wire 3):
            0. If se_override >= 0.75 → plan_verify (system critical)
            1. If symbolic input (Bio-Poetica/KRISPER) → symbolic_parse_verify
            2. classify() → code vs non-code
            3. If se_override >= 0.50 → upgrade to verified route
            4. Standard keyword-based routing
            5. If code → code_verify  (v1: all code routes verify by default)
        """
        from helix_substrate.symbolic_ir import is_symbolic_input

        # Wire 3: Se-based route escalation.
        # High Se overrides normal classification — system is stressed,
        # force maximum caution regardless of query content.
        if se_override is not None and se_override >= 0.75:
            return ROUTE_PLAN_VERIFY

        # Symbolic input gets strict IR pipeline — no freeform execution
        if is_symbolic_input(query):
            return ROUTE_SYMBOLIC_FULL

        target, confidence, debug = classify(query)
        is_code = target == ModelTarget.QWEN_CODER
        query_lower = query.lower()

        if not is_code:
            has_escalation = any(kw in query_lower for kw in _ESCALATION_KEYWORDS)
            if has_escalation:
                return ROUTE_PLAN_VERIFY
            # Wire 3: mid-Se escalates non-code to plan_verify
            if se_override is not None and se_override >= 0.50:
                return ROUTE_PLAN_VERIFY
            return ROUTE_DIRECT_PLAN

        has_complexity = any(kw in query_lower for kw in _COMPLEXITY_KEYWORDS)

        # Wire 3: mid-Se upgrades simple code to full pipeline
        if se_override is not None and se_override >= 0.50:
            return ROUTE_PLAN_CODE_VERIFY

        if has_complexity:
            return ROUTE_PLAN_CODE_VERIFY
        return ROUTE_CODE_VERIFY


# ---------------------------------------------------------------------------
# Step / Task results
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result from executing one lobe step."""
    lobe_name: str
    role: str
    output: str
    timing_ms: float
    tokens_generated: int
    model_used: Optional[str]  # ModelTarget.value or None
    dispatch_path: Optional[str]  # "fused", "naive", or None
    input_tokens: int = 0  # WO-F: actual input tokens fed to model
    verification_result: Optional[dict] = None  # VerificationResult.to_receipt() for verify steps


@dataclass
class TaskResult:
    """Complete result from executing a route."""
    route: Route
    steps: List[StepResult]
    final_output: str
    route_receipt: dict


# ---------------------------------------------------------------------------
# Lobe Scheduler
# ---------------------------------------------------------------------------

class LobeScheduler:
    """Multi-role orchestration layer.

    Sits on top of ModelManager (model loading/swapping) and basin_runtime
    (generation + dispatch receipts). No kernel changes.
    """

    def __init__(self, model_manager, morphsat_enabled: bool = False):
        """
        Args:
            model_manager: helix_substrate.model_manager.ModelManager instance.
            morphsat_enabled: When True, a MorphSATGate enforces task-state
                transitions between lobe steps. Illegal transitions abort the
                pipeline.  (WO-ECHO-MORPHSAT-INTEGRATION-01)
        """
        self.mgr = model_manager
        self.morphsat_enabled = morphsat_enabled
        self._execution_count = 0

    def execute(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        route_override: Optional[str] = None,
        memory_context: str = "",
        se_override: Optional[float] = None,
    ) -> TaskResult:
        """Route and execute a query through the lobe pipeline.

        Args:
            query: User request text
            max_tokens: Override max tokens per step (uses lobe default if None)
            route_override: Force a specific route name (bypasses Policy)
            memory_context: Recalled memory text to inject into first model step (WO-F)
            se_override: Wire 3 — current system Se from HomeostasisController.
                         When provided, biases route selection toward more cautious
                         routes as entropy increases.  (WO-HOMEOSTATIC-WIRE-01)

        Returns:
            TaskResult with final output and route receipt.
        """
        t_start = time.time()
        cpu_start = time.process_time()
        start_iso = datetime.now(timezone.utc).isoformat()

        # Select route (Wire 3: Se biases selection when provided)
        if route_override and route_override in _ALL_ROUTES:
            route = _ALL_ROUTES[route_override]
        else:
            route = Policy.select_route(query, se_override=se_override)

        # MorphSAT task-state enforcement gate (WO-ECHO-MORPHSAT-INTEGRATION-01)
        morphsat_gate = None
        if self.morphsat_enabled:
            from helix_substrate.morphsat_gate import (
                MorphSATGate, TaskEvent, classify_event,
            )
            morphsat_gate = MorphSATGate()
            # Fire NEW_TASK to transition IDLE → PLANNING
            morphsat_gate.step(TaskEvent.NEW_TASK)

        # Execute each step
        steps: List[StepResult] = []
        step_outputs: Dict[str, str] = {}  # lobe_name → output text
        memory_injected = False  # WO-F: inject memory into first model step only
        morphsat_blocked = False  # set True if gate blocks a transition

        for route_step in route.steps:
            lobe = get_lobe(route_step.lobe_name)

            # Non-model lobes: parser, compiler (rule-based) and memory (stub)
            if lobe.capability.model_target is None:
                if lobe.name == "parser":
                    result = self._execute_parser_step(query, route_step)
                elif lobe.name == "compiler":
                    result = self._execute_compiler_step(
                        query, steps, step_outputs, route_step,
                    )
                else:
                    # Memory stub
                    result = StepResult(
                        lobe_name=lobe.name,
                        role=route_step.role,
                        output="",
                        timing_ms=0.0,
                        tokens_generated=0,
                        model_used=None,
                        dispatch_path=None,
                    )
                steps.append(result)
                step_outputs[lobe.name] = result.output

                # MorphSAT gate check for non-model steps
                if morphsat_gate is not None and result.output:
                    event = classify_event(result.output, route_step.role)
                    _, was_legal, action = morphsat_gate.step(event)
                    if not was_legal:
                        morphsat_blocked = True
                        break

                continue

            # Build context from previous step
            context = ""
            if route_step.input_from and route_step.input_from in step_outputs:
                prev_output = step_outputs[route_step.input_from]
                if prev_output:
                    context = prev_output + "\n\n"

            # WO-F: Inject memory context into first model-bearing step
            if memory_context and not memory_injected:
                context = memory_context + "\n\n" + context
                memory_injected = True

            # Build prompt
            prompt = lobe.format_prompt(task=query, context=context)

            # Ensure model is loaded
            step_max = max_tokens if max_tokens else lobe.capability.max_tokens
            result = self._execute_step(lobe, route_step, prompt, step_max)

            # Rule-based verification for verify steps
            if route_step.role == "verify" and context:
                vr = self._run_rule_verification(
                    context, route_step.verification_type,
                )
                if vr is not None:
                    result.verification_result = vr.to_receipt()
                    from helix_substrate.verifier import StrictVerificationMode
                    StrictVerificationMode.enforce(vr)

            steps.append(result)
            step_outputs[lobe.name] = result.output

            # MorphSAT gate check for model steps
            if morphsat_gate is not None:
                event = classify_event(result.output, route_step.role)
                _, was_legal, action = morphsat_gate.step(event)
                if not was_legal:
                    morphsat_blocked = True
                    break

        # Final output is the last non-empty step
        final_output = ""
        if morphsat_blocked:
            # Gate blocked the pipeline — surface the block reason
            last_gate_entry = morphsat_gate.history[-1]
            final_output = (
                f"[MORPHSAT BLOCKED] {last_gate_entry['action']}: "
                f"{last_gate_entry['from']} + {last_gate_entry['event']} "
                f"is an illegal transition. Pipeline aborted."
            )
        else:
            for step in reversed(steps):
                if step.output:
                    final_output = step.output
                    break

        # Build route receipt
        receipt = self._build_route_receipt(
            query, route, steps, t_start, cpu_start, start_iso,
        )

        # Attach MorphSAT gate receipt if active
        if morphsat_gate is not None:
            receipt["morphsat"] = morphsat_gate.to_receipt()
            receipt["morphsat"]["blocked"] = morphsat_blocked

        self._execution_count += 1

        return TaskResult(
            route=route,
            steps=steps,
            final_output=final_output,
            route_receipt=receipt,
        )

    @staticmethod
    def _execute_parser_step(query: str, route_step: RouteStep) -> StepResult:
        """Execute the parser lobe: source text → SymbolicIR JSON."""
        import json as _json
        from helix_substrate.symbolic_ir import parse_symbolic

        t0 = time.perf_counter()
        ir = parse_symbolic(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return StepResult(
            lobe_name="parser",
            role=route_step.role,
            output=_json.dumps(ir),
            timing_ms=round(elapsed_ms, 3),
            tokens_generated=0,
            model_used=None,
            dispatch_path=None,
        )

    @staticmethod
    def _execute_compiler_step(
        query: str,
        steps: List[StepResult],
        step_outputs: Dict[str, str],
        route_step: RouteStep,
    ) -> StepResult:
        """Execute the compiler lobe: IR + verification → ActionPlan JSON.

        Reads parser output for IR, checks verifier result for approval,
        then compiles to an ActionPlan.
        """
        import json as _json
        from helix_substrate.symbolic_executor import compile_symbolic_ir

        t0 = time.perf_counter()

        # Get IR from parser step output
        ir_json = step_outputs.get("parser", "{}")
        try:
            ir_dict = _json.loads(ir_json)
        except _json.JSONDecodeError:
            ir_dict = {}

        # Check verification result from verifier step
        verifier_step = next(
            (s for s in steps if s.role == "verify"), None,
        )
        if verifier_step and verifier_step.verification_result:
            vr = verifier_step.verification_result
            if vr.get("action_allowed"):
                ir_dict["execution_allowed"] = True

        # Compile IR to action plan
        action_plan = compile_symbolic_ir(ir_dict)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return StepResult(
            lobe_name="compiler",
            role=route_step.role,
            output=_json.dumps(action_plan),
            timing_ms=round(elapsed_ms, 3),
            tokens_generated=0,
            model_used=None,
            dispatch_path=None,
        )

    @staticmethod
    def _run_rule_verification(content: str, verification_type: Optional[str]):
        """Run rule-based verification on content from previous step.

        Returns VerificationResult or None.
        """
        from helix_substrate.verifier import run_verification
        vtype = verification_type or "code_sanity"
        return run_verification(content, vtype)

    def _execute_step(
        self,
        lobe: Lobe,
        route_step: RouteStep,
        prompt: str,
        max_tokens: int,
    ) -> StepResult:
        """Execute a single lobe step."""
        model, tokenizer = self.mgr.ensure_model(lobe.capability.model_target)

        t0 = time.perf_counter()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        generated = output_ids.shape[1] - input_ids.shape[1]
        text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

        # Get dispatch path from first HelixLinear module
        dispatch_path = None
        from helix_substrate.helix_linear import HelixLinear
        for _, mod in model.named_modules():
            if isinstance(mod, HelixLinear):
                dispatch_path = mod._last_dispatch_path
                break

        return StepResult(
            lobe_name=lobe.name,
            role=route_step.role,
            output=text,
            timing_ms=round(elapsed_ms, 1),
            tokens_generated=generated,
            model_used=lobe.capability.model_target.value,
            dispatch_path=dispatch_path,
            input_tokens=input_ids.shape[1],  # WO-F: actual input token count
        )

    def _build_route_receipt(
        self,
        query: str,
        route: Route,
        steps: List[StepResult],
        t_start: float,
        cpu_start: float,
        start_iso: str,
    ) -> dict:
        """Build a structured route receipt."""
        now = datetime.now(timezone.utc).isoformat()
        query_sha256 = hashlib.sha256(query.encode()).hexdigest()

        step_receipts = []
        for step in steps:
            sr = {
                "lobe": step.lobe_name,
                "role": step.role,
                "model_used": step.model_used,
                "dispatch_path": step.dispatch_path,
                "input_tokens": step.input_tokens,  # WO-F
                "tokens_generated": step.tokens_generated,
                "timing_ms": step.timing_ms,
            }
            if step.verification_result is not None:
                sr["verification"] = step.verification_result
            step_receipts.append(sr)

        total_tokens = sum(s.tokens_generated for s in steps)
        total_timing = sum(s.timing_ms for s in steps)
        models_used = sorted(set(
            s.model_used for s in steps if s.model_used is not None
        ))

        receipt = {
            "schema": SCHEMA_ROUTE_RECEIPT,
            "query_sha256": query_sha256,
            "query_length": len(query),
            "route": {
                "name": route.name,
                "selected_lobes": route.selected_lobes,
                "route_reason": route.reason,
                "verification_required": route.verification_required,
                "n_steps": len(route.steps),
            },
            "steps": step_receipts,
            "summary": {
                "total_tokens_generated": total_tokens,
                "total_timing_ms": round(total_timing, 1),
                "models_used": models_used,
                "n_model_swaps": max(0, len(models_used) - 1),
                "all_fused": all(
                    s.dispatch_path == "fused"
                    for s in steps if s.dispatch_path is not None
                ),
            },
            "cost": {
                "wall_time_s": round(time.time() - t_start, 3),
                "cpu_time_s": round(time.process_time() - cpu_start, 3),
                "peak_memory_mb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
                ),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
                "timestamp_start": start_iso,
                "timestamp_end": now,
            },
        }

        if torch.cuda.is_available():
            receipt["cost"]["gpu"] = torch.cuda.get_device_name(0)

        # Add symbolic execution fields when compiler step present
        compiler_step = next(
            (s for s in steps if s.role == "compile"), None,
        )
        if compiler_step and compiler_step.output:
            import json as _json
            try:
                plan = _json.loads(compiler_step.output)
                receipt["symbolic"] = {
                    "source_symbolic_text": plan.get("source_symbolic_text", ""),
                    "ir_hash": plan.get("ir_hash", ""),
                    "action_plan_hash": plan.get("action_plan_hash", ""),
                    "compile_result": plan.get("compile_result", ""),
                    "execution_mode": plan.get("execution_mode", ""),
                    "compiler_target": plan.get("compiler_target", ""),
                    "execution_allowed": plan.get("execution_allowed", False),
                    "blocked_reason": plan.get("fallback_reason"),
                }
            except (_json.JSONDecodeError, TypeError):
                pass

        return receipt

    def status(self) -> dict:
        """Return scheduler status."""
        return {
            "lobes": list_lobes(),
            "routes": list(_ALL_ROUTES.keys()),
            "execution_count": self._execution_count,
            "model_manager": self.mgr.status(),
        }


# ---------------------------------------------------------------------------
# Route receipt validation
# ---------------------------------------------------------------------------

_REQUIRED_ROUTE_RECEIPT_TOP = (
    "schema", "query_sha256", "route", "steps", "summary", "cost",
)
_REQUIRED_ROUTE = (
    "name", "selected_lobes", "route_reason", "verification_required", "n_steps",
)
_REQUIRED_STEP = (
    "lobe", "role", "model_used", "dispatch_path", "tokens_generated", "timing_ms",
)
_REQUIRED_SUMMARY = (
    "total_tokens_generated", "total_timing_ms", "models_used",
    "n_model_swaps", "all_fused",
)
_REQUIRED_COST = (
    "wall_time_s", "cpu_time_s", "peak_memory_mb", "python_version",
    "hostname", "timestamp_start", "timestamp_end",
)


def validate_route_receipt(receipt: dict) -> List[str]:
    """Validate a route receipt for structural and semantic correctness.

    Returns list of issues (empty = valid). Never raises.
    """
    issues = []
    if not isinstance(receipt, dict):
        return ["receipt is not a dict"]

    # Top-level fields
    for f in _REQUIRED_ROUTE_RECEIPT_TOP:
        if f not in receipt:
            issues.append(f"missing top-level field: {f}")

    if issues:
        return issues

    # Schema
    if receipt["schema"] != SCHEMA_ROUTE_RECEIPT:
        issues.append(
            f"schema mismatch: got {receipt['schema']!r}, "
            f"expected {SCHEMA_ROUTE_RECEIPT!r}"
        )

    # Route block
    route = receipt.get("route")
    if isinstance(route, dict):
        for f in _REQUIRED_ROUTE:
            if f not in route:
                issues.append(f"missing route.{f}")
        # selected_lobes must be a list
        lobes = route.get("selected_lobes")
        if lobes is not None and not isinstance(lobes, list):
            issues.append("route.selected_lobes is not a list")
        # All lobes must be registered
        if isinstance(lobes, list):
            for lobe_name in lobes:
                if lobe_name not in LOBE_REGISTRY:
                    issues.append(f"unknown lobe in route: {lobe_name!r}")
    else:
        issues.append("route is not a dict")

    # Steps
    steps = receipt.get("steps")
    if isinstance(steps, list):
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                issues.append(f"steps[{i}] is not a dict")
                continue
            for f in _REQUIRED_STEP:
                if f not in step:
                    issues.append(f"missing steps[{i}].{f}")
    else:
        issues.append("steps is not a list")

    # Summary
    summary = receipt.get("summary")
    if isinstance(summary, dict):
        for f in _REQUIRED_SUMMARY:
            if f not in summary:
                issues.append(f"missing summary.{f}")
    else:
        issues.append("summary is not a dict")

    # Cost
    cost = receipt.get("cost")
    if isinstance(cost, dict):
        for f in _REQUIRED_COST:
            if f not in cost:
                issues.append(f"missing cost.{f}")
    else:
        issues.append("cost is not a dict")

    # Semantic: n_steps matches actual steps count
    if isinstance(route, dict) and isinstance(steps, list):
        if route.get("n_steps") != len(steps):
            issues.append(
                f"semantic: route.n_steps={route.get('n_steps')} "
                f"but {len(steps)} step receipts"
            )

    # Semantic: verification_required=True requires a verifier step
    if isinstance(route, dict) and isinstance(steps, list):
        if route.get("verification_required") is True:
            has_verifier = any(
                s.get("role") == "verify" for s in steps if isinstance(s, dict)
            )
            if not has_verifier:
                issues.append(
                    "semantic: verification_required=True but no verify step"
                )

    return issues
