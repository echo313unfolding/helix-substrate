"""
Budget Gate — preflight VRAM check before every execution.

Layer 2 in the four-layer stack:
    Layer 1: Se cognitive router (proposes route)
    Layer 2: Budget gate (can hardware afford it?)  ← THIS
    Layer 3: Verifier/policy (is route allowed?)
    Layer 4: Receipt (records everything)

The budget gate sits between route selection and execution. It consults
the proven KV budget tables (WO-B) to determine whether the proposed
route can execute without OOM.

Three verdicts:
    RESIDENT_OK   — target model is resident, context fits, proceed
    SWAP_REQUIRED — context exceeds coresident budget, need solo mode
    DENY_BUDGET   — context exceeds even solo budget, block execution

All decisions are pure lookups (< 1ms, no GPU calls).

Work Order: WO-C (Budget-Aware Governed Multi-Lobe Controller)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

from helix_substrate.query_classifier import ModelTarget
from helix_substrate.lobe_context_policy import (
    PROVEN_BUDGETS, ResidencyMode, LobeKVBudget,
    check_context_budget, ContextBudgetResult, TOTAL_VRAM_MB,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = "budget_gate:v1"


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

class BudgetVerdict(Enum):
    """Result of a budget gate check."""
    RESIDENT_OK = "resident_ok"
    SWAP_REQUIRED = "swap_required"
    DENY_BUDGET = "deny_budget"


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetDecision:
    """Full budget gate decision with reasoning."""
    verdict: BudgetVerdict
    reason: str
    target: str                    # ModelTarget.value
    requested_context: int
    mode: str                      # "solo" or "coresident"
    estimated_peak_vram_mb: float
    context_cap: int               # max safe context for this mode
    recommended_action: str        # "proceed", "evict_other", "deny"

    def to_receipt(self) -> dict:
        """Serialize for receipt embedding."""
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "target": self.target,
            "requested_context": self.requested_context,
            "mode": self.mode,
            "estimated_peak_vram_mb": self.estimated_peak_vram_mb,
            "context_cap": self.context_cap,
            "recommended_action": self.recommended_action,
        }


# ---------------------------------------------------------------------------
# Budget Gate
# ---------------------------------------------------------------------------

class BudgetGate:
    """Preflight VRAM budget check.

    Consults proven KV budget tables. No GPU calls, no side effects.
    Designed to be called between route selection and execution.

    Usage:
        gate = BudgetGate()
        decision = gate.check(ModelTarget.TINYLLAMA, 512, ["tinyllama", "qwen_coder"])
        if decision.verdict == BudgetVerdict.DENY_BUDGET:
            # fail-closed
            ...
    """

    def __init__(self, total_vram_mb: int = TOTAL_VRAM_MB):
        self.total_vram_mb = total_vram_mb

    def check(
        self,
        target: ModelTarget,
        requested_context: int,
        resident_models: List[str],
    ) -> BudgetDecision:
        """Check if a route can execute within budget.

        Pure query. < 1ms. No GPU calls.

        Args:
            target: Which model the route needs.
            requested_context: Estimated context tokens for the request.
            resident_models: Currently resident model target values.

        Returns:
            BudgetDecision with verdict and reasoning.
        """
        # Step 1: Check coresident budget (current state)
        other_resident = [m for m in resident_models if m != target.value]
        is_coresident = len(other_resident) > 0

        if is_coresident:
            cores_result = check_context_budget(
                target, requested_context, resident_models,
            )
            if cores_result.fits:
                return self._make_decision(
                    BudgetVerdict.RESIDENT_OK,
                    target, requested_context,
                    "coresident", cores_result,
                    "proceed",
                )

            # Coresident doesn't fit — check if solo would work
            solo_result = check_context_budget(
                target, requested_context, [target.value],
            )
            if solo_result.fits:
                return self._make_decision(
                    BudgetVerdict.SWAP_REQUIRED,
                    target, requested_context,
                    "solo", solo_result,
                    "evict_other",
                    extra_reason=(
                        f"Coresident mode insufficient "
                        f"(max {cores_result.max_safe_context} tokens). "
                        f"Solo mode supports {solo_result.max_safe_context} tokens. "
                        f"Evict {', '.join(other_resident)} to proceed."
                    ),
                )

            # Even solo doesn't fit — deny
            return self._make_decision(
                BudgetVerdict.DENY_BUDGET,
                target, requested_context,
                "solo", solo_result,
                "deny",
                extra_reason=(
                    f"Exceeds even solo budget "
                    f"(max {solo_result.max_safe_context} tokens, "
                    f"headroom {solo_result.headroom_mb} MB). "
                    f"Cannot execute {requested_context} tokens on this hardware."
                ),
            )

        # Not coresident — check solo budget directly
        solo_result = check_context_budget(
            target, requested_context, [target.value],
        )
        if solo_result.fits:
            return self._make_decision(
                BudgetVerdict.RESIDENT_OK,
                target, requested_context,
                "solo", solo_result,
                "proceed",
            )

        # Solo doesn't fit — deny
        return self._make_decision(
            BudgetVerdict.DENY_BUDGET,
            target, requested_context,
            "solo", solo_result,
            "deny",
            extra_reason=(
                f"Exceeds solo budget "
                f"(max {solo_result.max_safe_context} tokens, "
                f"headroom {solo_result.headroom_mb} MB). "
                f"Cannot execute {requested_context} tokens."
            ),
        )

    def check_route(
        self,
        route_steps: List[dict],
        resident_models: List[str],
        default_context: int = 128,
    ) -> List[BudgetDecision]:
        """Check budget for each step in a multi-step route.

        Args:
            route_steps: List of dicts with at minimum:
                - "lobe_name": str
                - "model_target": str (ModelTarget.value) or None
                - "context_tokens": int (optional, defaults to default_context)
            resident_models: Currently resident model target values.
            default_context: Default context if not specified per step.

        Returns:
            List of BudgetDecision, one per step that has a model_target.
        """
        decisions = []
        for step in route_steps:
            target_str = step.get("model_target")
            if target_str is None:
                continue  # Non-model lobe (memory, parser, compiler)

            try:
                target = ModelTarget(target_str)
            except ValueError:
                decisions.append(BudgetDecision(
                    verdict=BudgetVerdict.DENY_BUDGET,
                    reason=f"Unknown model target: {target_str}",
                    target=target_str,
                    requested_context=0,
                    mode="unknown",
                    estimated_peak_vram_mb=0,
                    context_cap=0,
                    recommended_action="deny",
                ))
                continue

            ctx = step.get("context_tokens", default_context)
            decision = self.check(target, ctx, resident_models)
            decisions.append(decision)

        return decisions

    def _make_decision(
        self,
        verdict: BudgetVerdict,
        target: ModelTarget,
        requested_context: int,
        mode: str,
        budget_result: ContextBudgetResult,
        action: str,
        extra_reason: Optional[str] = None,
    ) -> BudgetDecision:
        """Build a BudgetDecision from a ContextBudgetResult."""
        # Estimate peak VRAM
        budget_key = (target.value, mode)
        budget = PROVEN_BUDGETS.get(budget_key)
        if budget:
            peak = budget.weight_vram_mb + budget_result.estimated_kv_mb
            # If coresident, add other model weight
            if mode == "coresident":
                # Approximate: total idle VRAM includes both models
                peak = self.total_vram_mb - budget.headroom_mb + budget_result.estimated_kv_mb
            context_cap = budget.max_safe_context
        else:
            peak = 0
            context_cap = 0

        reason = extra_reason or budget_result.reason

        return BudgetDecision(
            verdict=verdict,
            reason=reason,
            target=target.value,
            requested_context=requested_context,
            mode=mode,
            estimated_peak_vram_mb=round(peak, 1),
            context_cap=context_cap,
            recommended_action=action,
        )
