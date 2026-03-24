"""
Adaptive Route Shaping — WO-G / WO-H

Uses budget pressure to modify route selection before execution.
Not just approve/deny — reshapes the route to fit, governed by
explicit :class:`ShapingPolicy` invariants.

Shaping rules (applied in order of increasing severity):
    1. cap_output       — reduce max_tokens on over-budget steps
    2. trim_memory      — reduce memory injection (top-1 then none)
    3. downgrade_route  — drop verifier, then planner
    4. deny             — no legal execution shape fits

Each rule is gated by the ShapingPolicy:
    - Risky queries lock verifier (cannot be dropped)
    - Symbolic routes lock all downgrades
    - Policy violations are receipted with explicit reasons

Work Order: WO-G (Adaptive Route Shaping), WO-H (Shaping Policy)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from .budget_gate import BudgetGate, BudgetVerdict
from .shaping_policy import ShapingPolicy
from .token_accountant import TokenAccountant, RouteTokenAccount


SCHEMA = "route_shaper:v2"

# Route downgrade chains: each entry maps a route to its lighter alternative.
DOWNGRADE_MAP = {
    "plan_code_verify": "code_verify",
    "code_verify": "direct_code",
    "plan_verify": "direct_plan",
}

# Output caps to try, in order of decreasing budget.
OUTPUT_CAPS = [128, 64, 32]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ShapingAction:
    """One shaping rule application."""
    rule: str         # e.g. "cap_output", "trim_memory_top1", "downgrade_route"
    reason: str       # why it was applied
    before: str       # previous state
    after: str        # new state
    policy_allowed: bool = True   # WO-H: was this allowed by policy?
    policy_reason: str = ""       # WO-H: if blocked, why


@dataclass
class ShapedResult:
    """Complete result of route shaping."""
    original_route_name: str
    shaped_route_name: str
    max_tokens: Optional[int]         # possibly reduced (None = use lobe default)
    memory_context: str               # possibly trimmed
    actions: List[ShapingAction]
    shaped: bool                      # True if any changes were made
    verdict: str                      # final verdict after shaping
    token_account: Optional[RouteTokenAccount]
    per_step_decisions: Optional[List[dict]]
    policy: Optional[ShapingPolicy] = None  # WO-H: policy that governed this shaping

    def to_receipt(self) -> dict:
        blocked = [a for a in self.actions if not a.policy_allowed]
        return {
            "schema": SCHEMA,
            "original_route": self.original_route_name,
            "shaped_route": self.shaped_route_name,
            "max_tokens": self.max_tokens,
            "memory_trimmed": any(a.rule.startswith("trim_memory") for a in self.actions
                                  if a.policy_allowed),
            "shaped": self.shaped,
            "n_actions": len(self.actions),
            "actions": [asdict(a) for a in self.actions],
            "verdict": self.verdict,
            "token_account": self.token_account.to_receipt() if self.token_account else None,
            "per_step_decisions": self.per_step_decisions,
            "policy": self.policy.to_receipt() if self.policy else None,
            "policy_constraints": self.policy.constraints_summary if self.policy else "none",
            "safety_constraints_applied": len(blocked),
            "safety_blocks": [
                {"rule": a.rule, "reason": a.policy_reason} for a in blocked
            ],
        }


# ---------------------------------------------------------------------------
# Core shaping function
# ---------------------------------------------------------------------------

def shape_route(
    route,
    route_name: str,
    query: str,
    memory_context: str,
    max_tokens: Optional[int],
    accountant: TokenAccountant,
    gate: BudgetGate,
    resident_models: List[str],
    policy: Optional[ShapingPolicy] = None,
) -> ShapedResult:
    """Shape a route to fit within budget constraints.

    Tries shaping rules in order of least to most severe, gated by
    the :class:`ShapingPolicy`.  Returns the first shape that fits.

    Args:
        route: The proposed Route object.
        route_name: Route name string.
        query: User query text.
        memory_context: Recalled memory text (may be empty).
        max_tokens: Current max_tokens override (None = use lobe defaults).
        accountant: TokenAccountant for real token counting.
        gate: BudgetGate for budget checks.
        resident_models: Currently resident model target values.
        policy: Shaping policy (WO-H).  If None, built from query context.

    Returns:
        ShapedResult with the (possibly modified) execution parameters.
    """
    from .lobe_scheduler import _ALL_ROUTES

    # ── WO-H: Build or use policy ──
    if policy is None:
        from .shaping_policy import ShapingPolicy
        policy = ShapingPolicy.for_query(query, route_name)

    actions: List[ShapingAction] = []

    def _result(name, mt, mem, v, ta, sd):
        return ShapedResult(
            original_route_name=route_name,
            shaped_route_name=name,
            max_tokens=mt,
            memory_context=mem,
            actions=actions,
            shaped=len(actions) > 0,
            verdict=v,
            token_account=ta,
            per_step_decisions=sd,
            policy=policy,
        )

    # ── Check original route ──
    verdict, token_acct, step_decs = _check_route_budget(
        route, query, memory_context, max_tokens,
        accountant, gate, resident_models,
    )

    if verdict in ("resident_ok", "swap_required"):
        return _result(route_name, max_tokens, memory_context,
                       verdict, token_acct, step_decs)

    # ── Rule 1: Cap output ──
    if policy.allow_output_cap:
        for cap in OUTPUT_CAPS:
            if max_tokens is not None and max_tokens <= cap:
                continue
            cap_ok, cap_reason = policy.check_output_cap(cap)
            if not cap_ok:
                actions.append(ShapingAction(
                    rule="cap_output",
                    reason=f"Would cap to {cap}",
                    before=f"max_tokens={max_tokens or 'default'}",
                    after=f"max_tokens={cap}",
                    policy_allowed=False,
                    policy_reason=cap_reason,
                ))
                continue

            v, ta, sd = _check_route_budget(
                route, query, memory_context, cap,
                accountant, gate, resident_models,
            )
            if v in ("resident_ok", "swap_required"):
                actions.append(ShapingAction(
                    rule="cap_output",
                    reason=f"Budget pressure: reduced per-step output to {cap} tokens",
                    before=f"max_tokens={max_tokens or 'default'}",
                    after=f"max_tokens={cap}",
                ))
                return _result(route_name, cap, memory_context, v, ta, sd)

    # ── Rule 2: Trim memory ──
    if memory_context:
        trim_ok, trim_reason = policy.check_memory_trim()
        if trim_ok:
            # Try top-1 only
            lines = memory_context.split("\n")
            if len(lines) > 2:
                trimmed = lines[0] + "\n" + lines[1]
                v, ta, sd = _check_route_budget(
                    route, query, trimmed, max_tokens,
                    accountant, gate, resident_models,
                )
                if v in ("resident_ok", "swap_required"):
                    actions.append(ShapingAction(
                        rule="trim_memory_top1",
                        reason="Budget pressure: trimmed memory to top-1 match",
                        before=f"memory_items={len(lines) - 1}",
                        after="memory_items=1",
                    ))
                    return _result(route_name, max_tokens, trimmed, v, ta, sd)

            # Try no memory
            v, ta, sd = _check_route_budget(
                route, query, "", max_tokens,
                accountant, gate, resident_models,
            )
            if v in ("resident_ok", "swap_required"):
                actions.append(ShapingAction(
                    rule="trim_memory_none",
                    reason="Budget pressure: dropped all memory injection",
                    before=f"memory_len={len(memory_context)}",
                    after="memory_len=0",
                ))
                return _result(route_name, max_tokens, "", v, ta, sd)
        else:
            actions.append(ShapingAction(
                rule="trim_memory",
                reason="Would trim memory",
                before=f"memory_len={len(memory_context)}",
                after="blocked",
                policy_allowed=False,
                policy_reason=trim_reason,
            ))

    # ── Rule 3: Downgrade route (+ combined rules) ──
    current_name = route_name
    downgrade_name = DOWNGRADE_MAP.get(current_name)

    while downgrade_name:
        if downgrade_name not in _ALL_ROUTES:
            break

        # WO-H: Check policy before downgrading
        dg_ok, dg_reason = policy.check_downgrade(current_name, downgrade_name)
        if not dg_ok:
            actions.append(ShapingAction(
                rule="downgrade_route",
                reason=f"Would downgrade {current_name} → {downgrade_name}",
                before=f"route={current_name}",
                after=f"route={downgrade_name}",
                policy_allowed=False,
                policy_reason=dg_reason,
            ))
            # Cannot continue down this chain — policy blocks it
            break

        downgraded = _ALL_ROUTES[downgrade_name]

        # Try downgraded route with current settings
        v, ta, sd = _check_route_budget(
            downgraded, query, memory_context, max_tokens,
            accountant, gate, resident_models,
        )
        if v in ("resident_ok", "swap_required"):
            actions.append(ShapingAction(
                rule="downgrade_route",
                reason=f"Budget pressure: {current_name} → {downgrade_name}",
                before=f"route={current_name}",
                after=f"route={downgrade_name}",
            ))
            return _result(downgrade_name, max_tokens, memory_context, v, ta, sd)

        # Try downgrade + output cap + no memory (combined pressure)
        for cap in OUTPUT_CAPS:
            cap_ok, _ = policy.check_output_cap(cap)
            mem_ok, _ = policy.check_memory_trim()
            if not cap_ok:
                continue

            mem_to_use = "" if mem_ok else memory_context
            v, ta, sd = _check_route_budget(
                downgraded, query, mem_to_use, cap,
                accountant, gate, resident_models,
            )
            if v in ("resident_ok", "swap_required"):
                actions.append(ShapingAction(
                    rule="downgrade_route",
                    reason=f"Budget pressure: {current_name} → {downgrade_name}",
                    before=f"route={current_name}",
                    after=f"route={downgrade_name}",
                ))
                if memory_context and mem_ok:
                    actions.append(ShapingAction(
                        rule="trim_memory_none",
                        reason="Combined with route downgrade",
                        before=f"memory_len={len(memory_context)}",
                        after="memory_len=0",
                    ))
                actions.append(ShapingAction(
                    rule="cap_output",
                    reason="Combined with route downgrade",
                    before=f"max_tokens={max_tokens or 'default'}",
                    after=f"max_tokens={cap}",
                ))
                return _result(downgrade_name, cap, mem_to_use, v, ta, sd)

        # Continue down the chain
        current_name = downgrade_name
        downgrade_name = DOWNGRADE_MAP.get(current_name)

    # ── Rule 4: Deny ──
    _, ta, sd = _check_route_budget(
        route, query, "", None,
        accountant, gate, resident_models,
    )

    # Build deny reason from policy constraints if any were hit
    blocked_rules = [a for a in actions if not a.policy_allowed]
    if blocked_rules:
        deny_reason = (
            f"No legal shape fits. Policy blocked: "
            f"{', '.join(a.rule + ' (' + a.policy_reason + ')' for a in blocked_rules)}"
        )
    else:
        deny_reason = "No legal execution shape fits within budget"

    actions.append(ShapingAction(
        rule="deny",
        reason=deny_reason,
        before=f"route={route_name}",
        after="DENIED",
    ))
    return _result(route_name, max_tokens, memory_context,
                   "deny_budget", ta, sd)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _check_route_budget(
    route,
    query: str,
    memory_context: str,
    max_tokens: Optional[int],
    accountant: TokenAccountant,
    gate: BudgetGate,
    resident_models: List[str],
) -> Tuple[str, Optional[RouteTokenAccount], Optional[List[dict]]]:
    """Check if a route fits within budget.

    Returns ``(verdict, token_account, per_step_decision_receipts)``.
    """
    from .lobe_scheduler import get_lobe

    try:
        token_acct = accountant.account_route(
            route, query,
            memory_context=memory_context,
            max_tokens=max_tokens,
        )

        route_steps = []
        for i, rs in enumerate(route.steps):
            lobe = get_lobe(rs.lobe_name)
            mt = lobe.capability.model_target
            if mt is None:
                continue
            route_steps.append({
                "lobe_name": rs.lobe_name,
                "model_target": mt.value,
                "context_tokens": token_acct.steps[i].total_budget_tokens,
            })

        if not route_steps:
            return ("resident_ok", token_acct, None)

        decisions = gate.check_route(route_steps, resident_models)
        per_step = [d.to_receipt() for d in decisions]

        if any(d.verdict == BudgetVerdict.DENY_BUDGET for d in decisions):
            return ("deny_budget", token_acct, per_step)
        elif any(d.verdict == BudgetVerdict.SWAP_REQUIRED for d in decisions):
            return ("swap_required", token_acct, per_step)
        else:
            return ("resident_ok", token_acct, per_step)

    except Exception:
        return ("error", None, None)
