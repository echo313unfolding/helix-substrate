"""
Transform law — explicit state machine for legal system transitions.

Formalizes the 4 independent state axes of the AI OS runtime and the coupling
rules that bind them. This is the system law that answers:
  - What transitions are legal?
  - What transitions require reset?
  - What transitions carry cache/KV safely?
  - What transitions poison state?

The 4 axes:
  Route:   COLD → WARM_CANDIDATE → INHERITED → UNSTABLE → RESET → COLD
  Cache:   MISS | HIT | KV_REUSE
  Quality: UNCHECKED | PASS | FAIL_*
  Budget:  NORMAL | CAPPED | DENIED

Degradation ladder (service levels, best → worst):
  FULL → INHERITED → KV_PREFIXED → CACHE_REPLAY → WEB_PASSTHROUGH →
  CAPPED → FALLBACK → ADMIT_UNKNOWN → DENIED

Quality floors per query type prevent degradation below a minimum level.

Work Order: WO-TRANSFORM-01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple


# ── Axis 1: Route State ──
# (mirrors route_inheritance.InheritanceState)

class RouteState(Enum):
    COLD = "cold"
    WARM_CANDIDATE = "warm_candidate"
    INHERITED = "inherited"
    UNSTABLE = "unstable"
    RESET = "reset"


# ── Axis 2: Cache Event ──
# (per-query outcome, not persistent state)

class CacheEvent(Enum):
    MISS = "miss"               # no cached response or KV prefix
    RESPONSE_HIT = "response_hit"  # response cache hit → skip generation
    KV_REUSE = "kv_reuse"      # KV prefix reused → skip prefix prefill


# ── Axis 3: Quality State ──
# (mirrors quality_gate.QualityVerdict but adds UNCHECKED)

class QualityState(Enum):
    UNCHECKED = "unchecked"     # before generation or on cache/web bypass
    PASS = "pass"
    FAIL_EMPTY = "empty"
    FAIL_REPETITION = "repetition"
    FAIL_OVERLONG = "overlong"
    FAIL_DEGENERATE = "degenerate"

    @property
    def failed(self) -> bool:
        return self not in (QualityState.UNCHECKED, QualityState.PASS)


# ── Axis 4: Budget State ──
# (mirrors session_budget.BudgetVerdict)

class BudgetState(Enum):
    NORMAL = "normal"
    CAPPED = "capped"
    DENIED = "denied"


# ── Service Level (Degradation Ladder) ──
# IntEnum so levels are orderable: higher = better service

class ServiceLevel(IntEnum):
    DENIED = 1            # budget exhausted, refuse to serve
    ADMIT_UNKNOWN = 2     # "I don't have enough information"
    FALLBACK = 3          # echo memory or web retrieval only, no LLM
    CAPPED = 4            # budget-capped generation (reduced max_tokens)
    WEB_PASSTHROUGH = 5   # web fact bypass, no LLM generation
    CACHE_REPLAY = 6      # cached response returned
    KV_PREFIXED = 7       # KV prefix reused + generate new tokens
    INHERITED = 8         # inherited route + full generation
    FULL = 9              # fresh route + full generation


# ── Quality Floor ──
# Minimum acceptable service level per query type.
# The runtime MUST NOT degrade below this level for a given query type.

QUALITY_FLOOR: Dict[str, ServiceLevel] = {
    "code": ServiceLevel.CAPPED,           # code needs generation (can cap, not skip)
    "graph": ServiceLevel.CAPPED,          # graph needs generation
    "analytical": ServiceLevel.INHERITED,  # analytical can inherit but needs LLM
    "factual": ServiceLevel.WEB_PASSTHROUGH,  # facts can come from web
    "neutral": ServiceLevel.FALLBACK,      # general chat can fall back to retrieval
}

# Default floor for unknown query types
DEFAULT_QUALITY_FLOOR = ServiceLevel.FALLBACK


def get_quality_floor(query_type: str) -> ServiceLevel:
    """Return the minimum acceptable service level for this query type."""
    return QUALITY_FLOOR.get(query_type, DEFAULT_QUALITY_FLOOR)


def can_degrade_to(query_type: str, target_level: ServiceLevel) -> bool:
    """Check whether degrading to target_level is legal for this query type."""
    floor = get_quality_floor(query_type)
    return target_level >= floor


# ── Composed System State ──

@dataclass
class SystemState:
    """Snapshot of the 4-axis system state at a point in time."""
    route: RouteState
    cache: CacheEvent
    quality: QualityState
    budget: BudgetState
    service_level: ServiceLevel

    def as_dict(self) -> dict:
        return {
            "route": self.route.value,
            "cache": self.cache.value,
            "quality": self.quality.value,
            "budget": self.budget.value,
            "service_level": self.service_level.name,
        }


# ── Coupling Rules ──
# Events in one axis that force transitions in other axes.

@dataclass
class CouplingEffect:
    """A forced state change caused by an event in another axis."""
    trigger: str          # what happened (e.g., "quality_fail")
    source_axis: str      # which axis fired (e.g., "quality")
    forced_axis: str      # which axis is forced (e.g., "route")
    forced_to: str        # what state it's forced to (e.g., "RESET")
    reason: str           # why


# Explicit coupling rules — these are the cross-axis invariants.
# Each rule says: "if X happens in axis A, then axis B MUST transition to Y."

COUPLING_RULES: List[CouplingEffect] = [
    CouplingEffect(
        trigger="quality_fail",
        source_axis="quality",
        forced_axis="route",
        forced_to="RESET",
        reason="Quality failure means the route may have been wrong. "
               "Reset to COLD to prevent inheriting a bad route.",
    ),
    CouplingEffect(
        trigger="model_swap",
        source_axis="route",
        forced_axis="route",
        forced_to="RESET",
        reason="GPU model swap invalidates all route assumptions. "
               "KV cache is from the old model.",
    ),
    CouplingEffect(
        trigger="model_swap",
        source_axis="route",
        forced_axis="cache",
        forced_to="CLEAR",
        reason="KV cache from model A cannot be used for model B. "
               "Response cache survives (keyed by model).",
    ),
    CouplingEffect(
        trigger="hard_drift",
        source_axis="route",
        forced_axis="route",
        forced_to="RESET",
        reason="Regime distance >= 0.60 means fundamentally different query. "
               "No state carries safely.",
    ),
    CouplingEffect(
        trigger="budget_denied",
        source_axis="budget",
        forced_axis="service_level",
        forced_to="DENIED",
        reason="Budget exhausted. No generation, no retrieval cost. "
               "Service level forced to DENIED.",
    ),
]


# ── Illegal State Combinations ──
# Combinations that should NEVER occur. If detected, something is broken.

def check_illegal_states(state: SystemState) -> List[str]:
    """Return list of violations. Empty list = legal state."""
    violations = []

    # 1. Cannot be INHERITED and have a cache MISS with quality UNCHECKED
    #    and budget DENIED. (Budget denied should prevent reaching generation.)
    if state.budget == BudgetState.DENIED and state.service_level > ServiceLevel.DENIED:
        if state.service_level not in (ServiceLevel.CACHE_REPLAY, ServiceLevel.ADMIT_UNKNOWN):
            violations.append(
                "Budget DENIED but service level requires generation "
                f"({state.service_level.name}). Only CACHE_REPLAY or "
                "ADMIT_UNKNOWN are legal when budget is DENIED."
            )

    # 2. Cannot have RESPONSE_HIT with quality FAIL_*.
    #    Cache hits bypass quality gate entirely.
    if state.cache == CacheEvent.RESPONSE_HIT and state.quality.failed:
        violations.append(
            f"Response cache HIT but quality is {state.quality.value}. "
            "Cache hits bypass the quality gate — quality should be UNCHECKED."
        )

    # 3. Cannot be KV_REUSE with route COLD.
    #    KV prefix reuse requires an established prior context.
    if state.cache == CacheEvent.KV_REUSE and state.route == RouteState.COLD:
        violations.append(
            "KV prefix reuse with route COLD. KV reuse requires "
            "a prior query in the same session to have stored the prefix."
        )

    # 4. Cannot be INHERITED service level with route COLD/RESET.
    if state.service_level == ServiceLevel.INHERITED:
        if state.route in (RouteState.COLD, RouteState.RESET):
            violations.append(
                f"Service level INHERITED but route is {state.route.value}. "
                "Inheritance requires route state INHERITED."
            )

    # 5. Quality FAIL with service level > FALLBACK.
    #    A quality failure should trigger fallback or retry, not continue serving.
    if state.quality.failed and state.service_level > ServiceLevel.FALLBACK:
        violations.append(
            f"Quality {state.quality.value} but service level "
            f"{state.service_level.name}. Quality failure should "
            "trigger fallback or retry, not serve at this level."
        )

    return violations


# ── Transition Validation ──

def required_transitions(event: str) -> List[CouplingEffect]:
    """Given an event, return the coupling rules that must fire."""
    return [r for r in COUPLING_RULES if r.trigger == event]


# ── Degradation Path ──

DEGRADATION_SEQUENCE = [
    ServiceLevel.FULL,
    ServiceLevel.INHERITED,
    ServiceLevel.KV_PREFIXED,
    ServiceLevel.CACHE_REPLAY,
    ServiceLevel.WEB_PASSTHROUGH,
    ServiceLevel.CAPPED,
    ServiceLevel.FALLBACK,
    ServiceLevel.ADMIT_UNKNOWN,
    ServiceLevel.DENIED,
]


def degradation_path(query_type: str) -> List[ServiceLevel]:
    """Return the legal degradation path for this query type.

    Filters the global sequence by the quality floor for the query type.
    """
    floor = get_quality_floor(query_type)
    # Include all levels down to and including the floor, plus ADMIT_UNKNOWN and DENIED
    # (which are always reachable as last resort)
    path = []
    for level in DEGRADATION_SEQUENCE:
        if level >= floor:
            path.append(level)
        elif level <= ServiceLevel.ADMIT_UNKNOWN:
            # ADMIT_UNKNOWN and DENIED are always legal (hard floor)
            path.append(level)
    return path


# ── Safe Carry Rules ──
# What state survives which transitions?

@dataclass
class CarryRule:
    """What data is safe to carry across a transition."""
    transition: str
    response_cache_safe: bool
    kv_cache_safe: bool
    route_safe: bool
    reason: str


CARRY_RULES: List[CarryRule] = [
    CarryRule(
        transition="same_model_sequential",
        response_cache_safe=True,
        kv_cache_safe=True,
        route_safe=True,
        reason="Same model, sequential queries — all state carries.",
    ),
    CarryRule(
        transition="model_swap",
        response_cache_safe=True,   # keyed by model, still valid
        kv_cache_safe=False,        # wrong model's KV
        route_safe=False,           # route assumptions invalid
        reason="Model swap invalidates KV and route. "
               "Response cache survives because it's keyed by model.",
    ),
    CarryRule(
        transition="quality_failure",
        response_cache_safe=True,   # cache didn't cause the failure
        kv_cache_safe=True,         # KV prefix is still valid
        route_safe=False,           # route may have been wrong
        reason="Quality failure resets route but cache/KV are safe. "
               "The output was bad, not the cache.",
    ),
    CarryRule(
        transition="hard_drift",
        response_cache_safe=True,   # keyed by content, still valid
        kv_cache_safe=False,        # KV prefix may be from different context
        route_safe=False,           # completely different regime
        reason="Hard drift (>0.60) means fundamentally different query. "
               "KV prefix unlikely to share tokens. Route is invalid.",
    ),
    CarryRule(
        transition="budget_exhausted",
        response_cache_safe=True,   # can still serve from cache
        kv_cache_safe=True,         # KV is allocated, just don't add
        route_safe=True,            # route is still valid
        reason="Budget exhaustion doesn't invalidate any state. "
               "It just prevents new generation.",
    ),
    CarryRule(
        transition="session_reset",
        response_cache_safe=False,  # new session, clear everything
        kv_cache_safe=False,
        route_safe=False,
        reason="Session reset clears all state.",
    ),
]


def get_carry_rule(transition: str) -> Optional[CarryRule]:
    """Look up the carry rule for a given transition type."""
    for rule in CARRY_RULES:
        if rule.transition == transition:
            return rule
    return None


# ── Law Summary (for receipts) ──

def law_summary() -> dict:
    """Return the complete transform law as a serializable dict."""
    return {
        "work_order": "WO-TRANSFORM-01",
        "axes": {
            "route": [s.value for s in RouteState],
            "cache": [s.value for s in CacheEvent],
            "quality": [s.value for s in QualityState],
            "budget": [s.value for s in BudgetState],
        },
        "service_levels": [
            {"name": s.name, "value": s.value}
            for s in sorted(ServiceLevel, reverse=True)
        ],
        "quality_floors": {
            k: v.name for k, v in QUALITY_FLOOR.items()
        },
        "coupling_rules": [
            {
                "trigger": r.trigger,
                "source": r.source_axis,
                "forced_axis": r.forced_axis,
                "forced_to": r.forced_to,
            }
            for r in COUPLING_RULES
        ],
        "carry_rules": [
            {
                "transition": r.transition,
                "response_cache": r.response_cache_safe,
                "kv_cache": r.kv_cache_safe,
                "route": r.route_safe,
            }
            for r in CARRY_RULES
        ],
        "degradation_paths": {
            qt: [s.name for s in degradation_path(qt)]
            for qt in QUALITY_FLOOR
        },
        "n_coupling_rules": len(COUPLING_RULES),
        "n_carry_rules": len(CARRY_RULES),
        "n_illegal_checks": 5,
    }
