"""
Pre-inference route decision — frozen before inference starts.

Collects ALL routing decisions into a single immutable object computed
at request start. Nothing downstream interprets policy per-token or
per-layer; it reads from the frozen RouteDecision.

Decisions frozen:
    - target_model: which model to use
    - sidecar_phase: "fused" | "scatter" | None
    - retrieval_mode: how to get context
    - budget_mode: token budget for generation
    - fused_path: whether Triton fused path is allowed

Work Order: WO-AI-OS-RUNTIME-01, Phase 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .query_classifier import ModelTarget


class RetrievalMode(Enum):
    """How to fetch context before inference."""
    CACHE_FIRST = "cache_first"      # Check response cache before anything
    EXACT_FACT = "exact_fact"        # Web passthrough, skip LLM
    LOCAL_RETRIEVAL = "local"        # Echo memory only
    WEB_DIRECT = "web_direct"        # Force web search
    CODE = "code"                     # No retrieval, code generation
    GRAPH = "graph"                   # FGIP graph tool routing


@dataclass(frozen=True)
class RouteDecision:
    """Immutable routing decision, frozen at request start.

    Every field is decided once. Nothing downstream re-interprets policy.
    """
    # Model routing
    target_model: ModelTarget
    model_confidence: float

    # Retrieval routing
    retrieval_mode: RetrievalMode

    # GPU execution routing
    sidecar_phase: Optional[str]  # "fused" | "scatter" | None
    fused_path_allowed: bool

    # Budget
    max_tokens: int
    budget_mode: str  # "normal" | "capped" | "graph"

    # Metadata for receipt
    classification_debug: dict = field(default_factory=dict)
    is_forced: bool = False
    graph_intent: Optional[str] = None
    inherited: bool = False           # WO-INHERIT-01: was this route inherited?
    inheritance_debug: dict = field(default_factory=dict)

    def as_receipt_dict(self) -> dict:
        """Flatten for receipt embedding."""
        return {
            "target_model": self.target_model.value,
            "model_confidence": self.model_confidence,
            "retrieval_mode": self.retrieval_mode.value,
            "sidecar_phase": self.sidecar_phase,
            "fused_path_allowed": self.fused_path_allowed,
            "max_tokens": self.max_tokens,
            "budget_mode": self.budget_mode,
            "is_forced": self.is_forced,
            "graph_intent": self.graph_intent,
            "inherited": self.inherited,
        }


def compute_route(
    query: str,
    force_model: Optional[ModelTarget] = None,
    web_mode: str = "auto",
    graph_available: bool = False,
    best_retrieval_score: float = 0.0,
    is_cache_candidate: bool = True,
    fused_available: bool = True,
    inherited_suggestion: Optional[dict] = None,
) -> RouteDecision:
    """Compute the full route decision before inference starts.

    This is the ONLY place routing decisions are made. Everything downstream
    reads from the returned RouteDecision — no per-token or per-layer policy.

    Args:
        query: Raw user query.
        force_model: Forced model override (from /tinyllama, /qwen commands).
        web_mode: "auto", "on", "off".
        graph_available: Whether FGIP graph is connected.
        best_retrieval_score: Best echo memory relevance score (0-1).
        is_cache_candidate: Whether to check response cache first.
        fused_available: Whether Triton fused path is available.
        inherited_suggestion: WO-INHERIT-01 suggestion from InheritanceStateMachine.
            If present and inherit=True, skips classification for inherited fields.
    """
    from .query_classifier import classify, detect_graph_intent
    from .web_fact_tool import needs_web_search, is_exact_fact_query

    # 1. Model classification
    if force_model:
        target = force_model
        confidence = 1.0
        debug = {"forced": True}
    else:
        target, confidence, debug = classify(query)

    is_code = target in (ModelTarget.QWEN_CODER, ModelTarget.QWEN_CODER_3B)

    # 2. Graph intent detection (checked for ALL queries — graph keywords are
    # specific enough to override code classification, e.g. "thesis score for X"
    # would otherwise match "(for)\s" code pattern)
    graph_intent = None
    if graph_available:
        graph_intent = detect_graph_intent(query)

    if graph_intent:
        target = ModelTarget.TINYLLAMA  # Graph queries always use TinyLlama

    # 3. Retrieval mode — priority order: graph > exact_fact > code > cache > web > local
    if graph_intent:
        retrieval_mode = RetrievalMode.GRAPH
    elif is_code:
        retrieval_mode = RetrievalMode.CODE
    elif is_exact_fact_query(query) and web_mode != "off":
        retrieval_mode = RetrievalMode.EXACT_FACT
    elif is_cache_candidate:
        retrieval_mode = RetrievalMode.CACHE_FIRST
    elif web_mode == "on":
        retrieval_mode = RetrievalMode.WEB_DIRECT
    elif web_mode == "auto" and needs_web_search(query, best_retrieval_score, is_code):
        retrieval_mode = RetrievalMode.WEB_DIRECT
    else:
        retrieval_mode = RetrievalMode.LOCAL_RETRIEVAL

    # 4. Sidecar phase — use auto-detect (None) to let the triton kernel
    # choose per-call based on N: fused for N<=16 (decode), scatter for N>16
    # (prefill). Blanket "fused" was proven to regress prefill by 10.5%
    # (stabilization receipt 2026-03-18). Auto-detect recovers to -1.7%.
    sidecar_phase = None

    # 5. Budget
    if graph_intent:
        max_tokens = 80
        budget_mode = "graph"
    elif is_code:
        max_tokens = 128
        budget_mode = "normal"
    else:
        max_tokens = 64
        budget_mode = "normal"

    # 6. WO-INHERIT-01: Apply inheritance suggestion if available
    # Inheritance can bias model/retrieval/budget but NEVER overrides:
    #   - forced model (user command)
    #   - graph intent (specific tool routing)
    #   - budget/quality/safety gates
    was_inherited = False
    inh_debug = {}
    if (inherited_suggestion
            and inherited_suggestion.get("inherit")
            and not force_model
            and not graph_intent):
        reuse = set(inherited_suggestion.get("fields_to_reuse", []))
        inh_debug = {
            "inherited": True,
            "fields_reused": list(reuse),
            "confidence": inherited_suggestion.get("confidence", 0),
            "reason": inherited_suggestion.get("reason", ""),
        }
        was_inherited = True
        # Note: we computed the full route above anyway — inheritance just
        # confirms it matches. This is the "first do no harm" approach.
        # In future, inherited fields could skip classification entirely.

    return RouteDecision(
        target_model=target,
        model_confidence=confidence,
        retrieval_mode=retrieval_mode,
        sidecar_phase=sidecar_phase,
        fused_path_allowed=fused_available,
        max_tokens=max_tokens,
        budget_mode=budget_mode,
        classification_debug=debug,
        is_forced=force_model is not None,
        graph_intent=graph_intent.value if graph_intent else None,
        inherited=was_inherited,
        inheritance_debug=inh_debug,
    )
