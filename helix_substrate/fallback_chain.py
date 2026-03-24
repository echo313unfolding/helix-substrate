"""
Explicit fallback chain with receipted steps.

Each retrieval attempt is logged with its outcome, latency, and reason for
advancing to the next step. No silent fallbacks — every step is visible.

Chain: local_retrieval → web_instant → web_search → page_fetch → admit_unknown

Work Order: WO-AI-OS-RUNTIME-01, Phase 4
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class FallbackStep(Enum):
    LOCAL_RETRIEVAL = "local_retrieval"
    WEB_INSTANT = "web_instant"
    WEB_SEARCH = "web_search"
    PAGE_FETCH = "page_fetch"
    ADMIT_UNKNOWN = "admit_unknown"


@dataclass
class StepResult:
    """Result of one fallback step."""
    step: FallbackStep
    succeeded: bool
    latency_ms: float
    reason: str  # Why this step was attempted
    outcome: str  # What happened ("got 3 facts", "no results", "error: timeout")
    content: Optional[str] = None  # Retrieved content (if succeeded)


@dataclass
class FallbackChainResult:
    """Complete chain execution result with full audit trail."""
    steps: List[StepResult] = field(default_factory=list)
    final_content: Optional[str] = None
    final_step: Optional[FallbackStep] = None
    total_latency_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.final_content is not None

    def as_receipt_dict(self) -> dict:
        return {
            "succeeded": self.succeeded,
            "final_step": self.final_step.value if self.final_step else None,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "steps": [
                {
                    "step": s.step.value,
                    "succeeded": s.succeeded,
                    "latency_ms": round(s.latency_ms, 1),
                    "reason": s.reason,
                    "outcome": s.outcome,
                }
                for s in self.steps
            ],
        }


class FallbackChain:
    """Execute retrieval with explicit fallback chain.

    Each step is receipted. Chain stops at first success or exhaustion.
    """

    def __init__(self, web_tool=None, echo_idx=None):
        self.web_tool = web_tool
        self.echo_idx = echo_idx

    def execute(
        self,
        query: str,
        skip_local: bool = False,
        skip_web: bool = False,
        min_local_score: float = 0.35,
    ) -> FallbackChainResult:
        """Run the full fallback chain.

        Args:
            query: User query.
            skip_local: Skip local retrieval step.
            skip_web: Skip all web steps.
            min_local_score: Minimum relevance score to accept local results.

        Returns:
            FallbackChainResult with full audit trail.
        """
        result = FallbackChainResult()
        t_total = time.perf_counter()

        # Step 1: Local retrieval
        if not skip_local and self.echo_idx is not None:
            t0 = time.perf_counter()
            try:
                chunks = self.echo_idx.search(query[:500], k=2)
                best_score = chunks[0].relevance_score if chunks else 0.0
                latency = (time.perf_counter() - t0) * 1000

                if chunks and best_score >= min_local_score:
                    content = "\n".join([c.text for c in chunks])
                    result.steps.append(StepResult(
                        step=FallbackStep.LOCAL_RETRIEVAL,
                        succeeded=True,
                        latency_ms=latency,
                        reason="first step in chain",
                        outcome=f"score={best_score:.3f}, {len(chunks)} chunks",
                        content=content,
                    ))
                    result.final_content = content
                    result.final_step = FallbackStep.LOCAL_RETRIEVAL
                    result.total_latency_ms = (time.perf_counter() - t_total) * 1000
                    return result
                else:
                    result.steps.append(StepResult(
                        step=FallbackStep.LOCAL_RETRIEVAL,
                        succeeded=False,
                        latency_ms=latency,
                        reason="first step in chain",
                        outcome=f"score={best_score:.3f} < threshold={min_local_score}",
                    ))
            except Exception as e:
                latency = (time.perf_counter() - t0) * 1000
                result.steps.append(StepResult(
                    step=FallbackStep.LOCAL_RETRIEVAL,
                    succeeded=False,
                    latency_ms=latency,
                    reason="first step in chain",
                    outcome=f"error: {e}",
                ))

        # Steps 2-4: Web retrieval (instant → search → fetch)
        if not skip_web and self.web_tool is not None:
            t0 = time.perf_counter()
            try:
                web_result = self.web_tool.search(query)
                latency = (time.perf_counter() - t0) * 1000

                if web_result and web_result.facts:
                    content = web_result.as_context(max_chars=400)
                    if content:
                        # Determine which web step succeeded based on method
                        method = web_result.facts[0].method if web_result.facts else "unknown"
                        if "instant" in method:
                            step = FallbackStep.WEB_INSTANT
                        elif "fetch" in method:
                            step = FallbackStep.PAGE_FETCH
                        else:
                            step = FallbackStep.WEB_SEARCH

                        result.steps.append(StepResult(
                            step=step,
                            succeeded=True,
                            latency_ms=latency,
                            reason="local retrieval insufficient",
                            outcome=f"{len(web_result.facts)} facts via {method}",
                            content=content,
                        ))
                        result.final_content = content
                        result.final_step = step
                        result.total_latency_ms = (time.perf_counter() - t_total) * 1000
                        return result
                    else:
                        result.steps.append(StepResult(
                            step=FallbackStep.WEB_SEARCH,
                            succeeded=False,
                            latency_ms=latency,
                            reason="local retrieval insufficient",
                            outcome=f"{len(web_result.facts)} facts but no useful context",
                        ))
                else:
                    result.steps.append(StepResult(
                        step=FallbackStep.WEB_SEARCH,
                        succeeded=False,
                        latency_ms=latency,
                        reason="local retrieval insufficient",
                        outcome="no results from web",
                    ))
            except Exception as e:
                latency = (time.perf_counter() - t0) * 1000
                result.steps.append(StepResult(
                    step=FallbackStep.WEB_SEARCH,
                    succeeded=False,
                    latency_ms=latency,
                    reason="local retrieval insufficient",
                    outcome=f"error: {e}",
                ))

        # Step 5: Admit unknown
        result.steps.append(StepResult(
            step=FallbackStep.ADMIT_UNKNOWN,
            succeeded=True,
            latency_ms=0.0,
            reason="all retrieval steps exhausted",
            outcome="no authoritative information found",
            content="I don't have enough information to answer this accurately.",
        ))
        result.final_content = "I don't have enough information to answer this accurately."
        result.final_step = FallbackStep.ADMIT_UNKNOWN
        result.total_latency_ms = (time.perf_counter() - t_total) * 1000
        return result
