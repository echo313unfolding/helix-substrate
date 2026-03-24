"""
Session-level token budget and accounting.

Tracks cumulative token usage across a conversation session and enforces
configurable limits. Prevents runaway sessions from generating unbounded tokens.

Budget modes:
    - NORMAL: full generation budget
    - CAPPED: reduced max_tokens (approaching limit)
    - DENIED: no generation (budget exhausted)

Work Order: WO-AI-OS-RUNTIME-01, Phase 4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BudgetVerdict(Enum):
    NORMAL = "normal"       # Full budget available
    CAPPED = "capped"       # Reduced max_tokens
    DENIED = "denied"       # Budget exhausted


@dataclass
class SessionBudget:
    """Session-level token budget governor.

    Args:
        max_tokens: Total session token budget.
        warn_pct: Percentage at which to warn (default 80%).
        cap_pct: Percentage at which to cap max_tokens (default 90%).
    """
    max_tokens: int = 10000
    warn_pct: float = 0.80
    cap_pct: float = 0.90

    # Counters
    prompt_tokens: int = 0
    generated_tokens: int = 0
    queries_served: int = 0
    queries_denied: int = 0
    queries_capped: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def usage_pct(self) -> float:
        return self.total_tokens / max(1, self.max_tokens)

    def check(self, requested_tokens: int = 64) -> BudgetVerdict:
        """Check if generation is allowed within budget.

        Args:
            requested_tokens: max_tokens the caller wants to generate.

        Returns:
            BudgetVerdict indicating whether to proceed, cap, or deny.
        """
        if self.remaining <= 0:
            return BudgetVerdict.DENIED
        if self.usage_pct >= self.cap_pct:
            return BudgetVerdict.CAPPED
        return BudgetVerdict.NORMAL

    def capped_max_tokens(self, requested: int) -> int:
        """Return the effective max_tokens after budget cap.

        If budget is sufficient, returns the full requested amount.
        If budget is tight, returns a reduced cap (minimum 16).
        """
        if self.remaining >= requested:
            return requested
        return max(16, min(requested, self.remaining))

    def record_query(self, prompt_tokens: int, generated_tokens: int,
                     was_capped: bool = False, was_denied: bool = False) -> None:
        """Record a completed query's token usage."""
        self.prompt_tokens += prompt_tokens
        self.generated_tokens += generated_tokens
        self.queries_served += 1
        if was_capped:
            self.queries_capped += 1
        if was_denied:
            self.queries_denied += 1

    def should_warn(self) -> bool:
        """Return True if usage is above warn threshold."""
        return self.usage_pct >= self.warn_pct

    def reset(self) -> None:
        """Reset all counters (new session)."""
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.queries_served = 0
        self.queries_denied = 0
        self.queries_capped = 0

    def as_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "remaining": self.remaining,
            "usage_pct": round(self.usage_pct * 100, 1),
            "queries_served": self.queries_served,
            "queries_capped": self.queries_capped,
            "queries_denied": self.queries_denied,
        }
