"""
Shaping Policy — WO-H

Explicit policy for route shaping decisions.  Turns WO-G's implicit
shaping rules into governed adaptation with hard invariants.

Policy controls:
    allow_output_cap     — can max_tokens be reduced?
    allow_memory_trim    — can memory injection be trimmed/dropped?
    allow_verifier_drop  — can verifier step be removed via downgrade?
    allow_planner_drop   — can planner step be removed via downgrade?
    min_output_tokens    — floor for output capping
    prefer_swap          — accept swap_required before trying downgrades?

Invariants:
    - Risky queries (delete, deploy, destroy...) lock verifier
    - Symbolic routes never bypass compile/verify
    - Policy violations are receipted with explicit reasons

Work Order: WO-H (Shaping Policy and Invariants)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple

SCHEMA = "shaping_policy:v1"

# ---------------------------------------------------------------------------
# Route topology helpers
# ---------------------------------------------------------------------------

# Routes that include a verifier step
_HAS_VERIFIER = frozenset({
    "plan_code_verify", "code_verify", "plan_verify",
    "symbolic_parse_verify", "symbolic_full",
})

# Routes that include a planner step
_HAS_PLANNER = frozenset({
    "plan_code_verify", "plan_verify", "direct_plan",
})

# Symbolic routes — never downgrade
_SYMBOLIC_ROUTES = frozenset({
    "symbolic_parse_verify", "symbolic_full",
})

# Keywords that signal risky/destructive intent
_RISKY_KEYWORDS = frozenset({
    "delete", "remove", "drop", "destroy", "production", "deploy",
    "migrate", "dangerous", "critical", "irreversible",
    "rollback", "revert", "rm", "kill", "truncate", "wipe",
})


# ---------------------------------------------------------------------------
# Policy dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShapingPolicy:
    """Explicit policy governing which shaping actions are allowed.

    Each field defaults to the most permissive setting.  Override via
    :meth:`for_query` to get query-appropriate invariants.
    """
    allow_output_cap: bool = True
    allow_memory_trim: bool = True
    allow_verifier_drop: bool = True
    allow_planner_drop: bool = True
    min_output_tokens: int = 32
    prefer_swap: bool = True  # accept swap_required before downgrading

    # ── factory ──

    @staticmethod
    def for_query(
        query: str,
        route_name: str,
        is_symbolic: bool = False,
    ) -> ShapingPolicy:
        """Build query-appropriate policy with hard invariants.

        Rules:
            1. Symbolic routes → lock all drops (never bypass compile/verify)
            2. Risky query keywords → lock verifier drop
            3. Everything else → fully permissive
        """
        # Symbolic routes: never downgrade
        if is_symbolic or route_name in _SYMBOLIC_ROUTES:
            return ShapingPolicy(
                allow_verifier_drop=False,
                allow_planner_drop=False,
            )

        # Risky queries: lock verifier
        query_lower = query.lower()
        is_risky = any(kw in query_lower for kw in _RISKY_KEYWORDS)
        if is_risky:
            return ShapingPolicy(
                allow_verifier_drop=False,
            )

        # Default: fully permissive
        return ShapingPolicy()

    # ── checks ──

    def check_output_cap(self, proposed_cap: int) -> Tuple[bool, str]:
        """Check if capping output to *proposed_cap* is allowed.

        Returns ``(allowed, reason)``.
        """
        if not self.allow_output_cap:
            return (False, "Output capping disabled by policy")
        if proposed_cap < self.min_output_tokens:
            return (False,
                    f"Proposed cap {proposed_cap} < min_output_tokens {self.min_output_tokens}")
        return (True, "")

    def check_memory_trim(self) -> Tuple[bool, str]:
        """Check if memory trimming is allowed.

        Returns ``(allowed, reason)``.
        """
        if not self.allow_memory_trim:
            return (False, "Memory trimming disabled by policy")
        return (True, "")

    def check_downgrade(
        self,
        from_route: str,
        to_route: str,
    ) -> Tuple[bool, str]:
        """Check if downgrading from *from_route* to *to_route* is allowed.

        Returns ``(allowed, reason)``.
        """
        loses_verifier = (from_route in _HAS_VERIFIER
                          and to_route not in _HAS_VERIFIER)
        loses_planner = (from_route in _HAS_PLANNER
                         and to_route not in _HAS_PLANNER)

        if loses_verifier and not self.allow_verifier_drop:
            return (False,
                    f"Policy forbids dropping verifier "
                    f"({from_route} → {to_route})")

        if loses_planner and not self.allow_planner_drop:
            return (False,
                    f"Policy forbids dropping planner "
                    f"({from_route} → {to_route})")

        return (True, "")

    # ── receipt ──

    def to_receipt(self) -> dict:
        d = asdict(self)
        d["schema"] = SCHEMA
        return d

    # ── diagnostics ──

    @property
    def constraints_summary(self) -> str:
        """Human-readable summary of active constraints."""
        parts = []
        if not self.allow_verifier_drop:
            parts.append("verifier_locked")
        if not self.allow_planner_drop:
            parts.append("planner_locked")
        if not self.allow_output_cap:
            parts.append("output_cap_locked")
        if not self.allow_memory_trim:
            parts.append("memory_trim_locked")
        if self.prefer_swap:
            parts.append("prefer_swap")
        return ", ".join(parts) if parts else "fully_permissive"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def is_risky_query(query: str) -> bool:
    """Quick check: does the query contain risky keywords?"""
    q = query.lower()
    return any(kw in q for kw in _RISKY_KEYWORDS)
