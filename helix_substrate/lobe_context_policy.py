"""
Per-Lobe Context Policy — KV budget tables for governed execution.

All numbers traced to:
    bench_kv_cache_pressure_20260314T114028.json

This module provides:
    LobeKVBudget   — dataclass: per-model per-mode VRAM/KV budget
    PROVEN_BUDGETS — dict keyed by (ModelTarget, mode) with receipted numbers
    check_context_budget() — pure-query function: can this request fit?

Work Order: WO-B (Budget-Aware Governed Multi-Lobe Controller)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from helix_substrate.query_classifier import ModelTarget


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = "lobe_context_policy:v1"
SOURCE_RECEIPT = "bench_kv_cache_pressure_20260314T114028.json"

# T2000 total VRAM — from receipt
TOTAL_VRAM_MB = 3899


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResidencyMode(Enum):
    """How many models are on GPU."""
    SOLO = "solo"
    CORESIDENT = "coresident"


# ---------------------------------------------------------------------------
# Budget dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LobeKVBudget:
    """Per-model per-mode VRAM and KV budget.

    All values are from receipted benchmarks, not estimates.
    """
    target: ModelTarget
    mode: ResidencyMode
    weight_vram_mb: int       # VRAM used by model weights alone
    kv_per_token_mb: float    # Approximate KV cache cost per context token
    max_safe_context: int     # Highest context proven safe (no OOM)
    headroom_mb: int          # VRAM headroom at idle (total - weights used)
    max_tested_fail: Optional[int] = None  # Lowest context proven to OOM (if any)

    @property
    def budget_key(self) -> Tuple[str, str]:
        return (self.target.value, self.mode.value)


# ---------------------------------------------------------------------------
# Proven budgets — ALL values from receipt
# ---------------------------------------------------------------------------
#
# Source: bench_kv_cache_pressure_20260314T114028.json
#
# Solo TinyLlama:
#   weight VRAM = 1494 MB
#   headroom = 3899 - 1494 = 2405 MB
#   128 tok OK (peak 1521, KV ~26 MB) => 0.20 MB/tok
#   512 tok OK (peak 1626, KV ~124 MB)
#   1024 tok OK (peak 1901, KV ~399 MB)
#   2048 tok OK (peak 2850, KV ~1347 MB)
#   max_safe_context = 2048 (all tested contexts passed)
#
# Solo Qwen:
#   weight VRAM = 2205 MB (load_meta from receipt, rounds to 2205)
#   headroom = 3899 - 2205 = 1694 MB
#   128 tok OK (peak 2237, KV ~24 MB) => 0.19 MB/tok
#   512 tok OK (peak 2308, KV ~95 MB)
#   2048 tok OK (peak 2860, KV ~647 MB)
#   4096 tok OOM (peak 3225)
#   max_safe_context = 2048, max_tested_fail = 4096
#
# Coresident (both loaded):
#   idle VRAM = 3707 MB, headroom = 3899 - 3707 = 192 MB
#   TinyLlama 128 tok OK (peak 3726, KV ~19 MB)
#   TinyLlama 512 tok OOM
#   Qwen 128 tok OK (peak 3731, KV ~24 MB)
#   Qwen 512 tok OK (peak 3801, KV ~94 MB)
#   Qwen 1024 tok OOM
#   TinyLlama max_safe = 128, fail = 512
#   Qwen max_safe = 512, fail = 1024

PROVEN_BUDGETS: Dict[Tuple[str, str], LobeKVBudget] = {}

_BUDGET_ENTRIES = [
    LobeKVBudget(
        target=ModelTarget.TINYLLAMA,
        mode=ResidencyMode.SOLO,
        weight_vram_mb=1494,
        kv_per_token_mb=0.20,
        max_safe_context=2048,
        headroom_mb=2405,
        max_tested_fail=None,
    ),
    LobeKVBudget(
        target=ModelTarget.TINYLLAMA,
        mode=ResidencyMode.CORESIDENT,
        weight_vram_mb=1494,
        kv_per_token_mb=0.20,
        max_safe_context=128,
        headroom_mb=192,
        max_tested_fail=512,
    ),
    LobeKVBudget(
        target=ModelTarget.QWEN_CODER,
        mode=ResidencyMode.SOLO,
        weight_vram_mb=2205,
        kv_per_token_mb=0.19,
        max_safe_context=2048,
        headroom_mb=1694,
        max_tested_fail=4096,
    ),
    LobeKVBudget(
        target=ModelTarget.QWEN_CODER,
        mode=ResidencyMode.CORESIDENT,
        weight_vram_mb=2205,
        kv_per_token_mb=0.19,
        max_safe_context=512,
        headroom_mb=192,
        max_tested_fail=1024,
    ),
]

for _b in _BUDGET_ENTRIES:
    PROVEN_BUDGETS[_b.budget_key] = _b


# ---------------------------------------------------------------------------
# Budget check
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextBudgetResult:
    """Result of a context budget check."""
    fits: bool
    mode: str           # "solo" or "coresident"
    target: str         # ModelTarget.value
    requested_context: int
    max_safe_context: int
    estimated_kv_mb: float
    headroom_mb: int
    reason: str


def check_context_budget(
    target: ModelTarget,
    requested_context: int,
    resident_models: List[str],
) -> ContextBudgetResult:
    """Check whether a context length fits within proven KV budgets.

    Pure query — no GPU calls, no side effects. < 1ms.

    Args:
        target: Which model will run inference.
        requested_context: How many context tokens the request needs.
        resident_models: List of currently resident model target values
            (e.g., ["tinyllama", "qwen_coder"]).

    Returns:
        ContextBudgetResult with fits=True/False and reason.
    """
    # Determine mode: solo if target is the only resident, coresident otherwise
    other_resident = [m for m in resident_models if m != target.value]
    if other_resident:
        mode = ResidencyMode.CORESIDENT
    else:
        mode = ResidencyMode.SOLO

    key = (target.value, mode.value)
    budget = PROVEN_BUDGETS.get(key)

    if budget is None:
        return ContextBudgetResult(
            fits=False,
            mode=mode.value,
            target=target.value,
            requested_context=requested_context,
            max_safe_context=0,
            estimated_kv_mb=0.0,
            headroom_mb=0,
            reason=f"No proven budget for ({target.value}, {mode.value})",
        )

    estimated_kv = requested_context * budget.kv_per_token_mb
    fits = requested_context <= budget.max_safe_context

    if fits:
        reason = (f"{requested_context} tokens <= {budget.max_safe_context} max safe "
                  f"({mode.value} mode, ~{estimated_kv:.0f} MB KV)")
    else:
        reason = (f"{requested_context} tokens > {budget.max_safe_context} max safe "
                  f"({mode.value} mode, ~{estimated_kv:.0f} MB KV, "
                  f"headroom={budget.headroom_mb} MB)")

    return ContextBudgetResult(
        fits=fits,
        mode=mode.value,
        target=target.value,
        requested_context=requested_context,
        max_safe_context=budget.max_safe_context,
        estimated_kv_mb=round(estimated_kv, 1),
        headroom_mb=budget.headroom_mb,
        reason=reason,
    )


def get_budget(target: ModelTarget, mode: ResidencyMode) -> Optional[LobeKVBudget]:
    """Get a specific budget entry. Returns None if not found."""
    return PROVEN_BUDGETS.get((target.value, mode.value))


def all_budgets() -> List[LobeKVBudget]:
    """Return all budget entries."""
    return list(PROVEN_BUDGETS.values())
