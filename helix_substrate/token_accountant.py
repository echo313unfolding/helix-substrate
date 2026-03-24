"""
Token-Accurate Budget Accountant — WO-E

Replaces the heuristic ``len(query) // 4`` context estimator with
real tokenizer-based token counting.

The formula::

    requested_context
    = query_tokens
    + memory_tokens
    + route_prompt_tokens
    + expected_generation_budget

All counts use the target model's actual tokenizer.
Degrades to heuristic (len // 4, min 1) if tokenizer unavailable.
The ``source`` field on every TokenAccount tells you which path was used.

Work Order: WO-E (Token-Accurate Budget Gate)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular import; we import at call time

log = logging.getLogger(__name__)

SCHEMA = "token_accountant:v1"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TokenAccount:
    """Token-accurate budget for a single execution step."""
    query_tokens: int
    memory_tokens: int
    prompt_tokens: int             # Template chrome (fixed text around placeholders)
    context_tokens: int            # Output carried from previous route step
    expected_output_tokens: int    # max_tokens for this step
    total_input_tokens: int        # Full tokenized input the model will see
    total_budget_tokens: int       # total_input + expected_output = peak KV
    source: str                    # "tokenizer" or "heuristic" or "none"

    def to_receipt(self) -> dict:
        return asdict(self)


@dataclass
class RouteTokenAccount:
    """Token-accurate budget for an entire route (all steps)."""
    steps: List[TokenAccount]
    peak_step_tokens: int          # max(total_budget_tokens) across steps
    total_output_tokens: int       # sum of expected outputs across steps

    def to_receipt(self) -> dict:
        return {
            "steps": [s.to_receipt() for s in self.steps],
            "peak_step_tokens": self.peak_step_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


# ---------------------------------------------------------------------------
# Accountant
# ---------------------------------------------------------------------------

_ZERO_ACCOUNT = TokenAccount(
    query_tokens=0, memory_tokens=0, prompt_tokens=0,
    context_tokens=0, expected_output_tokens=0,
    total_input_tokens=0, total_budget_tokens=0,
    source="none",
)


class TokenAccountant:
    """Counts actual tokens using model tokenizers.

    Falls back to heuristic (len // 4, min 1) when the tokenizer for a
    target model is unavailable.  The ``source`` field on every
    :class:`TokenAccount` tells the caller which path was used.
    """

    def __init__(self, model_manager=None):
        self._tokenizers: Dict = {}
        self._model_manager = model_manager
        self._fallback_count = 0

    # ── tokenizer access ──

    def _get_tokenizer(self, target):
        """Get cached tokenizer for a model target. Returns None on failure."""
        if target in self._tokenizers:
            return self._tokenizers[target]

        # Try model manager first (already has CPU-cached tokenizers)
        if self._model_manager is not None:
            try:
                tok = self._model_manager._get_tokenizer(target)
                self._tokenizers[target] = tok
                return tok
            except Exception:
                pass

        # Try direct load
        try:
            from transformers import AutoTokenizer
            from .model_manager import MODEL_CONFIGS
            cfg = MODEL_CONFIGS[target]
            tok = AutoTokenizer.from_pretrained(str(cfg["model_dir"]))
            self._tokenizers[target] = tok
            return tok
        except Exception as e:
            log.warning("TokenAccountant: no tokenizer for %s: %s", target, e)
            return None

    # ── counting primitives ──

    def count_tokens(self, text: str, target) -> tuple:
        """Count tokens in *text* for *target* model.

        Returns ``(count, source)`` where source is ``"tokenizer"`` or
        ``"heuristic"``.
        """
        if not text:
            return (0, "tokenizer")

        tokenizer = self._get_tokenizer(target)
        if tokenizer is not None:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
                return (len(ids), "tokenizer")
            except Exception:
                pass

        # Heuristic fallback
        self._fallback_count += 1
        return (max(len(text) // 4, 1), "heuristic")

    # ── per-step accounting ──

    def account_step(
        self,
        lobe_name: str,
        query: str,
        memory_context: str = "",
        prev_step_output: str = "",
        max_tokens_override: Optional[int] = None,
    ) -> TokenAccount:
        """Build token-accurate budget for a single lobe step.

        Args:
            lobe_name: Registered lobe name (e.g. ``"coder"``).
            query: Raw user query text.
            memory_context: Recalled memory text (may be empty).
            prev_step_output: Output from prior route step (may be empty).
            max_tokens_override: Override the lobe's default max_tokens.

        Returns:
            :class:`TokenAccount` with all fields populated.
        """
        from .lobe_scheduler import get_lobe

        lobe = get_lobe(lobe_name)
        target = lobe.capability.model_target

        if target is None:
            return _ZERO_ACCOUNT

        # ── count individual components ──
        query_count, q_src = self.count_tokens(query, target)
        memory_count, _ = self.count_tokens(memory_context, target)

        context_text = (prev_step_output + "\n\n") if prev_step_output else ""
        context_count, _ = self.count_tokens(context_text, target)

        # Prompt template chrome (fixed text around {task}/{context})
        empty_prompt = lobe.format_prompt(task="", context="")
        prompt_count, _ = self.count_tokens(empty_prompt, target)

        # Expected generation budget
        expected_output = max_tokens_override or lobe.capability.max_tokens

        # ── total input = full tokenized prompt as model will see it ──
        # Tokenizing the concatenated string avoids token-boundary skew
        # that summing per-component counts would introduce.
        full_prompt = lobe.format_prompt(task=query, context=context_text)
        if memory_context:
            full_prompt = memory_context + "\n" + full_prompt
        full_input_count, src = self.count_tokens(full_prompt, target)

        total_budget = full_input_count + expected_output

        return TokenAccount(
            query_tokens=query_count,
            memory_tokens=memory_count,
            prompt_tokens=prompt_count,
            context_tokens=context_count,
            expected_output_tokens=expected_output,
            total_input_tokens=full_input_count,
            total_budget_tokens=total_budget,
            source=src,
        )

    # ── per-route accounting ──

    def account_route(
        self,
        route,
        query: str,
        memory_context: str = "",
        max_tokens: Optional[int] = None,
    ) -> RouteTokenAccount:
        """Account for all steps in a route.

        For multi-step routes, intermediate outputs are estimated
        (max_tokens * 4 chars) to compute downstream context cost.
        """
        from .lobe_scheduler import get_lobe

        step_accounts: List[TokenAccount] = []
        prev_output_estimate = ""

        for route_step in route.steps:
            lobe = get_lobe(route_step.lobe_name)

            # Memory only applies to first model-bearing step
            mem = memory_context if not route_step.input_from else ""

            context = prev_output_estimate if route_step.input_from else ""

            account = self.account_step(
                route_step.lobe_name,
                query,
                memory_context=mem,
                prev_step_output=context,
                max_tokens_override=max_tokens,
            )
            step_accounts.append(account)

            # Estimate output text for next step's context
            out_tokens = max_tokens or lobe.capability.max_tokens
            prev_output_estimate = "x" * (out_tokens * 4)

        peak = max((s.total_budget_tokens for s in step_accounts), default=0)
        total_out = sum(s.expected_output_tokens for s in step_accounts)

        return RouteTokenAccount(
            steps=step_accounts,
            peak_step_tokens=peak,
            total_output_tokens=total_out,
        )

    # ── single-query shortcut ──

    def estimate_context(
        self,
        query: str,
        target,
        route_name: str = "direct_plan",
        memory_context: str = "",
        max_tokens: Optional[int] = None,
    ) -> int:
        """Return total_budget_tokens for a single primary step.

        This is the drop-in replacement for::

            estimated_context = max(len(query) // 4, 32)

        Returns the peak KV token count (input + output).
        """
        # Map route_name to its primary lobe
        lobe_name = self._lobe_for_route(route_name, target)

        account = self.account_step(
            lobe_name,
            query,
            memory_context=memory_context,
            max_tokens_override=max_tokens,
        )
        return account.total_budget_tokens

    def estimate_context_with_account(
        self,
        query: str,
        target,
        route_name: str = "direct_plan",
        memory_context: str = "",
        max_tokens: Optional[int] = None,
    ) -> tuple:
        """Like :meth:`estimate_context` but also returns the TokenAccount.

        Returns ``(total_budget_tokens, token_account)``.
        """
        lobe_name = self._lobe_for_route(route_name, target)
        account = self.account_step(
            lobe_name,
            query,
            memory_context=memory_context,
            max_tokens_override=max_tokens,
        )
        return (account.total_budget_tokens, account)

    @staticmethod
    def _lobe_for_route(route_name: str, target) -> str:
        """Determine the primary (most expensive) lobe for a route.

        For budget-gate purposes, we check the lobe that will use the
        most KV cache — typically the model-bearing lobe that receives
        the most context.
        """
        from .query_classifier import ModelTarget

        # Routes with known primary lobes
        _ROUTE_PRIMARY = {
            "direct_code": "coder",
            "code_verify": "coder",        # coder has more context than verifier
            "direct_plan": "planner",
            "plan_verify": "planner",
            "plan_code_verify": "coder",   # coder gets planner output as context
        }

        primary = _ROUTE_PRIMARY.get(route_name)
        if primary:
            return primary

        # Symbolic / unknown routes: use target model to pick lobe
        if target == ModelTarget.QWEN_CODER:
            return "coder"
        return "planner"

    # ── diagnostics ──

    def status(self) -> dict:
        """Current accountant state."""
        return {
            "loaded_tokenizers": [
                t.value if hasattr(t, "value") else str(t)
                for t in self._tokenizers.keys()
            ],
            "fallback_count": self._fallback_count,
            "schema": SCHEMA,
        }
