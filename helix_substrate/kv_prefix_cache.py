"""
KV prefix reuse for immediate follow-up turns.

When the user asks a follow-up question and the prompt prefix is unchanged
(same system prompt + same context), we can reuse the KV cache from the
previous turn instead of re-encoding the entire prefix.

This saves the prefill cost on the shared prefix, which is typically 80-90%
of the prompt for multi-turn conversations with injected context.

Safety contract:
    - Only reuse if prefix tokens match EXACTLY (no fuzzy matching).
    - Clear all slots on model swap (different model = different KV space).
    - Multi-slot LRU cache (default 4 slots) for concurrent conversation contexts.
    - Keyed by (model_target, prefix_hash) — prefix includes system prompt + context.

Work Order: WO-AI-OS-RUNTIME-01, Phase 2 → Track A-lite → Track A2
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch


@dataclass
class KVPrefixState:
    """Cached KV state for one prefix slot."""
    prefix_hash: str
    prefix_len: int                    # number of tokens in the prefix
    model_target: str
    past_key_values: object            # HF DynamicCache or tuple of (K, V) pairs
    created_at: float
    last_used_at: float
    reuse_count: int = 0


@dataclass
class KVCacheStats:
    """Statistics for receipt emission."""
    reused: int = 0
    rebuilt: int = 0
    evicted: int = 0
    cleared: int = 0
    total_prefix_tokens_saved: int = 0

    def as_dict(self) -> dict:
        return {
            "kv_reused": self.reused,
            "kv_rebuilt": self.rebuilt,
            "kv_evicted": self.evicted,
            "kv_cleared": self.cleared,
            "total_prefix_tokens_saved": self.total_prefix_tokens_saved,
            "reuse_rate": round(self.reused / max(1, self.reused + self.rebuilt), 3),
        }


def _hash_token_ids(token_ids: torch.Tensor) -> str:
    """Hash token IDs for prefix comparison."""
    raw = token_ids.cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:32]


class KVPrefixCache:
    """Multi-slot LRU KV prefix cache for follow-up turn reuse.

    Holds up to ``max_slots`` cached prefix KV states keyed by
    ``(model_target, prefix_hash)``.  On capacity overflow the
    least-recently-used slot is evicted.

    Usage::

        cache = KVPrefixCache(max_slots=4)

        # On each turn:
        prefix_ids = tokenizer.encode(prefix_text)
        cached_kv, prefix_len = cache.try_reuse(prefix_ids, model_target)
        if cached_kv is not None:
            # Feed only the NEW tokens (after prefix) to model, with past_key_values
            ...
        else:
            # Full encode from scratch, then store:
            cache.store(prefix_ids, model_target, past_key_values)
    """

    def __init__(self, max_slots: int = 4):
        self.max_slots = max_slots
        self._slots: Dict[str, KVPrefixState] = {}
        self.stats = KVCacheStats()

    # ── internal ──

    @staticmethod
    def _slot_key(model_target: str, prefix_hash: str) -> str:
        return f"{model_target}:{prefix_hash}"

    def _evict_lru(self) -> None:
        """Evict the least-recently-used slot.  Frees GPU memory."""
        if not self._slots:
            return
        lru_key = min(self._slots, key=lambda k: self._slots[k].last_used_at)
        self._slots[lru_key].past_key_values = None
        del self._slots[lru_key]
        self.stats.evicted += 1

    # ── public API (unchanged from single-slot) ──

    def try_reuse(
        self,
        prefix_ids: torch.Tensor,
        model_target: str,
    ) -> Tuple[Optional[object], int]:
        """Try to reuse cached KV for this prefix.

        Args:
            prefix_ids: Token IDs for the prompt prefix (1D or 2D tensor).
            model_target: Current model target string.

        Returns:
            (past_key_values, prefix_len) if reusable, else (None, 0).
        """
        prefix_hash = _hash_token_ids(prefix_ids)
        key = self._slot_key(model_target, prefix_hash)
        state = self._slots.get(key)

        if state is None:
            self.stats.rebuilt += 1
            return None, 0

        # Clone to prevent HF DynamicCache in-place mutation from corrupting
        # our cached copy.
        cloned = self._clone_kv(state.past_key_values)
        if cloned is None:
            # KV became invalid — remove the slot
            state.past_key_values = None
            del self._slots[key]
            self.stats.rebuilt += 1
            return None, 0

        state.last_used_at = time.time()
        state.reuse_count += 1
        self.stats.reused += 1
        self.stats.total_prefix_tokens_saved += state.prefix_len
        return cloned, state.prefix_len

    def store(
        self,
        prefix_ids: torch.Tensor,
        model_target: str,
        past_key_values: object,
    ) -> None:
        """Store KV state for potential reuse on next turn.

        Detaches and clones the KV tensors to prevent mutation.
        If the cache is full, the LRU slot is evicted first.
        """
        cloned_kv = self._clone_kv(past_key_values)
        if cloned_kv is None:
            return

        prefix_hash = _hash_token_ids(prefix_ids)
        key = self._slot_key(model_target, prefix_hash)

        # Evict LRU if at capacity and this is a new key
        if key not in self._slots and len(self._slots) >= self.max_slots:
            self._evict_lru()

        now = time.time()
        prefix_len = prefix_ids.shape[-1] if prefix_ids.dim() > 0 else len(prefix_ids)

        self._slots[key] = KVPrefixState(
            prefix_hash=prefix_hash,
            prefix_len=prefix_len,
            model_target=model_target,
            past_key_values=cloned_kv,
            created_at=now,
            last_used_at=now,
        )

    def clear(self, reason: str = "manual") -> int:
        """Clear all cached KV slots.  Returns count cleared.

        Call this on model swap — old model's KV tensors occupy GPU memory
        that the new model needs.
        """
        n = len(self._slots)
        for state in self._slots.values():
            state.past_key_values = None
        self._slots.clear()
        self.stats.cleared += n
        return n

    @property
    def slot_count(self) -> int:
        """Current number of occupied slots."""
        return len(self._slots)

    def slot_summary(self) -> list:
        """Per-slot summary for receipts / debugging."""
        out = []
        for key, state in self._slots.items():
            out.append({
                "key": key,
                "prefix_hash": state.prefix_hash[:12],
                "prefix_len": state.prefix_len,
                "model": state.model_target,
                "reuse_count": state.reuse_count,
                "age_s": round(time.time() - state.created_at, 1),
            })
        return sorted(out, key=lambda x: x["key"])

    @staticmethod
    def _clone_kv(past_key_values) -> Optional[object]:
        """Deep-clone KV cache to prevent in-place mutation.

        Handles DynamicCache (HF >= 4.36, including 4.57+ which dropped
        key_cache/value_cache in favor of __getitem__) and tuple-of-tuples.
        """
        if past_key_values is None:
            return None

        # DynamicCache — works with both old (key_cache/value_cache) and
        # new (layers + __getitem__) APIs by using len() + __getitem__ + update()
        try:
            from transformers.cache_utils import DynamicCache
            if isinstance(past_key_values, DynamicCache):
                cloned = DynamicCache()
                for layer_idx in range(len(past_key_values)):
                    k, v = past_key_values[layer_idx]
                    cloned.update(k.detach().clone(), v.detach().clone(), layer_idx)
                return cloned
        except (ImportError, AttributeError, TypeError):
            pass

        # Tuple-of-tuples legacy format
        if isinstance(past_key_values, tuple):
            try:
                return tuple(
                    tuple(t.detach().clone() for t in layer_kv)
                    for layer_kv in past_key_values
                )
            except (AttributeError, TypeError):
                pass

        return None
