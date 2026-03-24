"""
Short-TTL response cache for the local assistant runtime.

Keyed by:
    - normalized query text
    - target model (tinyllama / qwen_coder)
    - retrieval context hash (echo memory + web results)
    - generation mode (code / factual / graph)

Avoids re-generating identical answers within a configurable TTL window.
Thread-safe via simple dict + timestamp expiry (single-threaded REPL, but safe).

Work Order: WO-AI-OS-RUNTIME-01, Phase 2
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class CacheEntry:
    """Single cached response."""
    answer: str
    gen_time: float
    n_tokens: int
    created_at: float
    hit_count: int = 0


@dataclass
class CacheStats:
    """Cumulative cache statistics for receipts."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_saved_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hits / max(1, self.hits + self.misses), 3),
            "total_saved_ms": round(self.total_saved_ms, 1),
        }


def _normalize_query(query: str) -> str:
    """Normalize query for cache key: lowercase, collapse whitespace, strip punctuation."""
    q = query.lower().strip()
    q = re.sub(r'\s+', ' ', q)
    q = re.sub(r'[?.!,;:]+$', '', q)
    return q


def _make_cache_key(
    query: str,
    target_model: str,
    context_hash: str,
    generation_mode: str,
) -> str:
    """Build deterministic cache key from all inputs that affect output."""
    normalized = _normalize_query(query)
    raw = f"{normalized}|{target_model}|{context_hash}|{generation_mode}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def hash_context(context: str) -> str:
    """Hash retrieval context for cache key. Empty context → empty hash."""
    if not context:
        return "empty"
    return hashlib.sha256(context.encode()).hexdigest()[:16]


class ResponseCache:
    """Short-TTL response cache for local assistant.

    Args:
        ttl_seconds: Time-to-live for cache entries (default 300 = 5 minutes).
        max_entries: Maximum cache size (LRU eviction beyond this).
    """

    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 64):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    def get(
        self,
        query: str,
        target_model: str,
        context_hash: str,
        generation_mode: str,
    ) -> Optional[CacheEntry]:
        """Look up cached response. Returns None on miss or expiry."""
        key = _make_cache_key(query, target_model, context_hash, generation_mode)
        entry = self._cache.get(key)
        if entry is None:
            self.stats.misses += 1
            return None

        # Check TTL
        age = time.time() - entry.created_at
        if age > self.ttl:
            del self._cache[key]
            self.stats.misses += 1
            self.stats.evictions += 1
            return None

        entry.hit_count += 1
        self.stats.hits += 1
        self.stats.total_saved_ms += entry.gen_time * 1000
        return entry

    def put(
        self,
        query: str,
        target_model: str,
        context_hash: str,
        generation_mode: str,
        answer: str,
        gen_time: float,
        n_tokens: int,
    ) -> str:
        """Store a response in cache. Returns the cache key."""
        key = _make_cache_key(query, target_model, context_hash, generation_mode)

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_entries:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]
            self.stats.evictions += 1

        self._cache[key] = CacheEntry(
            answer=answer,
            gen_time=gen_time,
            n_tokens=n_tokens,
            created_at=time.time(),
        )
        return key

    def clear(self) -> int:
        """Clear all entries. Returns count cleared."""
        n = len(self._cache)
        self._cache.clear()
        return n

    def expire_stale(self) -> int:
        """Remove all expired entries. Returns count removed."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now - v.created_at > self.ttl]
        for k in expired:
            del self._cache[k]
        self.stats.evictions += len(expired)
        return len(expired)
