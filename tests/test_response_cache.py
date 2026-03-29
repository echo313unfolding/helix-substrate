"""Tests for response_cache.py — short-TTL LLM response cache."""

import time
from helix_substrate.response_cache import (
    ResponseCache, CacheEntry, CacheStats,
    _normalize_query, _make_cache_key, hash_context,
)


class TestNormalization:
    def test_normalize_basic(self):
        assert _normalize_query("  Hello World  ") == "hello world"

    def test_normalize_whitespace(self):
        assert _normalize_query("what   is\tthis") == "what is this"

    def test_normalize_trailing_punctuation(self):
        assert _normalize_query("what is this?") == "what is this"
        assert _normalize_query("hello!") == "hello"

    def test_same_key_for_equivalent_queries(self):
        k1 = _make_cache_key("What is Python?", "tinyllama", "ctx1", "factual")
        k2 = _make_cache_key("what is python", "tinyllama", "ctx1", "factual")
        assert k1 == k2

    def test_different_key_for_different_model(self):
        k1 = _make_cache_key("hello", "tinyllama", "ctx1", "factual")
        k2 = _make_cache_key("hello", "qwen_coder", "ctx1", "factual")
        assert k1 != k2

    def test_different_key_for_different_context(self):
        k1 = _make_cache_key("hello", "tinyllama", "ctx1", "factual")
        k2 = _make_cache_key("hello", "tinyllama", "ctx2", "factual")
        assert k1 != k2

    def test_hash_context_empty(self):
        assert hash_context("") == "empty"

    def test_hash_context_nonempty(self):
        h = hash_context("some retrieval context")
        assert len(h) == 16
        assert h != "empty"


class TestResponseCache:
    def test_put_and_get(self):
        cache = ResponseCache(ttl_seconds=60)
        cache.put("what is python", "tinyllama", "empty", "factual",
                  answer="A programming language", gen_time=0.5, n_tokens=10)
        entry = cache.get("what is python", "tinyllama", "empty", "factual")
        assert entry is not None
        assert entry.answer == "A programming language"
        assert entry.n_tokens == 10

    def test_miss(self):
        cache = ResponseCache()
        entry = cache.get("unknown query", "tinyllama", "empty", "factual")
        assert entry is None
        assert cache.stats.misses == 1

    def test_ttl_expiry(self):
        cache = ResponseCache(ttl_seconds=0.01)
        cache.put("q", "m", "c", "g", answer="a", gen_time=0.1, n_tokens=5)
        time.sleep(0.02)
        entry = cache.get("q", "m", "c", "g")
        assert entry is None
        assert cache.stats.evictions == 1

    def test_max_entries_eviction(self):
        cache = ResponseCache(max_entries=2)
        cache.put("q1", "m", "c", "g", answer="a1", gen_time=0.1, n_tokens=5)
        cache.put("q2", "m", "c", "g", answer="a2", gen_time=0.1, n_tokens=5)
        cache.put("q3", "m", "c", "g", answer="a3", gen_time=0.1, n_tokens=5)
        # Oldest (q1) should be evicted
        assert cache.get("q1", "m", "c", "g") is None
        assert cache.get("q3", "m", "c", "g") is not None

    def test_hit_count_increments(self):
        cache = ResponseCache()
        cache.put("q", "m", "c", "g", answer="a", gen_time=0.1, n_tokens=5)
        cache.get("q", "m", "c", "g")
        cache.get("q", "m", "c", "g")
        entry = cache.get("q", "m", "c", "g")
        assert entry.hit_count == 3

    def test_stats(self):
        cache = ResponseCache()
        cache.put("q", "m", "c", "g", answer="a", gen_time=0.5, n_tokens=5)
        cache.get("q", "m", "c", "g")  # hit
        cache.get("miss", "m", "c", "g")  # miss
        d = cache.stats.as_dict()
        assert d["hits"] == 1
        assert d["misses"] == 1
        assert d["hit_rate"] == 0.5

    def test_clear(self):
        cache = ResponseCache()
        cache.put("q", "m", "c", "g", answer="a", gen_time=0.1, n_tokens=5)
        cleared = cache.clear()
        assert cleared == 1
        assert cache.get("q", "m", "c", "g") is None

    def test_expire_stale(self):
        cache = ResponseCache(ttl_seconds=0.01)
        cache.put("q1", "m", "c", "g", answer="a1", gen_time=0.1, n_tokens=5)
        time.sleep(0.02)
        cache.put("q2", "m", "c", "g", answer="a2", gen_time=0.1, n_tokens=5)
        expired = cache.expire_stale()
        assert expired == 1  # q1 expired, q2 still alive
