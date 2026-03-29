"""Tests for kv_prefix_cache.py — KV prefix reuse for follow-up turns."""

import torch
from helix_substrate.kv_prefix_cache import (
    KVPrefixCache, KVPrefixState, KVCacheStats, _hash_token_ids,
)


class TestHashTokenIds:
    def test_deterministic(self):
        ids = torch.tensor([1, 2, 3, 4, 5])
        h1 = _hash_token_ids(ids)
        h2 = _hash_token_ids(ids)
        assert h1 == h2

    def test_different_ids_different_hash(self):
        h1 = _hash_token_ids(torch.tensor([1, 2, 3]))
        h2 = _hash_token_ids(torch.tensor([1, 2, 4]))
        assert h1 != h2

    def test_hash_length(self):
        h = _hash_token_ids(torch.tensor([1, 2, 3]))
        assert len(h) == 32


class TestKVPrefixCache:
    def _make_kv(self, n_layers=2, seq_len=10, hidden=16):
        """Create a fake tuple-of-tuples KV cache."""
        return tuple(
            (torch.randn(1, 2, seq_len, hidden), torch.randn(1, 2, seq_len, hidden))
            for _ in range(n_layers)
        )

    def test_store_and_reuse(self):
        cache = KVPrefixCache(max_slots=4)
        prefix_ids = torch.tensor([1, 2, 3, 4, 5])
        kv = self._make_kv()

        cache.store(prefix_ids, "tinyllama", kv)
        assert cache.slot_count == 1

        reused_kv, prefix_len = cache.try_reuse(prefix_ids, "tinyllama")
        assert reused_kv is not None
        assert prefix_len == 5
        assert cache.stats.reused == 1

    def test_miss_on_different_prefix(self):
        cache = KVPrefixCache(max_slots=4)
        prefix_ids = torch.tensor([1, 2, 3])
        kv = self._make_kv()
        cache.store(prefix_ids, "tinyllama", kv)

        other_ids = torch.tensor([4, 5, 6])
        reused_kv, _ = cache.try_reuse(other_ids, "tinyllama")
        assert reused_kv is None
        assert cache.stats.rebuilt == 1

    def test_miss_on_different_model(self):
        cache = KVPrefixCache(max_slots=4)
        prefix_ids = torch.tensor([1, 2, 3])
        kv = self._make_kv()
        cache.store(prefix_ids, "tinyllama", kv)

        reused_kv, _ = cache.try_reuse(prefix_ids, "qwen_coder")
        assert reused_kv is None

    def test_lru_eviction(self):
        cache = KVPrefixCache(max_slots=2)
        kv = self._make_kv()

        ids1 = torch.tensor([1, 1, 1])
        ids2 = torch.tensor([2, 2, 2])
        ids3 = torch.tensor([3, 3, 3])

        cache.store(ids1, "m", kv)
        cache.store(ids2, "m", kv)
        assert cache.slot_count == 2

        # Storing a 3rd should evict LRU (ids1)
        cache.store(ids3, "m", kv)
        assert cache.slot_count == 2
        assert cache.stats.evicted == 1

        # ids1 should be gone
        reused, _ = cache.try_reuse(ids1, "m")
        assert reused is None

        # ids3 should be present
        reused, _ = cache.try_reuse(ids3, "m")
        assert reused is not None

    def test_clear(self):
        cache = KVPrefixCache(max_slots=4)
        kv = self._make_kv()
        cache.store(torch.tensor([1, 2, 3]), "m", kv)
        cleared = cache.clear("model_swap")
        assert cleared == 1
        assert cache.slot_count == 0
        assert cache.stats.cleared == 1

    def test_reuse_returns_clone(self):
        """Reused KV must be a clone, not a reference to the stored copy."""
        cache = KVPrefixCache(max_slots=4)
        prefix_ids = torch.tensor([1, 2, 3])
        kv = self._make_kv()
        cache.store(prefix_ids, "m", kv)

        reused1, _ = cache.try_reuse(prefix_ids, "m")
        reused2, _ = cache.try_reuse(prefix_ids, "m")

        # Mutating reused1 should NOT affect reused2
        reused1[0][0].fill_(999.0)
        assert reused2[0][0].mean().item() != 999.0

    def test_stats_as_dict(self):
        cache = KVPrefixCache()
        d = cache.stats.as_dict()
        assert "kv_reused" in d
        assert "reuse_rate" in d
        assert d["reuse_rate"] == 0.0

    def test_slot_summary(self):
        cache = KVPrefixCache()
        kv = self._make_kv()
        cache.store(torch.tensor([1, 2, 3]), "tinyllama", kv)
        summary = cache.slot_summary()
        assert len(summary) == 1
        assert summary[0]["model"] == "tinyllama"
        assert summary[0]["prefix_len"] == 3
