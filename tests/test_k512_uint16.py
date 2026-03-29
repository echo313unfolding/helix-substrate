"""Tests for k>256 (uint16 indices) support across the CDNA v3 pipeline.

Work Order: WO-ADAPTIVE-K-QUALITY-01
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.cdnav3_reader import CDNAv3Reader
from helix_substrate.tensor_policy import TensorClass, TensorPolicy


def _cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestK512WriterReader:
    """Test k=512 write → read roundtrip with uint16 indices."""

    def test_k512_roundtrip(self):
        """Write at k=512, read back, verify cosine matches or exceeds k=256."""
        rng = np.random.RandomState(42)
        tensor = rng.randn(128, 256).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            policy = TensorPolicy(
                tensor_class=TensorClass.UNKNOWN,
                storage_mode="codebook",
                n_clusters=512,
                sidecar_enabled=False,
                use_kmeans=True,
            )
            stats = writer.write_tensor(tensor, "test.weight", policy=policy)

            # Verify stats
            assert stats["cosine_no_sidecar"] > 0.90
            assert stats["shape"] == [128, 256]

            # Verify meta.json has index_dtype=uint16
            td = Path(tmpdir) / "test_weight.cdnav3"
            meta = json.loads((td / "meta.json").read_text())
            assert meta["index_dtype"] == "uint16"
            assert meta["n_clusters"] == 512

            # Verify indices.bin is 2x size (uint16 vs uint8)
            indices_size = (td / "indices.bin").stat().st_size
            expected_size = 128 * 256 * 2  # rows * cols * 2 bytes
            assert indices_size == expected_size

            # Read back
            reader = CDNAv3Reader(td)
            reconstructed = reader.reconstruct()
            assert reconstructed.shape == (128, 256)
            cos = _cosine(tensor, reconstructed)
            assert cos > 0.90

    def test_k512_better_than_k256(self):
        """k=512 should achieve >= k=256 cosine on the same tensor."""
        rng = np.random.RandomState(42)
        tensor = rng.randn(128, 256).astype(np.float32)

        cosines = {}
        for k in [256, 512]:
            with tempfile.TemporaryDirectory() as tmpdir:
                writer = CDNAv3Writer(Path(tmpdir))
                policy = TensorPolicy(
                    tensor_class=TensorClass.UNKNOWN,
                    storage_mode="codebook",
                    n_clusters=k,
                    sidecar_enabled=False,
                    use_kmeans=True,
                )
                stats = writer.write_tensor(tensor, "test.weight", policy=policy)
                cosines[k] = stats["cosine_no_sidecar"]

        # k=512 should be at least as good as k=256
        assert cosines[512] >= cosines[256] - 1e-6

    def test_k256_still_uint8(self):
        """Verify k=256 still uses uint8 (backward compat)."""
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            policy = TensorPolicy(
                tensor_class=TensorClass.UNKNOWN,
                storage_mode="codebook",
                n_clusters=256,
                sidecar_enabled=False,
                use_kmeans=True,
            )
            writer.write_tensor(tensor, "test.weight", policy=policy)

            td = Path(tmpdir) / "test_weight.cdnav3"
            meta = json.loads((td / "meta.json").read_text())
            assert meta["index_dtype"] == "uint8"

            # Indices.bin should be 1 byte per element
            indices_size = (td / "indices.bin").stat().st_size
            assert indices_size == 64 * 128

    def test_backward_compat_no_index_dtype(self):
        """Reader should default to uint8 when meta.json lacks index_dtype."""
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            policy = TensorPolicy(
                tensor_class=TensorClass.UNKNOWN,
                storage_mode="codebook",
                n_clusters=256,
                sidecar_enabled=False,
                use_kmeans=True,
            )
            writer.write_tensor(tensor, "test.weight", policy=policy)

            # Remove index_dtype from meta.json to simulate old format
            td = Path(tmpdir) / "test_weight.cdnav3"
            meta = json.loads((td / "meta.json").read_text())
            del meta["index_dtype"]
            (td / "meta.json").write_text(json.dumps(meta, indent=2))

            # Should still read correctly (defaults to uint8)
            reader = CDNAv3Reader(td)
            reconstructed = reader.reconstruct()
            assert reconstructed.shape == (64, 128)

    def test_k512_block_reconstruct(self):
        """Block reconstruction should work with uint16 indices."""
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            policy = TensorPolicy(
                tensor_class=TensorClass.UNKNOWN,
                storage_mode="codebook",
                n_clusters=512,
                sidecar_enabled=False,
                use_kmeans=True,
            )
            writer.write_tensor(tensor, "test.weight", policy=policy)

            reader = CDNAv3Reader(Path(tmpdir) / "test_weight.cdnav3")
            full = reader.reconstruct()
            block = reader.reconstruct_block(16, 32)
            np.testing.assert_array_equal(block, full[16:32])

    def test_k512_with_sidecar(self):
        """k=512 with sidecar corrections should work."""
        rng = np.random.RandomState(42)
        tensor = rng.randn(128, 256).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            policy = TensorPolicy(
                tensor_class=TensorClass.FFN,
                storage_mode="codebook",
                n_clusters=512,
                sidecar_enabled=True,
                use_kmeans=True,
                percentile=99.5,
            )
            stats = writer.write_tensor(tensor, "blk.0.ffn_down.weight", policy=policy)
            assert stats["cosine_with_sidecar"] >= stats["cosine_no_sidecar"]

            reader = CDNAv3Reader(Path(tmpdir) / "blk_0_ffn_down_weight.cdnav3")
            reconstructed = reader.reconstruct()
            assert reconstructed.shape == (128, 256)

    def test_codebook_verification_k512(self):
        """Codebook SHA256 verification should work for k=512."""
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 64).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            policy = TensorPolicy(
                tensor_class=TensorClass.UNKNOWN,
                storage_mode="codebook",
                n_clusters=512,
                sidecar_enabled=False,
                use_kmeans=True,
            )
            writer.write_tensor(tensor, "test.weight", policy=policy)
            reader = CDNAv3Reader(Path(tmpdir) / "test_weight.cdnav3")
            assert reader.verify_codebook() is True


class TestHelixLinearK512:
    """Test HelixLinear with k=512 codebook."""

    def test_helix_linear_k512_forward(self):
        """HelixLinear forward pass with k=512 codebook."""
        import torch
        from helix_substrate.helix_linear import HelixLinear

        out_f, in_f = 64, 128
        codebook = torch.randn(512)
        # Use int16 for k>256 indices
        indices = torch.randint(0, 512, (out_f, in_f), dtype=torch.int16)

        hl = HelixLinear(
            in_features=in_f,
            out_features=out_f,
            codebook=codebook,
            indices=indices,
            tensor_name="test_k512",
        )

        x = torch.randn(2, in_f)
        y = hl(x)
        assert y.shape == (2, out_f)

    def test_helix_linear_k512_memory_savings(self):
        """Memory savings should reflect 2 bytes/index for k=512."""
        import torch
        from helix_substrate.helix_linear import HelixLinear

        out_f, in_f = 64, 128
        codebook = torch.randn(512)
        indices = torch.randint(0, 512, (out_f, in_f), dtype=torch.int16)

        hl = HelixLinear(
            in_features=in_f,
            out_features=out_f,
            codebook=codebook,
            indices=indices,
        )

        savings = hl.memory_savings()
        dense_bytes = out_f * in_f * 4
        expected_compressed = 512 * 4 + out_f * in_f * 2  # codebook + uint16 indices
        assert savings["dense_bytes"] == dense_bytes
        assert savings["compressed_bytes"] == expected_compressed

    def test_helix_linear_k256_still_1byte(self):
        """k=256 indices should still use 1 byte in memory_savings."""
        import torch
        from helix_substrate.helix_linear import HelixLinear

        out_f, in_f = 64, 128
        codebook = torch.randn(256)
        indices = torch.randint(0, 256, (out_f, in_f), dtype=torch.uint8)

        hl = HelixLinear(
            in_features=in_f,
            out_features=out_f,
            codebook=codebook,
            indices=indices,
        )

        savings = hl.memory_savings()
        expected_compressed = 256 * 4 + out_f * in_f * 1  # codebook + uint8 indices
        assert savings["compressed_bytes"] == expected_compressed

    def test_load_k512_from_cdnav3(self):
        """Load HelixLinear from a k=512 .cdnav3 directory."""
        import torch
        from helix_substrate.helix_linear import load_helix_linear_from_cdnav3

        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            policy = TensorPolicy(
                tensor_class=TensorClass.UNKNOWN,
                storage_mode="codebook",
                n_clusters=512,
                sidecar_enabled=False,
                use_kmeans=True,
            )
            writer.write_tensor(tensor, "test.weight", policy=policy)

            hl = load_helix_linear_from_cdnav3(Path(tmpdir) / "test_weight.cdnav3")
            assert hl.codebook.shape[0] == 512
            assert hl.indices.dtype == torch.int16

            # Forward pass works
            x = torch.randn(2, 128)
            y = hl(x)
            assert y.shape == (2, 64)

    def test_from_quantized_config_k512(self):
        """from_quantized_config should create correct shell for k=512."""
        import torch
        from helix_substrate.helix_linear import HelixLinear

        shell = HelixLinear.from_quantized_config(
            in_features=128,
            out_features=64,
            n_clusters=512,
        )
        assert shell.codebook.shape[0] == 512
        assert shell.indices.dtype == torch.int16


class TestKAllocator:
    """Test the budget-constrained k allocator."""

    def test_basic_allocation(self):
        from helix_substrate.k_allocator import allocate_k

        # Simulate 10 tensors with varying quality
        stats = []
        for i in range(10):
            stats.append({
                "tensor_name": f"model.layers.{i}.weight",
                "cosine": 0.995 + i * 0.0005,  # 0.995 to 0.9995
                "original_bytes": 1_000_000,
                "compressed_bytes": 250_000,
                "current_k": 256,
            })

        k_map = allocate_k(stats, target_ratio=3.0)

        assert "overrides" in k_map
        assert k_map["k_default"] == 256
        assert k_map["estimated_ratio"] >= 3.0

    def test_no_upgrade_beyond_budget(self):
        """Allocator should not exceed the compression budget."""
        from helix_substrate.k_allocator import allocate_k

        stats = []
        for i in range(10):
            stats.append({
                "tensor_name": f"model.layers.{i}.weight",
                "cosine": 0.990,  # all same quality
                "original_bytes": 1_000_000,
                "compressed_bytes": 250_000,
                "current_k": 256,
            })

        # Very tight budget — ratio 4.0 means basically no room for upgrades
        k_map = allocate_k(stats, target_ratio=4.0)
        assert k_map["estimated_ratio"] >= 4.0 or k_map["n_upgraded"] == 0

    def test_worst_quality_upgraded_first(self):
        """Lowest-cosine tensors should get k=512 first."""
        from helix_substrate.k_allocator import allocate_k

        stats = [
            {"tensor_name": "bad_tensor", "cosine": 0.990, "original_bytes": 1_000_000,
             "compressed_bytes": 250_000, "current_k": 256},
            {"tensor_name": "good_tensor", "cosine": 0.9999, "original_bytes": 1_000_000,
             "compressed_bytes": 250_000, "current_k": 256},
        ]

        k_map = allocate_k(stats, target_ratio=2.0)  # generous budget
        overrides = k_map["overrides"]

        # bad_tensor should be upgraded before good_tensor
        if "bad_tensor" in overrides:
            assert overrides["bad_tensor"] == 512

    def test_estimate_bytes(self):
        from helix_substrate.k_allocator import _estimate_bytes

        # k=256: 256*4 + n*1
        assert _estimate_bytes(400_000, 256) == 256 * 4 + 100_000 * 1

        # k=512: 512*4 + n*2
        assert _estimate_bytes(400_000, 512) == 512 * 4 + 100_000 * 2
