"""Tests for CDNA v3 writer and reader."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.cdnav3_reader import CDNAv3Reader
from helix_substrate.tensor_policy import TensorClass, TensorPolicy, get_default_policy


def _cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestRoundtrip:
    def test_synthetic_no_sidecar(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(128, 256).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            policy = TensorPolicy(
                tensor_class=TensorClass.UNKNOWN,
                storage_mode="codebook",
                n_clusters=256,
                sidecar_enabled=False,
                use_kmeans=True,
            )
            stats = writer.write_tensor(tensor, "test.weight", policy=policy)
            assert stats["cosine_no_sidecar"] > 0.90

            reader = CDNAv3Reader(Path(tmpdir) / "test_weight.cdnav3")
            reconstructed = reader.reconstruct()
            assert reconstructed.shape == (128, 256)
            assert _cosine(tensor, reconstructed) > 0.90

    def test_synthetic_with_sidecar(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(128, 256).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            stats = writer.write_tensor(tensor, "blk.0.ffn_down.weight")

            assert stats["cosine_with_sidecar"] > 0.99
            assert stats["num_outliers"] > 0

            reader = CDNAv3Reader(Path(tmpdir) / "blk_0_ffn_down_weight.cdnav3")
            reconstructed = reader.reconstruct()
            assert _cosine(tensor, reconstructed) > 0.99


class TestDirectoryStructure:
    def test_files_exist(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(tensor, "blk.0.attn_q.weight")

            td = Path(tmpdir) / "blk_0_attn_q_weight.cdnav3"
            assert td.exists()
            assert (td / "meta.json").exists()
            assert (td / "codebook.npy").exists()
            assert (td / "indices.bin").exists()
            assert (td / "stats.json").exists()

    def test_meta_json_contents(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 64).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(tensor, "blk.2.ffn_down.weight")

            td = Path(tmpdir) / "blk_2_ffn_down_weight.cdnav3"
            meta = json.loads((td / "meta.json").read_text())
            assert meta["format_version"] == "cdna_v3"
            assert meta["tensor_name"] == "blk.2.ffn_down.weight"
            assert meta["shape"] == [32, 64]
            assert meta["n_clusters"] == 256
            assert "codebook_sha256" in meta
            assert meta["tensor_class"] == "ffn"


class TestCodebookVerification:
    def test_verify_passes(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 64).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(tensor, "test.weight")
            reader = CDNAv3Reader(Path(tmpdir) / "test_weight.cdnav3")
            assert reader.verify_codebook() is True


class Test1DTensor:
    def test_saved_as_npy(self):
        tensor = np.random.randn(2048).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            stats = writer.write_tensor(tensor, "blk.0.attn_norm.weight")
            assert stats["storage_mode"] == "exact"
            assert (Path(tmpdir) / "blk_0_attn_norm_weight.npy").exists()


class TestReconstructBlock:
    def test_block_matches_full(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(tensor, "blk.0.ffn_down.weight")

            reader = CDNAv3Reader(Path(tmpdir) / "blk_0_ffn_down_weight.cdnav3")
            full = reader.reconstruct()

            # Block [16:32] should match slice of full
            block = reader.reconstruct_block(16, 32)
            assert block.shape == (16, 128)
            np.testing.assert_array_equal(block, full[16:32])

    def test_first_block(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(tensor, "blk.0.ffn_up.weight")

            reader = CDNAv3Reader(Path(tmpdir) / "blk_0_ffn_up_weight.cdnav3")
            full = reader.reconstruct()
            block = reader.reconstruct_block(0, 16)
            np.testing.assert_array_equal(block, full[0:16])


class TestAutoPolicy:
    def test_ffn_gets_sidecar(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            stats = writer.write_tensor(tensor, "blk.0.ffn_down.weight")
            assert stats["num_outliers"] > 0

    def test_norm_gets_exact(self):
        tensor = np.random.randn(2048).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            stats = writer.write_tensor(tensor, "blk.0.attn_norm.weight")
            assert stats["storage_mode"] == "exact"


# Integration test with real TinyLlama tensor
GGUF_PATH = Path("/home/voidstr3m33/helix-cdc/models/gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf")


@pytest.mark.skipif(not GGUF_PATH.exists(), reason="TinyLlama GGUF not found")
class TestRealTinyLlama:
    def test_single_tensor_roundtrip(self):
        pytest.importorskip("gguf")
        from gguf import GGUFReader
        from gguf.quants import dequantize

        reader = GGUFReader(str(GGUF_PATH))
        target_name = "blk.2.ffn_down.weight"

        # Find and dequantize tensor
        tensor_data = None
        for t in reader.tensors:
            if t.name == target_name:
                tensor_data = dequantize(t.data, t.tensor_type)
                tensor_data = tensor_data.reshape(t.shape)
                break

        assert tensor_data is not None, f"Tensor {target_name} not found"
        tensor_data = tensor_data.astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            stats = writer.write_tensor(
                tensor_data,
                target_name,
                source_artifact=str(GGUF_PATH),
            )

            print(f"\n  Shape: {tensor_data.shape}")
            print(f"  Original: {stats['original_bytes']} bytes")
            print(f"  Compressed: {stats['compressed_bytes']} bytes")
            print(f"  Ratio: {stats['compression_ratio']}x")
            print(f"  Cosine (no sidecar): {stats['cosine_no_sidecar']}")
            print(f"  Cosine (with sidecar): {stats['cosine_with_sidecar']}")
            print(f"  Outliers: {stats['num_outliers']}")

            # Verify roundtrip
            td = Path(tmpdir) / "blk_2_ffn_down_weight.cdnav3"
            v3reader = CDNAv3Reader(td)
            reconstructed = v3reader.reconstruct()

            assert reconstructed.shape == tensor_data.shape
            cosine = _cosine(tensor_data, reconstructed)
            print(f"  Verified cosine: {cosine:.6f}")
            assert cosine > 0.95, f"Cosine too low: {cosine}"

            # Verify codebook hash
            assert v3reader.verify_codebook()

            # Verify stats
            s = v3reader.get_stats()
            assert s["tensor_name"] == target_name
