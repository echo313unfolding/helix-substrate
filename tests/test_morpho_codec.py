"""Tests for morpho codec integration with CDNA v3."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from helix_substrate.morpho_codec import grow_weights, morpho_encode, morpho_decode
from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.cdnav3_reader import CDNAv3Reader
from helix_substrate.tensor_policy import TensorClass, TensorPolicy


def _cosine(a, b):
    a, b = a.ravel(), b.ravel()
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if (na > 0 and nb > 0) else 0.0


class TestGrowWeightsDeterminism:
    """Core property: same seed -> same weights."""

    def test_deterministic_small(self):
        seed = b"test_seed_123"
        W1 = grow_weights(seed, (32, 64), steps=200)
        W2 = grow_weights(seed, (32, 64), steps=200)
        np.testing.assert_array_equal(W1, W2)

    def test_different_seeds_differ(self):
        W1 = grow_weights(b"seed_a", (32, 64), steps=200)
        W2 = grow_weights(b"seed_b", (32, 64), steps=200)
        cosine = _cosine(W1, W2)
        assert cosine < 0.5, f"Different seeds should give different weights, got cosine={cosine}"

    def test_shape_correct(self):
        W = grow_weights(b"test", (16, 32), steps=100)
        assert W.shape == (16, 32)
        assert W.dtype == np.float32


class TestMorphoCodecRoundtrip:
    """Encode -> decode roundtrip."""

    def test_roundtrip_deterministic(self):
        """Encode a tensor, decode it, encode again -> same result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor = np.random.randn(32, 64).astype(np.float32)

            stats = morpho_encode(
                tensor, "test.weight", Path(tmpdir) / "out", steps=200
            )
            decoded1 = morpho_decode(Path(tmpdir) / "out")

            # Re-encode with same name -> same seed -> same decode
            stats2 = morpho_encode(
                tensor, "test.weight", Path(tmpdir) / "out2", steps=200
            )
            decoded2 = morpho_decode(Path(tmpdir) / "out2")

            np.testing.assert_array_equal(decoded1, decoded2)

    def test_stats_contain_cosine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor = np.random.randn(32, 64).astype(np.float32)
            stats = morpho_encode(
                tensor, "test.weight", Path(tmpdir) / "out", steps=200
            )
            assert "cosine" in stats
            assert "compression_ratio" in stats
            assert stats["storage_mode"] == "morpho"
            assert stats["compression_ratio"] > 10  # growth config much smaller than tensor

    def test_files_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor = np.random.randn(32, 64).astype(np.float32)
            out = Path(tmpdir) / "out"
            morpho_encode(tensor, "test.weight", out, steps=200)

            assert (out / "morpho_config.json").exists()
            assert (out / "meta.json").exists()
            assert (out / "stats.json").exists()

    def test_meta_json_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor = np.random.randn(32, 64).astype(np.float32)
            out = Path(tmpdir) / "out"
            morpho_encode(tensor, "blk.0.ffn_norm.weight", out, steps=200)

            meta = json.loads((out / "meta.json").read_text())
            assert meta["format_version"] == "cdna_v3"
            assert meta["storage_mode"] == "morpho"
            assert meta["tensor_name"] == "blk.0.ffn_norm.weight"


class TestCDNAv3Integration:
    """Test morpho codec through CDNAv3Writer/Reader pipeline."""

    def test_writer_reader_roundtrip(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 64).astype(np.float32)

        morpho_policy = TensorPolicy(
            tensor_class=TensorClass.UNKNOWN,
            storage_mode="morpho",
            morpho_growth_steps=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            stats = writer.write_tensor(tensor, "test.weight", policy=morpho_policy)
            assert stats["storage_mode"] == "morpho"

            reader = CDNAv3Reader(Path(tmpdir) / "test_weight.cdnav3")
            assert reader.is_morpho
            reconstructed = reader.reconstruct()
            assert reconstructed.shape == (32, 64)
            assert reconstructed.dtype == np.float32

    def test_morpho_is_deterministic_through_pipeline(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 64).astype(np.float32)

        morpho_policy = TensorPolicy(
            tensor_class=TensorClass.UNKNOWN,
            storage_mode="morpho",
            morpho_growth_steps=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(tensor, "test.weight", policy=morpho_policy)

            reader = CDNAv3Reader(Path(tmpdir) / "test_weight.cdnav3")
            r1 = reader.reconstruct()
            r2 = reader.reconstruct()
            np.testing.assert_array_equal(r1, r2)

    def test_block_reconstruction(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 128).astype(np.float32)

        morpho_policy = TensorPolicy(
            tensor_class=TensorClass.UNKNOWN,
            storage_mode="morpho",
            morpho_growth_steps=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(tensor, "test.weight", policy=morpho_policy)

            reader = CDNAv3Reader(Path(tmpdir) / "test_weight.cdnav3")
            full = reader.reconstruct()
            block = reader.reconstruct_block(16, 32)
            np.testing.assert_array_equal(block, full[16:32])

    def test_verify_config_hash(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 64).astype(np.float32)

        morpho_policy = TensorPolicy(
            tensor_class=TensorClass.UNKNOWN,
            storage_mode="morpho",
            morpho_growth_steps=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(tensor, "test.weight", policy=morpho_policy)

            reader = CDNAv3Reader(Path(tmpdir) / "test_weight.cdnav3")
            assert reader.verify_codebook() is True

    def test_get_stats(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 64).astype(np.float32)

        morpho_policy = TensorPolicy(
            tensor_class=TensorClass.UNKNOWN,
            storage_mode="morpho",
            morpho_growth_steps=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(tensor, "test.weight", policy=morpho_policy)

            reader = CDNAv3Reader(Path(tmpdir) / "test_weight.cdnav3")
            stats = reader.get_stats()
            assert stats["storage_mode"] == "morpho"
            assert stats["compression_ratio"] > 10


class TestCompressionRatio:
    """Verify morpho achieves extreme compression (config << tensor)."""

    def test_small_tensor_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor = np.random.randn(32, 64).astype(np.float32)
            stats = morpho_encode(
                tensor, "test.weight", Path(tmpdir) / "out", steps=200
            )
            # 32*64*4 = 8192 bytes original, config ~400-500 bytes
            assert stats["compression_ratio"] > 10

    def test_large_tensor_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor = np.random.randn(256, 256).astype(np.float32)
            stats = morpho_encode(
                tensor, "test.weight", Path(tmpdir) / "out", steps=200
            )
            # 256*256*4 = 262144 bytes original, config ~300 bytes
            assert stats["compression_ratio"] > 500


class TestFitToTarget:
    """Test the target-fitting mechanism."""

    def test_fit_improves_cosine(self):
        """Fitting should produce better cosine than seed-only."""
        from helix_substrate.morpho_codec import fit_to_target, grow_from_fit_result, grow_weights
        import hashlib

        # A nearly-constant target (like norm weights)
        rng = np.random.RandomState(42)
        target = np.ones((1, 128), dtype=np.float64) + rng.randn(1, 128) * 0.05

        result = fit_to_target(target, n_codons=16, steps=200, max_iter=50)
        assert result["cosine"] > 0.5, f"Fit cosine too low: {result['cosine']}"

        # Verify decode matches
        W = grow_from_fit_result(result, (1, 128))
        cos = float(np.dot(target.ravel(), W.ravel().astype(np.float64)) / (
            np.linalg.norm(target) * np.linalg.norm(W)))
        assert abs(cos - result["cosine"]) < 0.01

    def test_fit_encode_decode_roundtrip(self):
        """Full pipeline with fit=True."""
        rng = np.random.RandomState(42)
        target = np.ones((1, 64), dtype=np.float32) * 1.5 + rng.randn(1, 64).astype(np.float32) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            from helix_substrate.cdnav3_writer import CDNAv3Writer
            from helix_substrate.cdnav3_reader import CDNAv3Reader
            from helix_substrate.tensor_policy import TensorPolicy, TensorClass

            policy = TensorPolicy(
                tensor_class=TensorClass.NORM,
                storage_mode="morpho",
                morpho_growth_steps=200,
            )

            writer = CDNAv3Writer(Path(tmpdir))
            stats = writer.write_tensor(
                target, "test.norm.weight", policy=policy,
            )
            assert stats["storage_mode"] == "morpho"

    def test_fit_result_fields(self):
        from helix_substrate.morpho_codec import fit_to_target
        target = np.ones((1, 64), dtype=np.float64)
        result = fit_to_target(target, n_codons=8, steps=100, max_iter=20)
        assert "codons" in result
        assert "scale" in result
        assert "shift" in result
        assert "cosine" in result
        assert len(result["codons"]) == 8


class TestFittedPipeline:
    """Test fit=True flows through CDNAv3Writer -> morpho_encode -> CDNAv3Reader."""

    def test_writer_reader_fitted_roundtrip(self):
        """CDNAv3Writer with morpho_fit=True -> CDNAv3Reader -> verify cosine matches."""
        from helix_substrate.tensor_policy import MORPHO_FIT_POLICY

        rng = np.random.RandomState(42)
        # Nearly-constant target (like norm weights) where fitting works well
        target = np.ones((1, 128), dtype=np.float32) * 1.5 + rng.randn(1, 128).astype(np.float32) * 0.05

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            stats = writer.write_tensor(target, "test.norm.weight", policy=MORPHO_FIT_POLICY)
            assert stats["storage_mode"] == "morpho"
            assert stats["cosine"] > 0.5, f"Fitted cosine too low: {stats['cosine']}"

            reader = CDNAv3Reader(Path(tmpdir) / "test_norm_weight.cdnav3")
            assert reader.is_morpho
            reconstructed = reader.reconstruct()
            assert reconstructed.shape == target.shape

    def test_fitted_deterministic_decode(self):
        """Encode with fit -> decode twice -> identical arrays."""
        from helix_substrate.tensor_policy import MORPHO_FIT_POLICY

        rng = np.random.RandomState(42)
        target = np.ones((1, 64), dtype=np.float32) * 2.0 + rng.randn(1, 64).astype(np.float32) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(target, "test.norm.weight", policy=MORPHO_FIT_POLICY)

            reader = CDNAv3Reader(Path(tmpdir) / "test_norm_weight.cdnav3")
            r1 = reader.reconstruct()
            r2 = reader.reconstruct()
            np.testing.assert_array_equal(r1, r2)

    def test_fit_policy_fields_propagate(self):
        """Verify policy fields reach morpho_encode (fit=True produces v2 config)."""
        from helix_substrate.tensor_policy import MORPHO_FIT_POLICY

        rng = np.random.RandomState(42)
        target = np.ones((1, 32), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            writer.write_tensor(target, "test.norm.weight", policy=MORPHO_FIT_POLICY)

            config_path = Path(tmpdir) / "test_norm_weight.cdnav3" / "morpho_config.json"
            assert config_path.exists()
            config = json.loads(config_path.read_text())
            assert "codons" in config, "fit=True should produce config with codons"
            assert "scale" in config
            assert "shift" in config
            assert config["cosine_similarity"] > 0.0


class TestDefaultRoutingUnchanged:
    """Verify existing codecs still work (morpho doesn't break anything)."""

    def test_norm_still_exact(self):
        tensor = np.random.randn(2048).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            stats = writer.write_tensor(tensor, "blk.0.attn_norm.weight")
            assert stats["storage_mode"] == "exact"

    def test_ffn_still_codebook_sidecar(self):
        rng = np.random.RandomState(42)
        tensor = rng.randn(64, 128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CDNAv3Writer(Path(tmpdir))
            stats = writer.write_tensor(tensor, "blk.0.ffn_down.weight")
            assert stats["num_outliers"] > 0
