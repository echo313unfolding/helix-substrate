"""Tests for CDNA v3 model manifest builder."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.build_manifest_v3 import build_manifest, validate_manifest


class TestBuildManifest:
    def test_synthetic_three_tensors(self):
        rng = np.random.RandomState(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            writer = CDNAv3Writer(base)

            writer.write_tensor(rng.randn(64, 128).astype(np.float32), "blk.0.ffn_down.weight")
            writer.write_tensor(rng.randn(64, 128).astype(np.float32), "blk.0.attn_q.weight")
            writer.write_tensor(rng.randn(64).astype(np.float32), "blk.0.attn_norm.weight")

            manifest = build_manifest(base)
            assert manifest["tensor_count"] == 3
            assert "blk.0.ffn_down.weight" in manifest["tensors"]
            assert "blk.0.attn_q.weight" in manifest["tensors"]
            assert "blk.0.attn_norm.weight" in manifest["tensors"]

    def test_schema_fields(self):
        rng = np.random.RandomState(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            writer = CDNAv3Writer(base)
            writer.write_tensor(rng.randn(32, 64).astype(np.float32), "test.weight")

            manifest = build_manifest(base, source_info={"gguf": "test.gguf"})
            assert manifest["schema"] == "cdna_v3_manifest"
            assert "created_utc" in manifest
            assert manifest["format_version"] == 3
            assert manifest["source"] == {"gguf": "test.gguf"}
            assert "stats" in manifest

    def test_aggregate_stats(self):
        rng = np.random.RandomState(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            writer = CDNAv3Writer(base)
            writer.write_tensor(rng.randn(64, 128).astype(np.float32), "blk.0.ffn_down.weight")
            writer.write_tensor(rng.randn(32, 64).astype(np.float32), "blk.1.ffn_down.weight")

            manifest = build_manifest(base)
            stats = manifest["stats"]
            assert stats["tensor_count"] == 2
            assert stats["total_original_bytes"] > 0
            assert stats["total_compressed_bytes"] > 0
            assert stats["overall_ratio"] > 1.0


class TestValidateManifest:
    def test_valid_manifest_passes(self):
        rng = np.random.RandomState(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            writer = CDNAv3Writer(base)
            writer.write_tensor(rng.randn(32, 64).astype(np.float32), "blk.0.ffn_down.weight")
            writer.write_tensor(rng.randn(64).astype(np.float32), "blk.0.attn_norm.weight")

            build_manifest(base)
            result = validate_manifest(base / "manifest_v3.json")
            assert result["valid"] is True
            assert result["tensor_count"] == 2

    def test_missing_dir_fails(self):
        rng = np.random.RandomState(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            writer = CDNAv3Writer(base)
            writer.write_tensor(rng.randn(32, 64).astype(np.float32), "test.weight")

            build_manifest(base)

            # Delete the tensor directory
            import shutil
            shutil.rmtree(base / "test_weight.cdnav3")

            result = validate_manifest(base / "manifest_v3.json")
            assert result["valid"] is False
            assert len(result["errors"]) > 0
