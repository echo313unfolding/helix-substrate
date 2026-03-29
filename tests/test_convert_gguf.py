"""Tests for convert_gguf.py — GGUF converter components."""

import tempfile
from pathlib import Path

import numpy as np
from helix_substrate.convert_gguf import quantize_tensor, _sha256_file


class TestQuantizeTensor:
    def test_basic_quantization(self):
        tensor = np.random.randn(32, 64).astype(np.float32)
        indices, codebook = quantize_tensor(tensor, n_clusters=256)
        assert indices.shape == tensor.shape
        assert indices.dtype == np.uint8
        assert codebook.shape == (256,)
        assert codebook.dtype == np.float32

    def test_indices_in_range(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        indices, codebook = quantize_tensor(tensor, n_clusters=128)
        assert indices.max() < 128
        assert indices.min() >= 0

    def test_codebook_spans_range(self):
        tensor = np.array([-5.0, 0.0, 5.0], dtype=np.float32).reshape(1, 3)
        indices, codebook = quantize_tensor(tensor, n_clusters=256)
        assert codebook[0] <= -4.9
        assert codebook[-1] >= 4.9

    def test_constant_tensor(self):
        tensor = np.ones((4, 4), dtype=np.float32) * 3.0
        indices, codebook = quantize_tensor(tensor, n_clusters=256)
        # All values the same -> indices should all be the same
        assert len(np.unique(indices)) <= 2  # rounding may give 1-2 unique

    def test_large_tensor_chunked(self):
        # > 1M elements to exercise chunking
        tensor = np.random.randn(2048, 1024).astype(np.float32)
        indices, codebook = quantize_tensor(tensor, n_clusters=256)
        assert indices.shape == tensor.shape

    def test_reconstruction_quality(self):
        tensor = np.random.randn(64, 128).astype(np.float32)
        indices, codebook = quantize_tensor(tensor, n_clusters=256)
        reconstructed = codebook[indices]
        # Uniform quantization with 256 levels should be decent
        mse = np.mean((tensor - reconstructed) ** 2)
        assert mse < 0.1  # not great but reasonable for uniform


class TestSha256File:
    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"hello world test data")
            path = Path(f.name)
        h1 = _sha256_file(path)
        h2 = _sha256_file(path)
        assert h1 == h2
        assert len(h1) == 64
        path.unlink()

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"data A")
            path_a = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"data B")
            path_b = Path(f.name)
        assert _sha256_file(path_a) != _sha256_file(path_b)
        path_a.unlink()
        path_b.unlink()
