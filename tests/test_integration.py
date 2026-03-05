"""Integration tests for helix-substrate streaming functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_stream_matmul_small():
    """
    Verify Y = X @ W via streaming matches direct matmul.

    This tests the core claim: compute Y = X @ W without loading full W.
    """
    from helix_substrate.cdna_reader import CDNAv2Writer
    from helix_substrate.stream_matmul import stream_xw_from_cdna

    rng = np.random.RandomState(42)

    # Small tensor for fast test
    M, K = 64, 128
    W = rng.randn(M, K).astype(np.float32)

    # Create k-means quantized version
    # Simple uniform codebook for test
    codebook = np.linspace(W.min(), W.max(), 256).astype(np.float32)

    # Quantize W to indices
    indices = np.argmin(np.abs(W[:, :, None] - codebook[None, None, :]), axis=2).astype(np.uint8)

    # Write CDNAv2 file
    with tempfile.NamedTemporaryFile(suffix=".cdna2.hxz", delete=False) as f:
        cdna_path = Path(f.name)

    try:
        writer = CDNAv2Writer(codec="brotli", block_rows=16)
        writer.write(
            indices=indices,
            codebook=codebook,
            output_path=cdna_path,
            tensor_name="test_weight",
        )

        # Input activations
        X = rng.randn(4, M).astype(np.float32)  # [seq, M]

        # Compute via streaming
        Y_stream, receipt = stream_xw_from_cdna(X, cdna_path, emit_receipt=True)

        # Reconstruct W from codebook for ground truth
        W_quantized = codebook[indices]
        Y_direct = X @ W_quantized

        # Should match closely (both use quantized W)
        assert Y_stream.shape == Y_direct.shape, f"Shape mismatch: {Y_stream.shape} vs {Y_direct.shape}"

        cosine = np.dot(Y_stream.flat, Y_direct.flat) / (np.linalg.norm(Y_stream) * np.linalg.norm(Y_direct))
        assert cosine > 0.999, f"Streaming matmul diverged: cosine={cosine}"

        # Receipt should have valid stats
        assert len(receipt.blocks_touched) > 0
        assert receipt.tracemalloc_peak_mb >= 0

    finally:
        cdna_path.unlink(missing_ok=True)


def test_stream_matmul_transpose():
    """Verify Y = X @ W^T mode works for GQA-style projections."""
    from helix_substrate.cdna_reader import CDNAv2Writer
    from helix_substrate.stream_matmul import stream_xw_from_cdna

    rng = np.random.RandomState(123)

    # W stored as [rows=out_dim, cols=in_dim], compute X @ W^T
    out_dim, in_dim = 32, 64
    W = rng.randn(out_dim, in_dim).astype(np.float32)

    codebook = np.linspace(W.min(), W.max(), 256).astype(np.float32)
    indices = np.argmin(np.abs(W[:, :, None] - codebook[None, None, :]), axis=2).astype(np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".cdna2.hxz", delete=False) as f:
        cdna_path = Path(f.name)

    try:
        writer = CDNAv2Writer(codec="brotli", block_rows=8)
        writer.write(indices=indices, codebook=codebook, output_path=cdna_path)

        # X has shape [seq, in_dim], result should be [seq, out_dim]
        X = rng.randn(3, in_dim).astype(np.float32)

        Y_stream, receipt = stream_xw_from_cdna(X, cdna_path, transpose_w=True)

        W_quantized = codebook[indices]
        Y_direct = X @ W_quantized.T

        assert Y_stream.shape == (3, out_dim), f"Wrong output shape: {Y_stream.shape}"

        cosine = np.dot(Y_stream.flat, Y_direct.flat) / (np.linalg.norm(Y_stream) * np.linalg.norm(Y_direct))
        assert cosine > 0.999, f"Transpose mode diverged: cosine={cosine}"

    finally:
        cdna_path.unlink(missing_ok=True)


def test_cdna_v2_block_streaming():
    """Verify CDNAv2 block streaming works correctly."""
    from helix_substrate.cdna_reader import CDNAv2Writer, CDNAv2Reader

    rng = np.random.RandomState(777)

    M, K = 128, 64
    W = rng.randn(M, K).astype(np.float32)

    codebook = np.linspace(-3.0, 3.0, 256).astype(np.float32)
    indices = np.argmin(np.abs(W[:, :, None] - codebook[None, None, :]), axis=2).astype(np.uint8)

    block_rows = 16
    num_blocks = (M + block_rows - 1) // block_rows

    with tempfile.NamedTemporaryFile(suffix=".cdna2.hxz", delete=False) as f:
        cdna_path = Path(f.name)

    try:
        writer = CDNAv2Writer(codec="brotli", block_rows=block_rows)
        writer.write(indices=indices, codebook=codebook, output_path=cdna_path)

        reader = CDNAv2Reader(cdna_path)

        # Verify header
        assert reader.rows == M
        assert reader.cols == K
        assert reader.num_blocks == num_blocks

        # Stream all blocks and verify reconstruction
        blocks_seen = 0
        for block_indices, start_row in reader.stream_blocks():
            end_row = min(start_row + block_rows, M)
            expected = indices[start_row:end_row]

            assert np.array_equal(block_indices, expected), f"Block starting at row {start_row} mismatch"
            blocks_seen += 1

        assert blocks_seen == num_blocks

    finally:
        cdna_path.unlink(missing_ok=True)


def test_full_pipeline_encode_stream_decode():
    """
    End-to-end test: quantize → write CDNAv2 → stream matmul → verify.

    This is the production workflow:
    1. Take a weight matrix W
    2. Quantize to codebook + indices
    3. Write to CDNAv2 format
    4. Stream compute Y = X @ W without loading full W
    5. Verify result against original
    """
    from helix_substrate.cdna_reader import CDNAv2Writer
    from helix_substrate.stream_matmul import stream_xw_from_cdna

    rng = np.random.RandomState(999)

    # Realistic-ish dimensions (scaled down)
    M, K = 256, 512
    W = rng.randn(M, K).astype(np.float32)

    # Step 1: Quantize (simple k-means-like: uniform codebook)
    codebook = np.linspace(W.min(), W.max(), 256).astype(np.float32)
    indices = np.argmin(np.abs(W[:, :, None] - codebook[None, None, :]), axis=2).astype(np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".cdna2.hxz", delete=False) as f:
        cdna_path = Path(f.name)

    try:
        # Step 2: Write CDNAv2
        writer = CDNAv2Writer(codec="brotli", block_rows=32)
        stats = writer.write(indices=indices, codebook=codebook, output_path=cdna_path)
        assert stats["shape"] == [M, K]

        # Step 3: Stream compute
        X = rng.randn(8, M).astype(np.float32)  # [batch, M]
        Y_stream, receipt = stream_xw_from_cdna(X, cdna_path)

        # Step 4: Verify against quantized W (not original - quantization is lossy)
        W_quantized = codebook[indices]
        Y_quantized = X @ W_quantized

        # Should match exactly since both use quantized W
        cosine = np.dot(Y_stream.flat, Y_quantized.flat) / (
            np.linalg.norm(Y_stream) * np.linalg.norm(Y_quantized)
        )
        assert cosine > 0.999, f"Pipeline diverged: cosine={cosine}"

        # Also check against original (will be lower due to quantization)
        Y_original = X @ W
        cosine_original = np.dot(Y_stream.flat, Y_original.flat) / (
            np.linalg.norm(Y_stream) * np.linalg.norm(Y_original)
        )
        assert cosine_original > 0.90, f"Quantization too lossy: cosine={cosine_original}"

        # Receipt should capture the operation
        assert receipt is not None
        assert len(receipt.blocks_touched) > 0

    finally:
        cdna_path.unlink(missing_ok=True)
