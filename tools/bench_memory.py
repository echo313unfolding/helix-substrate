#!/usr/bin/env python3
"""
Benchmark: Peak memory usage for streaming matmul vs standard numpy matmul.

This script produces an actual measurement, not an estimate.
"""

import tempfile
import tracemalloc
from pathlib import Path

import numpy as np

from helix_substrate.cdna_reader import CDNAv2Writer, CDNAv2Reader


def measure_standard_matmul(
    X: np.ndarray,
    indices: np.ndarray,
    codebook: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Standard matmul - dequantizes full W into memory, then multiplies.

    This measures the real-world case: you have indices + codebook on disk,
    you load and dequantize the FULL matrix, then compute.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()

    # Dequantize full W into memory - THIS is what streaming avoids
    W = codebook[indices]

    # Compute matmul
    Y = X @ W

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return Y, peak


def measure_streaming_matmul(
    X: np.ndarray,
    cdna_path: Path,
    codebook: np.ndarray,
    block_rows: int
) -> tuple[np.ndarray, float]:
    """Streaming matmul - processes W block by block."""
    tracemalloc.start()
    tracemalloc.reset_peak()

    reader = CDNAv2Reader(cdna_path)
    rows, cols = reader.rows, reader.cols

    # Output accumulator
    Y = np.zeros((X.shape[0], cols), dtype=np.float32)

    # Stream blocks
    for block_indices, start_row in reader.stream_blocks():
        end_row = start_row + block_indices.shape[0]

        # Dequantize block
        W_block = codebook[block_indices]  # [block_rows, cols]

        # Accumulate: Y += X[:, start_row:end_row] @ W_block
        Y += X[:, start_row:end_row] @ W_block

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return Y, peak


def run_benchmark(M: int = 4096, K: int = 4096, block_rows: int = 64):
    """
    Benchmark streaming vs standard matmul on MxK weight matrix.

    Args:
        M: Weight matrix rows (input dimension)
        K: Weight matrix cols (output dimension)
        block_rows: Block size for streaming
    """
    print(f"Benchmark: {M}x{K} weight matrix, block_rows={block_rows}")
    print("=" * 60)

    rng = np.random.RandomState(42)

    # Create weight matrix
    W = rng.randn(M, K).astype(np.float32)
    W_bytes = W.nbytes
    print(f"W shape: {W.shape}, size: {W_bytes / 1024 / 1024:.2f} MB")

    # Quantize to 8-bit indices + codebook (chunk-wise to avoid OOM)
    codebook = np.linspace(W.min(), W.max(), 256).astype(np.float32)
    indices = np.zeros((M, K), dtype=np.uint8)
    chunk_size = 256  # Quantize 256 rows at a time
    for i in range(0, M, chunk_size):
        end_i = min(i + chunk_size, M)
        chunk = W[i:end_i]
        indices[i:end_i] = np.argmin(
            np.abs(chunk[:, :, None] - codebook[None, None, :]), axis=2
        ).astype(np.uint8)
    del W  # Free original W, we only need indices + codebook now

    # Write CDNA file
    with tempfile.NamedTemporaryFile(suffix=".cdna2.hxz", delete=False) as f:
        cdna_path = Path(f.name)

    writer = CDNAv2Writer(codec="brotli", block_rows=block_rows)
    writer.write(indices=indices, codebook=codebook, output_path=cdna_path)

    # Input activations (batch of 8 vectors)
    X = rng.randn(8, M).astype(np.float32)
    print(f"X shape: {X.shape}, size: {X.nbytes / 1024:.2f} KB")
    print()

    # Benchmark standard matmul (dequantizes full W, then multiplies)
    print("Standard matmul (dequantize full W, then multiply):")
    Y_standard, peak_standard = measure_standard_matmul(X, indices, codebook)
    print(f"  Peak memory: {peak_standard / 1024 / 1024:.2f} MB")

    # Benchmark streaming matmul
    print("Streaming matmul (block by block):")
    Y_streaming, peak_streaming = measure_streaming_matmul(X, cdna_path, codebook, block_rows)
    print(f"  Peak memory: {peak_streaming / 1024 / 1024:.2f} MB")

    # Verify correctness
    cosine = np.dot(Y_standard.flat, Y_streaming.flat) / (
        np.linalg.norm(Y_standard) * np.linalg.norm(Y_streaming)
    )
    print()
    print(f"Cosine similarity: {cosine:.6f}")

    # Calculate ratio
    if peak_streaming > 0:
        ratio = peak_standard / peak_streaming
        print(f"Memory ratio (standard/streaming): {ratio:.2f}x")
    else:
        print("Warning: streaming peak was 0, cannot compute ratio")

    # Cleanup
    cdna_path.unlink(missing_ok=True)

    print()
    print("=" * 60)
    print("RECEIPT:")
    print(f"  Matrix size: {M}x{K}")
    print(f"  Block size: {block_rows}")
    print(f"  Standard peak: {peak_standard / 1024 / 1024:.2f} MB")
    print(f"  Streaming peak: {peak_streaming / 1024 / 1024:.2f} MB")
    if peak_streaming > 0:
        print(f"  Measured ratio: {ratio:.2f}x")
    print(f"  Correctness: cosine={cosine:.6f}")

    return {
        "matrix_size": (M, K),
        "block_rows": block_rows,
        "standard_peak_bytes": peak_standard,
        "streaming_peak_bytes": peak_streaming,
        "ratio": ratio if peak_streaming > 0 else None,
        "cosine": cosine,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark streaming vs standard matmul memory usage")
    parser.add_argument("--rows", type=int, default=4096, help="Weight matrix rows")
    parser.add_argument("--cols", type=int, default=4096, help="Weight matrix cols")
    parser.add_argument("--block-rows", type=int, default=64, help="Block size for streaming")

    args = parser.parse_args()

    run_benchmark(M=args.rows, K=args.cols, block_rows=args.block_rows)
