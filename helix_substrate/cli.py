"""
helix-substrate CLI.

Usage:
    helix-substrate convert <model_id> --output <dir>
    helix-substrate bench --rows 16384 --cols 16384
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="helix-substrate",
        description="Model weight compression and streaming decode library"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert HuggingFace model to CDNA format"
    )
    convert_parser.add_argument(
        "model_id",
        help="HuggingFace model ID (e.g., mistralai/Mistral-7B-v0.1)"
    )
    convert_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for CDNA files"
    )
    convert_parser.add_argument(
        "--block-rows",
        type=int,
        default=64,
        help="Block size for streaming (default: 64)"
    )
    convert_parser.add_argument(
        "--codec",
        choices=["brotli", "zstd"],
        default="brotli",
        help="Compression codec (default: brotli)"
    )
    convert_parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Don't convert embedding layers"
    )
    convert_parser.add_argument(
        "--skip-lm-head",
        action="store_true",
        help="Don't convert the final lm_head projection"
    )
    convert_parser.add_argument(
        "--token",
        help="HuggingFace API token for gated models"
    )
    convert_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    # Bench command
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run memory benchmark"
    )
    bench_parser.add_argument(
        "--rows",
        type=int,
        default=4096,
        help="Weight matrix rows (default: 4096)"
    )
    bench_parser.add_argument(
        "--cols",
        type=int,
        default=4096,
        help="Weight matrix cols (default: 4096)"
    )
    bench_parser.add_argument(
        "--block-rows",
        type=int,
        default=64,
        help="Block size for streaming (default: 64)"
    )

    args = parser.parse_args()

    if args.command == "convert":
        from helix_substrate.convert import convert_huggingface_model
        from pathlib import Path

        convert_huggingface_model(
            model_id=args.model_id,
            output_dir=Path(args.output),
            block_rows=args.block_rows,
            codec=args.codec,
            skip_embeddings=args.skip_embeddings,
            skip_lm_head=args.skip_lm_head,
            token=args.token,
            verbose=not args.quiet,
        )

    elif args.command == "bench":
        # Import and run benchmark
        import tempfile
        import tracemalloc
        from pathlib import Path
        import numpy as np
        from helix_substrate.cdna_reader import CDNAv2Writer, CDNAv2Reader

        M, K = args.rows, args.cols
        block_rows = args.block_rows

        print(f"Benchmark: {M}x{K} weight matrix, block_rows={block_rows}")
        print("=" * 60)

        rng = np.random.RandomState(42)
        W = rng.randn(M, K).astype(np.float32)
        print(f"W shape: {W.shape}, size: {W.nbytes / 1024 / 1024:.2f} MB")

        # Quantize
        codebook = np.linspace(W.min(), W.max(), 256).astype(np.float32)
        indices = np.zeros((M, K), dtype=np.uint8)
        for i in range(0, M, 256):
            end_i = min(i + 256, M)
            chunk = W[i:end_i]
            indices[i:end_i] = np.argmin(
                np.abs(chunk[:, :, None] - codebook[None, None, :]), axis=2
            ).astype(np.uint8)
        del W

        # Write CDNA
        with tempfile.NamedTemporaryFile(suffix=".cdna2.hxz", delete=False) as f:
            cdna_path = Path(f.name)

        writer = CDNAv2Writer(codec="brotli", block_rows=block_rows)
        writer.write(indices=indices, codebook=codebook, output_path=cdna_path)

        X = rng.randn(8, M).astype(np.float32)
        print(f"X shape: {X.shape}")
        print()

        # Standard matmul
        print("Standard matmul (dequantize full W, then multiply):")
        tracemalloc.start()
        tracemalloc.reset_peak()
        W_full = codebook[indices]
        Y_standard = X @ W_full
        _, peak_standard = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del W_full
        print(f"  Peak memory: {peak_standard / 1024 / 1024:.2f} MB")

        # Streaming matmul
        print("Streaming matmul (block by block):")
        tracemalloc.start()
        tracemalloc.reset_peak()
        reader = CDNAv2Reader(cdna_path)
        Y_stream = np.zeros((X.shape[0], K), dtype=np.float32)
        for block_indices, start_row in reader.stream_blocks():
            end_row = start_row + block_indices.shape[0]
            W_block = codebook[block_indices]
            Y_stream += X[:, start_row:end_row] @ W_block
        _, peak_streaming = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  Peak memory: {peak_streaming / 1024 / 1024:.2f} MB")

        # Verify
        cosine = np.dot(Y_standard.flat, Y_stream.flat) / (
            np.linalg.norm(Y_standard) * np.linalg.norm(Y_stream)
        )
        ratio = peak_standard / peak_streaming if peak_streaming > 0 else 0

        print()
        print(f"Cosine similarity: {cosine:.6f}")
        print(f"Memory ratio: {ratio:.2f}x")

        cdna_path.unlink(missing_ok=True)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
