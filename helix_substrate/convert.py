"""
HuggingFace model converter for helix-substrate.

Converts HuggingFace models to CDNA format for streaming inference.

Usage:
    helix-substrate convert mistralai/Mistral-7B-v0.1 --output ./mistral-cdna
"""

import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def quantize_tensor(
    tensor: np.ndarray,
    n_clusters: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize tensor to uint8 indices + codebook.

    Uses uniform quantization (fast) rather than k-means (slow but better).
    For production, could use k-means for higher quality.
    """
    tensor = tensor.astype(np.float32)

    # Uniform codebook spanning the value range
    vmin, vmax = tensor.min(), tensor.max()
    codebook = np.linspace(vmin, vmax, n_clusters).astype(np.float32)

    # Quantize in chunks to avoid OOM on large tensors
    flat = tensor.reshape(-1)
    indices = np.zeros(flat.shape[0], dtype=np.uint8)

    chunk_size = 1024 * 1024  # 1M elements at a time
    for i in range(0, len(flat), chunk_size):
        chunk = flat[i:i + chunk_size]
        # Find nearest codebook entry for each value
        indices[i:i + chunk_size] = np.argmin(
            np.abs(chunk[:, None] - codebook[None, :]), axis=1
        ).astype(np.uint8)

    indices = indices.reshape(tensor.shape)
    return indices, codebook


def convert_huggingface_model(
    model_id: str,
    output_dir: Path,
    block_rows: int = 64,
    codec: str = "brotli",
    skip_embeddings: bool = False,
    skip_lm_head: bool = False,
    token: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Convert a HuggingFace model to CDNA format.

    Args:
        model_id: HuggingFace model ID (e.g., "mistralai/Mistral-7B-v0.1")
        output_dir: Directory to save CDNA files
        block_rows: Block size for streaming (smaller = less memory, more overhead)
        codec: Compression codec ("brotli" or "zstd")
        skip_embeddings: Don't convert embedding layers (they're accessed by index, not matmul)
        skip_lm_head: Don't convert the final lm_head projection
        token: HuggingFace API token for gated models
        verbose: Print progress

    Returns:
        Manifest dict with conversion stats
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        from safetensors import safe_open
    except ImportError:
        raise ImportError(
            "HuggingFace converter requires: pip install helix-substrate[hf]\n"
            "Or: pip install huggingface_hub safetensors"
        )

    from helix_substrate.cdna_reader import CDNAv2Writer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading {model_id}...")

    # Download model files
    model_path = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "config.json", "tokenizer*"],
        token=token,
    )
    model_path = Path(model_path)

    # Find safetensor files
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {model_id}")

    if verbose:
        print(f"Found {len(safetensor_files)} safetensor file(s)")

    # Copy config
    config_src = model_path / "config.json"
    if config_src.exists():
        config_dst = output_dir / "config.json"
        config_dst.write_text(config_src.read_text())

    # Process each safetensor file
    manifest = {
        "model_id": model_id,
        "format": "cdna_v2",
        "codec": codec,
        "block_rows": block_rows,
        "tensors": {},
    }

    writer = CDNAv2Writer(codec=codec, block_rows=block_rows)

    total_original_bytes = 0
    total_compressed_bytes = 0
    tensor_count = 0

    for sf_file in sorted(safetensor_files):
        if verbose:
            print(f"\nProcessing {sf_file.name}...")

        with safe_open(sf_file, framework="numpy") as f:
            for tensor_name in f.keys():
                # Skip patterns
                if skip_embeddings and "embed" in tensor_name.lower():
                    if verbose:
                        print(f"  Skipping embedding: {tensor_name}")
                    continue

                if skip_lm_head and "lm_head" in tensor_name.lower():
                    if verbose:
                        print(f"  Skipping lm_head: {tensor_name}")
                    continue

                # Load tensor
                tensor = f.get_tensor(tensor_name)
                original_bytes = tensor.nbytes
                total_original_bytes += original_bytes

                # Skip 1D tensors (biases, norms) - they're tiny
                if tensor.ndim == 1:
                    # Save as-is in a simple format
                    np.save(output_dir / f"{tensor_name}.npy", tensor)
                    manifest["tensors"][tensor_name] = {
                        "format": "npy",
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "bytes": original_bytes,
                    }
                    if verbose:
                        print(f"  {tensor_name}: {tensor.shape} (1D, saved as npy)")
                    continue

                # Reshape to 2D if needed
                original_shape = tensor.shape
                if tensor.ndim > 2:
                    tensor = tensor.reshape(-1, tensor.shape[-1])

                # Quantize
                indices, codebook = quantize_tensor(tensor)

                # Write CDNA
                safe_name = tensor_name.replace("/", "_").replace(".", "_")
                cdna_path = output_dir / f"{safe_name}.cdna2.hxz"

                stats = writer.write(
                    indices=indices,
                    codebook=codebook,
                    output_path=cdna_path,
                    tensor_name=tensor_name,
                )

                compressed_bytes = cdna_path.stat().st_size
                total_compressed_bytes += compressed_bytes
                ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0

                manifest["tensors"][tensor_name] = {
                    "format": "cdna_v2",
                    "file": str(cdna_path.name),
                    "original_shape": list(original_shape),
                    "cdna_shape": list(tensor.shape),
                    "original_bytes": original_bytes,
                    "compressed_bytes": compressed_bytes,
                    "compression_ratio": round(ratio, 2),
                }

                tensor_count += 1
                if verbose:
                    print(f"  {tensor_name}: {original_shape} → {ratio:.1f}x compression")

    # Write manifest
    manifest["stats"] = {
        "tensor_count": tensor_count,
        "original_bytes": total_original_bytes,
        "compressed_bytes": total_compressed_bytes,
        "overall_ratio": round(total_original_bytes / total_compressed_bytes, 2) if total_compressed_bytes > 0 else 0,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Conversion complete: {model_id}")
        print(f"  Tensors converted: {tensor_count}")
        print(f"  Original size: {total_original_bytes / 1e9:.2f} GB")
        print(f"  Compressed size: {total_compressed_bytes / 1e9:.2f} GB")
        print(f"  Overall ratio: {manifest['stats']['overall_ratio']:.1f}x")
        print(f"  Output: {output_dir}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to CDNA format for streaming inference"
    )
    parser.add_argument(
        "model_id",
        help="HuggingFace model ID (e.g., mistralai/Mistral-7B-v0.1)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for CDNA files"
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=64,
        help="Block size for streaming (default: 64)"
    )
    parser.add_argument(
        "--codec",
        choices=["brotli", "zstd"],
        default="brotli",
        help="Compression codec (default: brotli)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Don't convert embedding layers"
    )
    parser.add_argument(
        "--skip-lm-head",
        action="store_true",
        help="Don't convert the final lm_head projection"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token for gated models"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
