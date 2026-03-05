"""
GGUF model converter for helix-substrate.

Converts GGUF models to CDNA format for streaming inference.

Usage:
    helix-substrate convert-gguf ./model.gguf --output ./model-cdna

Supports all quantized formats (Q8_0, Q4_K_M, Q5_K, F16, F32, etc.)
via the gguf library's dequantize function.
"""

import json
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np


def quantize_tensor(
    tensor: np.ndarray,
    n_clusters: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize tensor to uint8 indices + codebook.

    Uses fast uniform quantization (linear codebook spanning value range).
    This is much faster than k-means and good enough for most use cases.
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
        # Use searchsorted for speed (codebook is sorted/uniform)
        # Scale to [0, n_clusters-1] range
        normalized = (chunk - vmin) / (vmax - vmin + 1e-10)
        indices[i:i + chunk_size] = np.clip(
            (normalized * (n_clusters - 1)).astype(np.int32),
            0, n_clusters - 1
        ).astype(np.uint8)

    return indices.reshape(tensor.shape), codebook


def convert_gguf_model(
    gguf_path: Path,
    output_dir: Path,
    block_rows: int = 16,
    codec: str = "brotli",
    verbose: bool = True,
) -> dict:
    """
    Convert a GGUF model to CDNA format.

    Args:
        gguf_path: Path to GGUF file
        output_dir: Directory to save CDNA files
        block_rows: Block size for streaming (smaller = less memory, more overhead)
        codec: Compression codec ("brotli" or "zstd")
        verbose: Print progress

    Returns:
        Manifest dict with conversion stats
    """
    try:
        from gguf import GGUFReader
        from gguf.quants import dequantize
    except ImportError:
        raise ImportError(
            "GGUF converter requires: pip install gguf\n"
            "Or: pip install helix-substrate[gguf]"
        )

    from helix_substrate.cdna_reader import CDNAv2Writer

    gguf_path = Path(gguf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading GGUF: {gguf_path}")

    reader = GGUFReader(str(gguf_path))

    # Get model metadata
    metadata = {}
    for kv in reader.fields.values():
        if hasattr(kv, 'parts') and len(kv.parts) > 0:
            try:
                # Try to extract simple values
                if hasattr(kv.parts[-1], 'tolist'):
                    val = kv.parts[-1].tolist()
                    if isinstance(val, list) and len(val) == 1:
                        val = val[0]
                    metadata[kv.name] = val
            except:
                pass

    if verbose:
        print(f"Found {len(reader.tensors)} tensors")

    # Process tensors
    manifest = {
        "source_gguf": str(gguf_path.name),
        "source_sha256": _sha256_file(gguf_path),
        "format": "cdna_v2",
        "codec": codec,
        "block_rows": block_rows,
        "metadata": metadata,
        "tensors": {},
    }

    writer = CDNAv2Writer(codec=codec, block_rows=block_rows)

    total_original_bytes = 0
    total_compressed_bytes = 0
    tensor_count = 0

    for t in reader.tensors:
        tensor_name = t.name

        # Dequantize tensor
        # CRITICAL: Use shape from dequantized data, NOT t.shape (raw GGUF format)
        # The gguf library's dequantize() returns correct PyTorch layout [out, in]
        # while t.shape gives raw GGUF storage format which may differ
        try:
            tensor = dequantize(t.data, t.tensor_type)
        except Exception as e:
            # Fallback for F16/F32 that don't need dequantization
            tensor = t.data.copy()
            if hasattr(t.tensor_type, 'name'):
                if t.tensor_type.name == "F16":
                    tensor = tensor.view(np.float16).astype(np.float32)
                elif t.tensor_type.name == "F32":
                    tensor = tensor.view(np.float32)

        tensor = tensor.astype(np.float32)
        original_bytes = tensor.nbytes
        total_original_bytes += original_bytes

        # Skip 1D tensors (norms, biases) - save as npy
        if tensor.ndim == 1:
            np.save(output_dir / f"{tensor_name.replace('.', '_')}.npy", tensor)
            manifest["tensors"][tensor_name] = {
                "format": "npy",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": original_bytes,
            }
            if verbose:
                print(f"  {tensor_name}: {tensor.shape} (1D, saved as npy)")
            continue

        # Reshape to 2D if needed (rare, but handle gracefully)
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
            "quant_type": str(t.tensor_type.name) if hasattr(t.tensor_type, 'name') else "UNKNOWN",
            "original_bytes": int(original_bytes),
            "compressed_bytes": int(compressed_bytes),
            "compression_ratio": round(ratio, 2),
        }

        tensor_count += 1
        if verbose:
            print(f"  [{tensor_count}] {tensor_name}: {original_shape} → {ratio:.1f}x")

    # Write manifest
    manifest["stats"] = {
        "tensor_count": tensor_count,
        "original_bytes": int(total_original_bytes),
        "compressed_bytes": int(total_compressed_bytes),
        "overall_ratio": round(total_original_bytes / total_compressed_bytes, 2) if total_compressed_bytes > 0 else 0,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Conversion complete: {gguf_path.name}")
        print(f"  Tensors converted: {tensor_count}")
        print(f"  Original size: {total_original_bytes / 1e9:.2f} GB")
        print(f"  Compressed size: {total_compressed_bytes / 1e9:.2f} GB")
        print(f"  Overall ratio: {manifest['stats']['overall_ratio']:.1f}x")
        print(f"  Output: {output_dir}")

    return manifest


def _sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF models to CDNA format for streaming inference"
    )
    parser.add_argument(
        "gguf_path",
        help="Path to GGUF file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for CDNA files"
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=16,
        help="Block size for streaming (default: 16)"
    )
    parser.add_argument(
        "--codec",
        choices=["brotli", "zstd"],
        default="brotli",
        help="Compression codec (default: brotli)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    convert_gguf_model(
        gguf_path=Path(args.gguf_path),
        output_dir=Path(args.output),
        block_rows=args.block_rows,
        codec=args.codec,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
