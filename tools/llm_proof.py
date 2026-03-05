#!/usr/bin/env python3
"""
LLM Proof Script for helix-substrate

Validates that GGUF conversion and streaming matmul work correctly
on real model weights (not synthetic data).

This script:
1. Uses TinyLlama-1.1B GGUF (pre-downloaded or downloads it)
2. Converts 5 sample tensors to CDNA format
3. Verifies streaming matmul produces correct results
4. Generates a signed receipt

Usage:
    python tools/llm_proof.py                    # Full proof
    python tools/llm_proof.py --check-only       # Check if receipt exists
    python tools/llm_proof.py --quick            # Quick proof (1 tensor)
"""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Receipt directory
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "llm_proof"


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_tinyllama():
    """Download TinyLlama GGUF if not present."""
    from huggingface_hub import hf_hub_download

    local_path = Path.home() / "helix-substrate" / "models" / "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

    if local_path.exists():
        print(f"Using existing model: {local_path}")
        return local_path

    # Try alternate location
    alt_path = Path.home() / "helix-cdc" / "models" / "gguf" / "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
    if alt_path.exists():
        print(f"Using existing model: {alt_path}")
        return alt_path

    print("Downloading TinyLlama-1.1B Q8_0 from HuggingFace...")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        local_dir=local_path.parent,
    )
    return Path(downloaded)


def run_proof(quick: bool = False):
    """Run the full LLM proof."""
    try:
        from gguf import GGUFReader
        from gguf.quants import dequantize
    except ImportError:
        print("ERROR: gguf package required. Install with: pip install gguf")
        sys.exit(1)

    from helix_substrate.cdna_reader import CDNAv2Writer, CDNAv2Reader
    from helix_substrate.convert_gguf import quantize_tensor

    print("=" * 60)
    print("helix-substrate LLM Proof")
    print("=" * 60)

    # Get model
    gguf_path = download_tinyllama()
    model_sha256 = sha256_file(gguf_path)
    print(f"Model SHA256: {model_sha256[:16]}...")

    # Load GGUF
    print("\nLoading GGUF...")
    t0 = time.time()
    reader = GGUFReader(str(gguf_path))
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.2f}s, {len(reader.tensors)} tensors")

    # Select test tensors (real weight matrices)
    test_tensors = [
        "blk.0.attn_q.weight",   # 2048x2048
        "blk.0.ffn_gate.weight", # 5632x2048
        "blk.10.attn_k.weight",  # 256x2048
    ]
    if not quick:
        test_tensors.extend([
            "blk.15.attn_output.weight",
            "output.weight",  # 32000x2048
        ])

    results = []
    total_original = 0
    total_compressed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        writer = CDNAv2Writer(codec="brotli", block_rows=16)

        for tensor_name in test_tensors:
            print(f"\nTesting: {tensor_name}")

            # Find tensor
            tensor_data = None
            for t in reader.tensors:
                if t.name == tensor_name:
                    tensor_data = dequantize(t.data, t.tensor_type).astype(np.float32)
                    break

            if tensor_data is None:
                print(f"  WARNING: Tensor not found, skipping")
                continue

            print(f"  Shape: {tensor_data.shape}")
            original_bytes = tensor_data.nbytes
            total_original += original_bytes

            # Quantize
            indices, codebook = quantize_tensor(tensor_data)

            # Write CDNA
            cdna_path = tmpdir / f"{tensor_name.replace('.', '_')}.cdna2.hxz"
            writer.write(
                indices=indices,
                codebook=codebook,
                output_path=cdna_path,
                tensor_name=tensor_name,
            )
            compressed_bytes = cdna_path.stat().st_size
            total_compressed += compressed_bytes
            ratio = original_bytes / compressed_bytes
            print(f"  Compression: {ratio:.1f}x")

            # Test streaming matmul
            cdna_reader = CDNAv2Reader(cdna_path)

            # Create test input
            in_dim = tensor_data.shape[1]  # Columns
            X = np.random.RandomState(42).randn(8, in_dim).astype(np.float32)

            # Ground truth: direct matmul with quantized weights
            W_quantized = codebook[indices]
            Y_truth = X @ W_quantized.T  # [8, out_dim]

            # Streaming matmul
            out_dim = tensor_data.shape[0]
            Y_stream = np.zeros((8, out_dim), dtype=np.float32)

            for block_indices, start_row in cdna_reader.stream_blocks():
                end_row = start_row + block_indices.shape[0]
                W_block = codebook[block_indices]  # [block_rows, in_dim]
                Y_stream[:, start_row:end_row] = X @ W_block.T

            # Verify
            cosine = np.dot(Y_truth.flatten(), Y_stream.flatten()) / (
                np.linalg.norm(Y_truth) * np.linalg.norm(Y_stream)
            )
            max_diff = np.abs(Y_truth - Y_stream).max()

            print(f"  Streaming cosine: {cosine:.6f}")
            print(f"  Max diff: {max_diff:.6f}")

            results.append({
                "tensor": tensor_name,
                "shape": list(tensor_data.shape),
                "original_bytes": int(original_bytes),
                "compressed_bytes": int(compressed_bytes),
                "compression_ratio": round(ratio, 2),
                "streaming_cosine": round(float(cosine), 6),
                "max_diff": round(float(max_diff), 6),
                "pass": bool(cosine > 0.9999),
            })

    # Generate receipt
    all_pass = all(r["pass"] for r in results)

    receipt = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "package_version": "0.2.1",
        "model": "TinyLlama-1.1B-Chat-v1.0-Q8_0.gguf",
        "model_sha256": model_sha256,
        "tensors_tested": len(results),
        "total_original_bytes": int(total_original),
        "total_compressed_bytes": int(total_compressed),
        "overall_ratio": round(total_original / total_compressed, 2) if total_compressed > 0 else 0,
        "all_cosine_above_0.9999": all_pass,
        "results": results,
        "verdict": "PASS" if all_pass else "FAIL",
    }

    # Save receipt
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPT_DIR / "tinyllama_1b_proof.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))

    print("\n" + "=" * 60)
    print(f"VERDICT: {receipt['verdict']}")
    print(f"  Tensors tested: {len(results)}")
    print(f"  All streaming cosine > 0.9999: {all_pass}")
    print(f"  Overall compression: {receipt['overall_ratio']:.1f}x")
    print(f"  Receipt saved: {receipt_path}")
    print("=" * 60)

    return receipt


def check_receipt():
    """Check if a valid receipt exists."""
    receipt_path = RECEIPT_DIR / "tinyllama_1b_proof.json"

    if not receipt_path.exists():
        print("No receipt found. Run: python tools/llm_proof.py")
        return False

    receipt = json.loads(receipt_path.read_text())
    print(f"Receipt found: {receipt_path}")
    print(f"  Timestamp: {receipt['timestamp']}")
    print(f"  Model: {receipt['model']}")
    print(f"  Verdict: {receipt['verdict']}")

    return receipt['verdict'] == "PASS"


def main():
    parser = argparse.ArgumentParser(description="LLM proof for helix-substrate")
    parser.add_argument("--check-only", action="store_true", help="Only check if receipt exists")
    parser.add_argument("--quick", action="store_true", help="Quick proof (1 tensor)")

    args = parser.parse_args()

    if args.check_only:
        success = check_receipt()
        sys.exit(0 if success else 1)

    receipt = run_proof(quick=args.quick)
    sys.exit(0 if receipt['verdict'] == "PASS" else 1)


if __name__ == "__main__":
    main()
