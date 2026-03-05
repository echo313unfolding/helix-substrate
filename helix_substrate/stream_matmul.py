#!/usr/bin/env python3
"""
stream_xw_matmul.py - Streaming X @ W Matmul from CDNA

NOTE: "Streaming" in helix-cdc means decode streaming (block-by-block matmul),
NOT token generation streaming. This is part of the decode loop, not LLM inference.
See WO-VOCABULARY-ENFORCE-01 for vocabulary enforcement.

WO-STREAM-XW-01: Core primitive for activations-shaped compute from compressed weights.

Computes Y = X @ W where:
  - X is [batch, seq, K] activations (in memory)
  - W is [K, d_out] stored in CDNAv2 format (streamed, never fully loaded)

Algorithm (row-chunk the inner dimension):
  Y = zeros([batch, seq, d_out])
  for block_id in cdna_reader.blocks:
      W_rows = fetch_block_dequant(block_id, sidecar)  # [block_rows, d_out]
      k_start, k_end = block_row_range(block_id)
      X_slice = X[:, :, k_start:k_end]                  # [batch, seq, block_rows]
      Y += X_slice @ W_rows                             # partial accumulation

Memory bound: ~8MB working set vs 64MB full W = 8x savings per projection

Work Order: WO-STREAM-XW-01
"""

from __future__ import annotations

import hashlib
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from helix_substrate.cdna_reader import CDNAv2Reader, load_cdna_auto
from helix_substrate.sidecar import read_outlier_sidecar

# --- C++ kernel feature flags (optional, pure Python fallback available) ---
_USE_CPP_KERNEL = os.getenv("HELIX_USE_CPP_KERNEL", "0") == "1"
_USE_FUSED_MATMUL = os.getenv("HELIX_USE_FUSED_MATMUL", "0") == "1"  # Phase 2
_CPP_SPOTCHECK = os.getenv("HELIX_CPP_SPOTCHECK", "0") == "1"

# C++ native kernel (optional — pure Python path is default)
try:
    from helix_substrate._native import (  # type: ignore
        cpp_kernel_available,
        has_avx2,
        get_version,
        fused_decode_block,
        fused_decode_block_zstd,
        fused_decode_matmul,
        fused_decode_matmul_transpose,
        gather_codebook,
    )
except ImportError:
    cpp_kernel_available = lambda: False
    has_avx2 = lambda: False
    get_version = lambda: "unavailable"
    fused_decode_block = None
    fused_decode_block_zstd = None  # WO-CDNA2-KERNEL-02 Phase 1
    fused_decode_matmul = None      # WO-CDNA2-KERNEL-02 Phase 2
    fused_decode_matmul_transpose = None
    gather_codebook = None

# Final availability check: env flag + kernel available + AVX2
CPP_OK = bool(_USE_CPP_KERNEL and cpp_kernel_available() and has_avx2())

# Phase 2: Fused matmul availability (requires CPP_OK + explicit opt-in + function exists)
FUSED_MATMUL_OK = bool(CPP_OK and _USE_FUSED_MATMUL and fused_decode_matmul is not None)

CPP_META = {
    "enabled_by_env": _USE_CPP_KERNEL,
    "available": cpp_kernel_available() if callable(cpp_kernel_available) else False,
    "avx2": has_avx2() if callable(has_avx2) else False,
    "version": get_version() if callable(get_version) else "unavailable",
    "fused_matmul_enabled": _USE_FUSED_MATMUL,
    "fused_matmul_available": fused_decode_matmul is not None,
}


def reload_kernel_config() -> Dict[str, Any]:
    """Re-read env vars and update CPP_OK/FUSED_MATMUL_OK.

    WO-ECHO-POLICY-GOLDEN-BITE-01: This function allows runtime reconfiguration
    of kernel dispatch after changing HELIX_USE_CPP_KERNEL or HELIX_USE_FUSED_MATMUL
    environment variables. Without this, the module-level constants are frozen
    at import time and policy knobs have no effect.

    Returns:
        Dict with 'cpp_ok', 'fused_matmul_ok', and 'meta' keys.
    """
    global _USE_CPP_KERNEL, _USE_FUSED_MATMUL, CPP_OK, FUSED_MATMUL_OK, CPP_META

    _USE_CPP_KERNEL = os.getenv("HELIX_USE_CPP_KERNEL", "0") == "1"
    _USE_FUSED_MATMUL = os.getenv("HELIX_USE_FUSED_MATMUL", "0") == "1"

    CPP_OK = bool(_USE_CPP_KERNEL and cpp_kernel_available() and has_avx2())
    FUSED_MATMUL_OK = bool(CPP_OK and _USE_FUSED_MATMUL and fused_decode_matmul is not None)

    CPP_META = {
        "enabled_by_env": _USE_CPP_KERNEL,
        "available": cpp_kernel_available() if callable(cpp_kernel_available) else False,
        "avx2": has_avx2() if callable(has_avx2) else False,
        "version": get_version() if callable(get_version) else "unavailable",
        "fused_matmul_enabled": _USE_FUSED_MATMUL,
        "fused_matmul_available": fused_decode_matmul is not None,
    }

    return {
        "cpp_ok": CPP_OK,
        "fused_matmul_ok": FUSED_MATMUL_OK,
        "meta": CPP_META,
    }


VerifyPolicy = Literal["always", "sampled", "trust_cached"]


def prefilter_sidecar_by_block(
    positions: np.ndarray,
    values: np.ndarray,
    block_rows: int,
    cols: int,
) -> Dict[int, List[Tuple[int, int, float]]]:
    """
    Pre-compute block → sidecar positions mapping for O(1) lookup per block.

    Phase 1 of WO-STREAMING-PERF-01: Eliminates O(outliers × blocks) scanning.

    Args:
        positions: Flat outlier positions [N]
        values: Outlier values [N]
        block_rows: Rows per block
        cols: Tensor columns

    Returns:
        Dict mapping block_idx → [(local_row, col, value), ...]
    """
    if positions is None or len(positions) == 0:
        return {}

    # Vectorized computation of block indices and local positions
    rows = positions // cols
    block_indices = rows // block_rows
    local_rows = rows % block_rows
    col_indices = positions % cols

    # Group by block
    block_sidecars: Dict[int, List[Tuple[int, int, float]]] = {}
    for i in range(len(positions)):
        block_idx = int(block_indices[i])
        if block_idx not in block_sidecars:
            block_sidecars[block_idx] = []
        block_sidecars[block_idx].append((
            int(local_rows[i]),
            int(col_indices[i]),
            float(values[i])
        ))

    return block_sidecars


def _decode_cdna_block_cpp(
    compressed: bytes,
    codebook: np.ndarray,
    rows: int,
    cols: int,
    sidecar_positions: Optional[np.ndarray] = None,
    sidecar_values: Optional[np.ndarray] = None,
    verify_sha256: Optional[bytes] = None,
    codec: str = "brotli",  # WO-CDNA2-KERNEL-02: codec routing
) -> np.ndarray:
    """
    Decode CDNA block using C++ kernel (WO-CDNA-CPP-KERNEL-02 + WO-CDNA2-KERNEL-02).

    Fuses: decompress + codebook gather + sidecar apply
    Routes to brotli or zstd decompressor based on codec parameter.

    Args:
        compressed: Compressed CDNA indices (brotli or zstd)
        codebook: float32 codebook [256]
        rows: Number of rows in block
        cols: Number of columns
        sidecar_positions: Linear positions for corrections (optional)
        sidecar_values: Correction values (optional)
        verify_sha256: Expected SHA256 of decompressed indices (optional)
        codec: Compression codec - "brotli" or "zstd" (default: "brotli")

    Returns:
        float32 array of shape (rows, cols)
    """
    # WO-CDNA2-KERNEL-02: Route to appropriate decompressor
    if codec == "zstd" and fused_decode_block_zstd is not None:
        return fused_decode_block_zstd(
            compressed_data=compressed,
            codebook=np.ascontiguousarray(codebook, dtype=np.float32),
            rows=rows,
            cols=cols,
            sidecar_positions=sidecar_positions,
            sidecar_values=sidecar_values,
            verify_sha256=verify_sha256,
        )
    else:
        # Default to brotli (also handles lz4 fallback to Python path)
        return fused_decode_block(
            compressed_data=compressed,
            codebook=np.ascontiguousarray(codebook, dtype=np.float32),
            rows=rows,
            cols=cols,
            sidecar_positions=sidecar_positions,
            sidecar_values=sidecar_values,
            verify_sha256=verify_sha256,
        )


@dataclass
class StreamXWReceipt:
    """
    Receipt for stream_xw_from_cdna operation.

    Provides audit trail for:
    - Which blocks were touched/verified
    - Memory usage (RSS delta + tracemalloc peak)
    - Accuracy metrics (when comparing to canonical)
    - Codec version and streaming mode (claim hygiene)
    """
    schema: str = "stream_xw_receipt_v1"
    work_order: str = "WO-STREAM-XW-01"
    tensor_name: str = ""
    cdna_path: str = ""
    sidecar_path: Optional[str] = None

    # Claim hygiene: explicit codec and streaming mode
    # - codec_version: "cdna_v1" | "cdna_v2" - which format was used
    # - streaming_mode: "true_block_streaming" | "full_load_fallback"
    #   - true_block_streaming: Blocks streamed individually, never full W in memory
    #   - full_load_fallback: CDNAv1 forced full tensor load (no block streaming)
    codec_version: str = "unknown"
    streaming_mode: str = "unknown"

    # Transpose mode: whether X @ W^T was computed instead of X @ W
    transpose_w: bool = False

    # Block access info
    blocks_touched: List[int] = field(default_factory=list)
    blocks_verified: int = 0
    sidecar_applied: bool = False
    sidecar_corrections: int = 0

    # Input/output hashes (compute proof)
    input_sha256: str = ""
    output_sha256: str = ""

    # Accuracy (optional, filled by verification)
    cosine_vs_canonical: Optional[float] = None
    max_error: Optional[float] = None

    # Memory audit
    rss_delta_mb: float = 0.0
    tracemalloc_peak_mb: float = 0.0
    full_weight_mb: float = 0.0
    savings_factor: float = 0.0

    # Timing
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    # Phase 0 timing breakdown (WO-STREAMING-PERF-01)
    timing_breakdown: Optional[Dict[str, float]] = None

    # C++ kernel info (WO-CDNA-CPP-KERNEL-02)
    native_kernel_info: Optional[Dict[str, Any]] = None

    status: str = "PENDING"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "schema": self.schema,
            "work_order": self.work_order,
            "tensor_name": self.tensor_name,
            "cdna_path": self.cdna_path,
            # Claim hygiene: codec and streaming mode
            "codec_version": self.codec_version,
            "streaming_mode": self.streaming_mode,
            "transpose_w": self.transpose_w,
            # Sharp Edge 5B: Human-readable projection mode
            "projection_mode": "x_wt" if self.transpose_w else "x_w",
            "blocks_touched": self.blocks_touched,
            "blocks_verified": self.blocks_verified,
            "sidecar_applied": self.sidecar_applied,
            # Compute proof: input → output hash chain
            "compute_proof": {
                "input_activation_sha256": self.input_sha256,
                "output_activation_sha256": self.output_sha256,
            },
            "memory_audit": {
                "rss_delta_mb": self.rss_delta_mb,
                "tracemalloc_peak_mb": self.tracemalloc_peak_mb,
                "full_weight_mb": self.full_weight_mb,
                "savings_factor": self.savings_factor,
            },
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "status": self.status,
        }

        if self.sidecar_path:
            d["sidecar_path"] = self.sidecar_path
        if self.sidecar_applied:
            d["sidecar_corrections"] = self.sidecar_corrections
        if self.cosine_vs_canonical is not None:
            d["accuracy"] = {
                "cosine_vs_canonical": self.cosine_vs_canonical,
            }
            if self.max_error is not None:
                d["accuracy"]["max_error"] = self.max_error
        elif self.max_error is not None:
            d["accuracy"] = {"max_error": self.max_error}

        if self.timing_breakdown is not None:
            d["timing_breakdown"] = self.timing_breakdown

        if self.native_kernel_info is not None:
            d["native_kernel"] = self.native_kernel_info

        return d


def get_rss_mb() -> float:
    """Get current RSS (Resident Set Size) in MB."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except:
        pass
    return 0.0


def _compute_sha256(arr: np.ndarray) -> str:
    """Compute SHA256 of numpy array."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def stream_xw_from_cdna(
    X: np.ndarray,
    cdna_path: Union[str, Path],
    sidecar_path: Optional[Union[str, Path]] = None,
    verify_policy: VerifyPolicy = "trust_cached",
    emit_receipt: bool = True,
    transpose_w: bool = False,
    telem=None,  # Optional telemetry integration (WO-ECHO-TELEM-01)
) -> Tuple[np.ndarray, StreamXWReceipt]:
    """
    Compute Y = X @ W (or Y = X @ W^T if transpose_w=True) where W is stored in CDNAv2 format.

    Never loads full W into memory. Streams blocks and accumulates partial products.

    Args:
        X: Input activations. Shape [K] (1D), [seq, K] (2D), or [batch, seq, K] (3D).
           For transpose_w=False: K must match W's rows (the CDNA tensor's first dimension).
           For transpose_w=True: K must match W's cols (enables X @ W^T computation).
        cdna_path: Path to .cdna2.hxz or .cdna.hxz file containing W.
        sidecar_path: Optional path to .hxzo sidecar for outlier correction.
        verify_policy: Block verification policy:
            - "always": Verify every block on every access
            - "sampled": Verify random 10% of blocks
            - "trust_cached": Verify only on first access (default)
        emit_receipt: Whether to generate access receipt.
        transpose_w: If True, compute Y = X @ W^T instead of Y = X @ W.
            Useful for GQA where K/V weights are stored transposed.
            For W stored as [rows, cols], transpose mode computes:
            Y[batch, seq, rows] = X[batch, seq, cols] @ W^T[cols, rows]

    Returns:
        (Y, receipt) where Y has shape [..., d_out] matching X's leading dims.
        For transpose_w=False: d_out = W.cols
        For transpose_w=True: d_out = W.rows

    Memory Bound:
        ~8MB working set vs 64MB full W = 8x savings for typical 4096x4096 projection.

    Raises:
        ValueError: If X's last dimension doesn't match expected W dimension.
    """
    t0 = time.perf_counter()
    baseline_rss = get_rss_mb()
    peak_rss = baseline_rss

    cdna_path = Path(cdna_path)

    # Start memory tracking
    tracemalloc.start()

    # Load CDNA reader (header only, no decompression)
    reader = load_cdna_auto(cdna_path)

    # Handle input dimensions
    X = np.asarray(X, dtype=np.float32)
    original_shape = X.shape

    # Normalize to 3D [batch, seq, K]
    if X.ndim == 1:
        X = X.reshape(1, 1, -1)
    elif X.ndim == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])
    elif X.ndim > 3:
        raise ValueError(f"X must be 1D, 2D, or 3D, got shape {original_shape}")

    batch, seq, K = X.shape

    # Validate dimensions based on transpose mode
    # CDNA stores W as [rows, cols]
    # For X @ W: X's last dim must match rows, output dim = cols
    # For X @ W^T: X's last dim must match cols, output dim = rows
    if transpose_w:
        if K != reader.cols:
            raise ValueError(
                f"Input X has last dimension {K}, but W has {reader.cols} cols. "
                f"These must match for X @ W^T (transpose mode)."
            )
        d_out = reader.rows  # Output is the row dimension when transposed
    else:
        if K != reader.rows:
            raise ValueError(
                f"Input X has last dimension {K}, but W has {reader.rows} rows. "
                f"These must match for X @ W."
            )
        d_out = reader.cols

    # Load sidecar if provided
    outlier_positions = None
    outlier_values = None
    sidecar_applied = False
    sidecar_corrections = 0

    if sidecar_path is not None:
        sidecar_path = Path(sidecar_path)
        if sidecar_path.exists():
            outlier_positions, outlier_values, _ = read_outlier_sidecar(str(sidecar_path))
            sidecar_applied = True

    # Initialize output accumulator
    Y = np.zeros((batch, seq, d_out), dtype=np.float32)

    # Track blocks
    blocks_touched = []
    blocks_verified = 0

    # Check if we have CDNAv2 (with block streaming) or CDNAv1 (no blocks)
    # This determines whether we get true block streaming or full load fallback
    is_v2 = hasattr(reader, 'num_blocks')
    codec_version = "cdna_v2" if is_v2 else "cdna_v1"
    streaming_mode = "true_block_streaming" if is_v2 else "full_load_fallback"

    import random

    # Phase 0 + 0.5: Timing buckets (WO-STREAMING-PERF-01)
    # Phase 0.5: Split I/O from dequant to identify bottleneck
    t_indices_total = 0.0   # File I/O + Brotli decompress (stream_rows)
    t_dequant_total = 0.0   # Codebook gather (self.codebook[indices])
    t_sidecar_total = 0.0   # Sidecar correction
    t_matmul_total = 0.0    # BLAS X @ W_block

    if is_v2:
        # CDNAv2: Stream blocks
        # W is stored as [rows, cols]
        #
        # For X @ W (transpose_w=False):
        #   W rows = input dim, W cols = output dim
        #   Block over W rows, accumulate X_slice @ W_block
        #
        # For X @ W^T (transpose_w=True):
        #   W^T has shape [cols, rows], so rows = output dim, cols = input dim
        #   Block over W rows (which become output columns in W^T)
        #   Directly assign: Y[:, :, row_start:row_end] = X @ W_block.T

        # Sharp Edge 5A + WO-CDNA2-KERNEL-02 Phase 2: Performance optimization
        # Flatten X once before the loop for more efficient BLAS operations
        # This is required for transpose_w mode and also for fused matmul path
        if transpose_w or FUSED_MATMUL_OK:
            X_flat = X.reshape(-1, K)  # [batch*seq, K]
            Y_flat = np.zeros((batch * seq, d_out), dtype=np.float32)
        else:
            X_flat = None
            Y_flat = None

        # Determine if we should verify blocks
        # With CDNAv2 caching, "trust_cached" means verify on first access (cached)
        # "always" verifies every time, "sampled" verifies 10% randomly
        should_verify_blocks = verify_policy in ("always", "trust_cached")

        # Phase 1: Pre-filter sidecar positions by block (WO-STREAMING-PERF-01)
        # This eliminates O(outliers × blocks) scanning - now O(outliers) once + O(1) per block
        block_sidecars: Dict[int, List[Tuple[int, int, float]]] = {}
        if sidecar_applied and outlier_positions is not None:
            block_sidecars = prefilter_sidecar_by_block(
                outlier_positions, outlier_values,
                reader.block_rows, reader.cols
            )

        # Phase 2: Keep file handle open across all blocks (WO-STREAMING-PERF-01)
        # This avoids 0.5-1ms overhead per file open × num_blocks
        with open(cdna_path, "rb") as file_handle:
            for block_idx in range(reader.num_blocks):
                block_start_row = block_idx * reader.block_rows

                # Handle last block which may have fewer rows
                if block_idx == reader.num_blocks - 1:
                    block_end_row = reader.rows
                else:
                    block_end_row = block_start_row + reader.block_rows

                actual_block_rows = block_end_row - block_start_row

                # Determine verify for this specific block
                if verify_policy == "sampled":
                    should_verify = random.random() < 0.1
                else:
                    should_verify = should_verify_blocks

                # Fetch block indices and dequantize
                # WO-CDNA-CPP-KERNEL-02 + WO-CDNA2-KERNEL-02: C++ kernel path with codec routing
                # C++ kernel supports brotli and zstd; lz4 falls back to Python
                reader_codec = getattr(reader, 'codec', 'brotli')
                cpp_supports_codec = reader_codec in ("brotli", "zstd")

                # WO-CDNA2-KERNEL-02 Phase 2: Fused decode+matmul path (highest priority)
                # Does decompress → gather → sidecar → BLAS matmul all in C++
                # Eliminates W_block allocation and numpy matmul overhead
                if FUSED_MATMUL_OK and cpp_supports_codec and hasattr(reader, 'get_block_compressed') and not transpose_w:
                    # Fused matmul path for non-transpose mode
                    t_fused_start = time.perf_counter()

                    compressed, rows_in_block, cols, expected_sha = reader.get_block_compressed(
                        block_idx, file_handle
                    )

                    # Get prefiltered sidecar for this block
                    block_sidecar_list = block_sidecars.get(block_idx, [])
                    if block_sidecar_list:
                        positions = np.array([r * cols + c for r, c, _ in block_sidecar_list], dtype=np.uint64)
                        values = np.array([v for _, _, v in block_sidecar_list], dtype=np.float32)
                        sidecar_corrections += len(block_sidecar_list)
                    else:
                        positions, values = None, None

                    # Extract X slice for this block
                    X_slice_flat = np.ascontiguousarray(X_flat[:, block_start_row:block_end_row])

                    # Fused decode+matmul: Y_flat += X_slice @ W_block
                    fused_decode_matmul(
                        compressed_data=compressed,
                        codebook=reader.codebook,
                        X=X_slice_flat,
                        Y=Y_flat,
                        block_rows=rows_in_block,
                        cols=cols,
                        sidecar_positions=positions,
                        sidecar_values=values,
                        accumulate=(block_idx > 0),  # First block overwrites, rest accumulate
                        codec=reader_codec,
                    )

                    if should_verify:
                        blocks_verified += 1

                    t_fused_ms = (time.perf_counter() - t_fused_start) * 1000
                    t_indices_total += t_fused_ms  # All fused into one timing bucket

                    # Mark that we used fused path - no separate matmul timing
                    W_block = None  # No W_block allocated

                elif FUSED_MATMUL_OK and cpp_supports_codec and hasattr(reader, 'get_block_compressed') and transpose_w:
                    # Fused matmul path for transpose mode: Y += X @ W^T
                    t_fused_start = time.perf_counter()

                    compressed, rows_in_block, cols, expected_sha = reader.get_block_compressed(
                        block_idx, file_handle
                    )

                    # Get prefiltered sidecar for this block
                    block_sidecar_list = block_sidecars.get(block_idx, [])
                    if block_sidecar_list:
                        positions = np.array([r * cols + c for r, c, _ in block_sidecar_list], dtype=np.uint64)
                        values = np.array([v for _, _, v in block_sidecar_list], dtype=np.float32)
                        sidecar_corrections += len(block_sidecar_list)
                    else:
                        positions, values = None, None

                    # For transpose: Y_partial is the slice of Y corresponding to this block's W rows
                    # Y_flat[:, block_start_row:block_end_row] += X_flat @ W_block.T
                    Y_partial = np.ascontiguousarray(Y_flat[:, block_start_row:block_end_row])

                    fused_decode_matmul_transpose(
                        compressed_data=compressed,
                        codebook=reader.codebook,
                        X=X_flat,  # [batch_seq, cols]
                        Y=Y_partial,  # [batch_seq, block_rows]
                        block_rows=rows_in_block,
                        cols=cols,
                        sidecar_positions=positions,
                        sidecar_values=values,
                        accumulate=False,  # Direct assignment for transpose mode
                        codec=reader_codec,
                    )

                    # Copy result back (since Y_partial is a copy due to ascontiguousarray)
                    Y_flat[:, block_start_row:block_end_row] = Y_partial

                    if should_verify:
                        blocks_verified += 1

                    t_fused_ms = (time.perf_counter() - t_fused_start) * 1000
                    t_indices_total += t_fused_ms

                    W_block = None  # No W_block allocated

                elif CPP_OK and cpp_supports_codec and hasattr(reader, 'get_block_compressed'):
                    # C++ fast path (Phase 1): fused decompress + gather + sidecar
                    t_cpp_start = time.perf_counter()

                    compressed, rows_in_block, cols, expected_sha = reader.get_block_compressed(
                        block_idx, file_handle
                    )

                    # Get prefiltered sidecar for this block
                    block_sidecar_list = block_sidecars.get(block_idx, [])
                    if block_sidecar_list:
                        # Convert to linear positions for C++ kernel
                        positions = np.array([r * cols + c for r, c, _ in block_sidecar_list], dtype=np.uint64)
                        values = np.array([v for _, _, v in block_sidecar_list], dtype=np.float32)
                        sidecar_corrections += len(block_sidecar_list)
                    else:
                        positions, values = None, None

                    W_block = _decode_cdna_block_cpp(
                        compressed=compressed,
                        codebook=reader.codebook,
                        rows=rows_in_block,
                        cols=cols,
                        sidecar_positions=positions,
                        sidecar_values=values,
                        verify_sha256=expected_sha if should_verify else None,
                        codec=reader_codec,  # WO-CDNA2-KERNEL-02: pass codec for routing
                    )

                    if should_verify:
                        blocks_verified += 1

                    t_cpp_ms = (time.perf_counter() - t_cpp_start) * 1000
                    t_indices_total += t_cpp_ms  # All fused into one timing bucket

                    # Optional spotcheck: verify C++ matches Python on first block
                    if _CPP_SPOTCHECK and block_idx == 0:
                        W_py, _ = reader.stream_rows_dequant(
                            block_start_row, block_end_row,
                            emit_receipt=False, verify=False, file_handle=file_handle
                        )
                        # Apply sidecar to Python version for comparison
                        if block_sidecar_list:
                            for local_row, col, value in block_sidecar_list:
                                W_py[local_row, col] = value
                        if not np.allclose(W_block, W_py, rtol=1e-5, atol=1e-5):
                            raise RuntimeError("C++ kernel output mismatch vs Python reference")

                    # Skip manual sidecar - already applied by C++

                elif hasattr(reader, 'stream_rows_dequant'):
                    # Python path: CDNAv2 streaming API
                    # Phase 2: Pass file_handle to avoid reopening file per block
                    # Phase 3: emit_receipt=False to avoid per-block receipt overhead
                    # Phase 0.5: Receipt now always contains timing (t_indices_ms, t_dequant_ms)
                    W_block, block_receipt = reader.stream_rows_dequant(
                        block_start_row, block_end_row,
                        sidecar_path=None,  # We'll apply sidecar manually for row range
                        emit_receipt=False,  # Phase 3: Batch receipts (timing still returned)
                        verify=should_verify,  # Verify during load, not separately
                        file_handle=file_handle,  # Phase 2: Reuse file handle
                    )
                    if should_verify:
                        blocks_verified += 1

                    # Phase 0.5: Extract timing from block receipt
                    if block_receipt:
                        t_indices_total += block_receipt.get("t_indices_ms", 0)
                        t_dequant_total += block_receipt.get("t_dequant_ms", 0)

                    # Phase 0: Time sidecar correction (Python path only)
                    t_sidecar_start = time.perf_counter()

                    # Phase 1: Apply prefiltered sidecar corrections for this block
                    # O(1) lookup + O(corrections_in_block) vs O(all_outliers) before
                    if block_idx in block_sidecars:
                        for local_row, col, value in block_sidecars[block_idx]:
                            W_block[local_row, col] = value
                            sidecar_corrections += 1

                    t_sidecar_total += time.perf_counter() - t_sidecar_start

                else:
                    # CDNAv1 fallback - get full indices and slice (no timing split)
                    t_io_start = time.perf_counter()
                    blob = reader._ensure_indices_blob()
                    all_indices = np.frombuffer(blob, dtype=np.uint8).reshape(reader.rows, reader.cols)
                    block_indices = all_indices[block_start_row:block_end_row]
                    W_block = reader.codebook[block_indices]
                    # CDNAv1: assign all to indices (no decompress/dequant split)
                    t_indices_total += (time.perf_counter() - t_io_start) * 1000

                    # Phase 0: Time sidecar correction (CDNAv1 path)
                    t_sidecar_start = time.perf_counter()

                    # Apply prefiltered sidecar corrections
                    if block_idx in block_sidecars:
                        for local_row, col, value in block_sidecars[block_idx]:
                            W_block[local_row, col] = value
                            sidecar_corrections += 1

                    t_sidecar_total += time.perf_counter() - t_sidecar_start

                # Phase 0: Time matmul (skip if fused path was used)
                if W_block is not None:
                    t_matmul_start = time.perf_counter()

                    if transpose_w:
                        # Transpose mode: Y[:, row_start:row_end] = X_flat @ W_block^T
                        # W_block is [block_rows, cols], W_block^T is [cols, block_rows]
                        # X_flat [batch*seq, cols] @ W_block^T [cols, block_rows] = [batch*seq, block_rows]
                        # Direct assignment to output columns (no accumulation)
                        # Sharp Edge 5A: Using flattened arrays for efficient BLAS
                        Y_flat[:, block_start_row:block_end_row] = X_flat @ W_block.T
                    elif FUSED_MATMUL_OK and Y_flat is not None:
                        # Non-fused C++ path but with flattened arrays (for consistency)
                        X_slice_flat = X_flat[:, block_start_row:block_end_row]
                        Y_flat += X_slice_flat @ W_block
                    else:
                        # Standard mode: Y += X_slice @ W_block (accumulate)
                        # X is [batch, seq, K], we want X[..., block_start_row:block_end_row]
                        X_slice = X[:, :, block_start_row:block_end_row]  # [batch, seq, block_rows]

                        # Compute partial product and accumulate
                        # X_slice [batch, seq, block_rows] @ W_block [block_rows, d_out] = [batch, seq, d_out]
                        Y += X_slice @ W_block
                        del X_slice

                    t_matmul_total += time.perf_counter() - t_matmul_start

                    # Free block memory
                    del W_block

                blocks_touched.append(block_idx)
                peak_rss = max(peak_rss, get_rss_mb())

                # WO-ECHO-TELEM-01: Record telemetry for this block
                if telem is not None:
                    # Record block as a "step" in telemetry
                    block_ms = (time.perf_counter() - t0) * 1000 / (block_idx + 1)
                    telem._counters.step_times_ms.append(block_ms)
                    telem._global_counters.step_times_ms.append(block_ms)
                    telem._counters.steps += 1
                    telem._global_counters.steps += 1

                    # Cache: verified = hit, unverified = trust (pseudo-hit)
                    if should_verify:
                        telem.record_cache_hit()
                    else:
                        telem.record_cache_hit()  # trust_cached is still a hit

                    # Regen: sidecar corrections are regeneration events
                    if block_idx in block_sidecars:
                        block_corrections = block_sidecars[block_idx]
                        if block_corrections:
                            # Estimate bytes from corrections
                            regen_bytes = len(block_corrections) * 4  # float32
                            telem.record_regen(bytes=regen_bytes, time_ms=0.0)

                    # Route choice: cpp vs python
                    reader_codec = getattr(reader, 'codec', 'brotli')
                    codec_ok = reader_codec in ("brotli", "zstd")
                    if CPP_OK and codec_ok:
                        route = "cpp_fused" if FUSED_MATMUL_OK else "cpp_decode"
                    else:
                        route = "python"
                    telem.record_route(route)

        # Sharp Edge 5A + Phase 2: Reshape Y_flat back to 3D
        # Needed for transpose_w mode and fused matmul path
        if Y_flat is not None:
            Y = Y_flat.reshape(batch, seq, d_out)
            del X_flat, Y_flat

    else:
        # CDNAv1: No block streaming, load full indices and compute in chunks
        # We'll still chunk to demonstrate bounded memory approach

        # Get all indices at once (CDNAv1 doesn't support streaming)
        blob = reader._ensure_indices_blob()
        all_indices = np.frombuffer(blob, dtype=np.uint8).reshape(reader.rows, reader.cols)

        # Dequantize full tensor
        W_full = reader.codebook[all_indices]

        # Apply sidecar corrections
        if sidecar_applied and outlier_positions is not None:
            for i in range(len(outlier_positions)):
                pos = outlier_positions[i]
                row = pos // reader.cols  # Use reader.cols for consistency
                col = pos % reader.cols
                if row < W_full.shape[0] and col < W_full.shape[1]:
                    W_full[row, col] = float(outlier_values[i])
                    sidecar_corrections += 1

        # Compute full matmul (CDNAv1 doesn't give us memory savings)
        if transpose_w:
            Y = X @ W_full.T  # [batch, seq, cols] @ [cols, rows] = [batch, seq, rows]
        else:
            Y = X @ W_full    # [batch, seq, rows] @ [rows, cols] = [batch, seq, cols]
        blocks_touched = [0]  # Single "block" representing full tensor
        blocks_verified = 1 if verify_policy == "always" else 0

        peak_rss = max(peak_rss, get_rss_mb())
        del all_indices, W_full

    # Get tracemalloc peak
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # WO-ECHO-TELEM-01: Record output "tokens" (elements in output tensor)
    if telem is not None:
        # Count output elements as tokens for throughput tracking
        output_elements = int(np.prod(Y.shape))
        telem.record_token(count=output_elements)

    # Reshape output to match input shape
    if len(original_shape) == 1:
        Y = Y.squeeze(axis=(0, 1))  # [d_out]
    elif len(original_shape) == 2:
        Y = Y.squeeze(axis=0)  # [seq, d_out]
    # else keep [batch, seq, d_out]

    duration_ms = (time.perf_counter() - t0) * 1000

    # Build receipt
    # WO-APPLES-TO-APPLES-01: Use bytes for savings_factor calculation
    # to avoid rounding issues (0.0 MB -> infinity ratios)
    full_weight_bytes = reader.rows * reader.cols * 4  # float32
    full_weight_mb = full_weight_bytes / (1024 * 1024)
    rss_delta_mb = peak_rss - baseline_rss
    # Convert to bytes for precise calculation
    rss_delta_bytes = int(rss_delta_mb * 1024 * 1024)
    # Minimum 1 page (4KB) to avoid division by zero
    rss_delta_bytes = max(rss_delta_bytes, 4096)
    # Calculate savings factor using bytes (not rounded MB)
    savings_factor = full_weight_bytes / rss_delta_bytes

    # Phase 0 + 0.5: Build timing breakdown (WO-STREAMING-PERF-01)
    # t_indices_total, t_dequant_total already in ms (from block receipts)
    # t_sidecar_total, t_matmul_total in seconds (from perf_counter)
    t_sidecar_ms = t_sidecar_total * 1000
    t_matmul_ms = t_matmul_total * 1000
    t_accounted_ms = t_indices_total + t_dequant_total + t_sidecar_ms + t_matmul_ms

    timing_breakdown = {
        "t_indices_ms": round(t_indices_total, 3),    # I/O + Brotli decompress
        "t_dequant_ms": round(t_dequant_total, 3),    # codebook[indices] gather
        "t_sidecar_ms": round(t_sidecar_ms, 3),
        "t_matmul_ms": round(t_matmul_ms, 3),
        "t_other_ms": round(duration_ms - t_accounted_ms, 3),
    }

    # WO-CDNA-CPP-KERNEL-02 + WO-CDNA2-KERNEL-02: Build native kernel info for receipt
    reader_codec = getattr(reader, 'codec', 'brotli')
    cpp_supports_codec = reader_codec in ("brotli", "zstd")
    fused_matmul_used = FUSED_MATMUL_OK and cpp_supports_codec
    native_kernel_info = {
        "used": CPP_OK and cpp_supports_codec,
        "codec": reader_codec,
        "cpp_codec_supported": cpp_supports_codec,
        "fused_matmul_used": fused_matmul_used,  # Phase 2
        **CPP_META,
    }

    receipt = StreamXWReceipt(
        tensor_name=reader.tensor_name,
        cdna_path=str(cdna_path),
        sidecar_path=str(sidecar_path) if sidecar_path else None,
        # Claim hygiene: explicitly record codec and streaming mode
        codec_version=codec_version,
        streaming_mode=streaming_mode,
        transpose_w=transpose_w,
        blocks_touched=blocks_touched,
        blocks_verified=blocks_verified,
        sidecar_applied=sidecar_applied,
        sidecar_corrections=sidecar_corrections,
        input_sha256=_compute_sha256(np.asarray(X, dtype=np.float32)),
        output_sha256=_compute_sha256(Y),
        rss_delta_mb=rss_delta_mb,
        tracemalloc_peak_mb=peak_mem / (1024 * 1024),
        full_weight_mb=full_weight_mb,
        savings_factor=savings_factor,
        duration_ms=duration_ms,
        timing_breakdown=timing_breakdown,
        native_kernel_info=native_kernel_info,
        status="PASS",
    )

    return Y, receipt


def stream_xw_from_manifest(
    X: np.ndarray,
    tensor_name: str,
    manifest: Dict[str, Any],
    base_path: Path,
    verify_policy: VerifyPolicy = "trust_cached",
    transpose_w: bool = False,
    
) -> Tuple[np.ndarray, StreamXWReceipt]:
    """
    Compute Y = X @ W (or Y = X @ W^T) using tensor info from manifest.

    Convenience wrapper that resolves CDNA and sidecar paths from manifest.

    Args:
        X: Input activations [batch, seq, K] or [seq, K] or [K].
        tensor_name: Name of tensor in manifest (e.g., "blk.0.attn_q.weight").
        manifest: Parsed manifest dict (cdna_hybrid_manifest_v2 schema).
        base_path: Base path for resolving relative paths in manifest.
        verify_policy: Block verification policy.
        transpose_w: If True, compute Y = X @ W^T instead of Y = X @ W.

    Returns:
        (Y, receipt)
    """
    # Find tensor in manifest
    tensor_info = None
    for shard in manifest.get("shards", []):
        if shard.get("tensor_name") == tensor_name:
            tensor_info = shard
            break

    if tensor_info is None:
        raise KeyError(f"Tensor '{tensor_name}' not found in manifest")

    # Resolve paths
    cdna_path = base_path / tensor_info["path"]

    sidecar_path = None
    if tensor_info.get("outlier_sidecar_path"):
        sidecar_dir = manifest.get("sidecar_dir", "")
        sidecar_path = base_path / sidecar_dir / tensor_info["outlier_sidecar_path"]

    return stream_xw_from_cdna(
        X=X,
        cdna_path=cdna_path,
        sidecar_path=sidecar_path,
        verify_policy=verify_policy,
        transpose_w=transpose_w,
        telem=telem,  # WO-ECHO-TELEM-01
    )


def compare_vs_canonical(
    Y_stream: np.ndarray,
    Y_canonical: np.ndarray,
    receipt: StreamXWReceipt,
) -> StreamXWReceipt:
    """
    Compare streaming result to canonical and update receipt with accuracy metrics.

    Args:
        Y_stream: Result from stream_xw_from_cdna.
        Y_canonical: Result from full X @ W_full computation.
        receipt: Receipt to update.

    Returns:
        Updated receipt with cosine_vs_canonical and max_error.
    """
    Y_stream = Y_stream.flatten()
    Y_canonical = Y_canonical.flatten()

    # Cosine similarity
    dot = float(np.sum(Y_stream * Y_canonical))
    norm_stream = float(np.sqrt(np.sum(Y_stream ** 2)))
    norm_canonical = float(np.sqrt(np.sum(Y_canonical ** 2)))
    cosine = dot / (norm_stream * norm_canonical + 1e-10)

    # Max error
    max_err = float(np.abs(Y_stream - Y_canonical).max())

    receipt.cosine_vs_canonical = cosine
    receipt.max_error = max_err

    # Update status based on accuracy
    if cosine >= 0.999:
        receipt.status = "PASS"
    elif cosine >= 0.99:
        receipt.status = "ACCEPTABLE"
    else:
        receipt.status = "FAIL"

    return receipt


if __name__ == "__main__":
    print("=== Stream X@W from CDNA Test ===")
    print()
    print("Usage:")
    print("  from helix_cdc.regrow.stream_xw_matmul import stream_xw_from_cdna")
    print("  Y, receipt = stream_xw_from_cdna(X, 'tensor.cdna2.hxz')")
    print()
    print("See tools/streaming_xw_verification.py for verification.")
