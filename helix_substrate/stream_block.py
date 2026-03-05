#!/usr/bin/env python3
"""
stream_transformer_block.py - Full Transformer Block from CDNA

NOTE: This is a decode loop component. "Streaming" refers to memory-efficient
block-wise decoding, NOT llama.cpp token streaming or chat inference.
See WO-VOCABULARY-ENFORCE-01 for vocabulary enforcement.

WO-STREAM-FULL-BLOCK-01: Complete transformer block where all 7 projections
(Q, K, V, O, gate, up, down) are streamed from CDNAv2 format - never fully loaded.

Pipeline (Pre-norm Transformer):
  1. X_norm = RMSNorm(X, attn_norm)
  2. attn_out = stream_attention_forward(X_norm)
  3. X_mid = X + attn_out                  # Residual 1
  4. X_mid_norm = RMSNorm(X_mid, ffn_norm)
  5. ffn_out = stream_ffn_forward(X_mid_norm)
  6. output = X_mid + ffn_out              # Residual 2

Memory Analysis (for Mistral blk.0, seq=256):
  - Input X: 4 MB ([1, 256, 4096])
  - Attention weights (Q,K,V,O): 160 MB - NEVER LOADED
  - FFN weights (gate,up,down): 672 MB - NEVER LOADED
  - Norm weights: 32 KB (loaded once, tiny)
  - Total working set: ~50 MB vs 832 MB full = 16x savings

Norm Weight Sources:
  - Option 1: GGUF file (default - norms not quantized)
  - Option 2: F16 shard files (seeds/shards_f16/blk.N.*.hxz)
  - Option 3: Passed directly as numpy arrays

Work Order: WO-STREAM-FULL-BLOCK-01
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from helix_substrate.stream_matmul import get_rss_mb, VerifyPolicy

from helix_substrate.stream_attention import (
    stream_attention_forward,
    AttentionLayerReceipt,
)
from helix_substrate.stream_ffn import (
    stream_ffn_forward,
    FFNLayerReceipt,
)


@dataclass
class TransformerBlockReceipt:
    """
    Receipt for full transformer block forward pass.

    Aggregates AttentionLayerReceipt and FFNLayerReceipt.
    """
    schema: str = "stream_transformer_block_receipt_v1"
    work_order: str = "WO-STREAM-FULL-BLOCK-01"
    block_index: int = 0

    # Component receipts
    attention_receipt: Optional[Dict[str, Any]] = None
    ffn_receipt: Optional[Dict[str, Any]] = None

    # Norm info
    norms_info: Dict[str, Any] = field(default_factory=dict)

    # Block config
    block_config: Dict[str, Any] = field(default_factory=dict)

    # Overall memory audit
    memory_audit: Dict[str, Any] = field(default_factory=dict)

    # Timing breakdown
    timing: Dict[str, float] = field(default_factory=dict)

    # Accuracy (optional)
    accuracy: Dict[str, Any] = field(default_factory=dict)

    # Status
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    status: str = "PENDING"

    def to_dict(self) -> Dict[str, Any]:
        # Aggregate claim hygiene from subreceipts
        attn_claim = {}
        ffn_claim = {}

        if self.attention_receipt:
            attn_claim = self.attention_receipt.get("claim_hygiene", {})
        if self.ffn_receipt:
            ffn_claim = self.ffn_receipt.get("claim_hygiene", {})

        # Aggregate codec versions and streaming modes
        all_codec_versions = set()
        all_streaming_modes = set()

        for src in [attn_claim, ffn_claim]:
            all_codec_versions.update(src.get("codec_versions_used", []))
            all_streaming_modes.update(src.get("streaming_modes_used", []))

        d = {
            "schema": self.schema,
            "work_order": self.work_order,
            "block_index": self.block_index,
            # Claim hygiene: top-level summary
            "claim_hygiene": {
                "codec_versions_used": list(all_codec_versions),
                "streaming_modes_used": list(all_streaming_modes),
                "all_true_block_streaming": all_streaming_modes == {"true_block_streaming"},
                "projections_streamed": 7,  # Q, K, V, O, gate, up, down
                "attention_gqa_handling": attn_claim.get("gqa_handling", "unknown"),
            },
            "components": {
                "attention": self.attention_receipt,
                "ffn": self.ffn_receipt,
            },
            "norms_info": self.norms_info,
            "block_config": self.block_config,
            "memory_audit": self.memory_audit,
            "timing": self.timing,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "status": self.status,
        }
        if self.accuracy:
            d["accuracy"] = self.accuracy
        return d


def rms_norm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5  # WO-CDNA-NORM-EPS-FIX-01: Match GGUF metadata
) -> np.ndarray:
    """
    RMSNorm: x * weight / RMS(x)

    Args:
        x: Input tensor [..., d_model]
        weight: Norm weight [d_model]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor [..., d_model]
    """
    # RMS = sqrt(mean(x^2))
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


# Cache for TensorAccessor and norm weights (WO-TOKENIZER-BRIDGE-01.1)
_ACCESSOR_CACHE: Dict[str, Any] = {}
_NORM_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _get_cached_accessor(gguf_path: Union[str, Path]) -> Any:
    """Get or create cached TensorAccessor for a GGUF file."""
    from helix_cdc.regrow.tensor_accessor import TensorAccessor

    key = str(Path(gguf_path).absolute())
    if key not in _ACCESSOR_CACHE:
        _ACCESSOR_CACHE[key] = TensorAccessor.from_gguf(gguf_path)
    return _ACCESSOR_CACHE[key]


def load_norm_weights_from_gguf(
    gguf_path: Union[str, Path],
    block_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load attention and FFN norm weights from GGUF file.

    Uses caching to avoid repeated loading (WO-TOKENIZER-BRIDGE-01.1).

    Args:
        gguf_path: Path to GGUF file
        block_index: Transformer block index

    Returns:
        (attn_norm_weight, ffn_norm_weight)
    """
    # Check norm cache first
    cache_key = f"{Path(gguf_path).absolute()}:blk.{block_index}"
    if cache_key in _NORM_CACHE:
        return _NORM_CACHE[cache_key]

    # Use cached accessor
    accessor = _get_cached_accessor(gguf_path)

    attn_norm_name = f"blk.{block_index}.attn_norm.weight"
    ffn_norm_name = f"blk.{block_index}.ffn_norm.weight"

    attn_norm = accessor.load_tensor(attn_norm_name).astype(np.float32)
    ffn_norm = accessor.load_tensor(ffn_norm_name).astype(np.float32)

    # Cache the norms
    _NORM_CACHE[cache_key] = (attn_norm, ffn_norm)

    return attn_norm, ffn_norm


def stream_transformer_block_forward(
    X: np.ndarray,
    manifest_path: Union[str, Path],
    block_index: int,
    d_head: int = 128,
    verify_policy: VerifyPolicy = "trust_cached",
    attn_norm_weight: Optional[np.ndarray] = None,
    ffn_norm_weight: Optional[np.ndarray] = None,
    gguf_path: Optional[Union[str, Path]] = None,
    norm_eps: float = 1e-5,  # WO-CDNA-NORM-EPS-FIX-01: Match GGUF metadata
    return_intermediates: bool = False,
    
) -> Tuple[np.ndarray, TransformerBlockReceipt, Optional[Dict[str, np.ndarray]]]:
    """
    Full transformer block from CDNA weights.

    Pipeline:
      1. X_norm = RMSNorm(X, attn_norm)
      2. attn_out = Attention(X_norm)
      3. X_mid = X + attn_out           (residual)
      4. X_mid_norm = RMSNorm(X_mid, ffn_norm)
      5. ffn_out = FFN(X_mid_norm)
      6. output = X_mid + ffn_out       (residual)

    NEVER loads full projection weights into memory.

    Args:
        X: Input activations [batch, seq, d_model] or [seq, d_model].
        manifest_path: Path to hybrid_manifest_v2.json.
        block_index: Transformer block index (0-31 for Mistral).
        d_head: Head dimension for attention scaling (default: 128 for Mistral).
        verify_policy: Block verification policy.
        attn_norm_weight: Pre-loaded attention norm weight [d_model]. If None, loads from GGUF.
        ffn_norm_weight: Pre-loaded FFN norm weight [d_model]. If None, loads from GGUF.
        gguf_path: Path to GGUF file for loading norm weights (required if norms not provided).
        norm_eps: Epsilon for RMSNorm (default: 1e-6).
        return_intermediates: If True, return intermediate tensors.

    Returns:
        (output [batch, seq, d_model], receipt, intermediates or None)
    """
    t0_total = time.perf_counter()
    baseline_rss = get_rss_mb()
    peak_rss = baseline_rss

    tracemalloc.start()

    manifest_path = Path(manifest_path)

    # Handle input dimensions
    X = np.asarray(X, dtype=np.float32)
    original_shape = X.shape

    if X.ndim == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])
    elif X.ndim != 3:
        raise ValueError(f"X must be 2D [seq, d_model] or 3D [batch, seq, d_model], got shape {original_shape}")

    batch, seq, d_model = X.shape

    timing = {}
    norms_info = {}

    # Load norm weights if not provided
    t0 = time.perf_counter()
    if attn_norm_weight is None or ffn_norm_weight is None:
        if gguf_path is None:
            raise ValueError(
                "Either provide attn_norm_weight/ffn_norm_weight directly, "
                "or provide gguf_path to load them from GGUF."
            )
        loaded_attn_norm, loaded_ffn_norm = load_norm_weights_from_gguf(gguf_path, block_index)
        if attn_norm_weight is None:
            attn_norm_weight = loaded_attn_norm
        if ffn_norm_weight is None:
            ffn_norm_weight = loaded_ffn_norm
        norms_info["source"] = "gguf"
        norms_info["gguf_path"] = str(gguf_path)
    else:
        norms_info["source"] = "provided"

    timing["norm_load_ms"] = (time.perf_counter() - t0) * 1000

    norms_info["attn_norm_shape"] = list(attn_norm_weight.shape)
    norms_info["ffn_norm_shape"] = list(ffn_norm_weight.shape)
    norms_info["attn_norm_range"] = [float(attn_norm_weight.min()), float(attn_norm_weight.max())]
    norms_info["ffn_norm_range"] = [float(ffn_norm_weight.min()), float(ffn_norm_weight.max())]
    norms_info["eps"] = norm_eps

    peak_rss = max(peak_rss, get_rss_mb())

    # Step 1: Apply attention norm
    t0 = time.perf_counter()
    X_norm = rms_norm(X, attn_norm_weight, norm_eps)
    timing["attn_norm_ms"] = (time.perf_counter() - t0) * 1000

    # Step 2: Attention forward
    t0 = time.perf_counter()
    attn_out, attn_receipt, attn_intermediates = stream_attention_forward(
        X_norm,
        manifest_path,
        block_index,
        d_head=d_head,
        verify_policy=verify_policy,
        return_intermediates=return_intermediates,
        telem=telem,  # WO-BASIN-TELEM-HOOK-01
    )
    timing["attention_ms"] = (time.perf_counter() - t0) * 1000
    peak_rss = max(peak_rss, get_rss_mb())

    # Ensure attn_out has correct shape
    if attn_out.ndim == 2:
        attn_out = attn_out.reshape(1, attn_out.shape[0], attn_out.shape[1])

    # Step 3: Residual connection 1
    t0 = time.perf_counter()
    X_mid = X + attn_out
    timing["residual_1_ms"] = (time.perf_counter() - t0) * 1000

    del X_norm, attn_out  # Free memory

    # Step 4: Apply FFN norm
    t0 = time.perf_counter()
    X_mid_norm = rms_norm(X_mid, ffn_norm_weight, norm_eps)
    timing["ffn_norm_ms"] = (time.perf_counter() - t0) * 1000

    # Step 5: FFN forward
    t0 = time.perf_counter()
    ffn_out, ffn_receipt, ffn_intermediates = stream_ffn_forward(
        X_mid_norm,
        manifest_path,
        block_index,
        verify_policy=verify_policy,
        return_intermediates=return_intermediates,
        telem=telem,  # WO-BASIN-TELEM-HOOK-01
    )
    timing["ffn_ms"] = (time.perf_counter() - t0) * 1000
    peak_rss = max(peak_rss, get_rss_mb())

    # Ensure ffn_out has correct shape
    if ffn_out.ndim == 2:
        ffn_out = ffn_out.reshape(1, ffn_out.shape[0], ffn_out.shape[1])

    # Step 6: Residual connection 2
    t0 = time.perf_counter()
    output = X_mid + ffn_out
    timing["residual_2_ms"] = (time.perf_counter() - t0) * 1000

    del X_mid_norm, ffn_out  # Free memory

    # Get tracemalloc peak
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate memory stats
    # WO-APPLES-TO-APPLES-01: Use bytes for savings_factor calculation
    # to avoid rounding issues (0.0 MB -> infinity ratios)
    attn_memory = attn_receipt.to_dict().get("memory_audit", {})
    ffn_memory = ffn_receipt.to_dict().get("memory_audit", {})

    attn_full_mb = attn_memory.get("full_weights_mb", 0)
    ffn_full_mb = ffn_memory.get("full_weights_mb", 0)
    full_weights_mb = attn_full_mb + ffn_full_mb

    rss_delta_mb = peak_rss - baseline_rss
    # Convert to bytes for precise calculation
    full_weights_bytes = int(full_weights_mb * 1024 * 1024)
    rss_delta_bytes = int(rss_delta_mb * 1024 * 1024)
    # Minimum 1 page (4KB) to avoid division by zero
    rss_delta_bytes = max(rss_delta_bytes, 4096)
    # Calculate savings factor using bytes (not rounded MB)
    savings_factor = full_weights_bytes / rss_delta_bytes

    memory_audit = {
        "rss_baseline_mb": baseline_rss,
        "rss_peak_mb": peak_rss,
        "rss_delta_mb": rss_delta_mb,
        "rss_delta_bytes": rss_delta_bytes,  # Store bytes for audit
        "tracemalloc_peak_mb": peak_mem / (1024 * 1024),
        "full_weights_mb": full_weights_mb,
        "full_weights_bytes": full_weights_bytes,  # Store bytes for audit
        "savings_factor": savings_factor,
        "component_breakdown": {
            "attention_full_mb": attn_full_mb,
            "ffn_full_mb": ffn_full_mb,
            "norms_kb": (attn_norm_weight.nbytes + ffn_norm_weight.nbytes) / 1024,
        },
    }

    # Reshape output to match input
    if len(original_shape) == 2:
        output = output.squeeze(axis=0)

    duration_ms = (time.perf_counter() - t0_total) * 1000

    # Build receipt
    receipt = TransformerBlockReceipt(
        block_index=block_index,
        attention_receipt=attn_receipt.to_dict(),
        ffn_receipt=ffn_receipt.to_dict(),
        norms_info=norms_info,
        block_config={
            "batch": batch,
            "seq": seq,
            "d_model": d_model,
            "d_head": d_head,
            "architecture": "pre_norm_transformer",
            "norm_type": "rms_norm",
            "ffn_type": "swiglu",
        },
        memory_audit=memory_audit,
        timing=timing,
        duration_ms=duration_ms,
        status="PASS" if attn_receipt.status == "PASS" and ffn_receipt.status == "PASS" else "FAIL",
    )

    # Prepare intermediates if requested
    intermediates = None
    if return_intermediates:
        intermediates = {
            "X_mid": X_mid,
            "attention": attn_intermediates,
            "ffn": ffn_intermediates,
        }

    return output, receipt, intermediates


def stream_multi_block_forward(
    X: np.ndarray,
    manifest_path: Union[str, Path],
    block_indices: List[int],
    d_head: int = 128,
    verify_policy: VerifyPolicy = "trust_cached",
    gguf_path: Optional[Union[str, Path]] = None,
    norm_eps: float = 1e-5,  # WO-CDNA-NORM-EPS-FIX-01: Match GGUF metadata
    trace_hidden: bool = False,
) -> Tuple[np.ndarray, List[TransformerBlockReceipt], Optional[Dict[int, np.ndarray]]]:
    """
    Run multiple transformer blocks sequentially.

    Args:
        X: Input activations [batch, seq, d_model] or [seq, d_model].
        manifest_path: Path to hybrid_manifest_v2.json.
        block_indices: List of block indices to run (e.g., [0, 1, 2]).
        d_head: Head dimension for attention scaling.
        verify_policy: Block verification policy.
        gguf_path: Path to GGUF file for loading norm weights.
        norm_eps: Epsilon for RMSNorm.
        trace_hidden: If True, capture hidden state before each block for debugging.

    Returns:
        (final_output, list of receipts, hidden_trace or None)

        hidden_trace is a dict mapping block_index -> hidden state BEFORE that block.
        Key -1 contains the final output (after all blocks).
    """
    receipts = []
    hidden_trace: Optional[Dict[int, np.ndarray]] = {} if trace_hidden else None

    current_X = X
    for block_idx in block_indices:
        # Capture hidden state BEFORE this block (for debugging divergence)
        if trace_hidden and hidden_trace is not None:
            hidden_trace[block_idx] = current_X.copy()

        output, receipt, _ = stream_transformer_block_forward(
            current_X,
            manifest_path,
            block_idx,
            d_head=d_head,
            verify_policy=verify_policy,
            gguf_path=gguf_path,
            norm_eps=norm_eps,
            return_intermediates=False,
        )
        receipts.append(receipt)
        current_X = output

    # Capture final output
    if trace_hidden and hidden_trace is not None:
        hidden_trace[-1] = current_X.copy()

    return current_X, receipts, hidden_trace


if __name__ == "__main__":
    print("=== Stream Transformer Block from CDNA ===")
    print()
    print("Usage:")
    print("  from helix_cdc.regrow.stream_transformer_block import stream_transformer_block_forward")
    print("  output, receipt, _ = stream_transformer_block_forward(")
    print("      X=input_activations,")
    print("      manifest_path='seeds/hybrid_manifest_v2.json',")
    print("      block_index=0,")
    print("      gguf_path='model.gguf',  # For norm weights")
    print("  )")
    print()
    print("See tools/stream_transformer_block_verification.py for verification.")
