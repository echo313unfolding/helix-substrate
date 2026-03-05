#!/usr/bin/env python3
"""
stream_attention_layer.py - Full Attention Layer from CDNA

WO-STREAM-ATTN-LAYER-01: Full attention forward pass where Wq, Wk, Wv, Wo
are all streamed from CDNAv2 format - never fully loaded.

Pipeline:
  1. Q = stream_xw_from_cdna(X, Wq)     # Never loads full Wq
  2. K = stream_xw_from_cdna(X, Wk)     # Never loads full Wk
  3. V = stream_xw_from_cdna(X, Wv)     # Never loads full Wv
  4. scores = Q @ K^T / sqrt(d)         # Standard attention
  5. attn = softmax(scores)             # Numerically stable (max-shift)
  6. context = attn @ V                 # Attention output
  7. output = stream_xw_from_cdna(context, Wo)  # Final projection

Memory Analysis (for Mistral blk.0, seq=256):
  - Input X: 4 MB ([1, 256, 4096])
  - Q, K, V, context: 4 MB each (intermediate, can be freed)
  - attn_weights: 0.25 MB ([1, 256, 256])
  - Wq (64MB), Wk (16MB), Wv (16MB), Wo (64MB) - NEVER LOADED
  - Total: ~20 MB vs 180 MB full = 9x savings

Work Order: WO-STREAM-ATTN-LAYER-01
"""

from __future__ import annotations

import json
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from helix_substrate.stream_matmul import (
    StreamXWReceipt,
    stream_xw_from_cdna,
    stream_xw_from_manifest,
    get_rss_mb,
)
from helix_substrate.rope import apply_rope_to_qk

VerifyPolicy = Literal["always", "sampled", "trust_cached"]


@dataclass
class AttentionLayerReceipt:
    """
    Receipt for full attention layer forward pass.

    Chains individual StreamXWReceipts for Q, K, V, O projections.
    """
    schema: str = "stream_attn_layer_receipt_v1"
    work_order: str = "WO-STREAM-ATTN-LAYER-01"
    block_index: int = 0

    # Per-projection receipts (each contains codec_version and streaming_mode)
    q_receipt: Optional[Dict[str, Any]] = None
    k_receipt: Optional[Dict[str, Any]] = None
    v_receipt: Optional[Dict[str, Any]] = None
    o_receipt: Optional[Dict[str, Any]] = None

    # Attention computation stats
    attention_config: Dict[str, Any] = field(default_factory=dict)
    softmax_stats: Dict[str, Any] = field(default_factory=dict)

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
        # Extract claim hygiene fields from attention_config for top-level visibility
        gqa_handling = self.attention_config.get("gqa_handling", "unknown")
        codec_versions = self.attention_config.get("codec_versions_used", ["unknown"])
        streaming_modes = self.attention_config.get("streaming_modes_used", ["unknown"])

        d = {
            "schema": self.schema,
            "work_order": self.work_order,
            "block_index": self.block_index,
            # Claim hygiene: top-level summary for easy auditing
            "claim_hygiene": {
                "gqa_handling": gqa_handling,
                "gqa_note": self.attention_config.get("gqa_note", ""),
                "codec_versions_used": codec_versions,
                "streaming_modes_used": streaming_modes,
                "all_true_block_streaming": streaming_modes == ["true_block_streaming"],
            },
            "projections": {
                "q": self.q_receipt,
                "k": self.k_receipt,
                "v": self.v_receipt,
                "o": self.o_receipt,
            },
            "attention_config": self.attention_config,
            "softmax_stats": self.softmax_stats,
            "memory_audit": self.memory_audit,
            "timing": self.timing,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "status": self.status,
        }
        if self.accuracy:
            d["accuracy"] = self.accuracy
        return d


def stable_softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax using max-shift method.

    Args:
        scores: Raw attention scores [..., seq, seq].
        axis: Axis to apply softmax over (default: -1).

    Returns:
        Attention weights with rows summing to 1.0.
    """
    # Subtract max for numerical stability
    scores_max = scores.max(axis=axis, keepdims=True)
    scores_shifted = scores - scores_max
    exp_scores = np.exp(scores_shifted)
    attn_weights = exp_scores / exp_scores.sum(axis=axis, keepdims=True)
    return attn_weights


def verify_softmax(attn_weights: np.ndarray) -> Dict[str, Any]:
    """Verify softmax is numerically stable."""
    row_sums = attn_weights.sum(axis=-1)
    return {
        "method": "max_shift",
        "numerically_stable": bool((row_sums >= 0.9999).all() and (row_sums <= 1.0001).all()),
        "row_sums_min": float(row_sums.min()),
        "row_sums_max": float(row_sums.max()),
        "all_non_negative": bool((attn_weights >= 0).all()),
    }


def stream_attention_forward(
    X: np.ndarray,
    manifest_path: Union[str, Path],
    block_index: int,
    d_head: int = 128,
    verify_policy: VerifyPolicy = "trust_cached",
    return_intermediates: bool = False,
    
) -> Tuple[np.ndarray, AttentionLayerReceipt, Optional[Dict[str, np.ndarray]]]:
    """
    Full attention layer from CDNA weights.

    Computes: output = softmax(Q @ K^T / sqrt(d_head)) @ V, then output @ Wo

    NEVER loads full Wq, Wk, Wv, or Wo into memory.

    Args:
        X: Input activations [batch, seq, d_model] or [seq, d_model].
        manifest_path: Path to hybrid_manifest_v2.json.
        block_index: Transformer block index (0-31 for Mistral).
        d_head: Head dimension for scaling (default: 128 for Mistral).
        verify_policy: Block verification policy.
        return_intermediates: If True, return Q, K, V, attn_weights in a dict.

    Returns:
        (output [batch, seq, d_model], receipt, intermediates or None)

    Note:
        This is single-head attention for simplicity. Multi-head would split
        Q, K, V along the head dimension and concatenate outputs.
    """
    t0_total = time.perf_counter()
    baseline_rss = get_rss_mb()
    peak_rss = baseline_rss

    tracemalloc.start()

    manifest_path = Path(manifest_path)

    # Load manifest from cache (WO-TOKENIZER-BRIDGE-01.1)
    from helix_cdc.regrow.cache import get_manifest_and_base
    manifest, base_path = get_manifest_and_base(manifest_path)

    # Handle input dimensions
    X = np.asarray(X, dtype=np.float32)
    original_shape = X.shape

    if X.ndim == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])
    elif X.ndim != 3:
        raise ValueError(f"X must be 2D [seq, d_model] or 3D [batch, seq, d_model], got shape {original_shape}")

    batch, seq, d_model = X.shape

    # Build tensor names for this block
    q_name = f"blk.{block_index}.attn_q.weight"
    k_name = f"blk.{block_index}.attn_k.weight"
    v_name = f"blk.{block_index}.attn_v.weight"
    o_name = f"blk.{block_index}.attn_output.weight"

    # Scale factor
    scale = 1.0 / np.sqrt(d_head)

    receipts = {}
    timing = {}

    # Get tensor shapes to handle GQA dimension mismatches
    # GQA (Grouped Query Attention) in Mistral uses fewer K/V heads than Q heads
    # This means Wk and Wv have different input dimensions than Wq
    tensor_shapes = {}
    for shard in manifest.get("shards", []):
        if shard["tensor_name"] in [q_name, k_name, v_name, o_name]:
            tensor_shapes[shard["tensor_name"]] = tuple(shard["shape"])

    # Track GQA handling mode for claim hygiene
    # "faithful_gqa" = proper GQA head grouping with K/V broadcast
    # Keys track which adjustments were made:
    # - k_transposed, v_transposed: K/V weights stored transposed, using X @ W^T
    # - o_adjusted: Output projection needed dimension adjustment (should not happen with faithful GQA)
    gqa_adjustments = {
        "k_transposed": False,
        "v_transposed": False,
        "o_adjusted": False,
    }

    # Step 1: Q = X @ Wq
    t0 = time.perf_counter()
    Q, q_receipt = stream_xw_from_manifest(X, q_name, manifest, base_path, verify_policy, telem=telem)
    timing["q_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["q"] = q_receipt.to_dict()
    peak_rss = max(peak_rss, get_rss_mb())

    # DEBUG: Compare Q projection against dense GGUF computation (WO-CDNA-DEBUG-PROJ-01)
    # Enabled via CDNA_DEBUG_PROJECTIONS=1 environment variable
    if block_index == 0 and os.environ.get("CDNA_DEBUG_PROJECTIONS"):
        from helix_cdc.regrow.tensor_accessor import TensorAccessor
        # Find GGUF path from manifest (top-level or metadata)
        gguf_path_for_debug = manifest.get("source_gguf") or manifest.get("metadata", {}).get("source_gguf")
        if gguf_path_for_debug:
            # Resolve relative path from manifest base
            gguf_full_path = base_path / gguf_path_for_debug
            if not gguf_full_path.exists():
                print(f"[DEBUG] GGUF not found at {gguf_full_path}, skipping projection debug")
            else:
                accessor = TensorAccessor.from_gguf(gguf_full_path)
                Wq_dense = accessor.load_tensor(q_name)
                # Q weight is [d_model, d_q] = [4096, 4096], so X @ W gives [batch, seq, d_q]
                # (NOT transposed - verified experimentally)
                Q_dense = X @ Wq_dense
                cos_q = np.dot(Q.flatten(), Q_dense.flatten()) / (
                    np.linalg.norm(Q) * np.linalg.norm(Q_dense) + 1e-10
                )
                max_err_q = np.abs(Q - Q_dense).max()
                print(f"[DEBUG] Block {block_index} Q projection: cosine={cos_q:.6f}, max_err={max_err_q:.6f}")
                if cos_q < 0.999:
                    print(f"[DEBUG]   WARNING: Q projection cosine {cos_q:.6f} < 0.999")
                    print(f"[DEBUG]   CDNA Q shape: {Q.shape}, Dense Q shape: {Q_dense.shape}")
                    print(f"[DEBUG]   CDNA Q mean: {Q.mean():.6f}, Dense Q mean: {Q_dense.mean():.6f}")

    # Step 2: K = X @ Wk (handle GQA where Wk may be stored transposed)
    # GQA models store K/V weights with fewer heads than Q.
    # If Wk shape is [d_k, d_model] instead of [d_model, d_k], use transpose mode.
    t0 = time.perf_counter()
    k_shape = tensor_shapes.get(k_name, (d_model, d_model))
    # Detect transposed storage: rows < d_model but cols == d_model
    needs_transpose_k = (k_shape[0] != d_model and k_shape[1] == d_model)
    K, k_receipt = stream_xw_from_manifest(
        X, k_name, manifest, base_path, verify_policy,
        transpose_w=needs_transpose_k, telem=telem,
    )
    if needs_transpose_k:
        gqa_adjustments["k_transposed"] = True
    timing["k_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["k"] = k_receipt.to_dict()
    peak_rss = max(peak_rss, get_rss_mb())

    # DEBUG: Compare K projection against dense GGUF computation (WO-CDNA-DEBUG-PROJ-01)
    if block_index == 0 and os.environ.get("CDNA_DEBUG_PROJECTIONS"):
        from helix_cdc.regrow.tensor_accessor import TensorAccessor
        gguf_path_for_debug = manifest.get("source_gguf") or manifest.get("metadata", {}).get("source_gguf")
        if gguf_path_for_debug:
            gguf_full_path = base_path / gguf_path_for_debug
            if gguf_full_path.exists():
                accessor = TensorAccessor.from_gguf(gguf_full_path)
                Wk_dense = accessor.load_tensor(k_name)
                # GGML K weight is [d_k, d_model], GQA: d_k = n_kv_heads * d_head
                # With transpose_w=True: X @ W^T where W is [d_k, d_model] gives [batch, seq, d_k]
                if needs_transpose_k:
                    K_dense = X @ Wk_dense.T
                else:
                    K_dense = X @ Wk_dense
                cos_k = np.dot(K.flatten(), K_dense.flatten()) / (
                    np.linalg.norm(K) * np.linalg.norm(K_dense) + 1e-10
                )
                max_err_k = np.abs(K - K_dense).max()
                print(f"[DEBUG] Block {block_index} K projection: cosine={cos_k:.6f}, max_err={max_err_k:.6f}, transpose={needs_transpose_k}")
                if cos_k < 0.999:
                    print(f"[DEBUG]   WARNING: K projection cosine {cos_k:.6f} < 0.999")
                    print(f"[DEBUG]   CDNA K shape: {K.shape}, Dense K shape: {K_dense.shape}")
                    print(f"[DEBUG]   Wk_dense shape: {Wk_dense.shape}")

    # Step 3: V = X @ Wv (handle GQA where Wv may be stored transposed)
    # Same as K: if Wv shape is [d_v, d_model], use transpose mode.
    t0 = time.perf_counter()
    v_shape = tensor_shapes.get(v_name, (d_model, d_model))
    # Detect transposed storage: rows < d_model but cols == d_model
    needs_transpose_v = (v_shape[0] != d_model and v_shape[1] == d_model)
    V, v_receipt = stream_xw_from_manifest(
        X, v_name, manifest, base_path, verify_policy,
        transpose_w=needs_transpose_v, telem=telem,
    )
    if needs_transpose_v:
        gqa_adjustments["v_transposed"] = True
    timing["v_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["v"] = v_receipt.to_dict()
    peak_rss = max(peak_rss, get_rss_mb())

    # DEBUG: Compare V projection against dense GGUF computation (WO-CDNA-DEBUG-PROJ-01)
    if block_index == 0 and os.environ.get("CDNA_DEBUG_PROJECTIONS"):
        from helix_cdc.regrow.tensor_accessor import TensorAccessor
        gguf_path_for_debug = manifest.get("source_gguf") or manifest.get("metadata", {}).get("source_gguf")
        if gguf_path_for_debug:
            gguf_full_path = base_path / gguf_path_for_debug
            if gguf_full_path.exists():
                accessor = TensorAccessor.from_gguf(gguf_full_path)
                Wv_dense = accessor.load_tensor(v_name)
                # GGML V weight is [d_v, d_model], GQA: d_v = n_kv_heads * d_head
                if needs_transpose_v:
                    V_dense = X @ Wv_dense.T
                else:
                    V_dense = X @ Wv_dense
                cos_v = np.dot(V.flatten(), V_dense.flatten()) / (
                    np.linalg.norm(V) * np.linalg.norm(V_dense) + 1e-10
                )
                max_err_v = np.abs(V - V_dense).max()
                print(f"[DEBUG] Block {block_index} V projection: cosine={cos_v:.6f}, max_err={max_err_v:.6f}, transpose={needs_transpose_v}")
                if cos_v < 0.999:
                    print(f"[DEBUG]   WARNING: V projection cosine {cos_v:.6f} < 0.999")
                    print(f"[DEBUG]   CDNA V shape: {V.shape}, Dense V shape: {V_dense.shape}")

    # Step 3.5: Apply RoPE (Rotary Position Embeddings) to Q and K
    # WO-CDNA-ROPE-01: Critical for positional awareness
    t0 = time.perf_counter()
    # WO-CDNA-ROPE-FIX-01: Use correct theta from model metadata
    # Mistral-7B uses rope_freq_base = 1,000,000 (NOT 10,000)
    # TODO: Read this from GGUF metadata instead of hardcoding
    rope_theta = 1000000.0
    Q, K = apply_rope_to_qk(Q, K, d_head, start_pos=0, theta=rope_theta)
    timing["rope_ms"] = (time.perf_counter() - t0) * 1000

    # Step 4-6: Faithful GQA Attention
    # GQA (Grouped Query Attention): K/V have fewer heads than Q
    # Each K/V head is shared across a group of Q heads
    #
    # Key shapes:
    # Q: [batch, seq, n_heads * d_head]
    # K: [batch, seq, n_kv_heads * d_head]
    # V: [batch, seq, n_kv_heads * d_head]
    #
    # For Mistral-7B: n_heads=32, n_kv_heads=8, d_head=128, group_size=4

    t0 = time.perf_counter()

    # Infer head counts from projection output dimensions
    d_q = Q.shape[-1]  # 4096 = n_heads * d_head
    d_k = K.shape[-1]  # 1024 = n_kv_heads * d_head
    d_v = V.shape[-1]  # 1024 = n_kv_heads * d_head

    n_heads = d_q // d_head      # 32
    n_kv_heads = d_k // d_head   # 8
    group_size = n_heads // n_kv_heads  # 4

    assert d_q % d_head == 0, f"Q dim {d_q} not divisible by d_head {d_head}"
    assert d_k % d_head == 0, f"K dim {d_k} not divisible by d_head {d_head}"
    assert n_heads % n_kv_heads == 0, f"n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads}"

    # Reshape into heads
    # Q: [batch, seq, n_heads * d_head] → [batch, n_heads, seq, d_head]
    Qh = Q.reshape(batch, seq, n_heads, d_head).transpose(0, 2, 1, 3)
    # K, V: [batch, seq, n_kv_heads * d_head] → [batch, n_kv_heads, seq, d_head]
    Kh = K.reshape(batch, seq, n_kv_heads, d_head).transpose(0, 2, 1, 3)
    Vh = V.reshape(batch, seq, n_kv_heads, d_head).transpose(0, 2, 1, 3)

    # Initialize context: [batch, n_heads, seq, d_head]
    context_heads = np.zeros((batch, n_heads, seq, d_head), dtype=Q.dtype)

    timing["head_reshape_ms"] = (time.perf_counter() - t0) * 1000

    # Compute attention per KV head group
    # Each KV head broadcasts across group_size Q heads
    t0 = time.perf_counter()

    all_attn_weights = []  # For verification

    for hkv in range(n_kv_heads):
        h0 = hkv * group_size  # Start Q head index
        h1 = (hkv + 1) * group_size  # End Q head index (exclusive)

        # Q for this group: [batch, group_size, seq, d_head]
        Qg = Qh[:, h0:h1, :, :]

        # Shared K/V for this group: [batch, 1, seq, d_head]
        Kshared = Kh[:, hkv:hkv+1, :, :]
        Vshared = Vh[:, hkv:hkv+1, :, :]

        # Attention scores: [batch, group_size, seq, seq]
        # Broadcast K across all Q heads in group
        # Qg @ Kshared^T: [batch, group_size, seq, d_head] @ [batch, 1, d_head, seq]
        scores = np.einsum('bgsd,bkdt->bgst', Qg, Kshared.transpose(0, 1, 3, 2)) * scale

        # WO-CDNA-CAUSAL-MASK-01: Apply causal mask for autoregressive attention
        # Each position can only attend to itself and earlier positions
        # mask[i,j] = -inf if j > i (future positions)
        causal_mask = np.triu(np.ones((seq, seq), dtype=np.float32), k=1) * -1e9
        scores = scores + causal_mask  # Broadcasting adds mask to all batches/heads

        # Softmax over key positions (last axis)
        weights = stable_softmax(scores, axis=-1)
        all_attn_weights.append(weights)

        # Context: [batch, group_size, seq, d_head]
        # weights @ Vshared: [batch, group_size, seq, seq] @ [batch, 1, seq, d_head]
        ctx = np.einsum('bgst,bktd->bgsd', weights, Vshared)

        context_heads[:, h0:h1, :, :] = ctx

    timing["attention_computation_ms"] = (time.perf_counter() - t0) * 1000
    peak_rss = max(peak_rss, get_rss_mb())

    # Reshape context back: [batch, n_heads, seq, d_head] → [batch, seq, n_heads * d_head]
    t0 = time.perf_counter()
    context = context_heads.transpose(0, 2, 1, 3).reshape(batch, seq, n_heads * d_head)
    timing["context_reshape_ms"] = (time.perf_counter() - t0) * 1000

    # Aggregate attention weights for verification (optional)
    attn_weights = np.concatenate(all_attn_weights, axis=1)  # [batch, n_heads, seq, seq]
    softmax_stats = verify_softmax(attn_weights)
    peak_rss = max(peak_rss, get_rss_mb())

    # Step 7: Output = context @ Wo
    # With faithful GQA, context is [batch, seq, n_heads * d_head] = [batch, seq, d_model]
    # This should naturally match Wo's input dimension (no padding needed)
    t0 = time.perf_counter()
    o_shape = tensor_shapes.get(o_name, (d_model, d_model))
    o_in_dim = o_shape[0]

    # Verify context dimension matches Wo input (should always match with faithful GQA)
    if context.shape[-1] != o_in_dim:
        # This shouldn't happen with faithful GQA, but handle gracefully
        if context.shape[-1] < o_in_dim:
            padded = np.zeros((batch, seq, o_in_dim), dtype=context.dtype)
            padded[:, :, :context.shape[-1]] = context
            output, o_receipt = stream_xw_from_manifest(padded, o_name, manifest, base_path, verify_policy, telem=telem)
            gqa_adjustments["o_adjusted"] = True
            gqa_adjustments["o_note"] = f"Unexpected: context dim {context.shape[-1]} < Wo in {o_in_dim}"
        else:
            context_sliced = context[:, :, :o_in_dim]
            output, o_receipt = stream_xw_from_manifest(context_sliced, o_name, manifest, base_path, verify_policy, telem=telem)
            gqa_adjustments["o_adjusted"] = True
            gqa_adjustments["o_note"] = f"Unexpected: context dim {context.shape[-1]} > Wo in {o_in_dim}"
    else:
        # Normal path: context dim matches Wo input
        output, o_receipt = stream_xw_from_manifest(context, o_name, manifest, base_path, verify_policy, telem=telem)

    timing["o_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["o"] = o_receipt.to_dict()
    peak_rss = max(peak_rss, get_rss_mb())

    # Get tracemalloc peak
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate memory stats
    # Full weight costs (if we loaded them all)
    full_wq_mb = q_receipt.full_weight_mb
    full_wk_mb = k_receipt.full_weight_mb
    full_wv_mb = v_receipt.full_weight_mb
    full_wo_mb = o_receipt.full_weight_mb
    full_weights_mb = full_wq_mb + full_wk_mb + full_wv_mb + full_wo_mb

    rss_delta = peak_rss - baseline_rss

    memory_audit = {
        "rss_baseline_mb": baseline_rss,
        "rss_peak_mb": peak_rss,
        "rss_delta_mb": rss_delta,
        "tracemalloc_peak_mb": peak_mem / (1024 * 1024),
        "full_weights_mb": full_weights_mb,
        "savings_factor": full_weights_mb / max(rss_delta, 0.1) if rss_delta > 0 else full_weights_mb,
        "weight_breakdown": {
            "Wq": full_wq_mb,
            "Wk": full_wk_mb,
            "Wv": full_wv_mb,
            "Wo": full_wo_mb,
        },
    }

    # Reshape output to match input
    if len(original_shape) == 2:
        output = output.squeeze(axis=0)

    duration_ms = (time.perf_counter() - t0_total) * 1000

    # Determine GQA handling mode
    # faithful_gqa: proper head grouping with K/V broadcast, transpose mode for transposed weights
    # unexpected_adjustment: if o_adjusted is True, something went wrong with dimension matching
    has_unexpected_adjustment = gqa_adjustments.get("o_adjusted", False)
    gqa_handling = "faithful_gqa" if not has_unexpected_adjustment else "faithful_gqa_with_unexpected_adjustment"

    # Aggregate codec/streaming info from projections for claim hygiene
    codec_versions = set()
    streaming_modes = set()
    for proj_key in ["q", "k", "v", "o"]:
        proj_receipt = receipts.get(proj_key, {})
        codec_versions.add(proj_receipt.get("codec_version", "unknown"))
        streaming_modes.add(proj_receipt.get("streaming_mode", "unknown"))

    # Build receipt
    receipt = AttentionLayerReceipt(
        block_index=block_index,
        q_receipt=receipts["q"],
        k_receipt=receipts["k"],
        v_receipt=receipts["v"],
        o_receipt=receipts["o"],
        attention_config={
            "d_head": d_head,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "group_size": group_size,
            "scale": scale,
            "batch": batch,
            "seq": seq,
            "d_model": d_model,
            "d_q": d_q,
            "d_k": d_k,
            "d_v": d_v,
            # Claim hygiene: GQA handling mode
            "gqa_handling": gqa_handling,
            "gqa_adjustments": gqa_adjustments,
            "gqa_note": (
                "Faithful GQA: K/V heads broadcast across Q head groups, transpose mode for K/V projections"
                if gqa_handling == "faithful_gqa" else
                f"Faithful GQA with unexpected output dimension adjustment: {gqa_adjustments.get('o_note', 'unknown')}"
            ),
            # Claim hygiene: codec and streaming mode summary
            "codec_versions_used": list(codec_versions),
            "streaming_modes_used": list(streaming_modes),
        },
        softmax_stats=softmax_stats,
        memory_audit=memory_audit,
        timing=timing,
        duration_ms=duration_ms,
        status="PASS" if softmax_stats["numerically_stable"] else "FAIL",
    )

    # Prepare intermediates if requested
    intermediates = None
    if return_intermediates:
        intermediates = {
            "Q": Q,
            "K": K,
            "V": V,
            "scores": scores,
            "attn_weights": attn_weights,
            "context": context,
        }

    return output, receipt, intermediates


def stream_attention_forward_from_paths(
    X: np.ndarray,
    wq_path: Union[str, Path],
    wk_path: Union[str, Path],
    wv_path: Union[str, Path],
    wo_path: Union[str, Path],
    wq_sidecar: Optional[Union[str, Path]] = None,
    wk_sidecar: Optional[Union[str, Path]] = None,
    wv_sidecar: Optional[Union[str, Path]] = None,
    wo_sidecar: Optional[Union[str, Path]] = None,
    d_head: int = 128,
    verify_policy: VerifyPolicy = "trust_cached",
) -> Tuple[np.ndarray, AttentionLayerReceipt]:
    """
    Full attention layer from explicit CDNA file paths.

    Like stream_attention_forward but takes paths directly instead of manifest.

    Args:
        X: Input activations [batch, seq, d_model] or [seq, d_model].
        wq_path, wk_path, wv_path, wo_path: Paths to CDNA files.
        wq_sidecar, wk_sidecar, wv_sidecar, wo_sidecar: Optional sidecar paths.
        d_head: Head dimension for scaling.
        verify_policy: Block verification policy.

    Returns:
        (output, receipt)
    """
    t0_total = time.perf_counter()
    baseline_rss = get_rss_mb()
    peak_rss = baseline_rss

    tracemalloc.start()

    # Handle input dimensions
    X = np.asarray(X, dtype=np.float32)
    original_shape = X.shape

    if X.ndim == 2:
        X = X.reshape(1, X.shape[0], X.shape[1])
    elif X.ndim != 3:
        raise ValueError(f"X must be 2D or 3D, got shape {original_shape}")

    batch, seq, d_model = X.shape
    scale = 1.0 / np.sqrt(d_head)

    receipts = {}
    timing = {}

    # Q projection
    t0 = time.perf_counter()
    Q, q_receipt = stream_xw_from_cdna(X, wq_path, wq_sidecar, verify_policy)
    timing["q_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["q"] = q_receipt.to_dict()
    peak_rss = max(peak_rss, get_rss_mb())

    # K projection
    t0 = time.perf_counter()
    K, k_receipt = stream_xw_from_cdna(X, wk_path, wk_sidecar, verify_policy)
    timing["k_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["k"] = k_receipt.to_dict()
    peak_rss = max(peak_rss, get_rss_mb())

    # V projection
    t0 = time.perf_counter()
    V, v_receipt = stream_xw_from_cdna(X, wv_path, wv_sidecar, verify_policy)
    timing["v_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["v"] = v_receipt.to_dict()
    peak_rss = max(peak_rss, get_rss_mb())

    # Attention scores
    t0 = time.perf_counter()
    scores = np.einsum('bsd,btd->bst', Q, K) * scale
    timing["attention_scores_ms"] = (time.perf_counter() - t0) * 1000

    # Softmax
    t0 = time.perf_counter()
    attn_weights = stable_softmax(scores, axis=-1)
    timing["softmax_ms"] = (time.perf_counter() - t0) * 1000
    softmax_stats = verify_softmax(attn_weights)

    # Context
    t0 = time.perf_counter()
    context = np.einsum('bst,btd->bsd', attn_weights, V)
    timing["context_computation_ms"] = (time.perf_counter() - t0) * 1000

    # Output projection
    t0 = time.perf_counter()
    output, o_receipt = stream_xw_from_cdna(context, wo_path, wo_sidecar, verify_policy)
    timing["o_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["o"] = o_receipt.to_dict()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory audit
    full_weights_mb = (
        q_receipt.full_weight_mb +
        k_receipt.full_weight_mb +
        v_receipt.full_weight_mb +
        o_receipt.full_weight_mb
    )
    rss_delta = peak_rss - baseline_rss

    memory_audit = {
        "rss_baseline_mb": baseline_rss,
        "rss_peak_mb": peak_rss,
        "rss_delta_mb": rss_delta,
        "tracemalloc_peak_mb": peak_mem / (1024 * 1024),
        "full_weights_mb": full_weights_mb,
        "savings_factor": full_weights_mb / max(rss_delta, 0.1) if rss_delta > 0 else full_weights_mb,
    }

    if len(original_shape) == 2:
        output = output.squeeze(axis=0)

    duration_ms = (time.perf_counter() - t0_total) * 1000

    receipt = AttentionLayerReceipt(
        block_index=-1,  # Not from manifest
        q_receipt=receipts["q"],
        k_receipt=receipts["k"],
        v_receipt=receipts["v"],
        o_receipt=receipts["o"],
        attention_config={
            "d_head": d_head,
            "scale": scale,
            "batch": batch,
            "seq": seq,
            "d_model": d_model,
        },
        softmax_stats=softmax_stats,
        memory_audit=memory_audit,
        timing=timing,
        duration_ms=duration_ms,
        status="PASS" if softmax_stats["numerically_stable"] else "FAIL",
    )

    return output, receipt


if __name__ == "__main__":
    print("=== Stream Attention Layer from CDNA ===")
    print()
    print("Usage:")
    print("  from helix_cdc.regrow.stream_attention_layer import stream_attention_forward")
    print("  output, receipt, _ = stream_attention_forward(")
    print("      X=input_activations,")
    print("      manifest_path='seeds/hybrid_manifest_v2.json',")
    print("      block_index=0,")
    print("  )")
    print()
    print("See tools/streaming_attention_layer_verification.py for verification.")
