#!/usr/bin/env python3
"""
stream_ffn_layer.py - Streaming MLP/FFN Layer from CDNA

WO-STREAM-FULL-BLOCK-01: Full FFN forward pass where gate, up, down weights
are all streamed from CDNAv2 format - never fully loaded.

Pipeline (Mistral SwiGLU):
  1. gate_out = X @ Wgate^T         # [batch, seq, 14336]
  2. gate_activated = silu(gate_out) # SiLU activation
  3. up_out = X @ Wup^T             # [batch, seq, 14336]
  4. intermediate = gate_activated * up_out  # Element-wise
  5. output = intermediate @ Wdown  # [batch, seq, 4096]

Memory Analysis (for Mistral blk.0, seq=256):
  - Input X: 4 MB ([1, 256, 4096])
  - gate_out, up_out: 14 MB each (intermediate, freed)
  - intermediate: 14 MB (freed after down projection)
  - Wgate (224MB), Wup (224MB), Wdown (224MB) - NEVER LOADED
  - Total: ~45 MB vs 672 MB full = 15x savings

Transpose Requirements:
  | Tensor   | Stored Shape   | Transpose? | Computation                    |
  |----------|----------------|------------|--------------------------------|
  | ffn_gate | [14336, 4096]  | YES        | X @ W^T → [batch, seq, 14336]  |
  | ffn_up   | [14336, 4096]  | YES        | X @ W^T → [batch, seq, 14336]  |
  | ffn_down | [4096, 14336]  | NO         | X @ W → [batch, seq, 4096]     |

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

from helix_substrate.stream_matmul import (
    StreamXWReceipt,
    stream_xw_from_cdna,
    stream_xw_from_manifest,
    get_rss_mb,
    VerifyPolicy,
)

@dataclass
class FFNLayerReceipt:
    """
    Receipt for full FFN layer forward pass.

    Chains individual StreamXWReceipts for gate, up, down projections.
    """
    schema: str = "stream_ffn_layer_receipt_v1"
    work_order: str = "WO-STREAM-FULL-BLOCK-01"
    block_index: int = 0

    # Per-projection receipts
    gate_receipt: Optional[Dict[str, Any]] = None
    up_receipt: Optional[Dict[str, Any]] = None
    down_receipt: Optional[Dict[str, Any]] = None

    # FFN computation stats
    ffn_config: Dict[str, Any] = field(default_factory=dict)
    activation_stats: Dict[str, Any] = field(default_factory=dict)

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
        # Extract claim hygiene fields for top-level visibility
        codec_versions = self.ffn_config.get("codec_versions_used", ["unknown"])
        streaming_modes = self.ffn_config.get("streaming_modes_used", ["unknown"])

        d = {
            "schema": self.schema,
            "work_order": self.work_order,
            "block_index": self.block_index,
            # Claim hygiene: top-level summary for easy auditing
            "claim_hygiene": {
                "codec_versions_used": codec_versions,
                "streaming_modes_used": streaming_modes,
                "all_true_block_streaming": streaming_modes == ["true_block_streaming"],
                "projection_modes": {
                    "gate": self.gate_receipt.get("projection_mode", "unknown") if self.gate_receipt else "unknown",
                    "up": self.up_receipt.get("projection_mode", "unknown") if self.up_receipt else "unknown",
                    "down": self.down_receipt.get("projection_mode", "unknown") if self.down_receipt else "unknown",
                },
            },
            "projections": {
                "gate": self.gate_receipt,
                "up": self.up_receipt,
                "down": self.down_receipt,
            },
            "ffn_config": self.ffn_config,
            "activation_stats": self.activation_stats,
            "memory_audit": self.memory_audit,
            "timing": self.timing,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "status": self.status,
        }
        if self.accuracy:
            d["accuracy"] = self.accuracy
        return d


def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU (Swish) activation: x * sigmoid(x)

    Numerically stable implementation.
    """
    # Use stable sigmoid: 1 / (1 + exp(-x)) = exp(x) / (exp(x) + 1) for positive x
    # This avoids overflow in exp(-x) for large positive x
    pos_mask = x >= 0
    neg_mask = ~pos_mask

    result = np.empty_like(x)

    # For positive x: x * exp(x) / (exp(x) + 1) = x * (1 - 1/(exp(x)+1))
    exp_pos = np.exp(np.minimum(x[pos_mask], 80))  # Clip to avoid overflow
    result[pos_mask] = x[pos_mask] * exp_pos / (exp_pos + 1)

    # For negative x: x * 1 / (1 + exp(-x)) = x * exp(x) / (1 + exp(x))
    exp_neg = np.exp(np.maximum(x[neg_mask], -80))  # Clip to avoid underflow
    result[neg_mask] = x[neg_mask] * exp_neg / (1 + exp_neg)

    return result


def _add_projection_mode(receipt_dict: Dict[str, Any], transpose_w: bool) -> Dict[str, Any]:
    """Add projection_mode field to receipt for human auditors (Sharp Edge 5B)."""
    receipt_dict["projection_mode"] = "x_wt" if transpose_w else "x_w"
    return receipt_dict


def stream_ffn_forward(
    X: np.ndarray,
    manifest_path: Union[str, Path],
    block_index: int,
    verify_policy: VerifyPolicy = "trust_cached",
    return_intermediates: bool = False,
    
) -> Tuple[np.ndarray, FFNLayerReceipt, Optional[Dict[str, np.ndarray]]]:
    """
    Full FFN layer from CDNA weights using SwiGLU activation.

    Computes: output = down(silu(gate(X)) * up(X))

    NEVER loads full Wgate, Wup, or Wdown into memory.

    Args:
        X: Input activations [batch, seq, d_model] or [seq, d_model].
        manifest_path: Path to hybrid_manifest_v2.json.
        block_index: Transformer block index (0-31 for Mistral).
        verify_policy: Block verification policy.
        return_intermediates: If True, return gate_out, up_out, intermediate in a dict.

    Returns:
        (output [batch, seq, d_model], receipt, intermediates or None)

    Shapes (Mistral-7B):
        X: [batch, seq, 4096]
        gate_out: [batch, seq, 14336]
        up_out: [batch, seq, 14336]
        intermediate: [batch, seq, 14336]
        output: [batch, seq, 4096]
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
    gate_name = f"blk.{block_index}.ffn_gate.weight"
    up_name = f"blk.{block_index}.ffn_up.weight"
    down_name = f"blk.{block_index}.ffn_down.weight"

    # Get tensor shapes from manifest to determine transpose mode
    tensor_shapes = {}
    for shard in manifest.get("shards", []):
        if shard["tensor_name"] in [gate_name, up_name, down_name]:
            tensor_shapes[shard["tensor_name"]] = tuple(shard["shape"])

    receipts = {}
    timing = {}

    # Step 1: Gate projection (needs transpose: [14336, 4096] stored)
    # X [batch, seq, 4096] @ Wgate^T [4096, 14336] = gate_out [batch, seq, 14336]
    t0 = time.perf_counter()
    gate_shape = tensor_shapes.get(gate_name, (14336, d_model))
    needs_transpose_gate = (gate_shape[0] != d_model and gate_shape[1] == d_model)

    gate_out, gate_receipt = stream_xw_from_manifest(
        X, gate_name, manifest, base_path, verify_policy,
        transpose_w=needs_transpose_gate, telem=telem,
    )
    timing["gate_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["gate"] = _add_projection_mode(gate_receipt.to_dict(), needs_transpose_gate)
    peak_rss = max(peak_rss, get_rss_mb())

    # Step 2: Apply SiLU activation to gate output
    t0 = time.perf_counter()
    gate_activated = silu(gate_out)
    timing["silu_activation_ms"] = (time.perf_counter() - t0) * 1000

    # Activation stats for verification
    activation_stats = {
        "gate_out_range": [float(gate_out.min()), float(gate_out.max())],
        "gate_activated_range": [float(gate_activated.min()), float(gate_activated.max())],
        "silu_reduction_factor": float(np.abs(gate_activated).mean() / (np.abs(gate_out).mean() + 1e-10)),
    }

    # Step 3: Up projection (needs transpose: [14336, 4096] stored)
    # X [batch, seq, 4096] @ Wup^T [4096, 14336] = up_out [batch, seq, 14336]
    t0 = time.perf_counter()
    up_shape = tensor_shapes.get(up_name, (14336, d_model))
    needs_transpose_up = (up_shape[0] != d_model and up_shape[1] == d_model)

    up_out, up_receipt = stream_xw_from_manifest(
        X, up_name, manifest, base_path, verify_policy,
        transpose_w=needs_transpose_up, telem=telem,
    )
    timing["up_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["up"] = _add_projection_mode(up_receipt.to_dict(), needs_transpose_up)
    peak_rss = max(peak_rss, get_rss_mb())

    activation_stats["up_out_range"] = [float(up_out.min()), float(up_out.max())]

    # Step 4: Element-wise multiply (gate_activated * up_out)
    t0 = time.perf_counter()
    intermediate = gate_activated * up_out
    timing["elementwise_multiply_ms"] = (time.perf_counter() - t0) * 1000

    activation_stats["intermediate_range"] = [float(intermediate.min()), float(intermediate.max())]
    peak_rss = max(peak_rss, get_rss_mb())

    # Free intermediate tensors
    del gate_out, gate_activated, up_out

    # Step 5: Down projection
    # intermediate [batch, seq, 14336] @ Wdown → output [batch, seq, 4096]
    # Wdown is stored as [4096, 14336] (rows=d_model, cols=d_intermediate)
    # For X @ W: X's last dim must match W's rows
    # Here intermediate has dim 14336, W has rows=4096, cols=14336
    # So we need transpose: X @ W^T = [batch, seq, 14336] @ [14336, 4096]
    t0 = time.perf_counter()
    down_shape = tensor_shapes.get(down_name, (d_model, 14336))
    d_intermediate = intermediate.shape[-1]
    # If W's rows don't match X's last dim, but W's cols do, use transpose
    needs_transpose_down = (down_shape[0] != d_intermediate and down_shape[1] == d_intermediate)

    output, down_receipt = stream_xw_from_manifest(
        intermediate, down_name, manifest, base_path, verify_policy,
        transpose_w=needs_transpose_down, telem=telem,
    )
    timing["down_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["down"] = _add_projection_mode(down_receipt.to_dict(), needs_transpose_down)
    peak_rss = max(peak_rss, get_rss_mb())

    # Get tracemalloc peak
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate memory stats
    full_gate_mb = gate_receipt.full_weight_mb
    full_up_mb = up_receipt.full_weight_mb
    full_down_mb = down_receipt.full_weight_mb
    full_weights_mb = full_gate_mb + full_up_mb + full_down_mb

    rss_delta = peak_rss - baseline_rss

    memory_audit = {
        "rss_baseline_mb": baseline_rss,
        "rss_peak_mb": peak_rss,
        "rss_delta_mb": rss_delta,
        "tracemalloc_peak_mb": peak_mem / (1024 * 1024),
        "full_weights_mb": full_weights_mb,
        "savings_factor": full_weights_mb / max(rss_delta, 0.1) if rss_delta > 0 else full_weights_mb,
        "weight_breakdown": {
            "Wgate": full_gate_mb,
            "Wup": full_up_mb,
            "Wdown": full_down_mb,
        },
    }

    # Reshape output to match input
    if len(original_shape) == 2:
        output = output.squeeze(axis=0)

    duration_ms = (time.perf_counter() - t0_total) * 1000

    # Aggregate codec/streaming info from projections for claim hygiene
    codec_versions = set()
    streaming_modes = set()
    for proj_key in ["gate", "up", "down"]:
        proj_receipt = receipts.get(proj_key, {})
        codec_versions.add(proj_receipt.get("codec_version", "unknown"))
        streaming_modes.add(proj_receipt.get("streaming_mode", "unknown"))

    # Build receipt
    receipt = FFNLayerReceipt(
        block_index=block_index,
        gate_receipt=receipts["gate"],
        up_receipt=receipts["up"],
        down_receipt=receipts["down"],
        ffn_config={
            "batch": batch,
            "seq": seq,
            "d_model": d_model,
            "d_intermediate": gate_shape[0] if needs_transpose_gate else gate_shape[1],
            "activation": "silu",
            "architecture": "swiglu",
            # Claim hygiene
            "codec_versions_used": list(codec_versions),
            "streaming_modes_used": list(streaming_modes),
            "transpose_modes": {
                "gate": needs_transpose_gate,
                "up": needs_transpose_up,
                "down": needs_transpose_down,
            },
        },
        activation_stats=activation_stats,
        memory_audit=memory_audit,
        timing=timing,
        duration_ms=duration_ms,
        status="PASS",
    )

    # Prepare intermediates if requested
    intermediates = None
    if return_intermediates:
        # Note: gate_out, gate_activated, up_out already freed
        # Only intermediate is preserved for return
        intermediates = {
            "intermediate": intermediate,
        }

    return output, receipt, intermediates


def stream_ffn_forward_from_paths(
    X: np.ndarray,
    wgate_path: Union[str, Path],
    wup_path: Union[str, Path],
    wdown_path: Union[str, Path],
    wgate_sidecar: Optional[Union[str, Path]] = None,
    wup_sidecar: Optional[Union[str, Path]] = None,
    wdown_sidecar: Optional[Union[str, Path]] = None,
    verify_policy: VerifyPolicy = "trust_cached",
) -> Tuple[np.ndarray, FFNLayerReceipt]:
    """
    Full FFN layer from explicit CDNA file paths.

    Like stream_ffn_forward but takes paths directly instead of manifest.

    Args:
        X: Input activations [batch, seq, d_model] or [seq, d_model].
        wgate_path, wup_path, wdown_path: Paths to CDNA files.
        wgate_sidecar, wup_sidecar, wdown_sidecar: Optional sidecar paths.
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

    receipts = {}
    timing = {}

    # Gate projection (assume transpose for typical Mistral layout)
    t0 = time.perf_counter()
    gate_out, gate_receipt = stream_xw_from_cdna(X, wgate_path, wgate_sidecar, verify_policy, transpose_w=True)
    timing["gate_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["gate"] = _add_projection_mode(gate_receipt.to_dict(), True)
    peak_rss = max(peak_rss, get_rss_mb())

    # SiLU activation
    t0 = time.perf_counter()
    gate_activated = silu(gate_out)
    timing["silu_activation_ms"] = (time.perf_counter() - t0) * 1000

    # Up projection
    t0 = time.perf_counter()
    up_out, up_receipt = stream_xw_from_cdna(X, wup_path, wup_sidecar, verify_policy, transpose_w=True)
    timing["up_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["up"] = _add_projection_mode(up_receipt.to_dict(), True)
    peak_rss = max(peak_rss, get_rss_mb())

    # Element-wise multiply
    t0 = time.perf_counter()
    intermediate = gate_activated * up_out
    timing["elementwise_multiply_ms"] = (time.perf_counter() - t0) * 1000
    peak_rss = max(peak_rss, get_rss_mb())

    del gate_out, gate_activated, up_out

    # Down projection (no transpose)
    t0 = time.perf_counter()
    output, down_receipt = stream_xw_from_cdna(intermediate, wdown_path, wdown_sidecar, verify_policy, transpose_w=False)
    timing["down_projection_ms"] = (time.perf_counter() - t0) * 1000
    receipts["down"] = _add_projection_mode(down_receipt.to_dict(), False)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory audit
    full_weights_mb = (
        gate_receipt.full_weight_mb +
        up_receipt.full_weight_mb +
        down_receipt.full_weight_mb
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

    receipt = FFNLayerReceipt(
        block_index=-1,  # Not from manifest
        gate_receipt=receipts["gate"],
        up_receipt=receipts["up"],
        down_receipt=receipts["down"],
        ffn_config={
            "batch": batch,
            "seq": seq,
            "d_model": d_model,
            "activation": "silu",
            "architecture": "swiglu",
        },
        memory_audit=memory_audit,
        timing=timing,
        duration_ms=duration_ms,
        status="PASS",
    )

    return output, receipt


if __name__ == "__main__":
    print("=== Stream FFN Layer from CDNA ===")
    print()
    print("Usage:")
    print("  from helix_cdc.regrow.stream_ffn_layer import stream_ffn_forward")
    print("  output, receipt, _ = stream_ffn_forward(")
    print("      X=input_activations,")
    print("      manifest_path='seeds/hybrid_manifest_v2.json',")
    print("      block_index=0,")
    print("  )")
    print()
    print("See tools/stream_ffn_layer_verification.py for verification.")
