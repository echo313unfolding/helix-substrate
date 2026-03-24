"""
HelixAffineLinear — int8 block-affine drop-in nn.Linear replacement.

Codec: W ≈ ((uint8 - 128) / 127) * scale + bias, per block of 256 weights.
Ported from helix-cdc/helix_cdc/quant/fp8_block_affine.py (encoder only).

Storage:
    indices:  [out, in] uint8  (1 byte per weight)
    scales:   [n_blocks] fp16  (2 bytes per 256 weights)
    biases:   [n_blocks] fp16  (2 bytes per 256 weights)
    Overhead: 4 bytes / 256 weights = 1.56% → ~3.94x compression

Forward:
    Chunked dequant + matmul (no codebook gather — pure arithmetic).
    Optional FP16 matmul on CUDA for ~2x throughput.

Prototype — not a replacement for HelixLinear. Separate codec comparison.

Work Order: WO-HELIX-AFFINE-AUDIT-01
"""

from __future__ import annotations

from typing import Dict, Optional, Set

import torch
import torch.nn as nn


class HelixAffineLinear(nn.Module):
    """
    Drop-in nn.Linear replacement backed by int8 block-affine quantization.

    W[i, j] = ((indices[i, j] - 128) / 127) * scale[block_id] + bias_ab[block_id]
    where block_id = i * blocks_per_row + j // block_size.

    No codebook gather — dequant is pure arithmetic (sub, div, mul, add).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        indices: torch.Tensor,       # [out, in] uint8
        scales: torch.Tensor,        # [n_blocks] fp16
        biases_ab: torch.Tensor,     # [n_blocks] fp16 (block-affine bias, NOT linear bias)
        block_size: int = 256,
        bias: Optional[torch.Tensor] = None,
        tensor_name: str = "",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.blocks_per_row = in_features // block_size
        self.tensor_name = tensor_name

        assert indices.dtype == torch.uint8
        self.register_buffer("indices", indices.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("biases_ab", biases_ab.contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        N = x_2d.shape[0]
        output = torch.zeros(N, self.out_features, device=x.device, dtype=torch.float32)

        CHUNK = 256
        use_fp16 = x.is_cuda
        x_compute = x_2d.half() if use_fp16 else x_2d.float()

        bpr = self.blocks_per_row
        bs = self.block_size

        for i in range(0, self.out_features, CHUNK):
            end = min(i + CHUNK, self.out_features)
            chunk_rows = end - i

            # Dequantize: (uint8 - 128) / 127 * scale + bias
            int8_vals = self.indices[i:end].reshape(chunk_rows, bpr, bs).to(torch.int16) - 128
            sb_start = i * bpr
            sb_end = end * bpr
            s = self.scales[sb_start:sb_end].float().reshape(chunk_rows, bpr, 1)
            b = self.biases_ab[sb_start:sb_end].float().reshape(chunk_rows, bpr, 1)
            W_chunk = ((int8_vals.float() / 127.0) * s + b).reshape(chunk_rows, self.in_features)

            if use_fp16:
                output[:, i:end] = (x_compute @ W_chunk.half().t()).float()
            else:
                output[:, i:end] = x_compute @ W_chunk.t()

        if self.bias is not None:
            output += self.bias.unsqueeze(0)

        return output.reshape(*orig_shape[:-1], self.out_features)

    def decode_weight(self) -> torch.Tensor:
        """Full weight reconstruction (for debugging/validation)."""
        with torch.no_grad():
            int8_vals = self.indices.reshape(
                self.out_features, self.blocks_per_row, self.block_size
            ).to(torch.int16) - 128
            s = self.scales.float().reshape(self.out_features, self.blocks_per_row, 1)
            b = self.biases_ab.float().reshape(self.out_features, self.blocks_per_row, 1)
            return ((int8_vals.float() / 127.0) * s + b).reshape(
                self.out_features, self.in_features
            )

    def memory_savings(self) -> dict:
        dense_bytes = self.out_features * self.in_features * 4
        compressed = (
            self.indices.numel() * 1       # uint8 indices
            + self.scales.numel() * 2      # fp16 scales
            + self.biases_ab.numel() * 2   # fp16 biases
        )
        return {
            "dense_bytes": dense_bytes,
            "compressed_bytes": compressed,
            "ratio": round(dense_bytes / max(1, compressed), 2),
            "savings_pct": round(100 * (1 - compressed / dense_bytes), 1),
        }

    def extra_repr(self) -> str:
        savings = self.memory_savings()
        parts = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"block_size={self.block_size}",
            f"compression={savings['ratio']}x",
        ]
        if self.bias is not None:
            parts.append("bias=True")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Encoder (from-scratch, no helix-cdc dependency)
# ---------------------------------------------------------------------------

@torch.no_grad()
def quantize_linear_to_affine(
    weight: torch.Tensor,
    block_size: int = 256,
) -> tuple:
    """
    Quantize a weight tensor to int8 block-affine.

    Args:
        weight: [out, in] float32 weight tensor
        block_size: elements per block (must divide in_features)

    Returns:
        (indices, scales, biases_ab, stats)
        indices: [out, in] uint8
        scales:  [n_blocks] fp16
        biases_ab: [n_blocks] fp16
        stats: dict with error metrics
    """
    rows, cols = weight.shape
    assert cols % block_size == 0, f"cols {cols} not divisible by block_size {block_size}"
    bpr = cols // block_size

    W = weight.float()
    blocks = W.reshape(rows, bpr, block_size)

    # Per-block bias (mean) and scale (max abs residual — raw spread, NOT /127)
    bias = blocks.mean(dim=2)                           # [rows, bpr]
    residual = blocks - bias.unsqueeze(2)                # [rows, bpr, bs]
    scale = residual.abs().amax(dim=2).clamp(min=1e-6)   # [rows, bpr]

    # Quantize to int8 range [-127, 127], store as uint8 with +128 offset
    # q = round(residual * 127 / scale), dequant = (q / 127) * scale + bias
    quantized = (residual * 127.0 / scale.unsqueeze(2)).round().clamp(-128, 127)
    indices = (quantized.to(torch.int16) + 128).to(torch.uint8).reshape(rows, cols)

    # Reconstruction error
    recon = ((quantized / 127.0) * scale.unsqueeze(2) + bias.unsqueeze(2)).reshape(rows, cols)
    err = (recon - W).abs()

    stats = {
        "n_blocks": rows * bpr,
        "block_size": block_size,
        "max_abs_error": err.max().item(),
        "mean_abs_error": err.mean().item(),
        "cosine": float(torch.nn.functional.cosine_similarity(
            W.reshape(1, -1), recon.reshape(1, -1)
        ).item()),
    }

    return indices, scale.half().reshape(-1), bias.half().reshape(-1), stats


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------

def swap_to_affine(
    model: nn.Module,
    block_size: int = 256,
    skip_modules: Optional[Set[str]] = None,
) -> tuple:
    """
    Replace nn.Linear modules with HelixAffineLinear (int8 block-affine).

    Args:
        model: PyTorch model
        block_size: block size for quantization
        skip_modules: set of module names to skip (e.g., {"lm_head"})

    Returns:
        (model, n_replaced, total_stats)
    """
    if skip_modules is None:
        skip_modules = {"lm_head"}

    replaced = 0
    total_stats = {
        "max_abs_error": 0.0,
        "sum_cosine": 0.0,
        "n_tensors": 0,
        "compressed_bytes": 0,
        "dense_bytes": 0,
    }

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name in skip_modules:
            continue

        weight = module.weight.data
        rows, cols = weight.shape

        if cols % block_size != 0:
            continue

        indices, scales, biases_ab, stats = quantize_linear_to_affine(weight, block_size)

        new_mod = HelixAffineLinear(
            in_features=cols,
            out_features=rows,
            indices=indices,
            scales=scales,
            biases_ab=biases_ab,
            block_size=block_size,
            bias=module.bias.data.clone() if module.bias is not None else None,
            tensor_name=name,
        )

        # Replace in parent
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

        replaced += 1
        total_stats["max_abs_error"] = max(total_stats["max_abs_error"], stats["max_abs_error"])
        total_stats["sum_cosine"] += stats["cosine"]
        total_stats["n_tensors"] += 1
        savings = new_mod.memory_savings()
        total_stats["compressed_bytes"] += savings["compressed_bytes"]
        total_stats["dense_bytes"] += savings["dense_bytes"]

    if total_stats["n_tensors"] > 0:
        total_stats["avg_cosine"] = total_stats["sum_cosine"] / total_stats["n_tensors"]
        total_stats["overall_ratio"] = round(
            total_stats["dense_bytes"] / max(1, total_stats["compressed_bytes"]), 2
        )

    return model, replaced, total_stats
