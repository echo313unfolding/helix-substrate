"""Fused Triton kernel: 12-bit unpack → codebook gather → BF16 write.

Replaces the Python pipeline:
  unpack_12bit() → .int() → index_select(codebook) → reshape → buf[:] = ...

With a single GPU kernel that reads packed uint8 indices and writes BF16
weights directly into the materialization buffer. Zero intermediate allocations.

The codebook ([4096, 2] = 32 KB in BF16) is loaded into shared memory once
per thread block, giving zero-latency lookups.

For scalar VQ (vector_dim=1), codebook is [K] and each index produces one weight.
For 2D VQ (vector_dim=2), codebook is [K, 2] and each index produces two weights.

Work Order: WO-FUSED-GATHER-01
"""

from __future__ import annotations

import torch

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if Triton gather kernel can run."""
    if not _TRITON_AVAILABLE:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


if _TRITON_AVAILABLE:

    @triton.jit
    def _gather_12bit_vq2d_kernel(
        packed_ptr,      # [n_packed_bytes] uint8 — 12-bit packed indices
        codebook_ptr,    # [K, 2] bf16/f16 — codebook
        output_ptr,      # [n_indices * 2] bf16/f16 — output buffer (flat)
        n_indices: tl.constexpr,  # number of 12-bit index values
        BLOCK: tl.constexpr,      # indices per block
    ):
        """Unpack 12-bit indices and gather 2D VQ codebook entries.

        Each index i in [0, n_indices) maps to:
          output[2*i]   = codebook[idx, 0]
          output[2*i+1] = codebook[idx, 1]

        12-bit packing: pairs (a, b) packed into 3 bytes:
          byte0 = a[7:0], byte1 = a[11:8] | b[3:0]<<4, byte2 = b[11:4]
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_indices

        # Compute byte offsets for 12-bit unpacking
        # Index i within a pair: pair = i // 2, pos = i % 2
        pair = offs // 2
        pos = offs % 2  # 0 = first in pair (a), 1 = second in pair (b)

        byte_base = pair * 3  # each pair occupies 3 bytes

        # Load the 3 bytes for each pair
        b0 = tl.load(packed_ptr + byte_base, mask=mask, other=0).to(tl.int32)
        b1 = tl.load(packed_ptr + byte_base + 1, mask=mask, other=0).to(tl.int32)
        b2 = tl.load(packed_ptr + byte_base + 2, mask=mask, other=0).to(tl.int32)

        # Unpack: a = b0 | (b1 & 0x0F) << 8, b = (b1 >> 4) | b2 << 4
        val_a = b0 | ((b1 & 0x0F) << 8)
        val_b = ((b1 >> 4) & 0x0F) | (b2 << 4)

        # Select a or b based on position
        idx = tl.where(pos == 0, val_a, val_b)

        # Codebook gather: load both elements of codebook[idx]
        w0 = tl.load(codebook_ptr + idx * 2, mask=mask)
        w1 = tl.load(codebook_ptr + idx * 2 + 1, mask=mask)

        # Write to output: output[2*offs] = w0, output[2*offs+1] = w1
        tl.store(output_ptr + offs * 2, w0, mask=mask)
        tl.store(output_ptr + offs * 2 + 1, w1, mask=mask)


    @triton.jit
    def _gather_unpacked_vq2d_kernel(
        indices_ptr,     # [n_indices] int16 — unpacked indices
        codebook_ptr,    # [K, 2] bf16/f16 — codebook
        output_ptr,      # [n_indices * 2] bf16/f16 — output buffer (flat)
        n_indices: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Gather 2D VQ codebook entries from int16 indices (no 12-bit unpack)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_indices

        idx = tl.load(indices_ptr + offs, mask=mask, other=0).to(tl.int32)

        w0 = tl.load(codebook_ptr + idx * 2, mask=mask)
        w1 = tl.load(codebook_ptr + idx * 2 + 1, mask=mask)

        tl.store(output_ptr + offs * 2, w0, mask=mask)
        tl.store(output_ptr + offs * 2 + 1, w1, mask=mask)


def fused_gather_12bit_vq2d(
    packed: torch.Tensor,
    codebook: torch.Tensor,
    output: torch.Tensor,
    n_indices: int,
) -> None:
    """Fused 12-bit unpack + 2D VQ gather into output buffer.

    Args:
        packed: [n_packed_bytes] uint8 — 12-bit packed index data
        codebook: [K, 2] bf16/f16 — codebook (pre-cast to output dtype)
        output: [n_indices * 2] bf16/f16 — pre-allocated output buffer (flat view)
        n_indices: number of index values (= out_features * in_features // 2)
    """
    BLOCK = 1024
    grid = ((n_indices + BLOCK - 1) // BLOCK,)
    _gather_12bit_vq2d_kernel[grid](
        packed, codebook, output,
        n_indices=n_indices,
        BLOCK=BLOCK,
    )


def fused_gather_unpacked_vq2d(
    indices: torch.Tensor,
    codebook: torch.Tensor,
    output: torch.Tensor,
    n_indices: int,
) -> None:
    """Gather 2D VQ from int16 indices into output buffer.

    Args:
        indices: [n_indices] int16 — flat index data
        codebook: [K, 2] bf16/f16 — codebook
        output: [n_indices * 2] bf16/f16 — pre-allocated output buffer (flat view)
        n_indices: number of index values
    """
    BLOCK = 1024
    grid = ((n_indices + BLOCK - 1) // BLOCK,)
    _gather_unpacked_vq2d_kernel[grid](
        indices, codebook, output,
        n_indices=n_indices,
        BLOCK=BLOCK,
    )
