"""
Triton fused affine-lattice matmul kernel.

For per-tensor affine quantization:
    W[o, i] = indices[o, i] * step + offset

The matmul Y = X @ W^T decomposes as:
    Y = step * (X @ indices^T) + offset * rowsum(X)

This kernel computes X @ indices^T with indices cast to BF16 per-tile
(no full-tensor cast, no codebook gather, no temporary allocation).
The step/offset scaling is applied on output store.

Compared to VQ gather-matmul:
- Eliminates codebook memory traffic (just 2 scalars vs 1KB table)
- Eliminates gather indirection (sequential loads vs random access)
- Same tl.dot accumulation structure as v3

Work Order: WO-AFFINE-LATTICE-MATMUL-01
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _affine_matmul_kernel(
        # Pointers
        x_ptr,          # [N, IN] float activations
        indices_ptr,    # [OUT, IN] uint8 quantized weights
        output_ptr,     # [N, OUT] float output
        rowsum_ptr,     # [N] float pre-computed rowsum(X)
        # Scalars
        step,           # float: codebook step size
        offset,         # float: codebook offset (vmin)
        # Dimensions
        N,              # batch size
        IN,             # input features
        OUT,            # output features
        # Strides
        stride_xn,
        stride_xi,
        stride_idx_o,
        stride_idx_i,
        stride_on,
        stride_oo,
        # Block sizes
        BLOCK_M: tl.constexpr,  # batch tile
        BLOCK_N: tl.constexpr,  # output tile
        BLOCK_K: tl.constexpr,  # reduction tile
    ):
        """
        Fused affine-lattice matmul: Y = step * (X @ indices^T) + offset * rowsum(X).

        No codebook gather. Indices are cast to BF16 per-tile in registers.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < N
        mask_n = offs_n < OUT

        # Accumulator for X @ indices^T (pre-scale)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, IN, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < IN

            # Load X tile: [BLOCK_M, BLOCK_K]
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xi,
                mask=mask_m[:, None] & mask_k[None, :], other=0.0
            )

            # Load indices tile: [BLOCK_N, BLOCK_K] as uint8
            idx_tile = tl.load(
                indices_ptr + offs_n[:, None] * stride_idx_o + offs_k[None, :] * stride_idx_i,
                mask=mask_n[:, None] & mask_k[None, :], other=0
            )

            # Cast to FP16 for tl.dot (two-step: uint8 → fp32 → fp16
            # avoids Triton LLIR issues on CC 7.5 Turing)
            w_tile = idx_tile.to(tl.float32).to(tl.float16)
            x_fp = x_tile.to(tl.float16)

            # Tiled matmul: acc += X_tile @ indices_tile^T
            acc = tl.dot(x_fp, tl.trans(w_tile), acc=acc)

        # Apply affine transform: Y = step * acc + offset * rowsum
        rowsum_tile = tl.load(rowsum_ptr + offs_m, mask=mask_m, other=0.0)
        result = step * acc + offset * rowsum_tile[:, None]

        # Store
        out_ptrs = output_ptr + offs_m[:, None] * stride_on + offs_n[None, :] * stride_oo
        tl.store(out_ptrs, result, mask=mask_m[:, None] & mask_n[None, :])


def fused_affine_matmul(
    x: torch.Tensor,            # [N, IN]
    indices: torch.Tensor,      # [OUT, IN] uint8
    step: float,
    offset: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused affine-lattice matmul.

    Y = step * (X @ indices^T) + offset * rowsum(X) [+ bias]

    Args:
        x:       [N, IN] float activations
        indices: [OUT, IN] uint8 quantized weights
        step:    per-tensor scale (codebook step size)
        offset:  per-tensor offset (codebook vmin)
        bias:    optional [OUT] bias vector

    Returns:
        [N, OUT] float output
    """
    assert x.is_cuda and indices.is_cuda, "fused_affine_matmul requires CUDA tensors"
    assert x.ndim == 2 and indices.ndim == 2
    assert x.shape[1] == indices.shape[1], f"IN mismatch: {x.shape[1]} vs {indices.shape[1]}"

    N, IN = x.shape
    OUT = indices.shape[0]

    # Pre-compute rowsum(X)
    rowsum = x.sum(dim=-1)  # [N]

    output = torch.empty(N, OUT, device=x.device, dtype=torch.float32)

    # Tile sizes
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (
        (N + BLOCK_M - 1) // BLOCK_M,
        (OUT + BLOCK_N - 1) // BLOCK_N,
    )

    _affine_matmul_kernel[grid](
        x, indices, output, rowsum,
        step, offset,
        N, IN, OUT,
        x.stride(0), x.stride(1),
        indices.stride(0), indices.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    if bias is not None:
        output += bias.unsqueeze(0)

    return output


def affine_matmul_reference(
    x: torch.Tensor,
    indices: torch.Tensor,
    step: float,
    offset: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reference (PyTorch) affine-lattice matmul. Works on CPU and CUDA.

    Y = step * (X @ indices_float^T) + offset * rowsum(X) [+ bias]
    """
    rowsum = x.sum(dim=-1, keepdim=True)  # [N, 1]
    idx_float = indices.to(x.dtype)       # [OUT, IN] — allocates temporary
    Y = step * (x @ idx_float.t()) + offset * rowsum
    if bias is not None:
        Y += bias.unsqueeze(0)
    return Y
