"""
Fused per-group affine matmul kernels for CC 7.5 (Turing).

Uses FP16 tl.dot (NOT BF16 — Turing has no BF16 tensor cores).

Two kernels:
  1. Tiled kernel for N>1 (prefill) — FP16 tensor cores via tl.dot
  2. Element-wise kernel for N=1 (decode) — no tl.dot, 1D accumulator

Math:
  W[o, i] = indices[o, i] * scale[o, i//G] + offset[o, i//G]
  Y = X @ W^T

Dequant happens in registers per group tile — no W materialization.

Work Order: WO-AFFINE-GROUP-MATMUL-01
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
    def _affine_group_tiled_kernel(
        # Pointers
        x_ptr,          # [N, IN] float activations
        indices_ptr,    # [OUT, IN] uint8 quantized weights
        scale_ptr,      # [OUT, n_groups] float16 per-group scales
        offset_ptr,     # [OUT, n_groups] float16 per-group offsets
        output_ptr,     # [N, OUT] float32 output
        # Dimensions
        N, IN, OUT, n_groups,
        # Strides
        stride_xn, stride_xi,
        stride_idx_o, stride_idx_i,
        stride_so, stride_sg,
        stride_on, stride_oo,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        """
        Fused per-group affine matmul for N>1 (prefill).

        Per group g, dequants indices in registers:
            w_tile = indices_tile * scale[:, None] + offset[:, None]
        Then accumulates via FP16 tl.dot:
            acc += x_fp16 @ w_tile^T
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < N
        mask_n = offs_n < OUT

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for g in range(n_groups):
            k_start = g * GROUP_SIZE
            offs_k = k_start + tl.arange(0, GROUP_SIZE)

            # Load X tile: [BLOCK_M, GROUP_SIZE]
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xi,
                mask=mask_m[:, None], other=0.0
            )

            # Load indices tile: [BLOCK_N, GROUP_SIZE]
            idx_tile = tl.load(
                indices_ptr + offs_n[:, None] * stride_idx_o + offs_k[None, :] * stride_idx_i,
                mask=mask_n[:, None], other=0
            )

            # Load per-group scale/offset: [BLOCK_N]
            scale = tl.load(
                scale_ptr + offs_n * stride_so + g * stride_sg,
                mask=mask_n, other=1.0
            )
            offset = tl.load(
                offset_ptr + offs_n * stride_so + g * stride_sg,
                mask=mask_n, other=0.0
            )

            # Dequant in registers: w = idx * scale[:, None] + offset[:, None]
            # Two-step cast: uint8 → fp32 → fp16 (avoids Triton LLIR issues)
            idx_fp = idx_tile.to(tl.float32).to(tl.float16)
            w_tile = idx_fp * scale[:, None].to(tl.float16) + offset[:, None].to(tl.float16)

            # X to FP16 for tl.dot
            x_fp = x_tile.to(tl.float16)

            # Tiled matmul: acc += X_fp16 @ W_fp16^T → FP32 accumulator
            acc = tl.dot(x_fp, tl.trans(w_tile), acc=acc)

        # Store
        out_ptrs = output_ptr + offs_m[:, None] * stride_on + offs_n[None, :] * stride_oo
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


    @triton.jit
    def _affine_group_decode_kernel(
        # Pointers
        x_ptr,          # [IN] float32 input vector (flattened from [1, IN])
        indices_ptr,    # [OUT, IN] uint8 quantized weights
        scale_ptr,      # [OUT, n_groups] float16 per-group scales
        offset_ptr,     # [OUT, n_groups] float16 per-group offsets
        output_ptr,     # [OUT] float32 output
        # Dimensions
        IN, OUT, n_groups,
        # Strides
        stride_idx_o, stride_idx_i,
        stride_so, stride_sg,
        # Tile sizes
        BLOCK_OUT: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        """
        Fused per-group affine decode for N=1.

        Uses 1D [BLOCK_OUT] accumulator with element-wise multiply
        (no tl.dot — avoids BLOCK_M=16 padding waste for single tokens).

        Per group g, algebraic decomposition:
            partial = sum(x_vec * indices_fp, axis=1)   [BLOCK_OUT]
            x_sum = sum(x_vec)                           scalar
            acc += scale * partial + offset * x_sum
        """
        pid = tl.program_id(0)
        offs_out = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        mask_out = offs_out < OUT

        acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

        for g in range(n_groups):
            k_start = g * GROUP_SIZE
            offs_k = k_start + tl.arange(0, GROUP_SIZE)

            # Load x vector: [GROUP_SIZE]
            x_vec = tl.load(x_ptr + offs_k)

            # Load indices: [BLOCK_OUT, GROUP_SIZE]
            idx_tile = tl.load(
                indices_ptr + offs_out[:, None] * stride_idx_o + offs_k[None, :] * stride_idx_i,
                mask=mask_out[:, None], other=0
            )

            # Element-wise dot product per output
            idx_fp = idx_tile.to(tl.float32)
            partial = tl.sum(x_vec[None, :] * idx_fp, axis=1)  # [BLOCK_OUT]
            x_sum = tl.sum(x_vec)  # scalar

            # Load per-group scale/offset: [BLOCK_OUT]
            scale = tl.load(
                scale_ptr + offs_out * stride_so + g * stride_sg,
                mask=mask_out, other=1.0
            )
            offset = tl.load(
                offset_ptr + offs_out * stride_so + g * stride_sg,
                mask=mask_out, other=0.0
            )

            # Accumulate: acc += scale * partial + offset * x_sum
            acc += scale.to(tl.float32) * partial + offset.to(tl.float32) * x_sum

        tl.store(output_ptr + offs_out, acc, mask=mask_out)


    @triton.jit
    def _affine_group_decode_packed6_kernel(
        # Pointers
        x_ptr,          # [IN] float input vector
        packed_ptr,     # [OUT, IN*3//4] uint8 packed 6-bit indices
        scale_ptr,      # [OUT, n_groups] float16
        offset_ptr,     # [OUT, n_groups] float16
        output_ptr,     # [OUT] float32 output
        # Dimensions
        IN, OUT, n_groups,
        # Strides
        packed_stride,  # stride between rows in packed array (= IN * 3 // 4)
        stride_so, stride_sg,
        # Tile sizes
        BLOCK_OUT: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        """
        Fused decode for N=1 with 6-bit packed indices.

        Unpacks 3 bytes → 4 values in-register. No separate unpack step.
        Loads: 3 × [BLOCK_OUT, GROUP_SIZE] uint8 from packed data (with L1 coalescing).
        """
        pid = tl.program_id(0)
        offs_out = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        mask_out = offs_out < OUT

        acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

        # Precompute unpack offsets (constant across groups)
        offs_k = tl.arange(0, GROUP_SIZE)  # [GROUP_SIZE]
        quad = offs_k // 4                  # which quad (0..31 for GS=128)
        sub = offs_k % 4                    # position in quad (0..3)
        byte_rel0 = quad * 3                # byte offsets within group
        byte_rel1 = quad * 3 + 1
        byte_rel2 = quad * 3 + 2

        for g in range(n_groups):
            packed_group_start = g * (GROUP_SIZE * 3 // 4)

            # Load packed bytes: [BLOCK_OUT, GROUP_SIZE] with coalesced access
            # (4 consecutive values share 3 bytes → L1 cache handles duplicates)
            b0 = tl.load(
                packed_ptr + offs_out[:, None] * packed_stride
                + (packed_group_start + byte_rel0)[None, :],
                mask=mask_out[:, None], other=0,
            )
            b1 = tl.load(
                packed_ptr + offs_out[:, None] * packed_stride
                + (packed_group_start + byte_rel1)[None, :],
                mask=mask_out[:, None], other=0,
            )
            b2 = tl.load(
                packed_ptr + offs_out[:, None] * packed_stride
                + (packed_group_start + byte_rel2)[None, :],
                mask=mask_out[:, None], other=0,
            )

            # Unpack 6-bit values in registers
            v = tl.where(sub[None, :] == 0, b0 & 0x3F,
                tl.where(sub[None, :] == 1, ((b0 >> 6) | (b1 << 2)) & 0x3F,
                tl.where(sub[None, :] == 2, ((b1 >> 4) | (b2 << 4)) & 0x3F,
                                            (b2 >> 2) & 0x3F)))

            # Load x vector slice
            x_vec = tl.load(x_ptr + g * GROUP_SIZE + offs_k)

            # Element-wise dot product
            idx_fp = v.to(tl.float32)
            partial = tl.sum(x_vec[None, :] * idx_fp, axis=1)
            x_sum = tl.sum(x_vec)

            # Scale/offset
            scale = tl.load(scale_ptr + offs_out * stride_so + g * stride_sg,
                            mask=mask_out, other=1.0)
            offset = tl.load(offset_ptr + offs_out * stride_so + g * stride_sg,
                             mask=mask_out, other=0.0)
            acc += scale.to(tl.float32) * partial + offset.to(tl.float32) * x_sum

        tl.store(output_ptr + offs_out, acc, mask=mask_out)


def fused_affine_group_matmul_packed6(
    x: torch.Tensor,            # [N, IN]
    packed_indices: torch.Tensor,  # [OUT * IN * 3 // 4] uint8 packed
    scales: torch.Tensor,       # [OUT, n_groups] float16
    offsets: torch.Tensor,      # [OUT, n_groups] float16
    out_features: int,
    in_features: int,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused per-group affine matmul with 6-bit packed indices. N=1 only."""
    assert x.is_cuda
    N, IN = x.shape
    OUT = out_features
    assert IN == in_features
    assert N == 1, "Packed 6-bit kernel currently supports N=1 decode only"
    n_groups = IN // group_size
    packed_stride = IN * 3 // 4  # bytes per row in packed array

    output = torch.empty(N, OUT, device=x.device, dtype=torch.float32)
    BLOCK_OUT = 64
    grid = ((OUT + BLOCK_OUT - 1) // BLOCK_OUT,)

    _affine_group_decode_packed6_kernel[grid](
        x.contiguous().view(-1),
        packed_indices,
        scales, offsets,
        output.view(-1),
        IN, OUT, n_groups,
        packed_stride,
        scales.stride(0), scales.stride(1),
        BLOCK_OUT=BLOCK_OUT, GROUP_SIZE=group_size,
    )

    if bias is not None:
        output += bias.unsqueeze(0)
    return output


def fused_affine_group_matmul(
    x: torch.Tensor,            # [N, IN]
    indices: torch.Tensor,      # [OUT, IN] uint8
    scales: torch.Tensor,       # [OUT, n_groups] float16
    offsets: torch.Tensor,      # [OUT, n_groups] float16
    group_size: int = 128,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused per-group affine matmul. No W materialization.

    Y[n, o] = sum_i X[n, i] * (indices[o, i] * scale[o, i//G] + offset[o, i//G])

    Dispatches to:
      - N=1: element-wise decode kernel
      - N>1: FP16 tl.dot tiled kernel
    """
    assert x.is_cuda and indices.is_cuda
    assert x.ndim == 2 and indices.ndim == 2
    N, IN = x.shape
    OUT = indices.shape[0]
    assert indices.shape[1] == IN
    n_groups = IN // group_size
    assert scales.shape == (OUT, n_groups)
    assert offsets.shape == (OUT, n_groups)

    output = torch.empty(N, OUT, device=x.device, dtype=torch.float32)

    if N == 1:
        # Decode path: element-wise, no tl.dot
        BLOCK_OUT = 64
        grid = ((OUT + BLOCK_OUT - 1) // BLOCK_OUT,)
        _affine_group_decode_kernel[grid](
            x.contiguous().view(-1),  # flatten to [IN]
            indices, scales, offsets,
            output.view(-1),  # flatten to [OUT]
            IN, OUT, n_groups,
            indices.stride(0), indices.stride(1),
            scales.stride(0), scales.stride(1),
            BLOCK_OUT=BLOCK_OUT, GROUP_SIZE=group_size,
        )
    else:
        # Prefill path: FP16 tl.dot
        BLOCK_M = 16
        BLOCK_N = 64
        grid = (
            (N + BLOCK_M - 1) // BLOCK_M,
            (OUT + BLOCK_N - 1) // BLOCK_N,
        )
        _affine_group_tiled_kernel[grid](
            x, indices, scales, offsets, output,
            N, IN, OUT, n_groups,
            x.stride(0), x.stride(1),
            indices.stride(0), indices.stride(1),
            scales.stride(0), scales.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, GROUP_SIZE=group_size,
        )

    if bias is not None:
        output += bias.unsqueeze(0)

    return output


def affine_group_matmul_reference(
    x: torch.Tensor,
    indices: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference (PyTorch) per-group affine matmul. CPU or CUDA."""
    OUT, IN = indices.shape
    n_groups = IN // group_size
    # Dequant W per group
    idx_groups = indices.reshape(OUT, n_groups, group_size).float()
    W = (idx_groups * scales[:, :, None].float() + offsets[:, :, None].float()).reshape(OUT, IN)
    Y = x.float() @ W.t()
    if bias is not None:
        Y += bias.float().unsqueeze(0)
    return Y
