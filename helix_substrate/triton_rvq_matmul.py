"""
Triton fused RVQ (Residual Vector Quantization) gather-matmul kernel.

Computes output = x @ W^T where W = codebook1[high_nibble] + codebook2[low_nibble],
with packed 4-bit nibble indices (two 4-bit indices per uint8 byte).

Packing format:
    packed_byte = (stage1_idx << 4) | stage2_idx
    stage1 (coarse): high nibble, values 0-15
    stage2 (residual): low nibble, values 0-15
    w[i,j] = codebook1[packed[i,j] >> 4] + codebook2[packed[i,j] & 0xF]

Storage per weight element: 1 byte (same as VQ-256)
Codebook storage: 2 * 16 * 4 = 128 bytes (vs 1024 for VQ-256)

Quality: matches VQ-256 cosine (0.9997+) with only 32 total centroids.
PPL: +0.10% on TinyLlama (proven, WO-RVQ-8X-01 Phase 1b receipt).
Mixed RVQ/8-bit: +0.05% PPL at 7.7x from FP32 (~3.8x from BF16).

Kernel algorithm:
    For each (BLOCK_M, BLOCK_N) output tile, iterate over BLOCK_K reduction tiles:
        1. Load packed indices [BLOCK_N, BLOCK_K] uint8
        2. Unpack: high = packed >> 4, low = packed & 0xF
        3. Gather: w1 = codebook1[high], w2 = codebook2[low]
        4. Sum: w = w1 + w2
        5. Cast to FP16, tl.dot with FP32 accumulate

Memory: only 2 codebooks (128B total) + packed indices (uint8) touched.
W never exists in global memory. Same memory model as VQ kernel.

Work Order: WO-RVQ-8X-01 Phase 2 (runtime kernel)
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Reuse shape-keyed config from VQ kernel (same shapes, same perf profile)
from helix_substrate.triton_vq_matmul import _get_config

# --- Kernel Version Constants ---
KERNEL_VERSION = "rvq_v1_tiled_fp16dot"
KERNEL_IMPL = "triton_rvq_matmul"


def get_kernel_metadata() -> dict:
    """Return version/dispatch metadata for receipt embedding."""
    meta = {
        "kernel_impl": KERNEL_IMPL,
        "kernel_version": KERNEL_VERSION,
        "dispatch_selected": KERNEL_VERSION,
        "triton_version": None,
        "torch_version": torch.__version__,
        "cuda_version": None,
        "compute_capability": None,
    }
    if HAS_TRITON:
        meta["triton_version"] = triton.__version__
    if torch.cuda.is_available():
        meta["cuda_version"] = torch.version.cuda
        props = torch.cuda.get_device_properties(0)
        meta["compute_capability"] = f"{props.major}.{props.minor}"
    return meta


# ---------------------------------------------------------------------------
# Nibble packing/unpacking utilities
# ---------------------------------------------------------------------------

def pack_nibbles(idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit index tensors into one uint8 tensor.

    Args:
        idx1: [*shape] int tensor, values 0-15 (coarse stage, stored in high nibble)
        idx2: [*shape] int tensor, values 0-15 (residual stage, stored in low nibble)

    Returns:
        [*shape] uint8 tensor where each byte = (idx1 << 4) | idx2
    """
    assert idx1.shape == idx2.shape
    return ((idx1 & 0xF) << 4 | (idx2 & 0xF)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a uint8 tensor into two 4-bit index tensors.

    Args:
        packed: [*shape] uint8 tensor

    Returns:
        (idx1, idx2) both [*shape] uint8, values 0-15
        idx1 = high nibble (coarse), idx2 = low nibble (residual)
    """
    # Cast to int32 for bitwise ops (uint8 >> 4 can overflow in some backends)
    p = packed.to(torch.int32)
    idx1 = ((p >> 4) & 0xF).to(torch.uint8)
    idx2 = (p & 0xF).to(torch.uint8)
    return idx1, idx2


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAS_TRITON:
    # Reuse sidecar kernel from VQ file
    from helix_substrate.triton_vq_matmul import _sidecar_fused_kernel

    @triton.jit
    def _rvq_gather_matmul_tiled_kernel(
        # Pointers
        x_ptr,              # [N, IN] float32 input activations
        codebook1_ptr,      # [16] float32 stage-1 (coarse) codebook
        codebook2_ptr,      # [16] float32 stage-2 (residual) codebook
        packed_idx_ptr,     # [OUT, IN] uint8 packed nibble indices
        output_ptr,         # [N, OUT] float32 output
        # Dimensions
        N,                  # batch size (flattened)
        IN,                 # input features
        OUT,                # output features
        # Strides
        stride_xn,          # x stride along N
        stride_xi,          # x stride along IN
        stride_idx_o,       # packed_idx stride along OUT
        stride_idx_i,       # packed_idx stride along IN
        stride_on,          # output stride along N
        stride_oo,          # output stride along OUT
        # Block sizes (constexpr for compilation)
        BLOCK_M: tl.constexpr,   # batch tile (16)
        BLOCK_N: tl.constexpr,   # output tile (64)
        BLOCK_K: tl.constexpr,   # reduction tile (32-64)
    ):
        """
        Tiled RVQ gather-matmul with nibble unpacking and tl.dot.

        For each (BLOCK_M, BLOCK_N) output tile, reduces over IN in BLOCK_K steps:
            1. Load packed uint8 indices → unpack high/low nibbles
            2. Gather from two codebooks → sum to reconstruct weight tile
            3. FP16 tl.dot with FP32 accumulate

        Same performance profile as VQ v3 kernel — the extra unpack + second
        gather is negligible vs the tl.dot and memory loads.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < N
        mask_n = offs_n < OUT

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, IN, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < IN

            # Load x tile: [BLOCK_M, BLOCK_K] float32
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xi,
                mask=mask_m[:, None] & mask_k[None, :], other=0.0
            )

            # Load packed indices: [BLOCK_N, BLOCK_K] uint8
            packed = tl.load(
                packed_idx_ptr + offs_n[:, None] * stride_idx_o + offs_k[None, :] * stride_idx_i,
                mask=mask_n[:, None] & mask_k[None, :], other=0
            )

            # Unpack nibbles (cast to int32 for bitwise + pointer arithmetic)
            packed_i32 = packed.to(tl.int32)
            idx_coarse = packed_i32 >> 4          # high nibble: stage 1
            idx_residual = packed_i32 & 0xF       # low nibble: stage 2

            # Two codebook gathers → sum
            w_coarse = tl.load(codebook1_ptr + idx_coarse)     # [BLOCK_N, BLOCK_K]
            w_residual = tl.load(codebook2_ptr + idx_residual)  # [BLOCK_N, BLOCK_K]
            w_tile = w_coarse + w_residual

            # FP16 compute, FP32 accumulate (proven on CC 7.5)
            x_f16 = x_tile.to(tl.float16)
            w_f16 = w_tile.to(tl.float16)
            acc = tl.dot(x_f16, tl.trans(w_f16), acc=acc)

        # Store output tile
        out_ptrs = output_ptr + offs_m[:, None] * stride_on + offs_n[None, :] * stride_oo
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def fused_rvq_matmul(
    x: torch.Tensor,
    codebook1: torch.Tensor,
    codebook2: torch.Tensor,
    packed_indices: torch.Tensor,
    sidecar_rows: Optional[torch.Tensor] = None,
    sidecar_cols: Optional[torch.Tensor] = None,
    sidecar_deltas: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    _dispatch_log: Optional[dict] = None,
    sidecar_phase: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute output = x @ W^T where W = codebook1[hi] + codebook2[lo],
    without materializing W in global memory.

    Phase 1 (Triton kernel): Unpack nibbles, dual gather, tl.dot
    Phase 2 (torch scatter):  Sidecar correction (sparse, <1K elements)
    Phase 3 (torch add):      Bias (optional)

    Args:
        x: [N, IN] input activations (float32/float16/bfloat16)
        codebook1: [16] float32 coarse stage codebook
        codebook2: [16] float32 residual stage codebook
        packed_indices: [OUT, IN] uint8 packed nibble indices
        sidecar_rows: [nnz] int64 precomputed output row indices
        sidecar_cols: [nnz] int64 precomputed input col indices
        sidecar_deltas: [nnz] float32 precomputed (exact - RVQ) deltas
        bias: [OUT] float32 bias
        _dispatch_log: optional dict for instrumentation
        sidecar_phase: "fused" | "scatter" | None (auto)

    Returns:
        [N, OUT] float32 output
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available. Use naive_rvq_matmul() fallback.")

    N, IN = x.shape
    OUT = packed_indices.shape[0]

    assert x.is_cuda, "fused_rvq_matmul requires CUDA tensors"
    assert packed_indices.dtype == torch.uint8, (
        f"Expected uint8 packed indices, got {packed_indices.dtype}"
    )
    assert codebook1.shape[0] == 16, f"Coarse codebook must have 16 entries, got {codebook1.shape[0]}"
    assert codebook2.shape[0] == 16, f"Residual codebook must have 16 entries, got {codebook2.shape[0]}"

    output = torch.empty(N, OUT, device=x.device, dtype=torch.float32)

    # Ensure x is FP32 (kernel casts to FP16 internally)
    x_f32 = x.float() if x.dtype != torch.float32 else x

    # Shape-keyed config dispatch (same perf profile as VQ kernel)
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K, num_warps, num_stages = _get_config(OUT, IN)

    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(OUT, BLOCK_N))

    if _dispatch_log is not None:
        _dispatch_log["dispatch_selected"] = KERNEL_VERSION
        _dispatch_log["block_config"] = f"M{BLOCK_M}_N{BLOCK_N}_K{BLOCK_K}_w{num_warps}_s{num_stages}"

    try:
        _rvq_gather_matmul_tiled_kernel[grid](
            x_f32, codebook1, codebook2, packed_indices, output,
            N, IN, OUT,
            x_f32.stride(0), x_f32.stride(1),
            packed_indices.stride(0), packed_indices.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )
    except Exception as e:
        raise RuntimeError(
            f"RVQ Triton kernel launch failed ({KERNEL_VERSION}): {e}. "
            f"N={N}, IN={IN}, OUT={OUT}, config=K{BLOCK_K}_w{num_warps}_s{num_stages}. "
            f"No silent fallback."
        ) from e

    # Phase 2: Sidecar correction (identical to VQ — sparse delta on weight matrix)
    _N_SIDECAR_FUSED_THRESHOLD = 16
    has_precomputed = sidecar_rows is not None

    if sidecar_phase is None:
        use_fused_sidecar = has_precomputed and N <= _N_SIDECAR_FUSED_THRESHOLD
    else:
        use_fused_sidecar = has_precomputed and sidecar_phase == "fused"

    if use_fused_sidecar:
        nnz = sidecar_rows.shape[0]
        BLOCK = min(1024, triton.next_power_of_2(nnz))
        grid_sc = (triton.cdiv(nnz, BLOCK),)
        _sidecar_fused_kernel[grid_sc](
            x_f32, output,
            sidecar_cols, sidecar_rows, sidecar_deltas,
            N, nnz,
            x_f32.stride(0), output.stride(0),
            BLOCK=BLOCK,
        )
    elif has_precomputed:
        x_at_cols = x_f32[:, sidecar_cols]
        corrections = x_at_cols * sidecar_deltas.unsqueeze(0)
        output.scatter_add_(1, sidecar_rows.unsqueeze(0).expand(N, -1), corrections)

    # Phase 3: Bias
    if bias is not None:
        output += bias.unsqueeze(0)

    return output


def naive_rvq_matmul(
    x: torch.Tensor,
    codebook1: torch.Tensor,
    codebook2: torch.Tensor,
    packed_indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CPU-friendly naive RVQ matmul. Materializes W, then matmuls.

    Used for testing and CPU fallback. NOT memory-efficient.

    Args:
        x: [N, IN] input
        codebook1: [16] float32 coarse codebook
        codebook2: [16] float32 residual codebook
        packed_indices: [OUT, IN] uint8 packed nibbles
        bias: [OUT] optional bias

    Returns:
        [N, OUT] float32 output
    """
    idx1, idx2 = unpack_nibbles(packed_indices)
    W = codebook1[idx1.long()] + codebook2[idx2.long()]  # [OUT, IN]
    output = x.float() @ W.t()
    if bias is not None:
        output += bias.unsqueeze(0)
    return output


def dequant_rvq_tile(
    codebook1: torch.Tensor,
    codebook2: torch.Tensor,
    packed_indices: torch.Tensor,
    start_row: int,
    end_row: int,
    sidecar_rows: Optional[torch.Tensor] = None,
    sidecar_cols: Optional[torch.Tensor] = None,
    sidecar_deltas: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize a tile of RVQ weight rows. Used for tiled CPU forward.

    Args:
        codebook1: [16] coarse codebook
        codebook2: [16] residual codebook
        packed_indices: [OUT, IN] packed nibbles
        start_row: first row (inclusive)
        end_row: last row (exclusive)
        sidecar_*: optional sparse corrections

    Returns:
        [end_row - start_row, IN] float32 weight tile
    """
    tile_packed = packed_indices[start_row:end_row]
    idx1, idx2 = unpack_nibbles(tile_packed)
    tile = codebook1[idx1.long()] + codebook2[idx2.long()]

    if sidecar_rows is not None:
        mask = (sidecar_rows >= start_row) & (sidecar_rows < end_row)
        if mask.any():
            tile = tile.clone()
            local_rows = sidecar_rows[mask] - start_row
            local_cols = sidecar_cols[mask]
            tile[local_rows, local_cols] += sidecar_deltas[mask]

    return tile


def is_available() -> bool:
    """Check if RVQ Triton kernel is available."""
    return HAS_TRITON and torch.cuda.is_available()
