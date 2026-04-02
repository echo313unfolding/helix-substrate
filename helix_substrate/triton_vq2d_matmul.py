"""
Triton fused 2D Vector Quantization gather-matmul kernel.

Computes output = x @ W^T where W is stored as 2D VQ:
  W[n, 2j:2j+2] = codebook[indices[n, j]]   (each index → 2 weights)

Codebook: [K, 2] float32 — each entry encodes a pair of adjacent weights.
Indices:  [OUT, IN//2] uint16 — one index per weight pair.

The kernel decomposes the matmul into paired reductions:
  output[m,n] = Σ_j (x[m,2j] * cb[idx[n,j], 0] + x[m,2j+1] * cb[idx[n,j], 1])
             = Σ_j dot(x_pair[m,j], cb[idx[n,j]])

This becomes two tl.dot calls per tile iteration at half the reduction dimension:
  acc += tl.dot(x_even, w_even^T)  +  tl.dot(x_odd, w_odd^T)

Key advantages over scalar VQ:
  - Half the index loads (IN//2 indices vs IN)
  - Codebook captures joint distribution of adjacent weights
  - Quality: 2D k=4096 beats scalar k=256 on 154/154 TinyLlama tensors (0.9994 vs 0.9993)
  - With packed 12-bit indices: 0.75 bytes/weight = 2.67x from BF16

Proven: WO-MULTIDIM-VQ-01b, receipt vq2d_k4096_20260330T173606.json
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

from helix_substrate.triton_vq_matmul import _get_config

KERNEL_VERSION = "vq2d_v4_tiled_fp16dot"
KERNEL_IMPL = "triton_vq2d_matmul"


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
# Triton kernel
# ---------------------------------------------------------------------------

if HAS_TRITON:
    from helix_substrate.triton_vq_matmul import _sidecar_fused_kernel

    @triton.jit
    def _vq2d_gather_matmul_tiled_kernel(
        # Pointers
        x_ptr,              # [N, IN] float32 input activations
        codebook_ptr,       # [K * 2] float32 flattened codebook (K entries × 2 dims)
        indices_ptr,        # [OUT, IN//2] uint16 pair indices
        output_ptr,         # [N, OUT] float32 output
        # Dimensions
        N,                  # batch size
        IN,                 # input features (must be even)
        OUT,                # output features
        IN_PAIRS,           # IN // 2
        # Strides
        stride_xn,          # x stride along N
        stride_xi,          # x stride along IN (typically 1)
        stride_idx_o,       # indices stride along OUT
        stride_idx_i,       # indices stride along IN_PAIRS (typically 1)
        stride_on,          # output stride along N
        stride_oo,          # output stride along OUT
        # Block sizes
        BLOCK_M: tl.constexpr,    # batch tile (16)
        BLOCK_N: tl.constexpr,    # output tile (64)
        BLOCK_KP: tl.constexpr,   # reduction tile in PAIRS (half of BLOCK_K)
    ):
        """
        Tiled 2D VQ gather-matmul with paired tl.dot.

        For each output tile [BLOCK_M, BLOCK_N], reduces over IN_PAIRS:
            1. Load x even/odd columns: x[:, 2j] and x[:, 2j+1]
            2. Load indices: [BLOCK_N, BLOCK_KP] uint16
            3. Gather codebook pairs: cb[idx*2] (even) and cb[idx*2+1] (odd)
            4. Two FP16 tl.dots, FP32 accumulate

        Correctness:
            acc[m,n] = Σ_j x[m,2j]*cb[idx[n,j],0] + x[m,2j+1]*cb[idx[n,j],1]
                     = (x_even @ w_even^T) + (x_odd @ w_odd^T)
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < N
        mask_n = offs_n < OUT

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kp_start in range(0, IN_PAIRS, BLOCK_KP):
            offs_kp = kp_start + tl.arange(0, BLOCK_KP)
            mask_kp = offs_kp < IN_PAIRS

            # Load x even columns: x[:, 2*kp] — [BLOCK_M, BLOCK_KP]
            x_even = tl.load(
                x_ptr + offs_m[:, None] * stride_xn + (offs_kp[None, :] * 2) * stride_xi,
                mask=mask_m[:, None] & mask_kp[None, :], other=0.0
            )
            # Load x odd columns: x[:, 2*kp + 1] — [BLOCK_M, BLOCK_KP]
            x_odd = tl.load(
                x_ptr + offs_m[:, None] * stride_xn + (offs_kp[None, :] * 2 + 1) * stride_xi,
                mask=mask_m[:, None] & mask_kp[None, :], other=0.0
            )

            # Load pair indices: [BLOCK_N, BLOCK_KP] uint16
            idx_tile = tl.load(
                indices_ptr + offs_n[:, None] * stride_idx_o + offs_kp[None, :] * stride_idx_i,
                mask=mask_n[:, None] & mask_kp[None, :], other=0
            )
            idx_i32 = idx_tile.to(tl.int32)

            # Gather codebook: cb[idx*2] = even component, cb[idx*2+1] = odd component
            # Codebook stored flat as [K*2] float32
            w_even = tl.load(codebook_ptr + idx_i32 * 2)        # [BLOCK_N, BLOCK_KP]
            w_odd = tl.load(codebook_ptr + idx_i32 * 2 + 1)     # [BLOCK_N, BLOCK_KP]

            # Two tl.dots at half reduction dim, FP16 inputs with FP32 accumulator.
            # Uses tensor cores for speed. FP16 range (±65504) is sufficient
            # because codebook values and activations are small.
            acc = tl.dot(x_even.to(tl.float16), tl.trans(w_even.to(tl.float16)), acc=acc)
            acc = tl.dot(x_odd.to(tl.float16), tl.trans(w_odd.to(tl.float16)), acc=acc)

        # Store output tile
        out_ptrs = output_ptr + offs_m[:, None] * stride_on + offs_n[None, :] * stride_oo
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def fused_vq2d_matmul(
    x: torch.Tensor,
    codebook: torch.Tensor,
    indices: torch.Tensor,
    sidecar_rows: Optional[torch.Tensor] = None,
    sidecar_cols: Optional[torch.Tensor] = None,
    sidecar_deltas: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    _dispatch_log: Optional[dict] = None,
    sidecar_phase: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute output = x @ W^T where W = codebook[indices] (2D VQ),
    without materializing W.

    Phase 1 (Triton kernel): Paired tl.dot with 2D codebook gather
    Phase 2 (torch scatter):  Sidecar correction (sparse)
    Phase 3 (torch add):      Bias (optional)

    Args:
        x: [N, IN] input activations (IN must be even)
        codebook: [K, 2] float32 2D codebook
        indices: [OUT, IN//2] uint16 pair indices
        sidecar_rows/cols/deltas: optional sparse corrections
        bias: [OUT] optional bias
        _dispatch_log: optional instrumentation dict
        sidecar_phase: "fused" | "scatter" | None

    Returns:
        [N, OUT] float32 output
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available. Use naive_vq2d_matmul() fallback.")

    N, IN = x.shape
    OUT, IN_PAIRS = indices.shape

    assert x.is_cuda, "fused_vq2d_matmul requires CUDA tensors"
    assert IN % 2 == 0, f"IN must be even for 2D VQ, got {IN}"
    assert IN_PAIRS == IN // 2, f"indices shape mismatch: {IN_PAIRS} != {IN}//2"
    assert codebook.shape[1] == 2, f"Codebook must be [K, 2], got {codebook.shape}"
    # Accept uint16 (natural storage dtype for k>256) and cast to int32 for Triton
    if indices.dtype == torch.uint16:
        indices = indices.to(torch.int32)
    assert indices.dtype in (torch.uint8, torch.int16, torch.int32), (
        f"Expected uint8/int16/int32 indices, got {indices.dtype}"
    )

    output = torch.empty(N, OUT, device=x.device, dtype=torch.float32)
    x_f32 = x.float() if x.dtype != torch.float32 else x

    # Flatten codebook for stride-2 gather in kernel
    codebook_flat = codebook.contiguous().view(-1)  # [K*2]

    # Shape-keyed config (use original IN for lookup since perf scales with total flops)
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K, num_warps, num_stages = _get_config(OUT, IN)
    BLOCK_KP = max(16, BLOCK_K // 2)  # min 16 for tl.dot

    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(OUT, BLOCK_N))

    if _dispatch_log is not None:
        _dispatch_log["dispatch_selected"] = KERNEL_VERSION
        _dispatch_log["block_config"] = f"M{BLOCK_M}_N{BLOCK_N}_KP{BLOCK_KP}_w{num_warps}_s{num_stages}"

    try:
        _vq2d_gather_matmul_tiled_kernel[grid](
            x_f32, codebook_flat, indices, output,
            N, IN, OUT, IN_PAIRS,
            x_f32.stride(0), x_f32.stride(1),
            indices.stride(0), indices.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_KP=BLOCK_KP,
            num_warps=num_warps, num_stages=num_stages,
        )
    except Exception as e:
        raise RuntimeError(
            f"2D VQ Triton kernel launch failed ({KERNEL_VERSION}): {e}. "
            f"N={N}, IN={IN}, OUT={OUT}, config=KP{BLOCK_KP}_w{num_warps}_s{num_stages}."
        ) from e

    # Phase 2: Sidecar (same as scalar VQ — sparse delta on weight matrix)
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


def naive_vq2d_matmul(
    x: torch.Tensor,
    codebook: torch.Tensor,
    indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CPU-friendly naive 2D VQ matmul. Materializes W, then matmuls.

    Args:
        x: [N, IN] input
        codebook: [K, 2] float32 codebook
        indices: [OUT, IN//2] uint16/int16 pair indices
        bias: [OUT] optional

    Returns:
        [N, OUT] float32 output
    """
    OUT, IN_PAIRS = indices.shape
    IN = IN_PAIRS * 2

    # Reconstruct W: [OUT, IN]
    # codebook[indices] → [OUT, IN//2, 2] → reshape to [OUT, IN]
    W = codebook[indices.long()].reshape(OUT, IN)  # [OUT, IN//2, 2] → [OUT, IN]
    output = x.float() @ W.t()
    if bias is not None:
        output += bias.unsqueeze(0)
    return output


def dequant_vq2d_tile(
    codebook: torch.Tensor,
    indices: torch.Tensor,
    start_row: int,
    end_row: int,
    sidecar_rows: Optional[torch.Tensor] = None,
    sidecar_cols: Optional[torch.Tensor] = None,
    sidecar_deltas: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize a tile of 2D VQ weight rows. Used for tiled CPU forward.

    Args:
        codebook: [K, 2] codebook
        indices: [OUT, IN//2] pair indices
        start_row, end_row: row range

    Returns:
        [end_row - start_row, IN] float32 weight tile
    """
    tile_idx = indices[start_row:end_row]  # [rows, IN//2]
    rows, in_pairs = tile_idx.shape
    tile = codebook[tile_idx.long()].reshape(rows, in_pairs * 2)  # [rows, IN]

    if sidecar_rows is not None:
        mask = (sidecar_rows >= start_row) & (sidecar_rows < end_row)
        if mask.any():
            tile = tile.clone()
            local_rows = sidecar_rows[mask] - start_row
            local_cols = sidecar_cols[mask]
            tile[local_rows, local_cols] += sidecar_deltas[mask]

    return tile


def is_available() -> bool:
    """Check if 2D VQ Triton kernel is available."""
    return HAS_TRITON and torch.cuda.is_available()
