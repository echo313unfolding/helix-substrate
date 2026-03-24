"""
Triton fused VQ gather-matmul kernel for HelixLinear.

Computes output = x @ W^T where W = codebook[indices], WITHOUT ever
materializing the full W in global memory.

Standard path:
    W = codebook[indices]         # [out, in] float32 -- FULL W in global memory
    output = x @ W.T              # standard matmul

Fused path (this kernel):
    For each output tile:
        Load indices tile -> gather codebook in registers -> tiled matmul
    Sidecar + SVD applied as small post-corrections.

Three kernel variants:
    v1 (original): Scalar k-loop -- one input column per iteration.
    v2 (blocked):  K_BLOCK=16 columns per iteration with unrolled inner loop.
                   Optional FP16 multiply with FP32 accumulate.
    v3 (tiled):    True tiled GEMM with tl.dot. Gathers codebook into [BLOCK_N, BLOCK_K]
                   tiles, casts to FP16, then uses tl.dot for tiled matmul.
                   3-5x faster than v2 on CC 7.5. Default kernel.

Memory: only codebook (1KB) + indices (uint8) touched. W never exists.
Peak VRAM delta: zero (vs out*in*4 bytes for naive path).

Work Order: WO-HELIX-LINEAR-01
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

# --- Kernel Version Constants ---
KERNEL_VERSION = "v4_manifest_fp16dot"
KERNEL_IMPL = "triton_vq_matmul"

# --- Static shape-keyed config manifest ---
# Derived from decode_kernel_sweep + decode_warps_sweep receipts.
# Key: (OUT, IN) → (BLOCK_K, num_warps, num_stages)
# Avoids @triton.autotune Python dispatch overhead (154 lookups/token).
# Default: K64/w4/s1 (best general config from sweeps).
_DEFAULT_CONFIG = (64, 4, 1)  # BLOCK_K, num_warps, num_stages
_SHAPE_CONFIGS = {
    # ffn_gate/up: OUT=5632, IN=2048 — w8 wins by ~5%
    (5632, 2048): (64, 8, 1),
    # ffn_down: OUT=2048, IN=5632 — w4 is best
    (2048, 5632): (64, 4, 1),
    # attn_q/o: OUT=2048, IN=2048 — w4 is best
    (2048, 2048): (64, 4, 1),
    # attn_k/v: OUT=256, IN=2048 — small tensor, K64/w4 fine
    (256, 2048): (64, 4, 1),
}


def _get_config(OUT: int, IN: int):
    """Look up static kernel config. O(1), no autotune overhead."""
    return _SHAPE_CONFIGS.get((OUT, IN), _DEFAULT_CONFIG)


def get_kernel_metadata() -> dict:
    """Return version/dispatch metadata for receipt embedding.

    Every benchmark receipt MUST include these fields.
    """
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


if HAS_TRITON:

    @triton.jit
    def _vq_gather_matmul_kernel(
        # Pointers
        x_ptr,          # [N, IN] float32 input activations
        codebook_ptr,   # [256] float32 cluster centers
        indices_ptr,    # [OUT, IN] uint8 cluster indices
        output_ptr,     # [N, OUT] float32 output
        # Dimensions
        N,              # batch size (flattened)
        IN,             # input features
        OUT,            # output features
        # Strides
        stride_xn,      # x stride along N
        stride_xi,      # x stride along IN
        stride_idx_o,   # indices stride along OUT
        stride_idx_i,   # indices stride along IN
        stride_on,      # output stride along N
        stride_oo,      # output stride along OUT
        # Block sizes (constexpr for compilation)
        BLOCK_N: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        """
        Fused codebook[indices] gather + matmul via outer products (v1 -- scalar k-loop).

        For each (BLOCK_N, BLOCK_OUT) output tile, accumulates over IN:
            acc[n, o] += x[n, k] * codebook[indices[o, k]]

        Uses element-wise outer products instead of tl.dot to avoid
        the gather+dot pattern that causes map::at on Triton 3.2/CC7.5.
        The codebook gather happens in registers -- full W never exists.
        """
        pid_n = tl.program_id(0)
        pid_o = tl.program_id(1)

        # Offsets for this tile
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_o = pid_o * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

        mask_n = offs_n < N
        mask_o = offs_o < OUT

        # Accumulator in float32
        acc = tl.zeros((BLOCK_N, BLOCK_OUT), dtype=tl.float32)

        # Process one k at a time: outer-product accumulation
        for k in range(0, IN):
            # Load x column: x[:, k] -> [BLOCK_N]
            x_col = tl.load(
                x_ptr + offs_n * stride_xn + k * stride_xi,
                mask=mask_n, other=0.0
            )

            # Load indices column: indices[:, k] -> [BLOCK_OUT] uint8
            idx_col = tl.load(
                indices_ptr + offs_o * stride_idx_o + k * stride_idx_i,
                mask=mask_o, other=0
            )

            # Gather from codebook: codebook[idx] -> [BLOCK_OUT] float32
            w_col = tl.load(codebook_ptr + idx_col.to(tl.int32))

            # Outer product: [BLOCK_N, 1] * [1, BLOCK_OUT] -> [BLOCK_N, BLOCK_OUT]
            acc += x_col[:, None] * w_col[None, :]

        # Store output: [BLOCK_N, BLOCK_OUT]
        out_ptrs = output_ptr + offs_n[:, None] * stride_on + offs_o[None, :] * stride_oo
        tl.store(out_ptrs, acc, mask=mask_n[:, None] & mask_o[None, :])


    @triton.jit
    def _vq_gather_matmul_blocked_kernel(
        # Pointers
        x_ptr,          # [N, IN] float32 input activations
        codebook_ptr,   # [256] float32 or float16 cluster centers
        indices_ptr,    # [OUT, IN] uint8 cluster indices
        output_ptr,     # [N, OUT] float32 output
        # Dimensions
        N,              # batch size (flattened)
        IN,             # input features
        OUT,            # output features
        # Strides
        stride_xn,      # x stride along N
        stride_xi,      # x stride along IN
        stride_idx_o,   # indices stride along OUT
        stride_idx_i,   # indices stride along IN
        stride_on,      # output stride along N
        stride_oo,      # output stride along OUT
        # Block sizes (constexpr for compilation)
        BLOCK_N: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
        K_BLOCK: tl.constexpr,
        USE_FP16: tl.constexpr,
    ):
        """
        Blocked variant of VQ gather-matmul (v2).

        Processes K_BLOCK input columns per outer-loop iteration with
        unrolled inner loop, reducing loop iterations by K_BLOCK x.
        Optional FP16 compute: multiply in float16, accumulate in float32.
        """
        pid_n = tl.program_id(0)
        pid_o = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_o = pid_o * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

        mask_n = offs_n < N
        mask_o = offs_o < OUT

        acc = tl.zeros((BLOCK_N, BLOCK_OUT), dtype=tl.float32)

        # Blocked k-loop: outer steps by K_BLOCK, inner is unrolled (K_BLOCK is constexpr).
        # Individual column loads in the inner loop — avoids unsupported 2D tile indexing.
        # Compiler unrolls the inner range(K_BLOCK) and can pipeline the loads.
        for k_start in range(0, IN, K_BLOCK):
            for kb in range(K_BLOCK):
                k = k_start + kb

                # Load x column: x[:, k] -> [BLOCK_N]
                x_col = tl.load(
                    x_ptr + offs_n * stride_xn + k * stride_xi,
                    mask=mask_n & (k < IN), other=0.0
                )

                # Load indices column: indices[:, k] -> [BLOCK_OUT] uint8
                idx_col = tl.load(
                    indices_ptr + offs_o * stride_idx_o + k * stride_idx_i,
                    mask=mask_o & (k < IN), other=0
                )

                # Gather from codebook
                w_col = tl.load(codebook_ptr + idx_col.to(tl.int32))

                # Outer product with optional FP16 compute
                if USE_FP16:
                    acc += (x_col.to(tl.float16)[:, None] * w_col.to(tl.float16)[None, :]).to(tl.float32)
                else:
                    acc += x_col[:, None] * w_col[None, :]

        # Store output tile (always FP32)
        out_ptrs = output_ptr + offs_n[:, None] * stride_on + offs_o[None, :] * stride_oo
        tl.store(out_ptrs, acc, mask=mask_n[:, None] & mask_o[None, :])


    @triton.jit
    def _vq_gather_matmul_tiled_kernel(
        # Pointers
        x_ptr,          # [N, IN] float32 input activations
        codebook_ptr,   # [256] float32 cluster centers
        indices_ptr,    # [OUT, IN] uint8 cluster indices
        output_ptr,     # [N, OUT] float32 output
        # Dimensions
        N,              # batch size (flattened)
        IN,             # input features
        OUT,            # output features
        # Strides
        stride_xn,      # x stride along N
        stride_xi,      # x stride along IN
        stride_idx_o,   # indices stride along OUT
        stride_idx_i,   # indices stride along IN
        stride_on,      # output stride along N
        stride_oo,      # output stride along OUT
        # Block sizes (constexpr for compilation)
        BLOCK_M: tl.constexpr,   # batch tile (16)
        BLOCK_N: tl.constexpr,   # output tile (64)
        BLOCK_K: tl.constexpr,   # reduction tile (32)
    ):
        """
        Tiled VQ gather-matmul with tl.dot (v3).

        Gathers codebook[indices] into [BLOCK_N, BLOCK_K] tiles, casts to FP16,
        then uses tl.dot for a proper tiled matrix multiply. FP16 multiply with
        FP32 accumulate.

        3-5x faster than v2 outer-product kernel on CC 7.5 because tl.dot
        enables better instruction scheduling even without tensor cores.

        Key insight: tl.dot fails with FP32 on CC 7.5 (map::at error), but
        works with FP16 inputs. The FP16 precision loss is negligible
        (rel_err < 3e-4, cosine = 1.000000).
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

            # Load x tile: [BLOCK_M, BLOCK_K]
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xi,
                mask=mask_m[:, None] & mask_k[None, :], other=0.0
            )

            # Load indices tile: [BLOCK_N, BLOCK_K] uint8
            idx_tile = tl.load(
                indices_ptr + offs_n[:, None] * stride_idx_o + offs_k[None, :] * stride_idx_i,
                mask=mask_n[:, None] & mask_k[None, :], other=0
            )

            # Dequant: gather codebook → [BLOCK_N, BLOCK_K] float32
            w_tile = tl.load(codebook_ptr + idx_tile.to(tl.int32))

            # Cast to FP16, tl.dot with FP32 accumulate
            x_f16 = x_tile.to(tl.float16)
            w_f16 = w_tile.to(tl.float16)
            acc = tl.dot(x_f16, tl.trans(w_f16), acc=acc)

        # Store output tile (always FP32)
        out_ptrs = output_ptr + offs_m[:, None] * stride_on + offs_n[None, :] * stride_oo
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


    # ─── v4: Autotuned kernel ──────────────────────────────────────────
    # Shape-keyed autotune: Triton explores configs per (N, IN, OUT) and
    # caches winners to ~/.triton/cache (persists across sessions).
    # First call per shape pays ~10s exploration cost; subsequent calls instant.
    #
    # Calibrated from decode_kernel_sweep + decode_warps_sweep receipts:
    #   - BLOCK_K=64 beats K=32 by 11-15% on most shapes
    #   - num_warps=4 is critical (up to 1.58x on ffn_down)
    #   - num_warps=8 occasionally wins on large OUT (ffn_gate/up)
    #   - num_stages has minimal effect on CC 7.5
    #   - BLOCK_M=16 is minimum for tl.dot on CC 7.5
    #   - BLOCK_N=64 is consistently best for decode
    _AUTOTUNE_CONFIGS = [
        # Primary candidates (from sweep winners)
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=1),
        # Wider output tile (for prefill with large N)
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=1),
        # Different warps for different occupancy profiles
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        # K=256 for small-OUT tensors (attn_k/v)
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 256}, num_warps=4, num_stages=1),
    ]

    @triton.autotune(
        configs=_AUTOTUNE_CONFIGS,
        key=['N', 'IN', 'OUT'],
    )
    @triton.jit
    def _vq_autotuned_kernel(
        x_ptr, codebook_ptr, indices_ptr, output_ptr,
        N, IN, OUT,
        stride_xn, stride_xi,
        stride_idx_o, stride_idx_i,
        stride_on, stride_oo,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Autotuned VQ gather-matmul (v4).

        Same algorithm as v3 tiled kernel, but Triton selects optimal
        (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) per shape.
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

            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xi,
                mask=mask_m[:, None] & mask_k[None, :], other=0.0
            )
            idx_tile = tl.load(
                indices_ptr + offs_n[:, None] * stride_idx_o + offs_k[None, :] * stride_idx_i,
                mask=mask_n[:, None] & mask_k[None, :], other=0
            )
            w_tile = tl.load(codebook_ptr + idx_tile.to(tl.int32))

            x_f16 = x_tile.to(tl.float16)
            w_f16 = w_tile.to(tl.float16)
            acc = tl.dot(x_f16, tl.trans(w_f16), acc=acc)

        out_ptrs = output_ptr + offs_m[:, None] * stride_on + offs_n[None, :] * stride_oo
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


    # ─── Fused sidecar kernel ─────────────────────────────────────────
    # Replaces 3 separate CUDA launches (index + mul + scatter_add) with
    # one Triton kernel. Saves 308 launches/token (2 per HelixLinear layer).
    @triton.jit
    def _sidecar_fused_kernel(
        x_ptr, output_ptr,
        cols_ptr, rows_ptr, deltas_ptr,
        N, nnz,
        stride_xn, stride_on,
        BLOCK: tl.constexpr,
    ):
        """Apply sparse sidecar corrections in one fused kernel."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < nnz

        col = tl.load(cols_ptr + offs, mask=mask, other=0)
        row = tl.load(rows_ptr + offs, mask=mask, other=0)
        delta = tl.load(deltas_ptr + offs, mask=mask, other=0.0)

        for n in range(N):
            x_val = tl.load(x_ptr + n * stride_xn + col, mask=mask, other=0.0)
            correction = x_val * delta
            tl.atomic_add(output_ptr + n * stride_on + row, correction, mask=mask)


def fused_vq_matmul(
    x: torch.Tensor,
    codebook: torch.Tensor,
    indices: torch.Tensor,
    sidecar_positions: Optional[torch.Tensor] = None,
    sidecar_values: Optional[torch.Tensor] = None,
    codebook_values_at_sidecar: Optional[torch.Tensor] = None,
    sidecar_rows: Optional[torch.Tensor] = None,
    sidecar_cols: Optional[torch.Tensor] = None,
    sidecar_deltas: Optional[torch.Tensor] = None,
    svd_U: Optional[torch.Tensor] = None,
    svd_s: Optional[torch.Tensor] = None,
    svd_Vt: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    codebook_f16: Optional[torch.Tensor] = None,
    _dispatch_log: Optional[dict] = None,
    sidecar_phase: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute output = x @ W^T where W = codebook[indices] + sidecar + SVD,
    without materializing W in global memory.

    Phase 1 (Triton kernel): output = x @ codebook[indices]^T  (tiled v3 with tl.dot)
    Phase 2 (torch scatter):  output += sidecar correction  (tiny, <1K elements)
    Phase 3 (torch matmul):   output += SVD residual         (rank-8, tiny)
    Phase 4 (torch add):      output += bias                 (optional)

    Args:
        x: [N, IN] input activations (float32 or float16)
        codebook: [256] float32 cluster centers
        indices: [OUT, IN] uint8 cluster assignments
        sidecar_positions: [nnz] int64 flat positions of outliers (legacy API)
        sidecar_values: [nnz] float32 exact outlier values (legacy API)
        codebook_values_at_sidecar: [nnz] float32 VQ values at outliers (legacy API)
        sidecar_rows: [nnz] int64 precomputed output row indices (preferred)
        sidecar_cols: [nnz] int64 precomputed input col indices (preferred)
        sidecar_deltas: [nnz] float32 precomputed (value - VQ) deltas (preferred)
        svd_U: [OUT, rank] float32 left singular vectors
        svd_s: [rank] float32 singular values
        svd_Vt: [rank, IN] float32 right singular vectors (transposed)
        bias: [OUT] float32 bias
        codebook_f16: [256] float16 codebook for FP16 compute path

    Returns:
        [N, OUT] float32 output
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available. Use naive forward path.")

    N, IN = x.shape
    OUT = indices.shape[0]

    # Ensure correct types
    assert x.is_cuda, "fused_vq_matmul requires CUDA tensors"
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), (
        f"Expected float32/float16/bfloat16, got {x.dtype}"
    )
    assert indices.dtype == torch.uint8, f"Expected uint8 indices, got {indices.dtype}"

    output = torch.empty(N, OUT, device=x.device, dtype=torch.float32)

    # Ensure x is FP32 for the kernel (kernel casts to FP16 internally)
    x_f32 = x.float() if x.dtype != torch.float32 else x

    # v4 manifest dispatch: static shape-keyed config from sweep results.
    # Zero Python overhead — just a dict lookup per call (vs @triton.autotune
    # which adds ~0.5ms/call dispatch overhead, catastrophic at 154 calls/token).
    BLOCK_M = 16  # minimum for tl.dot on CC 7.5
    BLOCK_N = 64  # consistently best for decode
    BLOCK_K, num_warps, num_stages = _get_config(OUT, IN)

    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(OUT, BLOCK_N))

    if _dispatch_log is not None:
        _dispatch_log["dispatch_selected"] = KERNEL_VERSION
        _dispatch_log["block_config"] = f"M{BLOCK_M}_N{BLOCK_N}_K{BLOCK_K}_w{num_warps}_s{num_stages}"

    try:
        _vq_gather_matmul_tiled_kernel[grid](
            x_f32, codebook, indices, output,
            N, IN, OUT,
            x_f32.stride(0), x_f32.stride(1),
            indices.stride(0), indices.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )
    except Exception as e:
        raise RuntimeError(
            f"Triton kernel launch failed ({KERNEL_VERSION}): {e}. "
            f"N={N}, IN={IN}, OUT={OUT}, config=K{BLOCK_K}_w{num_warps}_s{num_stages}. "
            f"No silent fallback."
        ) from e

    # Phase 2: Sidecar correction (sparse, typically <1K elements)
    # Phase-aware routing: fused Triton sidecar wins at small N (decode),
    # but atomic_add contention on CC 7.5 makes scatter_add faster at large N
    # (prefill). Crossover measured at N=16 (sidecar_phase_bench receipt).
    #
    # sidecar_phase: "fused" | "scatter" | None (auto-detect from N).
    # When frozen at request start, avoids per-layer decision overhead.
    _N_SIDECAR_FUSED_THRESHOLD = 16
    has_precomputed = sidecar_rows is not None
    has_legacy = sidecar_positions is not None and sidecar_values is not None

    if sidecar_phase is None:
        # Auto-detect (legacy path — still works, just not pre-frozen)
        use_fused_sidecar = has_precomputed and N <= _N_SIDECAR_FUSED_THRESHOLD
    else:
        use_fused_sidecar = has_precomputed and sidecar_phase == "fused"

    if use_fused_sidecar:
        # Fused sidecar: one Triton kernel, saves 2 CUDA launches per layer.
        # Optimal for decode (N=1) where launch overhead dominates.
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
        # PyTorch scatter_add: wins at large N (prefill) where atomic_add
        # contention in the fused kernel causes 3-9x slowdown at N=512.
        x_at_cols = x_f32[:, sidecar_cols]  # [N, nnz]
        corrections = x_at_cols * sidecar_deltas.unsqueeze(0)  # [N, nnz]
        output.scatter_add_(1, sidecar_rows.unsqueeze(0).expand(N, -1), corrections)
    elif has_legacy:
        # Legacy path: compute rows/cols/deltas from flat positions
        rows = sidecar_positions // IN
        cols = sidecar_positions % IN

        if codebook_values_at_sidecar is not None:
            deltas = sidecar_values - codebook_values_at_sidecar
        else:
            idx_flat = indices.reshape(-1)
            vq_vals = codebook[idx_flat[sidecar_positions].long()]
            deltas = sidecar_values - vq_vals

        x_at_cols = x_f32[:, cols]  # [N, nnz]
        corrections = x_at_cols * deltas.unsqueeze(0)  # [N, nnz]
        output.scatter_add_(1, rows.unsqueeze(0).expand(N, -1), corrections)

    # Phase 3: SVD residual correction (rank-8 typically)
    if svd_U is not None:
        # output += x @ Vt^T @ diag(s) @ U^T
        down = x_f32 @ svd_Vt.t()  # [N, rank]
        scaled = down * svd_s.unsqueeze(0)  # [N, rank]
        svd_correction = scaled @ svd_U.t()  # [N, OUT]
        output += svd_correction

    # Phase 4: Bias
    if bias is not None:
        output += bias.unsqueeze(0)

    return output


def is_available() -> bool:
    """Check if Triton fused kernel is available."""
    return HAS_TRITON and torch.cuda.is_available()
