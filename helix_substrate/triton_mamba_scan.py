"""
Triton fused Mamba SSM scan kernel.

Two kernels:
  1. triton_mamba_scan_chunk — fused prefill scan for a chunk of tokens
  2. triton_mamba_decode_step — fused single-token decode (state update + output)

Both eliminate Python loop overhead and fuse memory accesses. Same math as
mamba_scan_chunked.py but executed as GPU kernels instead of PyTorch ops.

Mathematical basis (linear recurrence):
    h[t] = a[t] * h[t-1] + b[t]   (state update per timestep)
    y[t] = (h[t] * C[t]).sum() + D * x[t]   (output readout)

Each Triton program handles one (batch, head, head_dim_element) triple,
processing all state_size dimensions in registers.

Shapes (Zamba2-2.7B example):
    heads=16, head_dim=64, state_size=64, chunk_size=32
    Grid: batch * heads * head_dim = 1024 programs

Work Order: WO-TRITON-MAMBA-SCAN
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


KERNEL_VERSION = "v1_sequential_chunk"


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1: Fused chunk scan (prefill)
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _mamba_scan_chunk_kernel(
        # Input pointers
        h_ptr,          # [batch, heads, head_dim, state_size] — initial state
        dt_ptr,         # [batch, chunk_size, heads] — timestep values (after softplus)
        A_ptr,          # [heads] — decay rates (negative)
        x_ptr,          # [batch, chunk_size, heads, head_dim] — hidden states
        B_ptr,          # [batch, chunk_size, heads, state_size] — input gate
        C_ptr,          # [batch, chunk_size, heads, state_size] — output gate
        D_ptr,          # [heads] — skip connection (broadcast over head_dim)
        # Output pointers
        y_ptr,          # [batch, chunk_size, heads, head_dim] — output
        h_out_ptr,      # [batch, heads, head_dim, state_size] — final state
        # Dimensions
        batch_size,
        chunk_size,
        heads,
        head_dim,
        state_size,
        # Strides for h: [batch, heads, head_dim, state_size]
        stride_h_b, stride_h_h, stride_h_d, stride_h_s,
        # Strides for dt: [batch, chunk_size, heads]
        stride_dt_b, stride_dt_t, stride_dt_h,
        # Strides for x: [batch, chunk_size, heads, head_dim]
        stride_x_b, stride_x_t, stride_x_h, stride_x_d,
        # Strides for B: [batch, chunk_size, heads, state_size]
        stride_B_b, stride_B_t, stride_B_h, stride_B_s,
        # Strides for C: [batch, chunk_size, heads, state_size]
        stride_C_b, stride_C_t, stride_C_h, stride_C_s,
        # Strides for y: [batch, chunk_size, heads, head_dim]
        stride_y_b, stride_y_t, stride_y_h, stride_y_d,
        # Constexpr
        STATE_SIZE: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
    ):
        """Fused Mamba scan for one chunk.

        One program per (batch, head, head_dim_element).
        Processes all timesteps in the chunk sequentially.
        State is held in registers — no global memory round-trips per step.
        """
        # Decompose program ID into (batch, head, head_dim)
        pid = tl.program_id(0)
        bid = pid // (heads * head_dim)
        remain = pid % (heads * head_dim)
        hid = remain // head_dim
        did = remain % head_dim

        # Load decay rate for this head
        A_val = tl.load(A_ptr + hid)

        # Load skip connection for this head
        D_val = tl.load(D_ptr + hid)

        # Load initial state: h[batch, head, hd, :state_size]
        h_base = h_ptr + bid * stride_h_b + hid * stride_h_h + did * stride_h_d
        state_offsets = tl.arange(0, STATE_SIZE)
        state = tl.load(h_base + state_offsets * stride_h_s,
                        mask=state_offsets < state_size, other=0.0)

        # Sequential scan over chunk (no break — Triton 2.x doesn't support it)
        for t in range(CHUNK_SIZE):
            # Mask for valid timesteps (CHUNK_SIZE may be rounded up)
            valid = t < chunk_size

            # Load dt[batch, t, head] — scalar (0 if invalid — exp(A*0)=1, no state change)
            dt_val = tl.load(dt_ptr + bid * stride_dt_b + t * stride_dt_t + hid * stride_dt_h,
                            mask=valid, other=0.0)

            # Compute dA = exp(A * dt) — decay for this timestep
            dA = tl.exp(A_val * dt_val)

            # Load x[batch, t, head, hd] — scalar
            x_val = tl.load(x_ptr + bid * stride_x_b + t * stride_x_t +
                           hid * stride_x_h + did * stride_x_d,
                           mask=valid, other=0.0)

            # Scaled input
            dx = dt_val * x_val

            # Load B[batch, t, head, :state_size]
            B_base = B_ptr + bid * stride_B_b + t * stride_B_t + hid * stride_B_h
            B_t = tl.load(B_base + state_offsets * stride_B_s,
                         mask=(state_offsets < state_size) & valid, other=0.0)

            # State update: h = dA * h + dx * B
            state = dA * state + dx * B_t

            # Load C[batch, t, head, :state_size]
            C_base = C_ptr + bid * stride_C_b + t * stride_C_t + hid * stride_C_h
            C_t = tl.load(C_base + state_offsets * stride_C_s,
                         mask=(state_offsets < state_size) & valid, other=0.0)

            # Output: y = sum(state * C) + D * x
            y_val = tl.sum(state * C_t) + D_val * x_val

            # Store y[batch, t, head, hd] — only for valid timesteps
            tl.store(y_ptr + bid * stride_y_b + t * stride_y_t +
                    hid * stride_y_h + did * stride_y_d, y_val, mask=valid)

        # Store final state: h_out[batch, head, hd, :state_size]
        h_out_base = h_out_ptr + bid * stride_h_b + hid * stride_h_h + did * stride_h_d
        tl.store(h_out_base + state_offsets * stride_h_s, state,
                mask=state_offsets < state_size)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2: Fused decode step (single token)
# ═══════════════════════════════════════════════════════════════════════════════

    @triton.jit
    def _mamba_decode_step_kernel(
        # Input/output pointers
        h_ptr,          # [batch, heads, head_dim, state_size] — state (in-place update)
        dt_ptr,         # [batch, heads] — timestep for this token
        A_ptr,          # [heads] — decay rates
        x_ptr,          # [batch, heads, head_dim] — input hidden state
        B_ptr,          # [batch, heads, state_size] — input gate
        C_ptr,          # [batch, heads, state_size] — output gate
        D_ptr,          # [heads] — skip connection
        y_ptr,          # [batch, heads, head_dim] — output
        # Dimensions
        batch_size, heads, head_dim, state_size,
        # Strides for h: [batch, heads, head_dim, state_size]
        stride_h_b, stride_h_h, stride_h_d, stride_h_s,
        # Strides for dt: [batch, heads]
        stride_dt_b, stride_dt_h,
        # Strides for x: [batch, heads, head_dim]
        stride_x_b, stride_x_h, stride_x_d,
        # Strides for B: [batch, heads, state_size]
        stride_B_b, stride_B_h, stride_B_s,
        # Strides for C: [batch, heads, state_size]
        stride_C_b, stride_C_h, stride_C_s,
        # Strides for y: [batch, heads, head_dim]
        stride_y_b, stride_y_h, stride_y_d,
        # Constexpr
        STATE_SIZE: tl.constexpr,
    ):
        """Fused Mamba decode step — one state update + output readout.

        One program per (batch, head, head_dim_element).
        Replaces 5+ separate PyTorch kernel launches with 1.
        """
        pid = tl.program_id(0)
        bid = pid // (heads * head_dim)
        remain = pid % (heads * head_dim)
        hid = remain // head_dim
        did = remain % head_dim

        # Load constants
        A_val = tl.load(A_ptr + hid)
        D_val = tl.load(D_ptr + hid)

        # Load dt for this (batch, head)
        dt_val = tl.load(dt_ptr + bid * stride_dt_b + hid * stride_dt_h)

        # Compute decay
        dA = tl.exp(A_val * dt_val)

        # Load input x[batch, head, hd]
        x_val = tl.load(x_ptr + bid * stride_x_b + hid * stride_x_h + did * stride_x_d)
        dx = dt_val * x_val

        # Load state[state_size]
        state_offsets = tl.arange(0, STATE_SIZE)
        h_base = h_ptr + bid * stride_h_b + hid * stride_h_h + did * stride_h_d
        state = tl.load(h_base + state_offsets * stride_h_s,
                       mask=state_offsets < state_size, other=0.0)

        # Load B[batch, head, :state_size]
        B_base = B_ptr + bid * stride_B_b + hid * stride_B_h
        B_vals = tl.load(B_base + state_offsets * stride_B_s,
                        mask=state_offsets < state_size, other=0.0)

        # State update
        state = dA * state + dx * B_vals

        # Store updated state
        tl.store(h_base + state_offsets * stride_h_s, state,
                mask=state_offsets < state_size)

        # Load C[batch, head, :state_size]
        C_base = C_ptr + bid * stride_C_b + hid * stride_C_h
        C_vals = tl.load(C_base + state_offsets * stride_C_s,
                        mask=state_offsets < state_size, other=0.0)

        # Output: y = sum(state * C) + D * x
        y_val = tl.sum(state * C_vals) + D_val * x_val

        # Store output
        tl.store(y_ptr + bid * stride_y_b + hid * stride_y_h + did * stride_y_d, y_val)


# ═══════════════════════════════════════════════════════════════════════════════
# Python wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def triton_mamba_scan_chunk(
    h: torch.Tensor,           # [batch, heads, head_dim, state_size]
    dt: torch.Tensor,          # [batch, chunk_size, heads]
    A: torch.Tensor,           # [heads]
    hidden_states: torch.Tensor,  # [batch, chunk_size, heads, head_dim]
    B: torch.Tensor,           # [batch, chunk_size, heads, state_size]
    C: torch.Tensor,           # [batch, chunk_size, heads, state_size]
    D: torch.Tensor,           # [heads] or [heads, 1]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated chunk scan. Drop-in replacement for chunked_scan().

    Returns:
        y: [batch, chunk_size, heads, head_dim] — output
        h_out: [batch, heads, head_dim, state_size] — final state
    """
    assert HAS_TRITON, "Triton not available"

    batch, chunk_size, heads = dt.shape
    head_dim = hidden_states.shape[-1]
    state_size = B.shape[-1]

    # Ensure float32 for numerical stability
    h = h.float().contiguous()
    dt = dt.float().contiguous()
    A = A.float().contiguous()
    hidden_states = hidden_states.float().contiguous()
    B = B.float().contiguous()
    C = C.float().contiguous()
    D_flat = D.float().flatten()[:heads].contiguous()

    # Allocate outputs
    y = torch.zeros(batch, chunk_size, heads, head_dim, device=h.device, dtype=torch.float32)
    h_out = torch.empty_like(h)

    # Round state_size up to power of 2 for Triton constexpr
    STATE_SIZE_CONST = triton.next_power_of_2(state_size)
    CHUNK_SIZE_CONST = triton.next_power_of_2(chunk_size)

    # Grid: one program per (batch, head, head_dim_element)
    grid = (batch * heads * head_dim,)

    _mamba_scan_chunk_kernel[grid](
        h, dt, A, hidden_states, B, C, D_flat,
        y, h_out,
        batch, chunk_size, heads, head_dim, state_size,
        # h strides
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        # dt strides
        dt.stride(0), dt.stride(1), dt.stride(2),
        # x strides
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states.stride(2), hidden_states.stride(3),
        # B strides
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        # C strides
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        # y strides
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        # constexpr
        STATE_SIZE=STATE_SIZE_CONST,
        CHUNK_SIZE=CHUNK_SIZE_CONST,
    )

    return y, h_out


def triton_mamba_decode_step(
    h: torch.Tensor,           # [batch, heads, head_dim, state_size] — MODIFIED IN PLACE
    dt: torch.Tensor,          # [batch, heads]
    A: torch.Tensor,           # [heads]
    x: torch.Tensor,           # [batch, heads, head_dim]
    B: torch.Tensor,           # [batch, heads, state_size]
    C: torch.Tensor,           # [batch, heads, state_size]
    D: torch.Tensor,           # [heads] or [heads, 1]
) -> torch.Tensor:
    """Triton-accelerated single decode step. Updates h in-place.

    Returns:
        y: [batch, heads, head_dim] — output for this token
    """
    assert HAS_TRITON, "Triton not available"

    batch, heads, head_dim, state_size = h.shape

    # Ensure float32 contiguous
    h_c = h.float().contiguous()
    dt = dt.float().contiguous()
    A = A.float().contiguous()
    x = x.float().contiguous()
    B = B.float().contiguous()
    C = C.float().contiguous()
    D_flat = D.float().flatten()[:heads].contiguous()

    y = torch.empty(batch, heads, head_dim, device=h.device, dtype=torch.float32)

    STATE_SIZE_CONST = triton.next_power_of_2(state_size)

    grid = (batch * heads * head_dim,)

    _mamba_decode_step_kernel[grid](
        h_c, dt, A, x, B, C, D_flat, y,
        batch, heads, head_dim, state_size,
        # h strides
        h_c.stride(0), h_c.stride(1), h_c.stride(2), h_c.stride(3),
        # dt strides
        dt.stride(0), dt.stride(1),
        # x strides
        x.stride(0), x.stride(1), x.stride(2),
        # B strides
        B.stride(0), B.stride(1), B.stride(2),
        # C strides
        C.stride(0), C.stride(1), C.stride(2),
        # y strides
        y.stride(0), y.stride(1), y.stride(2),
        STATE_SIZE=STATE_SIZE_CONST,
    )

    # Copy updated state back
    h.copy_(h_c)

    return y


# ═══════════════════════════════════════════════════════════════════════════════
# Full scan using Triton chunks (replaces chunked_scan from mamba_scan_chunked.py)
# ═══════════════════════════════════════════════════════════════════════════════

def triton_scan(
    h: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    hidden_states: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full sequence scan using Triton kernels.

    Drop-in replacement for chunked_scan() with identical API.
    Launches one Triton kernel per chunk (vs ~8 PyTorch ops per chunk).
    """
    batch, seq_len, heads = dt.shape
    head_dim = hidden_states.shape[-1]
    device = h.device

    y = torch.zeros(batch, seq_len, heads, head_dim, device=device, dtype=torch.float32)

    for cs in range(0, seq_len, chunk_size):
        ce = min(cs + chunk_size, seq_len)
        c_len = ce - cs

        # Chunk slices
        dt_c = dt[:, cs:ce].contiguous()
        x_c = hidden_states[:, cs:ce].contiguous()
        B_c = B[:, cs:ce].contiguous()
        C_c = C[:, cs:ce].contiguous()

        if c_len == 1:
            # Single token — use decode step kernel
            y_t = triton_mamba_decode_step(
                h, dt_c.squeeze(1), A,
                x_c.squeeze(1), B_c.squeeze(1), C_c.squeeze(1), D
            )
            y[:, cs] = y_t
        else:
            # Multi-token chunk — use scan kernel
            y_c, h = triton_mamba_scan_chunk(h, dt_c, A, x_c, B_c, C_c, D)
            y[:, cs:ce] = y_c

    return y, h


# ═══════════════════════════════════════════════════════════════════════════════
# Correctness verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_against_sequential(
    heads: int = 16,
    head_dim: int = 64,
    state_size: int = 64,
    seq_len: int = 128,
    chunk_size: int = 32,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Verify Triton scan matches sequential scan exactly.

    Uses controlled random inputs (NOT real model weights — use real prompts
    for final correctness test).
    """
    from helix_substrate.mamba_scan_chunked import sequential_scan, chunked_scan

    torch.manual_seed(seed)
    batch = 1

    # Generate inputs
    h = torch.randn(batch, heads, head_dim, state_size, device=device)
    dt = torch.rand(batch, seq_len, heads, device=device) * 0.1 + 0.01
    A = -torch.rand(heads, device=device) * 5
    hidden_states = torch.randn(batch, seq_len, heads, head_dim, device=device)
    B = torch.randn(batch, seq_len, heads, state_size, device=device)
    C = torch.randn(batch, seq_len, heads, state_size, device=device)
    D = torch.randn(heads, 1, device=device)

    # Reference: sequential scan
    h_seq = h.clone()
    y_seq, h_final_seq = sequential_scan(h_seq, dt, A, hidden_states, B, C, D)

    # PyTorch chunked scan
    h_chunked = h.clone()
    y_chunked, h_final_chunked = chunked_scan(h_chunked, dt, A, hidden_states, B, C, D, chunk_size)

    # Triton scan
    h_triton = h.clone()
    y_triton, h_final_triton = triton_scan(h_triton, dt, A, hidden_states, B, C, D, chunk_size)

    # Compare
    y_err_chunked = (y_seq - y_chunked).abs().max().item()
    y_err_triton = (y_seq - y_triton).abs().max().item()
    h_err_chunked = (h_final_seq - h_final_chunked).abs().max().item()
    h_err_triton = (h_final_seq - h_final_triton).abs().max().item()

    result = {
        "status": "PASS" if y_err_triton < 1e-3 and h_err_triton < 1e-3 else "FAIL",
        "y_max_abs_error_chunked_vs_seq": y_err_chunked,
        "y_max_abs_error_triton_vs_seq": y_err_triton,
        "h_max_abs_error_chunked_vs_seq": h_err_chunked,
        "h_max_abs_error_triton_vs_seq": h_err_triton,
        "shapes": {
            "batch": batch, "heads": heads, "head_dim": head_dim,
            "state_size": state_size, "seq_len": seq_len, "chunk_size": chunk_size,
        },
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata
# ═══════════════════════════════════════════════════════════════════════════════

def get_kernel_metadata() -> dict:
    """Return kernel metadata for receipts."""
    meta = {
        "kernel_impl": "triton_mamba_scan",
        "kernel_version": KERNEL_VERSION,
        "has_triton": HAS_TRITON,
        "torch_version": torch.__version__,
    }
    if HAS_TRITON:
        meta["triton_version"] = triton.__version__
    if torch.cuda.is_available():
        meta["cuda_version"] = torch.version.cuda
        props = torch.cuda.get_device_properties(0)
        meta["compute_capability"] = f"{props.major}.{props.minor}"
    return meta
