"""
Chunked Mamba2 scan: vectorized within chunks, sequential across chunks.

The key insight — the SSM recurrence h_t = a_t * h_{t-1} + b_t is a
LINEAR recurrence. Within a chunk, we can solve it in closed form:

    h_k = cumA_k * (h_init + Σ_{i=0}^{k} b_i / cumA_i)

where cumA_k = ∏_{j=0}^{k} a_j (cumulative product of decay factors).

This replaces the Python for-loop (6 kernel launches per token) with
vectorized tensor operations (~8 operations per chunk), reducing
kernel launch overhead by ~chunk_size factor.

                    Sequential              Chunked (chunk=32)
    seq_len=256:    256 iters × 6 = 1536    8 chunks × 8 = 64 launches
    seq_len=1:      1 iter (decode)         1 iter (same — decode is unchanged)

Memory per chunk (Zamba2 shapes, chunk_size=32):
    b tensor: [1, 32, 16, 64, 64] × 4 bytes ≈ 8 MB
    Total working set: ~32 MB — fits easily on T2000 (4 GB)

Usage:
    import helix_substrate.mamba_scan_chunked  # Replaces sequential scan patch

Work Order: WO-PREFETCH-PIPELINE-01
"""

import torch
import torch.nn.functional as F

# Default chunk size — tuned for T2000 (4 GB VRAM)
# Larger = more parallel work per chunk, but more memory
# 32 uses ~32 MB working set, gives 32x reduction in outer loop iterations
DEFAULT_CHUNK_SIZE = 32


# ─────────────────────────────────────────────────────────────────────
# Core scan functions (can be used independently for benchmarking)
# ─────────────────────────────────────────────────────────────────────

def sequential_scan(h, dt, A, hidden_states, B, C, D):
    """Token-by-token scan (baseline). Python loop, 6 kernel launches per token.

    This is what mamba_scan_patch.py does. Extracted here for comparison.

    Args:
        h: State [batch, heads, head_dim, state_size]
        dt: Timestep [batch, seq_len, heads]
        A: Decay rates [heads] (negative)
        hidden_states: Input [batch, seq_len, heads, head_dim]
        B: Input gate [batch, seq_len, heads, state_size]
        C: Output gate [batch, seq_len, heads, state_size]
        D: Skip connection [heads, 1]

    Returns:
        y: Output [batch, seq_len, heads, head_dim]
        h: Final state [batch, heads, head_dim, state_size]
    """
    batch, seq_len, heads = dt.shape
    head_dim = hidden_states.shape[-1]

    y = torch.zeros(batch, seq_len, heads, head_dim,
                    device=h.device, dtype=h.dtype)

    for t in range(seq_len):
        # ── One timestep ──
        # Step 1: Discretize decay
        dA_t = torch.exp(A * dt[:, t, :])          # [batch, heads]

        # Step 2: Input
        x_t = hidden_states[:, t, :, :]             # [batch, heads, head_dim]
        dx_t = dt[:, t, :, None] * x_t              # [batch, heads, head_dim]

        # Step 3: State update  h = decay * h + input ⊗ gate
        B_t = B[:, t, :, :]                          # [batch, heads, state_size]
        h = (dA_t[:, :, None, None] * h +
             dx_t[:, :, :, None] * B_t[:, :, None, :])

        # Step 4: Output readout
        C_t = C[:, t, :, :]
        y_t = (h * C_t[:, :, None, :]).sum(-1)       # [batch, heads, head_dim]
        y_t = y_t + D[None, :, :] * x_t              # skip connection
        y[:, t, :, :] = y_t

    return y, h


def chunked_scan(h, dt, A, hidden_states, B, C, D, chunk_size=DEFAULT_CHUNK_SIZE):
    """Vectorized chunked scan. No Python loop within chunks.

    Mathematical derivation:
    ─────────────────────────
    The recurrence:  h_t = a_t * h_{t-1} + b_t

    Unroll for a chunk of C tokens:
        h_0 = a_0 * h_init + b_0
        h_1 = a_1 * h_0 + b_1  =  a_1*a_0 * h_init + a_1*b_0 + b_1
        h_k = cumA_k * h_init + Σ_{i=0}^{k} (cumA_k / cumA_i) * b_i

    Factor out cumA_k:
        h_k = cumA_k * (h_init + Σ_{i=0}^{k} b_i / cumA_i)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^
                                  This is a CUMULATIVE SUM.

    So the entire chunk reduces to:
        1. cumA = cumprod(a)           — one cumsum in log space
        2. scaled_b = b / cumA         — element-wise division
        3. prefix = cumsum(scaled_b)   — one cumsum
        4. h_k = cumA_k * (h_init + prefix_k)   — element-wise multiply

    Four vectorized operations replace C iterations of the Python loop.

    Args:
        h: State [batch, heads, head_dim, state_size] (float32)
        dt: Timestep [batch, seq_len, heads]
        A: Decay rates [heads] (negative)
        hidden_states: [batch, seq_len, heads, head_dim]
        B: Input gate [batch, seq_len, heads, state_size]
        C: Output gate [batch, seq_len, heads, state_size]
        D: Skip connection [heads, 1]
        chunk_size: Tokens per vectorized chunk

    Returns:
        y: Output [batch, seq_len, heads, head_dim]
        h: Final state [batch, heads, head_dim, state_size]
    """
    batch, seq_len, heads = dt.shape
    head_dim = hidden_states.shape[-1]
    device = h.device
    dtype = h.dtype

    y = torch.zeros(batch, seq_len, heads, head_dim, device=device, dtype=dtype)

    # Precompute dA and dx for the full sequence (two kernel launches total
    # vs 2 per token in the sequential version)
    dA_all = torch.exp(A[None, None, :] * dt)               # [batch, seq_len, heads]
    dx_all = dt[:, :, :, None] * hidden_states               # [batch, seq_len, heads, head_dim]

    for cs in range(0, seq_len, chunk_size):
        ce = min(cs + chunk_size, seq_len)
        c_len = ce - cs

        if c_len == 1:
            # ── Decode path: single token, direct update ──
            # No benefit from vectorization here — just one state update.
            dA_t = dA_all[:, cs, :, None, None]
            dx_t = dx_all[:, cs]
            B_t = B[:, cs]
            h = dA_t * h + dx_t[:, :, :, None] * B_t[:, :, None, :]
            y_t = (h * C[:, cs, :, None, :]).sum(-1)
            y_t = y_t + D[None, :, :] * hidden_states[:, cs]
            y[:, cs] = y_t
            continue

        # ── Chunk slices ──
        dA_c = dA_all[:, cs:ce]                # [batch, c_len, heads]
        dx_c = dx_all[:, cs:ce]                # [batch, c_len, heads, head_dim]
        B_c = B[:, cs:ce]                      # [batch, c_len, heads, state_size]
        C_c = C[:, cs:ce]                      # [batch, c_len, heads, state_size]
        x_c = hidden_states[:, cs:ce]          # [batch, c_len, heads, head_dim]

        # ── Step 1: Cumulative decay in log-space ──
        log_dA = torch.log(dA_c.clamp(min=1e-10))    # [batch, c_len, heads]
        log_cumA = torch.cumsum(log_dA, dim=1)         # [batch, c_len, heads]

        # ── Safety check: fall back to sequential if decay range overflows ──
        # When cumulative decay spans >80 orders of magnitude, exp(-log_cumA)
        # overflows float32. This happens with strong per-step decay (dA << 1)
        # over many tokens. Fall back to sequential scan for correctness.
        log_range = log_cumA[:, -1, :] - log_cumA[:, 0, :]   # [batch, heads]
        if (log_range.abs() > 80).any():
            # Sequential fallback for this chunk (rare — strong decay)
            for t in range(c_len):
                ti = cs + t
                dA_t = dA_all[:, ti, :, None, None]
                dx_t = dx_all[:, ti]
                B_t = B[:, ti]
                h = dA_t * h + dx_t[:, :, :, None] * B_t[:, :, None, :]
                y_t = (h * C[:, ti, :, None, :]).sum(-1)
                y_t = y_t + D[None, :, :] * hidden_states[:, ti]
                y[:, ti] = y_t
            continue

        cumA = torch.exp(log_cumA)                     # [batch, c_len, heads]

        # ── Step 2: Input contributions ──
        b = dx_c[:, :, :, :, None] * B_c[:, :, :, None, :]
        # shape: [batch, c_len, heads, head_dim, state_size]

        # ── Step 3: Scale by inverse cumulative decay ──
        inv_cumA = torch.exp(-log_cumA)                # [batch, c_len, heads]
        scaled_b = b * inv_cumA[:, :, :, None, None]   # broadcast over hd, ss

        # ── Step 4: Prefix sum (THE KEY TRICK) ──
        cum_b = torch.cumsum(scaled_b, dim=1)
        # shape: [batch, c_len, heads, head_dim, state_size]

        # ── Step 5: Reconstruct all states in the chunk ──
        h_chunk = cumA[:, :, :, None, None] * (
            h[:, None, :, :, :] + cum_b
        )
        # shape: [batch, c_len, heads, head_dim, state_size]

        # ── Step 6: Output readout ──
        y_c = (h_chunk * C_c[:, :, :, None, :]).sum(-1)   # [batch, c_len, heads, head_dim]
        y_c = y_c + D[None, None, :, :] * x_c
        y[:, cs:ce] = y_c

        # ── Step 7: Carry state to next chunk ──
        h = h_chunk[:, -1]   # [batch, heads, head_dim, state_size]

    return y, h


# ─────────────────────────────────────────────────────────────────────
# Full forward function (drop-in for Zamba2MambaMixer.torch_forward)
# ─────────────────────────────────────────────────────────────────────

def _manual_conv1d(x, weight, bias, groups):
    """Manual grouped conv1d (avoids cuDNN issues on Turing GPUs)."""
    batch, channels, length = x.shape
    out_channels, in_ch_per_group, kernel_size = weight.shape
    group_out = out_channels // groups
    group_in = channels // groups

    x_unf = x.unfold(2, kernel_size, 1)
    out_len = x_unf.shape[2]
    x_unf = x_unf.reshape(batch, groups, group_in, out_len, kernel_size)
    w = weight.reshape(groups, group_out, group_in, kernel_size)
    out = torch.einsum('bgitk,gojk->bgojt', x_unf.float(), w.float())
    out = out.reshape(batch, out_channels, out_len)

    if bias is not None:
        out = out + bias[:, None]
    return out.to(x.dtype)


def _chunked_scan_forward(self, input_states, cache_params=None, attention_mask=None):
    """Memory-efficient + chunked-vectorized replacement for Zamba2 torch_forward.

    Same math as mamba_scan_patch.py but replaces the Python for-loop with
    chunked_scan() for prefill. Decode path (single token) is unchanged.
    """
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    # ── Projection (same as original) ──
    if cache_params is not None and cache_params.has_previous_state:
        projected_states = self.in_proj(input_states.squeeze(1))
    else:
        if attention_mask is not None and not torch.all(attention_mask == 1):
            input_states = (input_states * attention_mask[:, :, None]).to(dtype)
        projected_states = self.in_proj(input_states)

    d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size
             - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
    _, _, gate, hidden_states_raw, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
        dim=-1,
    )

    # ── Conv1d (manual padding to avoid cuDNN) ──
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx].clone().to(hidden_states_raw.device)
        if cache_params.has_previous_state:
            gate = gate.unsqueeze(1)
            conv_state = cache_params.conv_states[self.layer_idx]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = (hidden_states_raw[:, 0, :]
                                    if hidden_states_raw.ndim == 3
                                    else hidden_states_raw)
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            hidden_states_raw = torch.sum(
                conv_state.to(projected_states.device) * self.conv1d.weight[:, 0, :],
                dim=-1,
            )
            if self.use_conv_bias:
                hidden_states_raw += self.conv1d.bias
            hidden_states_raw = self.act(hidden_states_raw).to(dtype)[:, None, ...]
        else:
            hidden_states_t = hidden_states_raw.transpose(1, 2)
            conv_state = F.pad(hidden_states_t,
                               (self.conv_kernel_size - hidden_states_t.shape[-1], 0))
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            padded = F.pad(hidden_states_t, (self.conv_kernel_size - 1, 0))
            hidden_states_raw = self.act(
                _manual_conv1d(padded, self.conv1d.weight, self.conv1d.bias,
                               self.conv1d.groups).transpose(1, 2)
            )[:, :seq_len, :]
            if attention_mask is not None and not torch.all(attention_mask == 1):
                hidden_states_raw = (hidden_states_raw * attention_mask[:, :, None]).to(dtype)
    else:
        ssm_state = torch.zeros(
            (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
            device=hidden_states_raw.device, dtype=dtype,
        )
        hidden_states_t = hidden_states_raw.transpose(1, 2)
        padded = F.pad(hidden_states_t, (self.conv_kernel_size - 1, 0))
        hidden_states_raw = self.act(
            _manual_conv1d(padded, self.conv1d.weight, self.conv1d.bias,
                           self.conv1d.groups).transpose(1, 2)
        )[..., :seq_len, :]

    # ── Split into SSM components ──
    hidden_states, B, C = torch.split(
        hidden_states_raw,
        [self.intermediate_size, self.n_groups * self.ssm_state_size,
         self.n_groups * self.ssm_state_size],
        dim=-1,
    )

    A = -torch.exp(self.A_log.float())

    # ── Decode path (single token — uses cache, same as original) ──
    if cache_params is not None and cache_params.has_previous_state:
        dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
        dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
        dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)
        dt = torch.clamp(F.softplus(dt + dt_bias.to(dt.dtype)), self.time_step_min)

        A_exp = (A[..., None, None]
                 .expand(self.num_heads, self.head_dim, self.ssm_state_size)
                 .to(torch.float32))
        dA = torch.exp(dt[..., None] * A_exp)

        B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
        B = B.expand(batch_size, self.n_groups,
                     self.num_heads // self.n_groups, B.shape[-1]).contiguous()
        B = B.reshape(batch_size, -1, B.shape[-1])

        dB = dt[..., None] * B[..., None, :]
        hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
        dBx = dB * hidden_states[..., None]

        cache_params.ssm_states[self.layer_idx].copy_(
            cache_params.ssm_states[self.layer_idx] * dA + dBx
        )

        C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
        C = C.expand(batch_size, self.n_groups,
                     self.num_heads // self.n_groups, C.shape[-1]).contiguous()
        C = C.reshape(batch_size, -1, C.shape[-1])

        ssm_states = cache_params.ssm_states[self.layer_idx].to(C.dtype)
        ssm_states_reshaped = ssm_states.view(
            batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
        C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)
        y = (torch.bmm(ssm_states_reshaped, C_reshaped)
             .view(batch_size, self.num_heads, self.head_dim))

        D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
        y = (y + hidden_states * D).to(y.dtype)
        y = y.reshape(batch_size, -1)[:, None, ...]

    else:
        # ══════════════════════════════════════════════════════════════
        # CHUNKED SCAN — replaces the token-by-token Python for-loop
        # ══════════════════════════════════════════════════════════════
        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_min)

        hidden_states = hidden_states.reshape(
            batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat_interleave(
            self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        C = C.repeat_interleave(
            self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)

        D = self.D[..., None]   # [num_heads, 1]
        h = ssm_state.float()

        # ── THE CHANGE: chunked_scan instead of for-loop ──
        y, h = chunked_scan(h, dt, A, hidden_states, B, C, D,
                            chunk_size=DEFAULT_CHUNK_SIZE)

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(h.to(ssm_state.dtype))

        y = y.reshape(batch_size, seq_len, -1)

    # ── Norm + output projection ──
    scan_output = self.norm(y, gate)
    contextualized_states = self.out_proj(scan_output.to(dtype))
    return contextualized_states


# ─────────────────────────────────────────────────────────────────────
# Monkey-patch
# ─────────────────────────────────────────────────────────────────────

def apply_patch():
    """Monkey-patch HF Zamba2 to use chunked scan."""
    try:
        from transformers.models.zamba2.modeling_zamba2 import Zamba2MambaMixer
        Zamba2MambaMixer.torch_forward = _chunked_scan_forward
        print(f"[mamba_scan_chunked] Patched Zamba2MambaMixer with chunked scan "
              f"(chunk_size={DEFAULT_CHUNK_SIZE})")
        return True
    except ImportError:
        print("[mamba_scan_chunked] Zamba2 not found in transformers, patch skipped")
        return False


# Auto-apply on import
apply_patch()
