"""Memory-efficient Mamba2 SSM scan for small GPUs.

Monkey-patches HuggingFace's Zamba2 naive torch_forward to replace the
chunked SSD scan (which materializes GB-sized intermediates) with a
sequential token-by-token scan that uses O(state_size) memory.

Usage:
    import helix_substrate.mamba_scan_patch  # Apply patch before loading model

The naive HF path allocates:
    G_intermediate: [b, chunks, seq, state, heads, state] — GB-scale
    M, Y_diag: similarly huge

The patched path allocates:
    ssm_state: [b, heads, head_dim, state_size] — ~1 MB
    y: [b, seq_len, heads, head_dim] — small

This enables Zamba2-1.2B and 2.7B on 4 GB GPUs like the Quadro T2000.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sequential_scan_forward(self, input_states, cache_params=None, attention_mask=None):
    """Memory-efficient replacement for Zamba2's torch_forward.

    Replaces the chunked SSD scan with sequential token-by-token processing.
    Same math, O(state_size) memory instead of O(seq_len * state_size^2).
    """
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    # --- Projection (same as original) ---
    if cache_params is not None and cache_params.has_previous_state:
        projected_states = self.in_proj(input_states.squeeze(1))
    else:
        if attention_mask is not None and not torch.all(attention_mask == 1):
            input_states = (input_states * attention_mask[:, :, None]).to(dtype)
        projected_states = self.in_proj(input_states)

    d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size
             - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
    _, _, gate, hidden_states, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
        dim=-1,
    )

    # --- Conv1d (manual padding to avoid cuDNN) ---
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx].clone().to(hidden_states.device)
        if cache_params.has_previous_state:
            gate = gate.unsqueeze(1)
            conv_state = cache_params.conv_states[self.layer_idx]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = hidden_states[:, 0, :] if hidden_states.ndim == 3 else hidden_states
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            hidden_states = torch.sum(
                conv_state.to(projected_states.device) * self.conv1d.weight[:, 0, :], dim=-1
            )
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype)[:, None, ...]
        else:
            hidden_states_t = hidden_states.transpose(1, 2)
            conv_state = F.pad(hidden_states_t, (self.conv_kernel_size - hidden_states_t.shape[-1], 0))
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            # Manual conv1d: pad + grouped conv (avoids cuDNN)
            padded = F.pad(hidden_states_t, (self.conv_kernel_size - 1, 0))
            hidden_states = self.act(
                _manual_conv1d(padded, self.conv1d.weight, self.conv1d.bias,
                               self.conv1d.groups).transpose(1, 2)
            )[:, :seq_len, :]
            if attention_mask is not None and not torch.all(attention_mask == 1):
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    else:
        ssm_state = torch.zeros(
            (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
            device=hidden_states.device, dtype=dtype,
        )
        hidden_states_t = hidden_states.transpose(1, 2)
        padded = F.pad(hidden_states_t, (self.conv_kernel_size - 1, 0))
        hidden_states = self.act(
            _manual_conv1d(padded, self.conv1d.weight, self.conv1d.bias,
                           self.conv1d.groups).transpose(1, 2)
        )[..., :seq_len, :]

    # --- Split into SSM components ---
    hidden_states, B, C = torch.split(
        hidden_states,
        [self.intermediate_size, self.n_groups * self.ssm_state_size,
         self.n_groups * self.ssm_state_size],
        dim=-1,
    )

    A = -torch.exp(self.A_log.float())  # [num_heads]

    # --- Decode path (single token, same as original) ---
    if cache_params is not None and cache_params.has_previous_state:
        dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
        dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
        dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)
        dt = torch.clamp(F.softplus(dt + dt_bias.to(dt.dtype)), self.time_step_min)

        A_exp = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(torch.float32)
        dA = torch.exp(dt[..., None] * A_exp)

        B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
        B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
        B = B.reshape(batch_size, -1, B.shape[-1])

        dB = dt[..., None] * B[..., None, :]
        hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
        dBx = dB * hidden_states[..., None]

        cache_params.ssm_states[self.layer_idx].copy_(
            cache_params.ssm_states[self.layer_idx] * dA + dBx
        )

        C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
        C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
        C = C.reshape(batch_size, -1, C.shape[-1])

        ssm_states = cache_params.ssm_states[self.layer_idx].to(C.dtype)
        ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
        C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)
        y = torch.bmm(ssm_states_reshaped, C_reshaped).view(batch_size, self.num_heads, self.head_dim)

        D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
        y = (y + hidden_states * D).to(y.dtype)
        y = y.reshape(batch_size, -1)[:, None, ...]

    else:
        # ============================================================
        # SEQUENTIAL SCAN — replaces the chunked SSD that OOMs
        # ============================================================
        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_min)
        # dt: [batch, seq_len, num_heads]

        hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)

        # D skip connection
        D = self.D[..., None]  # [num_heads, 1]

        # Prepare output buffer
        y = torch.zeros(batch_size, seq_len, self.num_heads, self.head_dim,
                        device=hidden_states.device, dtype=hidden_states.dtype)

        # State: [batch, num_heads, head_dim, ssm_state_size]
        h = ssm_state.float()

        for t in range(seq_len):
            # Discretize A for this timestep
            # A: [num_heads], dt[:, t]: [batch, num_heads]
            dA_t = torch.exp(A * dt[:, t, :])  # [batch, num_heads]

            # Discretize input: dt * x
            x_t = hidden_states[:, t, :, :]  # [batch, num_heads, head_dim]
            dx_t = dt[:, t, :, None] * x_t   # [batch, num_heads, head_dim]

            # B for this timestep
            B_t = B[:, t, :, :]  # [batch, num_heads, ssm_state_size]

            # State update: h = dA * h + dx outer B
            # dA: [batch, num_heads, 1, 1] * h: [batch, num_heads, head_dim, ssm_state_size]
            # dx_t: [batch, num_heads, head_dim, 1] * B_t: [batch, num_heads, 1, ssm_state_size]
            h = dA_t[:, :, None, None] * h + dx_t[:, :, :, None] * B_t[:, :, None, :]

            # Output: y = C * h summed over state dim, plus D * x
            C_t = C[:, t, :, :]  # [batch, num_heads, ssm_state_size]
            # [batch, num_heads, head_dim] = sum over state of h * C
            y_t = (h * C_t[:, :, None, :]).sum(-1)  # [batch, num_heads, head_dim]
            y_t = y_t + D.unsqueeze(0) * x_t        # D skip

            y[:, t, :, :] = y_t

        # Store final state
        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(h.to(ssm_state.dtype))

        y = y.reshape(batch_size, seq_len, -1)

    # --- Norm + output projection (same as original) ---
    scan_output = self.norm(y, gate)
    contextualized_states = self.out_proj(scan_output.to(dtype))
    return contextualized_states


def _manual_conv1d(x, weight, bias, groups):
    """Manual grouped conv1d that avoids cuDNN.

    Uses unfold + matmul instead of F.conv1d to bypass cuDNN initialization issues
    on older drivers / Turing GPUs.
    """
    batch, channels, length = x.shape
    out_channels, in_ch_per_group, kernel_size = weight.shape
    group_out = out_channels // groups
    group_in = channels // groups

    # Unfold input into sliding windows
    x_unf = x.unfold(2, kernel_size, 1)  # [batch, channels, out_len, kernel_size]
    out_len = x_unf.shape[2]

    # Reshape for grouped matmul
    x_unf = x_unf.reshape(batch, groups, group_in, out_len, kernel_size)
    w = weight.reshape(groups, group_out, group_in, kernel_size)

    # Manual grouped conv via einsum
    out = torch.einsum('bgitk,gojk->bgojt', x_unf.float(), w.float())
    out = out.reshape(batch, out_channels, out_len)

    if bias is not None:
        out = out + bias[:, None]
    return out.to(x.dtype)


def apply_patch():
    """Monkey-patch HF Zamba2 to use memory-efficient scan."""
    try:
        from transformers.models.zamba2.modeling_zamba2 import Zamba2MambaMixer
        Zamba2MambaMixer.torch_forward = _sequential_scan_forward
        print("[mamba_scan_patch] Patched Zamba2MambaMixer with memory-efficient sequential scan")
        return True
    except ImportError:
        print("[mamba_scan_patch] Zamba2 not found in transformers, patch skipped")
        return False


# Auto-apply on import
apply_patch()
