#!/usr/bin/env python3
"""
rope.py - Rotary Position Embeddings for CDNA Streaming

WO-CDNA-ROPE-01: Implement RoPE for Mistral attention.

Mistral uses RoPE to encode positional information into Q and K.
Without RoPE, attention has no concept of token positions.

Reference: https://arxiv.org/abs/2104.09864 (RoFormer)

For Mistral-7B:
  - rope_theta = 10000.0
  - d_head = 128
  - RoPE is applied per-head to Q and K
"""

import numpy as np
from functools import lru_cache


@lru_cache(maxsize=32)
def _precompute_freqs_cis(
    d_head: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> np.ndarray:
    """
    Precompute complex exponentials for RoPE.

    Returns freqs_cis: [max_seq_len, d_head//2] complex64
    """
    # Build frequency array
    # freqs[i] = 1.0 / (theta ** (2i / d_head))
    dim = d_head // 2
    freqs = 1.0 / (theta ** (np.arange(0, dim, dtype=np.float32) / dim))

    # Build position array
    positions = np.arange(max_seq_len, dtype=np.float32)

    # Outer product: [seq_len, dim]
    angles = np.outer(positions, freqs)

    # Convert to complex: e^(i * angle) = cos(angle) + i * sin(angle)
    freqs_cis = np.exp(1j * angles).astype(np.complex64)

    return freqs_cis


def apply_rope(
    x: np.ndarray,
    start_pos: int = 0,
    theta: float = 10000.0,
) -> np.ndarray:
    """
    Apply Rotary Position Embeddings to Q or K tensor.

    Uses INTERLEAVED style (GPT-J/LLaMA/Mistral):
    - Pairs: (x[0], x[1]), (x[2], x[3]), ...

    Args:
        x: Input tensor [..., seq, d_head] (last dim is head dimension)
        start_pos: Starting position index (for KV cache, usually 0)
        theta: RoPE base frequency (10000.0 for Mistral)

    Returns:
        x_rope: Same shape as input with RoPE applied

    The RoPE formula (for each pair of dimensions):
        x_rope[..., 2i]   = x[..., 2i] * cos(m*θ_i)   - x[..., 2i+1] * sin(m*θ_i)
        x_rope[..., 2i+1] = x[..., 2i] * sin(m*θ_i)   + x[..., 2i+1] * cos(m*θ_i)

    Where:
        m = position index
        θ_i = 1 / (theta ** (2i / d_head))
    """
    *batch_dims, seq_len, d_head = x.shape

    # Get precomputed frequencies
    freqs_cis = _precompute_freqs_cis(d_head, start_pos + seq_len, theta)

    # Extract relevant positions
    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]  # [seq, d_head//2]

    # Reshape x to complex view: [..., seq, d_head//2, 2] -> [..., seq, d_head//2] complex
    x_reshaped = x.reshape(*batch_dims, seq_len, d_head // 2, 2)
    x_complex = x_reshaped[..., 0] + 1j * x_reshaped[..., 1]  # [..., seq, d_head//2]

    # Broadcast freqs_cis to match x_complex shape
    # freqs_cis: [seq, d_head//2]
    # x_complex: [..., seq, d_head//2]
    # Need to broadcast across batch dimensions
    for _ in batch_dims:
        freqs_cis = np.expand_dims(freqs_cis, 0)  # [1, ..., seq, d_head//2]

    # Apply rotation: multiply by complex exponential
    x_rotated = x_complex * freqs_cis

    # Convert back to real: [..., seq, d_head//2] -> [..., seq, d_head//2, 2] -> [..., seq, d_head]
    x_out = np.stack([x_rotated.real, x_rotated.imag], axis=-1)
    x_out = x_out.reshape(*batch_dims, seq_len, d_head).astype(x.dtype)

    return x_out


def apply_rope_to_qk(
    Q: np.ndarray,
    K: np.ndarray,
    d_head: int,
    start_pos: int = 0,
    theta: float = 10000.0,
) -> tuple:
    """
    Apply RoPE to Q and K tensors for attention.

    Handles multi-head attention by applying RoPE to each head independently.

    Args:
        Q: [batch, seq, n_heads * d_head] or [batch, n_heads, seq, d_head]
        K: [batch, seq, n_kv_heads * d_head] or [batch, n_kv_heads, seq, d_head]
        d_head: Head dimension (128 for Mistral)
        start_pos: Starting position (for KV cache)
        theta: RoPE base frequency

    Returns:
        (Q_rope, K_rope): Same shapes as inputs with RoPE applied
    """
    # Handle 3D case: [batch, seq, n_heads * d_head]
    if Q.ndim == 3:
        batch, seq, d_q = Q.shape
        n_heads = d_q // d_head

        # Reshape to 4D for per-head processing
        Q_4d = Q.reshape(batch, seq, n_heads, d_head)  # [batch, seq, n_heads, d_head]
        Q_rope_4d = np.zeros_like(Q_4d)

        for h in range(n_heads):
            Q_rope_4d[:, :, h, :] = apply_rope(Q_4d[:, :, h, :], start_pos, theta)

        Q_rope = Q_rope_4d.reshape(batch, seq, d_q)
    else:
        # 4D case: [batch, n_heads, seq, d_head]
        batch, n_heads, seq, d = Q.shape
        Q_rope = np.zeros_like(Q)
        for h in range(n_heads):
            Q_rope[:, h, :, :] = apply_rope(Q[:, h, :, :], start_pos, theta)

    # Same for K
    if K.ndim == 3:
        batch, seq, d_k = K.shape
        n_kv_heads = d_k // d_head

        K_4d = K.reshape(batch, seq, n_kv_heads, d_head)
        K_rope_4d = np.zeros_like(K_4d)

        for h in range(n_kv_heads):
            K_rope_4d[:, :, h, :] = apply_rope(K_4d[:, :, h, :], start_pos, theta)

        K_rope = K_rope_4d.reshape(batch, seq, d_k)
    else:
        batch, n_kv_heads, seq, d = K.shape
        K_rope = np.zeros_like(K)
        for h in range(n_kv_heads):
            K_rope[:, h, :, :] = apply_rope(K[:, h, :, :], start_pos, theta)

    return Q_rope, K_rope


if __name__ == "__main__":
    # Quick verification
    print("=== RoPE Implementation Test ===")

    # Simulate Mistral dimensions
    batch = 1
    seq = 10
    n_heads = 32
    n_kv_heads = 8
    d_head = 128

    # Random Q, K
    np.random.seed(42)
    Q = np.random.randn(batch, seq, n_heads * d_head).astype(np.float32)
    K = np.random.randn(batch, seq, n_kv_heads * d_head).astype(np.float32)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")

    # Apply RoPE
    Q_rope, K_rope = apply_rope_to_qk(Q, K, d_head, start_pos=0, theta=10000.0)

    print(f"Q_rope shape: {Q_rope.shape}")
    print(f"K_rope shape: {K_rope.shape}")

    # Verify RoPE changes the values
    q_diff = np.max(np.abs(Q_rope - Q))
    k_diff = np.max(np.abs(K_rope - K))

    print(f"Max Q change: {q_diff:.4f}")
    print(f"Max K change: {k_diff:.4f}")

    # Verify RoPE preserves magnitude (rotation should preserve L2 norm per head)
    Q_norm = np.linalg.norm(Q.reshape(batch, seq, n_heads, d_head), axis=-1)
    Q_rope_norm = np.linalg.norm(Q_rope.reshape(batch, seq, n_heads, d_head), axis=-1)
    norm_diff = np.max(np.abs(Q_norm - Q_rope_norm))

    print(f"Max norm difference: {norm_diff:.6f} (should be ~0)")

    if norm_diff < 1e-5 and q_diff > 0.01:
        print("\n✓ RoPE implementation looks correct!")
    else:
        print("\n✗ RoPE implementation may have issues")
