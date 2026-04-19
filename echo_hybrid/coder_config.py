"""
EchoHybridCoder: Production-scale hybrid SSM+Transformer for code generation.

Architecture: 28 blocks (24 Mamba + 4 Attention), d=1536
Attention donor: Qwen2.5-Coder-1.5B (GQA + SwiGLU + RoPE)
Mamba blocks: random init, d_inner=3072 (2x expand)

Block pattern: [M,M,M,M,M,A, M,M,M,M,M,A, M,M,M,M,M,A, M,M,M,M,M,A, M,M,M,M]
Attention at positions 5, 11, 17, 23 (evenly spaced, 1:6 ratio)

Work Orders: WO-ECHO-HYBRID-CODER-01
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _default_coder_pattern() -> List[str]:
    """28 blocks: attention at positions 5, 11, 17, 23."""
    pat = ["ssm"] * 28
    for i in [5, 11, 17, 23]:
        pat[i] = "attn"
    return pat


@dataclass
class EchoHybridCoderConfig:
    # Architecture
    block_pattern: List[str] = field(default_factory=_default_coder_pattern)
    hidden_size: int = 1536
    vocab_size: int = 151936  # Qwen2.5 tokenizer

    # SSM hparams (Mamba-style at 1536d)
    ssm_d_inner: int = 3072       # hidden_size * expand
    ssm_expand: int = 2
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_dt_rank: int = 96         # ceil(1536/16)
    ssm_use_conv_bias: bool = True
    ssm_use_bias: bool = False

    # Attention hparams (match Qwen2.5-Coder-1.5B exactly)
    attn_num_heads: int = 12
    attn_num_kv_heads: int = 2         # GQA: 2 KV heads
    attn_head_dim: int = 128           # 1536 / 12
    attn_intermediate_size: int = 8960  # SwiGLU FFN
    attn_rope_theta: float = 1000000.0
    attn_max_position_embeddings: int = 32768

    # Shared
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    @property
    def n_blocks(self) -> int:
        return len(self.block_pattern)

    @property
    def n_ssm(self) -> int:
        return sum(1 for b in self.block_pattern if b == "ssm")

    @property
    def n_attn(self) -> int:
        return sum(1 for b in self.block_pattern if b == "attn")


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def _build_rope_cache(seq_len: int, head_dim: int, theta: float = 1000000.0,
                      device: torch.device = None) -> torch.Tensor:
    """Precompute cos/sin for RoPE."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    dim = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (dim / head_dim))
    angles = torch.outer(pos, freqs)  # [seq, head_dim/2]
    return torch.cat([angles, angles], dim=-1)  # [seq, head_dim]


def _apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x: [batch, heads, seq, head_dim]."""
    seq_len = x.shape[2]
    angles = angles[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    cos = angles.cos()
    sin = angles.sin()
    # Rotate: split into two halves, apply rotation
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos[..., :d2] - x2 * sin[..., :d2],
                      x2 * cos[..., d2:] + x1 * sin[..., d2:]], dim=-1)


# ---------------------------------------------------------------------------
# Naive selective scan (pure PyTorch — same as config.py)
# ---------------------------------------------------------------------------

def selective_scan_naive(u, dt, A, B, C, D):
    batch, seq_len, d_inner = u.shape
    d_state = A.shape[1]
    dtA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    dtB_u = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(seq_len):
        x = dtA[:, i] * x + dtB_u[:, i]
        y = (x * C[:, i].unsqueeze(1)).sum(-1)
        ys.append(y)
    y = torch.stack(ys, dim=1)
    return y + u * D.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Mamba Block (same structure as config.py, scaled to 1536d)
# ---------------------------------------------------------------------------

class CoderMambaBlock(nn.Module):
    def __init__(self, cfg: EchoHybridCoderConfig):
        super().__init__()
        d = cfg.hidden_size
        d_inner = cfg.ssm_d_inner
        dt_rank = cfg.ssm_dt_rank
        d_state = cfg.ssm_d_state

        self.norm = RMSNorm(d, eps=cfg.layer_norm_eps)
        self.in_proj = nn.Linear(d, d_inner * 2, bias=cfg.ssm_use_bias)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=cfg.ssm_d_conv,
            padding=cfg.ssm_d_conv - 1, groups=d_inner,
            bias=cfg.ssm_use_conv_bias,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                       .unsqueeze(0).expand(d_inner, -1).clone())
        )
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d, bias=cfg.ssm_use_bias)

        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :x.shape[1]]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)
        ssm_input = self.x_proj(x_branch)
        dt, B_proj, C_proj = ssm_input.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        y = selective_scan_naive(x_branch, dt, A, B_proj, C_proj, self.D)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return residual + y


# ---------------------------------------------------------------------------
# Qwen-compatible Attention Block (GQA + SwiGLU + RoPE)
# ---------------------------------------------------------------------------

class CoderAttentionBlock(nn.Module):
    """Qwen2.5-style attention block: GQA + SwiGLU FFN + RoPE.

    Weight names match Qwen2.5 structure for direct donor transplant:
        self_attn.{q,k,v,o}_proj — GQA projections
        mlp.{gate,up,down}_proj  — SwiGLU FFN
    """

    def __init__(self, cfg: EchoHybridCoderConfig):
        super().__init__()
        d = cfg.hidden_size
        self.n_heads = cfg.attn_num_heads
        self.n_kv_heads = cfg.attn_num_kv_heads
        self.head_dim = cfg.attn_head_dim
        self.rope_theta = cfg.attn_rope_theta

        # GQA: Q has n_heads, K/V have n_kv_heads
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

        # SwiGLU FFN
        self.gate_proj = nn.Linear(d, cfg.attn_intermediate_size, bias=False)
        self.up_proj = nn.Linear(d, cfg.attn_intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.attn_intermediate_size, d, bias=False)

        # Norms
        self.input_layernorm = RMSNorm(d, eps=cfg.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(d, eps=cfg.layer_norm_eps)

        # RoPE cache (lazily extended)
        self._rope_cache = None
        self._rope_cache_len = 0

    def _get_rope(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._rope_cache is None or self._rope_cache_len < seq_len:
            self._rope_cache = _build_rope_cache(
                max(seq_len, 256), self.head_dim, self.rope_theta, device
            )
            self._rope_cache_len = max(seq_len, 256)
        return self._rope_cache.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x = self.input_layernorm(x)

        # GQA projections
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        angles = self._get_rope(L, x.device)
        q = _apply_rope(q, angles)
        k = _apply_rope(k, angles)

        # Expand KV heads for GQA: [B, n_kv, L, hd] → [B, n_heads, L, hd]
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, self.n_heads, L, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, self.n_heads, L, self.head_dim)

        # Scaled dot-product with causal mask
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.o_proj(out)
        x = residual + out

        # SwiGLU FFN
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return residual + x


# ---------------------------------------------------------------------------
# Block dispatcher
# ---------------------------------------------------------------------------

class CoderHybridBlock(nn.Module):
    def __init__(self, block_type: str, cfg: EchoHybridCoderConfig):
        super().__init__()
        self.block_type = block_type
        if block_type == "ssm":
            self.block = CoderMambaBlock(cfg)
        elif block_type == "attn":
            self.block = CoderAttentionBlock(cfg)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class EchoHybridCoderModel(nn.Module):
    """Born-compressed hybrid SSM+Transformer for code generation.

    28 blocks: 24 Mamba + 4 Qwen-compatible Attention
    d=1536, vocab=151936 (Qwen2.5 tokenizer)
    """

    def __init__(self, cfg: Optional[EchoHybridCoderConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = EchoHybridCoderConfig()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([
            CoderHybridBlock(bt, cfg) for bt in cfg.block_pattern
        ])
        self.norm_f = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.cfg.initializer_range)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}

    def n_params(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_word_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def __repr__(self) -> str:
        pat = "".join("S" if b == "ssm" else "A" for b in self.cfg.block_pattern)
        tied = " (tied)" if self.cfg.tie_word_embeddings else ""
        return (
            f"EchoHybridCoderModel(\n"
            f"  pattern=[{pat}], d={self.cfg.hidden_size}, "
            f"vocab={self.cfg.vocab_size}{tied}\n"
            f"  n_blocks={self.cfg.n_blocks} "
            f"({self.cfg.n_ssm} SSM, {self.cfg.n_attn} ATTN)\n"
            f"  params={self.n_params():,} total, "
            f"{self.n_params(exclude_embeddings=True):,} non-embedding\n"
            f")"
        )
