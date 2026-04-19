"""
WO-BORN-HYBRID-01: EchoHybridV2 -- Born-compressed Qwen-Mamba hybrid.

28-layer hybrid: 24 Mamba-1 + 4 Qwen2 ATTN+MLP at positions [6,13,20,27].
Dimensions from Qwen2.5-Coder-1.5B: d=1536, intermediate=8960, vocab=151936.
~778M params (24 Mamba + 4 ATTN+MLP + embeddings).

ATTN layers initialized from pretrained Qwen2.5-Coder-1.5B.
Mamba layers initialized randomly (initializer_range=0.02).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Try CUDA selective scan; fallback to naive
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_CUDA_AVAILABLE = True
except ImportError:
    MAMBA_CUDA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EchoHybridV2Config:
    """Configuration for Qwen-Mamba hybrid at 1.5B-class dimensions."""

    # Block layout: 24 "mamba" + 4 "attn" at positions 6,13,20,27
    block_pattern: List[str] = field(default_factory=lambda: (
        ["mamba"] * 6 + ["attn"] +     # 0-6
        ["mamba"] * 6 + ["attn"] +     # 7-13
        ["mamba"] * 6 + ["attn"] +     # 14-20
        ["mamba"] * 6 + ["attn"]       # 21-27
    ))

    # Qwen2.5-Coder-1.5B dimensions
    hidden_size: int = 1536
    vocab_size: int = 151936

    # Mamba-1 hparams
    ssm_expand: int = 2
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_use_conv_bias: bool = True
    ssm_use_bias: bool = False

    # Attention hparams (Qwen2 GQA)
    attn_num_heads: int = 12
    attn_num_kv_heads: int = 2
    attn_intermediate_size: int = 8960
    attn_bias: bool = True      # Qwen2.5-Coder-1.5B: q/k/v have bias
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768

    # Shared
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    @property
    def n_blocks(self) -> int:
        return len(self.block_pattern)

    @property
    def n_mamba(self) -> int:
        return sum(1 for b in self.block_pattern if b == "mamba")

    @property
    def n_attn(self) -> int:
        return sum(1 for b in self.block_pattern if b == "attn")

    @property
    def ssm_d_inner(self) -> int:
        return self.hidden_size * self.ssm_expand

    @property
    def ssm_dt_rank(self) -> int:
        return math.ceil(self.hidden_size / self.ssm_d_state)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.attn_num_heads

    @property
    def attn_positions(self) -> List[int]:
        return [i for i, b in enumerate(self.block_pattern) if b == "attn"]


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

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 32768, theta: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len <= self._max_len_cached:
            return
        self._max_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, seq_len: int):
        self._update_cache(seq_len, x.device, x.dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to q and k. cos/sin: [L, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Naive selective scan (CPU smoke test only)
# ---------------------------------------------------------------------------

def selective_scan_naive(u, dt, A, B, C, D):
    """Pure PyTorch scan. O(L) sequential -- unusable at scale. CPU smoke test only."""
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
# Mamba-1 Block
# ---------------------------------------------------------------------------

class Mamba1Block(nn.Module):
    """Mamba-1 SSM block at Qwen dimensions (d=1536, d_inner=3072).

    ~14.9M params per block. Uses CUDA selective scan when available.
    All nn.Linear layers are named for VQ compression.
    """

    def __init__(self, cfg: EchoHybridV2Config):
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

        # Project and split into x branch + gate z
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Conv1d on x branch: (B,L,D) -> (B,D,L) -> conv -> (B,L,D)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :residual.shape[1]]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # SSM projections
        ssm_input = self.x_proj(x_branch)
        dt_input, B_proj, C_proj = ssm_input.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt_input)

        A = -torch.exp(self.A_log.float())

        if MAMBA_CUDA_AVAILABLE and x.is_cuda:
            # CUDA kernel: expects (B,D,L) layout, applies softplus internally
            y = selective_scan_fn(
                x_branch.transpose(1, 2).contiguous(),
                dt.transpose(1, 2).contiguous(),
                A.contiguous(),
                B_proj.transpose(1, 2).contiguous(),
                C_proj.transpose(1, 2).contiguous(),
                self.D.float().contiguous(),
                z=None,
                delta_bias=None,
                delta_softplus=True,
                return_last_state=False,
            )
            y = y.transpose(1, 2)  # back to (B, L, D)
        else:
            dt = F.softplus(dt)
            y = selective_scan_naive(x_branch, dt, A, B_proj, C_proj, self.D)

        # Gate and project out
        y = y * F.silu(z)
        y = self.out_proj(y)
        return residual + y


# ---------------------------------------------------------------------------
# Qwen2 Attention + MLP Block
# ---------------------------------------------------------------------------

class Qwen2Block(nn.Module):
    """Qwen2-style block: GQA attention (RoPE) + SiLU-gated MLP.

    ~46.8M params per block. Matches Qwen2.5-Coder-1.5B layer layout exactly.
    """

    def __init__(self, cfg: EchoHybridV2Config):
        super().__init__()
        d = cfg.hidden_size
        n_heads = cfg.attn_num_heads
        n_kv_heads = cfg.attn_num_kv_heads
        head_dim = cfg.head_dim

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_kv_groups = n_heads // n_kv_heads

        # Attention sublayer
        self.input_layernorm = RMSNorm(d, eps=cfg.layer_norm_eps)
        self.q_proj = nn.Linear(d, n_heads * head_dim, bias=cfg.attn_bias)
        self.k_proj = nn.Linear(d, n_kv_heads * head_dim, bias=cfg.attn_bias)
        self.v_proj = nn.Linear(d, n_kv_heads * head_dim, bias=cfg.attn_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, d, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, cfg.max_position_embeddings, cfg.rope_theta)

        # MLP sublayer (SiLU-gated, Qwen2 style)
        self.post_attention_layernorm = RMSNorm(d, eps=cfg.layer_norm_eps)
        self.gate_proj = nn.Linear(d, cfg.attn_intermediate_size, bias=False)
        self.up_proj = nn.Linear(d, cfg.attn_intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.attn_intermediate_size, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # --- Attention sublayer ---
        residual = x
        x = self.input_layernorm(x)

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expand KV heads to match Q heads
        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        # Scaled dot-product attention with causal mask
        # Use F.scaled_dot_product_attention if available (Flash Attention)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        x = residual + out

        # --- MLP sublayer ---
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return residual + x


# ---------------------------------------------------------------------------
# Hybrid Block Dispatcher
# ---------------------------------------------------------------------------

class HybridBlock(nn.Module):
    """Routes to Mamba1Block or Qwen2Block based on block_type."""

    def __init__(self, block_type: str, cfg: EchoHybridV2Config):
        super().__init__()
        self.block_type = block_type
        if block_type == "mamba":
            self.block = Mamba1Block(cfg)
        elif block_type == "attn":
            self.block = Qwen2Block(cfg)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# EchoHybridV2Model
# ---------------------------------------------------------------------------

class EchoHybridV2Model(nn.Module):
    """Born-compressed Qwen-Mamba hybrid.

    28 blocks: 24 Mamba-1 + 4 Qwen2 ATTN+MLP.
    d=1536, vocab=151936 (Qwen2.5-Coder-1.5B tokenizer).
    ~778M total params with tied embeddings.
    """

    def __init__(self, cfg: Optional[EchoHybridV2Config] = None):
        super().__init__()
        if cfg is None:
            cfg = EchoHybridV2Config()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([
            HybridBlock(bt, cfg) for bt in cfg.block_pattern
        ])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Init Mamba blocks randomly; ATTN blocks get pretrained weights later
        self._init_mamba_weights()

    def _init_mamba_weights(self):
        """Initialize Mamba blocks with small normal. ATTN blocks left for pretrained."""
        for i, bt in enumerate(self.cfg.block_pattern):
            if bt != "mamba":
                continue
            block = self.blocks[i].block
            for name, p in block.named_parameters():
                if name in ("A_log", "D"):
                    continue  # keep their special init
                if p.dim() >= 2:
                    nn.init.normal_(p, std=self.cfg.initializer_range)
                elif "bias" in name:
                    nn.init.zeros_(p)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_checkpoint: bool = False,
    ):
        x = self.embed_tokens(input_ids)

        for block in self.blocks:
            if use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False,
                )
            else:
                x = block(x)

        x = self.norm(x)
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
            n -= self.embed_tokens.weight.numel()
            if not self.cfg.tie_word_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def get_linear_layers(self) -> Dict[str, nn.Linear]:
        """Return all compressible nn.Linear layers (excludes lm_head)."""
        layers = {}
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear) and name != "lm_head":
                layers[name] = mod
        return layers

    def __repr__(self) -> str:
        pat = "".join("M" if b == "mamba" else "A" for b in self.cfg.block_pattern)
        tied = " (tied)" if self.cfg.tie_word_embeddings else ""
        return (
            f"EchoHybridV2Model(\n"
            f"  pattern=[{pat}], d={self.cfg.hidden_size}, "
            f"vocab={self.cfg.vocab_size}{tied}\n"
            f"  n_blocks={self.cfg.n_blocks} "
            f"({self.cfg.n_mamba} Mamba, {self.cfg.n_attn} ATTN)\n"
            f"  params={self.n_params():,} total, "
            f"{self.n_params(exclude_embeddings=True):,} non-embedding\n"
            f"  CUDA scan: {MAMBA_CUDA_AVAILABLE}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Pretrained weight loading from Qwen2.5-Coder-1.5B
# ---------------------------------------------------------------------------

QWEN_ATTN_KEY_MAP = {
    "input_layernorm.weight": "model.layers.{}.input_layernorm.weight",
    "q_proj.weight": "model.layers.{}.self_attn.q_proj.weight",
    "q_proj.bias": "model.layers.{}.self_attn.q_proj.bias",
    "k_proj.weight": "model.layers.{}.self_attn.k_proj.weight",
    "k_proj.bias": "model.layers.{}.self_attn.k_proj.bias",
    "v_proj.weight": "model.layers.{}.self_attn.v_proj.weight",
    "v_proj.bias": "model.layers.{}.self_attn.v_proj.bias",
    "o_proj.weight": "model.layers.{}.self_attn.o_proj.weight",
    "post_attention_layernorm.weight": "model.layers.{}.post_attention_layernorm.weight",
    "gate_proj.weight": "model.layers.{}.mlp.gate_proj.weight",
    "up_proj.weight": "model.layers.{}.mlp.up_proj.weight",
    "down_proj.weight": "model.layers.{}.mlp.down_proj.weight",
}


def load_qwen_pretrained(
    model: EchoHybridV2Model,
    qwen_path: str = "Qwen/Qwen2.5-Coder-1.5B",
) -> Dict:
    """Load pretrained Qwen2.5-Coder-1.5B weights into ATTN layers + embeddings.

    Mamba blocks keep their random init. Attention blocks are loaded from
    the Qwen layer at the SAME position index (6, 13, 20, 27).

    Args:
        qwen_path: HuggingFace model name or local path.

    Returns:
        Dict with loading stats.
    """
    from transformers import AutoModelForCausalLM

    stats = {"attn_loaded": 0, "embed_loaded": 0, "skipped": [], "errors": []}

    print(f"  Loading Qwen2.5-Coder-1.5B from {qwen_path}...")
    qwen = AutoModelForCausalLM.from_pretrained(
        qwen_path, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    qwen_sd = qwen.state_dict()

    # --- Embeddings + final norm ---
    if "model.embed_tokens.weight" in qwen_sd:
        model.embed_tokens.weight.data.copy_(qwen_sd["model.embed_tokens.weight"])
        stats["embed_loaded"] += 1
        print(f"  Loaded embed_tokens: {model.embed_tokens.weight.shape}")

    if "model.norm.weight" in qwen_sd:
        model.norm.weight.data.copy_(qwen_sd["model.norm.weight"])
        stats["embed_loaded"] += 1
        print(f"  Loaded final norm")

    # --- Attention blocks from matching Qwen layers ---
    for block_idx, block_type in enumerate(model.cfg.block_pattern):
        if block_type != "attn":
            continue

        block = model.blocks[block_idx].block
        qwen_layer_idx = block_idx  # same position

        loaded_keys = []
        for our_key, qwen_template in QWEN_ATTN_KEY_MAP.items():
            qwen_key = qwen_template.format(qwen_layer_idx)
            if qwen_key not in qwen_sd:
                stats["skipped"].append(
                    f"blocks.{block_idx}.block.{our_key} (missing {qwen_key})"
                )
                continue

            parts = our_key.split(".")
            obj = block
            for p in parts[:-1]:
                obj = getattr(obj, p)
            param_name = parts[-1]

            if not hasattr(obj, param_name):
                stats["skipped"].append(
                    f"blocks.{block_idx}.block.{our_key} (no param)"
                )
                continue

            dst = getattr(obj, param_name)
            if dst is None:
                stats["skipped"].append(
                    f"blocks.{block_idx}.block.{our_key} (None)"
                )
                continue

            src = qwen_sd[qwen_key]
            if src.shape == dst.shape:
                dst.data.copy_(src)
                loaded_keys.append(our_key)
            else:
                stats["errors"].append(
                    f"blocks.{block_idx}.block.{our_key}: shape mismatch "
                    f"{src.shape} vs {dst.shape}"
                )

        stats["attn_loaded"] += len(loaded_keys)
        print(
            f"  ATTN block {block_idx} <- qwen layer {qwen_layer_idx}: "
            f"{len(loaded_keys)}/{len(QWEN_ATTN_KEY_MAP)} keys loaded"
        )

    del qwen, qwen_sd
    import gc; gc.collect()

    print(f"\n  Pretrained loading complete:")
    print(f"    ATTN params loaded: {stats['attn_loaded']}")
    print(f"    Embed/norm loaded: {stats['embed_loaded']}")
    if stats["skipped"]:
        print(f"    Skipped: {len(stats['skipped'])}")
    if stats["errors"]:
        print(f"    Errors: {stats['errors']}")

    return stats
