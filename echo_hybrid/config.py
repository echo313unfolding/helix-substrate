"""
WO-ECHO-HYBRID-01a: EchoHybrid config and model definition.

Born-compressed interleaved SSM + Transformer hybrid.
Phase 1: mamba-130m (d=768) + bert-base (d=768), exact dim match.

Block pattern: ["ssm","ssm","attn","ssm","ssm","attn","ssm","ssm","ssm"]
Total: 9 blocks (7 SSM, 2 ATTN), ~79M params.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EchoHybridConfig:
    # Architecture
    block_pattern: List[str] = field(
        default_factory=lambda: ["ssm", "ssm", "attn", "ssm", "ssm", "attn", "ssm", "ssm", "ssm"]
    )
    hidden_size: int = 768
    vocab_size: int = 50280   # mamba-130m tokenizer

    # SSM hparams (from mamba-130m)
    ssm_d_inner: int = 1536       # hidden_size * expand
    ssm_expand: int = 2
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_dt_rank: int = 48
    ssm_use_conv_bias: bool = True
    ssm_use_bias: bool = False

    # Attention hparams (from bert-base)
    attn_num_heads: int = 12
    attn_intermediate_size: int = 3072
    attn_dropout: float = 0.0     # no dropout for small distillation

    # Shared
    num_shared_attn_blocks: int = 0   # 0 = independent, >0 = weight-shared
    adapter_dim: Optional[int] = None  # None = exact dim match
    layer_norm_eps: float = 1e-5
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
# RMSNorm (shared by both block types for consistency)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Naive selective scan (pure PyTorch, correct but slow — Phase 1 only)
# ---------------------------------------------------------------------------

def selective_scan_naive(u, dt, A, B, C, D):
    """
    u:  (batch, seq, d_inner)
    dt: (batch, seq, d_inner)
    A:  (d_inner, d_state)
    B:  (batch, seq, d_state)
    C:  (batch, seq, d_state)
    D:  (d_inner,)
    Returns: (batch, seq, d_inner)
    """
    batch, seq_len, d_inner = u.shape
    d_state = A.shape[1]

    # Discretize: dA = exp(dt * A), dB_u = dt * B * u
    dtA = torch.exp(
        dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    )  # (B, L, d_inner, d_state)
    dtB_u = (
        dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    )  # (B, L, d_inner, d_state)

    # Sequential scan
    x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(seq_len):
        x = dtA[:, i] * x + dtB_u[:, i]
        y = (x * C[:, i].unsqueeze(1)).sum(-1)
        ys.append(y)

    y = torch.stack(ys, dim=1)
    return y + u * D.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# SSM Block (Mamba-style)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Single Mamba-style SSM block. All nn.Linear layers are named for
    later HelixLinear wrapping."""

    def __init__(self, cfg: EchoHybridConfig):
        super().__init__()
        d = cfg.hidden_size
        d_inner = cfg.ssm_d_inner
        dt_rank = cfg.ssm_dt_rank
        d_state = cfg.ssm_d_state

        self.norm = RMSNorm(d, eps=cfg.layer_norm_eps)

        # in_proj splits into x and gate (z)
        self.in_proj = nn.Linear(d, d_inner * 2, bias=cfg.ssm_use_bias)

        # Depthwise conv on x branch
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=cfg.ssm_d_conv,
            padding=cfg.ssm_d_conv - 1, groups=d_inner,
            bias=cfg.ssm_use_conv_bias,
        )

        # SSM projections
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # SSM parameters (not nn.Linear — these are small param tensors)
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

        # Project and split
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Conv1d on x branch: (B, L, D) -> (B, D, L) -> conv -> (B, D, L) -> (B, L, D)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :x.shape[1]]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # SSM projections
        ssm_input = self.x_proj(x_branch)
        dt, B_proj, C_proj = ssm_input.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        # SSM scan
        A = -torch.exp(self.A_log)
        y = selective_scan_naive(x_branch, dt, A, B_proj, C_proj, self.D)

        # Gate and project back
        y = y * F.silu(z)
        y = self.out_proj(y)

        return residual + y


# ---------------------------------------------------------------------------
# Attention Block (BERT-style multi-head self-attention + FFN)
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """Single Transformer attention block with causal mask."""

    def __init__(self, cfg: EchoHybridConfig):
        super().__init__()
        d = cfg.hidden_size
        n_heads = cfg.attn_num_heads
        assert d % n_heads == 0
        self.head_dim = d // n_heads
        self.n_heads = n_heads

        self.norm1 = RMSNorm(d, eps=cfg.layer_norm_eps)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

        self.norm2 = RMSNorm(d, eps=cfg.layer_norm_eps)
        self.ffn_up = nn.Linear(d, cfg.attn_intermediate_size)
        self.ffn_down = nn.Linear(cfg.attn_intermediate_size, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x = self.norm1(x)

        # Multi-head self-attention with causal mask
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product with causal mask
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        x = residual + out

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn_down(F.gelu(self.ffn_up(x)))
        return residual + x


# ---------------------------------------------------------------------------
# EchoHybridBlock — dispatcher
# ---------------------------------------------------------------------------

class EchoHybridBlock(nn.Module):
    """Routes to SSM or Attention based on block_type string."""

    def __init__(self, block_type: str, cfg: EchoHybridConfig):
        super().__init__()
        self.block_type = block_type
        if block_type == "ssm":
            self.block = MambaBlock(cfg)
        elif block_type == "attn":
            self.block = AttentionBlock(cfg)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# EchoHybridModel — full model
# ---------------------------------------------------------------------------

class EchoHybridModel(nn.Module):
    """Born-compressed interleaved SSM + Transformer hybrid.

    9 blocks: [SSM, SSM, ATTN, SSM, SSM, ATTN, SSM, SSM, SSM]
    d_model = 768, vocab = 50280 (mamba-130m tokenizer)
    """

    def __init__(self, cfg: Optional[EchoHybridConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = EchoHybridConfig()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([
            EchoHybridBlock(bt, cfg) for bt in cfg.block_pattern
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
            f"EchoHybridModel(\n"
            f"  pattern=[{pat}], d={self.cfg.hidden_size}, "
            f"vocab={self.cfg.vocab_size}{tied}\n"
            f"  n_blocks={self.cfg.n_blocks} "
            f"({self.cfg.n_ssm} SSM, {self.cfg.n_attn} ATTN)\n"
            f"  params={self.n_params():,} total, "
            f"{self.n_params(exclude_embeddings=True):,} non-embedding\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Pretrained weight initialization (WO-ECHO-HYBRID-05 Exp 4)
# ---------------------------------------------------------------------------

def load_pretrained_hybrid(
    model: EchoHybridModel,
    mamba_path: str = "/home/voidstr3m33/models/mamba-130m-hf-dense/model.safetensors",
    bert_name: str = "bert-base-uncased",
) -> dict:
    """Load pretrained weights from mamba-130m + bert-base into EchoHybrid.

    Mamba-130m (24 layers, d=768) → SSM blocks (7 blocks, cycled from layer 0-6)
    BERT-base (12 layers, d=768, intermediate=3072) → Attention blocks (2 blocks)

    Returns dict with loading stats.
    """
    from safetensors.torch import load_file

    stats = {"ssm_loaded": 0, "attn_loaded": 0, "skipped": [], "errors": []}

    # --- Load mamba-130m weights ---
    print(f"  Loading mamba-130m from {mamba_path}...")
    mamba_sd = load_file(mamba_path)

    # Embedding: backbone.embeddings.weight → model.embed.weight
    if "backbone.embeddings.weight" in mamba_sd:
        model.embed.weight.data.copy_(mamba_sd["backbone.embeddings.weight"])
        print(f"  Loaded embedding: {model.embed.weight.shape}")

    # Map SSM blocks: our blocks.{i}.block.{param} ← mamba backbone.layers.{j}.mixer.{param}
    # We have 7 SSM blocks among 9 total. Cycle through mamba's 24 layers.
    ssm_idx = 0
    mamba_n_layers = 24

    # Key mapping: EchoHybrid SSM ← Mamba-130m
    ssm_key_map = {
        "in_proj.weight": "mixer.in_proj.weight",
        "x_proj.weight": "mixer.x_proj.weight",
        "dt_proj.weight": "mixer.dt_proj.weight",
        "dt_proj.bias": "mixer.dt_proj.bias",
        "out_proj.weight": "mixer.out_proj.weight",
        "conv1d.weight": "mixer.conv1d.weight",
        "conv1d.bias": "mixer.conv1d.bias",
        "A_log": "mixer.A_log",
        "D": "mixer.D",
        "norm.weight": "norm.weight",
    }

    for block_idx, block_type in enumerate(model.cfg.block_pattern):
        if block_type != "ssm":
            continue

        mamba_layer_idx = ssm_idx % mamba_n_layers
        block = model.blocks[block_idx].block

        loaded_keys = []
        for our_key, mamba_suffix in ssm_key_map.items():
            mamba_key = f"backbone.layers.{mamba_layer_idx}.{mamba_suffix}"
            if mamba_key not in mamba_sd:
                stats["skipped"].append(f"blocks.{block_idx}.block.{our_key} (missing {mamba_key})")
                continue

            # Navigate to the right parameter
            parts = our_key.split(".")
            obj = block
            for p in parts[:-1]:
                obj = getattr(obj, p)

            param_name = parts[-1]
            src = mamba_sd[mamba_key]
            dst = getattr(obj, param_name)

            if src.shape == dst.shape:
                dst.data.copy_(src)
                loaded_keys.append(our_key)
            else:
                stats["errors"].append(
                    f"blocks.{block_idx}.block.{our_key}: shape mismatch "
                    f"{src.shape} vs {dst.shape}"
                )

        stats["ssm_loaded"] += len(loaded_keys)
        ssm_idx += 1
        print(f"  SSM block {block_idx} ← mamba layer {mamba_layer_idx}: "
              f"{len(loaded_keys)}/{len(ssm_key_map)} keys loaded")

    # --- Load BERT-base weights ---
    print(f"  Loading bert-base-uncased...")
    from transformers import BertModel
    bert = BertModel.from_pretrained(bert_name)
    bert_sd = bert.state_dict()
    bert_n_layers = 12

    # Key mapping: EchoHybrid Attention ← BERT
    attn_key_map = {
        "q_proj.weight": "encoder.layer.{}.attention.self.query.weight",
        "q_proj.bias": "encoder.layer.{}.attention.self.query.bias",
        "k_proj.weight": "encoder.layer.{}.attention.self.key.weight",
        "k_proj.bias": "encoder.layer.{}.attention.self.key.bias",
        "v_proj.weight": "encoder.layer.{}.attention.self.value.weight",
        "v_proj.bias": "encoder.layer.{}.attention.self.value.bias",
        "o_proj.weight": "encoder.layer.{}.attention.output.dense.weight",
        "o_proj.bias": "encoder.layer.{}.attention.output.dense.bias",
        "ffn_up.weight": "encoder.layer.{}.intermediate.dense.weight",
        "ffn_up.bias": "encoder.layer.{}.intermediate.dense.bias",
        "ffn_down.weight": "encoder.layer.{}.output.dense.weight",
        "ffn_down.bias": "encoder.layer.{}.output.dense.bias",
    }

    attn_idx = 0
    for block_idx, block_type in enumerate(model.cfg.block_pattern):
        if block_type != "attn":
            continue

        bert_layer_idx = attn_idx % bert_n_layers
        block = model.blocks[block_idx].block

        loaded_keys = []
        for our_key, bert_template in attn_key_map.items():
            bert_key = bert_template.format(bert_layer_idx)
            if bert_key not in bert_sd:
                stats["skipped"].append(f"blocks.{block_idx}.block.{our_key} (missing {bert_key})")
                continue

            parts = our_key.split(".")
            obj = block
            for p in parts[:-1]:
                obj = getattr(obj, p)

            param_name = parts[-1]
            src = bert_sd[bert_key]

            if not hasattr(obj, param_name):
                # Our model might not have bias for some layers
                stats["skipped"].append(f"blocks.{block_idx}.block.{our_key} (no param)")
                continue

            dst = getattr(obj, param_name)
            if dst is None:
                stats["skipped"].append(f"blocks.{block_idx}.block.{our_key} (param is None)")
                continue

            if src.shape == dst.shape:
                dst.data.copy_(src)
                loaded_keys.append(our_key)
            else:
                stats["errors"].append(
                    f"blocks.{block_idx}.block.{our_key}: shape mismatch "
                    f"{src.shape} vs {dst.shape}"
                )

        stats["attn_loaded"] += len(loaded_keys)
        attn_idx += 1
        print(f"  ATTN block {block_idx} ← bert layer {bert_layer_idx}: "
              f"{len(loaded_keys)}/{len(attn_key_map)} keys loaded")

    del bert, bert_sd  # free memory

    print(f"\n  Pretrained loading complete:")
    print(f"    SSM params loaded: {stats['ssm_loaded']}")
    print(f"    ATTN params loaded: {stats['attn_loaded']}")
    if stats["skipped"]:
        print(f"    Skipped: {len(stats['skipped'])}")
    if stats["errors"]:
        print(f"    Errors: {stats['errors']}")

    return stats
