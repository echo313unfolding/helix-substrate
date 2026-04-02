"""
Tensor classification and default compression policies for CDNA v3.

Classifies tensors by name and shape, then assigns per-class policies
controlling quantization, sidecar generation, and block layout.

Handles both GGUF-style names (blk.N.attn_q.weight) and HuggingFace-style
names (model.layers.N.self_attn.q_proj.weight).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional


class TensorClass(Enum):
    EMBEDDING = "embedding"
    ATTENTION_QK = "attention_qk"
    ATTENTION_VO = "attention_vo"
    FFN = "ffn"
    NORM = "norm"
    LM_HEAD = "lm_head"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TensorPolicy:
    tensor_class: TensorClass
    storage_mode: str  # "exact", "codebook", "codebook+sidecar", "morpho"
    n_clusters: int = 256
    percentile: float = 99.9
    use_kmeans: bool = True
    sidecar_enabled: bool = True
    block_rows: int = 16
    max_corrections: int = 512
    # Multi-dimensional VQ: 1=scalar (backward compat), 2=2D pairs, 4=4D quads.
    # With d>1, each uint8 index covers d weights → d× fewer indices.
    # Codebook shape becomes [k, d] instead of [k].
    vector_dim: int = 1
    # SVD residual correction (rank > 0 enables VQ + low-rank residual decoder)
    svd_residual_rank: int = 0
    # Morpho codec parameters (only used when storage_mode="morpho")
    morpho_growth_steps: int = 1000
    morpho_target_se: float = 1.5
    morpho_geometry: str = "fibpi3d"
    morpho_c: float = 0.35
    morpho_gamma: float = 0.02
    morpho_fit: bool = False
    morpho_n_codons: int = 32
    morpho_max_iter: int = 200
    morpho_verbose: bool = False
    morpho_spectral: bool = False  # Use FFT/spectral path (provably optimal for 1D)
    morpho_n_harmonics: int = 64   # Number of harmonics for spectral mode
    morpho_min_cosine: float = 0.0  # Quality gate: fallback to exact if below this


# GGUF name patterns
_GGUF_PATTERNS = [
    (re.compile(r"blk\.(\d+)\.attn_q\.weight"), TensorClass.ATTENTION_QK),
    (re.compile(r"blk\.(\d+)\.attn_k\.weight"), TensorClass.ATTENTION_QK),
    (re.compile(r"blk\.(\d+)\.attn_v\.weight"), TensorClass.ATTENTION_VO),
    (re.compile(r"blk\.(\d+)\.attn_output\.weight"), TensorClass.ATTENTION_VO),
    (re.compile(r"blk\.(\d+)\.ffn_gate\.weight"), TensorClass.FFN),
    (re.compile(r"blk\.(\d+)\.ffn_up\.weight"), TensorClass.FFN),
    (re.compile(r"blk\.(\d+)\.ffn_down\.weight"), TensorClass.FFN),
    (re.compile(r"blk\.(\d+)\.\w*norm\w*\.weight"), TensorClass.NORM),
    (re.compile(r"^token_embd\.weight$"), TensorClass.EMBEDDING),
    (re.compile(r"^output\.weight$"), TensorClass.LM_HEAD),
    (re.compile(r"^output_norm\.weight$"), TensorClass.NORM),
]

# HuggingFace name patterns
_HF_PATTERNS = [
    (re.compile(r"layers\.(\d+)\.self_attn\.q_proj\.weight"), TensorClass.ATTENTION_QK),
    (re.compile(r"layers\.(\d+)\.self_attn\.k_proj\.weight"), TensorClass.ATTENTION_QK),
    (re.compile(r"layers\.(\d+)\.self_attn\.v_proj\.weight"), TensorClass.ATTENTION_VO),
    (re.compile(r"layers\.(\d+)\.self_attn\.o_proj\.weight"), TensorClass.ATTENTION_VO),
    (re.compile(r"layers\.(\d+)\.mlp\.gate_proj\.weight"), TensorClass.FFN),
    (re.compile(r"layers\.(\d+)\.mlp\.up_proj\.weight"), TensorClass.FFN),
    (re.compile(r"layers\.(\d+)\.mlp\.down_proj\.weight"), TensorClass.FFN),
    (re.compile(r"layers\.(\d+)\.\w*norm\w*\.weight"), TensorClass.NORM),
    (re.compile(r"embed_tokens\.weight$"), TensorClass.EMBEDDING),
    (re.compile(r"lm_head\.weight$"), TensorClass.LM_HEAD),
    # Mamba SSM mixer layers
    (re.compile(r"layers\.(\d+)\.mixer\.in_proj\.weight"), TensorClass.FFN),
    (re.compile(r"layers\.(\d+)\.mixer\.out_proj\.weight"), TensorClass.FFN),
    (re.compile(r"layers\.(\d+)\.mixer\.x_proj\.weight"), TensorClass.UNKNOWN),
    (re.compile(r"layers\.(\d+)\.mixer\.dt_proj\.weight"), TensorClass.UNKNOWN),
    # Mamba-2 additional mixer layers
    (re.compile(r"layers\.(\d+)\.mixer\.B_proj\.weight"), TensorClass.FFN),
    (re.compile(r"layers\.(\d+)\.mixer\.C_proj\.weight"), TensorClass.FFN),
    (re.compile(r"layers\.(\d+)\.mixer\.A_log"), TensorClass.UNKNOWN),
    (re.compile(r"layers\.(\d+)\.mixer\.D$"), TensorClass.UNKNOWN),
    (re.compile(r"layers\.(\d+)\.mixer\.norm\.weight"), TensorClass.NORM),
    # Mamba embedding (Mamba-1: "embeddings", Mamba-2: "embedding")
    (re.compile(r"backbone\.embeddings?\.weight$"), TensorClass.EMBEDDING),
]


def classify_tensor(
    name: str,
    shape: Optional[tuple[int, ...]] = None,
) -> TensorClass:
    """
    Classify a tensor by its name and shape.

    1D tensors are always classified as NORM (biases, layer norms).
    2D tensors are classified by name pattern matching.

    Args:
        name: Tensor name (GGUF or HuggingFace format)
        shape: Tensor shape (optional, used for 1D detection)

    Returns:
        TensorClass enum value
    """
    if shape is not None and len(shape) == 1:
        return TensorClass.NORM

    for pattern, cls in _GGUF_PATTERNS:
        if pattern.search(name):
            return cls

    for pattern, cls in _HF_PATTERNS:
        if pattern.search(name):
            return cls

    return TensorClass.UNKNOWN


def parse_tensor_name(name: str) -> dict:
    """
    Extract structured info from a tensor name.

    Returns:
        Dict with keys: layer_idx (int or None), module_family (str), projection (str)
    """
    # Try GGUF format: blk.N.module.weight
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight", name)
    if m:
        layer_idx = int(m.group(1))
        raw = m.group(2)
        if raw.startswith("attn_"):
            return {"layer_idx": layer_idx, "module_family": "attention", "projection": raw[5:]}
        if raw.startswith("ffn_"):
            return {"layer_idx": layer_idx, "module_family": "ffn", "projection": raw[4:]}
        return {"layer_idx": layer_idx, "module_family": raw, "projection": ""}

    # Try HF format: model.layers.N.module.proj.weight
    m = re.search(r"layers\.(\d+)\.self_attn\.(\w+)\.weight", name)
    if m:
        return {"layer_idx": int(m.group(1)), "module_family": "attention", "projection": m.group(2).replace("_proj", "")}

    m = re.search(r"layers\.(\d+)\.mlp\.(\w+)\.weight", name)
    if m:
        return {"layer_idx": int(m.group(1)), "module_family": "ffn", "projection": m.group(2).replace("_proj", "")}

    # Special tensors
    if "embd" in name or "embed" in name:
        return {"layer_idx": None, "module_family": "embedding", "projection": ""}
    if name in ("output.weight", "lm_head.weight"):
        return {"layer_idx": None, "module_family": "lm_head", "projection": ""}
    if "norm" in name:
        return {"layer_idx": None, "module_family": "norm", "projection": ""}

    return {"layer_idx": None, "module_family": "unknown", "projection": ""}


# Default policies per tensor class
_DEFAULT_POLICIES = {
    TensorClass.NORM: TensorPolicy(
        tensor_class=TensorClass.NORM,
        storage_mode="exact",
        n_clusters=0,
        percentile=100.0,
        use_kmeans=False,
        sidecar_enabled=False,
        block_rows=0,
        max_corrections=0,
    ),
    TensorClass.EMBEDDING: TensorPolicy(
        tensor_class=TensorClass.EMBEDDING,
        storage_mode="exact",
        n_clusters=0,
        percentile=100.0,
        use_kmeans=False,
        sidecar_enabled=False,
        block_rows=0,
        max_corrections=0,
    ),
    TensorClass.ATTENTION_QK: TensorPolicy(
        tensor_class=TensorClass.ATTENTION_QK,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        percentile=99.95,
        use_kmeans=True,
        sidecar_enabled=True,
        block_rows=16,
        max_corrections=512,
    ),
    TensorClass.ATTENTION_VO: TensorPolicy(
        tensor_class=TensorClass.ATTENTION_VO,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        percentile=99.9,
        use_kmeans=True,
        sidecar_enabled=True,
        block_rows=16,
        max_corrections=512,
    ),
    TensorClass.FFN: TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        percentile=99.9,
        use_kmeans=True,
        sidecar_enabled=True,
        block_rows=16,
        max_corrections=512,
    ),
    TensorClass.LM_HEAD: TensorPolicy(
        tensor_class=TensorClass.LM_HEAD,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        percentile=99.9,
        use_kmeans=True,
        sidecar_enabled=True,
        block_rows=16,
        max_corrections=512,
    ),
    TensorClass.UNKNOWN: TensorPolicy(
        tensor_class=TensorClass.UNKNOWN,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        percentile=99.9,
        use_kmeans=True,
        sidecar_enabled=True,
        block_rows=32,
        max_corrections=256,
    ),
}


MORPHO_FIT_POLICY = TensorPolicy(
    tensor_class=TensorClass.NORM,
    storage_mode="morpho",
    morpho_fit=True,
    morpho_growth_steps=300,
    morpho_max_iter=150,
    morpho_n_codons=32,
)

# FFT-based morpho policy: direct Fourier projection (provably optimal for 1D norms)
# Cosine 0.96+ on 42/45 TinyLlama norms, <1ms encode, ~10x compression.
# Quality gate at 0.90: weak norms (blk.0-2.attn_norm) fall back to exact.
MORPHO_FFT_POLICY = TensorPolicy(
    tensor_class=TensorClass.NORM,
    storage_mode="morpho",
    morpho_fit=True,
    morpho_spectral=True,
    morpho_n_codons=32,
    morpho_min_cosine=0.90,
    morpho_max_iter=150,
    morpho_growth_steps=300,
)


def get_default_policy(tensor_class: TensorClass) -> TensorPolicy:
    """Get the default compression policy for a tensor class."""
    return _DEFAULT_POLICIES[tensor_class]


def get_policy(
    name: str,
    shape: tuple[int, ...],
    block_idx: int | None = None,
    kurtosis: float | None = None,
    n_blocks: int | None = None,
) -> TensorPolicy:
    """
    Tensor compression policy router.

    Returns the default VQ-256 policy for all tensors. SVD routing was
    disabled after crossover testing proved it adds zero value at any scale
    (1.5B, 3B, 7B). Plain k-means VQ-256 is optimal everywhere.

    Historical context: kurtosis-based SVD routing (kurtosis > 5 → SVD rank-8)
    improved per-tensor cosine but had no effect on model PPL at 1.5B/3B
    and actively hurt at 7B (+4%). The per-tensor metric does not predict
    model-level behavior.

    Receipts: receipts/scaling_analysis/ppl_eval_qwen2.5-*_20260327T*.json
    """
    tc = classify_tensor(name, shape=shape)
    base = get_default_policy(tc)

    # SVD routing DISABLED (2026-03-28).
    # Crossover test at 1.5B/3B/7B proved SVD adds zero value at any scale:
    #   1.5B: 15.31 with SVD → 15.31 without (noise)
    #   3B:   9.56 with SVD → 9.60 without (noise)
    #   7B:   7.68 with SVD → 7.39 without (SVD hurts +4%)
    # Plain k-means VQ-256 is optimal everywhere. Per-tensor cosine improvement
    # from SVD does not predict model-level PPL improvement.
    # Receipts: receipts/scaling_analysis/ppl_eval_qwen2.5-*_20260327T*.json
    return base
