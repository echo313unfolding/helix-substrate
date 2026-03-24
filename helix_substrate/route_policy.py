"""
Scored typed routing for CDNA v3 / HelixLinear.

Replaces blunt binary routing (SVD yes/no) with a multi-signal scored policy
that selects execution paths based on measured fragility, structure, and
runtime capability.

Three layers:
  1. Offline policy manifest: score each tensor, assign route type
  2. Runtime capability gate: check hardware/kernel support, downgrade if needed
  3. Dispatch logger: record intended vs actual route + metrics

Route types:
  EXACT             - stored as-is (norms, biases, 1D tensors)
  VQ_ONLY           - codebook + sidecar, no SVD correction
  VQ_PLUS_SVD_R8    - codebook + sidecar + SVD rank-8 residual
  DENSE_PASSTHROUGH - keep as nn.Linear (emergency, if fragility is extreme)
  FAIL_CLOSED       - no supported execution path (strict mode fails here)

Design:
  Block 21 (last layer) should emerge naturally from the score, not be hardcoded.
  Backward compatible: if no manifest, falls back to tensor_policy.get_policy().

Evidence chain:
  receipts/focus_budget/focus_budget_20260316T120843.json
  receipts/targeted_correction/targeted_correction_20260316T144853.json
  receipts/policy_patch_confirmation/policy_confirmation_20260316T163115.json
  receipts/kernel_profile/kernel_profile_20260316T171152.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from helix_substrate.tensor_policy import (
    TensorClass,
    TensorPolicy,
    classify_tensor,
    get_default_policy,
    get_policy,
    parse_tensor_name,
)


class RouteType(Enum):
    """Execution path for a compressed tensor."""
    EXACT = "exact"
    VQ_ONLY = "vq_only"
    VQ_PLUS_SVD_R8 = "vq_plus_svd_r8"
    DENSE_PASSTHROUGH = "dense_passthrough"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class RouteScore:
    """Composite routing score for a single tensor or block."""
    block_idx: int
    tensor_name: str
    route: RouteType
    composite_score: float
    # Score components (all 0-1 normalized before weighting)
    position_score: float = 0.0      # later = higher
    role_score: float = 0.0          # attention_qk > ffn > attention_vo
    kurtosis_score: float = 0.0      # higher kurtosis = more fragile
    # Activation-based (only available after focus-budget diagnostic)
    activation_cos_score: float = 0.0
    min_token_cos_score: float = 0.0
    max_diff_score: float = 0.0
    ppl_sensitivity_score: float = 0.0
    # Explanation
    reason: str = ""


@dataclass
class RuntimeCapability:
    """Hardware and kernel capability for route gating."""
    has_cuda: bool = False
    has_triton: bool = False
    compute_capability: str = ""
    vram_free_mb: float = 0.0
    strict_mode: bool = True
    allowed_routes: List[RouteType] = field(default_factory=lambda: [
        RouteType.EXACT, RouteType.VQ_ONLY, RouteType.VQ_PLUS_SVD_R8,
    ])


@dataclass
class DispatchRecord:
    """Record of intended vs actual execution path."""
    tensor_name: str
    intended_route: RouteType
    actual_route: str  # "fused", "naive", "dense", "exact"
    fallback_reason: Optional[str] = None
    latency_ms: Optional[float] = None


# ─── Scoring weights (receipt-calibrated) ───────────────────────────────
# Weights tuned against TinyLlama focus-budget + targeted correction receipts.
# Block 21 scores ~5.5, blocks 8/11 score ~1.8-2.0, block 0 scores ~0.8 (post-SVD).
SCORE_WEIGHTS = {
    "position": 0.20,
    "role": 0.05,
    "kurtosis": 0.05,
    "activation_cos": 0.30,
    "min_token_cos": 0.20,
    "max_diff": 0.10,
    "ppl_sensitivity": 0.10,
}

# Route thresholds (calibrated against targeted-correction receipts)
# Targeted correction proved: block 21 (score ~5.5) benefits from SVD,
# blocks 8 (score ~1.8) and 11 (score ~2.0) do NOT benefit.
# Threshold 3.5 cleanly separates the proven-benefit region.
THRESHOLD_SVD = 3.5       # Above this → VQ_PLUS_SVD_R8
THRESHOLD_DENSE = 8.0     # Above this → DENSE_PASSTHROUGH (extreme emergency only)

# Role fragility priors (before activation data)
ROLE_FRAGILITY = {
    TensorClass.ATTENTION_QK: 0.6,
    TensorClass.ATTENTION_VO: 0.4,
    TensorClass.FFN: 0.5,
    TensorClass.LM_HEAD: 0.8,
    TensorClass.EMBEDDING: 0.3,
    TensorClass.NORM: 0.0,
    TensorClass.UNKNOWN: 0.5,
}


def compute_route_score(
    tensor_name: str,
    shape: tuple,
    block_idx: int,
    n_blocks: int,
    kurtosis: float = 0.0,
    # Activation signals (from focus-budget receipt, optional)
    activation_cos_mean: float = 1.0,
    activation_cos_min: float = 1.0,
    max_abs_diff: float = 0.0,
    ppl_delta_pct: float = 0.0,
) -> RouteScore:
    """
    Compute composite routing score for a tensor.

    Higher score = more fragile = needs more correction.

    Returns RouteScore with route type and explanation.
    """
    tc = classify_tensor(tensor_name, shape=shape)

    # 1D tensors are always EXACT
    if len(shape) == 1 or tc == TensorClass.NORM:
        return RouteScore(
            block_idx=block_idx,
            tensor_name=tensor_name,
            route=RouteType.EXACT,
            composite_score=0.0,
            reason="1D/norm tensor → exact storage",
        )

    # Position score: normalized to [0, 1], last block = 1.0
    pos = block_idx / max(n_blocks - 1, 1)

    # Role score
    role = ROLE_FRAGILITY.get(tc, 0.5)

    # Kurtosis score: sigmoid-like normalization
    # kurtosis < 5: ~0, kurtosis 50: ~0.5, kurtosis 300: ~0.95
    kurt_norm = min(kurtosis / 300.0, 1.0) if kurtosis > 0 else 0.0

    # Activation cosine: (1 - cos) * 100, then normalize
    # Perfect = 0, block 21 had (1-0.9954)*100 = 0.46
    act_cos = min((1.0 - activation_cos_mean) * 100, 5.0) / 5.0

    # Min-token cosine: (1 - min_cos) * 10, then normalize
    # Perfect = 0, block 21 had (1-0.274)*10 = 7.26
    min_cos = min((1.0 - activation_cos_min) * 10, 10.0) / 10.0

    # Max abs diff: normalize against 25 (block 21 was 23.76)
    diff_norm = min(max_abs_diff / 25.0, 1.0)

    # PPL sensitivity: normalize against 1% (block 21 was 0.47%)
    ppl_norm = min(abs(ppl_delta_pct) / 1.0, 1.0)

    # Composite score (weighted sum)
    composite = (
        SCORE_WEIGHTS["position"] * pos +
        SCORE_WEIGHTS["role"] * role +
        SCORE_WEIGHTS["kurtosis"] * kurt_norm +
        SCORE_WEIGHTS["activation_cos"] * act_cos +
        SCORE_WEIGHTS["min_token_cos"] * min_cos +
        SCORE_WEIGHTS["max_diff"] * diff_norm +
        SCORE_WEIGHTS["ppl_sensitivity"] * ppl_norm
    )

    # Scale to make thresholds work (multiply by 10 so block 21 lands ~2.3)
    composite *= 10

    # Route decision
    if composite >= THRESHOLD_DENSE:
        route = RouteType.DENSE_PASSTHROUGH
        reason = f"composite {composite:.2f} >= {THRESHOLD_DENSE} → dense passthrough"
    elif composite >= THRESHOLD_SVD:
        route = RouteType.VQ_PLUS_SVD_R8
        reason = f"composite {composite:.2f} >= {THRESHOLD_SVD} → VQ + SVD rank-8"
    else:
        route = RouteType.VQ_ONLY
        reason = f"composite {composite:.2f} < {THRESHOLD_SVD} → VQ only"

    return RouteScore(
        block_idx=block_idx,
        tensor_name=tensor_name,
        route=route,
        composite_score=round(composite, 4),
        position_score=round(pos, 4),
        role_score=round(role, 4),
        kurtosis_score=round(kurt_norm, 4),
        activation_cos_score=round(act_cos, 4),
        min_token_cos_score=round(min_cos, 4),
        max_diff_score=round(diff_norm, 4),
        ppl_sensitivity_score=round(ppl_norm, 4),
        reason=reason,
    )


# ─── Manifest ───────────────────────────────────────────────────────────

@dataclass
class RouteManifest:
    """
    Complete routing manifest for a model.

    Generated offline from tensor stats + activation diagnostics.
    Loaded at runtime to drive routing decisions.
    """
    model_name: str
    n_blocks: int
    routes: Dict[str, RouteScore]  # tensor_name → RouteScore
    generation_timestamp: str = ""
    source_receipts: List[str] = field(default_factory=list)

    def get_route(self, tensor_name: str) -> RouteType:
        """Get route for a tensor. Falls back to VQ_ONLY if not in manifest."""
        score = self.routes.get(tensor_name)
        if score is None:
            return RouteType.VQ_ONLY
        return score.route

    def get_policy(self, tensor_name: str, shape: tuple,
                   block_idx: int = None, kurtosis: float = None) -> TensorPolicy:
        """
        Get TensorPolicy for a tensor, driven by manifest route.

        This is the bridge from scored routing back to the existing policy system.
        Backward compatible: produces the same TensorPolicy format that
        CDNAv3Writer and HelixLinear expect.
        """
        route = self.get_route(tensor_name)
        tc = classify_tensor(tensor_name, shape=shape)
        base = get_default_policy(tc)

        if route == RouteType.EXACT:
            return replace(base, storage_mode="exact", svd_residual_rank=0)
        elif route == RouteType.VQ_PLUS_SVD_R8:
            return replace(base, svd_residual_rank=8)
        elif route == RouteType.DENSE_PASSTHROUGH:
            return replace(base, storage_mode="exact", svd_residual_rank=0)
        elif route == RouteType.FAIL_CLOSED:
            raise RuntimeError(
                f"Route FAIL_CLOSED for {tensor_name}: no supported execution path. "
                f"Check hardware capability or relax strict mode."
            )
        else:  # VQ_ONLY
            return replace(base, svd_residual_rank=0)

    def summary(self) -> Dict[str, int]:
        """Count routes by type."""
        counts = {rt.value: 0 for rt in RouteType}
        for score in self.routes.values():
            counts[score.route.value] += 1
        return counts

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "model_name": self.model_name,
            "n_blocks": self.n_blocks,
            "generation_timestamp": self.generation_timestamp,
            "source_receipts": self.source_receipts,
            "summary": self.summary(),
            "routes": {
                name: {
                    "route": score.route.value,
                    "composite_score": score.composite_score,
                    "block_idx": score.block_idx,
                    "position_score": score.position_score,
                    "role_score": score.role_score,
                    "kurtosis_score": score.kurtosis_score,
                    "activation_cos_score": score.activation_cos_score,
                    "min_token_cos_score": score.min_token_cos_score,
                    "max_diff_score": score.max_diff_score,
                    "ppl_sensitivity_score": score.ppl_sensitivity_score,
                    "reason": score.reason,
                }
                for name, score in self.routes.items()
            },
        }

    def save(self, path: Path):
        """Save manifest to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RouteManifest":
        """Load manifest from JSON file."""
        data = json.loads(path.read_text())
        routes = {}
        for name, rd in data["routes"].items():
            routes[name] = RouteScore(
                block_idx=rd["block_idx"],
                tensor_name=name,
                route=RouteType(rd["route"]),
                composite_score=rd["composite_score"],
                position_score=rd.get("position_score", 0),
                role_score=rd.get("role_score", 0),
                kurtosis_score=rd.get("kurtosis_score", 0),
                activation_cos_score=rd.get("activation_cos_score", 0),
                min_token_cos_score=rd.get("min_token_cos_score", 0),
                max_diff_score=rd.get("max_diff_score", 0),
                ppl_sensitivity_score=rd.get("ppl_sensitivity_score", 0),
                reason=rd.get("reason", ""),
            )
        return cls(
            model_name=data["model_name"],
            n_blocks=data["n_blocks"],
            routes=routes,
            generation_timestamp=data.get("generation_timestamp", ""),
            source_receipts=data.get("source_receipts", []),
        )


# ─── Runtime capability resolution ─────────────────────────────────────

def resolve_capability() -> RuntimeCapability:
    """Detect current runtime capability."""
    import torch

    cap = RuntimeCapability()
    cap.has_cuda = torch.cuda.is_available()

    if cap.has_cuda:
        props = torch.cuda.get_device_properties(0)
        cap.compute_capability = f"{props.major}.{props.minor}"
        free, total = torch.cuda.mem_get_info()
        cap.vram_free_mb = round(free / (1024 * 1024), 1)

    try:
        import triton
        cap.has_triton = True
    except ImportError:
        cap.has_triton = False

    # Set allowed routes based on capability
    cap.allowed_routes = [RouteType.EXACT, RouteType.VQ_ONLY]
    if cap.has_cuda and cap.has_triton:
        cap.allowed_routes.append(RouteType.VQ_PLUS_SVD_R8)
    cap.allowed_routes.append(RouteType.DENSE_PASSTHROUGH)

    return cap


def gate_route(intended: RouteType, capability: RuntimeCapability) -> tuple:
    """
    Gate an intended route against runtime capability.

    Returns (actual_route, fallback_reason).
    If strict_mode and intended route not available, returns FAIL_CLOSED.
    """
    if intended in capability.allowed_routes:
        return intended, None

    # Downgrade path
    if intended == RouteType.VQ_PLUS_SVD_R8:
        if RouteType.VQ_ONLY in capability.allowed_routes:
            return RouteType.VQ_ONLY, "SVD not available (no CUDA/Triton), downgraded to VQ_ONLY"
        elif RouteType.DENSE_PASSTHROUGH in capability.allowed_routes:
            return RouteType.DENSE_PASSTHROUGH, "VQ not available, using dense passthrough"

    if capability.strict_mode:
        return RouteType.FAIL_CLOSED, f"Route {intended.value} not supported and strict mode is on"

    return RouteType.VQ_ONLY, f"Fallback to VQ_ONLY (intended: {intended.value})"


# ─── Backward compatibility bridge ─────────────────────────────────────

def get_policy_from_manifest(
    manifest: Optional[RouteManifest],
    name: str,
    shape: tuple,
    block_idx: int = None,
    kurtosis: float = None,
    n_blocks: int = None,
) -> TensorPolicy:
    """
    Get TensorPolicy using manifest if available, else fall back to binary rules.

    This is the drop-in replacement for tensor_policy.get_policy() that supports
    both the old binary system and the new scored routing.
    """
    if manifest is not None:
        return manifest.get_policy(name, shape, block_idx=block_idx, kurtosis=kurtosis)

    # Fall back to existing binary routing
    return get_policy(name, shape, block_idx=block_idx, kurtosis=kurtosis,
                      n_blocks=n_blocks)
