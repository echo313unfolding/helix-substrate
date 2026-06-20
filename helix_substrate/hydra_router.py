"""HXQ Hydra Router — multi-head codec routing per tensor.

One profiler trunk, multiple codec heads. The router decides which head
handles each tensor based on probed cosine similarity, kurtosis, tensor
type, and policy.

This is the planning layer. It does NOT perform quantization or kernel
dispatch. It produces a CompressionPlan that downstream tools consume.

Sidecar policy (2026-04-30): Sidecar is NOT used for uniform affine heads.
Sparse outlier correction does not help when quantization error is evenly
distributed (affine case). Sidecar is reserved for VQ/outlier-heavy codecs
where a small number of positions dominate error.
Receipt: receipts/hxq_mixed_lowbit_sidecar/sidecar_20260430T192842.json

Usage:
    from helix_substrate.hydra_router import HydraRouter, TensorProfile

    profiles = [TensorProfile(...), ...]
    router = HydraRouter(policy="edge_balanced")
    plan = router.route(profiles)
    print(plan.avg_bpw, plan.summary())

See docs/HXQ_HYDRA_ROUTER.md for the full spec.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Codec heads
# ═══════════════════════════════════════════════════════════════════════════

class Head(str, Enum):
    EXACT = "exact"
    AFFINE6 = "affine6"
    AFFINE5 = "affine5"
    AFFINE4 = "affine4"
    AFFINE3 = "affine3"
    SIDECAR_VQ = "sidecar_vq"  # sparse outlier repair for VQ codecs only (not affine)
    RVQ = "rvq"                # reserved

    @property
    def bpw(self) -> float:
        """Bits per weight for group_size=128."""
        return _HEAD_BPW.get(self, 0.0)


_HEAD_BPW = {
    Head.EXACT: 16.0,
    Head.AFFINE6: 6.25,
    Head.AFFINE5: 5.25,
    Head.AFFINE4: 4.25,
    Head.AFFINE3: 3.25,
    Head.RVQ: 6.0,  # placeholder
}

# Sidecar adds ~0.1-0.5 bpw depending on density
SIDECAR_BPW_ESTIMATE = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# Profiler trunk
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TensorProfile:
    """Per-tensor features from the profiler trunk."""
    tensor_name: str
    shape: tuple[int, ...]
    layer_index: int
    tensor_type: str             # q_proj, k_proj, gate_proj, embed, lm_head, norm, etc.
    n_params: int
    kurtosis: float = 0.0
    std: float = 0.0
    # Cosine similarity under each affine head
    affine6_cosine: float = 1.0
    affine5_cosine: float = 1.0
    affine4_cosine: float = 0.0
    affine3_cosine: float = 0.0
    # Error at reference head (affine6)
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Routing decision
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TensorPlan:
    """Routing decision for a single tensor."""
    tensor_name: str
    head: Head
    reason: list[str]
    bpw: float
    expected_cosine: float
    sidecar_budget: float = 0.0   # fraction of params stored as outliers
    fallback_head: Head = Head.EXACT

    def to_dict(self) -> dict:
        d = asdict(self)
        d["head"] = self.head.value
        d["fallback_head"] = self.fallback_head.value
        return d


@dataclass
class CompressionPlan:
    """Full model compression plan."""
    model: str
    policy: str
    tensors: list[TensorPlan] = field(default_factory=list)

    @property
    def avg_bpw(self) -> float:
        if not self.tensors:
            return 0.0
        total_bits = sum(t.bpw * self._param_count(t) for t in self.tensors)
        total_params = sum(self._param_count(t) for t in self.tensors)
        return total_bits / total_params if total_params > 0 else 0.0

    def _param_count(self, tp: TensorPlan) -> int:
        """Look up param count from the stored profiles."""
        return self._param_map.get(tp.tensor_name, 1)

    def summary(self) -> dict:
        head_counts = {}
        for t in self.tensors:
            h = t.head.value
            head_counts[h] = head_counts.get(h, 0) + 1
        return {
            "model": self.model,
            "policy": self.policy,
            "n_tensors": len(self.tensors),
            "avg_bpw": round(self.avg_bpw, 3),
            "head_distribution": head_counts,
        }

    def to_json(self, path: Optional[Path] = None) -> str:
        data = {
            "model": self.model,
            "policy": self.policy,
            "avg_bpw": round(self.avg_bpw, 3),
            "n_tensors": len(self.tensors),
            "plan": [t.to_dict() for t in self.tensors],
        }
        text = json.dumps(data, indent=2)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return text


# ═══════════════════════════════════════════════════════════════════════════
# Policies
# ═══════════════════════════════════════════════════════════════════════════

POLICIES = ("quality_first", "edge_balanced", "size_target", "experimental_lowbit")

# Tensor types that are always exact unless explicitly overridden
EXACT_TYPES = {"embed", "embed_tokens", "lm_head", "norm", "layernorm",
               "input_layernorm", "post_attention_layernorm", "final_layernorm",
               "model.norm", "model.embed_tokens", "lm_head"}

# High-risk attention projections
ATTENTION_FRAGILE = {"q_proj", "k_proj"}


def _is_exact_tensor(profile: TensorProfile) -> bool:
    """Check if tensor should always be exact."""
    ttype = profile.tensor_type.lower()
    tname = profile.tensor_name.lower()
    for exact in EXACT_TYPES:
        if exact in ttype or exact in tname:
            return True
    # 1D tensors (biases, norms) are always exact
    if len(profile.shape) < 2:
        return True
    return False


def _is_early_attention(profile: TensorProfile, early_threshold: int = 3) -> bool:
    """Check if tensor is an early-layer attention projection."""
    return (profile.layer_index < early_threshold and
            profile.tensor_type in ATTENTION_FRAGILE)


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════

class HydraRouter:
    """Multi-head codec router.

    Args:
        policy: One of POLICIES.
        kurtosis_threshold: Kurtosis above this triggers affine6 for attention.
        cosine_gate_5: Minimum affine5 cosine to allow affine5.
        cosine_gate_4: Minimum affine4 cosine to allow affine4.
        cosine_gate_3: Minimum affine3 cosine to allow affine3.
        bpw_target: Target avg bpw for size_target policy.
        sidecar_threshold: Cosine above this qualifies for sidecar repair
                           (tensor almost passes but doesn't quite).
    """

    def __init__(
        self,
        policy: str = "edge_balanced",
        kurtosis_threshold: float = 50.0,
        cosine_gate_5: float = 0.998,
        cosine_gate_4: float = 0.999,
        cosine_gate_3: float = 0.998,
        bpw_target: float = 5.0,
        sidecar_threshold: float = 0.996,
        sidecar_density: float = 0.0025,
    ):
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy: {policy}. Must be one of {POLICIES}")
        self.policy = policy
        self.kurtosis_threshold = kurtosis_threshold
        self.cosine_gate_5 = cosine_gate_5
        self.cosine_gate_4 = cosine_gate_4
        self.cosine_gate_3 = cosine_gate_3
        self.bpw_target = bpw_target
        self.sidecar_threshold = sidecar_threshold
        self.sidecar_density = sidecar_density

    def route(self, profiles: list[TensorProfile],
              model_name: str = "unknown") -> CompressionPlan:
        """Route all tensors according to the active policy."""
        plan = CompressionPlan(model=model_name, policy=self.policy)
        plan._param_map = {p.tensor_name: p.n_params for p in profiles}

        if self.policy == "quality_first":
            plan.tensors = [self._route_quality_first(p) for p in profiles]
        elif self.policy == "edge_balanced":
            plan.tensors = [self._route_edge_balanced(p) for p in profiles]
        elif self.policy == "size_target":
            plan.tensors = self._route_size_target(profiles)
        elif self.policy == "experimental_lowbit":
            plan.tensors = [self._route_experimental(p) for p in profiles]

        return plan

    def route_with_ghost(
        self,
        profiles: list[TensorProfile],
        ghost_preroute: "GhostPreRoute",
        ghost_features_map: dict[str, dict],
        arch: str,
        model_name: str = "unknown",
    ) -> tuple[CompressionPlan, "GhostPreRouteResult"]:
        """Route with Ghost pre-screening.

        For tensors where Ghost says SKIP_PROBE, uses the policy's default
        safe head (affine5 for edge_balanced/size_target, affine6 for
        quality_first) without requiring probe cosine data.

        For tensors where Ghost says PROBE_REQUIRED, uses normal
        probe-based routing (requires valid cosine fields in profile).

        Args:
            profiles: TensorProfiles (cosine fields may be unset for
                      tensors Ghost will pre-route)
            ghost_preroute: calibrated GhostPreRoute instance
            ghost_features_map: {tensor_name: {te, tr, mo, ac}}
            arch: architecture family for Ghost model selection
            model_name: model identifier

        Returns:
            (CompressionPlan, GhostPreRouteResult)
        """
        from helix_substrate.ghost_bridge import (
            GhostPreRouteResult, PreRouteAction,
        )

        plan = CompressionPlan(model=model_name, policy=self.policy)
        plan._param_map = {p.tensor_name: p.n_params for p in profiles}

        ghost_result = GhostPreRouteResult(n_total=len(profiles))
        tensor_plans = []

        for p in profiles:
            # Always-exact tensors bypass Ghost
            if _is_exact_tensor(p):
                tensor_plans.append(TensorPlan(
                    tensor_name=p.tensor_name, head=Head.EXACT,
                    reason=["exact_tensor_type"], bpw=Head.EXACT.bpw,
                    expected_cosine=1.0, fallback_head=Head.EXACT,
                ))
                continue

            # Check Ghost pre-route
            gf = ghost_features_map.get(p.tensor_name)
            if gf is not None:
                decision = ghost_preroute.decide(gf, arch)
                ghost_result.decisions.append({
                    "tensor_name": p.tensor_name,
                    "decision": decision,
                })

                if decision.action == PreRouteAction.SKIP_PROBE:
                    ghost_result.n_ghost_routed += 1
                    # Ghost says safe → use policy's default safe head
                    safe_head, safe_bpw = self._ghost_safe_head()
                    tensor_plans.append(TensorPlan(
                        tensor_name=p.tensor_name, head=safe_head,
                        reason=["ghost_preroute", f"confidence={decision.confidence:.3f}"],
                        bpw=safe_bpw,
                        expected_cosine=0.999,  # Ghost-estimated (conservative)
                        fallback_head=Head.AFFINE6,
                    ))
                    continue
                else:
                    ghost_result.n_probe_required += 1
            else:
                ghost_result.n_probe_required += 1

            # Normal probe-based routing
            if self.policy == "quality_first":
                tensor_plans.append(self._route_quality_first(p))
            elif self.policy == "edge_balanced":
                tensor_plans.append(self._route_edge_balanced(p))
            elif self.policy == "experimental_lowbit":
                tensor_plans.append(self._route_experimental(p))
            else:
                # size_target handles all tensors at once; fall through
                tensor_plans.append(self._route_edge_balanced(p))

        plan.tensors = tensor_plans
        return plan, ghost_result

    def _ghost_safe_head(self) -> tuple[Head, float]:
        """Default safe head for ghost-routed tensors under current policy."""
        if self.policy == "quality_first":
            return Head.AFFINE6, Head.AFFINE6.bpw
        elif self.policy in ("edge_balanced", "size_target"):
            return Head.AFFINE5, Head.AFFINE5.bpw
        elif self.policy == "experimental_lowbit":
            return Head.AFFINE5, Head.AFFINE5.bpw
        return Head.AFFINE6, Head.AFFINE6.bpw

    # ── Per-tensor policies ──

    def _route_quality_first(self, p: TensorProfile) -> TensorPlan:
        if _is_exact_tensor(p):
            return TensorPlan(
                tensor_name=p.tensor_name, head=Head.EXACT,
                reason=["exact_tensor_type"], bpw=Head.EXACT.bpw,
                expected_cosine=1.0, fallback_head=Head.EXACT,
            )

        if p.affine6_cosine < 0.999:
            return TensorPlan(
                tensor_name=p.tensor_name, head=Head.EXACT,
                reason=["affine6_cosine_too_low"],
                bpw=Head.EXACT.bpw, expected_cosine=1.0,
                fallback_head=Head.EXACT,
            )

        return TensorPlan(
            tensor_name=p.tensor_name, head=Head.AFFINE6,
            reason=["quality_first_default"], bpw=Head.AFFINE6.bpw,
            expected_cosine=p.affine6_cosine, fallback_head=Head.EXACT,
        )

    def _route_edge_balanced(self, p: TensorProfile) -> TensorPlan:
        if _is_exact_tensor(p):
            return TensorPlan(
                tensor_name=p.tensor_name, head=Head.EXACT,
                reason=["exact_tensor_type"], bpw=Head.EXACT.bpw,
                expected_cosine=1.0, fallback_head=Head.EXACT,
            )

        # Early attention or high kurtosis -> affine6
        reasons_6 = []
        if _is_early_attention(p):
            reasons_6.append("early_attention")
        if p.kurtosis > self.kurtosis_threshold and p.tensor_type in ATTENTION_FRAGILE:
            reasons_6.append("high_kurtosis_attention")
        if reasons_6:
            return TensorPlan(
                tensor_name=p.tensor_name, head=Head.AFFINE6,
                reason=reasons_6, bpw=Head.AFFINE6.bpw,
                expected_cosine=p.affine6_cosine, fallback_head=Head.EXACT,
            )

        # If affine5 passes cosine gate -> affine5
        if p.affine5_cosine >= self.cosine_gate_5:
            return TensorPlan(
                tensor_name=p.tensor_name, head=Head.AFFINE5,
                reason=["cosine_pass", "edge_default"],
                bpw=Head.AFFINE5.bpw, expected_cosine=p.affine5_cosine,
                fallback_head=Head.AFFINE6,
            )

        # Default -> affine6
        return TensorPlan(
            tensor_name=p.tensor_name, head=Head.AFFINE6,
            reason=["affine5_cosine_below_gate"], bpw=Head.AFFINE6.bpw,
            expected_cosine=p.affine6_cosine, fallback_head=Head.EXACT,
        )

    def _route_size_target(self, profiles: list[TensorProfile]) -> list[TensorPlan]:
        """Greedy: assign lowest viable bit to each tensor, respecting fragile."""
        plans = []
        for p in profiles:
            if _is_exact_tensor(p):
                plans.append(TensorPlan(
                    tensor_name=p.tensor_name, head=Head.EXACT,
                    reason=["exact_tensor_type"], bpw=Head.EXACT.bpw,
                    expected_cosine=1.0, fallback_head=Head.EXACT,
                ))
                continue

            # Try affine5 first
            if p.affine5_cosine >= self.cosine_gate_5:
                plans.append(TensorPlan(
                    tensor_name=p.tensor_name, head=Head.AFFINE5,
                    reason=["size_target", "cosine_pass"],
                    bpw=Head.AFFINE5.bpw, expected_cosine=p.affine5_cosine,
                    fallback_head=Head.AFFINE6,
                ))
            else:
                # Cosine below gate — fall back to affine6.
                # Sidecar does NOT help here: uniform affine error is
                # distributed, not sparse outliers. See sidecar probe receipt.
                plans.append(TensorPlan(
                    tensor_name=p.tensor_name, head=Head.AFFINE6,
                    reason=["size_target", "fragile_fallback"],
                    bpw=Head.AFFINE6.bpw, expected_cosine=p.affine6_cosine,
                    fallback_head=Head.EXACT,
                ))

        return plans

    def _route_experimental(self, p: TensorProfile) -> TensorPlan:
        if _is_exact_tensor(p):
            return TensorPlan(
                tensor_name=p.tensor_name, head=Head.EXACT,
                reason=["exact_tensor_type"], bpw=Head.EXACT.bpw,
                expected_cosine=1.0, fallback_head=Head.EXACT,
            )

        # Try affine3 (very rare)
        if p.affine3_cosine >= self.cosine_gate_3 and p.kurtosis < 5.0:
            return TensorPlan(
                tensor_name=p.tensor_name, head=Head.AFFINE3,
                reason=["experimental_lowbit", "affine3_gate_pass"],
                bpw=Head.AFFINE3.bpw, expected_cosine=p.affine3_cosine,
                fallback_head=Head.AFFINE5,
            )

        # Try affine4
        if p.affine4_cosine >= self.cosine_gate_4 and p.kurtosis < self.kurtosis_threshold:
            return TensorPlan(
                tensor_name=p.tensor_name, head=Head.AFFINE4,
                reason=["experimental_lowbit", "affine4_gate_pass"],
                bpw=Head.AFFINE4.bpw, expected_cosine=p.affine4_cosine,
                fallback_head=Head.AFFINE5,
            )

        # High kurtosis -> affine6 (no sidecar — affine error is distributed)
        if p.kurtosis > self.kurtosis_threshold:
            return TensorPlan(
                tensor_name=p.tensor_name, head=Head.AFFINE6,
                reason=["high_kurtosis"],
                bpw=Head.AFFINE6.bpw,
                expected_cosine=p.affine6_cosine,
                fallback_head=Head.EXACT,
            )

        # Default -> affine5
        if p.affine5_cosine >= self.cosine_gate_5:
            return TensorPlan(
                tensor_name=p.tensor_name, head=Head.AFFINE5,
                reason=["experimental_default"], bpw=Head.AFFINE5.bpw,
                expected_cosine=p.affine5_cosine, fallback_head=Head.AFFINE6,
            )

        return TensorPlan(
            tensor_name=p.tensor_name, head=Head.AFFINE6,
            reason=["experimental_fallback"], bpw=Head.AFFINE6.bpw,
            expected_cosine=p.affine6_cosine, fallback_head=Head.EXACT,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Profile builder (from mixed-lowbit probe receipt data)
# ═══════════════════════════════════════════════════════════════════════════

def profiles_from_probe_receipt(receipt_path: Path) -> list[TensorProfile]:
    """Load TensorProfiles from a mixed-lowbit probe receipt JSON.

    Expects the format produced by bench_hxq_mixed_lowbit_probe.py:
    phase1.tensor_records[].{name, layer, type, shape, kurtosis, std, strategies}
    """
    data = json.loads(receipt_path.read_text())
    records = data.get("phase1", {}).get("tensor_records", [])

    profiles = []
    for rec in records:
        strategies = rec.get("strategies", {})
        profiles.append(TensorProfile(
            tensor_name=rec["name"],
            shape=tuple(rec["shape"]),
            layer_index=rec.get("layer", 0),
            tensor_type=rec.get("type", "unknown"),
            n_params=rec.get("n_params", 0),
            kurtosis=rec.get("kurtosis", 0.0),
            std=rec.get("std", 0.0),
            affine6_cosine=strategies.get("affine6_g128", {}).get("cosine", 1.0),
            affine5_cosine=strategies.get("affine5_g128", {}).get("cosine", 1.0),
            affine4_cosine=strategies.get("affine4_g128", {}).get("cosine", 0.0),
            affine3_cosine=strategies.get("affine3_g128", {}).get("cosine", 0.0),
            max_abs_error=strategies.get("affine6_g128", {}).get("max_abs_error", 0.0),
            mean_abs_error=strategies.get("affine6_g128", {}).get("mean_abs_error", 0.0),
        ))

    return profiles
