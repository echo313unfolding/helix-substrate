"""
kurtosis_gate.py — Runtime SVD correction gate based on activation kurtosis.

For SVD-sensitive (Class 2) HelixLinear modules, kurtosis of the input
activation correlates with SVD need:
  high kurtosis → heavy tails / outliers → VQ struggles → SVD correction needed
  low kurtosis  → compact distribution → VQ handles it → skip SVD, save compute

Proven: WO-MULTITASK-DRIFT-01 (v3)
  q_proj:    r=0.6455, Pareto optimal (17% compute saving, 85% less degradation)
  gate_proj: r=0.8118, Pareto optimal (30% compute saving, 74% less degradation)

This module is used by HelixLinear.attach_kurtosis_gate() to enable
per-forward gating on Class 2 modules. Class 1 and non-SVD modules
have zero overhead (no gate attached).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch


def compute_kurtosis(x: torch.Tensor) -> float:
    """Excess kurtosis of a tensor. One pass, no SVD, cheap."""
    flat = x.detach().float().reshape(-1)
    if flat.numel() < 4:
        return 0.0
    m = flat.mean()
    s = flat.std()
    if s < 1e-8:
        return 0.0
    return ((flat - m) / s).pow(4).mean().item() - 3.0


@dataclass
class KurtosisGate:
    """
    Hysteresis gate that decides SVD enable/skip based on activation kurtosis.

    Attach to a HelixLinear module via module.attach_kurtosis_gate().
    The gate is queried on every forward() call for Class 2 modules.

    Attributes:
        threshold: Kurtosis above this → SVD correction needed.
                   Must be calibrated per-module from real traces.
        switch_to_svd_after: Consecutive high-kurtosis windows before enabling SVD.
        recover_to_skip_after: Consecutive low-kurtosis windows before disabling SVD.
        ema_decay: Smoothing factor for kurtosis EMA (lower = more responsive).
    """
    threshold: float
    switch_to_svd_after: int = 2
    recover_to_skip_after: int = 2
    ema_decay: float = 0.5
    # Amortization: compute kurtosis every `check_interval` forward calls.
    # Between checks, reuse the cached decision. Proven: ci=2 achieves 100%
    # agreement with ci=1 and gate overhead (1.1ms) < SVD marginal (3.1ms).
    check_interval: int = 2
    # Internal state
    _ema_kurtosis: float = 0.0
    _high_streak: int = 0
    _low_streak: int = 0
    _initialized: bool = False
    _enable_svd: bool = False
    _total_updates: int = 0
    _total_forwards: int = 0
    _total_svd_enables: int = 0
    _total_switches: int = 0
    # Bounded trace (last N entries for receipts)
    _trace_max: int = 100
    _trace: List[Dict[str, Any]] = field(default_factory=list)

    def step(self, x: torch.Tensor) -> bool:
        """
        Amortized gate step. Call on every forward().

        Computes kurtosis on the first call (cold-start) and then every
        `check_interval` calls thereafter. Between checks, returns the
        cached decision (zero cost).

        Args:
            x: Input activation tensor.

        Returns:
            True if SVD should be enabled for this forward pass.
        """
        self._total_forwards += 1
        if not self._initialized or self._total_forwards % self.check_interval == 0:
            kurt = compute_kurtosis(x)
            return self.update(kurt)
        # Between checks: count SVD enables for accurate stats
        if self._enable_svd:
            self._total_svd_enables += 1
        return self._enable_svd

    def update(self, raw_kurtosis: float) -> bool:
        """
        Feed a new kurtosis observation. Returns True if SVD should be enabled.

        Called by step() every check_interval forwards.
        Can also be called directly for non-amortized use (benchmarks).
        """
        if not self._initialized:
            self._ema_kurtosis = raw_kurtosis
            self._initialized = True
        else:
            self._ema_kurtosis = (self.ema_decay * self._ema_kurtosis
                                  + (1 - self.ema_decay) * raw_kurtosis)

        if self._ema_kurtosis >= self.threshold:
            self._high_streak += 1
            self._low_streak = 0
        else:
            self._low_streak += 1
            self._high_streak = 0

        old = self._enable_svd
        switched = False

        if not self._enable_svd and self._high_streak >= self.switch_to_svd_after:
            self._enable_svd = True
            switched = True

        elif self._enable_svd and self._low_streak >= self.recover_to_skip_after:
            self._enable_svd = False
            switched = True

        self._total_updates += 1
        if self._enable_svd:
            self._total_svd_enables += 1
        if switched:
            self._total_switches += 1

        # Bounded trace
        if len(self._trace) < self._trace_max:
            self._trace.append({
                "raw": round(raw_kurtosis, 1),
                "ema": round(self._ema_kurtosis, 1),
                "svd": self._enable_svd,
                "sw": switched,
            })

        return self._enable_svd

    def summary(self) -> Dict[str, Any]:
        """Receipt-ready summary of gate state."""
        return {
            "threshold": self.threshold,
            "hysteresis": {
                "switch_to_svd_after": self.switch_to_svd_after,
                "recover_to_skip_after": self.recover_to_skip_after,
            },
            "ema_decay": self.ema_decay,
            "check_interval": self.check_interval,
            "total_forwards": self._total_forwards,
            "total_updates": self._total_updates,
            "total_svd_enables": self._total_svd_enables,
            "total_switches": self._total_switches,
            "svd_rate": round(self._total_svd_enables / max(self._total_forwards, 1), 4),
            "current_enable_svd": self._enable_svd,
            "current_ema_kurtosis": round(self._ema_kurtosis, 1),
        }

    def trace(self) -> List[Dict[str, Any]]:
        """Return bounded trace for receipt logging."""
        return list(self._trace)

    def reset(self) -> None:
        """Reset gate state (e.g., between benchmark runs)."""
        self._ema_kurtosis = 0.0
        self._high_streak = 0
        self._low_streak = 0
        self._initialized = False
        self._enable_svd = False
        self._total_updates = 0
        self._total_forwards = 0
        self._total_svd_enables = 0
        self._total_switches = 0
        self._trace = []


# ============================================================================
# Class 2 module registry — which modules need kurtosis gating
# ============================================================================

# Calibrated thresholds from WO-MULTITASK-DRIFT-01 v3 receipt.
# Per-module because kurtosis distributions differ dramatically
# (q_proj: 745-1382, gate_proj: 37-472).
CLASS2_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    # TinyLlama layer 0
    "model.layers.0.self_attn.q_proj.weight": {
        "threshold": 1002.0,
        "correlation": 0.6455,
        "compute_saving_pct": 17.2,
    },
    "model.layers.0.mlp.gate_proj.weight": {
        "threshold": 59.9,
        "correlation": 0.8118,
        "compute_saving_pct": 29.7,
    },
}


def attach_kurtosis_gates(model, thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
                          hysteresis_n: int = 2, ema_decay: float = 0.5) -> Dict[str, Any]:
    """
    Attach kurtosis gates to all Class 2 HelixLinear modules in a model.

    Only modules listed in thresholds (or CLASS2_THRESHOLDS) get a gate.
    All other modules have zero overhead.

    Args:
        model: nn.Module containing HelixLinear modules
        thresholds: {tensor_name: {"threshold": float}} or None for defaults
        hysteresis_n: Consecutive windows for switching
        ema_decay: Kurtosis EMA decay

    Returns:
        Dict with counts of modules affected
    """
    if thresholds is None:
        thresholds = CLASS2_THRESHOLDS

    stats = {"attached": 0, "skipped_no_svd": 0, "skipped_class1": 0}

    for name, module in model.named_modules():
        if not hasattr(module, "has_svd") or not hasattr(module, "tensor_name"):
            continue

        tensor_name = module.tensor_name
        if tensor_name in thresholds:
            if not module.has_svd:
                stats["skipped_no_svd"] += 1
                continue

            cfg = thresholds[tensor_name]
            gate = KurtosisGate(
                threshold=cfg["threshold"],
                switch_to_svd_after=hysteresis_n,
                recover_to_skip_after=hysteresis_n,
                ema_decay=ema_decay,
            )
            module._kurtosis_gate = gate
            stats["attached"] += 1
        else:
            stats["skipped_class1"] += 1

    return stats


def collect_gate_summaries(model) -> Dict[str, Dict[str, Any]]:
    """Collect gate summaries from all gated modules for receipt logging."""
    summaries = {}
    for name, module in model.named_modules():
        gate = getattr(module, "_kurtosis_gate", None)
        if gate is not None:
            summaries[getattr(module, "tensor_name", name)] = gate.summary()
    return summaries
