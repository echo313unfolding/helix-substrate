"""
Structural Entropy (Se) Estimator for Tensor Routing

Computes Se = H x U x D for any tensor, producing a normalized 0-1 score
that captures how "complex" a tensor is from a compute-routing perspective.

Formula:
    Se = H x U x D (all normalized 0-1)

Where:
    H = (1 - energy_at_10pct)     -- Entropy: spread-out spectrum = high H
    U = (1 - neighbor_coherence)  -- Unstructuredness: no local adjacency = high U
    D = sqrt(rank_ratio)          -- Depth: high effective rank = high D

2D Routing Policy:
    Zone 1: Se < 0.30, C_struct >= 0.30 -> CPU (simple + organized)
    Zone 2: 0.30 <= Se < 0.70          -> GPU (parallelizable)
    Zone 3: Se >= 0.70, C_struct < 0.30 -> QPU (complex + unstructured)
    Zone 4: Se >= 0.70, C_struct >= 0.30 -> GPU (complex but structured)
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

# Routing thresholds
CPU_MAX_SE = 0.30
GPU_MAX_SE = 0.70


def _quick_coherence(W: np.ndarray, n_samples: int = 50) -> float:
    """Neighbor coherence via sampled row correlations."""
    m, n = W.shape
    if m <= 1:
        return 0.0

    correlations = []
    rng = np.random.RandomState(42)
    n_row = min(n_samples, m - 1)
    for i in rng.choice(m - 1, n_row, replace=False):
        r1 = W[i].flatten()
        r2 = W[i + 1].flatten()
        denom = np.std(r1) * np.std(r2)
        if denom > 1e-8:
            corr = np.corrcoef(r1, r2)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

    return float(np.mean(correlations)) if correlations else 0.0


def compute_tensor_se(
    tensor: np.ndarray,
    name: str = "tensor",
    gpu_vram_mb: float = 4000.0,
) -> Dict[str, Any]:
    """
    Compute Se = H x U x D for a tensor with 2D routing and gate overrides.

    Args:
        tensor: NumPy array (2D preferred, will reshape if needed)
        name: Tensor name for debugging
        gpu_vram_mb: GPU VRAM in MB (for size-based gating)

    Returns:
        Dict with Se score, components, routing decision, and gate info.
    """
    if tensor is None or tensor.size == 0:
        return {
            "Se": 0.0, "H": 0.0, "U": 0.0, "D": 0.0, "D_raw": 0.0,
            "C_struct": 0.0,
            "components": {
                "energy_at_10pct": 1.0, "neighbor_coherence": 1.0,
                "row_coherence": 0.0, "col_coherence": 0.0,
                "coherence_asymmetry": 0.0, "rank_ratio": 0.0,
            },
            "gates": {
                "condition_number": 1.0, "sufficient_k": True,
                "sparsity_ratio": 0.0, "sparsity_structure": "dense",
            },
            "gate_override": None, "routing_hint": "cpu",
            "routing_zone": 1, "size_mb": 0.0, "size_gated": False,
        }

    size_mb = tensor.size * tensor.itemsize / 1e6
    size_gated = size_mb > gpu_vram_mb * 0.8

    # Ensure 2D
    if tensor.ndim == 1:
        tensor_2d = tensor.reshape(-1, 1)
    elif tensor.ndim > 2:
        tensor_2d = tensor.reshape(tensor.shape[0], -1)
    else:
        tensor_2d = tensor

    m, n = tensor_2d.shape
    max_rank = min(m, n)
    tensor_f32 = tensor_2d.astype(np.float32)

    # SVD analysis
    if max_rank <= 512:
        _, S, _ = np.linalg.svd(tensor_f32, full_matrices=False)
        frob_sq = np.sum(S ** 2)
        idx_10pct = max(1, len(S) // 10)
        energy_at_10pct = float(np.sum(S[:idx_10pct] ** 2) / frob_sq) if frob_sq > 0 else 1.0

        cumulative = np.cumsum(S ** 2) / frob_sq if frob_sq > 0 else np.ones_like(S)
        rank_estimate = int(np.searchsorted(cumulative, 0.95) + 1)
        rank_ratio = rank_estimate / max_rank

        sigma_min = S[-1] if S[-1] > 1e-12 else 1e-12
        condition_number = float(min(S[0] / sigma_min, 1e12))
        sufficient_k = True
    else:
        energy_at_10pct = 0.5
        rank_ratio = 0.5
        condition_number = 1.0
        sufficient_k = False

    neighbor_coherence = _quick_coherence(tensor_f32)
    row_coherence = neighbor_coherence
    col_coherence = neighbor_coherence
    coherence_asymmetry = 0.0
    max_coherence = neighbor_coherence

    sparsity_ratio = float(np.mean(np.abs(tensor_f32) < 1e-6))
    sparsity_structure = "dense" if sparsity_ratio < 0.1 else "random"

    # Compute H, U, D
    H = float(max(0.0, min(1.0, 1.0 - energy_at_10pct)))
    U = float(max(0.0, min(1.0, 1.0 - neighbor_coherence)))
    D_raw = rank_ratio
    D = float(max(0.0, min(1.0, np.sqrt(D_raw))))
    C_struct = float(max(0.0, min(1.0, max_coherence)))

    Se = H * U * D

    # Gate thresholds
    KAPPA_THRESHOLD = 100.0
    SPARSITY_THRESHOLD = 0.40
    STRUCT_THRESHOLD = 0.30

    gate_kappa_fail = condition_number > KAPPA_THRESHOLD
    gate_rank_unstable = not sufficient_k
    gate_sparse_structured = sparsity_ratio > SPARSITY_THRESHOLD and sparsity_structure == "block"

    gates = {
        "condition_number": float(condition_number),
        "sufficient_k": bool(sufficient_k),
        "sparsity_ratio": float(sparsity_ratio),
        "sparsity_structure": sparsity_structure,
    }

    gate_override = None
    if size_gated:
        routing_hint, routing_zone, gate_override = "cpu", 1, "size"
    elif gate_kappa_fail:
        routing_hint, routing_zone, gate_override = "gpu", 2, "kappa"
    elif gate_sparse_structured:
        routing_hint, routing_zone, gate_override = "cpu", 1, "sparsity"
    elif gate_rank_unstable:
        routing_hint, routing_zone, gate_override = "cpu", 1, "sufficient_k"
    else:
        if Se >= GPU_MAX_SE:
            if C_struct >= STRUCT_THRESHOLD:
                routing_hint, routing_zone = "gpu", 4
            else:
                routing_hint, routing_zone = "qpu", 3
        elif Se >= CPU_MAX_SE:
            routing_hint, routing_zone = "gpu", 2
        else:
            routing_hint, routing_zone = "cpu", 1

    return {
        "Se": float(Se),
        "H": H,
        "U": U,
        "D": D,
        "D_raw": float(D_raw),
        "C_struct": C_struct,
        "components": {
            "energy_at_10pct": float(energy_at_10pct),
            "neighbor_coherence": float(neighbor_coherence),
            "row_coherence": float(row_coherence),
            "col_coherence": float(col_coherence),
            "coherence_asymmetry": float(coherence_asymmetry),
            "rank_ratio": float(rank_ratio),
        },
        "gates": gates,
        "gate_override": gate_override,
        "routing_hint": routing_hint,
        "routing_zone": routing_zone,
        "size_mb": float(size_mb),
        "size_gated": bool(size_gated),
    }


def compute_routing_decision(
    tensor: np.ndarray,
    state: Optional[dict] = None,
    gpu_vram_mb: float = 4000.0,
    name: str = "tensor",
) -> Dict[str, Any]:
    """Full routing decision with optional state override."""
    result = compute_tensor_se(tensor, name=name, gpu_vram_mb=gpu_vram_mb)

    override_applied = False
    if state and isinstance(state, dict):
        if "force_backend" in state:
            result["routing_hint"] = state["force_backend"]
            override_applied = True

    result["override_applied"] = override_applied
    return result
