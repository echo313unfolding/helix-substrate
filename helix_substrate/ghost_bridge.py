"""Ghost Bridge — compressed-domain pre-routing for Hydra Router.

Bridges Crystal Vault's Ghost signal (compressed-domain structural
features) into HXQ's Hydra Router (codec routing decisions).

Ghost reads encoded bytes WITHOUT decompression and extracts 4 features:
  te: transition_entropy (bigram entropy, normalized)
  tr: transition_rank (average row entropy × bigram coverage)
  mo: markov_order (bigram entropy / 2× unigram entropy)
  ac: index_autocorr (max of column and row autocorrelation)

Phase 0.17b proved: architecture-aware multi-feature model on these 4
features clears 53.8% of tensors at precision=0.955, recall=0.904.
Those tensors skip expensive codec probing.

The bridge does NOT decide final codec assignment. It decides:
  SKIP_PROBE → tensor is safe, route directly (saves probe time)
  PROBE_REQUIRED → tensor may be fragile, run full Hydra probe

Lineage:
  Phase 0.10: transition_entropy dominates (r=0.976)
  Phase 0.15: Ghost classifier 73.3% role (8.1x random)
  Phase 0.16: Ghost predicts execution R²=0.818
  Phase 0.17: Single-feature pre-route: 8.6% cleared (too low)
  Phase 0.17b: Arch-aware multi-feature: 53.8% cleared (PASS)
  Ghost Bridge: wires Phase 0.17b into Hydra production routing
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional
import json

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Ghost feature extraction (from encoded bytes, no decompression)
# ═══════════════════════════════════════════════════════════════════════════

def ghost_features_from_bytes(raw_bytes: bytes, shape: tuple = ()) -> dict:
    """Extract Ghost features from raw encoded bytes.

    Operates on uint8 VQ index bytes. Does NOT decompress.
    body_opened = false.

    Args:
        raw_bytes: Raw bytes from the encoded tensor body (.indices)
        shape: Tensor shape (used for row-wise autocorrelation)

    Returns:
        dict with keys: te, tr, mo, ac (all float, 0.0-1.0 range)
    """
    n = len(raw_bytes)
    if n < 64:
        return {"te": 0.0, "tr": 0.0, "mo": 0.0, "ac": 0.0}

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    # Sample for large tensors (deterministic)
    sample_size = min(n - 1, 500000)
    rng = np.random.default_rng(42)
    if sample_size < n - 1:
        idx = rng.choice(n - 1, sample_size, replace=False)
        idx.sort()
        pairs_a = arr[idx]
        pairs_b = arr[idx + 1]
    else:
        pairs_a = arr[:-1]
        pairs_b = arr[1:]

    pair_codes = pairs_a.astype(np.uint16) * 256 + pairs_b.astype(np.uint16)
    pair_counts = np.bincount(pair_codes, minlength=65536).astype(np.float64)
    total_pairs = len(pair_codes)

    # TE: transition entropy (normalized bigram entropy)
    pair_probs = pair_counts[pair_counts > 0] / total_pairs
    bigram_h = -float(np.sum(pair_probs * np.log2(pair_probs)))
    max_bigram_h = 2.0 * np.log2(256)
    te = bigram_h / max_bigram_h if max_bigram_h > 0 else 0.0

    # TR: transition rank (average row entropy × bigram coverage)
    transition_matrix = pair_counts.reshape(256, 256)
    row_sums = transition_matrix.sum(axis=1)
    row_entropies = []
    for i in range(256):
        if row_sums[i] > 10:
            rp = transition_matrix[i]
            rp = rp[rp > 0] / row_sums[i]
            row_entropies.append(-float(np.sum(rp * np.log2(rp))) / 8.0)
    nonzero_bigrams = int(np.sum(pair_counts > 0))
    bigram_coverage = nonzero_bigrams / 65536.0
    tr = float(np.mean(row_entropies)) * bigram_coverage if row_entropies else 0.0

    # MO: markov order (bigram entropy / 2× unigram entropy)
    byte_counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = byte_counts[byte_counts > 0] / n
    unigram_h = -float(np.sum(probs * np.log2(probs)))
    mo = bigram_h / (2.0 * unigram_h) if unigram_h > 0 else 1.0

    # AC: index autocorrelation (max of column-wise and row-wise)
    a = arr[:-1].astype(np.float64)
    b = arr[1:].astype(np.float64)
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    col_ac = float(np.mean((a - ma) * (b - mb)) / (sa * sb)) if sa > 1e-12 and sb > 1e-12 else 0.0

    row_ac = 0.0
    if len(shape) == 2 and shape[1] > 1:
        stride = shape[1]
        if n > stride:
            a2 = arr[:-stride].astype(np.float64)
            b2 = arr[stride:].astype(np.float64)
            ma2, mb2 = a2.mean(), b2.mean()
            sa2, sb2 = a2.std(), b2.std()
            if sa2 > 1e-12 and sb2 > 1e-12:
                row_ac = float(np.mean((a2 - ma2) * (b2 - mb2)) / (sa2 * sb2))

    ac = max(abs(col_ac), abs(row_ac))

    return {
        "te": round(te, 6),
        "tr": round(tr, 6),
        "mo": round(mo, 6),
        "ac": round(ac, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pre-route decision
# ═══════════════════════════════════════════════════════════════════════════

class PreRouteAction(str, Enum):
    SKIP_PROBE = "skip_probe"       # Ghost is confident → route directly
    PROBE_REQUIRED = "probe_required"  # Ghost uncertain → run full Hydra probe


@dataclass
class GhostDecision:
    """Pre-routing decision from Ghost features."""
    action: PreRouteAction
    confidence: float               # 0.0-1.0, how confident Ghost is
    ghost_features: dict             # te, tr, mo, ac
    arch: str                        # architecture used for decision
    fragility_score: float = 0.0     # predicted fragility (0=safe, 1=fragile)


@dataclass
class GhostPreRouteResult:
    """Summary of Ghost pre-routing across a model."""
    n_total: int = 0
    n_ghost_routed: int = 0          # tensors where Ghost said SKIP_PROBE
    n_probe_required: int = 0        # tensors where Ghost said PROBE_REQUIRED
    decisions: list = field(default_factory=list)

    @property
    def cleared_fraction(self) -> float:
        return self.n_ghost_routed / self.n_total if self.n_total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_ghost_routed": self.n_ghost_routed,
            "n_probe_required": self.n_probe_required,
            "cleared_fraction": round(self.cleared_fraction, 4),
            "decisions": [
                {
                    "tensor_name": d.get("tensor_name", ""),
                    "action": d["decision"].action.value,
                    "confidence": round(d["decision"].confidence, 4),
                    "fragility_score": round(d["decision"].fragility_score, 4),
                }
                for d in self.decisions
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Ghost pre-route engine
# ═══════════════════════════════════════════════════════════════════════════

class GhostPreRoute:
    """Architecture-aware pre-routing from Ghost features.

    Fits a simple linear model per architecture family. At routing time,
    predicts fragility score from Ghost features and applies a safety
    threshold. Tensors below the threshold are "safe" (SKIP_PROBE),
    tensors above are "fragile" (PROBE_REQUIRED).

    The model is conservative by design: when uncertain, it says
    PROBE_REQUIRED (not SKIP_PROBE). False negatives (fragile tensor
    marked safe) are the danger; false positives (safe tensor sent
    to probe) waste time but don't lose quality.
    """

    def __init__(self, arch_models: dict[str, dict] = None,
                 fragility_threshold: float = 0.5):
        """
        Args:
            arch_models: {arch_name: {"coefficients": [b0, b1, b2, b3, b4],
                                       "threshold": float}}
                         where prediction = b0 + b1*te + b2*tr + b3*mo + b4*ac
            fragility_threshold: default threshold if not per-arch
        """
        self.arch_models = arch_models or {}
        self.fragility_threshold = fragility_threshold

    def decide(self, ghost_features: dict, arch: str) -> GhostDecision:
        """Predict safe/fragile from Ghost features.

        Args:
            ghost_features: dict with te, tr, mo, ac
            arch: architecture family (e.g., "mamba", "transformer", "hybrid")

        Returns:
            GhostDecision with action and confidence
        """
        te = ghost_features.get("te", 0.0)
        tr = ghost_features.get("tr", 0.0)
        mo = ghost_features.get("mo", 0.0)
        ac = ghost_features.get("ac", 0.0)
        x = np.array([1.0, te, tr, mo, ac])

        model = self.arch_models.get(arch)
        if model is None:
            # Unknown architecture → always probe (conservative)
            return GhostDecision(
                action=PreRouteAction.PROBE_REQUIRED,
                confidence=0.0,
                ghost_features=ghost_features,
                arch=arch,
                fragility_score=0.5,
            )

        coeffs = np.array(model["coefficients"])
        threshold = model.get("threshold", self.fragility_threshold)

        raw_score = float(x @ coeffs)
        fragility = np.clip(raw_score, 0.0, 1.0)

        if fragility >= threshold:
            action = PreRouteAction.PROBE_REQUIRED
            confidence = fragility  # higher fragility → more confident it needs probing
        else:
            action = PreRouteAction.SKIP_PROBE
            # confidence = distance from threshold (how clearly safe)
            confidence = 1.0 - fragility / threshold if threshold > 0 else 1.0

        return GhostDecision(
            action=action,
            confidence=round(float(confidence), 4),
            ghost_features=ghost_features,
            arch=arch,
            fragility_score=round(float(fragility), 4),
        )

    @classmethod
    def calibrate(cls, data: list[dict],
                  min_precision_safe: float = 0.95,
                  min_recall_fragile: float = 0.90) -> "GhostPreRoute":
        """Fit per-architecture linear models from labeled data.

        Args:
            data: list of dicts with keys:
                  ghost (dict with te/tr/mo/ac), arch (str), fragile (bool)
            min_precision_safe: safety constraint
            min_recall_fragile: safety constraint

        Returns:
            Calibrated GhostPreRoute instance
        """
        archs = sorted(set(d["arch"] for d in data))
        arch_models = {}

        for arch in archs:
            arch_data = [d for d in data if d["arch"] == arch]
            if len(arch_data) < 10:
                continue

            X = np.array([[1.0, d["ghost"]["te"], d["ghost"]["tr"],
                           d["ghost"]["mo"], d["ghost"]["ac"]]
                          for d in arch_data])
            y = np.array([1.0 if d["fragile"] else 0.0 for d in arch_data])

            # Fit OLS on full data (not LOO — LOO was for evaluation only)
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue

            # Find optimal threshold meeting safety constraints
            predictions = np.clip(X @ beta, 0.0, 1.0)
            best_threshold = _find_safe_threshold(
                predictions, y.astype(int),
                min_precision_safe, min_recall_fragile,
            )

            if best_threshold is not None:
                arch_models[arch] = {
                    "coefficients": [round(float(b), 6) for b in beta],
                    "threshold": round(float(best_threshold), 6),
                    "n_train": len(arch_data),
                    "n_fragile": int(y.sum()),
                }

        return cls(arch_models=arch_models)

    def save(self, path: Path) -> None:
        """Save calibrated model to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": "ghost_bridge_v1",
            "fragility_threshold": self.fragility_threshold,
            "arch_models": self.arch_models,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GhostPreRoute":
        """Load calibrated model from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(
            arch_models=data.get("arch_models", {}),
            fragility_threshold=data.get("fragility_threshold", 0.5),
        )

    def to_dict(self) -> dict:
        return {
            "version": "ghost_bridge_v1",
            "fragility_threshold": self.fragility_threshold,
            "arch_models": self.arch_models,
        }


def _find_safe_threshold(predictions: np.ndarray, labels: np.ndarray,
                         min_precision: float, min_recall: float) -> Optional[float]:
    """Find threshold that maximizes cleared fraction while meeting safety.

    predictions: predicted fragility scores (0-1)
    labels: 1 = fragile, 0 = safe
    """
    best_threshold = None
    best_cleared = 0.0

    for t in np.linspace(0.0, 1.0, 200):
        ghost_fragile = predictions >= t
        ghost_safe = ~ghost_fragile
        truly_fragile = labels == 1
        truly_safe = labels == 0

        n_ghost_safe = ghost_safe.sum()
        if n_ghost_safe == 0:
            continue

        precision_safe = (ghost_safe & truly_safe).sum() / n_ghost_safe
        n_truly_fragile = truly_fragile.sum()
        recall_fragile = (ghost_fragile & truly_fragile).sum() / n_truly_fragile if n_truly_fragile > 0 else 1.0
        cleared = n_ghost_safe / len(labels)

        if precision_safe >= min_precision and recall_fragile >= min_recall:
            if cleared > best_cleared:
                best_cleared = cleared
                best_threshold = float(t)

    return best_threshold
