"""
Route inheritance — sustained-signal state machine for the AI OS.

Ports the Terminal 3 morphogenesis control law into routing:
  - Sustained evidence before route inheritance (not single-query)
  - Role earned from environment (query similarity), not assumed
  - Reset on regime change (model swap, domain shift, quality failure)
  - Neutral default (COLD = full recompute)

The state machine tracks a "route regime" fingerprint across queries.
When queries are similar enough for long enough, routing decisions
are inherited rather than recomputed from scratch.

Work Order: WO-INHERIT-01
Lineage: WO-TERMINAL-3-DIFF (morphogenesis control law) → here
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .query_classifier import ModelTarget
from .route_decision import RetrievalMode, RouteDecision


# ── State machine ──

class InheritanceState(Enum):
    """Route inheritance states (Terminal 3 control law).

    COLD: full recompute, no history.
    WARM_CANDIDATE: two similar queries in sequence, building confidence.
    INHERITED: route family is reused, only recompute drifting parts.
    UNSTABLE: mismatch detected, degrading confidence.
    RESET: hard boundary crossed, immediate return to COLD.
    """
    COLD = "cold"
    WARM_CANDIDATE = "warm_candidate"
    INHERITED = "inherited"
    UNSTABLE = "unstable"
    RESET = "reset"


# ── Regime fingerprint ──

@dataclass(frozen=True)
class RouteRegime:
    """Fingerprint of a routing decision for similarity comparison.

    Two queries are in the same "regime" when their route fingerprints
    are close. This is the morphogenesis gradient — not spatial distance,
    but semantic/route distance.
    """
    model_target: str          # "tinyllama" or "qwen_coder"
    retrieval_mode: str        # RetrievalMode value
    budget_class: str          # "normal", "capped", "graph"
    dominant_signal: str       # Se dominant: "code", "fact", "graph", "analytical", "neutral"
    se_bucket: str             # Se quantized: "low" (<0.15), "mid" (0.15-0.4), "high" (>0.4)

    @staticmethod
    def from_route(route: RouteDecision, se_dominant: str = "neutral",
                   se_value: float = 0.0) -> RouteRegime:
        """Build regime fingerprint from a frozen RouteDecision + Se signal."""
        if se_value < 0.15:
            se_bucket = "low"
        elif se_value < 0.40:
            se_bucket = "mid"
        else:
            se_bucket = "high"

        return RouteRegime(
            model_target=route.target_model.value,
            retrieval_mode=route.retrieval_mode.value,
            budget_class=route.budget_mode,
            dominant_signal=se_dominant,
            se_bucket=se_bucket,
        )

    def distance(self, other: RouteRegime) -> float:
        """Route gradient: weighted distance between two regimes.

        Returns 0.0 for identical regimes, up to 1.0 for maximally different.
        This is the Terminal 3 gradient — but in route space, not physical space.
        """
        score = 0.0
        total_weight = 0.0

        # Model target mismatch (heaviest — requires GPU swap)
        w = 0.35
        if self.model_target != other.model_target:
            score += w
        total_weight += w

        # Retrieval mode mismatch
        w = 0.25
        if self.retrieval_mode != other.retrieval_mode:
            score += w
        total_weight += w

        # Dominant signal mismatch
        w = 0.20
        if self.dominant_signal != other.dominant_signal:
            score += w
        total_weight += w

        # Se bucket mismatch
        w = 0.10
        if self.se_bucket != other.se_bucket:
            score += w
        total_weight += w

        # Budget class mismatch
        w = 0.10
        if self.budget_class != other.budget_class:
            score += w
        total_weight += w

        return score / total_weight if total_weight > 0 else 0.0


# ── Inherited route state ──

@dataclass
class InheritedRouteState:
    """Current inheritance state with diagnostics.

    This is the cell's "differentiation state" from Terminal 3,
    translated into route space.
    """
    state: InheritanceState = InheritanceState.COLD
    regime: Optional[RouteRegime] = None
    confidence: float = 0.0
    stable_ticks: int = 0        # consecutive low-gradient queries
    drift_score: float = 0.0     # last regime distance (the gradient)
    entered_at: Optional[float] = None   # timestamp when current state entered
    last_used_at: Optional[float] = None
    transition_reason: str = ""
    queries_inherited: int = 0    # total queries served via inheritance
    queries_total: int = 0

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "regime": {
                "model_target": self.regime.model_target,
                "retrieval_mode": self.regime.retrieval_mode,
                "budget_class": self.regime.budget_class,
                "dominant_signal": self.regime.dominant_signal,
                "se_bucket": self.regime.se_bucket,
            } if self.regime else None,
            "confidence": round(self.confidence, 3),
            "stable_ticks": self.stable_ticks,
            "drift_score": round(self.drift_score, 4),
            "transition_reason": self.transition_reason,
            "queries_inherited": self.queries_inherited,
            "queries_total": self.queries_total,
            "inheritance_rate": (
                round(self.queries_inherited / max(1, self.queries_total), 3)
            ),
        }


# ── Thresholds (Terminal 3 equivalents) ──

# Regime distance below this = "same regime" (low gradient)
REGIME_SIMILARITY_THRESHOLD = 0.20

# Consecutive similar queries before inheritance activates
# (DIFF_SUSTAIN_TICKS equivalent)
WARM_TICKS_REQUIRED = 2

# Consecutive similar queries in WARM before INHERITED
INHERIT_TICKS_REQUIRED = 3

# Regime distance above this = "hard drift" → RESET
HARD_DRIFT_THRESHOLD = 0.60

# Confidence decay per unstable tick
CONFIDENCE_DECAY = 0.3

# Confidence growth per stable tick
CONFIDENCE_GROWTH = 0.2

# Below this confidence in INHERITED state → downgrade to UNSTABLE
MIN_INHERITED_CONFIDENCE = 0.3


# ── State machine ──

class InheritanceStateMachine:
    """Route inheritance state machine (Terminal 3 control law).

    Tracks routing regime across queries. When queries are similar enough
    for long enough, inherits the route family instead of full recompute.

    Terminal 3 mapping:
      - COLD = generalist (no specialization)
      - WARM_CANDIDATE = gradient detected, building sustained signal
      - INHERITED = correction/compression_specialist (committed role)
      - UNSTABLE = gradient shifting, confidence dropping
      - RESET = daughter cell (neutral restart)

    The "gradient" is regime distance. The "sustained signal" is consecutive
    similar queries. The "differentiation" is route inheritance.
    """

    def __init__(self):
        self._state = InheritedRouteState()
        self._history: List[RouteRegime] = []  # recent regimes (bounded)
        self._reset_reasons: List[Dict[str, Any]] = []  # audit trail

    @property
    def state(self) -> InheritedRouteState:
        return self._state

    @property
    def current_state(self) -> InheritanceState:
        return self._state.state

    def suggest(self, new_regime: RouteRegime) -> Dict[str, Any]:
        """Suggest whether to inherit the previous route or recompute.

        Called BEFORE compute_route(). Returns a suggestion dict:
          - inherit: bool (True = skip full classification, reuse previous)
          - fields_to_reuse: list of route fields safe to inherit
          - fields_to_recompute: list of route fields that should be fresh
          - confidence: float (how sure we are)
          - reason: str

        IMPORTANT: This is a SUGGESTION. The caller decides whether to act on it.
        Budget, quality, and reset checks are NEVER skipped.
        """
        s = self._state
        now = time.time()

        if s.state == InheritanceState.COLD or s.regime is None:
            return {
                "inherit": False,
                "fields_to_reuse": [],
                "fields_to_recompute": ["all"],
                "confidence": 0.0,
                "reason": "cold start — no history",
                "state": s.state.value,
            }

        dist = s.regime.distance(new_regime)

        if s.state == InheritanceState.INHERITED and dist < REGIME_SIMILARITY_THRESHOLD:
            # Safe to inherit
            reuse = []
            recompute = []

            # Model target: safe if same regime
            if new_regime.model_target == s.regime.model_target:
                reuse.append("model_target")
            else:
                recompute.append("model_target")

            # Retrieval mode: safe if same
            if new_regime.retrieval_mode == s.regime.retrieval_mode:
                reuse.append("retrieval_mode")
            else:
                recompute.append("retrieval_mode")

            # Budget: safe if same
            if new_regime.budget_class == s.regime.budget_class:
                reuse.append("budget_class")
            else:
                recompute.append("budget_class")

            return {
                "inherit": True,
                "fields_to_reuse": reuse,
                "fields_to_recompute": recompute if recompute else ["none"],
                "confidence": s.confidence,
                "reason": f"inherited — {s.stable_ticks} stable ticks, drift={dist:.3f}",
                "state": s.state.value,
            }

        if s.state == InheritanceState.WARM_CANDIDATE:
            return {
                "inherit": False,
                "fields_to_reuse": [],
                "fields_to_recompute": ["all"],
                "confidence": s.confidence,
                "reason": f"warming — {s.stable_ticks}/{INHERIT_TICKS_REQUIRED} ticks",
                "state": s.state.value,
            }

        # UNSTABLE or high drift
        return {
            "inherit": False,
            "fields_to_reuse": [],
            "fields_to_recompute": ["all"],
            "confidence": s.confidence,
            "reason": f"not stable — state={s.state.value}, drift={dist:.3f}",
            "state": s.state.value,
        }

    def observe(
        self,
        regime: RouteRegime,
        quality_passed: bool = True,
        cache_hit: bool = False,
        was_inherited: bool = False,
    ) -> Dict[str, Any]:
        """Observe the outcome of a query and update state.

        Called AFTER a query completes. Drives the state machine transitions.

        Args:
            regime: The route regime of the completed query.
            quality_passed: Did the output pass quality checks?
            cache_hit: Was the response served from cache?
            was_inherited: Did we actually use inheritance for this query?

        Returns:
            Transition info dict for receipt/diagnostics.
        """
        s = self._state
        now = time.time()
        s.queries_total += 1
        if was_inherited:
            s.queries_inherited += 1

        old_state = s.state
        old_regime = s.regime

        # Compute gradient (regime distance)
        if s.regime is not None:
            dist = s.regime.distance(regime)
        else:
            dist = 1.0  # first query = maximum novelty

        s.drift_score = dist
        s.last_used_at = now

        # ── Hard reset triggers (Terminal 3: daughter resets to generalist) ──
        if not quality_passed:
            self._transition(InheritanceState.RESET, "quality_failure")
            self._transition(InheritanceState.COLD, "auto_reset_after_quality_failure")
            s.regime = regime
            s.confidence = 0.0
            s.stable_ticks = 0
            return self._transition_receipt(old_state, old_regime, dist)

        if dist >= HARD_DRIFT_THRESHOLD:
            self._transition(InheritanceState.RESET, f"hard_drift={dist:.3f}")
            self._transition(InheritanceState.COLD, "auto_reset_after_hard_drift")
            s.regime = regime
            s.confidence = 0.0
            s.stable_ticks = 0
            return self._transition_receipt(old_state, old_regime, dist)

        # ── State transitions (Terminal 3: sustained gradient → role change) ──
        low_gradient = dist < REGIME_SIMILARITY_THRESHOLD

        if s.state == InheritanceState.COLD:
            # First query establishes regime
            s.regime = regime
            s.stable_ticks = 1
            s.confidence = 0.1
            s.entered_at = now
            if low_gradient and old_regime is not None:
                self._transition(InheritanceState.WARM_CANDIDATE,
                                 f"first_similar_query, drift={dist:.3f}")
            # else: stay COLD, regime updated

        elif s.state == InheritanceState.WARM_CANDIDATE:
            if low_gradient:
                s.stable_ticks += 1
                s.confidence = min(1.0, s.confidence + CONFIDENCE_GROWTH)
                if s.stable_ticks >= INHERIT_TICKS_REQUIRED:
                    self._transition(InheritanceState.INHERITED,
                                     f"sustained_{s.stable_ticks}_ticks")
            else:
                # Gradient rose — back to COLD
                s.stable_ticks = 0
                s.confidence = 0.0
                s.regime = regime
                self._transition(InheritanceState.COLD,
                                 f"warm_interrupted, drift={dist:.3f}")

        elif s.state == InheritanceState.INHERITED:
            if low_gradient:
                s.stable_ticks += 1
                s.confidence = min(1.0, s.confidence + CONFIDENCE_GROWTH * 0.5)
            else:
                # Drift detected — don't immediately reset, degrade
                s.confidence = max(0.0, s.confidence - CONFIDENCE_DECAY)
                s.stable_ticks = 0
                if s.confidence < MIN_INHERITED_CONFIDENCE:
                    s.regime = regime
                    self._transition(InheritanceState.UNSTABLE,
                                     f"confidence_below_{MIN_INHERITED_CONFIDENCE}")

        elif s.state == InheritanceState.UNSTABLE:
            if low_gradient:
                s.stable_ticks += 1
                s.confidence = min(1.0, s.confidence + CONFIDENCE_GROWTH)
                if s.stable_ticks >= WARM_TICKS_REQUIRED:
                    self._transition(InheritanceState.WARM_CANDIDATE,
                                     "recovery_from_unstable")
            else:
                s.stable_ticks = 0
                s.confidence = max(0.0, s.confidence - CONFIDENCE_DECAY)
                s.regime = regime
                if s.confidence <= 0.0:
                    self._transition(InheritanceState.COLD,
                                     "unstable_confidence_exhausted")

        elif s.state == InheritanceState.RESET:
            # Should not stay in RESET (auto-transitions to COLD)
            s.regime = regime
            s.stable_ticks = 0
            s.confidence = 0.0
            self._transition(InheritanceState.COLD, "reset_to_cold")

        # Update regime to current (for INHERITED, keep the established one)
        if s.state != InheritanceState.INHERITED:
            s.regime = regime

        # Maintain bounded history
        self._history.append(regime)
        if len(self._history) > 20:
            self._history = self._history[-20:]

        return self._transition_receipt(old_state, old_regime, dist)

    def force_reset(self, reason: str) -> None:
        """Force immediate reset (model swap, user override, etc)."""
        self._transition(InheritanceState.RESET, f"forced: {reason}")
        self._transition(InheritanceState.COLD, "auto_reset")
        self._state.regime = None
        self._state.confidence = 0.0
        self._state.stable_ticks = 0

    def _transition(self, new_state: InheritanceState, reason: str) -> None:
        """Execute a state transition with audit trail."""
        old = self._state.state
        self._state.state = new_state
        self._state.transition_reason = reason
        if new_state != old:
            self._state.entered_at = time.time()
            self._reset_reasons.append({
                "from": old.value,
                "to": new_state.value,
                "reason": reason,
                "timestamp": time.time(),
            })
            # Keep bounded
            if len(self._reset_reasons) > 50:
                self._reset_reasons = self._reset_reasons[-50:]

    def _transition_receipt(
        self,
        old_state: InheritanceState,
        old_regime: Optional[RouteRegime],
        dist: float,
    ) -> Dict[str, Any]:
        """Build a receipt of the state transition for diagnostics."""
        s = self._state
        return {
            "old_state": old_state.value,
            "new_state": s.state.value,
            "changed": old_state != s.state,
            "drift_score": round(dist, 4),
            "confidence": round(s.confidence, 3),
            "stable_ticks": s.stable_ticks,
            "reason": s.transition_reason,
            "regime": asdict(s.regime) if s.regime else None,
        }

    def stats(self) -> Dict[str, Any]:
        """Summary statistics for diagnostics."""
        s = self._state
        return {
            "state": s.state.value,
            "confidence": round(s.confidence, 3),
            "stable_ticks": s.stable_ticks,
            "queries_total": s.queries_total,
            "queries_inherited": s.queries_inherited,
            "inheritance_rate": round(
                s.queries_inherited / max(1, s.queries_total), 3
            ),
            "recent_resets": len([
                r for r in self._reset_reasons[-10:]
                if r["to"] == "cold"
            ]),
        }
