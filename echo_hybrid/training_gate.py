"""
MorphSAT Training Gate — FSA enforcement for born-compressed training.

Hard constraint gate for training state transitions. Catches codebook collapse,
loss spikes, and dead centroids before they waste compute on rented GPUs.

States: INIT → TRAINING ↔ RECOMPRESSING → HALTED
Guards: codebook utilization, loss spike ratio, dead centroid fraction

Usage:
    from echo_hybrid.training_gate import TrainingGate

    gate = TrainingGate(util_floor=0.8, spike_ceil=2.0)
    gate.step(TrainEvent.COMPRESS_DONE, signals={"codebook_utilization": 0.95})
    # ... in training loop:
    state, legal, action = gate.step(TrainEvent.STEP_DONE, signals={...})
    if not legal:
        # HALTED — save receipt, stop training

Work Orders: WO-ECHO-HYBRID-07
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Training FSA
# ---------------------------------------------------------------------------

class TrainState(IntEnum):
    INIT = 0           # Before initial compression
    TRAINING = 1       # Running training steps
    RECOMPRESSING = 2  # Refreshing codebooks from shadow weights
    HALTED = 3         # Guard failed — training stopped


class TrainEvent(IntEnum):
    COMPRESS_DONE = 0     # Initial codebook compression complete
    STEP_DONE = 1         # Training step finished (includes vstep if any)
    RECOMPRESS_START = 2  # Schedule says recompress now
    RECOMPRESS_DONE = 3   # Recompression finished, guards evaluated
    GUARD_FAIL = 4        # A guard condition failed


N_STATES = len(TrainState)
N_EVENTS = len(TrainEvent)

STATE_NAMES = [s.name for s in TrainState]
EVENT_NAMES = [e.name for e in TrainEvent]

# Transition table: T[state, event] → next_state (-1 = illegal)
TRANSITION_TABLE = np.full((N_STATES, N_EVENTS), -1, dtype=np.int32)

# Legal transitions
TRANSITION_TABLE[TrainState.INIT,          TrainEvent.COMPRESS_DONE]    = TrainState.TRAINING
TRANSITION_TABLE[TrainState.TRAINING,      TrainEvent.STEP_DONE]       = TrainState.TRAINING
TRANSITION_TABLE[TrainState.TRAINING,      TrainEvent.RECOMPRESS_START] = TrainState.RECOMPRESSING
TRANSITION_TABLE[TrainState.RECOMPRESSING, TrainEvent.RECOMPRESS_DONE] = TrainState.TRAINING
TRANSITION_TABLE[TrainState.RECOMPRESSING, TrainEvent.GUARD_FAIL]      = TrainState.HALTED
TRANSITION_TABLE[TrainState.TRAINING,      TrainEvent.GUARD_FAIL]      = TrainState.HALTED
TRANSITION_TABLE[TrainState.INIT,          TrainEvent.GUARD_FAIL]      = TrainState.HALTED

N_ILLEGAL = int((TRANSITION_TABLE == -1).sum())
N_LEGAL = int((TRANSITION_TABLE >= 0).sum())


# ---------------------------------------------------------------------------
# Guard conditions
# ---------------------------------------------------------------------------

class GuardResult:
    """Result of evaluating guard conditions on training signals."""
    __slots__ = ("passed", "violations")

    def __init__(self):
        self.passed = True
        self.violations: List[str] = []

    def fail(self, reason: str):
        self.passed = False
        self.violations.append(reason)


def evaluate_guards(
    signals: Dict[str, float],
    util_floor: float = 0.8,
    spike_ceil: float = 2.0,
    dead_ceil: float = 0.3,
) -> GuardResult:
    """Evaluate guard conditions on training signals.

    Guards:
        1. codebook_utilization > util_floor — catches centroid collapse
        2. loss_spike_ratio < spike_ceil — catches catastrophic recompression
        3. dead_centroid_frac < dead_ceil — catches codebook underuse

    Args:
        signals: Dict with keys like "codebook_utilization", "loss_spike_ratio",
                 "dead_centroid_frac". Missing keys are skipped (no guard applied).
        util_floor: Minimum codebook utilization (fraction of centroids used).
        spike_ceil: Maximum loss ratio (current_loss / recent_avg_loss).
        dead_ceil: Maximum fraction of dead centroids (assigned to 0 weights).
    """
    result = GuardResult()

    util = signals.get("codebook_utilization")
    if util is not None and util < util_floor:
        result.fail(
            f"codebook_utilization={util:.3f} < floor={util_floor:.3f} — "
            f"centroid collapse detected"
        )

    spike = signals.get("loss_spike_ratio")
    if spike is not None and spike > spike_ceil:
        result.fail(
            f"loss_spike_ratio={spike:.3f} > ceil={spike_ceil:.3f} — "
            f"catastrophic loss spike after recompression"
        )

    dead = signals.get("dead_centroid_frac")
    if dead is not None and dead > dead_ceil:
        result.fail(
            f"dead_centroid_frac={dead:.3f} > ceil={dead_ceil:.3f} — "
            f"too many unused centroids"
        )

    return result


# ---------------------------------------------------------------------------
# Training Gate
# ---------------------------------------------------------------------------

class TrainingGate:
    """Hard FSA enforcement gate for born-compressed training transitions.

    Same pattern as MorphSATGate but for training dynamics instead of
    task lifecycle. Gates on codebook health signals at every recompression
    and optionally at every vstep.
    """

    def __init__(
        self,
        util_floor: float = 0.8,
        spike_ceil: float = 2.0,
        dead_ceil: float = 0.3,
        transition_table: Optional[np.ndarray] = None,
    ):
        self.T = (transition_table if transition_table is not None
                  else TRANSITION_TABLE).copy()
        self.state = TrainState.INIT
        self.util_floor = util_floor
        self.spike_ceil = spike_ceil
        self.dead_ceil = dead_ceil
        self.history: List[dict] = []
        self.illegal_caught = 0
        self.guard_failures = 0
        self.total_transitions = 0

    def step(
        self,
        event: TrainEvent,
        signals: Optional[Dict[str, float]] = None,
    ) -> Tuple[TrainState, bool, str]:
        """Attempt a state transition with optional guard evaluation.

        Args:
            event: The training event that occurred.
            signals: Optional dict of training signals for guard evaluation.
                     Guards are only checked on RECOMPRESS_DONE and STEP_DONE events.

        Returns:
            (new_state, was_legal, action_taken)

            action_taken is one of:
                "ALLOWED" — legal transition, state updated
                "FSA_BLOCKED" — illegal per FSA, state held
                "GUARD_BLOCKED" — guard condition failed, state → HALTED
        """
        self.total_transitions += 1
        old_state = self.state

        # Guard evaluation on recompression completion or step completion
        if signals and event in (TrainEvent.RECOMPRESS_DONE, TrainEvent.STEP_DONE):
            guard = evaluate_guards(
                signals,
                util_floor=self.util_floor,
                spike_ceil=self.spike_ceil,
                dead_ceil=self.dead_ceil,
            )
            if not guard.passed:
                self.guard_failures += 1
                self.state = TrainState.HALTED
                entry = {
                    "from": STATE_NAMES[old_state],
                    "event": EVENT_NAMES[event],
                    "action": "GUARD_BLOCKED",
                    "to": "HALTED",
                    "violations": guard.violations,
                    "signals": signals,
                }
                self.history.append(entry)
                return self.state, False, "GUARD_BLOCKED"

        # FSA check
        next_state = self.T[old_state, event]
        if next_state == -1:
            self.illegal_caught += 1
            self.history.append({
                "from": STATE_NAMES[old_state],
                "event": EVENT_NAMES[event],
                "action": "FSA_BLOCKED",
                "to": STATE_NAMES[old_state],
            })
            return self.state, False, "FSA_BLOCKED"

        # Legal transition
        self.state = TrainState(next_state)
        entry = {
            "from": STATE_NAMES[old_state],
            "event": EVENT_NAMES[event],
            "action": "ALLOWED",
            "to": STATE_NAMES[self.state],
        }
        if signals:
            entry["signals"] = signals
        self.history.append(entry)
        return self.state, True, "ALLOWED"

    def to_receipt(self) -> dict:
        """Export gate state and history as a receipt-compatible dict."""
        return {
            "final_state": STATE_NAMES[self.state],
            "total_transitions": self.total_transitions,
            "illegal_caught": self.illegal_caught,
            "guard_failures": self.guard_failures,
            "guard_thresholds": {
                "util_floor": self.util_floor,
                "spike_ceil": self.spike_ceil,
                "dead_ceil": self.dead_ceil,
            },
            "history": self.history,
        }
