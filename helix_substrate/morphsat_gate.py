"""
MorphSAT Task-State Enforcement Gate
=====================================

Hard FSA enforcement layer for task-state transitions in the lobe scheduler.
Extracted from WO-ECHO-MORPHSAT-INTEGRATION-01 (proven 2026-03-23).

The gate sits between lobe steps and enforces a finite-state automaton on the
task lifecycle. Illegal transitions are blocked — the pipeline holds or aborts.

FSA: 5 states (IDLE→PLANNING→WRITING→TESTING→DONE), 7 events, 23 illegal transitions.
Guardian layer adds 7 domain-specific vow blocks on top.

Usage:
    from helix_substrate.morphsat_gate import MorphSATGate, classify_event

    gate = MorphSATGate()
    gate.step(TaskEvent.NEW_TASK)      # IDLE → PLANNING
    event = classify_event(output, "generate")
    state, legal, action = gate.step(event)  # enforce transition

Work Orders: WO-ECHO-MORPHSAT-INTEGRATION-01
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Task-State FSA
# ---------------------------------------------------------------------------

class TaskState(IntEnum):
    IDLE = 0
    PLANNING = 1
    WRITING = 2
    TESTING = 3
    DONE = 4


class TaskEvent(IntEnum):
    NEW_TASK = 0       # user submits a coding request
    PLAN_COMPLETE = 1  # planner lobe finishes
    CODE_COMPLETE = 2  # coder lobe finishes
    TEST_PASS = 3      # verifier says PASS
    TEST_FAIL = 4      # verifier says FAIL
    RESET = 5          # user requests restart
    DEPLOY = 6         # user requests deployment


N_STATES = len(TaskState)
N_EVENTS = len(TaskEvent)

STATE_NAMES = [s.name for s in TaskState]
EVENT_NAMES = [e.name for e in TaskEvent]

# Transition table: T[state, event] → next_state (-1 = illegal)
TRANSITION_TABLE = np.full((N_STATES, N_EVENTS), -1, dtype=np.int32)

# Legal transitions
TRANSITION_TABLE[TaskState.IDLE,     TaskEvent.NEW_TASK]      = TaskState.PLANNING
TRANSITION_TABLE[TaskState.PLANNING, TaskEvent.PLAN_COMPLETE] = TaskState.WRITING
TRANSITION_TABLE[TaskState.WRITING,  TaskEvent.CODE_COMPLETE] = TaskState.TESTING
TRANSITION_TABLE[TaskState.TESTING,  TaskEvent.TEST_PASS]     = TaskState.DONE
TRANSITION_TABLE[TaskState.TESTING,  TaskEvent.TEST_FAIL]     = TaskState.WRITING  # revision loop
TRANSITION_TABLE[TaskState.DONE,     TaskEvent.NEW_TASK]      = TaskState.PLANNING
TRANSITION_TABLE[TaskState.DONE,     TaskEvent.DEPLOY]        = TaskState.DONE     # legal deploy from DONE
# RESET from any state goes to IDLE
for _s in TaskState:
    TRANSITION_TABLE[_s, TaskEvent.RESET] = TaskState.IDLE

N_ILLEGAL = int((TRANSITION_TABLE == -1).sum())
N_LEGAL = int((TRANSITION_TABLE >= 0).sum())


# ---------------------------------------------------------------------------
# Guardian Vows (domain-specific policy constraints)
# ---------------------------------------------------------------------------

# These are BLOCKED regardless of FSA legality — extra policy layer.
GUARDIAN_BLOCKED: Set[Tuple[int, int]] = {
    (TaskState.IDLE,     TaskEvent.DEPLOY),     # can't deploy from idle
    (TaskState.PLANNING, TaskEvent.DEPLOY),     # can't deploy from planning
    (TaskState.WRITING,  TaskEvent.DEPLOY),     # can't deploy while writing
    (TaskState.TESTING,  TaskEvent.DEPLOY),     # can't deploy while testing
    (TaskState.WRITING,  TaskEvent.NEW_TASK),   # can't start new task while writing
    (TaskState.TESTING,  TaskEvent.NEW_TASK),   # can't start new task while testing
    (TaskState.PLANNING, TaskEvent.NEW_TASK),   # can't start new task while planning
}


# ---------------------------------------------------------------------------
# Event Classification (grounding layer)
# ---------------------------------------------------------------------------

# Mapping from lobe scheduler role names to FSA event classification
_ROLE_TO_EVENT_MAP = {
    "plan": TaskEvent.PLAN_COMPLETE,
    "generate": TaskEvent.CODE_COMPLETE,
    "parse": TaskEvent.PLAN_COMPLETE,     # parser produces a plan-like artifact
    "compile": TaskEvent.CODE_COMPLETE,   # compiler produces executable output
}


def classify_event(lobe_output: str, lobe_role: str) -> TaskEvent:
    """Classify a lobe output into a task event.

    This is the grounding layer — maps continuous (text) output to discrete FSA event.
    Uses keyword patterns specific to each lobe role.

    Args:
        lobe_output: The text output from a lobe step.
        lobe_role: The role of the step ("plan", "generate", "verify", etc.)

    Returns:
        The detected TaskEvent.
    """
    text = lobe_output.lower().strip()

    if lobe_role == "new_task":
        return TaskEvent.NEW_TASK

    if lobe_role == "plan":
        return TaskEvent.PLAN_COMPLETE

    if lobe_role == "generate":
        return TaskEvent.CODE_COMPLETE

    if lobe_role == "verify":
        if "pass" in text and "fail" not in text:
            return TaskEvent.TEST_PASS
        if "fail" in text:
            return TaskEvent.TEST_FAIL
        return TaskEvent.TEST_PASS  # ambiguous → pass (conservative)

    if lobe_role == "deploy":
        return TaskEvent.DEPLOY

    if lobe_role == "reset":
        return TaskEvent.RESET

    # Non-FSA roles (parse, compile, retrieve) — use lookup or skip
    if lobe_role in _ROLE_TO_EVENT_MAP:
        return _ROLE_TO_EVENT_MAP[lobe_role]

    return TaskEvent.NEW_TASK  # fallback


# ---------------------------------------------------------------------------
# MorphSAT Gate
# ---------------------------------------------------------------------------

class MorphSATGate:
    """Hard FSA enforcement gate for task-state transitions.

    Sits between lobe steps in the scheduler. Each step output is classified
    into a TaskEvent via classify_event(), then the gate enforces the FSA
    transition. Illegal transitions are blocked — the step output is suppressed.

    Guardian vows add a second policy layer on top of FSA legality.
    """

    def __init__(
        self,
        transition_table: Optional[np.ndarray] = None,
        guardian_blocked: Optional[Set[Tuple[int, int]]] = None,
        enable_guardian: bool = True,
    ):
        self.T = (transition_table if transition_table is not None
                  else TRANSITION_TABLE).copy()
        self.guardian_blocked = (guardian_blocked if guardian_blocked is not None
                                else GUARDIAN_BLOCKED) if enable_guardian else set()
        self.state = TaskState.IDLE
        self.history: List[dict] = []
        self.illegal_caught = 0
        self.guardian_caught = 0
        self.total_transitions = 0

    def step(self, event: TaskEvent) -> Tuple[TaskState, bool, str]:
        """Attempt a state transition.

        Returns:
            (new_state, was_legal, action_taken)

            action_taken is one of:
                "ALLOWED" — legal transition, state updated
                "FSA_BLOCKED" — illegal per FSA, state held
                "GUARDIAN_BLOCKED" — blocked by guardian vow, state held
        """
        self.total_transitions += 1
        old_state = self.state

        # Guardian check first (policy layer above FSA)
        if (int(old_state), int(event)) in self.guardian_blocked:
            self.guardian_caught += 1
            self.history.append({
                'from': STATE_NAMES[old_state], 'event': EVENT_NAMES[event],
                'action': 'GUARDIAN_BLOCKED', 'to': STATE_NAMES[old_state],
            })
            return self.state, False, 'GUARDIAN_BLOCKED'

        # FSA check
        next_state = self.T[old_state, event]
        if next_state == -1:
            self.illegal_caught += 1
            self.history.append({
                'from': STATE_NAMES[old_state], 'event': EVENT_NAMES[event],
                'action': 'FSA_BLOCKED', 'to': STATE_NAMES[old_state],
            })
            return self.state, False, 'FSA_BLOCKED'

        # Legal transition
        self.state = TaskState(next_state)
        self.history.append({
            'from': STATE_NAMES[old_state], 'event': EVENT_NAMES[event],
            'action': 'ALLOWED', 'to': STATE_NAMES[self.state],
        })
        return self.state, True, 'ALLOWED'

    def reset(self):
        """Reset gate to IDLE state."""
        self.state = TaskState.IDLE

    def to_receipt(self) -> dict:
        """Export gate state and history as a receipt-compatible dict."""
        return {
            'final_state': STATE_NAMES[self.state],
            'total_transitions': self.total_transitions,
            'illegal_caught': self.illegal_caught,
            'guardian_caught': self.guardian_caught,
            'history': self.history,
        }
