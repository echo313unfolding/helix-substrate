"""
Hybrid CPU/GPU scheduler with R* headroom axes and hysteresis.

Decides when to promote work from CPU to GPU (or demote back) based on:
- Predicted gain from GPU execution
- R* axes: thermal, power, memory, queue headroom (each 0-1)
- Hysteresis: promote_threshold > demote_threshold prevents oscillation
- Cooldown: minimum interval between decisions

Usage::

    from helix_substrate.hybrid_scheduler import HybridScheduler

    sched = HybridScheduler(promote_threshold=1.5)
    telemetry = {"temp": 65.0, "mem_used": 1400, "mem_total": 4096}

    promote, info = sched.should_promote(
        predicted_gain=2.5,
        telemetry=telemetry,
    )
    if promote and sched.allocate():
        # run on GPU
        ...
        sched.release()

Ported from echo-box/ops/hybrid_scheduler.py (2025-11).
Adapted: removed QUBO-specific predict_gain heuristic -- caller supplies
predicted_gain directly (model-swap latency ratio, tensor size, etc.).
"""

import time
from typing import Dict, List, Tuple


class HybridScheduler:
    """CPU->GPU promotion scheduler with hysteresis and R* headroom."""

    def __init__(
        self,
        promote_threshold: float = 1.5,
        demote_threshold: float = 1.0,
        cooldown_seconds: float = 30.0,
        max_concurrent: int = 4,
        temp_max: float = 85.0,
        power_max: float = 250.0,
    ):
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_concurrent = max_concurrent
        self.temp_max = temp_max
        self.power_max = power_max

        self.active_gpu_jobs = 0
        self.last_decision_time = 0.0
        self.last_decision = "initial"
        self.decision_history: List[dict] = []

    def get_r_star_axes(self, telemetry: dict) -> dict:
        """
        Compute R* axes (0-1, higher = more headroom).

        telemetry keys: temp, power, mem_used, mem_total.
        Missing keys use safe defaults.
        """
        temp = telemetry.get("temp", 50.0)
        power = telemetry.get("power", 100.0)
        mem_used = telemetry.get("mem_used", 0)
        mem_total = telemetry.get("mem_total", 1)

        return {
            "thermal": max(0.0, 1.0 - (temp / self.temp_max)),
            "power": max(0.0, 1.0 - (power / self.power_max)),
            "memory": max(0.0, 1.0 - (mem_used / max(1, mem_total))),
            "queue": max(0.0, 1.0 - (self.active_gpu_jobs / self.max_concurrent)),
        }

    def should_promote(
        self,
        predicted_gain: float,
        telemetry: dict,
    ) -> Tuple[bool, dict]:
        """
        Decide whether to promote to GPU.

        Args:
            predicted_gain: Expected speedup factor on GPU vs CPU (caller computes).
            telemetry: Dict with temp, power, mem_used, mem_total.

        Returns:
            (should_promote, decision_info)
        """
        now = time.time()
        r_star = self.get_r_star_axes(telemetry)

        promotion_score = (
            predicted_gain
            * r_star["thermal"]
            * r_star["power"]
            * r_star["memory"]
            * r_star["queue"]
        )

        time_since_last = now - self.last_decision_time
        in_cooldown = time_since_last < self.cooldown_seconds

        # Hysteresis: use lower threshold when already on GPU
        if self.last_decision == "promote_to_gpu":
            threshold = self.demote_threshold
            decision_type = (
                "demote_from_gpu" if promotion_score < threshold else "stay_on_gpu"
            )
        else:
            threshold = self.promote_threshold
            decision_type = (
                "promote_to_gpu" if promotion_score > threshold else "cpu_start"
            )

        if in_cooldown:
            decision_type = self.last_decision

        # Diagnostic reasons
        reasons_not = []
        if promotion_score <= self.promote_threshold:
            reasons_not.append(
                f"score={promotion_score:.2f} <= threshold={self.promote_threshold}"
            )
        if r_star["thermal"] < 0.3:
            reasons_not.append(f"thermal_low (R={r_star['thermal']:.2f})")
        if r_star["power"] < 0.3:
            reasons_not.append(f"power_low (R={r_star['power']:.2f})")
        if r_star["queue"] == 0.0:
            reasons_not.append("queue_full")
        if in_cooldown:
            remaining = self.cooldown_seconds - time_since_last
            reasons_not.append(f"cooldown ({remaining:.1f}s remaining)")

        decision = {
            "decision": decision_type,
            "r_star_axes": r_star,
            "predicted_gain": predicted_gain,
            "promotion_score": promotion_score,
            "threshold": threshold,
            "in_cooldown": in_cooldown,
            "reasons_not_to_promote": (
                reasons_not if not decision_type.startswith("promote") else []
            ),
        }

        promote = decision_type == "promote_to_gpu"
        if not in_cooldown:
            self.last_decision = decision_type
            self.last_decision_time = now
            self.decision_history.append(
                {
                    "timestamp": now,
                    "decision": decision_type,
                    "score": promotion_score,
                    "predicted_gain": predicted_gain,
                }
            )

        return promote, decision

    def should_demote(self, temp: float, marginal_improvement: float) -> Tuple[bool, str]:
        """
        Decide whether to demote from GPU.

        Returns:
            (should_demote, reason)
        """
        if temp > (self.temp_max - 5.0):
            return True, "thermal_clamp"
        if marginal_improvement < 0.01:
            return True, "low_marginal_return"
        return False, ""

    def allocate(self) -> bool:
        """Try to allocate a GPU slot. Returns True if successful."""
        if self.active_gpu_jobs < self.max_concurrent:
            self.active_gpu_jobs += 1
            return True
        return False

    def release(self):
        """Release a GPU slot."""
        self.active_gpu_jobs = max(0, self.active_gpu_jobs - 1)
