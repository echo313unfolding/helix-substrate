"""
WO-ECHO-HYBRID-03: Sidecar-aware born-compressed training.

The sidecar residual (shadow_weight - quantized_weight) is a live quality signal.
Layers where the residual grows are layers where the codebook can't track weight
evolution. This trainer uses that signal to drive per-layer adaptive recompression.

Three modes:
1. UNIFORM  — baseline: recompress all layers on fixed schedule (Phase 1 behavior)
2. THRESHOLD — recompress a layer when its sidecar norm exceeds N× its initial value
3. TOPK     — every K steps, recompress only the top-M layers by sidecar norm growth

The hypothesis: adaptive recompression closes the born-compressed gap with dense
because compression budget is allocated where weights are actually changing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from echo_hybrid.config import EchoHybridConfig, EchoHybridModel
from echo_hybrid.train_phase1 import (
    compress_linear,
    compress_all_linears,
    STEQuantizer,
    load_wikitext_chunks,
)


def compute_eff_rank(weight: torch.Tensor) -> float:
    """Effective rank = exp(entropy of normalized singular values)."""
    import math
    W = weight.detach().float()
    if W.dim() == 1:
        return 1.0
    try:
        sv = torch.linalg.svdvals(W)
        sv = sv[sv > 1e-10]
        if len(sv) == 0:
            return 1.0
        p = sv / sv.sum()
        entropy = -(p * p.log()).sum().item()
        return math.exp(entropy)
    except Exception:
        return 1.0

RECEIPT_DIR = Path("receipts/echo_hybrid")
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


def cost_block(t_start, cpu_start, start_iso):
    return {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def loss_stats(losses):
    window = max(1, len(losses) // 10)
    smoothed = [np.mean(losses[max(0, i - window):i + 1]) for i in range(len(losses))]
    return {
        "first_loss": round(losses[0], 4),
        "last_loss": round(losses[-1], 4),
        "min_loss": round(min(losses), 4),
        "loss_delta": round(losses[0] - losses[-1], 4),
        "trend_down": bool(smoothed[-1] < smoothed[0]),
        "loss_curve": [round(l, 4) for l in losses],
    }


# ---------------------------------------------------------------------------
# Sidecar norm computation
# ---------------------------------------------------------------------------

def compute_sidecar_norms(
    model: nn.Module,
    compressed: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """Compute per-layer residual L2 norm between shadow weights and quantized weights.

    This is the training-time analog of the live sidecar norm from inference.
    A layer with a growing sidecar norm is a layer where the codebook is stale.
    """
    modules = dict(model.named_modules())
    norms = {}
    for name, factors in compressed.items():
        mod = modules[name]
        W_shadow = mod.weight.data.float()

        # Reconstruct quantized weight
        cb = factors["codebook"].float().to(W_shadow.device)
        idx = factors["indices"].to(W_shadow.device).long()
        W_q = cb[idx]

        # Apply sidecar corrections
        sp = factors["sidecar_positions"]
        sv = factors["sidecar_values"].float().to(W_shadow.device)
        W_q_flat = W_q.reshape(-1)
        W_q_flat[sp] += sv
        W_q = W_q_flat.reshape(W_shadow.shape)

        # Residual norm (RMS to normalize across layer sizes)
        residual = W_shadow - W_q
        norms[name] = residual.pow(2).mean().sqrt().item()

    return norms


# ---------------------------------------------------------------------------
# Sidecar-aware trainer
# ---------------------------------------------------------------------------

class SidecarAwareTrainer:
    """Born-compressed training with sidecar-driven adaptive recompression.

    Monitors the residual between shadow weights and quantized weights per layer.
    Recompresses layers selectively based on where the codebook is drifting most.
    """

    def __init__(
        self,
        model: EchoHybridModel,
        lr: float = 1e-4,
        n_clusters: int = 256,
        vector_dim: int = 1,
        device: str = "cpu",
        # Recompression policy
        mode: str = "threshold",  # "uniform", "threshold", "topk"
        # Uniform params
        uniform_every: int = 50,
        # Threshold params
        threshold_check_every: int = 10,
        threshold_multiplier: float = 2.0,  # recompress when norm > init_norm * multiplier
        # TopK params
        topk_every: int = 20,
        topk_k: int = 10,  # recompress top-K layers by norm growth
    ):
        self.model = model.to(device)
        self.device = device
        self.n_clusters = n_clusters
        self.vector_dim = vector_dim
        self.step_count = 0
        self.mode = mode

        # Policy params
        self.uniform_every = uniform_every
        self.threshold_check_every = threshold_check_every
        self.threshold_multiplier = threshold_multiplier
        self.topk_every = topk_every
        self.topk_k = topk_k

        # Initial compression
        vd_label = f"d={vector_dim}" if vector_dim > 1 else "scalar"
        print(f"  Initial compression ({n_clusters} clusters, {vd_label})...")
        self.compressed = compress_all_linears(model, n_clusters=n_clusters, vector_dim=vector_dim)
        self.codebook_init_verified = len(self.compressed) > 0
        print(f"  Compressed {len(self.compressed)} layers.")

        # Compute initial sidecar norms (baseline)
        self.init_norms = compute_sidecar_norms(model, self.compressed)
        self.norm_history: Dict[str, List[float]] = {name: [n] for name, n in self.init_norms.items()}

        # Tracking
        self.recompress_events: List[Dict] = []  # {step, layer, old_norm, new_norm, reason}
        self.total_recompressions = 0
        self.eff_rank_snapshots: List[Dict] = []  # {step, per_layer_eff_rank}
        self.utilization_log: List[Dict] = []  # {step, per_layer_utilization}
        self.eff_rank_every: int = 50  # log eff_rank every N steps

        self.ste = STEQuantizer(model, self.compressed)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        print(f"  Mode: {mode}")
        print(f"  Initial mean sidecar norm: {np.mean(list(self.init_norms.values())):.6f}")

    def train_step(self, input_ids: torch.Tensor) -> float:
        """One training step with STE + sidecar monitoring."""
        self.model.train()
        input_ids = input_ids.to(self.device)

        # STE forward
        self.ste.apply_quantized_weights()
        out = self.model(input_ids=input_ids, labels=input_ids)
        loss = out["loss"]

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # CRITICAL: restore shadow weights BEFORE optimizer step.
        # Gradient lives in weight.grad. Restore first so optimizer
        # updates the real shadow weights, not the STE-substituted W_q.
        self.ste.restore_shadow_weights()
        self.optimizer.step()
        self.step_count += 1

        # Adaptive recompression based on mode
        if self.mode == "uniform":
            self._policy_uniform()
        elif self.mode == "threshold":
            self._policy_threshold()
        elif self.mode == "topk":
            self._policy_topk()

        # Periodic eff_rank snapshot
        if self.eff_rank_every > 0 and self.step_count % self.eff_rank_every == 0:
            self._snapshot_eff_rank()

        return loss.item()

    def _compute_current_norms(self) -> Dict[str, float]:
        """Compute current sidecar norms and record history."""
        norms = compute_sidecar_norms(self.model, self.compressed)
        for name, n in norms.items():
            self.norm_history[name].append(n)
        return norms

    def _snapshot_eff_rank(self):
        """Log per-layer effective rank (one SVD per layer, cheap)."""
        modules = dict(self.model.named_modules())
        snapshot = {"step": self.step_count}
        for name in self.compressed:
            mod = modules[name]
            snapshot[name] = round(compute_eff_rank(mod.weight), 4)
        self.eff_rank_snapshots.append(snapshot)

    def _recompress_layer(self, name: str, reason: str):
        """Recompress a single layer and update the compressed dict."""
        modules = dict(self.model.named_modules())
        mod = modules[name]
        old_norm = self.norm_history[name][-1] if self.norm_history[name] else 0

        vd = self.compressed[name].get("vector_dim", 1)
        if isinstance(vd, torch.Tensor):
            vd = vd.item()
        self.compressed[name] = compress_linear(mod.weight, n_clusters=self.n_clusters, vector_dim=vd)
        self.ste.compressed = self.compressed

        new_norm = compute_sidecar_norms(self.model, {name: self.compressed[name]})[name]
        self.norm_history[name].append(new_norm)

        # Log codebook utilization
        util = self.compressed[name].get("codebook_utilization", None)
        n_active = self.compressed[name].get("n_active_centroids", None)

        self.recompress_events.append({
            "step": self.step_count,
            "layer": name,
            "old_norm": round(old_norm, 6),
            "new_norm": round(new_norm, 6),
            "reason": reason,
            "codebook_utilization": round(util, 4) if util is not None else None,
            "n_active_centroids": n_active,
        })
        self.total_recompressions += 1

    def _recompress_all(self, reason: str):
        """Recompress all layers."""
        self.compressed = compress_all_linears(self.model, n_clusters=self.n_clusters)
        self.ste.compressed = self.compressed

        norms = compute_sidecar_norms(self.model, self.compressed)
        for name, n in norms.items():
            self.norm_history[name].append(n)

        # Log per-layer utilization
        util_summary = {}
        for name, factors in self.compressed.items():
            util = factors.get("codebook_utilization", None)
            if util is not None:
                util_summary[name] = round(util, 4)

        self.recompress_events.append({
            "step": self.step_count,
            "layer": "ALL",
            "reason": reason,
            "per_layer_utilization": util_summary if util_summary else None,
        })
        self.total_recompressions += len(self.compressed)

    def _policy_uniform(self):
        """Classic: recompress everything on fixed schedule."""
        if self.uniform_every > 0 and self.step_count % self.uniform_every == 0:
            self._recompress_all(f"uniform_every_{self.uniform_every}")

    def _policy_threshold(self):
        """Recompress layers whose sidecar norm exceeds threshold_multiplier × init_norm."""
        if self.step_count % self.threshold_check_every != 0:
            return

        norms = self._compute_current_norms()
        triggered = []
        for name, current_norm in norms.items():
            init_norm = self.init_norms[name]
            if init_norm > 0 and current_norm > init_norm * self.threshold_multiplier:
                triggered.append((name, current_norm, init_norm))

        if triggered:
            for name, curr, init in triggered:
                self._recompress_layer(name, f"threshold_{self.threshold_multiplier}x (curr={curr:.6f}, init={init:.6f})")
            if self.step_count <= 100 or self.step_count % 50 == 0:
                print(f"    [sidecar] step {self.step_count}: recompressed {len(triggered)} layers (threshold)")

    def _policy_topk(self):
        """Every topk_every steps, recompress the top-K layers by norm growth ratio."""
        if self.step_count % self.topk_every != 0:
            return

        norms = self._compute_current_norms()
        growth_ratios = []
        for name, current_norm in norms.items():
            init_norm = self.init_norms[name]
            ratio = current_norm / max(init_norm, 1e-10)
            growth_ratios.append((name, ratio, current_norm))

        # Sort by growth ratio descending, take top-K
        growth_ratios.sort(key=lambda x: x[1], reverse=True)
        to_recompress = growth_ratios[:self.topk_k]

        for name, ratio, curr in to_recompress:
            self._recompress_layer(name, f"topk_{self.topk_k} (growth={ratio:.2f}x)")

        if self.step_count <= 100 or self.step_count % 50 == 0:
            print(f"    [sidecar] step {self.step_count}: recompressed top-{self.topk_k} layers "
                  f"(max growth={growth_ratios[0][1]:.2f}x)")

    def get_diagnostics(self) -> Dict:
        """Return diagnostic summary for receipt."""
        final_norms = compute_sidecar_norms(self.model, self.compressed)
        init_mean = np.mean(list(self.init_norms.values()))
        final_mean = np.mean(list(final_norms.values()))

        # Per-layer growth
        layer_growth = {}
        for name in self.init_norms:
            init_n = self.init_norms[name]
            final_n = final_norms.get(name, 0)
            layer_growth[name] = {
                "init_norm": round(init_n, 6),
                "final_norm": round(final_n, 6),
                "growth_ratio": round(final_n / max(init_n, 1e-10), 3),
            }

        # Which layers got recompressed most?
        recomp_counts = {}
        for event in self.recompress_events:
            layer = event["layer"]
            recomp_counts[layer] = recomp_counts.get(layer, 0) + 1

        # Initial codebook utilization
        init_utilization = {}
        for name, factors in self.compressed.items():
            util = factors.get("codebook_utilization", None)
            if util is not None:
                init_utilization[name] = round(util, 4)

        return {
            "mode": self.mode,
            "total_recompressions": self.total_recompressions,
            "init_mean_norm": round(init_mean, 6),
            "final_mean_norm": round(final_mean, 6),
            "norm_growth_ratio": round(final_mean / max(init_mean, 1e-10), 3),
            "recompress_event_count": len(self.recompress_events),
            "recompress_counts_by_layer": recomp_counts,
            "top5_growth_layers": sorted(
                layer_growth.items(),
                key=lambda x: x[1]["growth_ratio"],
                reverse=True,
            )[:5],
            "layer_growth": layer_growth,
            "eff_rank_snapshots": self.eff_rank_snapshots,
            "codebook_utilization": init_utilization,
        }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    mode: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    chunks: torch.Tensor,
    **trainer_kwargs,
) -> Dict:
    """Run one sidecar-aware training experiment."""
    print(f"\n  --- Mode: {mode} ({trainer_kwargs}) ---")

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)

    trainer = SidecarAwareTrainer(
        model=model,
        lr=lr,
        mode=mode,
        device="cpu",
        **trainer_kwargs,
    )

    losses = []
    chunk_idx = 0
    for step in range(steps):
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(batch_size)])
        chunk_idx += batch_size
        loss = trainer.train_step(batch)
        losses.append(loss)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"    step {step+1:4d}/{steps}  loss={loss:.4f}")

    stats = loss_stats(losses)
    diagnostics = trainer.get_diagnostics()

    print(f"  Loss: {stats['first_loss']} -> {stats['last_loss']} (delta={stats['loss_delta']})")
    print(f"  Total recompressions: {diagnostics['total_recompressions']}")
    print(f"  Norm growth: {diagnostics['init_mean_norm']:.6f} -> {diagnostics['final_mean_norm']:.6f} "
          f"({diagnostics['norm_growth_ratio']}x)")

    return {
        "mode": mode,
        "trainer_kwargs": {k: v for k, v in trainer_kwargs.items()},
        "model": f"EchoHybrid-{cfg.n_blocks}block-{cfg.hidden_size}d",
        "block_pattern": cfg.block_pattern,
        "training_steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        **stats,
        "diagnostics": diagnostics,
        "cost": cost_block(t_start, cpu_start, start_iso),
    }


def run_dense_baseline(steps, batch_size, seq_len, lr, chunks) -> Dict:
    """Dense baseline for gap comparison."""
    print(f"\n  --- Dense baseline ---")

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    losses = []
    chunk_idx = 0
    for step in range(steps):
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(batch_size)])
        chunk_idx += batch_size

        out = model(input_ids=batch, labels=batch)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 10 == 0 or step == 0:
            print(f"    step {step+1:4d}/{steps}  loss={loss.item():.4f}")

    stats = loss_stats(losses)
    return {
        "mode": "dense",
        "training_steps": steps,
        **stats,
        "cost": cost_block(t_start, cpu_start, start_iso),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WO-ECHO-HYBRID-03: Sidecar-aware training")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mode", type=str, default="all",
                        help="Mode: uniform, threshold, topk, or 'all' for comparison")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    max_chunks = args.steps * args.batch_size * 10
    chunks = load_wikitext_chunks(seq_len=args.seq_len, max_chunks=max_chunks)

    print("\n" + "=" * 70)
    print("WO-ECHO-HYBRID-03: Sidecar-Aware Born-Compressed Training")
    print("=" * 70)

    results = {}

    # Dense baseline
    print("\n" + "=" * 70)
    print("BASELINE: Dense (no compression)")
    print("=" * 70)
    results["dense"] = run_dense_baseline(args.steps, args.batch_size, args.seq_len, args.lr, chunks)

    if args.mode in ("uniform", "all"):
        print("\n" + "=" * 70)
        print("MODE: Uniform recompression (Phase 1 equivalent)")
        print("=" * 70)
        results["uniform_50"] = run_experiment(
            "uniform", args.steps, args.batch_size, args.seq_len, args.lr, chunks,
            uniform_every=50,
        )

    if args.mode in ("threshold", "all"):
        print("\n" + "=" * 70)
        print("MODE: Threshold-triggered recompression")
        print("=" * 70)
        # Test multiple thresholds
        for mult in [1.5, 2.0, 3.0]:
            results[f"threshold_{mult}x"] = run_experiment(
                "threshold", args.steps, args.batch_size, args.seq_len, args.lr, chunks,
                threshold_check_every=10,
                threshold_multiplier=mult,
            )

    if args.mode in ("topk", "all"):
        print("\n" + "=" * 70)
        print("MODE: Top-K recompression")
        print("=" * 70)
        # Test different K values and check intervals
        for k, every in [(5, 20), (10, 20), (10, 10)]:
            results[f"topk_{k}_every{every}"] = run_experiment(
                "topk", args.steps, args.batch_size, args.seq_len, args.lr, chunks,
                topk_every=every,
                topk_k=k,
            )

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON: Sidecar-Aware vs Dense vs Uniform")
    print("=" * 70)
    print(f"{'mode':<25} {'first':>8} {'last':>8} {'delta':>8} {'gap_vs_dense':>14} {'recomps':>8}")
    print("-" * 80)

    dense_last = results.get("dense", {}).get("last_loss", 0)
    for name, r in sorted(results.items()):
        last = r.get("last_loss", 0)
        gap = round(last - dense_last, 4) if dense_last else "?"
        recomps = r.get("diagnostics", {}).get("total_recompressions", "-")
        print(f"{name:<25} {r.get('first_loss', '?'):>8} {last:>8} {r.get('loss_delta', '?'):>8} {gap:>+14} {recomps!s:>8}")

    # Emit receipt
    receipt = {
        "wo": "WO-ECHO-HYBRID-03",
        "experiment": "sidecar_aware_training",
        "timestamp": time.strftime("%Y-%m-%d"),
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "results": results,
        "dense_last_loss": dense_last,
        "best_mode": min(
            [(k, v["last_loss"]) for k, v in results.items() if k != "dense"],
            key=lambda x: x[1],
        )[0] if len(results) > 1 else None,
        "best_gap_vs_dense": round(
            min(v["last_loss"] for k, v in results.items() if k != "dense") - dense_last, 4
        ) if len(results) > 1 else None,
        "notes": (
            "Sidecar-aware born-compressed training. Uses residual norm between "
            "shadow weights and quantized weights as a live signal to drive "
            "per-layer adaptive recompression. Tests threshold vs topK vs uniform."
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_03_sidecar_aware.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nRECEIPT: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
