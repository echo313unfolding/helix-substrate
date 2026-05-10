"""
WO-ECHO-HYBRID-04: Se-routed born-compressed training.

Uses weight geometry (eff_rank/k, kurtosis) to set per-layer compression thresholds.
Logs a full feature vector on every recompression event for routing function development.

Key insight: layers with low effective rank can tolerate more codebook drift because
their weight structure is low-dimensional. Layers with high effective rank need tighter
tracking because information is spread across more dimensions.

Feature vector (logged per recompression event):
  - layer_name, block_type (ssm/attn), step
  - eff_rank, se (eff_rank/k), kurtosis
  - sidecar_norm, drift_ratio (current/init norm)
  - loss_before, loss_after (per-step loss around recompression)
  - weight_rms, weight_std
"""

from __future__ import annotations

import argparse
import json
import math
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
from echo_hybrid.train_sidecar_aware import compute_sidecar_norms

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
# Weight geometry features
# ---------------------------------------------------------------------------

def compute_eff_rank(weight: torch.Tensor) -> float:
    """Effective rank = exp(entropy of normalized singular values)."""
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


def compute_kurtosis(weight: torch.Tensor) -> float:
    """Excess kurtosis of weight values."""
    W = weight.detach().float().reshape(-1)
    if len(W) < 4:
        return 0.0
    mean = W.mean()
    var = W.var()
    if var < 1e-10:
        return 0.0
    m4 = ((W - mean) ** 4).mean()
    return (m4 / (var ** 2) - 3.0).item()


def compute_layer_features(name: str, mod: nn.Module, block_pattern: list) -> Dict:
    """Compute full feature vector for a linear layer."""
    W = mod.weight

    # Determine block type
    block_type = "unknown"
    for i, bt in enumerate(block_pattern):
        if f"blocks.{i}." in name:
            block_type = bt
            break

    # Determine layer role
    role = "unknown"
    if "in_proj" in name:
        role = "in_proj"
    elif "out_proj" in name:
        role = "out_proj"
    elif "x_proj" in name:
        role = "x_proj"
    elif "dt_proj" in name:
        role = "dt_proj"
    elif "q_proj" in name:
        role = "q_proj"
    elif "k_proj" in name:
        role = "k_proj"
    elif "v_proj" in name:
        role = "v_proj"
    elif "o_proj" in name:
        role = "o_proj"
    elif "ffn_up" in name:
        role = "ffn_up"
    elif "ffn_down" in name:
        role = "ffn_down"

    eff_rank = compute_eff_rank(W)
    k = 256  # n_clusters
    se = eff_rank / k

    return {
        "name": name,
        "block_type": block_type,
        "role": role,
        "shape": list(W.shape),
        "n_params": W.numel(),
        "eff_rank": round(eff_rank, 4),
        "se": round(se, 6),
        "kurtosis": round(compute_kurtosis(W), 4),
        "weight_rms": round(W.float().pow(2).mean().sqrt().item(), 6),
        "weight_std": round(W.float().std().item(), 6),
    }


# ---------------------------------------------------------------------------
# Se-routed trainer
# ---------------------------------------------------------------------------

class SeRoutedTrainer:
    """Born-compressed training with Se (eff_rank/k) geometry-aware routing.

    Per-layer recompression threshold is set inversely proportional to Se:
    - Low Se (low-rank) layers: high threshold (tolerate more drift)
    - High Se (high-rank) layers: low threshold (recompress sooner)

    Formula: per_layer_threshold = base_threshold / (se_ratio ** sensitivity)
    where se_ratio = layer_se / median_se (normalized around 1.0)

    Logs full feature vector on every recompression event.
    """

    def __init__(
        self,
        model: EchoHybridModel,
        lr: float = 1e-4,
        n_clusters: int = 256,
        device: str = "cpu",
        # Routing params
        check_every: int = 10,
        base_threshold: float = 3.0,  # base norm growth multiplier
        se_sensitivity: float = 1.0,  # how strongly Se modulates threshold (0=uniform, 1=linear, 2=quadratic)
    ):
        self.model = model.to(device)
        self.device = device
        self.n_clusters = n_clusters
        self.step_count = 0
        self.check_every = check_every
        self.base_threshold = base_threshold
        self.se_sensitivity = se_sensitivity
        self.last_loss = None

        # Initial compression
        print(f"  Initial compression ({n_clusters} clusters)...")
        self.compressed = compress_all_linears(model, n_clusters=n_clusters)
        self.codebook_init_verified = len(self.compressed) > 0
        print(f"  Compressed {len(self.compressed)} layers.")

        # Compute per-layer features at init
        modules = dict(model.named_modules())
        self.layer_features: Dict[str, Dict] = {}
        self.init_features: Dict[str, Dict] = {}  # snapshot at init
        for name in self.compressed:
            features = compute_layer_features(name, modules[name], model.cfg.block_pattern)
            self.layer_features[name] = features
            self.init_features[name] = dict(features)

        # Compute Se-aware per-layer thresholds
        se_values = [f["se"] for f in self.layer_features.values()]
        self.median_se = float(np.median(se_values))
        self.per_layer_threshold: Dict[str, float] = {}

        print(f"\n  Per-layer Se routing (base_threshold={base_threshold}, sensitivity={se_sensitivity}):")
        print(f"  {'layer':<50} {'se':>8} {'threshold':>10} {'eff_rank':>10} {'kurtosis':>10}")
        print("  " + "-" * 92)

        for name, features in self.layer_features.items():
            se_ratio = features["se"] / max(self.median_se, 1e-10)
            # Higher Se → lower threshold (recompress sooner)
            # Lower Se → higher threshold (tolerate more drift)
            threshold = base_threshold / max(se_ratio ** se_sensitivity, 0.1)
            # Clamp to reasonable range
            threshold = max(1.2, min(threshold, 20.0))
            self.per_layer_threshold[name] = threshold

            short_name = name if len(name) <= 48 else "..." + name[-45:]
            print(f"  {short_name:<50} {features['se']:>8.4f} {threshold:>10.2f} "
                  f"{features['eff_rank']:>10.2f} {features['kurtosis']:>10.2f}")

        # Sidecar norm tracking
        self.init_norms = compute_sidecar_norms(model, self.compressed)
        self.norm_history: Dict[str, List[float]] = {name: [n] for name, n in self.init_norms.items()}

        # Full event log (the routing dataset)
        self.recomp_events: List[Dict] = []
        self.total_recompressions = 0

        self.ste = STEQuantizer(model, self.compressed)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        print(f"\n  Median Se: {self.median_se:.6f}")
        print(f"  Threshold range: [{min(self.per_layer_threshold.values()):.2f}, "
              f"{max(self.per_layer_threshold.values()):.2f}]")
        print(f"  Init mean sidecar norm: {np.mean(list(self.init_norms.values())):.6f}")

    def train_step(self, input_ids: torch.Tensor) -> float:
        """One training step with STE + Se-routed recompression."""
        self.model.train()
        input_ids = input_ids.to(self.device)

        # STE forward
        self.ste.apply_quantized_weights()
        out = self.model(input_ids=input_ids, labels=input_ids)
        loss = out["loss"]

        # Backward — restore BEFORE optimizer step (the Phase 1 bug fix)
        self.optimizer.zero_grad()
        loss.backward()
        self.ste.restore_shadow_weights()
        self.optimizer.step()

        self.step_count += 1
        loss_val = loss.item()
        self.last_loss = loss_val

        # Se-aware adaptive recompression
        if self.step_count % self.check_every == 0:
            self._se_route_recompress(loss_val)

        return loss_val

    def _se_route_recompress(self, current_loss: float):
        """Check each layer against its Se-weighted threshold. Recompress if exceeded."""
        norms = compute_sidecar_norms(self.model, self.compressed)
        for name, n in norms.items():
            self.norm_history[name].append(n)

        triggered = []
        for name, current_norm in norms.items():
            init_norm = self.init_norms[name]
            if init_norm < 1e-12:
                continue
            drift_ratio = current_norm / init_norm
            threshold = self.per_layer_threshold[name]

            if drift_ratio > threshold:
                triggered.append((name, drift_ratio, threshold, current_norm))

        if not triggered:
            return

        # Recompress triggered layers and log full feature vector
        modules = dict(self.model.named_modules())
        for name, drift_ratio, threshold, pre_norm in triggered:
            mod = modules[name]

            # Snapshot features BEFORE recompression
            pre_features = compute_layer_features(name, mod, self.model.cfg.block_pattern)

            # Recompress
            self.compressed[name] = compress_linear(mod.weight, n_clusters=self.n_clusters)
            self.ste.compressed = self.compressed
            self.total_recompressions += 1

            # Post-recompression norm
            post_norms = compute_sidecar_norms(self.model, {name: self.compressed[name]})
            post_norm = post_norms[name]
            self.norm_history[name].append(post_norm)

            # Update init norm baseline to post-recompression
            self.init_norms[name] = post_norm

            # Log full event
            event = {
                "step": self.step_count,
                "layer_name": name,
                "block_type": pre_features["block_type"],
                "role": pre_features["role"],
                "shape": pre_features["shape"],
                "n_params": pre_features["n_params"],
                # Geometry at time of recompression
                "eff_rank": pre_features["eff_rank"],
                "se": pre_features["se"],
                "kurtosis": pre_features["kurtosis"],
                "weight_rms": pre_features["weight_rms"],
                "weight_std": pre_features["weight_std"],
                # Drift signal
                "pre_sidecar_norm": round(pre_norm, 8),
                "post_sidecar_norm": round(post_norm, 8),
                "drift_ratio": round(drift_ratio, 4),
                "threshold_used": round(threshold, 4),
                # Training context
                "loss_at_recomp": round(current_loss, 4),
                # Geometry change from init
                "init_eff_rank": self.init_features[name]["eff_rank"],
                "eff_rank_delta": round(pre_features["eff_rank"] - self.init_features[name]["eff_rank"], 4),
                "init_kurtosis": self.init_features[name]["kurtosis"],
                "kurtosis_delta": round(pre_features["kurtosis"] - self.init_features[name]["kurtosis"], 4),
            }
            self.recomp_events.append(event)

        if self.step_count <= 100 or self.step_count % 50 == 0:
            se_triggered = [self.layer_features[name]["se"] for name, _, _, _ in triggered]
            print(f"    [se-route] step {self.step_count}: recompressed {len(triggered)} layers "
                  f"(mean Se={np.mean(se_triggered):.4f})")

    def get_diagnostics(self) -> Dict:
        """Return full diagnostics for receipt."""
        final_norms = compute_sidecar_norms(self.model, self.compressed)
        modules = dict(self.model.named_modules())

        # Final geometry
        final_features = {}
        for name in self.compressed:
            final_features[name] = compute_layer_features(
                name, modules[name], self.model.cfg.block_pattern
            )

        # Per-layer summary
        layer_summary = {}
        for name in self.compressed:
            recomp_count = sum(1 for e in self.recomp_events if e["layer_name"] == name)
            layer_summary[name] = {
                "init_se": self.init_features[name]["se"],
                "final_se": final_features[name]["se"],
                "se_delta": round(final_features[name]["se"] - self.init_features[name]["se"], 6),
                "threshold": round(self.per_layer_threshold[name], 4),
                "recomp_count": recomp_count,
                "init_norm": round(self.norm_history[name][0], 8),
                "final_norm": round(final_norms.get(name, 0), 8),
            }

        return {
            "total_recompressions": self.total_recompressions,
            "n_recomp_events": len(self.recomp_events),
            "median_se": round(self.median_se, 6),
            "base_threshold": self.base_threshold,
            "se_sensitivity": self.se_sensitivity,
            "threshold_range": [
                round(min(self.per_layer_threshold.values()), 4),
                round(max(self.per_layer_threshold.values()), 4),
            ],
            "layer_summary": layer_summary,
            "recomp_event_log": self.recomp_events,
        }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WO-ECHO-HYBRID-04: Se-routed training")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--base-threshold", type=float, default=3.0)
    parser.add_argument("--se-sensitivity", type=float, default=1.0,
                        help="0=uniform, 1=linear Se modulation, 2=quadratic")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    max_chunks = args.steps * args.batch_size * 10
    chunks = load_wikitext_chunks(seq_len=args.seq_len, max_chunks=max_chunks)

    print("\n" + "=" * 70)
    print("WO-ECHO-HYBRID-04: Se-Routed Born-Compressed Training")
    print("=" * 70)

    results = {}

    # Dense baseline
    print("\n--- Dense baseline ---")
    cfg = EchoHybridConfig()
    model_dense = EchoHybridModel(cfg)
    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=args.lr)
    model_dense.train()

    dense_losses = []
    chunk_idx = 0
    for step in range(args.steps):
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(args.batch_size)])
        chunk_idx += args.batch_size
        out = model_dense(input_ids=batch, labels=batch)
        loss = out["loss"]
        opt_dense.zero_grad()
        loss.backward()
        opt_dense.step()
        dense_losses.append(loss.item())
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  step {step+1:4d}/{args.steps}  loss={loss.item():.4f}")

    results["dense"] = loss_stats(dense_losses)
    print(f"Dense: {results['dense']['first_loss']} -> {results['dense']['last_loss']}")

    # Se-routed configs to test
    configs = [
        {"se_sensitivity": 0.0, "label": "se_flat"},      # baseline: Se doesn't modulate (same as uniform threshold)
        {"se_sensitivity": 0.5, "label": "se_sqrt"},       # mild modulation
        {"se_sensitivity": 1.0, "label": "se_linear"},     # linear modulation
        {"se_sensitivity": 2.0, "label": "se_quadratic"},  # strong modulation
    ]

    for config in configs:
        sens = config["se_sensitivity"]
        label = config["label"]
        print(f"\n{'='*70}")
        print(f"Se-routed: {label} (sensitivity={sens})")
        print(f"{'='*70}")

        cfg = EchoHybridConfig()
        model = EchoHybridModel(cfg)

        trainer = SeRoutedTrainer(
            model=model,
            lr=args.lr,
            n_clusters=256,
            device="cpu",
            check_every=args.check_every,
            base_threshold=args.base_threshold,
            se_sensitivity=sens,
        )

        losses = []
        chunk_idx = 0
        for step in range(args.steps):
            batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(args.batch_size)])
            chunk_idx += args.batch_size
            loss = trainer.train_step(batch)
            losses.append(loss)
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  step {step+1:4d}/{args.steps}  loss={loss:.4f}")

        stats = loss_stats(losses)
        diagnostics = trainer.get_diagnostics()

        results[label] = {
            **stats,
            "se_sensitivity": sens,
            "diagnostics_summary": {
                "total_recompressions": diagnostics["total_recompressions"],
                "n_recomp_events": diagnostics["n_recomp_events"],
                "median_se": diagnostics["median_se"],
                "threshold_range": diagnostics["threshold_range"],
            },
            "full_diagnostics": diagnostics,
        }

        print(f"\n  {label}: {stats['first_loss']} -> {stats['last_loss']} "
              f"(delta={stats['loss_delta']}, recomps={diagnostics['total_recompressions']})")

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON: Se-Routed vs Dense")
    print(f"{'='*70}")
    dense_last = results["dense"]["last_loss"]
    print(f"{'mode':<20} {'first':>8} {'last':>8} {'delta':>8} {'gap':>8} {'recomps':>8}")
    print("-" * 60)
    for label, r in results.items():
        last = r["last_loss"]
        gap = round(last - dense_last, 4)
        recomps = r.get("diagnostics_summary", {}).get("total_recompressions", "-")
        print(f"{label:<20} {r['first_loss']:>8} {last:>8} {r['loss_delta']:>8} {gap:>+8} {recomps!s:>8}")

    # Emit receipt
    receipt = {
        "wo": "WO-ECHO-HYBRID-04",
        "experiment": "se_routed_training",
        "timestamp": time.strftime("%Y-%m-%d"),
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "check_every": args.check_every,
        "base_threshold": args.base_threshold,
        "configs_tested": [c["label"] for c in configs],
        "results": results,
        "dense_last_loss": dense_last,
        "notes": (
            "Se-routed born-compressed training. Per-layer recompression threshold "
            "modulated by Se (eff_rank/k): high-rank layers recompress sooner, "
            "low-rank layers tolerate more drift. Full feature vector logged on every "
            "recompression event for routing function development."
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_04_se_routed.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nRECEIPT: {out_path}")

    # Also save the recomp event log as a separate JSONL for analysis
    event_log_path = RECEIPT_DIR / "wo_echo_hybrid_04_recomp_events.jsonl"
    with open(event_log_path, "w") as f:
        for config in configs:
            label = config["label"]
            if label in results and "full_diagnostics" in results[label]:
                for event in results[label]["full_diagnostics"].get("recomp_event_log", []):
                    event["config"] = label
                    f.write(json.dumps(event) + "\n")
    print(f"EVENT LOG: {event_log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
