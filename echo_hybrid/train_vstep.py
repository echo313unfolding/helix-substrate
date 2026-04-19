"""
WO-ECHO-HYBRID-05: V-step codebook refinement + convergence analysis.

Two experiments in one script:
  Exp 1 (--exp convergence): 500-step TopK-5 vs dense, with eff_rank + utilization logging
  Exp 3 (--exp vstep):       V-step EMA codebook update comparison (5 configs × 100 steps)

V-step: After each training step, update codebook entries via EMA of their assigned weights:
  c_j = (1 - α) * c_j + α * mean(w_i for assign(w_i) == j)
This is an online "soft recompression" at negligible cost. Full k-means recompression
still runs on the TopK schedule as a reset.

Usage:
    python -m echo_hybrid.train_vstep --exp convergence   # Exp 1: ~45 min
    python -m echo_hybrid.train_vstep --exp vstep          # Exp 3: ~50 min
    python -m echo_hybrid.train_vstep --exp all            # Both
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from echo_hybrid.config import EchoHybridConfig, EchoHybridModel, load_pretrained_hybrid
from echo_hybrid.train_phase1 import (
    compress_linear,
    compress_all_linears,
    STEQuantizer,
    load_wikitext_chunks,
)
from echo_hybrid.train_sidecar_aware import (
    SidecarAwareTrainer,
    compute_sidecar_norms,
    cost_block,
    loss_stats,
)
from echo_hybrid.train_se_routed import compute_eff_rank

RECEIPT_DIR = Path("receipts/echo_hybrid")
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# V-step codebook update (EMA + PV-Tuning reassign)
# ---------------------------------------------------------------------------

def vstep_update_codebooks(
    compressed: Dict[str, Dict[str, torch.Tensor]],
    model: nn.Module,
    alpha: float = 0.1,
    mode: str = "ema",
) -> Dict[str, int]:
    """Update codebook entries to track evolving shadow weights.

    Two modes (PV-Tuning inspired):
      "ema":      Soft update — c_j = (1-α)*c_j + α*mean(assigned_weights).
                  Cheap but doesn't handle weights crossing centroid boundaries.
      "reassign": One Lloyd's iteration — re-assign each weight to its nearest
                  centroid, then recompute centroids as mean of new assignments.
                  Properly handles boundary crossings. Dead centroids get
                  reinitialized from the largest cluster.

    Handles both scalar VQ (d=1) and grouped VQ (d>1).

    Returns dict of {layer_name: n_dead_centroids} for monitoring.
    """
    modules = dict(model.named_modules())
    dead_counts = {}

    for name, factors in compressed.items():
        mod = modules[name]
        vd = factors.get("vector_dim", 1)
        if isinstance(vd, torch.Tensor):
            vd = vd.item()

        if vd == 1:
            n_dead = _vstep_scalar(factors, mod, alpha, mode)
        else:
            n_dead = _vstep_grouped(factors, mod, vd, alpha, mode)

        dead_counts[name] = n_dead

    return dead_counts


def _vstep_scalar(factors, mod, alpha, mode):
    """V-step for scalar VQ (d=1). Uses fast searchsorted reassignment."""
    W = mod.weight.data.float().cpu().numpy().reshape(-1)
    cb = factors["codebook"].numpy().copy()
    idx = factors["indices"].numpy().reshape(-1).copy()

    if mode == "reassign":
        sort_order = np.argsort(cb)
        sorted_cb = cb[sort_order]
        insert_pos = np.searchsorted(sorted_cb, W)
        insert_pos = np.clip(insert_pos, 0, len(sorted_cb) - 1)
        left_pos = np.clip(insert_pos - 1, 0, len(sorted_cb) - 1)
        dist_right = np.abs(W - sorted_cb[insert_pos])
        dist_left = np.abs(W - sorted_cb[left_pos])
        best_sorted = np.where(dist_left < dist_right, left_pos, insert_pos)
        new_idx = sort_order[best_sorted].astype(idx.dtype)

        n_dead = 0
        for j in range(len(cb)):
            assigned = W[new_idx == j]
            if len(assigned) > 0:
                cb[j] = assigned.mean()
            else:
                n_dead += 1
                cluster_sizes = np.bincount(new_idx, minlength=len(cb))
                largest = cluster_sizes.argmax()
                members = W[new_idx == largest]
                if len(members) > 1:
                    dists = np.abs(members - cb[largest])
                    cb[j] = members[dists.argmax()]

        factors["indices"] = torch.from_numpy(new_idx.reshape(factors["indices"].shape))

    else:  # ema
        n_dead = 0
        for j in range(len(cb)):
            assigned = W[idx == j]
            if len(assigned) > 0:
                cb[j] = (1 - alpha) * cb[j] + alpha * assigned.mean()
            else:
                n_dead += 1

    factors["codebook"] = torch.from_numpy(cb)
    return n_dead


def _load_lloyd_lib():
    """Load hxq_lloyd shared library via ctypes. Returns lib or None."""
    import ctypes
    from pathlib import Path

    candidates = [
        Path("/home/voidstr3m33/hxq-native/lib/libhxq_lloyd.so"),
        Path(__file__).parent.parent / "lib" / "libhxq_lloyd.so",
    ]
    for p in candidates:
        if p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                lib.hxq_lloyd_reassign.restype = ctypes.c_int
                lib.hxq_lloyd_reassign.argtypes = [
                    ctypes.c_void_p,   # weights
                    ctypes.c_void_p,   # codebook
                    ctypes.c_void_p,   # indices
                    ctypes.c_size_t,   # n_vectors
                    ctypes.c_int,      # k
                    ctypes.c_int,      # d
                    ctypes.c_void_p,   # n_dead (int*)
                ]
                return lib
            except OSError:
                continue
    return None

# Try to load at import time — falls back to Python if not available
_lloyd_lib = _load_lloyd_lib()
if _lloyd_lib is not None:
    print("[vstep] Loaded hxq_lloyd C library — fast reassignment enabled")


def _vstep_grouped_c(factors, mod, vd):
    """V-step grouped reassign via C library (hxq_lloyd.so). ~3000x faster than Python."""
    import ctypes

    W = mod.weight.data.float()
    vectors = W.reshape(-1, vd).contiguous()
    cb = factors["codebook"].float().contiguous()
    k = cb.shape[0]
    n_vectors = vectors.shape[0]

    # Get numpy arrays (contiguous, float32)
    w_np = vectors.numpy()
    cb_np = cb.numpy().copy()  # copy because C updates in place
    idx_np = np.empty(n_vectors, dtype=np.uint8)

    n_dead = ctypes.c_int(0)
    rc = _lloyd_lib.hxq_lloyd_reassign(
        w_np.ctypes.data,
        cb_np.ctypes.data,
        idx_np.ctypes.data,
        ctypes.c_size_t(n_vectors),
        ctypes.c_int(k),
        ctypes.c_int(vd),
        ctypes.byref(n_dead),
    )
    if rc != 0:
        raise RuntimeError(f"hxq_lloyd_reassign failed: rc={rc}")

    factors["codebook"] = torch.from_numpy(cb_np)
    factors["indices"] = torch.from_numpy(idx_np).reshape(factors["indices"].shape)
    return n_dead.value


def _vstep_grouped(factors, mod, vd, alpha, mode):
    """V-step for grouped VQ (d>1). Uses C library if available, else torch BLAS.

    Centroid update is fully vectorized (no Python loop over k).
    """
    # Fast path: C library for reassign mode
    if mode == "reassign" and _lloyd_lib is not None:
        return _vstep_grouped_c(factors, mod, vd)

    W = mod.weight.data.float()  # stay in torch
    vectors = W.reshape(-1, vd)  # [N, d]
    cb_t = factors["codebook"].float()  # [k, d]
    k = cb_t.shape[0]

    if mode == "reassign":
        # Reassign: ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x@c.T (all torch)
        X_sq = (vectors * vectors).sum(dim=1, keepdim=True)  # [N, 1]
        C_sq = (cb_t * cb_t).sum(dim=1, keepdim=True).T       # [1, k]
        dists = X_sq + C_sq - 2.0 * (vectors @ cb_t.T)        # [N, k]
        new_idx = dists.argmin(dim=1)                          # [N]

        # Vectorized centroid update via scatter
        new_idx_np = new_idx.numpy()
        vectors_np = vectors.numpy()
        counts = np.bincount(new_idx_np, minlength=k)
        sums = np.zeros((k, vd), dtype=np.float64)
        np.add.at(sums, new_idx_np, vectors_np.astype(np.float64))
        alive = counts > 0
        cb_np = cb_t.numpy().copy()
        cb_np[alive] = (sums[alive] / counts[alive, None]).astype(np.float32)

        # Reinitialize dead centroids
        n_dead = int(np.sum(~alive))
        if n_dead > 0:
            largest = counts.argmax()
            members = vectors_np[new_idx_np == largest]
            if len(members) > 1:
                d2 = np.sum((members - cb_np[largest]) ** 2, axis=1)
                far_idx = np.argsort(d2)[-n_dead:]
                dead_ids = np.where(~alive)[0]
                for i, did in enumerate(dead_ids[:len(far_idx)]):
                    cb_np[did] = members[far_idx[i]]

        factors["codebook"] = torch.from_numpy(cb_np)
        factors["indices"] = new_idx.to(factors["indices"].dtype).reshape(factors["indices"].shape)

    else:  # ema
        idx_np = factors["indices"].numpy().reshape(-1)
        vectors_np = vectors.numpy()
        counts = np.bincount(idx_np, minlength=k)
        sums = np.zeros((k, vd), dtype=np.float64)
        np.add.at(sums, idx_np, vectors_np.astype(np.float64))
        alive = counts > 0
        cb_np = cb_t.numpy().copy()
        means = (sums[alive] / counts[alive, None]).astype(np.float32)
        cb_np[alive] = (1 - alpha) * cb_np[alive] + alpha * means
        n_dead = int(np.sum(~alive))
        factors["codebook"] = torch.from_numpy(cb_np)

    return n_dead


# ---------------------------------------------------------------------------
# VStepTrainer — extends SidecarAwareTrainer with online codebook EMA
# ---------------------------------------------------------------------------

class VStepTrainer(SidecarAwareTrainer):
    """SidecarAwareTrainer + online codebook update after each step.

    Two V-step modes:
      "ema":      Soft centroid update (cheap, may miss boundary crossings)
      "reassign": One Lloyd's iteration per step (tighter, handles drift)

    The TopK recompression schedule still runs as a full k-means reset.
    Between recompressions, V-step keeps codebooks tracking the weights.
    """

    def __init__(
        self,
        model: EchoHybridModel,
        vstep_alpha: float = 0.1,
        vstep_enabled: bool = True,
        vstep_mode: str = "ema",
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.vstep_alpha = vstep_alpha
        self.vstep_enabled = vstep_enabled
        self.vstep_mode = vstep_mode
        self.dead_centroid_log: List[Dict] = []

    def train_step(self, input_ids: torch.Tensor) -> float:
        """One step with STE + sidecar monitoring + V-step codebook update."""
        loss = super().train_step(input_ids)

        # V-step: online codebook update
        if self.vstep_enabled:
            dead_counts = vstep_update_codebooks(
                self.compressed, self.model,
                alpha=self.vstep_alpha,
                mode=self.vstep_mode,
            )
            # Update STE's reference
            self.ste.compressed = self.compressed

            # Log dead centroids periodically
            if self.step_count % 20 == 0:
                total_dead = sum(dead_counts.values())
                total_centroids = len(dead_counts) * self.n_clusters
                self.dead_centroid_log.append({
                    "step": self.step_count,
                    "total_dead": total_dead,
                    "total_centroids": total_centroids,
                    "dead_pct": round(100 * total_dead / max(total_centroids, 1), 2),
                    "per_layer_dead": {k: v for k, v in dead_counts.items() if v > 0},
                })

        return loss

    def get_diagnostics(self) -> Dict:
        diag = super().get_diagnostics()
        diag["vstep_alpha"] = self.vstep_alpha
        diag["vstep_enabled"] = self.vstep_enabled
        diag["vstep_mode"] = self.vstep_mode
        diag["dead_centroid_log"] = self.dead_centroid_log
        return diag


# ---------------------------------------------------------------------------
# Exp 1: 500-step convergence
# ---------------------------------------------------------------------------

def run_convergence(steps: int, batch_size: int, seq_len: int, lr: float):
    """Run TopK-5/every-20 + dense for `steps` steps. Log eff_rank + utilization."""
    print("\n" + "=" * 70)
    print(f"EXP 1: {steps}-step convergence (TopK-5 vs Dense)")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    max_chunks = steps * batch_size * 5
    chunks = load_wikitext_chunks(seq_len=seq_len, max_chunks=max_chunks)

    # --- Dense baseline ---
    print("\n--- Dense baseline ---")
    cfg = EchoHybridConfig()
    model_dense = EchoHybridModel(cfg)
    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=lr)
    model_dense.train()

    dense_losses = []
    dense_eff_rank_snapshots = []
    chunk_idx = 0
    modules_dense = dict(model_dense.named_modules())

    for step in range(steps):
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(batch_size)])
        chunk_idx += batch_size
        out = model_dense(input_ids=batch, labels=batch)
        loss = out["loss"]
        opt_dense.zero_grad()
        loss.backward()
        opt_dense.step()
        dense_losses.append(loss.item())

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  step {step+1:4d}/{steps}  loss={loss.item():.4f}")

        # eff_rank snapshot every 50 steps
        if (step + 1) % 50 == 0:
            snap = {"step": step + 1}
            for name, mod in modules_dense.items():
                if isinstance(mod, nn.Linear) and name != "lm_head":
                    snap[name] = round(compute_eff_rank(mod.weight), 4)
            dense_eff_rank_snapshots.append(snap)

    dense_stats = loss_stats(dense_losses)
    print(f"Dense: {dense_stats['first_loss']} -> {dense_stats['last_loss']}")

    # --- TopK-5/every-20 born-compressed ---
    print(f"\n--- TopK-5/every-20 born-compressed ({steps} steps) ---")
    cfg = EchoHybridConfig()
    model_bc = EchoHybridModel(cfg)

    trainer = SidecarAwareTrainer(
        model=model_bc,
        lr=lr,
        mode="topk",
        topk_every=20,
        topk_k=5,
        device="cpu",
    )

    bc_losses = []
    chunk_idx = 0
    for step in range(steps):
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(batch_size)])
        chunk_idx += batch_size
        loss = trainer.train_step(batch)
        bc_losses.append(loss)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  step {step+1:4d}/{steps}  loss={loss:.4f}")

    bc_stats = loss_stats(bc_losses)
    diagnostics = trainer.get_diagnostics()
    print(f"Born-compressed: {bc_stats['first_loss']} -> {bc_stats['last_loss']}")
    print(f"  Recompressions: {diagnostics['total_recompressions']}")

    # --- Gap trajectory ---
    checkpoints = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    gap_trajectory = {}
    for cp in checkpoints:
        if cp <= len(dense_losses) and cp <= len(bc_losses):
            window = min(10, cp)
            d_loss = np.mean(dense_losses[cp - window:cp])
            bc_loss = np.mean(bc_losses[cp - window:cp])
            gap = round(bc_loss - d_loss, 4)
            gap_trajectory[str(cp)] = {
                "dense_loss": round(d_loss, 4),
                "bc_loss": round(bc_loss, 4),
                "gap": gap,
            }

    print(f"\n  Gap trajectory:")
    print(f"  {'step':>6} {'dense':>8} {'bc':>8} {'gap':>8}")
    print("  " + "-" * 34)
    for cp in checkpoints:
        if str(cp) in gap_trajectory:
            g = gap_trajectory[str(cp)]
            print(f"  {cp:>6} {g['dense_loss']:>8} {g['bc_loss']:>8} {g['gap']:>+8}")

    gaps = [gap_trajectory[str(cp)]["gap"] for cp in checkpoints if str(cp) in gap_trajectory]
    gap_closing = bool(gaps[-1] < gaps[0]) if len(gaps) >= 2 else None

    print(f"\n  Gap closing: {gap_closing}")
    if len(gaps) >= 2:
        print(f"  Gap at first checkpoint: {gaps[0]:+.4f}")
        print(f"  Gap at last checkpoint: {gaps[-1]:+.4f}")

    # --- Codebook utilization summary from initial compression ---
    util_summary = diagnostics.get("codebook_utilization", {})
    if util_summary:
        util_vals = list(util_summary.values())
        print(f"\n  Codebook utilization: mean={np.mean(util_vals):.4f}, "
              f"min={min(util_vals):.4f}, max={max(util_vals):.4f}")
        low_util = {k: v for k, v in util_summary.items() if v < 0.7}
        if low_util:
            print(f"  LOW utilization (<70%): {len(low_util)} layers")
            for name, u in sorted(low_util.items(), key=lambda x: x[1]):
                print(f"    {name}: {u:.4f}")
        else:
            print(f"  All layers ≥70% utilization")

    # Emit receipt
    receipt = {
        "wo": "WO-ECHO-HYBRID-05a",
        "experiment": "convergence_500",
        "timestamp": time.strftime("%Y-%m-%d"),
        "training_steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "lr": lr,
        "mode": "topk_5_every20",
        "dense": dense_stats,
        "born_compressed": bc_stats,
        "gap_trajectory": gap_trajectory,
        "gap_closing": gap_closing,
        "initial_gap": gaps[0] if gaps else None,
        "final_gap": gaps[-1] if gaps else None,
        "diagnostics": diagnostics,
        "dense_eff_rank_snapshots": dense_eff_rank_snapshots,
        "notes": (
            f"{steps}-step convergence test. TopK-5/every-20 vs dense. "
            f"Includes eff_rank snapshots every 50 steps and codebook utilization."
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_05a_convergence.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nRECEIPT: {out_path}")
    return receipt


# ---------------------------------------------------------------------------
# Exp 3: V-step comparison
# ---------------------------------------------------------------------------

def run_vstep_comparison(steps: int, batch_size: int, seq_len: int, lr: float, config_filter: list = None):
    """Compare V-step EMA configs against TopK-5 baseline and dense."""
    print("\n" + "=" * 70)
    print(f"EXP 3: V-Step Codebook Refinement ({steps} steps × 5 configs)")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    max_chunks = steps * batch_size * 12
    chunks = load_wikitext_chunks(seq_len=seq_len, max_chunks=max_chunks)

    configs = [
        {"label": "topk5_no_vstep",   "vstep_enabled": False, "vstep_alpha": 0.0, "vstep_mode": "ema"},
        {"label": "topk5_ema_01",     "vstep_enabled": True,  "vstep_alpha": 0.1, "vstep_mode": "ema"},
        {"label": "topk5_ema_03",     "vstep_enabled": True,  "vstep_alpha": 0.3, "vstep_mode": "ema"},
        {"label": "topk5_reassign",   "vstep_enabled": True,  "vstep_alpha": 0.0, "vstep_mode": "reassign"},
        {"label": "reassign_only",    "vstep_enabled": True,  "vstep_alpha": 0.0, "vstep_mode": "reassign", "no_topk": True},
        # Dense baseline handled separately
    ]

    results = {}

    # Dense baseline
    print("\n--- Dense baseline ---")
    cfg = EchoHybridConfig()
    model_dense = EchoHybridModel(cfg)
    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=lr)
    model_dense.train()

    dense_losses = []
    chunk_idx = 0
    for step in range(steps):
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(batch_size)])
        chunk_idx += batch_size
        out = model_dense(input_ids=batch, labels=batch)
        loss = out["loss"]
        opt_dense.zero_grad()
        loss.backward()
        opt_dense.step()
        dense_losses.append(loss.item())
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  step {step+1:4d}/{steps}  loss={loss.item():.4f}")

    results["dense"] = loss_stats(dense_losses)
    dense_last = results["dense"]["last_loss"]
    print(f"Dense: {results['dense']['first_loss']} -> {dense_last}")

    # V-step configs (optionally filtered)
    if config_filter:
        configs = [c for c in configs if c["label"] in config_filter]
        print(f"  [filter] Running only: {[c['label'] for c in configs]}")
    for config in configs:
        label = config["label"]
        no_topk = config.get("no_topk", False)
        print(f"\n--- {label} ---")

        cfg = EchoHybridConfig()
        model = EchoHybridModel(cfg)

        trainer = VStepTrainer(
            model=model,
            lr=lr,
            mode="topk",
            topk_every=20 if not no_topk else 999999,  # effectively disable topk
            topk_k=5,
            device="cpu",
            vstep_alpha=config["vstep_alpha"],
            vstep_enabled=config["vstep_enabled"],
            vstep_mode=config.get("vstep_mode", "ema"),
        )

        losses = []
        chunk_idx = 0
        for step in range(steps):
            batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(batch_size)])
            chunk_idx += batch_size
            loss = trainer.train_step(batch)
            losses.append(loss)
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  step {step+1:4d}/{steps}  loss={loss:.4f}")

        stats = loss_stats(losses)
        diagnostics = trainer.get_diagnostics()

        gap = round(stats["last_loss"] - dense_last, 4)
        print(f"  {label}: {stats['first_loss']} -> {stats['last_loss']} "
              f"(gap={gap:+.4f}, recomps={diagnostics['total_recompressions']})")

        # Dead centroid summary
        if diagnostics.get("dead_centroid_log"):
            last_dead = diagnostics["dead_centroid_log"][-1]
            print(f"  Dead centroids at end: {last_dead['dead_pct']:.1f}%")

        results[label] = {
            **stats,
            "gap_vs_dense": gap,
            "vstep_alpha": config["vstep_alpha"],
            "vstep_enabled": config["vstep_enabled"],
            "no_topk": no_topk,
            "diagnostics": diagnostics,
        }

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON: V-Step Codebook Refinement")
    print(f"{'='*70}")
    print(f"{'config':<22} {'first':>8} {'last':>8} {'delta':>8} {'gap':>8} {'recomps':>8} {'dead%':>6}")
    print("-" * 74)

    for label, r in sorted(results.items()):
        last = r.get("last_loss", 0)
        gap = round(last - dense_last, 4)
        recomps = r.get("diagnostics", {}).get("total_recompressions", "-")
        dead_log = r.get("diagnostics", {}).get("dead_centroid_log", [])
        dead_pct = dead_log[-1]["dead_pct"] if dead_log else "-"
        print(f"{label:<22} {r.get('first_loss','?'):>8} {last:>8} "
              f"{r.get('loss_delta','?'):>8} {gap:>+8} {recomps!s:>8} {dead_pct!s:>6}")

    # Identify best non-dense config
    non_dense = {k: v for k, v in results.items() if k != "dense"}
    if non_dense:
        best_label = min(non_dense, key=lambda k: non_dense[k]["last_loss"])
        best_gap = non_dense[best_label]["gap_vs_dense"]
        baseline_gap = non_dense.get("topk5_no_vstep", {}).get("gap_vs_dense", None)

        print(f"\nBest config: {best_label} (gap={best_gap:+.4f})")
        if baseline_gap is not None:
            improvement = baseline_gap - best_gap
            print(f"V-step improvement over baseline: {improvement:+.4f} points")

    # Emit receipt
    receipt = {
        "wo": "WO-ECHO-HYBRID-05b",
        "experiment": "vstep_comparison",
        "timestamp": time.strftime("%Y-%m-%d"),
        "training_steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "lr": lr,
        "configs_tested": [c["label"] for c in configs],
        "results": results,
        "dense_last_loss": dense_last,
        "best_config": best_label if non_dense else None,
        "best_gap_vs_dense": best_gap if non_dense else None,
        "notes": (
            "V-step codebook refinement comparison. Online EMA update of codebook "
            "entries between full k-means recompressions. Tests whether continuous "
            "codebook tracking reduces the born-compressed gap."
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_05b_vstep.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nRECEIPT: {out_path}")
    return receipt


# ---------------------------------------------------------------------------
# Exp 4: Pretrained weight initialization
# ---------------------------------------------------------------------------

def run_pretrained_comparison(steps: int, batch_size: int, seq_len: int, lr: float):
    """Compare pretrained init vs random init, dense vs born-compressed vs V-step."""
    print("\n" + "=" * 70)
    print(f"EXP 4: Pretrained Weight Initialization ({steps} steps × 4 configs)")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    max_chunks = steps * batch_size * 10
    chunks = load_wikitext_chunks(seq_len=seq_len, max_chunks=max_chunks)

    results = {}

    # --- Config 1: random_born (sanity check, repeat of WO-03 winner) ---
    print("\n--- random_born (TopK-5, random init) ---")
    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)
    trainer = VStepTrainer(
        model=model, lr=lr, mode="topk", topk_every=20, topk_k=5,
        device="cpu", vstep_enabled=False,
    )
    losses = _run_training_loop(trainer, chunks, steps, batch_size)
    results["random_born"] = {
        **loss_stats(losses),
        "init": "random", "compression": "topk5",
        "diagnostics": trainer.get_diagnostics(),
    }
    print(f"  random_born: {results['random_born']['first_loss']} -> {results['random_born']['last_loss']}")

    # --- Config 2: pretrained_dense ---
    print("\n--- pretrained_dense ---")
    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)
    load_stats = load_pretrained_hybrid(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    chunk_idx = 0
    for step in range(steps):
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(batch_size)])
        chunk_idx += batch_size
        out = model(input_ids=batch, labels=batch)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  step {step+1:4d}/{steps}  loss={loss.item():.4f}")
    results["pretrained_dense"] = {
        **loss_stats(losses),
        "init": "pretrained", "compression": "none",
        "load_stats": load_stats,
    }
    print(f"  pretrained_dense: {results['pretrained_dense']['first_loss']} -> {results['pretrained_dense']['last_loss']}")

    # --- Config 3: pretrained_born (TopK-5) ---
    print("\n--- pretrained_born (TopK-5) ---")
    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)
    load_stats = load_pretrained_hybrid(model)
    trainer = VStepTrainer(
        model=model, lr=lr, mode="topk", topk_every=20, topk_k=5,
        device="cpu", vstep_enabled=False,
    )
    losses = _run_training_loop(trainer, chunks, steps, batch_size)
    results["pretrained_born"] = {
        **loss_stats(losses),
        "init": "pretrained", "compression": "topk5",
        "load_stats": load_stats,
        "diagnostics": trainer.get_diagnostics(),
    }
    print(f"  pretrained_born: {results['pretrained_born']['first_loss']} -> {results['pretrained_born']['last_loss']}")

    # --- Config 4: pretrained_vstep (TopK-5 + V-step EMA) ---
    print("\n--- pretrained_vstep (TopK-5 + V-step α=0.1) ---")
    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)
    load_stats = load_pretrained_hybrid(model)
    trainer = VStepTrainer(
        model=model, lr=lr, mode="topk", topk_every=20, topk_k=5,
        device="cpu", vstep_enabled=True, vstep_alpha=0.1,
    )
    losses = _run_training_loop(trainer, chunks, steps, batch_size)
    results["pretrained_vstep"] = {
        **loss_stats(losses),
        "init": "pretrained", "compression": "topk5_vstep01",
        "load_stats": load_stats,
        "diagnostics": trainer.get_diagnostics(),
    }
    print(f"  pretrained_vstep: {results['pretrained_vstep']['first_loss']} -> {results['pretrained_vstep']['last_loss']}")

    # --- Comparison ---
    print(f"\n{'='*70}")
    print("COMPARISON: Pretrained vs Random Init")
    print(f"{'='*70}")
    dense_last = results["pretrained_dense"]["last_loss"]
    print(f"{'config':<22} {'first':>8} {'last':>8} {'delta':>8} {'gap_vs_dense':>14}")
    print("-" * 62)
    for label, r in sorted(results.items()):
        last = r["last_loss"]
        gap = round(last - dense_last, 4)
        print(f"{label:<22} {r['first_loss']:>8} {last:>8} {r['loss_delta']:>8} {gap:>+14}")

    # Key analysis
    pretrained_gap = results["pretrained_born"]["last_loss"] - results["pretrained_dense"]["last_loss"]
    random_gap = results["random_born"]["last_loss"] - results["pretrained_dense"]["last_loss"]
    vstep_gap = results["pretrained_vstep"]["last_loss"] - results["pretrained_dense"]["last_loss"]

    print(f"\n  Random born gap vs pretrained dense: {random_gap:+.4f}")
    print(f"  Pretrained born gap vs pretrained dense: {pretrained_gap:+.4f}")
    print(f"  Pretrained vstep gap vs pretrained dense: {vstep_gap:+.4f}")
    print(f"  Pretrained init advantage: {random_gap - pretrained_gap:+.4f}")

    receipt = {
        "wo": "WO-ECHO-HYBRID-05c",
        "experiment": "pretrained_comparison",
        "timestamp": time.strftime("%Y-%m-%d"),
        "training_steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "lr": lr,
        "configs_tested": list(results.keys()),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "diagnostics"}
                    for k, v in results.items()},
        "pretrained_gap": round(pretrained_gap, 4),
        "random_gap": round(random_gap, 4),
        "vstep_gap": round(vstep_gap, 4),
        "notes": (
            "Pretrained weight initialization comparison. Tests whether pretrained "
            "weights (mamba-130m + bert-base) improve born-compressed training quality "
            "and whether V-step EMA further closes the gap."
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_05c_pretrained.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nRECEIPT: {out_path}")
    return receipt


def _run_training_loop(trainer, chunks, steps, batch_size):
    """Helper: run training loop and return losses."""
    losses = []
    chunk_idx = 0
    for step in range(steps):
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(batch_size)])
        chunk_idx += batch_size
        loss = trainer.train_step(batch)
        losses.append(loss)
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  step {step+1:4d}/{steps}  loss={loss:.4f}")
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WO-ECHO-HYBRID-05: V-step + convergence + pretrained")
    parser.add_argument("--exp", type=str, default="all",
                        help="convergence, vstep, pretrained, or all")
    parser.add_argument("--convergence-steps", type=int, default=500)
    parser.add_argument("--vstep-steps", type=int, default=100)
    parser.add_argument("--pretrained-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config labels to run (e.g. reassign_only). Default: all")
    args = parser.parse_args()

    if args.exp in ("convergence", "all"):
        run_convergence(args.convergence_steps, args.batch_size, args.seq_len, args.lr)

    if args.exp in ("vstep", "all"):
        cf = args.configs.split(",") if args.configs else None
        run_vstep_comparison(args.vstep_steps, args.batch_size, args.seq_len, args.lr, config_filter=cf)

    if args.exp in ("pretrained", "all"):
        run_pretrained_comparison(args.pretrained_steps, args.batch_size, args.seq_len, args.lr)

    print("\n" + "=" * 70)
    print("WO-ECHO-HYBRID-05 COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
