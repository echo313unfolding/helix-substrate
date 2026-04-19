"""
WO-ECHO-HYBRID-02: Experiment stack — closing the born-compressed gap.

Four experiments:
1. Compress schedule sweep — {10, 25, 50, 100} at 100 steps
2. 500-step convergence — born-compressed vs dense, does the gap close or widen?
3. Pattern search — systematic attention placement exploration
4. Per-layer compression — different cluster counts for SSM vs ATTN layers

All results go to receipts/echo_hybrid/wo_echo_hybrid_02_*.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from echo_hybrid.config import EchoHybridConfig, EchoHybridModel
from echo_hybrid.train_phase1 import (
    compress_all_linears,
    compress_linear,
    STEQuantizer,
    Phase1Trainer,
    load_wikitext_chunks,
)

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


def train_born_compressed(cfg, steps, batch_size, lr, chunks, compress_every, n_clusters=256):
    """Run born-compressed training and return losses list."""
    model = EchoHybridModel(cfg)
    trainer = Phase1Trainer(
        model=model,
        lr=lr,
        compress_schedule=compress_every,
        n_clusters=n_clusters,
        device="cpu",
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

    return losses, trainer


def train_dense(cfg, steps, batch_size, lr, chunks):
    """Run dense training and return losses list."""
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

    return losses


# ---------------------------------------------------------------------------
# Experiment 1: Compress schedule sweep
# ---------------------------------------------------------------------------

def exp1_compress_schedule(steps, batch_size, seq_len, lr, chunks):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Compress schedule sweep")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    cfg = EchoHybridConfig()
    schedules = [10, 25, 50, 100]
    results = {}

    for sched in schedules:
        print(f"\n  --- compress_every={sched} ---")
        losses, trainer = train_born_compressed(cfg, steps, batch_size, lr, chunks, sched)
        stats = loss_stats(losses)
        results[str(sched)] = {
            **stats,
            "n_recompressions": max(0, steps // sched - 1) if sched > 0 else 0,
        }
        print(f"  Loss: {stats['first_loss']} -> {stats['last_loss']} (delta={stats['loss_delta']})")

    # Comparison
    print(f"\n{'schedule':>10} {'first':>8} {'last':>8} {'delta':>8} {'recomps':>8}")
    print("-" * 50)
    for sched in schedules:
        r = results[str(sched)]
        print(f"{sched:>10} {r['first_loss']:>8} {r['last_loss']:>8} {r['loss_delta']:>8} {r['n_recompressions']:>8}")

    receipt = {
        "wo": "WO-ECHO-HYBRID-02",
        "experiment": "compress_schedule_sweep",
        "timestamp": time.strftime("%Y-%m-%d"),
        "model": f"EchoHybrid-{cfg.n_blocks}block-{cfg.hidden_size}d",
        "block_pattern": cfg.block_pattern,
        "training_steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "schedules_tested": schedules,
        "results": results,
        "best_schedule": min(results, key=lambda k: results[k]["last_loss"]),
        "notes": (
            f"Compress schedule sweep: tested recompression every {schedules} steps. "
            f"Question: does more frequent recompression close the gap with dense?"
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_02_compress_schedule.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"RECEIPT: {out_path}")
    return receipt


# ---------------------------------------------------------------------------
# Experiment 2: 500-step convergence
# ---------------------------------------------------------------------------

def exp2_convergence_500(batch_size, seq_len, lr, chunks):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: 500-step convergence (born-compressed vs dense)")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    steps = 500
    cfg = EchoHybridConfig()

    # Dense 500 steps
    print("\n  --- Dense (500 steps) ---")
    dense_losses = train_dense(cfg, steps, batch_size, lr, chunks)
    dense_stats = loss_stats(dense_losses)
    print(f"  Dense: {dense_stats['first_loss']} -> {dense_stats['last_loss']} (delta={dense_stats['loss_delta']})")

    # Born-compressed 500 steps (use best schedule from exp1 if available, else 50)
    # Default to compress-every-50 as a reasonable middle ground
    compress_every = 50
    exp1_path = RECEIPT_DIR / "wo_echo_hybrid_02_compress_schedule.json"
    if exp1_path.exists():
        with open(exp1_path) as f:
            exp1 = json.load(f)
        compress_every = int(exp1.get("best_schedule", 50))
        print(f"  Using best schedule from exp1: compress_every={compress_every}")

    print(f"\n  --- Born-compressed (500 steps, compress_every={compress_every}) ---")
    bc_losses, trainer = train_born_compressed(cfg, steps, batch_size, lr, chunks, compress_every)
    bc_stats = loss_stats(bc_losses)
    print(f"  Born-compressed: {bc_stats['first_loss']} -> {bc_stats['last_loss']} (delta={bc_stats['loss_delta']})")

    # Gap analysis at checkpoints
    checkpoints = [50, 100, 200, 300, 400, 500]
    gap_trajectory = {}
    for cp in checkpoints:
        if cp <= len(dense_losses) and cp <= len(bc_losses):
            d_loss = np.mean(dense_losses[max(0, cp-10):cp])
            bc_loss = np.mean(bc_losses[max(0, cp-10):cp])
            gap = round(bc_loss - d_loss, 4)
            gap_trajectory[str(cp)] = {
                "dense_loss": round(d_loss, 4),
                "bc_loss": round(bc_loss, 4),
                "gap": gap,
            }

    gap_closing = False
    gaps = [gap_trajectory[str(cp)]["gap"] for cp in checkpoints if str(cp) in gap_trajectory]
    if len(gaps) >= 2:
        gap_closing = bool(gaps[-1] < gaps[0])

    print(f"\n  Gap trajectory:")
    print(f"  {'step':>6} {'dense':>8} {'bc':>8} {'gap':>8}")
    print("  " + "-" * 34)
    for cp in checkpoints:
        if str(cp) in gap_trajectory:
            g = gap_trajectory[str(cp)]
            print(f"  {cp:>6} {g['dense_loss']:>8} {g['bc_loss']:>8} {g['gap']:>+8}")

    print(f"\n  Gap closing over time: {gap_closing}")
    print(f"  Gap at 100: {gap_trajectory.get('100', {}).get('gap', '?')}")
    print(f"  Gap at 500: {gap_trajectory.get('500', {}).get('gap', '?')}")

    receipt = {
        "wo": "WO-ECHO-HYBRID-02",
        "experiment": "convergence_500",
        "timestamp": time.strftime("%Y-%m-%d"),
        "model": f"EchoHybrid-{cfg.n_blocks}block-{cfg.hidden_size}d",
        "block_pattern": cfg.block_pattern,
        "training_steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "compress_every": compress_every,
        "dense": dense_stats,
        "born_compressed": bc_stats,
        "gap_trajectory": gap_trajectory,
        "gap_closing": gap_closing,
        "initial_gap": gaps[0] if gaps else None,
        "final_gap": gaps[-1] if gaps else None,
        "notes": (
            f"500-step convergence test. Does the born-compressed gap close, widen, or "
            f"hold steady vs dense? compress_every={compress_every}."
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_02_convergence.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"RECEIPT: {out_path}")
    return receipt


# ---------------------------------------------------------------------------
# Experiment 3: Pattern search
# ---------------------------------------------------------------------------

def generate_candidate_patterns() -> List[Tuple[str, List[str]]]:
    """Generate a set of attention-placement patterns to test.

    Fixed: 9 blocks, 7 SSM + 2 ATTN. We vary where the 2 attention blocks go.
    C(9,2) = 36 possible placements. We test a curated subset plus the reference.
    """
    candidates = []
    n_blocks = 9
    n_attn = 2

    # All 36 possible placements of 2 attention blocks in 9 positions
    all_patterns = []
    for positions in combinations(range(n_blocks), n_attn):
        pattern = ["ssm"] * n_blocks
        for p in positions:
            pattern[p] = "attn"
        name = "".join("A" if b == "attn" else "S" for b in pattern)
        all_patterns.append((name, pattern))

    # Curated subset: reference + evenly spaced + clustered + edge cases
    curated_names = {
        "SSASSASSS",  # reference (positions 2,5)
        "ASSSSSSAS",  # early + late
        "SSSASSSAS",  # mid + late
        "SSSSASSAS",  # late cluster
        "SASSSSSAS",  # early + very late
        "SSSSSSSAA",  # both at end
        "AASSSSSS",   # both at start — wait this is 8 chars
        "AASSSSSSS",  # both at start
        "SSSASSSSA",  # mid + end
        "ASSSASSSS",  # evenly spaced
        "SSSSASSSS",  # single mid (only 1 attn — skip)
    }

    # Actually just test all 36 — each is only 100 steps (~9 min each... too slow)
    # Test 12 representative ones instead
    curated_positions = [
        (0, 1),  # AASSSSSSS — both early
        (0, 4),  # ASSSSSSSS... wait let me just compute
        (0, 8),  # A at start, A at end
        (1, 7),  # near-start, near-end
        (2, 5),  # reference [SSASSASSS]
        (2, 6),  # shifted reference
        (3, 6),  # center-weighted
        (4, 5),  # clustered center
        (4, 8),  # center + end
        (0, 5),  # start + center
        (7, 8),  # SSSSSSSAA — both late
        (3, 5),  # slightly asymmetric center
    ]

    for positions in curated_positions:
        pattern = ["ssm"] * n_blocks
        for p in positions:
            pattern[p] = "attn"
        name = "".join("A" if b == "attn" else "S" for b in pattern)
        candidates.append((name, pattern))

    return candidates


def exp3_pattern_search(steps, batch_size, seq_len, lr, chunks):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Pattern search (attention placement)")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    candidates = generate_candidate_patterns()
    print(f"  Testing {len(candidates)} patterns at {steps} steps each")

    # Use best compress schedule if available
    compress_every = 50
    exp1_path = RECEIPT_DIR / "wo_echo_hybrid_02_compress_schedule.json"
    if exp1_path.exists():
        with open(exp1_path) as f:
            exp1 = json.load(f)
        compress_every = int(exp1.get("best_schedule", 50))

    results = {}
    for i, (name, pattern) in enumerate(candidates):
        print(f"\n  [{i+1}/{len(candidates)}] Pattern: {name}  (compress_every={compress_every})")
        cfg = EchoHybridConfig(block_pattern=pattern)
        losses, trainer = train_born_compressed(cfg, steps, batch_size, lr, chunks, compress_every)
        stats = loss_stats(losses)
        results[name] = {
            "pattern": pattern,
            "attn_positions": [j for j, b in enumerate(pattern) if b == "attn"],
            **stats,
        }
        print(f"  {name}: {stats['first_loss']} -> {stats['last_loss']} (delta={stats['loss_delta']})")

    # Rank by final loss
    ranked = sorted(results.items(), key=lambda x: x[1]["last_loss"])
    print(f"\n  Pattern ranking (by last_loss):")
    print(f"  {'rank':>4} {'pattern':>12} {'last_loss':>10} {'delta':>8} {'attn_pos':>12}")
    print("  " + "-" * 50)
    for rank, (name, r) in enumerate(ranked, 1):
        print(f"  {rank:>4} {name:>12} {r['last_loss']:>10} {r['loss_delta']:>8} {str(r['attn_positions']):>12}")

    best_name, best = ranked[0]
    ref_result = results.get("SSASSASSS", {})

    receipt = {
        "wo": "WO-ECHO-HYBRID-02",
        "experiment": "pattern_search",
        "timestamp": time.strftime("%Y-%m-%d"),
        "training_steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "compress_every": compress_every,
        "n_patterns_tested": len(candidates),
        "results": results,
        "ranking": [{"rank": i+1, "pattern": name, "last_loss": r["last_loss"]} for i, (name, r) in enumerate(ranked)],
        "best_pattern": best_name,
        "best_last_loss": best["last_loss"],
        "reference_pattern": "SSASSASSS",
        "reference_last_loss": ref_result.get("last_loss"),
        "best_vs_reference": round(best["last_loss"] - ref_result.get("last_loss", 0), 4) if ref_result else None,
        "notes": (
            f"Pattern search: {len(candidates)} attention placements in 9-block hybrid. "
            f"Best: {best_name} ({best['last_loss']}). "
            f"Reference SSASSASSS: {ref_result.get('last_loss', '?')}."
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_02_pattern_search.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"RECEIPT: {out_path}")
    return receipt


# ---------------------------------------------------------------------------
# Experiment 4: Per-layer compression
# ---------------------------------------------------------------------------

class PerLayerPhase1Trainer:
    """Born-compressed trainer with different cluster counts for SSM vs ATTN layers."""

    def __init__(
        self,
        model: EchoHybridModel,
        lr: float = 1e-4,
        compress_schedule: int = 100,
        ssm_clusters: int = 256,
        attn_clusters: int = 256,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.compress_schedule = compress_schedule
        self.ssm_clusters = ssm_clusters
        self.attn_clusters = attn_clusters
        self.step_count = 0

        # Determine which layers are SSM vs ATTN
        self.layer_cluster_map = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear) or name == "lm_head":
                continue
            # blocks.N.block.xxx — determine block type from pattern
            is_attn = False
            for i, bt in enumerate(model.cfg.block_pattern):
                if f"blocks.{i}." in name and bt == "attn":
                    is_attn = True
                    break
            self.layer_cluster_map[name] = attn_clusters if is_attn else ssm_clusters

        # Compress with per-layer cluster counts
        self.compressed = self._compress_per_layer()
        self.codebook_init_verified = len(self.compressed) > 0

        self.ste = STEQuantizer(model, self.compressed)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def _compress_per_layer(self):
        compressed = {}
        for name, mod in self.model.named_modules():
            if not isinstance(mod, nn.Linear) or name == "lm_head":
                continue
            n_clusters = self.layer_cluster_map.get(name, 256)
            compressed[name] = compress_linear(mod.weight, n_clusters=n_clusters)
        return compressed

    def train_step(self, input_ids: torch.Tensor) -> float:
        self.model.train()
        input_ids = input_ids.to(self.device)

        self.ste.apply_quantized_weights()
        out = self.model(input_ids=input_ids, labels=input_ids)
        loss = out["loss"]

        self.optimizer.zero_grad()
        loss.backward()

        # Restore shadow before optimizer step so gradients update shadow weights
        self.ste.restore_shadow_weights()
        self.optimizer.step()
        self.step_count += 1

        if self.compress_schedule > 0 and self.step_count % self.compress_schedule == 0:
            self.compressed = self._compress_per_layer()
            self.ste.compressed = self.compressed

        return loss.item()


def exp4_per_layer_compression(steps, batch_size, seq_len, lr, chunks):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Per-layer compression (SSM vs ATTN cluster counts)")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    compress_every = 50
    exp1_path = RECEIPT_DIR / "wo_echo_hybrid_02_compress_schedule.json"
    if exp1_path.exists():
        with open(exp1_path) as f:
            exp1 = json.load(f)
        compress_every = int(exp1.get("best_schedule", 50))

    # Configs to test: (ssm_clusters, attn_clusters)
    configs = [
        (256, 256),   # uniform baseline
        (128, 512),   # fewer SSM clusters, more ATTN
        (512, 128),   # more SSM clusters, fewer ATTN
        (512, 512),   # both high
        (128, 128),   # both low (more aggressive compression)
    ]

    cfg = EchoHybridConfig()
    results = {}

    for ssm_k, attn_k in configs:
        label = f"ssm{ssm_k}_attn{attn_k}"
        print(f"\n  --- {label} ---")

        model = EchoHybridModel(cfg)
        trainer = PerLayerPhase1Trainer(
            model=model,
            lr=lr,
            compress_schedule=compress_every,
            ssm_clusters=ssm_k,
            attn_clusters=attn_k,
            device="cpu",
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
        results[label] = {
            "ssm_clusters": ssm_k,
            "attn_clusters": attn_k,
            **stats,
        }
        print(f"  {label}: {stats['first_loss']} -> {stats['last_loss']} (delta={stats['loss_delta']})")

    # Rank
    ranked = sorted(results.items(), key=lambda x: x[1]["last_loss"])
    print(f"\n  Per-layer compression ranking:")
    print(f"  {'rank':>4} {'config':>20} {'last_loss':>10} {'delta':>8}")
    print("  " + "-" * 46)
    for rank, (label, r) in enumerate(ranked, 1):
        print(f"  {rank:>4} {label:>20} {r['last_loss']:>10} {r['loss_delta']:>8}")

    receipt = {
        "wo": "WO-ECHO-HYBRID-02",
        "experiment": "per_layer_compression",
        "timestamp": time.strftime("%Y-%m-%d"),
        "training_steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "compress_every": compress_every,
        "configs_tested": [{"ssm": s, "attn": a} for s, a in configs],
        "results": results,
        "ranking": [{"rank": i+1, "config": label, "last_loss": r["last_loss"]} for i, (label, r) in enumerate(ranked)],
        "best_config": ranked[0][0],
        "best_last_loss": ranked[0][1]["last_loss"],
        "notes": (
            f"Per-layer compression: different cluster counts for SSM vs ATTN layers. "
            f"Tests whether attention layers need finer codebooks than SSM layers."
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_02_per_layer.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"RECEIPT: {out_path}")
    return receipt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WO-ECHO-HYBRID-02: Experiment stack")
    parser.add_argument("--steps", type=int, default=100, help="Steps per config (except exp2 which is always 500)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--exp", type=str, default="all",
                        help="Which experiment: 1, 2, 3, 4, or 'all'")
    args = parser.parse_args()

    # Load data once — enough for 500 steps at batch_size=2 + pattern search
    max_chunks = max(500, args.steps) * args.batch_size * 20
    chunks = load_wikitext_chunks(seq_len=args.seq_len, max_chunks=max_chunks)

    if args.exp in ("1", "all"):
        exp1_compress_schedule(args.steps, args.batch_size, args.seq_len, args.lr, chunks)

    if args.exp in ("2", "all"):
        exp2_convergence_500(args.batch_size, args.seq_len, args.lr, chunks)

    if args.exp in ("3", "all"):
        exp3_pattern_search(args.steps, args.batch_size, args.seq_len, args.lr, chunks)

    if args.exp in ("4", "all"):
        exp4_per_layer_compression(args.steps, args.batch_size, args.seq_len, args.lr, chunks)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Receipts in: {RECEIPT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
