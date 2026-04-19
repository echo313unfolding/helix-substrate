"""
WO-ECHO-HYBRID-01c: Negative controls for born-compressed training.

Three baselines run in sequence:
1. Dense — same model, no compression at all
2. Post-training HXQ — train dense, compress after, eval
3. Broken interleave — shuffled block pattern, born-compressed

Each produces its own receipt JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from echo_hybrid.config import EchoHybridConfig, EchoHybridModel
from echo_hybrid.train_phase1 import (
    compress_all_linears,
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


# ---------------------------------------------------------------------------
# Baseline 1: Dense (no compression)
# ---------------------------------------------------------------------------

def run_dense(steps: int, batch_size: int, seq_len: int, lr: float, chunks: torch.Tensor):
    print("=" * 60)
    print("BASELINE 1: Dense (no compression)")
    print("=" * 60)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)
    print(model)

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
            print(f"  step {step+1:4d}/{steps}  loss={loss.item():.4f}")

    stats = loss_stats(losses)
    print(f"Loss: {stats['first_loss']} -> {stats['last_loss']} (delta={stats['loss_delta']})")

    receipt = {
        "wo": "WO-ECHO-HYBRID-01c",
        "method": "dense",
        "timestamp": time.strftime("%Y-%m-%d"),
        "status": "PASS",
        "model": f"EchoHybrid-{cfg.n_blocks}block-{cfg.hidden_size}d",
        "block_pattern": cfg.block_pattern,
        "training_steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "compressed": False,
        **stats,
        "notes": "Dense baseline — identical model, no compression, standard training.",
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_01c_dense.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"RECEIPT: {out_path}\n")
    return receipt


# ---------------------------------------------------------------------------
# Baseline 2: Post-training HXQ (train dense, compress after, eval)
# ---------------------------------------------------------------------------

def run_post_train_hxq(steps: int, batch_size: int, seq_len: int, lr: float, chunks: torch.Tensor):
    print("=" * 60)
    print("BASELINE 2: Post-training HXQ (train dense, compress after)")
    print("=" * 60)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)
    print(model)

    # Phase A: train dense for `steps` steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    train_losses = []
    chunk_idx = 0
    print(f"\n  Phase A: Dense training ({steps} steps)...")
    for step in range(steps):
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(batch_size)])
        chunk_idx += batch_size

        out = model(input_ids=batch, labels=batch)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (step + 1) % 10 == 0 or step == 0:
            print(f"    step {step+1:4d}/{steps}  loss={loss.item():.4f}")

    dense_last_loss = train_losses[-1]
    print(f"  Dense training done. Final loss: {dense_last_loss:.4f}")

    # Phase B: compress the trained model
    print(f"\n  Phase B: Compressing trained weights...")
    compressed = compress_all_linears(model, n_clusters=256)
    print(f"  Compressed {len(compressed)} layers.")

    # Phase C: eval with compressed weights (10 steps, no gradient)
    print(f"\n  Phase C: Eval with compressed weights (10 steps)...")
    ste = STEQuantizer(model, compressed)
    model.eval()

    eval_losses = []
    for step in range(10):
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(batch_size)])
        chunk_idx += batch_size

        ste.apply_quantized_weights()
        with torch.no_grad():
            out = model(input_ids=batch, labels=batch)
        ste.restore_shadow_weights()

        eval_losses.append(out["loss"].item())
        print(f"    eval step {step+1}/10  loss={out['loss'].item():.4f}")

    post_hxq_loss = np.mean(eval_losses)
    print(f"  Post-HXQ eval loss (mean): {post_hxq_loss:.4f}")
    print(f"  Compression degradation: {post_hxq_loss - dense_last_loss:+.4f}")

    receipt = {
        "wo": "WO-ECHO-HYBRID-01c",
        "method": "post_train_hxq",
        "timestamp": time.strftime("%Y-%m-%d"),
        "status": "PASS",
        "model": f"EchoHybrid-{cfg.n_blocks}block-{cfg.hidden_size}d",
        "block_pattern": cfg.block_pattern,
        "training_steps": steps,
        "eval_steps": 10,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dense_phase": {
            "first_loss": round(train_losses[0], 4),
            "last_loss": round(train_losses[-1], 4),
            "loss_delta": round(train_losses[0] - train_losses[-1], 4),
        },
        "first_loss": round(train_losses[0], 4),
        "last_loss": round(post_hxq_loss, 4),
        "min_loss": round(min(train_losses), 4),
        "loss_delta": round(train_losses[0] - post_hxq_loss, 4),
        "trend_down": bool(post_hxq_loss < train_losses[0]),
        "post_hxq_eval_loss": round(post_hxq_loss, 4),
        "compression_degradation": round(post_hxq_loss - dense_last_loss, 4),
        "n_compressed_layers": len(compressed),
        "loss_curve": [round(l, 4) for l in train_losses],
        "eval_losses": [round(l, 4) for l in eval_losses],
        "notes": (
            "Post-training HXQ: trained dense for 100 steps, then compressed, "
            "then evaluated compressed model for 10 steps. "
            "Compression degradation = post_hxq_eval_loss - dense_final_loss."
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_01c_posttrain.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"RECEIPT: {out_path}\n")
    return receipt


# ---------------------------------------------------------------------------
# Baseline 3: Broken interleave (shuffled block pattern, born-compressed)
# ---------------------------------------------------------------------------

def run_broken_interleave(steps: int, batch_size: int, seq_len: int, lr: float, chunks: torch.Tensor):
    print("=" * 60)
    print("BASELINE 3: Broken interleave (shuffled pattern, born-compressed)")
    print("=" * 60)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Shuffle the block pattern with a fixed seed for reproducibility
    original_pattern = ["ssm", "ssm", "attn", "ssm", "ssm", "attn", "ssm", "ssm", "ssm"]
    rng = random.Random(42)
    broken_pattern = original_pattern.copy()
    rng.shuffle(broken_pattern)
    print(f"  Original pattern: {original_pattern}")
    print(f"  Broken pattern:   {broken_pattern}")

    cfg = EchoHybridConfig(block_pattern=broken_pattern)
    model = EchoHybridModel(cfg)
    print(model)

    # Born-compressed training (same as Phase1Trainer)
    trainer = Phase1Trainer(
        model=model,
        lr=lr,
        compress_schedule=100,
        n_clusters=256,
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
            print(f"  step {step+1:4d}/{steps}  loss={loss:.4f}")

    stats = loss_stats(losses)
    print(f"Loss: {stats['first_loss']} -> {stats['last_loss']} (delta={stats['loss_delta']})")

    receipt = {
        "wo": "WO-ECHO-HYBRID-01c",
        "method": "broken_interleave",
        "timestamp": time.strftime("%Y-%m-%d"),
        "status": "PASS",
        "model": f"EchoHybrid-{cfg.n_blocks}block-{cfg.hidden_size}d",
        "block_pattern": broken_pattern,
        "original_pattern": original_pattern,
        "training_steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "compressed": True,
        "compressed_from_step": 0,
        "codebook_init_verified": trainer.codebook_init_verified,
        **stats,
        "notes": (
            f"Broken interleave: block pattern shuffled with seed=42. "
            f"Born-compressed (same as 01b). Tests whether the specific "
            f"SSM-heavy pattern matters or any interleaving works."
        ),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_01c_broken.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"RECEIPT: {out_path}\n")
    return receipt


# ---------------------------------------------------------------------------
# Main — run all three in sequence
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WO-ECHO-HYBRID-01c: Negative controls")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Load data once, reuse for all three
    chunks = load_wikitext_chunks(seq_len=args.seq_len, max_chunks=args.steps * args.batch_size * 4)

    # Run all three baselines
    r_dense = run_dense(args.steps, args.batch_size, args.seq_len, args.lr, chunks)
    r_post = run_post_train_hxq(args.steps, args.batch_size, args.seq_len, args.lr, chunks)
    r_broken = run_broken_interleave(args.steps, args.batch_size, args.seq_len, args.lr, chunks)

    # Load 01b receipt for comparison
    r_born = {}
    born_path = RECEIPT_DIR / "wo_echo_hybrid_01b.json"
    if born_path.exists():
        with open(born_path) as f:
            r_born = json.load(f)

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    rows = [
        ("born_compressed", r_born.get("first_loss", "?"), r_born.get("final_loss", "?"),
         round(r_born.get("first_loss", 0) - r_born.get("final_loss", 0), 4) if r_born else "?",
         r_born.get("trend_decreasing", "?")),
        ("dense", r_dense["first_loss"], r_dense["last_loss"], r_dense["loss_delta"], r_dense["trend_down"]),
        ("post_train_hxq", r_post["first_loss"], r_post["last_loss"], r_post["loss_delta"], r_post["trend_down"]),
        ("broken_interleave", r_broken["first_loss"], r_broken["last_loss"], r_broken["loss_delta"], r_broken["trend_down"]),
    ]
    print(f"{'method':<22} {'first_loss':>10} {'last_loss':>10} {'delta':>8} {'trend_down':>10}")
    print("-" * 70)
    for name, fl, ll, d, td in rows:
        print(f"{name:<22} {fl:>10} {ll:>10} {d:>8} {td!s:>10}")

    print(f"\nAll receipts in: {RECEIPT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
