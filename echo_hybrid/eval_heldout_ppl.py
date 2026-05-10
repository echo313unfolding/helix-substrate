"""
WO-ECHO-HYBRID-06: Held-out perplexity evaluation.

Trains 4 configs (dense, d=1+reassign, d=2+reassign, d=4+reassign) for 500 steps,
then evaluates WikiText-2 validation perplexity on each.

Each config writes its own receipt as it completes.
Final summary receipt written after all configs.

Usage:
    python3 -m echo_hybrid.eval_heldout_ppl [--steps 500] [--configs dense,d1,d2,d4]
"""

from __future__ import annotations

import argparse
import json
import math
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

from echo_hybrid.config import EchoHybridConfig, EchoHybridModel
from echo_hybrid.train_phase1 import (
    compress_all_linears,
    STEQuantizer,
    Phase1Trainer,
    load_wikitext_chunks,
)

RECEIPT_DIR = Path("receipts/echo_hybrid")
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Held-out perplexity (WikiText-2 validation)
# ---------------------------------------------------------------------------

def eval_ppl_dense(model, seq_len=64, max_chunks=500) -> float:
    """Perplexity on WikiText-2 validation for a dense (uncompressed) model."""
    chunks = _load_val_chunks(seq_len, max_chunks)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(len(chunks)):
            batch = chunks[i:i+1]
            out = model(input_ids=batch, labels=batch)
            total_loss += out["loss"].item() * (seq_len - 1)
            total_tokens += seq_len - 1
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def eval_ppl_compressed(model, compressed, seq_len=64, max_chunks=500) -> float:
    """Perplexity on WikiText-2 validation under compressed forward."""
    chunks = _load_val_chunks(seq_len, max_chunks)
    ste = STEQuantizer(model, compressed)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(len(chunks)):
            batch = chunks[i:i+1]
            ste.apply_quantized_weights()
            out = model(input_ids=batch, labels=batch)
            ste.restore_shadow_weights()
            total_loss += out["loss"].item() * (seq_len - 1)
            total_tokens += seq_len - 1
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


_val_chunks_cache = {}

def _load_val_chunks(seq_len, max_chunks):
    key = (seq_len, max_chunks)
    if key not in _val_chunks_cache:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = "\n\n".join([x for x in ds["text"] if x.strip()])
        tokens = tokenizer.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long)
        n_chunks = min(max_chunks, len(tokens) // seq_len)
        _val_chunks_cache[key] = tokens[:n_chunks * seq_len].reshape(n_chunks, seq_len)
        print(f"Loaded {n_chunks} validation chunks of {seq_len} tokens.")
    return _val_chunks_cache[key]


# ---------------------------------------------------------------------------
# Train + eval for each config
# ---------------------------------------------------------------------------

def run_config(label, vector_dim, vstep_mode, steps, batch_size, seq_len, lr, train_chunks):
    """Train one config and return (final_train_loss, held_out_ppl, wall_time)."""
    print(f"\n{'='*60}")
    print(f"CONFIG: {label}")
    print(f"{'='*60}")

    t0 = time.time()
    cpu0 = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)

    is_dense = (label == "dense")

    if is_dense:
        # Dense training: no compression
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        losses = []
        chunk_idx = 0
        for step in range(steps):
            batch = torch.stack([train_chunks[(chunk_idx + i) % len(train_chunks)] for i in range(batch_size)])
            chunk_idx += batch_size
            out = model(input_ids=batch, labels=batch)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (step + 1) % 50 == 0 or step == 0:
                print(f"  step {step+1:4d}/{steps}  loss={loss.item():.4f}")
        compressed = None
    else:
        # Born-compressed with reassign
        trainer = Phase1Trainer(
            model=model, lr=lr, compress_schedule=0,
            n_clusters=256, vector_dim=vector_dim,
            device="cpu", vstep_mode=vstep_mode,
        )
        losses = []
        chunk_idx = 0
        for step in range(steps):
            batch = torch.stack([train_chunks[(chunk_idx + i) % len(train_chunks)] for i in range(batch_size)])
            chunk_idx += batch_size
            loss = trainer.train_step(batch)
            losses.append(loss)
            if (step + 1) % 50 == 0 or step == 0:
                print(f"  step {step+1:4d}/{steps}  loss={loss:.4f}")
        compressed = trainer.compressed

    train_time = time.time() - t0
    final_train_loss = losses[-1]
    print(f"  Training done: {final_train_loss:.4f} ({train_time:.0f}s)")

    # Eval held-out perplexity
    print(f"  Evaluating held-out perplexity...")
    eval_t0 = time.time()
    if is_dense:
        ppl = eval_ppl_dense(model, seq_len=seq_len)
    else:
        ppl = eval_ppl_compressed(model, compressed, seq_len=seq_len)
    eval_time = time.time() - eval_t0
    print(f"  PPL: {ppl:.2f} (eval took {eval_time:.0f}s)")

    wall = time.time() - t0
    cost = {
        "wall_time_s": round(wall, 3),
        "cpu_time_s": round(time.process_time() - cpu0, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    result = {
        "label": label,
        "vector_dim": vector_dim,
        "vstep_mode": vstep_mode,
        "final_train_loss": round(final_train_loss, 4),
        "heldout_ppl": round(ppl, 2),
        "heldout_avg_loss": round(math.log(ppl), 4),
        "training_steps": steps,
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        "cost": cost,
    }

    # Write per-config receipt
    receipt_path = RECEIPT_DIR / f"wo_echo_hybrid_06_{label}.json"
    with open(receipt_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  RECEIPT: {receipt_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WO-ECHO-HYBRID-06: Held-out perplexity eval")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--configs", type=str, default="dense,d1,d2,d4",
                        help="Comma-separated configs to run")
    args = parser.parse_args()

    t_start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    config_map = {
        "dense": ("dense", 1, "none"),
        "d1":    ("d1_reassign", 1, "reassign"),
        "d2":    ("d2_reassign", 2, "reassign"),
        "d4":    ("d4_reassign", 4, "reassign"),
    }

    configs_to_run = [c.strip() for c in args.configs.split(",")]
    print(f"Configs: {configs_to_run}")
    print(f"Steps: {args.steps}, batch={args.batch_size}, seq={args.seq_len}")

    # Load training data once
    max_chunks = args.steps * args.batch_size * 2
    train_chunks = load_wikitext_chunks(seq_len=args.seq_len, max_chunks=max_chunks)

    results = {}
    for key in configs_to_run:
        if key not in config_map:
            print(f"Unknown config: {key}, skipping")
            continue
        label, vd, vstep = config_map[key]
        result = run_config(label, vd, vstep, args.steps, args.batch_size,
                           args.seq_len, args.lr, train_chunks)
        results[label] = result

    # Summary table
    print(f"\n{'='*60}")
    print("HELD-OUT PERPLEXITY COMPARISON")
    print(f"{'='*60}")
    print(f"{'config':<20} {'d':>3} {'train_loss':>12} {'val_ppl':>10} {'val_loss':>10}")
    print("-" * 60)

    dense_ppl = results.get("dense", {}).get("heldout_ppl", None)
    for label, r in results.items():
        ppl = r["heldout_ppl"]
        gap = f"(+{ppl - dense_ppl:.1f})" if dense_ppl and label != "dense" else ""
        print(f"{label:<20} {r['vector_dim']:>3} {r['final_train_loss']:>12} "
              f"{ppl:>10.2f} {r['heldout_avg_loss']:>10.4f} {gap}")

    # Write summary receipt
    summary = {
        "wo": "WO-ECHO-HYBRID-06",
        "experiment": "heldout_ppl_comparison",
        "timestamp": time.strftime("%Y-%m-%d"),
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "eval_split": "wikitext-2-raw-v1/validation",
        "configs": results,
        "notes": (
            "Held-out perplexity comparison. Each config trained for the same number "
            "of steps on WikiText-2 train, then evaluated on WikiText-2 validation. "
            "Perplexity = exp(avg cross-entropy loss per token)."
        ),
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_06_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSUMMARY RECEIPT: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
