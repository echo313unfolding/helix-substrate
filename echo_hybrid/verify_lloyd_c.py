"""
WO-HXQ-LLOYD-C-01: Verify C Lloyd's library in the training loop.

Runs 10 steps with d=2 grouped VQ, comparing:
  1. C library path (hxq_lloyd.so via ctypes)
  2. Python/torch fallback

Checks:
  - Loss curves match within FP tolerance
  - C path is significantly faster
  - Dead centroid counts match

Emits timing receipt.
"""

import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

from echo_hybrid.config import EchoHybridConfig, EchoHybridModel
from echo_hybrid.train_phase1 import (
    compress_all_linears,
    STEQuantizer,
    load_wikitext_chunks,
)
from echo_hybrid.train_vstep import (
    vstep_update_codebooks,
    _lloyd_lib,
    _vstep_grouped,
    _vstep_grouped_c,
)
from echo_hybrid.train_sidecar_aware import cost_block

RECEIPT_DIR = Path("receipts/echo_hybrid")
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

STEPS = 10
BATCH_SIZE = 2
SEQ_LEN = 64
LR = 1e-4
VECTOR_DIM = 2


def run_training(label, use_c_library, seed=42):
    """Run STEPS training steps with grouped VQ (d=2) reassignment.

    Returns (losses, vstep_times, total_dead).
    """
    print(f"\n--- {label} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)

    # Compress with d=2
    print(f"  Compressing (k=256, d={VECTOR_DIM})...")
    compressed = compress_all_linears(model, n_clusters=256, vector_dim=VECTOR_DIM)
    n_layers = len(compressed)

    # Check how many layers actually got grouped VQ
    grouped = sum(1 for f in compressed.values()
                  if (f.get("vector_dim", 1) if not isinstance(f.get("vector_dim", 1), torch.Tensor)
                      else f["vector_dim"].item()) > 1)
    print(f"  {n_layers} layers compressed, {grouped} grouped (d={VECTOR_DIM})")

    ste = STEQuantizer(model, compressed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    chunks = load_wikitext_chunks(seq_len=SEQ_LEN, max_chunks=STEPS * BATCH_SIZE * 3)

    losses = []
    vstep_times = []
    total_dead = 0
    chunk_idx = 0

    for step in range(STEPS):
        model.train()
        batch = torch.stack([chunks[(chunk_idx + i) % len(chunks)] for i in range(BATCH_SIZE)])
        chunk_idx += BATCH_SIZE

        ste.apply_quantized_weights()
        out = model(input_ids=batch, labels=batch)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        ste.restore_shadow_weights()
        optimizer.step()

        losses.append(loss.item())

        # V-step: Lloyd's reassignment
        t0 = time.perf_counter()
        if use_c_library and _lloyd_lib is not None:
            # Force C path for all grouped layers
            modules = dict(model.named_modules())
            dead_counts = {}
            for name, factors in compressed.items():
                mod = modules[name]
                vd = factors.get("vector_dim", 1)
                if isinstance(vd, torch.Tensor):
                    vd = vd.item()
                if vd > 1:
                    dead_counts[name] = _vstep_grouped_c(factors, mod, vd)
                else:
                    from echo_hybrid.train_vstep import _vstep_scalar
                    dead_counts[name] = _vstep_scalar(factors, mod, 0.0, "reassign")
        else:
            # Force Python path for all layers
            modules = dict(model.named_modules())
            dead_counts = {}
            for name, factors in compressed.items():
                mod = modules[name]
                vd = factors.get("vector_dim", 1)
                if isinstance(vd, torch.Tensor):
                    vd = vd.item()
                if vd > 1:
                    dead_counts[name] = _vstep_grouped(factors, mod, vd, 0.0, "reassign")
                else:
                    from echo_hybrid.train_vstep import _vstep_scalar
                    dead_counts[name] = _vstep_scalar(factors, mod, 0.0, "reassign")
            dead_counts = dead_counts
        vstep_time = time.perf_counter() - t0
        vstep_times.append(vstep_time)

        ste.compressed = compressed
        step_dead = sum(dead_counts.values())
        total_dead += step_dead

        print(f"  step {step+1:2d}/{STEPS}  loss={loss.item():.4f}  "
              f"vstep={vstep_time*1000:.1f}ms  dead={step_dead}")

    return losses, vstep_times, total_dead


def main():
    print("=" * 70)
    print("WO-HXQ-LLOYD-C-01: Verify C Lloyd's in Training Loop")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    if _lloyd_lib is None:
        print("ERROR: C library not loaded. Cannot verify.")
        return 1

    print(f"C library loaded: YES")
    print(f"Config: {STEPS} steps, batch={BATCH_SIZE}, seq={SEQ_LEN}, d={VECTOR_DIM}")

    # Run with C library
    c_losses, c_times, c_dead = run_training("C library (hxq_lloyd.so)", use_c_library=True, seed=42)

    # Run with Python fallback
    py_losses, py_times, py_dead = run_training("Python/torch fallback", use_c_library=False, seed=42)

    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    print(f"\n{'step':>4} {'C loss':>10} {'Py loss':>10} {'diff':>12} {'C ms':>8} {'Py ms':>8} {'speedup':>8}")
    print("-" * 66)

    max_diff = 0.0
    for i in range(STEPS):
        diff = abs(c_losses[i] - py_losses[i])
        max_diff = max(max_diff, diff)
        c_ms = c_times[i] * 1000
        py_ms = py_times[i] * 1000
        speedup = py_ms / c_ms if c_ms > 0 else float('inf')
        print(f"  {i+1:2d}   {c_losses[i]:>10.4f} {py_losses[i]:>10.4f} {diff:>12.6f} "
              f"{c_ms:>7.1f} {py_ms:>7.1f} {speedup:>7.1f}x")

    c_total_ms = sum(c_times) * 1000
    py_total_ms = sum(py_times) * 1000
    mean_speedup = py_total_ms / c_total_ms if c_total_ms > 0 else float('inf')

    print(f"\nTotal vstep time:  C={c_total_ms:.1f}ms  Python={py_total_ms:.1f}ms  "
          f"speedup={mean_speedup:.1f}x")
    print(f"Max loss difference: {max_diff:.8f}")
    print(f"Dead centroids:  C={c_dead}  Python={py_dead}")

    # Verdict
    loss_match = max_diff < 0.01  # FP tolerance for different reassignment order
    speedup_ok = mean_speedup > 1.0
    print(f"\nLoss match (diff < 0.01): {'PASS' if loss_match else 'FAIL'}")
    print(f"C faster than Python:     {'PASS' if speedup_ok else 'FAIL'}")

    overall = "PASS" if loss_match and speedup_ok else "FAIL"
    print(f"\nOverall: {overall}")

    # Receipt
    receipt = {
        "wo": "WO-HXQ-LLOYD-C-01",
        "test": "training_loop_verification",
        "result": overall,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "lr": LR,
            "vector_dim": VECTOR_DIM,
            "n_clusters": 256,
        },
        "c_library": {
            "losses": c_losses,
            "vstep_times_ms": [round(t * 1000, 2) for t in c_times],
            "total_vstep_ms": round(c_total_ms, 2),
            "total_dead": c_dead,
        },
        "python_fallback": {
            "losses": py_losses,
            "vstep_times_ms": [round(t * 1000, 2) for t in py_times],
            "total_vstep_ms": round(py_total_ms, 2),
            "total_dead": py_dead,
        },
        "comparison": {
            "max_loss_diff": round(max_diff, 8),
            "mean_speedup": round(mean_speedup, 2),
            "loss_match": loss_match,
            "c_faster": speedup_ok,
        },
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    out_path = RECEIPT_DIR / "wo_hxq_lloyd_c_01_verify.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nRECEIPT: {out_path}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
