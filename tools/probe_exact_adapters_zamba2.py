#!/usr/bin/env python3
"""
Probe: Does storing shared_transformer LoRA adapters exact reduce PPL?

Hypothesis: The shared_transformer is used by 6 hybrid layers (5,11,17,23,29,35).
VQ error in its 48 LoRA adapters gets amplified 6x. Storing them exact eliminates
that amplification at a cost of ~95 MB (ratio drops from 4.0x to ~3.7x).

Method: Temporarily hide the 48 adapter .cdnav3 dirs so load_cdna_factors() skips
them. Those modules stay as dense nn.Linear with original FP32 weights.

Work Order: WO-ADAPTIVE-K-QUALITY-01
"""

import gc
import json
import platform
import resource
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path.home() / "models" / "zamba2-1.2b"
CDNA_DIR = MODEL_DIR / "cdnav3"

# Known baseline
DENSE_PPL = 5.4583
HELIX_PPL_BASELINE = 5.6168  # +2.90%


def find_adapter_dirs(cdna_dir: Path) -> list[Path]:
    """Find all shared_transformer adapter .cdnav3 directories."""
    adapter_dirs = []
    for d in sorted(cdna_dir.glob("*.cdnav3")):
        if "shared_transformer" in d.name and "adapter" in d.name:
            adapter_dirs.append(d)
    return adapter_dirs


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    assert MODEL_DIR.exists(), f"Model not found: {MODEL_DIR}"
    assert CDNA_DIR.exists(), f"CDNA dir not found: {CDNA_DIR}"

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # ── Step 1: Find and hide adapter dirs ──
    adapter_dirs = find_adapter_dirs(CDNA_DIR)
    print(f"Found {len(adapter_dirs)} shared_transformer adapter tensors")

    # Calculate size impact
    total_exact_bytes = 0
    total_compressed_bytes = 0
    adapter_names = []
    for d in adapter_dirs:
        stats_path = d / "stats.json"
        if stats_path.exists():
            s = json.loads(stats_path.read_text())
            total_exact_bytes += s.get("original_bytes", 0)
            total_compressed_bytes += s.get("compressed_bytes", 0)
        meta_path = d / "meta.json"
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            adapter_names.append(m.get("tensor_name", d.name))

    extra_cost_mb = (total_exact_bytes - total_compressed_bytes) / 1e6
    print(f"  Adapter exact size:      {total_exact_bytes / 1e6:.1f} MB")
    print(f"  Adapter compressed size: {total_compressed_bytes / 1e6:.1f} MB")
    print(f"  Extra cost if exact:     {extra_cost_mb:.1f} MB")

    # Move adapter dirs to temp location
    stash_dir = CDNA_DIR / "_adapter_stash"
    stash_dir.mkdir(exist_ok=True)

    print(f"\n  Stashing {len(adapter_dirs)} adapter dirs → {stash_dir}")
    moved = []
    for d in adapter_dirs:
        dest = stash_dir / d.name
        shutil.move(str(d), str(dest))
        moved.append((dest, d))

    try:
        # ── Step 2: Load model with adapters as dense nn.Linear ──
        print(f"\n{'='*70}")
        print(f"  PPL Evaluation: {len(moved)} adapters EXACT, rest HelixLinear")
        print(f"{'='*70}")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from helix_substrate.helix_linear import load_cdna_factors, swap_to_helix, swap_summary

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        print(f"\n  Loading model + HelixLinear factors...", flush=True)
        t_load = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float32, trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        factors = load_cdna_factors(CDNA_DIR, model=model)
        n_factors = len(factors)
        model = swap_to_helix(model, factors)
        del factors
        gc.collect()
        summary = swap_summary(model)
        model.eval()
        load_s = round(time.time() - t_load, 1)
        print(f"  Loaded: {summary['helix_modules']} HelixLinear + "
              f"{summary['linear_modules']} nn.Linear, {load_s}s", flush=True)
        print(f"  (Expected: ~{136 - len(moved)} HelixLinear, ~{len(moved)} adapters stay dense)")

        # ── Step 3: PPL eval ──
        from tools.eval_ppl_cpu import eval_ppl
        print(f"\n  Evaluating PPL (WikiText-2, seq_len=2048, n_tokens=2048)...", flush=True)
        t_ppl = time.time()
        helix_ppl, n_tok = eval_ppl(model, tokenizer, device="cpu",
                                     n_tokens=2048, seq_len=2048)
        ppl_s = round(time.time() - t_ppl, 1)

        delta_pct = round((helix_ppl - DENSE_PPL) / DENSE_PPL * 100, 2)
        improvement = round(HELIX_PPL_BASELINE - helix_ppl, 4)

        del model
        gc.collect()

    finally:
        # ── Step 4: Restore adapter dirs ──
        print(f"\n  Restoring {len(moved)} adapter dirs...")
        for src, dst in moved:
            shutil.move(str(src), str(dst))
        if stash_dir.exists():
            stash_dir.rmdir()
        print(f"  Restored.")

    # ── Results ──
    # Estimate new compression ratio with adapters exact
    # Current: all compressed at 4.0x (1.149 GB compressed from 4.594 GB dense)
    # New: adapters exact + rest compressed
    current_total_compressed = 1.149e9  # from manifest
    total_dense = 4.594e9
    new_compressed = current_total_compressed - total_compressed_bytes + total_exact_bytes
    new_ratio = total_dense / new_compressed

    print(f"\n{'='*70}")
    print(f"  PROBE RESULTS: Exact Adapters")
    print(f"{'='*70}")
    print(f"  Dense PPL (known):       {DENSE_PPL}")
    print(f"  Helix PPL (all k=256):   {HELIX_PPL_BASELINE} (+2.90%)")
    print(f"  Helix PPL (adapters exact): {helix_ppl} ({delta_pct:+.2f}%)")
    print(f"  PPL improvement:         {improvement:+.4f}")
    print(f"  Eval: {n_tok} tokens, {ppl_s}s")
    print(f"")
    print(f"  Compression impact:")
    print(f"    Adapters exact:   {len(moved)} tensors ({total_exact_bytes / 1e6:.1f} MB)")
    print(f"    Adapters were:    {total_compressed_bytes / 1e6:.1f} MB compressed")
    print(f"    Extra cost:       {extra_cost_mb:.1f} MB")
    print(f"    New ratio:        {new_ratio:.2f}x (was 4.00x)")

    if delta_pct <= 2.0:
        verdict = "PASS"
        print(f"\n  VERDICT: PASS — exact adapters bring delta under 2%!")
        print(f"  → Update should_compress() to store shared_transformer adapters exact")
    elif helix_ppl < HELIX_PPL_BASELINE:
        verdict = "PARTIAL"
        print(f"\n  VERDICT: PARTIAL — improvement of {improvement:+.4f} but delta still {delta_pct:.2f}%")
        print(f"  → Consider also storing main shared_transformer weights exact")
    else:
        verdict = "FAIL"
        print(f"\n  VERDICT: FAIL — exact adapters don't help PPL")
        print(f"  → Accept +2.90% as the VQ hybrid cost")
    print(f"{'='*70}")

    # ── Receipt ──
    wall = round(time.time() - t_start, 3)
    receipt = {
        "work_order": "WO-ADAPTIVE-K-QUALITY-01",
        "question": "Does storing 48 shared_transformer LoRA adapters exact reduce Zamba2 PPL?",
        "verdict": verdict,
        "baseline": {
            "dense_ppl": DENSE_PPL,
            "helix_ppl_k256": HELIX_PPL_BASELINE,
            "delta_pct_k256": 2.90,
        },
        "probe": {
            "helix_ppl_adapters_exact": helix_ppl,
            "delta_pct": delta_pct,
            "improvement": improvement,
            "n_tokens": n_tok,
            "n_adapters_exact": len(moved),
            "adapter_exact_bytes": total_exact_bytes,
            "adapter_compressed_bytes": total_compressed_bytes,
            "extra_cost_bytes": total_exact_bytes - total_compressed_bytes,
            "new_ratio": round(new_ratio, 2),
        },
        "adapter_names": adapter_names,
        "cost": {
            "wall_time_s": wall,
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipts_dir = Path(__file__).resolve().parent.parent / "receipts" / "adaptive_k"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"exact_adapters_probe_zamba2_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
