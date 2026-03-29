#!/usr/bin/env python3
"""
Probe: Does k=512 reduce PPL on Zamba2-1.2B?

Recompresses the 5 lowest-cosine tensors at k=512, runs PPL eval.
If delta drops from +2.90% toward 2%, proceed with full adaptive-k.
If unchanged, STOP — k=512 is dead for this model.

Work Order: WO-ADAPTIVE-K-QUALITY-01
"""

import json
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

# 5 lowest-cosine tensors from Zamba2 stats.json (2026-03-28 compression)
WORST_TENSORS = [
    "model.layers.5.shared_transformer.self_attn.linear_q_adapter_list.2.1.weight",  # cos=0.998708
    "model.layers.5.shared_transformer.self_attn.linear_q_adapter_list.1.1.weight",  # cos=0.998797
    "model.layers.27.mamba.in_proj.weight",                                           # cos=0.998890
    "model.layers.5.shared_transformer.self_attn.linear_k_adapter_list.2.1.weight",  # cos=0.998937
    "model.layers.5.shared_transformer.self_attn.linear_k_adapter_list.1.1.weight",  # cos=0.998971
]

MODEL_DIR = Path.home() / "models" / "zamba2-1.2b"
CDNA_DIR = MODEL_DIR / "cdnav3"

# Known baseline from 2026-03-28 receipt
DENSE_PPL = 5.4583
HELIX_PPL_BASELINE = 5.6168  # +2.90%


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    assert MODEL_DIR.exists(), f"Model not found: {MODEL_DIR}"
    assert CDNA_DIR.exists(), f"CDNA dir not found: {CDNA_DIR}"

    # ── Step 1: Create k_map.json ──
    k_map = {
        "model": "zamba2-1.2b",
        "target_ratio": 3.5,
        "k_default": 256,
        "overrides": {name: 512 for name in WORST_TENSORS},
        "probe": True,
    }
    k_map_path = CDNA_DIR / "k_map_probe.json"
    k_map_path.write_text(json.dumps(k_map, indent=2))
    print(f"k-map: {k_map_path} ({len(WORST_TENSORS)} tensors → k=512)")

    # ── Step 2: Recompress with k-map ──
    # compress.py auto-detects k mismatch and recompresses only affected tensors
    print(f"\n{'='*70}")
    print(f"  Phase 1: Recompress 5 tensors at k=512")
    print(f"{'='*70}")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.compress import compress_model

    compress_model(
        MODEL_DIR,
        dry_run=False,
        force=False,
        k_map_file=k_map_path,
    )

    # ── Step 3: Verify k=512 was applied ──
    print(f"\nVerifying k=512 applied to target tensors...")
    for name in WORST_TENSORS:
        safe = name.replace("/", "_").replace(".", "_")
        meta_path = CDNA_DIR / f"{safe}.cdnav3" / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            k = meta.get("n_clusters", "?")
            idx_dtype = meta.get("index_dtype", "?")
            print(f"  k={k} idx={idx_dtype}  {name}")
        else:
            print(f"  MISSING  {name}")

    # ── Step 4: PPL eval (skip dense — already known) ──
    print(f"\n{'='*70}")
    print(f"  Phase 2: PPL Evaluation (Helix only, dense={DENSE_PPL} known)")
    print(f"{'='*70}")

    import gc
    import torch
    import numpy as np
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
    model = swap_to_helix(model, factors)
    del factors
    gc.collect()
    summary = swap_summary(model)
    model.eval()
    load_s = round(time.time() - t_load, 1)
    print(f"  Loaded: {summary['helix_modules']} HelixLinear, {load_s}s", flush=True)

    # PPL eval
    from tools.eval_ppl_cpu import eval_ppl
    print(f"\n  Evaluating PPL (WikiText-2, seq_len=2048, n_tokens=2048)...", flush=True)
    t_ppl = time.time()
    helix_ppl, n_tok = eval_ppl(model, tokenizer, device="cpu", n_tokens=2048, seq_len=2048)
    ppl_s = round(time.time() - t_ppl, 1)

    delta_pct = round((helix_ppl - DENSE_PPL) / DENSE_PPL * 100, 2)
    improvement = round(HELIX_PPL_BASELINE - helix_ppl, 4)

    print(f"\n{'='*70}")
    print(f"  PROBE RESULTS")
    print(f"{'='*70}")
    print(f"  Dense PPL (known):     {DENSE_PPL}")
    print(f"  Helix PPL (k=256):     {HELIX_PPL_BASELINE} (+2.90%)")
    print(f"  Helix PPL (k=512 x5):  {helix_ppl} ({delta_pct:+.2f}%)")
    print(f"  PPL improvement:       {improvement:+.4f}")
    print(f"  Eval: {n_tok} tokens, {ppl_s}s")

    if delta_pct <= 2.0:
        print(f"\n  VERDICT: PROBE PASSES — k=512 brings delta under 2%")
        print(f"  → Proceed to full adaptive-k allocation (Phase 3-4)")
    elif helix_ppl < HELIX_PPL_BASELINE:
        print(f"\n  VERDICT: PARTIAL — k=512 helps ({improvement:+.4f}) but delta still >{delta_pct:.2f}%")
        print(f"  → Consider upgrading more tensors or k=1024 on worst 2-3")
    else:
        print(f"\n  VERDICT: PROBE FAILS — k=512 does not reduce PPL")
        print(f"  → Accept +2.90% as hybrid VQ cost. Close work order.")
    print(f"{'='*70}")

    # ── Receipt ──
    wall = round(time.time() - t_start, 3)
    receipt = {
        "work_order": "WO-ADAPTIVE-K-QUALITY-01",
        "question": "Does k=512 on 5 worst tensors reduce Zamba2 PPL delta toward 2%?",
        "verdict": "PASS" if delta_pct <= 2.0 else "PARTIAL" if helix_ppl < HELIX_PPL_BASELINE else "FAIL",
        "baseline": {
            "dense_ppl": DENSE_PPL,
            "helix_ppl_k256": HELIX_PPL_BASELINE,
            "delta_pct_k256": 2.90,
        },
        "probe": {
            "helix_ppl_k512x5": helix_ppl,
            "delta_pct": delta_pct,
            "improvement": improvement,
            "n_tokens": n_tok,
            "upgraded_tensors": WORST_TENSORS,
        },
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
    receipt_path = receipts_dir / f"k512_probe_zamba2_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
