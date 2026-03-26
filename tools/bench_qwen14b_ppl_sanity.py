#!/usr/bin/env python3
"""
WO-QWEN14B-PPL-SANITY-01: Qwen 14B PPL Sanity Check
=====================================================
Verifies Qwen2.5-14B-Instruct compression after N_BLOCKS fix (40→48).
CPU-only (14B in BF16 is ~28 GB RAM).

Steps:
  1. Load Qwen 14B FP32 on CPU
  2. Compute baseline PPL on WikiText-2 (EVAL_TOKENS=4096)
  3. Load compressed via load_cdna_factors() + swap_to_helix()
  4. Compute compressed PPL, compare
  5. PASS if delta < 2%

Expected runtime: 2-4 hours (14B CPU PPL is slow).

Output: receipts/qwen14b_sanity/
"""

import gc
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR = Path.home() / "models" / "qwen2.5-14b-instruct"
EVAL_TOKENS = 4096
SEQ_LEN = 512  # Shorter seq_len for CPU to avoid OOM

# Auto-detect layer count from config.json
_config = json.load(open(MODEL_DIR / "config.json"))
N_BLOCKS = _config.get("num_hidden_layers") or _config.get("n_layer")
if N_BLOCKS is None:
    raise ValueError(f"Cannot detect layer count from {MODEL_DIR / 'config.json'}")


def compute_perplexity_cpu(model, tokenizer, n_tokens=EVAL_TOKENS, seq_len=SEQ_LEN):
    """Compute perplexity on WikiText-2 — CPU-only, loss on CPU."""
    from datasets import load_dataset

    print("  Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens + 1)
    input_ids = encodings.input_ids[:, :n_tokens + 1]

    nlls = []
    n_evaluated = 0

    print(f"  Evaluating PPL (seq_len={seq_len}, n_tokens={n_tokens})...")
    model.eval()
    with torch.no_grad():
        for i in range(0, input_ids.shape[1] - 1, seq_len):
            end = min(i + seq_len + 1, input_ids.shape[1])
            chunk = input_ids[:, i:end]
            if chunk.shape[1] < 2:
                break

            outputs = model(input_ids=chunk[:, :-1])
            logits = outputs.logits.float()

            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
            )
            n_tok = shift_labels.numel()
            nlls.append(loss.item() * n_tok)
            n_evaluated += n_tok

            if n_evaluated % 1024 == 0:
                print(f"    {n_evaluated}/{n_tokens} tokens...", flush=True)

            if n_evaluated >= n_tokens:
                break

    ppl = np.exp(sum(nlls) / n_evaluated)
    print(f"  PPL = {ppl:.4f} ({n_evaluated} tokens)")
    return round(float(ppl), 4), n_evaluated


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 70)
    print("  WO-QWEN14B-PPL-SANITY-01: Qwen 14B PPL Sanity Check")
    print(f"  Auto-detected: {N_BLOCKS} blocks (from config.json)")
    print("  Running on CPU (14B BF16 ~28 GB RAM)")
    print("=" * 70)

    assert MODEL_DIR.exists(), f"Model not found: {MODEL_DIR}"

    cdna_dir = MODEL_DIR / "cdnav3"
    assert cdna_dir.exists(), f"CDNA dir not found: {cdna_dir}"

    # Count compressed tensors
    n_cdna = len(list(cdna_dir.glob("*.cdnav3")))
    expected = N_BLOCKS * 7  # 7 tensor types for Qwen
    print(f"  CDNA tensors: {n_cdna} (expected {expected})")
    if n_cdna < expected:
        print(f"  WARNING: Only {n_cdna}/{expected} tensors compressed. Run compress_qwen14b_instruct.py first.")
        sys.exit(1)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from helix_substrate.helix_linear import load_cdna_factors, swap_to_helix, swap_summary

    print("\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    # ── Phase A: Baseline PPL ──
    print("\n  --- Phase A: Baseline (FP32, CPU) ---")
    print("  Loading model (BF16, low_cpu_mem_usage=True)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"  Model loaded: {load_time:.1f}s")

    ppl_baseline, n_tokens_base = compute_perplexity_cpu(model, tokenizer)
    print(f"  Baseline PPL (BF16): {ppl_baseline:.4f}")

    del model
    gc.collect()

    # ── Phase B: Compressed PPL ──
    print("\n  --- Phase B: Compressed (HelixLinear, CPU) ---")
    print("  Loading model shell (BF16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    shell_time = time.time() - t0

    print(f"  Loading CDNA v3 factors from {cdna_dir}...")
    t0 = time.time()
    factors = load_cdna_factors(cdna_dir, model=model)
    factor_time = time.time() - t0
    print(f"  Factors loaded: {len(factors)} tensors, {factor_time:.1f}s")

    print("  Swapping to HelixLinear...")
    t0 = time.time()
    model = swap_to_helix(model, factors)
    swap_time = time.time() - t0
    summary = swap_summary(model)
    print(f"  Swap complete: {summary['helix_modules']} HelixLinear, "
          f"{summary['linear_modules']} nn.Linear remaining, {swap_time:.1f}s")

    model.eval()

    ppl_compressed, n_tokens_comp = compute_perplexity_cpu(model, tokenizer)
    print(f"  Compressed PPL: {ppl_compressed:.4f}")

    del model
    gc.collect()

    # ── Results ──
    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    delta_pct = (ppl_compressed - ppl_baseline) / ppl_baseline * 100
    verdict = "PASS" if abs(delta_pct) < 2.0 else "FAIL"

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — WO-QWEN14B-PPL-SANITY-01")
    print(f"{'=' * 70}")
    print(f"  Model:          Qwen2.5-14B-Instruct ({N_BLOCKS} blocks)")
    print(f"  CDNA tensors:   {n_cdna}")
    print(f"  Baseline PPL:   {ppl_baseline:.4f}")
    print(f"  Compressed PPL: {ppl_compressed:.4f}")
    print(f"  Delta:          {delta_pct:+.2f}%")
    print(f"  Verdict:        {verdict}")
    print(f"  Wall time:      {wall:.0f}s ({wall/3600:.1f}h)")
    print(f"{'=' * 70}")

    receipt = {
        "work_order": "WO-QWEN14B-PPL-SANITY-01",
        "question": "Does Qwen2.5-14B-Instruct compressed PPL stay under 2% after N_BLOCKS fix?",
        "model": "Qwen2.5-14B-Instruct",
        "n_blocks": N_BLOCKS,
        "n_cdna_tensors": n_cdna,
        "expected_tensors": expected,
        "eval_dataset": "wikitext-2-raw-v1 (validation)",
        "eval_tokens": EVAL_TOKENS,
        "seq_len": SEQ_LEN,
        "baseline_ppl": ppl_baseline,
        "compressed_ppl": ppl_compressed,
        "delta_pct": round(delta_pct, 4),
        "verdict": verdict,
        "swap_summary": {
            "helix_modules": summary["helix_modules"],
            "linear_modules": summary["linear_modules"],
        },
        "cost": {
            "wall_time_s": round(wall, 3),
            "cpu_time_s": round(cpu, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipts_dir = Path(__file__).parent.parent / "receipts" / "qwen14b_sanity"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"qwen14b_ppl_sanity_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
