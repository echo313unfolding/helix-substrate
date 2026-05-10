#!/usr/bin/env python3
"""
HXQ_AFFINE_4 Edge Probe: Can affine correction rescue 4-bit (k=16) for edge?

Two phases:
  Phase 1 (fast): Tensor-level cosine on all 154 TinyLlama layers
    - Gate: mean cosine >= 0.999 AND min cosine >= 0.995
    - If FAIL, skip Phase 2 (no point measuring PPL)

  Phase 2 (slow): Model-level PPL on WikiText-2
    - Gate: PPL delta <= +3% vs FP32 dense
    - Compare against raw k=16 (+9.34% prior receipt)

Strategies tested:
  A. VQ-256 global (k=256, production HXQ baseline)
  B. Affine6 g128 (k=64, per-group-128 uniform — production HXQ_AFFINE_6)
  C. Affine4 g128 (k=16, per-group-128 uniform — THE MAIN TEST)
  D. Affine4 g64  (k=16, per-group-64 uniform — tighter groups)
  E. Affine4 g32  (k=16, per-group-32 uniform — maximum locality)
  F. VQ16 g128 k-means (k=16, per-group-128 k-means — VQ variant)

Work Order: WO-HXQ-AFFINE-4-EDGE-PROBE
"""

import gc
import json
import math
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

MODEL_DIR = Path.home() / "models" / "tinyllama-dense"
TOKENIZER_DIR = Path.home() / "models" / "tinyllama-1.1b-chat-v1.0"
SEQ_LEN = 512
MAX_EVAL_TOKENS = 20_000  # CPU eval — 20K tokens is enough for relative PPL

# TinyLlama layer patterns (22 layers × 7 types = 154 tensors)
HF_PATTERNS = {
    "q_proj":    "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj":    "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj":    "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj":    "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj":   "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}
N_LAYERS = 22

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def cosine_sim(a, b):
    a_f, b_f = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    dot = np.dot(a_f, b_f)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))


def kmeans_1d(data, k=16, max_iters=10):
    centroids = np.percentile(data, np.linspace(0, 100, k)).astype(np.float32)
    for _ in range(max_iters):
        indices = assign_nearest(data, centroids)
        for c in range(k):
            mask = indices == c
            if mask.any():
                centroids[c] = data[mask].mean()
    return centroids


def assign_nearest(flat, codebook, chunk_size=500_000):
    indices = np.empty(len(flat), dtype=np.uint8)
    for s in range(0, len(flat), chunk_size):
        e = min(s + chunk_size, len(flat))
        dists = np.abs(flat[s:e, None] - codebook[None, :])
        indices[s:e] = np.argmin(dists, axis=1).astype(np.uint8)
    return indices


# ---------------------------------------------------------------
# Encoding strategies (tensor-level)
# ---------------------------------------------------------------

def encode_vq_global(tensor_2d, k=256):
    """Global k-means VQ."""
    flat = tensor_2d.ravel().astype(np.float32)
    sample = flat if len(flat) <= 500_000 else flat[np.random.RandomState(42).choice(len(flat), 500_000, replace=False)]
    codebook = kmeans_1d(sample, k=k)
    indices = assign_nearest(flat, codebook)
    recon = codebook[indices].reshape(tensor_2d.shape)
    bpw = np.log2(k) + k * 32 / len(flat)  # index bits + codebook overhead
    return recon, {"cosine": cosine_sim(tensor_2d, recon), "bpw": bpw}


def encode_affine_group(tensor_2d, k=64, group_size=128):
    """Per-group uniform quantization (affine)."""
    rows, cols = tensor_2d.shape
    flat = tensor_2d.ravel().astype(np.float32)
    n = len(flat)

    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    else:
        padded = flat.copy()

    n_groups = len(padded) // group_size
    groups = padded.reshape(n_groups, group_size)

    g_min = groups.min(axis=1, keepdims=True)
    g_max = groups.max(axis=1, keepdims=True)
    g_range = np.maximum(g_max - g_min, 1e-10)

    normalized = (groups - g_min) / g_range
    indices = np.clip(np.round(normalized * (k - 1)), 0, k - 1).astype(np.int32)
    recon_groups = g_min + indices.astype(np.float32) / (k - 1) * g_range
    recon = recon_groups.ravel()[:n].reshape(tensor_2d.shape)

    bits_per_idx = np.log2(k)
    overhead_bpw = 2 * 16 / group_size  # scale + offset (FP16 each)
    bpw = bits_per_idx + overhead_bpw

    return recon, {"cosine": cosine_sim(tensor_2d, recon), "bpw": bpw}


def encode_vq_group_kmeans(tensor_2d, k=16, group_size=128):
    """Per-group k-means VQ."""
    rows, cols = tensor_2d.shape
    flat = tensor_2d.ravel().astype(np.float32)
    n = len(flat)

    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    else:
        padded = flat.copy()

    n_groups = len(padded) // group_size
    groups = padded.reshape(n_groups, group_size)
    recon_all = np.empty_like(padded)

    for g in range(n_groups):
        group = groups[g]
        codebook = kmeans_1d(group, k=k, max_iters=10)
        indices = assign_nearest(group, codebook)
        recon_all[g * group_size:(g + 1) * group_size] = codebook[indices]

    recon = recon_all[:n].reshape(tensor_2d.shape)
    bits_per_idx = np.log2(k)
    overhead_bpw = k * 32 / group_size  # per-group codebook
    bpw = bits_per_idx + overhead_bpw

    return recon, {"cosine": cosine_sim(tensor_2d, recon), "bpw": bpw}


# ---------------------------------------------------------------
# Phase 1: Tensor-level probe
# ---------------------------------------------------------------

def run_phase1():
    """Tensor-level cosine probe on all 154 TinyLlama layers."""
    from safetensors import safe_open

    sf_path = MODEL_DIR / "model.safetensors"
    print(f"\n  Loading tensors from {sf_path}")

    # Only vectorized (uniform affine) strategies — all O(n), no k-means loops.
    # VQ-256 global baseline from prior receipts (cos ~0.9998, PPL +0.78%).
    strategies = [
        ("affine8_g128",      lambda t: encode_affine_group(t, k=256, group_size=128)),
        ("affine6_g128",      lambda t: encode_affine_group(t, k=64, group_size=128)),
        ("affine4_g128",      lambda t: encode_affine_group(t, k=16, group_size=128)),
        ("affine4_g64",       lambda t: encode_affine_group(t, k=16, group_size=64)),
        ("affine4_g32",       lambda t: encode_affine_group(t, k=16, group_size=32)),
    ]

    # Accumulate per-strategy stats
    all_stats = {name: {"cosines": [], "bpws": []} for name, _ in strategies}

    import torch as _torch

    with safe_open(sf_path, framework="pt") as sf:
        tensor_names = []
        for i in range(N_LAYERS):
            for ptype, pattern in HF_PATTERNS.items():
                tensor_names.append(pattern.format(i=i))

        print(f"  {len(tensor_names)} tensors to process")
        t0 = time.time()

        for idx, tname in enumerate(tensor_names):
            tensor = sf.get_tensor(tname).float().numpy()
            if tensor.ndim == 1:
                continue  # skip biases

            for sname, encode_fn in strategies:
                _, stats = encode_fn(tensor)
                all_stats[sname]["cosines"].append(stats["cosine"])
                all_stats[sname]["bpws"].append(stats["bpw"])

            if (idx + 1) % 22 == 0:
                elapsed = time.time() - t0
                print(f"    {idx + 1}/{len(tensor_names)} tensors, {elapsed:.1f}s")

    phase1_time = time.time() - t0

    # Summary
    print(f"\n  Phase 1 complete in {phase1_time:.1f}s")
    print(f"\n  {'Strategy':<22} {'mean_cos':>10} {'min_cos':>10} {'p5_cos':>10} {'bpw':>6}")
    print("  " + "-" * 62)

    results = []
    for sname, _ in strategies:
        cosines = np.array(all_stats[sname]["cosines"])
        bpws = np.array(all_stats[sname]["bpws"])
        r = {
            "strategy": sname,
            "mean_cosine": round(float(cosines.mean()), 6),
            "min_cosine": round(float(cosines.min()), 6),
            "p5_cosine": round(float(np.percentile(cosines, 5)), 6),
            "std_cosine": round(float(cosines.std()), 6),
            "mean_bpw": round(float(bpws.mean()), 2),
            "n_tensors": len(cosines),
        }
        results.append(r)
        print(f"  {sname:<22} {r['mean_cosine']:>10.6f} {r['min_cosine']:>10.6f} "
              f"{r['p5_cosine']:>10.6f} {r['mean_bpw']:>6.2f}")

    # Gate check
    affine4_g128 = next(r for r in results if r["strategy"] == "affine4_g128")
    gate_pass = affine4_g128["mean_cosine"] >= 0.999 and affine4_g128["min_cosine"] >= 0.995

    print(f"\n  Phase 1 Gate: mean_cos >= 0.999 AND min_cos >= 0.995")
    print(f"  affine4_g128: mean={affine4_g128['mean_cosine']:.6f}, "
          f"min={affine4_g128['min_cosine']:.6f}")

    if gate_pass:
        print(f"  RESULT: PASS — proceeding to Phase 2 (PPL)")
    else:
        print(f"  RESULT: FAIL — tensor quality too low for PPL test")
        # Check if tighter groups pass
        for r in results:
            if "affine4" in r["strategy"] or "vq16" in r["strategy"]:
                gp = r["mean_cosine"] >= 0.999 and r["min_cosine"] >= 0.995
                status = "PASS" if gp else "FAIL"
                print(f"    {r['strategy']}: {status} (mean={r['mean_cosine']:.6f}, "
                      f"min={r['min_cosine']:.6f})")

    return results, gate_pass, phase1_time


# ---------------------------------------------------------------
# Phase 2: Model-level PPL
# ---------------------------------------------------------------

def run_phase2(phase1_pass_strategies):
    """Model-level PPL for strategies that passed Phase 1 tensor gate."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors import safe_open

    print(f"\n  Phase 2: Model-level PPL on CPU")
    print(f"  Strategies to test: {[s['strategy'] for s in phase1_pass_strategies]}")

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    text_raw = load_wikitext2()

    # Build tensor name → strategy → recon mapping
    sf_path = MODEL_DIR / "model.safetensors"

    # First: FP32 baseline PPL
    print(f"\n  Loading FP32 model for baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32, device_map="cpu",
    )
    model.eval()

    print(f"  Evaluating FP32 baseline PPL...")
    ppl_fp32, n_tok = eval_ppl_torch(model, tokenizer, text_raw)
    print(f"  FP32 PPL: {ppl_fp32:.4f} ({n_tok} tokens)")

    ppl_results = [{"strategy": "fp32_dense", "ppl": round(ppl_fp32, 4),
                     "eval_tokens": n_tok, "bpw": 32.0}]

    # For each strategy: replace weights, eval PPL, restore
    tensor_names = []
    for i in range(N_LAYERS):
        for ptype, pattern in HF_PATTERNS.items():
            tensor_names.append(pattern.format(i=i))

    # Build encode functions map
    encode_fns = {
        "affine8_g128":     lambda t: encode_affine_group(t, k=256, group_size=128),
        "affine6_g128":     lambda t: encode_affine_group(t, k=64, group_size=128),
        "affine4_g128":     lambda t: encode_affine_group(t, k=16, group_size=128),
        "affine4_g64":      lambda t: encode_affine_group(t, k=16, group_size=64),
        "affine4_g32":      lambda t: encode_affine_group(t, k=16, group_size=32),
    }

    # Always test affine6 and affine4_g128 (even if tensor gate failed — need PPL number)
    strategies_to_test = ["affine6_g128", "affine4_g128"]
    for s in phase1_pass_strategies:
        sn = s["strategy"]
        if sn not in strategies_to_test:
            strategies_to_test.append(sn)

    # Save original weights
    original_state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            for tname in tensor_names:
                # Convert HF tensor name to module path
                mod_name = tname.replace(".weight", "")
                if name == mod_name:
                    original_state[name] = mod.weight.data.clone()

    for strategy_name in strategies_to_test:
        if strategy_name not in encode_fns:
            continue

        print(f"\n  Strategy: {strategy_name}")
        t0 = time.time()
        encode_fn = encode_fns[strategy_name]

        # Replace weights
        n_replaced = 0
        with safe_open(sf_path, framework="pt") as sf:
            for tname in tensor_names:
                mod_name = tname.replace(".weight", "")
                # Navigate to module
                parts = mod_name.split(".")
                mod = model
                try:
                    for p in parts:
                        mod = getattr(mod, p)
                except AttributeError:
                    continue

                if not isinstance(mod, torch.nn.Linear):
                    continue

                tensor_np = sf.get_tensor(tname).float().numpy()
                recon, _ = encode_fn(tensor_np)
                mod.weight.data = torch.from_numpy(recon).to(mod.weight.dtype)
                n_replaced += 1

        quant_time = time.time() - t0
        print(f"    Replaced {n_replaced} layers in {quant_time:.1f}s")

        # Eval PPL
        ppl, n_tok = eval_ppl_torch(model, tokenizer, text_raw)
        delta = ((ppl / ppl_fp32) - 1) * 100
        print(f"    PPL: {ppl:.4f} ({delta:+.2f}% vs FP32)")

        ppl_results.append({
            "strategy": strategy_name,
            "ppl": round(ppl, 4),
            "eval_tokens": n_tok,
            "ppl_delta_pct": round(delta, 2),
        })

        # Restore original weights
        for name, orig_w in original_state.items():
            parts = name.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p)
            mod.weight.data = orig_w.clone()

    del model
    gc.collect()

    return ppl_results, ppl_fp32


def eval_ppl_torch(model, tokenizer, text):
    """Non-overlapping window PPL on CPU."""
    import torch

    encodings = tokenizer(text, return_tensors="pt")
    all_ids = encodings.input_ids.squeeze(0).tolist()
    if MAX_EVAL_TOKENS > 0:
        all_ids = all_ids[:MAX_EVAL_TOKENS]

    nlls = []
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_ids) - 1, SEQ_LEN):
            end = min(i + SEQ_LEN, len(all_ids))
            input_ids = torch.tensor(all_ids[i:end], dtype=torch.long).unsqueeze(0)
            outputs = model(input_ids, labels=input_ids)
            chunk_tokens = input_ids.shape[1] - 1
            nlls.append(outputs.loss.float().item() * chunk_tokens)
            total_tokens += chunk_tokens
            if end >= len(all_ids):
                break

    return math.exp(sum(nlls) / total_tokens), total_tokens


def load_wikitext2():
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join([t for t in ds["text"] if t.strip()])


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    t_global = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 65)
    print("  HXQ_AFFINE_4 EDGE PROBE")
    print(f"  Model: TinyLlama 1.1B (dense)")
    print(f"  Phase 1: Tensor cosine on 154 layers")
    print(f"  Phase 2: WikiText-2 PPL (seq={SEQ_LEN}, {MAX_EVAL_TOKENS} tok, CPU)")
    print(f"  {start_iso}")
    print("=" * 65)

    # Phase 1: fast tensor-level probe
    print("\n" + "=" * 50)
    print("  PHASE 1: TENSOR-LEVEL COSINE PROBE")
    print("=" * 50)
    phase1_results, phase1_gate, phase1_time = run_phase1()

    # Phase 2: model-level PPL (always run affine6 and affine4_g128)
    # Also run any 4-bit variant that passed Phase 1
    phase1_pass = [r for r in phase1_results
                   if r["mean_cosine"] >= 0.999 and r["min_cosine"] >= 0.995]

    print("\n" + "=" * 50)
    print("  PHASE 2: MODEL-LEVEL PPL")
    print("=" * 50)
    ppl_results, ppl_fp32 = run_phase2(phase1_pass)

    # ============================================================
    # Final summary
    # ============================================================
    wall_time = time.time() - t_global

    print("\n" + "=" * 80)
    print("  HXQ_AFFINE_4 EDGE PROBE — FINAL RESULTS")
    print("=" * 80)

    print(f"\n  Phase 1 — Tensor Cosine (154 layers):")
    print(f"  {'Strategy':<22} {'mean_cos':>10} {'min_cos':>10} {'bpw':>6}")
    print("  " + "-" * 52)
    for r in phase1_results:
        print(f"  {r['strategy']:<22} {r['mean_cosine']:>10.6f} {r['min_cosine']:>10.6f} "
              f"{r['mean_bpw']:>6.2f}")

    if ppl_results:
        print(f"\n  Phase 2 — WikiText-2 PPL (FP32 baseline: {ppl_fp32:.4f}):")
        print(f"  {'Strategy':<22} {'PPL':>10} {'Δ PPL%':>10}")
        print("  " + "-" * 44)
        for r in ppl_results:
            delta = f"{r.get('ppl_delta_pct', 0):+.2f}%" if "ppl_delta_pct" in r else "baseline"
            print(f"  {r['strategy']:<22} {r['ppl']:>10.4f} {delta:>10}")

    # Gate decision
    affine4_ppl = next((r for r in ppl_results if r["strategy"] == "affine4_g128"), None)
    if affine4_ppl:
        delta = affine4_ppl["ppl_delta_pct"]
        improvement = 9.34 - delta  # vs raw k=16
        print(f"\n  === GATE DECISION ===")
        print(f"  affine4_g128: PPL {affine4_ppl['ppl']:.4f}, delta {delta:+.2f}%")
        print(f"  Raw k=16 reference: +9.34% (prior receipt)")
        print(f"  Improvement over raw k=16: {improvement:+.2f} pp")

        if delta <= 3.0:
            verdict = f"PASS — {delta:+.2f}% <= 3% threshold"
        elif delta <= 5.0:
            verdict = f"MARGINAL — {delta:+.2f}%, reduced but above 3%"
        else:
            verdict = f"FAIL — {delta:+.2f}%, affine does not rescue k=16"
        print(f"  VERDICT: {verdict}")
    else:
        verdict = "SKIPPED — affine4_g128 not tested"
        delta = None
        print(f"\n  VERDICT: {verdict}")

    # Best 4-bit variant
    four_bit_ppl = [r for r in ppl_results
                    if "affine4" in r.get("strategy", "") or "vq16" in r.get("strategy", "")]
    if four_bit_ppl:
        best = min(four_bit_ppl, key=lambda r: r["ppl"])
        print(f"\n  Best 4-bit variant: {best['strategy']}")
        print(f"    PPL: {best['ppl']:.4f} ({best.get('ppl_delta_pct', 0):+.2f}%)")

    # Edge size estimate
    affine4_p1 = next((r for r in phase1_results if r["strategy"] == "affine4_g128"), None)
    if affine4_p1:
        bpw = affine4_p1["mean_bpw"]
        zamba_params_m = 2700
        est_mb = zamba_params_m * bpw / 8
        print(f"\n  Edge size estimate (Zamba2-2.7B at {bpw:.2f} bpw):")
        print(f"    Affine4: ~{est_mb:.0f} MB")
        print(f"    Affine6: ~{zamba_params_m * 6.25 / 8:.0f} MB")
        print(f"    Q4_K_M:  ~{zamba_params_m * 4.5 / 8:.0f} MB")

    # Receipt
    cost = {
        "wall_time_s": round(wall_time, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
    }

    receipt = {
        "experiment": "hxq_affine_4_edge_probe",
        "description": "Can per-group affine correction rescue k=16 (4-bit) for edge security?",
        "model": "TinyLlama-1.1B (dense FP32)",
        "phase1": {
            "description": "Tensor-level cosine on 154 layers",
            "gate": "mean_cos >= 0.999 AND min_cos >= 0.995",
            "results": phase1_results,
            "gate_pass": phase1_gate,
            "time_s": round(phase1_time, 1),
        },
        "phase2": {
            "description": f"WikiText-2 PPL (seq={SEQ_LEN}, {MAX_EVAL_TOKENS} tok, CPU)",
            "fp32_baseline_ppl": round(ppl_fp32, 4) if ppl_results else None,
            "results": ppl_results,
        },
        "gate": {
            "criterion": "affine4_g128 PPL delta <= +3% vs FP32",
            "raw_k16_reference": "+9.34% (prior receipt, global k-means)",
            "affine4_g128_ppl_delta_pct": delta,
            "verdict": verdict,
        },
        "cost": cost,
    }

    receipt_dir = Path(__file__).resolve().parent.parent / "receipts" / "hxq_affine4_probe"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"affine4_probe_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"\n  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
