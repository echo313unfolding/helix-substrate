#!/usr/bin/env python3
"""
HXQ_MIXED_LOWBIT_SIDECAR_PROBE: Can sparse sidecar push mixed-precision closer to affine6 quality?

Baseline receipts:
  - affine6_all:    6.25 bpw, +0.64% PPL
  - size_target:    5.25 bpw, +2.39% PPL  (152/154 at affine5, 2 fragile at affine6)
  - affine4_all:    4.25 bpw, +11.37% PPL (dead)

Sidecar concept:
  W ≈ affine_quantized_W + sparse_sidecar_delta
  residual = W_original - W_reconstructed
  keep top-k% absolute residuals as sparse corrections (row, col, fp16 value)

Policies:
  A. affine6_all (baseline)
  B. size_target_no_sidecar (from prior result)
  C. size_target_sidecar (affine5/6 routing + sidecar on low-cos tensors)
  D. aggressive_sidecar (push more tensors to affine4 + sidecar to recover quality)

Sidecar budgets: 0.1%, 0.25%, 0.5%, 1.0% of tensor elements

Work Order: WO-HXQ-MIXED-LOWBIT-SIDECAR-PROBE
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
MAX_EVAL_TOKENS = 20_000

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
SIDECAR_BUDGETS = [0.001, 0.0025, 0.005, 0.01]  # 0.1%, 0.25%, 0.5%, 1.0%

# ---------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------

def encode_affine_group(tensor_2d, k, group_size):
    """Per-group uniform quantization. Returns (recon, stats)."""
    flat = tensor_2d.ravel().astype(np.float32)
    n = len(flat)

    pad = (group_size - n % group_size) % group_size
    padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)]) if pad > 0 else flat.copy()

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
    overhead_bpw = 2 * 16 / group_size
    bpw = bits_per_idx + overhead_bpw

    return recon, bpw


def cosine_sim(a, b):
    a_f, b_f = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    dot = np.dot(a_f, b_f)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))


def apply_sidecar(original, recon, budget_frac):
    """Apply sparse sidecar correction to reconstruction.

    Args:
        original: original tensor (float32)
        recon: quantized reconstruction (float32)
        budget_frac: fraction of elements to keep as sparse corrections

    Returns:
        corrected: recon + sparse delta
        sidecar_bpw: bits per weight overhead from sidecar
        n_corrections: number of sparse entries
    """
    residual = original.ravel() - recon.ravel()
    n = len(residual)
    n_keep = max(1, int(n * budget_frac))

    # Find top-k absolute residuals
    abs_residual = np.abs(residual)
    if n_keep >= n:
        # Keep everything — but that defeats the purpose
        top_indices = np.arange(n)
    else:
        # Partial sort for top-k
        threshold_idx = np.argpartition(abs_residual, -n_keep)[-n_keep:]
        top_indices = threshold_idx

    # Build sparse correction
    corrected = recon.ravel().copy()
    corrected[top_indices] += residual[top_indices]
    corrected = corrected.reshape(original.shape)

    # Sidecar storage cost:
    # Each correction = position (log2(n) bits) + value (16 bits fp16)
    # In practice, delta-varint encoding compresses positions ~2x
    # Conservative estimate: 32 bits per position + 16 bits per value = 48 bits per correction
    # More realistic with delta-varint: ~24 bits per position + 16 bits per value = 40 bits
    bits_per_correction = 40  # conservative delta-varint estimate
    sidecar_total_bits = n_keep * bits_per_correction
    sidecar_bpw = sidecar_total_bits / n

    return corrected, sidecar_bpw, n_keep


def excess_kurtosis(data):
    n = len(data)
    if n < 4:
        return 0.0
    m, s = data.mean(), data.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((data - m) / s) ** 4) - 3.0)


# ---------------------------------------------------------------
# Phase 1: Per-tensor sidecar effectiveness probe
# ---------------------------------------------------------------

def run_phase1():
    from safetensors import safe_open

    sf_path = MODEL_DIR / "model.safetensors"
    print(f"\n  Loading tensors from {sf_path}")

    tensor_records = []

    with safe_open(sf_path, framework="pt") as sf:
        tensor_names = []
        for i in range(N_LAYERS):
            for ptype, pattern in HF_PATTERNS.items():
                tensor_names.append((pattern.format(i=i), i, ptype))

        print(f"  {len(tensor_names)} tensors")
        t0 = time.time()

        for idx, (tname, layer_idx, tensor_type) in enumerate(tensor_names):
            tensor = sf.get_tensor(tname).float().numpy()
            if tensor.ndim == 1:
                continue

            flat = tensor.ravel().astype(np.float32)
            kurt = excess_kurtosis(flat)
            n_params = int(np.prod(tensor.shape))

            record = {
                "name": tname,
                "layer": layer_idx,
                "type": tensor_type,
                "n_params": n_params,
                "kurtosis": round(kurt, 3),
                "strategies": {},
            }

            # Test each base quantization level
            for label, k, gs in [("affine6", 64, 128), ("affine5", 32, 128), ("affine4", 16, 128)]:
                recon, base_bpw = encode_affine_group(tensor, k=k, group_size=gs)
                cos_base = cosine_sim(tensor, recon)

                entry = {
                    "base_cosine": round(cos_base, 6),
                    "base_bpw": round(base_bpw, 2),
                    "sidecar": {},
                }

                # Test each sidecar budget
                for budget in SIDECAR_BUDGETS:
                    corrected, sc_bpw, n_corr = apply_sidecar(tensor, recon, budget)
                    cos_corrected = cosine_sim(tensor, corrected)
                    total_bpw = base_bpw + sc_bpw

                    budget_key = f"{budget*100:.1f}%"
                    entry["sidecar"][budget_key] = {
                        "cosine": round(cos_corrected, 6),
                        "total_bpw": round(total_bpw, 2),
                        "sidecar_bpw": round(sc_bpw, 3),
                        "n_corrections": n_corr,
                        "cosine_gain": round(cos_corrected - cos_base, 6),
                    }

                record["strategies"][label] = entry

            tensor_records.append(record)

            if (idx + 1) % 22 == 0:
                elapsed = time.time() - t0
                print(f"    {idx + 1}/{len(tensor_names)} tensors, {elapsed:.1f}s")

    phase1_time = time.time() - t0
    print(f"\n  Phase 1 complete in {phase1_time:.1f}s")

    # Summary: how much does sidecar improve each base quantization?
    for label in ["affine6", "affine5", "affine4"]:
        print(f"\n  {label} + sidecar summary:")
        print(f"    {'Budget':>8} {'mean_cos':>10} {'min_cos':>10} {'mean_bpw':>10} {'cos_gain':>10}")
        print("    " + "-" * 52)

        # Base (no sidecar)
        cosines = [r["strategies"][label]["base_cosine"] for r in tensor_records]
        bpws = [r["strategies"][label]["base_bpw"] for r in tensor_records]
        print(f"    {'base':>8} {np.mean(cosines):>10.6f} {np.min(cosines):>10.6f} "
              f"{np.mean(bpws):>10.2f} {'—':>10}")

        for budget in SIDECAR_BUDGETS:
            bk = f"{budget*100:.1f}%"
            cosines = [r["strategies"][label]["sidecar"][bk]["cosine"] for r in tensor_records]
            bpws_total = [r["strategies"][label]["sidecar"][bk]["total_bpw"] for r in tensor_records]
            gains = [r["strategies"][label]["sidecar"][bk]["cosine_gain"] for r in tensor_records]
            print(f"    {bk:>8} {np.mean(cosines):>10.6f} {np.min(cosines):>10.6f} "
                  f"{np.mean(bpws_total):>10.2f} {np.mean(gains):>10.6f}")

    return tensor_records, phase1_time


# ---------------------------------------------------------------
# Build policies with sidecar
# ---------------------------------------------------------------

def build_policies(tensor_records):
    """Build routing policies, some with sidecar."""

    policies = {}

    # A. affine6_all — no sidecar baseline
    policies["affine6_all"] = {
        r["name"]: {"base": "affine6", "sidecar_budget": None}
        for r in tensor_records
    }

    # B. size_target_no_sidecar — from prior result (affine5 unless fragile)
    size_no_sc = {}
    for r in tensor_records:
        cos5 = r["strategies"]["affine5"]["base_cosine"]
        size_no_sc[r["name"]] = {
            "base": "affine5" if cos5 >= 0.998 else "affine6",
            "sidecar_budget": None,
        }
    policies["size_target_no_sc"] = size_no_sc

    # C. size_target_sidecar — affine5 + sidecar on tensors where affine5 cos < 0.999
    # Use 0.5% sidecar budget on fragile tensors
    size_sc = {}
    for r in tensor_records:
        cos5 = r["strategies"]["affine5"]["base_cosine"]
        if cos5 >= 0.999:
            size_sc[r["name"]] = {"base": "affine5", "sidecar_budget": None}
        elif cos5 >= 0.998:
            size_sc[r["name"]] = {"base": "affine5", "sidecar_budget": 0.005}
        else:
            size_sc[r["name"]] = {"base": "affine6", "sidecar_budget": None}
    policies["size_target_sc"] = size_sc

    # D. aggressive_sidecar — try affine4 where sidecar at 0.5% restores cos >= 0.999
    # For tensors where affine4+sidecar(0.5%) cos >= 0.999: use affine4+sidecar
    # Otherwise fall back to affine5 or affine5+sidecar
    aggr = {}
    for r in tensor_records:
        cos4_sc = r["strategies"]["affine4"]["sidecar"]["0.5%"]["cosine"]
        cos5 = r["strategies"]["affine5"]["base_cosine"]

        if cos4_sc >= 0.999:
            aggr[r["name"]] = {"base": "affine4", "sidecar_budget": 0.005}
        elif cos5 >= 0.999:
            aggr[r["name"]] = {"base": "affine5", "sidecar_budget": None}
        elif cos5 >= 0.998:
            aggr[r["name"]] = {"base": "affine5", "sidecar_budget": 0.005}
        else:
            aggr[r["name"]] = {"base": "affine6", "sidecar_budget": None}
    policies["aggressive_sc"] = aggr

    # E. ultra_aggressive — try affine4 where sidecar at 1.0% restores cos >= 0.998
    ultra = {}
    for r in tensor_records:
        cos4_sc1 = r["strategies"]["affine4"]["sidecar"]["1.0%"]["cosine"]
        cos4_sc05 = r["strategies"]["affine4"]["sidecar"]["0.5%"]["cosine"]
        cos5 = r["strategies"]["affine5"]["base_cosine"]

        if cos4_sc05 >= 0.999:
            ultra[r["name"]] = {"base": "affine4", "sidecar_budget": 0.005}
        elif cos4_sc1 >= 0.998:
            ultra[r["name"]] = {"base": "affine4", "sidecar_budget": 0.01}
        elif cos5 >= 0.998:
            ultra[r["name"]] = {"base": "affine5", "sidecar_budget": None}
        else:
            ultra[r["name"]] = {"base": "affine6", "sidecar_budget": None}
    policies["ultra_aggressive_sc"] = ultra

    # Print summaries
    print("\n  Routing policies:")
    strat_params = {"affine6": (64, 128), "affine5": (32, 128), "affine4": (16, 128)}
    for pname, pmap in policies.items():
        base_counts = {}
        sc_count = 0
        total_bpw = 0
        total_params = 0

        for r in tensor_records:
            cfg = pmap[r["name"]]
            base = cfg["base"]
            base_counts[base] = base_counts.get(base, 0) + 1

            base_bpw = r["strategies"][base]["base_bpw"]
            sc_bpw = 0
            if cfg["sidecar_budget"] is not None:
                bk = f"{cfg['sidecar_budget']*100:.1f}%"
                sc_bpw = r["strategies"][base]["sidecar"][bk]["sidecar_bpw"]
                sc_count += 1

            total_bpw += (base_bpw + sc_bpw) * r["n_params"]
            total_params += r["n_params"]

        avg_bpw = total_bpw / total_params if total_params > 0 else 0
        dist = ", ".join(f"{s}={c}" for s, c in sorted(base_counts.items()))
        print(f"    {pname:<22} avg_bpw={avg_bpw:.2f}  sc_tensors={sc_count}  [{dist}]")

    return policies


# ---------------------------------------------------------------
# Phase 2: Model-level PPL
# ---------------------------------------------------------------

def run_phase2(tensor_records, policies):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors import safe_open

    print(f"\n  Phase 2: Model-level PPL on CPU")

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    print(f"  Loading FP32 model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.float32, device_map="cpu",
    )
    model.eval()
    text = load_wikitext2()

    print(f"  Evaluating FP32 baseline PPL...")
    ppl_fp32, n_tok = eval_ppl(model, tokenizer, text)
    print(f"  FP32 PPL: {ppl_fp32:.4f} ({n_tok} tokens)")

    ppl_results = [{"policy": "fp32_dense", "ppl": round(ppl_fp32, 4),
                     "avg_bpw": 32.0, "eval_tokens": n_tok}]

    sf_path = MODEL_DIR / "model.safetensors"
    strat_params = {"affine6": (64, 128), "affine5": (32, 128), "affine4": (16, 128)}
    rec_by_name = {r["name"]: r for r in tensor_records}

    # Save original weights
    original_weights = {}
    for r in tensor_records:
        mod_name = r["name"].replace(".weight", "")
        parts = mod_name.split(".")
        mod = model
        try:
            for p in parts:
                mod = getattr(mod, p)
        except AttributeError:
            continue
        if isinstance(mod, torch.nn.Linear):
            original_weights[r["name"]] = mod.weight.data.clone()

    for policy_name, policy_map in policies.items():
        print(f"\n  Policy: {policy_name}")
        t0 = time.time()

        # Compute avg bpw
        total_bpw = 0
        total_params = 0
        for r in tensor_records:
            cfg = policy_map[r["name"]]
            base_bpw = r["strategies"][cfg["base"]]["base_bpw"]
            sc_bpw = 0
            if cfg["sidecar_budget"] is not None:
                bk = f"{cfg['sidecar_budget']*100:.1f}%"
                sc_bpw = r["strategies"][cfg["base"]]["sidecar"][bk]["sidecar_bpw"]
            total_bpw += (base_bpw + sc_bpw) * r["n_params"]
            total_params += r["n_params"]
        avg_bpw = total_bpw / total_params

        # Replace weights
        n_replaced = 0
        n_sidecars = 0
        with safe_open(sf_path, framework="pt") as sf:
            for r in tensor_records:
                tname = r["name"]
                cfg = policy_map[tname]
                k, gs = strat_params[cfg["base"]]

                mod_name = tname.replace(".weight", "")
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
                recon, _ = encode_affine_group(tensor_np, k=k, group_size=gs)

                if cfg["sidecar_budget"] is not None:
                    corrected, _, _ = apply_sidecar(tensor_np, recon, cfg["sidecar_budget"])
                    mod.weight.data = torch.from_numpy(corrected).to(mod.weight.dtype)
                    n_sidecars += 1
                else:
                    mod.weight.data = torch.from_numpy(recon).to(mod.weight.dtype)
                n_replaced += 1

        quant_time = time.time() - t0
        print(f"    Replaced {n_replaced} layers ({n_sidecars} with sidecar), "
              f"{quant_time:.1f}s, avg_bpw={avg_bpw:.2f}")

        ppl, n_tok = eval_ppl(model, tokenizer, text)
        delta = ((ppl / ppl_fp32) - 1) * 100
        print(f"    PPL: {ppl:.4f} ({delta:+.2f}% vs FP32)")

        ppl_results.append({
            "policy": policy_name,
            "ppl": round(ppl, 4),
            "ppl_delta_pct": round(delta, 2),
            "avg_bpw": round(avg_bpw, 2),
            "n_sidecars": n_sidecars,
            "eval_tokens": n_tok,
        })

        # Restore
        for tname, orig_w in original_weights.items():
            mod_name = tname.replace(".weight", "")
            parts = mod_name.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p)
            mod.weight.data = orig_w.clone()

    del model
    gc.collect()
    return ppl_results, ppl_fp32


def eval_ppl(model, tokenizer, text):
    import torch
    encodings = tokenizer(text, return_tensors="pt")
    all_ids = encodings.input_ids.squeeze(0).tolist()
    if MAX_EVAL_TOKENS > 0:
        all_ids = all_ids[:MAX_EVAL_TOKENS]

    nlls = []
    total_tokens = 0
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

    print("=" * 70)
    print("  HXQ_MIXED_LOWBIT_SIDECAR_PROBE")
    print(f"  Model: TinyLlama 1.1B (dense)")
    print(f"  Phase 1: Per-tensor sidecar effectiveness")
    print(f"  Phase 2: Routed policy PPL (WikiText-2, {MAX_EVAL_TOKENS} tok, CPU)")
    print(f"  Sidecar budgets: {[f'{b*100:.1f}%' for b in SIDECAR_BUDGETS]}")
    print(f"  {start_iso}")
    print("=" * 70)

    # Phase 1
    print("\n" + "=" * 55)
    print("  PHASE 1: PER-TENSOR SIDECAR EFFECTIVENESS")
    print("=" * 55)
    tensor_records, phase1_time = run_phase1()

    # Build policies
    print("\n" + "=" * 55)
    print("  ROUTING POLICIES WITH SIDECAR")
    print("=" * 55)
    policies = build_policies(tensor_records)

    # Phase 2
    print("\n" + "=" * 55)
    print("  PHASE 2: MODEL-LEVEL PPL")
    print("=" * 55)
    ppl_results, ppl_fp32 = run_phase2(tensor_records, policies)

    # ============================================================
    # Final summary
    # ============================================================
    wall_time = time.time() - t_global

    print("\n" + "=" * 80)
    print("  HXQ_MIXED_LOWBIT_SIDECAR_PROBE — FINAL RESULTS")
    print("=" * 80)

    # PPL results
    print(f"\n  Policy PPL results (FP32 baseline: {ppl_fp32:.4f}):")
    print(f"  {'Policy':<24} {'PPL':>8} {'Δ PPL%':>8} {'bpw':>6} {'sc':>4}")
    print("  " + "-" * 54)
    for r in ppl_results:
        delta = f"{r.get('ppl_delta_pct', 0):+.2f}%" if "ppl_delta_pct" in r else "base"
        sc = f"{r.get('n_sidecars', 0)}" if "n_sidecars" in r else "—"
        print(f"  {r['policy']:<24} {r['ppl']:>8.4f} {delta:>8} {r['avg_bpw']:>6.2f} {sc:>4}")

    # Pareto
    print(f"\n  Pareto (bpw vs PPL delta):")
    non_base = [r for r in ppl_results if r["policy"] != "fp32_dense"]
    for r in sorted(non_base, key=lambda x: x["avg_bpw"]):
        marker = ""
        dppl = r.get("ppl_delta_pct", 99)
        if dppl <= 2.0 and r["avg_bpw"] < 6.0:
            marker = " ← STRONG"
        elif dppl <= 3.0 and r["avg_bpw"] < 6.0:
            marker = " ← PASS"
        print(f"    {r['avg_bpw']:.2f} bpw, {dppl:+.2f}% PPL  [{r['policy']}]{marker}")

    # Gate
    best = None
    for r in ppl_results:
        if r["policy"] == "fp32_dense":
            continue
        dppl = r.get("ppl_delta_pct", 99)
        if dppl <= 3.0 and r["avg_bpw"] < 6.25:
            if best is None or r["avg_bpw"] < best["avg_bpw"]:
                best = r

    print(f"\n  === GATE DECISION ===")
    if best:
        print(f"  Best: {best['policy']} at {best['avg_bpw']:.2f} bpw, "
              f"{best['ppl_delta_pct']:+.2f}% PPL")
        savings_bpw = 6.25 - best["avg_bpw"]
        zamba_mb_a6 = 2700 * 6.25 / 8
        zamba_mb_best = 2700 * best["avg_bpw"] / 8
        print(f"  Savings: {savings_bpw:.2f} bpw ({zamba_mb_a6:.0f} → {zamba_mb_best:.0f} MB for Zamba2-2.7B)")

        # Did sidecar help vs no-sidecar?
        no_sc = next((r for r in ppl_results if r["policy"] == "size_target_no_sc"), None)
        if no_sc and best["policy"] != "size_target_no_sc":
            ppl_improvement = no_sc["ppl_delta_pct"] - best["ppl_delta_pct"]
            bpw_cost = best["avg_bpw"] - no_sc["avg_bpw"]
            print(f"  Sidecar effect vs no-sidecar: {ppl_improvement:+.2f}pp quality, "
                  f"{bpw_cost:+.2f} bpw cost")

        if best["avg_bpw"] <= 4.75:
            verdict = f"PASS — {best['avg_bpw']:.2f} bpw below 4.75 target"
        elif best["avg_bpw"] < 6.0:
            verdict = f"PARTIAL — {best['avg_bpw']:.2f} bpw, quality OK, saves {savings_bpw:.2f} bpw vs affine6"
        else:
            verdict = f"MARGINAL — only {savings_bpw:.2f} bpw saved"
    else:
        verdict = "FAIL — no policy beats affine6 on size within quality gate"

    print(f"  VERDICT: {verdict}")

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
        "experiment": "hxq_mixed_lowbit_sidecar_probe",
        "description": "Can sparse sidecar improve mixed low-bit policy?",
        "model": "TinyLlama-1.1B (dense FP32)",
        "prior_baselines": {
            "affine6_all": {"bpw": 6.25, "ppl_delta_pct": 0.64},
            "size_target": {"bpw": 5.25, "ppl_delta_pct": 2.39},
            "affine4_all": {"bpw": 4.25, "ppl_delta_pct": 11.37},
        },
        "sidecar_budgets_tested": [f"{b*100:.1f}%" for b in SIDECAR_BUDGETS],
        "phase1": {
            "description": "Per-tensor sidecar effectiveness across 3 base levels × 4 budgets",
            "n_tensors": len(tensor_records),
            "time_s": round(phase1_time, 1),
            # Omit full tensor_records for size — key stats only
        },
        "phase2": {
            "fp32_baseline_ppl": round(ppl_fp32, 4),
            "results": ppl_results,
        },
        "gate": {
            "criterion": "avg_bpw < 6.25 AND PPL delta <= +3%",
            "verdict": verdict,
        },
        "cost": cost,
    }

    receipt_dir = Path(__file__).resolve().parent.parent / "receipts" / "hxq_mixed_lowbit_sidecar"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"sidecar_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"\n  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
