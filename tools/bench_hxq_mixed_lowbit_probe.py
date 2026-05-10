#!/usr/bin/env python3
"""
HXQ_MIXED_LOWBIT_PROBE: Can routed mixed-precision reach Q4_K_M size with acceptable quality?

NOT "3-bit everywhere." Routed: profiler decides per tensor.

Phase 1: Tensor sensitivity probe (154 TinyLlama layers)
  - For each tensor: affine6/affine4/affine3 cosine + metadata
  - Record kurtosis, shape, layer index, tensor type

Phase 2: Build routing policies from Phase 1 data, measure PPL
  A. all_affine6:      all tensors at 6-bit (baseline)
  B. safe_mixed:       affine6 unless affine4 passes cosine >= 0.999
  C. aggressive_mixed: affine6/4/3 based on cosine gate (0.999/0.998)
  D. size_target:      lowest bit per tensor targeting avg bpw <= 4.5

Pass: avg bpw <= 4.75 AND PPL delta <= +3%
      OR clear Pareto improvement over affine6

Work Order: WO-HXQ-MIXED-LOWBIT-PROBE
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

    orig_flat = tensor_2d.ravel().astype(np.float64)
    recon_flat = recon.ravel().astype(np.float64)
    dot = np.dot(orig_flat, recon_flat)
    na, nb = np.linalg.norm(orig_flat), np.linalg.norm(recon_flat)
    cos = float(dot / (na * nb)) if na > 1e-30 and nb > 1e-30 else 0.0

    err = np.abs(recon.ravel() - tensor_2d.ravel())

    bits_per_idx = np.log2(k)
    overhead_bpw = 2 * 16 / group_size
    bpw = bits_per_idx + overhead_bpw

    return recon, {
        "cosine": cos,
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
        "bpw": bpw,
    }


def excess_kurtosis(data):
    n = len(data)
    if n < 4:
        return 0.0
    m = data.mean()
    s = data.std()
    if s < 1e-12:
        return 0.0
    z = (data - m) / s
    return float(np.mean(z ** 4) - 3.0)


# ---------------------------------------------------------------
# Phase 1: Per-tensor sensitivity probe
# ---------------------------------------------------------------

def run_phase1():
    from safetensors import safe_open

    sf_path = MODEL_DIR / "model.safetensors"
    print(f"\n  Loading tensors from {sf_path}")

    # Strategies to probe per tensor
    strategies = [
        ("affine6_g128", 64, 128),
        ("affine5_g128", 32, 128),
        ("affine4_g128", 16, 128),
        ("affine4_g64",  16, 64),
        ("affine3_g128",  8, 128),
        ("affine3_g64",   8, 64),
        ("affine3_g32",   8, 32),
    ]

    tensor_records = []

    with safe_open(sf_path, framework="pt") as sf:
        tensor_names = []
        for i in range(N_LAYERS):
            for ptype, pattern in HF_PATTERNS.items():
                tensor_names.append((pattern.format(i=i), i, ptype))

        print(f"  {len(tensor_names)} tensors × {len(strategies)} strategies")
        t0 = time.time()

        for idx, (tname, layer_idx, tensor_type) in enumerate(tensor_names):
            tensor = sf.get_tensor(tname).float().numpy()
            if tensor.ndim == 1:
                continue

            flat = tensor.ravel().astype(np.float32)
            kurt = excess_kurtosis(flat)
            std = float(flat.std())
            shape = list(tensor.shape)

            record = {
                "name": tname,
                "layer": layer_idx,
                "type": tensor_type,
                "shape": shape,
                "n_params": int(np.prod(shape)),
                "kurtosis": round(kurt, 3),
                "std": round(std, 6),
                "strategies": {},
            }

            for sname, k, gs in strategies:
                _, stats = encode_affine_group(tensor, k=k, group_size=gs)
                record["strategies"][sname] = {
                    "cosine": round(stats["cosine"], 6),
                    "max_abs_error": round(stats["max_abs_error"], 6),
                    "mean_abs_error": round(stats["mean_abs_error"], 6),
                    "bpw": round(stats["bpw"], 2),
                }

            tensor_records.append(record)

            if (idx + 1) % 22 == 0:
                elapsed = time.time() - t0
                print(f"    {idx + 1}/{len(tensor_names)} tensors, {elapsed:.1f}s")

    phase1_time = time.time() - t0
    print(f"\n  Phase 1 complete in {phase1_time:.1f}s ({len(tensor_records)} tensors)")

    # Summary per strategy
    print(f"\n  {'Strategy':<16} {'mean_cos':>10} {'min_cos':>10} {'p5_cos':>10} {'bpw':>6}")
    print("  " + "-" * 56)
    for sname, k, gs in strategies:
        cosines = [r["strategies"][sname]["cosine"] for r in tensor_records]
        ca = np.array(cosines)
        bpw = tensor_records[0]["strategies"][sname]["bpw"]
        print(f"  {sname:<16} {ca.mean():>10.6f} {ca.min():>10.6f} "
              f"{np.percentile(ca, 5):>10.6f} {bpw:>6.2f}")

    return tensor_records, phase1_time


# ---------------------------------------------------------------
# Routing policies
# ---------------------------------------------------------------

def build_policies(tensor_records):
    """Build routing policies from Phase 1 per-tensor data."""

    policies = {}

    # A. all_affine6: baseline
    policies["all_affine6"] = {r["name"]: "affine6_g128" for r in tensor_records}

    # B. safe_mixed: affine4 if cosine >= 0.999, else affine6
    safe = {}
    for r in tensor_records:
        a4_cos = r["strategies"]["affine4_g128"]["cosine"]
        safe[r["name"]] = "affine4_g128" if a4_cos >= 0.999 else "affine6_g128"
    policies["safe_mixed"] = safe

    # C. aggressive_mixed: affine3 if cos >= 0.999, affine4 if cos >= 0.999, else affine6
    aggr = {}
    for r in tensor_records:
        a3_cos = r["strategies"]["affine3_g128"]["cosine"]
        a4_cos = r["strategies"]["affine4_g128"]["cosine"]
        if a3_cos >= 0.999:
            aggr[r["name"]] = "affine3_g128"
        elif a4_cos >= 0.999:
            aggr[r["name"]] = "affine4_g128"
        else:
            aggr[r["name"]] = "affine6_g128"
    policies["aggressive_mixed"] = aggr

    # D. size_target: lowest bit per tensor targeting avg bpw <= 4.5
    #    Try from lowest to highest, pick lowest that passes cosine >= 0.998
    size_t = {}
    candidates_ordered = [
        ("affine3_g128", 0.998),
        ("affine3_g64", 0.998),
        ("affine4_g128", 0.998),
        ("affine4_g64", 0.998),
        ("affine5_g128", 0.998),
        ("affine6_g128", 0.0),  # always passes
    ]
    for r in tensor_records:
        chosen = "affine6_g128"
        for sname, gate in candidates_ordered:
            if r["strategies"][sname]["cosine"] >= gate:
                chosen = sname
                break
        size_t[r["name"]] = chosen
    policies["size_target"] = size_t

    # Print policy summaries
    print("\n  Routing policies:")
    for pname, pmap in policies.items():
        counts = {}
        total_bpw = 0
        total_params = 0
        for r in tensor_records:
            strat = pmap[r["name"]]
            counts[strat] = counts.get(strat, 0) + 1
            total_bpw += r["strategies"][strat]["bpw"] * r["n_params"]
            total_params += r["n_params"]
        avg_bpw = total_bpw / total_params if total_params > 0 else 0

        dist = ", ".join(f"{s}={c}" for s, c in sorted(counts.items()))
        print(f"    {pname:<20} avg_bpw={avg_bpw:.2f}  [{dist}]")

    return policies


# ---------------------------------------------------------------
# Phase 2: Model-level PPL for each policy
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

    # Build name→record lookup
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

    # Strategy → (k, group_size)
    strat_params = {
        "affine6_g128": (64, 128),
        "affine5_g128": (32, 128),
        "affine4_g128": (16, 128),
        "affine4_g64":  (16, 64),
        "affine3_g128": (8, 128),
        "affine3_g64":  (8, 64),
        "affine3_g32":  (8, 32),
    }

    for policy_name, policy_map in policies.items():
        print(f"\n  Policy: {policy_name}")
        t0 = time.time()

        # Compute avg bpw for this policy
        total_bpw = 0
        total_params = 0
        for r in tensor_records:
            strat = policy_map[r["name"]]
            total_bpw += r["strategies"][strat]["bpw"] * r["n_params"]
            total_params += r["n_params"]
        avg_bpw = total_bpw / total_params

        # Replace weights
        n_replaced = 0
        with safe_open(sf_path, framework="pt") as sf:
            for r in tensor_records:
                tname = r["name"]
                strat = policy_map[tname]
                k, gs = strat_params[strat]

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
                mod.weight.data = torch.from_numpy(recon).to(mod.weight.dtype)
                n_replaced += 1

        quant_time = time.time() - t0
        print(f"    Replaced {n_replaced} layers in {quant_time:.1f}s, avg_bpw={avg_bpw:.2f}")

        # Eval PPL
        ppl, n_tok = eval_ppl(model, tokenizer, text)
        delta = ((ppl / ppl_fp32) - 1) * 100
        print(f"    PPL: {ppl:.4f} ({delta:+.2f}% vs FP32)")

        ppl_results.append({
            "policy": policy_name,
            "ppl": round(ppl, 4),
            "ppl_delta_pct": round(delta, 2),
            "avg_bpw": round(avg_bpw, 2),
            "eval_tokens": n_tok,
        })

        # Restore original weights
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
    print("  HXQ_MIXED_LOWBIT_PROBE")
    print(f"  Model: TinyLlama 1.1B (dense)")
    print(f"  Phase 1: Per-tensor sensitivity (154 layers × 7 strategies)")
    print(f"  Phase 2: Routed policy PPL (WikiText-2, {MAX_EVAL_TOKENS} tok, CPU)")
    print(f"  {start_iso}")
    print("=" * 70)

    # Phase 1
    print("\n" + "=" * 55)
    print("  PHASE 1: PER-TENSOR SENSITIVITY PROBE")
    print("=" * 55)
    tensor_records, phase1_time = run_phase1()

    # Build routing policies
    print("\n" + "=" * 55)
    print("  ROUTING POLICIES")
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
    print("  HXQ_MIXED_LOWBIT_PROBE — FINAL RESULTS")
    print("=" * 80)

    # Tensor sensitivity summary by type
    print(f"\n  Tensor sensitivity by type (affine4_g128 cosine):")
    by_type = {}
    for r in tensor_records:
        t = r["type"]
        by_type.setdefault(t, []).append(r["strategies"]["affine4_g128"]["cosine"])
    print(f"  {'Type':<12} {'mean_cos':>10} {'min_cos':>10} {'count':>6}")
    print("  " + "-" * 42)
    for t in sorted(by_type.keys()):
        arr = np.array(by_type[t])
        print(f"  {t:<12} {arr.mean():>10.6f} {arr.min():>10.6f} {len(arr):>6}")

    # Kurtosis vs compressibility
    print(f"\n  Kurtosis correlation with affine4 cosine:")
    kurts = np.array([r["kurtosis"] for r in tensor_records])
    cos4 = np.array([r["strategies"]["affine4_g128"]["cosine"] for r in tensor_records])
    if len(kurts) > 2:
        corr = np.corrcoef(kurts, cos4)[0, 1]
        print(f"    Pearson r = {corr:.3f}")
        # Top 5 most compressible (highest affine4 cos)
        sorted_recs = sorted(tensor_records, key=lambda r: r["strategies"]["affine4_g128"]["cosine"], reverse=True)
        print(f"\n    Top 5 most affine4-friendly:")
        for r in sorted_recs[:5]:
            print(f"      {r['name']:>50s}  cos={r['strategies']['affine4_g128']['cosine']:.6f}  "
                  f"kurt={r['kurtosis']:.1f}")
        print(f"    Bottom 5 (most fragile):")
        for r in sorted_recs[-5:]:
            print(f"      {r['name']:>50s}  cos={r['strategies']['affine4_g128']['cosine']:.6f}  "
                  f"kurt={r['kurtosis']:.1f}")

    # PPL results
    print(f"\n  Policy PPL results (FP32 baseline: {ppl_fp32:.4f}):")
    print(f"  {'Policy':<22} {'PPL':>8} {'Δ PPL%':>8} {'avg_bpw':>8}")
    print("  " + "-" * 50)
    for r in ppl_results:
        delta = f"{r.get('ppl_delta_pct', 0):+.2f}%" if "ppl_delta_pct" in r else "base"
        print(f"  {r['policy']:<22} {r['ppl']:>8.4f} {delta:>8} {r['avg_bpw']:>8.2f}")

    # Pareto analysis
    print(f"\n  Pareto analysis (bpw vs PPL delta):")
    non_base = [r for r in ppl_results if r["policy"] != "fp32_dense"]
    for r in sorted(non_base, key=lambda x: x["avg_bpw"]):
        marker = ""
        if r.get("ppl_delta_pct", 99) <= 3.0 and r["avg_bpw"] <= 4.75:
            marker = " ← PASS"
        elif r.get("ppl_delta_pct", 99) <= 3.0:
            marker = " ← quality OK, size not target"
        print(f"    {r['avg_bpw']:.2f} bpw, {r.get('ppl_delta_pct', 0):+.2f}% PPL  "
              f"[{r['policy']}]{marker}")

    # Gate decision
    best_routed = None
    for r in ppl_results:
        if r["policy"] in ("safe_mixed", "aggressive_mixed", "size_target"):
            if best_routed is None or r["avg_bpw"] < best_routed["avg_bpw"]:
                if r.get("ppl_delta_pct", 99) <= 3.0:
                    best_routed = r

    print(f"\n  === GATE DECISION ===")
    if best_routed:
        print(f"  Best viable routed: {best_routed['policy']}")
        print(f"    avg_bpw: {best_routed['avg_bpw']:.2f}")
        print(f"    PPL delta: {best_routed['ppl_delta_pct']:+.2f}%")
        if best_routed["avg_bpw"] <= 4.75:
            verdict = (f"PASS — {best_routed['policy']} at {best_routed['avg_bpw']:.2f} bpw, "
                       f"{best_routed['ppl_delta_pct']:+.2f}% PPL")
        else:
            verdict = (f"PARTIAL — quality OK but avg_bpw {best_routed['avg_bpw']:.2f} > 4.75 target")
    else:
        # Check if any routed policy beats affine6 on size without catastrophic PPL
        routed = [r for r in ppl_results if r["policy"] in ("safe_mixed", "aggressive_mixed", "size_target")]
        if routed:
            best_size = min(routed, key=lambda r: r["avg_bpw"])
            verdict = (f"FAIL — best routed ({best_size['policy']}) at {best_size['avg_bpw']:.2f} bpw "
                       f"has {best_size.get('ppl_delta_pct', 99):+.2f}% PPL")
        else:
            verdict = "FAIL — no routed policies tested"

    print(f"  VERDICT: {verdict}")

    # Edge size estimates
    affine6_r = next((r for r in ppl_results if r["policy"] == "all_affine6"), None)
    if best_routed and affine6_r:
        zamba_params_m = 2700
        a6_mb = zamba_params_m * affine6_r["avg_bpw"] / 8
        routed_mb = zamba_params_m * best_routed["avg_bpw"] / 8
        savings = a6_mb - routed_mb
        print(f"\n  Edge estimate (Zamba2-2.7B):")
        print(f"    affine6 all:  ~{a6_mb:.0f} MB")
        print(f"    {best_routed['policy']}: ~{routed_mb:.0f} MB (saves ~{savings:.0f} MB)")

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
        "experiment": "hxq_mixed_lowbit_probe",
        "description": "Routed mixed-precision: can per-tensor bit routing reach low bpw with acceptable quality?",
        "model": "TinyLlama-1.1B (dense FP32)",
        "phase1": {
            "description": "Per-tensor sensitivity probe (154 layers × 7 strategies)",
            "n_tensors": len(tensor_records),
            "time_s": round(phase1_time, 1),
            "tensor_records": tensor_records,
        },
        "phase2": {
            "description": f"WikiText-2 PPL (seq={SEQ_LEN}, {MAX_EVAL_TOKENS} tok, CPU)",
            "fp32_baseline_ppl": round(ppl_fp32, 4),
            "results": ppl_results,
        },
        "policies": {
            pname: {
                "distribution": {},
                "avg_bpw": None,
            }
            for pname in policies
        },
        "gate": {
            "criterion": "avg_bpw <= 4.75 AND PPL delta <= +3%",
            "verdict": verdict,
        },
        "cost": cost,
    }

    # Fill policy distributions
    for pname, pmap in policies.items():
        dist = {}
        total_bpw = 0
        total_params = 0
        for r in tensor_records:
            strat = pmap[r["name"]]
            dist[strat] = dist.get(strat, 0) + 1
            total_bpw += r["strategies"][strat]["bpw"] * r["n_params"]
            total_params += r["n_params"]
        receipt["policies"][pname]["distribution"] = dist
        receipt["policies"][pname]["avg_bpw"] = round(total_bpw / total_params, 2) if total_params else 0

    receipt_dir = Path(__file__).resolve().parent.parent / "receipts" / "hxq_mixed_lowbit_probe"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"mixed_lowbit_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"\n  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
