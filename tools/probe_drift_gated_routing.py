#!/usr/bin/env python3
"""
Probe 5: Does drift-gated routing outperform static exit?

WO-SENSING-PROBE-05: Test the ΔΣt prediction from the equation.

The equation says: ΔΣt ⋅ KR(λᵢ, ηᵢ) ⋅ Θ(cᵢ)

Translation: routing (KR) should only activate when accumulated spectral
drift (ΔΣt) is nonzero, gated by quality constraints (Θ). When drift is
low, skip routing — use static policy.

Concrete test:
  1. Measure spectral drift rate per layer (change in effective rank
     between consecutive layers)
  2. Define ΔΣt = cumulative drift from layer 0 to layer N
  3. When ΔΣt stabilizes (drift rate drops below threshold), the system
     has converged — trigger early exit (Θ gate opens)
  4. Compare exit quality vs fixed-layer exit and vs oracle (full model)

Prediction: Drift-gated exit should match or beat fixed-layer exit,
because it adapts to query difficulty automatically.

Control: Fixed exit at layer 17 (median convergence from Probe 2).

Model: TinyLlama-1.1B-Chat-v1.0 (dense FP32, 22 layers)
"""

import json
import math
import os
import sys
import time
import resource
import platform
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_substrate.rapl_meter import RaplMeter

# ── Cost tracking ──
t_start = time.time()
cpu_start = time.process_time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

# ── Same prompts as prior probes ──
PROMPTS = {
    "factual_short": "What is the capital of France?",

    "factual_long": (
        "The process of photosynthesis converts carbon dioxide and water into "
        "glucose and oxygen using energy from sunlight. This reaction occurs "
        "primarily in the chloroplasts of plant cells, where chlorophyll "
        "pigments absorb light energy. The light-dependent reactions take "
        "place in the thylakoid membranes, producing ATP and NADPH."
    ),

    "code": (
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
        "\n"
        "# Calculate the first 20 Fibonacci numbers\n"
        "for i in range(20):\n"
        "    print(f'F({i}) = {fibonacci(i)}')\n"
    ),

    "creative": (
        "In the dream, the ocean was made of glass and the fish swam through "
        "light instead of water. Each wave was a frozen moment, a sculpture "
        "of motion that never moved. The lighthouse keeper counted the colors "
        "of silence — seven, she decided, though the eighth was always hiding "
        "behind her left eye. The moon tasted like copper pennies and old songs."
    ),

    "adversarial_repetitive": (
        "the cat sat on the mat the cat sat on the mat the cat sat on the mat "
        "the cat sat on the mat the cat sat on the mat the cat sat on the mat "
        "the cat sat on the mat the cat sat on the mat the cat sat on the mat"
    ),

    "mixed_technical": (
        "The Transformer architecture uses multi-head self-attention where "
        "Q = XW_Q, K = XW_K, V = XW_V, and attention = softmax(QK^T/sqrt(d_k))V. "
        "For a model with 32 heads, d_model=4096, each head has d_k=128. "
        "The KV cache stores K and V tensors to avoid recomputation during "
        "autoregressive generation, trading memory for compute."
    ),
}


def effective_rank(matrix: np.ndarray) -> float:
    """Effective rank = exp(H(σ_norm))."""
    try:
        s = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return 1.0
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    h = -np.sum(p * np.log(p))
    return float(np.exp(h))


def main():
    out_dir = Path(__file__).resolve().parent.parent / "receipts" / "drift_gated_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROBE 5: Does drift-gated routing outperform static exit?")
    print("WO-SENSING-PROBE-05 — Testing ΔΣt from the equation")
    print("=" * 70)

    # ── Load model ──
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading {MODEL_ID} (dense, from HF cache)...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rapl = RaplMeter()
    rapl.__enter__()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    final_norm = model.model.norm
    lm_head = model.lm_head
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Model: {n_layers} layers, hidden={hidden_dim}")

    # ── Drift threshold sweep ──
    # Test multiple thresholds to find the optimal one
    DRIFT_THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    FIXED_EXIT_LAYER = 17  # Median from Probe 2
    # Minimum layers before allowing exit (avoid degenerate exits)
    MIN_LAYERS = 8

    # ── Run each prompt, collect hidden states ──
    all_results = {}

    for prompt_id, text in PROMPTS.items():
        print(f"\n── {prompt_id} ──")
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        seq_len = inputs["input_ids"].shape[1]
        print(f"  Tokens: {seq_len}")

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states  # (n_layers+1) tensors
        full_logits = outputs.logits[0, -1, :].cpu().float().numpy()
        full_top1 = int(np.argmax(full_logits))
        full_top5 = set(np.argsort(full_logits)[-5:].tolist())

        # ── Compute effective rank per layer ──
        eranks = []
        for i in range(n_layers + 1):
            hs_np = hidden_states[i][0].cpu().float().numpy()  # [seq_len, hidden]
            if seq_len >= 2:
                er = effective_rank(hs_np)
            else:
                er = 1.0
            eranks.append(er)

        # ── Compute ΔΣt: cumulative spectral drift ──
        # Drift at layer i = |eff_rank(i) - eff_rank(i-1)|
        # ΔΣt at layer i = sum of drifts from layer 1 to i
        drift_per_layer = []
        cumulative_drift = []
        drift_rate = []  # Instantaneous drift rate (change in ΔΣt)
        running_sum = 0.0

        for i in range(1, n_layers + 1):
            d = abs(eranks[i] - eranks[i - 1])
            drift_per_layer.append(d)
            running_sum += d
            cumulative_drift.append(running_sum)
            # Drift rate = how much drift is changing (second derivative)
            if len(drift_per_layer) >= 2:
                rate = abs(drift_per_layer[-1] - drift_per_layer[-2])
            else:
                rate = d
            drift_rate.append(rate)

        # ── Compute exit quality at each layer ──
        layer_quality = []
        for i in range(1, n_layers + 1):
            hs = hidden_states[i]
            with torch.no_grad():
                normed = final_norm(hs)
                exit_logits = lm_head(normed)
            exit_logits_np = exit_logits[0, -1, :].cpu().float().numpy()
            exit_top1 = int(np.argmax(exit_logits_np))
            exit_top5 = set(np.argsort(exit_logits_np)[-5:].tolist())

            # Cosine similarity
            dot = float(np.dot(full_logits, exit_logits_np))
            n1 = float(np.linalg.norm(full_logits))
            n2 = float(np.linalg.norm(exit_logits_np))
            cos = dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

            layer_quality.append({
                "layer": i,
                "logits_cos": round(cos, 6),
                "top1_match": exit_top1 == full_top1,
                "top5_overlap": len(exit_top5 & full_top5),
            })

        # ── Drift-gated exit: find where drift rate drops below threshold ──
        drift_exits = {}
        for threshold in DRIFT_THRESHOLDS:
            exit_layer = n_layers  # Default: use all layers
            for i in range(MIN_LAYERS - 1, len(drift_rate)):
                # Exit when instantaneous drift drops below threshold
                # AND stays low for 2 consecutive layers (avoid noise)
                if i >= 1 and drift_rate[i] < threshold and drift_rate[i - 1] < threshold:
                    exit_layer = i + 1  # +1 because drift_rate is 0-indexed from layer 1
                    break
            drift_exits[threshold] = exit_layer

        # ── Fixed exit at layer 17 ──
        fixed_exit = min(FIXED_EXIT_LAYER, n_layers)

        # ── Oracle: earliest layer with top1 match ──
        oracle_exit = n_layers
        for lq in layer_quality:
            if lq["top1_match"]:
                oracle_exit = lq["layer"]
                break

        # ── Report ──
        print(f"  Eff rank profile: [{eranks[0]:.1f} → {eranks[-1]:.1f}]")
        print(f"  ΔΣt (total drift): {cumulative_drift[-1]:.1f}")
        print(f"  Oracle exit: layer {oracle_exit}")
        print(f"  Fixed exit (L{FIXED_EXIT_LAYER}): "
              f"cos={layer_quality[fixed_exit-1]['logits_cos']:.4f}, "
              f"top1={'✓' if layer_quality[fixed_exit-1]['top1_match'] else '✗'}")

        for threshold in DRIFT_THRESHOLDS:
            dl = drift_exits[threshold]
            q = layer_quality[dl - 1]
            print(f"  Drift exit (τ={threshold}): layer {dl}, "
                  f"cos={q['logits_cos']:.4f}, "
                  f"top1={'✓' if q['top1_match'] else '✗'}")

        all_results[prompt_id] = {
            "seq_len": seq_len,
            "full_top1": full_top1,
            "eranks": [round(e, 2) for e in eranks],
            "drift_per_layer": [round(d, 3) for d in drift_per_layer],
            "cumulative_drift": [round(d, 3) for d in cumulative_drift],
            "drift_rate": [round(d, 3) for d in drift_rate],
            "layer_quality": layer_quality,
            "oracle_exit": oracle_exit,
            "fixed_exit": fixed_exit,
            "drift_exits": {str(t): v for t, v in drift_exits.items()},
        }

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS: Compare exit strategies
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON: Exit strategies across all queries")
    print("=" * 70)

    prompt_ids = list(all_results.keys())

    # Header
    print(f"\n{'Strategy':>25s}  {'Avg layer':>9s}  {'Avg cos':>8s}  "
          f"{'Top1 %':>7s}  {'Top5 avg':>8s}  {'Layers saved':>12s}")
    print("-" * 80)

    strategies = {}

    # Oracle
    oracle_layers = [all_results[pid]["oracle_exit"] for pid in prompt_ids]
    oracle_cos = [all_results[pid]["layer_quality"][all_results[pid]["oracle_exit"] - 1]["logits_cos"]
                  for pid in prompt_ids]
    oracle_top1 = [all_results[pid]["layer_quality"][all_results[pid]["oracle_exit"] - 1]["top1_match"]
                   for pid in prompt_ids]
    oracle_top5 = [all_results[pid]["layer_quality"][all_results[pid]["oracle_exit"] - 1]["top5_overlap"]
                   for pid in prompt_ids]
    strategies["oracle"] = {
        "avg_layer": np.mean(oracle_layers),
        "avg_cos": np.mean(oracle_cos),
        "top1_pct": sum(oracle_top1) / len(oracle_top1) * 100,
        "avg_top5": np.mean(oracle_top5),
    }
    print(f"{'Oracle (best possible)':>25s}  {np.mean(oracle_layers):>9.1f}  "
          f"{np.mean(oracle_cos):>8.4f}  {strategies['oracle']['top1_pct']:>6.1f}%  "
          f"{np.mean(oracle_top5):>8.1f}  "
          f"{(n_layers - np.mean(oracle_layers)) / n_layers * 100:>11.1f}%")

    # Fixed
    fixed_cos = [all_results[pid]["layer_quality"][FIXED_EXIT_LAYER - 1]["logits_cos"]
                 for pid in prompt_ids]
    fixed_top1 = [all_results[pid]["layer_quality"][FIXED_EXIT_LAYER - 1]["top1_match"]
                  for pid in prompt_ids]
    fixed_top5 = [all_results[pid]["layer_quality"][FIXED_EXIT_LAYER - 1]["top5_overlap"]
                  for pid in prompt_ids]
    strategies["fixed_17"] = {
        "avg_layer": FIXED_EXIT_LAYER,
        "avg_cos": np.mean(fixed_cos),
        "top1_pct": sum(fixed_top1) / len(fixed_top1) * 100,
        "avg_top5": np.mean(fixed_top5),
    }
    print(f"{'Fixed (layer 17)':>25s}  {FIXED_EXIT_LAYER:>9.1f}  "
          f"{np.mean(fixed_cos):>8.4f}  {strategies['fixed_17']['top1_pct']:>6.1f}%  "
          f"{np.mean(fixed_top5):>8.1f}  "
          f"{(n_layers - FIXED_EXIT_LAYER) / n_layers * 100:>11.1f}%")

    # Full model (baseline)
    strategies["full_model"] = {
        "avg_layer": n_layers,
        "avg_cos": 1.0,
        "top1_pct": 100.0,
        "avg_top5": 5.0,
    }
    print(f"{'Full model (layer 22)':>25s}  {n_layers:>9.1f}  "
          f"{1.0:>8.4f}  {100.0:>6.1f}%  {5.0:>8.1f}  {0.0:>11.1f}%")

    # Drift-gated at each threshold
    best_drift_threshold = None
    best_drift_score = -1

    for threshold in DRIFT_THRESHOLDS:
        t_key = str(threshold)
        d_layers = [all_results[pid]["drift_exits"][t_key] for pid in prompt_ids]
        d_cos = [all_results[pid]["layer_quality"][all_results[pid]["drift_exits"][t_key] - 1]["logits_cos"]
                 for pid in prompt_ids]
        d_top1 = [all_results[pid]["layer_quality"][all_results[pid]["drift_exits"][t_key] - 1]["top1_match"]
                  for pid in prompt_ids]
        d_top5 = [all_results[pid]["layer_quality"][all_results[pid]["drift_exits"][t_key] - 1]["top5_overlap"]
                  for pid in prompt_ids]

        avg_layer = np.mean(d_layers)
        avg_cos = np.mean(d_cos)
        top1_pct = sum(d_top1) / len(d_top1) * 100
        avg_top5 = np.mean(d_top5)

        strategies[f"drift_{threshold}"] = {
            "avg_layer": float(avg_layer),
            "avg_cos": float(avg_cos),
            "top1_pct": float(top1_pct),
            "avg_top5": float(avg_top5),
            "threshold": threshold,
        }

        # Score: top1_pct * savings (higher = better tradeoff)
        savings = (n_layers - avg_layer) / n_layers
        score = top1_pct * savings
        if score > best_drift_score:
            best_drift_score = score
            best_drift_threshold = threshold

        print(f"{'Drift (τ=' + str(threshold) + ')':>25s}  {avg_layer:>9.1f}  "
              f"{avg_cos:>8.4f}  {top1_pct:>6.1f}%  {avg_top5:>8.1f}  "
              f"{savings * 100:>11.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # Per-query drift profile
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ΔΣt PROFILES: Cumulative drift by query type")
    print("=" * 70)

    # Show drift at key layers for each query
    key_layers = [5, 10, 15, 20, 22]
    print(f"\n{'Query':>25s}", end="")
    for kl in key_layers:
        print(f"  {'L' + str(kl):>6s}", end="")
    print(f"  {'Total':>7s}")
    print("-" * 75)

    for pid in prompt_ids:
        cd = all_results[pid]["cumulative_drift"]
        print(f"{pid:>25s}", end="")
        for kl in key_layers:
            idx = min(kl - 1, len(cd) - 1)
            print(f"  {cd[idx]:>6.1f}", end="")
        print(f"  {cd[-1]:>7.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # KEY QUESTION: Does drift rate correlate with query difficulty?
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY QUESTION: Does total drift predict query difficulty?")
    print("=" * 70)

    total_drifts = [all_results[pid]["cumulative_drift"][-1] for pid in prompt_ids]
    oracle_exits = [all_results[pid]["oracle_exit"] for pid in prompt_ids]

    from scipy import stats as sp_stats
    if len(set(oracle_exits)) > 1:
        r_drift_exit, p_drift_exit = sp_stats.pearsonr(total_drifts, oracle_exits)
        print(f"\n  Total ΔΣt vs oracle exit layer: r={r_drift_exit:.3f}, p={p_drift_exit:.4f}")
    else:
        r_drift_exit = 0.0
        p_drift_exit = 1.0
        print(f"\n  All queries have same oracle exit — no variance")

    for pid in prompt_ids:
        print(f"  {pid:>25s}: ΔΣt={all_results[pid]['cumulative_drift'][-1]:>6.1f}, "
              f"oracle={all_results[pid]['oracle_exit']}")

    # ══════════════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Does drift-gated beat fixed?
    best_drift = strategies[f"drift_{best_drift_threshold}"]
    fixed = strategies["fixed_17"]

    drift_beats_fixed_quality = best_drift["top1_pct"] >= fixed["top1_pct"]
    drift_beats_fixed_savings = best_drift["avg_layer"] < fixed["avg_layer"]
    drift_beats_both = drift_beats_fixed_quality and drift_beats_fixed_savings

    if drift_beats_both:
        verdict = (f"ΔΣt CONFIRMED — drift-gated (τ={best_drift_threshold}) beats fixed exit: "
                   f"same or better quality at fewer layers")
        action = "PROCEED: Drift-gated routing is a viable early exit strategy"
    elif drift_beats_fixed_quality:
        verdict = (f"ΔΣt PARTIAL — drift-gated has better quality but uses more layers "
                   f"(τ={best_drift_threshold}: {best_drift['avg_layer']:.1f} vs fixed: {fixed['avg_layer']:.1f})")
        action = "MARGINAL: Drift gate improves quality but not efficiency"
    elif drift_beats_fixed_savings:
        verdict = (f"ΔΣt PARTIAL — drift-gated saves layers but loses quality "
                   f"(τ={best_drift_threshold}: {best_drift['top1_pct']:.0f}% vs fixed: {fixed['top1_pct']:.0f}%)")
        action = "MARGINAL: Drift gate saves compute but at quality cost"
    else:
        verdict = "ΔΣt NOT CONFIRMED — fixed exit matches or beats drift-gated on this model"
        action = "STOP: Static policy is sufficient for TinyLlama"

    # Does drift predict difficulty?
    if abs(r_drift_exit) > 0.5 and p_drift_exit < 0.1:
        drift_predicts = True
        verdict += f"\n  ΔΣt PREDICTS difficulty: r={r_drift_exit:.3f}"
    else:
        drift_predicts = False
        verdict += f"\n  ΔΣt does NOT predict difficulty: r={r_drift_exit:.3f}"

    print(f"\n{verdict}")
    print(f"Action: {action}")

    # ── Cleanup ──
    del model
    rapl.__exit__(None, None, None)

    # ── Receipt ──
    wall = round(time.time() - t_start, 3)
    cpu = round(time.process_time() - cpu_start, 3)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    receipt = {
        "work_order": "WO-SENSING-PROBE-05",
        "question": "Does drift-gated routing (ΔΣt) outperform static exit?",
        "equation_term": "ΔΣt ⋅ KR(λᵢ, ηᵢ) ⋅ Θ(cᵢ)",
        "verdict": verdict,
        "action": action,
        "model": "TinyLlama-1.1B-Chat-v1.0 (dense FP32)",
        "n_layers": n_layers,
        "n_prompts": len(prompt_ids),
        "fixed_exit_layer": FIXED_EXIT_LAYER,
        "best_drift_threshold": best_drift_threshold,
        "drift_predicts_difficulty": drift_predicts,
        "drift_difficulty_correlation": {
            "r": round(float(r_drift_exit), 4),
            "p": round(float(p_drift_exit), 4),
        },
        "strategies": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                           for kk, vv in v.items()}
                       for k, v in strategies.items()},
        "per_query": all_results,
        "cost": {
            "wall_time_s": wall,
            "cpu_time_s": cpu,
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    if rapl.available and rapl.joules is not None:
        receipt["cost"]["energy_joules"] = round(rapl.joules, 3)

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = out_dir / f"drift_gated_probe_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, cls=NumpyEncoder)

    print(f"\nReceipt: {receipt_path}")
    print(f"Cost: {wall}s wall, {cpu}s CPU, "
          f"{receipt['cost']['peak_memory_mb']} MB peak")


if __name__ == "__main__":
    main()
