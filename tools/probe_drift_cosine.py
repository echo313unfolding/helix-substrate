#!/usr/bin/env python3
"""
Probe 5b: ΔΣt as cosine distance between consecutive hidden states.

WO-SENSING-PROBE-05b: Re-test drift-gated routing with the correct signal.

Probe 5 failed because effective rank measures subspace SHAPE, not CONTENT.
Cosine distance between consecutive hidden states measures how much the
representation is actually changing — when it stops changing, exit.

ΔΣt = cumulative sum of (1 - cos(h[i], h[i-1])) from layer 1 to N.
When the instantaneous drift (1 - cos) drops below threshold for 2
consecutive layers, the gate opens: Θ fires, exit.

Comparison:
  - Oracle: earliest layer with correct top-1 prediction
  - Fixed: exit at layer 17 (median from Probe 2)
  - CALM-style: exit when 1-cos < threshold (instantaneous, no accumulation)
  - ΔΣt-gated: exit when drift RATE stabilizes (accumulated, gated)

The question: does drift accumulation (ΔΣt from the equation) add value
over CALM's simpler instantaneous confidence signal?

Model: TinyLlama-1.1B-Chat-v1.0 (dense FP32, 22 layers)
"""

import json
import math
import sys
import time
import resource
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_substrate.rapl_meter import RaplMeter

# ── Cost tracking ──
t_start = time.time()
cpu_start = time.process_time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

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

MIN_LAYERS = 8  # Don't allow exit before layer 8

# Thresholds for cosine distance (1-cos). Smaller = stricter convergence.
COS_THRESHOLDS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]


def main():
    out_dir = Path(__file__).resolve().parent.parent / "receipts" / "drift_gated_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROBE 5b: ΔΣt as cosine distance (hidden state convergence)")
    print("WO-SENSING-PROBE-05b")
    print("=" * 70)

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading {MODEL_ID}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rapl = RaplMeter()
    rapl.__enter__()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu",
    )
    model.eval()

    final_norm = model.model.norm
    lm_head = model.lm_head
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, hidden={model.config.hidden_size}")

    FIXED_EXIT = 17
    all_results = {}

    for prompt_id, text in PROMPTS.items():
        print(f"\n── {prompt_id} ──")
        inputs = tokenizer(text, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]
        print(f"  Tokens: {seq_len}")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states
        full_logits = outputs.logits[0, -1, :].cpu().float()
        full_top1 = int(torch.argmax(full_logits))
        full_top5 = set(torch.topk(full_logits, 5).indices.tolist())

        # ── Cosine distance between consecutive hidden states ──
        # Use the LAST TOKEN's hidden state (the one that predicts next token)
        last_token_hs = [hs[0, -1, :].cpu().float() for hs in hidden_states]

        cos_distances = []  # 1 - cos(h[i], h[i-1])
        cumulative_drift = []  # ΔΣt
        running_drift = 0.0

        for i in range(1, n_layers + 1):
            cos_sim = F.cosine_similarity(
                last_token_hs[i].unsqueeze(0),
                last_token_hs[i - 1].unsqueeze(0),
            ).item()
            dist = 1.0 - cos_sim
            cos_distances.append(dist)
            running_drift += dist
            cumulative_drift.append(running_drift)

        # ── Exit quality at each layer ──
        layer_quality = []
        for i in range(1, n_layers + 1):
            with torch.no_grad():
                normed = final_norm(hidden_states[i])
                exit_logits = lm_head(normed)[0, -1, :].cpu().float()

            exit_top1 = int(torch.argmax(exit_logits))
            exit_top5 = set(torch.topk(exit_logits, 5).indices.tolist())

            cos_with_full = F.cosine_similarity(
                full_logits.unsqueeze(0), exit_logits.unsqueeze(0)
            ).item()

            layer_quality.append({
                "layer": i,
                "logits_cos": round(cos_with_full, 6),
                "top1_match": exit_top1 == full_top1,
                "top5_overlap": len(exit_top5 & full_top5),
            })

        # ── Oracle exit ──
        oracle_exit = n_layers
        for lq in layer_quality:
            if lq["top1_match"]:
                oracle_exit = lq["layer"]
                break

        # ── CALM-style exit: instantaneous cos distance < threshold ──
        calm_exits = {}
        for threshold in COS_THRESHOLDS:
            exit_layer = n_layers
            for i in range(max(MIN_LAYERS - 1, 0), len(cos_distances)):
                if cos_distances[i] < threshold:
                    exit_layer = i + 1
                    break
            calm_exits[threshold] = exit_layer

        # ── ΔΣt-gated exit: drift RATE drops below threshold ──
        # Drift rate = change in instantaneous drift between consecutive layers
        # When the drift rate itself stabilizes (drift is no longer accelerating
        # or decelerating), the system has found its regime.
        drift_rates = []
        for i in range(1, len(cos_distances)):
            drift_rates.append(abs(cos_distances[i] - cos_distances[i - 1]))

        delta_sigma_exits = {}
        for threshold in COS_THRESHOLDS:
            exit_layer = n_layers
            for i in range(max(MIN_LAYERS - 2, 0), len(drift_rates)):
                # Exit when drift rate drops below threshold for 2 consecutive
                if i >= 1 and drift_rates[i] < threshold and drift_rates[i - 1] < threshold:
                    exit_layer = i + 2  # +2: drift_rates[0] = between layers 1-2
                    break
            delta_sigma_exits[threshold] = exit_layer

        # ── Print summary ──
        print(f"  Cos distance profile: [{min(cos_distances):.6f}, {max(cos_distances):.6f}]")
        print(f"  ΔΣt (total): {cumulative_drift[-1]:.4f}")
        print(f"  Oracle: layer {oracle_exit}")
        fq = layer_quality[FIXED_EXIT - 1]
        print(f"  Fixed L{FIXED_EXIT}: cos={fq['logits_cos']:.4f}, "
              f"top1={'Y' if fq['top1_match'] else 'N'}")

        # Show best CALM and best ΔΣt
        for threshold in [0.001, 0.005, 0.01]:
            cl = calm_exits[threshold]
            cq = layer_quality[cl - 1]
            dl = delta_sigma_exits[threshold]
            dq = layer_quality[dl - 1]
            print(f"  τ={threshold}: CALM→L{cl} "
                  f"(cos={cq['logits_cos']:.4f}, top1={'Y' if cq['top1_match'] else 'N'}) | "
                  f"ΔΣt→L{dl} "
                  f"(cos={dq['logits_cos']:.4f}, top1={'Y' if dq['top1_match'] else 'N'})")

        all_results[prompt_id] = {
            "seq_len": seq_len,
            "full_top1": full_top1,
            "oracle_exit": oracle_exit,
            "cos_distances": [round(d, 8) for d in cos_distances],
            "cumulative_drift": [round(d, 6) for d in cumulative_drift],
            "drift_rates": [round(d, 8) for d in drift_rates],
            "layer_quality": layer_quality,
            "calm_exits": {str(t): v for t, v in calm_exits.items()},
            "delta_sigma_exits": {str(t): v for t, v in delta_sigma_exits.items()},
        }

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON: All exit strategies")
    print("=" * 70)

    prompt_ids = list(all_results.keys())

    def eval_strategy(name, get_layer_fn):
        layers = [get_layer_fn(pid) for pid in prompt_ids]
        cos_vals = [all_results[pid]["layer_quality"][get_layer_fn(pid) - 1]["logits_cos"]
                    for pid in prompt_ids]
        top1s = [all_results[pid]["layer_quality"][get_layer_fn(pid) - 1]["top1_match"]
                 for pid in prompt_ids]
        top5s = [all_results[pid]["layer_quality"][get_layer_fn(pid) - 1]["top5_overlap"]
                 for pid in prompt_ids]
        return {
            "name": name,
            "avg_layer": float(np.mean(layers)),
            "avg_cos": float(np.mean(cos_vals)),
            "top1_pct": float(sum(top1s) / len(top1s) * 100),
            "avg_top5": float(np.mean(top5s)),
            "layers": layers,
        }

    strategies = []
    strategies.append(eval_strategy("Oracle", lambda pid: all_results[pid]["oracle_exit"]))
    strategies.append(eval_strategy("Fixed L17", lambda pid: min(FIXED_EXIT, n_layers)))
    strategies.append(eval_strategy("Full (L22)", lambda pid: n_layers))

    for t in COS_THRESHOLDS:
        tk = str(t)
        strategies.append(eval_strategy(
            f"CALM τ={t}", lambda pid, _t=tk: all_results[pid]["calm_exits"][_t]))
        strategies.append(eval_strategy(
            f"ΔΣt τ={t}", lambda pid, _t=tk: all_results[pid]["delta_sigma_exits"][_t]))

    print(f"\n{'Strategy':>20s}  {'Avg L':>5s}  {'Cos':>6s}  {'Top1%':>5s}  "
          f"{'T5':>3s}  {'Saved':>5s}  Per-query layers")
    print("-" * 85)

    for s in strategies:
        savings = (n_layers - s["avg_layer"]) / n_layers * 100
        layer_str = ",".join(str(l) for l in s["layers"])
        print(f"{s['name']:>20s}  {s['avg_layer']:>5.1f}  {s['avg_cos']:>6.3f}  "
              f"{s['top1_pct']:>5.1f}  {s['avg_top5']:>3.1f}  {savings:>4.1f}%  [{layer_str}]")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS: Does ΔΣt add value over CALM?
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS: ΔΣt vs CALM at matched quality levels")
    print("=" * 70)

    # Find threshold where each strategy first achieves >= 66% top-1
    # (matches fixed L17 performance)
    target_top1 = 66.0

    print(f"\nFirst threshold achieving ≥{target_top1}% top-1:")
    for method in ["CALM", "ΔΣt"]:
        for t in COS_THRESHOLDS:
            s = next(s for s in strategies if s["name"] == f"{method} τ={t}")
            if s["top1_pct"] >= target_top1:
                savings = (n_layers - s["avg_layer"]) / n_layers * 100
                print(f"  {method}: τ={t}, avg layer={s['avg_layer']:.1f}, "
                      f"top1={s['top1_pct']:.0f}%, saved={savings:.1f}%")
                break
        else:
            print(f"  {method}: never reaches {target_top1}% top-1")

    # Find the Pareto-optimal strategies
    print(f"\nPareto front (highest top1% for each layer budget):")
    seen_layers = set()
    for s in sorted(strategies, key=lambda x: (x["avg_layer"], -x["top1_pct"])):
        layer_bucket = round(s["avg_layer"])
        if layer_bucket not in seen_layers and s["top1_pct"] > 0:
            seen_layers.add(layer_bucket)
            savings = (n_layers - s["avg_layer"]) / n_layers * 100
            print(f"  L≈{layer_bucket:>2d}: {s['name']:>20s} → "
                  f"top1={s['top1_pct']:>5.1f}%, cos={s['avg_cos']:.3f}, "
                  f"saved={savings:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # KEY: Does ΔΣt accumulation predict query difficulty?
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY: Does cumulative drift (ΔΣt) predict oracle exit layer?")
    print("=" * 70)

    total_drifts = [all_results[pid]["cumulative_drift"][-1] for pid in prompt_ids]
    oracle_exits = [all_results[pid]["oracle_exit"] for pid in prompt_ids]

    from scipy import stats as sp_stats
    if len(set(oracle_exits)) > 1:
        r, p = sp_stats.pearsonr(total_drifts, oracle_exits)
        print(f"\n  ΔΣt vs oracle exit: r={r:.3f}, p={p:.4f}")
    else:
        r, p = 0.0, 1.0
        print(f"\n  All same oracle exit — no variance")

    # Also check: does max instantaneous drift predict difficulty?
    max_drifts = [max(all_results[pid]["cos_distances"]) for pid in prompt_ids]
    if len(set(oracle_exits)) > 1:
        r_max, p_max = sp_stats.pearsonr(max_drifts, oracle_exits)
        print(f"  Max single-layer drift vs oracle: r={r_max:.3f}, p={p_max:.4f}")
    else:
        r_max, p_max = 0.0, 1.0

    # Check: drift at layer 10 (midpoint) predicts exit?
    mid_drifts = [all_results[pid]["cumulative_drift"][9] for pid in prompt_ids]  # layer 10
    if len(set(oracle_exits)) > 1:
        r_mid, p_mid = sp_stats.pearsonr(mid_drifts, oracle_exits)
        print(f"  ΔΣt at L10 vs oracle: r={r_mid:.3f}, p={p_mid:.4f}")
    else:
        r_mid, p_mid = 0.0, 1.0

    for pid in prompt_ids:
        print(f"  {pid:>25s}: ΔΣt={all_results[pid]['cumulative_drift'][-1]:.4f}, "
              f"max_d={max(all_results[pid]['cos_distances']):.6f}, "
              f"oracle=L{all_results[pid]['oracle_exit']}")

    # ══════════════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Find best CALM and best ΔΣt at ≥50% top-1
    best_calm = None
    best_delta = None
    for s in strategies:
        if s["top1_pct"] >= 50:
            if "CALM" in s["name"] and (best_calm is None or s["avg_layer"] < best_calm["avg_layer"]):
                best_calm = s
            if "ΔΣt" in s["name"] and (best_delta is None or s["avg_layer"] < best_delta["avg_layer"]):
                best_delta = s

    if best_calm and best_delta:
        if best_delta["avg_layer"] < best_calm["avg_layer"] and best_delta["top1_pct"] >= best_calm["top1_pct"]:
            verdict = (f"ΔΣt BEATS CALM — fewer layers ({best_delta['avg_layer']:.1f} vs "
                       f"{best_calm['avg_layer']:.1f}) at same or better quality")
            action = "CONFIRMED: Drift accumulation adds value over instantaneous signal"
        elif best_delta["avg_layer"] == best_calm["avg_layer"]:
            verdict = "ΔΣt TIES CALM — same exit points"
            action = "NO ADDED VALUE: Accumulation doesn't help over instantaneous"
        else:
            verdict = (f"CALM BEATS ΔΣt — CALM exits at {best_calm['avg_layer']:.1f} vs "
                       f"ΔΣt at {best_delta['avg_layer']:.1f}")
            action = "ΔΣt NOT CONFIRMED: Simpler instantaneous signal is sufficient"
    elif best_calm:
        verdict = "Only CALM reaches 50% top-1; ΔΣt never does"
        action = "ΔΣt NOT CONFIRMED"
    elif best_delta:
        verdict = "Only ΔΣt reaches 50% top-1; CALM never does"
        action = "ΔΣt CONFIRMED (CALM fails)"
    else:
        verdict = "Neither strategy reaches 50% top-1 before full model"
        action = "BOTH FAIL: Full model required"

    # Drift prediction
    if abs(r) > 0.5 and p < 0.1:
        verdict += f"\n  ΔΣt PREDICTS difficulty: r={r:.3f}"
    else:
        verdict += f"\n  ΔΣt does NOT predict difficulty: r={r:.3f}"

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
        "work_order": "WO-SENSING-PROBE-05b",
        "question": "Does ΔΣt (cosine distance drift) outperform CALM-style instantaneous exit?",
        "equation_term": "ΔΣt ⋅ KR(λᵢ, ηᵢ) ⋅ Θ(cᵢ)",
        "verdict": verdict,
        "action": action,
        "model": "TinyLlama-1.1B-Chat-v1.0 (dense FP32)",
        "n_layers": n_layers,
        "n_prompts": len(prompt_ids),
        "fixed_exit_layer": FIXED_EXIT,
        "drift_predicts_difficulty": abs(r) > 0.5 and p < 0.1,
        "drift_difficulty_correlation": {"r": round(float(r), 4), "p": round(float(p), 4)},
        "max_drift_correlation": {"r": round(float(r_max), 4), "p": round(float(p_max), 4)},
        "mid_drift_correlation": {"r": round(float(r_mid), 4), "p": round(float(p_mid), 4)},
        "strategies_summary": {s["name"]: {
            "avg_layer": round(s["avg_layer"], 2),
            "avg_cos": round(s["avg_cos"], 4),
            "top1_pct": round(s["top1_pct"], 1),
            "avg_top5": round(s["avg_top5"], 1),
        } for s in strategies},
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
    receipt_path = out_dir / f"drift_cosine_probe_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, cls=NumpyEncoder)

    print(f"\nReceipt: {receipt_path}")
    print(f"Cost: {wall}s wall, {cpu}s CPU, "
          f"{receipt['cost']['peak_memory_mb']} MB peak")


if __name__ == "__main__":
    main()
