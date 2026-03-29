#!/usr/bin/env python3
"""
Probe: Does KV activation kurtosis vary by query type?

WO-SENSING-PROBE-01: The critical gate for runtime adaptive precision.

If per-layer KV kurtosis varies meaningfully across query types, then
runtime measurement can drive dynamic precision allocation (build on
DP-LLM, KVmix, KurTail with unified spectral framework).

If per-layer KV kurtosis is CONSTANT regardless of input, the static
policy (layer 0 exact, rest VQ) is optimal and runtime measurement
adds zero value. Stop here.

Measurements per layer per prompt:
  - Kurtosis (distribution shape — heavy tails vs smooth)
  - Shannon entropy (information density — binned histogram)
  - Value range (dynamic range — spread)

Analysis:
  - Per-layer: cross-query variance vs within-query noise
  - Layer ranking: does the "hardest to compress" layer change by query?
  - Signal strength: is cross-query variation larger than measurement noise?

Model: TinyLlama-1.1B (HelixLinear compressed, 22 layers)
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

# ── Diverse query types ──
# Each exercises different model behaviors and attention patterns.
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


def kurtosis(arr: np.ndarray) -> float:
    """Excess kurtosis (Fisher's definition)."""
    m = np.mean(arr)
    s = np.std(arr)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


def shannon_entropy_binned(arr: np.ndarray, n_bins: int = 256) -> float:
    """Shannon entropy of binned distribution (bits).

    Bins the values into n_bins uniform buckets and computes
    H = -sum(p * log2(p)) for non-zero bins.
    """
    if arr.size < 2:
        return 0.0
    hist, _ = np.histogram(arr, bins=n_bins)
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    return float(-np.sum(probs * np.log2(probs)))


def main():
    out_dir = Path(__file__).resolve().parent.parent / "receipts" / "kv_kurtosis_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROBE: Does KV activation kurtosis vary by query type?")
    print("WO-SENSING-PROBE-01")
    print("=" * 70)

    # ── Load model ──
    # Use dense TinyLlama (HF cached) for accurate KV measurements.
    # The probe measures KV activation *variation* across queries — this is
    # a structural property of the architecture, independent of weight values.
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
    print(f"  Model loaded, device={next(model.parameters()).device}")

    # ── Run each prompt ──
    results = {}
    n_layers = None

    for prompt_id, text in PROMPTS.items():
        print(f"\n── {prompt_id} ──")

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        seq_len = inputs["input_ids"].shape[1]
        print(f"  Tokens: {seq_len}")

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, return_dict=True)

        past_kv = outputs.past_key_values
        if n_layers is None:
            n_layers = len(past_kv)

        layer_measurements = []
        for i, (k, v) in enumerate(past_kv):
            k_np = k.cpu().float().numpy().ravel()
            v_np = v.cpu().float().numpy().ravel()

            layer_measurements.append({
                "layer": i,
                "k_kurtosis": round(kurtosis(k_np), 3),
                "v_kurtosis": round(kurtosis(v_np), 3),
                "k_entropy": round(shannon_entropy_binned(k_np), 3),
                "v_entropy": round(shannon_entropy_binned(v_np), 3),
                "k_std": round(float(np.std(k_np)), 6),
                "v_std": round(float(np.std(v_np)), 6),
                "k_range": round(float(np.ptp(k_np)), 4),
                "v_range": round(float(np.ptp(v_np)), 4),
            })

        results[prompt_id] = {
            "seq_len": seq_len,
            "layers": layer_measurements,
        }

        # Quick summary
        k_kurts = [m["k_kurtosis"] for m in layer_measurements]
        v_kurts = [m["v_kurtosis"] for m in layer_measurements]
        print(f"  K kurtosis: [{min(k_kurts):.1f}, {max(k_kurts):.1f}]  "
              f"mean={np.mean(k_kurts):.1f}")
        print(f"  V kurtosis: [{min(v_kurts):.1f}, {max(v_kurts):.1f}]  "
              f"mean={np.mean(v_kurts):.1f}")

    # ── Cross-prompt analysis ──
    print("\n" + "=" * 70)
    print("ANALYSIS: Cross-query kurtosis variation")
    print("=" * 70)

    prompt_ids = list(results.keys())

    # Build matrices: [n_prompts x n_layers] for K and V kurtosis
    k_matrix = np.array([
        [results[pid]["layers"][i]["k_kurtosis"] for i in range(n_layers)]
        for pid in prompt_ids
    ])
    v_matrix = np.array([
        [results[pid]["layers"][i]["v_kurtosis"] for i in range(n_layers)]
        for pid in prompt_ids
    ])
    k_entropy_matrix = np.array([
        [results[pid]["layers"][i]["k_entropy"] for i in range(n_layers)]
        for pid in prompt_ids
    ])
    v_entropy_matrix = np.array([
        [results[pid]["layers"][i]["v_entropy"] for i in range(n_layers)]
        for pid in prompt_ids
    ])

    # Per-layer cross-query statistics
    print(f"\n{'Layer':>5}  {'K kurt mean':>11}  {'K kurt std':>10}  {'K kurt CV':>9}  "
          f"{'V kurt mean':>11}  {'V kurt std':>10}  {'V kurt CV':>9}")
    print("-" * 80)

    k_cvs = []
    v_cvs = []
    analysis_per_layer = []

    for i in range(n_layers):
        k_vals = k_matrix[:, i]
        v_vals = v_matrix[:, i]
        k_ent_vals = k_entropy_matrix[:, i]
        v_ent_vals = v_entropy_matrix[:, i]

        k_mean, k_std = np.mean(k_vals), np.std(k_vals)
        v_mean, v_std = np.mean(v_vals), np.std(v_vals)
        k_cv = k_std / max(abs(k_mean), 1e-6)
        v_cv = v_std / max(abs(v_mean), 1e-6)
        k_cvs.append(k_cv)
        v_cvs.append(v_cv)

        layer_analysis = {
            "layer": i,
            "k_kurtosis_mean": round(float(k_mean), 3),
            "k_kurtosis_std": round(float(k_std), 3),
            "k_kurtosis_cv": round(float(k_cv), 4),
            "k_kurtosis_range": [round(float(k_vals.min()), 3), round(float(k_vals.max()), 3)],
            "v_kurtosis_mean": round(float(v_mean), 3),
            "v_kurtosis_std": round(float(v_std), 3),
            "v_kurtosis_cv": round(float(v_cv), 4),
            "v_kurtosis_range": [round(float(v_vals.min()), 3), round(float(v_vals.max()), 3)],
            "k_entropy_mean": round(float(np.mean(k_ent_vals)), 3),
            "k_entropy_std": round(float(np.std(k_ent_vals)), 3),
            "v_entropy_mean": round(float(np.mean(v_ent_vals)), 3),
            "v_entropy_std": round(float(np.std(v_ent_vals)), 3),
        }
        analysis_per_layer.append(layer_analysis)

        flag = " ***" if k_cv > 0.10 or v_cv > 0.10 else ""
        print(f"{i:>5}  {k_mean:>11.2f}  {k_std:>10.3f}  {k_cv:>9.4f}  "
              f"{v_mean:>11.2f}  {v_std:>10.3f}  {v_cv:>9.4f}{flag}")

    # ── Key questions ──
    print("\n" + "=" * 70)
    print("KEY QUESTIONS")
    print("=" * 70)

    # Q1: Does kurtosis vary meaningfully?
    # Coefficient of variation > 10% = meaningful variation
    k_varying = sum(1 for cv in k_cvs if cv > 0.10)
    v_varying = sum(1 for cv in v_cvs if cv > 0.10)
    print(f"\nQ1: Layers with >10% cross-query kurtosis CV:")
    print(f"  K: {k_varying}/{n_layers} layers")
    print(f"  V: {v_varying}/{n_layers} layers")

    # Q2: Does the "hardest layer" ranking change?
    rankings = {}
    for pid in prompt_ids:
        k_kurts = [results[pid]["layers"][i]["k_kurtosis"] for i in range(n_layers)]
        top_layer = int(np.argmax(k_kurts))
        rankings[pid] = top_layer
    unique_tops = len(set(rankings.values()))
    print(f"\nQ2: Highest-kurtosis K layer by query type:")
    for pid, top in rankings.items():
        k_val = results[pid]["layers"][top]["k_kurtosis"]
        print(f"  {pid:>25s}: layer {top} (kurtosis={k_val:.1f})")
    print(f"  Unique top layers: {unique_tops}/{len(prompt_ids)}")
    ranking_changes = unique_tops > 1

    # Q3: Signal vs noise — between-query variance vs within-layer baseline
    k_between = float(np.mean([np.std(k_matrix[:, i]) for i in range(n_layers)]))
    v_between = float(np.mean([np.std(v_matrix[:, i]) for i in range(n_layers)]))
    k_within = float(np.std([np.mean(k_matrix[:, i]) for i in range(n_layers)]))
    v_within = float(np.std([np.mean(v_matrix[:, i]) for i in range(n_layers)]))
    print(f"\nQ3: Between-query vs within-model variance:")
    print(f"  K: between-query std={k_between:.3f}, within-model std={k_within:.3f}, "
          f"ratio={k_between / max(k_within, 1e-6):.3f}")
    print(f"  V: between-query std={v_between:.3f}, within-model std={v_within:.3f}, "
          f"ratio={v_between / max(v_within, 1e-6):.3f}")

    # Q4: Does entropy tell a different story than kurtosis?
    k_ent_cvs = [float(np.std(k_entropy_matrix[:, i]) / max(abs(np.mean(k_entropy_matrix[:, i])), 1e-6))
                 for i in range(n_layers)]
    v_ent_cvs = [float(np.std(v_entropy_matrix[:, i]) / max(abs(np.mean(v_entropy_matrix[:, i])), 1e-6))
                 for i in range(n_layers)]
    k_ent_varying = sum(1 for cv in k_ent_cvs if cv > 0.05)
    v_ent_varying = sum(1 for cv in v_ent_cvs if cv > 0.05)
    print(f"\nQ4: Layers with >5% cross-query entropy CV:")
    print(f"  K: {k_ent_varying}/{n_layers} layers")
    print(f"  V: {v_ent_varying}/{n_layers} layers")

    # ── Verdict ──
    print("\n" + "=" * 70)
    signal_exists = (k_varying >= 3 or v_varying >= 3) and ranking_changes

    if signal_exists:
        verdict = "SIGNAL EXISTS — runtime kurtosis carries query-dependent information"
        action = "PROCEED: Build dynamic per-layer precision allocation"
        print(f"VERDICT: {verdict}")
        print(f"ACTION:  {action}")
    else:
        verdict = "NO SIGNAL — kurtosis is static regardless of query type"
        action = "STOP: Static policy is optimal. Runtime measurement adds zero value."
        print(f"VERDICT: {verdict}")
        print(f"ACTION:  {action}")

    # ── Cleanup ──
    del model
    rapl.__exit__(None, None, None)

    # ── Receipt ──
    wall = round(time.time() - t_start, 3)
    cpu = round(time.process_time() - cpu_start, 3)

    receipt = {
        "work_order": "WO-SENSING-PROBE-01",
        "question": "Does KV activation kurtosis vary by query type?",
        "verdict": verdict,
        "action": action,
        "model": "TinyLlama-1.1B-Chat-v1.0 (dense FP32, from HF cache)",
        "n_layers": n_layers,
        "n_prompts": len(prompt_ids),
        "prompt_ids": prompt_ids,
        "measurements": {
            "kurtosis_varying_layers_k": k_varying,
            "kurtosis_varying_layers_v": v_varying,
            "entropy_varying_layers_k": k_ent_varying,
            "entropy_varying_layers_v": v_ent_varying,
            "ranking_changes": ranking_changes,
            "unique_top_layers": unique_tops,
            "between_query_std_k": round(k_between, 4),
            "between_query_std_v": round(v_between, 4),
            "within_model_std_k": round(k_within, 4),
            "within_model_std_v": round(v_within, 4),
        },
        "analysis_per_layer": analysis_per_layer,
        "raw_results": results,
        "layer_rankings_by_query": rankings,
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
    receipt_path = out_dir / f"kv_kurtosis_probe_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nReceipt: {receipt_path}")
    print(f"Cost: {wall}s wall, {cpu}s CPU, "
          f"{receipt['cost']['peak_memory_mb']} MB peak")


if __name__ == "__main__":
    main()
