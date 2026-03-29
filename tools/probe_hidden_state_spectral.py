#!/usr/bin/env python3
"""
Probes 2+3: Hidden state spectral analysis for early exit and query separation.

WO-SENSING-PROBE-02: Does kurtosis on intermediate activations predict early
exit quality? If layer N's activation kurtosis stabilizes, can you skip
remaining layers without quality loss?

WO-SENSING-PROBE-03: Does a composite spectral signal (kurtosis + effective
rank + entropy) separate query types better than kurtosis alone?

Bonus: Does kurtosis correlate with effective rank? If so, kurtosis is a
cheap O(n) proxy for the O(n³) spectral decomposition.

Model: TinyLlama-1.1B-Chat-v1.0 (dense FP32, 22 layers)

Measurements per layer per prompt:
  - Kurtosis (distribution shape — heavy tails)
  - Shannon entropy (binned, information density)
  - Effective rank (exp(H(σ)) where σ = normalized singular values)
  - Stable rank (||A||_F² / ||A||_2²)
  - Early exit quality (logits cosine vs full model, top-1 agreement)

Analysis:
  - Probe 2: kurtosis vs early-exit quality correlation
  - Probe 3: metric discrimination power (which separates queries best?)
  - Bonus: kurtosis vs effective rank scatter (cheap proxy test)
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

# ── Same diverse query types as Probe 1 ──
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


# ── Measurement functions ──

def kurtosis(arr: np.ndarray) -> float:
    """Excess kurtosis (Fisher's definition)."""
    m = np.mean(arr)
    s = np.std(arr)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


def shannon_entropy_binned(arr: np.ndarray, n_bins: int = 256) -> float:
    """Shannon entropy of binned distribution (bits)."""
    if arr.size < 2:
        return 0.0
    hist, _ = np.histogram(arr, bins=n_bins)
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    return float(-np.sum(probs * np.log2(probs)))


def effective_rank(matrix: np.ndarray) -> float:
    """Effective rank = exp(H(σ_norm)) where σ are singular values.

    Measures how many dimensions carry significant signal.
    Range: 1 (rank-1) to min(m,n) (full rank, uniform spectrum).
    """
    # matrix is [seq_len, hidden_dim]
    try:
        s = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return 1.0
    s = s[s > 1e-10]  # Drop near-zero singular values
    if len(s) == 0:
        return 1.0
    # Normalize to a probability distribution
    p = s / s.sum()
    # Shannon entropy of singular value distribution
    h = -np.sum(p * np.log(p))
    return float(np.exp(h))


def stable_rank(matrix: np.ndarray) -> float:
    """Stable rank = ||A||_F² / ||A||_2².

    Cheaper than effective rank (no full SVD needed).
    ||A||_2 = largest singular value, ||A||_F = sqrt(sum of all σ²).
    """
    fro_sq = float(np.sum(matrix ** 2))
    # Only need largest singular value
    try:
        s_max = np.linalg.svd(matrix, compute_uv=False)[0]
    except np.linalg.LinAlgError:
        return 1.0
    if s_max < 1e-10:
        return 1.0
    return fro_sq / (s_max ** 2)


def main():
    out_dir = Path(__file__).resolve().parent.parent / "receipts" / "hidden_state_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROBES 2+3: Hidden state spectral analysis")
    print("WO-SENSING-PROBE-02 (early exit) + WO-SENSING-PROBE-03 (Se composite)")
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

    # Access final layernorm and lm_head for early exit simulation
    final_norm = model.model.norm
    lm_head = model.lm_head
    n_layers = model.config.num_hidden_layers
    print(f"  Model: {n_layers} layers, hidden={model.config.hidden_size}")

    # ── Run each prompt ──
    results = {}

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

        # outputs.hidden_states: tuple of (n_layers+1) tensors [1, seq_len, hidden]
        # Index 0 = embedding output, 1..n_layers = after each transformer block
        hidden_states = outputs.hidden_states
        full_logits = outputs.logits  # [1, seq_len, vocab]

        # Measure full-model logits for early exit comparison
        full_logits_np = full_logits[0, -1, :].cpu().float().numpy()  # last token
        full_top1 = int(np.argmax(full_logits_np))

        layer_measurements = []
        for i in range(n_layers + 1):
            hs = hidden_states[i]  # [1, seq_len, hidden]
            hs_np = hs[0].cpu().float().numpy()  # [seq_len, hidden]
            hs_flat = hs_np.ravel()

            # ── Spectral measurements ──
            kurt = kurtosis(hs_flat)
            entropy = shannon_entropy_binned(hs_flat)

            # Effective rank and stable rank on the [seq_len, hidden] matrix
            # For very short sequences, SVD is trivial
            if seq_len >= 2:
                eff_rank = effective_rank(hs_np)
                stab_rank = stable_rank(hs_np)
            else:
                eff_rank = 1.0
                stab_rank = 1.0

            # ── Early exit quality (skip for embedding layer 0) ──
            if i > 0:
                with torch.no_grad():
                    normed = final_norm(hs)
                    exit_logits = lm_head(normed)
                exit_logits_np = exit_logits[0, -1, :].cpu().float().numpy()
                exit_top1 = int(np.argmax(exit_logits_np))

                # Cosine similarity between exit logits and full logits
                dot = float(np.dot(full_logits_np, exit_logits_np))
                norm_full = float(np.linalg.norm(full_logits_np))
                norm_exit = float(np.linalg.norm(exit_logits_np))
                if norm_full > 0 and norm_exit > 0:
                    logits_cos = dot / (norm_full * norm_exit)
                else:
                    logits_cos = 0.0

                top1_match = exit_top1 == full_top1
            else:
                logits_cos = None
                top1_match = None
                exit_top1 = None

            # ── Composite Se for activations ──
            # Se_act = log1p(H_norm * C * K_norm)
            # H_norm = entropy / 8.0 (normalize to ~[0,1] for float32)
            # C = eff_rank / min(seq_len, hidden_dim) (fraction of active dims)
            # K_norm = min(kurt / 10.0, 1.0) (normalized kurtosis)
            h_norm = min(entropy / 8.0, 1.0)
            c_ratio = eff_rank / min(seq_len, model.config.hidden_size) if seq_len >= 2 else 0.0
            k_norm = min(abs(kurt) / 10.0, 1.0)
            se_act = math.log1p(h_norm * c_ratio * k_norm)

            layer_measurements.append({
                "layer": i,
                "kurtosis": round(kurt, 3),
                "entropy": round(entropy, 3),
                "effective_rank": round(eff_rank, 2),
                "stable_rank": round(stab_rank, 2),
                "se_act": round(se_act, 6),
                "logits_cos": round(logits_cos, 6) if logits_cos is not None else None,
                "top1_match": top1_match,
                "exit_top1": exit_top1,
            })

        results[prompt_id] = {
            "seq_len": seq_len,
            "full_top1": full_top1,
            "layers": layer_measurements,
        }

        # Quick summary
        kurts = [m["kurtosis"] for m in layer_measurements]
        eranks = [m["effective_rank"] for m in layer_measurements]
        cos_vals = [m["logits_cos"] for m in layer_measurements if m["logits_cos"] is not None]
        top1s = [m["top1_match"] for m in layer_measurements if m["top1_match"] is not None]
        first_match = next((m["layer"] for m in layer_measurements
                           if m["top1_match"] is True), n_layers)
        print(f"  Kurtosis: [{min(kurts):.1f}, {max(kurts):.1f}]")
        print(f"  Eff rank: [{min(eranks):.1f}, {max(eranks):.1f}]")
        print(f"  Exit cos: [{min(cos_vals):.4f}, {max(cos_vals):.4f}]")
        print(f"  Top1 match from layer: {first_match}/{n_layers}")

    # ══════════════════════════════════════════════════════════════════════
    # PROBE 2 ANALYSIS: Early Exit Quality vs Kurtosis
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PROBE 2: Does kurtosis predict early exit quality?")
    print("=" * 70)

    prompt_ids = list(results.keys())

    # Per-layer: average logits cosine across queries
    print(f"\n{'Layer':>5}  {'Avg cos':>8}  {'Top1 %':>7}  {'Avg kurt':>9}  "
          f"{'Avg eff_rank':>12}  {'Avg stable_rank':>15}")
    print("-" * 70)

    layer_avg_cos = []
    layer_avg_kurt = []
    layer_avg_erank = []
    layer_top1_pct = []

    for i in range(1, n_layers + 1):
        cos_vals = [results[pid]["layers"][i]["logits_cos"] for pid in prompt_ids]
        kurt_vals = [results[pid]["layers"][i]["kurtosis"] for pid in prompt_ids]
        erank_vals = [results[pid]["layers"][i]["effective_rank"] for pid in prompt_ids]
        srank_vals = [results[pid]["layers"][i]["stable_rank"] for pid in prompt_ids]
        top1_vals = [results[pid]["layers"][i]["top1_match"] for pid in prompt_ids]

        avg_cos = np.mean(cos_vals)
        avg_kurt = np.mean(kurt_vals)
        avg_erank = np.mean(erank_vals)
        avg_srank = np.mean(srank_vals)
        top1_pct = sum(top1_vals) / len(top1_vals) * 100

        layer_avg_cos.append(avg_cos)
        layer_avg_kurt.append(avg_kurt)
        layer_avg_erank.append(avg_erank)
        layer_top1_pct.append(top1_pct)

        flag = " ***" if top1_pct >= 100 else ""
        print(f"{i:>5}  {avg_cos:>8.4f}  {top1_pct:>6.1f}%  {avg_kurt:>9.2f}  "
              f"{avg_erank:>12.1f}  {avg_srank:>15.1f}{flag}")

    # Find earliest layer with 100% top-1 agreement (across all queries)
    first_perfect = next((i+1 for i, pct in enumerate(layer_top1_pct) if pct >= 100), n_layers)
    print(f"\nEarliest layer with 100% top-1 match across all queries: {first_perfect}/{n_layers}")
    if first_perfect < n_layers:
        skippable = n_layers - first_perfect
        savings = skippable / n_layers * 100
        print(f"  → Could skip {skippable} layers ({savings:.0f}% compute savings)")

    # Correlation: kurtosis vs logits cosine
    from scipy import stats as sp_stats
    kurt_arr = np.array(layer_avg_kurt)
    cos_arr = np.array(layer_avg_cos)
    erank_arr = np.array(layer_avg_erank)

    r_kurt_cos, p_kurt_cos = sp_stats.pearsonr(kurt_arr, cos_arr)
    r_erank_cos, p_erank_cos = sp_stats.pearsonr(erank_arr, cos_arr)

    print(f"\nCorrelation with exit quality (logits cosine):")
    print(f"  Kurtosis  ↔ cos: r={r_kurt_cos:.3f}, p={p_kurt_cos:.4f}")
    print(f"  Eff rank  ↔ cos: r={r_erank_cos:.3f}, p={p_erank_cos:.4f}")

    # Per-query: first layer where top1 matches
    print(f"\nPer-query earliest top1 match:")
    early_exit_layers = {}
    for pid in prompt_ids:
        layers = results[pid]["layers"]
        first = next((m["layer"] for m in layers if m["top1_match"] is True), n_layers)
        early_exit_layers[pid] = first
        print(f"  {pid:>25s}: layer {first}/{n_layers}")

    # Does kurtosis at early layers predict which queries can exit early?
    early_kurt = {}
    for pid in prompt_ids:
        # Kurtosis at layer 1 (earliest hidden state)
        early_kurt[pid] = results[pid]["layers"][1]["kurtosis"]

    print(f"\nDoes layer-1 kurtosis predict early exit layer?")
    exit_layers = [early_exit_layers[pid] for pid in prompt_ids]
    kurt_at_1 = [early_kurt[pid] for pid in prompt_ids]
    if len(set(exit_layers)) > 1:
        r_pred, p_pred = sp_stats.pearsonr(kurt_at_1, exit_layers)
        print(f"  r={r_pred:.3f}, p={p_pred:.4f}")
    else:
        print(f"  All queries exit at same layer — no variance to correlate")

    # ══════════════════════════════════════════════════════════════════════
    # PROBE 3 ANALYSIS: Metric discrimination power
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PROBE 3: Which metric separates query types best?")
    print("=" * 70)

    # For each metric, compute the average cross-query CV per layer
    metrics = ["kurtosis", "entropy", "effective_rank", "stable_rank", "se_act"]
    metric_cvs = {}

    print(f"\nAverage cross-query coefficient of variation per metric:")
    for metric in metrics:
        cvs = []
        for i in range(n_layers + 1):
            vals = [results[pid]["layers"][i][metric] for pid in prompt_ids]
            mean = np.mean(vals)
            std = np.std(vals)
            cv = std / max(abs(mean), 1e-6)
            cvs.append(cv)
        avg_cv = np.mean(cvs)
        metric_cvs[metric] = avg_cv
        print(f"  {metric:>20s}: avg CV = {avg_cv:.4f}")

    best_metric = max(metric_cvs, key=metric_cvs.get)
    print(f"\n  Best discriminator: {best_metric} (CV={metric_cvs[best_metric]:.4f})")

    # Can any metric distinguish "adversarial_repetitive" from others?
    print(f"\nAdversarial vs others — per-metric separation:")
    for metric in metrics:
        adv_vals = [results["adversarial_repetitive"]["layers"][i][metric]
                    for i in range(n_layers + 1)]
        other_vals = []
        for pid in prompt_ids:
            if pid == "adversarial_repetitive":
                continue
            other_vals.extend([results[pid]["layers"][i][metric]
                              for i in range(n_layers + 1)])
        adv_mean = np.mean(adv_vals)
        other_mean = np.mean(other_vals)
        pooled_std = np.std(adv_vals + other_vals)
        if pooled_std > 1e-6:
            cohens_d = abs(adv_mean - other_mean) / pooled_std
        else:
            cohens_d = 0.0
        print(f"  {metric:>20s}: adv={adv_mean:.3f} vs others={other_mean:.3f}, "
              f"Cohen's d={cohens_d:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # BONUS: Kurtosis vs Effective Rank correlation
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BONUS: Is kurtosis a cheap proxy for effective rank?")
    print("=" * 70)

    all_kurts = []
    all_eranks = []
    for pid in prompt_ids:
        for m in results[pid]["layers"]:
            if m["effective_rank"] is not None:
                all_kurts.append(m["kurtosis"])
                all_eranks.append(m["effective_rank"])

    r_proxy, p_proxy = sp_stats.pearsonr(all_kurts, all_eranks)
    r_spearman, p_spearman = sp_stats.spearmanr(all_kurts, all_eranks)
    print(f"  Pearson:  r={r_proxy:.3f}, p={p_proxy:.6f}")
    print(f"  Spearman: ρ={r_spearman:.3f}, p={p_spearman:.6f}")

    if abs(r_proxy) > 0.7:
        proxy_verdict = "STRONG — kurtosis is a viable proxy for effective rank"
    elif abs(r_proxy) > 0.4:
        proxy_verdict = "MODERATE — kurtosis captures some rank information"
    else:
        proxy_verdict = "WEAK — kurtosis and effective rank measure different things"
    print(f"  Verdict: {proxy_verdict}")

    # ══════════════════════════════════════════════════════════════════════
    # VERDICTS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VERDICTS")
    print("=" * 70)

    # Probe 2 verdict
    if first_perfect < n_layers * 0.8:
        p2_verdict = (f"SIGNAL EXISTS — top-1 converges at layer {first_perfect}/{n_layers}, "
                      f"{n_layers - first_perfect} layers skippable")
        p2_action = "PROCEED: Early exit at convergence layer saves compute"
    else:
        p2_verdict = (f"WEAK SIGNAL — top-1 needs {first_perfect}/{n_layers} layers, "
                      f"minimal savings")
        p2_action = "MARGINAL: Early exit saves < 20% compute"

    # Check if kurtosis predicts convergence
    if abs(r_kurt_cos) > 0.5 and p_kurt_cos < 0.05:
        p2_verdict += f"\n  Kurtosis predicts exit quality: r={r_kurt_cos:.3f}"
        p2_action += " — kurtosis is a viable convergence signal"
    else:
        p2_verdict += f"\n  Kurtosis does NOT predict exit quality: r={r_kurt_cos:.3f}"

    print(f"\nProbe 2 (early exit): {p2_verdict}")
    print(f"  Action: {p2_action}")

    # Probe 3 verdict
    if metric_cvs[best_metric] > 0.10:
        p3_verdict = f"SIGNAL EXISTS — {best_metric} has {metric_cvs[best_metric]:.3f} avg CV"
        if best_metric != "kurtosis" and metric_cvs[best_metric] > metric_cvs.get("kurtosis", 0) * 1.2:
            p3_action = f"PROCEED: {best_metric} separates queries better than kurtosis alone"
        else:
            p3_action = "KURTOSIS SUFFICIENT — composite adds marginal value"
    else:
        p3_verdict = "NO SIGNAL — no metric separates query types at >10% CV"
        p3_action = "STOP: Static policy is optimal"

    print(f"\nProbe 3 (query separation): {p3_verdict}")
    print(f"  Action: {p3_action}")

    # Bonus verdict
    print(f"\nBonus (cheap proxy): {proxy_verdict}")

    # ── Cleanup ──
    del model
    rapl.__exit__(None, None, None)

    # ── Receipt ──
    wall = round(time.time() - t_start, 3)
    cpu = round(time.process_time() - cpu_start, 3)

    receipt = {
        "work_orders": ["WO-SENSING-PROBE-02", "WO-SENSING-PROBE-03"],
        "probe_2_question": "Does kurtosis predict early exit quality?",
        "probe_2_verdict": p2_verdict,
        "probe_2_action": p2_action,
        "probe_3_question": "Does composite spectral signal separate query types better than kurtosis alone?",
        "probe_3_verdict": p3_verdict,
        "probe_3_action": p3_action,
        "bonus_question": "Is kurtosis a cheap proxy for effective rank?",
        "bonus_verdict": proxy_verdict,
        "bonus_correlation": {
            "pearson_r": round(r_proxy, 4),
            "pearson_p": round(p_proxy, 6),
            "spearman_rho": round(r_spearman, 4),
            "spearman_p": round(p_spearman, 6),
        },
        "model": "TinyLlama-1.1B-Chat-v1.0 (dense FP32)",
        "n_layers": n_layers,
        "n_prompts": len(prompt_ids),
        "prompt_ids": prompt_ids,
        "early_exit": {
            "first_perfect_layer": first_perfect,
            "skippable_layers": n_layers - first_perfect,
            "savings_pct": round((n_layers - first_perfect) / n_layers * 100, 1),
            "per_query_first_match": early_exit_layers,
            "kurt_cos_correlation": round(r_kurt_cos, 4),
            "erank_cos_correlation": round(r_erank_cos, 4),
        },
        "metric_discrimination": {
            "avg_cv_per_metric": {k: round(v, 4) for k, v in metric_cvs.items()},
            "best_discriminator": best_metric,
        },
        "layer_averages": {
            "logits_cos": [round(v, 4) for v in layer_avg_cos],
            "kurtosis": [round(v, 3) for v in layer_avg_kurt],
            "effective_rank": [round(v, 2) for v in layer_avg_erank],
        },
        "raw_results": results,
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

    # Convert numpy types for JSON serialization
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

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = out_dir / f"hidden_state_probe_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, cls=NumpyEncoder)

    print(f"\nReceipt: {receipt_path}")
    print(f"Cost: {wall}s wall, {cpu}s CPU, "
          f"{receipt['cost']['peak_memory_mb']} MB peak")


if __name__ == "__main__":
    main()
