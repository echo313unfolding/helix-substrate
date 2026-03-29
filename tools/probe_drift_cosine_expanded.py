#!/usr/bin/env python3
"""
Probe 5b+: ΔΣt expanded — 25 queries to test r=-0.811 significance.

WO-SENSING-PROBE-05b+: Same method as Probe 5b but with 25 diverse queries
to push p below 0.01 (or kill the finding). N=6 gave r=-0.811, p=0.0504.

Key question: Does cumulative cosine drift at layer 10 predict oracle exit layer?
If yes (p<0.01): real finding, proceed to scaling on Qwen 3B.
If no (p>0.05): the r=-0.811 was a fluke of small N. Kill the direction.

Model: TinyLlama-1.1B-Chat-v1.0 (dense FP32, 22 layers)
"""

import json
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

# ══════════════════════════════════════════════════════════════════════════
# 25 diverse prompts covering factual, code, creative, adversarial,
# technical, conversational, mathematical, multilingual, and edge cases.
# ══════════════════════════════════════════════════════════════════════════
PROMPTS = {
    # ── Factual (short) ──
    "factual_capital": "What is the capital of France?",
    "factual_element": "What is the chemical symbol for gold?",
    "factual_date": "When did World War II end?",

    # ── Factual (long context) ──
    "factual_long_bio": (
        "Marie Curie was a Polish-born physicist and chemist who conducted "
        "pioneering research on radioactivity. She was the first woman to win "
        "a Nobel Prize, the first person to win Nobel Prizes in two different "
        "sciences, and the first woman to become a professor at the University "
        "of Paris. Her achievements included the development of the theory of "
        "radioactivity, techniques for isolating radioactive isotopes, and the "
        "discovery of two elements, polonium and radium."
    ),
    "factual_long_process": (
        "The process of photosynthesis converts carbon dioxide and water into "
        "glucose and oxygen using energy from sunlight. This reaction occurs "
        "primarily in the chloroplasts of plant cells, where chlorophyll "
        "pigments absorb light energy. The light-dependent reactions take "
        "place in the thylakoid membranes, producing ATP and NADPH."
    ),

    # ── Code ──
    "code_fibonacci": (
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
        "\n"
        "for i in range(20):\n"
        "    print(f'F({i}) = {fibonacci(i)}')\n"
    ),
    "code_sorting": (
        "def quicksort(arr):\n"
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    pivot = arr[len(arr) // 2]\n"
        "    left = [x for x in arr if x < pivot]\n"
        "    middle = [x for x in arr if x == pivot]\n"
        "    right = [x for x in arr if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n"
    ),
    "code_class": (
        "class BinaryTree:\n"
        "    def __init__(self, value):\n"
        "        self.value = value\n"
        "        self.left = None\n"
        "        self.right = None\n"
        "\n"
        "    def insert(self, value):\n"
        "        if value < self.value:\n"
        "            if self.left is None:\n"
        "                self.left = BinaryTree(value)\n"
        "            else:\n"
        "                self.left.insert(value)\n"
        "        else:\n"
        "            if self.right is None:\n"
        "                self.right = BinaryTree(value)\n"
        "            else:\n"
        "                self.right.insert(value)\n"
    ),

    # ── Creative / narrative ──
    "creative_dream": (
        "In the dream, the ocean was made of glass and the fish swam through "
        "light instead of water. Each wave was a frozen moment, a sculpture "
        "of motion that never moved. The lighthouse keeper counted the colors "
        "of silence — seven, she decided, though the eighth was always hiding "
        "behind her left eye. The moon tasted like copper pennies and old songs."
    ),
    "creative_scifi": (
        "The colony ship drifted between stars for three hundred years before "
        "the AI woke the first sleeper. She opened her eyes to a ceiling of "
        "bioluminescent moss and the gentle hum of recycled air. The ship had "
        "changed while they slept, growing new corridors like roots seeking water."
    ),
    "creative_poetry": (
        "Roses are red, violets are blue, "
        "sugar is sweet, and so are you. "
        "The stars are bright, the moon is high, "
        "and somewhere a nightingale begins to cry."
    ),

    # ── Adversarial / repetitive ──
    "adversarial_repeat": (
        "the cat sat on the mat the cat sat on the mat the cat sat on the mat "
        "the cat sat on the mat the cat sat on the mat the cat sat on the mat "
        "the cat sat on the mat the cat sat on the mat the cat sat on the mat"
    ),
    "adversarial_nonsense": (
        "flurble glorpnax tintinnabulation quizzify blepstone morphwhistle "
        "fraxinating glimpoid snarkle durblefin crempwhistle zoxulate "
        "blithering plonkmaster quantifry whelkstone marblegurp"
    ),
    "adversarial_numbers": (
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 "
        "21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40"
    ),

    # ── Technical / mathematical ──
    "technical_transformer": (
        "The Transformer architecture uses multi-head self-attention where "
        "Q = XW_Q, K = XW_K, V = XW_V, and attention = softmax(QK^T/sqrt(d_k))V. "
        "For a model with 32 heads, d_model=4096, each head has d_k=128. "
        "The KV cache stores K and V tensors to avoid recomputation during "
        "autoregressive generation, trading memory for compute."
    ),
    "technical_calculus": (
        "The fundamental theorem of calculus states that if F is an antiderivative "
        "of f on [a,b], then the integral from a to b of f(x)dx equals F(b) - F(a). "
        "This connects differentiation and integration as inverse operations. "
        "For example, the integral of 3x^2 from 0 to 2 is x^3 evaluated from 0 to 2, "
        "which gives 8 - 0 = 8."
    ),
    "technical_chemistry": (
        "In organic chemistry, nucleophilic substitution reactions follow either "
        "SN1 or SN2 mechanisms. SN2 reactions are bimolecular, with the nucleophile "
        "attacking the electrophilic carbon simultaneously as the leaving group "
        "departs, resulting in inversion of stereochemistry (Walden inversion). "
        "SN1 reactions proceed through a carbocation intermediate."
    ),

    # ── Conversational / simple ──
    "conv_greeting": "Hello, how are you today?",
    "conv_opinion": "What do you think about the weather?",
    "conv_instruction": "Please tell me a short story about a dog.",

    # ── Mixed / multi-domain ──
    "mixed_legal": (
        "Under Section 230 of the Communications Decency Act, providers and "
        "users of interactive computer services are generally not treated as "
        "the publisher or speaker of information provided by another information "
        "content provider. This liability shield has been central to the growth "
        "of social media platforms and user-generated content websites."
    ),
    "mixed_medical": (
        "The human heart has four chambers: two atria and two ventricles. "
        "Deoxygenated blood enters the right atrium via the vena cava, "
        "passes to the right ventricle, and is pumped to the lungs through "
        "the pulmonary artery. Oxygenated blood returns to the left atrium "
        "via the pulmonary veins and is pumped to the body by the left ventricle."
    ),

    # ── Edge: very short ──
    "edge_single_word": "Hello",
    "edge_question_mark": "Why?",
}

MIN_LAYERS = 8
COS_THRESHOLDS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]


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


def main():
    out_dir = Path(__file__).resolve().parent.parent / "receipts" / "drift_gated_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROBE 5b+: ΔΣt expanded (25 queries, significance test)")
    print("WO-SENSING-PROBE-05b+")
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
    print(f"  {len(PROMPTS)} queries to process")

    FIXED_EXIT = 17
    all_results = {}

    for idx, (prompt_id, text) in enumerate(PROMPTS.items()):
        print(f"\n[{idx+1}/{len(PROMPTS)}] {prompt_id}")
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
        last_token_hs = [hs[0, -1, :].cpu().float() for hs in hidden_states]

        cos_distances = []
        cumulative_drift = []
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

        # ── CALM-style exit ──
        calm_exits = {}
        for threshold in COS_THRESHOLDS:
            exit_layer = n_layers
            for i in range(max(MIN_LAYERS - 1, 0), len(cos_distances)):
                if cos_distances[i] < threshold:
                    exit_layer = i + 1
                    break
            calm_exits[threshold] = exit_layer

        # ── ΔΣt-gated exit ──
        drift_rates = []
        for i in range(1, len(cos_distances)):
            drift_rates.append(abs(cos_distances[i] - cos_distances[i - 1]))

        delta_sigma_exits = {}
        for threshold in COS_THRESHOLDS:
            exit_layer = n_layers
            for i in range(max(MIN_LAYERS - 2, 0), len(drift_rates)):
                if i >= 1 and drift_rates[i] < threshold and drift_rates[i - 1] < threshold:
                    exit_layer = i + 2
                    break
            delta_sigma_exits[threshold] = exit_layer

        print(f"  Oracle: L{oracle_exit} | ΔΣt(total): {cumulative_drift[-1]:.4f} | "
              f"ΔΣt(L10): {cumulative_drift[9]:.4f}")

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
          f"{'T5':>3s}  {'Saved':>5s}")
    print("-" * 60)

    for s in strategies:
        savings = (n_layers - s["avg_layer"]) / n_layers * 100
        print(f"{s['name']:>20s}  {s['avg_layer']:>5.1f}  {s['avg_cos']:>6.3f}  "
              f"{s['top1_pct']:>5.1f}  {s['avg_top5']:>3.1f}  {savings:>4.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # KEY: Does ΔΣt accumulation predict oracle exit layer?
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY QUESTION: Does ΔΣt predict oracle exit layer? (N=%d)" % len(prompt_ids))
    print("Previous: r=-0.811, p=0.0504 (N=6). Need p<0.01 to confirm.")
    print("=" * 70)

    from scipy import stats as sp_stats

    oracle_exits = [all_results[pid]["oracle_exit"] for pid in prompt_ids]

    # Total drift
    total_drifts = [all_results[pid]["cumulative_drift"][-1] for pid in prompt_ids]
    if len(set(oracle_exits)) > 1:
        r_total, p_total = sp_stats.pearsonr(total_drifts, oracle_exits)
        rho_total, p_rho_total = sp_stats.spearmanr(total_drifts, oracle_exits)
    else:
        r_total, p_total, rho_total, p_rho_total = 0.0, 1.0, 0.0, 1.0

    print(f"\n  ΔΣt (total) vs oracle exit:")
    print(f"    Pearson  r = {r_total:.4f}, p = {p_total:.6f}")
    print(f"    Spearman ρ = {rho_total:.4f}, p = {p_rho_total:.6f}")

    # Drift at layer 10 (the key finding from Probe 5b)
    mid_drifts = [all_results[pid]["cumulative_drift"][9] for pid in prompt_ids]
    if len(set(oracle_exits)) > 1:
        r_mid, p_mid = sp_stats.pearsonr(mid_drifts, oracle_exits)
        rho_mid, p_rho_mid = sp_stats.spearmanr(mid_drifts, oracle_exits)
    else:
        r_mid, p_mid, rho_mid, p_rho_mid = 0.0, 1.0, 0.0, 1.0

    print(f"\n  ΔΣt at L10 vs oracle exit (KEY TEST):")
    print(f"    Pearson  r = {r_mid:.4f}, p = {p_mid:.6f}")
    print(f"    Spearman ρ = {rho_mid:.4f}, p = {p_rho_mid:.6f}")

    # Max single-layer drift
    max_drifts = [max(all_results[pid]["cos_distances"]) for pid in prompt_ids]
    if len(set(oracle_exits)) > 1:
        r_max, p_max = sp_stats.pearsonr(max_drifts, oracle_exits)
    else:
        r_max, p_max = 0.0, 1.0
    print(f"\n  Max single-layer drift vs oracle: r={r_max:.4f}, p={p_max:.6f}")

    # Drift at multiple checkpoints
    print(f"\n  Drift at various layers vs oracle exit:")
    checkpoint_correlations = {}
    for check_layer in [5, 8, 10, 12, 14, 16, 18]:
        if check_layer <= n_layers:
            check_drifts = [all_results[pid]["cumulative_drift"][check_layer - 1]
                           for pid in prompt_ids]
            if len(set(oracle_exits)) > 1:
                r_c, p_c = sp_stats.pearsonr(check_drifts, oracle_exits)
                print(f"    L{check_layer:>2d}: r={r_c:.4f}, p={p_c:.6f} {'***' if p_c < 0.01 else '**' if p_c < 0.05 else '*' if p_c < 0.1 else ''}")
                checkpoint_correlations[check_layer] = {"r": round(r_c, 4), "p": round(p_c, 6)}

    # ── Per-query detail ──
    print(f"\n  Per-query breakdown:")
    print(f"    {'Query':>25s}  {'Tokens':>6s}  {'Oracle':>6s}  {'ΔΣt(L10)':>8s}  {'ΔΣt(tot)':>8s}")
    print("    " + "-" * 60)

    sorted_prompts = sorted(prompt_ids, key=lambda pid: all_results[pid]["oracle_exit"])
    for pid in sorted_prompts:
        r = all_results[pid]
        print(f"    {pid:>25s}  {r['seq_len']:>6d}  L{r['oracle_exit']:>3d}    "
              f"{r['cumulative_drift'][9]:>8.4f}  {r['cumulative_drift'][-1]:>8.4f}")

    # ── Sequence length confound check ──
    seq_lens = [all_results[pid]["seq_len"] for pid in prompt_ids]
    if len(set(oracle_exits)) > 1 and len(set(seq_lens)) > 1:
        r_seq, p_seq = sp_stats.pearsonr(seq_lens, oracle_exits)
        r_drift_seq, p_drift_seq = sp_stats.pearsonr(mid_drifts, seq_lens)
        print(f"\n  Confound check:")
        print(f"    seq_len vs oracle: r={r_seq:.4f}, p={p_seq:.6f}")
        print(f"    ΔΣt(L10) vs seq_len: r={r_drift_seq:.4f}, p={p_drift_seq:.6f}")

        # Partial correlation: drift → oracle, controlling for seq_len
        if abs(r_seq) > 0.3:
            # Use partial correlation to check if drift effect survives
            from scipy.stats import pearsonr
            # Residualize both drift and oracle against seq_len
            slope_d, intercept_d, _, _, _ = sp_stats.linregress(seq_lens, mid_drifts)
            slope_o, intercept_o, _, _, _ = sp_stats.linregress(seq_lens, oracle_exits)
            resid_d = [d - (slope_d * s + intercept_d) for d, s in zip(mid_drifts, seq_lens)]
            resid_o = [o - (slope_o * s + intercept_o) for o, s in zip(oracle_exits, seq_lens)]
            r_partial, p_partial = pearsonr(resid_d, resid_o)
            print(f"    Partial r (drift→oracle | seq_len): r={r_partial:.4f}, p={p_partial:.6f}")
        else:
            r_partial, p_partial = r_mid, p_mid
            print(f"    seq_len not confounded (r={r_seq:.3f}), no partial needed")
    else:
        r_seq, p_seq = 0.0, 1.0
        r_partial, p_partial = r_mid, p_mid

    # ══════════════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Drift prediction verdict
    if p_mid < 0.01:
        drift_verdict = (f"CONFIRMED (p={p_mid:.6f} < 0.01): "
                        f"ΔΣt at L10 predicts oracle exit (r={r_mid:.3f})")
        drift_action = "PROCEED to Probe 4 on Qwen 3B — finding is real"
    elif p_mid < 0.05:
        drift_verdict = (f"SUGGESTIVE (p={p_mid:.6f} < 0.05): "
                        f"ΔΣt at L10 correlates with oracle exit (r={r_mid:.3f})")
        drift_action = "CAUTIOUS: consider Probe 4 but finding is borderline"
    elif p_mid < 0.1:
        drift_verdict = (f"MARGINAL (p={p_mid:.6f} < 0.1): "
                        f"Same as before — trend but not significant (r={r_mid:.3f})")
        drift_action = "LIKELY DEAD: r=-0.811 was inflated by small N"
    else:
        drift_verdict = (f"DEAD (p={p_mid:.6f}): "
                        f"No relationship between drift and oracle exit (r={r_mid:.3f})")
        drift_action = "KILL: ΔΣt does not predict oracle exit. Do not scale."

    print(f"\n  Drift prediction: {drift_verdict}")
    print(f"  Action: {drift_action}")

    # Exit strategy verdict
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
            exit_verdict = (f"ΔΣt BEATS CALM — fewer layers ({best_delta['avg_layer']:.1f} vs "
                           f"{best_calm['avg_layer']:.1f}) at same or better quality")
        elif best_delta["avg_layer"] == best_calm["avg_layer"]:
            exit_verdict = "ΔΣt TIES CALM — same exit points"
        else:
            exit_verdict = (f"CALM BEATS ΔΣt — {best_calm['avg_layer']:.1f} vs "
                           f"{best_delta['avg_layer']:.1f} layers")
    elif best_delta:
        exit_verdict = "Only ΔΣt reaches 50% top-1; CALM never does"
    elif best_calm:
        exit_verdict = "Only CALM reaches 50% top-1; ΔΣt never does"
    else:
        exit_verdict = "Neither reaches 50% — full model required"

    print(f"\n  Exit strategies: {exit_verdict}")

    # Oracle exit distribution
    from collections import Counter
    exit_dist = Counter(oracle_exits)
    print(f"\n  Oracle exit distribution: {dict(sorted(exit_dist.items()))}")
    print(f"  Oracle exit range: L{min(oracle_exits)}-L{max(oracle_exits)}")
    print(f"  Queries needing full model (L22): {exit_dist.get(n_layers, 0)}/{len(prompt_ids)}")

    # ── Cleanup ──
    del model
    rapl.__exit__(None, None, None)

    # ── Receipt ──
    wall = round(time.time() - t_start, 3)
    cpu = round(time.process_time() - cpu_start, 3)

    receipt = {
        "work_order": "WO-SENSING-PROBE-05b+",
        "question": "Does r=-0.811 (ΔΣt vs oracle exit) survive with N=25?",
        "previous_result": {"r": -0.811, "p": 0.0504, "n": 6},
        "expanded_result": {
            "total_drift": {"r": round(r_total, 4), "p": round(p_total, 6)},
            "mid_drift_L10": {"r": round(r_mid, 4), "p": round(p_mid, 6)},
            "spearman_L10": {"rho": round(rho_mid, 4), "p": round(p_rho_mid, 6)},
            "max_drift": {"r": round(r_max, 4), "p": round(p_max, 6)},
            "partial_r_controlling_seqlen": {"r": round(r_partial, 4), "p": round(p_partial, 6)},
            "checkpoint_correlations": checkpoint_correlations,
        },
        "confound_check": {
            "seq_len_vs_oracle": {"r": round(float(r_seq), 4), "p": round(float(p_seq), 6)},
        },
        "drift_verdict": drift_verdict,
        "drift_action": drift_action,
        "exit_verdict": exit_verdict,
        "model": "TinyLlama-1.1B-Chat-v1.0 (dense FP32)",
        "n_layers": n_layers,
        "n_prompts": len(prompt_ids),
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
    receipt_path = out_dir / f"drift_cosine_expanded_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, cls=NumpyEncoder)

    print(f"\nReceipt: {receipt_path}")
    print(f"Cost: {wall:.1f}s wall, {cpu:.1f}s CPU, "
          f"{receipt['cost']['peak_memory_mb']} MB peak")


if __name__ == "__main__":
    main()
