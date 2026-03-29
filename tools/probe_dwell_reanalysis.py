#!/usr/bin/env python3
"""
Probe Dwell Reanalysis — Deep analysis of n_transitions finding.

Reads the Probe 5b+ receipt (no new forward passes) and computes
quiet-zone episode statistics from per-layer cosine distances, then
correlates them with oracle exit layer.

Key question: Can partial n_transitions (first K layers only) predict
oracle exit, making this useful as an ACTUAL early-exit signal?

Work order: WO-SENSING-PROBE-05b+-DWELL
"""

import json
import time
import resource
import platform
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from itertools import groupby

import numpy as np
from scipy import stats

# ── Cost tracking ──────────────────────────────────────────────
t_start = time.time()
cpu_start = time.process_time()
start_iso = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

# ── Load receipt ───────────────────────────────────────────────
RECEIPT_PATH = "/home/voidstr3m33/helix-substrate/receipts/drift_gated_probe/drift_cosine_expanded_20260329T084037.json"
OUT_DIR = "/home/voidstr3m33/helix-substrate/receipts/drift_gated_probe"

with open(RECEIPT_PATH, 'r') as f:
    receipt = json.load(f)

per_query = receipt['per_query']
n_layers = receipt['n_layers']
n_prompts = receipt['n_prompts']

print(f"Loaded receipt: {n_prompts} queries, {n_layers} layers")
print(f"Model: {receipt['model']}")
print()

# ── Extract data ───────────────────────────────────────────────
query_names = []
cos_dist_matrix = []  # shape: (n_queries, n_layers)
oracle_exits = []

for qname, qdata in per_query.items():
    query_names.append(qname)
    cos_dist_matrix.append(qdata['cos_distances'])
    oracle_exits.append(qdata['oracle_exit'])

cos_dist_matrix = np.array(cos_dist_matrix)  # (24, 22)
oracle_exits = np.array(oracle_exits)

print(f"cos_dist_matrix shape: {cos_dist_matrix.shape}")
print(f"Oracle exits: min={oracle_exits.min()}, max={oracle_exits.max()}, "
      f"mean={oracle_exits.mean():.1f}, median={np.median(oracle_exits):.1f}")
print()

# ── Episode computation ────────────────────────────────────────
def compute_episode_stats(cos_dists, threshold, max_layer=None):
    """
    Compute quiet-zone episode statistics for a single query.

    cos_dists: array of cosine distances (layer i to layer i+1)
    threshold: below this = quiet
    max_layer: if set, only use cos_dists[:max_layer] (for partial analysis)

    Returns dict of metrics.
    """
    if max_layer is not None:
        cd = cos_dists[:max_layer]
    else:
        cd = cos_dists

    n = len(cd)
    below = (cd < threshold).astype(int)  # 1 = quiet, 0 = active

    # --- Quiet episodes: contiguous runs of below-threshold layers ---
    quiet_episodes = []
    current_run = 0
    current_start = None
    for i, b in enumerate(below):
        if b == 1:
            if current_run == 0:
                current_start = i
            current_run += 1
        else:
            if current_run > 0:
                quiet_episodes.append((current_start, current_run))
            current_run = 0
    if current_run > 0:
        quiet_episodes.append((current_start, current_run))

    n_quiet_episodes = len(quiet_episodes)
    episode_lengths = [ep[1] for ep in quiet_episodes]
    avg_quiet_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
    max_quiet_episode_length = max(episode_lengths) if episode_lengths else 0
    first_quiet_start = quiet_episodes[0][0] if quiet_episodes else n  # n = never
    total_quiet_layers = int(below.sum())

    # --- Transition events: below→above crossings (settling → disruption) ---
    n_transition_events = 0
    for i in range(1, n):
        if below[i - 1] == 1 and below[i] == 0:
            n_transition_events += 1

    # --- Also count above→below (disruption → settling) ---
    n_settling_events = 0
    for i in range(1, n):
        if below[i - 1] == 0 and below[i] == 1:
            n_settling_events += 1

    return {
        'n_quiet_episodes': n_quiet_episodes,
        'n_transition_events': n_transition_events,
        'n_settling_events': n_settling_events,
        'avg_quiet_episode_length': round(avg_quiet_episode_length, 3),
        'max_quiet_episode_length': max_quiet_episode_length,
        'first_quiet_start': first_quiet_start,
        'total_quiet_layers': total_quiet_layers,
        'quiet_fraction': round(total_quiet_layers / n, 4),
    }


# ── Section 1: Multi-threshold analysis ───────────────────────
print("=" * 72)
print("SECTION 1: Multi-threshold episode analysis (all 22 layers)")
print("=" * 72)

thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
metric_names = [
    'n_quiet_episodes', 'n_transition_events', 'n_settling_events',
    'avg_quiet_episode_length', 'max_quiet_episode_length',
    'first_quiet_start', 'total_quiet_layers', 'quiet_fraction'
]

all_results = {}  # threshold -> {metric_name -> (pearson_r, pearson_p, spearman_rho, spearman_p)}

for tau in thresholds:
    print(f"\n--- Threshold τ = {tau} ---")

    # Compute per-query metrics
    metric_arrays = {m: [] for m in metric_names}
    for i in range(len(query_names)):
        stats_i = compute_episode_stats(cos_dist_matrix[i], tau)
        for m in metric_names:
            metric_arrays[m].append(stats_i[m])

    # Correlate each metric with oracle_exit
    tau_results = {}
    for m in metric_names:
        arr = np.array(metric_arrays[m])
        # Check for constant arrays (no variance)
        if np.std(arr) < 1e-10:
            print(f"  {m:30s}: CONSTANT ({arr[0]:.3f}) — no correlation possible")
            tau_results[m] = {'pearson_r': None, 'pearson_p': None,
                              'spearman_rho': None, 'spearman_p': None,
                              'mean': float(arr[0]), 'std': 0.0}
            continue

        pr, pp = stats.pearsonr(arr, oracle_exits)
        sr, sp = stats.spearmanr(arr, oracle_exits)
        sig_marker = ""
        if pp < 0.01:
            sig_marker = " ***"
        elif pp < 0.05:
            sig_marker = " **"
        elif pp < 0.10:
            sig_marker = " *"

        print(f"  {m:30s}: Pearson r={pr:+.4f} (p={pp:.4f}), "
              f"Spearman ρ={sr:+.4f} (p={sp:.4f}){sig_marker}")

        tau_results[m] = {
            'pearson_r': round(pr, 5),
            'pearson_p': round(pp, 6),
            'spearman_rho': round(sr, 5),
            'spearman_p': round(sp, 6),
            'mean': round(float(np.mean(arr)), 4),
            'std': round(float(np.std(arr)), 4),
        }

    all_results[str(tau)] = tau_results

# ── Find best predictor ───────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 2: Best predictor identification")
print("=" * 72)

best_r = 0
best_metric = None
best_tau = None

for tau_str, tau_res in all_results.items():
    for m, vals in tau_res.items():
        if vals['pearson_r'] is not None and abs(vals['pearson_r']) > abs(best_r):
            best_r = vals['pearson_r']
            best_metric = m
            best_tau = float(tau_str)

print(f"\nBest predictor: {best_metric} at τ={best_tau}")
print(f"  Pearson r = {best_r:+.4f}")

# Compute the best metric array
best_metric_arr = []
for i in range(len(query_names)):
    s = compute_episode_stats(cos_dist_matrix[i], best_tau)
    best_metric_arr.append(s[best_metric])
best_metric_arr = np.array(best_metric_arr)

# Also find best by |Spearman|
best_rho = 0
best_metric_sp = None
best_tau_sp = None
for tau_str, tau_res in all_results.items():
    for m, vals in tau_res.items():
        if vals['spearman_rho'] is not None and abs(vals['spearman_rho']) > abs(best_rho):
            best_rho = vals['spearman_rho']
            best_metric_sp = m
            best_tau_sp = float(tau_str)

print(f"\nBest by Spearman: {best_metric_sp} at τ={best_tau_sp}")
print(f"  Spearman ρ = {best_rho:+.4f}")


# ── Section 3: Effect size & ROC for best predictor ───────────
print("\n" + "=" * 72)
print("SECTION 3: Effect size & discrimination analysis (best predictor)")
print("=" * 72)

# Split: early exit (oracle < 18) vs late exit (oracle >= 18)
cutoff = 18
early_mask = oracle_exits < cutoff
late_mask = oracle_exits >= cutoff

early_vals = best_metric_arr[early_mask]
late_vals = best_metric_arr[late_mask]

print(f"\nCutoff: oracle_exit < {cutoff} (early) vs >= {cutoff} (late)")
print(f"  Early group: n={early_mask.sum()}, {best_metric} mean={early_vals.mean():.3f} ± {early_vals.std():.3f}")
print(f"  Late group:  n={late_mask.sum()}, {best_metric} mean={late_vals.mean():.3f} ± {late_vals.std():.3f}")

# Cohen's d
pooled_std = np.sqrt((early_vals.std()**2 + late_vals.std()**2) / 2)
if pooled_std > 0:
    cohens_d = (late_vals.mean() - early_vals.mean()) / pooled_std
    print(f"  Cohen's d = {cohens_d:+.3f}")
else:
    cohens_d = 0.0
    print(f"  Cohen's d = N/A (zero variance)")

# Mann-Whitney U test
if len(early_vals) > 0 and len(late_vals) > 0:
    u_stat, u_p = stats.mannwhitneyu(early_vals, late_vals, alternative='two-sided')
    print(f"  Mann-Whitney U = {u_stat:.1f}, p = {u_p:.4f}")
else:
    u_stat, u_p = None, None

# ROC-like analysis: sweep thresholds on the metric to discriminate early vs late
print(f"\nROC-like analysis: predicting 'needs layer >= {cutoff}' from {best_metric}")
metric_min = best_metric_arr.min()
metric_max = best_metric_arr.max()

if metric_max > metric_min:
    roc_thresholds = np.linspace(metric_min - 0.5, metric_max + 0.5, 50)
    tpr_list = []
    fpr_list = []
    for t in roc_thresholds:
        # Predict "late" if metric >= t (or <= t if negative correlation)
        if best_r > 0:
            pred_late = best_metric_arr >= t
        else:
            pred_late = best_metric_arr <= t
        tp = np.sum(pred_late & late_mask)
        fp = np.sum(pred_late & early_mask)
        fn = np.sum(~pred_late & late_mask)
        tn = np.sum(~pred_late & early_mask)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # AUC via trapezoidal rule (sort by fpr)
    sorted_pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in sorted_pairs]
    tpr_sorted = [p[1] for p in sorted_pairs]
    auc = np.trapezoid(tpr_sorted, fpr_sorted)
    print(f"  AUC = {abs(auc):.4f}")
else:
    auc = 0.5
    print(f"  AUC = N/A (constant metric)")


# ── Section 4: Leave-one-out cross-validation ──────────────────
print("\n" + "=" * 72)
print("SECTION 4: Leave-one-out cross-validation (best predictor)")
print("=" * 72)

loo_errors = []
loo_predictions = []
for i in range(len(query_names)):
    # Train on all but i
    train_x = np.delete(best_metric_arr, i)
    train_y = np.delete(oracle_exits, i)
    test_x = best_metric_arr[i]
    test_y = oracle_exits[i]

    # Simple linear regression
    if np.std(train_x) > 1e-10:
        slope, intercept, _, _, _ = stats.linregress(train_x, train_y)
        pred = slope * test_x + intercept
    else:
        pred = train_y.mean()

    loo_predictions.append(pred)
    loo_errors.append(abs(pred - test_y))

loo_errors = np.array(loo_errors)
loo_predictions = np.array(loo_predictions)

print(f"\nLOO-CV with linear regression on {best_metric} (τ={best_tau}):")
print(f"  Mean absolute error: {loo_errors.mean():.2f} layers")
print(f"  Median absolute error: {np.median(loo_errors):.2f} layers")
print(f"  Max absolute error: {loo_errors.max():.2f} layers")
print(f"  RMSE: {np.sqrt(np.mean(loo_errors**2)):.2f} layers")

# Baseline: always predict mean oracle exit
baseline_errors = np.abs(oracle_exits - oracle_exits.mean())
print(f"\n  Baseline (predict mean): MAE={baseline_errors.mean():.2f} layers")
print(f"  Improvement over baseline: {(1 - loo_errors.mean()/baseline_errors.mean())*100:.1f}%")


# ── Section 5: Top predictors across all thresholds ───────────
print("\n" + "=" * 72)
print("SECTION 5: Ranked predictors (|Pearson r| across all τ)")
print("=" * 72)

ranked = []
for tau_str, tau_res in all_results.items():
    for m, vals in tau_res.items():
        if vals['pearson_r'] is not None:
            ranked.append((abs(vals['pearson_r']), vals['pearson_r'], vals['pearson_p'],
                           vals['spearman_rho'], vals['spearman_p'], m, float(tau_str)))

ranked.sort(key=lambda x: x[0], reverse=True)
print(f"\nTop 15 predictors:")
print(f"{'Rank':>4s} {'|r|':>6s} {'r':>8s} {'p':>8s} {'ρ':>8s} {'p(ρ)':>8s} {'Metric':>30s} {'τ':>6s}")
for i, (absr, r, p, rho, sp, m, tau) in enumerate(ranked[:15]):
    sig = ""
    if p < 0.01: sig = "***"
    elif p < 0.05: sig = "** "
    elif p < 0.10: sig = "*  "
    else: sig = "   "
    print(f"{i+1:4d} {absr:6.4f} {r:+8.4f} {p:8.4f} {rho:+8.4f} {sp:8.4f} {m:>30s} {tau:6.2f} {sig}")


# ── Section 6: CRITICAL — Partial-layer early exit viability ──
print("\n" + "=" * 72)
print("SECTION 6: PARTIAL-LAYER EARLY EXIT VIABILITY")
print("This is the key question: can we compute the signal from only")
print("the first K layers and still predict oracle exit?")
print("=" * 72)

# Test multiple K values and multiple thresholds
partial_K_values = [6, 8, 10, 12, 14, 16]
partial_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]

partial_results = {}

print(f"\n{'K':>3s} {'τ':>6s} {'Metric':>30s} {'r':>8s} {'p':>8s} {'ρ':>8s} {'p(ρ)':>8s} {'Sig':>4s}")
print("-" * 100)

for K in partial_K_values:
    for tau in partial_thresholds:
        for m in metric_names:
            arr = []
            for i in range(len(query_names)):
                s = compute_episode_stats(cos_dist_matrix[i], tau, max_layer=K)
                arr.append(s[m])
            arr = np.array(arr)

            if np.std(arr) < 1e-10:
                continue

            pr, pp = stats.pearsonr(arr, oracle_exits)
            sr, sp = stats.spearmanr(arr, oracle_exits)

            key = (K, tau, m)
            partial_results[key] = {
                'pearson_r': round(pr, 5),
                'pearson_p': round(pp, 6),
                'spearman_rho': round(sr, 5),
                'spearman_p': round(sp, 6),
            }

            if abs(pr) > 0.35 or pp < 0.10:
                sig = ""
                if pp < 0.01: sig = "***"
                elif pp < 0.05: sig = "** "
                elif pp < 0.10: sig = "*  "
                print(f"{K:3d} {tau:6.2f} {m:>30s} {pr:+8.4f} {pp:8.4f} {sr:+8.4f} {sp:8.4f} {sig}")

# Find best partial predictor
print(f"\n--- Best partial predictors (top 10 by |r|) ---")
partial_ranked = []
for (K, tau, m), vals in partial_results.items():
    partial_ranked.append((abs(vals['pearson_r']), vals['pearson_r'], vals['pearson_p'],
                           vals['spearman_rho'], vals['spearman_p'], m, K, tau))
partial_ranked.sort(key=lambda x: x[0], reverse=True)

print(f"\n{'Rank':>4s} {'K':>3s} {'τ':>6s} {'|r|':>6s} {'r':>8s} {'p':>8s} {'ρ':>8s} {'Metric':>30s}")
for i, (absr, r, p, rho, sp, m, K, tau) in enumerate(partial_ranked[:10]):
    sig = ""
    if p < 0.01: sig = "***"
    elif p < 0.05: sig = "** "
    elif p < 0.10: sig = "*  "
    print(f"{i+1:4d} {K:3d} {tau:6.2f} {absr:6.4f} {r:+8.4f} {p:8.4f} {rho:+8.4f} {m:>30s} {sig}")

# ── Section 6b: Compare full vs partial for best metric ────────
print(f"\n--- Degradation analysis: how does signal degrade as K shrinks? ---")
# For the best full-layer predictor, track r as K decreases
print(f"\nTracking {best_metric} at τ={best_tau} across K values:")
print(f"{'K':>4s} {'r':>8s} {'p':>8s} {'ρ':>8s} {'p(ρ)':>8s}")

degradation_data = {}
for K in range(4, 23):
    arr = []
    for i in range(len(query_names)):
        s = compute_episode_stats(cos_dist_matrix[i], best_tau, max_layer=K)
        arr.append(s[best_metric])
    arr = np.array(arr)

    if np.std(arr) < 1e-10:
        print(f"{K:4d}     CONSTANT")
        degradation_data[K] = {'r': None, 'p': None}
        continue

    pr, pp = stats.pearsonr(arr, oracle_exits)
    sr, sp = stats.spearmanr(arr, oracle_exits)
    sig = ""
    if pp < 0.01: sig = "***"
    elif pp < 0.05: sig = "** "
    elif pp < 0.10: sig = "*  "
    print(f"{K:4d} {pr:+8.4f} {pp:8.4f} {sr:+8.4f} {sp:8.4f} {sig}")
    degradation_data[K] = {'r': round(pr, 5), 'p': round(pp, 6)}

# ── Section 6c: LOO-CV with partial layers ──────────────────────
print(f"\n--- LOO-CV with partial layers (best full predictor: {best_metric} @ τ={best_tau}) ---")
for K in [8, 10, 12, 14]:
    partial_arr = []
    for i in range(len(query_names)):
        s = compute_episode_stats(cos_dist_matrix[i], best_tau, max_layer=K)
        partial_arr.append(s[best_metric])
    partial_arr = np.array(partial_arr)

    if np.std(partial_arr) < 1e-10:
        print(f"  K={K}: CONSTANT metric — cannot predict")
        continue

    loo_err = []
    for i in range(len(query_names)):
        tx = np.delete(partial_arr, i)
        ty = np.delete(oracle_exits, i)
        test_x = partial_arr[i]
        test_y = oracle_exits[i]
        if np.std(tx) > 1e-10:
            slope, intercept, _, _, _ = stats.linregress(tx, ty)
            pred = slope * test_x + intercept
        else:
            pred = ty.mean()
        loo_err.append(abs(pred - test_y))
    loo_err = np.array(loo_err)
    print(f"  K={K:2d}: LOO MAE = {loo_err.mean():.2f} layers "
          f"(baseline={baseline_errors.mean():.2f}, improvement={((1 - loo_err.mean()/baseline_errors.mean())*100):.1f}%)")


# ── Section 7: Per-query detail table ──────────────────────────
print("\n" + "=" * 72)
print("SECTION 7: Per-query detail (best predictor)")
print("=" * 72)
print(f"\n{best_metric} at τ={best_tau}")
print(f"{'Query':>30s} {'Oracle':>7s} {'Metric':>8s} {'LOO Pred':>9s} {'Error':>7s}")
for i in range(len(query_names)):
    s = compute_episode_stats(cos_dist_matrix[i], best_tau)
    val = s[best_metric]
    print(f"{query_names[i]:>30s} {oracle_exits[i]:7d} {val:8.3f} {loo_predictions[i]:9.2f} {loo_errors[i]:7.2f}")


# ── Section 8: Alternative — scan ALL metrics at ALL thresholds
# to find anything with p<0.05
print("\n" + "=" * 72)
print("SECTION 8: Statistically significant predictors (p < 0.05)")
print("=" * 72)

sig_results = []
for tau_str, tau_res in all_results.items():
    for m, vals in tau_res.items():
        if vals['pearson_p'] is not None and vals['pearson_p'] < 0.05:
            sig_results.append((vals['pearson_p'], vals['pearson_r'],
                                vals['spearman_rho'], vals['spearman_p'],
                                m, float(tau_str)))

if sig_results:
    sig_results.sort()
    print(f"\n{'p':>8s} {'r':>8s} {'ρ':>8s} {'p(ρ)':>8s} {'Metric':>30s} {'τ':>6s}")
    for p, r, rho, sp, m, tau in sig_results:
        print(f"{p:8.4f} {r:+8.4f} {rho:+8.4f} {sp:8.4f} {m:>30s} {tau:6.2f}")
    print(f"\nTotal: {len(sig_results)} significant at p<0.05 out of "
          f"{sum(1 for t in all_results.values() for v in t.values() if v['pearson_r'] is not None)} tested")

    # Bonferroni correction
    n_tests = sum(1 for t in all_results.values() for v in t.values() if v['pearson_r'] is not None)
    bonferroni_threshold = 0.05 / n_tests
    print(f"  Bonferroni-corrected threshold: p < {bonferroni_threshold:.6f}")
    bonf_sig = [s for s in sig_results if s[0] < bonferroni_threshold]
    print(f"  Survive Bonferroni: {len(bonf_sig)}")
else:
    print("\nNO statistically significant predictors at p < 0.05")


# ── Section 9: Correlation between metrics themselves ──────────
print("\n" + "=" * 72)
print("SECTION 9: Inter-metric correlations (τ=0.08)")
print("(How redundant are these metrics?)")
print("=" * 72)

tau_check = 0.08
metric_data = {}
for m in metric_names:
    arr = []
    for i in range(len(query_names)):
        s = compute_episode_stats(cos_dist_matrix[i], tau_check)
        arr.append(s[m])
    arr = np.array(arr)
    if np.std(arr) > 1e-10:
        metric_data[m] = arr

active_metrics = list(metric_data.keys())
print(f"\n{'':>30s}", end='')
for m2 in active_metrics:
    print(f" {m2[:8]:>8s}", end='')
print()

for m1 in active_metrics:
    print(f"{m1:>30s}", end='')
    for m2 in active_metrics:
        if m1 == m2:
            print(f"    1.00", end='')
        else:
            r, _ = stats.pearsonr(metric_data[m1], metric_data[m2])
            print(f" {r:+8.3f}", end='')
    print()


# ── Final verdict ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)

# Collect key findings
any_significant_full = len(sig_results) > 0 if sig_results else False

# Check partial viability at K=10
best_partial_at_10 = None
for (K, tau, m), vals in partial_results.items():
    if K == 10 and vals['pearson_p'] is not None:
        if best_partial_at_10 is None or vals['pearson_p'] < best_partial_at_10[1]:
            best_partial_at_10 = (vals['pearson_r'], vals['pearson_p'], m, tau)

print(f"\n1. FULL-LAYER best predictor: {best_metric} @ τ={best_tau}")
print(f"   r={best_r:+.4f}, p={all_results[str(best_tau)][best_metric]['pearson_p']:.6f}")
if all_results[str(best_tau)][best_metric]['pearson_p'] < 0.05:
    print(f"   STATUS: Significant (p < 0.05)")
else:
    print(f"   STATUS: NOT significant (p >= 0.05)")

print(f"\n2. LOO-CV MAE: {loo_errors.mean():.2f} layers (baseline {baseline_errors.mean():.2f})")
improvement_pct = (1 - loo_errors.mean() / baseline_errors.mean()) * 100
if improvement_pct > 10:
    print(f"   STATUS: Useful ({improvement_pct:.1f}% improvement)")
elif improvement_pct > 0:
    print(f"   STATUS: Marginal ({improvement_pct:.1f}% improvement)")
else:
    print(f"   STATUS: WORSE than baseline ({improvement_pct:.1f}%)")

if best_partial_at_10:
    print(f"\n3. PARTIAL (K=10) best predictor: {best_partial_at_10[2]} @ τ={best_partial_at_10[3]}")
    print(f"   r={best_partial_at_10[0]:+.4f}, p={best_partial_at_10[1]:.6f}")
    if best_partial_at_10[1] < 0.05:
        print(f"   STATUS: VIABLE for early exit at layer 10")
    else:
        print(f"   STATUS: NOT viable for early exit at layer 10")

print(f"\n4. EARLY EXIT VERDICT:")
# Check if any partial K<=12 has p<0.05
any_early_viable = False
for (K, tau, m), vals in partial_results.items():
    if K <= 12 and vals['pearson_p'] is not None and vals['pearson_p'] < 0.05:
        any_early_viable = True
        break

if any_early_viable:
    print(f"   Some signal survives at K<=12. Potentially useful, needs larger N to confirm.")
else:
    print(f"   NO signal at K<=12 with p<0.05. n_transitions is NOT useful as early exit signal.")
    print(f"   The correlation requires seeing most/all layers, defeating the purpose.")

print()

# ── Build receipt ──────────────────────────────────────────────
end_iso = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
cost = {
    'wall_time_s': round(time.time() - t_start, 3),
    'cpu_time_s': round(time.process_time() - cpu_start, 3),
    'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    'python_version': platform.python_version(),
    'hostname': platform.node(),
    'timestamp_start': start_iso,
    'timestamp_end': end_iso,
}

# Serialize partial_results with string keys
partial_results_serializable = {}
for (K, tau, m), vals in partial_results.items():
    key = f"K={K}_tau={tau}_{m}"
    partial_results_serializable[key] = vals

# Build compact per-query data
per_query_detail = {}
for i in range(len(query_names)):
    s_full = compute_episode_stats(cos_dist_matrix[i], best_tau)
    s_k10 = compute_episode_stats(cos_dist_matrix[i], best_tau, max_layer=10)
    per_query_detail[query_names[i]] = {
        'oracle_exit': int(oracle_exits[i]),
        'full_layer_stats': s_full,
        'partial_K10_stats': s_k10,
        'loo_prediction': round(float(loo_predictions[i]), 2),
        'loo_error': round(float(loo_errors[i]), 2),
    }

receipt_out = {
    'work_order': 'WO-SENSING-PROBE-05b+-DWELL',
    'description': 'Deep reanalysis of n_transitions / quiet-zone episode statistics',
    'source_receipt': RECEIPT_PATH,
    'model': receipt['model'],
    'n_layers': n_layers,
    'n_prompts': n_prompts,
    'best_full_predictor': {
        'metric': best_metric,
        'threshold': best_tau,
        'pearson_r': round(best_r, 5),
        'pearson_p': round(all_results[str(best_tau)][best_metric]['pearson_p'], 6),
        'spearman_rho': all_results[str(best_tau)][best_metric]['spearman_rho'],
        'spearman_p': all_results[str(best_tau)][best_metric]['spearman_p'],
        'cohens_d': round(cohens_d, 4),
        'roc_auc': round(abs(auc), 4) if isinstance(auc, float) else None,
        'loo_mae': round(float(loo_errors.mean()), 3),
        'loo_rmse': round(float(np.sqrt(np.mean(loo_errors**2))), 3),
        'baseline_mae': round(float(baseline_errors.mean()), 3),
        'improvement_pct': round(improvement_pct, 2),
    },
    'best_spearman_predictor': {
        'metric': best_metric_sp,
        'threshold': best_tau_sp,
        'spearman_rho': round(best_rho, 5),
    },
    'full_layer_correlations': all_results,
    'degradation_by_K': degradation_data,
    'partial_top10': [
        {
            'K': int(K), 'tau': tau, 'metric': m,
            'pearson_r': round(r, 5), 'pearson_p': round(p, 6),
            'spearman_rho': round(rho, 5),
        }
        for absr, r, p, rho, sp, m, K, tau in partial_ranked[:10]
    ],
    'significant_full_count': len(sig_results) if sig_results else 0,
    'bonferroni_survivors': len(bonf_sig) if sig_results else 0,
    'early_exit_viable_K12': any_early_viable,
    'verdict': {
        'full_layer_signal': 'SIGNIFICANT' if (all_results[str(best_tau)][best_metric]['pearson_p'] < 0.05) else 'NOT_SIGNIFICANT',
        'early_exit_useful': any_early_viable,
        'summary': '',  # filled below
    },
    'per_query': per_query_detail,
    'cost': cost,
}

# Summary
if any_early_viable:
    receipt_out['verdict']['summary'] = (
        f"Partial signal at K<=12: early exit MAY be viable. "
        f"Best full: {best_metric}@τ={best_tau} r={best_r:+.4f}. "
        f"LOO MAE={loo_errors.mean():.2f} vs baseline {baseline_errors.mean():.2f}. "
        f"Needs larger N."
    )
else:
    receipt_out['verdict']['summary'] = (
        f"No partial signal at K<=12: n_transitions NOT useful for early exit. "
        f"Best full: {best_metric}@τ={best_tau} r={best_r:+.4f} but requires all layers. "
        f"LOO MAE={loo_errors.mean():.2f} vs baseline {baseline_errors.mean():.2f}."
    )

# Save receipt
timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
receipt_path = os.path.join(OUT_DIR, f"dwell_reanalysis_{timestamp}.json")
with open(receipt_path, 'w') as f:
    json.dump(receipt_out, f, indent=2, default=str)

print(f"Receipt saved: {receipt_path}")
print(f"Cost: wall={cost['wall_time_s']}s, cpu={cost['cpu_time_s']}s, mem={cost['peak_memory_mb']}MB")
print(f"SHA256: ", end='')
with open(receipt_path, 'rb') as f:
    print(hashlib.sha256(f.read()).hexdigest())
