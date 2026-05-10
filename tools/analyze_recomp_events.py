"""
WO-ECHO-HYBRID-05d: Analyze the 609-event recompression dataset.

Questions:
1. Which features predict high drift_ratio (= codebook staleness)?
2. Can a simple decision tree beat the TopK-5 policy?
3. Does Se or kurtosis rank above sidecar_norm in feature importance?

Data: receipts/echo_hybrid/wo_echo_hybrid_04_recomp_events.jsonl
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

EVENTS_PATH = Path("receipts/echo_hybrid/wo_echo_hybrid_04_recomp_events.jsonl")
RECEIPT_DIR = Path("receipts/echo_hybrid")
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


def load_events():
    events = []
    with open(EVENTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 70)
    print("WO-ECHO-HYBRID-05d: Recompression Event Analysis")
    print("=" * 70)

    events = load_events()
    print(f"Loaded {len(events)} events from {EVENTS_PATH}")

    # Parse into feature matrix
    configs = sorted(set(e.get("config", "unknown") for e in events))
    print(f"Configs: {configs}")

    # Feature columns
    feature_names = [
        "eff_rank", "se", "kurtosis", "weight_rms",
        "pre_sidecar_norm", "drift_ratio", "step", "n_params",
    ]
    categorical_features = ["block_type", "role"]

    # Build arrays per config
    results_by_config = {}

    for config_name in configs:
        cfg_events = [e for e in events if e.get("config") == config_name]
        if len(cfg_events) < 10:
            print(f"  Skipping {config_name}: only {len(cfg_events)} events")
            continue

        print(f"\n--- Config: {config_name} ({len(cfg_events)} events) ---")

        # Build feature matrix
        X_list = []
        y_list = []  # drift_ratio as target
        meta_list = []

        for e in cfg_events:
            row = [float(e.get(f, 0)) for f in feature_names]
            X_list.append(row)
            y_list.append(float(e.get("drift_ratio", 0)))
            meta_list.append({
                "layer": e.get("layer_name", ""),
                "block_type": e.get("block_type", ""),
                "role": e.get("role", ""),
            })

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)

        print(f"  Feature matrix: {X.shape}")
        print(f"  drift_ratio: mean={y.mean():.4f}, std={y.std():.4f}, "
              f"min={y.min():.4f}, max={y.max():.4f}")

        # --- Correlation analysis ---
        print(f"\n  Feature correlations with drift_ratio:")
        correlations = {}
        for i, fname in enumerate(feature_names):
            if fname == "drift_ratio":
                continue
            col = X[:, i]
            if col.std() < 1e-12:
                correlations[fname] = 0.0
                continue
            corr = np.corrcoef(col, y)[0, 1]
            correlations[fname] = round(float(corr), 4)
            print(f"    {fname:<25} r = {corr:+.4f}")

        # --- Per block_type analysis ---
        print(f"\n  Per block_type drift_ratio:")
        block_type_stats = {}
        for bt in ["ssm", "attn"]:
            bt_mask = [m["block_type"] == bt for m in meta_list]
            bt_y = y[bt_mask]
            if len(bt_y) > 0:
                block_type_stats[bt] = {
                    "count": int(len(bt_y)),
                    "mean_drift": round(float(bt_y.mean()), 4),
                    "std_drift": round(float(bt_y.std()), 4),
                }
                print(f"    {bt}: n={len(bt_y)}, mean={bt_y.mean():.4f}, std={bt_y.std():.4f}")

        # --- Per role analysis ---
        print(f"\n  Per role drift_ratio:")
        role_stats = {}
        for role in sorted(set(m["role"] for m in meta_list)):
            role_mask = [m["role"] == role for m in meta_list]
            role_y = y[role_mask]
            if len(role_y) > 0:
                role_stats[role] = {
                    "count": int(len(role_y)),
                    "mean_drift": round(float(role_y.mean()), 4),
                    "std_drift": round(float(role_y.std()), 4),
                }
                print(f"    {role:<15}: n={len(role_y)}, mean={role_y.mean():.4f}, std={role_y.std():.4f}")

        # --- Decision tree classifier ---
        # Binary target: drift > median = "should recompress"
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score

        median_drift = float(np.median(y))
        y_binary = (y > median_drift).astype(int)

        # Remove drift_ratio from features (it's the target)
        feat_idx_no_drift = [i for i, f in enumerate(feature_names) if f != "drift_ratio"]
        X_no_drift = X[:, feat_idx_no_drift]
        feat_names_no_drift = [feature_names[i] for i in feat_idx_no_drift]

        tree_results = {}
        for depth in [3, 5]:
            clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
            scores = cross_val_score(clf, X_no_drift, y_binary, cv=5, scoring="accuracy")

            # Fit on full data for feature importances
            clf.fit(X_no_drift, y_binary)
            importances = dict(zip(feat_names_no_drift, [round(float(x), 4) for x in clf.feature_importances_]))

            tree_results[f"depth_{depth}"] = {
                "cv_accuracy_mean": round(float(scores.mean()), 4),
                "cv_accuracy_std": round(float(scores.std()), 4),
                "feature_importances": importances,
                "top3_features": sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3],
            }

            print(f"\n  DecisionTree depth={depth}: accuracy={scores.mean():.4f}±{scores.std():.4f}")
            print(f"    Top features: {tree_results[f'depth_{depth}']['top3_features']}")

        # --- Does Se or kurtosis rank above sidecar_norm? ---
        best_tree = tree_results.get("depth_5", tree_results.get("depth_3", {}))
        importances = best_tree.get("feature_importances", {})
        se_rank = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        se_position = next((i for i, (f, _) in enumerate(se_rank) if f == "se"), -1)
        sidecar_position = next((i for i, (f, _) in enumerate(se_rank) if f == "pre_sidecar_norm"), -1)
        kurtosis_position = next((i for i, (f, _) in enumerate(se_rank) if f == "kurtosis"), -1)

        se_vs_sidecar = "Se above sidecar" if se_position < sidecar_position else "sidecar above Se"
        print(f"\n  Se rank: #{se_position+1}, sidecar_norm rank: #{sidecar_position+1}, "
              f"kurtosis rank: #{kurtosis_position+1}")
        print(f"  => {se_vs_sidecar}")

        results_by_config[config_name] = {
            "n_events": len(cfg_events),
            "drift_ratio_stats": {
                "mean": round(float(y.mean()), 4),
                "std": round(float(y.std()), 4),
                "min": round(float(y.min()), 4),
                "max": round(float(y.max()), 4),
                "median": round(float(median_drift), 4),
            },
            "correlations_with_drift": correlations,
            "block_type_stats": block_type_stats,
            "role_stats": role_stats,
            "decision_tree": tree_results,
            "se_vs_sidecar_ranking": se_vs_sidecar,
            "feature_ranking": [f for f, _ in se_rank],
        }

    # --- Cross-config summary ---
    print(f"\n{'='*70}")
    print("CROSS-CONFIG SUMMARY")
    print(f"{'='*70}")

    for cfg_name, r in results_by_config.items():
        print(f"\n  {cfg_name}:")
        print(f"    Events: {r['n_events']}")
        print(f"    Drift: {r['drift_ratio_stats']}")
        d5 = r["decision_tree"].get("depth_5", {})
        print(f"    Tree accuracy: {d5.get('cv_accuracy_mean', '?')}")
        print(f"    Feature ranking: {r['feature_ranking'][:5]}")
        print(f"    Se vs sidecar: {r['se_vs_sidecar_ranking']}")

    # Emit receipt
    receipt = {
        "wo": "WO-ECHO-HYBRID-05d",
        "experiment": "recomp_routing_analysis",
        "timestamp": time.strftime("%Y-%m-%d"),
        "events_path": str(EVENTS_PATH),
        "total_events": len(events),
        "configs_analyzed": list(results_by_config.keys()),
        "results": results_by_config,
        "notes": (
            "Offline analysis of 609 recompression events from WO-04. "
            "Tests whether Se/kurtosis/sidecar_norm predict which layers need "
            "recompression. Decision tree classifiers at depth 3 and 5."
        ),
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    out_path = RECEIPT_DIR / "wo_echo_hybrid_05d_routing_analysis.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nRECEIPT: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
