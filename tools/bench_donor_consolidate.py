#!/usr/bin/env python3
"""WO-DONOR-SURVEY-01 Step 5: Consolidate Donor Selection Receipt

Reads receipts from Steps 1-4 and produces a single donor selection receipt
specifying:
  1. SSM donor model + specific layers to borrow
  2. Attention donor model + specific layers to borrow
  3. Block pattern
  4. VQ dimension per block type
  5. Estimated training budget

This receipt becomes the input spec for the GPU rental session.

Receipt includes cost block per WO-RECEIPT-COST-01.
"""

import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

t_start = time.time()
cpu_start = time.process_time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "donor_survey"
FINAL_DIR = Path(__file__).parent.parent / "receipts" / "donor_selection"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_receipt(pattern: str) -> dict:
    """Load the most recent receipt matching a glob pattern."""
    matches = sorted(RECEIPT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        return {"error": f"no receipt matching {pattern}"}
    path = matches[-1]
    print(f"  Loading: {path.name}")
    return json.loads(path.read_text())


def main():
    print("WO-DONOR-SURVEY-01 Step 5: Donor Selection Consolidation")
    print("=" * 70)

    # Load all step receipts
    print("\nLoading step receipts...")
    step1 = load_latest_receipt("sidecar_survey_*.json")
    step2 = load_latest_receipt("spectral_profile_*.json")
    step3 = load_latest_receipt("vq_compat_*.json")
    step4 = load_latest_receipt("pattern_eval_*.json")

    missing = []
    for name, data in [("step1_sidecar", step1), ("step2_spectral", step2),
                        ("step3_vq", step3), ("step4_pattern", step4)]:
        if "error" in data:
            missing.append(name)
            print(f"  WARNING: {name} — {data['error']}")

    if missing:
        print(f"\nMISSING RECEIPTS: {missing}")
        print("Run the missing steps first. Proceeding with available data.\n")

    # === Extract Step 1: SSM and Attention donor rankings ===
    ssm_ranking = step1.get("ssm_donor_ranking", [])
    attn_ranking = step1.get("attention_donor_ranking", [])

    ssm_top = ssm_ranking[0] if ssm_ranking else {"model": "mamba2-1.3b-helix", "sidecar_energy_ratio": "unknown"}
    attn_top = attn_ranking[0] if attn_ranking else {"model": "qwen2.5-coder-1.5b-helix", "sidecar_energy_ratio": "unknown"}

    print(f"\n--- Step 1: Sidecar Rankings ---")
    print(f"  Best SSM donor: {ssm_top.get('model')} (energy_ratio={ssm_top.get('sidecar_energy_ratio')})")
    print(f"  Best ATTN donor: {attn_top.get('model')} (energy_ratio={attn_top.get('sidecar_energy_ratio')})")

    # === Extract Step 2: Spectral profiles ===
    spectral_comparison = step2.get("cross_donor_comparison", {})
    print(f"\n--- Step 2: Spectral Profiles ---")
    for role in ["in_proj", "out_proj", "q_proj", "v_proj"]:
        entries = spectral_comparison.get(role, [])
        if entries:
            for e in entries:
                if e.get("is_primary"):
                    print(f"  {role}: {e['model']} kurtosis={e['kurtosis_mean']:.4f}")

    # === Extract Step 3: VQ compatibility ===
    vq_recommendations = step3.get("recommendations", {})
    print(f"\n--- Step 3: VQ Compatibility ---")
    for model, roles in vq_recommendations.items():
        print(f"  {model}:")
        for role, rec in sorted(roles.items()):
            print(f"    {role}: {rec['recommended_dim']} (d2_cos={rec['d2_cos']:.4f})")

    # === Extract Step 4: Pattern evaluation ===
    pattern_ranking = step4.get("ranking_by_compress_quality", [])
    best_pattern = pattern_ranking[0] if pattern_ranking else {"pattern": "SSASSASSS"}
    zamba2_ref = step4.get("zamba2_reference", {})

    print(f"\n--- Step 4: Pattern Evaluation ---")
    print(f"  Best pattern: {best_pattern.get('pattern')} (cos={best_pattern.get('compress_cos', '?')})")
    print(f"  Zamba2 reference: {zamba2_ref.get('closest_9block', 'SSSASSSAS')}")
    for r in pattern_ranking[:3]:
        print(f"    #{r['rank']}: {r['pattern']} cos={r.get('compress_cos', '?')} ratio={r.get('ratio', '?')}")

    # === Build consolidated donor selection ===
    print(f"\n{'='*70}")
    print("DONOR SELECTION DECISION")
    print(f"{'='*70}")

    # Determine VQ dim per block type
    ssm_vq_dim = "d=1"  # default
    attn_vq_dim = "d=1"
    mamba2_recs = vq_recommendations.get("mamba2-1.3b-helix", {})
    qwen_recs = vq_recommendations.get("qwen2.5-coder-1.5b-helix", {})

    if mamba2_recs:
        in_proj_rec = mamba2_recs.get("in_proj", {}).get("recommended_dim", "d=1")
        out_proj_rec = mamba2_recs.get("out_proj", {}).get("recommended_dim", "d=1")
        ssm_vq_dim = in_proj_rec  # Use in_proj recommendation as representative
    if qwen_recs:
        q_rec = qwen_recs.get("q_proj", {}).get("recommended_dim", "d=1")
        attn_vq_dim = q_rec

    # Build final config
    selected_pattern = best_pattern.get("pattern", "SSASSASSS")
    if isinstance(selected_pattern, list):
        pattern_str = "".join("A" if b == "attn" else "S" for b in selected_pattern)
    else:
        pattern_str = selected_pattern

    config = {
        "ssm_donor": {
            "model": "mamba2-1.3b-helix",
            "hidden_size": 2048,
            "architecture": "mamba_v2",
            "key_layers": ["in_proj", "out_proj"],
            "sidecar_energy_ratio": ssm_top.get("sidecar_energy_ratio"),
            "vq_dim": ssm_vq_dim,
            "notes": "Strongest SSM in zoo. d=2048, needs projection layer for hybrid.",
        },
        "attention_donor": {
            "model": "qwen2.5-coder-1.5b-helix",
            "hidden_size": 1536,
            "architecture": "transformer",
            "key_layers": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "sidecar_energy_ratio": attn_top.get("sidecar_energy_ratio"),
            "vq_dim": attn_vq_dim,
            "notes": "Strong coder model. d=1536, needs projection layer for hybrid.",
        },
        "architecture": {
            "block_pattern": selected_pattern if isinstance(selected_pattern, list) else list(selected_pattern.replace("A", "attn,").replace("S", "ssm,").rstrip(",").split(",")),
            "pattern_string": pattern_str,
            "n_blocks": len(pattern_str),
            "ssm_to_attn_ratio": f"{pattern_str.count('S')}:{pattern_str.count('A')}",
            "zamba2_ratio_reference": "5:1",
            "boundary_projection": {
                "ssm_dim": 2048,
                "attn_dim": 1536,
                "method": "linear projection at SSM↔ATTN boundaries",
                "projection_params": 2048 * 1536 * 2,  # two projection matrices
            },
        },
        "compression": {
            "n_clusters": 256,
            "ssm_vq_dim": ssm_vq_dim,
            "attn_vq_dim": attn_vq_dim,
            "sidecar_enabled": True,
        },
        "training_budget": {
            "phase1_steps": 500,
            "phase2_steps": 2000,
            "batch_size": 4,
            "seq_len": 128,
            "lr": 1e-4,
            "compress_schedule": 25,
            "estimated_gpu_hours": "2-4h on A100 or equivalent",
            "data": "wikitext-103 or The Stack v2 subset",
        },
    }

    # Print the final config
    for section, values in config.items():
        print(f"\n  {section}:")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"    {k}: {v}")

    # Predictions (falsifiable)
    predictions = {
        "p1": f"Mamba2-1.3B SSM blocks at {ssm_vq_dim} will achieve cosine > 0.998 during training",
        "p2": f"Qwen2.5-Coder-1.5B attention blocks at {attn_vq_dim} will achieve cosine > 0.998 during training",
        "p3": f"Pattern {pattern_str} will produce gap < 1.0 vs dense at 500 steps",
        "p4": "Projection layers (d=2048↔1536) will add < 5% parameter overhead",
        "p5": "Born-compressed hybrid will converge to loss < 7.0 within 500 steps",
    }

    receipt = {
        "work_order": "WO-DONOR-SURVEY-01",
        "step": "5_donor_selection",
        "status": "COMPLETE" if not missing else f"PARTIAL (missing: {missing})",
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "question": "Which donor models and configuration should we use for born-compressed GPU training?",
        "config": config,
        "predictions": predictions,
        "evidence_chain": {
            "step1_sidecar_survey": {
                "status": "COMPLETE" if "error" not in step1 else "MISSING",
                "ssm_top": ssm_top,
                "attn_top": attn_top,
            },
            "step2_spectral_profile": {
                "status": "COMPLETE" if "error" not in step2 else "MISSING",
                "n_models_profiled": step2.get("n_models_profiled"),
            },
            "step3_vq_compatibility": {
                "status": "COMPLETE" if "error" not in step3 else "MISSING",
                "recommendations": vq_recommendations,
            },
            "step4_pattern_evaluation": {
                "status": "COMPLETE" if "error" not in step4 else "MISSING",
                "best_pattern": best_pattern,
                "zamba2_reference": zamba2_ref,
            },
        },
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = FINAL_DIR / f"donor_selection_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\nFINAL RECEIPT: {receipt_path}")

    return receipt_path


if __name__ == "__main__":
    main()
