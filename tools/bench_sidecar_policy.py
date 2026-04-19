#!/usr/bin/env python3
"""
WO-SIDECAR-POLICY-01 — Policy Frontier Benchmark

Tests whether thresholded sidecar application creates a measurable
speed/quality operating frontier on zamba2-1.2b-helix.

Runs 5 modes as subprocesses (subprocess isolation is mandatory —
threshold is read at model instantiation, in-process swap is a
silent-fail trap):

  A   always_on            (baseline quality)
  B   always_off           (baseline speed ceiling)
  C1  threshold = 0.10
  C2  threshold = 0.15
  C3  threshold = 0.20

Per mode collects: PPL, wall time, tok/s, peak VRAM, apply/skip counts,
per-layer sidecar norm stats.

Phase B: 50-chunk subset, run each chunk in always_off AND threshold=0
to compute per-chunk quality delta vs per-chunk mean sidecar norm, then
Pearson correlation. A tier pass without ρ ≥ 0.3 is reported as
"frontier exists but not norm-driven".

Honesty rule (from WO): this WO proves the POLICY FRONTIER only. Speed
gain in threshold modes comes from skipping the sidecar ADD, NOT from
skipping construction. Construction-skip is WO-SIDECAR-FASTPATH-01.

Usage:
    # Driver — runs all 5 modes + phase B, writes receipt:
    python3 tools/bench_sidecar_policy.py

    # Worker — runs ONE mode (invoked by driver via subprocess):
    python3 tools/bench_sidecar_policy.py \
        --worker --mode threshold --threshold 0.10 \
        --output /tmp/sidecar_C1.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Frozen parameters (WO-SIDECAR-POLICY-01)
# ---------------------------------------------------------------------------

WO_ID = "WO-SIDECAR-POLICY-01"
MODEL_DIR = Path.home() / "models" / "zamba2-1.2b-helix"
DATASET = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"
N_CHUNKS = 200
MAX_LENGTH = 512
SEED = 42
MIN_FREE_VRAM_MB = 2000

RECEIPT_DIR = Path.home() / "helix-substrate" / "receipts" / "sidecar_routing"

MODES = [
    ("A_always_on",   "always_on",  0.0),
    ("B_always_off",  "always_off", 0.0),
    ("C1_p25_6.72",   "threshold",  6.72),   # April 9 calib p25
    ("C2_p50_14.59",  "threshold",  14.59),  # April 9 calib p50
    ("C3_p75_29.97",  "threshold",  29.97),  # April 9 calib p75
]

PHASE_B_N_CHUNKS = 200

# Restrict policy to in_proj layers only — matches ρ=0.574 methodology.
# Other HelixLinear layers keep default behavior (always_on).
LAYER_FILTER = "in_proj"


# ---------------------------------------------------------------------------
# Determinism (seed)
# ---------------------------------------------------------------------------

def seed_all(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VRAM preflight (copy from falsifier_live_sidecar_norm.py)
# ---------------------------------------------------------------------------

def vram_preflight() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"free_mb": None, "total_mb": None}
    free_b, total_b = torch.cuda.mem_get_info()
    free_mb = free_b / 1e6
    total_mb = total_b / 1e6
    if free_mb < MIN_FREE_VRAM_MB:
        sys.stderr.write(
            f"[FATAL] Only {free_mb:.0f} MB VRAM free "
            f"(need >= {MIN_FREE_VRAM_MB}). Free VRAM and re-run.\n"
        )
        sys.exit(2)
    return {"free_mb": free_mb, "total_mb": total_mb}


# ---------------------------------------------------------------------------
# Model & dataset loaders
# ---------------------------------------------------------------------------

def load_model():
    """Load zamba2-1.2b-helix via HXQ from_pretrained path."""
    try:
        from mamba_scan_lite import patch  # noqa: F401
        patch.apply_patch()
    except ImportError:
        pass

    import helix_substrate.hf_quantizer  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return tokenizer, model


def load_wikitext_test_chunks(tokenizer, n_chunks: int, max_length: int) -> List[torch.Tensor]:
    """Fixed N chunks from wikitext-2-raw TEST split, each length max_length."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", DATASET, split=DATASET_SPLIT)
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)

    chunks: List[torch.Tensor] = []
    for i in range(n_chunks):
        start = i * max_length
        end = start + max_length
        if end > len(ids_t):
            break
        chunks.append(ids_t[start:end])
    if len(chunks) < n_chunks:
        raise RuntimeError(
            f"wikitext test split only yielded {len(chunks)} chunks, need {n_chunks}"
        )
    return chunks


# ---------------------------------------------------------------------------
# Policy wiring
# ---------------------------------------------------------------------------

def enumerate_helix_linear(model) -> List[tuple]:
    """Return [(name, module)] for HelixLinear modules matching LAYER_FILTER.

    LAYER_FILTER="in_proj" restricts to in_proj layers only, matching the
    ρ=0.574 methodology. LAYER_FILTER="" returns all HelixLinear modules.
    """
    from helix_substrate.helix_linear import HelixLinear
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, HelixLinear) and (not LAYER_FILTER or LAYER_FILTER in name)
    ]


def apply_policy_to_all(modules, mode: str, threshold: float) -> int:
    """Set sidecar policy on every HelixLinear. Returns number touched."""
    n = 0
    for _, mod in modules:
        mod.set_sidecar_policy(mode, threshold)
        mod.reset_sidecar_policy_stats()
        n += 1
    return n


def gather_policy_stats(modules) -> Dict[str, Any]:
    """Aggregate per-layer policy stats into single summary."""
    apply_total = 0
    skip_total = 0
    all_norms: List[float] = []
    layer_stats: Dict[str, Any] = {}
    for name, mod in modules:
        s = mod.get_sidecar_policy_stats()
        layer_stats[name] = s
        apply_total += s.get("apply_count", 0)
        skip_total += s.get("skip_count", 0)
        norms = mod._sidecar_norms
        if norms:
            all_norms.extend(norms)

    norms_arr = np.asarray(all_norms, dtype=np.float64) if all_norms else np.empty(0)
    denom = apply_total + skip_total
    return {
        "apply_count": int(apply_total),
        "skip_count": int(skip_total),
        "apply_rate": float(apply_total / denom) if denom else 0.0,
        "mean_norm": float(norms_arr.mean()) if norms_arr.size else 0.0,
        "p50_norm":  float(np.percentile(norms_arr, 50)) if norms_arr.size else 0.0,
        "p95_norm":  float(np.percentile(norms_arr, 95)) if norms_arr.size else 0.0,
        "n_layers": len(modules),
        "per_layer": layer_stats,
    }


# ---------------------------------------------------------------------------
# PPL evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ppl(model, chunks: List[torch.Tensor], device: str) -> Dict[str, Any]:
    """Standard chunk-wise cross-entropy. PPL = exp(mean ce)."""
    import torch.nn.functional as F

    model.eval()
    ce_per_chunk: List[float] = []
    total_tokens = 0

    t_infer = time.time()
    for chunk in chunks:
        inp = chunk.unsqueeze(0).to(device)
        out = model(inp)
        logits = out.logits  # [1, S, V]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inp[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            reduction="mean",
        )
        ce_per_chunk.append(float(ce.item()))
        total_tokens += shift_labels.numel()
    t_infer_elapsed = time.time() - t_infer

    mean_ce = float(np.mean(ce_per_chunk)) if ce_per_chunk else float("nan")
    ppl = float(math.exp(mean_ce)) if math.isfinite(mean_ce) else float("nan")
    tok_s = total_tokens / t_infer_elapsed if t_infer_elapsed > 0 else 0.0

    return {
        "ppl": ppl,
        "mean_ce": mean_ce,
        "ce_per_chunk": ce_per_chunk,
        "wall_time_s": round(t_infer_elapsed, 3),
        "tok_s": round(tok_s, 2),
        "total_tokens": int(total_tokens),
    }


# ---------------------------------------------------------------------------
# Cost block (WO-RECEIPT-COST-01)
# ---------------------------------------------------------------------------

def _cost_block(t_start: float, cpu_start: float, ts_start: str) -> Dict[str, Any]:
    return {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s":  round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": ts_start,
        "timestamp_end": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


# ---------------------------------------------------------------------------
# Worker — single mode
# ---------------------------------------------------------------------------

def run_worker(mode: str, threshold: float, output_path: Path) -> int:
    t_start = time.time()
    cpu_start = time.process_time()
    ts_start = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    seed_all(SEED)
    vram_pre = vram_preflight()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer, model = load_model()
    modules = enumerate_helix_linear(model)
    n_set = apply_policy_to_all(modules, mode, threshold)
    chunks = load_wikitext_test_chunks(tokenizer, N_CHUNKS, MAX_LENGTH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppl_out = eval_ppl(model, chunks, device)
    policy_out = gather_policy_stats(modules)

    peak_vram_mib = None
    if torch.cuda.is_available():
        peak_vram_mib = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

    result = {
        "wo_id": WO_ID,
        "mode": mode,
        "threshold": threshold,
        "model": "zamba2-1.2b-helix",
        "dataset": DATASET,
        "dataset_split": DATASET_SPLIT,
        "n_chunks": N_CHUNKS,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "n_helix_linear_modules": n_set,
        "ppl": ppl_out["ppl"],
        "mean_ce": ppl_out["mean_ce"],
        "ce_per_chunk": ppl_out["ce_per_chunk"],
        "wall_time_s": ppl_out["wall_time_s"],
        "tok_s": ppl_out["tok_s"],
        "total_tokens": ppl_out["total_tokens"],
        "peak_vram_mib": peak_vram_mib,
        "vram_pre": vram_pre,
        "policy": policy_out,
        "cost": _cost_block(t_start, cpu_start, ts_start),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"[worker] {mode} threshold={threshold} -> ppl={result['ppl']:.4f} "
          f"tok/s={result['tok_s']:.2f} wall={result['wall_time_s']:.1f}s "
          f"apply={policy_out['apply_count']} skip={policy_out['skip_count']}")
    return 0


# ---------------------------------------------------------------------------
# Worker — phase B (50-chunk correlation)
# ---------------------------------------------------------------------------

def run_phase_b(output_path: Path) -> int:
    from scipy.stats import pearsonr

    t_start = time.time()
    cpu_start = time.process_time()
    ts_start = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    seed_all(SEED)
    vram_preflight()

    tokenizer, model = load_model()
    modules = enumerate_helix_linear(model)
    chunks = load_wikitext_test_chunks(tokenizer, N_CHUNKS, MAX_LENGTH)[:PHASE_B_N_CHUNKS]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import torch.nn.functional as F

    ce_on: List[float] = []
    ce_off: List[float] = []
    mean_norms: List[float] = []

    for i, chunk in enumerate(chunks):
        # ---- always_off pass ----
        apply_policy_to_all(modules, "always_off", 0.0)
        with torch.no_grad():
            inp = chunk.unsqueeze(0).to(device)
            out = model(inp)
            logits = out.logits[:, :-1, :].contiguous()
            labels = inp[:, 1:].contiguous()
            off_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1),
                reduction="mean",
            )
        ce_off.append(float(off_ce.item()))

        # ---- threshold=0.0 pass (applies everything AND records norms) ----
        apply_policy_to_all(modules, "threshold", 0.0)
        with torch.no_grad():
            out = model(inp)
            logits = out.logits[:, :-1, :].contiguous()
            on_ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1),
                reduction="mean",
            )
        ce_on.append(float(on_ce.item()))

        # Collect per-chunk mean sidecar norm (averaged across all layers & calls)
        per_layer_means: List[float] = []
        for _, mod in modules:
            if mod._sidecar_norms:
                per_layer_means.append(float(np.mean(mod._sidecar_norms)))
        mean_norms.append(float(np.mean(per_layer_means)) if per_layer_means else 0.0)

    benefits = [o - n for o, n in zip(ce_off, ce_on)]
    if len(benefits) >= 2 and float(np.std(mean_norms)) > 0 and float(np.std(benefits)) > 0:
        rho, pval = pearsonr(mean_norms, benefits)
    else:
        rho, pval = float("nan"), float("nan")

    result = {
        "wo_id": WO_ID,
        "phase": "B",
        "n_chunks_analyzed": len(chunks),
        "ce_off_per_chunk": ce_off,
        "ce_on_per_chunk":  ce_on,
        "benefit_per_chunk": benefits,
        "mean_norm_per_chunk": mean_norms,
        "correlation_norm_vs_benefit": float(rho),
        "p_value": float(pval),
        "cost": _cost_block(t_start, cpu_start, ts_start),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"[phase_b] rho={rho:.4f} p={pval:.3e} n={len(chunks)}")
    return 0


# ---------------------------------------------------------------------------
# Driver — spawn 5 modes + phase B, aggregate receipt
# ---------------------------------------------------------------------------

def run_driver() -> int:
    t_start = time.time()
    cpu_start = time.process_time()
    ts_start = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = RECEIPT_DIR / f"tmp_{ts_tag}"
    work_dir.mkdir(parents=True, exist_ok=True)

    me = str(Path(__file__).resolve())

    mode_results: Dict[str, Any] = {}
    for label, mode, threshold in MODES:
        out_path = work_dir / f"{label}.json"
        print(f"[driver] === {label} mode={mode} threshold={threshold} ===")
        cmd = [
            sys.executable, me,
            "--worker",
            "--mode", mode,
            "--threshold", str(threshold),
            "--output", str(out_path),
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[driver] {label} failed rc={rc}", file=sys.stderr)
            return rc
        mode_results[label] = json.loads(out_path.read_text())

    # Phase B
    print("[driver] === phase_b ===")
    phase_b_path = work_dir / "phase_b.json"
    rc = subprocess.call([
        sys.executable, me,
        "--phase-b",
        "--output", str(phase_b_path),
    ])
    phase_b = json.loads(phase_b_path.read_text()) if rc == 0 else {"error": f"rc={rc}"}

    # Aggregate deltas
    a = mode_results["A_always_on"]
    b = mode_results["B_always_off"]
    deltas_vs_a: Dict[str, Any] = {}
    recovered: Dict[str, Any] = {}
    for label in ("C1_t_0.10", "C2_t_0.15", "C3_t_0.20"):
        c = mode_results[label]
        q_delta_pct = 100.0 * (c["ppl"] - a["ppl"]) / a["ppl"] if a["ppl"] > 0 else float("nan")
        speed_gain_pct = 100.0 * (a["wall_time_s"] - c["wall_time_s"]) / a["wall_time_s"] \
            if a["wall_time_s"] > 0 else float("nan")
        deltas_vs_a[label] = {
            "quality_delta_pct": round(q_delta_pct, 4),
            "speed_gain_pct": round(speed_gain_pct, 4),
        }
        gap = a["wall_time_s"] - b["wall_time_s"]
        recovered[label] = round(
            (c["wall_time_s"] - b["wall_time_s"]) / gap, 4
        ) if gap != 0 else float("nan")

    # Gate evaluation
    gate_result = "fail"
    rho = phase_b.get("correlation_norm_vs_benefit", float("nan"))
    for label in ("C1_t_0.10", "C2_t_0.15", "C3_t_0.20"):
        q = deltas_vs_a[label]["quality_delta_pct"]
        rec = recovered[label]
        if q is None or rec is None:
            continue
        if rec >= 0.5 and q < 5.0:
            gate_result = "strong"
            break
        if rec >= 0.2 and gate_result not in ("strong",):
            gate_result = "tier_2"
        if deltas_vs_a[label]["speed_gain_pct"] > 0 and q < 5.0 and gate_result == "fail":
            gate_result = "tier_1"

    if gate_result in ("tier_1", "tier_2", "strong"):
        if not (rho == rho) or rho < 0.3:  # NaN or below floor
            conclusion = (
                f"Frontier exists ({gate_result}) but is NOT driven by sidecar norm "
                f"(rho={rho:.3f}). Investigate other causes."
            )
        else:
            conclusion = (
                f"Policy frontier receipted at {gate_result} with rho={rho:.3f}. "
                f"Threshold gating is a real signal."
            )
    else:
        conclusion = f"No policy frontier (rho={rho}). Sidecar routing is not a win on this model."

    receipt = {
        "wo_id": WO_ID,
        "model": "zamba2-1.2b-helix",
        "dataset": DATASET,
        "dataset_split": DATASET_SPLIT,
        "n_chunks": N_CHUNKS,
        "seeds": {"torch": SEED, "numpy": SEED, "random": SEED},
        "modes": {
            label: {
                "ppl": r["ppl"],
                "wall_time_s": r["wall_time_s"],
                "tok_s": r["tok_s"],
                "peak_vram_mib": r["peak_vram_mib"],
                "apply_count": r["policy"]["apply_count"],
                "skip_count":  r["policy"]["skip_count"],
                "apply_rate":  r["policy"]["apply_rate"],
                "mean_norm":   r["policy"]["mean_norm"],
                "p50_norm":    r["policy"]["p50_norm"],
                "p95_norm":    r["policy"]["p95_norm"],
            }
            for label, r in mode_results.items()
        },
        "deltas_vs_A": deltas_vs_a,
        "recovered_speed_fraction": recovered,
        "phase_b": {
            "n_chunks_analyzed": phase_b.get("n_chunks_analyzed"),
            "correlation_norm_vs_benefit": phase_b.get("correlation_norm_vs_benefit"),
            "p_value": phase_b.get("p_value"),
            "best_threshold_by_pareto": None,  # filled by Pareto analysis — TODO
        },
        "gate_result": gate_result,
        "conclusion": conclusion,
        "cost": _cost_block(t_start, cpu_start, ts_start),
    }

    receipt_path = RECEIPT_DIR / f"sidecar_policy_{ts_tag}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n[driver] receipt: {receipt_path}")
    print(f"[driver] gate:    {gate_result}")
    print(f"[driver] rho:     {rho}")
    print(f"[driver] {conclusion}")
    return 0 if gate_result != "fail" else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=WO_ID)
    p.add_argument("--worker", action="store_true",
                   help="run ONE mode (internal — invoked by driver)")
    p.add_argument("--phase-b", action="store_true",
                   help="run Phase B correlation worker")
    p.add_argument("--mode", choices=("default", "always_on", "always_off", "threshold"),
                   default="default")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--output", type=Path, default=Path("/tmp/sidecar_policy_out.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase_b:
        return run_phase_b(args.output)
    if args.worker:
        return run_worker(args.mode, args.threshold, args.output)
    return run_driver()


if __name__ == "__main__":
    sys.exit(main())
