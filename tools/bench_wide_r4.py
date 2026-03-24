#!/usr/bin/env python3
"""
WO-WIDE-R4-01: Kurtosis Routing Economic Justification

Tests the null hypothesis: uniform SVD rank-4 on ALL tensors matches or
beats kurtosis-routed SVD rank-8 on selected tensors.

Three strategies compared on same model (TinyLlama), same eval (WikiText-2):

  Strategy A — VQ-only:       No SVD anywhere (cheapest baseline)
  Strategy B — Kurtosis-routed: SVD rank-8 on kurtosis>5 tensors only (current policy)
  Strategy C — Wide r4:       SVD rank-4 on ALL 2D tensors (null hypothesis)

If B beats C at lower SVD byte cost, the routing gate is economically justified.
If C ties or beats B, the routing gate is vacuous — uniform low-rank wins.

Usage:
    python tools/bench_wide_r4.py
    python tools/bench_wide_r4.py --tokens 4096

Work Order: WO-WIDE-R4-01
"""

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kurtosis as scipy_kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.helix_linear import (
    load_helix_linear_from_cdnav3,
    swap_summary,
    swap_to_helix,
)
from helix_substrate.tensor_policy import (
    TensorPolicy,
    classify_tensor,
    get_default_policy,
    get_policy,
)

MODEL_DIR = Path.home() / "models" / "tinyllama_fp32"
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "wide_r4"

BLOCK_TENSOR_TYPES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

HF_PATTERNS = {
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}

N_BLOCKS = 22


@dataclass
class Strategy:
    name: str
    description: str
    svd_tensors: int = 0
    svd_bytes: int = 0
    total_bytes: int = 0
    dense_bytes: int = 0
    mean_cosine: float = 0.0
    ppl: float = 0.0
    compression_time_s: float = 0.0


def get_module(model, block_idx, tensor_type):
    """Get the nn.Linear module for a tensor in a block."""
    layer = model.model.layers[block_idx]
    if tensor_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, tensor_type)
    return getattr(layer.mlp, tensor_type)


def compute_perplexity(model, eval_tokens, seq_len=2048):
    """Compute perplexity. Returns (ppl, nll, n_tokens)."""
    model.eval()
    nlls = []
    n_tokens = 0

    for i in range(0, len(eval_tokens) - 1, seq_len):
        end = min(i + seq_len, len(eval_tokens))
        input_ids = torch.tensor(
            eval_tokens[i:end], dtype=torch.long
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        chunk_tokens = input_ids.shape[1] - 1
        nlls.append(outputs.loss.item() * chunk_tokens)
        n_tokens += chunk_tokens

        if end >= len(eval_tokens):
            break

    mean_nll = sum(nlls) / n_tokens
    return float(np.exp(mean_nll)), mean_nll, n_tokens


def compress_and_load(
    sf,
    model_template,
    tmpdir: Path,
    strategy_name: str,
    kurtosis_map: dict,
    make_policy_fn,
):
    """
    Compress all tensors with a given policy function, load as HelixLinear.

    Args:
        sf: safetensors file handle
        model_template: model to clone for module surgery
        tmpdir: temp directory for CDNA output
        strategy_name: label for this strategy
        kurtosis_map: dict of tensor_name -> kurtosis value
        make_policy_fn: callable(name, shape, block_idx, kurtosis) -> TensorPolicy

    Returns:
        (model_with_helix, strategy_stats)
    """
    import copy
    model = copy.deepcopy(model_template)

    cdna_dir = tmpdir / strategy_name
    writer = CDNAv3Writer(cdna_dir)

    t0 = time.time()
    helix_modules = {}
    total_svd_bytes = 0
    total_compressed_bytes = 0
    total_dense_bytes = 0
    svd_tensor_count = 0
    cosines = []

    for block_idx in range(N_BLOCKS):
        for tensor_type in BLOCK_TENSOR_TYPES:
            hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
            tensor_np = sf.get_tensor(hf_name).astype(np.float32)
            kurt = kurtosis_map[hf_name]

            policy = make_policy_fn(hf_name, tensor_np.shape, block_idx, kurt)

            stats = writer.write_tensor(tensor_np, hf_name, policy=policy)

            # Track SVD usage
            svd_b = stats.get("svd_bytes", 0)
            total_svd_bytes += svd_b
            if svd_b > 0:
                svd_tensor_count += 1

            total_compressed_bytes += stats.get("compressed_bytes", 0)
            total_dense_bytes += stats.get("original_bytes", 0)
            cosines.append(stats.get("cosine_with_svd", stats.get("cosine_with_sidecar", 0)))

            # Load as HelixLinear
            safe_name = hf_name.replace("/", "_").replace(".", "_")
            tensor_dir = cdna_dir / f"{safe_name}.cdnav3"

            module_path = hf_name.replace(".weight", "")
            mod = get_module(model, block_idx, tensor_type)
            bias = mod.bias.data.clone() if mod.bias is not None else None

            helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
            helix_modules[module_path] = helix_mod

    compression_time = time.time() - t0

    # Swap
    model = swap_to_helix(model, helix_modules)

    strat = Strategy(
        name=strategy_name,
        description="",
        svd_tensors=svd_tensor_count,
        svd_bytes=total_svd_bytes,
        total_bytes=total_compressed_bytes,
        dense_bytes=total_dense_bytes,
        mean_cosine=float(np.mean(cosines)),
        compression_time_s=round(compression_time, 2),
    )

    return model, strat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        print(f"ERROR: TinyLlama not found at {MODEL_DIR}")
        sys.exit(1)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("WO-WIDE-R4-01: Kurtosis Routing Economic Justification")
    print("=" * 70)

    # --- Load model and tokenizer ---
    print("\n[1/6] Loading TinyLlama...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model_template = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model_template.eval()

    # --- Load eval tokens ---
    print(f"[2/6] Loading WikiText-2 ({args.tokens} tokens)...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:args.tokens]
    print(f"  Tokens: {len(eval_tokens)}")

    # --- Baseline perplexity ---
    print("[3/6] Computing baseline (dense) perplexity...")
    ppl_baseline, _, n_tokens = compute_perplexity(model_template, eval_tokens)
    print(f"  Baseline PPL: {ppl_baseline:.4f}")

    # --- Precompute kurtosis for all tensors ---
    print("[4/6] Computing kurtosis for all tensors...")
    sf_path = MODEL_DIR / "model.safetensors"
    kurtosis_map = {}

    with safe_open(str(sf_path), framework="numpy") as sf:
        for block_idx in range(N_BLOCKS):
            for tensor_type in BLOCK_TENSOR_TYPES:
                hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                tensor_np = sf.get_tensor(hf_name).astype(np.float32)
                kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
                kurtosis_map[hf_name] = kurt

    n_high = sum(1 for k in kurtosis_map.values() if k > 5)
    print(f"  {len(kurtosis_map)} tensors, {n_high} with kurtosis > 5")

    # --- Run three strategies ---
    strategies = []

    with safe_open(str(sf_path), framework="numpy") as sf:

        # Strategy A: VQ-only (no SVD)
        print("\n[5/6] Compressing with three strategies...")
        print("  Strategy A: VQ-only (no SVD)...")

        def policy_vq_only(name, shape, block_idx, kurt):
            tc = classify_tensor(name, shape=shape)
            return get_default_policy(tc)  # svd_residual_rank=0 by default

        model_a, strat_a = compress_and_load(
            sf, model_template, Path(tempfile.mkdtemp()),
            "vq_only", kurtosis_map, policy_vq_only,
        )
        strat_a.description = "VQ + sidecar, no SVD"

        ppl_a, _, _ = compute_perplexity(model_a, eval_tokens)
        strat_a.ppl = ppl_a
        del model_a
        strategies.append(strat_a)
        print(f"    PPL: {ppl_a:.4f}, SVD tensors: 0, SVD bytes: 0")

        # Strategy B: Kurtosis-routed (current policy)
        print("  Strategy B: Kurtosis-routed SVD rank-8...")

        def policy_routed(name, shape, block_idx, kurt):
            return get_policy(name, shape, block_idx=block_idx,
                              kurtosis=kurt, n_blocks=N_BLOCKS)

        model_b, strat_b = compress_and_load(
            sf, model_template, Path(tempfile.mkdtemp()),
            "kurtosis_routed", kurtosis_map, policy_routed,
        )
        strat_b.description = "VQ + sidecar + SVD rank-8 on kurtosis>5 tensors"

        ppl_b, _, _ = compute_perplexity(model_b, eval_tokens)
        strat_b.ppl = ppl_b
        del model_b
        strategies.append(strat_b)
        print(f"    PPL: {ppl_b:.4f}, SVD tensors: {strat_b.svd_tensors}, "
              f"SVD bytes: {strat_b.svd_bytes:,}")

        # Strategy C: Wide rank-4 (null hypothesis)
        print("  Strategy C: Wide SVD rank-4 on ALL tensors...")

        def policy_wide_r4(name, shape, block_idx, kurt):
            from dataclasses import replace
            tc = classify_tensor(name, shape=shape)
            base = get_default_policy(tc)
            if len(shape) == 2 and base.storage_mode not in ("exact", "morpho"):
                return replace(base, svd_residual_rank=4)
            return base

        model_c, strat_c = compress_and_load(
            sf, model_template, Path(tempfile.mkdtemp()),
            "wide_r4", kurtosis_map, policy_wide_r4,
        )
        strat_c.description = "VQ + sidecar + SVD rank-4 on ALL 2D tensors"

        ppl_c, _, _ = compute_perplexity(model_c, eval_tokens)
        strat_c.ppl = ppl_c
        del model_c
        strategies.append(strat_c)
        print(f"    PPL: {ppl_c:.4f}, SVD tensors: {strat_c.svd_tensors}, "
              f"SVD bytes: {strat_c.svd_bytes:,}")

    # --- Results ---
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Baseline (dense) PPL: {ppl_baseline:.4f}")
    print()

    print(f"  {'Strategy':<30} {'PPL':>8} {'Δ PPL':>10} {'Δ%':>8} "
          f"{'SVD tensors':>12} {'SVD MB':>8} {'Cosine':>8} {'Ratio':>6}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*6}")

    for s in strategies:
        delta = s.ppl - ppl_baseline
        pct = 100 * delta / ppl_baseline
        ratio = s.dense_bytes / max(1, s.total_bytes)
        print(f"  {s.description:<30} {s.ppl:>8.4f} {delta:>+10.4f} {pct:>+7.2f}% "
              f"{s.svd_tensors:>12} {s.svd_bytes / 1024 / 1024:>8.2f} "
              f"{s.mean_cosine:>8.6f} {ratio:>6.2f}x")

    # --- Verdict ---
    print()
    b_wins = strat_b.ppl < strat_c.ppl
    b_cheaper = strat_b.svd_bytes < strat_c.svd_bytes

    if b_wins and b_cheaper:
        verdict = "ROUTING_WINS_DECISIVE"
        explanation = (
            f"Kurtosis routing beats wide_r4 on quality (PPL {strat_b.ppl:.4f} vs "
            f"{strat_c.ppl:.4f}) AND costs less SVD storage "
            f"({strat_b.svd_bytes:,} vs {strat_c.svd_bytes:,} bytes)"
        )
    elif b_wins:
        verdict = "ROUTING_WINS_QUALITY"
        explanation = (
            f"Kurtosis routing beats wide_r4 on quality (PPL {strat_b.ppl:.4f} vs "
            f"{strat_c.ppl:.4f}) but uses more SVD storage"
        )
    elif abs(strat_b.ppl - strat_c.ppl) < 0.01:
        verdict = "TIE"
        explanation = (
            f"PPL within 0.01 ({strat_b.ppl:.4f} vs {strat_c.ppl:.4f}). "
            f"Routing gate is measurement science, not economic advantage"
        )
    else:
        verdict = "WIDE_R4_WINS"
        explanation = (
            f"Wide_r4 beats routing on quality (PPL {strat_c.ppl:.4f} vs "
            f"{strat_b.ppl:.4f}). Routing gate is vacuous — uniform low-rank wins"
        )

    print(f"  VERDICT: {verdict}")
    print(f"  {explanation}")

    # --- Receipt ---
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "work_order": "WO-WIDE-R4-01",
        "description": "Kurtosis routing economic justification",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "eval_tokens": len(eval_tokens),
        "baseline_ppl": round(ppl_baseline, 6),
        "n_tensors": len(kurtosis_map),
        "n_kurtosis_above_5": n_high,
        "strategies": [],
        "verdict": verdict,
        "explanation": explanation,
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
        },
    }

    for s in strategies:
        delta = s.ppl - ppl_baseline
        pct = 100 * delta / ppl_baseline
        receipt["strategies"].append({
            "name": s.name,
            "description": s.description,
            "ppl": round(s.ppl, 6),
            "ppl_delta": round(delta, 6),
            "ppl_delta_pct": round(pct, 4),
            "svd_tensors": s.svd_tensors,
            "svd_bytes": s.svd_bytes,
            "total_compressed_bytes": s.total_bytes,
            "dense_equivalent_bytes": s.dense_bytes,
            "compression_ratio": round(s.dense_bytes / max(1, s.total_bytes), 2),
            "mean_cosine": round(s.mean_cosine, 6),
            "compression_time_s": s.compression_time_s,
        })

    receipt_path = RECEIPT_DIR / f"wide_r4_{time.strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
