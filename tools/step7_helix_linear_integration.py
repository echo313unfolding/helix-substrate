#!/usr/bin/env python3
"""
Step 7: HelixLinear integration test.

Validates that HelixLinear (drop-in nn.Linear replacement) produces the same
perplexity as Step 5's weight-surgery approach. This proves that "the compressed
form IS the executable" — no decompression step needed.

Usage:
    python tools/step7_helix_linear_integration.py
    python tools/step7_helix_linear_integration.py --tokens 2048

Prerequisites:
    - TinyLlama FP32 model at ~/models/tinyllama_fp32/
    - helix_substrate.helix_linear module

Work Order: WO-HELIX-LINEAR-01
"""

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kurtosis as scipy_kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.helix_linear import (
    HelixLinear,
    load_cdna_factors,
    load_helix_linear_from_cdnav3,
    swap_summary,
    swap_to_helix,
)
from helix_substrate.tensor_policy import classify_tensor, get_policy

MODEL_DIR = Path.home() / "models" / "tinyllama_fp32"
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "step7_helix_linear"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        print(f"ERROR: TinyLlama not found at {MODEL_DIR}")
        print("Download with: huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        sys.exit(1)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("Step 7: HelixLinear Integration Test")
    print("=" * 70)

    # --- Load model and tokenizer ---
    print("\n[1/5] Loading TinyLlama...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model.eval()

    # --- Load eval tokens ---
    print(f"[2/5] Loading WikiText-2 ({args.tokens} tokens)...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:args.tokens]
    print(f"  Tokens: {len(eval_tokens)}")

    # --- Baseline perplexity ---
    print("[3/5] Computing baseline perplexity...")
    ppl_baseline, nll_baseline, n_tokens = compute_perplexity(model, eval_tokens)
    print(f"  Baseline PPL: {ppl_baseline:.4f} (NLL: {nll_baseline:.6f})")

    # --- Compress all blocks to CDNA v3, build HelixLinear modules ---
    print(f"[4/5] Compressing {N_BLOCKS} blocks → HelixLinear...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        writer = CDNAv3Writer(tmpdir)

        # Read weights from safetensors
        sf_path = MODEL_DIR / "model.safetensors"
        helix_modules = {}
        total_tensors = 0
        total_savings = 0

        with safe_open(str(sf_path), framework="numpy") as sf:
            for block_idx in range(N_BLOCKS):
                for tensor_type in BLOCK_TENSOR_TYPES:
                    hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                    tensor_np = sf.get_tensor(hf_name).astype(np.float32)

                    # Get routed policy
                    kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
                    policy = get_policy(
                        hf_name, tensor_np.shape,
                        block_idx=block_idx, kurtosis=kurt
                    )

                    # Write to CDNA v3
                    stats = writer.write_tensor(
                        tensor_np, hf_name, policy=policy
                    )

                    # Load as HelixLinear
                    safe_name = hf_name.replace("/", "_").replace(".", "_")
                    tensor_dir = tmpdir / f"{safe_name}.cdnav3"

                    # Get bias from model
                    module_path = hf_name.replace(".weight", "")
                    mod = get_module(model, block_idx, tensor_type)
                    bias = mod.bias.data.clone() if mod.bias is not None else None

                    helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
                    helix_modules[module_path] = helix_mod

                    total_tensors += 1
                    savings = helix_mod.memory_savings()
                    total_savings += savings["dense_bytes"] - savings["compressed_bytes"]

                if (block_idx + 1) % 5 == 0 or block_idx == N_BLOCKS - 1:
                    print(f"  Block {block_idx + 1}/{N_BLOCKS} done "
                          f"({total_tensors} tensors, "
                          f"savings: {total_savings / 1024 / 1024:.0f} MB)")

        # --- Swap nn.Linear → HelixLinear ---
        print(f"\n  Swapping {len(helix_modules)} modules...")
        model = swap_to_helix(model, helix_modules)
        summary = swap_summary(model)
        print(f"  {summary['helix_modules']} HelixLinear, "
              f"{summary['linear_modules']} nn.Linear remaining")
        print(f"  Overall compression: {summary['overall_ratio']}x")

        # --- Compressed perplexity ---
        print("[5/5] Computing HelixLinear perplexity...")
        ppl_helix, nll_helix, _ = compute_perplexity(model, eval_tokens)
        print(f"  HelixLinear PPL: {ppl_helix:.4f} (NLL: {nll_helix:.6f})")

    # --- Results ---
    ppl_delta = ppl_helix - ppl_baseline
    ppl_pct = 100 * ppl_delta / ppl_baseline

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"  Baseline PPL:    {ppl_baseline:.4f}")
    print(f"  HelixLinear PPL: {ppl_helix:.4f}")
    print(f"  Delta:           {ppl_delta:+.4f} ({ppl_pct:+.2f}%)")
    print(f"  Compression:     {summary['overall_ratio']}x")
    print(f"  Tensors swapped: {total_tensors}")

    # --- Receipt ---
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "helix_linear_integration_v1",
        "work_order": "WO-HELIX-LINEAR-01",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "eval_tokens": len(eval_tokens),
        "baseline_ppl": round(ppl_baseline, 6),
        "helix_ppl": round(ppl_helix, 6),
        "ppl_delta": round(ppl_delta, 6),
        "ppl_delta_pct": round(ppl_pct, 4),
        "tensors_swapped": total_tensors,
        "compression_ratio": summary["overall_ratio"],
        "compressed_bytes": summary["compressed_bytes"],
        "dense_bytes": summary["dense_equivalent_bytes"],
        "helix_modules": summary["helix_modules"],
        "linear_modules_remaining": summary["linear_modules"],
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
        "verdict": "PASS" if abs(ppl_pct) < 2.0 else "FAIL",
    }

    receipt_path = RECEIPT_DIR / f"helix_linear_integration_{time.strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")
    print(f"  Verdict: {receipt['verdict']}")


if __name__ == "__main__":
    main()
