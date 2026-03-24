#!/usr/bin/env python3
"""
Qwen2.5-Coder-1.5B Phase 3: End-to-End Proof

Full compression + perplexity validation + GPU fit + coding prompts.

Phase A: CPU perplexity comparison (baseline BF16 vs HelixLinear)
Phase B: GPU fit test (load compressed model, measure VRAM)
Phase C: Coding-style qualitative prompts

Work Order: WO-QWEN-COMPRESS-01, Phase 3
"""

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
    load_helix_linear_from_cdnav3,
    swap_summary,
    swap_to_helix,
)
from helix_substrate.tensor_policy import get_policy

MODEL_DIR = Path.home() / "models" / "qwen2.5-coder-1.5b-instruct"
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "qwen_phase3"

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

N_BLOCKS = 28
EVAL_TOKENS = 2048  # shorter than TinyLlama's 4096 to keep CPU eval reasonable

CODING_PROMPTS = [
    {
        "name": "python_function",
        "prompt": "<|im_start|>user\nWrite a Python function that checks if a string is a valid IPv4 address.<|im_end|>\n<|im_start|>assistant\n",
    },
    {
        "name": "bug_fix",
        "prompt": "<|im_start|>user\nFix the bug in this code:\n```python\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result\n```<|im_end|>\n<|im_start|>assistant\n",
    },
    {
        "name": "controller_tool",
        "prompt": "<|im_start|>user\nGiven these available tools: [search_docs(query), run_test(file), get_status()], write the code to search for 'authentication', then run the auth test file.<|im_end|>\n<|im_start|>assistant\n",
    },
]


def get_module(model, block_idx, tensor_type):
    """Get the nn.Linear module for a tensor in a block."""
    layer = model.model.layers[block_idx]
    if tensor_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, tensor_type)
    return getattr(layer.mlp, tensor_type)


def compute_perplexity(model, eval_tokens, device="cpu", seq_len=2048):
    """Compute perplexity. Returns (ppl, nll, n_tokens)."""
    model.eval()
    nlls = []
    n_tokens = 0

    for i in range(0, len(eval_tokens) - 1, seq_len):
        end = min(i + seq_len, len(eval_tokens))
        input_ids = torch.tensor(
            eval_tokens[i:end], dtype=torch.long
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        chunk_tokens = input_ids.shape[1] - 1
        nlls.append(outputs.loss.item() * chunk_tokens)
        n_tokens += chunk_tokens

        if end >= len(eval_tokens):
            break

    mean_nll = sum(nlls) / n_tokens
    return float(np.exp(mean_nll)), mean_nll, n_tokens


def generate_text(model, tokenizer, prompt, device="cpu", max_new=128):
    """Generate text from a prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        t0 = time.time()
        output = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_time = time.time() - t0

    new_tokens = output.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    tok_s = new_tokens / gen_time if gen_time > 0 else 0
    return text, new_tokens, gen_time, tok_s


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("  Qwen2.5-Coder-1.5B: Phase 3 End-to-End Proof")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open

    # ================================================================
    # PHASE A: CPU Perplexity Comparison
    # ================================================================
    print("\n[Phase A] CPU Perplexity Comparison")
    print("-" * 70)

    # Load model in FP32 for compression compatibility
    print("[A1] Loading Qwen2.5-Coder-1.5B (FP32)...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model.eval()

    # Load eval tokens
    print(f"[A2] Loading WikiText-2 ({EVAL_TOKENS} tokens)...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:EVAL_TOKENS]
    print(f"  Tokens: {len(eval_tokens)}")

    # Baseline perplexity
    print("[A3] Computing baseline perplexity (CPU, may take a few minutes)...")
    t_baseline = time.time()
    ppl_baseline, nll_baseline, n_tokens = compute_perplexity(model, eval_tokens)
    baseline_time = time.time() - t_baseline
    print(f"  Baseline PPL: {ppl_baseline:.4f} (NLL: {nll_baseline:.6f}, {baseline_time:.1f}s)")

    # Compress all blocks
    print(f"\n[A4] Compressing all {N_BLOCKS} blocks × {len(BLOCK_TENSOR_TYPES)} types "
          f"= {N_BLOCKS * len(BLOCK_TENSOR_TYPES)} tensors...")

    sf_path = MODEL_DIR / "model.safetensors"
    compress_start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        writer = CDNAv3Writer(tmpdir)

        helix_modules = {}
        total_tensors = 0
        total_savings = 0
        per_tensor_cosines = []

        with safe_open(str(sf_path), framework="pt") as sf:
            for block_idx in range(N_BLOCKS):
                for tensor_type in BLOCK_TENSOR_TYPES:
                    hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                    tensor_np = sf.get_tensor(hf_name).float().numpy()

                    shape = tensor_np.shape
                    kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
                    policy = get_policy(
                        hf_name, shape,
                        block_idx=block_idx, kurtosis=kurt
                    )

                    stats = writer.write_tensor(tensor_np, hf_name, policy=policy)

                    # Load as HelixLinear
                    safe_name = hf_name.replace("/", "_").replace(".", "_")
                    tensor_dir = tmpdir / f"{safe_name}.cdnav3"

                    module_path = hf_name.replace(".weight", "")
                    mod = get_module(model, block_idx, tensor_type)
                    bias = mod.bias.data.clone() if mod.bias is not None else None

                    helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
                    helix_modules[module_path] = helix_mod

                    total_tensors += 1
                    savings = helix_mod.memory_savings()
                    total_savings += savings["dense_bytes"] - savings["compressed_bytes"]

                    # Track cosine from stats.json
                    stats_file = tensor_dir / "stats.json"
                    if stats_file.exists():
                        s = json.loads(stats_file.read_text())
                        cos = s.get("cosine_with_svd") or s.get("cosine_with_sidecar") or s.get("cosine_no_sidecar", 0)
                        per_tensor_cosines.append({
                            "name": hf_name, "block": block_idx,
                            "type": tensor_type, "cosine": cos,
                        })

                if (block_idx + 1) % 7 == 0 or block_idx == N_BLOCKS - 1:
                    print(f"  Block {block_idx + 1}/{N_BLOCKS} done "
                          f"({total_tensors} tensors, "
                          f"savings: {total_savings / 1024 / 1024:.0f} MB)")

        compress_time = time.time() - compress_start
        print(f"  Compression: {compress_time:.0f}s")

        # Swap nn.Linear → HelixLinear
        print(f"\n[A5] Swapping {len(helix_modules)} modules...")
        model = swap_to_helix(model, helix_modules)
        summary = swap_summary(model)
        print(f"  {summary['helix_modules']} HelixLinear, "
              f"{summary['linear_modules']} nn.Linear remaining")
        print(f"  Compression ratio: {summary['overall_ratio']}x")
        print(f"  Compressed size: {summary['compressed_bytes'] / 1e6:.0f} MB")
        print(f"  Dense equivalent: {summary['dense_equivalent_bytes'] / 1e6:.0f} MB")

        # Compressed perplexity
        print("\n[A6] Computing HelixLinear perplexity (CPU)...")
        t_helix = time.time()
        ppl_helix, nll_helix, _ = compute_perplexity(model, eval_tokens)
        helix_time = time.time() - t_helix
        print(f"  HelixLinear PPL: {ppl_helix:.4f} (NLL: {nll_helix:.6f}, {helix_time:.1f}s)")

    ppl_delta = ppl_helix - ppl_baseline
    ppl_pct = 100 * ppl_delta / ppl_baseline

    print(f"\n  {'=' * 60}")
    print(f"  PHASE A RESULTS")
    print(f"  {'=' * 60}")
    print(f"  Baseline PPL:    {ppl_baseline:.4f}")
    print(f"  HelixLinear PPL: {ppl_helix:.4f}")
    print(f"  Delta:           {ppl_delta:+.4f} ({ppl_pct:+.2f}%)")
    print(f"  Compression:     {summary['overall_ratio']}x")
    print(f"  Tensors swapped: {total_tensors}")

    # ================================================================
    # PHASE B: GPU Fit Test
    # ================================================================
    gpu_results = None
    if torch.cuda.is_available():
        print(f"\n[Phase B] GPU Fit Test")
        print("-" * 70)

        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"  GPU: {gpu_name} ({gpu_mem:.0f} MB)")

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            print("[B1] Moving HelixLinear model to GPU...")
            model = model.cuda()
            model.eval()

            vram_model = torch.cuda.memory_allocated() / 1e6
            print(f"  VRAM model: {vram_model:.0f} MB ({vram_model / gpu_mem * 100:.0f}%)")

            # Quick forward pass to measure peak
            print("[B2] Forward pass (peak VRAM)...")
            test_ids = torch.tensor(eval_tokens[:128], dtype=torch.long).unsqueeze(0).cuda()
            with torch.no_grad():
                _ = model(test_ids)
            vram_peak = torch.cuda.max_memory_allocated() / 1e6
            print(f"  VRAM peak: {vram_peak:.0f} MB ({vram_peak / gpu_mem * 100:.0f}%)")
            print(f"  Headroom: {gpu_mem - vram_peak:.0f} MB")

            # GPU perplexity for speed comparison
            print("[B3] GPU perplexity...")
            t_gpu = time.time()
            ppl_gpu, nll_gpu, _ = compute_perplexity(model, eval_tokens, device="cuda")
            gpu_ppl_time = time.time() - t_gpu
            print(f"  GPU PPL: {ppl_gpu:.4f} (matches CPU: {abs(ppl_gpu - ppl_helix) < 0.01}, {gpu_ppl_time:.1f}s)")

            gpu_results = {
                "gpu_name": gpu_name,
                "gpu_mem_mb": round(gpu_mem),
                "vram_model_mb": round(vram_model),
                "vram_peak_mb": round(vram_peak),
                "vram_headroom_mb": round(gpu_mem - vram_peak),
                "vram_utilization_pct": round(vram_peak / gpu_mem * 100, 1),
                "gpu_ppl": round(ppl_gpu, 6),
                "gpu_ppl_matches_cpu": abs(ppl_gpu - ppl_helix) < 0.01,
                "gpu_ppl_time_s": round(gpu_ppl_time, 1),
                "fit": True,
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  GPU OOM: {e}")
                gpu_results = {"fit": False, "error": str(e)}
                torch.cuda.empty_cache()
                model = model.cpu()
            else:
                raise

    # ================================================================
    # PHASE C: Coding Qualitative Prompts
    # ================================================================
    print(f"\n[Phase C] Coding Qualitative Prompts")
    print("-" * 70)

    device = "cuda" if gpu_results and gpu_results.get("fit") else "cpu"
    prompt_results = []

    for p in CODING_PROMPTS:
        print(f"\n  --- {p['name']} ---")
        text, n_tok, gen_time, tok_s = generate_text(
            model, tokenizer, p["prompt"], device=device, max_new=128
        )
        print(f"  Tokens: {n_tok}, Time: {gen_time:.1f}s, Speed: {tok_s:.1f} tok/s")
        print(f"  Output:\n    {text[:300]}{'...' if len(text) > 300 else ''}")

        prompt_results.append({
            "name": p["name"],
            "tokens_generated": n_tok,
            "generation_time_s": round(gen_time, 2),
            "tok_s": round(tok_s, 1),
            "output_preview": text[:500],
            "device": device,
        })

    # ================================================================
    # RECEIPT
    # ================================================================
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

    # Cosine summary
    if per_tensor_cosines:
        cosines = [c["cosine"] for c in per_tensor_cosines]
        cosine_summary = {
            "avg": round(float(np.mean(cosines)), 6),
            "min": round(float(np.min(cosines)), 6),
            "max": round(float(np.max(cosines)), 6),
            "below_999": sum(1 for c in cosines if c < 0.999),
            "below_998": sum(1 for c in cosines if c < 0.998),
        }
    else:
        cosine_summary = {}

    receipt = {
        "schema": "qwen_e2e_proof_v1",
        "work_order": "WO-QWEN-COMPRESS-01",
        "phase": 3,
        "model": "Qwen2.5-Coder-1.5B-Instruct",
        "model_params": 1_543_714_304,
        "n_blocks": N_BLOCKS,
        "eval_tokens": len(eval_tokens),
        "phase_a": {
            "baseline_ppl": round(ppl_baseline, 6),
            "helix_ppl": round(ppl_helix, 6),
            "ppl_delta": round(ppl_delta, 6),
            "ppl_delta_pct": round(ppl_pct, 4),
            "baseline_time_s": round(baseline_time, 1),
            "helix_time_s": round(helix_time, 1),
            "compression_time_s": round(compress_time, 1),
            "tensors_swapped": total_tensors,
            "compression_ratio": summary["overall_ratio"],
            "compressed_bytes": summary["compressed_bytes"],
            "dense_bytes": summary["dense_equivalent_bytes"],
            "helix_modules": summary["helix_modules"],
            "linear_modules_remaining": summary["linear_modules"],
            "cosine_summary": cosine_summary,
        },
        "phase_b": gpu_results,
        "phase_c": prompt_results,
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

    # Verdict
    ppl_ok = abs(ppl_pct) < 2.0
    gpu_ok = gpu_results and gpu_results.get("fit", False) if gpu_results else False
    prompts_ok = all(p["tokens_generated"] > 10 for p in prompt_results)

    if ppl_ok and gpu_ok and prompts_ok:
        receipt["verdict"] = "PASS"
    elif ppl_ok and not gpu_ok:
        receipt["verdict"] = "PARTIAL — PPL ok, GPU failed"
    elif not ppl_ok:
        receipt["verdict"] = f"FAIL — PPL delta {ppl_pct:+.2f}% exceeds 2%"
    else:
        receipt["verdict"] = "PARTIAL"

    receipt_path = RECEIPT_DIR / f"qwen_e2e_{time.strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))

    print(f"\n{'=' * 70}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Model:           Qwen2.5-Coder-1.5B-Instruct")
    print(f"  Baseline PPL:    {ppl_baseline:.4f}")
    print(f"  HelixLinear PPL: {ppl_helix:.4f}")
    print(f"  Delta:           {ppl_delta:+.4f} ({ppl_pct:+.2f}%)")
    print(f"  Compression:     {summary['overall_ratio']}x")
    print(f"  Tensors:         {total_tensors} swapped, {summary['linear_modules']} remaining")
    if gpu_results and gpu_results.get("fit"):
        print(f"  VRAM model:      {gpu_results['vram_model_mb']} MB")
        print(f"  VRAM peak:       {gpu_results['vram_peak_mb']} MB")
        print(f"  GPU speed:       {prompt_results[0]['tok_s'] if prompt_results else '?'} tok/s")
    print(f"  Verdict:         {receipt['verdict']}")
    print(f"  Receipt:         {receipt_path}")
    print(f"  Wall time:       {receipt['cost']['wall_time_s']:.0f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
