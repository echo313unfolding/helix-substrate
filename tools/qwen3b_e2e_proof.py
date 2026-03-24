#!/usr/bin/env python3
"""
Qwen2.5-Coder-3B-Instruct: CDNA v3 End-to-End Proof

Full compression + perplexity validation + GPU fit (zero-copy) + coding prompts.

Phase A: CPU perplexity comparison (baseline FP32 vs HelixLinear)
Phase B: GPU fit test via zero-copy (projected ~1488 MB VRAM)
Phase C: Coding-style qualitative prompts

WO-RECEIPT-COST-01 compliant.
"""

import gc
import json
import platform
import resource
import sys
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

MODEL_DIR = Path.home() / "models" / "qwen2.5-coder-3b-instruct"
CDNA_DIR = MODEL_DIR / "cdnav3"  # persistent — reusable after first run
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "qwen3b_proof"

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

N_BLOCKS = 36
EVAL_TOKENS = 2048  # keep CPU eval reasonable

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


def compute_perplexity_cpu_offload(model, eval_tokens, device="cuda", seq_len=2048):
    """Compute perplexity with CPU-offloaded loss computation.

    For models with large vocab (151K), FP32 logits OOM on 4GB GPU.
    Solution: forward pass on GPU, loss computation on CPU.
    """
    model.eval()
    nlls = []
    n_tokens = 0

    for i in range(0, len(eval_tokens) - 1, seq_len):
        end = min(i + seq_len, len(eval_tokens))
        input_ids = torch.tensor(
            eval_tokens[i:end], dtype=torch.long
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            if device == "cpu":
                outputs = model(input_ids, labels=input_ids)
                chunk_tokens = input_ids.shape[1] - 1
                nlls.append(outputs.loss.item() * chunk_tokens)
            else:
                # Forward without labels — get logits on GPU
                outputs = model(input_ids)
                # Move logits to CPU for loss computation (151K vocab OOMs on 4GB)
                logits_cpu = outputs.logits.float().cpu()
                labels_cpu = input_ids.cpu()

                shift_logits = logits_cpu[..., :-1, :].contiguous()
                shift_labels = labels_cpu[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                chunk_tokens = shift_labels.numel()
                nlls.append(loss.item() * chunk_tokens)

                del logits_cpu, labels_cpu, shift_logits, shift_labels
                torch.cuda.empty_cache()

        n_tokens += (input_ids.shape[1] - 1)
        del input_ids
        if device != "cpu":
            torch.cuda.empty_cache()

        if end >= len(eval_tokens):
            break

    mean_nll = sum(nlls) / n_tokens
    return float(np.exp(mean_nll)), mean_nll, n_tokens


def generate_text(model, tokenizer, prompt, device="cpu", max_new=128):
    """Generate text from a prompt."""
    from helix_substrate.helix_linear import freeze_sidecar_phase
    freeze_sidecar_phase(model, None)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.time()
        output = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        if device != "cpu":
            torch.cuda.synchronize()
        gen_time = time.time() - t0

    new_tokens = output.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    tok_s = new_tokens / gen_time if gen_time > 0 else 0
    return text, new_tokens, gen_time, tok_s


def compress_all_blocks(model, cdna_dir, n_blocks):
    """Compress all blocks to persistent CDNA v3 directory. Skips already-compressed.

    Takes a loaded model (handles sharded safetensors transparently).
    """
    cdna_dir.mkdir(parents=True, exist_ok=True)
    writer = CDNAv3Writer(cdna_dir)

    n_tensors = 0
    n_skipped = 0
    total_dense = 0
    total_compressed = 0
    per_tensor_cosines = []

    for block_idx in range(n_blocks):
        for tensor_type in BLOCK_TENSOR_TYPES:
            hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
            safe_name = hf_name.replace("/", "_").replace(".", "_")
            tensor_dir = cdna_dir / f"{safe_name}.cdnav3"

            # Skip already compressed
            if tensor_dir.exists() and (tensor_dir / "codebook.npy").exists():
                n_tensors += 1
                n_skipped += 1
                stats_file = tensor_dir / "stats.json"
                if stats_file.exists():
                    s = json.loads(stats_file.read_text())
                    cos = s.get("cosine_with_svd") or s.get("cosine_with_sidecar") or s.get("cosine_no_sidecar", 0)
                    per_tensor_cosines.append({
                        "name": hf_name, "block": block_idx,
                        "type": tensor_type, "cosine": cos,
                    })
                continue

            # Extract weight from loaded model
            mod = get_module(model, block_idx, tensor_type)
            tensor_np = mod.weight.data.float().numpy()

            shape = tensor_np.shape
            kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
            policy = get_policy(hf_name, shape, block_idx=block_idx, kurtosis=kurt)

            stats = writer.write_tensor(tensor_np, hf_name, policy=policy)
            n_tensors += 1
            total_dense += np.prod(shape) * 4
            total_compressed += stats.get("compressed_bytes", 0)

            # Read cosine
            stats_file = tensor_dir / "stats.json"
            if stats_file.exists():
                s = json.loads(stats_file.read_text())
                cos = s.get("cosine_with_svd") or s.get("cosine_with_sidecar") or s.get("cosine_no_sidecar", 0)
                per_tensor_cosines.append({
                    "name": hf_name, "block": block_idx,
                    "type": tensor_type, "cosine": cos,
                })

            del tensor_np

        if (block_idx + 1) % 9 == 0 or block_idx == n_blocks - 1:
            print(f"  Block {block_idx + 1}/{n_blocks} "
                  f"({n_tensors} tensors, {n_skipped} cached)", flush=True)

    return n_tensors, n_skipped, total_dense, total_compressed, per_tensor_cosines


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("  Qwen2.5-Coder-3B-Instruct: CDNA v3 End-to-End Proof")
    print("=" * 70)

    # Check model files exist (sharded or single)
    has_single = (MODEL_DIR / "model.safetensors").exists()
    has_sharded = (MODEL_DIR / "model.safetensors.index.json").exists()
    if not has_single and not has_sharded:
        print(f"ERROR: No model files found in {MODEL_DIR}. Download first.")
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ================================================================
    # PHASE A: CPU Perplexity Comparison
    # ================================================================
    print("\n[Phase A] CPU Perplexity Comparison")
    print("-" * 70)

    # A1: Load model in FP32 (handles sharded safetensors transparently)
    print("[A1] Loading Qwen2.5-Coder-3B (FP32)...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model.eval()

    # A2: Compress to persistent CDNA v3 dir (from loaded model weights)
    print(f"\n[A2] Compressing {N_BLOCKS} blocks × {len(BLOCK_TENSOR_TYPES)} types "
          f"= {N_BLOCKS * len(BLOCK_TENSOR_TYPES)} tensors...")
    compress_t0 = time.time()
    n_tensors, n_skipped, total_dense, total_compressed, per_tensor_cosines = \
        compress_all_blocks(model, CDNA_DIR, N_BLOCKS)
    compress_time = time.time() - compress_t0
    print(f"  Compression: {compress_time:.0f}s ({n_skipped} cached, "
          f"{n_tensors - n_skipped} new)")

    # A3: Baseline PPL BEFORE swapping
    print("\n[A3] Loading WikiText-2...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:EVAL_TOKENS]
    print(f"  Tokens: {len(eval_tokens)}")

    print("[A4] Computing baseline perplexity (CPU, FP32 dense)...")
    t_baseline = time.time()
    ppl_baseline, nll_baseline, n_eval = compute_perplexity_cpu_offload(
        model, eval_tokens, device="cpu")
    baseline_time = time.time() - t_baseline
    print(f"  Baseline PPL: {ppl_baseline:.4f} (NLL: {nll_baseline:.6f}, {baseline_time:.1f}s)")

    # Load HelixLinear modules from persistent CDNA v3
    print("\n[A5] Loading HelixLinear modules from CDNA v3...")
    helix_modules = {}
    for block_idx in range(N_BLOCKS):
        for tensor_type in BLOCK_TENSOR_TYPES:
            hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
            safe_name = hf_name.replace("/", "_").replace(".", "_")
            tensor_dir = CDNA_DIR / f"{safe_name}.cdnav3"
            if not tensor_dir.exists():
                continue
            module_path = hf_name.replace(".weight", "")
            mod = get_module(model, block_idx, tensor_type)
            bias = mod.bias.data.clone() if mod.bias is not None else None
            helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
            helix_modules[module_path] = helix_mod

    print(f"  Swapping {len(helix_modules)} modules to HelixLinear...")
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
    ppl_helix, nll_helix, _ = compute_perplexity_cpu_offload(
        model, eval_tokens, device="cpu")
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
    print(f"  Tensors swapped: {summary['helix_modules']}")

    # ================================================================
    # PHASE B: GPU Fit Test (Zero-Copy)
    # ================================================================
    gpu_results = None
    if torch.cuda.is_available():
        print(f"\n[Phase B] GPU Fit Test (Zero-Copy)")
        print("-" * 70)

        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"  GPU: {gpu_name} ({gpu_mem:.0f} MB)")

        # Free CPU model memory before GPU load
        del model
        gc.collect()

        try:
            # Reload model fresh for GPU with zero-copy
            print("[B1] Loading model with zero-copy HelixLinear indices...")
            from helix_substrate.zerocopy import pin_indices_from_file

            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR), torch_dtype=torch.float32
            )
            model.eval()

            # Reload HelixLinear modules
            helix_modules = {}
            for block_idx in range(N_BLOCKS):
                for tensor_type in BLOCK_TENSOR_TYPES:
                    hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                    safe_name = hf_name.replace("/", "_").replace(".", "_")
                    tensor_dir = CDNA_DIR / f"{safe_name}.cdnav3"
                    if not tensor_dir.exists():
                        continue
                    module_path = hf_name.replace(".weight", "")
                    mod = get_module(model, block_idx, tensor_type)
                    bias = mod.bias.data.clone() if mod.bias is not None else None
                    helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
                    helix_modules[module_path] = helix_mod

            model = swap_to_helix(model, helix_modules)

            # Move to GPU
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = model.cuda()
            model.eval()

            vram_after_load = torch.cuda.memory_allocated() / 1e6

            # Replace indices with zero-copy host-pinned versions
            print("[B2] Pinning HelixLinear indices to host memory...")
            pinned_buffers = []
            n_pinned = 0
            for name, module in model.named_modules():
                if not isinstance(module, HelixLinear):
                    continue
                tensor_name = name + ".weight"
                safe_name = tensor_name.replace("/", "_").replace(".", "_")
                tensor_dir = CDNA_DIR / f"{safe_name}.cdnav3"
                indices_path = tensor_dir / "indices.bin"
                if not indices_path.exists():
                    continue
                shape = tuple(module.indices.shape)
                zc_indices, pinned = pin_indices_from_file(indices_path, shape)
                pinned_buffers.append(pinned)
                module.indices = zc_indices
                n_pinned += 1

            gc.collect()
            torch.cuda.empty_cache()

            vram_zc = torch.cuda.memory_allocated() / 1e6
            pinned_mb = sum(b.nbytes for b in pinned_buffers) / 1024 / 1024
            print(f"  {n_pinned} indices pinned to host ({pinned_mb:.0f} MB)")
            print(f"  VRAM after ZC: {vram_zc:.0f} MB ({vram_zc / gpu_mem * 100:.0f}%)")
            print(f"  VRAM freed: {vram_after_load - vram_zc:.0f} MB (indices moved to host)")

            # Forward pass to check peak
            print("[B3] Forward pass (peak VRAM)...")
            test_ids = torch.tensor(eval_tokens[:128], dtype=torch.long).unsqueeze(0).cuda()
            with torch.no_grad():
                _ = model(test_ids)
            vram_peak = torch.cuda.max_memory_allocated() / 1e6
            print(f"  VRAM peak: {vram_peak:.0f} MB ({vram_peak / gpu_mem * 100:.0f}%)")
            print(f"  Headroom: {gpu_mem - vram_peak:.0f} MB")
            del test_ids
            torch.cuda.empty_cache()

            # GPU PPL with CPU-offloaded loss
            print("[B4] GPU perplexity (CPU-offloaded loss)...")
            t_gpu = time.time()
            ppl_gpu, nll_gpu, _ = compute_perplexity_cpu_offload(
                model, eval_tokens, device="cuda")
            gpu_ppl_time = time.time() - t_gpu
            ppl_match = abs(ppl_gpu - ppl_helix) < 0.5  # wider tolerance for offloaded loss
            print(f"  GPU PPL: {ppl_gpu:.4f} (matches CPU: {ppl_match}, {gpu_ppl_time:.1f}s)")

            gpu_results = {
                "gpu_name": gpu_name,
                "gpu_mem_mb": round(gpu_mem),
                "vram_after_load_mb": round(vram_after_load),
                "vram_zc_mb": round(vram_zc),
                "vram_peak_mb": round(vram_peak),
                "vram_headroom_mb": round(gpu_mem - vram_peak),
                "vram_utilization_pct": round(vram_peak / gpu_mem * 100, 1),
                "n_pinned": n_pinned,
                "pinned_host_mb": round(pinned_mb),
                "gpu_ppl": round(ppl_gpu, 6),
                "gpu_ppl_matches_cpu": ppl_match,
                "gpu_ppl_time_s": round(gpu_ppl_time, 1),
                "fit": True,
                "mode": "zero_copy",
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  GPU OOM: {e}")
                gpu_results = {"fit": False, "error": str(e), "mode": "zero_copy"}
                torch.cuda.empty_cache()
                # Reload on CPU for Phase C
                model = AutoModelForCausalLM.from_pretrained(
                    str(MODEL_DIR), torch_dtype=torch.float32
                )
                model.eval()
                helix_modules = {}
                for block_idx in range(N_BLOCKS):
                    for tensor_type in BLOCK_TENSOR_TYPES:
                        hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                        safe_name = hf_name.replace("/", "_").replace(".", "_")
                        tensor_dir = CDNA_DIR / f"{safe_name}.cdnav3"
                        if not tensor_dir.exists():
                            continue
                        module_path = hf_name.replace(".weight", "")
                        mod = get_module(model, block_idx, tensor_type)
                        bias = mod.bias.data.clone() if mod.bias is not None else None
                        helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
                        helix_modules[module_path] = helix_mod
                model = swap_to_helix(model, helix_modules)
            else:
                raise

    # ================================================================
    # PHASE C: Coding Qualitative Prompts
    # ================================================================
    print(f"\n[Phase C] Coding Qualitative Prompts")
    print("-" * 70)

    device = "cuda" if gpu_results and gpu_results.get("fit") else "cpu"
    if device == "cpu" and not hasattr(model, 'device'):
        # Model still exists from Phase A or OOM fallback
        pass

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
        "schema": "qwen3b_e2e_proof_v1",
        "model": "Qwen2.5-Coder-3B-Instruct",
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
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
            "tensors_swapped": summary["helix_modules"],
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
    ppl_ok = abs(ppl_pct) < 3.0  # wider threshold for larger model (proven: 0.78-1.73%)
    gpu_ok = gpu_results and gpu_results.get("fit", False) if gpu_results else False
    prompts_ok = all(p["tokens_generated"] > 10 for p in prompt_results)

    if ppl_ok and gpu_ok and prompts_ok:
        receipt["verdict"] = "PASS"
    elif ppl_ok and not gpu_ok:
        receipt["verdict"] = "PARTIAL — PPL ok, GPU failed"
    elif not ppl_ok:
        receipt["verdict"] = f"FAIL — PPL delta {ppl_pct:+.2f}% exceeds 3%"
    else:
        receipt["verdict"] = "PARTIAL"

    receipt_path = RECEIPT_DIR / f"qwen3b_e2e_{time.strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))

    # Manifest for persistent CDNA dir
    manifest = {
        "model": "Qwen2.5-Coder-3B-Instruct",
        "n_blocks": N_BLOCKS,
        "n_tensors": n_tensors,
        "compression_ratio": summary["overall_ratio"],
        "ppl_baseline": round(ppl_baseline, 4),
        "ppl_helix": round(ppl_helix, 4),
        "ppl_delta_pct": round(ppl_pct, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (CDNA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 70}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Model:           Qwen2.5-Coder-3B-Instruct")
    print(f"  Baseline PPL:    {ppl_baseline:.4f}")
    print(f"  HelixLinear PPL: {ppl_helix:.4f}")
    print(f"  Delta:           {ppl_delta:+.4f} ({ppl_pct:+.2f}%)")
    print(f"  Compression:     {summary['overall_ratio']}x")
    print(f"  Tensors:         {summary['helix_modules']} swapped, "
          f"{summary['linear_modules']} remaining")
    if gpu_results and gpu_results.get("fit"):
        print(f"  VRAM (ZC):       {gpu_results['vram_zc_mb']} MB")
        print(f"  VRAM peak:       {gpu_results['vram_peak_mb']} MB")
        print(f"  Host pinned:     {gpu_results['pinned_host_mb']} MB")
        print(f"  GPU speed:       {prompt_results[0]['tok_s'] if prompt_results else '?'} tok/s")
    elif gpu_results:
        print(f"  GPU:             OOM (zero-copy failed)")
    print(f"  Verdict:         {receipt['verdict']}")
    print(f"  Receipt:         {receipt_path}")
    print(f"  CDNA dir:        {CDNA_DIR}")
    print(f"  Wall time:       {receipt['cost']['wall_time_s']:.0f}s")
    print(f"{'=' * 70}")

    # Cleanup pinned buffers if they exist
    if gpu_results and gpu_results.get("fit"):
        for buf in pinned_buffers:
            buf.unregister()


if __name__ == "__main__":
    main()
