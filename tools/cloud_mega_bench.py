#!/usr/bin/env python3
"""
Cloud Mega-Benchmark — 5-test suite for RTX 4090.

Runs sequentially on a single GPU instance:
  Test 1: Dense vs Compressed Qwen 7B head-to-head (PPL + speed)
  Test 2: WO-WIDE-R4-01 at 7B scale (VQ-only / kurtosis-routed / wide-r4)
  Test 3: Dual-model load test (Qwen 7B + Qwen 3B simultaneously)
  Test 4: Mamba-130m GPU tok/s (vs 2.36 tok/s CPU baseline)
  Test 5: PPL comparison matrix (summary of all results)

Emits WO-RECEIPT-COST-01 compliant receipts per test + summary.

Usage:
    python3 tools/cloud_mega_bench.py \\
        --model-dir /workspace/models/qwen2.5-7b-instruct \\
        --model-dir-3b /workspace/models/qwen2.5-3b-instruct \\
        --mamba-dir /workspace/models/mamba-130m-hf \\
        --output-dir /workspace/receipts
"""

import argparse
import gc
import json
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Project path ──
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from helix_substrate.helix_linear import (
    HelixLinear,
    load_cdna_factors,
    load_helix_linear_from_cdnav3,
    swap_to_helix,
    swap_summary,
)
from helix_substrate.tensor_policy import (
    TensorPolicy,
    classify_tensor,
    get_default_policy,
    get_policy,
)
from helix_substrate.cdnav3_writer import CDNAv3Writer


# ── Utilities ──

def jsonable(obj):
    """JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def get_gpu_info():
    """Collect GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024),
        "capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
        "driver": torch.version.cuda,
    }


def make_cost_block(t_start, cpu_start, start_iso):
    """Build a WO-RECEIPT-COST-01 cost block."""
    return {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
    }


def write_receipt(receipt, output_dir, filename):
    """Write a JSON receipt to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2, default=jsonable)
    print(f"  Receipt written: {path}")
    return path


def gpu_cleanup():
    """Free GPU memory between tests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_vram_mb():
    """Current VRAM allocated in MB."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
    return 0


def get_vram_peak_mb():
    """Peak VRAM allocated in MB."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    return 0


# ── Model loading (shared) ──

def load_model_compressed(model_dir: Path, device="cuda"):
    """Load HF model shell + swap in HelixLinear from CDNA v3."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print("  Loading model shell...")
    t0 = time.time()
    has_weights = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "model.safetensors.index.json").exists()
        or (model_dir / "pytorch_model.bin").exists()
    )
    if has_weights:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to_empty(device="cpu")
        model = model.to(torch.float32)
    shell_time = time.time() - t0
    print(f"  Shell loaded: {shell_time:.1f}s (weights={'found' if has_weights else 'empty/CDNA-only'})")

    cdna_dir = model_dir / "cdnav3"
    print(f"  Loading CDNA v3 factors from {cdna_dir}...")
    t0 = time.time()
    factors = load_cdna_factors(cdna_dir, model=model)
    load_time = time.time() - t0
    print(f"  Factors loaded: {len(factors)} tensors, {load_time:.1f}s")

    print("  Swapping to HelixLinear...")
    t0 = time.time()
    model = swap_to_helix(model, factors)
    swap_time = time.time() - t0

    summary = swap_summary(model)
    print(f"  Swap complete: {summary['helix_modules']} HelixLinear, "
          f"{summary['linear_modules']} nn.Linear remaining, {swap_time:.1f}s")

    # Cast remaining non-HelixLinear params to float32
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()

    print(f"  Moving to {device}...")
    t0 = time.time()
    model = model.to(device).eval()
    move_time = time.time() - t0

    vram_mb = get_vram_mb()
    print(f"  On GPU: {vram_mb:.0f} MB VRAM, {move_time:.1f}s")

    return model, tokenizer, {
        "shell_time_s": round(shell_time, 2),
        "factor_load_time_s": round(load_time, 2),
        "swap_time_s": round(swap_time, 2),
        "move_time_s": round(move_time, 2),
        "helix_modules": summary["helix_modules"],
        "linear_modules": summary["linear_modules"],
        "vram_mb": round(vram_mb, 1),
        "compressed_mb": round(summary.get("compressed_bytes", 0) / 1024 / 1024, 1),
        "dense_equivalent_mb": round(summary.get("dense_equivalent_bytes", 0) / 1024 / 1024, 1),
    }


def load_model_dense(model_dir: Path, device="cuda", dtype=torch.bfloat16):
    """Load dense HF model for baseline comparison."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading dense baseline ({dtype})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device).eval()
    load_time = time.time() - t0

    vram_mb = get_vram_mb()
    print(f"  Dense loaded: {vram_mb:.0f} MB VRAM, {load_time:.1f}s")
    return model, tokenizer, {"vram_mb": round(vram_mb, 1), "load_time_s": round(load_time, 2), "dtype": str(dtype)}


# ── Perplexity evaluation (shared) ──

def eval_perplexity(model, tokenizer, device="cuda", n_tokens=8192, seq_len=2048):
    """Compute perplexity on WikiText-2 test set."""
    from datasets import load_dataset

    print("  Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens + 1)
    input_ids = encodings.input_ids[:, :n_tokens + 1].to(device)

    nlls = []
    n_evaluated = 0

    print(f"  Evaluating PPL (seq_len={seq_len}, n_tokens={n_tokens})...")
    with torch.no_grad():
        for i in range(0, input_ids.shape[1] - 1, seq_len):
            end = min(i + seq_len + 1, input_ids.shape[1])
            chunk = input_ids[:, i:end]
            if chunk.shape[1] < 2:
                break

            outputs = model(input_ids=chunk[:, :-1])
            logits = outputs.logits

            # Compute loss on CPU to avoid OOM with large vocab
            shift_logits = logits.float().cpu()
            shift_labels = chunk[:, 1:].cpu()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            n_tok = shift_labels.numel()
            nlls.append(loss.item() * n_tok)
            n_evaluated += n_tok

            if n_evaluated >= n_tokens:
                break

    ppl = np.exp(sum(nlls) / n_evaluated)
    print(f"  PPL = {ppl:.4f} ({n_evaluated} tokens)")
    return {"ppl": round(float(ppl), 4), "n_tokens": n_evaluated, "seq_len": seq_len}


# ── Decode speed benchmark (shared) ──

def bench_decode_speed(model, tokenizer, device="cuda", n_warmup=2, n_runs=5, max_tokens=128):
    """Benchmark decode speed. Returns tok/s and timing details."""
    prompt = "def fibonacci(n):\n    "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=4, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    gen_tokens = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_gen = out.shape[1] - prompt_len
        times.append(elapsed)
        gen_tokens.append(n_gen)

    avg_time = float(np.mean(times))
    avg_tokens = float(np.mean(gen_tokens))
    tok_s = avg_tokens / avg_time if avg_time > 0 else 0

    return {
        "prompt_tokens": prompt_len,
        "avg_gen_tokens": round(avg_tokens, 1),
        "avg_time_s": round(avg_time, 3),
        "tok_s": round(tok_s, 1),
        "n_runs": n_runs,
    }


# ═══════════════════════════════════════════════════════════════════
# Test 1: Dense vs Compressed Qwen 7B Head-to-Head
# ═══════════════════════════════════════════════════════════════════

def test1_dense_vs_compressed(model_dir: Path, output_dir: Path, device="cuda",
                               ppl_tokens=8192):
    """Dense bfloat16 vs HelixLinear compressed on same hardware, same dataset."""
    print("\n" + "=" * 70)
    print("[1/5] Dense vs Compressed Qwen 7B Head-to-Head")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    receipt = {
        "test": "test1_dense_vs_compressed",
        "model_dir": str(model_dir),
        "gpu": get_gpu_info(),
    }

    # Phase A: Compressed model
    print("\n  --- Compressed (HelixLinear) ---")
    gpu_cleanup()
    model, tokenizer, load_info = load_model_compressed(model_dir, device=device)
    receipt["compressed_load"] = load_info

    # Smoke test
    test_ids = tokenizer("Hello", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(test_ids)
    assert not torch.isnan(out.logits).any(), "NaN in compressed output"
    print("  Smoke test passed.")

    # PPL
    ppl_compressed = eval_perplexity(model, tokenizer, device=device, n_tokens=ppl_tokens)
    receipt["compressed_ppl"] = ppl_compressed

    # Speed
    speed_compressed = bench_decode_speed(model, tokenizer, device=device)
    receipt["compressed_speed"] = speed_compressed
    print(f"  Compressed decode: {speed_compressed['tok_s']} tok/s")

    receipt["compressed_vram_peak_mb"] = get_vram_peak_mb()

    # Free compressed
    del model
    gpu_cleanup()

    # Phase B: Dense baseline
    print("\n  --- Dense (bfloat16) ---")
    dense_model, _, dense_load = load_model_dense(model_dir, device=device)
    receipt["dense_load"] = dense_load

    ppl_dense = eval_perplexity(dense_model, tokenizer, device=device, n_tokens=ppl_tokens)
    receipt["dense_ppl"] = ppl_dense

    speed_dense = bench_decode_speed(dense_model, tokenizer, device=device)
    receipt["dense_speed"] = speed_dense
    print(f"  Dense decode: {speed_dense['tok_s']} tok/s")

    receipt["dense_vram_peak_mb"] = get_vram_peak_mb()

    del dense_model
    gpu_cleanup()

    # Deltas
    ppl_delta = (ppl_compressed["ppl"] - ppl_dense["ppl"]) / ppl_dense["ppl"] * 100
    receipt["ppl_delta_pct"] = round(ppl_delta, 4)
    speed_ratio = speed_compressed["tok_s"] / max(speed_dense["tok_s"], 0.01)
    receipt["speed_ratio"] = round(speed_ratio, 3)

    print(f"\n  PPL: compressed {ppl_compressed['ppl']:.4f} vs dense {ppl_dense['ppl']:.4f} "
          f"(delta {ppl_delta:+.2f}%)")
    print(f"  Speed: compressed {speed_compressed['tok_s']} vs dense {speed_dense['tok_s']} tok/s "
          f"(ratio {speed_ratio:.2f}x)")

    receipt["cost"] = make_cost_block(t_start, cpu_start, start_iso)

    ts = time.strftime("%Y%m%dT%H%M%S")
    write_receipt(receipt, output_dir / "cloud_bench",
                  f"test1_dense_vs_compressed_{ts}.json")
    return receipt


# ═══════════════════════════════════════════════════════════════════
# Test 2: WO-WIDE-R4-01 at 7B Scale
# ═══════════════════════════════════════════════════════════════════

N_BLOCKS_7B = 28

BLOCK_TENSOR_TYPES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

HF_PATTERNS_7B = {
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}


@dataclass
class StrategyResult:
    name: str
    description: str
    svd_tensors: int = 0
    svd_bytes: int = 0
    total_bytes: int = 0
    dense_bytes: int = 0
    mean_cosine: float = 0.0
    ppl: float = 0.0
    compression_time_s: float = 0.0


def _get_module_7b(model, block_idx, tensor_type):
    """Get nn.Linear module for a tensor in a Qwen block."""
    layer = model.model.layers[block_idx]
    if tensor_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, tensor_type)
    return getattr(layer.mlp, tensor_type)


def _compress_strategy_7b(model_dir, model_template, tokenizer, tmpdir,
                          strategy_name, kurtosis_map, make_policy_fn,
                          eval_tokens, device="cuda"):
    """
    Compress Qwen 7B with a given policy, load as HelixLinear, measure PPL.

    Memory-efficient: loads tensor data from sharded safetensors on-the-fly,
    avoids deepcopy by reloading model shell for each strategy.
    """
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open

    print(f"    Loading fresh model shell for strategy '{strategy_name}'...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    cdna_dir = tmpdir / strategy_name
    writer = CDNAv3Writer(cdna_dir)

    t0 = time.time()
    helix_modules = {}
    total_svd_bytes = 0
    total_compressed_bytes = 0
    total_dense_bytes = 0
    svd_tensor_count = 0
    cosines = []

    # Build shard -> tensor name mapping from index
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        # Single safetensors file
        weight_map = None

    # Group tensors by shard file to minimize file open/close
    shard_tensors = {}
    for block_idx in range(N_BLOCKS_7B):
        for tensor_type in BLOCK_TENSOR_TYPES:
            hf_name = HF_PATTERNS_7B[tensor_type].format(i=block_idx)
            if weight_map is not None:
                shard = weight_map.get(hf_name)
                if shard is None:
                    print(f"    WARNING: {hf_name} not in weight map, skipping")
                    continue
            else:
                shard = "model.safetensors"
            shard_tensors.setdefault(shard, []).append(
                (block_idx, tensor_type, hf_name)
            )

    for shard_file, tensor_list in shard_tensors.items():
        shard_path = model_dir / shard_file
        with safe_open(str(shard_path), framework="numpy") as sf:
            for block_idx, tensor_type, hf_name in tensor_list:
                tensor_np = sf.get_tensor(hf_name).astype(np.float32)
                kurt = kurtosis_map.get(hf_name, 0.0)

                policy = make_policy_fn(hf_name, tensor_np.shape, block_idx, kurt)
                stats = writer.write_tensor(tensor_np, hf_name, policy=policy)

                svd_b = stats.get("svd_bytes", 0)
                total_svd_bytes += svd_b
                if svd_b > 0:
                    svd_tensor_count += 1

                total_compressed_bytes += stats.get("compressed_bytes", 0)
                total_dense_bytes += stats.get("original_bytes", 0)
                cosines.append(stats.get("cosine_with_svd",
                                        stats.get("cosine_with_sidecar", 0)))

                # Build HelixLinear module
                safe_name = hf_name.replace("/", "_").replace(".", "_")
                tensor_dir = cdna_dir / f"{safe_name}.cdnav3"
                module_path = hf_name.replace(".weight", "")
                mod = _get_module_7b(model, block_idx, tensor_type)
                bias = mod.bias.data.clone() if mod.bias is not None else None
                helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
                helix_modules[module_path] = helix_mod

    compression_time = time.time() - t0

    # Swap
    model = swap_to_helix(model, helix_modules)

    # Cast non-HelixLinear to float32
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()

    model = model.to(device).eval()

    # Evaluate PPL
    ppl_result = eval_perplexity(model, tokenizer, device=device, n_tokens=4096)

    del model
    gpu_cleanup()

    # Clean up temp CDNA dir
    shutil.rmtree(cdna_dir, ignore_errors=True)

    strat = StrategyResult(
        name=strategy_name,
        description="",
        svd_tensors=svd_tensor_count,
        svd_bytes=total_svd_bytes,
        total_bytes=total_compressed_bytes,
        dense_bytes=total_dense_bytes,
        mean_cosine=float(np.mean(cosines)) if cosines else 0.0,
        ppl=ppl_result["ppl"],
        compression_time_s=round(compression_time, 2),
    )

    return strat


def test2_wide_r4_7b(model_dir: Path, output_dir: Path, device="cuda"):
    """WO-WIDE-R4-01 at 7B scale: three compression strategies compared."""
    from scipy.stats import kurtosis as scipy_kurtosis
    from safetensors import safe_open
    from transformers import AutoTokenizer

    print("\n" + "=" * 70)
    print("[2/5] WO-WIDE-R4-01 at 7B Scale")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    receipt = {
        "test": "test2_wide_r4_7b",
        "work_order": "WO-WIDE-R4-01",
        "model": "Qwen2.5-7B-Instruct",
        "n_blocks": N_BLOCKS_7B,
        "model_dir": str(model_dir),
    }

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    # Load eval tokens
    print("  Loading WikiText-2 eval tokens...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:4096]
    print(f"  Eval tokens: {len(eval_tokens)}")

    # Precompute kurtosis for all block tensors
    print("  Computing kurtosis for all tensors...")
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        weight_map = None

    kurtosis_map = {}

    # Group by shard
    shard_tensors = {}
    for block_idx in range(N_BLOCKS_7B):
        for tensor_type in BLOCK_TENSOR_TYPES:
            hf_name = HF_PATTERNS_7B[tensor_type].format(i=block_idx)
            if weight_map is not None:
                shard = weight_map.get(hf_name, "model.safetensors")
            else:
                shard = "model.safetensors"
            shard_tensors.setdefault(shard, []).append(hf_name)

    for shard_file, names in shard_tensors.items():
        shard_path = model_dir / shard_file
        with safe_open(str(shard_path), framework="numpy") as sf:
            for hf_name in names:
                tensor_np = sf.get_tensor(hf_name).astype(np.float32)
                kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))
                kurtosis_map[hf_name] = kurt

    n_high = sum(1 for k in kurtosis_map.values() if k > 5)
    print(f"  {len(kurtosis_map)} tensors, {n_high} with kurtosis > 5")
    receipt["n_tensors"] = len(kurtosis_map)
    receipt["n_kurtosis_above_5"] = n_high

    # Dense baseline PPL (on GPU)
    print("  Computing dense baseline PPL...")
    from transformers import AutoModelForCausalLM
    dense_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device).eval()

    ppl_baseline_result = eval_perplexity(dense_model, tokenizer, device=device, n_tokens=4096)
    ppl_baseline = ppl_baseline_result["ppl"]
    receipt["baseline_ppl"] = ppl_baseline
    del dense_model
    gpu_cleanup()

    tmpdir = Path(tempfile.mkdtemp(prefix="wide_r4_7b_"))

    # Strategy A: VQ-only
    print("\n  Strategy A: VQ-only (no SVD)...")

    def policy_vq_only(name, shape, block_idx, kurt):
        tc = classify_tensor(name, shape=shape)
        return get_default_policy(tc)

    strat_a = _compress_strategy_7b(
        model_dir, None, tokenizer, tmpdir,
        "vq_only", kurtosis_map, policy_vq_only, eval_tokens, device=device,
    )
    strat_a.description = "VQ + sidecar, no SVD"
    print(f"    PPL: {strat_a.ppl:.4f}, SVD tensors: 0, SVD bytes: 0")

    # Strategy B: Kurtosis-routed
    print("  Strategy B: Kurtosis-routed SVD rank-8...")

    def policy_routed(name, shape, block_idx, kurt):
        return get_policy(name, shape, block_idx=block_idx,
                          kurtosis=kurt, n_blocks=N_BLOCKS_7B)

    strat_b = _compress_strategy_7b(
        model_dir, None, tokenizer, tmpdir,
        "kurtosis_routed", kurtosis_map, policy_routed, eval_tokens, device=device,
    )
    strat_b.description = "VQ + sidecar + SVD rank-8 on kurtosis>5 tensors"
    print(f"    PPL: {strat_b.ppl:.4f}, SVD tensors: {strat_b.svd_tensors}, "
          f"SVD bytes: {strat_b.svd_bytes:,}")

    # Strategy C: Wide r4
    print("  Strategy C: Wide SVD rank-4 on ALL tensors...")

    def policy_wide_r4(name, shape, block_idx, kurt):
        from dataclasses import replace
        tc = classify_tensor(name, shape=shape)
        base = get_default_policy(tc)
        if len(shape) == 2 and base.storage_mode not in ("exact", "morpho"):
            return replace(base, svd_residual_rank=4)
        return base

    strat_c = _compress_strategy_7b(
        model_dir, None, tokenizer, tmpdir,
        "wide_r4", kurtosis_map, policy_wide_r4, eval_tokens, device=device,
    )
    strat_c.description = "VQ + sidecar + SVD rank-4 on ALL 2D tensors"
    print(f"    PPL: {strat_c.ppl:.4f}, SVD tensors: {strat_c.svd_tensors}, "
          f"SVD bytes: {strat_c.svd_bytes:,}")

    # Clean up tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Results table
    strategies = [strat_a, strat_b, strat_c]
    print(f"\n  {'Strategy':<40} {'PPL':>8} {'dPPL%':>8} {'SVD#':>6} {'SVD MB':>8} {'Ratio':>6}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*6}")
    for s in strategies:
        delta = (s.ppl - ppl_baseline) / ppl_baseline * 100
        ratio = s.dense_bytes / max(1, s.total_bytes)
        print(f"  {s.description:<40} {s.ppl:>8.4f} {delta:>+7.2f}% {s.svd_tensors:>6} "
              f"{s.svd_bytes / 1024 / 1024:>8.2f} {ratio:>6.2f}x")

    # Verdict
    b_wins = strat_b.ppl < strat_c.ppl
    b_cheaper = strat_b.svd_bytes < strat_c.svd_bytes
    if b_wins and b_cheaper:
        verdict = "ROUTING_WINS_DECISIVE"
    elif b_wins:
        verdict = "ROUTING_WINS_QUALITY"
    elif abs(strat_b.ppl - strat_c.ppl) < 0.05:
        verdict = "TIE"
    else:
        verdict = "WIDE_R4_WINS"

    receipt["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    receipt["strategies"] = []
    for s in strategies:
        delta = (s.ppl - ppl_baseline) / ppl_baseline * 100
        receipt["strategies"].append({
            "name": s.name,
            "description": s.description,
            "ppl": round(s.ppl, 6),
            "ppl_delta_pct": round(delta, 4),
            "svd_tensors": s.svd_tensors,
            "svd_bytes": s.svd_bytes,
            "total_compressed_bytes": s.total_bytes,
            "dense_equivalent_bytes": s.dense_bytes,
            "compression_ratio": round(s.dense_bytes / max(1, s.total_bytes), 2),
            "mean_cosine": round(s.mean_cosine, 6),
            "compression_time_s": s.compression_time_s,
        })

    receipt["cost"] = make_cost_block(t_start, cpu_start, start_iso)

    ts = time.strftime("%Y%m%dT%H%M%S")
    write_receipt(receipt, output_dir / "wide_r4",
                  f"wide_r4_7b_{ts}.json")
    return receipt


# ═══════════════════════════════════════════════════════════════════
# Test 3: Dual-Model Load Test
# ═══════════════════════════════════════════════════════════════════

def test3_dual_model_load(model_dir_7b: Path, model_dir_3b: Path,
                          output_dir: Path, device="cuda"):
    """Load compressed Qwen 7B + Qwen 3B simultaneously, prove both fit."""
    print("\n" + "=" * 70)
    print("[3/5] Dual-Model Load Test (Qwen 7B + Qwen 3B)")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    receipt = {
        "test": "test3_dual_model_load",
        "gpu": get_gpu_info(),
        "model_7b": str(model_dir_7b),
        "model_3b": str(model_dir_3b),
    }

    prompt = "Write a Python function that checks if a number is prime:\n"

    # Load Qwen 7B compressed
    print("\n  Loading Qwen 7B compressed...")
    gpu_cleanup()
    model_7b, tok_7b, load_7b = load_model_compressed(model_dir_7b, device=device)
    vram_after_7b = get_vram_mb()
    receipt["load_7b"] = load_7b
    receipt["vram_after_7b_mb"] = vram_after_7b
    print(f"  VRAM after 7B: {vram_after_7b:.0f} MB")

    # Load Qwen 3B compressed (on same GPU)
    print("\n  Loading Qwen 3B compressed...")
    model_3b, tok_3b, load_3b = load_model_compressed(model_dir_3b, device=device)
    vram_after_both = get_vram_mb()
    receipt["load_3b"] = load_3b
    receipt["vram_after_both_mb"] = vram_after_both
    print(f"  VRAM after both: {vram_after_both:.0f} MB")

    gpu_total = get_gpu_info().get("total_mb", 0)
    receipt["vram_headroom_mb"] = round(gpu_total - vram_after_both, 1) if gpu_total else 0

    # Generate from Qwen 7B
    print("\n  Generating 64 tokens from Qwen 7B...")
    input_ids_7b = tok_7b(prompt, return_tensors="pt").input_ids.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_7b = model_7b.generate(input_ids_7b, max_new_tokens=64, do_sample=False, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_time_7b = time.perf_counter() - t0
    gen_tokens_7b = out_7b.shape[1] - input_ids_7b.shape[1]
    tok_s_7b = gen_tokens_7b / gen_time_7b if gen_time_7b > 0 else 0
    text_7b = tok_7b.decode(out_7b[0, input_ids_7b.shape[1]:], skip_special_tokens=True)

    receipt["gen_7b"] = {
        "tokens": gen_tokens_7b,
        "time_s": round(gen_time_7b, 3),
        "tok_s": round(tok_s_7b, 1),
        "output_preview": text_7b[:300],
    }
    print(f"  7B: {gen_tokens_7b} tokens, {tok_s_7b:.1f} tok/s")

    # Generate from Qwen 3B
    print("  Generating 64 tokens from Qwen 3B...")
    input_ids_3b = tok_3b(prompt, return_tensors="pt").input_ids.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_3b = model_3b.generate(input_ids_3b, max_new_tokens=64, do_sample=False, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_time_3b = time.perf_counter() - t0
    gen_tokens_3b = out_3b.shape[1] - input_ids_3b.shape[1]
    tok_s_3b = gen_tokens_3b / gen_time_3b if gen_time_3b > 0 else 0
    text_3b = tok_3b.decode(out_3b[0, input_ids_3b.shape[1]:], skip_special_tokens=True)

    receipt["gen_3b"] = {
        "tokens": gen_tokens_3b,
        "time_s": round(gen_time_3b, 3),
        "tok_s": round(tok_s_3b, 1),
        "output_preview": text_3b[:300],
    }
    print(f"  3B: {gen_tokens_3b} tokens, {tok_s_3b:.1f} tok/s")

    receipt["vram_peak_mb"] = get_vram_peak_mb()
    print(f"\n  Combined VRAM peak: {receipt['vram_peak_mb']:.0f} MB")
    print(f"  Both models fit: {'YES' if receipt['vram_peak_mb'] < gpu_total else 'NO'}")

    receipt["both_fit"] = receipt["vram_peak_mb"] < gpu_total

    del model_7b, model_3b
    gpu_cleanup()

    receipt["cost"] = make_cost_block(t_start, cpu_start, start_iso)

    ts = time.strftime("%Y%m%dT%H%M%S")
    write_receipt(receipt, output_dir / "dual_model_load",
                  f"dual_model_load_{ts}.json")
    return receipt


# ═══════════════════════════════════════════════════════════════════
# Test 4: Mamba-130m GPU tok/s
# ═══════════════════════════════════════════════════════════════════

def test4_mamba_gpu(mamba_dir: Path, output_dir: Path, device="cuda"):
    """Load compressed Mamba-130m on GPU, benchmark decode speed."""
    print("\n" + "=" * 70)
    print("[4/5] Mamba-130m GPU tok/s")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    receipt = {
        "test": "test4_mamba_gpu",
        "model": "state-spaces/mamba-130m-hf",
        "model_dir": str(mamba_dir),
        "gpu": get_gpu_info(),
        "cpu_baseline_tok_s": 2.36,  # From existing receipt
    }

    gpu_cleanup()

    # Load Mamba compressed
    print("  Loading Mamba-130m compressed...")
    model, tokenizer, load_info = load_model_compressed(mamba_dir, device=device)
    receipt["load_info"] = load_info

    # Smoke test
    test_ids = tokenizer("The future of", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(test_ids)
    assert not torch.isnan(out.logits).any(), "NaN in Mamba output"
    print("  Smoke test passed.")

    # PPL
    print("  Evaluating PPL...")
    ppl_result = eval_perplexity(model, tokenizer, device=device, n_tokens=4096)
    receipt["ppl"] = ppl_result

    # Decode benchmark
    print("  Benchmarking decode speed (128 tokens, 5 runs)...")
    prompt = "The future of artificial intelligence"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=4, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    run_results = []
    for run_i in range(5):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=128, do_sample=False, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_gen = out.shape[1] - prompt_len
        tok_s = n_gen / elapsed if elapsed > 0 else 0
        run_results.append({"tokens": n_gen, "time_s": round(elapsed, 3),
                           "tok_s": round(tok_s, 1)})
        print(f"    Run {run_i+1}: {tok_s:.1f} tok/s ({n_gen} tokens in {elapsed:.2f}s)")

    avg_tok_s = float(np.mean([r["tok_s"] for r in run_results]))
    speedup = avg_tok_s / 2.36  # vs CPU baseline

    receipt["decode_runs"] = run_results
    receipt["avg_tok_s"] = round(avg_tok_s, 1)
    receipt["speedup_vs_cpu"] = round(speedup, 2)
    receipt["vram_peak_mb"] = get_vram_peak_mb()

    # Capture last generation text
    gen_text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    receipt["generation_preview"] = gen_text[:300]

    print(f"\n  Avg GPU: {avg_tok_s:.1f} tok/s vs CPU: 2.36 tok/s ({speedup:.1f}x speedup)")
    print(f"  VRAM peak: {receipt['vram_peak_mb']:.0f} MB")

    del model
    gpu_cleanup()

    receipt["cost"] = make_cost_block(t_start, cpu_start, start_iso)

    ts = time.strftime("%Y%m%dT%H%M%S")
    write_receipt(receipt, output_dir / "mamba_gpu",
                  f"mamba_gpu_{ts}.json")
    return receipt


# ═══════════════════════════════════════════════════════════════════
# Test 5: PPL Comparison Matrix (Summary)
# ═══════════════════════════════════════════════════════════════════

def test5_summary(results: dict, output_dir: Path):
    """Summarize all PPL results into a comparison table."""
    print("\n" + "=" * 70)
    print("[5/5] PPL Comparison Matrix")
    print("=" * 70)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    rows = []

    # Test 1: Dense vs Compressed Qwen 7B
    if "test1" in results and results["test1"] is not None:
        r = results["test1"]
        if "compressed_ppl" in r and "dense_ppl" in r:
            rows.append({
                "model": "Qwen2.5-7B-Instruct",
                "variant": "Dense bfloat16",
                "ppl": r["dense_ppl"]["ppl"],
                "source": "Test 1",
            })
            rows.append({
                "model": "Qwen2.5-7B-Instruct",
                "variant": "HelixLinear compressed",
                "ppl": r["compressed_ppl"]["ppl"],
                "delta_pct": r.get("ppl_delta_pct"),
                "source": "Test 1",
            })

    # Test 2: Wide R4 strategies
    if "test2" in results and results["test2"] is not None:
        r = results["test2"]
        if "baseline_ppl" in r:
            rows.append({
                "model": "Qwen2.5-7B-Instruct",
                "variant": "Dense baseline (wide_r4 eval)",
                "ppl": r["baseline_ppl"],
                "source": "Test 2",
            })
        for s in r.get("strategies", []):
            rows.append({
                "model": "Qwen2.5-7B-Instruct",
                "variant": s["description"],
                "ppl": s["ppl"],
                "delta_pct": s["ppl_delta_pct"],
                "compression_ratio": s.get("compression_ratio"),
                "source": "Test 2",
            })

    # Test 4: Mamba
    if "test4" in results and results["test4"] is not None:
        r = results["test4"]
        if "ppl" in r:
            rows.append({
                "model": "Mamba-130m",
                "variant": "HelixLinear GPU",
                "ppl": r["ppl"]["ppl"],
                "gpu_tok_s": r.get("avg_tok_s"),
                "cpu_tok_s": r.get("cpu_baseline_tok_s"),
                "speedup": r.get("speedup_vs_cpu"),
                "source": "Test 4",
            })

    # Print table
    print(f"\n  {'Model':<30} {'Variant':<45} {'PPL':>8} {'dPPL%':>8}")
    print(f"  {'-'*30} {'-'*45} {'-'*8} {'-'*8}")
    for row in rows:
        delta_str = f"{row.get('delta_pct', 0):+.2f}%" if row.get("delta_pct") is not None else "baseline"
        print(f"  {row['model']:<30} {row['variant']:<45} {row['ppl']:>8.4f} {delta_str:>8}")

    # Speed summary
    print(f"\n  Speed Summary:")
    if "test1" in results and results["test1"] is not None:
        r = results["test1"]
        cs = r.get("compressed_speed", {})
        ds = r.get("dense_speed", {})
        if cs and ds:
            print(f"    Qwen 7B compressed: {cs.get('tok_s', '?')} tok/s")
            print(f"    Qwen 7B dense:      {ds.get('tok_s', '?')} tok/s")
    if "test3" in results and results["test3"] is not None:
        r = results["test3"]
        print(f"    Dual-model VRAM:    {r.get('vram_after_both_mb', '?')} MB "
              f"(peak {r.get('vram_peak_mb', '?')} MB)")
        print(f"    Qwen 7B (dual):     {r.get('gen_7b', {}).get('tok_s', '?')} tok/s")
        print(f"    Qwen 3B (dual):     {r.get('gen_3b', {}).get('tok_s', '?')} tok/s")
    if "test4" in results and results["test4"] is not None:
        r = results["test4"]
        print(f"    Mamba GPU:          {r.get('avg_tok_s', '?')} tok/s "
              f"({r.get('speedup_vs_cpu', '?')}x vs CPU)")

    receipt = {
        "test": "test5_summary",
        "description": "PPL comparison matrix from all mega-bench tests",
        "rows": rows,
        "gpu": get_gpu_info(),
        "tests_completed": [k for k, v in results.items() if v is not None],
        "tests_failed": [k for k, v in results.items() if v is None],
        "cost": make_cost_block(t_start, cpu_start, start_iso),
    }

    ts = time.strftime("%Y%m%dT%H%M%S")
    write_receipt(receipt, output_dir, f"mega_bench_summary_{ts}.json")
    return receipt


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cloud Mega-Benchmark: 5-test suite for RTX 4090"
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to Qwen 7B with dense weights and cdnav3/")
    parser.add_argument("--model-dir-3b", type=str, required=True,
                        help="Path to Qwen 3B with dense weights and cdnav3/")
    parser.add_argument("--mamba-dir", type=str, required=True,
                        help="Path to Mamba-130m with cdnav3/")
    parser.add_argument("--output-dir", type=str, default="receipts",
                        help="Base output directory for all receipts")
    parser.add_argument("--skip-test", type=int, action="append", default=[],
                        help="Skip specific test numbers (1-5), can repeat")
    parser.add_argument("--ppl-tokens", type=int, default=8192,
                        help="Number of tokens for PPL eval (Test 1)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir_3b = Path(args.model_dir_3b)
    mamba_dir = Path(args.mamba_dir)
    output_dir = Path(args.output_dir)
    skip_tests = set(args.skip_test)

    mega_start = time.time()
    mega_cpu_start = time.process_time()
    mega_start_iso = datetime.now(timezone.utc).isoformat()

    gpu_info = get_gpu_info()

    print("=" * 70)
    print("  HELIX CLOUD MEGA-BENCHMARK")
    print("=" * 70)
    print(f"  GPU:       {gpu_info.get('name', 'N/A')} ({gpu_info.get('total_mb', '?')} MB)")
    print(f"  Qwen 7B:   {model_dir}")
    print(f"  Qwen 3B:   {model_dir_3b}")
    print(f"  Mamba:     {mamba_dir}")
    print(f"  Output:    {output_dir}")
    print(f"  Skip:      {skip_tests or 'none'}")
    print(f"  Torch:     {torch.__version__}")
    print(f"  CUDA:      {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print("=" * 70)

    # Validate dirs exist
    for label, d in [("model-dir", model_dir), ("model-dir-3b", model_dir_3b),
                     ("mamba-dir", mamba_dir)]:
        if not d.exists():
            print(f"ERROR: {label} does not exist: {d}")
            sys.exit(1)
        cdna = d / "cdnav3"
        if not cdna.exists():
            print(f"ERROR: {label}/cdnav3 not found: {cdna}")
            sys.exit(1)

    results = {}
    errors = {}

    # ── Test 1 ──
    if 1 not in skip_tests:
        try:
            results["test1"] = test1_dense_vs_compressed(
                model_dir, output_dir, device=args.device,
                ppl_tokens=args.ppl_tokens)
        except Exception as e:
            print(f"\n  ERROR in Test 1: {e}")
            traceback.print_exc()
            results["test1"] = None
            errors["test1"] = str(e)
    else:
        print("\n  [1/5] SKIPPED")
        results["test1"] = None

    # ── Test 2 ──
    if 2 not in skip_tests:
        try:
            results["test2"] = test2_wide_r4_7b(
                model_dir, output_dir, device=args.device)
        except Exception as e:
            print(f"\n  ERROR in Test 2: {e}")
            traceback.print_exc()
            results["test2"] = None
            errors["test2"] = str(e)
    else:
        print("\n  [2/5] SKIPPED")
        results["test2"] = None

    # ── Test 3 ──
    if 3 not in skip_tests:
        try:
            results["test3"] = test3_dual_model_load(
                model_dir, model_dir_3b, output_dir, device=args.device)
        except Exception as e:
            print(f"\n  ERROR in Test 3: {e}")
            traceback.print_exc()
            results["test3"] = None
            errors["test3"] = str(e)
    else:
        print("\n  [3/5] SKIPPED")
        results["test3"] = None

    # ── Test 4 ──
    if 4 not in skip_tests:
        try:
            results["test4"] = test4_mamba_gpu(
                mamba_dir, output_dir, device=args.device)
        except Exception as e:
            print(f"\n  ERROR in Test 4: {e}")
            traceback.print_exc()
            results["test4"] = None
            errors["test4"] = str(e)
    else:
        print("\n  [4/5] SKIPPED")
        results["test4"] = None

    # ── Test 5: Summary ──
    if 5 not in skip_tests:
        try:
            results["test5"] = test5_summary(results, output_dir)
        except Exception as e:
            print(f"\n  ERROR in Test 5: {e}")
            traceback.print_exc()
            results["test5"] = None
            errors["test5"] = str(e)
    else:
        print("\n  [5/5] SKIPPED")
        results["test5"] = None

    # ── Final Summary ──
    total_time = time.time() - mega_start

    print("\n" + "=" * 70)
    print("  MEGA-BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"  Total wall time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Tests run:       {sum(1 for v in results.values() if v is not None)}/5")
    if errors:
        print(f"  Errors:          {list(errors.keys())}")
    print(f"  Receipts in:     {output_dir}")

    # Write master receipt
    master = {
        "benchmark": "cloud_mega_bench",
        "gpu": gpu_info,
        "system": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "tests_completed": [k for k, v in results.items() if v is not None],
        "tests_failed": list(errors.keys()),
        "errors": errors,
        "cost": make_cost_block(mega_start, mega_cpu_start, mega_start_iso),
    }
    ts = time.strftime("%Y%m%dT%H%M%S")
    write_receipt(master, output_dir, f"mega_bench_master_{ts}.json")


if __name__ == "__main__":
    main()
