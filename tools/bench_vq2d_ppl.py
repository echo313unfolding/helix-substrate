#!/usr/bin/env python3
"""
WO-MULTIDIM-VQ Phase 1c: PPL comparison — scalar k=256 vs 2D VQ k=4096.

Head-to-head on any model: same eval corpus, same everything except the codec.
Proves whether the cosine improvement translates to perplexity improvement.

Architecture-agnostic: works on Transformer, Mamba, or any HF model.

Usage:
    python3 tools/bench_vq2d_ppl.py ~/models/tinyllama-dense
    python3 tools/bench_vq2d_ppl.py ~/models/mamba-130m-hf-dense
    python3 tools/bench_vq2d_ppl.py ~/models/tinyllama-dense --tokens 8192
"""

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
from dataclasses import replace as dc_replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.cdnav3_reader import CDNAv3Reader
from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.tensor_policy import classify_tensor, get_policy, TensorClass

RECEIPT_DIR = Path(__file__).resolve().parent.parent / "receipts" / "vq2d_ppl"

# Skip patterns for tensors that should remain exact
SKIP_PATTERNS = ["layernorm", "layer_norm", "norm", "bias", "embed_tokens", "lm_head",
                 "embedding"]


def should_compress(name, shape):
    """Should this tensor be VQ-compressed? Same logic as compress.py."""
    if len(shape) != 2:
        return False
    if shape[0] * shape[1] < 256:
        return False
    name_lower = name.lower()
    return not any(p in name_lower for p in SKIP_PATTERNS)


def compute_perplexity(model, eval_tokens, seq_len=2048):
    model.eval()
    nlls = []
    n_tokens = 0
    for i in range(0, len(eval_tokens) - 1, seq_len):
        end = min(i + seq_len, len(eval_tokens))
        input_ids = torch.tensor(eval_tokens[i:end], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        chunk_tokens = input_ids.shape[1] - 1
        nlls.append(outputs.loss.item() * chunk_tokens)
        n_tokens += chunk_tokens
        if end >= len(eval_tokens):
            break
    mean_nll = sum(nlls) / n_tokens
    return float(np.exp(mean_nll)), mean_nll, n_tokens


def compress_tensor(W, name, vector_dim=1, n_clusters=256):
    """Compress one tensor. Returns (W_hat, info_dict)."""
    from scipy.stats import kurtosis as scipy_kurtosis
    kurt = float(scipy_kurtosis(W.ravel(), fisher=True))
    policy = get_policy(name, W.shape, kurtosis=kurt)
    policy = dc_replace(policy, n_clusters=n_clusters)
    if vector_dim > 1 and len(W.shape) == 2 and W.shape[1] % vector_dim == 0:
        policy = dc_replace(policy, vector_dim=vector_dim)

    # Use a sanitized name for the cdna directory
    safe_name = name.replace(".", "_").replace("/", "_")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        writer = CDNAv3Writer(tmpdir)
        stats = writer.write_tensor(W, safe_name, policy=policy)

        tensor_dir = tmpdir / f"{safe_name}.cdnav3"

        if stats.get("storage_mode") == "exact":
            W_hat = W.copy()
        else:
            reader = CDNAv3Reader(tensor_dir)
            W_hat = reader.reconstruct()

    cos = float(np.dot(W.ravel(), W_hat.ravel()) / (
        np.linalg.norm(W.ravel()) * np.linalg.norm(W_hat.ravel()) + 1e-30
    ))

    return W_hat, {
        "name": name,
        "shape": list(W.shape),
        "storage_mode": stats.get("storage_mode", "codebook"),
        "compressed_bytes": stats.get("compressed_bytes", W.nbytes),
        "original_bytes": int(W.nbytes),
        "weight_cosine": round(cos, 6),
        "vector_dim": vector_dim,
        "n_clusters": n_clusters,
    }


def find_param_module(model, param_name):
    """Given 'model.layers.0.self_attn.q_proj.weight', return the module and attr."""
    parts = param_name.split(".")
    attr_name = parts[-1]  # 'weight' or 'bias'
    module_path = ".".join(parts[:-1])
    try:
        module = model.get_submodule(module_path)
        return module, attr_name
    except (AttributeError, torch.nn.modules.module.ModuleNotFoundError):
        return None, None


def compress_and_eval(model, original_weights, sf_paths, eval_tokens,
                       vector_dim, n_clusters, label):
    """Compress, replace weights, eval PPL. sf_paths is a list of shard paths."""
    from safetensors import safe_open

    print(f"\n{'=' * 80}", flush=True)
    print(f"  {label}: vector_dim={vector_dim}, n_clusters={n_clusters}", flush=True)
    print(f"{'=' * 80}", flush=True)

    t0 = time.time()
    all_stats = []
    total_orig = 0
    total_comp = 0
    n_compressed = 0
    n_exact = 0

    # Restore original weights
    print("  Restoring original weights...", flush=True)
    for name, orig_data in original_weights.items():
        module, attr = find_param_module(model, name)
        if module is not None:
            param = getattr(module, attr, None)
            if param is not None and isinstance(param, (torch.nn.Parameter, torch.Tensor)):
                with torch.no_grad():
                    param.data.copy_(orig_data)

    # Compress all eligible tensors across all shards
    # Use torch framework to handle BF16 tensors, then convert to numpy
    print("  Compressing...", flush=True)
    global_idx = 0
    total_keys = 0
    # Count total keys across shards for progress
    for sp in sf_paths:
        with safe_open(str(sp), framework="pt") as f:
            total_keys += len(f.keys())
    for sp in sf_paths:
      with safe_open(str(sp), framework="pt") as f:
        all_keys = sorted(f.keys())
        for key in all_keys:
            W_t = f.get_tensor(key)
            W = W_t.float().numpy()  # handle BF16/FP16 → FP32 → numpy

            if should_compress(key, W.shape):
                W_hat, info = compress_tensor(W, key, vector_dim=vector_dim,
                                               n_clusters=n_clusters)
                # Replace in model
                module, attr = find_param_module(model, key)
                if module is not None:
                    with torch.no_grad():
                        getattr(module, attr).data = torch.from_numpy(W_hat).float()

                total_orig += info["original_bytes"]
                total_comp += info["compressed_bytes"]
                all_stats.append(info)
                n_compressed += 1
            else:
                total_orig += W.nbytes
                total_comp += W.nbytes  # exact
                n_exact += 1

            global_idx += 1
            if global_idx % 30 == 0 or global_idx == total_keys:
                elapsed = time.time() - t0
                print(f"    {global_idx}/{total_keys} tensors ({n_compressed} compressed, "
                      f"{n_exact} exact, {elapsed:.0f}s)", flush=True)

    compress_time = time.time() - t0
    cos_vals = [s["weight_cosine"] for s in all_stats if s["storage_mode"] != "exact"]
    ratio = total_orig / total_comp if total_comp > 0 else 0

    print(f"\n  Compression: {ratio:.2f}x ({total_orig/1e9:.3f} GB -> {total_comp/1e9:.3f} GB)", flush=True)
    if cos_vals:
        print(f"  Cosine: min={min(cos_vals):.6f}  mean={np.mean(cos_vals):.6f}  "
              f"({n_compressed} tensors, {compress_time:.0f}s)", flush=True)

    # Eval PPL
    print(f"  Evaluating perplexity...", flush=True)
    t1 = time.time()
    ppl, nll, n_tok = compute_perplexity(model, eval_tokens)
    eval_time = time.time() - t1
    print(f"  PPL: {ppl:.4f}  NLL: {nll:.6f}  ({eval_time:.0f}s)", flush=True)

    return ppl, nll, {
        "label": label,
        "vector_dim": vector_dim,
        "n_clusters": n_clusters,
        "ppl": round(ppl, 4),
        "nll": round(nll, 6),
        "compression_ratio": round(ratio, 2),
        "total_original_bytes": total_orig,
        "total_compressed_bytes": total_comp,
        "n_compressed": n_compressed,
        "n_exact": n_exact,
        "cos_mean": round(float(np.mean(cos_vals)), 6) if cos_vals else None,
        "cos_min": round(float(min(cos_vals)), 6) if cos_vals else None,
        "compress_time_s": round(compress_time, 1),
        "eval_time_s": round(eval_time, 1),
        "tensor_stats": all_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="PPL comparison: scalar k=256 vs 2D VQ k=4096")
    parser.add_argument("model_dir", type=Path, help="Path to dense model directory")
    parser.add_argument("--tokens", type=int, default=4096,
                        help="Number of eval tokens (default: 4096)")
    args = parser.parse_args()

    model_dir = args.model_dir
    sf_path = model_dir / "model.safetensors"
    if sf_path.exists():
        sf_paths = [sf_path]
    else:
        sf_paths = sorted(model_dir.glob("model-*.safetensors"))
        if not sf_paths:
            print(f"ERROR: no model*.safetensors found in {model_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"  Found {len(sf_paths)} safetensor shards", flush=True)

    t_wall_start = time.time()
    t_cpu_start = time.process_time()
    ts_start = datetime.now(timezone.utc).isoformat()

    model_name = model_dir.name

    print("=" * 80, flush=True)
    print(f"WO-MULTIDIM-VQ Phase 1c: PPL Comparison on {model_name}", flush=True)
    print(f"  Scalar k=256 (production) vs 2D VQ k=4096 (new)", flush=True)
    print("=" * 80, flush=True)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load tokenizer + eval data
    print("Loading tokenizer and WikiText-2...", flush=True)
    # Map known model types to their canonical tokenizer
    TOKENIZER_MAP = {
        "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mamba": "EleutherAI/gpt-neox-20b",
        "mamba2": "EleutherAI/gpt-neox-20b",
        "qwen2": "Qwen/Qwen2.5-1.5B",
    }
    cfg = json.loads((model_dir / "config.json").read_text())
    model_type = cfg.get("model_type", "unknown")
    tok_source = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    except (ValueError, OSError, TypeError):
        tok_source = cfg.get("_name_or_path") or TOKENIZER_MAP.get(model_type, str(model_dir))
        print(f"  Local tokenizer failed, loading from: {tok_source}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    all_tokens = tokenizer.encode(text)
    eval_tokens = all_tokens[:args.tokens]
    print(f"  Eval tokens: {len(eval_tokens)}", flush=True)

    # Load model
    print(f"Loading {model_name}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    model.eval()

    # Save original weights
    print("Saving original weights...", flush=True)
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()

    # Baseline PPL
    print("\nBaseline (dense) perplexity...", flush=True)
    t0 = time.time()
    baseline_ppl, baseline_nll, n_tok = compute_perplexity(model, eval_tokens)
    print(f"  Baseline: ppl={baseline_ppl:.4f}  nll={baseline_nll:.6f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Run A: Scalar k=256 (production)
    ppl_scalar, nll_scalar, stats_scalar = compress_and_eval(
        model, original_weights, sf_paths, eval_tokens,
        vector_dim=1, n_clusters=256, label="Scalar k=256"
    )

    # Run B: 2D VQ k=4096
    ppl_2d, nll_2d, stats_2d = compress_and_eval(
        model, original_weights, sf_paths, eval_tokens,
        vector_dim=2, n_clusters=4096, label="2D VQ k=4096"
    )

    # Summary
    delta_scalar = ppl_scalar - baseline_ppl
    delta_2d = ppl_2d - baseline_ppl
    pct_scalar = (delta_scalar / baseline_ppl) * 100
    pct_2d = (delta_2d / baseline_ppl) * 100

    print(f"\n{'=' * 80}", flush=True)
    print(f"  RESULTS — {model_name}", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"  {'Config':<25} {'PPL':>10} {'Delta':>10} {'%':>8} {'Cos mean':>10} {'Cos min':>10} {'Ratio':>8}", flush=True)
    print(f"  {'-'*83}", flush=True)
    print(f"  {'Baseline (dense)':<25} {baseline_ppl:>10.4f}", flush=True)
    sc = stats_scalar
    print(f"  {'Scalar k=256':<25} {ppl_scalar:>10.4f} {delta_scalar:>+10.4f} {pct_scalar:>+7.3f}% "
          f"{sc['cos_mean']:>10.6f} {sc['cos_min']:>10.6f} {sc['compression_ratio']:>7.2f}x", flush=True)
    vq = stats_2d
    print(f"  {'2D VQ k=4096':<25} {ppl_2d:>10.4f} {delta_2d:>+10.4f} {pct_2d:>+7.3f}% "
          f"{vq['cos_mean']:>10.6f} {vq['cos_min']:>10.6f} {vq['compression_ratio']:>7.2f}x", flush=True)

    # Gate
    if ppl_2d <= ppl_scalar + 0.001:
        gate = "PASS"
        detail = f"2D VQ PPL {ppl_2d:.4f} <= scalar PPL {ppl_scalar:.4f}"
    else:
        gate = "FAIL"
        detail = f"2D VQ PPL {ppl_2d:.4f} > scalar PPL {ppl_scalar:.4f}"

    print(f"\n  GATE: 2D VQ k=4096 PPL <= scalar k=256 PPL", flush=True)
    print(f"  Result: {gate} — {detail}", flush=True)

    # Cost
    cost = {
        "wall_time_s": round(time.time() - t_wall_start, 3),
        "cpu_time_s": round(time.process_time() - t_cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": ts_start,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
    }

    receipt = {
        "work_order": "WO-MULTIDIM-VQ-01c-PPL",
        "question": "Does 2D VQ k=4096 cosine advantage translate to PPL advantage?",
        "model": model_name,
        "model_dir": str(model_dir),
        "gate": f"2D VQ PPL <= scalar PPL: {gate}",
        "verdict": gate,
        "baseline_ppl": round(baseline_ppl, 4),
        "baseline_nll": round(baseline_nll, 6),
        "eval_tokens": len(eval_tokens),
        "configs": {
            "scalar_k256": {k: v for k, v in stats_scalar.items() if k != "tensor_stats"},
            "vq2d_k4096": {k: v for k, v in stats_2d.items() if k != "tensor_stats"},
        },
        "scalar_tensor_stats": stats_scalar["tensor_stats"],
        "vq2d_tensor_stats": stats_2d["tensor_stats"],
        "cost": cost,
    }

    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    receipt_path = RECEIPT_DIR / f"vq2d_ppl_{safe_model}_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}", flush=True)
    print(f"Cost: {cost['wall_time_s']:.1f}s wall, {cost['peak_memory_mb']:.0f}MB peak", flush=True)


if __name__ == "__main__":
    main()
