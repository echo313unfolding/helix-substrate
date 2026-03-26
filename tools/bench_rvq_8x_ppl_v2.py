#!/usr/bin/env python3
"""
WO-RVQ-8X-01 Phase 1b v2: PPL Evaluation WITH Outlier Sidecar

v1 FAILED because it omitted the production outlier sidecar correction.
The sidecar stores exact FP32 values for the top-percentile outliers.
Without it, even k=256 VQ gives PPL 274 (vs production 6.22).

This v2 adds the sidecar correction to match production fidelity.

Strategies tested:
  1. Baseline (FP32, no compression)
  2. 8-bit VQ (k=256) + sidecar — production control
  3. 4-bit VQ (k=16) + sidecar — true 8x
  4. 4-bit RVQ (k=16+16) + sidecar — 4x with better quality
  5. Mixed-rate (kurt>5 → 8-bit, else → 4-bit flat) + sidecar — blended ~7x
  6. Mixed-rate (kurt>5 → 8-bit, else → RVQ) + sidecar — blended ~4x premium

Output: receipts/rvq_8x_ppl/
"""

import copy
import gc
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR = Path.home() / "models" / "tinyllama_fp32"
EVAL_TOKENS = 4096
SEQ_LEN = 2048
N_BLOCKS = 22
KURTOSIS_THRESHOLD = 5.0
OUTLIER_PERCENTILE = 99.9  # Match production default

TENSOR_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
HF_PATTERNS = {
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}


def compute_perplexity(model, eval_tokens, seq_len=SEQ_LEN):
    """Compute perplexity on token sequence."""
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
    mean_nll = sum(nlls) / n_tokens
    return float(np.exp(mean_nll)), mean_nll, n_tokens


def kurtosis_1d(data):
    """Fisher excess kurtosis (matches scipy)."""
    mu = data.mean()
    var = ((data - mu) ** 2).mean()
    if var < 1e-30:
        return 0.0
    m4 = ((data - mu) ** 4).mean()
    return float(m4 / (var ** 2) - 3.0)


def kmeans_1d(data, k=16, max_iters=10, seed=42):
    """1D k-means matching production _simple_kmeans."""
    n_unique = len(np.unique(data[:min(500_000, len(data))]))
    k = min(k, n_unique)

    # Subsample for k-means training (production uses up to 1M)
    rng = np.random.RandomState(seed)
    n_sample = min(1_000_000, len(data))
    if n_sample < len(data):
        sample = data[rng.choice(len(data), n_sample, replace=False)]
    else:
        sample = data

    centroids = np.percentile(sample, np.linspace(0, 100, k)).astype(np.float32)

    for _ in range(max_iters):
        # Chunked assignment to avoid OOM
        assignments = np.empty(len(sample), dtype=np.int32)
        chunk = 2_000_000
        for s in range(0, len(sample), chunk):
            e = min(s + chunk, len(sample))
            dists = np.abs(sample[s:e, None] - centroids)
            assignments[s:e] = np.argmin(dists, axis=1)

        new_centroids = np.empty_like(centroids)
        for i in range(k):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = sample[mask].mean()
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(centroids, new_centroids, atol=1e-7):
            break
        centroids = new_centroids

    return centroids


def assign_all(flat, codebook):
    """Assign all elements to nearest centroid, chunked."""
    indices = np.empty(len(flat), dtype=np.int32)
    chunk = 2_000_000
    for s in range(0, len(flat), chunk):
        e = min(s + chunk, len(flat))
        dists = np.abs(flat[s:e, None] - codebook)
        indices[s:e] = np.argmin(dists, axis=1)
    return indices


def find_outliers(flat, percentile=OUTLIER_PERCENTILE):
    """Find outlier positions and values (matching production sidecar)."""
    abs_vals = np.abs(flat)
    threshold = np.percentile(abs_vals, percentile)
    mask = abs_vals > threshold
    positions = np.where(mask)[0].astype(np.int32)
    values = flat[positions].astype(np.float32)
    return positions, values


def apply_sidecar(recon_flat, positions, values):
    """Patch outlier positions with exact values."""
    patched = recon_flat.copy()
    patched[positions] = values
    return patched


def encode_8bit(weight_np):
    """8-bit VQ (k=256) + outlier sidecar. Returns reconstructed weight."""
    flat = weight_np.ravel().astype(np.float32)
    codebook = kmeans_1d(flat, k=256)
    indices = assign_all(flat, codebook)
    recon = codebook[indices]

    # Outlier sidecar
    positions, values = find_outliers(flat)
    recon = apply_sidecar(recon, positions, values)

    return recon.reshape(weight_np.shape), len(positions)


def encode_4bit(weight_np):
    """4-bit VQ (k=16) + outlier sidecar. Returns reconstructed weight."""
    flat = weight_np.ravel().astype(np.float32)
    codebook = kmeans_1d(flat, k=16)
    indices = assign_all(flat, codebook)
    recon = codebook[indices]

    # Outlier sidecar
    positions, values = find_outliers(flat)
    recon = apply_sidecar(recon, positions, values)

    return recon.reshape(weight_np.shape), len(positions)


def encode_rvq(weight_np):
    """Residual VQ (k=16+16) + outlier sidecar. Returns reconstructed weight."""
    flat = weight_np.ravel().astype(np.float32)

    # Stage 1: coarse
    cb1 = kmeans_1d(flat, k=16, seed=42)
    idx1 = assign_all(flat, cb1)
    coarse_recon = cb1[idx1]

    # Stage 2: residual
    residual = flat - coarse_recon
    cb2 = kmeans_1d(residual, k=16, seed=43)
    idx2 = assign_all(residual, cb2)

    recon = coarse_recon + cb2[idx2]

    # Outlier sidecar
    positions, values = find_outliers(flat)
    recon = apply_sidecar(recon, positions, values)

    return recon.reshape(weight_np.shape), len(positions)


def get_module(model, block_idx, tensor_type):
    """Get nn.Linear module for a given block and tensor type."""
    layer = model.model.layers[block_idx]
    if tensor_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, tensor_type)
    return getattr(layer.mlp, tensor_type)


def set_module_weight(model, block_idx, tensor_type, new_weight_np):
    """Replace a module's weight with reconstructed weight."""
    mod = get_module(model, block_idx, tensor_type)
    new_w = torch.from_numpy(new_weight_np.astype(np.float32))
    mod.weight.data.copy_(new_w)


def run_strategy(model_base, sf, eval_tokens, name, encode_fn, per_tensor_kurtosis=None,
                 mixed_mode=False):
    """Encode all tensors with a strategy, swap weights, measure PPL."""
    print(f"\n  [{name}] Encoding 154 tensors...")
    model = copy.deepcopy(model_base)
    cosines = []
    n_8bit = 0
    n_4bit = 0
    total_outliers = 0

    for block_idx in range(N_BLOCKS):
        for tensor_type in TENSOR_TYPES:
            hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
            weight_np = sf.get_tensor(hf_name).astype(np.float32)

            if mixed_mode and per_tensor_kurtosis is not None:
                kurt = per_tensor_kurtosis.get(hf_name, 0.0)
                if kurt > KURTOSIS_THRESHOLD:
                    recon, n_out = encode_8bit(weight_np)
                    n_8bit += 1
                else:
                    recon, n_out = encode_fn(weight_np)
                    n_4bit += 1
            else:
                recon, n_out = encode_fn(weight_np)
                n_4bit += 1

            total_outliers += n_out

            # Cosine
            orig_flat = weight_np.ravel()
            recon_flat = recon.ravel()
            cos = float(np.dot(orig_flat, recon_flat) /
                        (np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat) + 1e-30))
            cosines.append(cos)

            set_module_weight(model, block_idx, tensor_type, recon)
            del weight_np, recon

        if (block_idx + 1) % 11 == 0 or block_idx == N_BLOCKS - 1:
            print(f"    Block {block_idx + 1}/{N_BLOCKS}", flush=True)

    print(f"  [{name}] Computing PPL... (outliers patched: {total_outliers})")
    ppl, nll, n_tok = compute_perplexity(model, eval_tokens)

    result = {
        "mean_cosine": round(float(np.mean(cosines)), 6),
        "min_cosine": round(float(min(cosines)), 6),
        "ppl": round(ppl, 6),
        "nll": round(nll, 6),
        "n_tokens": n_tok,
        "total_outliers_patched": total_outliers,
    }

    if mixed_mode:
        result["n_8bit"] = n_8bit
        result["n_4bit"] = n_4bit
        total_weights = N_BLOCKS * len(TENSOR_TYPES)
        result["blended_ratio"] = round(
            1.0 / ((n_8bit / total_weights) * (1/4) + (n_4bit / total_weights) * (1/8)), 2)

    print(f"  [{name}] PPL: {ppl:.4f}, mean cos: {np.mean(cosines):.6f}, "
          f"min cos: {min(cosines):.6f}")

    del model
    gc.collect()
    return result


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 70)
    print("  WO-RVQ-8X-01 Phase 1b v2: PPL Evaluation (WITH Sidecar)")
    print("  4-bit VQ / RVQ / Mixed-Rate on TinyLlama")
    print("  FIX: outlier sidecar correction (percentile={:.1f})".format(OUTLIER_PERCENTILE))
    print("=" * 70)

    assert MODEL_DIR.exists(), f"Model not found: {MODEL_DIR}"

    # Load tokenizer + eval data
    print("\n  Loading tokenizer + WikiText-2...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors import safe_open

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:EVAL_TOKENS]
    print(f"  Eval tokens: {len(eval_tokens)}")

    # Load model
    print("  Loading model (FP32)...")
    model_base = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model_base.eval()

    # Baseline PPL
    print("  Computing baseline PPL...")
    ppl_base, nll_base, n_tok = compute_perplexity(model_base, eval_tokens)
    print(f"  Baseline PPL: {ppl_base:.4f}")

    # Pre-compute kurtosis for mixed-rate routing
    print("  Computing per-tensor kurtosis...")
    sf = safe_open(str(MODEL_DIR / "model.safetensors"), framework="numpy")
    per_tensor_kurt = {}
    for block_idx in range(N_BLOCKS):
        for tensor_type in TENSOR_TYPES:
            hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
            w = sf.get_tensor(hf_name).astype(np.float32)
            per_tensor_kurt[hf_name] = kurtosis_1d(w.ravel())
            del w

    n_high_kurt = sum(1 for k in per_tensor_kurt.values() if k > KURTOSIS_THRESHOLD)
    print(f"  High-kurtosis tensors (>{KURTOSIS_THRESHOLD}): {n_high_kurt}/154")
    print(f"  Sidecar percentile: {OUTLIER_PERCENTILE}")

    # Strategy 1: 8-bit VQ (k=256) + sidecar — production control
    r_8bit = run_strategy(model_base, sf, eval_tokens, "8-bit VQ (k=256)+sidecar", encode_8bit)

    # Strategy 2: 4-bit VQ (k=16) + sidecar — true 8x
    r_4bit = run_strategy(model_base, sf, eval_tokens, "4-bit VQ (k=16)+sidecar", encode_4bit)

    # Strategy 3: RVQ (k=16+16) + sidecar — 4x with better quality
    r_rvq = run_strategy(model_base, sf, eval_tokens, "RVQ (k=16+16)+sidecar", encode_rvq)

    # Strategy 4: Mixed-rate (kurt>5 → 8-bit, else → 4-bit flat) + sidecar
    r_mixed = run_strategy(model_base, sf, eval_tokens, "Mixed (kurt>5→8bit, else→4bit)+sc",
                           encode_4bit, per_tensor_kurtosis=per_tensor_kurt, mixed_mode=True)

    # Strategy 5: Mixed-rate with RVQ rescue (kurt>5 → 8-bit, else → RVQ) + sidecar
    r_mixed_rvq = run_strategy(model_base, sf, eval_tokens,
                               "Mixed (kurt>5→8bit, else→RVQ)+sc",
                               encode_rvq, per_tensor_kurtosis=per_tensor_kurt,
                               mixed_mode=True)

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    # Results table
    print(f"\n{'=' * 70}")
    print("  RESULTS — WO-RVQ-8X-01 PPL v2 (WITH Outlier Sidecar)")
    print(f"{'=' * 70}")
    print(f"  Baseline PPL: {ppl_base:.4f}")
    print(f"  Sidecar: percentile={OUTLIER_PERCENTILE} (top {100-OUTLIER_PERCENTILE:.1f}% values exact)")
    print()

    strategies = [
        ("8-bit VQ (k=256)+sc", r_8bit, "4.0x"),
        ("4-bit VQ (k=16)+sc", r_4bit, "~8x"),
        ("RVQ (k=16+16)+sc", r_rvq, "4.0x"),
        ("Mixed 4bit/8bit+sc", r_mixed, f"{r_mixed.get('blended_ratio', '?')}x"),
        ("Mixed RVQ/8bit+sc", r_mixed_rvq, f"{r_mixed_rvq.get('blended_ratio', '?')}x"),
    ]

    print(f"  {'Strategy':<25} {'PPL':>8} {'Δ%':>8} {'Mean cos':>10} {'Min cos':>10} {'Ratio':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*7}")
    for name, r, ratio in strategies:
        delta_pct = 100 * (r["ppl"] - ppl_base) / ppl_base
        print(f"  {name:<25} {r['ppl']:>8.4f} {delta_pct:>+7.2f}% "
              f"{r['mean_cosine']:>10.6f} {r['min_cosine']:>10.6f} {ratio:>7}")

    # Verdict
    print()
    for name, r, ratio in strategies:
        delta_pct = 100 * (r["ppl"] - ppl_base) / ppl_base
        verdict = "PASS" if abs(delta_pct) < 2.0 else "FAIL"
        print(f"  {name:<25} → {verdict} ({delta_pct:+.2f}%)")

    print(f"\n  Wall: {wall:.0f}s, CPU: {cpu:.0f}s")

    # Receipt
    receipt_dir = Path(__file__).parent.parent / "receipts" / "rvq_8x_ppl"
    receipt_dir.mkdir(parents=True, exist_ok=True)

    receipt = {
        "work_order": "WO-RVQ-8X-01",
        "phase": "1b_ppl_eval_v2",
        "question": "Do 4-bit VQ strategies maintain PPL within 2% on TinyLlama (with outlier sidecar)?",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "eval_dataset": "wikitext-2-raw-v1 (validation)",
        "eval_tokens": len(eval_tokens),
        "sidecar_percentile": OUTLIER_PERCENTILE,
        "v1_bug": "v1 omitted production outlier sidecar, causing 8-bit VQ PPL 274 (vs production 6.22)",
        "baseline": {
            "ppl": round(ppl_base, 6),
            "nll": round(nll_base, 6),
        },
        "strategies": {
            "vq8_k256_sidecar": r_8bit,
            "vq4_k16_sidecar": r_4bit,
            "rvq4_k16_16_sidecar": r_rvq,
            "mixed_4bit_8bit_sidecar": r_mixed,
            "mixed_rvq_8bit_sidecar": r_mixed_rvq,
        },
        "kurtosis_threshold": KURTOSIS_THRESHOLD,
        "n_high_kurtosis": n_high_kurt,
        "n_total_tensors": 154,
        "verdicts": {},
        "cost": {
            "wall_time_s": round(wall, 3),
            "cpu_time_s": round(cpu, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    for name, r, ratio in strategies:
        key = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("+", "_")
        delta_pct = 100 * (r["ppl"] - ppl_base) / ppl_base
        receipt["verdicts"][key] = {
            "verdict": "PASS" if abs(delta_pct) < 2.0 else "FAIL",
            "delta_pct": round(delta_pct, 4),
            "ratio": ratio,
        }

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"rvq_8x_ppl_v2_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
