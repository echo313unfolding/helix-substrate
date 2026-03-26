#!/usr/bin/env python3
"""
WO-K64-CROSS-MAMBA-01: k=64 PPL Cross-Validation on Mamba-130m
================================================================
Tests whether k=64 (6-bit) VQ holds under 2% PPL on Mamba-130m (SSM).
Cross-model + cross-architecture validation.

Mamba has 4 linear per block (in_proj, x_proj, dt_proj, out_proj) + embedding.
Module access: model.backbone.layers[i].mixer.{proj}

Strategies:
  A. Baseline (FP32, no compression)
  B. k=256 (8-bit) + sidecar — production control
  C. k=64 (6-bit) + sidecar — THE TEST

Output: receipts/k64_cross_validation/
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

MODEL_DIR = Path.home() / "models" / "mamba-130m-hf"
EVAL_TOKENS = 4096
SEQ_LEN = 2048
OUTLIER_PERCENTILE = 99.9

# Auto-detect layer count from config.json
_config = json.load(open(MODEL_DIR / "config.json"))
N_BLOCKS = _config.get("num_hidden_layers") or _config.get("n_layer")
if N_BLOCKS is None:
    raise ValueError(f"Cannot detect layer count from {MODEL_DIR / 'config.json'}")

TENSOR_TYPES = ["in_proj", "x_proj", "dt_proj", "out_proj"]
HF_PATTERNS = {
    "in_proj":  "backbone.layers.{i}.mixer.in_proj.weight",
    "x_proj":   "backbone.layers.{i}.mixer.x_proj.weight",
    "dt_proj":  "backbone.layers.{i}.mixer.dt_proj.weight",
    "out_proj": "backbone.layers.{i}.mixer.out_proj.weight",
}
EMBEDDING_NAME = "backbone.embeddings.weight"


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


def kmeans_1d(data, k=256, max_iters=10, seed=42):
    """1D k-means matching production _simple_kmeans."""
    n_unique = len(np.unique(data[:min(500_000, len(data))]))
    k = min(k, n_unique)

    rng = np.random.RandomState(seed)
    n_sample = min(1_000_000, len(data))
    if n_sample < len(data):
        sample = data[rng.choice(len(data), n_sample, replace=False)]
    else:
        sample = data

    centroids = np.percentile(sample, np.linspace(0, 100, k)).astype(np.float32)

    for _ in range(max_iters):
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


def encode_vq(weight_np, k=256):
    """Generic VQ (k centroids) + outlier sidecar. Returns reconstructed weight."""
    flat = weight_np.ravel().astype(np.float32)
    codebook = kmeans_1d(flat, k=k)
    indices = assign_all(flat, codebook)
    recon = codebook[indices]

    positions, values = find_outliers(flat)
    recon = apply_sidecar(recon, positions, values)

    return recon.reshape(weight_np.shape), len(positions)


def get_module(model, block_idx, tensor_type):
    """Get nn.Linear module for a given Mamba block and tensor type."""
    layer = model.backbone.layers[block_idx]
    return getattr(layer.mixer, tensor_type)


def set_module_weight(model, block_idx, tensor_type, new_weight_np):
    """Replace a module's weight with reconstructed weight."""
    mod = get_module(model, block_idx, tensor_type)
    new_w = torch.from_numpy(new_weight_np.astype(np.float32))
    mod.weight.data.copy_(new_w)


def run_strategy(model_base, sf, eval_tokens, name, k_value):
    """Encode all mixer tensors + embedding, swap weights, measure PPL."""
    n_mixer = N_BLOCKS * len(TENSOR_TYPES)
    print(f"\n  [{name}] Encoding {n_mixer} mixer + 1 embedding tensors (k={k_value})...")
    model = copy.deepcopy(model_base)
    cosines = []
    total_outliers = 0
    per_tensor_cosines = {}

    # Encode embedding
    if EMBEDDING_NAME in sf.keys():
        weight_np = sf.get_tensor(EMBEDDING_NAME).astype(np.float32)
        recon, n_out = encode_vq(weight_np, k=k_value)
        total_outliers += n_out

        orig_flat = weight_np.ravel()
        recon_flat = recon.ravel()
        cos = float(np.dot(orig_flat, recon_flat) /
                    (np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat) + 1e-30))
        cosines.append(cos)
        per_tensor_cosines[EMBEDDING_NAME] = cos

        new_w = torch.from_numpy(recon.astype(np.float32))
        model.backbone.embeddings.weight.data.copy_(new_w)
        del weight_np, recon

    # Encode mixer layers
    for block_idx in range(N_BLOCKS):
        for tensor_type in TENSOR_TYPES:
            hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
            weight_np = sf.get_tensor(hf_name).astype(np.float32)

            recon, n_out = encode_vq(weight_np, k=k_value)
            total_outliers += n_out

            orig_flat = weight_np.ravel()
            recon_flat = recon.ravel()
            cos = float(np.dot(orig_flat, recon_flat) /
                        (np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat) + 1e-30))
            cosines.append(cos)
            per_tensor_cosines[hf_name] = cos

            set_module_weight(model, block_idx, tensor_type, recon)
            del weight_np, recon

        if (block_idx + 1) % 6 == 0 or block_idx == N_BLOCKS - 1:
            print(f"    Block {block_idx + 1}/{N_BLOCKS}", flush=True)

    print(f"  [{name}] Computing PPL... (outliers patched: {total_outliers})")
    ppl, nll, n_tok = compute_perplexity(model, eval_tokens)

    worst = sorted(per_tensor_cosines.items(), key=lambda x: x[1])[:5]

    result = {
        "k": k_value,
        "mean_cosine": round(float(np.mean(cosines)), 6),
        "min_cosine": round(float(min(cosines)), 6),
        "ppl": round(ppl, 6),
        "nll": round(nll, 6),
        "n_tokens": n_tok,
        "n_tensors_encoded": len(cosines),
        "total_outliers_patched": total_outliers,
        "worst_5_tensors": [{"tensor": t, "cosine": round(c, 6)} for t, c in worst],
    }

    print(f"  [{name}] PPL: {ppl:.4f}, mean cos: {np.mean(cosines):.6f}, "
          f"min cos: {min(cosines):.6f}")

    del model
    gc.collect()
    return result


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    n_total = N_BLOCKS * len(TENSOR_TYPES) + 1  # +1 for embedding
    print("=" * 70)
    print("  WO-K64-CROSS-MAMBA-01: k=64 Cross-Validation on Mamba-130m")
    print(f"  Auto-detected: {N_BLOCKS} blocks, {n_total} tensors")
    print("  Question: Does k=64 (6-bit) hold under 2% PPL on Mamba (SSM)?")
    print("=" * 70)

    assert MODEL_DIR.exists(), f"Model not found: {MODEL_DIR}"

    print("\n  Loading tokenizer + WikiText-2...")
    from transformers import AutoTokenizer, MambaForCausalLM
    from safetensors import safe_open

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:EVAL_TOKENS]
    print(f"  Eval tokens: {len(eval_tokens)}")

    print("  Loading model (FP32) via MambaForCausalLM...")
    model_base = MambaForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model_base.eval()

    print("  Computing baseline PPL...")
    ppl_base, nll_base, n_tok = compute_perplexity(model_base, eval_tokens)
    print(f"  Baseline PPL: {ppl_base:.4f}")

    sf = safe_open(str(MODEL_DIR / "model.safetensors"), framework="numpy")

    # Strategy B: k=256 (8-bit) + sidecar — production control
    r_k256 = run_strategy(model_base, sf, eval_tokens, "k=256 (8-bit)+sc", k_value=256)

    # Strategy C: k=64 (6-bit) + sidecar — THE TEST
    r_k64 = run_strategy(model_base, sf, eval_tokens, "k=64 (6-bit)+sc", k_value=64)

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    # Results table
    print(f"\n{'=' * 70}")
    print("  RESULTS — WO-K64-CROSS-MAMBA-01")
    print(f"{'=' * 70}")
    print(f"  Model: Mamba-130m ({N_BLOCKS} blocks, SSM architecture)")
    print(f"  Baseline PPL: {ppl_base:.4f}")
    print(f"  Sidecar: percentile={OUTLIER_PERCENTILE}")
    print()

    strategies = [
        ("k=256 (8-bit)+sc", r_k256, "4.0x"),
        ("k=64 (6-bit)+sc", r_k64, "5.3x"),
    ]

    print(f"  {'Strategy':<20} {'PPL':>8} {'delta%':>8} {'Mean cos':>10} {'Min cos':>10} {'Ratio':>7}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*7}")
    for name, r, ratio in strategies:
        delta_pct = 100 * (r["ppl"] - ppl_base) / ppl_base
        print(f"  {name:<20} {r['ppl']:>8.4f} {delta_pct:>+7.2f}% "
              f"{r['mean_cosine']:>10.6f} {r['min_cosine']:>10.6f} {ratio:>7}")

    # Verdict
    print()
    for name, r, ratio in strategies:
        delta_pct = 100 * (r["ppl"] - ppl_base) / ppl_base
        verdict = "PASS" if abs(delta_pct) < 2.0 else "FAIL"
        print(f"  {name:<20} -> {verdict} ({delta_pct:+.2f}%)")

    k64_delta = 100 * (r_k64["ppl"] - ppl_base) / ppl_base
    print(f"\n  === KEY ANSWER ===")
    if abs(k64_delta) < 2.0:
        print(f"  k=64 PASSES on Mamba-130m at {k64_delta:+.2f}% -> cross-architecture validated")
    else:
        print(f"  k=64 FAILS on Mamba-130m at {k64_delta:+.2f}% -> transformer-only")

    print(f"\n  Wall: {wall:.0f}s, CPU: {cpu:.0f}s")

    # Receipt
    receipt_dir = Path(__file__).parent.parent / "receipts" / "k64_cross_validation"
    receipt_dir.mkdir(parents=True, exist_ok=True)

    receipt = {
        "work_order": "WO-K64-CROSS-MAMBA-01",
        "question": "Does k=64 (6-bit) flat VQ hold under 2% PPL on Mamba-130m?",
        "model": "Mamba-130m",
        "architecture": "MambaForCausalLM",
        "n_blocks": N_BLOCKS,
        "n_tensors": n_total,
        "eval_dataset": "wikitext-2-raw-v1 (validation)",
        "eval_tokens": len(eval_tokens),
        "sidecar_percentile": OUTLIER_PERCENTILE,
        "baseline": {
            "ppl": round(ppl_base, 6),
            "nll": round(nll_base, 6),
        },
        "strategies": {
            "k256_8bit_sidecar": r_k256,
            "k64_6bit_sidecar": r_k64,
        },
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
        key = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("+", "_").replace("=", "eq")
        delta_pct = 100 * (r["ppl"] - ppl_base) / ppl_base
        receipt["verdicts"][key] = {
            "verdict": "PASS" if abs(delta_pct) < 2.0 else "FAIL",
            "delta_pct": round(delta_pct, 4),
            "ratio": ratio,
        }

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"k64_mamba_130m_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
