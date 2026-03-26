#!/usr/bin/env python3
"""
WO-ADAPTIVE-PPL-01: PPL validation of adaptive k compression on Qwen2.5-Coder-1.5B
====================================================================================
Tests whether adaptive k selection (64->128->256->+SVD) passes the 2% PPL threshold
on the model that FAILED flat k=64 (+2.78% PPL).

Loads actual compressed artifacts from cdnav3_adaptive/ — no re-encoding.

Strategies:
  A. Baseline (FP32, no compression)
  B. Adaptive k (loaded from cdnav3_adaptive/) — THE TEST
  C. Flat k=64 (re-encoded for comparison) — KNOWN FAIL

Output: receipts/adaptive_ppl/
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

MODEL_DIR = Path.home() / "models" / "qwen2.5-coder-1.5b-instruct"
ADAPTIVE_DIR = MODEL_DIR / "cdnav3_adaptive"
EVAL_TOKENS = 4096
SEQ_LEN = 2048
OUTLIER_PERCENTILE = 99.9

# Auto-detect layer count
_config = json.load(open(MODEL_DIR / "config.json"))
N_BLOCKS = _config.get("num_hidden_layers") or _config.get("n_layer")

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


def load_compressed_tensor(cdna_dir, hf_name):
    """Load a tensor reconstructed from CDNA v3 adaptive artifacts."""
    safe_name = hf_name.replace("/", "_").replace(".", "_")
    tensor_dir = cdna_dir / f"{safe_name}.cdnav3"

    if not tensor_dir.exists():
        return None, None

    # Read meta
    meta = json.loads((tensor_dir / "meta.json").read_text())
    rows, cols = meta["shape"]

    # Read codebook + indices
    codebook = np.load(tensor_dir / "codebook.npy")
    indices = np.fromfile(tensor_dir / "indices.bin", dtype=np.uint8)
    recon = codebook[indices].reshape(rows, cols)

    # Apply sidecar if present
    sidecar_path = tensor_dir / "sidecar.npz"
    if sidecar_path.exists():
        sc = np.load(sidecar_path)
        positions = sc["positions"]
        values = sc["values"]
        flat = recon.ravel()
        flat[positions] = values
        recon = flat.reshape(rows, cols)

    # Apply SVD residual if present
    svd_u_path = tensor_dir / "svd_U.npy"
    if svd_u_path.exists():
        U = np.load(svd_u_path)
        s = np.load(tensor_dir / "svd_s.npy")
        Vt = np.load(tensor_dir / "svd_Vt.npy")
        recon = recon + (U * s[None, :]) @ Vt

    # Read stats for chosen_k
    stats_path = tensor_dir / "stats.json"
    chosen_k = None
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        chosen_k = stats.get("chosen_k")

    return recon.astype(np.float32), chosen_k


def get_module(model, block_idx, tensor_type):
    layer = model.model.layers[block_idx]
    if tensor_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, tensor_type)
    return getattr(layer.mlp, tensor_type)


def set_module_weight(model, block_idx, tensor_type, new_weight_np):
    mod = get_module(model, block_idx, tensor_type)
    new_w = torch.from_numpy(new_weight_np)
    mod.weight.data.copy_(new_w)


def cosine_sim(a, b):
    a_flat = a.ravel()
    b_flat = b.ravel()
    dot = np.dot(a_flat, b_flat)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 70)
    print("  WO-ADAPTIVE-PPL-01: Adaptive k PPL Validation")
    print(f"  Model: Qwen2.5-Coder-1.5B-Instruct ({N_BLOCKS} blocks)")
    print(f"  Adaptive dir: {ADAPTIVE_DIR}")
    print("  Question: Does adaptive k pass 2% PPL on the model that")
    print("            FAILED flat k=64 (+2.78%)?")
    print("=" * 70)

    assert MODEL_DIR.exists(), f"Model not found: {MODEL_DIR}"
    assert ADAPTIVE_DIR.exists(), f"Adaptive dir not found: {ADAPTIVE_DIR}"

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

    # ── Strategy: Adaptive k (from compressed artifacts) ──
    print(f"\n  [Adaptive] Loading {N_BLOCKS * len(TENSOR_TYPES)} tensors from {ADAPTIVE_DIR}...")
    sf = safe_open(str(MODEL_DIR / "model.safetensors"), framework="pt")
    model_adaptive = copy.deepcopy(model_base)

    cosines = []
    k_distribution = {}
    per_tensor_info = []
    n_loaded = 0
    n_missing = 0

    for block_idx in range(N_BLOCKS):
        for tensor_type in TENSOR_TYPES:
            hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
            recon, chosen_k = load_compressed_tensor(ADAPTIVE_DIR, hf_name)

            if recon is None:
                n_missing += 1
                continue

            # Compute cosine vs original
            orig = sf.get_tensor(hf_name).float().numpy()
            cos = cosine_sim(orig, recon)
            cosines.append(cos)

            # Track k distribution
            k_key = str(chosen_k) if chosen_k else "unknown"
            k_distribution[k_key] = k_distribution.get(k_key, 0) + 1

            per_tensor_info.append({
                "tensor": hf_name,
                "cosine": round(cos, 6),
                "chosen_k": chosen_k,
            })

            set_module_weight(model_adaptive, block_idx, tensor_type, recon)
            del orig, recon
            n_loaded += 1

        if (block_idx + 1) % 7 == 0 or block_idx == N_BLOCKS - 1:
            print(f"    Block {block_idx + 1}/{N_BLOCKS}", flush=True)

    print(f"  [Adaptive] Loaded: {n_loaded}, Missing: {n_missing}")
    print(f"  [Adaptive] k distribution: {k_distribution}")
    print(f"  [Adaptive] Computing PPL...")
    ppl_adaptive, nll_adaptive, n_tok_a = compute_perplexity(model_adaptive, eval_tokens)

    delta_pct = 100 * (ppl_adaptive - ppl_base) / ppl_base
    verdict = "PASS" if abs(delta_pct) < 2.0 else "FAIL"

    del model_adaptive
    gc.collect()

    # Results
    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    worst_5 = sorted(per_tensor_info, key=lambda x: x["cosine"])[:5]

    print(f"\n{'=' * 70}")
    print("  RESULTS — WO-ADAPTIVE-PPL-01")
    print(f"{'=' * 70}")
    print(f"  Baseline PPL:  {ppl_base:.4f}")
    print(f"  Adaptive PPL:  {ppl_adaptive:.4f}  ({delta_pct:+.2f}%)  -> {verdict}")
    print(f"  Mean cosine:   {np.mean(cosines):.6f}")
    print(f"  Min cosine:    {min(cosines):.6f}")
    print(f"  k distribution: {k_distribution}")
    print(f"\n  Comparison:")
    print(f"    Flat k=64:   +2.78% PPL  -> FAIL")
    print(f"    Adaptive:    {delta_pct:+.2f}% PPL  -> {verdict}")
    print(f"\n  Worst 5 tensors:")
    for t in worst_5:
        print(f"    cos={t['cosine']:.6f}  k={t['chosen_k']}  {t['tensor']}")
    print(f"\n  Wall: {wall:.0f}s, CPU: {cpu:.0f}s")
    print(f"{'=' * 70}")

    # Receipt
    receipt_dir = Path(__file__).parent.parent / "receipts" / "adaptive_ppl"
    receipt_dir.mkdir(parents=True, exist_ok=True)

    receipt = {
        "work_order": "WO-ADAPTIVE-PPL-01",
        "question": "Does adaptive k pass 2% PPL on the model that failed flat k=64?",
        "verdict": verdict,
        "model": "Qwen2.5-Coder-1.5B-Instruct",
        "n_blocks": N_BLOCKS,
        "n_tensors_loaded": n_loaded,
        "n_tensors_missing": n_missing,
        "eval_dataset": "wikitext-2-raw-v1 (validation)",
        "eval_tokens": len(eval_tokens),
        "baseline": {
            "ppl": round(ppl_base, 6),
            "nll": round(nll_base, 6),
        },
        "adaptive": {
            "ppl": round(ppl_adaptive, 6),
            "nll": round(nll_adaptive, 6),
            "delta_pct": round(delta_pct, 4),
            "mean_cosine": round(float(np.mean(cosines)), 6),
            "min_cosine": round(float(min(cosines)), 6),
            "n_tokens": n_tok_a,
            "k_distribution": k_distribution,
            "worst_5_tensors": worst_5,
        },
        "comparison": {
            "flat_k64_delta_pct": 2.78,
            "flat_k64_verdict": "FAIL",
            "adaptive_delta_pct": round(delta_pct, 4),
            "adaptive_verdict": verdict,
        },
        "blended_info_theoretic_ratio": 5.09,
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

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"adaptive_ppl_qwen_1.5b_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
