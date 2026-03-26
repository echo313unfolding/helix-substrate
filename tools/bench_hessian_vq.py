#!/usr/bin/env python3
"""
WO-GPTQ-HELIX-HYBRID-01: Hessian-Informed VQ Compression

Tests whether GPTQ-style Hessian calibration improves CDNA v3 VQ quality.
Runs on TinyLlama-1.1B on CPU.

Three strategies compared:

  Strategy A — Naive VQ:      k-means codebook, round-to-nearest assignment
  Strategy B — Kurtosis VQ:   Current routing (SVD rank-8 on kurtosis>5)
  Strategy C — Hessian VQ:    GPTQ-style H=X^T@X importance-weighted VQ
                               + OBS error compensation across columns

If C beats B, Hessian calibration closes the gap vs GPTQ.
If B ties C, kurtosis is a sufficient cheap proxy.

Usage:
    python3 tools/bench_hessian_vq.py
    python3 tools/bench_hessian_vq.py --n-calibration 64 --tokens 4096

Work Order: WO-GPTQ-HELIX-HYBRID-01
"""

import argparse
import gc
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

from helix_substrate.cdnav3_writer import CDNAv3Writer, _cosine
from helix_substrate.helix_linear import (
    load_helix_linear_from_cdnav3,
    swap_to_helix,
)
from helix_substrate.tensor_policy import (
    classify_tensor,
    get_default_policy,
    get_policy,
)

MODEL_DIR = Path.home() / "models" / "tinyllama_fp32"
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "hessian_vq"

N_BLOCKS = 22
BLOCK_TENSOR_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
HF_PATTERNS = {
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}

# ────────────────────────────────────────────────────────────
# Hessian collection
# ────────────────────────────────────────────────────────────

class HessianCollector:
    """Collect activation covariance (Hessian) for each linear layer."""

    def __init__(self):
        self.hessians = {}  # module_name -> (H, n_samples)
        self._hooks = []

    def register(self, model):
        """Register forward hooks on all nn.Linear modules."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()  # [batch, seq, d_in]
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])  # [batch*seq, d_in]

            # Accumulate H = X^T @ X (unnormalized, we'll normalize later)
            h = x.T @ x  # [d_in, d_in]

            if name not in self.hessians:
                self.hessians[name] = (h.cpu(), x.shape[0])
            else:
                old_h, old_n = self.hessians[name]
                self.hessians[name] = (old_h + h.cpu(), old_n + x.shape[0])

        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_hessian(self, name):
        """Return normalized Hessian H = (1/N) * X^T @ X."""
        if name not in self.hessians:
            return None
        h, n = self.hessians[name]
        return (h / n).numpy()

    def get_importance(self, name):
        """Return per-column importance = diag(H)."""
        h = self.get_hessian(name)
        if h is None:
            return None
        return np.diag(h)


# ────────────────────────────────────────────────────────────
# Hessian-informed VQ encoder
# ────────────────────────────────────────────────────────────

def hessian_vq_encode(
    weight: np.ndarray,
    hessian: np.ndarray,
    n_clusters: int = 256,
    blocksize: int = 128,
    percdamp: float = 0.01,
):
    """
    GPTQ-style VQ encoding: quantize columns sequentially,
    compensate remaining columns using inverse Hessian.

    Args:
        weight: [d_out, d_in] weight matrix
        hessian: [d_in, d_in] activation covariance matrix
        n_clusters: VQ codebook size
        blocksize: columns per block (GPTQ default: 128)
        percdamp: Hessian damping factor

    Returns:
        dict with codebook, indices, cosine, stats
    """
    d_out, d_in = weight.shape
    W = weight.copy()

    # Build codebook from weight distribution (same as standard VQ)
    flat = W.ravel()
    from helix_substrate.cdna_encoder import _simple_kmeans
    n_sample = min(len(flat), 500_000)
    rng = np.random.RandomState(42)
    sample = flat[rng.choice(len(flat), n_sample, replace=False)]
    codebook, _ = _simple_kmeans(sample, n_clusters, max_iters=10)
    codebook = codebook.astype(np.float32)

    # Dampen Hessian and compute inverse via Cholesky
    H = hessian.copy().astype(np.float64)
    damp = percdamp * np.mean(np.diag(H))
    H += damp * np.eye(d_in)

    try:
        # Cholesky of H, then inverse, then upper Cholesky of H^{-1}
        L = np.linalg.cholesky(H)
        H_inv = np.linalg.solve(L @ L.T, np.eye(d_in))
        # Upper Cholesky of H_inv
        Hinv = np.linalg.cholesky(H_inv + 1e-10 * np.eye(d_in)).T
    except np.linalg.LinAlgError:
        # Fallback: just use diagonal importance weighting
        print("    WARNING: Cholesky failed, falling back to diagonal importance")
        Hinv = np.diag(1.0 / np.sqrt(np.diag(H) + 1e-10))

    # Quantized output
    Q = np.zeros_like(W)
    indices = np.zeros((d_out, d_in), dtype=np.uint8)

    # Process in blocks
    for i1 in range(0, d_in, blocksize):
        i2 = min(i1 + blocksize, d_in)
        count = i2 - i1

        W1 = W[:, i1:i2].copy()
        Q1 = np.zeros_like(W1)
        Idx1 = np.zeros((d_out, count), dtype=np.uint8)
        Err1 = np.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]  # [d_out]
            d = Hinv1[i, i]

            if d < 1e-10:
                # Skip near-zero importance columns
                q = w.copy()
                idx = np.argmin(np.abs(w[:, None] - codebook), axis=1).astype(np.uint8)
            else:
                # VQ quantize: assign each weight to nearest centroid
                idx = np.argmin(np.abs(w[:, None] - codebook), axis=1).astype(np.uint8)
                q = codebook[idx]

                # Compute scaled error and compensate remaining columns
                err = (w - q) / d  # [d_out]
                W1[:, i:] -= np.outer(err, Hinv1[i, i:])
                Err1[:, i] = err

            Q1[:, i] = q
            Idx1[:, i] = idx

        Q[:, i1:i2] = Q1
        indices[:, i1:i2] = Idx1

        # Lazy batch update: propagate error to remaining columns
        if i2 < d_in:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    # Compute fidelity
    flat_orig = weight.ravel()
    flat_q = Q.ravel()
    cos = float(_cosine(flat_orig, flat_q))
    mse = float(np.mean((flat_orig - flat_q) ** 2))

    return {
        "codebook": codebook,
        "indices": indices.ravel(),
        "cosine": cos,
        "mse": mse,
        "reconstructed": Q,
    }


# ────────────────────────────────────────────────────────────
# Helper: get module from model
# ────────────────────────────────────────────────────────────

def get_module(model, block_idx, tensor_type):
    layer = model.model.layers[block_idx]
    if tensor_type in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return getattr(layer.self_attn, tensor_type)
    return getattr(layer.mlp, tensor_type)


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


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--n-calibration", type=int, default=64,
                        help="Number of calibration sequences for Hessian")
    parser.add_argument("--cal-seq-len", type=int, default=512,
                        help="Sequence length for calibration")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        print(f"ERROR: TinyLlama not found at {MODEL_DIR}")
        sys.exit(1)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("WO-GPTQ-HELIX-HYBRID-01: Hessian-Informed VQ Compression")
    print("=" * 70)

    # ── Load model ──
    print("\n[1/7] Loading TinyLlama...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model.eval()

    # ── Load eval tokens ──
    print(f"[2/7] Loading WikiText-2...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:args.tokens]
    print(f"  Eval tokens: {len(eval_tokens)}")

    # ── Baseline PPL ──
    print("[3/7] Computing baseline PPL...")
    ppl_baseline, _, _ = compute_perplexity(model, eval_tokens)
    print(f"  Baseline PPL: {ppl_baseline:.4f}")

    # ── Collect Hessians ──
    print(f"[4/7] Collecting Hessians ({args.n_calibration} × {args.cal_seq_len} tokens)...")
    collector = HessianCollector()
    collector.register(model)

    # Use C4 or WikiText for calibration (different from eval)
    try:
        cal_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        cal_text = "\n\n".join([t for t in cal_ds["text"] if t.strip()])
    except Exception:
        cal_text = text  # fallback to same text

    cal_tokens = tokenizer.encode(cal_text)
    n_cal = 0
    for i in range(args.n_calibration):
        start = np.random.randint(0, max(1, len(cal_tokens) - args.cal_seq_len))
        seq = cal_tokens[start:start + args.cal_seq_len]
        input_ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            model(input_ids)
        n_cal += 1
        if (i + 1) % 16 == 0:
            print(f"  Calibration: {i + 1}/{args.n_calibration}", flush=True)

    collector.remove_hooks()
    print(f"  Collected Hessians for {len(collector.hessians)} layers")

    # ── Build module name mapping ──
    # Map HF tensor names to model module paths
    module_map = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            module_map[name] = mod

    # ── Strategy A: Naive VQ (no SVD, no Hessian) ──
    print("\n[5/7] Strategy A: Naive VQ...")
    import copy

    sf_path = MODEL_DIR / "model.safetensors"
    results = {"A": {}, "B": {}, "C": {}}

    with safe_open(str(sf_path), framework="numpy") as sf:
        tmpdir_a = Path(tempfile.mkdtemp())
        writer_a = CDNAv3Writer(tmpdir_a)
        model_a = copy.deepcopy(model)
        helix_a = {}
        cosines_a = []

        for block_idx in range(N_BLOCKS):
            for tensor_type in BLOCK_TENSOR_TYPES:
                hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                tensor_np = sf.get_tensor(hf_name).astype(np.float32)

                tc = classify_tensor(hf_name, shape=tensor_np.shape)
                policy = get_default_policy(tc)

                stats = writer_a.write_tensor(tensor_np, hf_name, policy=policy)
                cosines_a.append(stats.get("cosine_with_sidecar", 0))

                safe_name = hf_name.replace("/", "_").replace(".", "_")
                tensor_dir = tmpdir_a / f"{safe_name}.cdnav3"
                module_path = hf_name.replace(".weight", "")
                mod = get_module(model_a, block_idx, tensor_type)
                bias = mod.bias.data.clone() if mod.bias is not None else None
                helix_a[module_path] = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)

        model_a = swap_to_helix(model_a, helix_a)
        ppl_a, _, _ = compute_perplexity(model_a, eval_tokens)
        results["A"] = {"ppl": ppl_a, "mean_cosine": float(np.mean(cosines_a)),
                        "svd_tensors": 0, "svd_bytes": 0}
        print(f"  PPL: {ppl_a:.4f}, mean cosine: {np.mean(cosines_a):.6f}")
        del model_a; gc.collect()

    # ── Strategy B: Kurtosis-routed VQ ──
    print("[6/7] Strategy B: Kurtosis-routed VQ...")

    with safe_open(str(sf_path), framework="numpy") as sf:
        tmpdir_b = Path(tempfile.mkdtemp())
        writer_b = CDNAv3Writer(tmpdir_b)
        model_b = copy.deepcopy(model)
        helix_b = {}
        cosines_b = []
        svd_count_b = 0
        svd_bytes_b = 0

        for block_idx in range(N_BLOCKS):
            for tensor_type in BLOCK_TENSOR_TYPES:
                hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                tensor_np = sf.get_tensor(hf_name).astype(np.float32)
                kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))

                policy = get_policy(hf_name, tensor_np.shape,
                                    block_idx=block_idx, kurtosis=kurt,
                                    n_blocks=N_BLOCKS)

                stats = writer_b.write_tensor(tensor_np, hf_name, policy=policy)
                cosines_b.append(stats.get("cosine_with_svd",
                                           stats.get("cosine_with_sidecar", 0)))
                if stats.get("svd_bytes", 0) > 0:
                    svd_count_b += 1
                    svd_bytes_b += stats["svd_bytes"]

                safe_name = hf_name.replace("/", "_").replace(".", "_")
                tensor_dir = tmpdir_b / f"{safe_name}.cdnav3"
                module_path = hf_name.replace(".weight", "")
                mod = get_module(model_b, block_idx, tensor_type)
                bias = mod.bias.data.clone() if mod.bias is not None else None
                helix_b[module_path] = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)

        model_b = swap_to_helix(model_b, helix_b)
        ppl_b, _, _ = compute_perplexity(model_b, eval_tokens)
        results["B"] = {"ppl": ppl_b, "mean_cosine": float(np.mean(cosines_b)),
                        "svd_tensors": svd_count_b, "svd_bytes": svd_bytes_b}
        print(f"  PPL: {ppl_b:.4f}, mean cosine: {np.mean(cosines_b):.6f}, "
              f"SVD tensors: {svd_count_b}")
        del model_b; gc.collect()

    # ── Strategy C: Hessian-informed VQ ──
    print("[7/7] Strategy C: Hessian-informed VQ (GPTQ-style)...")

    with safe_open(str(sf_path), framework="numpy") as sf:
        tmpdir_c = Path(tempfile.mkdtemp())
        model_c = copy.deepcopy(model)
        helix_c = {}
        cosines_c = []

        for block_idx in range(N_BLOCKS):
            for tensor_type in BLOCK_TENSOR_TYPES:
                hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                tensor_np = sf.get_tensor(hf_name).astype(np.float32)

                # Get module path for Hessian lookup
                module_path = hf_name.replace(".weight", "")

                # Try to get Hessian for this layer
                H = collector.get_hessian(module_path)

                if H is not None and H.shape[0] == tensor_np.shape[1]:
                    # Run Hessian-informed VQ
                    result = hessian_vq_encode(
                        tensor_np, H,
                        n_clusters=256,
                        blocksize=128,
                        percdamp=0.01,
                    )
                    cosines_c.append(result["cosine"])

                    # Write CDNA v3 directory manually
                    safe_name = hf_name.replace("/", "_").replace(".", "_")
                    out_dir = tmpdir_c / f"{safe_name}.cdnav3"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    np.save(out_dir / "codebook.npy", result["codebook"])
                    result["indices"].astype(np.uint8).tofile(out_dir / "indices.bin")

                    rows, cols = tensor_np.shape
                    meta = {
                        "format_version": "cdna_v3",
                        "tensor_name": hf_name,
                        "shape": [rows, cols],
                        "dtype": "float32",
                        "storage_mode": "codebook",
                        "n_clusters": 256,
                        "hessian_informed": True,
                    }
                    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

                    stats = {
                        "tensor_name": hf_name,
                        "shape": [rows, cols],
                        "original_bytes": tensor_np.nbytes,
                        "compressed_bytes": int(
                            (out_dir / "codebook.npy").stat().st_size
                            + (out_dir / "indices.bin").stat().st_size
                        ),
                        "cosine_no_sidecar": round(result["cosine"], 6),
                        "cosine_with_sidecar": round(result["cosine"], 6),
                        "num_outliers": 0,
                        "sidecar_bytes": 0,
                        "svd_residual_rank": 0,
                        "svd_bytes": 0,
                    }
                    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
                else:
                    # Fallback: standard VQ
                    writer_c = CDNAv3Writer(tmpdir_c)
                    tc = classify_tensor(hf_name, shape=tensor_np.shape)
                    policy = get_default_policy(tc)
                    stats = writer_c.write_tensor(tensor_np, hf_name, policy=policy)
                    cosines_c.append(stats.get("cosine_with_sidecar", 0))

                # Load as HelixLinear
                safe_name = hf_name.replace("/", "_").replace(".", "_")
                tensor_dir = tmpdir_c / f"{safe_name}.cdnav3"
                mod = get_module(model_c, block_idx, tensor_type)
                bias = mod.bias.data.clone() if mod.bias is not None else None
                helix_c[module_path] = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)

            if (block_idx + 1) % 5 == 0:
                print(f"  Block {block_idx + 1}/{N_BLOCKS}", flush=True)

        model_c = swap_to_helix(model_c, helix_c)
        ppl_c, _, _ = compute_perplexity(model_c, eval_tokens)
        results["C"] = {"ppl": ppl_c, "mean_cosine": float(np.mean(cosines_c)),
                        "svd_tensors": 0, "svd_bytes": 0}
        print(f"  PPL: {ppl_c:.4f}, mean cosine: {np.mean(cosines_c):.6f}")
        del model_c; gc.collect()

    # ── Results ──
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Baseline PPL: {ppl_baseline:.4f}\n")

    strategies = [
        ("A: Naive VQ", results["A"]),
        ("B: Kurtosis-routed", results["B"]),
        ("C: Hessian-informed VQ", results["C"]),
    ]

    print(f"  {'Strategy':<30} {'PPL':>8} {'Δ%':>8} {'Cosine':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10}")
    for name, r in strategies:
        delta_pct = 100 * (r["ppl"] - ppl_baseline) / ppl_baseline
        print(f"  {name:<30} {r['ppl']:>8.4f} {delta_pct:>+7.2f}% {r['mean_cosine']:>10.6f}")

    # ── Verdict ──
    ppl_c_val = results["C"]["ppl"]
    ppl_b_val = results["B"]["ppl"]
    gap = ppl_c_val - ppl_b_val

    print()
    if ppl_c_val < ppl_b_val - 0.01:
        verdict = "HESSIAN_WINS"
        explanation = (f"Hessian VQ beats kurtosis routing by {-gap:.4f} PPL. "
                       f"GPTQ-style calibration improves CDNA v3.")
    elif abs(gap) <= 0.01:
        verdict = "TIE"
        explanation = (f"Gap is {gap:.4f} PPL. Kurtosis is a sufficient proxy — "
                       f"Hessian calibration adds negligible value for VQ.")
    else:
        verdict = "KURTOSIS_WINS"
        explanation = (f"Kurtosis routing beats Hessian VQ by {gap:.4f} PPL. "
                       f"Hessian error compensation may not suit VQ codebooks.")

    print(f"  VERDICT: {verdict}")
    print(f"  {explanation}")

    # ── Receipt ──
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "work_order": "WO-GPTQ-HELIX-HYBRID-01",
        "description": "Hessian-informed VQ compression vs kurtosis routing",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "eval_tokens": len(eval_tokens),
        "n_calibration": args.n_calibration,
        "cal_seq_len": args.cal_seq_len,
        "baseline_ppl": round(ppl_baseline, 6),
        "strategies": {
            name: {
                "ppl": round(r["ppl"], 6),
                "ppl_delta_pct": round(100 * (r["ppl"] - ppl_baseline) / ppl_baseline, 4),
                "mean_cosine": round(r["mean_cosine"], 6),
                "svd_tensors": r.get("svd_tensors", 0),
                "svd_bytes": r.get("svd_bytes", 0),
            }
            for name, r in strategies
        },
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

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"hessian_vq_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
