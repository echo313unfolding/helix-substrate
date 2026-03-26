#!/usr/bin/env python3
"""
WO-GPTQ-HELIX-HYBRID-02: Hessian Diagonal Routing + Weighted K-Means

Successor to -01 (which failed catastrophically with OBS compensation loop).

Key insight from -01 failure: GPTQ's OBS sequential column compensation
assumes independent scalar quantization errors. VQ errors are correlated
(weights sharing a centroid have coupled errors), so compensation cascades
destroy quality (PPL 313 vs baseline 6.17).

This experiment uses the Hessian diagonal ONLY, for two purposes:

  1. ROUTING — Per-tensor Hessian diagonal norm replaces kurtosis threshold.
     High H_diag_norm tensors get SVD rank-8. Low norm get VQ-only.

  2. WEIGHTED K-MEANS — During k-means centroid computation, each weight's
     contribution is weighted by its Hessian diagonal value. High-sensitivity
     weights pull centroids toward themselves more strongly.

No sequential column loop. No H^{-1} compensation. No Cholesky.

Four strategies compared:
  A — Naive VQ:           k-means, round-to-nearest, no routing
  B — Kurtosis-routed:    Current production policy (SVD r8 on kurtosis>5)
  C — Hessian-routed:     H_diag_norm threshold for SVD r8 routing
  D — Hessian-weighted:   Weighted k-means + H_diag_norm routing

Also computes Spearman rho between per-tensor kurtosis and H_diag_norm.

Usage:
    python3 tools/bench_hessian_vq_v2.py
    python3 tools/bench_hessian_vq_v2.py --n-calibration 64 --tokens 4096

Work Order: WO-GPTQ-HELIX-HYBRID-02
"""

import argparse
import copy
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
from scipy.stats import kurtosis as scipy_kurtosis, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.cdnav3_writer import CDNAv3Writer, _cosine
from helix_substrate.helix_linear import (
    load_helix_linear_from_cdnav3,
    swap_to_helix,
)
from helix_substrate.tensor_policy import (
    TensorPolicy,
    classify_tensor,
    get_default_policy,
    get_policy,
)

MODEL_DIR = Path.home() / "models" / "tinyllama_fp32"
RECEIPT_DIR = Path(__file__).parent.parent / "receipts" / "hessian_vq_v2"

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
# Hessian collection (reused from v1, proven correct)
# ────────────────────────────────────────────────────────────

class HessianCollector:
    """Collect activation covariance (Hessian) for each linear layer."""

    def __init__(self):
        self.hessians = {}  # module_name -> (H_diag, n_samples)
        self._hooks = []

    def register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])

            # Only accumulate diagonal: diag(X^T @ X) = sum of x_i^2 per column
            h_diag = (x * x).sum(dim=0)  # [d_in]

            if name not in self.hessians:
                self.hessians[name] = (h_diag.cpu(), x.shape[0])
            else:
                old_h, old_n = self.hessians[name]
                self.hessians[name] = (old_h + h_diag.cpu(), old_n + x.shape[0])

        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_diag(self, name):
        """Return normalized Hessian diagonal: (1/N) * diag(X^T @ X)."""
        if name not in self.hessians:
            return None
        h_diag, n = self.hessians[name]
        return (h_diag / n).numpy()

    def get_diag_norm(self, name):
        """Return L2 norm of Hessian diagonal — scalar importance per tensor."""
        d = self.get_diag(name)
        if d is None:
            return None
        return float(np.linalg.norm(d))


# ────────────────────────────────────────────────────────────
# Importance-weighted k-means
# ────────────────────────────────────────────────────────────

def weighted_kmeans(
    data: np.ndarray,
    weights: np.ndarray,
    n_clusters: int = 256,
    max_iters: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means where each sample has an importance weight.

    During centroid update, centroids are the weighted mean of assigned points.
    Assignment (nearest centroid) is still unweighted — we want fidelity for
    ALL weights, but high-importance weights steer centroid placement.

    Args:
        data: 1D array of weight values
        weights: 1D array of importance weights (same length as data)
        n_clusters: Number of centroids
        max_iters: Maximum iterations

    Returns:
        (centroids, assignments)
    """
    n_clusters = min(n_clusters, len(np.unique(data)))

    # Initialize centroids using percentiles
    percentiles = np.linspace(0, 100, n_clusters)
    centroids = np.percentile(data, percentiles).astype(np.float32)

    for _ in range(max_iters):
        # Assign points to nearest centroid (unweighted distances)
        dists = np.abs(data[:, np.newaxis] - centroids)
        assignments = np.argmin(dists, axis=1).astype(np.uint8)

        # Update centroids using weighted mean
        new_centroids = np.zeros_like(centroids)
        for i in range(n_clusters):
            mask = assignments == i
            if np.any(mask):
                w = weights[mask]
                w_sum = w.sum()
                if w_sum > 0:
                    new_centroids[i] = np.average(data[mask], weights=w)
                else:
                    new_centroids[i] = np.mean(data[mask])
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Final assignment
    dists = np.abs(data[:, np.newaxis] - centroids)
    assignments = np.argmin(dists, axis=1).astype(np.uint8)

    return centroids, assignments


def weighted_vq_encode(
    weight: np.ndarray,
    h_diag: np.ndarray,
    n_clusters: int = 256,
):
    """
    VQ encode a weight matrix using importance-weighted k-means.

    The Hessian diagonal h_diag[j] tells us how sensitive the output is
    to perturbations in column j. We expand this to per-element importance
    by broadcasting: importance[i,j] = h_diag[j] for all rows i.

    Args:
        weight: [d_out, d_in] weight matrix
        h_diag: [d_in] Hessian diagonal (per-input-feature importance)
        n_clusters: VQ codebook size

    Returns:
        dict with codebook, indices, cosine, reconstructed
    """
    d_out, d_in = weight.shape
    flat = weight.ravel()

    # Expand h_diag to per-element importance
    # weight[i,j] gets importance h_diag[j] (column importance)
    importance = np.tile(h_diag, d_out)  # [d_out * d_in]

    # Normalize importance to avoid numerical issues
    imp_max = importance.max()
    if imp_max > 0:
        importance = importance / imp_max

    # Subsample for k-means (same as standard pipeline)
    n_sample = min(len(flat), 500_000)
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(len(flat), n_sample, replace=False)
    sample_data = flat[idx_sample]
    sample_weights = importance[idx_sample]

    codebook, _ = weighted_kmeans(sample_data, sample_weights, n_clusters, max_iters=10)
    codebook = codebook.astype(np.float32)

    # Assign all elements to nearest centroid
    indices = np.argmin(np.abs(flat[:, None] - codebook), axis=1).astype(np.uint8)
    reconstructed = codebook[indices].reshape(d_out, d_in)

    cos = float(_cosine(weight.ravel(), reconstructed.ravel()))

    return {
        "codebook": codebook,
        "indices": indices,
        "cosine": cos,
        "reconstructed": reconstructed,
    }


# ────────────────────────────────────────────────────────────
# Helpers
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


def write_cdnav3_manually(out_dir, hf_name, codebook, indices, shape):
    """Write a CDNA v3 directory from raw codebook + indices."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "codebook.npy", codebook)
    indices.astype(np.uint8).tofile(out_dir / "indices.bin")

    meta = {
        "format_version": "cdna_v3",
        "tensor_name": hf_name,
        "shape": list(shape),
        "dtype": "float32",
        "storage_mode": "codebook",
        "n_clusters": len(codebook),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    cb_bytes = (out_dir / "codebook.npy").stat().st_size
    idx_bytes = (out_dir / "indices.bin").stat().st_size
    stats = {
        "tensor_name": hf_name,
        "shape": list(shape),
        "original_bytes": int(np.prod(shape) * 4),
        "compressed_bytes": cb_bytes + idx_bytes,
        "cosine_no_sidecar": 0,
        "cosine_with_sidecar": 0,
        "num_outliers": 0,
        "sidecar_bytes": 0,
        "svd_residual_rank": 0,
        "svd_bytes": 0,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--n-calibration", type=int, default=64)
    parser.add_argument("--cal-seq-len", type=int, default=512)
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        print(f"ERROR: TinyLlama not found at {MODEL_DIR}")
        sys.exit(1)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("WO-GPTQ-HELIX-HYBRID-02: Hessian Diagonal Routing + Weighted K-Means")
    print("  No OBS loop. No Cholesky. Only diagonal importance.")
    print("=" * 70)

    # ── Load model ──
    print("\n[1/9] Loading TinyLlama...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model.eval()

    # ── Load eval tokens ──
    print(f"[2/9] Loading WikiText-2...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:args.tokens]
    print(f"  Eval tokens: {len(eval_tokens)}")

    # ── Baseline PPL ──
    print("[3/9] Computing baseline PPL...")
    ppl_baseline, _, _ = compute_perplexity(model, eval_tokens)
    print(f"  Baseline PPL: {ppl_baseline:.4f}")

    # ── Collect Hessian diagonals ──
    print(f"[4/9] Collecting Hessian diagonals "
          f"({args.n_calibration} × {args.cal_seq_len} tokens)...")
    collector = HessianCollector()
    collector.register(model)

    try:
        cal_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        cal_text = "\n\n".join([t for t in cal_ds["text"] if t.strip()])
    except Exception:
        cal_text = text

    cal_tokens = tokenizer.encode(cal_text)
    for i in range(args.n_calibration):
        start = np.random.randint(0, max(1, len(cal_tokens) - args.cal_seq_len))
        seq = cal_tokens[start:start + args.cal_seq_len]
        input_ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            model(input_ids)
        if (i + 1) % 16 == 0:
            print(f"  Calibration: {i + 1}/{args.n_calibration}", flush=True)

    collector.remove_hooks()
    print(f"  Collected diagonals for {len(collector.hessians)} layers")

    # ── Compute per-tensor stats ──
    print("[5/9] Computing per-tensor kurtosis + Hessian diagonal norm...")
    sf_path = MODEL_DIR / "model.safetensors"

    per_tensor = []  # [{name, kurtosis, h_diag_norm, h_diag}]
    with safe_open(str(sf_path), framework="numpy") as sf:
        for block_idx in range(N_BLOCKS):
            for tensor_type in BLOCK_TENSOR_TYPES:
                hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                tensor_np = sf.get_tensor(hf_name).astype(np.float32)
                kurt = float(scipy_kurtosis(tensor_np.ravel(), fisher=True))

                module_path = hf_name.replace(".weight", "")
                h_diag = collector.get_diag(module_path)
                h_diag_norm = collector.get_diag_norm(module_path)

                per_tensor.append({
                    "name": hf_name,
                    "block_idx": block_idx,
                    "tensor_type": tensor_type,
                    "kurtosis": kurt,
                    "h_diag_norm": h_diag_norm if h_diag_norm is not None else 0.0,
                    "h_diag": h_diag,
                    "shape": tensor_np.shape,
                })
                del tensor_np

    # ── Spearman rho: kurtosis vs H_diag_norm ──
    kurtoses = [t["kurtosis"] for t in per_tensor]
    h_norms = [t["h_diag_norm"] for t in per_tensor]
    rho, p_value = spearmanr(kurtoses, h_norms)
    print(f"  Spearman rho (kurtosis vs H_diag_norm): {rho:.4f}, p={p_value:.2e}")
    print(f"  Kurtosis range: [{min(kurtoses):.2f}, {max(kurtoses):.2f}]")
    print(f"  H_diag_norm range: [{min(h_norms):.4f}, {max(h_norms):.4f}]")

    # ── Determine Hessian routing threshold ──
    # Use same routing fraction as kurtosis: count how many tensors kurtosis>5
    n_kurtosis_svd = sum(1 for k in kurtoses if k > 5)
    # Sort by H_diag_norm descending, pick threshold that routes same count
    sorted_norms = sorted(h_norms, reverse=True)
    if n_kurtosis_svd > 0 and n_kurtosis_svd <= len(sorted_norms):
        h_threshold = sorted_norms[n_kurtosis_svd - 1]
    else:
        h_threshold = sorted_norms[0] + 1  # route nothing

    n_hessian_svd = sum(1 for h in h_norms if h >= h_threshold)
    print(f"  Kurtosis routes {n_kurtosis_svd} tensors to SVD r8")
    print(f"  Hessian routes {n_hessian_svd} tensors to SVD r8 "
          f"(threshold={h_threshold:.4f})")

    # Show which tensors differ between routing strategies
    kurtosis_set = set()
    hessian_set = set()
    for t in per_tensor:
        if t["kurtosis"] > 5:
            kurtosis_set.add(t["name"])
        if t["h_diag_norm"] >= h_threshold:
            hessian_set.add(t["name"])
    only_kurtosis = kurtosis_set - hessian_set
    only_hessian = hessian_set - kurtosis_set
    if only_kurtosis or only_hessian:
        print(f"  Routing differences:")
        for name in sorted(only_kurtosis):
            t = next(x for x in per_tensor if x["name"] == name)
            print(f"    KURTOSIS-only: {name} (kurt={t['kurtosis']:.1f}, h_norm={t['h_diag_norm']:.4f})")
        for name in sorted(only_hessian):
            t = next(x for x in per_tensor if x["name"] == name)
            print(f"    HESSIAN-only:  {name} (kurt={t['kurtosis']:.1f}, h_norm={t['h_diag_norm']:.4f})")

    # ── Strategy A: Naive VQ ──
    print(f"\n[6/9] Strategy A: Naive VQ...")
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
                del tensor_np

        model_a = swap_to_helix(model_a, helix_a)
        ppl_a, _, _ = compute_perplexity(model_a, eval_tokens)
        print(f"  PPL: {ppl_a:.4f}, mean cosine: {np.mean(cosines_a):.6f}")
        del model_a; gc.collect()

    # ── Strategy B: Kurtosis-routed VQ ──
    print(f"[7/9] Strategy B: Kurtosis-routed VQ...")
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
                del tensor_np

        model_b = swap_to_helix(model_b, helix_b)
        ppl_b, _, _ = compute_perplexity(model_b, eval_tokens)
        print(f"  PPL: {ppl_b:.4f}, mean cosine: {np.mean(cosines_b):.6f}, "
              f"SVD tensors: {svd_count_b}, SVD bytes: {svd_bytes_b}")
        del model_b; gc.collect()

    # ── Strategy C: Hessian-routed VQ (same k-means, different routing) ──
    print(f"[8/9] Strategy C: Hessian-routed VQ...")
    with safe_open(str(sf_path), framework="numpy") as sf:
        tmpdir_c = Path(tempfile.mkdtemp())
        writer_c = CDNAv3Writer(tmpdir_c)
        model_c = copy.deepcopy(model)
        helix_c = {}
        cosines_c = []
        svd_count_c = 0
        svd_bytes_c = 0

        for t in per_tensor:
            hf_name = t["name"]
            block_idx = t["block_idx"]
            tensor_type = t["tensor_type"]
            tensor_np = sf.get_tensor(hf_name).astype(np.float32)

            tc = classify_tensor(hf_name, shape=tensor_np.shape)
            base_policy = get_default_policy(tc)

            # Route by Hessian diagonal norm instead of kurtosis
            if t["h_diag_norm"] >= h_threshold:
                from dataclasses import replace
                policy = replace(base_policy, svd_residual_rank=8)
            else:
                policy = base_policy

            stats = writer_c.write_tensor(tensor_np, hf_name, policy=policy)
            cosines_c.append(stats.get("cosine_with_svd",
                                       stats.get("cosine_with_sidecar", 0)))
            if stats.get("svd_bytes", 0) > 0:
                svd_count_c += 1
                svd_bytes_c += stats["svd_bytes"]

            safe_name = hf_name.replace("/", "_").replace(".", "_")
            tensor_dir = tmpdir_c / f"{safe_name}.cdnav3"
            module_path = hf_name.replace(".weight", "")
            mod = get_module(model_c, block_idx, tensor_type)
            bias = mod.bias.data.clone() if mod.bias is not None else None
            helix_c[module_path] = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
            del tensor_np

        model_c = swap_to_helix(model_c, helix_c)
        ppl_c, _, _ = compute_perplexity(model_c, eval_tokens)
        print(f"  PPL: {ppl_c:.4f}, mean cosine: {np.mean(cosines_c):.6f}, "
              f"SVD tensors: {svd_count_c}, SVD bytes: {svd_bytes_c}")
        del model_c; gc.collect()

    # ── Strategy D: Hessian-weighted k-means + Hessian routing ──
    print(f"[9/9] Strategy D: Hessian-weighted k-means + Hessian routing...")
    with safe_open(str(sf_path), framework="numpy") as sf:
        tmpdir_d = Path(tempfile.mkdtemp())
        model_d = copy.deepcopy(model)
        helix_d = {}
        cosines_d = []
        svd_count_d = 0
        svd_bytes_d = 0

        for t in per_tensor:
            hf_name = t["name"]
            block_idx = t["block_idx"]
            tensor_type = t["tensor_type"]
            tensor_np = sf.get_tensor(hf_name).astype(np.float32)
            h_diag = t["h_diag"]

            safe_name = hf_name.replace("/", "_").replace(".", "_")
            out_dir = tmpdir_d / f"{safe_name}.cdnav3"

            if h_diag is not None and len(h_diag) == tensor_np.shape[1]:
                # Weighted VQ encode
                result = weighted_vq_encode(tensor_np, h_diag, n_clusters=256)
                cosines_d.append(result["cosine"])

                write_cdnav3_manually(
                    out_dir, hf_name,
                    result["codebook"], result["indices"],
                    tensor_np.shape,
                )

                # Also handle SVD routing
                if t["h_diag_norm"] >= h_threshold:
                    # Add SVD residual on top of weighted VQ
                    residual = tensor_np - result["reconstructed"]
                    try:
                        U, S, Vt = np.linalg.svd(residual, full_matrices=False)
                        rank = 8
                        U8 = U[:, :rank].astype(np.float32)
                        S8 = S[:rank].astype(np.float32)
                        V8 = Vt[:rank, :].astype(np.float32)
                        np.save(out_dir / "svd_U.npy", U8)
                        np.save(out_dir / "svd_s.npy", S8)
                        np.save(out_dir / "svd_Vt.npy", V8)

                        # Update meta
                        meta = json.loads((out_dir / "meta.json").read_text())
                        meta["svd_residual_rank"] = rank
                        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

                        # Recompute cosine with SVD correction
                        svd_correction = (U8 * S8) @ V8
                        recon_svd = result["reconstructed"] + svd_correction
                        cos_svd = float(_cosine(tensor_np.ravel(), recon_svd.ravel()))
                        cosines_d[-1] = cos_svd

                        svd_bytes = ((out_dir / "svd_U.npy").stat().st_size +
                                     (out_dir / "svd_s.npy").stat().st_size +
                                     (out_dir / "svd_Vt.npy").stat().st_size)
                        svd_count_d += 1
                        svd_bytes_d += svd_bytes

                        # Update stats
                        stats = json.loads((out_dir / "stats.json").read_text())
                        stats["svd_residual_rank"] = rank
                        stats["svd_bytes"] = svd_bytes
                        stats["cosine_with_svd"] = round(cos_svd, 6)
                        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
                    except Exception as e:
                        print(f"    SVD failed for {hf_name}: {e}")
            else:
                # Fallback: standard VQ via CDNAv3Writer
                writer_d = CDNAv3Writer(tmpdir_d)
                tc = classify_tensor(hf_name, shape=tensor_np.shape)
                policy = get_default_policy(tc)
                stats = writer_d.write_tensor(tensor_np, hf_name, policy=policy)
                cosines_d.append(stats.get("cosine_with_sidecar", 0))

            # Load as HelixLinear
            module_path = hf_name.replace(".weight", "")
            mod = get_module(model_d, block_idx, tensor_type)
            bias = mod.bias.data.clone() if mod.bias is not None else None
            helix_d[module_path] = load_helix_linear_from_cdnav3(out_dir, bias=bias)
            del tensor_np

            if (per_tensor.index(t) + 1) % 22 == 0:
                print(f"  Tensor {per_tensor.index(t) + 1}/{len(per_tensor)}", flush=True)

        model_d = swap_to_helix(model_d, helix_d)
        ppl_d, _, _ = compute_perplexity(model_d, eval_tokens)
        print(f"  PPL: {ppl_d:.4f}, mean cosine: {np.mean(cosines_d):.6f}, "
              f"SVD tensors: {svd_count_d}, SVD bytes: {svd_bytes_d}")
        del model_d; gc.collect()

    # ── Results ──
    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    print(f"\n{'=' * 70}")
    print("RESULTS — WO-GPTQ-HELIX-HYBRID-02")
    print(f"{'=' * 70}")
    print(f"  Baseline PPL: {ppl_baseline:.4f}")
    print(f"  Spearman rho (kurtosis vs H_diag_norm): {rho:.4f}, p={p_value:.2e}\n")

    strategies = [
        ("A: Naive VQ", ppl_a, np.mean(cosines_a), 0, 0),
        ("B: Kurtosis-routed", ppl_b, np.mean(cosines_b), svd_count_b, svd_bytes_b),
        ("C: Hessian-routed", ppl_c, np.mean(cosines_c), svd_count_c, svd_bytes_c),
        ("D: Hessian-weighted+routed", ppl_d, np.mean(cosines_d), svd_count_d, svd_bytes_d),
    ]

    print(f"  {'Strategy':<30} {'PPL':>8} {'Δ%':>8} {'Cosine':>10} {'SVD#':>5} {'SVD MB':>7}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*5} {'-'*7}")
    for name, ppl, cos, svd_n, svd_b in strategies:
        delta_pct = 100 * (ppl - ppl_baseline) / ppl_baseline
        print(f"  {name:<30} {ppl:>8.4f} {delta_pct:>+7.2f}% {cos:>10.6f} {svd_n:>5} {svd_b/1e6:>7.2f}")

    # ── Verdict ──
    print()
    best_name = min(strategies, key=lambda x: x[1])[0]
    best_ppl = min(s[1] for s in strategies)
    gap_cd = ppl_d - ppl_c
    gap_db = ppl_d - ppl_b
    gap_cb = ppl_c - ppl_b

    if ppl_d < ppl_b - 0.01:
        verdict = "HESSIAN_WEIGHTED_WINS"
        explanation = (f"Strategy D (weighted k-means + Hessian routing) beats "
                       f"kurtosis routing by {-gap_db:.4f} PPL.")
    elif ppl_c < ppl_b - 0.01:
        verdict = "HESSIAN_ROUTING_WINS"
        explanation = (f"Strategy C (Hessian routing) beats kurtosis routing "
                       f"by {-gap_cb:.4f} PPL. Weighted k-means doesn't add value.")
    elif abs(gap_db) <= 0.01 and abs(gap_cb) <= 0.01:
        verdict = "TIE"
        explanation = (f"All strategies within 0.01 PPL. Kurtosis is a sufficient "
                       f"proxy for Hessian importance. rho={rho:.4f} confirms "
                       f"{'strong' if abs(rho) > 0.5 else 'moderate' if abs(rho) > 0.3 else 'weak'} "
                       f"correlation.")
    else:
        verdict = "KURTOSIS_WINS"
        explanation = (f"Kurtosis routing remains best. Hessian routing gap: "
                       f"{gap_cb:+.4f}, weighted: {gap_db:+.4f}.")

    print(f"  VERDICT: {verdict}")
    print(f"  {explanation}")
    print(f"\n  Wall: {wall:.0f}s, CPU: {cpu:.0f}s")

    # ── Receipt ──
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-tensor detail for correlation analysis
    per_tensor_detail = []
    for t in per_tensor:
        per_tensor_detail.append({
            "name": t["name"],
            "block_idx": t["block_idx"],
            "tensor_type": t["tensor_type"],
            "kurtosis": round(t["kurtosis"], 4),
            "h_diag_norm": round(t["h_diag_norm"], 6),
            "kurtosis_routes_svd": t["kurtosis"] > 5,
            "hessian_routes_svd": t["h_diag_norm"] >= h_threshold,
        })

    receipt = {
        "work_order": "WO-GPTQ-HELIX-HYBRID-02",
        "description": "Hessian diagonal routing + weighted k-means vs kurtosis routing",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "eval_tokens": len(eval_tokens),
        "n_calibration": args.n_calibration,
        "cal_seq_len": args.cal_seq_len,
        "baseline_ppl": round(ppl_baseline, 6),
        "spearman": {
            "rho": round(rho, 6),
            "p_value": float(f"{p_value:.6e}"),
            "description": "kurtosis vs Hessian diagonal norm, per tensor",
        },
        "routing": {
            "h_threshold": round(h_threshold, 6),
            "n_kurtosis_svd": n_kurtosis_svd,
            "n_hessian_svd": n_hessian_svd,
            "only_kurtosis": sorted(only_kurtosis),
            "only_hessian": sorted(only_hessian),
        },
        "strategies": {
            "A_naive_vq": {
                "ppl": round(ppl_a, 6),
                "ppl_delta_pct": round(100 * (ppl_a - ppl_baseline) / ppl_baseline, 4),
                "mean_cosine": round(float(np.mean(cosines_a)), 6),
                "svd_tensors": 0,
                "svd_bytes": 0,
            },
            "B_kurtosis_routed": {
                "ppl": round(ppl_b, 6),
                "ppl_delta_pct": round(100 * (ppl_b - ppl_baseline) / ppl_baseline, 4),
                "mean_cosine": round(float(np.mean(cosines_b)), 6),
                "svd_tensors": svd_count_b,
                "svd_bytes": svd_bytes_b,
            },
            "C_hessian_routed": {
                "ppl": round(ppl_c, 6),
                "ppl_delta_pct": round(100 * (ppl_c - ppl_baseline) / ppl_baseline, 4),
                "mean_cosine": round(float(np.mean(cosines_c)), 6),
                "svd_tensors": svd_count_c,
                "svd_bytes": svd_bytes_c,
            },
            "D_hessian_weighted": {
                "ppl": round(ppl_d, 6),
                "ppl_delta_pct": round(100 * (ppl_d - ppl_baseline) / ppl_baseline, 4),
                "mean_cosine": round(float(np.mean(cosines_d)), 6),
                "svd_tensors": svd_count_d,
                "svd_bytes": svd_bytes_d,
            },
        },
        "verdict": verdict,
        "explanation": explanation,
        "per_tensor": per_tensor_detail,
        "cost": {
            "wall_time_s": round(wall, 3),
            "cpu_time_s": round(cpu, 3),
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
    receipt_path = RECEIPT_DIR / f"hessian_vq_v2_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
