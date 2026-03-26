#!/usr/bin/env python3
"""
WO-GPTQ-HELIX-HYBRID-02 — Strategy D only (rerun after SVD file fix).

Strategies A/B/C already proven across 2 runs (stable results):
  A: PPL 6.2180, B: PPL 6.2030, C: PPL 6.2115

This script runs ONLY Strategy D (Hessian-weighted k-means + Hessian routing)
to avoid re-running the 60+ min A/B/C computation.

Uses same calibration, same thresholds, same methodology as full v2.
"""

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
from helix_substrate.tensor_policy import classify_tensor, get_default_policy

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


class HessianCollector:
    def __init__(self):
        self.hessians = {}
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
            h_diag = (x * x).sum(dim=0)
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
        if name not in self.hessians:
            return None
        h_diag, n = self.hessians[name]
        return (h_diag / n).numpy()

    def get_diag_norm(self, name):
        d = self.get_diag(name)
        if d is None:
            return None
        return float(np.linalg.norm(d))


def weighted_kmeans(data, weights, n_clusters=256, max_iters=10):
    n_clusters = min(n_clusters, len(np.unique(data)))
    percentiles = np.linspace(0, 100, n_clusters)
    centroids = np.percentile(data, percentiles).astype(np.float32)

    for _ in range(max_iters):
        dists = np.abs(data[:, np.newaxis] - centroids)
        assignments = np.argmin(dists, axis=1).astype(np.uint8)

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

    dists = np.abs(data[:, np.newaxis] - centroids)
    assignments = np.argmin(dists, axis=1).astype(np.uint8)
    return centroids, assignments


def weighted_vq_encode(weight, h_diag, n_clusters=256):
    d_out, d_in = weight.shape
    flat = weight.ravel()
    importance = np.tile(h_diag, d_out)
    imp_max = importance.max()
    if imp_max > 0:
        importance = importance / imp_max

    n_sample = min(len(flat), 500_000)
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(len(flat), n_sample, replace=False)

    codebook, _ = weighted_kmeans(flat[idx_sample], importance[idx_sample],
                                  n_clusters, max_iters=10)
    codebook = codebook.astype(np.float32)

    indices = np.argmin(np.abs(flat[:, None] - codebook), axis=1).astype(np.uint8)
    reconstructed = codebook[indices].reshape(d_out, d_in)
    cos = float(_cosine(weight.ravel(), reconstructed.ravel()))

    return {"codebook": codebook, "indices": indices, "cosine": cos,
            "reconstructed": reconstructed}


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
    return float(np.exp(sum(nlls) / n_tokens))


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("WO-GPTQ-HELIX-HYBRID-02 — Strategy D Only (SVD fix rerun)")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open

    print("[1/5] Loading model + data...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
    model.eval()

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_tokens = tokenizer.encode(text)[:4096]

    ppl_baseline = compute_perplexity(model, eval_tokens)
    print(f"  Baseline PPL: {ppl_baseline:.4f}")

    print("[2/5] Collecting Hessian diagonals (64 × 512)...")
    collector = HessianCollector()
    collector.register(model)

    cal_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    cal_text = "\n\n".join([t for t in cal_ds["text"] if t.strip()])
    cal_tokens = tokenizer.encode(cal_text)

    for i in range(64):
        start = np.random.randint(0, max(1, len(cal_tokens) - 512))
        seq = cal_tokens[start:start + 512]
        input_ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            model(input_ids)
        if (i + 1) % 16 == 0:
            print(f"  Calibration: {i + 1}/64", flush=True)

    collector.remove_hooks()

    print("[3/5] Computing per-tensor stats...")
    sf_path = MODEL_DIR / "model.safetensors"
    per_tensor = []
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
                    "name": hf_name, "block_idx": block_idx,
                    "tensor_type": tensor_type, "kurtosis": kurt,
                    "h_diag_norm": h_diag_norm or 0.0,
                    "h_diag": h_diag, "shape": tensor_np.shape,
                })
                del tensor_np

    kurtoses = [t["kurtosis"] for t in per_tensor]
    h_norms = [t["h_diag_norm"] for t in per_tensor]
    rho, p_value = spearmanr(kurtoses, h_norms)

    n_kurtosis_svd = sum(1 for k in kurtoses if k > 5)
    sorted_norms = sorted(h_norms, reverse=True)
    h_threshold = sorted_norms[n_kurtosis_svd - 1] if n_kurtosis_svd > 0 else sorted_norms[0] + 1

    print(f"  rho={rho:.4f}, threshold={h_threshold:.4f}")
    print(f"  Kurtosis routes {n_kurtosis_svd}, Hessian routes {sum(1 for h in h_norms if h >= h_threshold)}")

    print("[4/5] Strategy D: Hessian-weighted k-means + routing...")
    with safe_open(str(sf_path), framework="numpy") as sf:
        tmpdir_d = Path(tempfile.mkdtemp())
        model_d = copy.deepcopy(model)
        helix_d = {}
        cosines_d = []
        svd_count_d = 0
        svd_bytes_d = 0

        for ti, t in enumerate(per_tensor):
            hf_name = t["name"]
            block_idx = t["block_idx"]
            tensor_type = t["tensor_type"]
            tensor_np = sf.get_tensor(hf_name).astype(np.float32)
            h_diag = t["h_diag"]

            safe_name = hf_name.replace("/", "_").replace(".", "_")
            out_dir = tmpdir_d / f"{safe_name}.cdnav3"

            if h_diag is not None and len(h_diag) == tensor_np.shape[1]:
                result = weighted_vq_encode(tensor_np, h_diag, n_clusters=256)
                cosines_d.append(result["cosine"])

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
                }

                stats = {
                    "tensor_name": hf_name,
                    "shape": [rows, cols],
                    "original_bytes": int(np.prod(tensor_np.shape) * 4),
                    "compressed_bytes": 0,
                    "cosine_no_sidecar": round(result["cosine"], 6),
                    "cosine_with_sidecar": round(result["cosine"], 6),
                    "num_outliers": 0, "sidecar_bytes": 0,
                    "svd_residual_rank": 0, "svd_bytes": 0,
                }

                if t["h_diag_norm"] >= h_threshold:
                    residual = tensor_np - result["reconstructed"]
                    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
                    rank = 8
                    U8 = U[:, :rank].astype(np.float32)
                    S8 = S[:rank].astype(np.float32)
                    V8 = Vt[:rank, :].astype(np.float32)
                    np.save(out_dir / "svd_U.npy", U8)
                    np.save(out_dir / "svd_s.npy", S8)
                    np.save(out_dir / "svd_Vt.npy", V8)
                    meta["svd_residual_rank"] = rank

                    svd_correction = (U8 * S8) @ V8
                    recon_svd = result["reconstructed"] + svd_correction
                    cos_svd = float(_cosine(tensor_np.ravel(), recon_svd.ravel()))
                    cosines_d[-1] = cos_svd

                    svd_bytes = ((out_dir / "svd_U.npy").stat().st_size +
                                 (out_dir / "svd_s.npy").stat().st_size +
                                 (out_dir / "svd_Vt.npy").stat().st_size)
                    svd_count_d += 1
                    svd_bytes_d += svd_bytes
                    stats["svd_residual_rank"] = rank
                    stats["svd_bytes"] = svd_bytes

                (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
            else:
                writer_d = CDNAv3Writer(tmpdir_d)
                tc = classify_tensor(hf_name, shape=tensor_np.shape)
                policy = get_default_policy(tc)
                stats = writer_d.write_tensor(tensor_np, hf_name, policy=policy)
                cosines_d.append(stats.get("cosine_with_sidecar", 0))

            module_path = hf_name.replace(".weight", "")
            mod = get_module(model_d, block_idx, tensor_type)
            bias = mod.bias.data.clone() if mod.bias is not None else None
            helix_d[module_path] = load_helix_linear_from_cdnav3(out_dir, bias=bias)
            del tensor_np

            if (ti + 1) % 22 == 0:
                print(f"  Tensor {ti + 1}/{len(per_tensor)}", flush=True)

        model_d = swap_to_helix(model_d, helix_d)
        ppl_d = compute_perplexity(model_d, eval_tokens)
        print(f"  PPL D: {ppl_d:.4f}, cosine: {np.mean(cosines_d):.6f}, "
              f"SVD: {svd_count_d} tensors, {svd_bytes_d} bytes")
        del model_d; gc.collect()

    # ── Combined results ──
    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    # Hardcoded A/B/C from two stable runs
    ppl_a = 6.2180
    ppl_b = 6.2030
    ppl_c = 6.2115

    print(f"\n{'=' * 70}")
    print("COMBINED RESULTS — WO-GPTQ-HELIX-HYBRID-02")
    print(f"{'=' * 70}")
    print(f"  Baseline: {ppl_baseline:.4f}")
    print(f"  rho(kurtosis, H_diag_norm) = {rho:.4f}, p={p_value:.2e}\n")

    strategies = [
        ("A: Naive VQ", ppl_a),
        ("B: Kurtosis-routed", ppl_b),
        ("C: Hessian-routed", ppl_c),
        ("D: Hessian-weighted+routed", ppl_d),
    ]

    print(f"  {'Strategy':<30} {'PPL':>8} {'Δ%':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for name, ppl in strategies:
        delta = 100 * (ppl - ppl_baseline) / ppl_baseline
        print(f"  {name:<30} {ppl:>8.4f} {delta:>+7.2f}%")

    gap_db = ppl_d - ppl_b
    if ppl_d < ppl_b - 0.01:
        verdict = "HESSIAN_WEIGHTED_WINS"
    elif abs(gap_db) <= 0.01:
        verdict = "TIE"
    else:
        verdict = "KURTOSIS_WINS"

    print(f"\n  VERDICT: {verdict}")
    print(f"  D vs B gap: {gap_db:+.4f}")

    # [5/5] Receipt
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "work_order": "WO-GPTQ-HELIX-HYBRID-02",
        "description": "Hessian diagonal routing + weighted k-means vs kurtosis routing",
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "eval_tokens": 4096,
        "baseline_ppl": round(ppl_baseline, 6),
        "spearman_rho": round(rho, 6),
        "spearman_p": float(f"{p_value:.6e}"),
        "h_threshold": round(h_threshold, 6),
        "strategies": {
            "A_naive_vq": {"ppl": round(ppl_a, 6), "source": "prior_run"},
            "B_kurtosis_routed": {"ppl": round(ppl_b, 6), "source": "prior_run"},
            "C_hessian_routed": {"ppl": round(ppl_c, 6), "source": "prior_run"},
            "D_hessian_weighted": {
                "ppl": round(ppl_d, 6),
                "mean_cosine": round(float(np.mean(cosines_d)), 6),
                "svd_tensors": svd_count_d,
                "svd_bytes": svd_bytes_d,
                "source": "this_run",
            },
        },
        "verdict": verdict,
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
    receipt_path = RECEIPT_DIR / f"hessian_vq_v2_d_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
