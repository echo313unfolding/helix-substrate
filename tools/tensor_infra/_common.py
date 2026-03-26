"""Shared utilities for tensor infrastructure toy experiments."""
import json, time, resource, platform, shutil
from pathlib import Path
import numpy as np

HELIX_ROOT = Path(__file__).resolve().parent.parent.parent  # helix-substrate/
RECEIPT_BASE = HELIX_ROOT / "receipts" / "tensor_infra"
MODEL_DIR = Path("/home/voidstr3m33/models/tinyllama_fp32")

# ── Add helix_substrate to path ──
import sys
sys.path.insert(0, str(HELIX_ROOT))

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.cdnav3_reader import CDNAv3Reader
from helix_substrate.tensor_policy import TensorPolicy, TensorClass

# ── Cost Tracking (WO-RECEIPT-COST-01) ──
def start_cost():
    return time.time(), time.process_time(), time.strftime('%Y-%m-%dT%H:%M:%S')

def finish_cost(t_start, cpu_start, start_iso):
    return {
        'wall_time_s': round(time.time() - t_start, 3),
        'cpu_time_s': round(time.process_time() - cpu_start, 3),
        'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
        'timestamp_start': start_iso,
        'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

# ── Metrics ──
def cosine_sim(a, b):
    a_f, b_f = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    dot = np.dot(a_f, b_f)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0

def mse(a, b):
    return float(np.mean((a.ravel().astype(np.float64) - b.ravel().astype(np.float64)) ** 2))

def max_abs_err(a, b):
    return float(np.max(np.abs(a.ravel().astype(np.float64) - b.ravel().astype(np.float64))))

def kurtosis(a):
    """Excess kurtosis (Fisher definition)."""
    a_f = a.ravel().astype(np.float64)
    mu = np.mean(a_f)
    var = np.var(a_f)
    if var == 0: return 0.0
    return float(np.mean(((a_f - mu) / np.sqrt(var)) ** 4) - 3.0)

# ── Compress + Reconstruct Helper ──
def compress_tensor(tensor_2d, name, out_dir, policy):
    """Compress a 2D float32 tensor. Returns (stats, reconstructed)."""
    out_path = out_dir / name
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    writer = CDNAv3Writer(out_path)
    stats = writer.write_tensor(tensor_2d.astype(np.float32), name, policy=policy)

    tensor_dir = out_path / f"{name}.cdnav3"
    if not tensor_dir.exists():
        return stats, tensor_2d.copy()  # exact storage (1D/tiny)

    reader = CDNAv3Reader(tensor_dir)
    reconstructed = reader.reconstruct()
    return stats, reconstructed

# ── Policies ──
def policy_vq(k=256, sidecar=True, svd_rank=0, percentile=99.9, max_corr=512):
    return TensorPolicy(
        tensor_class=TensorClass.UNKNOWN,
        storage_mode="codebook" if not sidecar else "codebook+sidecar",
        n_clusters=k, use_kmeans=True,
        sidecar_enabled=sidecar, percentile=percentile,
        max_corrections=max_corr, svd_residual_rank=svd_rank,
    )

# ── Receipt Writer ──
def write_receipt(experiment, domain, results, cost, receipt_dir=None):
    if receipt_dir is None:
        receipt_dir = RECEIPT_BASE / domain
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "experiment": experiment,
        "domain": domain,
        "results": results,
        "cost": cost,
    }
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = receipt_dir / f"{domain}_{ts}.json"
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"  Receipt: {path}")
    return path

# ── Model Loader ──
def load_tinyllama_weights():
    """Load TinyLlama FP32 weights as dict of numpy arrays."""
    from safetensors.numpy import load_file
    return load_file(str(MODEL_DIR / "model.safetensors"))

def load_tinyllama_model():
    """Load TinyLlama as a PyTorch model (CPU, FP32)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer

def load_wikitext2_batch(tokenizer, n_tokens=128):
    """Load a batch from WikiText-2 for backward passes."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = " ".join([x["text"] for x in ds if x["text"].strip()])
    tokens = tokenizer(text, return_tensors="pt", max_length=n_tokens, truncation=True)
    return tokens

# ── Verdict Logic ──
def verdict(cosines, strong=0.999, passing=0.99, weak=0.95):
    worst = min(cosines)
    if worst >= strong: return "STRONG PASS", worst
    if worst >= passing: return "PASS", worst
    if worst >= weak: return "WEAK PASS", worst
    return "FAIL", worst
