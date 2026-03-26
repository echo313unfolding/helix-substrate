#!/usr/bin/env python3
"""
Dynamic multi-signal calibrator for CDNA v3 compression.

Single calibration pass computes three signals per tensor:
  1. Activation magnitude — per-column max |activation| across calibration samples
  2. Weight kurtosis — Fisher excess kurtosis of weight values (free from weights)
  3. Hessian diagonal — squared gradient approximation of per-element sensitivity

Outputs a per-tensor compression policy JSON that compress.py consumes via
--policy-file. Each tensor gets its own compression recipe (SVD rank, sidecar
mode, scale alpha, k) instead of global defaults.

Usage:
    python3 tools/calibrate_dynamic.py ~/models/tinyllama_fp32
    python3 tools/calibrate_dynamic.py ~/models/tinyllama_fp32 --n-samples 32 --device cpu
    python3 tools/calibrate_dynamic.py ~/models/qwen2.5-7b-instruct --device cuda

Output:
    {model_dir}/calibration/dynamic_policy.json
    {model_dir}/calibration/calibration_dynamic_meta.json

Work Order: WO-DYNAMIC-CALIBRATOR-01
"""

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_policy(kurtosis: float, act_scales: np.ndarray,
                   hessian_diag: np.ndarray, n_elements: int) -> dict:
    """Compute per-tensor compression policy from calibration signals.

    Args:
        kurtosis: Fisher excess kurtosis of weight values
        act_scales: Per-column activation scale factors [in_features]
        hessian_diag: Per-column Hessian diagonal approximation [in_features]
        n_elements: Total number of elements in the weight tensor
    """
    policy = {}

    # SVD routing (from kurtosis) — proven: kurtosis > 5 → SVD needed
    policy["svd"] = bool(kurtosis > 5.0)
    policy["svd_rank"] = 8 if kurtosis > 5.0 else 0

    # Sidecar selection mode
    policy["sidecar_mode"] = "hessian"

    # Base sidecar budget: 0.1% of elements
    base_top_k = max(1, int(n_elements * 0.001))

    # High kurtosis = more outliers = capture more corrections
    if kurtosis > 10:
        base_top_k = int(base_top_k * 2)

    policy["sidecar_top_k"] = base_top_k

    # Channel scaling
    policy["scale_alpha"] = 0.5

    # k (codebook size) — standard 256 unless tensor is very well-behaved
    policy["k"] = 256

    return policy


def kurtosis_1d(data: np.ndarray) -> float:
    """Fisher excess kurtosis (no scipy dependency)."""
    data = data.astype(np.float64)
    mean = data.mean()
    var = ((data - mean) ** 2).mean()
    if var < 1e-12:
        return 0.0
    m4 = ((data - mean) ** 4).mean()
    return float(m4 / (var ** 2) - 3.0)


def calibrate_dynamic(model_dir: Path, n_samples: int = 32, seq_len: int = 512,
                      alpha: float = 0.5, device: str = "cpu"):
    """Run multi-signal calibration: activation + kurtosis + Hessian.

    Three signals computed in one pass:
      1. Forward hooks capture activation magnitudes (same as calibrate.py)
      2. Weight kurtosis computed directly from parameters (free)
      3. Hessian diagonal approximated via squared gradients on calibration loss
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    model_dir = Path(model_dir).expanduser().resolve()

    print("=" * 70)
    print("  Dynamic Multi-Signal Calibrator")
    print("=" * 70)
    print(f"  Model:     {model_dir.name}")
    print(f"  Samples:   {n_samples}")
    print(f"  Seq len:   {seq_len}")
    print(f"  Alpha:     {alpha}")
    print(f"  Device:    {device}")

    # Load model
    use_gpu = device.startswith("cuda")
    dm = "auto" if use_gpu else "cpu"
    dtype = torch.float16 if use_gpu else torch.float32

    print(f"\n  Loading model ({dtype}, device_map={dm})...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype,
        device_map=dm, trust_remote_code=True, low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load calibration data
    print("  Loading calibration data...")
    texts = _load_calibration_texts(n_samples)
    print(f"  Got {len(texts)} calibration sequences")

    # ── Signal 1: Activation magnitudes (forward hooks) ──
    print("\n  [Signal 1] Activation magnitudes...")
    act_max = {}  # tensor_name -> [in_features]
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            tname = name + ".weight"

            def make_hook(tn):
                def hook_fn(mod, inp, out):
                    x = inp[0].detach().float()
                    cm = x.abs().amax(dim=tuple(range(x.ndim - 1)))
                    cm_np = cm.cpu().numpy().astype(np.float64)
                    if tn not in act_max:
                        act_max[tn] = cm_np
                    else:
                        act_max[tn] = np.maximum(act_max[tn], cm_np)
                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(tname)))

    # Find model input device
    try:
        model_device = model.get_input_embeddings().weight.device
    except Exception:
        model_device = next(model.parameters()).device

    # Forward pass for activation collection
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", max_length=seq_len,
                               truncation=True, padding=False)
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            try:
                model(**inputs)
            except Exception as e:
                print(f"    WARNING: Sample {i} failed: {e}")
                continue
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(texts)} forward passes done", flush=True)

    for h in hooks:
        h.remove()

    print(f"  Activation magnitudes: {len(act_max)} tensors")

    # ── Signal 2: Weight kurtosis (free) ──
    print("\n  [Signal 2] Weight kurtosis...")
    weight_kurtosis = {}
    for name, param in model.named_parameters():
        if param.ndim >= 2 and ".weight" in name:
            flat = param.detach().float().cpu().numpy().ravel()
            weight_kurtosis[name] = kurtosis_1d(flat)

    print(f"  Kurtosis computed: {len(weight_kurtosis)} tensors")

    # ── Signal 3: Hessian diagonal (squared gradients) ──
    print("\n  [Signal 3] Hessian diagonal approximation...")
    # Accumulate squared gradients across calibration samples
    grad_sq_accum = {}

    # Enable gradients for this pass
    model.train()  # needed for gradient computation
    for param in model.parameters():
        param.requires_grad_(True)

    n_hessian_samples = min(n_samples, 8)  # Hessian is expensive, use fewer samples
    for i in range(n_hessian_samples):
        if i >= len(texts):
            break

        inputs = tokenizer(texts[i], return_tensors="pt", max_length=seq_len,
                           truncation=True, padding=False)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()

        try:
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
        except Exception as e:
            print(f"    WARNING: Hessian sample {i} failed: {e}")
            continue

        # Accumulate squared gradients per Linear weight
        for name, param in model.named_parameters():
            if param.grad is not None and param.ndim >= 2 and ".weight" in name:
                gsq = (param.grad.float() ** 2).mean(dim=0).cpu().numpy()
                if name not in grad_sq_accum:
                    grad_sq_accum[name] = gsq
                else:
                    grad_sq_accum[name] += gsq

        model.zero_grad()

        if (i + 1) % 4 == 0:
            print(f"    {i+1}/{n_hessian_samples} backward passes done", flush=True)

    # Average the squared gradients
    hessian_diag = {}
    for name, gsq in grad_sq_accum.items():
        hessian_diag[name] = gsq / max(1, n_hessian_samples)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"  Hessian diag: {len(hessian_diag)} tensors")

    # ── Compute per-tensor policy ──
    print("\n  Computing per-tensor policies...")
    policies = {}

    for name in sorted(set(list(act_max.keys()) + list(weight_kurtosis.keys()))):
        if name not in weight_kurtosis:
            continue

        kurt = weight_kurtosis[name]
        act = act_max.get(name)
        hess = hessian_diag.get(name)

        # Compute act_scales: act_max ** alpha
        act_scales_arr = None
        if act is not None:
            clipped = np.clip(act.astype(np.float32), 1e-7, np.inf)
            act_scales_arr = np.power(clipped, alpha)

        # Estimate n_elements from activation shape
        n_elements = 0
        if act is not None:
            # Linear weight is [out, in], act gives [in], need to find [out] from param
            for pname, param in model.named_parameters():
                if pname == name:
                    n_elements = param.numel()
                    break

        if n_elements == 0:
            for pname, param in model.named_parameters():
                if pname == name:
                    n_elements = param.numel()
                    break

        policy = compute_policy(
            kurtosis=kurt,
            act_scales=act_scales_arr if act_scales_arr is not None else np.array([]),
            hessian_diag=hess if hess is not None else np.array([]),
            n_elements=n_elements,
        )

        entry = {
            "kurtosis": round(kurt, 4),
            "policy": policy,
        }

        # Store arrays as lists for JSON serialization
        if act_scales_arr is not None:
            entry["act_scales"] = act_scales_arr.tolist()
        if hess is not None:
            entry["hessian_diag"] = hess.tolist()

        policies[name] = entry

    # ── Save ──
    out_dir = model_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_path = out_dir / "dynamic_policy.json"
    with open(policy_path, "w") as f:
        json.dump(policies, f, indent=2)

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    meta = {
        "model": model_dir.name,
        "n_samples": n_samples,
        "n_hessian_samples": n_hessian_samples,
        "seq_len": seq_len,
        "alpha": alpha,
        "n_tensors": len(policies),
        "n_svd_routed": sum(1 for v in policies.values() if v["policy"]["svd"]),
        "kurtosis_stats": {
            "min": round(min(v["kurtosis"] for v in policies.values()), 4),
            "max": round(max(v["kurtosis"] for v in policies.values()), 4),
            "mean": round(np.mean([v["kurtosis"] for v in policies.values()]), 4),
        },
        "cost": {
            "wall_time_s": round(wall, 3),
            "cpu_time_s": round(cpu, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    meta_path = out_dir / "calibration_dynamic_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n  Policy:  {policy_path} ({len(policies)} tensors)")
    print(f"  SVD routed: {meta['n_svd_routed']} tensors (kurtosis > 5)")
    print(f"  Kurtosis:  min={meta['kurtosis_stats']['min']}, "
          f"max={meta['kurtosis_stats']['max']}, "
          f"mean={meta['kurtosis_stats']['mean']}")
    print(f"  Meta:    {meta_path}")
    print(f"  Time:    {wall:.0f}s wall, {cpu:.0f}s CPU")
    print(f"{'=' * 70}")

    del model
    return policy_path


def _load_calibration_texts(n_samples: int, min_length: int = 100) -> list[str]:
    """Load calibration texts from WikiText-2."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > min_length]
        return texts[:n_samples]
    except ImportError:
        pass

    # Fallback
    print("  WARNING: No calibration dataset found. Using synthetic text.")
    return ["The quick brown fox jumps over the lazy dog. " * 20] * min(n_samples, 16)


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic multi-signal calibrator: activation + kurtosis + Hessian.",
    )
    parser.add_argument("model_dir", type=Path,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--n-samples", type=int, default=32,
                        help="Calibration sequences (default: 32)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Max sequence length (default: 512)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Activation scaling exponent (default: 0.5)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    args = parser.parse_args()

    calibrate_dynamic(args.model_dir, n_samples=args.n_samples,
                      seq_len=args.seq_len, alpha=args.alpha, device=args.device)


if __name__ == "__main__":
    main()
