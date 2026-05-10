#!/usr/bin/env python3
"""
Layer-wise divergence probe: measure how quantization error propagates through depth.

Hypothesis: SSM recurrence low-pass filters quantization error (2D VQ correlated
pairs align with smoothing), while transformer attention compounds correlated
error across heads.

Test: Run same input through dense and compressed models, measure per-layer
activation divergence (cosine distance). If hypothesis holds:
  - Transformer: 2D VQ divergence grows FASTER with depth than scalar VQ
  - SSM: 2D VQ divergence grows SLOWER with depth than scalar VQ

Usage:
    python3 tools/divergence_probe.py --model ~/models/tinyllama-dense \
        --scalar ~/models/tinyllama-1.1b-helix \
        --vq2d ~/models/tinyllama-vq2d-helix

    # Or with a single model dir that has both compressed variants:
    python3 tools/divergence_probe.py --model ~/models/tinyllama-dense \
        --scalar ~/models/tinyllama-1.1b-helix

Receipt: receipts/divergence_probe/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import resource
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _cosine_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - cosine_similarity between two activation tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    dot = torch.dot(a_flat, b_flat)
    na = torch.norm(a_flat)
    nb = torch.norm(b_flat)
    if na < 1e-30 or nb < 1e-30:
        return 1.0
    cos = (dot / (na * nb)).item()
    return 1.0 - cos


def _l2_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 distance: ||a - b|| / ||a||."""
    diff = (a.float() - b.float()).norm().item()
    ref = a.float().norm().item()
    return diff / max(ref, 1e-30)


def collect_layer_activations(model, tokenizer, text: str, device: str = "cpu"):
    """Run forward pass and collect hidden state after each layer via hooks."""
    activations = {}

    def _make_hook(name):
        def hook(module, input, output):
            # Different architectures return different output types
            if isinstance(output, tuple):
                act = output[0]
            elif isinstance(output, torch.Tensor):
                act = output
            else:
                return
            activations[name] = act.detach().cpu()
        return hook

    hooks = []
    # Register hooks on all "layer" modules (covers transformer blocks, mamba layers)
    for name, module in model.named_modules():
        # Match patterns like model.layers.0, model.backbone.layers.0, etc.
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
            hooks.append(module.register_forward_hook(_make_hook(name)))

    # Run forward
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    return activations


def run_divergence_probe(
    dense_path: str,
    scalar_path: Optional[str] = None,
    vq2d_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: str = "cpu",
    text: str = "The quick brown fox jumps over the lazy dog. In mathematics, a prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. The first few prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23, and 29.",
) -> dict:
    """Run the divergence probe."""
    import warnings
    warnings.filterwarnings("ignore")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Try to register helix quantizer
    try:
        import helix_substrate.hf_quantizer
    except ImportError:
        pass

    # Find a path with tokenizer files — dense models often lack them
    tok_path = tokenizer_path or dense_path
    if not os.path.exists(os.path.join(tok_path, "tokenizer_config.json")):
        for fallback in [scalar_path, vq2d_path]:
            if fallback and os.path.exists(os.path.join(fallback, "tokenizer_config.json")):
                tok_path = fallback
                break
    print(f"Loading tokenizer from {tok_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    results = {"dense_path": dense_path, "text_len": len(text)}

    # Load dense model
    print(f"Loading dense model...", flush=True)
    dense_model = AutoModelForCausalLM.from_pretrained(
        dense_path, torch_dtype=torch.float32
    ).to(device).eval()
    dense_acts = collect_layer_activations(dense_model, tokenizer, text, device)
    layer_names = sorted(dense_acts.keys(), key=lambda n: int(n.split(".")[-1]))
    results["n_layers"] = len(layer_names)
    print(f"  Collected {len(layer_names)} layer activations", flush=True)

    # Get model architecture info
    config = dense_model.config
    results["model_type"] = getattr(config, "model_type", "unknown")
    results["architectures"] = getattr(config, "architectures", [])

    del dense_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _compare(model_path, label):
        print(f"Loading {label} model from {model_path}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32
        ).to(device).eval()
        acts = collect_layer_activations(model, tokenizer, text, device)

        cosine_dists = []
        l2_rels = []
        for name in layer_names:
            if name not in acts:
                cosine_dists.append(None)
                l2_rels.append(None)
                continue
            cd = _cosine_dist(dense_acts[name], acts[name])
            lr = _l2_rel(dense_acts[name], acts[name])
            cosine_dists.append(cd)
            l2_rels.append(lr)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return {
            "path": model_path,
            "cosine_dist_per_layer": [round(x, 8) if x is not None else None for x in cosine_dists],
            "l2_rel_per_layer": [round(x, 8) if x is not None else None for x in l2_rels],
            "cosine_dist_first": cosine_dists[0] if cosine_dists else None,
            "cosine_dist_last": cosine_dists[-1] if cosine_dists else None,
            "cosine_dist_ratio": (
                cosine_dists[-1] / max(cosine_dists[0], 1e-30)
                if cosine_dists and cosine_dists[0] is not None and cosine_dists[-1] is not None
                else None
            ),
        }

    if scalar_path:
        results["scalar"] = _compare(scalar_path, "scalar k=256")
        print(f"  Scalar: layer 0 cos_dist={results['scalar']['cosine_dist_first']:.2e}, "
              f"last={results['scalar']['cosine_dist_last']:.2e}, "
              f"ratio={results['scalar']['cosine_dist_ratio']:.1f}x", flush=True)

    if vq2d_path:
        results["vq2d"] = _compare(vq2d_path, "2D VQ k=4096")
        print(f"  2D VQ:  layer 0 cos_dist={results['vq2d']['cosine_dist_first']:.2e}, "
              f"last={results['vq2d']['cosine_dist_last']:.2e}, "
              f"ratio={results['vq2d']['cosine_dist_ratio']:.1f}x", flush=True)

    # Compare amplification rates
    if scalar_path and vq2d_path:
        sr = results["scalar"]["cosine_dist_ratio"]
        vr = results["vq2d"]["cosine_dist_ratio"]
        if sr and vr:
            if vr > sr:
                results["verdict"] = "2D VQ amplifies MORE (bad for this arch)"
            else:
                results["verdict"] = "2D VQ amplifies LESS (good for this arch)"
            results["amplification_ratio"] = round(vr / sr, 4)
            print(f"\n  Verdict: {results['verdict']}", flush=True)
            print(f"  2D/scalar amplification ratio: {results['amplification_ratio']:.2f}x", flush=True)

    results["layer_names"] = layer_names
    return results


def main():
    parser = argparse.ArgumentParser(description="Layer-wise divergence probe")
    parser.add_argument("--model", required=True, help="Dense model path")
    parser.add_argument("--scalar", default=None, help="Scalar k=256 compressed model path")
    parser.add_argument("--vq2d", default=None, help="2D VQ k=4096 compressed model path")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (default: auto-detect)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    if not args.scalar and not args.vq2d:
        parser.error("At least one of --scalar or --vq2d is required")

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    results = run_divergence_probe(
        dense_path=args.model,
        scalar_path=args.scalar,
        vq2d_path=args.vq2d,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    results["cost"] = cost

    # Save receipt
    receipts_dir = Path(__file__).resolve().parent.parent / "receipts" / "divergence_probe"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model).name
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"divergence_{model_name}_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nReceipt: {receipt_path}", flush=True)
    print(f"Cost: {cost['wall_time_s']}s wall, {cost['peak_memory_mb']} MB peak", flush=True)


if __name__ == "__main__":
    main()
