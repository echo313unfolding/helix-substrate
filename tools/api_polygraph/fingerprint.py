"""Model fingerprint generator — local ground truth for swap detection.

Two modes:
  1. Direct mode: Run HuggingFace model, record logprobs per probe prompt.
     Requires: torch, transformers. No HXQ dependency.

  2. HXQ mode: Run through HelixLinear, add sidecar norm weighting.
     Requires: torch, transformers, helix_substrate.
     Adds: per-layer sidecar norms that weight which tokens are most diagnostic.

The fingerprint is a JSON file containing the model's predicted probability
distribution over each probe prompt. This is the LOCAL GROUND TRUTH that
API responses are compared against.

Usage:
    # Direct mode (no HXQ needed)
    python3 -m api_polygraph.fingerprint --model meta-llama/Llama-3.2-3B-Instruct

    # HXQ mode (adds sidecar weighting)
    python3 -m api_polygraph.fingerprint --model /path/to/hxq/model --hxq
"""

import hashlib
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .probe_corpus import PROBES, CORPUS_VERSION


def _compute_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def generate_fingerprint_direct(
    model_name: str,
    output_path: Optional[str] = None,
    device: str = "auto",
    dtype: str = "float16",
) -> Dict[str, Any]:
    """Generate fingerprint using HuggingFace model directly.

    Runs each probe prompt through the model at temperature=0,
    records the top-k logprobs for each generated token.

    Args:
        model_name: HuggingFace model ID or local path
        output_path: Where to save fingerprint JSON (default: auto)
        device: "auto", "cuda", "cpu"
        dtype: "float16", "bfloat16", "float32"

    Returns:
        Fingerprint dict (also saved to disk)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    t_start = time.time()
    cpu_start = time.process_time()
    ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Load model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )
    if device != "auto":
        model = model.to(device)
    model.eval()

    load_time = time.time() - t_start
    print(f"Model loaded in {load_time:.1f}s")

    # Run probes
    probe_results = []
    for i, (probe_id, prompt, category, max_tokens) in enumerate(PROBES):
        print(f"  Probe {i+1}/{len(PROBES)}: {probe_id}...", end=" ", flush=True)
        probe_start = time.time()

        result = _run_probe_direct(
            model, tokenizer, prompt, max_tokens, device
        )
        result["probe_id"] = probe_id
        result["category"] = category
        result["prompt_hash"] = _compute_prompt_hash(prompt)
        result["elapsed_ms"] = round((time.time() - probe_start) * 1000, 1)

        probe_results.append(result)
        print(f"{result['elapsed_ms']:.0f}ms, {len(result['tokens'])} tokens")

    # Build fingerprint
    fingerprint = {
        "fingerprint_version": "1.0",
        "corpus_version": CORPUS_VERSION,
        "model_declared": model_name,
        "mode": "direct",
        "n_probes": len(probe_results),
        "n_params": sum(p.numel() for p in model.parameters()),
        "probes": probe_results,
        "profile": _compute_profile(probe_results),
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "model_load_s": round(load_time, 3),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": ts_start,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": device,
            "dtype": dtype,
        },
    }

    # Add CUDA stats if available
    if device == "cuda" and torch.cuda.is_available():
        fingerprint["cost"]["cuda_device"] = torch.cuda.get_device_name(0)
        fingerprint["cost"]["cuda_peak_mb"] = round(
            torch.cuda.max_memory_allocated() / (1024 * 1024), 1
        )

    # Save
    if output_path is None:
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        output_path = f"fingerprint_{safe_name}.json"
    with open(output_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    print(f"\nFingerprint saved: {output_path}")

    return fingerprint


def _run_probe_direct(model, tokenizer, prompt, max_tokens, device):
    """Run a single probe through the model, capture logprobs."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    tokens = []
    logprobs = []
    top_k_per_position = []

    with torch.no_grad():
        # Generate one token at a time to capture logprobs
        input_ids = inputs["input_ids"]

        for step in range(max_tokens):
            outputs = model(input_ids)
            next_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            # Softmax to get probabilities
            log_probs = torch.nn.functional.log_softmax(next_logits, dim=-1)

            # Greedy decode (temperature=0)
            next_token_id = next_logits.argmax(dim=-1)

            # Record
            token_str = tokenizer.decode(next_token_id[0])
            token_logprob = float(log_probs[0, next_token_id[0]].item())
            tokens.append(token_str)
            logprobs.append(round(token_logprob, 6))

            # Top-5 alternatives
            top_vals, top_ids = log_probs[0].topk(5)
            top_k = [
                {
                    "token": tokenizer.decode(tid),
                    "logprob": round(float(tv), 6),
                }
                for tid, tv in zip(top_ids.tolist(), top_vals.tolist())
            ]
            top_k_per_position.append(top_k)

            # Stop on EOS
            if next_token_id[0].item() == tokenizer.eos_token_id:
                break

            # Append for next step
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

    return {
        "prompt": prompt,
        "tokens": tokens,
        "logprobs": logprobs,
        "top_k": top_k_per_position,
        "mean_logprob": round(sum(logprobs) / len(logprobs), 6) if logprobs else 0.0,
    }


def generate_fingerprint_hxq(
    model_path: str,
    output_path: Optional[str] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """Generate fingerprint using HXQ-compressed model with sidecar norms.

    Same as direct mode, but also records per-layer sidecar norms.
    These norms weight which tokens are most diagnostic for swap detection:
    - LOW sidecar norm = model is confident = MOST diagnostic
    - HIGH sidecar norm = model is uncertain = LESS diagnostic

    The sidecar weighting is the upgrade over timing-only detection.

    Args:
        model_path: Path to HXQ-compressed model
        output_path: Where to save fingerprint JSON
        device: "auto", "cuda", "cpu"

    Returns:
        Fingerprint dict (also saved to disk)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    t_start = time.time()
    cpu_start = time.process_time()
    ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading HXQ model from {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )
    if device != "auto":
        model = model.to(device)
    model.eval()

    load_time = time.time() - t_start
    print(f"Model loaded in {load_time:.1f}s")

    # Find HelixLinear layers and reset sidecar stats
    helix_layers = _find_helix_layers(model)
    if helix_layers:
        print(f"Found {len(helix_layers)} HelixLinear layers — recording sidecar norms")
        for name, layer in helix_layers:
            layer.set_sidecar_policy("always_on")
            layer.reset_sidecar_policy_stats()
    else:
        print("No HelixLinear layers found — running in direct mode")

    # Run probes
    probe_results = []
    for i, (probe_id, prompt, category, max_tokens) in enumerate(PROBES):
        print(f"  Probe {i+1}/{len(PROBES)}: {probe_id}...", end=" ", flush=True)
        probe_start = time.time()

        # Reset sidecar stats per probe
        for name, layer in helix_layers:
            layer.reset_sidecar_policy_stats()

        result = _run_probe_direct(model, tokenizer, prompt, max_tokens, device)
        result["probe_id"] = probe_id
        result["category"] = category
        result["prompt_hash"] = _compute_prompt_hash(prompt)
        result["elapsed_ms"] = round((time.time() - probe_start) * 1000, 1)

        # Collect sidecar norms
        if helix_layers:
            sidecar_stats = {}
            all_norms = []
            for name, layer in helix_layers:
                stats = layer.get_sidecar_policy_stats()
                ns = stats.get("norm_stats", {})
                if ns.get("n", 0) > 0:
                    sidecar_stats[name] = {
                        "n": ns["n"],
                        "mean": ns["mean"],
                        "p50": ns["p50"],
                        "p95": ns["p95"],
                    }
                    all_norms.append(ns["mean"])

            result["sidecar_stats"] = sidecar_stats
            if all_norms:
                import numpy as np
                mean_norm = float(np.mean(all_norms))
                # Confidence: low norm = high confidence = more diagnostic
                result["sidecar_confidence"] = round(max(0.0, 1.0 - mean_norm), 4)
                # Diagnostic weight: inverse of norm (confident predictions matter more)
                result["diagnostic_weight"] = round(1.0 / (1.0 + mean_norm), 4)

        probe_results.append(result)
        conf_str = f", conf={result.get('sidecar_confidence', 'N/A')}" if helix_layers else ""
        print(f"{result['elapsed_ms']:.0f}ms, {len(result['tokens'])} tokens{conf_str}")

    # Build fingerprint
    fingerprint = {
        "fingerprint_version": "1.0",
        "corpus_version": CORPUS_VERSION,
        "model_declared": model_path,
        "mode": "hxq",
        "n_probes": len(probe_results),
        "n_helix_layers": len(helix_layers),
        "probes": probe_results,
        "profile": _compute_profile(probe_results),
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "model_load_s": round(load_time, 3),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": ts_start,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": device,
        },
    }

    if device == "cuda":
        import torch
        if torch.cuda.is_available():
            fingerprint["cost"]["cuda_device"] = torch.cuda.get_device_name(0)
            fingerprint["cost"]["cuda_peak_mb"] = round(
                torch.cuda.max_memory_allocated() / (1024 * 1024), 1
            )

    # Save
    if output_path is None:
        safe_name = Path(model_path).name
        output_path = f"fingerprint_{safe_name}_hxq.json"
    with open(output_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    print(f"\nFingerprint saved: {output_path}")

    return fingerprint


def _find_helix_layers(model):
    """Find all HelixLinear layers in a model."""
    layers = []
    try:
        from helix_substrate.helix_linear import HelixLinear
        for name, module in model.named_modules():
            if isinstance(module, HelixLinear):
                layers.append((name, module))
    except ImportError:
        pass
    return layers


def _compute_profile(probe_results: List[Dict]) -> Dict:
    """Compute aggregate profile statistics from probe results."""
    all_logprobs = []
    timing = []
    for r in probe_results:
        all_logprobs.extend(r.get("logprobs", []))
        timing.append(r.get("elapsed_ms", 0))

    profile = {
        "n_total_tokens": len(all_logprobs),
        "mean_logprob": round(sum(all_logprobs) / len(all_logprobs), 6) if all_logprobs else 0,
        "mean_probe_ms": round(sum(timing) / len(timing), 1) if timing else 0,
    }

    # Sidecar profile if available
    confidences = [r.get("sidecar_confidence") for r in probe_results if "sidecar_confidence" in r]
    if confidences:
        profile["mean_sidecar_confidence"] = round(sum(confidences) / len(confidences), 4)
        profile["sidecar_weighted"] = True
    else:
        profile["sidecar_weighted"] = False

    return profile


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate model fingerprint for API polygraph"
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--hxq", action="store_true", help="Use HXQ mode with sidecar norms")
    parser.add_argument("--output", help="Output path for fingerprint JSON")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--dtype", default="float16", help="Dtype (float16/bfloat16/float32)")
    args = parser.parse_args()

    if args.hxq:
        generate_fingerprint_hxq(args.model, args.output, args.device)
    else:
        generate_fingerprint_direct(args.model, args.output, args.device, args.dtype)
