#!/usr/bin/env python3
"""Domain 1: Gradient Compression for Distributed Training.
Compress real gradients from TinyLlama backward pass on WikiText-2."""

import torch, numpy as np
from pathlib import Path
from _common import *

def main():
    t_start, cpu_start, start_iso = start_cost()
    print("=" * 72)
    print("DOMAIN 1: Gradient Compression for Distributed Training")
    print("=" * 72)

    # Load model + tokenizer
    model, tokenizer = load_tinyllama_model()
    batch = load_wikitext2_batch(tokenizer, n_tokens=128)

    # Forward + backward (no optimizer step)
    model.train()
    input_ids = batch["input_ids"]
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    # Extract gradients for all 2D weight tensors
    out_dir = HELIX_ROOT / "tensor_infra_scratch" / "gradients"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_k256 = policy_vq(k=256, sidecar=True)
    policy_k64 = policy_vq(k=64, sidecar=True)

    results = []
    for name, param in model.named_parameters():
        if param.grad is None or param.grad.ndim != 2:
            continue

        grad_np = param.grad.detach().cpu().numpy().astype(np.float32)
        weight_np = param.data.detach().cpu().numpy().astype(np.float32)
        kurt = kurtosis(grad_np)

        for policy_name, policy in [("vq_k256_sidecar", policy_k256), ("vq_k64_sidecar", policy_k64)]:
            safe_name = name.replace(".", "_")
            stats, recon = compress_tensor(grad_np, safe_name, out_dir / policy_name, policy)

            cos = cosine_sim(grad_np, recon)
            err = mse(grad_np, recon)

            # SGD step comparison: W_new = W - lr * grad
            lr = 0.001
            w_exact = weight_np - lr * grad_np
            w_compressed = weight_np - lr * recon
            step_cos = cosine_sim(w_exact, w_compressed)

            results.append({
                "tensor": name,
                "shape": list(grad_np.shape),
                "kurtosis": round(kurt, 2),
                "codec": policy_name,
                "gradient_cosine": round(cos, 6),
                "mse": round(err, 10),
                "sgd_step_cosine": round(step_cos, 6),
                "compression_ratio": stats.get("compression_ratio", 1.0),
                "original_bytes": grad_np.nbytes,
            })

    # Print summary
    for codec in ["vq_k256_sidecar", "vq_k64_sidecar"]:
        codec_results = [r for r in results if r["codec"] == codec]
        cosines = [r["gradient_cosine"] for r in codec_results]
        step_cosines = [r["sgd_step_cosine"] for r in codec_results]
        v, worst = verdict(cosines)
        print(f"\n  {codec}: {len(codec_results)} tensors")
        print(f"    Gradient cosine: min={min(cosines):.6f}, mean={np.mean(cosines):.6f}")
        print(f"    SGD step cosine: min={min(step_cosines):.6f}, mean={np.mean(step_cosines):.6f}")
        print(f"    Verdict: {v} (worst cosine={worst:.6f})")

    # Total bandwidth savings
    total_orig = sum(r["original_bytes"] for r in results if r["codec"] == "vq_k256_sidecar")
    total_ratio = np.mean([r["compression_ratio"] for r in results if r["codec"] == "vq_k256_sidecar"])
    print(f"\n  Total gradient bytes (original): {total_orig / 1e6:.1f} MB")
    print(f"  Mean compression ratio: {total_ratio:.2f}x")

    cost = finish_cost(t_start, cpu_start, start_iso)
    write_receipt("tensor_infra_domain_1", "gradient_compress", {
        "n_tensors": len([r for r in results if r["codec"] == "vq_k256_sidecar"]),
        "per_tensor": results,
        "total_original_mb": round(total_orig / 1e6, 1),
        "mean_ratio": round(total_ratio, 2),
        "data_source": "REAL — TinyLlama backward on WikiText-2",
    }, cost)

if __name__ == "__main__":
    main()
