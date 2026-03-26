#!/usr/bin/env python3
"""Domain 4: Federated Learning Communication Compression.
Compress model weight deltas from a single SGD step."""

import torch, numpy as np
from pathlib import Path
from _common import *

def main():
    t_start, cpu_start, start_iso = start_cost()
    print("=" * 72)
    print("DOMAIN 4: Federated Learning Delta Compression")
    print("=" * 72)

    model, tokenizer = load_tinyllama_model()
    batch = load_wikitext2_batch(tokenizer, n_tokens=128)

    # Save original weights
    orig_weights = {}
    for name, param in model.named_parameters():
        if param.ndim == 2:
            orig_weights[name] = param.data.clone().cpu().numpy()

    # Single SGD step
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"].clone())
    outputs.loss.backward()
    optimizer.step()

    # Compute deltas
    out_dir = HELIX_ROOT / "tensor_infra_scratch" / "federated"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_k256 = policy_vq(k=256, sidecar=True)
    policy_k64 = policy_vq(k=64, sidecar=True)  # deltas cluster near zero → fewer centroids needed

    results = []
    total_delta_bytes = 0
    total_comp_bytes_k256 = 0
    total_comp_bytes_k64 = 0

    for name, param in model.named_parameters():
        if name not in orig_weights:
            continue
        new_w = param.data.detach().cpu().numpy()
        delta = (new_w - orig_weights[name]).astype(np.float32)
        kurt = kurtosis(delta)
        total_delta_bytes += delta.nbytes

        # Delta sparsity: % of values near zero
        sparsity = float(np.mean(np.abs(delta) < 1e-8))

        for policy_name, policy in [("k256", policy_k256), ("k64", policy_k64)]:
            safe_name = name.replace(".", "_")
            stats, recon_delta = compress_tensor(delta, safe_name, out_dir / policy_name, policy)

            delta_cos = cosine_sim(delta, recon_delta)
            # Reconstructed weight = original + compressed delta
            recon_w = orig_weights[name] + recon_delta
            weight_cos = cosine_sim(new_w, recon_w)
            comp_bytes = stats.get("compressed_bytes", delta.nbytes)

            if policy_name == "k256":
                total_comp_bytes_k256 += comp_bytes
            else:
                total_comp_bytes_k64 += comp_bytes

            results.append({
                "tensor": name,
                "shape": list(delta.shape),
                "kurtosis": round(kurt, 2),
                "sparsity": round(sparsity, 4),
                "codec": policy_name,
                "delta_cosine": round(delta_cos, 6),
                "weight_cosine": round(weight_cos, 8),
                "compression_ratio": stats.get("compression_ratio", 1.0),
                "delta_l2_norm": round(float(np.linalg.norm(delta)), 6),
            })

    # Summary per codec
    for codec in ["k256", "k64"]:
        cr = [r for r in results if r["codec"] == codec]
        d_cos = [r["delta_cosine"] for r in cr]
        w_cos = [r["weight_cosine"] for r in cr]
        ratios = [r["compression_ratio"] for r in cr]
        print(f"\n  {codec}: {len(cr)} tensors")
        print(f"    Delta cosine: min={min(d_cos):.6f}, mean={np.mean(d_cos):.6f}")
        print(f"    Weight cosine: min={min(w_cos):.8f}, mean={np.mean(w_cos):.8f}")
        print(f"    Compression: mean={np.mean(ratios):.2f}x")

    bw_ratio_k64 = total_delta_bytes / max(1, total_comp_bytes_k64)
    print(f"\n  Bandwidth savings (k64): {total_delta_bytes/1e6:.1f} MB → {total_comp_bytes_k64/1e6:.1f} MB ({bw_ratio_k64:.1f}x)")
    print(f"  FLAG: Single SGD step — delta magnitudes are realistic but training dynamics simplified")

    cost = finish_cost(t_start, cpu_start, start_iso)
    write_receipt("tensor_infra_domain_4", "federated_deltas", {
        "per_tensor": results,
        "total_delta_mb": round(total_delta_bytes / 1e6, 1),
        "bandwidth_ratio_k256": round(total_delta_bytes / max(1, total_comp_bytes_k256), 2),
        "bandwidth_ratio_k64": round(bw_ratio_k64, 2),
        "data_source": "PARTIAL — real TinyLlama weights + real WikiText-2 SGD step",
        "flag": "Single SGD step, not full fine-tuning",
    }, cost)

if __name__ == "__main__":
    main()
