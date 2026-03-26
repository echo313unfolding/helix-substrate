#!/usr/bin/env python3
"""Domain 9: Continual Learning Snapshots.
Create model checkpoint sequence, compress each, verify restoration."""

import torch, numpy as np
from pathlib import Path
from _common import *

def main():
    t_start, cpu_start, start_iso = start_cost()
    print("=" * 72)
    print("DOMAIN 9: Continual Learning Snapshots")
    print("=" * 72)

    model, tokenizer = load_tinyllama_model()
    batch = load_wikitext2_batch(tokenizer, n_tokens=128)

    out_dir = HELIX_ROOT / "tensor_infra_scratch" / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = policy_vq(k=256, sidecar=True)

    # Snapshot 0: baseline (no training)
    snapshots = {}
    snapshots[0] = {name: param.data.detach().cpu().numpy().copy()
                    for name, param in model.named_parameters() if param.ndim == 2}
    n_tensors = len(snapshots[0])

    # Create 3 more snapshots via SGD steps on different data slices
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for snap_id in range(1, 4):
        outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"].clone())
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        snapshots[snap_id] = {name: param.data.detach().cpu().numpy().copy()
                              for name, param in model.named_parameters() if param.ndim == 2}
        print(f"  Snapshot {snap_id}: loss={outputs.loss.item():.4f}")

    # Compress each snapshot (full compression)
    full_results = []
    for snap_id, snap_weights in snapshots.items():
        snap_dir = out_dir / f"snapshot_{snap_id}"
        snap_bytes = 0
        orig_bytes = 0
        cosines = []

        for name, tensor in snap_weights.items():
            safe = name.replace(".", "_")
            stats, recon = compress_tensor(tensor.astype(np.float32), safe, snap_dir, policy)
            cos = cosine_sim(tensor, recon)
            cosines.append(cos)
            orig_bytes += tensor.nbytes
            snap_bytes += stats.get("compressed_bytes", tensor.nbytes)

        ratio = orig_bytes / max(1, snap_bytes)
        full_results.append({
            "snapshot": snap_id,
            "n_tensors": len(snap_weights),
            "original_mb": round(orig_bytes / 1e6, 1),
            "compressed_mb": round(snap_bytes / 1e6, 1),
            "ratio": round(ratio, 2),
            "cosine_min": round(min(cosines), 6),
            "cosine_mean": round(float(np.mean(cosines)), 6),
        })
        print(f"  Snapshot {snap_id}: {orig_bytes/1e6:.1f} MB → {snap_bytes/1e6:.1f} MB "
              f"({ratio:.2f}x), cos_min={min(cosines):.6f}")

    # Delta compression: compress diff from snapshot 0
    delta_results = []
    base = snapshots[0]
    for snap_id in range(1, 4):
        snap = snapshots[snap_id]
        delta_dir = out_dir / f"delta_{snap_id}"
        delta_bytes = 0
        delta_orig_bytes = 0
        cosines = []

        for name in base:
            delta = (snap[name] - base[name]).astype(np.float32)
            safe = name.replace(".", "_")
            # Use k=64 for deltas (sparse, cluster near zero)
            delta_policy = policy_vq(k=64, sidecar=True)
            stats, recon_delta = compress_tensor(delta, safe, delta_dir, delta_policy)

            recon_weight = base[name] + recon_delta
            cos = cosine_sim(snap[name], recon_weight)
            cosines.append(cos)
            delta_orig_bytes += delta.nbytes
            delta_bytes += stats.get("compressed_bytes", delta.nbytes)

        ratio = delta_orig_bytes / max(1, delta_bytes)
        delta_results.append({
            "snapshot": snap_id,
            "delta_original_mb": round(delta_orig_bytes / 1e6, 1),
            "delta_compressed_mb": round(delta_bytes / 1e6, 1),
            "delta_ratio": round(ratio, 2),
            "restored_cosine_min": round(min(cosines), 6),
        })
        print(f"  Delta {snap_id}: {delta_orig_bytes/1e6:.1f} MB → {delta_bytes/1e6:.1f} MB "
              f"({ratio:.2f}x), restored_cos_min={min(cosines):.6f}")

    # Total storage comparison
    total_uncompressed = sum(r["original_mb"] for r in full_results)
    total_full_compressed = sum(r["compressed_mb"] for r in full_results)
    base_compressed = full_results[0]["compressed_mb"]
    total_delta_compressed = base_compressed + sum(r["delta_compressed_mb"] for r in delta_results)

    print(f"\n  Storage comparison for {len(snapshots)} snapshots:")
    print(f"    Uncompressed: {total_uncompressed:.1f} MB")
    print(f"    Full-compressed: {total_full_compressed:.1f} MB ({total_uncompressed/total_full_compressed:.1f}x)")
    print(f"    Base + deltas: {total_delta_compressed:.1f} MB ({total_uncompressed/total_delta_compressed:.1f}x)")
    print(f"  FLAG: Single SGD steps, not full training epochs")

    cost = finish_cost(t_start, cpu_start, start_iso)
    write_receipt("tensor_infra_domain_9", "continual_snapshots", {
        "n_snapshots": len(snapshots),
        "full_compression": full_results,
        "delta_compression": delta_results,
        "total_uncompressed_mb": round(total_uncompressed, 1),
        "total_full_compressed_mb": round(total_full_compressed, 1),
        "total_delta_compressed_mb": round(total_delta_compressed, 1),
        "data_source": "PARTIAL — TinyLlama base + 3 SGD-step snapshots",
        "flag": "Single SGD steps, not full training epochs",
    }, cost)

if __name__ == "__main__":
    main()
