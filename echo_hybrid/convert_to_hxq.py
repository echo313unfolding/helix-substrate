"""
WO-BORN-HYBRID-01: Convert trained born-compressed model to HXQ format.

Takes a training checkpoint (shadow weights + compressed codebooks/indices)
and exports the final model in HXQ format:
  - Per-layer .cdna files (codebook + indices)
  - Per-layer .hxzo sidecar files (outlier corrections)
  - manifest.json (metadata)

Usage:
    python -m echo_hybrid.convert_to_hxq \\
        --ckpt checkpoints/echo_hybrid_v2/born_d2/step_050000.pt \\
        --output models/echo_hybrid_v2_born_d2
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from echo_hybrid.config_v2 import EchoHybridV2Config, EchoHybridV2Model
from echo_hybrid.train_phase1 import compress_linear
from echo_hybrid.train_sidecar_aware import cost_block


def final_recompress(
    model: nn.Module,
    compressed: Dict[str, Dict],
    n_clusters: int = 256,
    vector_dim: int = 2,
) -> Dict[str, Dict]:
    """Final Lloyd's pass: recompress all layers from shadow weights.

    This ensures codebooks are optimally fitted to the final trained weights.
    """
    modules = dict(model.named_modules())
    final = {}

    for name, factors in compressed.items():
        mod = modules[name]
        W = mod.weight.data.cpu()
        vd = vector_dim if W.shape[1] % vector_dim == 0 else 1
        final[name] = compress_linear(W, n_clusters=n_clusters, vector_dim=vd)

    return final


def export_hxq(
    model: EchoHybridV2Model,
    compressed: Dict[str, Dict],
    output_dir: str,
    step: int = 0,
    vector_dim: int = 2,
):
    """Export model in HXQ format (codebooks + indices + sidecars).

    Directory layout:
        {output_dir}/
            manifest.json
            embed_tokens.safetensors  (FP16 embedding)
            norm.safetensors          (FP16 final norm)
            layers/
                blocks.0.block.in_proj.cdna    (codebook + indices)
                blocks.0.block.in_proj.hxzo    (sidecar)
                ...
    """
    from safetensors.torch import save_file
    from helix_substrate.sidecar import write_outlier_sidecar

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers_dir = out / "layers"
    layers_dir.mkdir(exist_ok=True)

    cfg = model.cfg
    manifest = {
        "format": "hxq",
        "version": "0.3.0",
        "quant_method": "hxq",
        "model_type": "echo_hybrid_v2",
        "architecture": {
            "hidden_size": cfg.hidden_size,
            "vocab_size": cfg.vocab_size,
            "n_blocks": cfg.n_blocks,
            "block_pattern": cfg.block_pattern,
            "n_mamba": cfg.n_mamba,
            "n_attn": cfg.n_attn,
            "ssm_d_inner": cfg.ssm_d_inner,
            "ssm_d_state": cfg.ssm_d_state,
            "ssm_dt_rank": cfg.ssm_dt_rank,
            "attn_num_heads": cfg.attn_num_heads,
            "attn_num_kv_heads": cfg.attn_num_kv_heads,
            "attn_intermediate_size": cfg.attn_intermediate_size,
            "tie_word_embeddings": cfg.tie_word_embeddings,
        },
        "compression": {
            "vector_dim": vector_dim,
            "n_clusters": 256,
            "born_compressed": True,
            "training_step": step,
        },
        "n_params": model.n_params(),
        "n_compressed_layers": len(compressed),
        "layers": {},
    }

    # --- Save embeddings and norms (uncompressed, FP16) ---
    embed_tensors = {"weight": model.embed_tokens.weight.data.half().cpu()}
    save_file(embed_tensors, str(out / "embed_tokens.safetensors"))
    print(f"  Saved embed_tokens: {model.embed_tokens.weight.shape}")

    norm_tensors = {"weight": model.norm.weight.data.half().cpu()}
    save_file(norm_tensors, str(out / "norm.safetensors"))
    print(f"  Saved final norm")

    # --- Save compressed layers ---
    for name, factors in compressed.items():
        safe_name = name.replace(".", "_")
        cb = factors["codebook"].cpu().numpy()
        idx = factors["indices"].cpu().numpy()
        vd = factors.get("vector_dim", 1)
        if isinstance(vd, torch.Tensor):
            vd = vd.item()

        # Save codebook + indices as .cdna
        cdna_path = layers_dir / f"{safe_name}.cdna"
        np.savez_compressed(
            str(cdna_path),
            codebook=cb,
            indices=idx.astype(np.uint8),
            vector_dim=np.array([vd]),
        )

        # Get the module shape for sidecar
        modules = dict(model.named_modules())
        mod = modules[name]
        shape = tuple(mod.weight.shape)

        # Save sidecar as .hxzo
        sp = factors["sidecar_positions"].cpu().numpy().astype(np.int64)
        sv = factors["sidecar_values"].cpu().numpy().astype(np.float32)

        if len(sp) > 0:
            hxzo_path = str(layers_dir / f"{safe_name}.hxzo")
            write_outlier_sidecar(
                positions=sp,
                values=sv,
                tensor_name=name,
                threshold_policy={"method": "topk", "n": len(sp)},
                shape=shape,
                output_path=hxzo_path,
            )

        # Compute reconstruction quality
        W_q = torch.from_numpy(cb[idx.astype(np.int64)])
        if vd > 1:
            W_q = W_q.reshape(shape)
        W_orig = mod.weight.data.cpu().float()
        cos_sim = torch.nn.functional.cosine_similarity(
            W_orig.reshape(1, -1), W_q.reshape(1, -1).float()
        ).item()

        manifest["layers"][name] = {
            "shape": list(shape),
            "vector_dim": vd,
            "codebook_shape": list(cb.shape),
            "indices_shape": list(idx.shape),
            "n_sidecar": len(sp),
            "cos_sim": round(cos_sim, 6),
            "utilization": round(factors.get("codebook_utilization", 0), 4),
        }

    # --- Save non-compressed parameters (conv1d, A_log, D, biases, norms) ---
    uncompressed = {}
    for name, param in model.named_parameters():
        # Skip compressed layers and embeddings (already saved)
        if any(name.startswith(cn) for cn in compressed.keys()):
            continue
        if name in ("embed_tokens.weight", "lm_head.weight", "norm.weight"):
            continue
        uncompressed[name] = param.data.half().cpu()

    if uncompressed:
        save_file(uncompressed, str(out / "uncompressed.safetensors"))
        manifest["n_uncompressed_params"] = sum(p.numel() for p in uncompressed.values())
        print(f"  Saved {len(uncompressed)} uncompressed params "
              f"({manifest['n_uncompressed_params']:,} values)")

    # --- Write manifest ---
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    total_cdna_bytes = sum(f.stat().st_size for f in layers_dir.glob("*.cdna.npz"))
    total_hxzo_bytes = sum(f.stat().st_size for f in layers_dir.glob("*.hxzo"))
    total_bytes = total_cdna_bytes + total_hxzo_bytes
    if (out / "uncompressed.safetensors").exists():
        total_bytes += (out / "uncompressed.safetensors").stat().st_size
    if (out / "embed_tokens.safetensors").exists():
        total_bytes += (out / "embed_tokens.safetensors").stat().st_size
    if (out / "norm.safetensors").exists():
        total_bytes += (out / "norm.safetensors").stat().st_size

    print(f"\n  HXQ export complete: {out}/")
    print(f"    Compressed layers: {len(compressed)}")
    print(f"    Total size: {total_bytes / 1e6:.1f} MB")
    print(f"    Manifest: {out / 'manifest.json'}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Convert trained model to HXQ format")
    parser.add_argument("--ckpt", type=str, required=True, help="Training checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--vector-dim", type=int, default=2, help="VQ group size for final compression")
    parser.add_argument("--n-clusters", type=int, default=256)
    parser.add_argument("--skip-recompress", action="store_true",
                        help="Use checkpoint codebooks as-is (skip final recompression)")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Load model from checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = EchoHybridV2Config()
    model = EchoHybridV2Model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    step = ckpt.get("step", 0)
    compressed = ckpt.get("compressed", None)

    if compressed is None:
        print("  No compressed artifacts in checkpoint -- running full compression...")
        from echo_hybrid.train_phase1 import compress_all_linears
        compressed = compress_all_linears(model, n_clusters=args.n_clusters, vector_dim=args.vector_dim)
    elif not args.skip_recompress:
        print("  Running final recompression from shadow weights...")
        compressed = final_recompress(model, compressed, n_clusters=args.n_clusters,
                                      vector_dim=args.vector_dim)

    # Export
    manifest = export_hxq(model, compressed, args.output, step=step, vector_dim=args.vector_dim)

    # Receipt
    receipt = {
        "wo": "WO-BORN-HYBRID-01-CONVERT",
        "timestamp": time.strftime("%Y-%m-%d"),
        "checkpoint": args.ckpt,
        "output": args.output,
        "step": step,
        "n_compressed_layers": len(compressed),
        "vector_dim": args.vector_dim,
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    receipt_dir = Path("receipts/echo_hybrid")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / "wo_born_hybrid_convert.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nRECEIPT: {receipt_path}")


if __name__ == "__main__":
    main()
