"""
WO-BORN-HYBRID-01: Born-compressed Qwen-Mamba hybrid training on A100.

Two modes:
  --mode born     Born-compressed training (grouped VQ d=2, STE, Lloyd's every step)
  --mode dense    Dense baseline (same architecture, no compression)

Usage:
    # Smoke test on CPU (seq_len=64, 10 steps)
    python -m echo_hybrid.train_born_hybrid --mode born --device cpu --steps 10 --seq-len 64 --batch-size 1

    # A100 born-compressed training
    python -m echo_hybrid.train_born_hybrid --mode born --device cuda --steps 50000 \\
        --batch-size 8 --grad-accum 4 --seq-len 2048 --data-dir data/code_tokens

    # A100 dense baseline
    python -m echo_hybrid.train_born_hybrid --mode dense --device cuda --steps 50000 \\
        --batch-size 8 --grad-accum 4 --seq-len 2048 --data-dir data/code_tokens
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from echo_hybrid.config_v2 import (
    EchoHybridV2Config,
    EchoHybridV2Model,
    load_qwen_pretrained,
    MAMBA_CUDA_AVAILABLE,
)
from echo_hybrid.train_phase1 import compress_linear, compress_all_linears
from echo_hybrid.train_sidecar_aware import compute_sidecar_norms, cost_block, loss_stats


RECEIPT_DIR = Path("receipts/echo_hybrid")
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# GPU-native Lloyd's reassignment (no numpy, stays on CUDA)
# ---------------------------------------------------------------------------

def lloyd_reassign_gpu(
    compressed: Dict[str, Dict[str, torch.Tensor]],
    model: nn.Module,
    device: torch.device,
) -> Dict[str, int]:
    """One Lloyd's iteration per layer on GPU. No CPU round-trip.

    For each compressed layer:
      1. Compute ||w - c||^2 for all weights vs all centroids (GPU matmul)
      2. Reassign each weight to nearest centroid
      3. Update centroids as mean of assigned weights
      4. Reinitialize dead centroids from largest cluster

    Returns {layer_name: n_dead_centroids}.
    """
    modules = dict(model.named_modules())
    dead_counts = {}

    for name, factors in compressed.items():
        mod = modules[name]
        vd = factors.get("vector_dim", 1)
        if isinstance(vd, torch.Tensor):
            vd = vd.item()
        vd = int(vd)

        W = mod.weight.data.float()
        cb = factors["codebook"].float().to(device)
        k = cb.shape[0]

        if vd == 1:
            # Scalar: searchsorted is faster than full distance matrix
            flat = W.reshape(-1)
            sorted_cb, sort_idx = cb.sort()
            insert = torch.searchsorted(sorted_cb, flat).clamp(0, k - 1)
            left = (insert - 1).clamp(0)
            d_right = (flat - sorted_cb[insert]).abs()
            d_left = (flat - sorted_cb[left]).abs()
            best_sorted = torch.where(d_left < d_right, left, insert)
            new_idx = sort_idx[best_sorted]

            # Centroid update
            new_cb = torch.zeros_like(cb)
            counts = torch.zeros(k, device=device)
            new_cb.scatter_add_(0, new_idx, flat)
            counts.scatter_add_(0, new_idx, torch.ones_like(flat))
            alive = counts > 0
            new_cb[alive] /= counts[alive]
            new_cb[~alive] = cb[~alive]  # keep dead centroids for now

            n_dead = int((~alive).sum().item())
            if n_dead > 0:
                largest = counts.argmax()
                members = flat[new_idx == largest]
                if len(members) > 1:
                    dists = (members - new_cb[largest]).abs()
                    far_idx = dists.argsort(descending=True)[:n_dead]
                    dead_ids = torch.where(~alive)[0]
                    for i in range(min(n_dead, len(far_idx))):
                        new_cb[dead_ids[i]] = members[far_idx[i]]

            factors["codebook"] = new_cb
            factors["indices"] = new_idx.to(torch.uint8).reshape(factors["indices"].shape)

        else:
            # Grouped VQ: d-dimensional distance via matmul trick
            vectors = W.reshape(-1, vd)  # [N, d]
            N = vectors.shape[0]

            # ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x@c.T
            X_sq = (vectors * vectors).sum(dim=1, keepdim=True)   # [N, 1]
            C_sq = (cb * cb).sum(dim=1, keepdim=True).T           # [1, k]

            # Chunked assignment to avoid [N, k] OOM for large layers
            CHUNK = 500_000
            new_idx = torch.empty(N, device=device, dtype=torch.long)
            for start in range(0, N, CHUNK):
                end = min(start + CHUNK, N)
                dists = X_sq[start:end] + C_sq - 2.0 * (vectors[start:end] @ cb.T)
                new_idx[start:end] = dists.argmin(dim=1)

            # Centroid update via scatter_add (no Python loop)
            idx_expand = new_idx.unsqueeze(1).expand(-1, vd)
            sums = torch.zeros(k, vd, device=device, dtype=torch.float64)
            sums.scatter_add_(0, idx_expand, vectors.double())
            counts = torch.zeros(k, device=device, dtype=torch.long)
            counts.scatter_add_(0, new_idx, torch.ones(N, device=device, dtype=torch.long))

            alive = counts > 0
            new_cb = cb.clone()
            alive_counts = counts[alive].unsqueeze(1).float()
            new_cb[alive] = (sums[alive] / alive_counts.double()).float()

            n_dead = int((~alive).sum().item())
            if n_dead > 0:
                largest = counts.argmax()
                members = vectors[new_idx == largest]
                if len(members) > 1:
                    d2 = ((members - new_cb[largest]) ** 2).sum(dim=1)
                    far_idx = d2.argsort(descending=True)[:n_dead]
                    dead_ids = torch.where(~alive)[0]
                    for i in range(min(n_dead, len(far_idx))):
                        new_cb[dead_ids[i]] = members[far_idx[i]]

            factors["codebook"] = new_cb
            factors["indices"] = new_idx.to(torch.uint8).reshape(factors["indices"].shape)

        dead_counts[name] = n_dead

    return dead_counts


# ---------------------------------------------------------------------------
# TopK sidecar-aware recompression (CPU k-means, infrequent)
# ---------------------------------------------------------------------------

def topk_recompress(
    model: nn.Module,
    compressed: Dict[str, Dict[str, torch.Tensor]],
    k: int = 5,
    n_clusters: int = 256,
    vector_dim: int = 2,
    device: torch.device = None,
):
    """Recompress top-K layers by sidecar norm growth.

    Uses CPU k-means (compress_linear) then moves results to device.
    Called every 20 steps -- CPU cost is amortized.
    """
    # Compute current sidecar norms
    norms = compute_sidecar_norms(model, compressed)

    # Rank by norm (higher = more stale)
    ranked = sorted(norms.items(), key=lambda x: x[1], reverse=True)
    to_recompress = [name for name, _ in ranked[:k]]

    modules = dict(model.named_modules())
    for name in to_recompress:
        mod = modules[name]
        W = mod.weight.data.cpu()
        vd = vector_dim if W.shape[1] % vector_dim == 0 else 1
        factors = compress_linear(W, n_clusters=n_clusters, vector_dim=vd)
        # Move to GPU
        if device is not None:
            factors["codebook"] = factors["codebook"].to(device)
            factors["indices"] = factors["indices"].to(device)
            factors["sidecar_positions"] = factors["sidecar_positions"].to(device)
            factors["sidecar_values"] = factors["sidecar_values"].to(device)
        compressed[name] = factors

    return to_recompress


# ---------------------------------------------------------------------------
# STE Quantizer (GPU-aware)
# ---------------------------------------------------------------------------

class STEQuantizerV2:
    """STE quantization for V2 model. Keeps codebooks/indices on model device."""

    def __init__(self, model: nn.Module, compressed: Dict[str, Dict[str, torch.Tensor]]):
        self.model = model
        self.compressed = compressed
        self._shadow_cache: Dict[str, torch.Tensor] = {}
        self._module_cache: Dict[str, nn.Module] = dict(model.named_modules())

    def apply_quantized_weights(self):
        """Replace weights with STE-quantized versions."""
        self._shadow_cache.clear()
        for name, factors in self.compressed.items():
            mod = self._module_cache[name]
            self._shadow_cache[name] = mod.weight.data.clone()

            cb = factors["codebook"].to(mod.weight.device)
            idx = factors["indices"].to(mod.weight.device).long()
            W_q = cb[idx]

            vd = factors.get("vector_dim", 1)
            if isinstance(vd, torch.Tensor):
                vd = vd.item()
            if vd > 1:
                W_q = W_q.reshape(mod.weight.shape)

            # Apply sidecar corrections
            sp = factors["sidecar_positions"].to(mod.weight.device)
            sv = factors["sidecar_values"].to(mod.weight.device)
            W_q_flat = W_q.reshape(-1)
            W_q_flat[sp.long()] += sv
            W_q = W_q_flat.reshape(mod.weight.shape)

            # STE: forward sees W_q, backward flows to W_shadow
            mod.weight.data = mod.weight.data + (W_q - mod.weight.data).detach()

    def restore_shadow_weights(self):
        """Restore shadow weights before optimizer step."""
        for name, shadow in self._shadow_cache.items():
            mod = self._module_cache[name]
            mod.weight.data = shadow
        self._shadow_cache.clear()


# ---------------------------------------------------------------------------
# Cosine warmup scheduler
# ---------------------------------------------------------------------------

def cosine_warmup_schedule(step: int, warmup: int, total: int, lr_max: float, lr_min: float = 0.0) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_batch(data, batch_size: int, seq_len: int, step: int, device: str):
    """Get a batch from memory-mapped data."""
    n_seqs = len(data) // seq_len
    # Deterministic but shuffled indices based on step
    rng = np.random.RandomState(step)
    indices = rng.randint(0, n_seqs, size=batch_size)
    batch = np.stack([
        data[i * seq_len:(i + 1) * seq_len].astype(np.int64)
        for i in indices
    ])
    return torch.from_numpy(batch).to(device)


def load_data(data_dir: str):
    """Load memory-mapped token data."""
    data_dir = Path(data_dir)
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"

    if not train_path.exists():
        raise FileNotFoundError(f"No train.bin in {data_dir}. Run prepare_data.py first.")

    train_data = np.memmap(str(train_path), dtype=np.uint32, mode="r")
    val_data = np.memmap(str(val_path), dtype=np.uint32, mode="r") if val_path.exists() else None

    meta_path = data_dir / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}

    print(f"  Train: {len(train_data):,} tokens")
    if val_data is not None:
        print(f"  Val:   {len(val_data):,} tokens")

    return train_data, val_data, meta


def load_wikitext_fallback(seq_len: int = 128, max_chunks: int = 5000):
    """Fallback for smoke testing without prepared data."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([x for x in ds["text"] if x.strip()])
    tokens = tokenizer.encode(text)
    tokens = np.array(tokens, dtype=np.uint32)

    n_chunks = min(max_chunks, len(tokens) // seq_len)
    tokens = tokens[:n_chunks * seq_len]
    print(f"  WikiText fallback: {len(tokens):,} tokens ({n_chunks} chunks of {seq_len})")
    return tokens, None, {"dataset": "wikitext-2", "seq_len": seq_len}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, val_data, seq_len: int, batch_size: int, device: str,
                 n_batches: int = 50, use_checkpoint: bool = False) -> float:
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0.0
    n_tokens = 0

    for i in range(n_batches):
        batch = get_batch(val_data, batch_size, seq_len, step=100000 + i, device=device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=(device != "cpu")):
            out = model(input_ids=batch, labels=batch, use_checkpoint=use_checkpoint)
        total_loss += out["loss"].item() * (batch.shape[0] * (seq_len - 1))
        n_tokens += batch.shape[0] * (seq_len - 1)

    avg_loss = total_loss / max(n_tokens, 1)
    model.train()
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    compressed: Optional[Dict],
    step: int,
    loss: float,
    path: str,
):
    """Save training checkpoint."""
    ckpt = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if compressed is not None:
        # Save compressed artifacts (codebooks, indices, sidecars)
        ckpt["compressed"] = {
            name: {k: v.cpu() if isinstance(v, torch.Tensor) else v
                   for k, v in factors.items()}
            for name, factors in compressed.items()
        }
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: {path} (step {step})")


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, device: str):
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    compressed = ckpt.get("compressed", None)
    if compressed is not None:
        # Move to device
        for name, factors in compressed.items():
            for k, v in factors.items():
                if isinstance(v, torch.Tensor):
                    factors[k] = v.to(device)
    return ckpt["step"], ckpt["loss"], compressed


# ---------------------------------------------------------------------------
# Born-compressed training loop
# ---------------------------------------------------------------------------

def train_born_compressed(args):
    """Born-compressed training with grouped VQ d=2, STE, Lloyd's every step."""
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    device = args.device
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- Build model ---
    print("Building EchoHybridV2...")
    cfg = EchoHybridV2Config()
    model = EchoHybridV2Model(cfg)
    print(model)

    # --- Load Qwen pretrained weights ---
    if not args.skip_pretrained:
        load_stats = load_qwen_pretrained(model, args.qwen_path)
    else:
        load_stats = {"attn_loaded": 0, "embed_loaded": 0, "note": "skipped"}

    model = model.to(device)
    print(f"  Model on {device}")

    # --- Load data ---
    print("Loading data...")
    if args.data_dir and Path(args.data_dir).exists():
        train_data, val_data, data_meta = load_data(args.data_dir)
    else:
        print("  No data dir found, using WikiText fallback")
        train_data, val_data, data_meta = load_wikitext_fallback(args.seq_len)

    # --- Initial compression (VQ d=2) ---
    print(f"Compressing all linear layers (VQ d={args.vector_dim}, k={args.n_clusters})...")
    compressed = compress_all_linears(model.cpu(), n_clusters=args.n_clusters, vector_dim=args.vector_dim)
    model = model.to(device)

    # Move compressed artifacts to device
    for name, factors in compressed.items():
        for k, v in factors.items():
            if isinstance(v, torch.Tensor):
                factors[k] = v.to(device)

    n_compressed = len(compressed)
    print(f"  Compressed {n_compressed} layers")

    # --- Optimizer with differential LR ---
    mamba_params = []
    attn_params = []
    for block_idx, bt in enumerate(cfg.block_pattern):
        block = model.blocks[block_idx]
        if bt == "mamba":
            mamba_params.extend(block.parameters())
        else:
            attn_params.extend(block.parameters())
    # Embeddings + final norm go with attention LR
    attn_params.extend(model.embed_tokens.parameters())
    attn_params.extend(model.norm.parameters())

    optimizer = torch.optim.AdamW([
        {"params": mamba_params, "lr": args.lr_mamba},
        {"params": attn_params, "lr": args.lr_attn},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # --- STE ---
    ste = STEQuantizerV2(model, compressed)

    # --- Resume from checkpoint ---
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        start_step, _, loaded_compressed = load_checkpoint(args.resume, model, optimizer, device)
        if loaded_compressed is not None:
            compressed = loaded_compressed
            ste.compressed = compressed
        print(f"  Resumed at step {start_step}")

    # --- Wandb init ---
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project="echo-hybrid-v2",
                name=f"born-d{args.vector_dim}-{time.strftime('%m%d-%H%M')}",
                config=vars(args),
            )
        except ImportError:
            print("  wandb not installed, skipping")
            args.wandb = False

    # --- Training loop ---
    model.train()
    losses = []
    val_ppls = []
    lloyd_total_dead = 0
    recompress_count = 0

    print(f"\n{'='*70}")
    print(f"Born-Compressed Training: {args.steps} steps")
    print(f"  VQ d={args.vector_dim}, k={args.n_clusters}")
    print(f"  LR mamba={args.lr_mamba}, LR attn={args.lr_attn}")
    print(f"  Batch={args.batch_size}, grad_accum={args.grad_accum}, "
          f"eff_batch={args.batch_size * args.grad_accum}")
    print(f"  Seq_len={args.seq_len}")
    print(f"  Lloyd's every step, TopK-{args.topk_k} recompress every {args.recompress_every} steps")
    print(f"  CUDA scan: {MAMBA_CUDA_AVAILABLE}")
    print(f"  Gradient checkpointing: {args.grad_checkpoint}")
    print(f"{'='*70}\n")

    optimizer.zero_grad()

    for step in range(start_step, args.steps):
        step_t0 = time.time()

        # --- LR schedule ---
        lr_m = cosine_warmup_schedule(step, args.warmup_steps, args.steps, args.lr_mamba)
        lr_a = cosine_warmup_schedule(step, args.warmup_steps, args.steps, args.lr_attn)
        optimizer.param_groups[0]["lr"] = lr_m
        optimizer.param_groups[1]["lr"] = lr_a

        # --- Gradient accumulation ---
        accum_loss = 0.0
        for micro in range(args.grad_accum):
            batch = get_batch(train_data, args.batch_size, args.seq_len,
                              step * args.grad_accum + micro, device)

            # Apply STE quantization
            ste.apply_quantized_weights()

            # Forward with mixed precision
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=(device != "cpu")):
                out = model(input_ids=batch, labels=batch,
                            use_checkpoint=args.grad_checkpoint)
                loss = out["loss"] / args.grad_accum

            # Backward
            loss.backward()

            # Restore shadow weights after backward
            ste.restore_shadow_weights()

            accum_loss += loss.item() * args.grad_accum

        # --- Gradient clipping + optimizer step ---
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(accum_loss)

        # --- Lloyd's reassignment every step (GPU) ---
        dead_counts = lloyd_reassign_gpu(compressed, model, torch.device(device))
        step_dead = sum(dead_counts.values())
        lloyd_total_dead += step_dead

        # --- TopK recompression ---
        if (step + 1) % args.recompress_every == 0:
            recompressed = topk_recompress(
                model, compressed, k=args.topk_k,
                n_clusters=args.n_clusters, vector_dim=args.vector_dim,
                device=torch.device(device),
            )
            ste.compressed = compressed
            recompress_count += len(recompressed)

        # --- Logging ---
        step_time = time.time() - step_t0
        if (step + 1) % args.log_every == 0 or step == start_step:
            tokens_per_sec = (args.batch_size * args.grad_accum * args.seq_len) / max(step_time, 1e-6)
            vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device != "cpu" else 0
            print(f"  step {step+1:6d}/{args.steps} | loss={accum_loss:.4f} | "
                  f"lr_m={lr_m:.2e} | grad={grad_norm:.2f} | "
                  f"dead={step_dead} | {tokens_per_sec:,.0f} tok/s | "
                  f"VRAM={vram_gb:.1f}GB | {step_time:.2f}s")

            if args.wandb:
                import wandb
                wandb.log({
                    "loss": accum_loss,
                    "lr_mamba": lr_m,
                    "lr_attn": lr_a,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "dead_centroids": step_dead,
                    "tokens_per_sec": tokens_per_sec,
                    "vram_gb": vram_gb,
                }, step=step + 1)

        # --- Validation ---
        if val_data is not None and (step + 1) % args.eval_every == 0:
            ppl = evaluate_ppl(model, val_data, args.seq_len, args.batch_size, device,
                               n_batches=args.eval_batches, use_checkpoint=args.grad_checkpoint)
            val_ppls.append({"step": step + 1, "ppl": round(ppl, 4)})
            print(f"  >>> VAL PPL at step {step+1}: {ppl:.4f}")
            if args.wandb:
                import wandb
                wandb.log({"val_ppl": ppl}, step=step + 1)

        # --- Checkpoint ---
        if (step + 1) % args.save_every == 0:
            ckpt_dir = Path(args.ckpt_dir) / f"born_d{args.vector_dim}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model, optimizer, compressed, step + 1, accum_loss,
                str(ckpt_dir / f"step_{step+1:06d}.pt"),
            )

    # --- Final VRAM ---
    vram_peak = torch.cuda.max_memory_allocated() / 1e9 if device != "cpu" else 0

    # --- Receipt ---
    receipt = {
        "wo": "WO-BORN-HYBRID-01",
        "mode": "born_compressed",
        "timestamp": time.strftime("%Y-%m-%d"),
        "model": f"EchoHybridV2-{cfg.n_blocks}block-{cfg.hidden_size}d",
        "n_params": model.n_params(),
        "n_params_non_embed": model.n_params(exclude_embeddings=True),
        "block_pattern_summary": f"{cfg.n_mamba}M+{cfg.n_attn}A",
        "compression": {
            "vector_dim": args.vector_dim,
            "n_clusters": args.n_clusters,
            "n_compressed_layers": n_compressed,
            "lloyd_per_step": True,
            "topk_k": args.topk_k,
            "recompress_every": args.recompress_every,
            "total_recompressions": recompress_count,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "seq_len": args.seq_len,
            "lr_mamba": args.lr_mamba,
            "lr_attn": args.lr_attn,
            "warmup_steps": args.warmup_steps,
            "grad_checkpoint": args.grad_checkpoint,
        },
        "results": {
            **loss_stats(losses),
            "val_ppls": val_ppls,
        },
        "load_stats": load_stats,
        "data_meta": data_meta,
        "cuda_scan": MAMBA_CUDA_AVAILABLE,
        "vram_peak_gb": round(vram_peak, 2),
        "device": device,
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    receipt_path = RECEIPT_DIR / f"wo_born_hybrid_born_d{args.vector_dim}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nRECEIPT: {receipt_path}")
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    if val_ppls:
        print(f"  Final val PPL: {val_ppls[-1]['ppl']}")
    print(f"  VRAM peak: {vram_peak:.2f} GB")
    print(f"  Recompressions: {recompress_count}")

    return receipt


# ---------------------------------------------------------------------------
# Dense baseline training loop
# ---------------------------------------------------------------------------

def train_dense(args):
    """Dense baseline -- same architecture, no compression."""
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    device = args.device
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- Build model ---
    print("Building EchoHybridV2 (dense baseline)...")
    cfg = EchoHybridV2Config()
    model = EchoHybridV2Model(cfg)
    print(model)

    if not args.skip_pretrained:
        load_stats = load_qwen_pretrained(model, args.qwen_path)
    else:
        load_stats = {"attn_loaded": 0, "embed_loaded": 0, "note": "skipped"}

    model = model.to(device)

    # --- Load data ---
    print("Loading data...")
    if args.data_dir and Path(args.data_dir).exists():
        train_data, val_data, data_meta = load_data(args.data_dir)
    else:
        train_data, val_data, data_meta = load_wikitext_fallback(args.seq_len)

    # --- Optimizer with differential LR ---
    mamba_params = []
    attn_params = []
    for block_idx, bt in enumerate(cfg.block_pattern):
        block = model.blocks[block_idx]
        if bt == "mamba":
            mamba_params.extend(block.parameters())
        else:
            attn_params.extend(block.parameters())
    attn_params.extend(model.embed_tokens.parameters())
    attn_params.extend(model.norm.parameters())

    optimizer = torch.optim.AdamW([
        {"params": mamba_params, "lr": args.lr_mamba},
        {"params": attn_params, "lr": args.lr_attn},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # --- Resume ---
    start_step = 0
    if args.resume:
        start_step, _, _ = load_checkpoint(args.resume, model, optimizer, device)

    # --- Wandb ---
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project="echo-hybrid-v2",
                name=f"dense-{time.strftime('%m%d-%H%M')}",
                config=vars(args),
            )
        except ImportError:
            args.wandb = False

    # --- Training ---
    model.train()
    losses = []
    val_ppls = []

    print(f"\n{'='*70}")
    print(f"Dense Baseline Training: {args.steps} steps")
    print(f"{'='*70}\n")

    optimizer.zero_grad()

    for step in range(start_step, args.steps):
        step_t0 = time.time()

        lr_m = cosine_warmup_schedule(step, args.warmup_steps, args.steps, args.lr_mamba)
        lr_a = cosine_warmup_schedule(step, args.warmup_steps, args.steps, args.lr_attn)
        optimizer.param_groups[0]["lr"] = lr_m
        optimizer.param_groups[1]["lr"] = lr_a

        accum_loss = 0.0
        for micro in range(args.grad_accum):
            batch = get_batch(train_data, args.batch_size, args.seq_len,
                              step * args.grad_accum + micro, device)

            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=(device != "cpu")):
                out = model(input_ids=batch, labels=batch,
                            use_checkpoint=args.grad_checkpoint)
                loss = out["loss"] / args.grad_accum

            loss.backward()
            accum_loss += loss.item() * args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(accum_loss)

        step_time = time.time() - step_t0
        if (step + 1) % args.log_every == 0 or step == start_step:
            tokens_per_sec = (args.batch_size * args.grad_accum * args.seq_len) / max(step_time, 1e-6)
            vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device != "cpu" else 0
            print(f"  step {step+1:6d}/{args.steps} | loss={accum_loss:.4f} | "
                  f"lr_m={lr_m:.2e} | grad={grad_norm:.2f} | "
                  f"{tokens_per_sec:,.0f} tok/s | VRAM={vram_gb:.1f}GB")

            if args.wandb:
                import wandb
                wandb.log({
                    "loss": accum_loss, "lr_mamba": lr_m, "lr_attn": lr_a,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "tokens_per_sec": tokens_per_sec,
                }, step=step + 1)

        if val_data is not None and (step + 1) % args.eval_every == 0:
            ppl = evaluate_ppl(model, val_data, args.seq_len, args.batch_size, device,
                               n_batches=args.eval_batches, use_checkpoint=args.grad_checkpoint)
            val_ppls.append({"step": step + 1, "ppl": round(ppl, 4)})
            print(f"  >>> VAL PPL at step {step+1}: {ppl:.4f}")

        if (step + 1) % args.save_every == 0:
            ckpt_dir = Path(args.ckpt_dir) / "dense"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model, optimizer, None, step + 1, accum_loss,
                str(ckpt_dir / f"step_{step+1:06d}.pt"),
            )

    vram_peak = torch.cuda.max_memory_allocated() / 1e9 if device != "cpu" else 0

    receipt = {
        "wo": "WO-BORN-HYBRID-01",
        "mode": "dense_baseline",
        "timestamp": time.strftime("%Y-%m-%d"),
        "model": f"EchoHybridV2-{cfg.n_blocks}block-{cfg.hidden_size}d",
        "n_params": model.n_params(),
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "seq_len": args.seq_len,
            "lr_mamba": args.lr_mamba,
            "lr_attn": args.lr_attn,
        },
        "results": {
            **loss_stats(losses),
            "val_ppls": val_ppls,
        },
        "load_stats": load_stats,
        "vram_peak_gb": round(vram_peak, 2),
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    receipt_path = RECEIPT_DIR / "wo_born_hybrid_dense.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nRECEIPT: {receipt_path}")
    return receipt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WO-BORN-HYBRID-01: Train born-compressed Qwen-Mamba hybrid")

    # Mode
    parser.add_argument("--mode", type=str, default="born", choices=["born", "dense"],
                        help="Training mode: born (compressed) or dense (baseline)")

    # Model
    parser.add_argument("--qwen-path", type=str, default="Qwen/Qwen2.5-Coder-1.5B",
                        help="HuggingFace path or local dir for Qwen pretrained weights")
    parser.add_argument("--skip-pretrained", action="store_true",
                        help="Skip loading Qwen pretrained weights (random init)")

    # Compression
    parser.add_argument("--vector-dim", type=int, default=2, help="VQ group size (d=2 for production)")
    parser.add_argument("--n-clusters", type=int, default=256, help="VQ codebook size")
    parser.add_argument("--topk-k", type=int, default=5, help="TopK recompression: number of layers")
    parser.add_argument("--recompress-every", type=int, default=20, help="TopK recompression interval")

    # Training
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr-mamba", type=float, default=3e-4, help="LR for Mamba blocks (random init)")
    parser.add_argument("--lr-attn", type=float, default=1e-5, help="LR for ATTN blocks (pretrained)")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM)")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/code_tokens",
                        help="Directory with train.bin / val.bin from prepare_data.py")

    # Infra
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/echo_hybrid_v2")
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50, help="Number of batches for val PPL")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    if args.mode == "born":
        train_born_compressed(args)
    else:
        train_dense(args)


if __name__ == "__main__":
    main()
