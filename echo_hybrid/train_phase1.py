"""
WO-ECHO-HYBRID-01b: Phase 1 born-compressed training scaffold.

Dense shadow weights maintained for gradient computation.
Codebooks fixed at init, refreshed offline on schedule.
HXQ forward path active from step zero via STE (Straight-Through Estimator).
No claim about STE through discrete codebook — this is Phase 1.

Usage:
    python echo_hybrid/train_phase1.py [--steps 100] [--compress-every 100] [--batch-size 2] [--seq-len 128]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Force unbuffered output for background runs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from echo_hybrid.config import EchoHybridConfig, EchoHybridModel


# ---------------------------------------------------------------------------
# In-memory VQ compression (no disk I/O)
# ---------------------------------------------------------------------------

def compress_linear(
    weight: torch.Tensor,
    n_clusters: int = 256,
    sidecar_threshold: float = 0.0,
    vector_dim: int = 1,
) -> Dict[str, torch.Tensor]:
    """Compress a 2D weight tensor via k-means VQ.

    Args:
        vector_dim: Group size. 1=scalar (legacy), 2/4/8=grouped (production HXQ).
                    in_features must be divisible by vector_dim.

    Returns dict with codebook, indices, sidecar_positions, sidecar_values, vector_dim.
    """
    W = weight.detach().float().cpu().numpy()
    out_dim, in_dim = W.shape

    assert in_dim % vector_dim == 0, \
        f"in_features ({in_dim}) must be divisible by vector_dim ({vector_dim})"

    if vector_dim == 1:
        # Scalar path (existing)
        flat = W.reshape(-1, 1)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            max_iter=10,
            batch_size=min(10000, flat.shape[0]),
            n_init=1,
            random_state=42,
        )
        kmeans.fit(flat)
        codebook = kmeans.cluster_centers_.squeeze(-1).astype(np.float32)  # [k]
        indices = kmeans.labels_.astype(np.uint8).reshape(W.shape)         # [out, in]
        W_q = codebook[indices]
    else:
        # Grouped path: d-dimensional vectors
        from helix_substrate.cdna_encoder import _vector_kmeans
        vectors = W.reshape(-1, vector_dim)  # [out * in/d, d]
        n_fit = min(len(vectors), 200000)
        if len(vectors) > n_fit:
            rng = np.random.RandomState(42)
            fit_idx = rng.choice(len(vectors), n_fit, replace=False)
            fit_vectors = vectors[fit_idx]
        else:
            fit_vectors = vectors
        codebook, _ = _vector_kmeans(fit_vectors, n_clusters, max_iters=10)
        codebook = codebook.astype(np.float32)  # [k, d]

        # Assign all vectors: ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x@c.T
        CHUNK = 50000
        indices_flat = np.empty(len(vectors), dtype=np.uint8)
        C_sq = np.sum(codebook ** 2, axis=1, keepdims=True).T  # [1, k]
        for start in range(0, len(vectors), CHUNK):
            end = min(start + CHUNK, len(vectors))
            X_chunk = vectors[start:end].astype(np.float32)
            X_sq = np.sum(X_chunk ** 2, axis=1, keepdims=True)
            dists = X_sq + C_sq - 2.0 * (X_chunk @ codebook.T)
            indices_flat[start:end] = np.argmin(dists, axis=1).astype(np.uint8)

        indices = indices_flat.reshape(out_dim, in_dim // vector_dim)  # [out, in/d]
        W_q = codebook[indices].reshape(W.shape)  # [out, in/d, d] → [out, in]

    # Compute residuals for sidecar
    residual = W - W_q
    flat_residual = residual.reshape(-1)

    if sidecar_threshold > 0:
        mask = np.abs(flat_residual) > sidecar_threshold
        positions = np.where(mask)[0].astype(np.int64)
        values = flat_residual[mask].astype(np.float32)
    else:
        # Top-256 outliers (match HXQ default)
        n_sidecar = min(256, flat_residual.size)
        top_idx = np.argpartition(np.abs(flat_residual), -n_sidecar)[-n_sidecar:]
        positions = top_idx.astype(np.int64)
        values = flat_residual[top_idx].astype(np.float32)

    # Codebook utilization
    n_active = len(np.unique(indices))
    utilization = n_active / n_clusters

    return {
        "codebook": torch.from_numpy(codebook),
        "indices": torch.from_numpy(indices),
        "sidecar_positions": torch.from_numpy(positions),
        "sidecar_values": torch.from_numpy(values),
        "n_active_centroids": n_active,
        "codebook_utilization": utilization,
        "vector_dim": vector_dim,
    }


def compress_all_linears(
    model: nn.Module,
    n_clusters: int = 256,
    vector_dim: int = 1,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compress all nn.Linear layers (except embeddings/lm_head) in model."""
    compressed = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        # Skip embedding-tied lm_head
        if name == "lm_head":
            continue
        # Only use grouped VQ if in_features is divisible by vector_dim
        vd = vector_dim if mod.weight.shape[1] % vector_dim == 0 else 1
        compressed[name] = compress_linear(mod.weight, n_clusters=n_clusters, vector_dim=vd)
    return compressed


# ---------------------------------------------------------------------------
# STE quantization hook
# ---------------------------------------------------------------------------

class STEQuantizer:
    """Applies STE quantization before each forward pass.

    The shadow weights are the model's actual nn.Parameter tensors.
    Before forward: weight.data = W_shadow + (W_q - W_shadow).detach()
    This makes the forward value = W_q, but backward treats it as identity
    (gradients flow to W_shadow, not through the quantization).
    """

    def __init__(self, model: nn.Module, compressed: Dict[str, Dict[str, torch.Tensor]]):
        self.model = model
        self.compressed = compressed
        self._shadow_cache: Dict[str, torch.Tensor] = {}

    def apply_quantized_weights(self):
        """Replace weights with STE-quantized versions. Save shadows for restore."""
        self._shadow_cache.clear()
        for name, factors in self.compressed.items():
            mod = dict(self.model.named_modules())[name]
            # Cache original weight for restore
            self._shadow_cache[name] = mod.weight.data.clone()
            # Reconstruct quantized weight
            cb = factors["codebook"].to(mod.weight.device)
            idx = factors["indices"].to(mod.weight.device).long()
            W_q = cb[idx]
            # For grouped VQ: [out, in/d, d] → [out, in]
            vd = factors.get("vector_dim", 1)
            if vd > 1:
                W_q = W_q.reshape(mod.weight.shape)
            # Apply sidecar corrections
            sp = factors["sidecar_positions"]
            sv = factors["sidecar_values"].to(mod.weight.device)
            W_q_flat = W_q.reshape(-1)
            W_q_flat[sp] += sv
            W_q = W_q_flat.reshape(mod.weight.shape)
            # STE: forward sees W_q, backward flows to W_shadow
            mod.weight.data = mod.weight.data + (W_q - mod.weight.data).detach()

    def restore_shadow_weights(self):
        """Restore original shadow weights after backward pass."""
        for name, shadow in self._shadow_cache.items():
            mod = dict(self.model.named_modules())[name]
            mod.weight.data = shadow
        self._shadow_cache.clear()


# ---------------------------------------------------------------------------
# Phase 1 Trainer
# ---------------------------------------------------------------------------

class Phase1Trainer:
    """Born-compressed training with dense shadow weights and STE forward.

    Key invariant: compressed forward path is active from step 0.
    Codebooks are initialized BEFORE the first training step.
    """

    def __init__(
        self,
        model: EchoHybridModel,
        lr: float = 1e-4,
        compress_schedule: int = 100,
        n_clusters: int = 256,
        vector_dim: int = 1,
        device: str = "cpu",
        vstep_mode: str = "none",
        gate_enabled: bool = False,
        gate_util_floor: float = 0.8,
        gate_spike_ceil: float = 2.0,
        gate_dead_ceil: float = 0.3,
    ):
        self.model = model.to(device)
        self.device = device
        self.compress_schedule = compress_schedule
        self.n_clusters = n_clusters
        self.vector_dim = vector_dim
        self.vstep_mode = vstep_mode
        self.step_count = 0
        self.halted = False
        self.halt_reason = None

        # MorphSAT training gate
        self.gate = None
        if gate_enabled:
            from echo_hybrid.training_gate import TrainingGate, TrainEvent
            self.gate = TrainingGate(
                util_floor=gate_util_floor,
                spike_ceil=gate_spike_ceil,
                dead_ceil=gate_dead_ceil,
            )

        # Compress all linear layers at init (BEFORE step 0)
        vd_label = f"d={vector_dim}" if vector_dim > 1 else "scalar"
        print(f"Compressing {sum(1 for n,m in model.named_modules() if isinstance(m, nn.Linear) and n != 'lm_head')} linear layers ({vd_label})...")
        self.compressed = compress_all_linears(model, n_clusters=n_clusters, vector_dim=vector_dim)
        self.codebook_init_verified = len(self.compressed) > 0
        print(f"  Compressed {len(self.compressed)} layers, codebooks initialized.")

        # Gate: COMPRESS_DONE with initial utilization signals
        if self.gate is not None:
            signals = self._codebook_signals()
            state, legal, action = self.gate.step(TrainEvent.COMPRESS_DONE, signals=signals)
            print(f"  [gate] INIT → {action} (util={signals.get('codebook_utilization', 0):.3f})")
            if not legal:
                self.halted = True
                self.halt_reason = f"Gate blocked at init: {action}"
                print(f"  [gate] HALTED: {self.halt_reason}")

        self.ste = STEQuantizer(model, self.compressed)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def _codebook_signals(self, recent_losses: list = None, current_loss: float = None) -> Dict[str, float]:
        """Compute guard signals from current codebook state."""
        # Aggregate codebook utilization across all layers (minimum = weakest link)
        utils = [f["codebook_utilization"] for f in self.compressed.values()]
        min_util = min(utils) if utils else 0.0

        # Dead centroid fraction (max across layers)
        dead_fracs = []
        for f in self.compressed.values():
            n_active = f["n_active_centroids"]
            dead_fracs.append(1.0 - n_active / self.n_clusters)
        max_dead = max(dead_fracs) if dead_fracs else 0.0

        signals = {
            "codebook_utilization": round(min_util, 4),
            "dead_centroid_frac": round(max_dead, 4),
        }

        # Loss spike ratio: current_loss / avg(recent_losses)
        if recent_losses and current_loss is not None and len(recent_losses) >= 10:
            recent_avg = sum(recent_losses[-20:]) / len(recent_losses[-20:])
            if recent_avg > 0:
                signals["loss_spike_ratio"] = round(current_loss / recent_avg, 4)

        return signals

    def train_step(self, input_ids: torch.Tensor, recent_losses: list = None) -> float:
        """One training step with compressed forward path.

        Args:
            input_ids: Input token IDs.
            recent_losses: Recent loss history for spike detection (optional,
                           only used when gate is enabled).

        Returns:
            Loss value. If gate halts training, returns the loss but sets
            self.halted = True.
        """
        if self.halted:
            return float("nan")

        import time as _time
        _t0 = _time.monotonic()
        self.model.train()
        input_ids = input_ids.to(self.device)

        # Apply STE quantization
        self.ste.apply_quantized_weights()
        _t1 = _time.monotonic()

        # Forward (compressed path)
        out = self.model(input_ids=input_ids, labels=input_ids)
        loss = out["loss"]
        _t2 = _time.monotonic()

        # Backward (gradients flow to shadow weights via STE)
        self.optimizer.zero_grad()
        loss.backward()
        _t3 = _time.monotonic()

        # CRITICAL: restore shadow weights BEFORE optimizer step.
        self.ste.restore_shadow_weights()
        self.optimizer.step()
        _t4 = _time.monotonic()

        self.step_count += 1

        # V-step: per-step codebook update (Lloyd's reassign or EMA)
        if self.vstep_mode != "none":
            from echo_hybrid.train_vstep import vstep_update_codebooks
            vstep_update_codebooks(
                self.compressed, self.model,
                alpha=0.1 if self.vstep_mode == "ema" else 0.0,
                mode=self.vstep_mode,
            )
            self.ste.compressed = self.compressed
        _t5 = _time.monotonic()

        # Timing (first 5 steps only)
        if self.step_count <= 5:
            print(f"  [timing] step {self.step_count}: "
                  f"ste={_t1-_t0:.2f}s fwd={_t2-_t1:.2f}s bwd={_t3-_t2:.2f}s "
                  f"opt={_t4-_t3:.2f}s vstep={_t5-_t4:.2f}s total={_t5-_t0:.2f}s")

        # Gate: STEP_DONE with signals (check every step when vstep is active)
        loss_val = loss.item()
        if self.gate is not None and self.vstep_mode != "none":
            from echo_hybrid.training_gate import TrainEvent
            signals = self._codebook_signals(recent_losses, loss_val)
            state, legal, action = self.gate.step(TrainEvent.STEP_DONE, signals=signals)
            if not legal:
                self.halted = True
                self.halt_reason = f"Gate blocked at step {self.step_count}: {action}"
                print(f"  [gate] HALTED at step {self.step_count}: {self.gate.history[-1]}")

        # Periodic recompression
        if not self.halted and self.compress_schedule > 0 and self.step_count % self.compress_schedule == 0:
            self._recompress(recent_losses, loss_val)

        return loss_val

    def _recompress(self, recent_losses: list = None, current_loss: float = None):
        """Refresh codebooks from updated shadow weights."""
        # Gate: RECOMPRESS_START
        if self.gate is not None:
            from echo_hybrid.training_gate import TrainEvent
            self.gate.step(TrainEvent.RECOMPRESS_START)

        print(f"  Recompressing at step {self.step_count}...")
        self.compressed = compress_all_linears(self.model, n_clusters=self.n_clusters, vector_dim=self.vector_dim)
        self.ste.compressed = self.compressed

        # Gate: RECOMPRESS_DONE with post-recompression signals
        if self.gate is not None:
            signals = self._codebook_signals(recent_losses, current_loss)
            state, legal, action = self.gate.step(TrainEvent.RECOMPRESS_DONE, signals=signals)
            print(f"  [gate] RECOMPRESS_DONE → {action} (util={signals.get('codebook_utilization', 0):.3f})")
            if not legal:
                self.halted = True
                self.halt_reason = f"Gate blocked after recompress at step {self.step_count}: {action}"
                print(f"  [gate] HALTED: {self.halt_reason}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wikitext_chunks(seq_len: int = 128, max_chunks: int = 2000) -> torch.Tensor:
    """Load WikiText-2 validation as contiguous token chunks."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    text = "\n\n".join([x for x in ds["text"] if x.strip()])
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)

    # Chunk into non-overlapping segments
    n_chunks = min(max_chunks, len(tokens) // seq_len)
    tokens = tokens[:n_chunks * seq_len].reshape(n_chunks, seq_len)
    print(f"Loaded {n_chunks} chunks of {seq_len} tokens from WikiText-2 train.")
    return tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WO-ECHO-HYBRID-01b: Phase 1 training")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--compress-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-clusters", type=int, default=256)
    parser.add_argument("--vector-dim", type=int, default=1,
                        help="VQ group size: 1=scalar, 2/4/8=grouped (production HXQ)")
    parser.add_argument("--vstep-mode", type=str, default="none",
                        choices=["none", "ema", "reassign"],
                        help="Per-step codebook update: none, ema, or reassign (Lloyd's)")
    parser.add_argument("--gate", action="store_true",
                        help="Enable MorphSAT training gate (codebook health enforcement)")
    parser.add_argument("--gate-util-floor", type=float, default=0.8,
                        help="Min codebook utilization before gate halts (default 0.8)")
    parser.add_argument("--gate-spike-ceil", type=float, default=2.0,
                        help="Max loss spike ratio before gate halts (default 2.0)")
    parser.add_argument("--gate-dead-ceil", type=float, default=0.3,
                        help="Max dead centroid fraction before gate halts (default 0.3)")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Track VRAM if CUDA
    vram_peak = 0.0
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Build model
    cfg = EchoHybridConfig()
    model = EchoHybridModel(cfg)
    print(model)

    # Load data
    chunks = load_wikitext_chunks(seq_len=args.seq_len, max_chunks=args.steps * args.batch_size * 2)

    # Initialize trainer (compresses before step 0)
    trainer = Phase1Trainer(
        model=model,
        lr=args.lr,
        compress_schedule=args.compress_every,
        n_clusters=args.n_clusters,
        vector_dim=args.vector_dim,
        device=args.device,
        vstep_mode=args.vstep_mode,
        gate_enabled=args.gate,
        gate_util_floor=args.gate_util_floor,
        gate_spike_ceil=args.gate_spike_ceil,
        gate_dead_ceil=args.gate_dead_ceil,
    )

    # Training loop
    losses = []
    print(f"\nTraining {args.steps} steps, batch={args.batch_size}, seq={args.seq_len}")
    print(f"Compress schedule: every {args.compress_every} steps")
    print(f"V-step mode: {args.vstep_mode}")
    print(f"Device: {args.device}")
    print("-" * 60)

    chunk_idx = 0
    for step in range(args.steps):
        # Build batch
        batch_chunks = []
        for _ in range(args.batch_size):
            batch_chunks.append(chunks[chunk_idx % len(chunks)])
            chunk_idx += 1
        batch = torch.stack(batch_chunks)

        loss = trainer.train_step(batch, recent_losses=losses)
        losses.append(loss)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  step {step+1:4d}/{args.steps}  loss={loss:.4f}")

        if trainer.halted:
            print(f"\n  [gate] Training halted at step {step+1}: {trainer.halt_reason}")
            break

    # VRAM tracking
    if args.device == "cuda" and torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Loss curve analysis
    first_loss = losses[0]
    last_loss = losses[-1]
    min_loss = min(losses)
    monotone_decreasing = bool(all(losses[i] >= losses[i+1] for i in range(len(losses)-1)))
    # Check "believably" decreasing: smoothed trend goes down
    window = max(1, len(losses) // 10)
    smoothed = [np.mean(losses[max(0,i-window):i+1]) for i in range(len(losses))]
    trend_down = bool(smoothed[-1] < smoothed[0])

    print("-" * 60)
    print(f"Loss: {first_loss:.4f} -> {last_loss:.4f} (min={min_loss:.4f})")
    print(f"Trend decreasing: {trend_down}")
    print(f"Strictly monotone: {monotone_decreasing}")

    # Dense cheat check: verify codebooks were initialized before step 0
    dense_cheat = not trainer.codebook_init_verified
    print(f"Dense cheat: {dense_cheat}")
    print(f"Codebook init verified: {trainer.codebook_init_verified}")

    # Emit receipt
    receipt = {
        "wo": "WO-ECHO-HYBRID-01b",
        "terminal": "T_this",
        "timestamp": time.strftime("%Y-%m-%d"),
        "status": "PASS" if (trend_down and trainer.codebook_init_verified and not dense_cheat) else "FAIL",
        "model": f"EchoHybrid-{cfg.n_blocks}block-{cfg.hidden_size}d",
        "block_pattern": cfg.block_pattern,
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.lr,
        "n_clusters": args.n_clusters,
        "vector_dim": args.vector_dim,
        "compress_schedule": args.compress_every,
        "vstep_mode": args.vstep_mode,
        "n_compressed_layers": len(trainer.compressed),
        "first_loss": round(first_loss, 4),
        "final_loss": round(last_loss, 4),
        "min_loss": round(min_loss, 4),
        "trend_decreasing": trend_down,
        "compressed_from_step": 0,
        "codebook_init_verified": trainer.codebook_init_verified,
        "dense_cheat": dense_cheat,
        "vram_peak_mb": round(vram_peak, 1),
        "device": args.device,
        "loss_curve": [round(l, 4) for l in losses],
        "halted": trainer.halted,
        "halt_reason": trainer.halt_reason,
        "gate_receipt": trainer.gate.to_receipt() if trainer.gate is not None else None,
        "notes": (
            f"Phase 1 born-compressed training. {len(trainer.compressed)} linear layers "
            f"compressed via k-means ({args.n_clusters} clusters) with STE forward. "
            f"Codebooks initialized before step 0. "
            f"Naive selective scan (pure PyTorch)."
        ),
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    out_dir = Path("receipts/echo_hybrid")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Distinct receipt name when vstep is active
    suffix = f"_vstep_{args.vstep_mode}" if args.vstep_mode != "none" else ""
    vd_suffix = f"_d{args.vector_dim}" if args.vector_dim > 1 else ""
    out_path = out_dir / f"wo_echo_hybrid_01b{vd_suffix}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nRECEIPT: {out_path}")
    print(f"STATUS: {receipt['status']}")

    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
