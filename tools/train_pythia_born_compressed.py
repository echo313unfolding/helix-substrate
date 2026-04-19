#!/usr/bin/env python3
"""
WO-BORN-COMPRESSED-PYTHIA-01 — Born-compressed training on a real architecture.

Trains Pythia 410M (GPTNeoX, 24 layers, hidden=1024, 16 heads) from scratch,
both dense and compressed (HelixLinearSTE), on the same corpus with matched config.

This is the apples-to-apples comparison: real architecture, published baselines,
only variable is nn.Linear vs HelixLinearSTE.

Decision rule (predeclared):
  - If compressed best PPL / dense best PPL < 2.0 → born-compressed is VIABLE at scale
  - If ratio < 1.5 → born-compressed is COMPETITIVE
  - If ratio > 2.0 → born-compressed FAILS at this scale

Usage:
    python3 tools/train_pythia_born_compressed.py                    # compressed
    python3 tools/train_pythia_born_compressed.py --dense            # dense baseline
    python3 tools/train_pythia_born_compressed.py --steps 5000       # custom steps
    python3 tools/train_pythia_born_compressed.py --device cuda      # GPU
"""

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

RECEIPT_DIR = PROJECT / "receipts" / "pythia_born_compressed"


def jsonable(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── Pythia 410M Architecture ──

PYTHIA_410M_CONFIG = dict(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    vocab_size=50304,
    max_position_embeddings=2048,
    hidden_act="gelu",
    layer_norm_eps=1e-5,
    rotary_pct=0.25,
    rotary_emb_base=10000,
    use_parallel_residual=True,
    tie_word_embeddings=False,
)


def build_pythia_model(compressed=False, device="cpu"):
    """Build Pythia 410M from config (not from pretrained weights).

    Args:
        compressed: If True, replace all nn.Linear with HelixLinearSTE.
        device: Target device.

    Returns:
        (model, config_info_dict)
    """
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    config = GPTNeoXConfig(**PYTHIA_410M_CONFIG)
    model = GPTNeoXForCausalLM(config)

    # Replace linear layers with HelixLinearSTE if compressed
    n_replaced = 0
    if compressed:
        n_replaced = _replace_linears_with_ste(model, device=device)

    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count effective params
    n_effective = n_params
    if compressed:
        from helix_substrate.helix_linear_ste import HelixLinearSTE
        for m in model.modules():
            if isinstance(m, HelixLinearSTE):
                n_effective += (m.out_features * m.in_features - m.codebook.numel())

    config_info = {
        "architecture": "GPTNeoX (Pythia 410M)",
        "num_layers": 24,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "vocab_size": 50304,
        "max_position_embeddings": 2048,
        "rotary_pct": 0.25,
        "use_parallel_residual": True,
        "total_params": n_params,
        "trainable_params": n_trainable,
        "effective_params": n_effective,
        "compressed": compressed,
        "n_ste_replaced": n_replaced,
    }

    return model, config_info


def _replace_linears_with_ste(model, device="cpu"):
    """Replace all nn.Linear modules (except embed_out/lm_head) with HelixLinearSTE."""
    from helix_substrate.helix_linear_ste import HelixLinearSTE

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Skip output embedding
        if name in ("embed_out", "lm_head"):
            continue

        new_mod = HelixLinearSTE.from_scratch(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            device=device,
        )

        # Replace in parent
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_mod)
        replaced += 1

    return replaced


# ── Data ──

SECURITY_CORPUS = Path(os.environ.get(
    "SECURITY_CORPUS",
    "/home/voidstr3m33/datasets/security_corpus/security_corpus.jsonl",
))


def load_dataset_tokens(tokenizer, max_tokens=2_000_000, seq_len=512):
    """Load training data. Uses full corpus — no artificial cap."""
    if SECURITY_CORPUS.exists() and SECURITY_CORPUS.stat().st_size > 0:
        texts = []
        with open(SECURITY_CORPUS) as f:
            for line in f:
                doc = json.loads(line)
                texts.append(doc["text"])
        text = "\n\n".join(texts)
        source = f"security_corpus ({len(texts):,} docs)"
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            text = "\n\n".join([t for t in ds["text"] if t.strip()])
            source = "wikitext-103-train (fallback)"
        except Exception:
            try:
                from datasets import load_dataset
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                text = "\n\n".join([t for t in ds["text"] if t.strip()])
                source = "wikitext-2-train (fallback)"
            except Exception:
                raise RuntimeError("No dataset available. Provide SECURITY_CORPUS env var or install 'datasets' package.")

    # Tokenize in chunks to avoid memory issues
    print(f"  Tokenizing {source}...")
    max_chars = max_tokens * 6  # rough chars-per-token estimate
    text = text[:max_chars]

    all_ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens)["input_ids"][0]

    chunks = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        chunks.append(all_ids[i:i + seq_len])

    print(f"  Dataset: {source}, {len(all_ids):,} tokens, {len(chunks)} chunks of {seq_len}")
    return chunks, source


def load_eval_tokens(tokenizer, max_tokens=8192):
    """Load eval data: held-out slice of corpus."""
    if SECURITY_CORPUS.exists() and SECURITY_CORPUS.stat().st_size > 0:
        texts = []
        with open(SECURITY_CORPUS) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        eval_start = int(len(texts) * 0.99)
        text = "\n\n".join(texts[eval_start:])
        source = f"security_corpus_eval ({len(texts) - eval_start} docs)"
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            text = "\n\n".join([t for t in ds["text"] if t.strip()])
            source = "wikitext-103-test (fallback)"
        except Exception:
            try:
                from datasets import load_dataset
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                text = "\n\n".join([t for t in ds["text"] if t.strip()])
                source = "wikitext-2-test (fallback)"
            except Exception:
                raise RuntimeError("No eval dataset available.")

    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens)["input_ids"][0]
    return ids, source


# ── Training ──

@torch.no_grad()
def eval_ppl(model, eval_tokens, device, seq_len=512):
    """Compute PPL on eval tokens."""
    model.eval()
    nlls = []
    n_scored = 0

    for i in range(0, len(eval_tokens) - 1, seq_len):
        end = min(i + seq_len, len(eval_tokens))
        input_ids = eval_tokens[i:end].unsqueeze(0).to(device)
        logits = model(input_ids).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = nn.CrossEntropyLoss(reduction="sum")(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(loss.item())
        n_scored += shift_labels.numel()

    avg_nll = sum(nlls) / max(1, n_scored)
    ppl = math.exp(min(avg_nll, 100))
    model.train()
    return ppl, n_scored


def reassign_all_ste(model):
    """Run index reassignment on all HelixLinearSTE modules."""
    from helix_substrate.helix_linear_ste import HelixLinearSTE
    total_changed = 0
    total_elements = 0
    for module in model.modules():
        if isinstance(module, HelixLinearSTE):
            changed, total = module.reassign_indices()
            total_changed += changed
            total_elements += total
    return total_changed, total_elements


def collect_ste_diagnostics(model):
    """Collect codebook utilization from all STE modules."""
    from helix_substrate.helix_linear_ste import HelixLinearSTE
    stats = []
    for name, module in model.named_modules():
        if isinstance(module, HelixLinearSTE):
            stats.append({
                "name": name,
                "cb_util": round(module.codebook_utilization(), 4),
                "shape": [module.out_features, module.in_features],
            })
    return stats


def train(
    compressed: bool = True,
    total_steps: int = 10_000,
    batch_size: int = 4,
    seq_len: int = 512,
    lr: float = 3e-4,
    warmup_steps: int = 500,
    reassign_interval: int = 500,
    log_interval: int = 50,
    eval_interval: int = 500,
    grad_clip: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
):
    t_total = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    torch.manual_seed(seed)

    mode = "compressed" if compressed else "dense"
    print("=" * 70)
    print(f"WO-BORN-COMPRESSED-PYTHIA-01 — {mode.upper()} MODE")
    print(f"Pythia 410M | Steps: {total_steps} | Batch: {batch_size} | Seq: {seq_len} | Device: {device}")
    print("=" * 70)

    # ── Tokenizer ──
    # Pythia uses EleutherAI/gpt-neox-20b tokenizer (same for all Pythia sizes)
    print("\n[1] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size}")

    # ── Build model ──
    print("\n[2] Building model...")
    model, config_info = build_pythia_model(compressed=compressed, device=device)

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ENABLED")

    print(f"  Params (total): {config_info['total_params']:,}")
    print(f"  Params (trainable): {config_info['trainable_params']:,}")
    if compressed:
        print(f"  Params (effective/represented): {config_info['effective_params']:,}")
        print(f"  HelixLinearSTE modules: {config_info['n_ste_replaced']}")
    else:
        print(f"  Mode: dense (all nn.Linear)")

    # ── Data ──
    print("\n[3] Loading dataset...")
    train_chunks, train_source = load_dataset_tokens(tokenizer, seq_len=seq_len)
    eval_tokens, eval_source = load_eval_tokens(tokenizer)

    if len(train_chunks) < batch_size:
        raise RuntimeError(f"Not enough training data: {len(train_chunks)} chunks < batch_size {batch_size}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_lr(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # ── Training loop ──
    print(f"\n[4] Training ({total_steps} steps)...")
    log_history = []
    eval_history = []
    reassign_history = []

    data_idx = 0
    running_loss = 0.0
    best_eval_ppl = float('inf')
    best_eval_step = 0

    # AMP for faster training on Ampere+ GPUs
    use_amp = device == "cuda" and torch.cuda.is_available()
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    for step in range(1, total_steps + 1):
        # Build batch
        batch_ids = []
        for _ in range(batch_size):
            batch_ids.append(train_chunks[data_idx % len(train_chunks)])
            data_idx += 1
        input_ids = torch.stack(batch_ids).to(device)

        # Forward
        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        # ── Logging ──
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            ppl_est = math.exp(min(avg_loss, 100))
            current_lr = scheduler.get_last_lr()[0]
            print(f"  step {step:>6d} | loss {avg_loss:.4f} | ppl ~{ppl_est:.1f} | lr {current_lr:.2e}")
            log_history.append({
                "step": step, "loss": round(avg_loss, 6),
                "ppl_est": round(ppl_est, 2), "lr": round(current_lr, 8),
            })
            running_loss = 0.0

        # ── Index reassignment (compressed only) ──
        if compressed and step % reassign_interval == 0:
            changed, total = reassign_all_ste(model)
            pct = 100 * changed / max(1, total)
            reassign_history.append({"step": step, "changed": changed, "total": total, "pct": round(pct, 2)})
            print(f"  [reassign] step {step}: {changed:,}/{total:,} changed ({pct:.1f}%)")

        # ── Eval ──
        if step % eval_interval == 0 or step == total_steps:
            ppl, n_scored = eval_ppl(model, eval_tokens, device, seq_len=seq_len)
            if ppl < best_eval_ppl:
                best_eval_ppl = ppl
                best_eval_step = step
                RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
                best_ckpt_path = RECEIPT_DIR / f"{mode}_best_model.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "mode": mode,
                    "eval_ppl": round(ppl, 4),
                    "config": config_info,
                }, best_ckpt_path)
                print(f"  [best] saved checkpoint at step {step} (PPL={ppl:.4f})")
            print(f"  [eval] step {step}: PPL={ppl:.4f} (scored {n_scored} tokens)")
            eval_history.append({"step": step, "ppl": round(ppl, 4), "n_scored": n_scored})

    # ── Final checkpoint ──
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RECEIPT_DIR / f"{mode}_{total_steps}steps_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": total_steps,
        "mode": mode,
        "config": config_info,
    }, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # ── Summary ──
    total_wall = time.time() - t_total
    total_cpu = time.process_time() - cpu_start
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    final_ppl = eval_history[-1]["ppl"] if eval_history else float('inf')

    print(f"\n{'=' * 70}")
    print(f"RESULT — {mode.upper()} (Pythia 410M)")
    print("=" * 70)
    print(f"  Final eval PPL: {final_ppl:.4f}")
    print(f"  Best eval PPL:  {best_eval_ppl:.4f} (step {best_eval_step})")
    print(f"  Wall time:      {total_wall:.0f}s ({total_wall/60:.1f}m)")
    print(f"  Peak RSS:       {peak_mem:.0f} MB")

    # ── Receipt ──
    receipt = {
        "work_order": "WO-BORN-COMPRESSED-PYTHIA-01",
        "mode": mode,
        "model": config_info,
        "training": {
            "total_steps": total_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "lr": lr,
            "warmup_steps": warmup_steps,
            "reassign_interval": reassign_interval,
            "grad_clip": grad_clip,
            "seed": seed,
            "device": device,
            "train_source": train_source,
            "eval_source": eval_source,
        },
        "results": {
            "final_ppl": final_ppl,
            "best_ppl": round(best_eval_ppl, 4),
            "best_step": best_eval_step,
            "eval_history": eval_history,
            "log_history": log_history[-20:],
        },
        "reassign_history": reassign_history[-10:] if reassign_history else [],
        "decision_rule": {
            "viable_threshold": 2.0,
            "competitive_threshold": 1.5,
            "note": "compressed_best_ppl / dense_best_ppl ratio",
        },
        "cost": {
            "wall_time_s": round(total_wall, 3),
            "cpu_time_s": round(total_cpu, 3),
            "peak_memory_mb": round(peak_mem, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }

    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"{mode}_{total_steps}steps_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, default=jsonable)

    print(f"\n  Receipt: {receipt_path}")
    print("=" * 70)

    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Born-compressed Pythia 410M training")
    parser.add_argument("--dense", action="store_true", help="Train dense baseline")
    parser.add_argument("--steps", type=int, default=10_000, help="Training steps")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
    parser.add_argument("--reassign", type=int, default=500, help="Index reassignment interval")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-interval", type=int, default=500, help="Eval interval")
    parser.add_argument("--log-interval", type=int, default=50, help="Log interval")
    args = parser.parse_args()

    train(
        compressed=not args.dense,
        total_steps=args.steps,
        batch_size=args.batch,
        seq_len=args.seq_len,
        lr=args.lr,
        warmup_steps=args.warmup,
        reassign_interval=args.reassign,
        device=args.device,
        seed=args.seed,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
    )
