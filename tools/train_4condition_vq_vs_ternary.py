#!/usr/bin/env python3
"""
WO-VQ-VS-TERNARY-01 — 4-condition from-scratch pretraining comparison.

The question: "Does learnable VQ codebook pretraining outperform fixed-grid ternary
(BitNet) at comparable bits-per-weight?"

Conditions (all same architecture, data, budget):
  1. dense     — standard nn.Linear (FP16/FP32 baseline)
  2. bitnet    — BitNet b1.58 ternary {-1,0,+1} with STE (1.58 bpw)
  3. vq1       — HXQ scalar VQ, 256-entry codebook, vector_dim=1 (8 bpw index)
  4. vq4       — HXQ grouped VQ, 256-entry codebook, vector_dim=4 (2 bpw effective)

Architecture: GPTNeoX 160M (Pythia-160M config) — real published architecture.
12 layers, hidden=768, 12 heads, intermediate=3072, vocab=50304.

Decision rules (predeclared):
  - VQ viable if best_ppl / dense_best_ppl < 2.0
  - VQ competitive if ratio < 1.5
  - VQ beats ternary if vq_best_ppl < bitnet_best_ppl

Usage:
    python3 tools/train_4condition_vq_vs_ternary.py --mode dense    --device cuda
    python3 tools/train_4condition_vq_vs_ternary.py --mode bitnet   --device cuda
    python3 tools/train_4condition_vq_vs_ternary.py --mode vq1      --device cuda
    python3 tools/train_4condition_vq_vs_ternary.py --mode vq4      --device cuda
    python3 tools/train_4condition_vq_vs_ternary.py --mode all      --device cuda  # sequential
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
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

RECEIPT_DIR = PROJECT / "receipts" / "vq_vs_ternary"


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


# ═══════════════════════════════════════════════════════════════════════
# BitNet b1.58 — Ternary weights with STE
# ═══════════════════════════════════════════════════════════════════════

class _WeightQuant(torch.autograd.Function):
    """Ternary weight quantization: w -> {-1, 0, +1} * scale, STE backward."""
    @staticmethod
    def forward(ctx, weight):
        scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        return (weight * scale).round().clamp(-1, 1) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ActQuant(torch.autograd.Function):
    """Per-token symmetric 8-bit activation quantization with STE."""
    @staticmethod
    def forward(ctx, x):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (x * scale).round().clamp(-128, 127) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BitLinear(nn.Module):
    """BitNet b1.58 linear layer — ternary weights, 8-bit activations, STE."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, x):
        w = _WeightQuant.apply(self.weight)
        x_q = _ActQuant.apply(x)
        out = F.linear(x_q, w, self.bias)
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bits=1.58"


# ═══════════════════════════════════════════════════════════════════════
# Model builder — GPTNeoX 160M (Pythia-160M architecture)
# ═══════════════════════════════════════════════════════════════════════

PYTHIA_160M_CONFIG = dict(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    vocab_size=50304,
    max_position_embeddings=2048,
    hidden_act="gelu",
    layer_norm_eps=1e-5,
    rotary_pct=0.25,
    rotary_emb_base=10000,
    use_parallel_residual=True,
    tie_word_embeddings=False,
)


def build_model(mode="dense", device="cpu"):
    """Build GPTNeoX 160M with the specified weight representation.

    Args:
        mode: "dense" | "bitnet" | "vq1" | "vq4"
        device: target device
    """
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    config = GPTNeoXConfig(**PYTHIA_160M_CONFIG)
    model = GPTNeoXForCausalLM(config)

    n_replaced = 0
    if mode == "bitnet":
        n_replaced = _replace_with_bitlinear(model)
    elif mode == "vq1":
        n_replaced = _replace_with_ste(model, vector_dim=1, device=device)
    elif mode == "vq4":
        n_replaced = _replace_with_ste(model, vector_dim=4, device=device)

    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Effective params for compressed modes
    n_effective = n_params
    if mode in ("vq1", "vq4"):
        from helix_substrate.helix_linear_ste import HelixLinearSTE
        for m in model.modules():
            if isinstance(m, HelixLinearSTE):
                n_effective += (m.out_features * m.in_features - m.codebook.numel())

    # Bits per weight
    if mode == "dense":
        bpw = 32.0
    elif mode == "bitnet":
        bpw = 1.58
    elif mode == "vq1":
        bpw = 8.0  # 8-bit index per scalar weight
    elif mode == "vq4":
        bpw = 2.0  # 8-bit index per 4-element group = 2 bpw

    config_info = {
        "architecture": "GPTNeoX 160M (Pythia-160M)",
        "mode": mode,
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "vocab_size": 50304,
        "total_params": n_params,
        "trainable_params": n_trainable,
        "effective_params": n_effective,
        "n_replaced": n_replaced,
        "bits_per_weight": bpw,
    }

    return model, config_info


def _replace_with_bitlinear(model):
    """Replace nn.Linear with BitLinear (except embed_out)."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name in ("embed_out", "lm_head"):
            continue
        new_mod = BitLinear(module.in_features, module.out_features,
                           bias=module.bias is not None)
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_mod)
        replaced += 1
    return replaced


def _replace_with_ste(model, vector_dim=1, device="cpu"):
    """Replace nn.Linear with HelixLinearSTE."""
    from helix_substrate.helix_linear_ste import HelixLinearSTE
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name in ("embed_out", "lm_head"):
            continue
        # Skip layers where in_features not divisible by vector_dim
        if module.in_features % vector_dim != 0:
            continue
        new_mod = HelixLinearSTE.from_scratch(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            vector_dim=vector_dim,
            device=device,
        )
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_mod)
        replaced += 1
    return replaced


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

SECURITY_CORPUS = Path(os.environ.get(
    "SECURITY_CORPUS",
    "/home/voidstr3m33/datasets/security_corpus/security_corpus.jsonl",
))


def load_train_tokens(tokenizer, max_tokens=2_000_000, seq_len=512):
    """Load training data from security corpus or WikiText fallback."""
    if SECURITY_CORPUS.exists() and SECURITY_CORPUS.stat().st_size > 0:
        texts = []
        with open(SECURITY_CORPUS) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        text = "\n\n".join(texts)
        source = f"security_corpus ({len(texts):,} docs)"
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            text = "\n\n".join([t for t in ds["text"] if t.strip()])
            source = "wikitext-103-train"
        except Exception:
            try:
                from datasets import load_dataset
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                text = "\n\n".join([t for t in ds["text"] if t.strip()])
                source = "wikitext-2-train"
            except Exception:
                raise RuntimeError("No dataset. Set SECURITY_CORPUS or install datasets.")

    text = text[:max_tokens * 6]
    print(f"  Tokenizing {source}...")
    all_ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens)["input_ids"][0]

    chunks = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        chunks.append(all_ids[i:i + seq_len])

    print(f"  Dataset: {source}, {len(all_ids):,} tokens, {len(chunks)} chunks of {seq_len}")
    return chunks, source


def load_eval_tokens(tokenizer, max_tokens=8192):
    """Load held-out eval tokens."""
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
            source = "wikitext-103-test"
        except Exception:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join([t for t in ds["text"] if t.strip()])
            source = "wikitext-2-test"

    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens)["input_ids"][0]
    return ids, source


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_ppl(model, eval_tokens, device, seq_len=512):
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
    from helix_substrate.helix_linear_ste import HelixLinearSTE
    total_changed = 0
    total_elements = 0
    for module in model.modules():
        if isinstance(module, HelixLinearSTE):
            changed, total = module.reassign_indices()
            total_changed += changed
            total_elements += total
    return total_changed, total_elements


def train_one_condition(
    mode: str = "dense",
    total_steps: int = 5000,
    batch_size: int = 8,
    seq_len: int = 512,
    lr: float = 3e-4,
    warmup_steps: int = 300,
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
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    print("=" * 70)
    print(f"WO-VQ-VS-TERNARY-01 — {mode.upper()}")
    print(f"GPTNeoX 160M | Steps: {total_steps} | Batch: {batch_size} | Seq: {seq_len}")
    print("=" * 70)

    # ── Tokenizer ──
    print("\n[1] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer)}")

    # ── Model ──
    print("\n[2] Building model...")
    model, config_info = build_model(mode=mode, device=device)

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ENABLED")

    print(f"  Mode: {mode}")
    print(f"  Params (total): {config_info['total_params']:,}")
    print(f"  Params (trainable): {config_info['trainable_params']:,}")
    if mode in ("vq1", "vq4"):
        print(f"  Params (effective): {config_info['effective_params']:,}")
    print(f"  Bits per weight: {config_info['bits_per_weight']}")
    print(f"  Layers replaced: {config_info['n_replaced']}")

    # ── Data ──
    print("\n[3] Loading dataset...")
    train_chunks, train_source = load_train_tokens(tokenizer, seq_len=seq_len)
    eval_tokens, eval_source = load_eval_tokens(tokenizer)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_lr(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # ── Training ──
    print(f"\n[4] Training ({total_steps} steps)...")
    log_history = []
    eval_history = []
    reassign_history = []

    data_idx = 0
    running_loss = 0.0
    best_eval_ppl = float('inf')
    best_eval_step = 0

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    amp_dtype = torch.float16

    for step in range(1, total_steps + 1):
        batch_ids = []
        for _ in range(batch_size):
            batch_ids.append(train_chunks[data_idx % len(train_chunks)])
            data_idx += 1
        input_ids = torch.stack(batch_ids).to(device)

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

        # VQ index reassignment
        if mode in ("vq1", "vq4") and step % reassign_interval == 0:
            changed, total = reassign_all_ste(model)
            pct = 100 * changed / max(1, total)
            reassign_history.append({"step": step, "changed": changed, "total": total, "pct": round(pct, 2)})
            print(f"  [reassign] step {step}: {changed:,}/{total:,} changed ({pct:.1f}%)")

        # Eval
        if step % eval_interval == 0 or step == total_steps:
            ppl, n_scored = eval_ppl(model, eval_tokens, device, seq_len=seq_len)
            if ppl < best_eval_ppl:
                best_eval_ppl = ppl
                best_eval_step = step
                RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step, "mode": mode,
                    "eval_ppl": round(ppl, 4),
                    "config": config_info,
                }, RECEIPT_DIR / f"{mode}_best_model.pt")
                print(f"  [best] saved at step {step} (PPL={ppl:.4f})")
            print(f"  [eval] step {step}: PPL={ppl:.4f} (scored {n_scored} tokens)")
            eval_history.append({"step": step, "ppl": round(ppl, 4), "n_scored": n_scored})

    # ── Summary ──
    total_wall = time.time() - t_total
    total_cpu = time.process_time() - cpu_start
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    final_ppl = eval_history[-1]["ppl"] if eval_history else float('inf')

    print(f"\n{'=' * 70}")
    print(f"RESULT — {mode.upper()} (GPTNeoX 160M)")
    print("=" * 70)
    print(f"  Final eval PPL: {final_ppl:.4f}")
    print(f"  Best eval PPL:  {best_eval_ppl:.4f} (step {best_eval_step})")
    print(f"  Bits per weight: {config_info['bits_per_weight']}")
    print(f"  Wall time:      {total_wall:.0f}s ({total_wall/60:.1f}m)")
    print(f"  Peak RSS:       {peak_mem:.0f} MB")

    # ── Receipt ──
    receipt = {
        "work_order": "WO-VQ-VS-TERNARY-01",
        "mode": mode,
        "model": config_info,
        "training": {
            "total_steps": total_steps, "batch_size": batch_size,
            "seq_len": seq_len, "lr": lr, "warmup_steps": warmup_steps,
            "reassign_interval": reassign_interval, "grad_clip": grad_clip,
            "seed": seed, "device": device,
            "train_source": train_source, "eval_source": eval_source,
        },
        "results": {
            "final_ppl": final_ppl,
            "best_ppl": round(best_eval_ppl, 4),
            "best_step": best_eval_step,
            "eval_history": eval_history,
            "log_history": log_history[-20:],
        },
        "reassign_history": reassign_history[-10:] if reassign_history else [],
        "decision_rules": {
            "viable_threshold": 2.0,
            "competitive_threshold": 1.5,
            "note": "compressed_best_ppl / dense_best_ppl",
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

    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = RECEIPT_DIR / f"{mode}_{total_steps}steps_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, default=jsonable)
    print(f"  Receipt: {receipt_path}")
    print("=" * 70)

    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-condition VQ vs ternary pretraining")
    parser.add_argument("--mode", type=str, default="dense",
                        choices=["dense", "bitnet", "vq1", "vq4", "all"],
                        help="Training mode")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--reassign", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    modes = ["dense", "bitnet", "vq1", "vq4"] if args.mode == "all" else [args.mode]

    all_results = {}
    for mode in modes:
        print(f"\n\n{'#' * 70}")
        print(f"# CONDITION: {mode.upper()}")
        print(f"{'#' * 70}\n")

        result = train_one_condition(
            mode=mode,
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
        all_results[mode] = result

        # Free GPU memory between conditions
        if args.device == "cuda":
            torch.cuda.empty_cache()
            import gc; gc.collect()

    # ── Comparison summary ──
    if len(all_results) > 1:
        print(f"\n\n{'=' * 70}")
        print("COMPARISON SUMMARY")
        print("=" * 70)
        dense_best = all_results.get("dense", {}).get("results", {}).get("best_ppl", float('inf'))
        for mode, result in all_results.items():
            best = result["results"]["best_ppl"]
            bpw = result["model"]["bits_per_weight"]
            ratio = best / dense_best if dense_best < float('inf') else float('inf')
            verdict = "BASELINE" if mode == "dense" else (
                "COMPETITIVE" if ratio < 1.5 else "VIABLE" if ratio < 2.0 else "FAIL"
            )
            print(f"  {mode:>8s}: best_ppl={best:>10.2f}  bpw={bpw:>5.2f}  ratio={ratio:>5.2f}x  [{verdict}]")
        print("=" * 70)

        # Save comparison receipt
        comparison = {
            "work_order": "WO-VQ-VS-TERNARY-01",
            "type": "comparison",
            "conditions": {m: {"best_ppl": r["results"]["best_ppl"],
                              "bpw": r["model"]["bits_per_weight"],
                              "wall_time_s": r["cost"]["wall_time_s"]}
                          for m, r in all_results.items()},
            "dense_best_ppl": dense_best,
            "ratios": {m: round(r["results"]["best_ppl"] / dense_best, 4)
                      for m, r in all_results.items() if dense_best < float('inf')},
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        comp_path = RECEIPT_DIR / f"comparison_{time.strftime('%Y%m%dT%H%M%S')}.json"
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"  Comparison receipt: {comp_path}")
