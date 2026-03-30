#!/usr/bin/env python3
"""Batch GPU BF16 PPL evaluation for all HuggingFace model zoo models.

Evaluates both dense and helix models on GPU with bfloat16, consistent methodology.
Outputs one JSON receipt per model to receipts/gpu_eval/.

Usage:
    python3 batch_gpu_eval.py                    # all models
    python3 batch_gpu_eval.py --model tinyllama   # single model
    python3 batch_gpu_eval.py --list              # show model list
"""

import argparse
import json
import os
import sys
import time
import platform
import gc
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset


# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "tinyllama": {
        "dense_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "helix_id": "EchoLabs33/tinyllama-1.1b-helix",
        "type": "transformer",
        "max_length": 2048,
    },
    "qwen2.5-3b": {
        "dense_id": "Qwen/Qwen2.5-3B-Instruct",
        "helix_id": "EchoLabs33/qwen2.5-3b-instruct-helix",
        "type": "transformer",
        "max_length": 2048,
    },
    "qwen2.5-coder-1.5b": {
        "dense_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "helix_id": "EchoLabs33/qwen2.5-coder-1.5b-helix",
        "type": "transformer",
        "max_length": 2048,
    },
    "qwen2.5-coder-3b": {
        "dense_id": "Qwen/Qwen2.5-Coder-3B",
        "helix_id": "EchoLabs33/qwen2.5-coder-3b-helix",
        "type": "transformer",
        "max_length": 2048,
    },
    "qwen2.5-7b": {
        "dense_id": "Qwen/Qwen2.5-7B-Instruct",
        "helix_id": "EchoLabs33/qwen2.5-7b-instruct-helix",
        "type": "transformer",
        "max_length": 2048,
    },
    "qwen2.5-14b": {
        "dense_id": "Qwen/Qwen2.5-14B-Instruct",
        "helix_id": "EchoLabs33/qwen2.5-14b-instruct-helix",
        "type": "transformer",
        "max_length": 2048,  # was 1024 due to VRAM, try 2048 first
    },
    "zamba2-1.2b": {
        "dense_id": "Zyphra/Zamba2-1.2B",
        "helix_id": "EchoLabs33/zamba2-1.2b-helix",
        "type": "hybrid",
        "max_length": 2048,
    },
    "zamba2-2.7b": {
        "dense_id": "Zyphra/Zamba2-2.7B-instruct",
        "helix_id": "EchoLabs33/zamba2-2.7b-instruct-helix",
        "type": "hybrid",
        "max_length": 2048,
    },
    "zamba2-7b": {
        "dense_id": "Zyphra/Zamba2-7B-Instruct",
        "helix_id": "EchoLabs33/zamba2-7b-instruct-helix",
        "type": "hybrid",
        "max_length": 2048,
    },
    "mamba2-1.3b": {
        "dense_id": "state-spaces/mamba2-1.3b",
        "helix_id": "EchoLabs33/mamba2-1.3b-helix",
        "type": "mamba",
        "max_length": 2048,
        "note": "embedding bug in HF quantizer - nn.Embedding not replaced",
    },
    "mamba-130m": {
        "dense_id": "state-spaces/mamba-130m",
        "helix_id": "EchoLabs33/mamba-130m-helix",
        "type": "mamba",
        "max_length": 2048,
        "note": "may have same embedding bug as mamba2-1.3b",
    },
}

STRIDE = 512
DTYPE = torch.bfloat16
DEVICE = "cuda"


def get_wikitext_encodings(tokenizer, max_length):
    """Load WikiText-2 test split and encode."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    return encodings


def eval_ppl(model, encodings, max_length, stride, device):
    """Sliding-window PPL evaluation."""
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls = []
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end

        input_chunk = input_ids[:, begin:end]
        target_chunk = input_chunk.clone()
        target_chunk[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            nll = outputs.loss.float().item()

        nlls.append(nll * target_len)
        n_tokens += target_len
        prev_end = end

        if end == seq_len:
            break

    ppl = float(np.exp(sum(nlls) / n_tokens))
    return ppl, n_tokens


def free_gpu():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def eval_transformer_pair(spec, receipt_dir):
    """Evaluate a transformer or hybrid model pair (dense + helix)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = spec["dense_id"].split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    result = {
        "model": name,
        "dtype": "bfloat16",
        "device": DEVICE,
        "max_length": spec["max_length"],
        "stride": STRIDE,
        "dataset": "wikitext-2-raw-v1",
    }

    # Load tokenizer (use helix tokenizer — it's identical)
    print(f"Loading tokenizer from {spec['helix_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(spec["helix_id"])
    encodings = get_wikitext_encodings(tokenizer, spec["max_length"])
    print(f"Encoded {encodings.input_ids.size(1)} tokens")

    # ── Dense eval ──
    t0 = time.time()
    ts_start = time.strftime('%Y-%m-%dT%H:%M:%S')
    print(f"\nLoading dense model: {spec['dense_id']}...")
    try:
        dense_model = AutoModelForCausalLM.from_pretrained(
            spec["dense_id"], torch_dtype=DTYPE
        ).to(DEVICE).eval()

        print("Evaluating dense PPL...")
        dense_ppl, n_tokens = eval_ppl(
            dense_model, encodings, spec["max_length"], STRIDE, DEVICE
        )
        print(f"Dense PPL: {dense_ppl:.4f} ({n_tokens} tokens)")
        result["dense_ppl"] = round(dense_ppl, 4)
        result["n_tokens"] = n_tokens

        del dense_model
        free_gpu()

    except Exception as e:
        print(f"Dense eval FAILED: {e}")
        result["dense_ppl"] = None
        result["dense_error"] = str(e)
        free_gpu()

    # ── Helix eval ──
    print(f"\nLoading helix model: {spec['helix_id']}...")
    import helix_substrate  # registers quantizer

    helix_model = AutoModelForCausalLM.from_pretrained(
        spec["helix_id"], torch_dtype=DTYPE
    ).to(DEVICE).eval()

    # Count helix modules
    from helix_substrate.helix_linear import HelixLinear
    n_helix = sum(1 for m in helix_model.modules() if isinstance(m, HelixLinear))
    print(f"HelixLinear modules: {n_helix}")
    result["helix_modules"] = n_helix

    print("Evaluating helix PPL...")
    helix_ppl, n_tokens = eval_ppl(
        helix_model, encodings, spec["max_length"], STRIDE, DEVICE
    )
    print(f"Helix PPL: {helix_ppl:.4f}")
    result["helix_ppl"] = round(helix_ppl, 4)

    del helix_model
    free_gpu()

    # ── Compute delta ──
    if result.get("dense_ppl"):
        delta = (result["helix_ppl"] - result["dense_ppl"]) / result["dense_ppl"] * 100
        result["delta_pct"] = round(delta, 2)
        print(f"Delta: +{delta:.2f}%")

    ts_end = time.strftime('%Y-%m-%dT%H:%M:%S')
    result["cost"] = {
        "wall_time_s": round(time.time() - t0, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": ts_start,
        "timestamp_end": ts_end,
    }

    # Save receipt
    os.makedirs(receipt_dir, exist_ok=True)
    slug = name.lower().replace(".", "_").replace("-", "_")
    receipt_path = os.path.join(receipt_dir, f"{slug}_gpu_bf16.json")
    with open(receipt_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Receipt: {receipt_path}")

    return result


def eval_mamba_pair(spec, receipt_dir):
    """Evaluate Mamba models. Dense uses mamba_ssm loader, helix uses HF + embedding fix."""
    name = spec["dense_id"].split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Evaluating (Mamba): {name}")
    print(f"{'='*60}")

    result = {
        "model": name,
        "dtype": "bfloat16",
        "device": DEVICE,
        "max_length": spec["max_length"],
        "stride": STRIDE,
        "dataset": "wikitext-2-raw-v1",
        "note": spec.get("note", ""),
    }

    from transformers import AutoTokenizer

    # Use helix tokenizer
    print(f"Loading tokenizer from {spec['helix_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(spec["helix_id"])
    encodings = get_wikitext_encodings(tokenizer, spec["max_length"])
    print(f"Encoded {encodings.input_ids.size(1)} tokens")

    t0 = time.time()
    ts_start = time.strftime('%Y-%m-%dT%H:%M:%S')

    # ── Dense eval via mamba_ssm ──
    print(f"\nLoading dense model via mamba_ssm: {spec['dense_id']}...")
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        dense_model = MambaLMHeadModel.from_pretrained(
            spec["dense_id"], dtype=DTYPE, device=DEVICE
        )
        dense_model.eval()

        # Manual PPL eval for mamba_ssm models (no labels kwarg)
        print("Evaluating dense PPL (mamba_ssm path)...")
        input_ids = encodings.input_ids.to(DEVICE)
        seq_len = input_ids.size(1)
        nlls = []
        n_tokens = 0
        prev_end = 0

        for begin in range(0, seq_len, STRIDE):
            end = min(begin + spec["max_length"], seq_len)
            target_len = end - prev_end

            input_chunk = input_ids[:, begin:end]
            with torch.no_grad():
                logits = dense_model(input_chunk).logits

            # Compute cross-entropy loss manually
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_chunk[:, 1:].contiguous()

            # Only score the new tokens
            if target_len < end - begin:
                offset = (end - begin) - target_len - 1
                shift_logits = shift_logits[:, offset:, :]
                shift_labels = shift_labels[:, offset:]

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).float().item()

            actual_tokens = shift_labels.numel()
            nlls.append(loss * actual_tokens)
            n_tokens += actual_tokens
            prev_end = end

            if end == seq_len:
                break

        dense_ppl = float(np.exp(sum(nlls) / n_tokens))
        print(f"Dense PPL: {dense_ppl:.4f} ({n_tokens} tokens)")
        result["dense_ppl"] = round(dense_ppl, 4)
        result["n_tokens"] = n_tokens
        result["dense_loader"] = "mamba_ssm"

        # Save dense embedding for helix fix
        if hasattr(dense_model, 'backbone') and hasattr(dense_model.backbone, 'embeddings'):
            dense_emb_weight = dense_model.backbone.embeddings.weight.detach().clone().cpu()
        elif hasattr(dense_model, 'backbone') and hasattr(dense_model.backbone, 'embedding'):
            dense_emb_weight = dense_model.backbone.embedding.weight.detach().clone().cpu()
        else:
            dense_emb_weight = None
            print("WARNING: Could not find dense embedding to fix helix model")

        del dense_model
        free_gpu()

    except ImportError:
        print("mamba_ssm not available — skipping dense eval")
        result["dense_ppl"] = None
        result["dense_error"] = "mamba_ssm not installed"
        dense_emb_weight = None
    except Exception as e:
        print(f"Dense eval FAILED: {e}")
        result["dense_ppl"] = None
        result["dense_error"] = str(e)
        dense_emb_weight = None
        free_gpu()

    # ── Helix eval via HF quantizer + embedding fix ──
    print(f"\nLoading helix model: {spec['helix_id']}...")
    import helix_substrate
    from transformers import AutoModelForCausalLM

    helix_model = AutoModelForCausalLM.from_pretrained(
        spec["helix_id"], torch_dtype=DTYPE
    )

    # Fix embedding if we have the dense weights
    emb_fixed = False
    if dense_emb_weight is not None:
        if hasattr(helix_model, 'backbone') and hasattr(helix_model.backbone, 'embeddings'):
            emb_module = helix_model.backbone.embeddings
        elif hasattr(helix_model, 'backbone') and hasattr(helix_model.backbone, 'embedding'):
            emb_module = helix_model.backbone.embedding
        else:
            emb_module = None

        if emb_module is not None and hasattr(emb_module, 'weight'):
            # Check if embedding looks random (cosine with dense < 0.5)
            cos = torch.nn.functional.cosine_similarity(
                emb_module.weight.detach().cpu().flatten().float(),
                dense_emb_weight.flatten().float(),
                dim=0
            ).item()
            print(f"Embedding cosine with dense: {cos:.4f}")
            if cos < 0.9:
                print("Embedding appears random — injecting dense embedding")
                emb_module.weight.data = dense_emb_weight.to(DTYPE)
                emb_fixed = True
                result["embedding_fix"] = "injected_dense_embedding"
            else:
                print("Embedding matches dense — no fix needed")

    helix_model = helix_model.to(DEVICE).eval()

    from helix_substrate.helix_linear import HelixLinear
    n_helix = sum(1 for m in helix_model.modules() if isinstance(m, HelixLinear))
    print(f"HelixLinear modules: {n_helix}")
    result["helix_modules"] = n_helix

    # PPL eval using manual path (mamba models don't support labels kwarg in HF)
    print("Evaluating helix PPL...")
    input_ids = encodings.input_ids.to(DEVICE)
    seq_len = input_ids.size(1)
    nlls = []
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, STRIDE):
        end = min(begin + spec["max_length"], seq_len)
        target_len = end - prev_end

        input_chunk = input_ids[:, begin:end]
        with torch.no_grad():
            outputs = helix_model(input_chunk)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_chunk[:, 1:].contiguous()

        if target_len < end - begin:
            offset = (end - begin) - target_len - 1
            shift_logits = shift_logits[:, offset:, :]
            shift_labels = shift_labels[:, offset:]

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).float().item()

        actual_tokens = shift_labels.numel()
        nlls.append(loss * actual_tokens)
        n_tokens += actual_tokens
        prev_end = end

        if end == seq_len:
            break

    helix_ppl = float(np.exp(sum(nlls) / n_tokens))
    print(f"Helix PPL: {helix_ppl:.4f}")
    result["helix_ppl"] = round(helix_ppl, 4)

    del helix_model
    free_gpu()

    # ── Compute delta ──
    if result.get("dense_ppl"):
        delta = (result["helix_ppl"] - result["dense_ppl"]) / result["dense_ppl"] * 100
        result["delta_pct"] = round(delta, 2)
        print(f"Delta: +{delta:.2f}%")

    ts_end = time.strftime('%Y-%m-%dT%H:%M:%S')
    result["cost"] = {
        "wall_time_s": round(time.time() - t0, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": ts_start,
        "timestamp_end": ts_end,
    }

    # Save receipt
    os.makedirs(receipt_dir, exist_ok=True)
    slug = name.lower().replace(".", "_").replace("-", "_")
    receipt_path = os.path.join(receipt_dir, f"{slug}_gpu_bf16.json")
    with open(receipt_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Receipt: {receipt_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch GPU BF16 PPL evaluation")
    parser.add_argument("--model", type=str, help="Evaluate single model (key name)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--receipt-dir", type=str, default="receipts/gpu_eval",
                        help="Output directory for receipts")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Skip dense eval (only eval helix)")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for key, spec in MODELS.items():
            note = f" ({spec['note']})" if spec.get('note') else ""
            print(f"  {key:25s} {spec['type']:12s} {spec['dense_id']}{note}")
        return

    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(MODELS.keys())}")
            sys.exit(1)
        models_to_eval = {args.model: MODELS[args.model]}
    else:
        models_to_eval = MODELS

    print(f"Batch GPU BF16 evaluation")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"Models: {len(models_to_eval)}")
    print(f"Receipt dir: {args.receipt_dir}")

    results = {}
    for key, spec in models_to_eval.items():
        try:
            if spec["type"] == "mamba":
                result = eval_mamba_pair(spec, args.receipt_dir)
            else:
                result = eval_transformer_pair(spec, args.receipt_dir)
            results[key] = result
        except Exception as e:
            print(f"\nFATAL ERROR evaluating {key}: {e}")
            import traceback
            traceback.print_exc()
            results[key] = {"model": key, "error": str(e)}
            free_gpu()

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'Model':30s} {'Dense':>10s} {'Helix':>10s} {'Delta':>10s}")
    print(f"{'='*70}")
    for key, r in results.items():
        dense = f"{r['dense_ppl']:.4f}" if r.get('dense_ppl') else "—"
        helix = f"{r['helix_ppl']:.4f}" if r.get('helix_ppl') else "ERROR"
        delta = f"+{r['delta_pct']:.2f}%" if r.get('delta_pct') else "—"
        print(f"{key:30s} {dense:>10s} {helix:>10s} {delta:>10s}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
