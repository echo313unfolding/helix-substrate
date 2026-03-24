#!/usr/bin/env python3
"""
WO-MAMBA-HELIX-01 — Verification benchmark for Mamba-130m through HelixLinear.

Exercises:
  1. Load MambaForCausalLM shell from HuggingFace (local or remote)
  2. Swap nn.Linear modules to HelixLinear via CDNA v3 factors
  3. Verify: module counts, no NaN, finite logits
  4. Perplexity on WikiText-2 test (or hardcoded fallback corpus)
  5. Memory usage and tok/s for short generation
  6. Emit receipt to receipts/mamba_compress/

CPU-only. Mamba-130m fits comfortably in RAM.

Usage:
    python3 tools/bench_mamba_helix.py
"""

import json
import os
import platform
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

# ── Project path ──
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

# ── Constants ──
MODEL_LOCAL = Path.home() / "models" / "mamba-130m-hf"
MODEL_HF_ID = "state-spaces/mamba-130m-hf"
CDNA_DIR = MODEL_LOCAL / "cdnav3"

# Hardcoded fallback corpus (~2000 tokens) if WikiText-2 unavailable.
# Source: opening of Wikipedia article on information theory (public domain).
FALLBACK_CORPUS = (
    "Information theory is the mathematical study of the quantification, storage, "
    "and communication of information. The field was originally established by the "
    "works of Harry Nyquist and Ralph Hartley, in the 1920s, and Claude Shannon in "
    "the 1940s. The field is at the intersection of probability theory, statistics, "
    "computer science, statistical mechanics, information engineering, and electrical "
    "engineering. A key measure in information theory is entropy. Entropy quantifies "
    "the amount of uncertainty involved in the value of a random variable or the "
    "outcome of a random process. For example, identifying the outcome of a fair "
    "coin flip provides less information than specifying the outcome from a roll of "
    "a die. Shannon's main result, the noisy-channel coding theorem, showed that "
    "reliable communication is possible over noisy channels provided that the rate "
    "of communication is below the channel capacity. The channel capacity can be "
    "approached in practice by using appropriate encoding and decoding systems. "
    "Information theory has found applications in many fields including physics, "
    "computer science, linguistics, and electrical engineering. It has had a "
    "profound impact on the design of digital communication systems, data "
    "compression algorithms, and error-correcting codes. The concept of information "
    "entropy was introduced by Claude Shannon in his 1948 paper 'A Mathematical "
    "Theory of Communication'. Shannon is considered the father of information "
    "theory. The basic idea is that the information content of a message can be "
    "measured in bits, and that the entropy of a source is the average information "
    "per symbol. Shannon showed that the entropy of a source sets a fundamental "
    "limit on lossless data compression. This result has had enormous practical "
    "significance. Modern compression algorithms like Huffman coding and arithmetic "
    "coding are direct applications of Shannon's theory. In addition to compression, "
    "information theory provides the theoretical foundation for error-correcting "
    "codes. These codes add redundancy to transmitted data so that errors introduced "
    "by the channel can be detected and corrected. Turbo codes and low-density "
    "parity-check codes approach the Shannon limit on channel capacity. The field "
    "has also found unexpected applications in biology, where information-theoretic "
    "measures are used to quantify the information content of DNA sequences and "
    "protein structures. In machine learning, concepts from information theory such "
    "as mutual information and the information bottleneck principle are used to "
    "understand and improve learning algorithms. The connections between information "
    "theory and statistical physics have led to deep insights in both fields. The "
    "maximum entropy principle, which states that the probability distribution that "
    "best represents the current state of knowledge is the one with largest entropy, "
    "has applications in both physics and machine learning. Information theory "
    "continues to be an active area of research with new applications emerging in "
    "quantum information theory, network information theory, and the study of "
    "complex systems. The fundamental limits established by Shannon remain relevant "
    "as we develop new communication and computation technologies."
)


def jsonable(obj):
    """JSON serializer for numpy/torch types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_perplexity_corpus(tokenizer, max_tokens=2048):
    """Load WikiText-2 test split; fall back to hardcoded corpus."""
    source = "wikitext-2-test"
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([t for t in ds["text"] if t.strip()])
        if len(text) < 100:
            raise ValueError("WikiText-2 text too short")
    except Exception as e:
        print(f"  WikiText-2 unavailable ({e}), using hardcoded fallback corpus.")
        text = FALLBACK_CORPUS
        source = "hardcoded-fallback"

    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = encodings["input_ids"]
    n_tokens = input_ids.shape[1]
    print(f"  Corpus source: {source}")
    print(f"  Tokens: {n_tokens}")
    return input_ids, n_tokens, source


def compute_perplexity(model, input_ids):
    """Compute perplexity via sliding window (stride=512)."""
    import torch
    import math

    seq_len = input_ids.shape[1]
    stride = 512
    max_length = 1024  # Mamba doesn't have a strict context limit but keep it bounded
    nlls = []
    n_tokens_scored = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_begin = max(begin, stride) if begin > 0 else 0
        input_chunk = input_ids[:, begin:end]
        target_len = end - max(begin, stride) if begin > 0 else end - begin

        with torch.no_grad():
            outputs = model(input_chunk)
            logits = outputs.logits

        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_chunk[:, 1:].contiguous()

        # Only score the non-overlapping portion
        if begin > 0:
            offset = max(begin, stride) - begin
            shift_logits = shift_logits[:, offset:, :]
            shift_labels = shift_labels[:, offset:]

        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        nll = loss_fn(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(nll.item())
        n_tokens_scored += shift_labels.numel()

        if end >= seq_len:
            break

    avg_nll = sum(nlls) / max(1, n_tokens_scored)
    ppl = math.exp(avg_nll)
    return ppl, avg_nll, n_tokens_scored


def run_bench():
    import torch
    import torch.nn as nn

    t_total_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    print("=" * 65)
    print("WO-MAMBA-HELIX-01 — Mamba-130m HelixLinear Benchmark (CPU)")
    print("=" * 65)

    errors = []
    receipt_data = {}

    # ── Phase 1: Load Model Shell ──
    print("\n[Phase 1] Loading Mamba-130m model shell...")
    t_load = time.time()

    from transformers import AutoTokenizer, MambaForCausalLM

    model_path = str(MODEL_LOCAL) if MODEL_LOCAL.exists() else MODEL_HF_ID
    print(f"  Model source: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MambaForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()

    load_time = time.time() - t_load
    print(f"  Load time: {load_time:.2f}s")

    # Count original nn.Linear modules
    orig_linears = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    print(f"  Original nn.Linear count: {orig_linears}")

    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    receipt_data["phase1_load"] = {
        "model_source": model_path,
        "load_time_s": round(load_time, 3),
        "original_linear_count": orig_linears,
        "total_params": n_params,
    }

    # ── Phase 2: Swap to HelixLinear ──
    print("\n[Phase 2] Loading CDNA v3 factors and swapping to HelixLinear...")
    t_swap = time.time()

    from helix_substrate.helix_linear import (
        HelixLinear, load_cdna_factors, swap_to_helix, swap_summary,
    )

    if not CDNA_DIR.exists():
        raise FileNotFoundError(f"CDNA directory not found: {CDNA_DIR}")

    helix_modules = load_cdna_factors(CDNA_DIR, model)
    n_factors_loaded = len(helix_modules)
    print(f"  CDNA factors loaded: {n_factors_loaded}")
    print(f"  Factor names (first 5): {list(helix_modules.keys())[:5]}")

    model = swap_to_helix(model, helix_modules)
    swap_time = time.time() - t_swap
    print(f"  Swap time: {swap_time:.2f}s")

    # Count post-swap modules
    summary = swap_summary(model)
    n_helix = summary["helix_modules"]
    n_linear_remaining = summary["linear_modules"]
    compression_ratio = summary["overall_ratio"]
    compressed_mb = summary["compressed_bytes"] / (1024 * 1024)
    dense_mb = summary["dense_equivalent_bytes"] / (1024 * 1024)

    print(f"  HelixLinear modules: {n_helix}")
    print(f"  Remaining nn.Linear: {n_linear_remaining}")
    print(f"  Compressed: {compressed_mb:.1f} MB vs dense: {dense_mb:.1f} MB ({compression_ratio}x)")

    receipt_data["phase2_swap"] = {
        "cdna_dir": str(CDNA_DIR),
        "factors_loaded": n_factors_loaded,
        "helix_modules": n_helix,
        "linear_remaining": n_linear_remaining,
        "compressed_mb": round(compressed_mb, 2),
        "dense_equivalent_mb": round(dense_mb, 2),
        "compression_ratio": compression_ratio,
        "swap_time_s": round(swap_time, 3),
    }

    # ── Phase 3: Sanity Checks ──
    print("\n[Phase 3] Sanity checks (NaN, finite logits, forward pass)...")
    t_check = time.time()

    test_text = "The quick brown fox jumps over the lazy dog."
    test_ids = tokenizer(test_text, return_tensors="pt")["input_ids"]
    print(f"  Test input: {test_ids.shape[1]} tokens")

    with torch.no_grad():
        outputs = model(test_ids)
        logits = outputs.logits

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    all_finite = torch.isfinite(logits).all().item()
    logits_shape = list(logits.shape)
    logits_min = logits.min().item()
    logits_max = logits.max().item()
    logits_mean = logits.mean().item()

    check_time = time.time() - t_check

    print(f"  Logits shape: {logits_shape}")
    print(f"  NaN present: {has_nan}")
    print(f"  Inf present: {has_inf}")
    print(f"  All finite: {all_finite}")
    print(f"  Logits range: [{logits_min:.4f}, {logits_max:.4f}], mean={logits_mean:.4f}")

    sanity_pass = all_finite and not has_nan and not has_inf
    verdict = "PASS" if sanity_pass else "FAIL"
    print(f"  Sanity verdict: {verdict}")
    if not sanity_pass:
        errors.append("Sanity check FAILED: logits contain NaN or Inf")

    receipt_data["phase3_sanity"] = {
        "test_tokens": test_ids.shape[1],
        "logits_shape": logits_shape,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "all_finite": all_finite,
        "logits_min": round(logits_min, 4),
        "logits_max": round(logits_max, 4),
        "logits_mean": round(logits_mean, 4),
        "verdict": verdict,
        "check_time_s": round(check_time, 3),
    }

    # ── Phase 4: Perplexity ──
    print("\n[Phase 4] Perplexity evaluation...")
    t_ppl = time.time()

    input_ids, n_corpus_tokens, corpus_source = load_perplexity_corpus(tokenizer, max_tokens=2048)

    ppl, avg_nll, n_scored = compute_perplexity(model, input_ids)
    ppl_time = time.time() - t_ppl

    print(f"  Perplexity: {ppl:.4f}")
    print(f"  Avg NLL: {avg_nll:.6f}")
    print(f"  Tokens scored: {n_scored}")
    print(f"  Eval time: {ppl_time:.2f}s")

    receipt_data["phase4_perplexity"] = {
        "corpus_source": corpus_source,
        "corpus_tokens": n_corpus_tokens,
        "tokens_scored": n_scored,
        "perplexity": round(ppl, 4),
        "avg_nll": round(avg_nll, 6),
        "eval_time_s": round(ppl_time, 3),
        "proof_layer": 3 if corpus_source == "wikitext-2-test" else 2,
        "proof_note": (
            "Layer 3: real data (WikiText-2 test)" if corpus_source == "wikitext-2-test"
            else "Layer 2: hardcoded fallback corpus (not real benchmark data)"
        ),
    }

    # ── Phase 5: Generation and tok/s ──
    print("\n[Phase 5] Generation benchmark (tok/s, memory)...")
    t_gen = time.time()

    prompt = "The future of artificial intelligence"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]
    gen_tokens = 64

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB

    t_gen_start = time.time()
    with torch.no_grad():
        gen_output = model.generate(
            prompt_ids,
            max_new_tokens=gen_tokens,
            do_sample=False,
            temperature=1.0,
        )
    gen_wall = time.time() - t_gen_start

    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    n_new_tokens = gen_output.shape[1] - prompt_len
    tok_per_sec = n_new_tokens / gen_wall if gen_wall > 0 else 0

    generated_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)
    gen_time = time.time() - t_gen

    # Check generation is finite / non-degenerate
    gen_has_nan = torch.isnan(gen_output.float()).any().item()
    unique_tokens = len(set(gen_output[0].tolist()[prompt_len:]))

    print(f"  Prompt: '{prompt}' ({prompt_len} tokens)")
    print(f"  Generated: {n_new_tokens} new tokens in {gen_wall:.2f}s")
    print(f"  Throughput: {tok_per_sec:.2f} tok/s")
    print(f"  Unique tokens in output: {unique_tokens}/{n_new_tokens}")
    print(f"  Peak RSS: {mem_after:.1f} MB")
    print(f"  Output preview: {generated_text[:200]}...")

    receipt_data["phase5_generation"] = {
        "prompt": prompt,
        "prompt_tokens": prompt_len,
        "new_tokens": n_new_tokens,
        "gen_wall_s": round(gen_wall, 3),
        "tok_per_sec": round(tok_per_sec, 2),
        "unique_output_tokens": unique_tokens,
        "gen_has_nan": gen_has_nan,
        "peak_rss_mb": round(mem_after, 1),
        "output_preview": generated_text[:300],
        "gen_time_s": round(gen_time, 3),
    }

    # ── Phase 6: Per-module compression summary ──
    print("\n[Phase 6] Per-module compression stats...")
    module_stats = []
    for name, mod in model.named_modules():
        if isinstance(mod, HelixLinear):
            savings = mod.memory_savings()
            module_stats.append({
                "name": name,
                "in": mod.in_features,
                "out": mod.out_features,
                "ratio": savings["ratio"],
                "svd_rank": mod.rank,
                "has_sidecar": mod.sidecar_positions is not None,
            })

    # Show a few representative
    if module_stats:
        for ms in module_stats[:4]:
            print(f"  {ms['name']}: ({ms['out']}, {ms['in']}) "
                  f"ratio={ms['ratio']}x svd_rank={ms['svd_rank']} "
                  f"sidecar={ms['has_sidecar']}")
        if len(module_stats) > 4:
            print(f"  ... and {len(module_stats) - 4} more HelixLinear modules")

    ratios = [m["ratio"] for m in module_stats]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    print(f"  Average per-module compression: {avg_ratio:.2f}x")

    receipt_data["phase6_module_stats"] = {
        "n_helix_modules": len(module_stats),
        "avg_compression_ratio": round(avg_ratio, 2),
        "min_ratio": round(min(ratios), 2) if ratios else 0,
        "max_ratio": round(max(ratios), 2) if ratios else 0,
        "modules": module_stats,
    }

    # ── Emit Receipt ──
    total_wall = time.time() - t_total_start
    total_cpu = time.process_time() - cpu_start
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB

    print(f"\n{'=' * 65}")
    print("RECEIPT SUMMARY")
    print("=" * 65)
    print(f"  HelixLinear modules: {n_helix}")
    print(f"  Remaining nn.Linear: {n_linear_remaining}")
    print(f"  Compression: {compression_ratio}x ({compressed_mb:.1f} MB / {dense_mb:.1f} MB)")
    print(f"  Sanity: {verdict}")
    print(f"  Perplexity: {ppl:.4f} ({corpus_source})")
    print(f"  Generation: {tok_per_sec:.2f} tok/s ({n_new_tokens} tokens)")
    print(f"  Peak RSS: {peak_mem:.1f} MB")
    print(f"  Errors: {len(errors)}")
    if errors:
        for e in errors:
            print(f"    ERROR: {e}")

    receipt_dir = PROJECT / "receipts" / "mamba_compress"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"mamba_helix_bench_{ts}.json"

    receipt = {
        "work_order": "WO-MAMBA-HELIX-01",
        "timestamp": ts,
        "model": "state-spaces/mamba-130m-hf",
        "architecture": "MambaForCausalLM",
        "model_config": {
            "n_layers": 24,
            "d_model": 768,
            "d_inner": 1536,
            "state_size": 16,
            "vocab_size": 50280,
        },
        "summary": {
            "helix_modules": n_helix,
            "linear_remaining": n_linear_remaining,
            "compression_ratio": compression_ratio,
            "compressed_mb": round(compressed_mb, 2),
            "dense_equivalent_mb": round(dense_mb, 2),
            "sanity_pass": sanity_pass,
            "perplexity": round(ppl, 4),
            "corpus_source": corpus_source,
            "tok_per_sec": round(tok_per_sec, 2),
            "peak_rss_mb": round(peak_mem, 1),
            "errors": errors,
        },
        "phases": receipt_data,
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

    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, default=jsonable)

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Total wall time: {total_wall:.1f}s")
    print(f"  Total CPU time: {total_cpu:.1f}s")
    print("=" * 65)

    return receipt


if __name__ == "__main__":
    run_bench()
