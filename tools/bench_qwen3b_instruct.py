#!/usr/bin/env python3
"""
Verification benchmark for Qwen2.5-3B-Instruct through HelixLinear.

Architecture: Qwen2ForCausalLM, 36 layers, 2048 hidden, 16 heads, 2 KV heads, vocab 151936
Compressed to CDNA v3: 252 tensors (36 blocks x 7 projections)
Target device: Quadro T2000 (4 GB VRAM), expected ~1200 MB VRAM.

Phases:
  A — Verify: count HelixLinear modules, forward pass finite, 0 NaN
  B — Perplexity on WikiText-2 test (CPU-offloaded loss for 151K vocab OOM safety)
  C — Inference speed: decode tok/s and prefill tok/s
  D — Generation quality: 3 FGIP-style analytical prompts
  E — VRAM measurement

Emits receipt to receipts/qwen3b_instruct_compress/qwen3b_instruct_bench_{ts}.json
with WO-RECEIPT-COST-01 cost block.

Usage:
    python3 tools/bench_qwen3b_instruct.py
"""

from __future__ import annotations

import gc
import json
import os
import platform
import resource
import sys
import time
import warnings
from pathlib import Path

# ── Project path ──
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

# ── Capture warnings ──
captured_warnings: list[dict] = []
_orig_showwarning = warnings.showwarning


def _capture_warning(message, category, filename, lineno, file=None, line=None):
    captured_warnings.append({
        "message": str(message),
        "category": category.__name__,
        "filename": str(filename),
        "lineno": lineno,
    })
    _orig_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _capture_warning

# ── Constants ──
MODEL_DIR = Path.home() / "models" / "qwen2.5-3b-instruct"
CDNA_DIR = MODEL_DIR / "cdnav3"
N_BLOCKS = 36
N_TENSOR_TYPES = 7  # q, k, v, o, gate, up, down
EXPECTED_HELIX_MODULES = N_BLOCKS * N_TENSOR_TYPES  # 252
VOCAB_SIZE = 151936

# FGIP-style analytical prompts
QUALITY_PROMPTS = [
    {
        "name": "cooling_systems",
        "prompt": "What are the main components of a data center cooling system?",
    },
    {
        "name": "tier_classification",
        "prompt": "Explain the difference between Tier III and Tier IV data center classifications.",
    },
    {
        "name": "location_factors",
        "prompt": "What factors determine the optimal location for a new data center facility?",
    },
]


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


def run_bench(skip_ppl: bool = False, cached_ppl: float | None = None):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from helix_substrate.helix_linear import (
        HelixLinear,
        load_cdna_factors,
        swap_to_helix,
        swap_summary,
    )
    from helix_substrate.zerocopy import pin_indices_from_file

    t_total_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    receipt = {
        "benchmark": "qwen3b_instruct_helix_linear",
        "model": "Qwen2.5-3B-Instruct",
        "architecture": "Qwen2ForCausalLM",
        "n_layers": N_BLOCKS,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_kv_heads": 2,
        "vocab_size": VOCAB_SIZE,
        "phases": {},
        "warnings": [],
    }

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA device — running on CPU (slow).")
    print("=" * 70)
    print("Qwen2.5-3B-Instruct — HelixLinear Verification Benchmark")
    print(f"Device: {device}")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────────
    # Load model and swap to HelixLinear
    # ──────────────────────────────────────────────────────────────────────
    print("\n[Load] Loading base model from", MODEL_DIR)
    t_load_start = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"  Base model loaded in {time.time() - t_load_start:.1f}s")

    # Load CDNA factors (pass model for bias extraction)
    print("[Load] Loading CDNA v3 factors from", CDNA_DIR)
    t_cdna_start = time.time()
    helix_modules = load_cdna_factors(CDNA_DIR, model=model)
    print(f"  {len(helix_modules)} CDNA factors loaded in {time.time() - t_cdna_start:.1f}s")

    # Swap to HelixLinear
    print("[Load] Swapping nn.Linear → HelixLinear...")
    model = swap_to_helix(model, helix_modules)
    summary = swap_summary(model)
    print(f"  Swap complete: {summary['helix_modules']} HelixLinear, "
          f"{summary['linear_modules']} nn.Linear remaining, "
          f"ratio={summary['overall_ratio']}x")

    # Pin-before-move: pin HelixLinear indices to host memory so they appear
    # as CUDA tensors but live in host RAM (zero VRAM for indices).
    pinned_buffers = []
    if device.startswith("cuda"):
        n_pinned = 0
        for name, module in model.named_modules():
            if not isinstance(module, HelixLinear):
                continue
            tensor_name = name + ".weight"
            safe_name = tensor_name.replace("/", "_").replace(".", "_")
            tensor_dir = CDNA_DIR / f"{safe_name}.cdnav3"
            indices_path = tensor_dir / "indices.bin"
            if not indices_path.exists():
                continue
            shape = tuple(module.indices.shape)
            zc_indices, pinned = pin_indices_from_file(indices_path, shape)
            pinned_buffers.append(pinned)
            module._buffers["indices"] = zc_indices
            n_pinned += 1
        print(f"  Pinned {n_pinned} index tensors to host RAM (zero-copy)")

    # Move to GPU — indices are already "CUDA" (pinned host), rest moves normally
    print(f"[Load] Moving to {device}...")
    t_gpu_start = time.time()
    model = model.to(device)
    model.eval()
    gpu_move_time = time.time() - t_gpu_start
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(0)
    print(f"  GPU move: {gpu_move_time:.1f}s")

    # Free CPU copies
    del helix_modules
    gc.collect()

    total_load_time = time.time() - t_load_start
    receipt["load_time_s"] = round(total_load_time, 1)
    receipt["swap_summary"] = summary

    # ══════════════════════════════════════════════════════════════════════
    # Phase A: Verify
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("[Phase A] Verification — module count, forward pass, NaN check")
    print("=" * 70)

    # Count HelixLinear modules
    helix_count = sum(1 for m in model.modules() if isinstance(m, HelixLinear))
    linear_count = sum(1 for m in model.modules()
                       if isinstance(m, torch.nn.Linear))
    print(f"  HelixLinear modules: {helix_count} (expect {EXPECTED_HELIX_MODULES})")
    print(f"  nn.Linear remaining: {linear_count}")

    helix_ok = helix_count == EXPECTED_HELIX_MODULES
    if not helix_ok:
        print(f"  WARNING: Expected {EXPECTED_HELIX_MODULES}, got {helix_count}")

    # Forward pass finite check
    print("  Running forward pass (short probe)...")
    probe_ids = tokenizer.encode("Hello, world!", return_tensors="pt").to(device)
    with torch.no_grad():
        probe_out = model(probe_ids)
    logits = probe_out.logits

    is_finite = torch.isfinite(logits).all().item()
    has_nan = torch.isnan(logits).any().item()
    logits_shape = list(logits.shape)
    logits_min = logits.min().item()
    logits_max = logits.max().item()
    logits_mean = logits.float().mean().item()

    print(f"  Logits shape: {logits_shape}")
    print(f"  Finite: {is_finite}, NaN: {has_nan}")
    print(f"  Range: [{logits_min:.4f}, {logits_max:.4f}], mean={logits_mean:.4f}")

    phase_a = {
        "helix_modules": helix_count,
        "expected_helix_modules": EXPECTED_HELIX_MODULES,
        "helix_count_ok": helix_ok,
        "linear_remaining": linear_count,
        "forward_pass_finite": is_finite,
        "forward_pass_nan": has_nan,
        "logits_shape": logits_shape,
        "logits_min": logits_min,
        "logits_max": logits_max,
        "logits_mean": logits_mean,
        "pass": helix_ok and is_finite and not has_nan,
    }
    receipt["phases"]["A_verify"] = phase_a
    status_a = "PASS" if phase_a["pass"] else "FAIL"
    print(f"  Phase A: {status_a}")

    del probe_out, logits, probe_ids
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # Phase B: Perplexity on WikiText-2 test
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("[Phase B] Perplexity — WikiText-2 test (CPU-offloaded loss)")
    print("=" * 70)

    if skip_ppl and cached_ppl is not None:
        print(f"  SKIPPED (using cached PPL={cached_ppl:.4f})")
        phase_b = {
            "dataset": "wikitext-2-raw-v1",
            "split": "test",
            "perplexity": cached_ppl,
            "cached": True,
            "pass": True,
        }
        receipt["phases"]["B_perplexity"] = phase_b
        status_b = "PASS (cached)"
        print(f"  Phase B: {status_b}")
    elif skip_ppl:
        print("  SKIPPED (no cached value)")
        phase_b = {"skipped": True, "pass": True}
        receipt["phases"]["B_perplexity"] = phase_b
        status_b = "SKIPPED"
        print(f"  Phase B: {status_b}")
    else:
        t_ppl_start = time.time()
        try:
            from datasets import load_dataset

            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join([t for t in dataset["text"] if t.strip()])

            encodings = tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids  # [1, seq_len]
            seq_len = input_ids.shape[1]
            print(f"  WikiText-2 test: {seq_len} tokens")

            # Chunked eval — max_length governs stride to avoid OOM on 4GB card
            max_length = 512
            stride = 256
            nlls = []
            n_tokens_eval = 0

            n_chunks = max(1, (seq_len - max_length) // stride + 1)
            print(f"  Evaluating {n_chunks} chunks (max_length={max_length}, stride={stride})...")

            for i in range(0, seq_len - 1, stride):
                begin = max(0, i)
                end = min(begin + max_length, seq_len)
                chunk_ids = input_ids[:, begin:end].to(device)

                # Target: shift by 1
                target_ids = chunk_ids.clone()
                # Mask context (only score the stride window, not the overlap)
                if i > 0:
                    target_ids[:, :max_length - stride] = -100

                with torch.no_grad():
                    outputs = model(chunk_ids)
                    # CPU-offloaded loss to avoid 151K vocab OOM
                    logits_cpu = outputs.logits.float().cpu()
                    labels_cpu = target_ids.cpu()

                    shift_logits = logits_cpu[..., :-1, :].contiguous().view(-1, VOCAB_SIZE)
                    shift_labels = labels_cpu[..., 1:].contiguous().view(-1)

                    # Only compute loss on non-masked tokens
                    mask = shift_labels != -100
                    if mask.any():
                        loss = F.cross_entropy(
                            shift_logits[mask], shift_labels[mask], reduction="sum"
                        )
                        n_valid = mask.sum().item()
                        nlls.append(loss.item())
                        n_tokens_eval += n_valid

                del chunk_ids, target_ids, outputs, logits_cpu, labels_cpu
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

                if end >= seq_len:
                    break

            total_nll = sum(nlls)
            ppl = float("inf")
            if n_tokens_eval > 0:
                import math
                avg_nll = total_nll / n_tokens_eval
                ppl = math.exp(avg_nll)

            ppl_time = time.time() - t_ppl_start
            print(f"  Tokens evaluated: {n_tokens_eval}")
            print(f"  Perplexity: {ppl:.4f}")
            print(f"  Time: {ppl_time:.1f}s")

            phase_b = {
                "dataset": "wikitext-2-raw-v1",
                "split": "test",
                "total_tokens": seq_len,
                "tokens_evaluated": n_tokens_eval,
                "max_length": max_length,
                "stride": stride,
                "perplexity": round(ppl, 4),
                "avg_nll": round(avg_nll, 6) if n_tokens_eval > 0 else None,
                "cpu_offloaded_loss": True,
                "time_s": round(ppl_time, 1),
                "pass": ppl < 50.0 and ppl > 1.0,  # sanity bounds
            }
        except Exception as e:
            print(f"  ERROR in Phase B: {e}")
            phase_b = {"error": str(e), "pass": False}

        receipt["phases"]["B_perplexity"] = phase_b
        status_b = "PASS" if phase_b.get("pass") else "FAIL"
        print(f"  Phase B: {status_b}")

    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # Phase C: Inference speed — decode tok/s and prefill tok/s
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("[Phase C] Inference speed — decode tok/s, prefill tok/s")
    print("=" * 70)

    # Prefill measurement: forward pass on a prompt, measure time
    prefill_prompt = "The primary considerations for building a modern data center include"
    prefill_ids = tokenizer.encode(prefill_prompt, return_tensors="pt").to(device)
    prefill_len = prefill_ids.shape[1]

    # Warmup
    with torch.no_grad():
        _ = model(prefill_ids)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Prefill timing (3 runs, take median)
    prefill_times = []
    for _ in range(3):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(prefill_ids)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        prefill_times.append(time.perf_counter() - t0)

    prefill_median = sorted(prefill_times)[1]
    prefill_tps = prefill_len / prefill_median
    print(f"  Prefill: {prefill_len} tokens in {prefill_median*1000:.1f} ms "
          f"({prefill_tps:.1f} tok/s)")

    # Decode measurement: generate tokens, measure total time
    decode_prompt = "Explain briefly what a data center is:"
    decode_ids = tokenizer.encode(decode_prompt, return_tensors="pt").to(device)
    decode_prompt_len = decode_ids.shape[1]
    max_new = 64

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t_gen_start = time.perf_counter()
    with torch.no_grad():
        gen_out = model.generate(
            decode_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_gen_end = time.perf_counter()

    gen_tokens = gen_out.shape[1] - decode_prompt_len
    gen_time = t_gen_end - t_gen_start
    decode_tps = gen_tokens / gen_time if gen_time > 0 else 0
    print(f"  Decode: {gen_tokens} tokens in {gen_time*1000:.1f} ms "
          f"({decode_tps:.1f} tok/s)")

    phase_c = {
        "prefill_tokens": prefill_len,
        "prefill_median_ms": round(prefill_median * 1000, 1),
        "prefill_tok_s": round(prefill_tps, 1),
        "prefill_runs": len(prefill_times),
        "decode_prompt_tokens": decode_prompt_len,
        "decode_new_tokens": gen_tokens,
        "decode_time_ms": round(gen_time * 1000, 1),
        "decode_tok_s": round(decode_tps, 1),
        "max_new_tokens": max_new,
        "pass": decode_tps > 0.1,  # at least generating something
    }
    receipt["phases"]["C_speed"] = phase_c
    status_c = "PASS" if phase_c["pass"] else "FAIL"
    print(f"  Phase C: {status_c}")

    del prefill_ids, decode_ids, gen_out
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    # Phase D: Generation quality — 3 FGIP-style analytical prompts
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("[Phase D] Generation quality — FGIP-style analytical prompts")
    print("=" * 70)

    generation_results = []
    for entry in QUALITY_PROMPTS:
        name = entry["name"]
        prompt = entry["prompt"]

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        chat_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(chat_input, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        gen_time_prompt = time.perf_counter() - t0

        new_tokens = output_ids.shape[1] - prompt_len
        response_text = tokenizer.decode(
            output_ids[0, prompt_len:], skip_special_tokens=True
        )

        # Basic quality checks
        is_nonempty = len(response_text.strip()) > 10
        is_coherent = not response_text.strip().startswith(response_text.strip()[:20] * 3)
        has_no_nan = "nan" not in response_text.lower()[:50]

        result = {
            "name": name,
            "prompt": prompt,
            "response": response_text[:500],  # truncate for receipt
            "new_tokens": new_tokens,
            "gen_time_s": round(gen_time_prompt, 2),
            "tok_s": round(new_tokens / gen_time_prompt, 1) if gen_time_prompt > 0 else 0,
            "nonempty": is_nonempty,
            "coherent": is_coherent,
            "no_nan": has_no_nan,
        }
        generation_results.append(result)
        print(f"\n  [{name}] {new_tokens} tokens, {gen_time_prompt:.2f}s, "
              f"{result['tok_s']} tok/s")
        # Print first 200 chars of response
        preview = response_text[:200].replace("\n", " ")
        print(f"    Response: {preview}...")

        del input_ids, output_ids
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    all_quality_pass = all(
        r["nonempty"] and r["coherent"] and r["no_nan"]
        for r in generation_results
    )
    phase_d = {
        "prompts": generation_results,
        "all_nonempty": all(r["nonempty"] for r in generation_results),
        "all_coherent": all(r["coherent"] for r in generation_results),
        "all_no_nan": all(r["no_nan"] for r in generation_results),
        "pass": all_quality_pass,
    }
    receipt["phases"]["D_quality"] = phase_d
    status_d = "PASS" if phase_d["pass"] else "FAIL"
    print(f"\n  Phase D: {status_d}")

    # ══════════════════════════════════════════════════════════════════════
    # Phase E: VRAM measurement
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("[Phase E] VRAM measurement")
    print("=" * 70)

    if device.startswith("cuda"):
        vram_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        vram_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        headroom_mb = gpu_total_mb - vram_peak

        print(f"  GPU: {gpu_name} ({gpu_total_mb:.0f} MB total)")
        print(f"  Allocated: {vram_allocated:.0f} MB")
        print(f"  Reserved:  {vram_reserved:.0f} MB")
        print(f"  Peak:      {vram_peak:.0f} MB")
        print(f"  Headroom:  {headroom_mb:.0f} MB")

        phase_e = {
            "gpu_name": gpu_name,
            "gpu_total_mb": round(gpu_total_mb),
            "vram_allocated_mb": round(vram_allocated),
            "vram_reserved_mb": round(vram_reserved),
            "vram_peak_mb": round(vram_peak),
            "headroom_mb": round(headroom_mb),
            "pass": vram_peak < 3500,  # must fit in T2000 with headroom
        }
    else:
        print("  (CPU mode — no VRAM to measure)")
        phase_e = {"gpu_name": "cpu", "pass": True}

    receipt["phases"]["E_vram"] = phase_e
    status_e = "PASS" if phase_e["pass"] else "FAIL"
    print(f"  Phase E: {status_e}")

    # ══════════════════════════════════════════════════════════════════════
    # Summary and receipt
    # ══════════════════════════════════════════════════════════════════════
    all_pass = all(
        receipt["phases"][k].get("pass", False)
        for k in receipt["phases"]
    )

    # Cost block (WO-RECEIPT-COST-01)
    end_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    receipt["cost"] = {
        "wall_time_s": round(time.time() - t_total_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": end_iso,
    }

    receipt["warnings"] = captured_warnings
    receipt["all_pass"] = all_pass
    receipt["device"] = device

    # Emit receipt
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_dir = PROJECT / "receipts" / "qwen3b_instruct_compress"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"qwen3b_instruct_bench_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, default=jsonable)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Phase A (Verify):     {status_a}")
    print(f"  Phase B (Perplexity): {status_b}"
          + (f"  PPL={phase_b.get('perplexity', '?')}" if 'perplexity' in phase_b else ""))
    print(f"  Phase C (Speed):      {status_c}"
          + f"  decode={phase_c.get('decode_tok_s', '?')} tok/s"
          + f"  prefill={phase_c.get('prefill_tok_s', '?')} tok/s")
    print(f"  Phase D (Quality):    {status_d}")
    print(f"  Phase E (VRAM):       {status_e}"
          + (f"  peak={phase_e.get('vram_peak_mb', '?')} MB" if 'vram_peak_mb' in phase_e else ""))
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAIL'}")
    print(f"  Receipt: {receipt_path}")
    print(f"  Wall time: {receipt['cost']['wall_time_s']:.1f}s")

    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ppl", action="store_true", help="Skip PPL eval (use --cached-ppl for value)")
    parser.add_argument("--cached-ppl", type=float, default=None, help="Cached PPL value to use with --skip-ppl")
    args = parser.parse_args()
    run_bench(skip_ppl=args.skip_ppl, cached_ppl=args.cached_ppl)
