#!/usr/bin/env python3
"""
Break the born-compressed model — adversarial eval suite.

Probes:
1. In-domain generation: feed CVE prefix, see if output is plausible security text
2. Out-of-distribution: feed non-security text, measure PPL degradation
3. Memorization check: feed exact training sequences, compare PPL vs novel
4. Codebook collapse: analyze per-layer codebook usage, dead entries, entropy
5. Gradient flow: check if STE gradients actually reach all layers
6. Temperature sweep: generate at various temperatures, check coherence

Usage:
    python3 tools/break_compressed_model.py --checkpoint receipts/compressed_native_training/compressed_2000steps_model.pt
    python3 tools/break_compressed_model.py  # auto-finds latest checkpoint
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)


def find_latest_checkpoint():
    """Find most recent compressed model checkpoint."""
    ckpt_dir = PROJECT / "receipts" / "compressed_native_training"
    candidates = sorted(ckpt_dir.glob("compressed_*_model.pt"))
    if candidates:
        return candidates[-1]
    # Try any .pt file
    candidates = sorted(ckpt_dir.glob("*.pt"))
    return candidates[-1] if candidates else None


def load_model(ckpt_path, device="cuda"):
    """Rebuild the hybrid model and load trained weights."""
    # Import from training script
    sys.path.insert(0, str(PROJECT / "tools"))
    from train_compressed_hybrid import build_hybrid_model

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model, config_info = build_hybrid_model(
        compressed=ckpt["config"]["compressed"],
        device=device,
        vocab_size=tokenizer.vocab_size,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Model loaded: {config_info['pattern']}, {config_info['total_params']:,} params")
    print(f"  Mode: {'compressed' if ckpt['config']['compressed'] else 'dense'}")
    print(f"  Trained for {ckpt['step']} steps")

    return model, tokenizer, config_info


@torch.no_grad()
def measure_ppl(model, tokenizer, text, device="cuda", max_len=128):
    """Measure perplexity on a text string."""
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    if tokens.shape[1] < 2:
        return float('inf'), 0
    outputs = model(tokens, labels=tokens)
    loss = outputs.loss.item()
    n_tokens = tokens.shape[1] - 1
    return math.exp(loss), n_tokens


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new=64, temperature=1.0, device="cuda"):
    """Generate text from a prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    for _ in range(max_new):
        # Only feed last 128 tokens to stay within trained seq_len
        context = generated[:, -128:]
        outputs = model(context)
        logits = outputs.logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def test_in_domain(model, tokenizer, device):
    """Probe 1: In-domain security text generation."""
    print("\n" + "=" * 70)
    print("PROBE 1: IN-DOMAIN GENERATION")
    print("=" * 70)

    prompts = [
        "CVE-2024-",
        "A buffer overflow vulnerability in",
        "The use-after-free bug allows",
        "An attacker can exploit this",
        "Linux kernel commit fixing",
    ]

    results = []
    for prompt in prompts:
        print(f"\n  Prompt: '{prompt}'")
        for temp in [0.0, 0.7, 1.0]:
            text = generate_text(model, tokenizer, prompt, max_new=50, temperature=temp, device=device)
            generated_part = text[len(prompt):]
            print(f"    T={temp}: {generated_part[:120]}")
            results.append({
                "prompt": prompt,
                "temperature": temp,
                "generated": generated_part[:200],
            })

    return {"probe": "in_domain_generation", "results": results}


def test_out_of_distribution(model, tokenizer, device):
    """Probe 2: Out-of-distribution text — measure PPL on non-security text."""
    print("\n" + "=" * 70)
    print("PROBE 2: OUT-OF-DISTRIBUTION PPL")
    print("=" * 70)

    in_domain_texts = [
        "CVE-2024-1234: A heap buffer overflow in the Linux kernel's netfilter subsystem allows local privilege escalation.",
        "Use-after-free vulnerability in the io_uring implementation allows an attacker to execute arbitrary code in kernel context.",
        "An out-of-bounds write in the ext4 filesystem driver could lead to memory corruption and denial of service.",
    ]

    ood_texts = [
        "The quick brown fox jumps over the lazy dog. It was a beautiful sunny day in the park.",
        "To make chocolate cake, preheat the oven to 350 degrees. Mix flour, sugar, and cocoa powder.",
        "The stock market rallied today on news of better-than-expected earnings from major tech companies.",
        "In quantum mechanics, the Schrödinger equation describes how the quantum state of a physical system changes.",
        "SELECT * FROM users WHERE id = 1; DROP TABLE users; -- this is a SQL injection example",
    ]

    results = {"in_domain": [], "out_of_domain": []}

    print("\n  In-domain PPL:")
    for text in in_domain_texts:
        ppl, n = measure_ppl(model, tokenizer, text, device)
        print(f"    PPL={ppl:,.1f} ({n} tok): {text[:80]}...")
        results["in_domain"].append({"text": text[:80], "ppl": round(ppl, 2), "n_tokens": n})

    print("\n  Out-of-domain PPL:")
    for text in ood_texts:
        ppl, n = measure_ppl(model, tokenizer, text, device)
        print(f"    PPL={ppl:,.1f} ({n} tok): {text[:80]}...")
        results["out_of_domain"].append({"text": text[:80], "ppl": round(ppl, 2), "n_tokens": n})

    in_avg = sum(r["ppl"] for r in results["in_domain"]) / len(results["in_domain"])
    ood_avg = sum(r["ppl"] for r in results["out_of_domain"]) / len(results["out_of_domain"])
    ratio = ood_avg / in_avg if in_avg > 0 else float('inf')

    print(f"\n  In-domain avg PPL:  {in_avg:,.1f}")
    print(f"  OOD avg PPL:        {ood_avg:,.1f}")
    print(f"  OOD/In-domain ratio: {ratio:.2f}x")
    print(f"  Verdict: {'SPECIALIZED' if ratio > 2.0 else 'NOT SPECIALIZED'} (ratio {'>' if ratio > 2.0 else '<='} 2x)")

    results["in_domain_avg_ppl"] = round(in_avg, 2)
    results["ood_avg_ppl"] = round(ood_avg, 2)
    results["ood_ratio"] = round(ratio, 2)
    results["specialized"] = ratio > 2.0

    return {"probe": "out_of_distribution", "results": results}


def test_memorization(model, tokenizer, device):
    """Probe 3: Memorization check — does the model reproduce training data verbatim?"""
    print("\n" + "=" * 70)
    print("PROBE 3: MEMORIZATION CHECK")
    print("=" * 70)

    # Load a few real training examples
    corpus_path = Path(os.environ.get(
        "SECURITY_CORPUS",
        "/home/voidstr3m33/datasets/security_corpus/security_corpus.jsonl"
    ))

    train_texts = []
    if corpus_path.exists():
        with open(corpus_path) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                doc = json.loads(line)
                train_texts.append(doc["text"])

    if not train_texts:
        print("  No training corpus found, skipping memorization check")
        return {"probe": "memorization", "results": {"skipped": True}}

    results = []
    for text in train_texts[:5]:
        # Feed first half, check if model generates second half
        words = text.split()
        if len(words) < 10:
            continue
        prefix = " ".join(words[:len(words)//2])
        expected_suffix = " ".join(words[len(words)//2:len(words)//2 + 20])

        generated = generate_text(model, tokenizer, prefix, max_new=30, temperature=0.0, device=device)
        gen_suffix = generated[len(prefix):].strip()

        # Check overlap
        expected_words = set(expected_suffix.lower().split())
        gen_words = set(gen_suffix.lower().split())
        overlap = len(expected_words & gen_words) / max(len(expected_words), 1)

        print(f"\n  Prefix: {prefix[:60]}...")
        print(f"  Expected: {expected_suffix[:80]}...")
        print(f"  Got:      {gen_suffix[:80]}...")
        print(f"  Word overlap: {overlap:.1%}")

        results.append({
            "prefix": prefix[:60],
            "expected": expected_suffix[:80],
            "generated": gen_suffix[:80],
            "word_overlap": round(overlap, 3),
        })

    avg_overlap = sum(r["word_overlap"] for r in results) / max(len(results), 1)
    print(f"\n  Avg word overlap: {avg_overlap:.1%}")
    print(f"  Verdict: {'MEMORIZING' if avg_overlap > 0.5 else 'NOT MEMORIZING'}")

    return {"probe": "memorization", "results": results, "avg_overlap": round(avg_overlap, 3)}


def test_codebook_health(model):
    """Probe 4: Codebook health — utilization, entropy, dead entries per layer."""
    print("\n" + "=" * 70)
    print("PROBE 4: CODEBOOK HEALTH")
    print("=" * 70)

    from helix_substrate.helix_linear_ste import HelixLinearSTE

    results = []
    total_dead = 0
    total_entries = 0

    for name, module in model.named_modules():
        if not isinstance(module, HelixLinearSTE):
            continue

        indices = module.indices.long().reshape(-1)
        k = module.codebook.shape[0]
        total_entries += k

        # Usage counts
        counts = torch.bincount(indices, minlength=k).float()
        used = (counts > 0).sum().item()
        dead = k - used
        total_dead += dead

        # Assignment entropy
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log2()).sum().item()
        max_entropy = math.log2(k)
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0

        # Codebook value spread
        cb = module.codebook.data
        cb_std = cb.std().item()
        cb_range = (cb.max() - cb.min()).item()

        results.append({
            "name": name,
            "k": k,
            "used": used,
            "dead": dead,
            "entropy": round(entropy, 2),
            "max_entropy": round(max_entropy, 2),
            "entropy_ratio": round(entropy_ratio, 3),
            "cb_std": round(cb_std, 6),
            "cb_range": round(cb_range, 6),
        })

        status = "OK" if dead == 0 else f"DEAD={dead}"
        print(f"  {name:50s} used={used}/{k} entropy={entropy:.2f}/{max_entropy:.2f} ({entropy_ratio:.1%}) {status}")

    print(f"\n  Total dead entries: {total_dead}/{total_entries} ({total_dead/max(total_entries,1):.1%})")
    avg_entropy = sum(r["entropy_ratio"] for r in results) / max(len(results), 1)
    print(f"  Avg entropy ratio: {avg_entropy:.1%}")
    print(f"  Verdict: {'HEALTHY' if total_dead < total_entries * 0.1 and avg_entropy > 0.8 else 'DEGRADED'}")

    return {
        "probe": "codebook_health",
        "results": results,
        "total_dead": total_dead,
        "total_entries": total_entries,
        "avg_entropy_ratio": round(avg_entropy, 3),
    }


def test_layer_gradients(model, tokenizer, device):
    """Probe 5: Gradient flow — do STE gradients actually reach all layers?"""
    print("\n" + "=" * 70)
    print("PROBE 5: GRADIENT FLOW")
    print("=" * 70)

    from helix_substrate.helix_linear_ste import HelixLinearSTE

    # One forward-backward pass
    text = "CVE-2024-9999: A critical buffer overflow in the network stack allows remote code execution."
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64).to(device)

    model.train()
    outputs = model(tokens, labels=tokens)
    outputs.loss.backward()

    results = []
    for name, module in model.named_modules():
        if not isinstance(module, HelixLinearSTE):
            continue

        cb_grad = module.codebook.grad
        if cb_grad is None:
            print(f"  {name:50s} NO GRADIENT!")
            results.append({"name": name, "has_grad": False, "grad_norm": 0, "grad_max": 0})
        else:
            norm = cb_grad.norm().item()
            maxval = cb_grad.abs().max().item()
            nonzero = (cb_grad.abs() > 1e-10).sum().item()
            total = cb_grad.numel()
            print(f"  {name:50s} norm={norm:.6f} max={maxval:.6f} nonzero={nonzero}/{total}")
            results.append({
                "name": name,
                "has_grad": True,
                "grad_norm": round(norm, 6),
                "grad_max": round(maxval, 6),
                "nonzero_frac": round(nonzero / total, 3),
            })

    model.zero_grad()
    model.eval()

    all_have_grad = all(r["has_grad"] for r in results)
    print(f"\n  All layers receive gradients: {all_have_grad}")
    print(f"  Verdict: {'FLOWING' if all_have_grad else 'BLOCKED'}")

    return {"probe": "gradient_flow", "results": results, "all_flowing": all_have_grad}


def test_repetition(model, tokenizer, device):
    """Probe 6: Repetition / degeneration check."""
    print("\n" + "=" * 70)
    print("PROBE 6: REPETITION / DEGENERATION")
    print("=" * 70)

    prompts = [
        "The vulnerability allows an attacker to",
        "CVE-2025-0001:",
        "In the Linux kernel,",
    ]

    results = []
    for prompt in prompts:
        text = generate_text(model, tokenizer, prompt, max_new=100, temperature=0.7, device=device)
        generated = text[len(prompt):]
        words = generated.split()

        # Check for repeated n-grams
        if len(words) >= 3:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            most_common = trigram_counts.most_common(1)[0] if trigram_counts else ("", 0)
            repeat_ratio = most_common[1] / max(len(trigrams), 1) if most_common[1] > 1 else 0
        else:
            repeat_ratio = 0
            most_common = ("", 0)

        # Unique token ratio
        unique_ratio = len(set(words)) / max(len(words), 1)

        print(f"\n  Prompt: '{prompt}'")
        print(f"  Generated ({len(words)} words): {generated[:150]}...")
        print(f"  Unique word ratio: {unique_ratio:.2f}")
        print(f"  Most repeated trigram: '{most_common[0]}' x{most_common[1]}")
        print(f"  Repeat ratio: {repeat_ratio:.2f}")

        degenerate = repeat_ratio > 0.3 or unique_ratio < 0.3
        print(f"  Verdict: {'DEGENERATE' if degenerate else 'OK'}")

        results.append({
            "prompt": prompt,
            "generated_len": len(words),
            "unique_ratio": round(unique_ratio, 3),
            "repeat_ratio": round(repeat_ratio, 3),
            "most_repeated": most_common[0],
            "degenerate": degenerate,
        })

    any_degenerate = any(r["degenerate"] for r in results)
    print(f"\n  Overall: {'DEGENERATE' if any_degenerate else 'STABLE'}")

    return {"probe": "repetition", "results": results, "any_degenerate": any_degenerate}


def main():
    parser = argparse.ArgumentParser(description="Adversarial eval of born-compressed model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    print("=" * 70)
    print("BREAK THE BORN-COMPRESSED MODEL — ADVERSARIAL EVAL")
    print("=" * 70)

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_latest_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        print(f"ERROR: No checkpoint found. Run training with checkpoint saving first.")
        sys.exit(1)

    t_start = time.time()

    # Load
    print("\n[0] Loading model...")
    model, tokenizer, config_info = load_model(ckpt_path, args.device)

    # Run all probes
    all_results = {"checkpoint": str(ckpt_path), "config": config_info}

    print("\n[1] In-domain generation...")
    all_results["probe_1"] = test_in_domain(model, tokenizer, args.device)

    print("\n[2] Out-of-distribution PPL...")
    all_results["probe_2"] = test_out_of_distribution(model, tokenizer, args.device)

    print("\n[3] Memorization check...")
    all_results["probe_3"] = test_memorization(model, tokenizer, args.device)

    print("\n[4] Codebook health...")
    all_results["probe_4"] = test_codebook_health(model)

    print("\n[5] Gradient flow...")
    all_results["probe_5"] = test_layer_gradients(model, tokenizer, args.device)

    print("\n[6] Repetition check...")
    all_results["probe_6"] = test_repetition(model, tokenizer, args.device)

    # Summary
    wall = time.time() - t_start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    verdicts = {
        "in_domain": "see generation samples",
        "ood_specialized": all_results["probe_2"]["results"].get("specialized", "unknown"),
        "memorizing": all_results["probe_3"].get("avg_overlap", 0) > 0.5 if "avg_overlap" in all_results["probe_3"] else "skipped",
        "codebook_healthy": all_results["probe_4"]["total_dead"] < all_results["probe_4"]["total_entries"] * 0.1,
        "gradients_flowing": all_results["probe_5"]["all_flowing"],
        "degenerate": all_results["probe_6"]["any_degenerate"],
    }
    for k, v in verdicts.items():
        print(f"  {k:25s}: {v}")

    all_results["verdicts"] = verdicts
    all_results["wall_time_s"] = round(wall, 1)

    # Save receipt
    receipt_dir = PROJECT / "receipts" / "compressed_native_training"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"adversarial_eval_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Wall time: {wall:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
