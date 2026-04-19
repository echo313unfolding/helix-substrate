"""
WO-BORN-HYBRID-01: Evaluation for born-compressed Qwen-Mamba hybrid.

Computes:
  1. Held-out code PPL (from prepared data)
  2. HumanEval pass@1 (via bigcode-evaluation-harness)
  3. MBPP pass@1

Usage:
    # PPL only (fast)
    python -m echo_hybrid.eval_born_hybrid --ckpt checkpoints/echo_hybrid_v2/born_d2/step_050000.pt --ppl-only

    # Full eval (HumanEval + MBPP)
    python -m echo_hybrid.eval_born_hybrid --ckpt checkpoints/echo_hybrid_v2/born_d2/step_050000.pt

    # Compare born vs dense
    python -m echo_hybrid.eval_born_hybrid --compare \\
        --born checkpoints/echo_hybrid_v2/born_d2/step_050000.pt \\
        --dense checkpoints/echo_hybrid_v2/dense/step_050000.pt
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from echo_hybrid.config_v2 import EchoHybridV2Config, EchoHybridV2Model
from echo_hybrid.train_sidecar_aware import cost_block

RECEIPT_DIR = Path("receipts/echo_hybrid")
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_from_checkpoint(ckpt_path: str, device: str = "cuda") -> EchoHybridV2Model:
    """Load model from training checkpoint."""
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = EchoHybridV2Config()
    model = EchoHybridV2Model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    step = ckpt.get("step", 0)
    print(f"  Loaded step {step}, {model.n_params():,} params")
    return model, step


# ---------------------------------------------------------------------------
# PPL evaluation on held-out data
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ppl(
    model: EchoHybridV2Model,
    data_path: str,
    seq_len: int = 2048,
    stride: int = 1024,
    max_seqs: int = 500,
    device: str = "cuda",
) -> Dict:
    """Compute perplexity on held-out data with sliding window."""
    model.eval()
    data = np.memmap(data_path, dtype=np.uint32, mode="r")
    n_tokens = len(data)

    total_nll = 0.0
    total_count = 0
    n_seqs = 0

    for start in range(0, n_tokens - seq_len, stride):
        if n_seqs >= max_seqs:
            break

        tokens = torch.from_numpy(data[start:start + seq_len].astype(np.int64)).unsqueeze(0).to(device)

        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=(device != "cpu")):
            out = model(input_ids=tokens, labels=tokens)

        # Only count tokens in the non-overlapping region
        if start == 0:
            count = seq_len - 1
        else:
            count = stride
        total_nll += out["loss"].item() * count
        total_count += count
        n_seqs += 1

    avg_nll = total_nll / max(total_count, 1)
    ppl = math.exp(avg_nll)

    return {
        "ppl": round(ppl, 4),
        "avg_nll": round(avg_nll, 6),
        "n_tokens_scored": total_count,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "stride": stride,
    }


# ---------------------------------------------------------------------------
# Code generation (for HumanEval / MBPP)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: EchoHybridV2Model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    """Simple autoregressive generation for code completion."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=(device != "cpu")):
            out = model(input_ids=input_ids)
        logits = out["logits"][:, -1, :] / max(temperature, 1e-8)

        # Top-p sampling
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = probs.cumsum(dim=-1)
        mask = cumsum - probs > top_p
        sorted_logits[mask] = float("-inf")
        probs = torch.softmax(sorted_logits, dim=-1)
        next_token_sorted = torch.multinomial(probs, num_samples=1)
        next_token = sorted_idx.gather(-1, next_token_sorted)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def eval_humaneval(model, device: str = "cuda", n_samples: int = 1) -> Dict:
    """Run HumanEval evaluation.

    Requires: pip install human-eval
    """
    try:
        from human_eval.data import write_jsonl, read_problems
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        print("  human-eval not installed. Install with: pip install human-eval")
        return {"status": "SKIPPED", "reason": "human-eval not installed"}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)

    problems = read_problems()
    print(f"  HumanEval: {len(problems)} problems, {n_samples} samples each")

    samples = []
    for task_id, problem in problems.items():
        prompt = problem["prompt"]
        for _ in range(n_samples):
            completion = generate(model, tokenizer, prompt, max_new_tokens=512,
                                  temperature=0.2, device=device)
            # Extract just the completion (remove prompt)
            completion = completion[len(prompt):]
            samples.append({"task_id": task_id, "completion": completion})

    # Write samples and evaluate
    tmp_path = "/tmp/humaneval_samples.jsonl"
    write_jsonl(tmp_path, samples)
    results = evaluate_functional_correctness(tmp_path)

    return {
        "pass_at_1": round(results.get("pass@1", 0), 4),
        "n_problems": len(problems),
        "n_samples": n_samples,
    }


def eval_mbpp(model, device: str = "cuda", n_samples: int = 1) -> Dict:
    """Run MBPP evaluation.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return {"status": "SKIPPED", "reason": "datasets not installed"}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)

    ds = load_dataset("mbpp", "sanitized", split="test")
    print(f"  MBPP: {len(ds)} problems")

    correct = 0
    total = 0

    for item in ds:
        prompt = item["prompt"] + "\n"
        completion = generate(model, tokenizer, prompt, max_new_tokens=512,
                              temperature=0.0, device=device)
        code = completion

        # Run test assertions
        test_code = "\n".join(item["test_list"])
        try:
            exec_globals = {}
            exec(code + "\n" + test_code, exec_globals)
            correct += 1
        except Exception:
            pass
        total += 1

    return {
        "pass_at_1": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate born-compressed hybrid")
    parser.add_argument("--ckpt", type=str, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ppl-only", action="store_true", help="Only compute PPL")
    parser.add_argument("--data-dir", type=str, default="data/code_tokens")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-seqs", type=int, default=500)

    # Compare mode
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--born", type=str, help="Born-compressed checkpoint")
    parser.add_argument("--dense", type=str, help="Dense baseline checkpoint")

    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    if args.compare:
        results = {}
        for label, path in [("born", args.born), ("dense", args.dense)]:
            if not path:
                continue
            print(f"\n{'='*70}")
            print(f"Evaluating: {label}")
            print(f"{'='*70}")
            model, step = load_model_from_checkpoint(path, args.device)
            r = {"step": step}

            val_path = Path(args.data_dir) / "val.bin"
            if val_path.exists():
                r["ppl"] = eval_ppl(model, str(val_path), args.seq_len,
                                    max_seqs=args.max_seqs, device=args.device)
                print(f"  PPL: {r['ppl']['ppl']}")

            if not args.ppl_only:
                r["humaneval"] = eval_humaneval(model, args.device)
                r["mbpp"] = eval_mbpp(model, args.device)
                print(f"  HumanEval: {r.get('humaneval', {}).get('pass_at_1', 'N/A')}")
                print(f"  MBPP: {r.get('mbpp', {}).get('pass_at_1', 'N/A')}")

            results[label] = r
            del model; gc.collect(); torch.cuda.empty_cache()

        # Gap analysis
        if "born" in results and "dense" in results:
            born_ppl = results["born"].get("ppl", {}).get("ppl", 0)
            dense_ppl = results["dense"].get("ppl", {}).get("ppl", 0)
            ppl_gap = born_ppl - dense_ppl if born_ppl and dense_ppl else None
            print(f"\n{'='*70}")
            print(f"GAP ANALYSIS")
            print(f"  Born PPL:  {born_ppl}")
            print(f"  Dense PPL: {dense_ppl}")
            if ppl_gap is not None:
                print(f"  Gap:       {ppl_gap:+.4f}")
            print(f"{'='*70}")

    else:
        model, step = load_model_from_checkpoint(args.ckpt, args.device)
        results = {"step": step}

        val_path = Path(args.data_dir) / "val.bin"
        if val_path.exists():
            results["ppl"] = eval_ppl(model, str(val_path), args.seq_len,
                                      max_seqs=args.max_seqs, device=args.device)
            print(f"  PPL: {results['ppl']['ppl']}")

        if not args.ppl_only:
            results["humaneval"] = eval_humaneval(model, args.device)
            results["mbpp"] = eval_mbpp(model, args.device)

    # Receipt
    receipt = {
        "wo": "WO-BORN-HYBRID-01-EVAL",
        "timestamp": time.strftime("%Y-%m-%d"),
        "results": results,
        "cost": cost_block(t_start, cpu_start, start_iso),
    }

    receipt_path = RECEIPT_DIR / "wo_born_hybrid_eval.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nRECEIPT: {receipt_path}")


import gc

if __name__ == "__main__":
    main()
