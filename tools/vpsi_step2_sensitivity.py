"""
Step 2: Gradient-Free Importance via Output Perturbation
========================================================
For each HelixLinear module, zero its output and measure KL divergence
of final logits from unperturbed baseline. Produces 154-element
sensitivity map.

Reads Step 1 receipt for activation magnitude data.
Outputs combined receipt with both columns.

WO-VPSI-01 Step 2

Usage:
    python3 tools/vpsi_step2_sensitivity.py [--n-seqs 20] [--seq-len 128]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import resource
import platform
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seqs", type=int, default=20,
                        help="Number of sequences for sensitivity eval")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--step1-receipt", type=str, default=None,
                        help="Path to Step 1 receipt (auto-detects if not given)")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    # Load compressed TinyLlama
    print("Loading compressed TinyLlama...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from helix_substrate.helix_linear import (
        HelixLinear, load_cdna_factors, swap_to_helix, swap_summary
    )

    model_path = str(Path.home() / "models" / "tinyllama_fp32")
    cdna_path = str(Path.home() / "models" / "tinyllama_fp32" / "cdnav3")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    with torch.no_grad():
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        factors = load_cdna_factors(cdna_path, model)
        model = swap_to_helix(model, factors)
        summary = swap_summary(model)
        print("  Swapped: %d HelixLinear, %d nn.Linear" % (
            summary["helix_modules"], summary["linear_modules"]))

    # Collect all HelixLinear module names
    helix_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, HelixLinear):
            helix_modules[name] = module

    print("  %d HelixLinear modules found" % len(helix_modules))

    # Load WikiText-2
    print("\nLoading WikiText-2...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if len(t.strip()) > 50])
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

    # Prepare input batches
    inputs = []
    for i in range(args.n_seqs):
        start_idx = i * args.seq_len
        if start_idx + args.seq_len > len(tokens):
            break
        inputs.append(tokens[start_idx:start_idx + args.seq_len].unsqueeze(0))
    print("  Prepared %d sequences" % len(inputs))

    # Phase 1: Get baseline logits for all sequences
    print("\nPhase 1: Computing baseline logits...")
    baseline_logits = []
    with torch.no_grad():
        for i, inp in enumerate(inputs):
            out = model(inp)
            # Store log-softmax of last token logits
            baseline_logits.append(
                F.log_softmax(out.logits[0, -1, :].float(), dim=-1)
            )
    print("  Done (%d baselines)" % len(baseline_logits))

    # Phase 2: For each module, zero its output and measure KL divergence
    print("\nPhase 2: Perturbation sensitivity (%d modules x %d seqs)..." % (
        len(helix_modules), len(inputs)))

    sensitivity = {}  # module_name -> mean KL divergence

    total_modules = len(helix_modules)
    for mod_idx, (name, module) in enumerate(helix_modules.items()):
        # Install a zeroing hook
        def zero_hook(mod, inp, out):
            return torch.zeros_like(out)

        handle = module.register_forward_hook(zero_hook)

        kl_divs = []
        with torch.no_grad():
            for seq_idx, inp in enumerate(inputs):
                out = model(inp)
                perturbed = F.log_softmax(out.logits[0, -1, :].float(), dim=-1)
                # KL(baseline || perturbed)
                baseline = baseline_logits[seq_idx]
                kl = F.kl_div(perturbed, baseline.exp(), reduction='sum',
                              log_target=False).item()
                kl_divs.append(kl)

        handle.remove()

        arr = np.array(kl_divs)
        sensitivity[name] = {
            "mean_kl": float(arr.mean()),
            "std_kl": float(arr.std()),
            "max_kl": float(arr.max()),
            "min_kl": float(arr.min()),
        }

        if (mod_idx + 1) % 20 == 0 or mod_idx == total_modules - 1:
            elapsed = time.time() - t_start
            rate = (mod_idx + 1) / elapsed
            eta = (total_modules - mod_idx - 1) / rate
            print("  %d/%d modules (%.1f/s, ETA %.0fs) — last: %s kl=%.4f" % (
                mod_idx + 1, total_modules, rate, eta, name.split(".")[-2],
                sensitivity[name]["mean_kl"]))

    # Results
    print("\n" + "=" * 70)
    print("SENSITIVITY RANKING (Top 20 and Bottom 20)")
    print("=" * 70)

    ranked = sorted(sensitivity.items(), key=lambda x: x[1]["mean_kl"], reverse=True)

    print("\n--- TOP 20 (most sensitive — zeroing causes largest KL divergence) ---")
    print("%4s %-55s %10s %10s" % ("Rank", "Module", "Mean KL", "Std KL"))
    for i, (name, stats) in enumerate(ranked[:20]):
        print("%4d %-55s %10.4f %10.4f" % (i + 1, name, stats["mean_kl"], stats["std_kl"]))

    print("\n--- BOTTOM 20 (least sensitive — safe to skip/lazy-load) ---")
    for i, (name, stats) in enumerate(ranked[-20:]):
        rank = len(ranked) - 20 + i + 1
        print("%4d %-55s %10.4f %10.4f" % (rank, name, stats["mean_kl"], stats["std_kl"]))

    # By functional type
    print("\n" + "=" * 70)
    print("SENSITIVITY BY FUNCTIONAL TYPE")
    print("=" * 70)

    by_type = defaultdict(list)
    for name, stats in sensitivity.items():
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]:
            if proj in name:
                by_type[proj].append(stats["mean_kl"])
                break

    print("%12s %10s %10s %10s %10s" % ("Type", "Mean KL", "Std", "Min", "Max"))
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]:
        vals = np.array(by_type[proj])
        print("%12s %10.4f %10.4f %10.4f %10.4f" % (
            proj, vals.mean(), vals.std(), vals.min(), vals.max()))

    # Correlation: activation magnitude (from Step 1) vs sensitivity
    step1_receipt = args.step1_receipt
    if step1_receipt is None:
        # Auto-detect latest Step 1 receipt
        receipt_dir = Path(__file__).resolve().parent.parent / "receipts" / "vpsi"
        step1_files = sorted(receipt_dir.glob("step1_*.json"))
        if step1_files:
            step1_receipt = str(step1_files[-1])

    act_mag_data = None
    if step1_receipt and Path(step1_receipt).exists():
        print("\nLoading Step 1 receipt: %s" % step1_receipt)
        with open(step1_receipt) as f:
            act_mag_data = json.load(f)["module_stats"]

        # Correlate
        paired_act = []
        paired_sens = []
        for name in sensitivity:
            if name in act_mag_data:
                paired_act.append(act_mag_data[name]["mean_l2"])
                paired_sens.append(sensitivity[name]["mean_kl"])

        if len(paired_act) > 5:
            from scipy.stats import spearmanr
            rho, p = spearmanr(paired_act, paired_sens)
            print("  Activation magnitude vs sensitivity: rho=%.4f, p=%.4e (n=%d)" % (
                rho, p, len(paired_act)))

    # Save receipt
    cost = {
        'wall_time_s': round(time.time() - t_start, 3),
        'cpu_time_s': round(time.process_time() - cpu_start, 3),
        'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
        'timestamp_start': start_iso,
        'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    receipt = {
        "step": "vpsi_step2_sensitivity",
        "model": "TinyLlama-1.1B-compressed",
        "n_sequences": len(inputs),
        "seq_length": args.seq_len,
        "n_modules": len(sensitivity),
        "sensitivity": sensitivity,
        "cost": cost,
    }

    receipt_dir = Path(__file__).resolve().parent.parent / "receipts" / "vpsi"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%dT%H%M%S')
    receipt_path = receipt_dir / ("step2_sensitivity_%s.json" % ts)
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)

    print("\nReceipt: %s" % receipt_path)
    print("Cost: %s" % json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()
