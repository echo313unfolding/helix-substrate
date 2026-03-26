#!/usr/bin/env python3
"""Domain 7: LoRA Adapter Compression and Composition.
Create a real LoRA adapter, compress its matrices, verify merged weight fidelity."""

import torch, numpy as np, tempfile, sys
from pathlib import Path

# Block awq import to avoid PytorchGELUTanh conflict with transformers
import importlib
_fake_awq = type(sys)('awq')
_fake_awq.__spec__ = importlib.machinery.ModuleSpec('awq', None)
sys.modules['awq'] = _fake_awq
_fake_linear = type(sys)('awq.modules.linear')
class _DummyWQLinear: pass
_fake_linear.WQLinear_GEMM = _DummyWQLinear
sys.modules['awq.modules'] = type(sys)('awq.modules')
sys.modules['awq.modules.linear'] = _fake_linear

from _common import *

def main():
    t_start, cpu_start, start_iso = start_cost()
    print("=" * 72)
    print("DOMAIN 7: LoRA Adapter Compression")
    print("=" * 72)

    from peft import get_peft_model, LoraConfig, TaskType

    model, tokenizer = load_tinyllama_model()
    batch = load_wikitext2_batch(tokenizer, n_tokens=128)

    # Apply LoRA (rank=16 targeting q_proj, v_proj)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Train 5 steps
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(5):
        outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"].clone())
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step+1}/5: loss={outputs.loss.item():.4f}")

    # Extract LoRA matrices
    out_dir = HELIX_ROOT / "tensor_infra_scratch" / "lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_matrices = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            lora_matrices[name] = param.data.detach().cpu().numpy().astype(np.float32)

    print(f"\n  Extracted {len(lora_matrices)} LoRA matrices")

    results = []
    for name, matrix in lora_matrices.items():
        kurt = kurtosis(matrix)
        # LoRA matrices can be very small (e.g., 16x2048 or 2048x16)
        # Use k=64 for small matrices, k=256 for larger ones
        min_dim = min(matrix.shape)
        k_val = 64 if min_dim <= 32 else 256
        policy = policy_vq(k=k_val, sidecar=True)

        safe = name.replace(".", "_")
        stats, recon = compress_tensor(matrix, safe, out_dir, policy)
        cos = cosine_sim(matrix, recon)

        results.append({
            "name": name,
            "shape": list(matrix.shape),
            "kurtosis": round(kurt, 2),
            "k": k_val,
            "cosine": round(cos, 6),
            "compression_ratio": stats.get("compression_ratio", 1.0),
            "original_bytes": matrix.nbytes,
            "storage_mode": stats.get("storage_mode", "codebook"),
        })
        print(f"    {name}: shape={matrix.shape}, cos={cos:.6f}, "
              f"ratio={stats.get('compression_ratio', 1.0):.2f}x")

    # Also compress merged deltas: B @ A for each layer's q_proj and v_proj
    print(f"\n  Merged delta compression:")
    merged_results = []
    # Group by layer and module
    lora_a = {n: m for n, m in lora_matrices.items() if "lora_A" in n}
    lora_b = {n: m for n, m in lora_matrices.items() if "lora_B" in n}

    for a_name, a_mat in lora_a.items():
        # Find matching B matrix
        b_name = a_name.replace("lora_A", "lora_B")
        if b_name not in lora_b:
            continue
        b_mat = lora_b[b_name]
        merged = (b_mat @ a_mat).astype(np.float32)  # (out_features, in_features)
        kurt = kurtosis(merged)

        policy = policy_vq(k=256, sidecar=True)
        safe = a_name.replace(".", "_").replace("lora_A", "merged")
        stats, recon = compress_tensor(merged, safe, out_dir / "merged", policy)
        cos = cosine_sim(merged, recon)

        merged_results.append({
            "pair": f"{a_name} + {b_name}",
            "merged_shape": list(merged.shape),
            "kurtosis": round(kurt, 2),
            "cosine": round(cos, 6),
            "compression_ratio": stats.get("compression_ratio", 1.0),
        })
        print(f"    merged {merged.shape}: cos={cos:.6f}, ratio={stats.get('compression_ratio', 1.0):.2f}x")

    # Verdict
    all_cosines = [r["cosine"] for r in results]
    v, worst = verdict(all_cosines, strong=0.999, passing=0.99)
    print(f"\n  Verdict: {v} (worst matrix cosine={worst:.6f})")
    print(f"  FLAG: 5-step training — adapter weights are real but not from converged fine-tune")

    cost = finish_cost(t_start, cpu_start, start_iso)
    write_receipt("tensor_infra_domain_7", "lora_compress", {
        "n_lora_matrices": len(results),
        "per_matrix": results,
        "merged_deltas": merged_results,
        "verdict": v,
        "data_source": "PARTIAL — PEFT LoRA on TinyLlama, 5 steps on WikiText-2",
        "flag": "5-step training, not converged",
    }, cost)

if __name__ == "__main__":
    main()
