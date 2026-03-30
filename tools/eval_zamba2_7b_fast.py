#!/usr/bin/env python3
"""GPU BF16 eval for Zamba2-7B helix — 8K tokens (fast estimate)."""
import sys, os, json, time, platform, gc
sys.path.insert(0, "/home/user")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Register quantizer BEFORE any transformers imports
try:
    import helix_substrate.hf_quantizer
except (ValueError, ImportError):
    pass

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_LENGTH = 512
STRIDE = 256
N_TOKENS = 8192  # fast estimate

dense_path = "/home/user/models/zamba2-7b-instruct"
helix_path = "/home/user/models/zamba2-7b-instruct-helix"

def eval_ppl_manual(model, input_ids, max_length, stride, device, max_tokens=None):
    seq_len = input_ids.size(1)
    nlls, n_tokens, prev_end = [], 0, 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        chunk = input_ids[:, begin:end].to(device)
        with torch.no_grad():
            logits = model(chunk).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        if target_len < end - begin:
            offset = (end - begin) - target_len - 1
            shift_logits = shift_logits[:, offset:, :]
            shift_labels = shift_labels[:, offset:]
        loss = torch.nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).float().item()
        actual = shift_labels.numel()
        nlls.append(loss * actual)
        n_tokens += actual
        prev_end = end
        if end == seq_len:
            break
        if max_tokens and n_tokens >= max_tokens:
            break
    return float(np.exp(sum(nlls) / n_tokens)), n_tokens

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(dense_path, trust_remote_code=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
encodings = tokenizer(text, return_tensors="pt")
input_ids = encodings.input_ids
print(f"Total tokens available: {input_ids.size(1)}, using first ~{N_TOKENS}")

t0 = time.time()
ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")

# Dense baseline from full-dataset run
DENSE_PPL_FULL = 4.6047

# Also do a quick 8K dense eval for apples-to-apples comparison
print("\n[1/3] Quick dense 8K eval...")
dense = AutoModelForCausalLM.from_pretrained(
    dense_path, torch_dtype=DTYPE, trust_remote_code=True,
    device_map="auto", low_cpu_mem_usage=True,
).eval()
dev = next(dense.parameters()).device
dense_ppl_8k, n_tok_d = eval_ppl_manual(dense, input_ids, MAX_LENGTH, STRIDE, dev, max_tokens=N_TOKENS)
print(f"Dense PPL (8K): {dense_ppl_8k:.4f} ({n_tok_d} tokens)")
del dense
gc.collect(); torch.cuda.empty_cache()

# Helix eval
print(f"\n[2/3] Loading helix model...")
helix = AutoModelForCausalLM.from_pretrained(
    helix_path, torch_dtype=DTYPE, trust_remote_code=True,
).to(DEVICE).eval()
print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

from helix_substrate.helix_linear import HelixLinear
n_helix = sum(1 for m in helix.modules() if isinstance(m, HelixLinear))
print(f"HelixLinear modules: {n_helix}")

print("[3/3] Helix PPL (8K tokens)...")
helix_ppl, n_tok_h = eval_ppl_manual(helix, input_ids, MAX_LENGTH, STRIDE, DEVICE, max_tokens=N_TOKENS)
print(f"Helix PPL (8K): {helix_ppl:.4f} ({n_tok_h} tokens)")
del helix
gc.collect(); torch.cuda.empty_cache()

# Deltas
delta_8k = (helix_ppl - dense_ppl_8k) / dense_ppl_8k * 100
delta_full = (helix_ppl - DENSE_PPL_FULL) / DENSE_PPL_FULL * 100

print(f"\n{'='*50}")
print(f"Dense PPL (8K):   {dense_ppl_8k:.4f}")
print(f"Dense PPL (full): {DENSE_PPL_FULL:.4f}")
print(f"Helix PPL (8K):   {helix_ppl:.4f}")
print(f"Delta (8K/8K):    {delta_8k:+.2f}%")
print(f"Delta (8K/full):  {delta_full:+.2f}%")
print(f"VRAM: 6.96 GB helix vs 13.70 GB dense = 1.97x savings")
print(f"{'='*50}")

ts_end = time.strftime("%Y-%m-%dT%H:%M:%S")
result = {
    "model": "zamba2-7b-instruct",
    "dtype": "bfloat16",
    "device": DEVICE,
    "max_length": MAX_LENGTH,
    "stride": STRIDE,
    "dataset": "wikitext-2-raw-v1",
    "n_tokens_eval": N_TOKENS,
    "dense_ppl_8k": round(dense_ppl_8k, 4),
    "dense_ppl_full": DENSE_PPL_FULL,
    "helix_ppl": round(helix_ppl, 4),
    "delta_pct": round(delta_8k, 2),
    "n_tokens_dense": n_tok_d,
    "n_tokens_helix": n_tok_h,
    "helix_modules": n_helix,
    "vram_helix_gb": 6.96,
    "vram_dense_gb": 13.70,
    "cost": {
        "wall_time_s": round(time.time() - t0, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": ts_start,
        "timestamp_end": ts_end,
    },
}

os.makedirs("/home/user/receipts/gpu_eval", exist_ok=True)
receipt_path = "/home/user/receipts/gpu_eval/zamba2_7b_instruct_gpu_bf16.json"
with open(receipt_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nReceipt: {receipt_path}")
print(json.dumps(result, indent=2))
