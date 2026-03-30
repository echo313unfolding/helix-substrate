#!/usr/bin/env python3
"""GPU BF16 eval for Zamba2-7B helix model only."""
import sys, os, json, time, platform, gc
sys.path.insert(0, "/home/user")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from datasets import load_dataset

DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_LENGTH = 512
STRIDE = 256

dense_path = "/home/user/models/zamba2-7b-instruct"
helix_path = "/home/user/models/zamba2-7b-instruct-helix"

def eval_ppl_manual(model, input_ids, max_length, stride, device):
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
        if n_tokens % 2048 < stride:
            print(f"  {n_tokens} tokens...", flush=True)
    return float(np.exp(sum(nlls) / n_tokens)), n_tokens

# Register quantizer FIRST before any transformers model loading
try:
    import helix_substrate.hf_quantizer
except ValueError:
    pass  # already registered

from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(dense_path, trust_remote_code=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
encodings = tokenizer(text, return_tensors="pt")
input_ids = encodings.input_ids
print(f"Tokens: {input_ids.size(1)}")

t0 = time.time()
ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")

# Dense result from v3 run
DENSE_PPL = 4.6047

print(f"\nLoading helix model from {helix_path}...")
helix = AutoModelForCausalLM.from_pretrained(
    helix_path, torch_dtype=DTYPE, trust_remote_code=True,
).to(DEVICE).eval()
print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

from helix_substrate.helix_linear import HelixLinear
n_helix = sum(1 for m in helix.modules() if isinstance(m, HelixLinear))
print(f"HelixLinear modules: {n_helix}")

print("Evaluating helix PPL...")
helix_ppl, n_tok = eval_ppl_manual(helix, input_ids, MAX_LENGTH, STRIDE, DEVICE)
print(f"Helix PPL: {helix_ppl:.4f} ({n_tok} tokens)")
del helix
gc.collect(); torch.cuda.empty_cache()

delta = (helix_ppl - DENSE_PPL) / DENSE_PPL * 100
print(f"\nDense PPL: {DENSE_PPL:.4f}")
print(f"Helix PPL: {helix_ppl:.4f}")
print(f"Delta: {delta:+.2f}%")

ts_end = time.strftime("%Y-%m-%dT%H:%M:%S")
result = {
    "model": "zamba2-7b-instruct",
    "dtype": "bfloat16",
    "device": DEVICE,
    "max_length": MAX_LENGTH,
    "stride": STRIDE,
    "dataset": "wikitext-2-raw-v1",
    "dense_ppl": DENSE_PPL,
    "helix_ppl": round(helix_ppl, 4),
    "delta_pct": round(delta, 2),
    "n_tokens": n_tok,
    "helix_modules": n_helix,
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
