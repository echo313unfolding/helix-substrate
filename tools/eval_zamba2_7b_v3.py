#!/usr/bin/env python3
"""GPU BF16 eval for Zamba2-7B — device_map=auto for dense, direct GPU for helix."""
import sys, os, json, time, platform, gc
sys.path.insert(0, "/home/user")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_LENGTH = 512
STRIDE = 256

dense_path = "/home/user/models/zamba2-7b-instruct"
helix_path = "/home/user/models/zamba2-7b-instruct-helix"

def eval_ppl_manual(model, input_ids, max_length, stride, device):
    """Manual PPL — works even without labels support."""
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
        if n_tokens % 1024 < stride:
            print(f"  {n_tokens} tokens...", flush=True)
    return float(np.exp(sum(nlls) / n_tokens)), n_tokens

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(dense_path, trust_remote_code=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
encodings = tokenizer(text, return_tensors="pt")
input_ids = encodings.input_ids
print(f"Tokens: {input_ids.size(1)}")

result = {"model": "zamba2-7b-instruct", "dtype": "bfloat16", "max_length": MAX_LENGTH}
t0 = time.time()
ts_start = time.strftime("%Y-%m-%dT%H:%M:%S")

# Dense eval with device_map=auto (CPU/GPU split)
print("\n[1/2] Loading dense model with device_map=auto...")
try:
    dense = AutoModelForCausalLM.from_pretrained(
        dense_path, torch_dtype=DTYPE, trust_remote_code=True,
        device_map="auto", low_cpu_mem_usage=True,
    ).eval()
    dev = next(dense.parameters()).device
    print(f"Dense model device: {dev}, VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print("Evaluating dense PPL...")
    dense_ppl, n_tok = eval_ppl_manual(dense, input_ids, MAX_LENGTH, STRIDE, dev)
    print(f"Dense PPL: {dense_ppl:.4f} ({n_tok} tokens)")
    result["dense_ppl"] = round(dense_ppl, 4)
    result["n_tokens"] = n_tok
except Exception as e:
    print(f"Dense eval FAILED: {e}")
    import traceback; traceback.print_exc()
    result["dense_ppl"] = None
    result["dense_error"] = str(e)

# Clean up dense
try:
    del dense
except:
    pass
gc.collect(); torch.cuda.empty_cache()
print(f"VRAM after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Helix eval
print("\n[2/2] Loading helix model...")
import helix_substrate.hf_quantizer
helix = AutoModelForCausalLM.from_pretrained(
    helix_path, torch_dtype=DTYPE, trust_remote_code=True,
).to(DEVICE).eval()
print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

from helix_substrate.helix_linear import HelixLinear
n_helix = sum(1 for m in helix.modules() if isinstance(m, HelixLinear))
print(f"HelixLinear modules: {n_helix}")
result["helix_modules"] = n_helix

print("Evaluating helix PPL...")
helix_ppl, n_tok_h = eval_ppl_manual(helix, input_ids, MAX_LENGTH, STRIDE, DEVICE)
print(f"Helix PPL: {helix_ppl:.4f} ({n_tok_h} tokens)")
result["helix_ppl"] = round(helix_ppl, 4)
del helix
gc.collect(); torch.cuda.empty_cache()

# Delta
if result.get("dense_ppl"):
    delta = (result["helix_ppl"] - result["dense_ppl"]) / result["dense_ppl"] * 100
    result["delta_pct"] = round(delta, 2)
    print(f"\nDelta: {delta:+.2f}%")

ts_end = time.strftime("%Y-%m-%dT%H:%M:%S")
result["cost"] = {
    "wall_time_s": round(time.time() - t0, 1),
    "python_version": platform.python_version(),
    "hostname": platform.node(),
    "timestamp_start": ts_start,
    "timestamp_end": ts_end,
}

os.makedirs("/home/user/receipts/gpu_eval", exist_ok=True)
receipt_path = "/home/user/receipts/gpu_eval/zamba2_7b_instruct_gpu_bf16.json"
with open(receipt_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nReceipt: {receipt_path}")
print(json.dumps(result, indent=2))
