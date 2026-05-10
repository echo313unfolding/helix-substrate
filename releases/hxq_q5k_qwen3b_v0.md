# HXQ Meta-Codec Q5_K Release — Qwen2.5-Coder-3B-Instruct

## Artifact

```
File:   ~/models/qwen2.5-coder-3b-instruct-hxq-q5k-v3.gguf
Size:   2,536,743,072 bytes (2.36 GiB)
SHA256: c25d242df60b44be1d9e95c32da5647f1b973e2b35258c69daf7eb413876c67f
```

## Architecture

```
Model:        qwen2
Parameters:   3.09 B
Layers:       36
Hidden:       2048
Vocab:        151936
Tensors:      434 (252 Q5_K + 181 F32 bias/norm + 1 F16 embedding)
Effective bpw: ~5.5 (weight matrices), ~6.6 overall (F16 embed + F32 norms)
```

## Performance (T2000, CUDA, ngl=99)

| Model | Size (GiB) | pp128 (t/s) | tg32 (t/s) |
|-------|-----------|-------------|------------|
| Q4_K_M stock | 1.95 | 243.28 | 33.33 |
| Q5_K_M stock | 2.07 | 240.32 | 33.18 |
| **HXQ Q5_K v3** | **2.36** | **239.72** | **33.51** |

## Behavior Tests (llama-server, CUDA, ngl=99, ctx=2048)

| Test | Response | Tokens | Speed |
|------|----------|--------|-------|
| "Hello! How are you?" | "Hello! I'm just a large language model created by Alibaba Cloud," | 14 | 36.9 tok/s |
| "Write Fibonacci function" | Correct iterative Python implementation | 70 | 34.4 tok/s |
| "Capital of France?" | "The capital of France is Paris." | 8 | 38.3 tok/s |

All responses coherent, on-topic, correctly formatted. Model answers normally.

## Compatibility

- Runtime: stock llama.cpp (any build with Q5_K support)
- Custom kernels: NONE required
- GGUF format: standard V3, no extensions
- Tensor types: Q5_K (weight matrices), F16 (embedding), F32 (bias/norm)
- Chat template: Qwen2 ChatML (im_start/im_end)
- Tokenizer: GPT-2 BPE with qwen2 pre-tokenizer

## Routing Policy (Variant A)

```
Weight matrices (2D, in_dim % 256 == 0) → Q5_K
Embedding                               → F16 (Q5_K crashes CUDA binbcast)
Biases                                  → F32 (required by llama.cpp binary_op)
Norms (RMSNorm)                         → F32 (required by llama.cpp binary_op)
```

## Validation

- Tensor cosine gate: all validated tensors >= 0.998 (from manifest)
- Byte-level Q5_K compatibility: 6/6 audit tests PASS, 0.00 decode error
- llama.cpp load: PASS
- llama-bench: PASS (performance within noise of stock)
- llama-server: PASS (chat completions, coherent responses)
- Tokenize/detokenize roundtrip: PASS
- Full-model manifest: 434 tensors routed, 0 failures

## Key Fixes (from v0 iterations)

1. **Token types**: BPE tokens must be type=1 (NORMAL), not type=0. Type=0 breaks detokenization.
2. **Bias/norm dtype**: Must be F32. F16 causes CUDA binbcast assertion failure.
3. **Embedding dtype**: Must be F16 (not Q5_K). Q5_K embedding crashes CUDA during inference.
4. **Vocab padding**: vocab.json (151643) padded to config.vocab_size (151936) with [PAD{i}] type=4.
5. **Tensor names**: HF names mapped to GGUF names via TensorNameMap.
6. **Tokenizer metadata**: Requires `tokenizer.ggml.pre=qwen2`, `add_bos_token=false`, chat_template.

## Reproduction

```bash
# Fast path (rename + fix from existing HXQ GGUF, ~11s):
python3 ~/tools/hxq_gguf_rename_tensors.py

# From scratch (slow, ~45 min, re-encodes all tensors):
python3 ~/tools/hxq_meta_emit_gguf.py

# Verify (bench):
cd ~/llama.cpp
LD_LIBRARY_PATH=build/bin build/bin/llama-bench \
  -m ~/models/qwen2.5-coder-3b-instruct-hxq-q5k-v3.gguf \
  -ngl 99 -p 128 -n 32 -r 3

# Verify (behavior):
LD_LIBRARY_PATH=build/bin build/bin/llama-server \
  -m ~/models/qwen2.5-coder-3b-instruct-hxq-q5k-v3.gguf \
  -ngl 99 --port 18090 -np 1 -c 2048
curl -s http://127.0.0.1:18090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":32,"temperature":0}'
```

## Receipts

- `receipts/hxq_meta_gguf_emit_v0_20260504.json` (emit + smoke + GPU bench)
- `receipts/hxq_q5h_runtime_alias_audit_20260504.json` (byte compatibility)
- `receipts/hxq_meta_qwen3b_manifest_20260504.json` (full-model routing)
- `receipts/hxq_q5_hierarchical_proof_v0_20260504.json` (hierarchy proof)

## What HXQ adds over stock Q5_K_M

1. **Consistent quantization**: all weight matrices at Q5_K (stock mixes Q4/Q5/Q6)
2. **Routing receipts**: per-tensor validation with cosine gates
3. **Meta-codec pipeline**: manifest → validate → emit → verify
4. **Provenance**: every tensor has a documented routing decision and quality proof

## Size Note

Current size (2.36 GiB) is larger than stock Q5_K_M (2.07 GiB) due to:
- F16 embedding (593 MB) vs stock Q6_K embedding (smaller)
- F32 biases/norms vs stock mixed precision

Stock Q5_K_M achieves smaller size by using Q6_K for embedding and mixed Q4/Q5/Q6
for different layer types. HXQ uses uniform Q5_K for all weight matrices but keeps
embedding/bias at higher precision for runtime safety.

## Date

2026-05-04 (initial), 2026-05-05 (v3 with behavior tests)
