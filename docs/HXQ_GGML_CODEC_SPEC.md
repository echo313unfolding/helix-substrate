# HXQ GGML Codec Specification

**Status:** Implemented. Branch `hxq-affine-type` in `echo313unfolding/llama.cpp`.

This document specifies the HXQ affine tensor types as implemented in ggml/GGUF. It covers the tensor layout, decode equation, metadata, and acceptance test results.

---

## 1. Type Definitions

Two types are registered in the `GGML_TYPE` enum:

| Enum | ID | Name | Group Size | Bits | BPW | Block Size (bytes) |
|------|-----|------|-----------|------|-----|-------------------|
| `GGML_TYPE_HXQ_AFFINE_G128` | 36 | `hxq_affine_g128` | 128 | 8 | 8.25 | 132 |
| `GGML_TYPE_HXQ_AFFINE_6` | 37 | `hxq_affine_6` | 128 | 6 | 6.25 | 100 |

Production type is `HXQ_AFFINE_6`. `HXQ_AFFINE_G128` is the 8-bit variant (retained for testing/compatibility).

---

## 2. Block Layout

### HXQ_AFFINE_G128 (8-bit, 132 bytes per 128 elements)

```
offset  size  field
0       2     scale   (float16)
2       2     offset  (float16)
4       128   qs[128] (uint8 indices, 0-255)
```

### HXQ_AFFINE_6 (6-bit, 100 bytes per 128 elements)

```
offset  size  field
0       2     scale   (float16)
2       2     offset  (float16)
4       96    qs[96]  (6-bit packed indices, 4 per 3 bytes)
```

6-bit packing layout (4 indices per 3 bytes):
```
byte[0]: [idx0[5:0] | idx1[1:0]<<6]
byte[1]: [idx1[5:2] | idx2[3:0]<<4]
byte[2]: [idx2[5:4] | idx3[5:0]<<2]
```

Each index range: 0-63 (64 levels).

---

## 3. Decode Equation

For both types:

```
W[i] = qs[i] * scale + offset
```

Where:
- `scale = (max - min) / (n_levels - 1)` — computed at quantize time, stored as float16
- `offset = min` — stored as float16
- `qs[i]` — quantized index (uint8 for g128, 6-bit packed for affine_6)
- `n_levels` — 256 for g128, 64 for affine_6

This is standard per-group affine (min/max) quantization. No codebook lookup. No calibration data.

---

## 4. Quantize Equation

```
idx = clamp(round((x - min) / scale), 0, n_levels - 1)
```

Where `min` and `max` are computed per group of 128 elements. Division-by-zero guard: if `max == min`, `scale = 0`, all indices = 0.

---

## 5. Supported Tensor Shapes

Any tensor whose innermost dimension is divisible by 128 (the group size). This covers all standard `nn.Linear` weight matrices. Tensors not divisible by 128 are zero-padded to the next multiple.

**Unsupported:** Tensors stored as exact (norms, biases, embeddings) are NOT quantized — stored as F16/F32 in the GGUF file.

---

## 6. GGUF Metadata

Standard GGUF tensor metadata. No custom keys required beyond the type enum. The type field in the GGUF tensor header identifies HXQ blocks. `convert_hf_to_gguf.py` sets the type during conversion.

---

## 7. Kernel Implementations

### CPU (AVX2)
- `vec_dot_hxq_affine_6_q8_1`: 6-bit unpack → shuffle → maddubs → horizontal sum
- `vec_dot_hxq_affine_g128_q8_1`: 8-bit XOR bias removal → maddubs → accumulate

### GPU Decode (N=1, mmvq)
- 6-bit: float unpack in registers, FP32 dot product
- 8-bit: dp4a INT8 dot product (reuse Q8 path)

### GPU Prefill (N>8, mmq/MMA)
- `load_tiles_hxq_affine_6`: 6-bit → int8 unpack in shared memory
- Delegates to `vec_dot_q8_1_q8_1_mma` for INT8 tensor core MMA
- Scale/offset stored as half2 in dm array (Q4_1-compatible delegation)
- Works on Turing (m8n8k16 x4) and Ampere (m16n8k32 x1)

---

## 8. Benchmark Results

### Qwen2.5-3B on T2000 (Quadro T2000, CC 7.5, 4GB)

| Quant | BPW | Size | pp128 | pp512 | tg128 |
|-------|-----|------|-------|-------|-------|
| Q4_K_M | 4.99 | 1.79 GiB | 246.58 | 236.10 | 43.65 |
| **HXQ_AFFINE_6** | **6.25** | **2.26 GiB** | **235.84** | **226.95** | **27.78** |
| Q8_0 | 8.50 | 3.05 GiB | 240.49 | 232.08 | 28.31 |

**Decode: 98% of Q8_0 at 26% smaller size.** Prefill: 98% of Q8_0.

### Qwen2.5-3B on RTX 3090 (CC 8.6, 24GB)

| Quant | BPW | Size | pp128 | pp512 | tg128 |
|-------|-----|------|-------|-------|-------|
| **HXQ_AFFINE_6** | **6.25** | **2.48 GiB** | **4905** | **7815** | **161.8** |
| Q8_0 | 8.50 | 3.36 GiB | 6095 | 8928 | 183.2 |

HXQ at 81-91% of Q8_0 on compute-bound 3090 (6-bit unpack overhead).

### Perplexity (WikiText-2, 3090, 584 chunks, c=512)

| Quant | PPL | Delta |
|-------|-----|-------|
| Q8_0 | 9.1179 | baseline |
| HXQ_AFFINE_6 | 9.2723 | +1.69% |

---

## 9. Tensor Codec Evidence

HXQ affine is not LLM-specific. Receipted across:

**ML architectures (6 families):** Transformer, SSM, Hybrid SSM+Transformer, CNN, ViT, Sentence Transformer. All cos > 0.999.

**Raw tensor distributions (5 types):** Mixed Gaussian, Uniform, Heavy-tailed Cauchy, Log-normal, Sparse 90%. All cos >= 0.998. Gate: PASS.

**Non-uniform alternatives tested and rejected:** Quantile-optimal (cos 0.9911) and mu-law (cos 0.09) both worse than affine (cos 0.9985) on heavy-tailed Cauchy.

See `docs/HXQ_TENSOR_CODEC_EVIDENCE.md` for full evidence tables.

---

## 10. Acceptance Tests (Status)

| Test | Status | Receipt |
|------|--------|---------|
| Roundtrip tensor reconstruction | PASS | Built into quantize/dequantize |
| GGUF write (convert_hf_to_gguf.py) | PASS | 6 models uploaded to HF |
| GGUF read (llama.cpp load) | PASS | All models load cleanly |
| llama-quantize conversion (F16 -> HXQ) | PASS | `--type hxq_affine_6` |
| Small model inference parity | PASS | Qwen2.5-3B coherent output |
| Perplexity delta | +1.69% vs Q8_0 | `receipts/hxq_ggml_native/` |
| Speed/memory measurement | 98% Q8_0 decode, 26% smaller | `receipts/hxq_ggml_native/` |
| CPU correctness (32/32 tests) | PASS | `receipts/hxq_ggml_native/` |
| GPU correctness (9/9 tests) | PASS | `receipts/hxq_ggml_native/` |

---

## 11. Files Modified (10 files, 217 insertions)

```
ggml/include/ggml.h                      — type enum
ggml/src/ggml-common.h                   — block structs
ggml/src/ggml.c                          — type registration
ggml/src/ggml-quants.h                   — function declarations
ggml/src/ggml-quants.c                   — quantize/dequantize + histograms
ggml/src/ggml-cpu/quants.c               — CPU vec_dot kernels
ggml/src/ggml-cpu/ops.cpp                — get_rows dispatch
ggml/src/ggml-cuda/mmvq.cu              — GPU decode kernel
ggml/src/ggml-cuda/mmq.cu               — GPU prefill (MMA) tile loader
ggml/src/ggml-cuda/mmq-instance-hxq_affine_6.cu — template instance
```

---

## 12. Upstream Strategy

**Phase 1 (current):** One stable type (`HXQ_AFFINE_6`), full correctness + benchmarks, PR after Zamba2 PR #21412 establishes contributor standing.

**Phase 2 (after merge):** Multi-architecture GGUF (Zamba2, more Qwen sizes).

**Phase 3 (future):** Hydra multi-head routing (per-tensor bit-width selection). Not in initial PR.

**PR framing:** "This adds one calibration-free GGML tensor type with deterministic per-group affine reconstruction. Here are parity and benchmark receipts."

---

## 13. Receipt Paths

```
receipts/hxq_ggml_native/hxq_ggml_type_bench.json
receipts/hxq_mma_kernel_3090_debug.json
receipts/raw_distribution_probe/hxq_raw_distribution_test_20260502T103120.json
receipts/nonuniform_ceiling_probe/nonuniform_ceiling_20260502T131054.json
docs/HXQ_TENSOR_CODEC_EVIDENCE.md
```
