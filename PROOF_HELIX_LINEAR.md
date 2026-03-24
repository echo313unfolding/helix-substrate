# Proof: Compressed Form IS the Executable

**Date:** 2026-03-10
**Claim:** HelixLinear runs TinyLlama 1.1B directly from CDNA v3 compressed representation with no decompression step, achieving +0.78% perplexity at 4.26x persistent memory savings.

---

## What Was Proven

HelixLinear is a drop-in `nn.Linear` replacement that stores weights as:
- **codebook**: [256] float32 cluster centers (1 KB)
- **indices**: [out, in] uint8 cluster assignments (1/4 of float32)
- **sidecar**: sparse outlier corrections (positions + exact values)
- **SVD factors**: optional rank-8 residual (U, s, Vt)

The full float32 weight matrix W exists only as a **temporary during forward()**, never as a persistent allocation.

---

## Result

| Metric | Value |
|--------|-------|
| Baseline perplexity | 6.1717 |
| **HelixLinear perplexity** | **6.2196** |
| **Perplexity delta** | **+0.0479 (+0.78%)** |
| Dense weight memory | 3.95 GB |
| **Compressed memory** | **926 MB** |
| **Memory ratio** | **4.26x** |
| Modules swapped | 154 HelixLinear |
| Modules remaining | 1 nn.Linear (lm_head) |
| Eval tokens | 4096 (WikiText-2 validation) |

### Comparison with Step 5 (weight-swap approach)

| | Step 5 | Step 7 (HelixLinear) |
|---|--------|---------------------|
| Method | Reconstruct W_hat, copy into nn.Linear | Run directly from compressed buffers |
| PPL delta | +0.87% | +0.78% |
| Persistent memory | Full float32 (dense) | Compressed (4.26x smaller) |
| Tensors compressed | 156 (incl. embed + lm_head) | 154 (block tensors only) |

Step 7 is slightly better on perplexity because it does not compress embed_tokens and lm_head.

---

## Forward Pass Architecture

```
forward(x):
    1. GATHER:  W_vq = codebook[indices]       # temporary float32, only during forward
    2. PATCH:   W_vq[positions] = values        # sparse outlier correction
    3. SVD:     W = W_vq + (U * s) @ Vt         # low-rank residual (rank-8)
    4. MATMUL:  output = x @ W.T + bias         # standard F.linear

    After return: W is freed. Only compressed buffers persist.
```

Two execution paths:
- **CPU/naive**: Reconstruct W as temporary, then `F.linear(x, W, bias)`
- **GPU/Triton fused**: `_vq_gather_matmul_kernel` — codebook gather in registers, W never in global memory. Sidecar + SVD applied as small post-corrections.

---

## Triton Fused Kernel

The Triton kernel (`triton_vq_matmul.py`) computes `output = x @ codebook[indices]^T` without materializing W:

- **Outer-product accumulation**: For each k in IN, load x[:, k] and codebook[indices[:, k]], accumulate outer product
- **Memory**: 0.1 MB peak vs 64 MB for full W (at 4096x4096)
- **Accuracy**: max diff vs naive 6.71e-04 at 4096x4096
- **Compatibility**: Works on Triton 3.2.0 / Compute Capability 7.5 (Turing)
- **Tests**: 7/7 pass

---

## Limitations

- **Naive path used for Step 7 receipt.** The Triton fused path was tested in unit tests but the perplexity benchmark ran on CPU (naive path). GPU perplexity with fused kernel not yet receipted.

- **Temporary W still exists.** In the naive path, the full W is materialized as a temporary during forward(). True zero-materialization requires the fused Triton path. On CPU, peak memory still includes one full weight matrix at a time.

- **lm_head not swapped.** The 1 remaining nn.Linear (lm_head / output projection) is kept dense. Could be swapped for additional savings.

- **No KV cache accounting.** The 4.26x ratio covers weight memory only. KV cache grows with sequence length and is not compressed.

- **Single model.** Only TinyLlama 1.1B tested.

---

## Files

| File | Purpose |
|------|---------|
| `helix_substrate/helix_linear.py` | HelixLinear module + swap utilities |
| `helix_substrate/triton_vq_matmul.py` | Fused VQ gather-matmul kernel |
| `tools/step7_helix_linear_integration.py` | Integration proof script |

---

## GPU Viability (Step 8)

Tested on Quadro T2000 (4 GB VRAM).

| Metric | Dense FP32 | HelixLinear |
|--------|-----------|-------------|
| Fits on 4 GB GPU | **NO (OOM)** | **YES** |
| VRAM (model) | N/A | 1493 MB (38%) |
| VRAM (peak) | N/A | 2870 MB (74%) |
| VRAM headroom | 0 | 2394 MB |
| Prompt throughput | N/A | 97 tok/s |
| Decode throughput | N/A | 3.3 tok/s |
| Triton fused | N/A | Active |
| Perplexity | N/A | 6.2196 (matches CPU) |

The HelixLinear model uses 38% of VRAM, leaving 2.4 GB for KV cache, tools, RAG, or other workloads.

## Receipts

- Step 7: `receipts/step7_helix_linear/helix_linear_integration_20260310T214642.json`
- Step 8: `receipts/step8_gpu_viability/gpu_viability_20260311T013916.json`

## Reproduction

```bash
# CPU integration test
python tools/step7_helix_linear_integration.py --tokens 4096

# GPU viability benchmark
python tools/step8_gpu_viability.py
```
