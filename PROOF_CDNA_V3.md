# CDNA v3 Compression Proof — TinyLlama 1.1B

**Date:** 2026-03-10
**Model:** TinyLlama 1.1B FP32 (4.40 GB, 201 tensors, 22 layers)
**Eval:** WikiText-2 validation (4096 tokens)
**Result:** 4.40 GB → 1.10 GB (3.99x) at +0.87% perplexity

---

## Architecture

```
classify(name, shape) → TensorClass (7 classes)
    ↓
get_policy(name, shape, kurtosis) → TensorPolicy
    ↓
CDNAv3Writer.write_tensor(W, policy) → codebook + indices + sidecar [+ SVD factors]
    ↓
CDNAv3Reader.reconstruct() → W_hat
```

**Codec families (2):**
- **VQ_ONLY** — k-means codebook (k=256) + uint8 indices + sidecar outlier corrections. Default for all blocks 1-21.
- **VQ_SVD_R8** — VQ_ONLY + rank-8 SVD residual correction. Routed for block 0 (anomalous kurtosis) and any tensor with kurtosis > 50.

**Routing rule:**
- Block 0 2D tensors → VQ_SVD_R8 (always)
- Kurtosis > 50 → VQ_SVD_R8 (safety net)
- Everything else → VQ_ONLY

**Storage format per tensor:**
```
{name}.cdnav3/
  meta.json        — shape, name, policy, codebook hash
  codebook.npy     — [256] float32 centroids
  indices.bin      — [rows×cols] uint8 cluster assignments
  sidecar.npz      — outlier positions + exact values (optional)
  svd_U.npy        — [rows, rank] left singular vectors (optional)
  svd_s.npy        — [rank] singular values (optional)
  svd_Vt.npy       — [rank, cols] right singular vectors (optional)
  stats.json       — encoding fidelity metrics
```

---

## Proof Staircase

| Step | What | Result | Receipt |
|------|------|--------|---------|
| **0** | Activation-aware evaluation harness | 36-tensor baseline, mean act_cos=0.9995 | `receipts/activation_baseline/*.json` |
| **1** | Low-rank residual benchmark (5 weakest tensors) | VQ+SVD(r=8) improves all 5; low-rank alone rejected | `receipts/step1_lowrank/lowrank_benchmark_*.json` |
| **1.5** | Routing rule validation (16 tensors) | 100% precision, 0 false positives | `receipts/step1_lowrank/routing_confirmation_*.json` |
| **2** | Mixed decoder integration into pipeline | 5/5 tests pass, streaming verified (max_diff=1.5e-8) | `receipts/step2_integration/*.json` |
| **3** | Router benchmark (24 tensors, static vs routed) | 0 regressions, 0 false positives, 0.42% SVD overhead | `receipts/step3_routing/*.json` |
| **4** | Per-block sensitivity (22 blocks, 1 at a time) | ALL 22 negligible. Max delta: block 21 at +0.47% | `receipts/step4_sensitivity/*.json` |
| **5** | Full model (all 22 blocks compressed at once) | **+0.87% perplexity at 3.99x compression** | `receipts/step5_full_model/*.json` |
| **6** | Decode latency benchmark (10-tensor sample) | Median 16.6 ms/tensor, full model ~3.7s CPU | `receipts/benchmarks/benchmark_20260310T*.json` |
| **7** | HelixLinear integration (compressed executable) | **+0.78% perplexity, 4.26x memory, 154/155 modules swapped** | `receipts/step7_helix_linear/*.json` |
| **8** | GPU viability (Quadro T2000, 4 GB) | **FITS at 38% VRAM. Dense: OOM. 97 tok/s prompt, 3.3 tok/s decode.** | `receipts/step8_gpu_viability/*.json` |

---

## Final Result (Steps 5-8)

| Metric | Value |
|--------|-------|
| Baseline perplexity | 6.1717 |
| Compressed perplexity | 6.2256 |
| **Perplexity delta** | **+0.0539 (+0.87%)** |
| **Compression ratio** | **3.99x** |
| Original size | 4.40 GB |
| Compressed size | 1.10 GB |
| Size savings | 3.30 GB (75%) |
| Tensors compressed | 156 |
| Tensors exact (norms) | 45 |
| SVD-upgraded (block 0) | 7 |
| KL divergence (mean) | 0.0095 |
| KL divergence (p99) | 0.0556 |
| Compounding factor | 1.17x (individual deltas sum to +0.046, actual is +0.054) |
| Decode latency (median) | 16.6 ms/tensor (CPU cold-decode) |
| Decode latency (max) | 57.9 ms/tensor |
| Full model decode estimate | 3.67 seconds |
| **HelixLinear PPL** | **6.2196 (+0.78%)** |
| **HelixLinear memory ratio** | **4.26x** (926 MB vs 3.95 GB dense) |
| **Modules swapped** | **154 HelixLinear, 1 nn.Linear (lm_head)** |
| **GPU VRAM (model)** | **1493 MB (38% of 4 GB)** |
| **GPU VRAM (peak)** | **2870 MB (74%)** |
| **Prompt throughput** | **97 tok/s** (512 tokens, Triton fused) |
| **Decode throughput** | **3.3 tok/s** (300 ms/token) |
| **Dense on same GPU** | **OOM** (does not fit) |

---

## Key Findings

1. **No fragile blocks.** All 22 blocks individually tolerate compression with <0.5% perplexity delta. The architecture does not depend on preserving any single critical layer.

2. **Errors compound minimally.** The compounding factor is 1.17x — nearly perfectly additive. No catastrophic interaction effects between compressed blocks.

3. **Block 0 anomaly fully neutralized.** Block 0 has pathological kurtosis (298.8 on k_proj) but the VQ+SVD(r=8) mixed decoder reduces its perplexity impact to 0.000.

4. **Activation-aware evaluation diverges from weight cosine.** Some tensors with high weight cosine (0.9997) have lower activation cosine (0.994), and vice versa. Activation-conditioned output cosine is the correct proxy for behavioral fidelity.

5. **Value distribution drives compression, not geometry.** k-means codebooks (value-space clustering) at 4x dominate every geometric/algebraic decomposition tested (FFT, SVD-only, wave/morpho, spiral). Indices are spatially independent (autocorrelation = 0.000).

6. **Compressed form IS the executable.** HelixLinear (Step 7) runs the model directly from codebook + uint8 indices + sidecar + SVD factors. No decompression step. 154/155 linear modules swapped, +0.78% perplexity, 4.26x persistent memory savings. The compressed representation is not just storage — it is the runtime format.

7. **4 GB GPU viability proven.** (Step 8) TinyLlama 1.1B FP32 does NOT fit on a Quadro T2000 (4 GB). HelixLinear loads at 1493 MB (38% VRAM), runs inference with Triton fused kernel at 97 tok/s prompt and 3.3 tok/s decode, with 2.4 GB VRAM headroom for KV cache, tools, or RAG.

---

## Limitations

- **Single model tested.** Only TinyLlama 1.1B. Results may not transfer to larger models, different architectures (Mistral, Llama 3, etc.), or models with more extreme weight distributions.

- **Small eval corpus.** 4096 tokens from WikiText-2 validation. Sufficient for ranking and delta measurement but not for publication-quality perplexity numbers. A full eval on the complete validation set (297K tokens) would take ~10 hours on CPU.

- **GPU decode is slow.** Triton fused kernel works (97 tok/s prompt, 3.3 tok/s decode on Quadro T2000) but uses outer-product accumulation (not tiled dot) due to Triton 3.2/CC7.5 gather+dot limitation. Tiled dot would be ~10-50x faster.

- **No downstream task evaluation.** Perplexity is a proxy. Actual task accuracy (MMLU, HellaSwag, ARC, etc.) not tested.

- **Fixed k=256.** Adaptive-k (proven to save 19.7% in isolation) not applied in the full-model test. The 3.99x ratio could improve to ~4.8x with sub-byte index packing.

- **Norm tensors stored exact.** 45 norm tensors (368 KB total, 0.008% of model) pass through without compression. Morpho/FFT codec exists for norms (42/45 at cos>0.9) but not applied here.

---

## What Remains Open

1. **~~Latency/throughput benchmarking.~~** DONE. CPU cold-decode: median 16.6 ms/tensor, full model ~3.7s. See `receipts/benchmarks/benchmark_20260310T204711.json`.

2. **Second model validation.** Test on a different architecture (Mistral-7B, Llama 3 8B) to verify generalization.

3. **Adaptive-k integration.** Per-tensor optimal k (64-256) with sub-byte index packing for 20% additional savings.

4. **Full eval corpus.** WikiText-2 full validation (297K tokens) and/or downstream tasks.

5. **~~GPU decode path.~~** DONE. Triton fused kernel works on Quadro T2000. Outer-product accumulation (not tiled dot). See `receipts/step8_gpu_viability/`.

---

## Reproduction

```bash
# From helix-substrate root
python tools/step5_full_model_perplexity.py --tokens 4096
```

Or use the standalone benchmark script:

```bash
python tools/benchmark_cdnav3.py
```

**Requirements:**
- Python 3.10+
- numpy, scipy, safetensors, torch, transformers, datasets
- TinyLlama FP32 model at `~/models/tinyllama_fp32/model.safetensors`

---

## Files

| File | Purpose |
|------|---------|
| `helix_substrate/tensor_policy.py` | Tensor classification + codec routing |
| `helix_substrate/cdnav3_writer.py` | Encode pipeline (VQ + sidecar + SVD) |
| `helix_substrate/cdnav3_reader.py` | Decode pipeline (streaming-capable) |
| `helix_substrate/generate_sidecars_v3.py` | Outlier detection + sidecar generation |
| `tools/step5_full_model_perplexity.py` | Full-model proof script |
| `tools/benchmark_cdnav3.py` | Standalone reproducible benchmark |
| `helix_substrate/helix_linear.py` | Drop-in nn.Linear replacement (compressed executable) |
| `tools/step7_helix_linear_integration.py` | HelixLinear integration proof script |
| `helix_substrate/triton_vq_matmul.py` | Fused VQ gather-matmul Triton kernel |
| `tools/step8_gpu_viability.py` | GPU viability benchmark |
| `PROOF_HELIX_LINEAR.md` | HelixLinear + GPU proof document |
| `BLOCK_GENOME_BLUEPRINT.md` | Architecture design document |
