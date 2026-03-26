# WO-RVQ-8X-01: 8x Executable Compression via Sub-Byte VQ

**Status:** IN PROGRESS
**Date:** 2026-03-24
**Depends on:** HelixLinear (PROVEN), Kurtosis Routing (PROVEN), Triton VQ kernel (PROVEN)

---

## Question

Can HelixLinear achieve 8x compression while remaining executable (no dequantization), with PPL delta under 2%?

## Math Correction (Honest Accounting)

The original framing ("two 4-bit stages packed into 1 byte = 8x") is **4x, not 8x**:
- Two 4-bit indices for ONE weight = 8 bits = 1 byte = 4x vs FP32
- Residual VQ improves *quality at 4x*, not *ratio to 8x*

**Actual path to 8x:** 4 bits per weight = 0.5 bytes/weight = 8x vs FP32.

Three executable mechanisms that achieve this:

| Mechanism | Bits/weight | Codebook | Kernel change | Quality risk |
|-----------|------------|----------|---------------|-------------|
| **4-bit scalar VQ** | 4 | k=16 (coarse) | Nibble extract + gather | HIGH — 16 centroids |
| **Grouped VQ (g=2)** | 4 | k=256, 2D entries | Gather outputs 2 values | MODERATE — vector codebook |
| **Mixed-rate** | 4-8 adaptive | k=16 or k=256 per tensor | Two kernel paths | LOW — kurtosis selects |

## Chosen Architecture: Mixed-Rate with Residual Enhancement

**Insight:** Not all tensors need 256 centroids. Kurtosis routing already identifies which tensors are sensitive.

### Rate allocation:
- **Low-kurtosis tensors (majority):** 4-bit VQ, k=16. Packed 2 per byte. **8x.**
- **High-kurtosis tensors (few):** 8-bit VQ, k=256 (current). **4x.**
- **Residual correction (optional):** For 4-bit tensors that need it, a second 4-bit stage quantizes the residual. Two nibbles packed into 1 byte. Still 8x per weight but with 16×16=256 effective levels.

### Blended ratio:
If 80% of tensors go 4-bit and 20% stay 8-bit:
- Blended = 0.8 × 8x + 0.2 × 4x = **7.2x** (codebook overhead negligible)
- With residual on 4-bit tensors: still 8x storage for those tensors, better quality

### Storage format:

```
4-bit tensor (.cdnav3):
  meta.json:      {"storage_mode": "codebook_4bit", "n_clusters": 16, ...}
  codebook.npy:   [16] float32 (64 bytes — negligible)
  indices.bin:    [rows, cols/2] uint8 (two weights packed per byte)

4-bit residual tensor (.cdnav3):
  meta.json:      {"storage_mode": "residual_vq_4bit", "n_clusters": [16, 16], ...}
  codebook.npy:   [16] float32 (coarse)
  codebook2.npy:  [16] float32 (residual)
  indices.bin:    [rows, cols/2] uint8 (high nibble = coarse, low nibble = residual)

8-bit tensor (.cdnav3):  (unchanged from current)
  codebook.npy:   [256] float32
  indices.bin:    [rows, cols] uint8
```

### Triton kernel paths:

```
4-bit scalar VQ:
  idx_byte = indices[row, col // 2]
  idx = (idx_byte >> 4) if col_even else (idx_byte & 0xF)
  w = codebook[idx]

4-bit residual VQ:
  idx_byte = indices[row, col // 2]
  coarse_idx = idx_byte >> 4
  fine_idx = idx_byte & 0xF
  w = codebook[coarse_idx] + codebook2[fine_idx]

8-bit VQ (existing):
  w = codebook[indices[row, col]]
```

## Implementation Plan

### Phase 1: 4-bit scalar VQ encoder + decoder (CPU)
- k=16 k-means encoding
- Nibble packing: two 4-bit indices per byte
- Nibble unpacking in HelixLinear forward (CPU path)
- Cosine fidelity measurement vs 8-bit

### Phase 2: Residual VQ encoder
- Stage 1: k=16 coarse codebook
- Stage 2: k=16 on residual (weight - codebook[coarse_idx])
- Pack coarse|fine into single byte
- Reconstruct = codebook[high] + codebook2[low]

### Phase 3: Mixed-rate tensor policy
- Extend `tensor_policy.py`: kurtosis threshold decides 4-bit vs 8-bit
- Low kurtosis → 4-bit (or 4-bit residual)
- High kurtosis → 8-bit + SVD sidecar (current production path)

### Phase 4: Triton kernel for 4-bit gather-matmul
- Nibble extract in kernel
- For residual: two gathers + add, still no materialization
- Benchmark vs 8-bit kernel

### Phase 5: Full model proof
- TinyLlama: encode all 154 tensors with mixed-rate policy
- Measure blended compression ratio
- PPL eval (target: <2% delta)
- VRAM measurement

## Predeclared Success Criteria

1. **4-bit VQ (k=16) encodes and reconstructs** — cosine > 0.99 on low-kurtosis tensors
2. **Residual VQ recovers fidelity** — cosine matches or exceeds 8-bit k=256 on same tensor
3. **Mixed-rate achieves >6x blended compression** on TinyLlama (154 tensors)
4. **PPL delta < 2%** on WikiText-2 (same methodology as all prior proofs)
5. **No dequantization** — gather-only kernel, weight matrix never materialized
6. **Kurtosis routing is calibration-free** — no Hessian, no calibration data needed

## What Could Kill This

- **k=16 too coarse for any tensor class** — even low-kurtosis tensors might need >16 centroids
- **Nibble packing overhead** — extract ops might slow the Triton kernel enough to lose the speed argument
- **Residual codebook doesn't help** — if residuals are uniformly distributed, second stage adds noise
- **Blended ratio stuck at 5-6x** — if too many tensors need 8-bit, the 8x headline dies

## Prior Art

| WO | Result | Relevance |
|----|--------|-----------|
| WO-GPTQ-HELIX-HYBRID-01 | PPL 313 (catastrophic) | OBS compensation incompatible with VQ |
| WO-GPTQ-HELIX-HYBRID-02 | Kurtosis wins (rho=0.25) | Hessian routing doesn't beat kurtosis |
| Step 7 (HelixLinear) | +0.78% PPL at 4x | Current 8-bit baseline |
| Kurtosis routing | Cross-architecture validated | Rate allocation signal |
