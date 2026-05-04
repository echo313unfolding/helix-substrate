# HXQ Codec Ladder v0 — Frozen (2026-05-04)

Three tiers. No gaps. No speculation.

## The Ladder

| Tier | Format | bpw | Layout | Status |
|------|--------|-----|--------|--------|
| Speed | Q4_K_M | ~4.5 | Hierarchical (llama.cpp native) | Production default |
| Middle | HXQ_Q5_H | 5.5 | Hierarchical (Q5_K-style) | **Adopt** — validated |
| Quality | HXQ_AF6 | 6.25 | Flat g128 | Keep current design |

## Why these three

### Speed: Q4_K_M (existing)

- 4.5 bpw, llama.cpp native, CUDA-optimized
- 12-44 tok/s depending on hardware
- Production default for all deployed models
- No HXQ differentiation at this tier — use upstream

### Middle: HXQ_Q5_H (new — adopt Q5_K layout)

- 5.5 bpw via hierarchical super-block:
  - 256-element super-block, 8 sub-groups of 32
  - 6-bit quantized scales/mins (12 bytes packed)
  - FP16 super-block headers (4 bytes)
  - 5-bit values (160 bytes)
  - Total: 176 bytes / 256 = 5.5 bpw
- Gets g32 quality (0.999224) at g64 cost — saves 0.5 bpw free
- Hierarchy is the mechanism: +0.000342 over flat-g64 at same bpw
- Iterative importance-weighted fitting adds only +0.000079 (marginal)
- One-shot min/max with hierarchy is 95% of the benefit

### Quality: HXQ_AF6 flat g128 (keep)

- 6.25 bpw, flat per-group-128 affine quantization
- 64 levels (6-bit), min/max scaling
- Already achieves 0.999672 output cosine
- Hierarchy tested and shelved: only +0.000055 gain (below noise)
- Reason: 64 levels already have so little error that local fitting barely helps

## Shelved / Dead paths

| Path | bpw | Why dead |
|------|-----|----------|
| Q4S2 shadow | 5.3-6.4 | Q5 flat beats Q4+shadow at matched bpw |
| AF6 hierarchical | 6.5 | Marginal gain (+0.000055), not worth complexity |
| k-means / sorted-affine | 5.5 | Catastrophic (0.982) — wrong for Gaussian weights |
| Sidecar correction | any | Routing dead (ρ=-0.023), distributed error not sparse |
| Affine-4 | 4.25 | +11.37% PPL — too coarse |

## The mechanism (why hierarchy helps Q5 but not AF6)

Hierarchy's benefit = proportional to quantization error being recovered.

```
Q5 (32 levels): g128→g32 quality gap = 0.000578
  → hierarchy recovers 0.000342 of that gap (59%) at only +0.25 bpw
  → NET POSITIVE

AF6 (64 levels): g128→g32 quality gap = 0.000140
  → hierarchy recovers 0.000055 of that gap (39%) at +0.25 bpw
  → NOT WORTH IT
```

The 6-bit codec already distributes error so evenly that per-sub-group fitting
has almost nothing to recover. The 5-bit codec has enough residual error that
local scale/min fitting is genuinely useful.

## What each tier is for

| Tier | Use case |
|------|----------|
| Q4_K_M | Default inference, chat, deployed models, speed-critical |
| HXQ_Q5_H | Storage-quality middle ground, fine-tunable via LoRA, archival |
| HXQ_AF6 | Maximum fidelity, research, embedding preservation, sensitivity tests |

## Runtime / kernel status

| Format | llama.cpp | Triton | CUDA native | Status |
|--------|-----------|--------|-------------|--------|
| Q4_K_M | Native | N/A | Native | Production |
| HXQ_Q5_H | Not yet | Not yet | Not yet | **Next engineering step** |
| HXQ_AF6 | mmvq kernel (hxq-affine-type branch) | Fused gather-matmul | dp4a vec_dot | Research (50% speed gap) |

### HXQ_Q5_H runtime path

Since Q5_K already exists in llama.cpp with full CUDA/Metal/AVX2 support,
HXQ_Q5_H can either:
1. **Use Q5_K directly** — map HXQ weights to Q5_K block format at conversion time
2. **Fork with HXQ improvements** — if we find algorithmic improvements worth shipping

Current evidence says option 1 is correct: the layout IS the trick, the fitting
is marginal. Convert HXQ weights to Q5_K blocks using naive min/max, get 99.5%
of iterative quality.

## Proof receipts

| Proof | Receipt | Key number |
|-------|---------|------------|
| Q5 hierarchy works | `hxq_q5_hierarchical_proof_v0_20260504.json` | +0.000342 vs flat-g64 |
| AF6 hierarchy marginal | `hxq_af6_hierarchical_proof_v0_20260504.json` | +0.000055 (below gate) |
| Q4S2 loses to Q5 | `hxq_q4s2_vs_q5_h2h_v0_20260504.json` | Q5 0.998646 > Q4+Sh 0.997998 |
| Shadow chain full | `hxq_q4s2_*_20260504.json` (5 receipts) | Chain closed |
| AF6 quality baseline | `affine_6bit_g128_quality_gate.json` | cos > 0.999 universal |

## What "boring and coherent" means

This ladder has:
- No overlapping tiers competing at the same bpw
- No unvalidated speculative positions
- No "maybe if we add X" hedges
- Clear mechanism for why each tier exists
- Clear mechanism for why alternatives are dead
- One next engineering step (wire Q5_H to existing Q5_K runtime)
