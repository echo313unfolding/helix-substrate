# HXQ Tensor Codec Evidence

**Status:** Receipted evidence as of 2026-05-02.

HXQ uses per-group affine quantization (group_size=128, 6-bit / 64 levels). This document collects the evidence that it works as a general tensor codec, not only for LLM weights.

---

## 1. Tested ML Architectures

| Model | Type | Key metric | Receipt |
|-------|------|------------|---------|
| Qwen2.5-3B | Transformer (LLM) | cos > 0.999, PPL +0.53% | `receipts/qwen3b_compress/` |
| SmolLM3-3B | Transformer (LLM) | cos > 0.999, PPL +1.47% | `receipts/cloud_bench/` |
| Zamba2-2.7B | Hybrid SSM+Transformer | cos > 0.999, PPL +0.89% | `receipts/mamba2_compress/` |
| all-MiniLM-L6-v2 | Sentence Transformer | cos 0.9989, rank preserved | `receipts/non_llm_proof/non_llm_proof_20260312T103756.json` |
| CLIP ViT-B/32 | Vision-Language | logits cos 0.9999 | `receipts/non_llm_proof/non_llm_proof_20260312T103756.json` |
| ResNet-18 | CNN | cos 0.9999, 100% pred match | `receipts/non_llm_proof/non_llm_proof_20260312T103756.json` |

Additional: 1024 MS MARCO embeddings via HXQ affine6, cos_min 0.999558, 1024/1024 PASS.

---

## 2. Tested Raw Tensor Distributions

Tensor size: 4096x4096. Method: per-group affine g128 6-bit. Gate: cos >= 0.998.

| Distribution | Kurtosis | Cosine | RMS | Gate |
|---|---|---|---|---|
| Mixed Gaussian | 3.01 | 0.9998 | 0.018 | PASS |
| Uniform [-1, 1] | 1.80 | 0.9999 | 0.009 | PASS |
| Heavy-tailed Cauchy | 5765 | 0.9985 | 0.613 | PASS |
| Log-normal | 33.89 | 0.9996 | 0.076 | PASS |
| Sparse (90% zeros) | 98.90 | 0.9988 | 0.015 | PASS |

All five pass. Heavy-tailed Cauchy is the hardest case (highest kurtosis, lowest cosine, highest RMS) but still clears the 0.998 gate.

Receipt: `receipts/raw_distribution_probe/hxq_raw_distribution_test_20260502T103120.json`

---

## 3. Failed Alternatives

Tested on heavy-tailed Cauchy (kurtosis 5765, 4096x4096) at the same 6-bit budget:

| Method | Global cosine | Per-group worst | RMS | Verdict |
|---|---|---|---|---|
| **Affine (current HXQ)** | **0.9985** | **0.9964** | **0.61** | **Winner** |
| Quantile-optimal | 0.9911 | 0.9150 | 1.50 | Worse |
| Mu-law companded | 0.0906 | -0.1296 | 24.16 | Catastrophic |

**Mechanism:** Cosine similarity is dominated by high-magnitude tail values. Non-uniform quantization allocates most levels to the dense center (where data clusters) and starves the tails of resolution. Since tails carry most of the signal energy, cosine drops. Uniform (affine) spacing gives proportional resolution across the full range, preserving tail magnitude.

Per-group analysis: 0 out of 131,072 groups fell below cos 0.99 with affine. Range-cosine correlation r = -0.60 (wider groups are worse, but not badly enough to fail).

Receipt: `receipts/nonuniform_ceiling_probe/nonuniform_ceiling_20260502T131054.json`

### Also tested and closed

- **Outlier sidecar for affine:** 0.02pp PPL gain for 0.20 bpw cost. Affine error is distributed, not sparse. Not worth the overhead.
- **Ternary fuzzy outlier pre-filter:** Historical ternary work was symbolic encoding (BloomCode pipeline), not numerical compression. Per-group already isolates outlier damage sufficiently.

---

## 4. Honest Framing

**What the evidence supports:**

- HXQ is calibration-free (no per-model or per-distribution tuning)
- HXQ works across tested ML architectures (transformer, SSM, hybrid, CNN, ViT, embedding)
- HXQ works across tested raw tensor distributions (Gaussian, uniform, heavy-tailed, log-normal, sparse)
- Per-group affine empirically outperforms tested non-uniform alternatives on heavy-tailed data

**What the evidence does not support:**

- Universal optimality across all possible tensor distributions
- Superiority over all quantization methods (untested: GPTQ, AWQ, QuIP# on the same distributions)
- Theoretical proof of affine optimality (empirical result only)
- Performance on distributions not tested (e.g., bimodal with distant modes, adversarial constructions)

---

## 5. Receipt Paths

```
receipts/raw_distribution_probe/
  hxq_raw_distribution_test_20260502T103120.json

receipts/nonuniform_ceiling_probe/
  nonuniform_ceiling_20260502T131054.json

receipts/non_llm_proof/
  non_llm_proof_20260312T103756.json
```

Test scripts:
```
tools/test_raw_distributions.py
tools/non_llm_proof.py
```
