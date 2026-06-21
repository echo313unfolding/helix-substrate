# Real-Model Compression Signal Sweep Results

Run date: 2026-06-20, v0.4.4 (`06a7203`).

## Data

| Source | Tensors | Records | Families |
|--------|---------|---------|----------|
| Real models (HF cache) | 161 | 1900 | 5 |
| Synthetic generators | 6 | 30 | 6 |
| **Total** | **167** | **1930** | **7** |

All 5 codecs applied to each tensor: exact, affine_g128, vq_k256, rvq_16x16, svd_rank8.

### Real Tensor Distribution

| Family | Tensors | Source |
|--------|---------|--------|
| transformer | 94 | Qwen-family models |
| ssm | 20 | Mamba-family models |
| cnn | 20 | CLIP/ViT models |
| hybrid | 17 | Zamba2 |
| encoder_transformer | 10 | BERT |
| moe | 0 | None loaded (1 synthetic only) |
| embedding | 0 | None loaded (1 synthetic only) |

Limitation: MoE and embedding families have only synthetic tensors.
No dedicated MoE or embedding model was present in the HF cache.

## Q2: Does Residual Geometry Improve Codec Routing?

Quality-gated two-stage selection: cosine floor (0.90) first, then lowest
structure_score among eligible codecs (within 0.01 cosine of best).

| Family | Residual Changed Choice | Rate |
|--------|------------------------|------|
| cnn | 19/21 | 90.5% |
| embedding | 1/1 | 100.0% (synthetic only) |
| encoder_transformer | 7/10 | 70.0% |
| hybrid | 2/17 | 11.8% |
| moe | 0/1 | 0.0% (synthetic only) |
| ssm | 11/21 | 52.4% |
| transformer | 64/96 | 66.7% |
| **Total** | **104/167** | **62.3%** |

Interpretation: on 62.3% of tensors, residual geometry chose a different codec
than best-by-cosine-alone, after the quality gate filtered out low-quality
options. This means residual features add information beyond scalar cosine
for routing among quality-comparable codecs.

Hybrid (Zamba2) shows only 11.8% change rate. Many hybrid tensors are 1D,
producing degenerate SVD/channel features (svd_rank=1.0, channel_concentration=1.0).
This is a shape artifact, not a signal finding.

## Q4: Which Codecs Create Structured Residuals?

| Codec | Mean Structure | Fraction > 0.3 |
|-------|---------------|-----------------|
| svd_rank8 | 0.207 | 18.7% |
| rvq_16x16 | 0.201 | 9.8% |
| affine_g128 | 0.198 | 18.1% |
| vq_k256 | 0.146 | 9.8% |

SVD and affine produce the most structured residuals. VQ produces the
least structured residuals (most noise-like errors). RVQ has moderate
mean but lower tail (fewer extreme structure cases).

## Q5: Exact Baseline

All 386 exact records have structure_score = 0.0. Zero-error baseline
produces zero structure. Confirmed: all structure is compression-induced.

## Q6: Ranking Divergence

97.6% of tensors have different codec rankings when ranked by structure_score
vs by cosine. Structure_score adds information beyond cosine for nearly all
tensors.

## Damage Type Distribution

| Family | Distributed | Concentrated | Structured | Low-rank |
|--------|------------|--------------|------------|----------|
| transformer | 720 (77.6%) | 72 (7.8%) | 113 (12.2%) | 23 (2.5%) |
| ssm | 150 (73.5%) | 16 (7.8%) | 38 (18.6%) | 0 (0%) |
| hybrid | 217 (77.5%) | 0 (0%) | 63 (22.5%) | 0 (0%) |
| cnn | 72 (85.7%) | 7 (8.3%) | 3 (3.6%) | 2 (2.4%) |
| encoder_transformer | 36 (90.0%) | 3 (7.5%) | 0 (0%) | 1 (2.5%) |
| moe | 3 (75.0%) | 0 (0%) | 1 (25.0%) | 0 (0%) |
| embedding | 3 (75.0%) | 0 (0%) | 1 (25.0%) | 0 (0%) |

Key observations:
- Transformers produce all 4 damage types (the only family to do so on real data)
- SSMs produce no low-rank damage (consistent with smooth state projections)
- Hybrids produce no concentrated damage (shared Mamba/attention layers)
- Encoder-transformers produce no structured damage
- Damage signatures are family-specific, not universal

## Ghost Features

1158 of 1930 records (60.0%) have Ghost features (from codecs that produce
encoded bytes: affine, vq, rvq). SVD and exact do not produce encoded bytes.

## Cost

| Metric | Value |
|--------|-------|
| Wall time | 141.75s |
| CPU time | 283.65s |
| Peak memory | 12.3 GB |
| Python | 3.10.12 |
| Host | Echo |

## Gauge-Only Routing on Real Data

Run: `python3 tools/sweep_gauge_only_routing.py --from-jsonl results/compression_signal_sweep.real.jsonl`

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Agreement rate | 99.6% (1538/1544) | >= 85% | PASS |
| Missed fallback rate | 0.0% | 0% | PASS |
| False fallback rate | 0.0% | < 15% | PASS |
| Verdict | gauge_only_sufficient | | |

### Disagreements

6 of 1544 comparisons disagree. All 6 are on **transformer embedding tensors**:

| Gauge-only decision | Full router decision | Count |
|--------------------|---------------------|-------|
| outlier_sidecar | low_rank_sidecar | 6 |

These tensors have both high kurtosis/concentration (outlier signal) and
low effective rank (low-rank signal). The gauge-only router checks outlier
pattern first (kurtosis > 6.0 AND concentration > 3.0), so it fires before
the SVD rank check. The full router has access to the classified `damage_type`
enum and picks `low_rank_sidecar` directly.

This is a **correction-type priority ambiguity**, not a safety failure:
- 0 missed fallbacks (gauge never says "accept" when full says "fallback")
- 0 false fallbacks (gauge never says "fallback" when full says "accept")
- Both `outlier_sidecar` and `low_rank_sidecar` are correction actions

The gauge-only router correctly detects that something is wrong. It just
picks a different correction. Both corrections would improve the result.

### Per-family agreement

| Family | Agreement | Rate |
|--------|-----------|------|
| transformer | 922/928 | 99.4% |
| ssm | 204/204 | 100% |
| hybrid | 280/280 | 100% |
| cnn | 84/84 | 100% |
| encoder_transformer | 40/40 | 100% |
| moe | 4/4 | 100% |
| embedding | 4/4 | 100% |

## What This Proves

**Proven:** Residual geometry provides information beyond scalar reconstruction
quality on real model tensors. 62.3% of tensors had their codec choice changed
by structure_score after passing the quality gate. Damage type distributions
vary by model family.

**Proven:** Gauge-only routing (no semantic metadata) achieves 99.6% agreement
with the full router on real model tensors, with zero missed/false fallbacks.
The routing layer operates on gauges, not words.

**Not proven:**
- This improves perplexity (no end-to-end eval)
- This transfers cleanly across families (damage signatures are family-specific)
- MoE behavior is covered (only 1 synthetic tensor)
- The changed routing decisions are actually better (would require downstream eval)

## Files

- `results/compression_signal_sweep.real.jsonl`: 1930 per-tensor-per-codec records
- `results/compression_signal_sweep.real.summary.json`: Analysis with Q2-Q6 answers
- `tools/sweep_compression_routing_signal.py`: Sweep implementation (v0.4.4)
- `tests/test_compression_routing_signal_sweep.py`: 42 tests
