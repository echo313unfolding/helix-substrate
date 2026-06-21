# Cross-Model Compression-Signal Sweep

Tests whether compression-induced routing signals are LLM-specific,
neural-weight-specific, codec-specific, or a broader property of lossy
encoding under heterogeneous structure.

## Hypothesis

Lossy compression creates measurable failure modes. Those failure modes
can be used as routing signals. This is especially true when:

1. The input has structure
2. The codec has an inductive bias
3. The compression is strong enough to reveal mismatch
4. The residual is not pure noise
5. The routing system can act conservatively

## Research Questions

| # | Question | Method |
|---|----------|--------|
| Q1 | Can residual features predict tensor role across model families? | Damage-type distribution per family |
| Q2 | Can they predict which codec wins? | Best-by-cosine vs best-by-structure agreement |
| Q3 | Do features transfer from one model family to another? | Cross-family damage patterns |
| Q4 | Which codecs create useful routing signals? | Structure score distribution per codec |
| Q5 | Are signals stronger in lossy vs lossless? | Exact baseline comparison |
| Q6 | Do residual features add info beyond cosine/error? | Ranking divergence analysis |

## Experiment Design

### Model Families

Synthetic generators cover five families. Real model loading is optional:

- Dense Transformer (MLP with mild outliers)
- SSM/Mamba (smooth, low-frequency state projections)
- MoE (sparse expert FFN)
- CNN/ViT (spatial/frequency patterns)
- Embedding (clustered rows)
- Attention (low-rank QKV)

### Codec Families

| Codec | What it tests | bpw |
|-------|---------------|-----|
| exact | Zero-error baseline | 32.0 |
| affine_g128 | Block scale/offset tolerance | ~6.25 |
| vq_k256 | Cluster/codebook friendliness | ~8.0 |
| rvq_16x16 | Staged residual repair | ~8.0 |
| svd_rank8 | Low-rank approximation | varies |

### Signals Measured

Per codec result:

**Scalar metrics:** cosine, rms_error, max_abs_error, bits_per_weight

**Residual profile (12 features):** kurtosis, sparsity, acf_lag1, acf_lag10,
spectral_ratio, svd_rank_ratio, top10_explained, channel_concentration,
structure_score, damage_type

**Ghost features (4, when encoded bytes available):** transition_entropy,
transition_rank, markov_order, index_autocorr

## Possible Outcomes

**Strong result:** Across multiple families, residual geometry predicts codec
suitability better than scalar reconstruction metrics alone.

**Stronger if Ghost transfers:** Encoded-body features provide
pre-decompression routing signals that generalize across model families.

**Partial result:** Compression-induced routing signals are not universal in
form, but recur when structured tensors are compressed by codecs with
mismatched inductive biases.

**Null result:** Residual features do not add information beyond cosine/error
magnitude. Structure score and cosine produce identical codec rankings.

## Usage

```bash
# Synthetic only (no model downloads)
python tools/sweep_compression_routing_signal.py --synthetic-only

# With real models
python tools/sweep_compression_routing_signal.py --models-dir /path/to/safetensors/

# Custom output
python tools/sweep_compression_routing_signal.py --synthetic-only --output receipts/sweep.jsonl
```

## Output

- `receipts/compression_signal_sweep.jsonl`: One record per tensor x codec
- `receipts/compression_signal_sweep.summary.json`: Analysis with Q2-Q6 answers

## Relationship to Prior Work

See `docs/RELATED_WORK_COMPRESSION_ROUTING.md` for literature positioning.

HXQ does not claim to invent compression-aware routing. The contribution is
a cross-codec routing stack that combines encoded-body pre-routing with
residual-damage post-routing. This sweep tests whether that signal generalizes.

## Evidence

- `tools/sweep_compression_routing_signal.py`: sweep implementation
- `tests/test_compression_routing_signal_sweep.py`: validation
- Ghost Bridge: `helix_substrate/ghost_bridge.py`
- Residual Contract: `helix_substrate/residual_contract.py`
- Residual Router: `helix_substrate/residual_router.py`
- Hydra Router: `helix_substrate/hydra_router.py`
