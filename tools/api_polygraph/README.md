# API Polygraph

Detect cloud provider model substitutions using local ground truth comparison.

## What it does (honestly)

Three detection layers, each requiring more infrastructure:

| Layer | Detects | Requires |
|-------|---------|----------|
| 1. Timing | Size-class swaps (3B served as 11B) | API access only |
| 2. Distribution | Family swaps (Llama served as Qwen) | API logprobs + local model fingerprint |
| 3. Sidecar-weighted | Amplifies Layer 2 on confident tokens (theoretical: untested on real API) | HXQ-compressed local model |

**Critical:** Detection requires LOCAL GROUND TRUTH. You must run the open-weights
model yourself. The tool compares what you measured locally against what the API
returned. Without the local reference, there is nothing to compare against.

## What it does NOT do

- Fingerprint models from API output alone (needs local model)
- Detect swaps without logprobs (Layer 1 timing-only fallback)
- Replace metadata-based detection (complementary, not a replacement)
- Work with closed-weight models (no local ground truth possible)

## How the sidecar weighting works (Layer 3)

The sidecar measures reconstruction error between original weights and their
HXQ-compressed representation. Low sidecar norm = the codebook captured that
weight well = the model is CONFIDENT about that weight's contribution.

```
diagnostic_weight = 1.0 / (1.0 + sidecar_mean_norm)
```

Confident predictions are most diagnostic because:
- Same model should agree on confident tokens
- Different models disagree most on tokens they're each confident about

The amplification metric: `weighted_KL / unweighted_KL`. If confident tokens
diverge MORE than uncertain tokens (amplification > 1.5x), strong swap signal.

The sidecar is a WEIGHTING TOOL for divergence tests. It tells you which tokens
to trust most in the comparison. It is not a standalone swap detector.

## Usage

```bash
# Step 1: Fingerprint local model (ground truth)
python3 -m api_polygraph fingerprint --model meta-llama/Llama-3.2-3B-Instruct

# Step 1 (HXQ mode, adds sidecar Layer 3):
python3 -m api_polygraph fingerprint --model /path/to/hxq/model --hxq

# Step 2: Probe the cloud API
python3 -m api_polygraph probe \
    --api-url https://api.deepinfra.com/v1/openai \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --api-key $DEEPINFRA_API_KEY

# Step 2 (Ollama local test):
python3 -m api_polygraph probe --model qwen2.5-coder:3b --ollama

# Step 3: Compare
python3 -m api_polygraph compare \
    --fingerprint fingerprint_llama3b.json \
    --probe api_probe_results.json
```

## Verdict grades

| Grade | Meaning |
|-------|---------|
| CONSISTENT | All layers agree: same model |
| SUSPICIOUS | Timing or distribution anomaly, not conclusive |
| SWAP_DETECTED | Multiple layers confirm different model served |

## Verification (2026-04-18)

Comparator math verified on synthetic same-model and different-model pairs:
- Same-model KL divergence: 0.000000, token match: 100%
- Different-model KL divergence: 2.989, token match: 0%
- Verdict aggregation grades correctly across all test cases

**NOT YET VERIFIED**: Against a real API that is performing substitutions.
Pat's DeepInfra evidence (helix-inference-os) shows the substitutions are real.
This tool needs to be tested against the same provider to confirm Layer 2/3
catches what his Layer 1 (metadata comparison) already catches.

## Probe corpus

20 fixed probes across 8 categories (factual, math, code, translation,
reasoning, completion, technical, format). Versioned — never change probes
after fingerprinting, or fingerprints become invalid.

## Receipt format

All outputs include WO-RECEIPT-COST-01 cost blocks:
```json
{
  "cost": {
    "wall_time_s": 12.34,
    "cpu_time_s": 11.02,
    "peak_memory_mb": 245.6,
    "python_version": "3.10.12",
    "hostname": "echo-labs",
    "timestamp_start": "2026-04-18T17:52:22",
    "timestamp_end": "2026-04-18T17:52:34"
  }
}
```

## Connection to helix-inference-os

Pat's MerkleDAG receipt system (helix-inference-os) seals `requested_model`
vs `actual_model` per call — metadata-based detection (Layer 1). This tool
adds distribution comparison (Layer 2) and sidecar-weighted divergence
(Layer 3) on top. The two systems are complementary:

- His receipts PROVE substitution happened (cryptographic evidence)
- This tool QUANTIFIES how different the served model is (statistical evidence)

## License

MIT — same as helix-substrate.

## Files

| File | Purpose |
|------|---------|
| `probe_corpus.py` | 20 fixed probes, versioned (never change after fingerprinting) |
| `fingerprint.py` | Local ground truth (direct mode + HXQ sidecar mode) |
| `api_probe.py` | Send probes to OpenAI-compatible API or Ollama |
| `compare.py` | Three-layer divergence scoring + verdict |
| `__main__.py` | CLI entry point |
