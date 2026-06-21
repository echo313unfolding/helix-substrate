# Gauge-Only Routing

Tests whether compression-induced routing signals work without semantic
labels, tensor names, architecture names, or human-readable roles.

## Core Principle

The routing layer is intentionally non-semantic. It operates on gauges,
not words.

```
The system should not route because it understands words.
It should route because the gauges move.
```

Like an operator in a plant room watching pressure, heat, vibration, flow,
and alarms — then opening the right valve.

## Why This Matters

If the router uses tensor names, model names, or human categories, someone
can object: "You are not discovering a compression signal. You are just
using metadata."

The gauge-only ablation proves the signal is structural, not semantic
leakage.

## Gauge Vector

The blind router receives ONLY numeric features:

| Gauge | Source | What it measures |
|-------|--------|-----------------|
| rms_error | Residual | Damage magnitude |
| cosine | Residual | Reconstruction fidelity |
| kurtosis | Residual | Tail heaviness (outliers) |
| sparsity | Residual | Error concentration |
| acf_lag1 | Residual | Short-range correlation |
| acf_lag10 | Residual | Medium-range correlation |
| spectral_ratio | Residual | Frequency concentration |
| svd_rank_ratio | Residual | Rank structure |
| top10_explained | Residual | Variance concentration |
| channel_concentration | Residual | Row-wise error skew |
| structure_score | Residual | Composite structure |
| ghost_te | Encoded body | Transition entropy |
| ghost_tr | Encoded body | Transition rank |
| ghost_mo | Encoded body | Markov order |
| ghost_ac | Encoded body | Index autocorrelation |
| confidence | Derived | Signal reliability |

## Forbidden Inputs

The gauge-only router may NOT use:

- tensor_name
- model_name
- model_family
- layer_index
- tensor_role (string)
- architecture label
- words from parameter names

Hash identifiers are allowed only for receipts, not routing decisions.

## Routing Logic

```
pressure    = structure_score     → accept / probe / fallback
heat        = kurtosis + channel_concentration → outlier correction
vibration   = spectral_ratio + acf_lag10 → structure detection
flow        = bpw / codec metrics → (future: cost-aware routing)
alarm       = confidence          → conservative fallback on low signal
```

## Comparison Protocol

For each tensor x codec:

1. Compute ResidualProfile (full information)
2. Build GaugeVector (numeric features only, no names)
3. Run gauge-only router → GaugeRouteDecision
4. Run full residual router → ResidualRouteDecision
5. Compare: agree / missed fallback / false fallback

## Pass/Fail Criteria

| Metric | Pass | Fail |
|--------|------|------|
| Agreement rate | >= 85% | < 70% |
| Missed fallback rate | 0% | > 0% |
| False fallback rate | < 15% | > 30% |

Missed fallbacks are the critical safety metric. A gauge-only router that
misses a fallback the full router would have caught is unsafe. False
fallbacks waste compute but don't lose quality.

## Usage

```bash
python tools/sweep_gauge_only_routing.py
python tools/sweep_gauge_only_routing.py --output receipts/gauge_routing.jsonl
```

## Output

- `receipts/gauge_only_routing.jsonl`: Per-comparison records
- `receipts/gauge_only_routing.summary.json`: Agreement analysis + verdict

## Relationship to Literature

Tang (2025): reconstruction error as intrinsic routing signal — a gauge.
Ye et al. (2026): control signal vs content channel — gauges vs words.
Expert Choice Routing: numeric router decisions, not English.

HXQ: the compressed object is a control-plane sensor. The routing layer
reads gauges from compression artifacts and opens the right valve.

## Evidence

- `tools/sweep_gauge_only_routing.py`: ablation implementation
- `tests/test_gauge_only_routing.py`: validation
- `helix_substrate/residual_contract.py`: gauge source (residual features)
- `helix_substrate/ghost_bridge.py`: gauge source (encoded-body features)
- `helix_substrate/residual_router.py`: full router for comparison
