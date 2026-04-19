# WO-SIDECAR-POLICY-01

**Priority:** High
**Sequence:** First of two. WO-SIDECAR-FASTPATH-01 follows if this gate passes.
**Authored:** 2026-04-07
**Target runner:** T3 (local T2000 box)

---

## Claim to receipt

> Thresholded application of sidecar corrections defines a measurable
> speed/quality operating frontier. Routed sidecar application is faster
> than always-on and preserves most of the quality benefit.

## Honesty rule — READ THIS FIRST

This WO proves the **policy frontier**, NOT full compute elimination from
skipping sidecar construction. Sidecar norm is measured from the realized
correction during this benchmark. The fast path (cheap proxy → skip
construction entirely) is WO-SIDECAR-FASTPATH-01 and is explicitly out of
scope here.

If the benchmark reports speed gain from skipping construction in this WO,
that is a bug or a misreport. The only legitimate speed delta here comes
from skipping the `output += sidecar_correction` step, not the construction
of `sidecar_correction` itself.

---

## Frozen parameters

### Model
**`zamba2-1.2b-helix`** — HXQ `from_pretrained` path.

Rationale:
1. The ρ=0.574 sidecar signal was measured on Zamba2-1.2B Mamba in_proj
   layers (see `sidecar-confidence-proven.md`). Same model keeps the
   signal chain intact.
2. The 2.7B runs via GGUF/llama.cpp, which has zero HXQ sidecar
   instrumentation. `self._last_sidecar_norm` does not exist in that
   runtime.
3. 1.2B-HXQ via `from_pretrained` is the only config where
   `HelixLinear` sidecar instrumentation can be exercised on T2000 without
   displacing the benchmark harness.

**Do not substitute models. Do not run on 2.7B. Do not run on GGUF path.**

### Dataset
- wikitext-2-raw, test split
- fixed chunk count: 200 chunks
- fixed max length: 512 tokens
- fixed prompt order across all modes

### Seeds
- `torch.manual_seed(42)`
- `numpy.random.seed(42)`
- `random.seed(42)`
- `torch.use_deterministic_algorithms(True)` if supported

### Isolation
**Each mode runs as a fresh subprocess.** Do not reuse one interpreter.
Thresholds are read in `HelixLinear.__init__`; in-process env-var swap is
a silent-fail trap.

### Hardware
- Quadro T2000 4GB
- `OMP_NUM_THREADS=8`
- Log nvidia-smi before each mode

---

## Required HelixLinear instrumentation

Before running the benchmark, wire these into `HelixLinear.forward()`:

```python
# In __init__:
self._sidecar_mode = os.environ.get("HELIX_SIDECAR_MODE", "always_on")
self._sidecar_threshold = float(os.environ.get("HELIX_SIDECAR_THRESHOLD", "0.0"))
self._sidecar_norms = []
self._sidecar_applied_count = 0
self._sidecar_skipped_count = 0
self._last_sidecar_norm = 0.0

# In forward(), after computing sidecar_correction:
if self.has_sidecar:
    sidecar_correction = self._compute_sidecar(...)
    sidecar_norm = sidecar_correction.norm().item()
    self._last_sidecar_norm = sidecar_norm
    self._sidecar_norms.append(sidecar_norm)

    if self._sidecar_mode == "always_on":
        output = output + sidecar_correction
        self._sidecar_applied_count += 1
    elif self._sidecar_mode == "always_off":
        self._sidecar_skipped_count += 1
    elif self._sidecar_mode == "threshold":
        if sidecar_norm > self._sidecar_threshold:
            output = output + sidecar_correction
            self._sidecar_applied_count += 1
        else:
            self._sidecar_skipped_count += 1
```

NOTE: construction is NOT skipped in this WO. Only application is gated.
This is deliberate. See honesty rule above.

---

## Modes

| Mode | `HELIX_SIDECAR_MODE` | `HELIX_SIDECAR_THRESHOLD` |
|------|----------------------|---------------------------|
| A    | always_on            | —                         |
| B    | always_off           | —                         |
| C1   | threshold            | 0.10                      |
| C2   | threshold            | 0.15                      |
| C3   | threshold            | 0.20                      |

---

## Phase A — policy frontier

For each mode, collect:

**Quality**
- PPL on wikitext-2-raw test
- mean per-chunk cross-entropy

**Performance**
- wall time total
- wall time per chunk
- tok/s
- peak VRAM (MiB)

**Routing**
- sidecar apply count
- sidecar skip count
- apply rate = applied / (applied + skipped)
- mean sidecar norm
- p50 sidecar norm
- p95 sidecar norm

**Deltas vs A (always_on)**
- quality delta %
- speed gain %

**Deltas vs B (always_off)**
- speed recovered fraction % = (routed_time - B_time) / (A_time - B_time)

---

## Phase B — oracle usefulness analysis

This is the piece that closes the causal loop. Without it, reviewers can
say the threshold result is accidental.

For a fixed subset of 50 chunks:

1. Run each chunk twice: once sidecar-on, once sidecar-off
2. Record per-chunk quality delta (ce_off - ce_on)
3. Record per-chunk mean sidecar norm (measured during the sidecar-on run)
4. Compute Pearson correlation between mean sidecar norm and quality
   delta across chunks

**Required output:** scatter plot data + correlation coefficient.

Interpretation target:
- ρ > 0.3 → signal is actionable (bigger norm predicts bigger benefit)
- ρ < 0.1 → threshold result is luck, frontier is noise
- 0.1 ≤ ρ ≤ 0.3 → weak signal, report honestly

---

## Gates

Tiered. Report which tier you hit.

- **Pass tier 1 (weak):** routed mode faster than always_on AND quality
  delta < 5% vs always_on
- **Pass tier 2 (moderate):** routed mode recovers ≥ 20% of the speed gap
  between always_on and always_off
- **Strong receipt:** routed mode recovers ≥ 50% of the speed gap with
  quality delta < 5% vs always_on

Phase B correlation threshold:
- ρ ≥ 0.3 is required for any tier to count as a real signal. A tier 2
  pass with ρ < 0.1 is reported as "frontier exists but is not driven by
  sidecar norm — investigate other causes."

---

## Output

Write receipt to:

```
~/helix-substrate/receipts/sidecar_routing/sidecar_policy_<ISO8601>.json
```

Schema (required fields):

```json
{
  "wo_id": "WO-SIDECAR-POLICY-01",
  "model": "zamba2-1.2b-helix",
  "dataset": "wikitext-2-raw",
  "n_chunks": 200,
  "seeds": {"torch": 42, "numpy": 42, "random": 42},
  "modes": {
    "A_always_on":    { "ppl": ..., "wall_time_s": ..., "tok_s": ..., "peak_vram_mib": ..., "apply_count": ..., "skip_count": ..., "apply_rate": ..., "mean_norm": ..., "p50_norm": ..., "p95_norm": ... },
    "B_always_off":   { ... },
    "C1_t_0.10":      { ... },
    "C2_t_0.15":      { ... },
    "C3_t_0.20":      { ... }
  },
  "deltas_vs_A": {
    "C1": { "quality_delta_pct": ..., "speed_gain_pct": ... },
    "C2": { ... },
    "C3": { ... }
  },
  "recovered_speed_fraction": {
    "C1": ..., "C2": ..., "C3": ...
  },
  "phase_b": {
    "n_chunks_analyzed": 50,
    "correlation_norm_vs_benefit": ...,
    "p_value": ...,
    "best_threshold_by_pareto": ...
  },
  "gate_result": "tier_1 | tier_2 | strong | fail",
  "conclusion": "one-paragraph plain text",
  "cost": {
    "wall_time_s": ...,
    "cpu_time_s": ...,
    "peak_memory_mb": ...,
    "python_version": ...,
    "hostname": ...,
    "timestamp_start": ...,
    "timestamp_end": ...
  }
}
```

The `cost` block is WO-RECEIPT-COST-01 compliant and is non-negotiable.

---

## Definition of done

- [ ] `HelixLinear` instrumentation landed and tested
- [ ] 5 modes completed on same fixed prompt set, subprocess-isolated
- [ ] Phase B correlation computed on 50-chunk subset
- [ ] Receipt written with all required fields
- [ ] `gate_result` recorded honestly
- [ ] One-paragraph conclusion: frontier exists / exists but not
      norm-driven / does not exist

---

## Out of scope

- Skipping sidecar construction (that is WO-SIDECAR-FASTPATH-01)
- Multi-model generalization (1.2B only)
- Transformer-only models (SSM-dominant only)
- Any change to the default `HELIX_SIDECAR_MODE` (default stays `always_on`)

---

## Follow-on: WO-SIDECAR-FASTPATH-01

Only run after this WO passes at least tier 1 with ρ ≥ 0.3.

Goal: replace realized-sidecar-norm gating with a cheaper proxy available
BEFORE materializing the sidecar correction, so sidecar construction
itself can be skipped. Candidate proxies:
- input activation statistics (kurtosis, norm, rank estimate)
- pre-sidecar codebook residual magnitude
- cached per-layer Se score

That WO will produce the actual compute savings receipt. This one only
proves the frontier is real.
