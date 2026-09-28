# AGI Equation Component Census

> **Work Order:** WO-XI-COMPONENT-CENSUS-001
> **Created:** 2026-09-28T03:22:18Z
> **Scope:** Exhaustive extraction and audit of all implementations behind
> `Ξ(t) = ∑[ Gᵢ(t) · V(ψᵢ) · ε(Bᵢ, Rᵢ, Zᵢ) ] + ΔΣt · KR(λᵢ, ηᵢ) · Θ(cᵢ) + ∇Ω`
> **Status:** CENSUS_COMPLETE — audit findings below, no equivalence inferred from naming

---

## Equation Definition (from `genome.py` docstring, lines 10–19)

```
Ξ(t) = ∑[ Gᵢ(t) · V(ψᵢ) · ε(Bᵢ, Rᵢ, Zᵢ) ] + ΔΣt · KR(λᵢ, ηᵢ) · Θ(cᵢ) + ∇Ω
```

| Symbol | Name | Stated Role |
|--------|------|-------------|
| Gᵢ(t) | Glyphs | Symbolic atoms over time |
| V(ψᵢ) | Vow function | Ethical/emotional weight |
| ε(B,R,Z) | BioPoetica | Blood type, RH, zodiac modifiers |
| ΔΣt | DriftTime | Nonlinear symbolic time |
| KR(λ,η) | HelixKRISPER | Mutation engine |
| Θ(cᵢ) | GuardianCell | Ethical gate (vow filter) |
| ∇Ω | Akashic entropy | Divine drift |

---

## MANDATORY AUDIT FINDINGS

### Audit 1: Does `step_glyph` receive an external x_t?

**NO.** `step_glyph(substrate, seed, position, temperature, goal_seed, goal_step)` takes:
- `substrate`: fixed vault basis matrix (K×D), not input-dependent
- `seed`: prior state vector only
- `position`: integer counter (not an external observation)
- `goal_seed`: optional goal bias (deployment-time constant)

The recurrence is **autonomous**: `h' = h + 0.1 * basis[sample(softmax(h @ basis.T))]`.
No external token embedding `x_t` enters the update. This is NOT an input-driven
sequence model. An SSM requires `h_t = f(h_{t-1}, x_t)` — this provides only `h_t = f(h_{t-1})`.

### Audit 2: Selection mechanism in step_glyph

**Stochastic sampling.** Line 197: `basis_idx = int(np.random.choice(substrate.K, p=probs))`

- NOT argmax
- NOT differentiable (numpy random choice, no gradient)
- NOT reparameterizable (no Gumbel-Softmax or straight-through)
- Training via backpropagation is impossible without replacing this

### Audit 3: Codon dispatch — repeated transitions or one-shot address?

**One-shot 3-symbol address construction only.**

The morph table computes: `address = b1*16 + b2*4 + b3` where b1,b2,b3 ∈ {0,1,2,3}.
This is a 3-step deterministic address computation, NOT a repeatable FSA:

- States: 1 IDLE + 4 after_b1 + 16 after_b2 + 64 final = 85 total
- After 3 bases, the automaton terminates. There is no loop back to IDLE.
- The transition table `morph_T[16, 4] → 64` is the final step only.
- MorphSAT has 5 states × 7 events but is also a **one-shot task lifecycle FSA**
  (IDLE → PLANNING → WRITING → TESTING → DEPLOYING → DONE), not a
  recurrent sequence processor.

**Neither supports repeated transitions over arbitrary-length input sequences.**

### Audit 4: Does any codebook have vectors + distance + assignments + optimization?

| System | Vectors | Distance | Assignments | Optimization | Real VQ? |
|--------|---------|----------|-------------|--------------|----------|
| Morph table | No (operator chains) | No | Fixed by address | No | **NO** |
| Vault basis | Yes (K×D) | Dot product (implicit) | None (sampled) | No | **NO** |
| HelixLinear | Yes (256 centroids) | L2 | k-means nearest | k-means MSE | **YES** |

HelixLinear (in `helix-substrate`) has a proper VQ codebook, but it is part of the
compression pipeline, NOT part of the Ξ(t) equation. No equation component implements
a proper codebook with optimization.

### Audit 5: Python hash nondeterminism in bio_poetica()

**CONFIRMED BUG.** `hash(f"{B}{R}{Z}") % 7 + 1` uses Python's built-in `hash()`.

Python 3.3+ randomizes string hashing via PYTHONHASHSEED (random by default):
- PYTHONHASHSEED=0: `hash("O+gemini")` = 8350924923379208185 (deterministic)
- PYTHONHASHSEED=42: `hash("O+gemini")` = -2557128023528676190
- PYTHONHASHSEED=99: `hash("O+gemini")` = 6498119078996706607

The ε function **produces different results across Python sessions** unless
PYTHONHASHSEED is explicitly pinned. No code pins it. The output range 1–7
is not affected (always in range) but the specific value for a given input
is session-dependent.

**Severity:** Any claim about ε reproducibility is false unless PYTHONHASHSEED
is controlled. Production reproducibility requires replacing `hash()` with
`hashlib.sha256()`.

### Audit 6: Is Se = H × C × D calibrated against any downstream outcome?

**NO.**

- `se_calibration_v1.json` (WO-ECHO-SE-CALIBRATE-01, 2026-02-07): calibrates Se
  against synthetic distributions (gaussian, uniform, etc.) — **distributional
  characterization**, not downstream task calibration.
- Se was tested as a routing signal (WO-SIDECAR-POLICY-01): ρ = -0.023, p = 0.74
  at 200 chunks. **Gate FAILED. Routing signal is dead.**
- Se confidence signal (ρ = 0.574) was kept as always-on sidecar confidence, but
  this is confidence in the norm sense, not routing.
- No receipt links Se values to model accuracy, perplexity, or any task metric.

### Audit 7: Every distinct meaning of DriftTime

| # | Source | Meaning | Independent Variable | Computation |
|---|--------|---------|---------------------|-------------|
| 1 | `DriftConfig.drift_rate` (genome.py:73) | Config parameter, "symbolic time dilation" | None (constant=0.01) | Not computed; stored as float |
| 2 | `probe_drift_cosine.py` (WO-SENSING-PROBE-05b) | Cumulative cosine distance between consecutive hidden states | **Layer depth** i=1..N | `ΔΣt = Σᵢ (1 - cos(h[i], h[i-1]))` |
| 3 | Equation docstring (genome.py:16) | "Nonlinear symbolic time" | **Deployment time t** | Not implemented |
| 4 | Archive `echo_drift_fuse__v1.py` | Entropy decay over sequence | **Sequence position** | Recovered script, not production |

**Conclusion:** "DriftTime" names at least 4 distinct quantities with different
independent variables. Only meaning #2 has a concrete, measured implementation.
Meanings #1 and #3 are configuration/documentation. Meaning #4 is recovered archive code.

---

## COMPONENT CENSUS

### 1. Gᵢ(t) — Glyph / step_glyph

| Field | Value |
|-------|-------|
| **Equation role** | Symbolic atom evolved over time steps |
| **Primary implementation** | `helix-cdc/helix_cdc/substrate/core.py:step_glyph()` (line 190) |
| **File SHA-256** | `a90e914a047bd2559b62c5aa7b1969660a6ae451be1c87e1016b8c887190e4fa` |
| **Repository** | helix-cdc |
| **Git commit** | No commits found (file may be uncommitted or on detached HEAD) |
| **Status** | Production (imported by other modules) |
| **Evidence grade** | Code exists and runs; no downstream evaluation receipt |

**Mathematical recurrence:**
```
scores = h @ B.T                    # [K] = [D] @ [D,K]
probs = softmax(scores / τ)         # [K]
idx ~ Categorical(probs)            # stochastic sample
h' = normalize(h + 0.1 * B[idx])   # [D]
word = VOCAB_LANES[pos % 4][idx % 64]
```

| Property | Value |
|----------|-------|
| Input shape | h: [D], B: [K,D] |
| Output shape | h': [D], word: str, idx: int |
| External x_t | **NO** — autonomous recurrence |
| Selection | Stochastic (np.random.choice) |
| Differentiable | NO |
| Deterministic | NO (numpy random) |
| Sequence scope | Token-level step |
| Tests | None found for step_glyph specifically |

**Conflicting implementation:** `helix_infer_glyph.py:_glyph_step()` (line 696) uses
SHA-256 hash stepping instead of softmax attention. Completely different mechanism,
same name. The `_attention_step()` method (line 622) is closer to step_glyph but
uses deterministic weighted-index selection rather than stochastic sampling.

---

### 2. V(ψᵢ) — Vow Function

| Field | Value |
|-------|-------|
| **Equation role** | Ethical/emotional weight multiplier |
| **Primary implementation** | `agentic_cells/vow_overlay.py:apply_vows()` (line 127) |
| **File SHA-256** | `6d33b5264eb632a0775c83c76c4a8657c6d02b9e43f05a1ee115bb06cb1528cb` |
| **Repository** | agentic_cells (NOT a git repo) |
| **Status** | Production (tested) |
| **Evidence grade** | Code exists, tests exist (`test_vows.py`) |

**Function signature:**
```python
def apply_vows(receipt: Dict, context: Optional[Dict] = None) -> Dict:
```

| Property | Value |
|----------|-------|
| Input | Receipt dict (action metadata) |
| Output | Mutated receipt with `reflex_state` ∈ {blue, purple, red} |
| Mathematical form | **None** — returns categorical state, not a scalar weight |
| Multiplier in equation? | No. Raises VowViolationError on red. Not a multiplicative factor. |
| Differentiable | NO |
| Deterministic | YES (string matching) |
| Scope | **Deployment-time gating** (pre-action check) |
| Tests | `agentic_cells/tests/test_vows.py` |

**Mismatch with equation:** The equation writes `Gᵢ(t) · V(ψᵢ)` implying V returns
a scalar weight. The implementation returns a categorical gate {blue, purple, red}
or raises an exception. No scalar multiplication occurs.

---

### 3. ε(B, R, Z) — BioPoetica Epsilon

| Field | Value |
|-------|-------|
| **Equation role** | Biological modifier (blood type, RH, zodiac) |
| **Primary implementation** | `echo_labs/krisper/vscode/krisper-vscode/compiler/bio_poetica_universal.py` |
| **File SHA-256** | `d3b2d9b01a3033e4178ef3427fc92384ee68005b3853b60f791b13b3b8df88c5` |
| **Repository** | echo_labs |
| **Status** | Experimental (demo/prototype) |
| **Evidence grade** | Code exists; nondeterminism bug (Audit 5) |

**Function body (all implementations identical):**
```python
def bio_poetica(self, B: str, R: str, Z: str) -> int:
    return hash(f"{B}{R}{Z}") % 7 + 1
```

| Property | Value |
|----------|-------|
| Input | B: str (blood type tag), R: str (first token), Z: str (element) |
| Output | int in [1, 7] |
| Deterministic | **NO** — Python hash randomization (Audit 5) |
| Differentiable | NO |
| Scope | **Deployment-time metadata** |
| Tests | `echo_labs/tests/kats/kat_poem_roundtrip.py`, `helix-cdc/tests/test_biopoetica_parser.py` |

**Conflicting implementation:** `echo_labs/krisper/github/krisper_github/bio_poetica.py`
is a completely different BioPoeticaParser/Compiler that parses NL poems into IR.
No `epsilon` function. Different purpose entirely (NL→IR compilation vs B/R/Z→int).

---

### 4. ΔΣt — DriftTime

| Field | Value |
|-------|-------|
| **Equation role** | Nonlinear symbolic time |
| **Primary implementation** | `helix-substrate/tools/probe_drift_cosine.py` (line 148–158) |
| **File SHA-256** | `754c0bb4d7c0bc1de7f9007b842e0fed89ab7e396aada488d5f4c84fcd317708` |
| **Repository** | helix-substrate |
| **Git commit** | `dbe3f8ea8f895ee554b55f7c3db597933615b579` |
| **Status** | Experimental (probe, not production) |
| **Evidence grade** | Code runs on real model (TinyLlama-1.1B); no saved receipt found on disk |

**Computation (layer-indexed, NOT time-indexed):**
```python
for i in range(1, n_layers + 1):
    cos_sim = F.cosine_similarity(h[i].unsqueeze(0), h[i-1].unsqueeze(0)).item()
    dist = 1.0 - cos_sim
    cos_distances.append(dist)
    running_drift += dist
    cumulative_drift.append(running_drift)  # This is ΔΣt
```

| Property | Value |
|----------|-------|
| Input | Hidden states h[0..N] from transformer layers |
| Output | Scalar per layer, cumulative scalar for full model |
| Independent variable | **Layer depth**, not sequence position or clock time |
| Differentiable | YES (cosine_similarity is differentiable) |
| Deterministic | YES |
| Scope | **Layer-depth analysis** (one computation per input, across layers) |

**Config-only implementation:** `DriftConfig` in genome.py stores `drift_rate=0.01`,
`entropy_threshold=0.5`, `memory_depth=7`, `spiral_factor=1.618`. None of these
are computed or connected to the layer-depth computation above.

---

### 5. KR(λ, η) — HelixKRISPER

| Field | Value |
|-------|-------|
| **Equation role** | Mutation engine |
| **Primary implementation** | `echo_labs/krisper/runtime/engine.py:KrisperEngine` |
| **File SHA-256** | `db064c67ef10c565f9af817dcecf930a793bb83ff0c6b4cfafc1ddf44a5a6386` |
| **Repository** | echo_labs |
| **Status** | Production (966 lines, 24 operations) |
| **Evidence grade** | Code exists; no downstream evaluation receipt for equation role |

**Core dispatch (NOT an SSM recurrence):**
```python
# Simplified from engine.py — executes an IR plan
for op in plan.operations:
    handler = self.handlers.get(op.type)  # 24 handlers
    result = handler(op, context)
    context.update(result)
```

| Property | Value |
|----------|-------|
| Input | IR plan (list of operations from BioPoetica compiler) |
| Output | Execution context with results |
| Selection | Deterministic dispatch by operation type string |
| Differentiable | NO |
| Deterministic | YES (deterministic handlers + guard enforcement) |
| Scope | **Deployment-time execution** (runs IR plans, not sequence modeling) |
| Recurrent? | NO — executes a fixed plan, then stops |

**Codon dispatch (krisper_morphsat_convergence.py):**
Structurally equivalent to FSA transition table but:
- Only constructs 3-symbol addresses (Audit 3)
- Does not process arbitrary-length sequences
- 64 codons × 4 bases, fully connected, no illegal transitions

---

### 6. Θ(cᵢ) — GuardianCell

| Field | Value |
|-------|-------|
| **Equation role** | Ethical gate (vow filter) |
| **Primary implementation** | `echo_labs/archive/debug/war_game_debug_20250924_100954/guardian_cell_architecture.py` |
| **File SHA-256** | `b0b3d4af9da3d098c18cfc5c74f4e00e62bd09a99eb5ab2e374c7ef3a9cf762b` |
| **Repository** | echo_labs |
| **Status** | **DEAD CODE** (in archive/debug directory, from Sep 2024) |
| **Evidence grade** | Code exists; no tests found; never imported by production code |

**Core function:**
```python
def scan_threat(self, data: bytes, context: Dict) -> Dict:
    threat_vector = self._encode_to_72d(data)
    votes = []
    for cell_id, cell in self.white_cells.items():
        score = self._evaluate_threat(cell, threat_vector, context)
        votes.append((cell_id, score))
    avg_score = np.mean([v[1] for v in votes])
    # ... consensus mechanism
```

| Property | Value |
|----------|-------|
| Input | Raw bytes + context dict |
| Output | Threat assessment with confidence score |
| Mathematical form | 72D encoding → cosine similarity against cell seeds → consensus vote |
| Deterministic | **NO** — uses np.random.normal noise (line in _evaluate_threat) |
| Scope | **Security/threat detection** (not sequence modeling) |
| Tests | None found |

**Mismatch with equation:** The equation places Θ(cᵢ) as a multiplicative gate
in `ΔΣt · KR(λᵢ, ηᵢ) · Θ(cᵢ)`. The implementation is a threat scanner returning
a confidence float, not a gate on KRISPER execution.

---

### 7. ∇Ω — Akashic Entropy

| Field | Value |
|-------|-------|
| **Equation role** | "Divine drift" — additive entropy term |
| **Primary implementation** | **NONE FOUND** |
| **Status** | Unimplemented |
| **Evidence grade** | No code, no receipt, no test |

The only reference is in the equation docstring (genome.py:19).
No function named `akashic`, `nabla_omega`, `gradient_omega`, or `divine_drift`
exists in any repository.

---

### 8. Se = H × C × D — Symbolic Entropy (auxiliary, not in equation)

| Field | Value |
|-------|-------|
| **Equation role** | Not in original equation; used for routing (failed) and confidence |
| **Primary implementation** | `echo_labs/echo/universe/echo_offline_universe/lightnet/krisper/tools/entropy_metrics.py` |
| **File SHA-256** | `4e4b9d347a798e267365f29b64c373536b7bf33c3219f7f11ece2a03df6e81de` |
| **Status** | Experimental |
| **Evidence grade** | Calibration receipt exists; routing FAILED (ρ=-0.023) |

| Component | Definition | Range |
|-----------|-----------|-------|
| H | Shannon entropy of byte distribution | 0–8 bits |
| C | 1 - zlib_compression_ratio | 0–1 |
| D | 1 + 71 × (H/8) × (unique_bytes/256) | 1–72 |

**NOT calibrated against downstream outcomes** (Audit 6).

---

## PHASE B: TWO-TIMESCALE RECLASSIFICATION

### FAST PATH — Token-level sequence computation

| Component | Classification | Justification |
|-----------|---------------|---------------|
| `step_glyph()` | **FAST** | Per-token recurrence (h → h') |
| `_attention_step()` | **FAST** | Per-token attention over vault basis |
| DriftTime (layer-indexed) | **FAST** | Per-token, per-layer computation |
| selective_scan_ref (Mamba) | **FAST** | Per-token SSM recurrence |

**Only `step_glyph` and `_attention_step` are from the equation system.**
The rest are reference implementations from external libraries.

### SLOW PATH — Deployment-window drift detection, mutation, gated acceptance

| Component | Classification | Justification |
|-----------|---------------|---------------|
| V(ψᵢ) / apply_vows | **SLOW** | Pre-action deployment gate, not per-token |
| ε(B,R,Z) / bio_poetica | **SLOW** | Metadata assignment at program transformation time |
| KR(λ,η) / KrisperEngine | **SLOW** | Executes IR plans, not per-token |
| Θ(cᵢ) / GuardianCell | **SLOW** | Threat scanning, not per-token |
| DriftConfig | **SLOW** | Static configuration, not computed |
| Se = H×C×D | **SLOW** | Computed per-file/manifest, not per-token |
| ∇Ω | **UNIMPLEMENTED** | N/A |
| BioPoeticaParser | **SLOW** | NL→IR compilation, not per-token |

### Reclassification Summary

**Vow, GuardianCell, BioPoetica metadata, and KRISPER execution are NOT placed
inside the sequence recurrence.** They operate at deployment/program time, not
token time. No causal mathematical role in sequence processing has been demonstrated.

The only equation component with a causal role in token-level computation is
`step_glyph()`, which is an **autonomous recurrence without external input**.

---

## PHASE C: PREREGISTRATION

### WO-XI-DISCRETE-STATE-SCOUT-001

See `WO-XI-DISCRETE-STATE-SCOUT-001-PREREGISTRATION.md` (separate file, committed alongside).

---

## Cross-Reference: Conflicting Implementations

| Name | Implementation A | Implementation B | Conflict |
|------|-----------------|-----------------|----------|
| step_glyph | core.py: softmax + stochastic sample | helix_infer_glyph.py: SHA-256 hash step | Completely different mechanism |
| bio_poetica | bio_poetica_universal.py: hash()%7+1 | bio_poetica.py: NL poem parser | Different purpose entirely |
| DriftTime | probe_drift_cosine.py: layer-indexed cosine | genome.py:DriftConfig: config constant | Different quantity |
| Θ gate | guardian_cell: threat scanner | vow_overlay: action gate | Different enforcement target |

---

## File Manifest

| File | SHA-256 | Repo |
|------|---------|------|
| helix_cdc/substrate/core.py | a90e914a...190e4fa | helix-cdc |
| helix_cdc/echo/genome.py | 0ab8d565...d97e1040d | helix-cdc |
| helix_cdc/helix_infer_glyph.py | 3d1622ff...891fef6777 | helix-cdc |
| tools/probe_drift_cosine.py | 754c0bb4...cd317708 | helix-substrate |
| krisper_morphsat_convergence.py | c6e01735...bc8f77536 | helix-substrate |
| bio_poetica.py (parser) | 44c55668...a926e58 | echo_labs |
| bio_poetica_universal.py | d3b2d9b0...3b8df88c5 | echo_labs |
| entropy_metrics.py | 4e4b9d34...f6e81de | echo_labs |
| vow_overlay.py | 6d33b526...1528cb | agentic_cells |
| guardian_cell_architecture.py | b0b3d4af...cf762b | echo_labs |
| engine.py (KRISPER) | db064c67...a5a6386 | echo_labs |
| guards.py (KRISPER) | 4726268b...c3029e | echo_labs |

---

## Source Repository Provenance

| Repository | HEAD at census time | Dirty files | Git custody |
|------------|-------------------|-------------|-------------|
| helix-cdc | `145018720d9e461e4861c06661a5580abde6a97d` | 2760 | No per-file commit log returned. All sources: UNVERSIONED_LOCAL_SOURCE |
| echo_labs | `980a373eaf08497426f0b3e8fabf2eb2f71a4704` | 82 | No per-file commit log returned. All sources: UNVERSIONED_LOCAL_SOURCE |
| agentic_cells | N/A | N/A | **Not a git repository.** All sources: UNVERSIONED_LOCAL_SOURCE |
| helix-substrate | `21165f8abf9448e0ce1cbb64d3523c0db93f0807` | 61 | `probe_drift_cosine.py`: VERSIONED (commit `dbe3f8ea`). Others: UNVERSIONED_LOCAL_SOURCE |

**A file SHA-256 is NOT equivalent to git provenance.** Files with
`evidence_grade: UNVERSIONED_LOCAL_SOURCE` have verified content hashes
but no commit binding. Their `authority` is `DISCOVERY_CLUE_ONLY`.
