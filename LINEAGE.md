# LINEAGE.md — HelixCode Origin to Production Code Mapping

Independent derivation of vector quantization and state-space model architectures
from biological first principles. Code-to-code mappings with dates.

## Timeline

| Date | Phase | Key Event |
|------|-------|-----------|
| Sep 2025 | Bio-prototype | CrystalVaultDNACompressor, KrisperSystem, BioPoetica interpreter |
| Oct 2025 | Formalization | `export_morph_table.py`: 64 codons, codon addresses, inflation ceiling |
| Nov 2025 | Architecture | `seed_morph_runtime.py`, `guardian_cell.py`, `krisper_cell.py` |
| Dec 2025 | Integration | FlowTorch engine with embedded MorphTable + MorphSAT classes |
| Mar 2026 | Production | CDNA v3, HelixLinear, MorphSAT standalone, Symbolic Control Plane |

## 1. EchoGlyph64 Palette → VQ Codebook (CDNA v3)

**Origin (mid-2025):** Hand-built a 64-symbol palette mapping 6-bit entropy blocks to glyphs.

```python
# BioPoetica Deep Dive (ChatGPT archive)
PALETTE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abcdefghijklmnopqrstuvwxyz") + ...
chunks = [binary[i:i+6] for i in range(0, len(binary), 6)]
glyph = PALETTE[index % 64]  # index → value lookup
```

**Production (Mar 2026):** k-means VQ codebook in CDNA v3.

```python
# helix_substrate/cdnav3_writer.py
codebook = self._build_kmeans_codebook(flat, n_clusters)  # k=256 centroids
indices = self._chunked_assign(flat, codebook)            # nearest-neighbor lookup
# helix_substrate/helix_linear.py
tensor = codebook[indices]  # same pattern: index → value
```

**Structural equivalence:** Fixed-size lookup table mapping integer indices to scalar values.
A 64-glyph palette is a k=64 codebook. k-means learning automates what the glyph palette
did by hand. The biological intuition (DNA codons as an addressing scheme) turned out to be
architecturally identical to vector quantization.

---

## 2. Codon Table → VQ Codebook Builder

**Origin (Sep-Oct 2025):**

```python
# echo_labs/crystal/demos/crystal_vault_64_cubed_decompressor.py
def _build_codon_table():  # 64 codons from 4^3 (ATCG triplets)

# repos/echo-box/ops/export_morph_table.py
address = base1*16 + base2*4 + base3  # 64 codon addresses
non_inflation_ceiling = 10  # MiB hard limit (budget constraint)
```

**Production:** `helix_substrate/k_allocator.py` — budget-constrained codebook allocation.
The inflation ceiling became `target_ratio`. The codon table became the codebook. The
64-entry dispatch table became a 256-entry VQ codebook. Same pattern: input → table lookup
→ output, with a hard budget ceiling.

---

## 3. GuardianCell → MorphSAT FSAGate

**Origin (Nov 2025):**

```python
# repos/echo-box/helix_cdc/pssh/guardian_cell.py
class GuardianCell:
    def check(self, state, action) -> GuardianResult(legal, score, violations):
        ...  # halt on violation, structured rejection

    @classmethod
    def from_preset(cls, name): ...  # load from built-in preset
    @classmethod
    def from_spec(cls, spec): ...    # load from JSON spec
```

**Production (Mar 2026):**

```python
# morphsat/morphsat/core.py
class MorphSATGate:
    def step(self, event) -> (state, legal, action):
        ...  # halt on violation, structured rejection

    @classmethod
    def from_preset(cls, name): ...  # WO-07: ported from GuardianCell
    @classmethod
    def from_spec(cls, spec): ...    # WO-07: ported from GuardianCell
    @classmethod
    def from_json(cls, path): ...    # WO-07: file-based loading
```

**Structural equivalence:** Gate that checks legality before allowing execution. Returns
structured result with violation details. Fail-closed policy (block on unknown).
The `from_preset`/`from_spec` constructors were ported directly (WO-07, 2026-03-28).

**What changed:** Domain-specific constraint checker (puzzle.hanoi) became a universal
finite-state automaton with configurable states/events/transitions. Guardian vows became
a policy layer (`GUARDIAN_BLOCKED`) above FSA legality.

---

## 4. KRISPERcell.propose() → MorphSATGate.propose()

**Origin (Nov 2025):**

```python
# repos/echo-box/helix_cdc/pssh/krisper_cell.py
class KRISPERcell:
    def propose(self, state, goal, guardian, max_candidates=32):
        raw_actions = _generate_actions(state)
        candidates = []
        for act in raw_actions:
            result = guardian.check(state, act)
            if not result.legal:
                continue  # filtered by guardian
            candidates.append(CandidateAction(action=act, guardian=result, cost=1.0))
        return candidates
```

**Production (Mar 2026):**

```python
# morphsat/morphsat/core.py
class MorphSATGate:
    def propose(self, max_candidates=3) -> List[CandidateTransition]:
        for ev in range(n_events):
            if T[state, ev] == -1: continue    # FSA illegal
            if (state, ev) in guardian: continue  # guardian blocked
            candidates.append(CandidateTransition(event=ev, cost=...))
        return sorted(candidates, key=cost)[:max_candidates]

    def step_or_propose(self, event, max_candidates=3):
        state, legal, action = self.step(event)
        if legal: return state, legal, action, []
        return state, legal, action, self.propose(max_candidates)
```

**What this enables:** MorphSAT goes from "enforcement only" (gate says no) to
"enforcement + steering" (gate says no and proposes ranked alternatives). This is the
difference between a constraint checker and a constrained generation system.
Ported directly (WO-08, 2026-03-28).

---

## 5. seed_morph_forward() = Hand-Crafted SSM

**Origin (Nov 2025):**

```python
# repos/echo-box/tools/seed_morph_runtime.py
def seed_morph_forward(seed, morph_table, sat_lane):
    h = initial_phase  # hidden state
    for token in input:
        ops = morph_table[codon(token)]           # A: state transition (morph table lookup)
        h = apply_ops_to_phase(h, ops)            # h' = A(h) + B(x)  (arctan2 input proj)
        output = sin(h)                           # C: readout (sin projection)
        output = sat_lane.constrain(output)       # hard constraint enforcement
```

**Mamba SSM equivalent:**

```
h' = A @ h + B @ x    # state update (learned A, B)
y  = C @ h'            # readout (learned C)
```

**Mapping:** h = phase angle = hidden state. A = morph table = state transition matrix.
B = arctan2 = input projection. C = sin = output projection. This was built from DNA
codon semantics (2025) before knowing what SSMs were (Mamba published Dec 2023, not
encountered until early 2026).

**What MorphSAT adds that no SSM has:** Hard symbolic transition constraints via
finite-state automata. SSMs learn soft transitions; MorphSAT enforces hard ones.
Results: S3 grammar 100% vs 17%, multi-counter generalization 100% vs 9%,
Python lexer 99.9% vs 94%.

---

## 6. FlowTorch MorphSAT → MorphSAT Package

**Origin (Dec 2025):**

```python
# repos/echo-box/runtimes/flowtorch/engine.py
class MorphTable:   # maps sequences to compressed representations
class MorphSAT:     # SAT constraint solver for next-token masking
    def _allowed_set(last_token, pos_mod4, V):  # position-modular adjacency
    def mask_logits():   # soft penalty for disallowed tokens
    def temperature():   # cosine annealing T0 → Tmin
```

**Production (Mar 2026):**

```python
# morphsat/morphsat/token.py
class MorphSATScorer:
    def _allowed_set(last_token, pos_mod4, vocab_size):  # same signature
    def mask_scores():   # soft penalty (renamed from mask_logits)
    def temperature():   # same cosine annealing
```

**Direct port.** The FlowTorch engine's MorphSAT class was extracted, cleaned up, and
given a 4-lane semantic structure (ENTITY→ACTION→QUALITY→RELATION). The `_allowed_set`
function with `prev+pos` mode carries over unchanged. Token adjacency scoring is the
formalized version of the original entropy-validated sequence scoring.

---

## 7. .hxz Vault → HelixLinear

**Origin (mid-2025, ChatGPT archive):**

> "each capsule is in effect a computational gene — a self-contained piece of
> algorithmic DNA that 'expresses' its functionality when injected into a compatible
> execution environment (lobe)"

> ".hxz Vaults: Executable, compressed, mutation-ready symbolic runtime capsules"

**Production (Mar 2026):** `helix_substrate/helix_linear.py`

The `.cdnav3` directory IS the `.hxz vault`. It's compressed (4.0x), executable
without decompression (`codebook[indices]` runs directly), and "expresses" when
injected into a PyTorch model via `swap_to_helix()`. The biological metaphor of a
computational gene that self-expresses in a compatible host is architecturally exact.

---

## 8. fold_view(block=16) → Block-Wise Streaming Decode

**Origin (mid-2025):**

```python
# ChatGPT archive (echo_gold_excerpts)
def fold_view(glyphs: str, block=16):
    for i in range(0, len(glyphs), block):
        chunk = glyphs[i:i+block]
        ...
```

**Production:** `helix_substrate/cdnav3_reader.py:reconstruct_block(start_row, end_row)`

Fixed-size block processing through a lookup table. The original processed 16-glyph
blocks; the production system processes configurable row blocks. The "never fully
decompress" concept from the early runtime decoder became HelixLinear's design:
compressed form IS the executable, tensors decode block-by-block on demand.

---

## 9. FlowTorch Lobe Routing → Lobe Scheduler + Query Classifier

**Origin (mid-2025):**

> "leftbrain/ → logical, math, precision models"
> "rightbrain/ → creative, generative, language models"

```
flowtorch-router: Central braid fusion engine
krisper-daemon: Entropy scoring, drift triggers
```

**Production (Mar 2026):**
- `helix_substrate/lobe_scheduler.py` — 6 lobes, 7 routes, fail-closed (231/231 tests)
- `helix_substrate/query_classifier.py` — coding→Qwen, factual→TinyLlama

The FlowTorch router dispatching to specialized brain lobes became the lobe scheduler
and the query classifier. The left/right brain split (logical vs creative) became the
coding/factual model routing in the dual-model local stack.

---

## 10. Stigmergy + Pheromone Board (Not Yet Ported)

**Origin (Nov 2025):**

```python
# repos/echo-box/helix_cdc/world/pheromone_board.py
class PheromoneBoard:
    def deposit(location, intensity, tag): ...
    def read(location): ...
    def evaporate(rate): ...
```

Used in `experiments/stigmergy_compression_demo.py` for multi-agent compression routing.

**Status:** Not ported. The routing function was absorbed by the lobe scheduler. The
pheromone board pattern (deposit/read/evaporate) would be relevant for distributed
inference coordination but is not needed for the current single-box architecture.

---

## What Survived, What Didn't

| Concept | Survived? | Notes |
|---------|-----------|-------|
| Codon table / glyph palette | **YES** → VQ codebook | Core of CDNA v3 |
| Guardian constraint gate | **YES** → MorphSAT FSAGate | Generalized from domain-specific to universal FSA |
| KRISPERcell propose | **YES** → MorphSATGate.propose() | Ported WO-08 (2026-03-28) |
| seed_morph_forward (SSM) | **YES** → paper contribution | Independent SSM derivation from bio first principles |
| .hxz capsule (compressed-native) | **YES** → HelixLinear | Compressed form IS the executable |
| Block-wise decode | **YES** → reconstruct_block() | Fixed-size block processing through lookup |
| Lobe routing | **YES** → lobe_scheduler + query_classifier | Specialized execution domains |
| Token adjacency scoring | **YES** → MorphSATScorer | Formalized with 4-lane structure |
| Crystal vault DNA compression | **NO** | Disproven (fingerprints, not encoding) |
| Fibonacci/golden-ratio scheduling | **NO** | Falsified (geometry irrelevant for scalar VQ) |
| Wave PDE morpho codec | **NO** | Disproven (wave contributes nothing vs k-means) |
| 487B regeneration | **NO** | Disproven (seeds are PRG, not encoder) |
| Helix64 glyph hierarchy | **NO** | Abandoned (VQ codebook is simpler and better) |
| Pheromone board | **NO** | Absorbed by lobe scheduler; single-box doesn't need it |

---

## The Independent Derivation Claim

Two architectures were independently derived from biological first principles before
encountering them in the ML literature:

1. **Vector Quantization** — derived from DNA codon tables (64-entry lookup mapping
   triplet indices to amino acid values). Encountered k-means VQ later; recognized the
   structural equivalence and adopted it with k=256 centroids.

2. **State-Space Models** — derived from morph table phase dynamics
   (`seed_morph_forward()` with h=phase, A=morph table, B=arctan2, C=sin readout).
   Encountered Mamba/S4 later; recognized the SSM equivalence.

Both derivations are documented in timestamped code (`repos/echo-box/`, Sep-Dec 2025)
and ChatGPT conversation archives (`chatgpt_book/`, `chatgpt_for_claude/`).

The novel contribution beyond independent derivation: **hard symbolic transition
constraints (MorphSAT FSAGate)** on top of SSM dynamics. No production SSM has this.
The constraint layer comes from the GuardianCell vow enforcement concept, which has no
analogue in the SSM literature.
