# WO-XI-DISCRETE-STATE-SCOUT-001: Preregistration

> **Status:** PREREGISTERED — awaiting review before execution
> **Created:** 2026-09-28T03:25:48Z
> **Scope:** Bounded mechanism scout (cannot establish "better than Mamba")
> **Type:** MECHANISM_CLAIM_ONLY

---

## Hypothesis

At matched parameter count, an explicit finite-state transition side-channel
improves length-extrapolated state tracking over a positive-decay continuous
recurrence.

## Conditions

| ID | Name | Description |
|----|------|-------------|
| A | Positive-decay Mamba reference | Standard Mamba SSM: `x_t = exp(Δ·A) x_{t-1} + Δ·B u_t`, A diagonal negative (guaranteed decay) |
| B | Negative/complex-transition ref | SSM with A containing negative real or complex eigenvalues (oscillatory/growing modes) |
| C | Learned finite-state channel | Discrete state s_t ∈ {1..K}, learned transition T[s, input_class] → s'. Continuous hidden h augmented with discrete s. |
| D | Fixed codon transition topology | Same as C but transition table initialized from KRISPER 64-codon morph table (no learning of transition structure) |
| E | Continuous + learned discrete hybrid | Full Mamba recurrence + parallel learned discrete state channel, outputs concatenated before projection |

## Tasks

| Task | Description | State-tracking complexity |
|------|-------------|--------------------------|
| Parity | Running XOR of binary input sequence | 2-state DFA |
| Modulo-3 counting | Running sum mod 3 | 3-state DFA |
| Held-out random DFA | Random 8-state DFA generated per run, train on examples, test on held-out sequences | Arbitrary DFA |
| Permutation composition | Compose permutations from a fixed group, report final permutation class | Group theory |
| Length extrapolation | All tasks above: train on length L, test on 4L | Generalization |

## Controls

- **Seeds:** 5 fixed seeds: [42, 137, 256, 1337, 31415]
- **Parameter budget:** Matched within 5% across all conditions (report exact counts)
- **Optimizer:** AdamW, lr=1e-3, weight_decay=0.01
- **Training tokens:** 500K tokens per task (fixed across conditions)
- **Training steps:** 5000 steps, batch size 64, sequence length 128
- **Test lengths:** [128 (train), 256, 512] for length extrapolation
- **Shuffled-transition negative control:** Condition D with random permutation of
  transition table rows (destroys structure, preserves parameter count)
- **State-label permutation:** For conditions C/D/E, randomly permute state labels
  and verify results are invariant (topology, not labels, should matter)
- **No task-specific codon semantics:** Codon symbols assigned to input classes
  via fixed hash, not by domain knowledge

## Metrics

| Metric | Unit | How measured |
|--------|------|-------------|
| Exact accuracy | % | Hard match on final state/output at each position |
| Cross-entropy loss | nats | Standard log-loss on output distribution |
| Throughput | tokens/sec | Wall-clock, averaged over 3 runs |
| Peak memory | MB | `torch.cuda.max_memory_allocated()` or CPU RSS |
| Length extrapolation accuracy | % | Accuracy on 4× longer sequences than training |

## Success Criteria (ALL required to justify scaling)

1. Hybrid (E) beats both continuous references (A, B) on **at least 3 of 4 tasks**
2. Improvement survives state-label permutation (topology, not labels)
3. Improvement survives 4× length extrapolation
4. No more than 10% throughput regression vs condition A
5. All-seed effect: improvement holds for all 5 seeds with pre-defined CI:
   mean(hybrid - baseline) > 0 with 95% bootstrap CI excluding zero

## Failure Criteria (ANY triggers negative result)

1. Advantage disappears under state-label relabeling
2. Only the codon-matched task (if any) improves
3. Learned finite states (C) perform no better than continuous references (A, B)
4. Result depends on one seed (other 4 show no effect)
5. Throughput overhead exceeds the accuracy gain (Pareto-dominated)
6. Fixed codon topology (D) equals or beats learned topology (C) — would
   indicate the discrete channel is just extra parameters, not structure

## Architecture Specifications

### Condition A: Positive-Decay Mamba Reference

```python
# Standard Mamba selective scan
# A: diagonal, negative real (positive decay guaranteed)
# B, C: input-dependent projections
# Δ: input-dependent discretization step
x_t = exp(Δ_t * A) * x_{t-1} + Δ_t * B_t * u_t
y_t = C_t @ x_t + D * u_t
```

- State dimension: 16
- Model dimension: 64
- Layers: 2
- Total params: ~50K (target, adjust to match)

### Condition B: Negative/Complex-Transition Reference

Same as A but:
- A initialized with complex eigenvalues (HiPPO-LegS style)
- Allows oscillatory modes, not purely decaying

### Condition C: Learned Finite-State Channel

```python
# Continuous: same as A
x_t = exp(Δ_t * A) * x_{t-1} + Δ_t * B_t * u_t

# Discrete: learned transition table
# K discrete states, input quantized to Q classes
input_class = quantize(u_t, Q)  # Q = 4 or 8
s_t = T_learned[s_{t-1}, input_class]  # lookup + straight-through gradient
s_embed = embedding(s_t)  # [discrete_dim]

# Combined output
y_t = proj([C_t @ x_t; s_embed]) + D * u_t
```

- Discrete states K: 16
- Input quantization Q: 8
- Discrete embedding dim: 16
- Transition table: learned via Gumbel-Softmax or straight-through

### Condition D: Fixed Codon Transition Topology

Same as C but:
- Transition table initialized from KRISPER morph table (64 codons)
- Transition structure frozen (only embeddings learned)
- Codon→input mapping via `hashlib.sha256(input_class_bytes).digest()[0] % 64`

### Condition E: Continuous + Learned Discrete Hybrid

Full condition A recurrence + full condition C discrete channel.
Outputs concatenated before final projection. Parameter budget shared.

## Implementation Plan

1. Create `research/agi_equation/scout/` directory
2. `models.py`: All 5 conditions as PyTorch modules
3. `tasks.py`: 4 task generators (parity, mod3, random DFA, permutation)
4. `train.py`: Training loop with all controls
5. `evaluate.py`: Evaluation including length extrapolation
6. `run_scout.py`: Full experiment runner (5 seeds × 5 conditions × 4 tasks)
7. `analyze.py`: Statistical analysis and receipt generation

## Hardware

- **Primary:** T2000 (bounded duration, mechanism scout)
- **Fallback:** RunPod 3090 if wall-clock exceeds 2 hours on T2000
- Estimated: ~50K params × 5000 steps × 5 seeds × 5 conditions = ~30 min on T2000

## What This Scout CANNOT Establish

- "Better than Mamba" — requires real NLP benchmarks, real model scale
- Language modeling quality — these are toy state-tracking tasks
- Scaling laws — single small parameter budget
- Practical utility — mechanism only, not application

It can establish: whether discrete state channels mechanistically improve
length-extrapolated state tracking at matched parameter count.

---

## Pre-Execution Checklist

- [ ] Census document committed
- [ ] This preregistration committed
- [ ] Both SHA-256 hashes recorded below
- [ ] User review received
- [ ] No code changes to preregistration after commit

## Frozen Hashes

Hashes are of the corrected artifacts in helix-substrate, computed after Commit A.
See `XI_SCOUT_FREEZE_MANIFEST.json` for the binding.
