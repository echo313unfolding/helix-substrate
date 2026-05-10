# FlowTorch Teacher-Student Distillation — Historical Finding

**Status:** ARCHIVED — weight-space reconstruction only, does not replace LoRA
**Date reviewed:** 2026-04-30
**Source receipts (SSD backup):**
- `echo-backup-2026-04-25/helix-cdc-receipts/kat_flowtorch_mistral_distill_lobe_pytorch.json`
- `echo-backup-2026-04-25/helix-cdc-receipts/kat_flowtorch_mistral_distill_sweep.json`
- `echo-backup-2026-04-25/helix-cdc-receipts/kat_flowtorch_teacher_student_toy.json`
- `echo-backup-2026-04-25/helix-cdc-receipts/kat_neworg_grow_mistral_mini.json`

## What It Did

Teacher-student weight-space distillation via FlowTorch (Dec 2025):
- Initialize random student weights
- Optimize: `loss = MSE(W_student, W_teacher)`
- Se (symbolic entropy) steering on learning rate

## Results

| Experiment | Scale | Correlation | Status |
|-----------|-------|-------------|--------|
| Toy network (16x32x8) | 4K params | -0.08 -> 0.64 | PARTIAL |
| Single lobe (4096x4096) | 16.7M params | -0.008 -> 0.9999 | PASS |
| 8-block sweep (Q/K/V/O + FFN) | 234M params | -0.005 -> 0.9952 | PASS |
| **Composed full block** | **all lobes** | **0.84 avg -> 0.24 composed** | **FAIL** |

## Why It Failed

Per-tensor weight matching does not preserve composed transformer behavior. When individually-matched lobes were assembled into a full block:
- Correlation dropped from 0.84 (per-lobe avg) to 0.24 (composed)
- Attention entropy = log(16) = uniform — no learned attention patterns
- The system's own diagnosis: "lobe_distillation_creates_new_organism_not_clone"

## Why It Does Not Replace LoRA

LoRA trains on **task loss** (behavioral). FlowTorch distillation trained on **weight MSE** (reconstruction).

| Property | FlowTorch Distill | LoRA |
|----------|-------------------|------|
| Loss function | MSE on weight matrices | Cross-entropy on task outputs |
| What it learns | Weight values | Task behavior |
| Composed behavior | Fails (0.24 correlation) | Preserves + adapts |
| Sentinel Hard20 | N/A | +15% (69 -> 84%) |

## Conclusion

1. Weight-space MSE distillation reached high per-block correlation (0.9952).
2. It failed to preserve composed transformer behavior (0.24 block correlation).
3. It does not replace LoRA for Sentinel or any task adaptation.
4. The gap was correctly identified in the original work: "stop trying to reconstruct weight values, start trying to reconstruct weight behaviors."

## Future Viable Branch (Not Built)

Behavioral distillation — KL divergence or task loss on model outputs, not MSE on weights:
- Train student to match teacher's output distribution (KL on logits)
- Or train student directly on labeled task data (cross-entropy)
- This is standard knowledge distillation (Hinton et al. 2015), not novel

HelixLinearSTE (born-compressed training) is closer to this: it trains on real task loss through the compressed representation. That path is active. FlowTorch distillation is not.
