# HXQ Hydra Router

**Status:** SPEC
**Date:** 2026-04-30
**Receipts backing this design:**
- `receipts/hxq_mixed_lowbit_probe/mixed_lowbit_20260430T190238.json`
- `receipts/hxq_affine4_probe/affine4_probe_20260430T182909.json`
- `receipts/router_quality_sweep.json` (7-model sweep)
- `receipts/affine_board_2026-04-24.json` (Qwen frozen board)
- `receipts/affine_board_zamba2_2026-04-25.json` (Zamba2 frozen board)

## Core Concept

HXQ is not one quantization method. It is a family of codec heads sharing a common profiler trunk, routed per tensor.

```
                    ┌──────────────────────┐
                    │   Tensor Profiler    │
                    │  (shared trunk)      │
                    │                      │
                    │  kurtosis            │
                    │  cosine probes       │
                    │  layer_index         │
                    │  tensor_type         │
                    │  shape / n_params    │
                    │  max_abs_error       │
                    │  sensitivity est.    │
                    └─────────┬────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         ┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐
         │  exact  │   │  affine6  │   │  affine5  │
         └─────────┘   └───────────┘   └───────────┘
              │               │               │
         ┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐
         │ affine4 │   │  affine3  │   │  sidecar  │
         └─────────┘   └───────────┘   └───────────┘
                              │
                        ┌─────▼─────┐
                        │ rvq (rsv) │
                        └───────────┘
```

## Why Not One Head

Uniform affine4 and affine3 are dead globally:

| Strategy | avg bpw | PPL delta | Verdict |
|----------|---------|-----------|---------|
| affine6_g128 | 6.25 | +0.64% | PASS |
| affine4_g128 | 4.25 | +11.37% | FAIL |
| size_target (routed 5/6) | 5.25 | +2.39% | PARTIAL |

Source: `affine4_probe_20260430T182909.json`, `mixed_lowbit_20260430T190238.json`

But **routed low-bit works**. The size_target policy (152 tensors at affine5, 2 fragile at affine6) hit +2.39% PPL at 5.25 bpw. The router saves what uniform quantization destroys.

## Codec Heads

| Head | Bits | bpw (g128) | When |
|------|------|------------|------|
| `exact` | 16/32 | 16/32 | embed, lm_head, norms, fragile if budget allows |
| `affine6` | 6 | 6.25 | default safe path, high-kurtosis attention, early q/k |
| `affine5` | 5 | 5.25 | most tensors when cosine >= 0.998 and not high-risk |
| `affine4` | 4 | 4.25 | only when cosine gate >= 0.999 AND kurtosis below threshold |
| `affine3` | 3 | 3.25 | never default; only if cosine gate passes (rare) |
| `sidecar_vq` | sparse | variable | reserved for VQ/outlier-heavy codecs only — NOT used for uniform affine (see below) |
| `rvq` | variable | variable | reserved for future per-group VQ rescue |

## Profiler Trunk

Per-tensor features computed before routing:

```python
@dataclass
class TensorProfile:
    tensor_name: str
    shape: tuple[int, int]
    layer_index: int
    tensor_type: str          # q_proj, k_proj, gate_proj, etc.
    n_params: int
    kurtosis: float
    std: float
    # Cosine similarity under each head (probed once)
    affine6_cosine: float
    affine5_cosine: float
    affine4_cosine: float
    affine3_cosine: float
    # Error metrics at affine6 (reference head)
    max_abs_error: float
    mean_abs_error: float
```

These features exist today in the mixed-lowbit probe output. The router formalizes them.

## Routing Policies

### 1. `quality_first`
Minimize quality loss. Prefer affine6/exact. Sidecar on fragile tensors.

Rules:
- embed/lm_head/norm -> exact
- all others -> affine6
- if affine6_cosine < 0.999 -> exact or affine6+sidecar

Expected: ~6.25 bpw, <+1% PPL

### 2. `edge_balanced`
Balance size and quality for edge deployment (T2000).

Rules:
- embed/lm_head/norm -> exact (unless flagged safe)
- if affine5_cosine >= 0.998 AND kurtosis < threshold -> affine5
- if early attention (layer < 3) OR high kurtosis -> affine6
- fallback -> affine6

Expected: ~5.5 bpw, <+2.5% PPL

### 3. `size_target`
Hit an average bpw target. No sidecar — affine error is distributed, not sparse.

Rules:
- embed/lm_head/norm -> exact (always)
- if affine5_cosine >= 0.998 -> affine5
- else -> affine6 (fragile fallback)

Expected: ~5.25 bpw (152/154 at affine5, 2 fragile at affine6 on TinyLlama)

### 4. `experimental_lowbit`
Research mode. Try affine4/3 only when tensor probes pass. No sidecar rescue.

Rules:
- embed/lm_head/norm -> exact
- if affine3_cosine >= 0.998 AND kurtosis < 5.0 -> affine3 (extremely rare)
- if affine4_cosine >= 0.999 AND kurtosis < threshold -> affine4
- high kurtosis -> affine6 (no sidecar — affine error is distributed)
- default -> affine5

Expected: ~4.5-5.0 bpw, quality depends on model

## Output: CompressionPlan

```json
{
  "model": "TinyLlama-1.1B",
  "policy": "edge_balanced",
  "avg_bpw": 5.42,
  "n_tensors": 154,
  "plan": [
    {
      "tensor": "model.layers.0.self_attn.q_proj.weight",
      "head": "affine6",
      "reason": ["high_kurtosis", "early_attention"],
      "bpw": 6.25,
      "expected_cosine": 0.9992,
      "sidecar_budget": 0.0,
      "fallback_head": "exact"
    },
    {
      "tensor": "model.layers.5.mlp.gate_proj.weight",
      "head": "affine5",
      "reason": ["cosine_pass", "mlp_tensor"],
      "bpw": 5.25,
      "expected_cosine": 0.9987,
      "sidecar_budget": 0.0,
      "fallback_head": "affine6"
    }
  ]
}
```

## Sidecar — VQ/Outlier Only (NOT for Affine)

Sidecar is a sparse residual correction stored as HXZO format. It is
**reserved for VQ and outlier-heavy codecs only**. It is NOT used for
uniform affine heads.

**Why:** Uniform affine quantization distributes error evenly across all
positions. Sidecar works when a small number of outlier positions dominate
error (as in VQ). With affine, there are no dominant outliers to patch.

**Sidecar probe (2026-04-30):** 0.5% sidecar on affine5 gives 0.02pp PPL
gain for 0.20 bpw cost. DEAD for affine.
Receipt: `receipts/hxq_mixed_lowbit_sidecar/sidecar_20260430T192842.json`

- Sparse residual correction stored as HXZO format (delta-varint positions + fp16 values)
- Applied only when codec_family is VQ or residual_type is sparse_outlier
- Budget: percentage of tensor params stored as exact outliers
- Existing infrastructure: `helix_substrate/sidecar.py` (HXZO v2)

## Routing Rules (formalized)

```
R1: embed/lm_head/norm -> exact UNLESS policy explicitly allows
R2: affine5_cosine >= 0.998 AND tensor NOT high-risk -> affine5
R3: affine4_cosine >= 0.999 AND kurtosis < threshold -> affine4
R4: high kurtosis OR early q/k projection -> affine6 or affine6+sidecar
R5: affine3 ONLY if cosine gate passes; NEVER default
R6: sidecar selected ONLY for VQ/outlier codecs, NEVER for uniform affine heads
R7: rvq reserved for future; not routable in v0
```

"High kurtosis" threshold: calibrate per-architecture from sweep data. Initial: kurtosis > 50 for attention, > 20 for MLP (from TinyLlama probe).

## What This Does NOT Do

- Does NOT implement new kernels. Kernel dispatch (Triton affine group matmul, cuBLAS, mmvq) is separate.
- Does NOT replace the existing quant router at `tools/router/`. That router maps models to hardware. This router maps tensors to codec heads within a single model.
- Does NOT make claims about compression ratios beyond what receipts show.

## Relationship to Existing Code

| Existing | Role | Hydra Relationship |
|----------|------|--------------------|
| `tools/router/quant_router.py` | Model -> hardware routing | Hydra is per-tensor, lives below this |
| `helix_substrate/kurtosis_gate.py` | Runtime activation kurtosis | Hydra uses weight kurtosis at compress-time |
| `helix_substrate/sidecar.py` | HXZO outlier format | Hydra's sidecar head writes this format |
| `helix_substrate/hf_quantizer.py` | HF integration | Hydra feeds CompressionPlan into this |
| `tools/bench_hxq_mixed_lowbit_probe.py` | Mixed-bit probe | Hydra formalizes the policies this probe tested |
