# WO-CPU-FORWARD-BUG: HelixLinear _forward_naive produces systematically worse results than GPU path

**Filed:** 2026-03-30
**Severity:** High — all CPU-evaluated PPL numbers are suspect
**Component:** `helix_substrate/helix_linear.py` → `_forward_naive()`

## Summary

The CPU tiled forward path (`_forward_naive`) in HelixLinear produces systematically worse perplexity than the GPU Triton fused path (`_forward_fused`). The error compounds across layers — deeper models show larger quality gaps.

## Evidence

| Model | Layers | CPU FP32 Delta | GPU BF16 Delta | Gap |
|-------|--------|---------------|---------------|-----|
| Zamba2-2.7B | 54 | +37.45% | **+6.59%** | 30.86pp |
| Zamba2-1.2B | 38 | +2.90% | +5.61% | -2.71pp* |
| Mamba2-1.3B | 48 | +8.0% | +7.94% | 0.06pp |

*Zamba2-1.2B shows the opposite pattern — CPU delta is lower. This is confounded by FP32-vs-BF16 dtype difference (dense baselines: 5.46 FP32 vs 7.06 BF16).

**Clear case:** Zamba2-2.7B at 54 layers shows a 30pp gap that is unambiguously a CPU forward path bug. The dense CPU baseline was also higher (8.00 CPU FP32 vs 5.33 GPU BF16), but the helix degradation is disproportionate (+37.45% vs +6.59%).

## Root Cause Hypothesis

`_forward_naive()` (line ~490 in helix_linear.py) uses a tiled matmul approach: `codebook[indices] @ x_chunk` processed in row tiles. Potential issues:

1. **FP32 accumulation precision** — tiling changes the order of floating-point additions vs the fused kernel's approach
2. **Sidecar application** — the naive path may apply sidecar corrections differently than the Triton path
3. **Tile boundary effects** — row-tiled processing may introduce discontinuities

The error compounds across layers because each HelixLinear's output feeds the next layer's input, so small per-layer errors multiply.

## Impact

- All CPU-evaluated PPL numbers in model cards are potentially inflated
- The `eval_ppl_cpu.py` tool hardcodes `device = "cpu"` and `torch.float32`
- All cloud pipeline runs (ssm_compress_pipeline.py Stage 3) used CPU eval

## Models Affected

Every model evaluated with `eval_ppl_cpu.py`:
- TinyLlama-1.1B (+0.78% — needs GPU re-eval)
- Qwen2.5-3B (+0.69% — needs GPU re-eval)
- Qwen2.5-Coder-1.5B (+1.73% — needs GPU re-eval)
- Qwen2.5-Coder-3B (+1.92% — needs GPU re-eval)
- All pipeline-evaluated models

## Fix Plan

1. **Immediate:** Add `--device cuda --dtype bfloat16` flags to `eval_ppl_cpu.py` (rename to `eval_ppl.py`)
2. **Root cause:** Diff `_forward_naive` vs `_forward_fused` codepaths to find the numerical divergence
3. **Test:** Write a unit test that compares naive vs fused output on the same input, assert max absolute error < threshold
4. **Re-eval:** Re-run all model PPL evals on GPU with BF16

## Receipts

- `receipts/cloud_bench/zamba2_2.7b_gpu_ppl_20260330.json` — GPU vs CPU comparison
- `receipts/pipeline/zamba2-2.7b-instruct_20260330T003524.json` — original CPU pipeline result
