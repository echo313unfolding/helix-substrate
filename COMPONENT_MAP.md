# COMPONENT_MAP.md

> Full codebase audit — 2026-03-28.
> Source of truth for file status before HF integration or runtime changes.
> SVD routing DISABLED 2026-03-28. Plain VQ-256 is the locked codec.

## Legend

| Status | Meaning |
|--------|---------|
| **LIVE** | Active production path |
| **LIVE-UTIL** | Active utility/tool, not in inference path |
| **DEAD** | No longer referenced or needed; zero runtime cost |
| **NEEDS-UPDATE** | Has stale assumptions (SVD/kurtosis/channel_scales) |
| **RESEARCH** | Experimental/benchmark only, not shipped |

---

## 1. helix_substrate/ — Core Runtime (60 files)

### Compression & Codec

| File | Status | Description | SVD/Kurtosis Notes |
|------|--------|-------------|---------------------|
| `helix_linear.py` | **LIVE** | Drop-in nn.Linear replacement. Codebook+indices+sidecar decode. Triton fused or CPU naive. | SVD buffers loaded if present (backward compat). Kurtosis gate attachable but NEVER attached. Channel_scales LIVE for --scale-file models. Zero overhead from dead paths. |
| `triton_vq_matmul.py` | **LIVE** | Fused VQ gather-matmul kernel (v4 manifest dispatch). 4 kernel variants. Phase-aware sidecar (fused N<=16, scatter N>16). | Correctly handles None SVD params. No stale assumptions. |
| `cdnav3_writer.py` | **LIVE** | Writes CDNA v3 tensor dirs. Codebook+indices+sidecar+optional SVD. | Only writes SVD when policy.svd_residual_rank>0 (now always 0). Safe. |
| `cdnav3_reader.py` | **LIVE** | Reads CDNA v3 tensors. Reconstructs with optional SVD. | Reads historical SVD files correctly. Graceful when absent. |
| `tensor_policy.py` | **LIVE** | Tensor classification + compression policy. get_policy() routes by name/shape. | **SVD routing DISABLED (line 296-304).** Always returns base VQ-256 policy. kurtosis param in signature unused but harmless. |
| `sidecar.py` | **LIVE** | Sidecar generation (outlier corrections). | Clean. No SVD/kurtosis deps. |
| `generate_sidecars_v3.py` | **LIVE-UTIL** | Offline sidecar generation for CDNA v3 tensors. | Clean. |
| `cdna_encoder.py` | **LIVE** | Low-level VQ encoding (k-means, assignment, sidecar). | Clean. |
| `cdna_reader.py` | **LIVE** | Legacy CDNA v2 reader. | No SVD support (v2 predates it). |
| `morpho_codec.py` | **LIVE** | Morpho codec (norm compression, FFT spectral path). | Independent of SVD. Used for NORM tensors only. |
| `se.py` | **LIVE** | Spectral entropy measurement. | Diagnostic/preflight only. Not routing. |
| `convert.py` | **LIVE-UTIL** | HF safetensors to CDNA conversion. | Clean. |
| `convert_gguf.py` | **LIVE-UTIL** | GGUF format conversion. | Clean. |
| `build_manifest_v3.py` | **LIVE-UTIL** | Builds CDNA v3 manifest.json from tensor dirs. | Clean. |
| `zerocopy.py` | **LIVE** | Zero-copy GPU tensor infra (pinned host for uint8 indices). | Clean. No SVD/kurtosis. |

### Routing & Policy

| File | Status | Description | SVD/Kurtosis Notes |
|------|--------|-------------|---------------------|
| `route_policy.py` | **LIVE** | Scored route selection (VQ_ONLY vs VQ_PLUS_SVD_R8 vs EXACT). | **VQ_PLUS_SVD_R8 variant exists but NEVER selected.** kurtosis_score field unused. Dead code, harmless. |
| `route_decision.py` | **LIVE** | Pre-inference route decision (frozen at request start). | Clean. No SVD/kurtosis. |
| `route_inheritance.py` | **LIVE** | Route inheritance tracking (deps, blockers). | Clean. |
| `route_shaper.py` | **LIVE** | Route shaping policy. | Clean. |
| `routing_ledger.py` | **LIVE** | Route logging/audit. | Clean. |
| `preflight_router.py` | **LIVE** | Preflight structural analysis. | Uses Se measurement (diagnostic, not SVD routing). |
| `kurtosis_gate.py` | **DEAD** | Runtime SVD gating based on activation kurtosis. | **Entire module OBSOLETE.** Never attached in current usage. Zero overhead. |
| `helix_affine_linear.py` | **DEAD** | Int8 block-affine runtime. Killed for speed (1.0 tok/s, 3.7x slower than VQ). | Receipt: `receipts/affine_codec_audit/`. |
| `helix_linear_ste.py` | **DEAD** | Straight-through estimator variant. Research only. | Never shipped. |
| `morphsat_gate.py` | **LIVE** | Morpho saturation gate. | Independent of SVD. |

### Symbolic Control Plane

| File | Status | Description |
|------|--------|-------------|
| `lobe_scheduler.py` | **LIVE** | 6-lobe FSM scheduler. 231/231 tests. |
| `symbolic_ir.py` | **LIVE** | Symbolic intermediate representation. |
| `symbolic_executor.py` | **LIVE** | Executor for symbolic actions (6 lobes, 7 routes, 8 action types). |
| `verifier.py` | **LIVE** | Symbolic output verifier (fail-closed). |
| `lobe_context_policy.py` | **LIVE** | Lobe context selection policy. |
| `executor_registry.py` | **LIVE** | Executor type registry. |
| `fallback_chain.py` | **LIVE** | Fallback chain for failed routes. |

### Model & Session Management

| File | Status | Description |
|------|--------|-------------|
| `model_adapter.py` | **LIVE** | HF AutoModel loader + config parsing. |
| `model_manager.py` | **LIVE** | Model lifecycle (load/swap/unload). Used in local_assistant.py. |
| `basin_runtime.py` | **LIVE** | Runtime for HelixLinear backend (load, generate, receipts). No SVD assumptions. |
| `query_classifier.py` | **LIVE** | Query intent classifier (factual/coding/exact-fact). Keyword+regex, <1ms. |
| `web_fact_tool.py` | **LIVE** | DDG web fact retrieval. |
| `response_cache.py` | **LIVE** | Response caching layer. |
| `kv_prefix_cache.py` | **LIVE** | Multi-slot LRU prefix reuse for follow-up turns. |
| `session_advisor.py` | **LIVE** | Session-level advisory. |
| `session_budget.py` | **LIVE** | Token/compute budget tracking. |
| `token_accountant.py` | **LIVE** | Token usage accounting. |
| `device_utils.py` | **LIVE** | VRAM tracking, device sync. |
| `rope.py` | **LIVE** | RoPE positional encoding. |

### Streaming Inference

| File | Status | Description |
|------|--------|-------------|
| `stream_matmul.py` | **LIVE** | Streaming matrix multiplication. |
| `stream_attention.py` | **LIVE** | Streaming attention (multi-head, RoPE-aware). |
| `stream_block.py` | **LIVE** | Block-level streaming. |
| `stream_ffn.py` | **LIVE** | FFN streaming. |

### Quality & Budget Gates

| File | Status | Description |
|------|--------|-------------|
| `quality_gate.py` | **LIVE** | Quality gate for decode outputs. |
| `budget_gate.py` | **LIVE** | Compute budget gate. |
| `shaping_policy.py` | **LIVE** | Output shaping policy. |
| `transform_law.py` | **LIVE** | 4-axis state machine (route/cache/quality/budget). 44/44 tests. |
| `backtest_integrity.py` | **LIVE** | Integrity backtesting. |
| `market_realism.py` | **LIVE** | Market realism checks. |

### Other

| File | Status | Description |
|------|--------|-------------|
| `receipt.py` | **LIVE** | Receipt builder (WO-RECEIPT-COST-01 compliant). |
| `thesis_tracer.py` | **LIVE** | Thesis/claim tracing. |
| `cli.py` | **LIVE** | CLI entry point. |
| `fgip_bridge.py` | **LIVE** | Bridge to FGIP engine (Bucket B). |
| `trading_executors.py` | **LIVE** | Trading execution (Bucket B). |
| `__init__.py` | **LIVE** | Package init. |

---

## 2. tools/ — Compression, Eval, Benchmarks (254 files)

### Production Pipeline

| File | Status | Description |
|------|--------|-------------|
| `compress.py` | **LIVE** | Universal CDNA v3 compressor. Codec lock: plain VQ-256, no flags needed. Saves biases, excludes embed/lm_head. SVD comment updated 2026-03-28. |
| `eval_ppl_cpu.py` | **LIVE** | CPU PPL evaluation (WikiText-2). Bias bug FIXED: passes model= to load_cdna_factors(). |
| `calibrate.py` | **LIVE-UTIL** | Calibration data generation. Optional (not needed for plain VQ). |

### Step Proofs (All LIVE-UTIL)

| File | Description | Receipt |
|------|-------------|---------|
| `step0_activation_aware_eval.py` | Activation-aware evaluation | `receipts/step0_*/` |
| `step1_mixed_decoder.py` | Mixed decoder proof | `receipts/step1_*/` |
| `step2_codec_router.py` | Codec router proof | `receipts/step2_*/` |
| `step3_sensitivity.py` | Sensitivity analysis | `receipts/step3_*/` |
| `step4_per_block.py` | Per-block PPL proof | `receipts/step4_*/` |
| `step5_full_model_perplexity.py` | Full model PPL proof (+0.87%) | `receipts/step5_*/` |
| `step6_decode_latency.py` | Decode latency (16.6ms/tensor) | `receipts/step6_*/` |
| `step7_helix_linear_integration.py` | HelixLinear integration (+0.78% PPL) | `receipts/step7_*/` |
| `step8_gpu_viability.py` | GPU viability (T2000 4GB fit) | `receipts/step8_*/` |

### Benchmarks (RESEARCH)

| File | Description |
|------|-------------|
| `bench_cross_arch_composition.py` | Transformer+SSM composition (FALSIFIED: PPL 96K) |
| `bench_mamba_helix.py` | Mamba2 CDNA v3 compression |
| `bench_kv_compression.py` | KV cache CDNA v3 compression |
| `bench_kv_attention_fidelity.py` | KV cache attention fidelity |
| `bench_stabilization.py` | GPU stabilization benchmark |
| ~40+ other `bench_*.py` | Various research benchmarks (properly isolated) |

### Utilities (LIVE-UTIL)

| File | Description |
|------|-------------|
| `echo_probe.py` | Echo system health/map/ask |
| `local_assistant.py` | Dual-model REPL (TinyLlama+Qwen) |
| `dump_kv_cache.py` | KV cache extraction |

---

## 3. helix-online-kv/ — Compressed KV Cache (10 files)

| File | Status | Description |
|------|--------|-------------|
| `compressed_cache.py` | **LIVE** | Drop-in DynamicCache replacement. Calibrates on first 128 tokens, then VQ-assigns. Proven: 0% PPL delta, 1.9x memory at 16K. |
| `codebook.py` | **LIVE** | Scalar (1D) online VQ codebook. K-means on calibration, nearest-centroid streaming. |
| `vector_codebook.py` | **LIVE** | Vector (head_dim-D) VQ codebook. Higher fidelity than scalar. Not used in CDC-03. |
| `product_codebook.py` | **LIVE** | Product quantization (16 subspaces x 256 centroids). Critical for CDC-03 Path G. |
| `compressed_attention.py` | **LIVE** | Paths A-G reference implementations + CDC-03 (Path G hybrid). Proven: cos=0.9997, 12.5% coverage, ~0.12x compute. |
| `layer_state.py` | **LIVE** | Per-layer compression FSM (calibrating -> streaming -> aged). |
| `aging_policy.py` | **LIVE** | Tier 0/1 token aging (FP16 recent, uint8 old). LRU. |
| `config.py` | **LIVE** | OnlineKVConfig dataclass. exact_layers=[0] (layer 0 high-kurtosis sink). |
| `triton_attention.py` | **LIVE** | Fused Triton kernel for Path A (scalar VQ decompress). PoC. |
| `__init__.py` | **LIVE** | Package init. |

**SVD/Kurtosis notes:** No SVD dependencies. config.py references kurtosis only as documentation for why layer 0 is exact. Clean.

---

## 4. echo_runtime/ — Unified Runtime (7 files)

| File | Status | Description |
|------|--------|-------------|
| `config.py` | **LIVE** | EchoConfig dataclass. Model path, CDNA dir, weights/KV/attention/generation configs. YAML loader. |
| `model_wrapper.py` | **LIVE** | Integration hub. load() -> _swap_to_helix() -> _init_kv_cache() -> _patch_attention(). Wires all three compressed layers. |
| `optimize.py` | **LIVE** | One-command model optimization: load -> compress -> validate -> output CDNA v3. |
| `prove_e2e.py` | **LIVE-UTIL** | Full e2e proof with cost block. |
| `demo.py` | **LIVE-UTIL** | End-to-end demo harness. |
| `smoke_test.py` | **LIVE-UTIL** | Smoke test (load -> generate -> check). |
| `__init__.py` | **LIVE** | Package init. |

**SVD/Kurtosis notes:** No SVD dependencies. model_wrapper.py calls load_cdna_factors() which handles SVD absence gracefully.

---

## 5. SVD/Kurtosis Stale Code Summary

Everything below is **SAFE** (zero runtime cost, backward compat only) but documents where dead code lives.

| Location | Dead Content | Risk | Action |
|----------|-------------|------|--------|
| `tensor_policy.py:276` | `kurtosis` parameter in get_policy() | None (ignored) | Optional cleanup |
| `tensor_policy.py:296-303` | SVD routing comment block | None (documents why disabled) | Keep |
| `route_policy.py:53` | `VQ_PLUS_SVD_R8` enum variant | None (never selected) | Keep for compat |
| `route_policy.py:68` | `kurtosis_score` field | None (never populated) | Optional cleanup |
| `helix_linear.py:87-93` | SVD buffer registration | None (buffers empty when rank=0) | Keep for old checkpoints |
| `helix_linear.py:109-114` | Kurtosis gate attachment point | None (gate always None) | Keep |
| `helix_linear.py:217-222` | Forward() gate step | None (skipped when gate=None) | Keep |
| `helix_linear.py:543-552` | Auto-load SVD files in load_cdna_factors() | None (backward compat for old dirs) | Keep |
| `helix_linear.py:556-560` | Auto-load channel_scales.npy | **LIVE** (needed for --scale-file models) | Keep |
| `cdnav3_writer.py:220` | SVD write gate (`if rank > 0`) | None (rank always 0) | Keep |
| `kurtosis_gate.py` (entire) | Runtime kurtosis gate | None (never attached) | Mark DEPRECATED |
| `helix_affine_linear.py` (entire) | Int8 block-affine runtime | None (killed for speed) | Mark DEPRECATED |
| `helix_linear_ste.py` (entire) | STE training variant | None (research only) | Mark DEPRECATED |

---

## 6. Dependency Graph (Production Path)

```
compress.py
  -> tensor_policy.py (classify, get_policy -> always VQ-256)
  -> cdna_encoder.py (k-means, assignment)
  -> sidecar.py (outlier corrections)
  -> cdnav3_writer.py (write codebook+indices+sidecar)

eval_ppl_cpu.py
  -> helix_linear.py (load_cdna_factors, swap_to_helix)
  -> transformers (AutoModel, AutoTokenizer)
  -> datasets (WikiText-2)

helix_linear.py (inference)
  -> cdnav3_reader.py (load tensors)
  -> triton_vq_matmul.py (GPU fused path)
  -> device_utils.py (VRAM, sync)
  -> zerocopy.py (pinned host indices)

echo_runtime/model_wrapper.py
  -> helix_linear.py (weight compression)
  -> helix_online_kv/compressed_cache.py (KV compression)
  -> helix_online_kv/compressed_attention.py (CDC-03)
  -> transformers (model loading)

basin_runtime.py
  -> helix_linear.py
  -> model_adapter.py
  -> receipt.py
```

---

## 7. Models Registry — Bias Status

| Model | Path | Tensors | Bias Files | Pipeline | Status |
|-------|------|---------|------------|----------|--------|
| TinyLlama 1.1B | `~/models/tinyllama_fp32/cdnav3/` | 154 | Needs audit | Pre-bias fix | **VERIFY** |
| Qwen 1.5B-Coder | `~/models/qwen2.5-coder-1.5b-instruct/cdnav3/` | 196 | 168 | 2026-03-28 clean | OK |
| Qwen 3B-Instruct | `~/models/qwen2.5-3b-instruct/cdnav3/` | 252 | 216 | 2026-03-28 clean | OK |
| Qwen 7B-Instruct | `~/models/qwen2.5-7b-instruct/cdnav3/` | 196 | 168 | Pre-2026-03-28 | **VERIFY** |
| Mamba2 1.3B | `~/models/mamba2-1.3b/cdnav3/` | — | Needs audit | — | **VERIFY** |

---

## 8. Scaling Table (Clean Pipeline, 2026-03-28)

Per-tensor ratio is FP32→compressed (~4x for k=256). File ratio is from BF16 source model.

| Model | Dense PPL | Helix PPL | Delta | Per-Tensor Ratio | File Ratio (from BF16) | Receipt |
|-------|-----------|-----------|-------|-----------------|----------------------|---------|
| TinyLlama 1.1B | 6.1717 | 6.2196 | +0.78% | 3.99x | 4.0x (FP32 source) | step7 |
| Qwen 1.5B-Coder | 15.064 | 15.309 | +1.63% | 4.0x | 1.5x | `ppl_eval_*1.5b*_20260328T102100` |
| Qwen 3B-Instruct | 9.443 | 9.604 | +1.71% | 4.0x | 1.6x | `ppl_eval_*3b*_20260328T111417` |
| Qwen 7B-Instruct | 6.948 | 7.388 | +6.34% | 4.0x | 2.2x | `ppl_eval_*7b*_20260327T182345` |
| Qwen 14B-Instruct | OOM | 3.782 | ~+14% est | 4.0x | 3.4x | cloud benchmark |

---

*Generated 2026-03-28. Four parallel audit agents. No files modified.*
