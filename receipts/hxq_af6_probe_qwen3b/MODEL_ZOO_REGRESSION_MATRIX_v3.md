# HXQ Model Zoo Regression Matrix

> Purpose: Map every compressed artifact to its codec configuration, encoder provenance,
> runtime dispatch, and baseline measurements. For any proposed codec change, this matrix
> identifies which artifacts need compatibility checks vs quality re-evaluation.
>
> Status: v3 RECONCILED — receipt reconciliation 2026-10-27 applied.
> Prior: v2 CORRECTED 2026-10-20 (6 misassociations fixed per review).
> v3 corrections: 4 source-binding errors fixed per READ-ONLY reconciliation.
> See Section 11 for reconciliation details and corrected metric bindings.

---

## 1. Architecture Families and Integration Surfaces

Each family exercises a different combination of encoder, codec, and runtime path.
A regression suite must cover all families that are affected by a change.

| Family | Representative models | Codec config | Encoder path | Runtime dispatch (supported) | GGUF path? |
|--------|----------------------|-------------|--------------|-----------------|------------|
| Dense transformer (small) | TinyLlama-1.1B | scalar VQ k=256 | tools/compress.py → CDNAv3Writer | d=1: fused supported, buffered available | No |
| Dense transformer (medium, d=2) | Qwen2.5-Coder-3B-Instruct | **2D VQ k=256** (codebook [256,2]) | tools/compress.py (universal) | d=2: vq2d fused supported, buffered available | Yes (AF6, separate artifact) |
| Dense transformer (large) | Qwen2.5-7B-Instruct | scalar VQ k=256 | tools/compress.py | d=1: fused supported, buffered available | Yes (AF6) |
| Hybrid MoE (Granite) | Granite-4.0-H-Micro | scalar VQ k=256 | tools/compress.py (universal) | d=1: fused supported, buffered available | No |
| Hybrid (Mamba+Transformer) | Zamba2-1.2B, Zamba2-2.7B-Instruct, Zamba2-7B-Instruct | scalar k=256 OR mixed (SSM→2D, transformer→scalar) | tools/compress.py --auto-codec | d=1/d=2: fused supported per tensor, buffered available | Yes (AF6 for Zamba2-2.7B) |
| Pure SSM | Mamba-130M | 2D VQ k=4096 (scalar CATASTROPHIC) | tools/compress.py --auto-codec 2d | d=2: vq2d fused supported, buffered available | No |
| Hybrid (Jamba-style) | Jamba-tiny-dev | scalar VQ k=256 | tools/compress.py | d=1: fused supported, buffered available | No |
| Hybrid (Falcon-H1) | Falcon-H1-0.5B | scalar VQ k=256 | tools/compress.py | d=1: fused supported, buffered available | No |
| Custom (STE training) | EchoHybrid-9block | scalar/2D/4D VQ (STE) | helix_linear_ste.py | d>2: buffered only (fused blocked by _check_fused_available) | No |

**Notes on the families table:**
- TinyLlama and Mamba-130M have **no local cdnav3 artifacts**. Baselines are from pod evaluations. Re-compression needed for local compatibility checks.
- Qwen2.5-Coder-3B-Instruct was originally expected to be scalar d=1 but the on-disk artifact is **d=2 k=256** (compression ratio 7.99x). See Section 9.
- Granite-4.0-H-Micro is a MoE hybrid (GraniteMoeHybridForCausalLM) — a different architecture family from the Zamba2 hybrids.
- SmolLM3-3B has GGUF AF6 only (GG04), no cdnav3 artifact.

---

## 2. Change-to-Artifact Impact Map

For each type of code change, which artifacts require which checks.

### Change categories:

| Change scope | Affected artifacts | Required checks |
|-------------|-------------------|----------------|
| **VQ encoder** (`hxq_encoder.py`, `_simple_kmeans`, `_vector_kmeans`) | All VQ-compressed models | New-encoding quality (re-compress + eval representative per family) |
| **Sidecar** (`generate_sidecars_v3.py`) | All models with sidecar enabled | New-encoding quality for sidecar-dependent models |
| **tensor_policy** (`tensor_policy.py`, `get_policy`) | All models (routing decisions change) | New-encoding quality if routing changes; compatibility if only classification changes |
| **CDNAv3Writer** (`hxq_writer_v3.py`) | All VQ-compressed models | New-encoding quality (serialization format) |
| **HelixLinear** (`helix_linear.py`) | All Python/Triton inference | Old-artifact compatibility (load + forward) + dispatch verification |
| **Triton VQ kernel** (`triton_vq_matmul.py`) | d=1 fused-path models | Old-artifact compatibility (numerical equivalence of forward pass) |
| **Triton VQ2D kernel** (`triton_vq2d_matmul.py`) | d=2 fused-path models: PT03 (Qwen-3B d=2), PT07 (Zamba2-1.2B mixed SSM tensors), Mamba-130M | Old-artifact compatibility |
| **Triton affine kernel** (`triton_affine_matmul.py`) | Affine-quantized tensors | Old-artifact compatibility |
| **C99/CUDA decoder** (`hxq.c`, `hxq_cuda.cu`) | All GGUF artifacts | Old-artifact compatibility (llama.cpp load + decode) |
| **GGUF converter** (`convert_gguf.py`) | All GGUF artifacts | New-encoding quality (re-convert + eval) |
| **KurtosisGate** (`kurtosis_gate.py`) | Class 2 modules only | Old-artifact compatibility (SVD gating behavior unchanged) |
| **ESG ablation** (isolated experiment) | NONE in production | Evaluate designated model only; does not affect zoo |

---

## 3. Artifact Registry

> All fields verified from on-disk manifest.json, tensor meta.json, codebook.npy shapes,
> and receipt files. Compression ratios and tensor counts from manifest; vector_dim and k
> from per-tensor codebook arrays. Baseline PPL from receipted evaluations.

### 3a. Python/Triton Artifacts (cdnav3 directories)

| artifact_id | model | params | arch_family | artifact_path | codec | vector_dim | k | sidecar | n_compressed / n_exact | compressed_gb | dense_gb | ratio | compressor | compress_date | baseline_ppl | baseline_delta_pct | eval_tokens | eval_hardware | dispatch_path | receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PT01 | Falcon-H1-0.5B | 0.5B | hybrid_h1 | `~/models/falcon-h1-0.5b-base/cdnav3` | scalar_vq | 1 | 256 | yes | 324 / 147 | 0.45 | 1.82 | 3.99 | compress.py (universal) | 2026-09-05 | 7.660 (HXQ) / 7.633 (dense) | +0.36% [0.33–0.40] | 366,305 test | RTX 4090 | d=1 supported; dispatch unobserved (Step 9 receipt does not record fused/buffered) | Step 9: `HXQ_HYBRID_BREADTH_STEP9_001.json` — custody-verified |
| PT02 | Jamba-tiny-dev | tiny | hybrid_jamba | `~/models/jamba-tiny-dev/cdnav3` | scalar_vq | 1 | 256 | yes | 302 / 119 | 0.25 | 1.01 | 3.99 | compress.py (universal) | 2026-09-06 | 21.985 (HXQ) / 21.908 (dense) | +0.35% [0.31–0.39] | 323,629 test | RTX 4090 | d=1 supported; dispatch unobserved (Step 9 receipt does not record fused/buffered) | Step 9: `HXQ_HYBRID_BREADTH_STEP9_001.json` — custody-verified |
| PT03 | Qwen2.5-Coder-3B-Instruct | 3B | dense_d2 | `~/models/qwen2.5-coder-3b-instruct/cdnav3` | **vq2d** | **2** | 256 | yes | 252 / 182 | 1.39 | 11.10 | **7.99** | compress.py (universal) | 2026-04-24 | **UNRESOLVED** | **UNRESOLVED** | — | — | d=2 supported (vq2d kernel available), actual dispatch unobserved | **NONE** — see note below |
| PT04 | Zamba2-2.7B-Instruct | 2.7B | hybrid | `~/models/zamba2-2.7b-instruct-dense/cdnav3` | scalar_vq | 1 | 256 | yes | 147 / 222 | 2.58 | 10.31 | 4.00 | compress.py (universal) | 2026-03-31 | **UNRESOLVED** | **UNRESOLVED** | — | — | d=1 supported, actual dispatch unobserved | **NONE** for this codec — see related evidence below |
| PT05 | Zamba2-7B-Instruct | 7B | hybrid | `~/models/zamba2-7b-instruct/cdnav3` | scalar_vq | 1 | 256 | yes | 213 / 330 (+45 cached) | 7.24 | 28.95 | 4.00 | compress.py (universal) | 2026-03-30 | 6.418 (HXQ) / 5.944 (dense) | +7.97% | 334,661 test | RTX 3090 | d=1 supported; dispatch unobserved | `gpu_eval/zamba2_7b_instruct_helix.json` |
| PT06 | Granite-4.0-H-Micro | ~3B | hybrid_moe | `~/models/granite-4.0-h-micro/cdnav3` | scalar_vq | 1 | 256 | yes | 168 / 190 (+20 cached) | 2.99 | 11.94 | 4.00 | compress.py (universal) | 2026-04-19 | **UNRESOLVED** | **UNRESOLVED** | — | — | d=1 supported; dispatch unobserved | **NONE** (no receipted model-level evaluation) |
| PT07 | Zamba2-1.2B (mixed) | 1.2B | hybrid_mixed | `~/models/zamba2-1.2b/cdnav3_mixed` | **mixed**: SSM→vq2d, xfmr→scalar | **2** (SSM) / **1** (xfmr) | 4096 (SSM) / 256 (xfmr) | yes | **11 local** (layers 0,1,10,11,12 only) — **PARTIAL**, no manifest | 0.25 | — | — | compress.py --auto-codec | — | **UNRESOLVED** — see note below | **UNRESOLVED** | — | — | d=2 supported (SSM), d=1 supported (xfmr), actual dispatch unobserved | **NONE** paired to this artifact |

**Artifact-to-receipt association rules:**
- A receipt is paired to an artifact ONLY when model identity (revision), codec configuration, AND artifact custody (hash or documented link) all match.
- Same-model/different-codec receipts are listed as **related evidence**, not baselines.
- Dispatch labels use: **supported** (kernel exists for this d), **configured** (artifact metadata enables path), **observed** (receipt records actual dispatch used). No receipt in this registry records actual dispatch path.

**PT03 identity note:** The artifact `qwen2.5-coder-3b-instruct` (manifest: `model: qwen2.5-coder-3b-instruct`) has no paired receipt. The nearby receipt `gpu_eval/qwen2_5_3b_instruct_gpu_bf16.json` identifies its model as `Qwen2.5-3B-Instruct` (non-Coder). These are different models. **Related evidence only:** Qwen2.5-3B-Instruct scalar VQ d=1 evaluates at +1.12% delta. The Coder d=2 artifact's quality is unknown.

**PT04 codec note:** The artifact uses **scalar VQ k=256**. The only Python/Triton evaluation for Zamba2-2.7B is `affine_v4_ppl_zamba2.json` which tests **6-bit affine g128** — a different codec. The scalar VQ baseline for this artifact remains unresolved. **Related evidence only:** Affine_v4 evaluates at +0.89% (4096 tok, T2000). GGUF AF6 evaluates at PPL 22.6526 (335K tok, RTX 3090) — also a different codec.

**PT07 completeness note:** The local directory contains **11 compressed tensors** (layers 0, 1, 10, 11, 12 — 6 of 54 layers). The evaluation receipt `zamba2_1_2b_mixed_codec.json` reports 136 `helix_modules` (60 scalar + 76 2D VQ). **This is a partial directory**, not the full artifact that was evaluated. The +3.45% result cannot be attributed to what is on disk. The complete artifact is missing.

**Artifacts with NO local cdnav3 directory (pod-only or lost):**

| model | status | best available data |
|---|---|---|
| TinyLlama-1.1B | **cdnav3_mixed_rvq in HF cache.** Path: `~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/59f6f375/cdnav3_mixed_rvq/`. 154 compressed, 47 exact, ratio 4.0, compressed 2026-06-02. **TWO mismatches vs receipt:** (1) model is `intermediate-step-1431k-3T`, receipt names `Chat-v1.0` — different revisions; (2) codec is mixed RVQ, receipt evaluated scalar VQ. **Neither artifact nor receipt can be paired.** | **Related evidence only:** TinyLlama-1.1B-Chat-v1.0 scalar VQ PPL 7.011 / dense 7.097 = -1.21% @ 341,469 tokens (pod). Receipt: `gpu_eval/tinyllama_1_1b_chat_v1_0_gpu_bf16.json`. Cannot establish baseline for the local artifact. |
| Qwen2.5-Coder-1.5B | **No cdnav3 on disk.** Adaptive eval only. | Baseline 8.583, adaptive +3.23%, flat-k64 +2.78% @ 4,096 tokens. Receipt: `adaptive_ppl/adaptive_ppl_qwen_1.5b_*.json` |
| Mamba-130M | **No cdnav3 on disk.** Evaluated on pod. | 2D VQ +4.63%, scalar +14.06% @ 4,096 tokens. Receipt: `vq2d_ppl/vq2d_ppl_mamba-130m-hf-dense_*.json` |

### 3b. GGUF Artifacts (AF6, llama.cpp path)

All AF6 GGUFs use uniform quantization (np.linspace codebook), ggml_type 37. NOT k-means.

| artifact_id | model | params | artifact_path | size_gb | baseline_ppl | vs_Q6_K | decode_tok_s | eval_hardware | receipt |
|---|---|---|---|---|---|---|---|---|---|
| GG01 | Qwen2.5-Coder-3B | 3B | `~/models/qwen2.5-coder-3b-hxq-affine6.gguf` | 2.3 | 9.9538 | < Q6 (9.9644) | 226.53 | RTX 3090 | `hxq_runtime_3090_qwen3b_20260508/bench_receipt.json` |
| GG02 | Zamba2-2.7B-Instruct | 2.7B | `~/models/zamba2-2.7b-instruct-hxq-af6.gguf` | 2.8 | 22.6526 | > Q6 (22.573) by 0.08 | 45.87 | RTX 3090 | `hxq_runtime_3090_zamba2_2.7b_20260509/bench_receipt.json` |
| GG03 | Qwen2.5-7B-Instruct | 7B | `~/models/qwen2.5-7b-instruct-hxq-affine6.gguf` | 5.6 | 7.9822 | < Q6 (8.1161) | 114.02 | RTX 3090 Ti | `hxq_runtime_3090ti_qwen7b_instruct_20260509/bench_receipt.json` |
| GG04 | SmolLM3-3B | 3B | `~/models/smollm3-3b-hxq-affine6.gguf` | 2.3 | 9.5195 | (vs Q8: +1.28%) | 28.29 (T2000) | T2000 | `smollm3_three_way_ppl_comparison.json` |
| GG05 | Qwen2.5-Coder-7B | 7B | `~/models/qwen2.5-coder-7b-hxq-affine6-native.gguf` | 5.6 | — | — | — | — | (no receipted benchmark) |
| GG06 | Qwen2.5-Sentinel | 3B | `~/models/qwen2.5-sentinel-hxq-affine6.gguf` | 2.3 | 11.7405 (old-build, requant) | — | — | T2000 (old build) | `hxq_three_way_ppl_comparison.json` — **NOT** prod_validation_v3 (see Section 11, correction 1) |

**Additional GGUF variants (non-AF6 HXQ):**

| artifact_path | codec | size_gb | note |
|---|---|---|---|
| `qwen2.5-coder-3b-hxq-g128-native.gguf` | HXQ G128 | 3.0 | Group-128 quantization variant |
| `qwen2.5-coder-7b-hxq-g128-native.gguf` | HXQ G128 | 7.3 | Group-128 quantization variant |
| `qwen2.5-sentinel-hxq-g128.gguf` | HXQ G128 | 3.0 | Sentinel variant |
| `qwen2.5-coder-3b-instruct-hxq-q5k-*.gguf` (6 versions) | HXQ Q5K | 2.0–2.4 | Multiple iterations of Q5K pipeline |
| `qwen2.5-coder-1.5b-hxq-affine6.gguf` | AF6 | 0.006 | **SUSPECT** — 5.7MB is too small for a 1.5B model. Likely corrupt or partial. |

### 3c. Experimental / Non-Standard Artifacts

| artifact_id | model | config | purpose | on_disk | receipt |
|---|---|---|---|---|---|
| EX01 | TinyLlama-1.1B | d=8 k=256 + sidecar | Sidecar ablation | Pod-only | `rvq_8x_ppl/*.json` |
| EX02 | TinyLlama-1.1B | RVQ k=16+16 + sidecar | RVQ proof | Pod-only | `rvq_8x_ppl/*.json` |
| EX03 | TinyLlama-1.1B | d=2 k=4096 | 2D VQ on dense | Pod-only | `vq2d_ppl/vq2d_ppl_tinyllama-dense_*.json` |
| EX04 | Qwen2.5-3B | d=2 k=4096 | 2D VQ on dense | Pod-only | `vq2d_ppl/vq2d_ppl_qwen2.5-3b-dense_*.json` |
| EX05 | EchoHybrid-9block | d=1/2/4 STE | STE training proof | Pod-only | `echo_hybrid/wo_echo_hybrid_06_summary.json` |
| EX06 | Zamba2-1.2B | d=2 k=4096 | 2D VQ on hybrid SSM | Pod-only | `vq2d_ppl/vq2d_ppl_zamba2-1.2b_*.json` |

---

## 4. Regression Test Plan Template

For a given change, fill in this plan before executing:

```
Change description: _______________________________________________
Files modified:     _______________________________________________
Change category:    [ ] VQ encoder  [ ] sidecar  [ ] tensor_policy
                    [ ] CDNAv3Writer  [ ] HelixLinear  [ ] Triton kernel
                    [ ] C99/CUDA decoder  [ ] GGUF converter
                    [ ] KurtosisGate  [ ] ESG experiment (isolated)

OLD-ARTIFACT COMPATIBILITY (load existing compressed artifacts on updated runtime):
  Required artifacts: [list artifact_ids from registry]
  Check: load → forward → compare output hash/PPL against baseline
  Hardware: T2000 (mechanical correctness, <5 min per model)
  Pass criterion: bitwise-identical logit hash OR PPL within ±0.01% of baseline

  NOTE: Missing historical PPL does NOT block this check. Run the SAME artifact
  on baseline runtime (pre-change) and candidate runtime (post-change) with
  identical inputs. If outputs match, the change is compatible — no dense-model
  delta needed. The comparison is runtime-vs-runtime on a fixed artifact.

NEW-ENCODING QUALITY (compress fresh from updated encoder):
  Required models: [list representative per affected family]
  Check: compress → load → eval PPL → compare against baseline
  Hardware: Pod (for models >1.5B or >10 trial eval)
  Pass criterion: PPL delta within ±0.5% of previous compression's delta

DISPATCH VERIFICATION:
  No existing receipt records which dispatch path was actually used.
  Before regression runs, establish observed dispatch for each artifact:
  Check: dispatch_metadata() on each loaded model reports which path is active
  Record: for each artifact, whether fused or buffered was OBSERVED (not inferred)
  Verify: d=1 artifacts CAN dispatch fused if Triton available (not silent fallback)
  Verify: d=2 artifacts CAN dispatch vq2d fused (not misrouted to scalar fused)
  Verify: d>2 artifacts use buffered (fused blocked by _check_fused_available)
  A regression test checks the path that was actually used, not the path
  that is theoretically available.

REPRESENTATIVE SELECTION (conditional on changed execution path):
  Selection depends on which CODE PATH the change touches, not just which
  models exist. A VQ encoder change requires re-encoding; a Triton kernel
  change requires loading existing artifacts on the new kernel.

  For VQ encoder / CDNAv3Writer / tensor_policy / sidecar changes (re-encode):
    Tier 1: Falcon-H1-0.5B [PT01] — fast, scalar d=1, on-disk, Step 9 baseline
    Tier 2: Qwen2.5-Coder-3B-Instruct [PT03] — d=2 k=256 (if d>1 encoder touched)
    Tier 2: Zamba2-2.7B-Instruct [PT04] — scalar on hybrid (if hybrid paths touched)
    Tier 3: Zamba2-7B-Instruct [PT05] — 7B scale (if large-model paths affected)

  For HelixLinear / Triton kernel changes (load existing artifacts):
    Tier 1: Falcon-H1-0.5B [PT01] — scalar d=1, receipted baseline
    Tier 2: Qwen2.5-Coder-3B-Instruct [PT03] — d=2 (if vq2d kernel touched)
    Tier 3: Granite-4.0-H-Micro [PT06] — MoE hybrid (architecture coverage)
    Note: PT03, PT04, PT06 have no receipted baselines yet. Establish baselines
    before using them for regression comparison.

  For C99/CUDA / GGUF converter changes:
    Tier 1: GG01 Qwen-3B-Coder AF6 — receipted PPL 9.9538 + speed
    Tier 2: GG02 Zamba2-2.7B AF6 — hybrid architecture
    Tier 3: GG03 Qwen-7B AF6 — 7B scale

  MISSING original artifacts (not on local disk):
    TinyLlama-1.1B-Chat-v1.0 scalar VQ
    Mamba-130M 2D VQ k=4096
    Zamba2-1.2B full mixed-codec (136 modules)

  NOTE on missing artifacts: These artifacts are not on local disk. Re-compression
  MAY reproduce identical bytes (k-means is deterministic given same input order
  and seed), but a re-compressed artifact CANNOT substitute for the original
  without verifying identity — hash the output and compare. If hashes differ,
  the re-compressed artifact is a new artifact with its own baseline, not a
  stand-in for the missing one.

  NOTE on runtime regression without historical PPL: Missing historical PPL
  does NOT block runtime regression testing for these models. If the original
  artifact is recovered (or re-compressed and identity-verified), run it on
  baseline runtime and candidate runtime with identical inputs. Output match
  proves compatibility without needing the original dense-model delta.

  Falcon-H1-0.5B [PT01] replaces TinyLlama as the fast Tier 1 representative
  (on-disk, custody-verified via Step 9).
```

---

## 5. Reusable Evidence Inventory

Existing receipts that do NOT need re-running for changes that don't touch their integration surface:

| Receipt | What it proves | Reusable unless |
|---------|---------------|-----------------|
| Step 9 (Falcon-H1 +0.36%, Jamba +0.35%) | Scalar VQ quality on hybrid at 366K+ tokens | VQ encoder, sidecar, or HelixLinear forward changes |
| Qwen-3B-Coder GGUF (PPL 9.9538, 226 tok/s) | AF6 quality + runtime performance | C99/CUDA decoder, convert_gguf.py, or AF6 kernel changes |
| Zamba2-2.7B GGUF (PPL 22.6526, 45.87 tok/s) | Hybrid AF6 quality + runtime | Same as above |
| d=4 kernel correctness (15/15 PASS) | Triton VQ kernel at d=4 in isolation | triton_vq_matmul.py changes |
| Sidecar routing Phase B (rho=-0.023) | Routing signal is dead | Sidecar detection mode changes (confidence signal still valid) |
| SmolLM3 three-way (PPL 9.52, 28.3 tok/s T2000) | AF6 quality vs Q8/Q4 on new model | C99/CUDA decoder or AF6 kernel changes |
| Production validation v3 (SmolLM3 9.4915, Qwen-Coder-3B 10.6621 — **NOT Sentinel**, see Section 11) | Bitwise arm4-equivalence on production builds; Gate 7 throughput is SmolLM3 only | hxq.c, hxq_cuda.cu, or llama.cpp integration changes |
| Zamba2-2.7B T2000 matched (AF6 19.36, Q6K 19.25, Q4KM 19.95) | AF6 quality + throughput on hybrid, stage1 arm4 | hxq.c, hxq_cuda.cu, Zamba2 C++ runtime, or AF6 kernel changes |
| Qwen-Coder-3B T2000 arm4 (PPL 9.97, pp512=241.6, tg128=37.0) | AF6 quality + throughput on stage1 arm4 build | Same as GG01 |
| GAP-HXQ-009 compiled codec parity (7/7 >= 0.999, NO_ROUTE_DIVERSITY) | AF6 tensor-level fidelity for Qwen; uniform AF6 sufficient | AF6 encoder or quantize_row_hxq_affine_6_ref changes |
| Qwen2.5-7B AF6 (PPL 7.9822, 114 tok/s) | AF6 at 7B scale, best-of-4 | Same as GG01 |
| ESG correctness (67/67 PASS) | ESG permutation + weighted k-means code correct | esg_ablation_core.py changes |

---

## 6. ESG Experiment Boundary

The ESG experiment (WP-HXQ-ESG-ABLATION) is isolated from the production zoo:
- It uses `esg_ablation_core.py`, NOT `tools/compress.py`
- It does NOT modify `hxq_encoder.py`, `tensor_policy.py`, or `hxq_writer_v3.py`
- Its results evaluate one proposed improvement; they do not validate existing artifacts
- If ESG is later integrated into the production encoder, THAT integration is a shared codec change requiring full zoo regression

---

## 7. Resolved Questions

- [x] **Artifact paths**: All 7 cdnav3 directories and 19 GGUF files located and listed in Section 3.
- [x] **GGUF vs Python-only**: 6 AF6 GGUFs + 3 G128 GGUFs + 7 Q5K GGUFs. Falcon-H1, Jamba-tiny-dev, Granite have Python-only.
- [x] **Non-default codec configs**: Qwen2.5-Coder-3B-Instruct uses **vd=2 k=256** (NOT scalar). Zamba2-1.2B mixed uses **vd=2 k=4096** for SSM tensors + **vd=1 k=256** for transformer tensors. All others are scalar vd=1 k=256.
- [x] **Total disk footprint**: ~16 GB cdnav3 + ~75 GB GGUF (all variants including non-HXQ baselines).
- [x] **Suspect artifact**: `qwen2.5-coder-1.5b-hxq-affine6.gguf` is 5.7MB — too small for a 1.5B model. Likely corrupt.

## 8. Remaining Gaps

- [ ] **Encoder commit SHA**: Manifests record timestamp but not git commit. Commit SHA is **unknown** for all artifacts. Do not infer from timestamps — compress.py changed multiple times in overlapping date ranges.
- [ ] **AWQ channel scaling**: No manifest records a `--scale-file` flag. Cannot confirm or deny from artifact metadata alone.
- [ ] **TinyLlama-1.1B-Chat-v1.0 scalar VQ artifact**: MISSING. The HF cache contains `intermediate-step-1431k-3T` (different revision) with mixed RVQ (different codec). Neither identity nor codec matches the -1.21% receipt. Re-compression may reproduce identical bytes (verify via hash), but cannot substitute without identity verification. Runtime regression still possible if artifact is recovered: compare baseline-vs-candidate runtime on fixed artifact.
- [ ] **Mamba-130M artifact**: MISSING from disk. Same re-compression and runtime-regression rules as TinyLlama above.
- [ ] **Zamba2-1.2B mixed-codec full artifact**: Local directory is PARTIAL (11 of 136 modules). The complete artifact that produced +3.45% is MISSING. Recovery from pod storage or re-compression with hash verification needed.
- [ ] **Granite-4.0-H-Micro baseline PPL**: No receipted model-level evaluation exists. Artifact is on disk and complete — new-encoding quality can be established but requires a pod run.
- [ ] **PT03 (Qwen-Coder-3B d=2) baseline PPL**: No receipt matches this artifact's model identity and codec. Requires evaluation of the on-disk artifact.
- [ ] **PT04 (Zamba2-2.7B scalar VQ) Python/Triton baseline**: The affine_v4 receipt tests a different codec. The scalar VQ artifact's quality via HelixLinear is unestablished. Related evidence: affine_v4 +0.89% (different codec), GGUF AF6 22.6526 (different codec and runtime).
- [ ] **GG06 (Sentinel) production-era evidence**: MISSING (v3 correction). The v2 matrix claimed "PPL in prod_validation_v3" but hash verification shows prod_v3 tests Qwen-Coder-3B (99fcb109), not Sentinel (edda3697). Sentinel has only pre-arm4/requant PPL (11.7405). Production compatibility gate + matched PPL needed. Reference baselines (F16, Q8_0, Q4_K_M) exist on disk. See Section 11.
- [x] **GG02 (Zamba2-2.7B) scientific comparison**: RESOLVED by stage1 arm4 matched benchmark (2026-10-24). Production binary binding remains unverified (different question from scientific comparison).
- [x] **GG01 (Qwen-Coder-3B) AF6 tensor fidelity**: RESOLVED by GAP-HXQ-009 compiled codec parity (2026-10-27). 7/7 near-threshold tensors >= 0.999. NO_ROUTE_DIVERSITY.

---

## 9. Key Discoveries

### 9a. PT03 (Qwen2.5-Coder-3B-Instruct) uses d=2

The prior draft listed this model as scalar VQ d=1. The actual on-disk artifact uses:
- Codebook shape: `(256, 2)` — 2D vector quantization
- Compression ratio: 7.99x (vs ~4x for scalar) — consistent with halving the index count
- Manifest says "mode: fixed, k: 256" but does NOT record vector_dim

This means changes to `triton_vq2d_matmul.py` affect this model. The d=2 fused dispatch path is **supported** for this artifact but has never been **observed** in a receipt.

**Identity caveat:** The artifact is `Qwen2.5-Coder-3B-Instruct`. The only nearby evaluation receipt names `Qwen2.5-3B-Instruct` (non-Coder). These are different models. The Coder d=2 artifact has **no receipted baseline**.

### 9b. PT07 (Zamba2-1.2B mixed) is partial

The local directory contains 11 compressed tensors (6 layers). The evaluation receipt reports 136 helix_modules (60 scalar + 76 2D VQ) across all 54 layers. The full artifact is missing — likely on pod storage.

### 9c. Three historical artifacts are missing locally

TinyLlama-1.1B-Chat-v1.0 scalar VQ, Mamba-130M, and full Zamba2-1.2B mixed-codec are not on local disk. Their receipted measurements came from pod runs. Re-compression may produce identical bytes (k-means is deterministic given same input order and seed), but the output must be hash-verified against the original before it can substitute. Without hash verification, a re-compressed artifact is a new artifact with its own baseline.

Missing historical PPL does NOT block runtime regression: if the artifact is recovered (or re-compressed and identity-verified), run it on baseline runtime vs candidate runtime with identical inputs. Output match proves compatibility without needing the dense-model delta.

### 9d. Dispatch path never recorded

No receipt in this registry records whether fused or buffered dispatch was actually used. All dispatch labels in the matrix are "supported" (kernel exists), not "observed." Establishing actual dispatch for receipted runs requires either re-running with instrumentation or inspecting the evaluation scripts for explicit dispatch forcing.

---

## 10. Disk Inventory Summary

```
Python/Triton artifacts (cdnav3):
  569M  falcon-h1-0.5b-base/cdnav3         (324 compressed, vd=1, k=256)
  375M  jamba-tiny-dev/cdnav3               (302 compressed, vd=1, k=256)
  1.9G  qwen2.5-coder-3b-instruct/cdnav3   (252 compressed, vd=2, k=256)  ← d=2
  2.6G  zamba2-2.7b-instruct-dense/cdnav3   (147 compressed, vd=1, k=256)
  3.2G  granite-4.0-h-micro/cdnav3          (168 compressed, vd=1, k=256)
  7.0G  zamba2-7b-instruct/cdnav3           (213 compressed, vd=1, k=256)
  246M  zamba2-1.2b/cdnav3_mixed            (11 tensors, PARTIAL — 6 of 54 layers)
  ────
  ~16G total

GGUF artifacts (HXQ only):
  2.3G  qwen2.5-coder-3b-hxq-affine6.gguf
  2.8G  zamba2-2.7b-instruct-hxq-af6.gguf
  5.6G  qwen2.5-7b-instruct-hxq-affine6.gguf
  5.6G  qwen2.5-coder-7b-hxq-affine6-native.gguf
  2.3G  smollm3-3b-hxq-affine6.gguf
  2.3G  qwen2.5-sentinel-hxq-affine6.gguf
  3.0G  qwen2.5-coder-3b-hxq-g128-native.gguf
  7.3G  qwen2.5-coder-7b-hxq-g128-native.gguf
  3.0G  qwen2.5-sentinel-hxq-g128.gguf
  ────
  ~34G HXQ GGUFs
```

---

## 11. Reconciliation v3 — 2026-10-27

READ-ONLY receipt reconciliation against all receipts created after the v2 matrix (2026-10-20).
No models were run. Four source-binding errors corrected.

### Correction 1: GG06 Sentinel / prod_validation_v3 misassociation

**Error:** Matrix v2 stated GG06 (Sentinel) has "PPL in prod_validation_v3".

**Finding:** Hash verification disproves this. The "qwen" model in prod_validation_v3 has SHA256
`99fcb109d3ce819a55b47f0733e0541ef26223f61e796172fb30d8db399fb9d6`, which matches
`qwen2.5-coder-3b-hxq-affine6.gguf` (GG01). Sentinel is `edda3697c1921a84f130fe46e39450b9f29ffc72d37135751a713c3b6e3aef70`.

**Correction:** prod_validation_v3 tests SmolLM3 (b51273f3) and Qwen-Coder-3B (99fcb109).
Sentinel has NO evidence in that receipt. GG06 line updated. Sentinel's only PPL receipt is
`hxq_three_way_ppl_comparison.json` (T2000, pre-arm4 old build, requantized from Q8_0: 11.7405).

**Sentinel production-era evidence status:** MISSING. No production-runtime (f3d6e00d3) test exists.

### Correction 2: Qwen vs SmolLM3 throughput separation

**Error:** prod_validation_v3 Gate 7 throughput (pp512=241.65, tg128=37.59) was attributed to
both GG01 (Qwen-Coder-3B) and GG04 (SmolLM3) without distinguishing source.

**Finding:** Gate 7 compares production throughput to the arm4_characterization_receipt reference,
which is explicitly `"model": "SmolLM3-3B"`. The arm4 reference numbers (pp512=242.12,
tg128=37.18) match SmolLM3. Gate 7 throughput is SmolLM3, not Qwen.

**Corrected bindings:**

| Model | Metric | Value | Receipt | Build | Eval config |
|---|---|---|---|---|---|
| **GG04 SmolLM3** | PPL | 9.4915 | prod_validation_v3 Gate 6 | production f3d6e00d3 | 564 chunks, ctx 512, full wikitext-2 |
| **GG04 SmolLM3** | pp512 | 241.65 | prod_validation_v3 Gate 7 | production f3d6e00d3 | llama-bench -p 512 |
| **GG04 SmolLM3** | tg128 | 37.59 | prod_validation_v3 Gate 7 | production f3d6e00d3 | llama-bench -n 128 |
| **GG01 Qwen-Coder-3B** | PPL | 10.6621 | prod_validation_v3 Gate 6 | production f3d6e00d3 | 583 chunks, ctx 512, full wikitext-2 |
| **GG01 Qwen-Coder-3B** | PPL | 9.9716 | HXQ_T2000_ARM4_RESULTS.json | stage1 arm4 1217ec3b5 | 50 chunks, ctx 512 (short eval) |
| **GG01 Qwen-Coder-3B** | pp512 | 241.6 | HXQ_T2000_ARM4_RESULTS.json | stage1 arm4 1217ec3b5 | llama-bench -p 512 |
| **GG01 Qwen-Coder-3B** | tg128 | 37.0 | HXQ_T2000_ARM4_RESULTS.json | stage1 arm4 1217ec3b5 | llama-bench -n 128 |
| **GG01 Qwen-Coder-3B** | PPL | 9.9538 | bench_receipt_qwen3b_3090.json | old 580e9a213 (MMVQ-only) | 3090, full wikitext-2 |

Note: Qwen has NO production-runtime throughput measurement. The stage1 arm4 throughput (241.6/37.0)
is the closest available but is a different build from production.

### Correction 3: Zamba2 prefill statistic labeling

**Error:** Matrix listed GG02 Zamba2-2.7B pp512=137.22 without specifying which statistic.

**Finding:** The matched benchmark receipt (`zamba2_2.7b_t2000_matched_20261024T164048Z.json`)
reports two pp512 values:
- `throughput_tok_s.pp512 = 137.22` — raw mean of all 6 interleaved runs (includes cold run index 3 at 114.84)
- `throughput_warm_only.pp512 = 141.69` — warm-only mean (excluding index 3, system event)

**Correction:** For matched AF6 vs Q6_K comparison, use warm-only statistics consistently:

| Quant | pp512 (warm) | tg128 | PPL (50-chunk) |
|---|---|---|---|
| HXQ AF6 | 141.69 | 16.27 | 19.3588 |
| Q6_K | 143.77 (raw, no cold anomaly observed) | 15.73 | 19.249 |
| Q4_K_M | 144.96 (raw) | 18.73 | 19.9489 |

The raw mean (137.22) is retained for the cold-inclusive throughput floor. Both are valid statistics
answering different questions. Q6_K and Q4_K_M show no cold anomaly; their raw means are usable
without exclusion.

### Correction 4: Zamba2 rerun classification

**Error:** Zamba2-2.7B was classified as "REQUIRES RERUN: YES (production branch)" without
distinguishing what the rerun would answer.

**Finding:** The October 24 matched benchmark on stage1 arm4 already established the scientific
model comparison:
- AF6 vs Q6_K: ~4.7% smaller, +3.4% decode, ~1.4% slower prefill (warm), PPL within stderr
- Build: af6-mmq-stage1, arm4 reconstructed-sum default, libggml_cuda ae8d52d2

What is NOT established: whether the production binary (f3d6e00d3 on af6-mmq-production) produces
identical results. These are different branches with different commits.

**Corrected classification:**

```
GG02 Zamba2-2.7B-Instruct AF6:
  SCIENTIFIC_MODEL_COMPARISON = COMPLETE (stage1 arm4 matched benchmark, 2026-10-24)
  CURRENT_PRODUCTION_BINARY_BINDING = UNVERIFIED (f3d6e00d3 not tested with Zamba2)
```

A production-branch rerun answers only the binary binding question, not the scientific comparison.

### Post-10/20 receipts ingested

| Receipt | Date | Evidence added |
|---|---|---|
| `zamba2_2.7b_t2000_matched_20261024T164048Z.json` | 2026-10-24 | GG02: T2000 PPL/prefill/decode/VRAM on stage1 arm4 |
| `HXQ_T2000_ARM4_RESULTS.json` | 2026-10-23 | GG01: T2000 PPL/prefill/decode on stage1 arm4 (Qwen, 50-chunk) |
| `compiled_codec_parity_receipt.json` | 2026-10-27 | GG01: AF6 codec fidelity 7/7 >= 0.999, GAP-HXQ-009 NO_ROUTE_DIVERSITY |
| `af6_probe_qwen3b_20261027T101245.json` | 2026-10-27 | GG01: 252-tensor AF6 probe, worst cosine 0.999097 |

### Sentinel reference baseline availability

On-disk artifacts for matched Sentinel evaluation:

| File | Size | Purpose |
|---|---|---|
| `qwen2.5-sentinel-f16.gguf` | 6.18 GB | F16 source (can create Q6_K reference) |
| `qwen2.5-sentinel-q8_0.gguf` | 3.29 GB | Q8_0 reference (existing) |
| `qwen2.5-sentinel-q4_k_m.gguf` | 1.93 GB | Q4_K_M reference (existing) |
| `qwen2.5-sentinel-hxq-affine6.gguf` | 2.43 GB | AF6 under test |
| `qwen2.5-sentinel-merged-dense/` | ~6 GB | Dense safetensors source |

A matched comparison (AF6 vs Q8_0 vs Q4_K_M) is feasible on the production runtime without
creating any new artifacts. Q6_K can be created from F16 if needed.

The existing three-way receipt (`hxq_three_way_ppl_comparison.json`) used Q8_0-requantized AF6
on a pre-arm4/pre-MMQ build. A production-runtime matched comparison would supersede it.

### Next experiment: Sentinel production compatibility gate

Sequencing (do not skip ahead):

1. **Compatibility gate** (cheap, ~2 min): Load `qwen2.5-sentinel-hxq-affine6.gguf` on
   production runtime f3d6e00d3. Run short deterministic logit test. Record model SHA +
   binary SHA + runtime dispatch. If FAIL → stop, diagnose incompatibility.

2. **Matched PPL** (if gate passes, ~75 min for 3-way): Run AF6/Q8_0/Q4_K_M on production
   runtime with identical eval config (wikitext-2, ctx 512, full chunks). This establishes
   production-era PPL and matched compression-quality delta.

3. **Throughput bench** (if PPL passes, ~15 min): llama-bench pp32/pp128/pp512/tg128 for
   all three quants. Establishes T2000 deployment throughput profile.
