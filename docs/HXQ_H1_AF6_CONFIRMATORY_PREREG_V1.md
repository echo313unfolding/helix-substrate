# HXQ H1 — AF6 Confirmatory Extension: Preregistration v1 (CANDIDATE — not frozen)

Status: **candidate for review.** This is the **primary** HXQ study. It
extends the published HXQ_AF6 GGUF results rather than repeating them.
Nothing has been executed.

Companions:
* H1A (`HXQ_H1A_Q5K_ENCODER_PREREG_V1.md`) tests the Q5_K compatibility
  lane. It is secondary and optional.
* H1B (`HXQ_H1_MATCHED_RATE_PREREG_V1.md`), the decoded-weight study of
  the older HXQ formats, is deferred.

## 0. The published result being extended

Source: the Hugging Face model cards (read 2026-10-13). Fork
`echo313unfolding/llama.cpp`, branch `hxq-affine-type` @ `580e9a2`;
`llama-perplexity` on WikiText-2-raw test at ctx = 512; full GPU offload.

| Model (card) | GPU | Chunks | Q4_K_M | Q5_K_M | HXQ_AF6 | Q6_K |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-3B-Instruct | RTX 3090 | **50** | 10.072 | 10.004 | **9.954** | 9.964 |
| Qwen2.5-7B-Instruct | RTX 3090 Ti | **50** | 8.315 | 8.184 | **7.982** | 8.116 |
| Zamba2-2.7B-Instruct | RTX 3090 | **654 (full)** | 23.278 | 22.743 | 22.653 | **22.573** |

AF6 artifacts on Hugging Face (byte sizes):
* `qwen2.5-coder-3b-instruct-hxq-affine6.gguf`: 2,429,779,040
* `qwen2.5-7b-instruct-hxq-affine6.gguf`: 5,978,016,256
* `zamba2-2.7b-instruct-hxq-affine6.gguf`: 3,001,120,000

**Audit findings that motivate this extension:**
1. **Sample size.** The Qwen runs scored about 12,800 tokens each (50
   chunks × 256 scored tokens). The only full-test run, Zamba2, is the
   one where AF6 ranked second.
2. **No dense reference.** The Qwen-7B AF6 vs Q6_K gap (−1.65%) is
   unusually large for a 6-bit comparison. A dense baseline is needed to
   tell codec quality apart from sample noise or tensor-policy
   differences.
3. **No confidence intervals, and no matched-rate comparison.** AF6 is
   compared with adjacent formats rather than with the K-quant
   rate–distortion curve at AF6's own bpw.
4. **Receipts not public.** The receipts the cards cite are in neither
   the GitHub repo nor the Hugging Face repos.
   * `hxq_runtime_3090_qwen3b_20260508`
   * `hxq_runtime_3090ti_qwen7b_instruct_20260509`
   * `hxq_runtime_3090_zamba2_2.7b_20260509`
5. **HumanEval.** The difference (84.1 vs 83.5 base) is one problem out
   of 164. It is outside H1's scope and is not treated as evidence.

## 1. Questions

* **Q1 (confirmation).** On the **full** WikiText-2 test, is AF6's NLL
  lower than Q6_K's on each model?
* **Q2 (matched rate).** At AF6's whole-file effective bpw, is AF6 below
  the llama.cpp K-quant rate–distortion curve?
* **Q3 (codec vs policy).** Does the AF6 advantage, where one exists,
  survive against K-quant controls that store the non-codec tensors
  (embedding, output, norms) exactly as the AF6 file does?

Speed is out of scope. The published speed numbers stand as published.

## 2. Phase 0 — artifacts and tensor-policy audit (before any evaluation)

1. **AF6 artifacts.** Download the three published AF6 GGUFs. Record their
   sha256 and verify them against the Hugging Face LFS hashes.
2. **Dense.** For each model, convert a dense GGUF at the pinned HF
   revision with the fork's `convert_hf_to_gguf.py`. Record the revision.
   * Use F16 if the model's released weights are F16; BF16 if they are
     BF16.
   * Use the fork for Zamba2.
3. **K-quants from that dense GGUF.** Default-mix Q4_K_M, Q5_K_M, Q6_K and
   Q8_0, produced by the fork's `llama-quantize` at `580e9a2`.
4. **Per-tensor type table.** List the ggml type of every tensor in every
   file. Then record for the AF6 file:
   * the types of the token embedding, the output head (if untied), and
     the norms and biases;
   * the types of the weight matrices.
5. **Policy-matched controls (Q3).** Q6_K-P and Q5_K-P use `--pure` K-quant
   weight matrices, with `--token-embedding-type` and
   `--output-tensor-type` set to the AF6 file's types. They must have the
   same tensor set and shapes as AF6.
6. **Effective bpw.** For every file, effective bpw = 8 × file bytes /
   parameter count, with the parameter count from the dense GGUF.

## 3. Evaluator

* **Binary.** The fork's `llama-perplexity` @ `580e9a2`: the published
  binary, which supports AF6 and Zamba2.
* **Device.** One GPU model per model, matching the published runs: RTX
  3090 for Coder-3B and Zamba2, and RTX 3090 Ti for 7B. Full offload. The
  dense, quantized and AF6 files of a model all run on the same device.
* **Text and settings.** WikiText-2-raw-v1 **test, complete**, at
  **ctx = 512**, the published setting, kept so the numbers are directly
  comparable. **All chunks.**
* **Per-chunk NLL** for every file, at full precision. If the binary does
  not emit it, a minimal harness against `libllama` at `580e9a2`
  reproduces llama-perplexity's chunking and scoring.
* **Secondary metrics:** KL(dense ‖ X) and same-top-token rate, via
  `--kl-divergence-base` / `--kl-divergence`.

## 4. Engine checks

* **E1:** dense evaluated twice gives identical per-chunk NLL.
* **E2:** within each model, every file's embedded tokenizer turns the
  test text into the identical token sequence (sha256 of the ids).
* **E3:** artifact hashes match Phase 0 and Hugging Face.
* **E4:** if the harness is used, its mean NLL reproduces
  llama-perplexity's final PPL for dense to a relative error ≤ 1e-6.
* **E5:** Q8_0 has \|ΔNLL vs dense\| ≤ 0.005 nats/token. This sanity-checks
  the pipeline.

Any failure means stop and report. Nothing is tuned.

## 5. Decision rules

δ = **0.0025 nats/token** (D3). All CIs are 95% paired chunk bootstraps:
10,000 resamples with a frozen seed, chunks resampled jointly across all
files of a model.

**Q1, per model.** Let Δ = NLL_Q6_K − NLL_AF6 per chunk; positive means
AF6 is better.

| Result | Conclusion |
|---|---|
| CI lower > 0 and point ≥ δ | **Confirmed:** AF6 beats Q6_K on the full test |
| CI within (−δ, +δ) | **Equivalent** to Q6_K at fewer bytes |
| CI upper < 0 and point ≤ −δ | **Reversed:** Q6_K beats AF6 |
| otherwise | **Inconclusive** at δ |

**Q2, per model.** The K-quant curve is the piecewise-linear interpolation,
in effective bpw, through the Q4_K_M, Q5_K_M, Q6_K and Q8_0 points
(ΔNLL vs dense). A = ΔNLL_curve(bpw_AF6) − ΔNLL_AF6. Apply the Q1 table
to A.

**Q3, per model.** Repeat Q1 with Q6_K-P in place of Q6_K. If Q1 confirms
but Q3 does not, the conclusion is: **"AF6's advantage comes from
tensor-policy choices, not the weight codec."**

**Pooled across the three models.** An equal-weight mean of each Δ and A.
The cross-architecture claim "AF6 is at or below the K-quant curve across
transformer and hybrid" requires the pooled A to be confirmed or
equivalent, **and** no single model reversed.

**Card-correction commitment (frozen now).** Whatever the outcome, the
three cards are updated with:
* the full-test numbers, the dense reference and the CIs;
* the Q2 and Q3 verdicts;
* a corrected format description.

The receipts are made public.

## 6. Procedure

1. The user resolves D1–D5 and freezes this document.
2. Phase 0 scripts, the harness (if needed) and the run scripts are
   committed **before** any evaluation.
3. Phase 0, then E1–E5, then each file of each model evaluated **once**.
   One receipt per model contains the hashes, the type tables, the bpw,
   the token-sequence hash, all per-chunk NLLs, the KL statistics and the
   bootstrap results.
4. Apply §5 mechanically, report, update the cards, and stop.

## 7. Decisions needed before freeze

* **D1 Models.** The three published models are proposed. Optional: other
  zoo models with AF6 GGUFs (Qwen2.5-3B, Coder-1.5B). The 14B dense GGUF
  (~29 GB) does not fit a 24 GB GPU; it is excluded unless a CPU
  evaluation is accepted as a separate device.
* **D2 Context.** ctx = 512 is proposed for continuity with the cards.
  Adding ctx = 2048 as a pre-declared secondary would double the
  compute.
* **D3 δ.** 0.0025 nats/token (about 0.25% PPL) is proposed.
* **D4 Q3 controls.** Include Q6_K-P and Q5_K-P (recommended), or drop Q3.
* **D5 Hardware.** The home box's 3090 / 3090 Ti, the published devices.
  Confirm they are still available.
