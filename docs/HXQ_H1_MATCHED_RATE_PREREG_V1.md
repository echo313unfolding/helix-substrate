# HXQ H1B — Decoded-Weight Multi-Format Codec Study: Preregistration v1 (CANDIDATE — not frozen)

Companion: H1A (`docs/HXQ_H1A_Q5K_ENCODER_PREREG_V1.md`) separately tests the released Q5_K artifact's encoder against stock Q5_K at identical layout.

Status: **candidate for review.** Nothing in H1 has been executed. Earlier
HXQ receipts are background only. The 14B result (README) stays
**unverified** until a dense baseline has been reproduced in this evaluator.

## 0. Question and ceiling

**Question.** At equal whole-model effective bits per weight, does an HXQ
format give better language-model fidelity than the llama.cpp K-quant
rate–distortion curve?

**Out of scope for H1** (deferred to H2, a native-runtime study): speed,
memory, time to first token, paging and bytes touched. They are measured
separately so that codec fidelity and runtime engineering are not
confounded.

**Ceiling.** A positive result may support only: "On these models and this
evaluation set, format X reached lower language-modelling loss than the
llama.cpp K-quant rate–distortion curve at the same effective bpw." It
says nothing about kernels, speed or other tasks.

## 1. Phase 0 — format audit and effective bpw (before any evaluation)

**1. Freeze each candidate format** from code and serialized artifacts, not
from descriptions:
* the format id;
* the encoder entry point and its parameters;
* which tensors are excluded or kept dense;
* sidecar rules;
* the code commit.

**2. Effective bpw** is measured from the serialized artifact:

  effective_bpw = 8 × (total serialized model bytes) / (original parameter count)

* **Included:** indices, codebooks, group scales and offsets, sparse-sidecar
  values and coordinates, low-rank factors, excluded or dense tensors,
  padding, headers and format metadata.
* **Excluded:** nothing. The tokenizer and config are not model bytes.
* Each baseline is computed the same way from its own artifact.

**3. Manifest.** For each (model, format), record:
* the artifact sha256, byte count and effective bpw;
* encoder commit, model revision and the excluded-tensor list.

## 2. Candidates and baselines

| Family | Formats | Role |
|---|---|---|
| HXQ | scalar VQ-256 (+ sidecar as frozen in Phase 0); AF6; 2D VQ-4096; RVQ (**only** if Phase 0 shows it is operational end to end) | candidates |
| llama.cpp K-quants | Q5_K_M, Q6_K, Q8_0 | K-quant rate–distortion (R–D) baseline curve |
| Uniform, simple | per-channel symmetric round-to-nearest INT8 | INT8 reference point |
| Dense | the model's released dtype | loss reference (ΔNLL = 0) |

## 3. Evaluator: one pipeline, decoded weights

Running each format in its own runtime would confound codec error with
evaluator differences. H1 therefore evaluates **every** candidate the same
way:
1. Encode it with its own encoder, into its real serialized format.
2. Decode it back to dense weights with its own reference decoder:
   * HXQ via its reader;
   * GGUF k-quants via the gguf-py reference dequantizer;
   * INT8 via its scales.
3. Load the decoded weights into the **same** Hugging Face model class.

**Frozen settings:**
* WikiText-2-raw-v1 **test split, complete**: the full text joined with
  "\n\n", tokenized once with the model's tokenizer.
* Non-overlapping windows of 2048 tokens, stride 2048. The final partial
  window is dropped. The window count is recorded.
* The forward dtype is identical for the dense model and every candidate
  (D2).
* Sampling is deterministic; there are no prompts beyond the evaluation
  text.

**Metrics:**
* **Primary:** per-window mean token NLL, and **ΔNLL = NLL(candidate) −
  NLL(dense)**, paired by window.
* **Secondary:**
  * mean per-token KL(dense ‖ candidate) over all evaluated positions;
  * top-1 next-token agreement with dense;
  * perplexity (derived).

## 4. Rate–distortion comparison and decision rule

**Baseline curve.** For each model, the llama.cpp K-quant R–D curve is the
piecewise-linear interpolation in effective bpw through the Q5_K_M, Q6_K and
Q8_0 points (ΔNLL against bpw). INT8 is also reported as its own point.

**Advantage.** For an HXQ point at bpw b:

  A(b) = ΔNLL_curve(b) − ΔNLL_HXQ(b)

A positive A means HXQ is better.
* The CI is a paired window bootstrap: 10,000 resamples with a frozen seed,
  resampling windows jointly across all formats of the model.
* If b falls outside the curve's range, that point is reported as
  **not comparable**, not extrapolated.

**Minimum material advantage:** δ = **0.0025 nats/token**, about 0.25% PPL
(D3).

**Per-format verdicts.** A format "beats" the curve when the pooled
equal-weight mean of A over the evaluated models has a 95% CI lower bound
> 0 and a point estimate ≥ δ, **and** on no single model is the A CI upper
bound < 0.

| Format | Survives as a codec contribution if | Otherwise |
|---|---|---|
| scalar VQ-256 | it beats the curve **and** INT8 at matched effective bpw | storage/implementation format only |
| AF6 | it beats the curve | an implementation format equivalent to Q6_K; no scientific claim |
| 2D VQ-4096 | it beats the curve **and** AF6 | dropped as a codec direction |
| RVQ (if run) | it beats the curve | dropped |

**Decision (applied mechanically after H1):**
* If any format survives, improve that codec next.
* If none survives, stop positioning HXQ as a superior weight codec. Run H2
  (paging and working set) and the recurrent-state studies instead.

## 5. Engine checks (before any candidate is evaluated)

* **E1:** the dense model evaluated twice gives bit-identical per-window NLL.
* **E2:** a Q8_0 round trip gives \|ΔNLL\| ≤ 0.005 nats/token. This catches
  a broken dequantizer or loader.
* **E3:** every decoded tensor has the original shape and dtype, and no
  NaN or Inf.
* **E4:** effective bpw recomputed from the artifact equals the manifest
  value.

Any failure means stop and report. Nothing is tuned.

## 6. Procedure

1. The user resolves D1–D5 and freezes this document.
2. Implement Phase 0 and the evaluator. Commit **before** any evaluation.
3. Run Phase 0, then E1–E4, then every (model, format) evaluation **once**.
   Write one receipt with the manifest, all per-window NLLs and the
   bootstrap results.
4. Apply §4 mechanically. Report every point, including the
   not-comparable ones and the losses. Then stop.

## 7. Decisions needed before freeze

* **D1 Models.** Proposed:
  * TinyLlama-1.1B (transformer);
  * Qwen2.5-1.5B (transformer);
  * Zamba2-2.7B (hybrid);
  * one pure Mamba (mamba-2.8b-hf).

  Qwen2.5-7B is optional; it needs a GPU with ≥ 16 GB for the dense
  baseline.
* **D2 Forward dtype.** Proposed fp32 for models ≤ 3B (removes fp16 noise
  from ΔNLL), and fp16 for 7B with the dense baseline in fp16 too.
* **D3 δ.** Proposed 0.0025 nats/token. Earlier receipts put AF6 vs Q6_K
  within about 0.001.
* **D4 INT8 baseline.** Plain per-channel round-to-nearest is proposed.
  An INT8 with the same 0.1% sidecar would make the scalar-VQ comparison
  isolate the codebook from the sidecar. Add it?
* **D5 Hardware.** Fidelity evaluation needs the dense model in the
  forward dtype: about 11 GB at fp32 for 2.7B, which is beyond the T2000.
  Proposed: run H1 on the home box or RunPod.

## 8. Implementation dependencies to verify in Phase 0

* gguf-py's numpy dequantization covers Q5_K, Q6_K and Q8_0, including the
  tensor types used by Mamba and Zamba2 GGUF conversions.
* HXQ readers exist and decode all four HXQ formats to dense.
* All candidates load into the same HF classes for Mamba and Zamba2.

If any dependency is missing for a model, that model is dropped from H1
before evaluation, and the drop is recorded.
