# HXQ H1A — Q5_K Encoder Reconciliation: Preregistration v1 (CANDIDATE — not frozen)

Status: **candidate for review; SECONDARY / optional.**
* The primary HXQ study is H1
  (`docs/HXQ_H1_AF6_CONFIRMATORY_PREREG_V1.md`), which extends the
  published HXQ_AF6 results.
* H1A covers only the Q5_K stock-compatibility lane.
* The decoded-weight study (H1B) is deferred.
* Nothing in H1A has been executed.

## 0. Background and question

The released artifact `qwen2.5-coder-3b-instruct-hxq-q5k-v3.gguf`
(`releases/hxq_q5k_qwen3b_v0.md`; 2,536,743,072 bytes; sha256
`c25d242df60b44be1d9e95c32da5647f1b973e2b35258c69daf7eb413876c67f`) has:
* **252 standard Q5_K weight matrices** (~5.5 bpw), emitted by HXQ's own
  HXQ_Q5_H encoder;
* an **F16 embedding** and **F32 biases and norms**;
* about 6.6 bpw for the whole file.

It runs in stock llama.cpp with no custom kernels.

The public record (audited 2026-10-13) establishes byte compatibility,
tensor cosine, load, speed and three chat prompts. **No perplexity or
logit comparison against dense or stock quantization exists.** The four
receipts the release cites are not committed to the repository. The
README's Qwen2.5-Coder-3B "+1.92%" belongs to the earlier scalar-VQ Python
path, not to this artifact.

The output format is standard Q5_K, so the only quality contribution HXQ
can make inside this artifact is **the encoder**: its choice of Q5_K block
parameters.

**Question.** With identical tensor layout and types, does HXQ_Q5_H produce
lower language-modelling loss than llama.cpp's own Q5_K encoder, both
without and with importance-matrix (imatrix) calibration?

**Ceiling.** Any claim covers this one model, this evaluation text and
this llama.cpp build. It is a claim about encoder quality only, not speed.
Speed is identical by construction: the same tensor types in the same
kernels.

## 1. Candidates

| ID | Artifact | Role |
|---|---|---|
| D | dense F16 GGUF from `convert_hf_to_gguf.py` at the pinned llama.cpp commit and the pinned HF model revision | loss reference |
| H | the released HXQ_Q5_H artifact, used **as is** (sha256 above) | candidate |
| S | `llama-quantize --pure` Q5_K from D, with `--token-embedding-type f16`; biases and norms F32 | **primary comparator:** stock non-calibrated encoder, same layout |
| I | as S, plus `--imatrix` computed by `llama-imatrix` from D on the calibration text (D3) | stronger comparator: stock calibrated encoder, same layout |
| K | stock Q4_K_M, Q5_K_M, Q6_K (default mixes) | rate–distortion context only; no criterion |

**Layout identity (V1, verified before any evaluation).** H, S and I must
have:
* identical tensor names, order, shapes and ggml types;
* identical tensor-data byte counts.

Only metadata may differ. Any other difference means that comparator is
invalid. Stop and report; do not adjust.

## 2. Evaluator

* **One binary.** `llama-perplexity` from one pinned llama.cpp commit, on
  one backend and device, with full offload. The same binary and backend
  are used for D, H, S, I and K (D4).
* **Text.** The WikiText-2-raw-v1 **test** split, complete
  (`wiki.test.raw`).
* **Settings.** `-c 2048 -b 2048`; all other options at the pinned
  defaults. That means llama-perplexity's standard chunking, which scores
  the second half of each chunk.
* **Per-chunk values.** Paired inference needs a per-chunk mean NLL for
  every model.
  * Phase 0 checks whether the pinned binary emits these at full
    precision.
  * If not, a minimal harness against `libllama` at the same commit
    reproduces llama-perplexity's chunking and scoring exactly and writes
    per-chunk NLL.
  * Engine check E4 validates the harness.
* **Secondary metrics,** from `--kl-divergence-base` (D's logits) and
  `--kl-divergence`, for H, S and I:
  * mean KL(D ‖ X);
  * same-top-token rate;
  * mean ln(PPL(X)/PPL(D)).

## 3. Engine checks (before any comparison is read)

* **E1:** D evaluated twice gives identical per-chunk NLL.
* **E2 (tokenization identity):** tokenizing `wiki.test.raw` with each of
  D, H, S and I's embedded tokenizers yields the identical token sequence
  (sha256 of the id array). H was built through a different conversion
  path (renamed tensors, padded vocab, token-type fixes), so this matters.
* **E3 (layout identity):** V1 holds for H, S and I.
* **E4:** if the harness is used, its mean NLL over all chunks reproduces
  llama-perplexity's final PPL for D to a relative error ≤ 1e-6.
* **E5:** H's sha256 equals the release value.

Any failure means stop and report. Nothing is tuned.

## 4. Decision rule

Per-chunk paired differences, with **positive meaning H is better:**
* Δ_HS = NLL_S − NLL_H;
* Δ_HI = NLL_I − NLL_H.

CIs are 95% paired chunk bootstraps: 10,000 resamples with a frozen seed.
The minimum material difference is δ = **0.0025 nats/token** (the same as
H1B).

| Result for Δ_HS | Conclusion |
|---|---|
| CI lower > 0 and point ≥ δ | **"HXQ_Q5_H is a better non-calibrated Q5_K encoder than stock, on this model."** |
| CI entirely within (−δ, +δ) | **Equivalent to stock Q5_K.** The release's value is the provenance and routing pipeline, not codec quality |
| CI upper < 0 and point ≤ −δ | **Worse than stock Q5_K.** |
| otherwise | **Inconclusive** at δ |

If Δ_HS shows HXQ is better, Δ_HI is read with the same table. Beating I
as well is the stronger claim: "better than stock *calibrated* Q5_K",
achieved calibration-free.

**Context (no criterion).** D, H, S, I and K plotted as PPL against
whole-file effective bpw (8 × file bytes / parameters). H's whole file is
larger than stock Q5_K_M (2.36 vs 2.07 GiB). So a quality win by H over
Q5_K_M is **not** a matched-rate win and is not reported as one.

## 5. Procedure

1. The user reviews and freezes this document, after resolving D1–D5.
2. Build D, S, I and K. Implement the per-chunk extraction (or harness).
   Commit the scripts, the llama.cpp commit and the artifact hashes
   **before** any evaluation.
3. Run E1–E5, then evaluate D, H, S, I and K **once** each. Write one
   receipt containing:
   * the manifest (all hashes and byte counts);
   * the layout check and token-sequence hash;
   * all per-chunk NLLs;
   * the KL statistics and bootstrap results.
4. Apply §4 mechanically, report, and stop.

## 6. Decisions needed before freeze

* **D1 Model revision.** H is used as released, so D, S and I must come
  from the **same** HF revision that H was encoded from. Record that
  revision. If it cannot be established, H1A must re-emit H from a pinned
  revision with `hxq_meta_emit_gguf.py` (about 45 min) instead of using
  the released file.
* **D2 llama.cpp commit.** Pin one stock commit, the build H was
  validated with if it is known. AF6 is not part of H1A, so no fork is
  needed.
* **D3 imatrix calibration text.** It must be disjoint from the
  evaluation text. Options:
  * WikiText-2 **train**: in-domain, the common llama.cpp practice, and it
    gives I a domain advantage;
  * a general public calibration set.

  The choice and the exact token count are frozen.
* **D4 Backend.** Every model must be fully offloaded on one device. The
  dense 3B F16 file (≈ 6.2 GB) does not fit the 4 GB T2000. Proposed: the
  home box GPU for all candidates. Alternative: CPU for all, which is
  slower but uses the identical backend.
* **D5 δ.** 0.0025 nats/token is proposed, matching H1B.
