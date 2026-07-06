# Test Coverage Analysis

_Generated 2026-07-06 on branch `claude/test-coverage-analysis-4zn9ld`._

## TL;DR

- The test suite was **effectively dead on arrival**: `import helix_substrate` raised
  `ModuleNotFoundError` because of a half-finished `cdna* → hxq*` rebrand, so **52 of 55
  test files failed at collection** (only 22 tests could even be collected).
- This analysis includes a **mechanical fix** for that breakage (import renames only, no
  logic changes). After the fix the suite runs: **1185 passed, 7 failed, 47 skipped**.
- Measured line coverage after the fix: **`helix_substrate` 49%**, **`echo_hybrid` 3%**,
  **overall 38%**. Before the fix it was unmeasurable.
- The 7 remaining failures are **real defects**, not test bugs (see §3).
- Biggest gaps: the **streaming-decode core** (the library's headline feature) and the
  **model-load / conversion path** are almost entirely untested, and **`echo_hybrid` has
  no tests at all**.

## 1. Method

```bash
pip install numpy safetensors brotli pytest pytest-cov scipy scikit-learn torch
python -m pytest tests/ --cov=helix_substrate --cov=echo_hybrid
python -m coverage report --sort=cover
```

Notes on the environment:
- `torch` is declared **optional** in `pyproject.toml`, but `helix_substrate/__init__.py`
  unconditionally imports `convert`, which imports `torch` at module top level. So the
  package **cannot be imported with the declared core dependencies alone** — torch is a de
  facto hard dependency. 14 test files also import torch directly.
- `scikit-learn` is imported by `echo_hybrid/train_phase1.py` (and transitively by
  `tests/test_grouped_vq_ste.py`) but is **not declared anywhere** in `pyproject.toml`.

## 2. The blocking issue: a broken rebrand (fixed in this branch)

The project was renamed `cdna* → hxq*` at the *file* level (`cdna_encoder.py →
hxq_encoder.py`, `cdnav3_writer.py → hxq_writer_v3.py`, `cdnav3_reader.py →
hxq_reader_v3.py`, `cdna_reader.py → hxq_reader.py`) but **the import statements were
never updated** in the package entry point, several source modules, and 11 test files.
The *class* names kept their old `CDNAv2*/CDNAv3*` spelling, which masked the problem.

Dangling references that were fixed (import-path renames only):

| File | Stale import | Fixed to |
|---|---|---|
| `helix_substrate/__init__.py` | `cdna_encoder`, `cdnav3_writer`, `cdnav3_reader` | `hxq_encoder`, `hxq_writer_v3`, `hxq_reader_v3` |
| `helix_substrate/stream_matmul.py` | `cdna_reader` | `hxq_reader` |
| `helix_substrate/cli.py`, `convert.py`, `convert_gguf.py` | `cdna_reader` | `hxq_reader` |
| `helix_substrate/hxq_writer_v3.py` | `cdna_encoder` | `hxq_encoder` |
| `echo_hybrid/train_phase1.py` | `cdna_encoder` | `hxq_encoder` |
| 11 test files | `cdna_encoder` / `cdna_reader` / `cdnav3_*` | `hxq_*` equivalents |

Impact of the fix: collectable tests went from **22 → 1192**, passing from ~unknown →
**1185**, and `helix_substrate` line coverage from unmeasurable → **49%**.

> **Still broken (not part of the shipped package, left as follow-up):** ~40 scripts under
> `tools/` still import the old `cdnav3_writer` / `cdnav3_reader` / `cdna_encoder` paths.
> They are not on the test path and `pyproject.toml` only packages `helix_substrate*`, so
> they don't affect the library — but every one of them is currently a broken script.

## 3. The 7 remaining failures are real bugs, not flaky tests

| Test(s) | Root cause | Severity |
|---|---|---|
| `test_compression_routing_signal_sweep` (×3) | `tools/sweep_compression_routing_signal.py` imports `helix_substrate.rvq_codec.encode_rvq_tensor` — **that module/function does not exist anywhere in the repo**. | Dead reference to a never-created module. |
| `test_integration` (×3, stream matmul / full pipeline) | `hxq_reader.py:1225` imports `helix_cdc.regrow.cdna_stream` — **the `helix_cdc` package is not present**. The v1 CDNA read path is unreachable. | Dead cross-package dependency in a core reader. |
| `test_k512_uint16::test_k512_better_than_k256` | Asserts more codebook entries never reconstruct worse (`cosine[512] >= cosine[256]`); actually gets `0.9961 < 0.9998`. **Deterministic** across re-runs. | Real quantization-quality regression (or a wrong invariant). |

These deserve tickets in their own right — the first two are dangling references that CI
would have caught immediately, the third is a genuine numerical invariant that the k512
writer violates today.

## 4. Coverage landscape (after fix)

`helix_substrate` modules by band (63 measured modules):

| Band | Count |
|---|---|
| 0% | 21 |
| 1–49% | 16 |
| 50–79% | 16 |
| 80–100% | 31 |

Well-covered core (good news): `receipt` 81%, `verifier` 96%, `symbolic_ir`/`symbolic_executor`
95–97%, `route_inheritance` 90%, `hydra_router` 91%, `residual_contract` 98%,
`session_advisor` 99%, `thesis_tracer`/`routing_ledger`/`response_cache` 100%,
`market_realism` 94%, `backtest_integrity` 96%, `trading_executors` 70%.

The routing / receipts / trading subsystems are genuinely well tested. The gaps are
concentrated in the **compute and I/O core**.

## 5. Where to improve tests — prioritized

### P0 — Stand up CI and lock in what we just recovered
There is **no `.github/workflows`** and **no `[tool.pytest]` / coverage config**. None of
the breakage above would survive a single `pytest` run in CI, yet it shipped in `0.3.3`.
- Add a GitHub Actions workflow running `pytest` on 3.10–3.12 with the real dependency set.
- Declare the *actual* test dependencies in `[project.optional-dependencies].dev`
  (`torch`, `scikit-learn`, `scipy` are all used but under-declared).
- Add a `tests/conftest.py` with a `torch`/GPU-availability fixture so GPU-only tests skip
  cleanly instead of erroring, and add an import-smoke test (`import helix_substrate`) that
  would have caught the rebrand instantly.

### P1 — The streaming-decode core (the headline feature, almost untested)
The README sells "streaming decode: block-by-block `Y = X @ W` without loading full
weights." That path is the least covered code in the repo:

| Module | Cover | Stmts |
|---|---|---|
| `stream_attention.py` | 0% | 300 |
| `stream_block.py` | 0% | 172 |
| `stream_ffn.py` | 25% | 182 |
| `stream_matmul.py` | 21% | 375 |
| `model_manager.py` | 0% | 305 |
| `model_adapter.py` | 0% | 62 |
| `prefetch_pipeline.py` | 0% | 143 |
| `tile_scheduler.py` | 0% | 74 |

None of these need a GPU. Add CPU tests that: build a tiny quantized block, run
`stream_xw_from_cdna` and assert numerical parity against a dense `X @ W` within tolerance;
exercise `stream_ffn` / `stream_attention` on toy shapes; and drive `model_manager` /
`prefetch_pipeline` with a synthetic sidecar. The `test_integration` failures in §3 mean
even the parity tests that *exist* don't run — fixing the `helix_cdc` reference unblocks them.

### P2 — Model load & conversion path (0–24%)
`convert_safetensors` 0%, `hf_integration` 0%, `cli` 0%, `hf_quantizer` 19%,
`convert_gguf` 24%, `hxq_reader` 38%, `layer_chain_receipt` 21%. This is the "get a model
in and back out" path — the first thing any user touches. Add round-trip tests
(safetensors → hxq → decode → compare) on a small synthetic tensor set, and a CLI smoke
test (`helix-substrate --help` and one convert/inspect subcommand on a fixture).

### P3 — Quantization codecs: raise assertions from "runs" to "correct"
`triton_vq_matmul` 15%, `triton_vq2d_matmul` 34%, `triton_rvq_matmul` 39%,
`helix_linear` 53%. The GPU Triton kernels (`triton_affine_matmul`,
`triton_affine_group_matmul`, `triton_gather_12bit`, `triton_mamba_scan`, all 0%) are
legitimately GPU-gated — but they should have **CPU reference-parity tests** that skip on
no-GPU rather than being invisible. The k512 failure (§3) shows codec-quality invariants
are worth asserting; extend that pattern (monotonic quality vs codebook size, MSE bounds).

### P4 — `echo_hybrid` training/eval (3%, effectively 0)
16 modules, ~4,200 statements, **essentially no tests**. Full training runs aren't
CI-friendly, but the pure pieces are: `training_gate`, `coder_config`/`config`/`config_v2`
(config validation), `prepare_data`, and `compress_linear` in `train_phase1`. Add unit
tests for the config/gate logic and a 1-step "does the training loop assemble and take a
gradient step on toy data" smoke test.

### P5 — Feature modules sitting at 0%
`route_policy` (143), `routing_index` (161), `fgip_bridge` (235), `morphsat_gate` (85),
`kurtosis_scan` (161), `mamba_scan_patch`/`mamba_scan_chunked` (279 combined),
`web_fact_tool` (14%). Several already have sibling modules that are well tested
(`route_shaper`, `route_inheritance`, `residual_router`), so the routing gaps are the
cheapest wins — mirror the existing routing tests.

## 6. Suggested next actions

1. Land the import-rename fix in this branch (done here) so the suite is runnable.
2. File the three §3 defects (`rvq_codec` missing, `helix_cdc` missing, k512 invariant).
3. Add CI + declare dev deps (P0) — this is the highest leverage single change.
4. Work P1→P2 for the biggest coverage-per-effort gains on user-facing code.
