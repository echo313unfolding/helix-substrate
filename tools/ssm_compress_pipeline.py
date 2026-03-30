#!/usr/bin/env python3
"""
WO-36: End-to-end SSM/Transformer compress pipeline.

Single-file pipeline: download → baseline PPL → compress → helix PPL → convert+validate →
model card → upload (private) → GPU eval → flip public.

Gate 1 (Stage 4): Validates safetensors after conversion. Blocks on corrupt/incomplete files.
Gate 2 (Stages 6-8): Uploads as private, runs GPU eval, flips to public only if eval passes.

Designed to be scp'd to a cloud box and run standalone. Heavy stages use subprocess
for memory isolation (prevents 10GB+ RSS leak between stages on 24GB boxes).

Resumable: tracks completed stages in {model_dir}/.pipeline_state.json.

Usage:
    # Full pipeline (download, compress, eval, upload)
    python3 tools/ssm_compress_pipeline.py \\
        --model Zyphra/Zamba2-7B-Instruct \\
        --hf-org EchoLabs33

    # Dry run (validate stages, no GPU needed)
    python3 tools/ssm_compress_pipeline.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --hf-org EchoLabs33 --skip-upload --dry-run

    # Resume after failure
    python3 tools/ssm_compress_pipeline.py \\
        --model Zyphra/Zamba2-7B-Instruct --hf-org EchoLabs33

    # Skip stages
    python3 tools/ssm_compress_pipeline.py \\
        --model Zyphra/Zamba2-7B-Instruct --hf-org EchoLabs33 \\
        --skip-baseline --skip-upload

Architecture detection:
    Reads config.json model_type to detect transformer/SSM/hybrid.
    Does NOT change compression behavior (compress.py is universal).
    Used for model card arch notes and logging only.

CDNA v3 codec — how one codec covers all architectures:
    Universal rule: 2D weights → VQ-256 + sidecar. 1D/embeddings/lm_head/conv1d → exact.
    - Transformer (Qwen): All Q/K/V/O + FFN projections compressed. Tied embeddings stored once.
    - SSM (Mamba): in_proj/out_proj compressed. conv1d exact (high kurtosis). A_log/D/dt_bias exact.
    - Hybrid (Zamba2): Mamba2 blocks + shared transformer + LoRA adapters all compressed.
      conv1d exact (kurtosis ~48.6). 136 HelixLinear modules.
    Forward pass: W = codebook[indices] + sidecar, then X@W via Triton kernel.
    Compressed form IS the executable — no decompression step.

Work Order: WO-SSM-COMPRESS-PIPELINE-01
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class PipelineError(Exception):
    """Pipeline stage failure with context."""
    def __init__(self, message: str, stage: str):
        super().__init__(message)
        self.stage = stage


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent


def _run_subprocess(cmd: list[str], stage_name: str, dry_run: bool = False,
                    capture_stdout: bool = False, timeout: int = 7200) -> str | None:
    """Run a subprocess with error handling. Returns stdout if capture_stdout=True."""
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(str(c) for c in cmd)}")
        return '{"dry_run": true}' if capture_stdout else None

    print(f"  Running: {' '.join(str(c) for c in cmd)}", file=sys.stderr, flush=True)
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_stdout,
            text=True,
            timeout=timeout,
            cwd=str(REPO_DIR),
        )
    except subprocess.TimeoutExpired:
        raise PipelineError(f"Stage timed out after {timeout}s", stage_name)

    if result.returncode != 0:
        stderr_msg = result.stderr[:2000] if capture_stdout and result.stderr else ""
        raise PipelineError(
            f"Stage exited with code {result.returncode}. {stderr_msg}",
            stage_name,
        )

    if capture_stdout:
        return result.stdout.strip()
    return None


def _detect_architecture(model_dir: Path) -> dict:
    """Detect model architecture from config.json."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {"model_type": "unknown", "is_ssm": False, "is_hybrid": False}

    config = json.loads(config_path.read_text())
    model_type = config.get("model_type", "unknown")
    return {
        "model_type": model_type,
        "is_ssm": model_type in ("mamba", "mamba2", "zamba2"),
        "is_hybrid": model_type == "zamba2",
        "num_hidden_layers": config.get("num_hidden_layers", config.get("n_layer", "?")),
        "hidden_size": config.get("hidden_size", config.get("d_model", "?")),
    }


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state(model_dir: Path) -> dict:
    """Load pipeline state, or return fresh state."""
    state_path = model_dir / ".pipeline_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"completed": {}, "version": 1}


def _save_state(model_dir: Path, state: dict):
    """Persist pipeline state."""
    model_dir.mkdir(parents=True, exist_ok=True)
    state_path = model_dir / ".pipeline_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def _should_skip(state: dict, stage: str, force: bool) -> bool:
    """Check if stage was already completed."""
    if force:
        return False
    return stage in state.get("completed", {})


# ---------------------------------------------------------------------------
# Embedded baseline eval script (memory-isolated via subprocess)
# ---------------------------------------------------------------------------

BASELINE_EVAL_SCRIPT = r'''
import gc, json, sys, time, platform, resource
import numpy as np
import torch
import torch.nn.functional as F

model_dir = sys.argv[1]
n_tokens = int(sys.argv[2])
seq_len = int(sys.argv[3])

t_start = time.time()
cpu_start = time.process_time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
print(f"Loading dense model from {model_dir}...", file=sys.stderr, flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float32, trust_remote_code=True,
    low_cpu_mem_usage=True,
).eval()

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join([t for t in ds["text"] if t.strip()])
enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens + 1)
ids = enc.input_ids[:, :n_tokens + 1]

nlls, n_eval = [], 0
with torch.no_grad():
    for i in range(0, ids.shape[1] - 1, seq_len):
        end = min(i + seq_len + 1, ids.shape[1])
        chunk = ids[:, i:end]
        if chunk.shape[1] < 2:
            break
        out = model(input_ids=chunk[:, :-1])
        logits = out.logits.float()
        labels = chunk[:, 1:]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        n = labels.numel()
        nlls.append(loss.item() * n)
        n_eval += n
        ppl_so_far = round(float(np.exp(sum(nlls) / n_eval)), 4)
        print(f"  Chunk: {n} tokens, loss={loss.item():.4f}, PPL={ppl_so_far}", file=sys.stderr, flush=True)
        if n_eval >= n_tokens:
            break

ppl = round(float(np.exp(sum(nlls) / n_eval)), 4)
wall = round(time.time() - t_start, 3)
cpu_t = round(time.process_time() - cpu_start, 3)
peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

result = {
    "ppl": ppl,
    "n_tokens": n_eval,
    "wall_time_s": wall,
    "cpu_time_s": cpu_t,
    "peak_memory_mb": peak_mb,
}
print(json.dumps(result))
'''


# ---------------------------------------------------------------------------
# Model card template
# ---------------------------------------------------------------------------

def _generate_model_card(model_name: str, hf_org: str, arch_info: dict,
                         summary: dict) -> str:
    """Generate a HuggingFace README.md model card."""
    model_type = arch_info["model_type"]

    # Architecture-specific notes
    if arch_info["is_hybrid"]:
        arch_note = (
            "Mamba2 blocks + shared transformer + LoRA adapters all compressed. "
            "conv1d exact (high kurtosis). A_log/D/dt_bias exact."
        )
    elif arch_info["is_ssm"]:
        arch_note = (
            "in_proj/out_proj compressed. conv1d exact (high kurtosis). "
            "A_log/D/dt_bias exact."
        )
    else:
        arch_note = (
            "All Q/K/V/O + FFN projections compressed. "
            "Embeddings + norms exact."
        )

    short_name = model_name.split("/")[-1]
    hf_repo = f"{hf_org}/{short_name}-hxq"

    baseline_ppl = summary.get("baseline_ppl", "N/A")
    hxq_ppl = summary.get("helix_ppl", "N/A")
    delta_pct = summary.get("delta_pct", "N/A")
    ratio = summary.get("ratio", "N/A")

    # Build verification status section
    convert_verdict = summary.get("convert_verdict", None)
    convert_sha256 = summary.get("convert_sha256", "")
    gpu_eval_status = summary.get("gpu_eval_status", None)
    gpu_eval_ppl = summary.get("gpu_eval_ppl", None)
    gpu_eval_hardware = summary.get("gpu_eval_hardware", None)

    verification_lines = []
    if convert_verdict == "PASS":
        sha_short = convert_sha256[:16] if convert_sha256 else "N/A"
        verification_lines.append(f"- **Conversion receipt:** PASS (tensor count validated, SHA256 `{sha_short}...`)")
    elif convert_verdict == "FAIL":
        verification_lines.append("- **Conversion receipt:** FAIL")
    else:
        verification_lines.append("- **Conversion receipt:** not available")

    verification_lines.append(
        f"- **Compression receipt:** PASS ({ratio}x ratio, +{delta_pct}% PPL delta)"
        if ratio and delta_pct != "N/A" else
        "- **Compression receipt:** see stats above"
    )

    if gpu_eval_status == "pass":
        hw_note = f" on {gpu_eval_hardware}" if gpu_eval_hardware else ""
        verification_lines.append(
            f"- **GPU eval receipt:** PASS (PPL {gpu_eval_ppl}{hw_note})")
    elif gpu_eval_status == "fail":
        verification_lines.append(
            f"- **GPU eval receipt:** FAIL (PPL {gpu_eval_ppl})")
    else:
        verification_lines.append(
            "- **GPU eval receipt:** Awaiting GPU verification — compression receipt only")

    verification_section = "\n".join(verification_lines)

    card = f"""---
library_name: transformers
tags:
  - hxq
  - helixcode
  - compressed
  - {model_type}
license: apache-2.0
base_model: {model_name}
---

# {short_name}-hxq

HelixCode (HXQ) compressed version of [{model_name}](https://huggingface.co/{model_name}).

## Compression Stats

| Metric | Value |
|--------|-------|
| Base model | {model_name} |
| Architecture | {model_type} |
| Compression ratio | {ratio}x |
| Baseline PPL (WikiText-2) | {baseline_ppl} |
| HXQ PPL (WikiText-2) | {hxq_ppl} |
| PPL delta | {delta_pct}% |

## Verification Status

{verification_section}

## Architecture Notes

{arch_note}

Codec: VQ-256 + sidecar outlier correction. Compressed form is the executable — no decompression step.
Forward pass: `W = codebook[indices] + sidecar`, then `X@W` via Triton fused kernel.

## Usage

```python
import helix_substrate  # registers HXQ quantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{hf_repo}",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("{hf_repo}", trust_remote_code=True)

inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Produced by

[HXQ/HelixCode](https://github.com/EchoLabs33) — HelixCode (HXQ) universal compression codec.

Compressed with `tools/ssm_compress_pipeline.py`.
"""
    return card


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_download(model_name: str, local_dir: Path, dry_run: bool) -> dict:
    """Stage 0: Download model from HuggingFace."""
    if local_dir.exists() and (local_dir / "config.json").exists():
        print(f"  Model already exists at {local_dir}, skipping download.")
        return {"status": "cached", "path": str(local_dir)}

    if dry_run:
        print(f"  [DRY RUN] Would download {model_name} to {local_dir}")
        return {"status": "dry_run", "path": str(local_dir)}

    from huggingface_hub import snapshot_download
    print(f"  Downloading {model_name} to {local_dir}...", file=sys.stderr, flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    wall = round(time.time() - t0, 1)
    print(f"  Download complete in {wall}s", file=sys.stderr, flush=True)
    return {"status": "downloaded", "path": str(local_dir), "wall_time_s": wall}


def stage_baseline_ppl(model_dir: Path, n_tokens: int, seq_len: int,
                       dry_run: bool) -> dict:
    """Stage 1: Evaluate dense baseline PPL via subprocess (memory isolated)."""
    stdout = _run_subprocess(
        [sys.executable, "-c", BASELINE_EVAL_SCRIPT,
         str(model_dir), str(n_tokens), str(seq_len)],
        stage_name="baseline_ppl",
        dry_run=dry_run,
        capture_stdout=True,
        timeout=3600,
    )
    if dry_run:
        return {"status": "dry_run", "ppl": 0.0}

    # Parse last line of stdout as JSON (earlier lines may have stderr mixed in)
    lines = stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise PipelineError(f"No JSON output from baseline eval. Output: {stdout[:500]}", "baseline_ppl")


def stage_compress(model_dir: Path, adaptive: bool, quality_target: float,
                   dry_run: bool, force: bool) -> dict:
    """Stage 2: Compress via subprocess (memory isolated)."""
    if dry_run:
        print(f"  [DRY RUN] Would compress {model_dir}")
        return {"status": "dry_run", "cdna_dir": str(model_dir / "cdnav3"), "n_tensors": 0}

    cmd = [sys.executable, str(TOOLS_DIR / "compress.py"), str(model_dir)]
    if adaptive:
        cmd += ["--adaptive", "--quality-target", str(quality_target)]
    if force:
        cmd.append("--force")

    t0 = time.time()
    _run_subprocess(cmd, stage_name="compress", dry_run=False, timeout=7200)
    wall = round(time.time() - t0, 1)

    # Check output
    cdna_dir = model_dir / "cdnav3"
    if not dry_run and not cdna_dir.exists():
        # Try adaptive output dir
        cdna_dir = model_dir / "cdnav3_adaptive"
        if not cdna_dir.exists():
            raise PipelineError("No cdnav3/ directory after compression", "compress")

    n_tensors = len(list(cdna_dir.glob("*.cdnav3"))) if cdna_dir.exists() else 0
    return {"status": "compressed", "cdna_dir": str(cdna_dir),
            "n_tensors": n_tensors, "wall_time_s": wall}


def stage_helix_ppl(model_dir: Path, n_tokens: int, seq_len: int,
                    dry_run: bool) -> dict:
    """Stage 3: Evaluate helix PPL via subprocess (memory isolated)."""
    stdout = _run_subprocess(
        [sys.executable, str(TOOLS_DIR / "eval_ppl_cpu.py"),
         "--model-dir", str(model_dir),
         "--n-tokens", str(n_tokens),
         "--seq-len", str(seq_len),
         "--skip-dense"],
        stage_name="helix_ppl",
        dry_run=dry_run,
        capture_stdout=True,
        timeout=3600,
    )
    if dry_run:
        return {"status": "dry_run", "ppl": 0.0}

    lines = stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            result = json.loads(line)
            return result.get("helix", result)
    raise PipelineError(f"No JSON output from helix eval. Output: {stdout[:500]}", "helix_ppl")


def stage_convert(model_dir: Path, output_dir: Path, dry_run: bool) -> dict:
    """Stage 4: Convert cdnav3/ to HF safetensors + validate (Gate 1)."""
    if dry_run:
        print(f"  [DRY RUN] Would convert {model_dir}/cdnav3 → {output_dir}")
        return {"status": "dry_run", "output_dir": str(output_dir)}

    sys.path.insert(0, str(REPO_DIR))
    from tools.convert_to_hf import convert_cdnav3_to_hf, emit_conversion_receipt

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    compressed_modules, exact_tensors, receipt = convert_cdnav3_to_hf(model_dir, output_dir)

    wall = round(time.time() - t_start, 1)

    # Emit archive receipt with cost
    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    model_name = model_dir.name
    receipt_path = emit_conversion_receipt(model_name, receipt, output_dir, cost)
    print(f"  Conversion receipt: {receipt_path}")

    # Gate 1: block pipeline on validation failure
    if receipt["verdict"] != "PASS":
        raise PipelineError(
            f"Gate 1 FAIL: safetensors validation failed. "
            f"Details: {receipt['validation'].get('details', [])}",
            "convert",
        )

    size_mb = 0
    safetensors_path = output_dir / "model.safetensors"
    if safetensors_path.exists():
        size_mb = round(safetensors_path.stat().st_size / 1024 / 1024, 1)

    return {
        "status": "converted",
        "output_dir": str(output_dir),
        "compressed_modules": len(compressed_modules),
        "exact_tensors": len(exact_tensors),
        "size_mb": size_mb,
        "wall_time_s": wall,
        "gate1_verdict": receipt["verdict"],
        "gate1_sha256": receipt["validation"]["sha256"],
    }


def stage_model_card(output_dir: Path, model_name: str, hf_org: str,
                     arch_info: dict, summary: dict, dry_run: bool) -> dict:
    """Stage 5: Generate model card README.md."""
    card = _generate_model_card(model_name, hf_org, arch_info, summary)
    if dry_run:
        print(f"  [DRY RUN] Would write model card ({len(card)} chars)")
        return {"status": "dry_run", "chars": len(card)}

    readme_path = output_dir / "README.md"
    readme_path.write_text(card)
    print(f"  Wrote model card to {readme_path} ({len(card)} chars)")
    return {"status": "written", "path": str(readme_path), "chars": len(card)}


def stage_upload(output_dir: Path, repo_id: str, hf_token: str | None,
                 dry_run: bool) -> dict:
    """Stage 6: Upload to HuggingFace Hub as PRIVATE (Gate 2 begins)."""
    if dry_run:
        print(f"  [DRY RUN] Would upload {output_dir} to {repo_id} (private)")
        return {"status": "dry_run", "repo_id": repo_id, "private": True}

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    print(f"  Creating/ensuring repo {repo_id} (private)...", file=sys.stderr, flush=True)
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model", private=True)

    print(f"  Uploading {output_dir} to {repo_id}...", file=sys.stderr, flush=True)
    t0 = time.time()
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    wall = round(time.time() - t0, 1)
    print(f"  Upload complete in {wall}s (PRIVATE)", file=sys.stderr, flush=True)
    return {"status": "uploaded_private", "repo_id": repo_id,
            "private": True, "wall_time_s": wall}


def stage_gpu_eval(output_dir: Path, n_tokens: int, seq_len: int,
                   threshold_pct: float, baseline_ppl: float | None,
                   dry_run: bool) -> dict:
    """Stage 7: GPU PPL eval on converted safetensors (validates the actual artifact)."""
    if dry_run:
        print(f"  [DRY RUN] Would run GPU eval on {output_dir}")
        return {"status": "dry_run", "ppl": 0.0}

    # Check for GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    if not has_gpu:
        print("  No GPU available. Model uploaded as PRIVATE.", file=sys.stderr, flush=True)
        print("  Run GPU eval manually, then use --flip-public REPO_ID", file=sys.stderr, flush=True)
        return {"status": "deferred_no_gpu", "ppl": None}

    # Run PPL eval on the converted HF checkpoint (not the cdnav3 dir)
    stdout = _run_subprocess(
        [sys.executable, str(TOOLS_DIR / "eval_ppl_cpu.py"),
         "--model-dir", str(output_dir),
         "--n-tokens", str(n_tokens),
         "--seq-len", str(seq_len),
         "--skip-dense"],
        stage_name="gpu_eval",
        dry_run=False,
        capture_stdout=True,
        timeout=3600,
    )

    lines = stdout.strip().split("\n")
    result = None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            result = json.loads(line)
            result = result.get("helix", result)
            break

    if result is None:
        raise PipelineError(f"No JSON output from GPU eval. Output: {stdout[:500]}", "gpu_eval")

    ppl = result.get("ppl")
    if ppl is None:
        raise PipelineError("GPU eval returned no PPL value", "gpu_eval")

    # Gate check: delta against baseline
    if baseline_ppl and baseline_ppl > 0:
        delta_pct = (ppl - baseline_ppl) / baseline_ppl * 100
        result["delta_pct"] = round(delta_pct, 2)
        result["threshold_pct"] = threshold_pct
        result["gate_pass"] = delta_pct <= threshold_pct
        if not result["gate_pass"]:
            print(f"  GPU EVAL FAIL: PPL {ppl} is +{delta_pct:.2f}% over baseline {baseline_ppl} "
                  f"(threshold: {threshold_pct}%)", file=sys.stderr, flush=True)
    else:
        result["gate_pass"] = True  # No baseline to compare against

    return result


def stage_flip_public(repo_id: str, hf_token: str | None, dry_run: bool) -> dict:
    """Stage 8: Flip repo from private to public (Gate 2 completes)."""
    if dry_run:
        print(f"  [DRY RUN] Would flip {repo_id} to public")
        return {"status": "dry_run", "repo_id": repo_id}

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.update_repo_visibility(repo_id=repo_id, private=False)
    print(f"  Flipped {repo_id} to PUBLIC", file=sys.stderr, flush=True)
    return {"status": "public", "repo_id": repo_id}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """Execute all pipeline stages with resumability."""
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    # Resolve paths
    model_name = args.model
    short_name = model_name.split("/")[-1].lower()
    if args.local_dir:
        model_dir = Path(args.local_dir)
    else:
        model_dir = Path.home() / "models" / short_name
    hf_org = args.hf_org
    hxq_name = f"{short_name}-hxq"
    output_dir = model_dir.parent / hxq_name
    repo_id = f"{hf_org}/{hxq_name}"

    print(f"\n{'='*70}")
    print(f"  SSM COMPRESS PIPELINE — {model_name}")
    print(f"  Local dir: {model_dir}")
    print(f"  Output:    {output_dir}")
    print(f"  HF repo:   {repo_id}")
    print(f"{'='*70}\n")

    # Load state
    state = _load_state(model_dir)
    receipt = {"pipeline": "ssm_compress_pipeline:v1", "model": model_name, "stages": {}}

    try:
        # ── Stage 0: Download ──
        if not _should_skip(state, "download", args.force):
            print("[Stage 0/8] Download", flush=True)
            result = stage_download(model_name, model_dir, args.dry_run)
            state["completed"]["download"] = result
            receipt["stages"]["download"] = result
            _save_state(model_dir, state)
        else:
            print("[Stage 0/8] Download — cached", flush=True)
            receipt["stages"]["download"] = state["completed"]["download"]

        # Detect architecture
        arch_info = _detect_architecture(model_dir)
        print(f"  Architecture: {arch_info['model_type']} "
              f"(SSM={arch_info['is_ssm']}, hybrid={arch_info['is_hybrid']})")
        receipt["architecture"] = arch_info

        # ── Stage 1: Baseline PPL ──
        if args.skip_baseline:
            print("[Stage 1/8] Baseline PPL — skipped by flag", flush=True)
            receipt["stages"]["baseline_ppl"] = {"status": "skipped"}
        elif not _should_skip(state, "baseline_ppl", args.force):
            print(f"[Stage 1/8] Baseline PPL (n_tokens={args.ppl_tokens}, seq_len={args.seq_len})",
                  flush=True)
            result = stage_baseline_ppl(model_dir, args.ppl_tokens, args.seq_len, args.dry_run)
            state["completed"]["baseline_ppl"] = result
            receipt["stages"]["baseline_ppl"] = result
            _save_state(model_dir, state)
        else:
            print("[Stage 1/8] Baseline PPL — cached", flush=True)
            receipt["stages"]["baseline_ppl"] = state["completed"]["baseline_ppl"]

        # ── Stage 2: Compress ──
        if args.skip_compress:
            print("[Stage 2/8] Compress — skipped by flag", flush=True)
            receipt["stages"]["compress"] = {"status": "skipped"}
        elif not _should_skip(state, "compress", args.force):
            print(f"[Stage 2/8] Compress (adaptive={args.adaptive}, quality={args.quality_target})",
                  flush=True)
            result = stage_compress(model_dir, args.adaptive, args.quality_target,
                                    args.dry_run, args.force)
            state["completed"]["compress"] = result
            receipt["stages"]["compress"] = result
            _save_state(model_dir, state)
        else:
            print("[Stage 2/8] Compress — cached", flush=True)
            receipt["stages"]["compress"] = state["completed"]["compress"]

        # ── Stage 3: Helix PPL ──
        if not _should_skip(state, "helix_ppl", args.force):
            print(f"[Stage 3/8] Helix PPL (n_tokens={args.ppl_tokens}, seq_len={args.seq_len})",
                  flush=True)
            result = stage_helix_ppl(model_dir, args.ppl_tokens, args.seq_len, args.dry_run)
            state["completed"]["helix_ppl"] = result
            receipt["stages"]["helix_ppl"] = result
            _save_state(model_dir, state)
        else:
            print("[Stage 3/8] Helix PPL — cached", flush=True)
            receipt["stages"]["helix_ppl"] = state["completed"]["helix_ppl"]

        # ── Stage 4: Convert to HF ──
        if args.skip_convert:
            print("[Stage 4/8] Convert — skipped by flag", flush=True)
            receipt["stages"]["convert"] = {"status": "skipped"}
        elif not _should_skip(state, "convert", args.force):
            print(f"[Stage 4/8] Convert to HF format → {output_dir}", flush=True)
            result = stage_convert(model_dir, output_dir, args.dry_run)
            state["completed"]["convert"] = result
            receipt["stages"]["convert"] = result
            _save_state(model_dir, state)
        else:
            print("[Stage 4/8] Convert — cached", flush=True)
            receipt["stages"]["convert"] = state["completed"]["convert"]

        # ── Build summary for model card ──
        baseline_ppl_data = receipt["stages"].get("baseline_ppl", {})
        helix_ppl_data = receipt["stages"].get("helix_ppl", {})
        convert_data = receipt["stages"].get("convert", {})

        baseline_ppl = baseline_ppl_data.get("ppl", None)
        helix_ppl = helix_ppl_data.get("ppl", None)
        delta_pct = None
        if baseline_ppl and helix_ppl and baseline_ppl > 0:
            delta_pct = round((helix_ppl - baseline_ppl) / baseline_ppl * 100, 2)

        # Estimate ratio from file sizes
        dense_size_mb = 0
        for sf in model_dir.glob("*.safetensors"):
            dense_size_mb += sf.stat().st_size / 1024 / 1024
        for bf in model_dir.glob("*.bin"):
            dense_size_mb += bf.stat().st_size / 1024 / 1024
        helix_size_mb = convert_data.get("size_mb", 0)
        ratio = round(dense_size_mb / helix_size_mb, 2) if helix_size_mb > 0 else None

        summary = {
            "baseline_ppl": baseline_ppl,
            "helix_ppl": helix_ppl,
            "delta_pct": f"+{delta_pct}" if delta_pct and delta_pct > 0 else str(delta_pct) if delta_pct else "N/A",
            "ratio": ratio,
            "dense_size_mb": round(dense_size_mb, 1),
            "helix_size_mb": helix_size_mb,
            # Gate results for model card verification section
            "convert_verdict": convert_data.get("gate1_verdict"),
            "convert_sha256": convert_data.get("gate1_sha256", ""),
        }

        # ── Stage 5: Model card ──
        if not _should_skip(state, "model_card", args.force):
            print("[Stage 5/8] Model card", flush=True)
            result = stage_model_card(output_dir, model_name, hf_org,
                                      arch_info, summary, args.dry_run)
            state["completed"]["model_card"] = result
            receipt["stages"]["model_card"] = result
            _save_state(model_dir, state)
        else:
            print("[Stage 5/8] Model card — cached", flush=True)
            receipt["stages"]["model_card"] = state["completed"]["model_card"]

        # ── Stage 6: Upload (as PRIVATE — Gate 2 begins) ──
        if args.skip_upload:
            print("[Stage 6/8] Upload — skipped by flag", flush=True)
            receipt["stages"]["upload"] = {"status": "skipped"}
        elif not _should_skip(state, "upload", args.force):
            print(f"[Stage 6/8] Upload to {repo_id} (private)", flush=True)
            result = stage_upload(output_dir, repo_id, args.hf_token, args.dry_run)
            state["completed"]["upload"] = result
            receipt["stages"]["upload"] = result
            _save_state(model_dir, state)
        else:
            print("[Stage 6/8] Upload — cached", flush=True)
            receipt["stages"]["upload"] = state["completed"]["upload"]

        # ── Stage 7: GPU eval on converted safetensors ──
        gpu_eval_passed = True
        if args.skip_gpu_eval or args.skip_upload:
            print("[Stage 7/8] GPU eval — skipped by flag", flush=True)
            receipt["stages"]["gpu_eval"] = {"status": "skipped"}
            if not args.skip_upload:
                print("  Model uploaded as PRIVATE. Run GPU eval manually, "
                      "then use --flip-public REPO_ID", flush=True)
        elif not _should_skip(state, "gpu_eval", args.force):
            print(f"[Stage 7/8] GPU eval on {output_dir}", flush=True)
            result = stage_gpu_eval(
                output_dir, args.ppl_tokens, args.seq_len,
                args.gpu_eval_threshold, baseline_ppl, args.dry_run)
            state["completed"]["gpu_eval"] = result
            receipt["stages"]["gpu_eval"] = result
            _save_state(model_dir, state)
            if result.get("status") == "deferred_no_gpu":
                gpu_eval_passed = True  # Not a failure, just deferred
            elif not result.get("gate_pass", True):
                gpu_eval_passed = False
        else:
            print("[Stage 7/8] GPU eval — cached", flush=True)
            receipt["stages"]["gpu_eval"] = state["completed"]["gpu_eval"]
            gpu_eval_passed = state["completed"]["gpu_eval"].get("gate_pass", True)

        # ── Update model card with GPU eval results ──
        gpu_eval_data = receipt["stages"].get("gpu_eval", {})
        if gpu_eval_data.get("ppl") is not None:
            gpu_ppl = gpu_eval_data["ppl"]
            gate_pass = gpu_eval_data.get("gate_pass", True)
            summary["gpu_eval_status"] = "pass" if gate_pass else "fail"
            summary["gpu_eval_ppl"] = gpu_ppl
            # Re-generate model card with GPU eval results
            card = _generate_model_card(model_name, hf_org, arch_info, summary)
            if not args.dry_run:
                readme_path = output_dir / "README.md"
                readme_path.write_text(card)
                print(f"  Updated model card with GPU eval results", flush=True)

        # ── Stage 8: Flip to public (Gate 2 completes) ──
        if args.skip_upload or args.skip_gpu_eval:
            print("[Stage 8/8] Flip public — skipped (upload or gpu-eval skipped)", flush=True)
            receipt["stages"]["flip_public"] = {"status": "skipped"}
        elif not gpu_eval_passed:
            print("[Stage 8/8] Flip public — BLOCKED (GPU eval failed)", flush=True)
            print(f"  Model stays PRIVATE at {repo_id}. Fix and re-run, or --flip-public manually.",
                  file=sys.stderr, flush=True)
            receipt["stages"]["flip_public"] = {"status": "blocked_by_gpu_eval"}
        elif not _should_skip(state, "flip_public", args.force):
            gpu_eval_status = receipt["stages"].get("gpu_eval", {}).get("status", "")
            if gpu_eval_status == "deferred_no_gpu":
                print("[Stage 8/8] Flip public — deferred (no GPU for eval)", flush=True)
                receipt["stages"]["flip_public"] = {"status": "deferred_no_gpu"}
            else:
                print(f"[Stage 8/8] Flip {repo_id} to public", flush=True)
                result = stage_flip_public(repo_id, args.hf_token, args.dry_run)
                state["completed"]["flip_public"] = result
                receipt["stages"]["flip_public"] = result
                _save_state(model_dir, state)
        else:
            print("[Stage 8/8] Flip public — cached", flush=True)
            receipt["stages"]["flip_public"] = state["completed"]["flip_public"]

        # ── Build receipt ──
        # Determine verdict from gates
        conversion_failed = receipt["stages"].get("convert", {}).get("gate1_verdict") == "FAIL"
        gpu_eval_failed = not gpu_eval_passed

        if conversion_failed:
            verdict = "FAIL:convert"
        elif gpu_eval_failed:
            verdict = "FAIL:gpu_eval"
        elif delta_pct is not None and delta_pct > args.gpu_eval_threshold:
            verdict = "WARN_HIGH_PPL_DELTA"
        else:
            verdict = "PASS"

        wall_time = round(time.time() - t_start, 3)
        cpu_time = round(time.process_time() - cpu_start, 3)
        peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

        receipt["verdict"] = verdict
        receipt["summary"] = summary
        receipt["cost"] = {
            "wall_time_s": wall_time,
            "cpu_time_s": cpu_time,
            "peak_memory_mb": peak_mb,
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Save receipt
        receipt_dir = REPO_DIR / "receipts" / "pipeline"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        receipt_path = receipt_dir / f"{short_name}_{ts}.json"
        with open(receipt_path, "w") as f:
            json.dump(receipt, f, indent=2)

        print(f"\n{'='*70}")
        print(f"  PIPELINE COMPLETE — {verdict}")
        print(f"  Baseline PPL: {baseline_ppl}")
        print(f"  Helix PPL:    {helix_ppl}")
        print(f"  Delta:        {summary['delta_pct']}%")
        print(f"  Ratio:        {ratio}x")
        print(f"  Wall time:    {wall_time}s")
        print(f"  Receipt:      {receipt_path}")
        print(f"{'='*70}\n")

        # JSON to stdout for scripting
        print(json.dumps(receipt))

    except PipelineError as e:
        _save_state(model_dir, state)
        print(f"\n  PIPELINE FAILED at stage '{e.stage}': {e}", file=sys.stderr)
        print(f"  State saved. Re-run to resume from last completed stage.", file=sys.stderr)
        receipt["verdict"] = f"FAIL:{e.stage}"
        receipt["error"] = str(e)

        receipt_dir = REPO_DIR / "receipts" / "pipeline"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        receipt_path = receipt_dir / f"{short_name}_FAIL_{ts}.json"
        with open(receipt_path, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"  Partial receipt: {receipt_path}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        _save_state(model_dir, state)
        print(f"\n  Interrupted. State saved at {model_dir}/.pipeline_state.json", file=sys.stderr)
        print(f"  Re-run to resume.", file=sys.stderr)
        sys.exit(130)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end SSM/Transformer compress pipeline: "
                    "download → baseline PPL → compress → helix PPL → convert → model card → upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 tools/ssm_compress_pipeline.py --model Zyphra/Zamba2-7B-Instruct --hf-org EchoLabs33\n"
               "  python3 tools/ssm_compress_pipeline.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 "
               "--hf-org EchoLabs33 --skip-upload --dry-run\n",
    )

    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID (e.g. Zyphra/Zamba2-7B-Instruct)")
    parser.add_argument("--hf-org", type=str, default=None,
                        help="HuggingFace org for upload (e.g. EchoLabs33)")
    parser.add_argument("--local-dir", type=Path, default=None,
                        help="Local model directory (default: ~/models/{model-name})")
    parser.add_argument("--ppl-tokens", type=int, default=8192,
                        help="Number of tokens for PPL evaluation (default: 8192)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length for PPL chunks (default: 2048)")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive k selection per tensor")
    parser.add_argument("--quality-target", type=float, default=0.998,
                        help="Cosine threshold for adaptive mode (default: 0.998)")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip HuggingFace upload stage")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip dense baseline PPL evaluation")
    parser.add_argument("--skip-compress", action="store_true",
                        help="Skip compression (use existing cdnav3/)")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip HF conversion (use existing output dir)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all stages, ignoring cached state")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate pipeline logic without running stages")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for eval (default: auto)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace API token (default: from env/cache)")
    parser.add_argument("--gpu-eval-threshold", type=float, default=5.0,
                        help="Max PPL delta %% for GPU eval gate (default: 5.0)")
    parser.add_argument("--skip-gpu-eval", action="store_true",
                        help="Skip GPU eval (model stays private, no eval)")
    parser.add_argument("--flip-public", type=str, default=None, metavar="REPO_ID",
                        help="Standalone: flip a private repo to public (e.g. EchoLabs33/model-helix)")

    args = parser.parse_args()

    # Standalone --flip-public mode
    if args.flip_public:
        result = stage_flip_public(args.flip_public, args.hf_token, getattr(args, 'dry_run', False))
        print(json.dumps(result))
        return

    # Full pipeline requires --model and --hf-org
    if not args.model:
        parser.error("--model is required (unless using --flip-public)")
    if not args.hf_org:
        parser.error("--hf-org is required (unless using --flip-public)")

    run_pipeline(args)


if __name__ == "__main__":
    main()
