"""
Basin runtime: capability check, model loading, generation, and dispatch receipts
for HelixLinear + fused CDNA v3 inference.

This module provides the runtime library for the Basin `helix_fused` backend.
No HTTP concerns — just model lifecycle, generation, and receipt building.

Usage:
    from helix_substrate.basin_runtime import (
        check_cdnav3_capability,
        load_helix_model,
        generate_prompt,
        build_dispatch_receipt,
        StrictBenchmarkMode,
    )

    cap = check_cdnav3_capability(cdna_dir)
    if not cap["valid"]:
        print(cap["issues"])

    model, tokenizer, meta = load_helix_model(model_dir)
    text, timing = generate_prompt(model, tokenizer, "Hello", max_tokens=64)
    receipt = build_dispatch_receipt(model, timing, {"prompt": "Hello"}, meta)

Work Order: WO-BASIN-HELIX-FUSED-01
Work Order: WO-BASIN-HARDENING-01 (validate_receipt, build_startup_receipt, build_hardening_summary)
Work Order: WO-BASIN-RECEIPT-SEMANTICS-01 (canonical runtime_path, semantic validation)
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .device_utils import (
    resolve_device, synchronize_device, memory_allocated,
    reset_peak_memory, get_device_name, device_info,
)


# ---------------------------------------------------------------------------
# 1a. Capability check
# ---------------------------------------------------------------------------

_REQUIRED_MANIFEST_FIELDS = ("model", "n_blocks", "n_tensors", "compression_ratio")


def check_cdnav3_capability(cdna_dir) -> dict:
    """
    Validate a CDNA v3 directory for HelixLinear readiness.

    Never raises — returns issues as strings so the caller decides.

    Returns:
        {
            "valid": bool,
            "issues": list[str],
            "model_name": str | None,
            "n_tensors": int | None,
            "n_blocks": int | None,
            "compression_ratio": float | None,
            "manifest_sha256": str | None,
            "tensor_dirs_found": int,
            "triton_available": bool,
            "cuda_available": bool,
        }
    """
    cdna_dir = Path(cdna_dir)
    issues = []
    result = {
        "valid": False,
        "issues": issues,
        "model_name": None,
        "n_tensors": None,
        "n_blocks": None,
        "compression_ratio": None,
        "manifest_sha256": None,
        "tensor_dirs_found": 0,
        "triton_available": False,
        "cuda_available": False,
    }

    # Check dir exists
    if not cdna_dir.exists():
        issues.append(f"CDNA dir not found: {cdna_dir}")
        return result
    if not cdna_dir.is_dir():
        issues.append(f"CDNA path is not a directory: {cdna_dir}")
        return result

    # Load manifest
    manifest_path = cdna_dir / "manifest.json"
    if not manifest_path.exists():
        issues.append(f"manifest.json not found in {cdna_dir}")
        return result

    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (json.JSONDecodeError, OSError) as e:
        issues.append(f"Failed to read manifest.json: {e}")
        return result

    # SHA256 the manifest for receipt provenance
    result["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()

    # Check required fields
    for field in _REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            issues.append(f"manifest.json missing required field: {field}")

    if issues:
        return result

    result["model_name"] = manifest["model"]
    result["n_tensors"] = manifest["n_tensors"]
    result["n_blocks"] = manifest["n_blocks"]
    result["compression_ratio"] = manifest["compression_ratio"]

    # Count .cdnav3 subdirs
    tensor_dirs = sorted(cdna_dir.glob("*.cdnav3"))
    tensor_dirs = [d for d in tensor_dirs if d.is_dir()]
    result["tensor_dirs_found"] = len(tensor_dirs)

    expected = manifest["n_tensors"]
    if len(tensor_dirs) != expected:
        issues.append(
            f"Expected {expected} tensor dirs, found {len(tensor_dirs)}"
        )

    # Spot-check first tensor's meta.json
    if tensor_dirs:
        meta_path = tensor_dirs[0] / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                fmt = meta.get("format_version")
                if fmt != "cdna_v3":
                    issues.append(
                        f"Tensor {tensor_dirs[0].name}: format_version={fmt}, expected cdna_v3"
                    )
            except (json.JSONDecodeError, OSError) as e:
                issues.append(f"Failed to read {meta_path}: {e}")
        else:
            issues.append(f"meta.json missing from {tensor_dirs[0].name}")

    # Check runtime
    resolved = resolve_device()
    result["device_info"] = device_info(resolved)
    result["cuda_available"] = torch.cuda.is_available()
    try:
        from helix_substrate.triton_vq_matmul import is_available
        result["triton_available"] = is_available()
    except ImportError:
        result["triton_available"] = False

    if resolved == "cpu":
        issues.append("No accelerator available (CUDA/MPS both unavailable)")
    if not result["triton_available"]:
        issues.append("Triton fused kernel not available")

    result["valid"] = len(issues) == 0
    return result


# ---------------------------------------------------------------------------
# 1b. BF16 lm_head wrapper
# ---------------------------------------------------------------------------

class _BF16Linear(nn.Module):
    """
    Wraps nn.Linear with BF16 weights for 2x faster cuBLAS matmul.

    HelixLinear outputs FP32, but lm_head doesn't need FP32 precision —
    BF16 produces token-identical greedy decode output (verified by receipt:
    remaining_levers_20260317T004515.json).

    Cast flow: FP32 input → BF16 → matmul → FP32 output.
    Memory: 125 MB vs 250 MB for FP32 (TinyLlama 32K vocab).
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data.bfloat16(), requires_grad=False)
        self.bias_param = nn.Parameter(linear.bias.data.bfloat16(), requires_grad=False) if linear.bias is not None else None
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x.bfloat16(), self.weight, self.bias_param).float()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, dtype=bf16"


# ---------------------------------------------------------------------------
# 1c. Model loading
# ---------------------------------------------------------------------------

def load_helix_model(
    model_dir,
    device: str = "auto",
) -> Tuple[nn.Module, object, dict]:
    """
    Load a HuggingFace model with HelixLinear swap from persistent CDNA v3.

    Follows the proven pattern from model_manager.py:_load().

    Args:
        model_dir: Path to HF model directory (must contain cdnav3/ subdir)
        device: Target device ("auto", "cuda:0", "mps", "cpu")

    Returns:
        (model, tokenizer, load_metadata)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from helix_substrate.helix_linear import load_cdna_factors, swap_to_helix, swap_summary

    model_dir = Path(model_dir)
    cdna_dir = model_dir / "cdnav3"

    if not cdna_dir.exists():
        raise FileNotFoundError(
            f"Pre-compressed CDNA dir not found: {cdna_dir}\n"
            f"Run: python tools/precompress_models.py"
        )

    # SHA256 manifest for provenance
    manifest_path = cdna_dir / "manifest.json"
    manifest_sha256 = None
    manifest_data = {}
    if manifest_path.exists():
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_data = json.loads(manifest_bytes)

    device = resolve_device(device)
    t0 = time.time()

    # Load base model on CPU (FP32)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float32
    )
    model.eval()

    # Load CDNA v3 factors and swap
    helix_modules = load_cdna_factors(cdna_dir, model=model)
    model = swap_to_helix(model, helix_modules)
    summary = swap_summary(model)

    # Optimize lm_head: BF16 is 2x faster than FP32 on cuBLAS with
    # token-identical output (measured in remaining_levers_bench).
    # Wrap in CastingLinear to handle FP32 hidden states from HelixLinear.
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        model.lm_head = _BF16Linear(model.lm_head)

    # Move to device
    reset_peak_memory(device)
    model = model.to(device)
    model.eval()

    load_time = time.time() - t0
    vram_mb = memory_allocated(device)

    load_metadata = {
        "model_dir": str(model_dir),
        "cdna_dir": str(cdna_dir),
        "device": device,
        "helix_modules": summary["helix_modules"],
        "linear_modules": summary["linear_modules"],
        "compression_ratio": summary["overall_ratio"],
        "compressed_bytes": summary["compressed_bytes"],
        "dense_equivalent_bytes": summary["dense_equivalent_bytes"],
        "vram_mb": round(vram_mb, 1),
        "load_time_s": round(load_time, 1),
        "manifest_sha256": manifest_sha256,
        "manifest_model": manifest_data.get("model"),
        "manifest_n_tensors": manifest_data.get("n_tensors"),
    }

    return model, tokenizer, load_metadata


# ---------------------------------------------------------------------------
# 1c. Generation
# ---------------------------------------------------------------------------

def generate_prompt(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 64,
    device: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Generate text from a prompt with timing instrumentation.

    Returns:
        (generated_text, timing_dict)
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_tokens = input_ids.shape[1]

    model.eval()
    with torch.no_grad():
        # Prefill: single forward pass to measure latency
        device_str = str(device)
        t_prefill_start = time.perf_counter()
        _ = model(input_ids)
        synchronize_device(device_str)
        prefill_ms = (time.perf_counter() - t_prefill_start) * 1000

        # Decode: full generation
        t_decode_start = time.perf_counter()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        synchronize_device(device_str)
        decode_ms = (time.perf_counter() - t_decode_start) * 1000

    generated_tokens = output_ids.shape[1] - prompt_tokens
    total_ms = prefill_ms + decode_ms
    tok_s = generated_tokens * 1000.0 / decode_ms if decode_ms > 0 else 0
    prefill_tok_s = prompt_tokens * 1000.0 / prefill_ms if prefill_ms > 0 else 0

    text = tokenizer.decode(output_ids[0, prompt_tokens:], skip_special_tokens=True)

    timing = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prefill_ms": round(prefill_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "total_ms": round(total_ms, 1),
        "tok_s": round(tok_s, 1),
        "prefill_tok_s": round(prefill_tok_s, 1),
    }

    return text, timing


def greedy_decode(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 64,
    device: Optional[str] = None,
    eos_token_id: Optional[int] = None,
) -> Tuple[str, dict]:
    """
    Custom greedy decode loop — bypasses HF generate() overhead.

    HF generate() adds ~1,400 CUDA launches per token from framework
    bookkeeping (slice, unsqueeze, view, copy, sync). This loop does
    only the essential work: forward pass → argmax → append.

    Greedy only (do_sample=False equivalent). For sampling, use generate_prompt().

    Returns:
        (generated_text, timing_dict)
    """
    if device is None:
        device = next(model.parameters()).device
    device_str = str(device)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    model.eval()
    generated_ids = []

    with torch.no_grad():
        # Prefill: process all prompt tokens, get KV cache
        t_prefill_start = time.perf_counter()
        outputs = model(input_ids, use_cache=True)
        synchronize_device(device_str)
        prefill_ms = (time.perf_counter() - t_prefill_start) * 1000

        # First decode token from prefill logits
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        generated_ids.append(next_token.item())

        # Decode loop: one token at a time with KV cache
        t_decode_start = time.perf_counter()
        for _ in range(max_tokens - 1):
            if eos_token_id is not None and generated_ids[-1] == eos_token_id:
                break

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            generated_ids.append(next_token.item())

        synchronize_device(device_str)
        decode_ms = (time.perf_counter() - t_decode_start) * 1000

    n_generated = len(generated_ids)
    total_ms = prefill_ms + decode_ms
    tok_s = n_generated * 1000.0 / (prefill_ms + decode_ms) if total_ms > 0 else 0
    decode_tok_s = (n_generated - 1) * 1000.0 / decode_ms if decode_ms > 0 and n_generated > 1 else 0
    prefill_tok_s = prompt_len * 1000.0 / prefill_ms if prefill_ms > 0 else 0

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    timing = {
        "prompt_tokens": prompt_len,
        "generated_tokens": n_generated,
        "prefill_ms": round(prefill_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "total_ms": round(total_ms, 1),
        "tok_s": round(tok_s, 1),
        "decode_tok_s": round(decode_tok_s, 1),
        "prefill_tok_s": round(prefill_tok_s, 1),
        "decode_loop": "custom_greedy",
    }

    return text, timing


# ---------------------------------------------------------------------------
# 1e. Dispatch receipt
# ---------------------------------------------------------------------------

def build_dispatch_receipt(
    model: nn.Module,
    timing: dict,
    prompt_info: dict,
    load_metadata: dict,
    cost_start_time: Optional[float] = None,
    cost_cpu_start: Optional[float] = None,
    cost_start_iso: Optional[str] = None,
) -> dict:
    """
    Build a structured dispatch receipt after a generation call.

    Walks model.named_modules(), collects dispatch_metadata() from each
    HelixLinear, and builds the receipt.

    Args:
        model: Model after forward pass
        timing: From generate_prompt()
        prompt_info: Caller-provided dict (prompt text/hash, max_tokens, etc.)
        load_metadata: From load_helix_model()
        cost_start_time: time.time() at start of request (for cost block)
        cost_cpu_start: time.process_time() at start of request
        cost_start_iso: ISO timestamp at start of request
    """
    from helix_substrate.helix_linear import HelixLinear

    # Collect dispatch metadata from all HelixLinear modules
    total_helix = 0
    fused_count = 0
    naive_count = 0
    unknown_count = 0

    for _name, module in model.named_modules():
        if isinstance(module, HelixLinear):
            total_helix += 1
            path = module._last_dispatch_path
            if path == "fused":
                fused_count += 1
            elif path == "naive":
                naive_count += 1
            else:
                unknown_count += 1

    # Determine runtime_path (WO-BASIN-RECEIPT-SEMANTICS-01: canonical labels)
    if total_helix == 0:
        runtime_path = "unsupported_manifest"
    elif fused_count == total_helix:
        runtime_path = "fused_triton_v3"
    elif naive_count == total_helix:
        runtime_path = "naive_tiled"
    elif fused_count > 0 and naive_count > 0:
        runtime_path = "mixed_dispatch"
    else:
        runtime_path = "unsupported_manifest"

    # Fallback reason
    fallback_reason = None
    if naive_count > 0 and fused_count == 0:
        if not torch.cuda.is_available() and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            fallback_reason = "No accelerator available"
        elif not torch.cuda.is_available():
            fallback_reason = "CUDA not available (MPS/CPU path)"
        else:
            try:
                from helix_substrate.triton_vq_matmul import is_available
                if not is_available():
                    fallback_reason = "Triton fused kernel not available"
            except ImportError:
                fallback_reason = "triton_vq_matmul not importable"
    elif naive_count > 0:
        fallback_reason = f"{naive_count}/{total_helix} modules fell to naive path"

    # Kernel metadata (if any module ran fused)
    kernel_metadata = None
    if fused_count > 0:
        try:
            from helix_substrate.triton_vq_matmul import get_kernel_metadata
            kernel_metadata = get_kernel_metadata()
        except ImportError:
            pass

    # Prompt info with SHA256
    prompt_text = prompt_info.get("prompt", "")
    prompt_sha256 = hashlib.sha256(prompt_text.encode()).hexdigest()

    # Cost block (WO-RECEIPT-COST-01)
    now = time.time()
    cost = {
        "wall_time_s": round(now - cost_start_time, 3) if cost_start_time else None,
        "cpu_time_s": round(time.process_time() - cost_cpu_start, 3) if cost_cpu_start else None,
        "peak_memory_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": cost_start_iso,
        "timestamp_end": datetime.now(timezone.utc).isoformat(),
    }
    try:
        device_str = str(next(model.parameters()).device)
    except StopIteration:
        device_str = "cpu"
    cost["device"] = get_device_name(device_str)

    receipt = {
        "schema": SCHEMA_REQUEST_RECEIPT,
        "model_id": load_metadata.get("manifest_model", "unknown"),
        "manifest_sha256": load_metadata.get("manifest_sha256"),
        "prompt_info": {
            "prompt_sha256": prompt_sha256,
            "prompt_length": len(prompt_text),
            "max_tokens": prompt_info.get("max_tokens"),
            "seed": prompt_info.get("seed", 42),
        },
        "timing": {
            **timing,
            "backend": "helix_fused",
        },
        "dispatch_summary": {
            "total_helix_modules": total_helix,
            "fused_count": fused_count,
            "naive_count": naive_count,
            "unknown_count": unknown_count,
            "all_fused": fused_count == total_helix and total_helix > 0,
            "runtime_path": runtime_path,
        },
        "kernel_metadata": kernel_metadata,
        "fallback_reason": fallback_reason,
        "load_metadata": {
            "helix_modules": load_metadata.get("helix_modules"),
            "compression_ratio": load_metadata.get("compression_ratio"),
            "vram_mb": load_metadata.get("vram_mb"),
            "load_time_s": load_metadata.get("load_time_s"),
        },
        "cost": cost,
    }

    return receipt


# ---------------------------------------------------------------------------
# 1e. Strict benchmark mode
# ---------------------------------------------------------------------------

@contextmanager
def StrictBenchmarkMode():
    """
    Context manager that raises RuntimeError if any HelixLinear forward
    dispatches to naive path when the input is on CUDA.

    CPU inputs going to naive path are expected and allowed.

    Usage:
        with StrictBenchmarkMode():
            output = model(x_cuda)  # raises if any module fell to naive
    """
    from helix_substrate.helix_linear import HelixLinear

    _original_forward = HelixLinear.forward

    def _strict_forward(self, x):
        output = _original_forward(self, x)
        if x.is_cuda and self._last_dispatch_path == "naive":
            raise RuntimeError(
                f"StrictBenchmarkMode: HelixLinear '{self.tensor_name}' "
                f"dispatched to naive path with CUDA input. "
                f"Expected fused Triton kernel. "
                f"Triton available: {self._use_fused}"
            )
        return output

    HelixLinear.forward = _strict_forward
    try:
        yield
    finally:
        HelixLinear.forward = _original_forward


# ---------------------------------------------------------------------------
# 2a. Receipt validation (WO-BASIN-HARDENING-01 + WO-BASIN-RECEIPT-SEMANTICS-01)
# ---------------------------------------------------------------------------

# Canonical runtime_path taxonomy (WO-BASIN-RECEIPT-SEMANTICS-01)
RUNTIME_PATH_FUSED = "fused_triton_v3"
RUNTIME_PATH_NAIVE = "naive_tiled"
RUNTIME_PATH_MIXED = "mixed_dispatch"
RUNTIME_PATH_UNSUPPORTED = "unsupported_manifest"
RUNTIME_PATH_STARTUP_FAIL = "startup_capability_failure"
RUNTIME_PATH_RECEIPT_FAIL = "receipt_validation_failure"

_VALID_RUNTIME_PATHS = (
    RUNTIME_PATH_FUSED,
    RUNTIME_PATH_NAIVE,
    RUNTIME_PATH_MIXED,
    RUNTIME_PATH_UNSUPPORTED,
    RUNTIME_PATH_STARTUP_FAIL,
    RUNTIME_PATH_RECEIPT_FAIL,
)

# Schema version constants
SCHEMA_REQUEST_RECEIPT = "basin_helix_fused:v1"
SCHEMA_STARTUP_RECEIPT = "basin_startup_helix_fused:v1"
SCHEMA_HARDENING_SUMMARY = "basin_hardening_summary:v1"

_REQUIRED_RECEIPT_TOP = (
    "schema", "model_id", "manifest_sha256", "prompt_info",
    "timing", "dispatch_summary", "kernel_metadata",
    "fallback_reason", "cost",
)
_REQUIRED_DISPATCH = (
    "total_helix_modules", "fused_count", "naive_count", "all_fused", "runtime_path",
)
_REQUIRED_TIMING = (
    "prompt_tokens", "generated_tokens", "prefill_ms", "decode_ms",
    "total_ms", "tok_s", "backend",
)
_REQUIRED_COST = (
    "wall_time_s", "cpu_time_s", "peak_memory_mb", "python_version",
    "hostname", "timestamp_start", "timestamp_end",
)


def validate_receipt(receipt: dict) -> list:
    """
    Validate a dispatch receipt for structural completeness AND semantic consistency.

    Structural checks: required fields present.
    Semantic checks (WO-BASIN-RECEIPT-SEMANTICS-01):
        - schema must be SCHEMA_REQUEST_RECEIPT
        - fused_triton_v3 requires kernel_metadata not None
        - all_fused=True requires naive_count=0
        - fallback_reason=None requires naive_count=0 (no silent pass-through)
        - naive_count>0 requires fallback_reason not None

    Returns list of issues (empty = valid). Never raises.
    """
    issues = []
    if not isinstance(receipt, dict):
        return ["receipt is not a dict"]

    # --- Structural checks ---

    # Top-level fields (all must be present; kernel_metadata and fallback_reason may be None)
    for field in _REQUIRED_RECEIPT_TOP:
        if field not in receipt:
            issues.append(f"missing top-level field: {field}")

    # dispatch_summary sub-fields
    ds = receipt.get("dispatch_summary")
    if isinstance(ds, dict):
        for f in _REQUIRED_DISPATCH:
            if f not in ds:
                issues.append(f"missing dispatch_summary.{f}")
        rp = ds.get("runtime_path")
        if rp not in _VALID_RUNTIME_PATHS:
            issues.append(f"invalid runtime_path: {rp!r}")
    elif ds is not None:
        issues.append("dispatch_summary is not a dict")

    # timing sub-fields
    tm = receipt.get("timing")
    if isinstance(tm, dict):
        for f in _REQUIRED_TIMING:
            if f not in tm:
                issues.append(f"missing timing.{f}")
    elif tm is not None:
        issues.append("timing is not a dict")

    # cost sub-fields
    cost = receipt.get("cost")
    if isinstance(cost, dict):
        for f in _REQUIRED_COST:
            if f not in cost:
                issues.append(f"missing cost.{f}")
    elif cost is not None:
        issues.append("cost is not a dict")

    # Bail early if structural issues prevent semantic checks
    if issues:
        return issues

    # --- Schema version assertion (WO-BASIN-RECEIPT-SEMANTICS-01 task 3) ---
    schema = receipt.get("schema")
    if schema != SCHEMA_REQUEST_RECEIPT:
        issues.append(
            f"schema mismatch: got {schema!r}, expected {SCHEMA_REQUEST_RECEIPT!r}"
        )

    # --- Semantic invariant checks (WO-BASIN-RECEIPT-SEMANTICS-01 task 2) ---

    rp = ds.get("runtime_path")
    fused_count = ds.get("fused_count", 0)
    naive_count = ds.get("naive_count", 0)
    all_fused = ds.get("all_fused")
    km = receipt.get("kernel_metadata")
    fr = receipt.get("fallback_reason")

    # S1: fused_triton_v3 requires kernel_metadata
    if rp == RUNTIME_PATH_FUSED and km is None:
        issues.append(
            f"semantic: runtime_path={rp!r} but kernel_metadata is None"
        )

    # S2: all_fused=True requires naive_count=0
    if all_fused is True and naive_count != 0:
        issues.append(
            f"semantic: all_fused=True but naive_count={naive_count}"
        )

    # S3: fallback_reason=None requires naive_count=0 (no silent pass-through)
    if fr is None and naive_count > 0:
        issues.append(
            f"semantic: fallback_reason is None but naive_count={naive_count} "
            f"(missing fallback explanation)"
        )

    # S4: naive_count>0 requires fallback_reason not None
    #     (redundant with S3 but makes the invariant explicit for audit)
    if naive_count > 0 and fr is None:
        # Already caught by S3, don't double-report
        pass

    # S5: all_fused consistency with counts
    if all_fused is True and fused_count == 0:
        issues.append("semantic: all_fused=True but fused_count=0")

    if all_fused is False and fused_count > 0 and naive_count == 0:
        # all modules were fused but all_fused was set to False
        total = ds.get("total_helix_modules", 0)
        unknown = ds.get("unknown_count", 0)
        if fused_count == total and unknown == 0:
            issues.append(
                f"semantic: all_fused=False but fused_count={fused_count} "
                f"== total_helix_modules={total} with naive_count=0"
            )

    return issues


def validate_startup_receipt(receipt: dict) -> list:
    """
    Validate a startup receipt for structural and semantic correctness.

    Schema assertion: must be SCHEMA_STARTUP_RECEIPT.
    """
    issues = []
    if not isinstance(receipt, dict):
        return ["startup receipt is not a dict"]

    required = (
        "schema", "backend", "model_id", "model_dir", "strict_mode",
        "capability_valid", "capability_issues", "triton_available",
        "cuda_available", "manifest_sha256", "timestamp", "hostname",
    )
    for field in required:
        if field not in receipt:
            issues.append(f"missing startup field: {field}")

    if issues:
        return issues

    # Schema version assertion
    if receipt["schema"] != SCHEMA_STARTUP_RECEIPT:
        issues.append(
            f"schema mismatch: got {receipt['schema']!r}, "
            f"expected {SCHEMA_STARTUP_RECEIPT!r}"
        )

    # Semantic: capability_issues must be a list
    ci = receipt.get("capability_issues")
    if not isinstance(ci, list):
        issues.append(f"capability_issues is not a list: {type(ci).__name__}")

    # Semantic: if capability_valid=False, capability_issues should be non-empty
    if receipt.get("capability_valid") is False and not ci:
        issues.append(
            "semantic: capability_valid=False but capability_issues is empty"
        )

    return issues


def validate_hardening_summary(summary: dict) -> list:
    """
    Validate a hardening summary receipt for structural and semantic correctness.

    Schema assertion: must be SCHEMA_HARDENING_SUMMARY.
    Semantic: request_count == success_count + len(failure_reasons)
    """
    issues = []
    if not isinstance(summary, dict):
        return ["hardening summary is not a dict"]

    required = (
        "schema", "startup_decision", "strict_mode", "request_count",
        "success_count", "fused_count", "naive_count",
        "mean_decode_tok_s", "mean_prefill_tok_s", "failure_reasons",
        "models_tested", "timestamp", "cost",
    )
    for field in required:
        if field not in summary:
            issues.append(f"missing summary field: {field}")

    if issues:
        return issues

    # Schema version assertion
    if summary["schema"] != SCHEMA_HARDENING_SUMMARY:
        issues.append(
            f"schema mismatch: got {summary['schema']!r}, "
            f"expected {SCHEMA_HARDENING_SUMMARY!r}"
        )

    # Semantic: request_count == success_count + len(failure_reasons)
    rc = summary["request_count"]
    sc = summary["success_count"]
    fr = summary.get("failure_reasons", [])
    if rc != sc + len(fr):
        issues.append(
            f"semantic: request_count({rc}) != "
            f"success_count({sc}) + len(failure_reasons)({len(fr)})"
        )

    # Semantic: fused + naive <= success_count
    fc = summary["fused_count"]
    nc = summary["naive_count"]
    if fc + nc > sc:
        issues.append(
            f"semantic: fused_count({fc}) + naive_count({nc}) > success_count({sc})"
        )

    # Semantic: strict_mode=True with naive_count>0 in successful requests is a violation
    if summary["strict_mode"] is True and nc > 0:
        issues.append(
            f"semantic: strict_mode=True but naive_count={nc} in successful requests"
        )

    return issues


# ---------------------------------------------------------------------------
# 2b. Startup receipt (WO-BASIN-HARDENING-01)
# ---------------------------------------------------------------------------

def build_startup_receipt(
    capability: dict,
    model_id: str,
    model_dir,
    strict_mode: bool,
) -> dict:
    """
    Build a structured startup receipt for the helix_fused backend.

    Captures the decision made at Basin startup for audit trail.
    """
    return {
        "schema": SCHEMA_STARTUP_RECEIPT,
        "backend": "helix_fused",
        "model_id": model_id,
        "model_dir": str(model_dir),
        "strict_mode": strict_mode,
        "capability_valid": capability.get("valid", False),
        "capability_issues": list(capability.get("issues", [])),
        "triton_available": capability.get("triton_available", False),
        "cuda_available": capability.get("cuda_available", False),
        "manifest_sha256": capability.get("manifest_sha256"),
        "model_name": capability.get("model_name"),
        "n_tensors": capability.get("n_tensors"),
        "n_blocks": capability.get("n_blocks"),
        "compression_ratio": capability.get("compression_ratio"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
    }


# ---------------------------------------------------------------------------
# 2c. Hardening summary receipt (WO-BASIN-HARDENING-01)
# ---------------------------------------------------------------------------

def build_hardening_summary(
    startup_receipt: dict,
    request_receipts: list,
    failures: list,
    models_tested: list,
    wall_time_s: float,
) -> dict:
    """
    Build an aggregate summary receipt from a hardening run.

    Args:
        startup_receipt: From build_startup_receipt()
        request_receipts: List of per-request dispatch receipts
        failures: List of {"request_index": int, "reason": str} dicts
        models_tested: List of model_id strings tested
        wall_time_s: Total wall time for the entire hardening run
    """
    fused_count = 0
    naive_count = 0
    decode_tok_s = []
    prefill_tok_s = []

    for r in request_receipts:
        ds = r.get("dispatch_summary", {})
        if ds.get("runtime_path") == RUNTIME_PATH_FUSED:
            fused_count += 1
        elif ds.get("runtime_path") == RUNTIME_PATH_NAIVE:
            naive_count += 1
        tm = r.get("timing", {})
        if tm.get("tok_s"):
            decode_tok_s.append(tm["tok_s"])
        if tm.get("prefill_tok_s"):
            prefill_tok_s.append(tm["prefill_tok_s"])

    return {
        "schema": SCHEMA_HARDENING_SUMMARY,
        "startup_decision": startup_receipt.get("backend", "unknown"),
        "strict_mode": startup_receipt.get("strict_mode", False),
        "request_count": len(request_receipts) + len(failures),
        "success_count": len(request_receipts),
        "fused_count": fused_count,
        "naive_count": naive_count,
        "mean_decode_tok_s": round(sum(decode_tok_s) / len(decode_tok_s), 1) if decode_tok_s else 0,
        "mean_prefill_tok_s": round(sum(prefill_tok_s) / len(prefill_tok_s), 1) if prefill_tok_s else 0,
        "latency_spread": {
            "min_decode_tok_s": round(min(decode_tok_s), 1) if decode_tok_s else 0,
            "max_decode_tok_s": round(max(decode_tok_s), 1) if decode_tok_s else 0,
        },
        "failure_reasons": [f["reason"] for f in failures],
        "models_tested": models_tested,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "cost": {
            "wall_time_s": round(wall_time_s, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
        },
    }
