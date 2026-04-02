#!/usr/bin/env python3
"""
Step 2: Convert CDNA v3 compressed model to HuggingFace-compatible safetensors.

Takes a model dir with cdnav3/ artifacts and produces a safetensors file with
HF-compatible key names for each HelixLinear component:
  - {module}.codebook         [256] float32
  - {module}.indices          [out, in] uint8
  - {module}.sidecar_indices  [N] int64  (flat positions of outliers)
  - {module}.sidecar_values   [N] float32 (exact values at those positions)
  - {module}.bias             [out] float32 (if present)

Exact-stored tensors (embeddings, norms, biases) are saved directly.

Also produces a config.json with quantization_config for HF auto-loading.

Usage:
    python3 tools/convert_to_hf.py \
        --model-dir ~/models/zamba2-1.2b \
        --output-dir ~/models/zamba2-1.2b-helix
"""

import argparse
import hashlib
import json
import platform
import re
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert_cdnav3_to_hf(model_dir: Path, output_dir: Path):
    """Convert CDNA v3 directory to HF safetensors format."""
    cdna_dir = model_dir / "cdnav3"
    if not cdna_dir.exists():
        print(f"ERROR: {cdna_dir} not found", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    compressed_modules = []
    exact_tensors = []

    # Process .cdnav3 directories (compressed tensors)
    for tensor_path in sorted(cdna_dir.glob("*.cdnav3")):
        if not tensor_path.is_dir():
            continue

        meta_path = tensor_path / "meta.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        tensor_name = meta["tensor_name"]  # e.g. "model.layers.0.self_attn.q_proj.weight"

        # Strip .weight suffix to get module path
        if tensor_name.endswith(".weight"):
            module_path = tensor_name[:-len(".weight")]
        else:
            module_path = tensor_name

        rows, cols = meta["shape"]
        vector_dim = meta.get("vector_dim", 1)

        # Codebook: [k] float32 for scalar, [k, d] for vector VQ
        codebook = np.load(tensor_path / "codebook.npy").astype(np.float32)
        tensors[f"{module_path}.codebook"] = torch.from_numpy(codebook)

        # Indices: [rows, cols] for scalar, [rows, cols/d] for vector VQ
        idx_dtype_str = meta.get("index_dtype", "uint8")
        np_idx_dtype = np.uint16 if idx_dtype_str == "uint16" else np.uint8
        raw_indices = np.fromfile(tensor_path / "indices.bin", dtype=np_idx_dtype)
        idx_cols = cols // vector_dim if vector_dim > 1 else cols
        indices = raw_indices.reshape(rows, idx_cols)

        k = meta.get("n_clusters", 256)
        if k > 256 and indices.max() < 4096:
            # 12-bit packing: 25% smaller index storage
            from helix_substrate.index_packing import pack_12bit
            idx_tensor = torch.from_numpy(indices.copy().astype(np.int16))
            tensors[f"{module_path}.indices"] = pack_12bit(idx_tensor)
        else:
            tensors[f"{module_path}.indices"] = torch.from_numpy(indices.copy())

        # Sidecar: optional outlier corrections
        sidecar_path = tensor_path / "sidecar.npz"
        if sidecar_path.exists():
            sidecar_data = np.load(sidecar_path)
            positions = sidecar_data["positions"].astype(np.int64)
            values = sidecar_data["values"].astype(np.float32)
            tensors[f"{module_path}.sidecar_positions"] = torch.from_numpy(positions.copy())
            tensors[f"{module_path}.sidecar_values"] = torch.from_numpy(values.copy())

        # SVD residual factors: DEAD (killed 2026-03-27, zero value at all scales).
        # Deliberately not packed — stripping dead artifacts reduces checkpoint size
        # and eliminates false signal during debugging.

        compressed_modules.append({
            "module": module_path,
            "shape": [rows, cols],
            "n_clusters": meta.get("n_clusters", 256),
            "vector_dim": vector_dim,
            "sidecar": sidecar_path.exists(),
        })

    # Process .npy files (exact-stored tensors: embeddings, norms, biases)
    for meta_path in sorted(cdna_dir.glob("*.npy.meta.json")):
        meta = json.loads(meta_path.read_text())
        tensor_name = meta.get("tensor_name", "")
        if not tensor_name:
            continue

        npy_path = meta_path.parent / meta_path.name.replace(".meta.json", "")
        if not npy_path.exists():
            continue

        data = np.load(npy_path)
        # Preserve native dtype (FP16 for new compressions, FP32 for legacy)
        tensors[tensor_name] = torch.from_numpy(data.copy())
        exact_tensors.append(tensor_name)

    # --- Copy missing "skip" tensors from original model ---
    # compress.py skips 1D tensors (A_log, D, dt_bias, norms, biases) without
    # storing them in cdnav3/. We must copy them from the original safetensors.
    orig_sf_files = sorted(model_dir.glob("model*.safetensors"))
    if not orig_sf_files:
        pt_path = model_dir / "pytorch_model.bin"
        if pt_path.exists():
            orig_state = torch.load(pt_path, map_location="cpu", weights_only=True)
            orig_keys = set(orig_state.keys())
        else:
            orig_state = None
            orig_keys = set()
    else:
        from safetensors import safe_open
        orig_keys = set()
        for sf in orig_sf_files:
            with safe_open(str(sf), framework="pt") as f:
                orig_keys.update(f.keys())
        orig_state = None  # loaded on-demand below

    if orig_keys:
        # Keys covered by compressed modules (codebook/indices replace .weight)
        compressed_weight_keys = {m["module"] + ".weight" for m in compressed_modules}
        output_keys = set(tensors.keys())
        covered = set()
        for k in output_keys:
            covered.add(k)
            for suffix in (".codebook", ".indices", ".sidecar_positions", ".sidecar_values"):
                if k.endswith(suffix):
                    covered.add(k.rsplit(".", 1)[0] + ".weight")
        covered.update(compressed_weight_keys)

        missing = orig_keys - covered
        if missing:
            # Copy missing tensors from original model (skip tensors: 1D params)
            copied = 0
            if orig_sf_files:
                from safetensors import safe_open
                for sf in orig_sf_files:
                    with safe_open(str(sf), framework="pt") as f:
                        for k in list(missing):
                            if k in f.keys():
                                # Preserve native dtype (BF16/FP16) — don't inflate to FP32
                                tensors[k] = f.get_tensor(k)
                                exact_tensors.append(k)
                                missing.discard(k)
                                copied += 1
            elif orig_state is not None:
                for k in list(missing):
                    if k in orig_state:
                        tensors[k] = orig_state[k]
                        exact_tensors.append(k)
                        missing.discard(k)
                        copied += 1

            print(f"  Copied {copied} skip tensors from original model")
            if missing:
                print(f"\n  ERROR: {len(missing)} tensors still missing after copy!")
                for k in sorted(missing)[:20]:
                    print(f"    MISSING: {k}")
                sys.exit(1)
        else:
            print(f"  Validation: all {len(orig_keys)} original tensors accounted for")

    # Save safetensors (after all tensors collected: compressed + exact + skip)
    safetensors_path = output_dir / "model.safetensors"
    save_file(tensors, str(safetensors_path))
    print(f"  Saved {len(tensors)} tensors to {safetensors_path}")
    print(f"    {len(compressed_modules)} compressed modules")
    print(f"    {len(exact_tensors)} exact tensors (including skip)")

    # Copy original config.json and add quantization_config
    orig_config_path = model_dir / "config.json"
    if orig_config_path.exists():
        config = json.loads(orig_config_path.read_text())
    else:
        config = {}

    # Determine if any module uses 12-bit packing
    has_12bit = any(m.get("n_clusters", 256) > 256 for m in compressed_modules)

    config["quantization_config"] = {
        "quant_method": "hxq",
        "codebook_size": 256,
        "sidecar_enabled": True,
        "exact_patterns": ["embed_tokens", "embed_positions", "wte", "wpe",
                           "lm_head", "layernorm", "layer_norm", "norm",
                           "backbone.embedding"],
        "compressed_modules": [m["module"] for m in compressed_modules],
    }
    if has_12bit:
        config["quantization_config"]["index_packing"] = "12bit"

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config with quantization_config to {config_path}")

    # Copy tokenizer files
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                     "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = model_dir / tok_file
        if src.exists():
            shutil.copy2(src, output_dir / tok_file)

    # Copy any custom modeling code (trust_remote_code models)
    for py_file in model_dir.glob("*.py"):
        shutil.copy2(py_file, output_dir / py_file.name)

    # Summary
    total_bytes = safetensors_path.stat().st_size
    print(f"\n  Output: {output_dir}")
    print(f"  Size: {total_bytes / 1024**2:.1f} MB")
    print(f"  Ready for: AutoModelForCausalLM.from_pretrained('{output_dir}')")

    # ── Gate 1: Validate the safetensors we just wrote ──
    receipt = validate_safetensors(safetensors_path, compressed_modules, exact_tensors)

    # Save receipt alongside model (travels with it)
    receipt_local = output_dir / "conversion_receipt.json"
    with open(receipt_local, "w") as f:
        json.dump(receipt, f, indent=2)

    if receipt["verdict"] == "PASS":
        print(f"\n  GATE 1 PASS — {receipt['validation']['total_keys']} keys validated, "
              f"SHA256={receipt['validation']['sha256'][:16]}...")
    else:
        print(f"\n  GATE 1 FAIL — conversion validation failed:", file=sys.stderr)
        for check, passed in receipt["validation"]["checks"].items():
            if not passed:
                print(f"    FAIL: {check}", file=sys.stderr)
        for detail in receipt["validation"].get("details", []):
            print(f"    {detail}", file=sys.stderr)

    # ── Gate 2: Completeness validation against original dense model ──
    gate2 = validate_completeness_gate(safetensors_path, model_dir, compressed_modules)

    # Save gate2 receipt alongside model
    gate2_path = output_dir / "completeness_gate_receipt.json"
    with open(gate2_path, "w") as f:
        json.dump(gate2, f, indent=2)

    if gate2["verdict"] == "PASS":
        print(f"\n  GATE 2 PASS — {gate2['summary']['dense_tensors']} dense tensors accounted for, "
              f"{gate2['summary']['skip_tensors_found']} skip tensors verified present")
    else:
        print(f"\n  GATE 2 FAIL — completeness validation failed:", file=sys.stderr)
        for detail in gate2.get("failures", []):
            print(f"    FAIL: {detail}", file=sys.stderr)
        raise RuntimeError(
            f"GATE 2 FAIL: {len(gate2.get('failures', []))} completeness checks failed. "
            f"Output safetensors is missing tensors from the dense model. "
            f"See {gate2_path} for details."
        )

    return compressed_modules, exact_tensors, receipt


def validate_completeness_gate(
    safetensors_path: Path,
    dense_model_dir: Path,
    compressed_modules: list = None,
) -> dict:
    """Validate that the converted safetensors contains ALL tensors from the dense model.

    This is the hard gate: if ANY tensor from the dense model is unaccounted for
    in the output, the pipeline MUST stop. This catches the class of bugs where
    skip tensors (norms, SSM params, embeddings, output heads) silently vanish
    during conversion.

    Can be called:
      1. Automatically at the end of convert_cdnav3_to_hf() (Gate 2)
      2. Standalone for auditing existing converted models:
           python3 tools/convert_to_hf.py --audit \\
               --safetensors ~/models/foo-helix/model.safetensors \\
               --dense-model ~/models/foo

    Args:
        safetensors_path: Path to the output .safetensors file
        dense_model_dir: Path to the original dense model directory
        compressed_modules: List of compressed module dicts (from conversion).
            If None, inferred from safetensors keys.

    Returns:
        dict with verdict (PASS/FAIL), summary, failures list.
    """
    failures = []

    # ── Load output safetensors keys ──
    try:
        with safe_open(str(safetensors_path), framework="pt") as f:
            output_keys = set(f.keys())
    except Exception as e:
        return {
            "verdict": "FAIL",
            "failures": [f"Cannot open output safetensors: {e}"],
            "summary": {},
        }

    # ── Load dense model tensor list ──
    dense_keys = set()
    orig_sf_files = sorted(dense_model_dir.glob("model*.safetensors"))
    if orig_sf_files:
        for sf in orig_sf_files:
            with safe_open(str(sf), framework="pt") as f:
                dense_keys.update(f.keys())
    else:
        pt_path = dense_model_dir / "pytorch_model.bin"
        if pt_path.exists():
            import torch as _torch
            state = _torch.load(pt_path, map_location="cpu", weights_only=True)
            dense_keys = set(state.keys())
            del state
        else:
            return {
                "verdict": "FAIL",
                "failures": ["No dense model weights found (no safetensors or pytorch_model.bin)"],
                "summary": {},
            }

    if not dense_keys:
        return {
            "verdict": "FAIL",
            "failures": ["Dense model has zero tensors — corrupt or empty"],
            "summary": {},
        }

    # ── Build the coverage map ──
    # For each dense tensor, determine how it should appear in the output:
    #   - Compressed weight: replaced by {module}.codebook + {module}.indices (+ optional sidecar)
    #   - Everything else: must appear with its original key name

    # Discover which original weight keys were compressed by examining output keys
    # A compressed module "{module}" produces keys like:
    #   {module}.codebook, {module}.indices, {module}.sidecar_positions, {module}.sidecar_values
    # Build set of original dense keys that were compressed.
    # Compressed modules store "module" = tensor_name minus ".weight" suffix.
    # For tensors that originally ended in ".weight", the dense key is module + ".weight".
    # For tensors that did NOT end in ".weight" (rare: 2D SSM params), the dense key
    # is the module path itself. We check both possibilities against the dense keys.
    compressed_weight_keys = set()
    if compressed_modules:
        for m in compressed_modules:
            candidate_with_weight = m["module"] + ".weight"
            if candidate_with_weight in dense_keys:
                compressed_weight_keys.add(candidate_with_weight)
            elif m["module"] in dense_keys:
                compressed_weight_keys.add(m["module"])
            else:
                # Fallback: assume .weight suffix (the common case)
                compressed_weight_keys.add(candidate_with_weight)
    else:
        # Infer from output keys: any key ending in .codebook implies compression
        for k in output_keys:
            if k.endswith(".codebook"):
                module_path = k[: -len(".codebook")]
                candidate_with_weight = module_path + ".weight"
                if candidate_with_weight in dense_keys:
                    compressed_weight_keys.add(candidate_with_weight)
                elif module_path in dense_keys:
                    compressed_weight_keys.add(module_path)
                else:
                    compressed_weight_keys.add(candidate_with_weight)

    # Map dense keys to their expected output representation
    accounted = set()
    skip_tensor_report = []

    # Known skip tensor patterns (tensors that compress.py marks "skip" and
    # are NOT stored in cdnav3/ — they must be copied verbatim from dense model)
    SKIP_PATTERNS = [
        # LayerNorm / RMSNorm
        (r"\.norm.*\.weight$", "layernorm"),
        (r"\.layernorm.*\.weight$", "layernorm"),
        (r"\.layer_norm.*\.weight$", "layernorm"),
        (r"\.rmsnorm.*\.weight$", "layernorm"),
        (r"\.input_layernorm\.weight$", "layernorm"),
        (r"\.post_attention_layernorm\.weight$", "layernorm"),
        (r"\.final_layernorm\.weight$", "layernorm"),
        (r"model\.norm\.weight$", "layernorm"),
        (r"model\.final_layernorm\.weight$", "layernorm"),
        # Mamba / Zamba SSM parameters (1D, non-weight)
        (r"\.A_log$", "ssm_param"),
        (r"\.D$", "ssm_param"),
        (r"\.dt_bias$", "ssm_param"),
        # Embeddings
        (r"embed_tokens\.weight$", "embedding"),
        (r"embed_positions\.weight$", "embedding"),
        (r"\.wte\.weight$", "embedding"),
        (r"\.wpe\.weight$", "embedding"),
        (r"backbone\.embedding.*\.weight$", "embedding"),
        # Output head
        (r"lm_head\.weight$", "output_head"),
        # Biases (1D)
        (r"\.bias$", "bias"),
    ]
    _skip_compiled = [(re.compile(pat), label) for pat, label in SKIP_PATTERNS]

    def classify_skip(name: str):
        """Return skip category if name matches a known skip pattern, else None."""
        for pat, label in _skip_compiled:
            if pat.search(name):
                return label
        return None

    # Check every dense tensor
    missing_from_output = []
    skip_found = {}     # category -> list of tensor names
    skip_missing = {}   # category -> list of tensor names

    for dense_key in sorted(dense_keys):
        if dense_key in compressed_weight_keys:
            # This weight was compressed — check that codebook+indices exist
            module_path = dense_key[: -len(".weight")] if dense_key.endswith(".weight") else dense_key
            cb_key = f"{module_path}.codebook"
            idx_key = f"{module_path}.indices"
            if cb_key in output_keys and idx_key in output_keys:
                accounted.add(dense_key)
            else:
                missing_parts = []
                if cb_key not in output_keys:
                    missing_parts.append(f"codebook ({cb_key})")
                if idx_key not in output_keys:
                    missing_parts.append(f"indices ({idx_key})")
                failures.append(
                    f"Compressed tensor {dense_key} missing parts: {', '.join(missing_parts)}"
                )
                missing_from_output.append(dense_key)
        elif dense_key in output_keys:
            # Present as exact copy
            accounted.add(dense_key)
            cat = classify_skip(dense_key)
            if cat:
                skip_found.setdefault(cat, []).append(dense_key)
        else:
            # NOT in output at all — this is a failure
            accounted.discard(dense_key)  # not accounted
            missing_from_output.append(dense_key)
            cat = classify_skip(dense_key)
            if cat:
                skip_missing.setdefault(cat, []).append(dense_key)
            failures.append(f"DROPPED: {dense_key} (category: {cat or 'unknown'})")

    # ── Verify expected skip categories are present ──
    # If the dense model has tensors matching skip patterns, at least some must appear
    # in the output. If a whole category vanishes, that's a red flag.
    expected_categories = set()
    for dense_key in dense_keys:
        cat = classify_skip(dense_key)
        if cat:
            expected_categories.add(cat)

    category_verdict = {}
    for cat in sorted(expected_categories):
        n_found = len(skip_found.get(cat, []))
        n_missing = len(skip_missing.get(cat, []))
        n_total = n_found + n_missing
        if n_missing > 0:
            category_verdict[cat] = f"INCOMPLETE ({n_found}/{n_total})"
            # Already logged individual failures above
        else:
            category_verdict[cat] = f"OK ({n_found}/{n_total})"

    # ── Build summary ──
    n_skip_total = sum(len(v) for v in skip_found.values())
    summary = {
        "dense_tensors": len(dense_keys),
        "output_tensors": len(output_keys),
        "compressed_weights": len(compressed_weight_keys),
        "accounted": len(accounted),
        "missing": len(missing_from_output),
        "skip_tensors_found": n_skip_total,
        "skip_categories": category_verdict,
    }

    verdict = "PASS" if len(failures) == 0 else "FAIL"

    return {
        "verdict": verdict,
        "summary": summary,
        "failures": failures,
        "missing_tensors": missing_from_output[:50],  # cap for readability
        "skip_found": {cat: names for cat, names in skip_found.items()},
    }


def validate_safetensors(safetensors_path: Path, compressed_modules: list,
                         exact_tensors: list) -> dict:
    """Reopen saved safetensors and validate structural integrity.

    Checks:
      1. Readable — safe_open() succeeds
      2. Compressed modules complete — codebook + indices exist, indices in range
      3. Exact tensors valid — no NaN/Inf
      4. Key count matches expected
      5. SHA256 of entire file

    Returns dict with verdict (PASS/FAIL), checks, details, sha256.
    """
    checks = {
        "readable": False,
        "compressed_complete": True,
        "indices_in_range": True,
        "exact_no_nan": True,
        "key_count_match": True,
    }
    details = []
    sha256_hex = ""

    # 1. Readable
    try:
        f = safe_open(str(safetensors_path), framework="pt")
        all_keys = set(f.keys())
        checks["readable"] = True
    except Exception as e:
        checks["readable"] = False
        details.append(f"safe_open failed: {e}")
        return {
            "verdict": "FAIL",
            "validation": {"checks": checks, "details": details,
                           "total_keys": 0, "compressed_modules": len(compressed_modules),
                           "exact_tensors": len(exact_tensors), "sha256": ""},
        }

    # 2. Compressed modules complete + indices in range
    for mod in compressed_modules:
        module_path = mod["module"]
        k = mod.get("n_clusters", 256)

        cb_key = f"{module_path}.codebook"
        idx_key = f"{module_path}.indices"

        if cb_key not in all_keys:
            checks["compressed_complete"] = False
            details.append(f"MISSING codebook: {cb_key}")
            continue
        if idx_key not in all_keys:
            checks["compressed_complete"] = False
            details.append(f"MISSING indices: {idx_key}")
            continue

        # Validate codebook shape
        cb = f.get_tensor(cb_key)
        vd = mod.get("vector_dim", 1)
        if vd > 1:
            if cb.ndim != 2 or cb.shape[0] != k or cb.shape[1] != vd:
                checks["compressed_complete"] = False
                details.append(f"BAD codebook shape {cb_key}: {cb.shape}, expected [{k}, {vd}]")
        else:
            if cb.ndim != 1 or cb.shape[0] != k:
                checks["compressed_complete"] = False
                details.append(f"BAD codebook shape {cb_key}: {cb.shape}, expected [{k}]")
        del cb

        # Validate indices shape and range
        idx = f.get_tensor(idx_key)
        rows, cols = mod["shape"]
        idx_cols = cols // vd if vd > 1 else cols

        if idx.dtype == torch.uint8 and idx.ndim == 1 and k > 256:
            # 12-bit packed format: validate packed size
            expected_packed = rows * idx_cols * 3 // 2
            if idx.shape[0] != expected_packed:
                checks["compressed_complete"] = False
                details.append(f"BAD packed indices size {idx_key}: {idx.shape[0]}, expected {expected_packed}")
            # Unpack to validate range
            from helix_substrate.index_packing import unpack_12bit
            idx_unpacked = unpack_12bit(idx, rows * idx_cols)
            max_idx = int(idx_unpacked.to(torch.int32).max())
        else:
            # Standard format
            expected_shape = (rows, idx_cols)
            if idx.shape != expected_shape:
                checks["compressed_complete"] = False
                details.append(f"BAD indices shape {idx_key}: {idx.shape}, expected {expected_shape}")
            max_idx = int(idx.to(torch.int32).max())

        if max_idx >= k:
            checks["indices_in_range"] = False
            details.append(f"INDEX OUT OF RANGE {idx_key}: max={max_idx}, codebook_size={k}")
        del idx

        # Sidecar keys (if module has sidecar)
        if mod.get("sidecar", False):
            pos_key = f"{module_path}.sidecar_positions"
            val_key = f"{module_path}.sidecar_values"
            if pos_key not in all_keys:
                checks["compressed_complete"] = False
                details.append(f"MISSING sidecar_positions: {pos_key}")
            if val_key not in all_keys:
                checks["compressed_complete"] = False
                details.append(f"MISSING sidecar_values: {val_key}")

    # 3. Exact tensors — no NaN/Inf
    for tensor_name in exact_tensors:
        if tensor_name not in all_keys:
            checks["exact_no_nan"] = False
            details.append(f"MISSING exact tensor: {tensor_name}")
            continue
        t = f.get_tensor(tensor_name)
        if not torch.isfinite(t).all():
            checks["exact_no_nan"] = False
            n_bad = int((~torch.isfinite(t)).sum())
            details.append(f"NaN/Inf in {tensor_name}: {n_bad} bad values")
        del t

    # 4. Key count
    # Expected: for each compressed module: codebook + indices + optional sidecar_positions + sidecar_values
    expected_keys = 0
    for mod in compressed_modules:
        expected_keys += 2  # codebook + indices
        if mod.get("sidecar", False):
            expected_keys += 2  # sidecar_positions + sidecar_values
    expected_keys += len(exact_tensors)

    if len(all_keys) != expected_keys:
        checks["key_count_match"] = False
        details.append(f"Key count mismatch: safetensors has {len(all_keys)}, expected {expected_keys}")

    # Close the safetensors handle (no explicit close needed, but release ref)
    del f

    # 5. SHA256 (chunked 8MB reads)
    h = hashlib.sha256()
    with open(safetensors_path, "rb") as fh:
        while True:
            chunk = fh.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    sha256_hex = h.hexdigest()

    verdict = "PASS" if all(checks.values()) else "FAIL"

    return {
        "verdict": verdict,
        "validation": {
            "checks": checks,
            "details": details,
            "total_keys": len(all_keys),
            "compressed_modules": len(compressed_modules),
            "exact_tensors": len(exact_tensors),
            "sha256": sha256_hex,
        },
    }


def emit_conversion_receipt(model_name: str, validation_result: dict,
                            output_dir: Path, cost: dict) -> Path:
    """Emit WO-RECEIPT-COST-01 compliant conversion receipt to archive dir."""
    receipt = {
        "work_order": f"convert-hf-{model_name}",
        "question": "Does the HF safetensors pass structural validation?",
        "verdict": validation_result["verdict"],
        "validation": validation_result["validation"],
        "cost": cost,
    }

    # Archive copy
    receipt_dir = Path(__file__).resolve().parent.parent / "receipts" / "convert"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipt_dir / f"{model_name}_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    return receipt_path


def _run_standalone_audit(safetensors_path: Path, dense_model_dir: Path):
    """Run completeness gate as a standalone audit on an existing converted model.

    Usage:
        python3 tools/convert_to_hf.py --audit \
            --safetensors ~/models/foo-helix/model.safetensors \
            --dense-model ~/models/foo
    """
    t_start = time.time()
    cpu_start = time.process_time()

    print(f"{'=' * 70}")
    print(f"  Completeness Gate — Standalone Audit")
    print(f"{'=' * 70}")
    print(f"  Safetensors:  {safetensors_path}")
    print(f"  Dense model:  {dense_model_dir}")
    print()

    if not safetensors_path.exists():
        print(f"  ERROR: {safetensors_path} does not exist", file=sys.stderr)
        sys.exit(1)
    if not dense_model_dir.exists():
        print(f"  ERROR: {dense_model_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    result = validate_completeness_gate(safetensors_path, dense_model_dir)

    # Print summary
    summary = result.get("summary", {})
    print(f"  Dense tensors:       {summary.get('dense_tensors', '?')}")
    print(f"  Output tensors:      {summary.get('output_tensors', '?')}")
    print(f"  Compressed weights:  {summary.get('compressed_weights', '?')}")
    print(f"  Accounted:           {summary.get('accounted', '?')}")
    print(f"  Missing:             {summary.get('missing', '?')}")
    print(f"  Skip tensors found:  {summary.get('skip_tensors_found', '?')}")

    # Print skip category breakdown
    categories = summary.get("skip_categories", {})
    if categories:
        print(f"\n  Skip tensor categories:")
        for cat, status in sorted(categories.items()):
            print(f"    {cat:20s}  {status}")

    # Print failures
    failures = result.get("failures", [])
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f_msg in failures[:30]:
            print(f"    {f_msg}")
        if len(failures) > 30:
            print(f"    ... and {len(failures) - 30} more")

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    print(f"\n  Time: {wall:.1f}s wall, {cpu:.1f}s CPU")

    if result["verdict"] == "PASS":
        print(f"\n  GATE PASS — all {summary.get('dense_tensors', '?')} dense tensors accounted for")
    else:
        print(f"\n  GATE FAIL — {len(failures)} tensors missing or incomplete", file=sys.stderr)

    # Save receipt
    receipt_path = safetensors_path.parent / "audit_completeness_gate.json"
    result["cost"] = {
        "wall_time_s": round(wall, 3),
        "cpu_time_s": round(cpu, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(receipt_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Receipt: {receipt_path}")

    sys.exit(0 if result["verdict"] == "PASS" else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert CDNA v3 to HF safetensors (with completeness gate)",
        epilog="Standalone audit mode:\n"
               "  python3 tools/convert_to_hf.py --audit \\\n"
               "      --safetensors ~/models/foo-helix/model.safetensors \\\n"
               "      --dense-model ~/models/foo\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Conversion mode args
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Model directory containing cdnav3/ subfolder (conversion mode)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for HF-compatible checkpoint (conversion mode)")

    # Audit mode args
    parser.add_argument("--audit", action="store_true",
                        help="Run completeness gate as standalone audit (no conversion)")
    parser.add_argument("--safetensors", type=Path, default=None,
                        help="Path to output .safetensors file (audit mode)")
    parser.add_argument("--dense-model", type=Path, default=None,
                        help="Path to original dense model directory (audit mode)")

    args = parser.parse_args()

    # ── Audit mode ──
    if args.audit:
        if not args.safetensors or not args.dense_model:
            parser.error("--audit requires --safetensors and --dense-model")
        _run_standalone_audit(args.safetensors, args.dense_model)
        return  # _run_standalone_audit calls sys.exit

    # ── Conversion mode ──
    if not args.model_dir or not args.output_dir:
        parser.error("Conversion mode requires --model-dir and --output-dir")

    # Timing instrumentation (WO-RECEIPT-COST-01)
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    print(f"Converting {args.model_dir} → {args.output_dir}")
    compressed_modules, exact_tensors, receipt = convert_cdnav3_to_hf(
        args.model_dir, args.output_dir)

    # Build cost block
    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Emit archive receipt
    model_name = args.model_dir.name
    receipt_path = emit_conversion_receipt(model_name, receipt, args.output_dir, cost)
    print(f"  Receipt: {receipt_path}")

    if receipt["verdict"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
