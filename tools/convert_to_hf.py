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
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
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

        # Codebook: [256] float32
        codebook = np.load(tensor_path / "codebook.npy").astype(np.float32)
        tensors[f"{module_path}.codebook"] = torch.from_numpy(codebook)

        # Indices: [rows, cols] uint8
        raw_indices = np.fromfile(tensor_path / "indices.bin", dtype=np.uint8)
        indices = raw_indices.reshape(rows, cols)
        tensors[f"{module_path}.indices"] = torch.from_numpy(indices.copy())

        # Sidecar: optional outlier corrections
        sidecar_path = tensor_path / "sidecar.npz"
        if sidecar_path.exists():
            sidecar_data = np.load(sidecar_path)
            positions = sidecar_data["positions"].astype(np.int64)
            values = sidecar_data["values"].astype(np.float32)
            tensors[f"{module_path}.sidecar_positions"] = torch.from_numpy(positions.copy())
            tensors[f"{module_path}.sidecar_values"] = torch.from_numpy(values.copy())

        # SVD residual factors: optional (dead at all scales, but support for completeness)
        if (tensor_path / "svd_U.npy").exists():
            tensors[f"{module_path}.svd_U"] = torch.from_numpy(
                np.load(tensor_path / "svd_U.npy").astype(np.float32).copy()
            )
            tensors[f"{module_path}.svd_s"] = torch.from_numpy(
                np.load(tensor_path / "svd_s.npy").astype(np.float32).copy()
            )
            tensors[f"{module_path}.svd_Vt"] = torch.from_numpy(
                np.load(tensor_path / "svd_Vt.npy").astype(np.float32).copy()
            )

        compressed_modules.append({
            "module": module_path,
            "shape": [rows, cols],
            "n_clusters": meta.get("n_clusters", 256),
            "sidecar": sidecar_path.exists(),
            "svd_rank": meta.get("svd_residual_rank", 0),
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

        data = np.load(npy_path).astype(np.float32)
        tensors[tensor_name] = torch.from_numpy(data.copy())
        exact_tensors.append(tensor_name)

    # Save safetensors
    safetensors_path = output_dir / "model.safetensors"
    save_file(tensors, str(safetensors_path))
    print(f"  Saved {len(tensors)} tensors to {safetensors_path}")
    print(f"    {len(compressed_modules)} compressed modules")
    print(f"    {len(exact_tensors)} exact tensors")

    # Copy original config.json and add quantization_config
    orig_config_path = model_dir / "config.json"
    if orig_config_path.exists():
        config = json.loads(orig_config_path.read_text())
    else:
        config = {}

    config["quantization_config"] = {
        "quant_method": "helix",
        "codebook_size": 256,
        "sidecar_enabled": True,
        "exact_patterns": ["embed_tokens", "embed_positions", "wte", "wpe",
                           "lm_head", "layernorm", "layer_norm", "norm",
                           "backbone.embedding"],
        "compressed_modules": [m["module"] for m in compressed_modules],
    }

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

    return compressed_modules, exact_tensors


def main():
    parser = argparse.ArgumentParser(description="Convert CDNA v3 to HF safetensors")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Model directory containing cdnav3/ subfolder")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for HF-compatible checkpoint")
    args = parser.parse_args()

    print(f"Converting {args.model_dir} → {args.output_dir}")
    convert_cdnav3_to_hf(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
