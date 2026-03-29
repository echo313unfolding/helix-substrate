#!/usr/bin/env python3
"""
Fixup: copy missing tensors from original model into helix safetensors.

Some models' cdnav3/ dirs don't store exact tensors as .npy files,
so the convert script misses norms, biases, embeddings, etc.
This script identifies what's missing and copies from the original.
"""

import argparse
import glob
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def load_original_tensors(model_dir: Path) -> dict:
    """Load all tensors from the original model (safetensors or pytorch_model.bin)."""
    tensors = {}

    # Try safetensors first
    sf_files = sorted(glob.glob(str(model_dir / "model*.safetensors")))
    if sf_files:
        for sf in sf_files:
            with safe_open(sf, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        return tensors

    # Fall back to pytorch_model.bin
    pt_path = model_dir / "pytorch_model.bin"
    if pt_path.exists():
        return torch.load(pt_path, map_location="cpu", weights_only=True)

    print(f"ERROR: No model files found in {model_dir}", file=sys.stderr)
    sys.exit(1)


def fixup(model_dir: Path, helix_dir: Path):
    """Copy missing tensors from original model into helix safetensors."""
    helix_path = helix_dir / "model.safetensors"
    if not helix_path.exists():
        print(f"ERROR: {helix_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load helix tensors
    helix_tensors = load_file(str(helix_path))

    # Determine which original keys are covered by compressed modules
    helix_modules = set()
    for k in helix_tensors:
        parts = k.rsplit(".", 1)
        if len(parts) == 2 and parts[1] in (
            "codebook", "indices", "sidecar_positions", "sidecar_values",
            "svd_U", "svd_s", "svd_Vt",
        ):
            helix_modules.add(parts[0] + ".weight")

    # Keys already present as exact
    exact_keys = {k for k in helix_tensors if not any(
        k.endswith(s) for s in (".codebook", ".indices", ".sidecar_positions",
                                 ".sidecar_values", ".svd_U", ".svd_s", ".svd_Vt")
    )}

    covered = helix_modules | exact_keys

    # Load original and find missing
    orig_tensors = load_original_tensors(model_dir)
    missing = {k: v for k, v in orig_tensors.items() if k not in covered}

    if not missing:
        print(f"  No missing tensors — {helix_dir.name} is complete.")
        return

    print(f"  Found {len(missing)} missing tensors in {helix_dir.name}:")
    for k in sorted(missing)[:10]:
        print(f"    {k} {list(missing[k].shape)}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")

    # Merge and resave
    merged = {**helix_tensors}
    for k, v in missing.items():
        merged[k] = v.float()  # ensure float32 for compatibility

    save_file(merged, str(helix_path))
    total_mb = helix_path.stat().st_size / 1024**2
    print(f"  Resaved with {len(merged)} total tensors ({total_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Fix missing tensors in helix conversion")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Original model directory")
    parser.add_argument("--helix-dir", type=Path, required=True,
                        help="Helix output directory")
    args = parser.parse_args()
    fixup(args.model_dir, args.helix_dir)


if __name__ == "__main__":
    main()
