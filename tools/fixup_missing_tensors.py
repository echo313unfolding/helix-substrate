#!/usr/bin/env python3
"""
Fixup: copy missing tensors from original model into helix safetensors.

Some models' cdnav3/ dirs don't store exact tensors as .npy files,
so the convert script misses norms, biases, embeddings, etc.
This script identifies what's missing and copies from the original.

Also supports --strip-svd to remove dead SVD artifacts (svd_U, svd_s, svd_Vt)
from existing helix checkpoints. SVD routing was killed 2026-03-27.
"""

import argparse
import glob
import json
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


SVD_SUFFIXES = (".svd_U", ".svd_s", ".svd_Vt")
HELIX_SUFFIXES = (".codebook", ".indices", ".sidecar_positions", ".sidecar_values") + SVD_SUFFIXES


def strip_svd(helix_dir: Path):
    """Remove dead SVD artifact keys from a helix safetensors file."""
    helix_path = helix_dir / "model.safetensors"
    if not helix_path.exists():
        print(f"ERROR: {helix_path} not found", file=sys.stderr)
        sys.exit(1)

    helix_tensors = load_file(str(helix_path))
    svd_keys = [k for k in helix_tensors if any(k.endswith(s) for s in SVD_SUFFIXES)]

    if not svd_keys:
        print(f"  No SVD keys found in {helix_dir.name} — already clean.")
        return

    print(f"  Stripping {len(svd_keys)} dead SVD keys from {helix_dir.name}:")
    for k in sorted(svd_keys)[:10]:
        print(f"    {k} {list(helix_tensors[k].shape)}")
    if len(svd_keys) > 10:
        print(f"    ... and {len(svd_keys) - 10} more")

    cleaned = {k: v for k, v in helix_tensors.items() if k not in set(svd_keys)}
    old_mb = helix_path.stat().st_size / 1024**2
    save_file(cleaned, str(helix_path))
    new_mb = helix_path.stat().st_size / 1024**2
    print(f"  Resaved: {len(cleaned)} tensors ({new_mb:.1f} MB, was {old_mb:.1f} MB, saved {old_mb - new_mb:.1f} MB)")


EMBEDDING_PATTERNS = ("embed_tokens", "embed_positions", "wte", "wpe",
                      "backbone.embedding", "backbone.embeddings")


def _find_vqd_embeddings(helix_tensors: dict) -> list[str]:
    """Find embedding modules that were VQ'd (codebook+indices) but should be exact."""
    vqd = []
    for k in helix_tensors:
        if not k.endswith(".codebook"):
            continue
        module = k[:-len(".codebook")]
        if any(pat in module for pat in EMBEDDING_PATTERNS):
            vqd.append(module)
    return sorted(vqd)


def fixup(model_dir: Path, helix_dir: Path, do_strip_svd: bool = False,
          fix_embeddings: bool = False):
    """Copy missing tensors from original model into helix safetensors."""
    helix_path = helix_dir / "model.safetensors"
    if not helix_path.exists():
        print(f"ERROR: {helix_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load helix tensors
    helix_tensors = load_file(str(helix_path))
    modified = False

    # Optionally strip SVD first
    if do_strip_svd:
        svd_keys = [k for k in helix_tensors if any(k.endswith(s) for s in SVD_SUFFIXES)]
        if svd_keys:
            print(f"  Stripping {len(svd_keys)} dead SVD keys...")
            helix_tensors = {k: v for k, v in helix_tensors.items() if k not in set(svd_keys)}
            modified = True

    # Fix VQ'd embeddings: replace codebook/indices with exact weight from original
    vqd_embeddings = _find_vqd_embeddings(helix_tensors) if fix_embeddings else []
    if vqd_embeddings:
        orig_tensors = load_original_tensors(model_dir)
        for module in vqd_embeddings:
            weight_key = module + ".weight"
            if weight_key not in orig_tensors:
                # Try singular/plural variation (backbone.embeddings vs backbone.embedding)
                if module.endswith("s"):
                    alt_key = module[:-1] + ".weight"
                else:
                    alt_key = module + "s.weight"
                if alt_key in orig_tensors:
                    weight_key = alt_key
                else:
                    print(f"  WARNING: {module}.weight not found in original (tried {alt_key} too) — skipping")
                    continue
            # Remove VQ keys for this embedding
            for suffix in (".codebook", ".indices", ".sidecar_positions", ".sidecar_values"):
                vq_key = module + suffix
                if vq_key in helix_tensors:
                    del helix_tensors[vq_key]
            # Insert exact embedding
            helix_tensors[weight_key] = orig_tensors[weight_key].float()
            print(f"  Fixed embedding: {module} (VQ → exact, {list(orig_tensors[weight_key].shape)})")
            modified = True

        # Patch config.json: remove fixed embeddings from compressed_modules
        config_path = helix_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            qcfg = config.get("quantization_config", {})
            old_modules = qcfg.get("compressed_modules", [])
            new_modules = [m for m in old_modules if m not in set(vqd_embeddings)]
            if len(new_modules) < len(old_modules):
                qcfg["compressed_modules"] = new_modules
                config["quantization_config"] = qcfg
                config_path.write_text(json.dumps(config, indent=2))
                removed = len(old_modules) - len(new_modules)
                print(f"  Patched config.json: removed {removed} embedding(s) from compressed_modules")

    # Determine which original keys are covered by compressed modules
    helix_modules = set()
    for k in helix_tensors:
        parts = k.rsplit(".", 1)
        if len(parts) == 2 and parts[1] in (
            "codebook", "indices", "sidecar_positions", "sidecar_values",
        ):
            helix_modules.add(parts[0] + ".weight")

    # Keys already present as exact
    exact_keys = {k for k in helix_tensors if not any(
        k.endswith(s) for s in HELIX_SUFFIXES
    )}

    covered = helix_modules | exact_keys

    # Load original and find missing
    if not vqd_embeddings:
        orig_tensors = load_original_tensors(model_dir)
    missing = {k: v for k, v in orig_tensors.items() if k not in covered}

    if not missing and not modified:
        print(f"  No missing tensors — {helix_dir.name} is complete.")
        return

    if not missing and modified:
        save_file(helix_tensors, str(helix_path))
        total_mb = helix_path.stat().st_size / 1024**2
        print(f"  Resaved: {len(helix_tensors)} tensors ({total_mb:.1f} MB)")
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
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Original model directory (required unless --strip-svd-only)")
    parser.add_argument("--helix-dir", type=Path, required=True,
                        help="Helix output directory")
    parser.add_argument("--strip-svd", action="store_true",
                        help="Remove dead SVD keys (svd_U, svd_s, svd_Vt) from checkpoint")
    parser.add_argument("--fix-embeddings", action="store_true",
                        help="Replace VQ'd embeddings with exact from original model")
    parser.add_argument("--strip-svd-only", action="store_true",
                        help="Only strip SVD keys, don't fix missing tensors (no --model-dir needed)")
    args = parser.parse_args()

    if args.strip_svd_only:
        strip_svd(args.helix_dir)
        return

    if args.model_dir is None:
        parser.error("--model-dir is required unless using --strip-svd-only")

    fixup(args.model_dir, args.helix_dir, do_strip_svd=args.strip_svd,
          fix_embeddings=args.fix_embeddings)


if __name__ == "__main__":
    main()
