"""Convert CDNA v3 directory format to a single HuggingFace-compatible safetensors file.

Reads all .cdnav3 directories and .npy exact tensors from a CDNA v3 output directory,
packs them into a safetensors file with a naming convention that HelixHfQuantizer
can load back into HelixLinear modules.

Naming convention inside safetensors:
    Compressed tensors get multiple keys per original tensor:
        {name}.codebook     → [256] float32
        {name}.indices      → [out, in] uint8
        {name}.sidecar_positions → [N] int64  (optional)
        {name}.sidecar_values    → [N] float32 (optional)
        {name}.svd_U        → [out, rank] float32 (optional)
        {name}.svd_s        → [rank] float32 (optional)
        {name}.svd_Vt       → [rank, in] float32 (optional)
        {name}.channel_scales → [in] float32 (optional)

    Exact tensors (norms, embeddings) keep their original name.

Also writes quantization_config into config.json for HF AutoModel loading.

Usage:
    python -m helix_substrate.convert_safetensors \\
        --cdnav3-dir ~/models/zamba2-1.2b/cdnav3/ \\
        --model-dir ~/models/zamba2-1.2b/ \\
        --output-dir ~/models/zamba2-1.2b-helix/
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from safetensors.numpy import save_file as np_save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import torch
    from safetensors.torch import save_file as torch_save_file
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _load_cdnav3_tensors(tensor_dir: Path) -> Dict[str, np.ndarray]:
    """Load all components of a .cdnav3 directory as numpy arrays.

    Returns dict with keys like:
        "model.layers.0.mamba.in_proj.codebook" → np.ndarray
        "model.layers.0.mamba.in_proj.indices"  → np.ndarray
        etc.
    """
    meta = json.loads((tensor_dir / "meta.json").read_text())
    tensor_name = meta["tensor_name"]
    rows, cols = meta["shape"]

    # Strip .weight suffix for module-level naming
    base_name = tensor_name
    if base_name.endswith(".weight"):
        base_name = base_name[:-7]

    result = {}

    # Codebook: [256] float32
    codebook = np.load(tensor_dir / "codebook.npy").astype(np.float32)
    result[f"{base_name}.codebook"] = codebook

    # Indices: [rows, cols] uint8
    raw = np.fromfile(tensor_dir / "indices.bin", dtype=np.uint8)
    indices = raw.reshape(rows, cols)
    result[f"{base_name}.indices"] = indices

    # Sidecar: optional
    sidecar_path = tensor_dir / "sidecar.npz"
    if sidecar_path.exists():
        sidecar = np.load(sidecar_path)
        if "positions" in sidecar and sidecar["positions"].size > 0:
            result[f"{base_name}.sidecar_positions"] = sidecar["positions"].astype(np.int64)
            result[f"{base_name}.sidecar_values"] = sidecar["values"].astype(np.float32)

    # SVD residual
    if (tensor_dir / "svd_U.npy").exists():
        result[f"{base_name}.svd_U"] = np.load(tensor_dir / "svd_U.npy").astype(np.float32)
        result[f"{base_name}.svd_s"] = np.load(tensor_dir / "svd_s.npy").astype(np.float32)
        result[f"{base_name}.svd_Vt"] = np.load(tensor_dir / "svd_Vt.npy").astype(np.float32)

    # Channel scales
    if (tensor_dir / "channel_scales.npy").exists():
        result[f"{base_name}.channel_scales"] = np.load(
            tensor_dir / "channel_scales.npy"
        ).astype(np.float32)

    return result


def _load_exact_tensor(npy_path: Path) -> tuple[str, np.ndarray]:
    """Load an exact (non-compressed) tensor from .npy + .meta.json."""
    meta_path = npy_path.parent / f"{npy_path.name}.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        tensor_name = meta.get("tensor_name", npy_path.stem)
    else:
        # Reconstruct name from filename
        tensor_name = npy_path.stem.replace("_", ".")
    return tensor_name, np.load(npy_path)


def _load_missing_from_original(
    all_tensors: Dict[str, np.ndarray],
    original_safetensors: Path,
    compressed_modules: list[str],
) -> int:
    """Load tensors from the original model that aren't in the CDNA v3 output.

    The compressor only exports weight matrices (as .cdnav3) and 1D tensors
    (as .npy). Architecture-specific non-weight tensors like Mamba's A_log,
    D, dt_bias are not exported. This function fills the gaps.

    Skips keys that belong to compressed modules (their .weight is already
    represented as .codebook + .indices).

    Returns number of tensors added.
    """
    import torch
    from safetensors.torch import load_file

    # Build set of module paths whose .weight is compressed
    compressed_weight_keys = {f"{m}.weight" for m in compressed_modules}

    original = load_file(str(original_safetensors))
    added = 0

    for key, tensor in original.items():
        # Skip if already present (exact tensors, norms, etc.)
        if key in all_tensors:
            continue
        # Skip compressed weight tensors (stored as codebook+indices instead)
        if key in compressed_weight_keys:
            continue
        # Convert to float32 numpy
        all_tensors[key] = tensor.float().numpy()
        added += 1

    return added


def convert_cdnav3_to_safetensors(
    cdnav3_dir: Path,
    output_path: Path,
    original_model_path: Optional[Path] = None,
) -> dict:
    """Convert a CDNA v3 directory to a safetensors file.

    Args:
        cdnav3_dir: Path to directory containing .cdnav3/ dirs and .npy files
        output_path: Path for the output safetensors file
        original_model_path: Optional path to original model.safetensors
            (to pull architecture-specific params not in CDNA v3 output)

    Returns:
        Conversion stats dict.
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors is required: pip install safetensors")

    cdnav3_dir = Path(cdnav3_dir)
    output_path = Path(output_path)

    all_tensors: Dict[str, np.ndarray] = {}
    n_compressed = 0
    n_exact = 0
    n_from_original = 0
    compressed_names = []

    # Load manifest for metadata
    manifest_path = cdnav3_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    # 1. Load all .cdnav3 compressed tensors
    for tensor_path in sorted(cdnav3_dir.glob("*.cdnav3")):
        if not tensor_path.is_dir():
            continue
        meta_path = tensor_path / "meta.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        if meta.get("storage_mode") == "exact":
            continue

        tensors = _load_cdnav3_tensors(tensor_path)
        all_tensors.update(tensors)
        n_compressed += 1

        tensor_name = meta["tensor_name"]
        base = tensor_name[:-7] if tensor_name.endswith(".weight") else tensor_name
        compressed_names.append(base)

    # 2. Load all .npy exact tensors (norms, embeddings, biases)
    for npy_path in sorted(cdnav3_dir.glob("*.npy")):
        name, data = _load_exact_tensor(npy_path)
        all_tensors[name] = data.astype(np.float32) if data.dtype != np.float32 else data
        n_exact += 1

    # 3. Pull missing tensors from original model (if available)
    if original_model_path is not None and original_model_path.exists():
        n_from_original = _load_missing_from_original(
            all_tensors, original_model_path, compressed_names
        )

    # 4. Save as safetensors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np_save_file(all_tensors, str(output_path))

    total_bytes = output_path.stat().st_size

    stats = {
        "n_compressed": n_compressed,
        "n_exact": n_exact,
        "n_from_original": n_from_original,
        "n_total_keys": len(all_tensors),
        "output_bytes": total_bytes,
        "output_mb": round(total_bytes / 1024 / 1024, 1),
        "compressed_modules": compressed_names,
    }

    return stats


def write_quantization_config(
    model_config_path: Path,
    output_config_path: Path,
    compressed_modules: list[str],
    manifest: Optional[dict] = None,
) -> None:
    """Add quantization_config to a model's config.json.

    Args:
        model_config_path: Original config.json
        output_config_path: Where to write updated config
        compressed_modules: List of module paths that are HelixLinear
        manifest: Optional CDNA v3 manifest for extra metadata
    """
    config = json.loads(model_config_path.read_text())

    quant_config = {
        "quant_method": "cdna_v3",
        "bits": 8,  # uint8 indices
        "n_clusters": 256,
        "compressed_modules": compressed_modules,
    }
    if manifest:
        quant_config["compression_ratio"] = manifest.get("compression_ratio", 4.0)
        quant_config["n_svd_routed"] = manifest.get("n_svd_routed", 0)

    config["quantization_config"] = quant_config

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(json.dumps(config, indent=2) + "\n")


def convert_model(
    cdnav3_dir: str,
    model_dir: str,
    output_dir: str,
) -> dict:
    """Full conversion: CDNA v3 dir → HF-compatible directory with safetensors + config.

    Args:
        cdnav3_dir: Path to CDNA v3 output directory
        model_dir: Path to original model directory (for config.json, tokenizer, etc.)
        output_dir: Path for output HF-compatible directory

    Returns:
        Conversion stats.
    """
    cdnav3_path = Path(cdnav3_dir)
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1. Convert tensors to safetensors
    # Find original model safetensors to pull non-weight params (Mamba A_log, D, etc.)
    original_st = None
    for candidate in [
        model_path / "model.safetensors",
        model_path / "model-00001-of-00001.safetensors",
    ]:
        if candidate.exists():
            original_st = candidate
            break
    if original_st is None:
        # Multi-shard: find all shards
        shards = sorted(model_path.glob("model-*.safetensors"))
        if shards:
            original_st = shards[0]  # First shard; we'll handle multi-shard later

    safetensors_path = output_path / "model.safetensors"
    stats = convert_cdnav3_to_safetensors(cdnav3_path, safetensors_path, original_st)

    # 2. Write config.json with quantization_config
    config_src = model_path / "config.json"
    if config_src.exists():
        manifest = {}
        manifest_path = cdnav3_path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

        write_quantization_config(
            config_src,
            output_path / "config.json",
            stats["compressed_modules"],
            manifest,
        )
    else:
        print(f"Warning: no config.json found at {config_src}")

    # 3. Copy tokenizer files
    for pattern in [
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "special_tokens_map.json", "added_tokens.json",
        "generation_config.json",
    ]:
        src = model_path / pattern
        if src.exists():
            shutil.copy2(src, output_path / pattern)

    # 4. Copy manifest
    manifest_src = cdnav3_path / "manifest.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, output_path / "cdnav3_manifest.json")

    stats["wall_time_s"] = round(time.time() - t0, 2)
    stats["output_dir"] = str(output_path)

    # Write conversion receipt
    receipt_path = output_path / "conversion_receipt.json"
    receipt_path.write_text(json.dumps(stats, indent=2) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert CDNA v3 compressed model to HF safetensors format"
    )
    parser.add_argument(
        "--cdnav3-dir", required=True,
        help="Path to CDNA v3 output directory (containing .cdnav3/ dirs)",
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to original model directory (for config.json, tokenizer)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Path for output HF-compatible directory",
    )
    args = parser.parse_args()

    stats = convert_model(args.cdnav3_dir, args.model_dir, args.output_dir)
    print(f"Conversion complete:")
    print(f"  Compressed tensors:    {stats['n_compressed']}")
    print(f"  Exact tensors (npy):   {stats['n_exact']}")
    print(f"  From original model:   {stats.get('n_from_original', 0)}")
    print(f"  Total keys:            {stats['n_total_keys']}")
    print(f"  Output size:           {stats['output_mb']} MB")
    print(f"  Time:                  {stats['wall_time_s']}s")
    print(f"  Output:                {stats['output_dir']}")


if __name__ == "__main__":
    main()
