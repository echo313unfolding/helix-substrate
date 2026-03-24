"""
CDNA v3 model manifest builder.

Scans a model directory for v3 tensor directories and .npy files,
then builds a manifest_v3.json indexing all tensors with their
classes, policies, and aggregate stats.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from helix_substrate.tensor_policy import classify_tensor, TensorClass


def build_manifest(
    model_dir: Path,
    source_info: Optional[dict] = None,
) -> dict:
    """
    Build a v3 model manifest from tensor directories.

    Args:
        model_dir: Directory containing .cdnav3/ dirs and .npy files
        source_info: Optional dict with source_path, sha256, etc.

    Returns:
        Manifest dict (also written to model_dir/manifest_v3.json)
    """
    model_dir = Path(model_dir)
    tensors = {}
    total_original = 0
    total_compressed = 0

    # Scan for .cdnav3 directories
    for td in sorted(model_dir.glob("*.cdnav3")):
        meta_path = td / "meta.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        stats_path = td / "stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

        name = meta["tensor_name"]
        tensors[name] = {
            "format": "cdna_v3",
            "dir": td.name,
            "shape": meta["shape"],
            "tensor_class": meta.get("tensor_class", "unknown"),
            "storage_mode": meta.get("storage_mode", "codebook+sidecar"),
            "has_sidecar": (td / "sidecar.npz").exists(),
            "stats": stats,
        }

        total_original += stats.get("original_bytes", 0)
        total_compressed += stats.get("compressed_bytes", 0)

    # Scan for .npy files (exact tensors)
    import numpy as np
    for npy in sorted(model_dir.glob("*.npy")):
        arr = np.load(npy)
        original_bytes = arr.nbytes
        compressed_bytes = npy.stat().st_size

        # Read companion meta if it exists (written by CDNAv3Writer)
        meta_path = Path(str(npy) + ".meta.json")
        if meta_path.exists():
            npy_meta = json.loads(meta_path.read_text())
            name = npy_meta["tensor_name"]
        else:
            name = npy.stem.replace("_", ".")

        tc = classify_tensor(name, shape=arr.shape)
        tensors[name] = {
            "format": "npy",
            "file": npy.name,
            "shape": list(arr.shape),
            "tensor_class": tc.value,
            "storage_mode": "exact",
            "has_sidecar": False,
            "stats": {
                "original_bytes": original_bytes,
                "compressed_bytes": compressed_bytes,
            },
        }
        total_original += original_bytes
        total_compressed += compressed_bytes

    manifest = {
        "schema": "cdna_v3_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": 3,
        "source": source_info or {},
        "tensor_count": len(tensors),
        "tensors": tensors,
        "stats": {
            "tensor_count": len(tensors),
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "overall_ratio": round(
                total_original / max(1, total_compressed), 2
            ),
        },
    }

    out_path = model_dir / "manifest_v3.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def validate_manifest(manifest_path: Path) -> dict:
    """
    Validate a v3 manifest: check that all referenced files/dirs exist.

    Returns:
        Dict with "valid" (bool), "errors" (list), "tensor_count" (int)
    """
    manifest_path = Path(manifest_path)
    model_dir = manifest_path.parent

    manifest = json.loads(manifest_path.read_text())
    errors = []

    for name, info in manifest.get("tensors", {}).items():
        fmt = info.get("format")
        if fmt == "cdna_v3":
            td = model_dir / info["dir"]
            if not td.exists():
                errors.append(f"Missing directory: {info['dir']}")
            else:
                for required in ("meta.json", "codebook.npy", "indices.bin"):
                    if not (td / required).exists():
                        errors.append(f"{info['dir']}/{required} missing")
        elif fmt == "npy":
            if not (model_dir / info["file"]).exists():
                errors.append(f"Missing file: {info['file']}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "tensor_count": len(manifest.get("tensors", {})),
    }
