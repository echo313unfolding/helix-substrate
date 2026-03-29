"""
Budget-constrained k allocation for CDNA v3 adaptive compression.

Given per-tensor compression stats (cosine, bytes), allocates codebook
resolution (k) per tensor to minimize quality loss within a compression
budget.

Algorithm: Greedy marginal-gain
  1. Start all tensors at k=256
  2. Sort by cosine ascending (worst quality first)
  3. Greedily upgrade worst tensors to k=512 until budget exhausted
  4. Optionally downgrade best tensors to k=128 to recover budget
  5. Output k-map JSON

Usage:
    python3 -m helix_substrate.k_allocator \\
        ~/models/zamba2-1.2b/cdnav3 \\
        --target-ratio 3.5 \\
        --output k_map.json

Work Order: WO-ADAPTIVE-K-QUALITY-01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def _estimate_bytes(original_bytes: int, k: int) -> int:
    """Estimate compressed bytes for a tensor at a given k.

    Layout: codebook (k * 4 bytes) + indices (n_elements * bytes_per_index)
    where bytes_per_index = 1 for k<=256, 2 for k>256.
    """
    # original_bytes = n_elements * 4 (float32)
    n_elements = original_bytes // 4
    bytes_per_index = 2 if k > 256 else 1
    return k * 4 + n_elements * bytes_per_index


def load_tensor_stats(cdna_dir: Path) -> list[dict]:
    """Load stats.json from all .cdnav3 directories.

    Returns list of dicts with keys: tensor_name, cosine, original_bytes,
    compressed_bytes, current_k.
    """
    cdna_dir = Path(cdna_dir)
    stats_list = []

    for tensor_path in sorted(cdna_dir.glob("*.cdnav3")):
        stats_path = tensor_path / "stats.json"
        meta_path = tensor_path / "meta.json"
        if not stats_path.exists():
            continue

        stats = json.loads(stats_path.read_text())

        # Best available cosine
        cosine = stats.get("cosine_with_svd",
                 stats.get("cosine_with_sidecar",
                 stats.get("cosine_no_sidecar", 0)))

        # Current k from meta or stats
        current_k = 256
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            current_k = meta.get("n_clusters", 256)

        stats_list.append({
            "tensor_name": stats["tensor_name"],
            "cosine": cosine,
            "original_bytes": stats.get("original_bytes", 0),
            "compressed_bytes": stats.get("compressed_bytes", 0),
            "current_k": current_k,
            "dir": str(tensor_path),
        })

    return stats_list


def allocate_k(
    tensor_stats: list[dict],
    target_ratio: float = 3.5,
    k_ladder: list[int] = None,
    total_dense_bytes: Optional[int] = None,
    max_upgrade_fraction: float = 0.3,
    allow_downgrade: bool = True,
) -> dict:
    """Allocate codebook size per tensor within a compression budget.

    Args:
        tensor_stats: List of dicts from load_tensor_stats()
        target_ratio: Minimum acceptable compression ratio
        k_ladder: Available k values (default [128, 256, 512])
        total_dense_bytes: Total dense model size. If None, sum from stats.
        max_upgrade_fraction: Max fraction of tensors to upgrade to k=512
        allow_downgrade: If True, downgrade best tensors to k=128 to free budget

    Returns:
        k-map dict ready for JSON serialization
    """
    if k_ladder is None:
        k_ladder = [128, 256, 512]

    if total_dense_bytes is None:
        total_dense_bytes = sum(s["original_bytes"] for s in tensor_stats)

    # Start: all tensors at their current k (typically 256)
    assignments = {}
    for s in tensor_stats:
        assignments[s["tensor_name"]] = s.get("current_k", 256)

    # Calculate current total compressed bytes
    def total_compressed():
        total = 0
        for s in tensor_stats:
            k = assignments[s["tensor_name"]]
            total += _estimate_bytes(s["original_bytes"], k)
        return total

    # Budget: maximum compressed bytes to stay at target_ratio
    budget = total_dense_bytes / target_ratio

    # Sort by cosine ascending (worst quality first)
    sorted_by_quality = sorted(tensor_stats, key=lambda s: s["cosine"])

    max_upgrades = int(len(tensor_stats) * max_upgrade_fraction)
    n_upgraded = 0

    # Phase 1: Upgrade worst-quality tensors to k=512
    for s in sorted_by_quality:
        if n_upgraded >= max_upgrades:
            break

        name = s["tensor_name"]
        current_k = assignments[name]

        # Only upgrade from 256 to 512
        if current_k >= 512:
            continue

        # Check if upgrade fits in budget
        old_bytes = _estimate_bytes(s["original_bytes"], current_k)
        new_bytes = _estimate_bytes(s["original_bytes"], 512)
        delta = new_bytes - old_bytes

        if total_compressed() + delta <= budget:
            assignments[name] = 512
            n_upgraded += 1

    # Phase 2 (optional): Downgrade best-quality tensors to k=128
    if allow_downgrade:
        sorted_by_quality_desc = sorted(tensor_stats, key=lambda s: s["cosine"], reverse=True)
        for s in sorted_by_quality_desc:
            if total_compressed() <= budget:
                break  # Already within budget

            name = s["tensor_name"]
            current_k = assignments[name]

            # Only downgrade from 256 to 128 (never downgrade upgraded tensors)
            if current_k != 256:
                continue

            # Only downgrade if tensor quality is excellent
            if s["cosine"] < 0.9995:
                continue

            assignments[name] = 128

    # Build overrides (only non-default entries)
    default_k = 256
    overrides = {name: k for name, k in assignments.items() if k != default_k}

    estimated_ratio = total_dense_bytes / max(1, total_compressed())

    return {
        "model": "auto",
        "target_ratio": target_ratio,
        "k_default": default_k,
        "overrides": overrides,
        "estimated_ratio": round(estimated_ratio, 2),
        "n_upgraded": sum(1 for k in overrides.values() if k > default_k),
        "n_downgraded": sum(1 for k in overrides.values() if k < default_k),
        "total_tensors": len(tensor_stats),
        "budget_bytes": int(budget),
        "estimated_compressed_bytes": int(total_compressed()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Allocate per-tensor codebook size (k) within a compression budget.",
    )
    parser.add_argument("cdna_dir", type=Path,
                        help="Path to cdnav3/ directory with compressed tensors")
    parser.add_argument("--target-ratio", type=float, default=3.5,
                        help="Minimum compression ratio (default: 3.5)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output k_map.json path (default: cdna_dir/k_map.json)")
    parser.add_argument("--max-upgrade-fraction", type=float, default=0.3,
                        help="Max fraction of tensors to upgrade to k=512 (default: 0.3)")
    parser.add_argument("--no-downgrade", action="store_true",
                        help="Don't downgrade high-quality tensors to k=128")
    args = parser.parse_args()

    cdna_dir = args.cdna_dir.expanduser().resolve()
    assert cdna_dir.is_dir(), f"Not a directory: {cdna_dir}"

    stats = load_tensor_stats(cdna_dir)
    if not stats:
        print(f"No .cdnav3 directories found in {cdna_dir}")
        return

    print(f"Loaded {len(stats)} tensor stats from {cdna_dir}")
    print(f"Target ratio: {args.target_ratio}x")

    k_map = allocate_k(
        stats,
        target_ratio=args.target_ratio,
        max_upgrade_fraction=args.max_upgrade_fraction,
        allow_downgrade=not args.no_downgrade,
    )

    # Print summary
    print(f"\nAllocation results:")
    print(f"  Upgraded to k=512:  {k_map['n_upgraded']} tensors")
    print(f"  Downgraded to k=128: {k_map['n_downgraded']} tensors")
    print(f"  Estimated ratio:     {k_map['estimated_ratio']}x")
    print(f"  Budget:              {k_map['budget_bytes'] / 1e9:.3f} GB")
    print(f"  Estimated size:      {k_map['estimated_compressed_bytes'] / 1e9:.3f} GB")

    if k_map["overrides"]:
        print(f"\nOverrides (top 10 by name):")
        for name, k in sorted(k_map["overrides"].items())[:10]:
            s = next((s for s in stats if s["tensor_name"] == name), {})
            cos = s.get("cosine", 0)
            print(f"  k={k:4d}  cos={cos:.4f}  {name}")
        if len(k_map["overrides"]) > 10:
            print(f"  ... and {len(k_map['overrides']) - 10} more")

    # Write output
    output_path = args.output or (cdna_dir / "k_map.json")
    output_path.write_text(json.dumps(k_map, indent=2))
    print(f"\nk-map written to: {output_path}")


if __name__ == "__main__":
    main()
