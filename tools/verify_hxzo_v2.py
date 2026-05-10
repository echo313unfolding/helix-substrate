#!/usr/bin/env python3
"""
HXZO v2 routing_context round-trip verification.

Done condition (WO-HXZO-V2-ROUTING-CONTEXT):
  Compress a tensor, decompress it, read back routing_context.eff_rank
  and routing_context.pre_sidecar_norm, confirm they match what was
  computed at compression time. Emit receipt.

Also verifies:
  - v1 backward compat (write v1, read with v2 reader)
  - v2 header inspection without payload decompression
  - build_routing_context() produces valid dict
"""

import json
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Ensure helix_substrate is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix_substrate.sidecar import (
    build_routing_context,
    clear_sidecar_cache,
    inspect_hxzo_header,
    read_outlier_sidecar,
    write_outlier_sidecar,
    HXZO_VERSION,
)


def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    rng = np.random.default_rng(42)
    clear_sidecar_cache()

    # ── Simulate compression-time signals ──
    # These would come from RouteScore + weight measurements at compress time
    eff_rank_at_compress = 0.4217
    pre_sidecar_norm_at_compress = 0.003142
    post_sidecar_norm_at_compress = 0.000821

    routing_ctx = build_routing_context(
        tensor_class="FFN",
        block_type="ssm",
        role="in_proj",
        block_idx=6,
        arch="zamba2",
        eff_rank=eff_rank_at_compress,
        kurtosis=3.7,
        se=0.81,
        weight_rms=0.0023,
        route="VQ_ONLY",
        composite_score=2.1,
        confidence=1.0 - np.mean([pre_sidecar_norm_at_compress, post_sidecar_norm_at_compress]),
        pre_sidecar_norm=pre_sidecar_norm_at_compress,
        post_sidecar_norm=post_sidecar_norm_at_compress,
        compression_step=0,
        drift_ratio=None,
        recomp_count=0,
    )

    # ── Test 1: v2 round-trip with routing_context ──
    n_outliers = 500
    positions = np.sort(rng.choice(100000, n_outliers, replace=False)).astype(np.int64)
    values = rng.standard_normal(n_outliers).astype(np.float32)
    threshold_policy = {"method": "percentile", "percentile": 99.9}

    with tempfile.NamedTemporaryFile(suffix=".hxzo", delete=False) as f:
        v2_path = f.name

    receipt_write = write_outlier_sidecar(
        positions=positions,
        values=values,
        tensor_name="model.layers.6.mamba.in_proj.weight",
        threshold_policy=threshold_policy,
        shape=(5120, 2560),
        output_path=v2_path,
        routing_context=routing_ctx,
    )

    # Read back
    pos_read, val_read, meta = read_outlier_sidecar(v2_path, use_cache=False)

    # Verify positions (exact)
    assert np.array_equal(positions, pos_read), "FAIL: positions mismatch"

    # Verify routing_context exists and fields match
    rc = meta["routing_context"]
    assert rc is not None, "FAIL: routing_context missing from read metadata"

    # DONE CONDITION: eff_rank and pre_sidecar_norm match
    assert abs(rc["eff_rank"] - eff_rank_at_compress) < 1e-5, (
        f"FAIL: eff_rank mismatch: wrote {eff_rank_at_compress}, read {rc['eff_rank']}"
    )
    assert abs(rc["pre_sidecar_norm"] - pre_sidecar_norm_at_compress) < 1e-5, (
        f"FAIL: pre_sidecar_norm mismatch: wrote {pre_sidecar_norm_at_compress}, read {rc['pre_sidecar_norm']}"
    )

    # Verify all 17 fields present
    expected_fields = {
        "tensor_class", "block_type", "role", "block_idx", "arch",
        "eff_rank", "kurtosis", "se", "weight_rms",
        "route", "composite_score", "confidence",
        "pre_sidecar_norm", "post_sidecar_norm",
        "compression_step", "drift_ratio", "recomp_count",
    }
    actual_fields = set(rc.keys())
    assert expected_fields == actual_fields, (
        f"FAIL: field mismatch. Missing: {expected_fields - actual_fields}, "
        f"Extra: {actual_fields - expected_fields}"
    )

    # Verify specific values
    assert rc["tensor_class"] == "FFN"
    assert rc["block_type"] == "ssm"
    assert rc["route"] == "VQ_ONLY"
    assert rc["block_idx"] == 6
    assert rc["arch"] == "zamba2"
    assert rc["drift_ratio"] is None
    assert rc["recomp_count"] == 0

    print("PASS: v2 round-trip — routing_context.eff_rank and .pre_sidecar_norm match")

    # ── Test 2: Header inspection (no payload decompression) ──
    hdr = inspect_hxzo_header(v2_path)
    assert hdr["valid"] is True
    assert hdr["version"] == 2
    assert "routing_context" in hdr
    assert hdr["routing_context"]["eff_rank"] == rc["eff_rank"]
    assert hdr["schema"] == "hxzo_outlier_sidecar_v2"
    print("PASS: v2 header inspection — routing_context readable without decompression")

    # ── Test 3: v1 backward compatibility ──
    with tempfile.NamedTemporaryFile(suffix=".hxzo", delete=False) as f:
        v1_path = f.name

    clear_sidecar_cache()

    # Write without routing_context → should produce v1 schema string
    write_outlier_sidecar(
        positions=positions,
        values=values,
        tensor_name="test_v1_compat",
        threshold_policy=threshold_policy,
        shape=(100, 1000),
        output_path=v1_path,
    )

    pos_v1, val_v1, meta_v1 = read_outlier_sidecar(v1_path, use_cache=False)
    assert np.array_equal(positions, pos_v1), "FAIL: v1 compat positions mismatch"
    assert meta_v1["routing_context"] is None, "FAIL: v1 should have no routing_context"

    hdr_v1 = inspect_hxzo_header(v1_path)
    assert hdr_v1["schema"] == "hxzo_outlier_sidecar_v1"
    print("PASS: v1 backward compatibility — no routing_context, reads clean")

    # Cleanup
    Path(v2_path).unlink()
    Path(v1_path).unlink()

    # ── Receipt ──
    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    receipt = {
        "schema": "hxzo_v2_routing_context_verification",
        "work_order": "WO-HXZO-V2-ROUTING-CONTEXT",
        "hxzo_version_written": HXZO_VERSION,
        "tests_passed": [
            "v2_roundtrip_routing_context",
            "eff_rank_match",
            "pre_sidecar_norm_match",
            "all_17_fields_present",
            "header_inspection_no_decompress",
            "v1_backward_compatibility",
        ],
        "done_condition": {
            "eff_rank_written": eff_rank_at_compress,
            "eff_rank_read": rc["eff_rank"],
            "pre_sidecar_norm_written": pre_sidecar_norm_at_compress,
            "pre_sidecar_norm_read": rc["pre_sidecar_norm"],
            "match": True,
        },
        "routing_context_written": routing_ctx,
        "routing_context_read": rc,
        "cost": cost,
    }

    # Write receipt
    receipt_dir = Path(__file__).resolve().parent.parent / "receipts" / "hxzo_v2"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"hxzo_v2_verification_{time.strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))

    print(f"\nReceipt: {receipt_path}")
    print(f"Cost: {cost['wall_time_s']}s wall, {cost['peak_memory_mb']} MB peak")
    print("\nALL TESTS PASSED — HXZO v2 routing_context round-trip verified")


if __name__ == "__main__":
    main()
