#!/usr/bin/env python3
"""
AF6 codec parity check: Python probe vs production C codec.

Compares the Python AF6 reconstruction (FP32 scale/offset) against a
faithful Python reimplementation of the C codec (FP16 scale/offset).

Scope: only the 7 tensors with probe affine6_cosine < 0.9995.
Decision gate: if any production-codec cosine < 0.999, Hydra has a real split.

Work Order: GAP-HXQ-009 parity verification
"""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import resource
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODEL_DIR = Path.home() / "models" / "qwen2.5-coder-3b-instruct-dense"
PROBE_RECEIPT = (
    Path(__file__).parent.parent
    / "receipts"
    / "hxq_af6_probe_qwen3b"
    / "af6_probe_qwen3b_20261027T101245.json"
)
QK = 128  # group size
K = 64    # 6-bit: 0..63


def fp32_to_fp16(v: float) -> float:
    """Round-trip FP32 → FP16 → FP32, matching GGML_FP32_TO_FP16/FP16_TO_FP32."""
    packed = struct.pack("e", np.float16(v))
    return float(struct.unpack("e", packed)[0])


def af6_encode_decode_probe(tensor_2d: np.ndarray) -> np.ndarray:
    """Python probe method: FP32 scale/offset, same as bench_hxq_qwen_af6_probe.py."""
    flat = tensor_2d.ravel().astype(np.float32)
    n = len(flat)
    pad = (QK - n % QK) % QK
    if pad > 0:
        padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    else:
        padded = flat.copy()

    n_groups = len(padded) // QK
    groups = padded.reshape(n_groups, QK)
    g_min = groups.min(axis=1, keepdims=True)
    g_max = groups.max(axis=1, keepdims=True)
    g_range = np.maximum(g_max - g_min, 1e-10)

    normalized = (groups - g_min) / g_range
    indices = np.clip(np.round(normalized * (K - 1)), 0, K - 1).astype(np.int32)
    recon = g_min + indices.astype(np.float32) / (K - 1) * g_range
    return recon.ravel()[:n].reshape(tensor_2d.shape)


def af6_encode_decode_codec(tensor_2d: np.ndarray) -> np.ndarray:
    """Faithful reimplementation of C codec: FP16 scale/offset, C-style rounding."""
    flat = tensor_2d.ravel().astype(np.float32)
    n = len(flat)
    pad = (QK - n % QK) % QK
    if pad > 0:
        padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    else:
        padded = flat.copy()

    n_groups = len(padded) // QK
    recon = np.empty_like(padded)

    for i in range(n_groups):
        group = padded[i * QK:(i + 1) * QK]
        vmin = float(group.min())
        vmax = float(group.max())

        range_f = vmax - vmin
        scale_f32 = range_f / 63.0
        inv_scale = (1.0 / scale_f32) if scale_f32 > 1e-10 else 0.0

        # FP16 round-trip (the critical difference)
        scale_f16 = fp32_to_fp16(scale_f32)
        offset_f16 = fp32_to_fp16(vmin)

        # Quantize indices using FP32 inv_scale (matches C code)
        for j in range(QK):
            v = float((group[j] - vmin) * inv_scale)
            v = max(0.0, min(63.0, v))
            idx = int(v + 0.5)  # C-style: (uint8_t)(v + 0.5f)
            # Reconstruct using FP16 scale/offset (matches dequantize)
            recon[i * QK + j] = idx * scale_f16 + offset_f16

    return recon[:n].reshape(tensor_2d.shape)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.ravel().astype(np.float64)
    b64 = b.ravel().astype(np.float64)
    dot = np.dot(a64, b64)
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))


def nmse(orig: np.ndarray, recon: np.ndarray) -> float:
    o = orig.ravel().astype(np.float64)
    r = recon.ravel().astype(np.float64)
    return float(np.mean((o - r) ** 2) / np.mean(o ** 2))


def main():
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    cpu_start = time.process_time()

    # Load probe receipt and select tensors below 0.9995
    probe = json.loads(PROBE_RECEIPT.read_text())
    records = probe["phase1"]["tensor_records"]
    targets = []
    for r in records:
        cos = r["strategies"]["affine6_g128"]["cosine"]
        if cos < 0.9995:
            targets.append((r["name"], cos))

    targets.sort(key=lambda x: x[1])
    print(f"Selected {len(targets)} tensors with probe cosine < 0.9995:")
    for name, cos in targets:
        print(f"  {cos:.6f}  {name}")
    print()

    # Load tensors from dense source
    from safetensors import safe_open

    shards = sorted(MODEL_DIR.glob("model*.safetensors"))
    shard_map: dict[str, Path] = {}
    for shard_path in shards:
        with safe_open(shard_path, framework="pt") as sf:
            for name in sf.keys():
                shard_map[name] = shard_path

    results = []
    for tname, probe_cos in targets:
        shard_path = shard_map[tname]
        with safe_open(str(shard_path), framework="pt") as sf:
            W = sf.get_tensor(tname).float().numpy()

        recon_probe = af6_encode_decode_probe(W)
        recon_codec = af6_encode_decode_codec(W)

        cos_probe = cosine_sim(W, recon_probe)
        cos_codec = cosine_sim(W, recon_codec)
        cos_ab = cosine_sim(recon_probe, recon_codec)

        diff_ab = np.abs(recon_probe.ravel() - recon_codec.ravel())
        diff_w_probe = np.abs(W.ravel() - recon_probe.ravel())
        diff_w_codec = np.abs(W.ravel() - recon_codec.ravel())

        entry = {
            "tensor": tname,
            "shape": list(W.shape),
            "n_params": int(np.prod(W.shape)),
            "probe_cosine": round(cos_probe, 7),
            "codec_cosine": round(cos_codec, 7),
            "probe_vs_codec_cosine": round(cos_ab, 9),
            "probe_max_abs_error": round(float(diff_w_probe.max()), 7),
            "codec_max_abs_error": round(float(diff_w_codec.max()), 7),
            "probe_mean_abs_error": round(float(diff_w_probe.mean()), 7),
            "codec_mean_abs_error": round(float(diff_w_codec.mean()), 7),
            "probe_nmse": round(nmse(W, recon_probe), 9),
            "codec_nmse": round(nmse(W, recon_codec), 9),
            "ab_max_abs_diff": round(float(diff_ab.max()), 7),
            "ab_mean_abs_diff": round(float(diff_ab.mean()), 7),
            "codec_above_0999": cos_codec >= 0.999,
        }
        results.append(entry)

        print(f"{tname}")
        print(f"  probe cos: {cos_probe:.7f}  codec cos: {cos_codec:.7f}  "
              f"delta: {cos_codec - cos_probe:+.2e}")
        print(f"  A vs B cos: {cos_ab:.9f}  max|A-B|: {diff_ab.max():.2e}  "
              f"mean|A-B|: {diff_ab.mean():.2e}")
        print(f"  codec >= 0.999: {cos_codec >= 0.999}")
        print()

        del W, recon_probe, recon_codec
        gc.collect()

    # Decision
    all_above = all(r["codec_above_0999"] for r in results)
    any_below = any(not r["codec_above_0999"] for r in results)
    worst_codec = min(r["codec_cosine"] for r in results)
    worst_tensor = min(results, key=lambda r: r["codec_cosine"])["tensor"]

    if all_above:
        decision = "NO_ROUTE_DIVERSITY"
        decision_detail = (
            f"All {len(results)} near-threshold tensors have codec cosine >= 0.999. "
            f"Worst: {worst_codec:.7f} ({worst_tensor}). "
            "quality_first candidate is semantically uniform AF6."
        )
    else:
        below = [r for r in results if not r["codec_above_0999"]]
        decision = "ROUTE_DIVERSITY_EXISTS"
        decision_detail = (
            f"{len(below)} tensor(s) have codec cosine < 0.999. "
            f"Worst: {worst_codec:.7f} ({worst_tensor}). "
            "Regenerate Hydra plan using production-codec measurements."
        )

    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "numpy_version": np.__version__,
    }

    receipt = {
        "gate": "GAP-HXQ-009-QWEN-AF6-CODEC-PARITY",
        "work_order": "GAP-HXQ-009",
        "scope": "7 tensors with probe affine6_cosine < 0.9995",
        "probe_receipt_sha256": hashlib.sha256(PROBE_RECEIPT.read_bytes()).hexdigest(),
        "model_dir": str(MODEL_DIR),
        "codec_differences": [
            "Python probe: FP32 scale/offset throughout",
            "C codec: FP16 round-trip on scale and offset before dequantize",
            "C codec: index rounding via (uint8_t)(v + 0.5f) vs np.round",
        ],
        "threshold": 0.999,
        "n_tensors_checked": len(results),
        "results": results,
        "worst_codec_cosine": worst_codec,
        "worst_tensor": worst_tensor,
        "decision": decision,
        "decision_detail": decision_detail,
        "cost": cost,
    }

    out_dir = Path(__file__).parent.parent / "receipts" / "hxq_af6_probe_qwen3b"
    out_path = out_dir / "codec_parity_receipt.json"
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    receipt_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print("=" * 60)
    print(f"DECISION: {decision}")
    print(f"  {decision_detail}")
    print(f"Receipt: {out_path}")
    print(f"SHA256:  {receipt_sha}")

    return 0 if all_above else 1


if __name__ == "__main__":
    raise SystemExit(main())
