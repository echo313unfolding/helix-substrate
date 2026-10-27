#!/usr/bin/env python3
"""
HXQ AF6 probe for Qwen2.5-Coder-3B-Instruct (dense).

Produces a per-tensor sensitivity receipt compatible with
profiles_from_probe_receipt() in hydra_router.py.

Phase 1 only: no PPL measurement, no routing, no quantization.
Output: JSON receipt with per-tensor affine6_cosine, kurtosis, std,
        max_abs_error, mean_abs_error for all 252 weight tensors.

Work Order: GAP-HXQ-009 Phase 1 (probe only)
"""

from __future__ import annotations

import gc
import json
import math
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

MODEL_DIR = Path.home() / "models" / "qwen2.5-coder-3b-instruct-dense"
N_LAYERS = 36

HF_PATTERNS = {
    "q_proj":    "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj":    "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj":    "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj":    "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj":   "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}

STRATEGIES = [
    ("affine6_g128", 64, 128),
    ("affine5_g128", 32, 128),
    ("affine4_g128", 16, 128),
    ("affine3_g128",  8, 128),
]


def encode_affine_group(tensor_2d, k, group_size):
    """Per-group uniform quantization. Returns (recon, stats)."""
    flat = tensor_2d.ravel().astype(np.float32)
    n = len(flat)

    pad = (group_size - n % group_size) % group_size
    padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)]) if pad > 0 else flat.copy()

    n_groups = len(padded) // group_size
    groups = padded.reshape(n_groups, group_size)

    g_min = groups.min(axis=1, keepdims=True)
    g_max = groups.max(axis=1, keepdims=True)
    g_range = np.maximum(g_max - g_min, 1e-10)

    normalized = (groups - g_min) / g_range
    indices = np.clip(np.round(normalized * (k - 1)), 0, k - 1).astype(np.int32)
    recon_groups = g_min + indices.astype(np.float32) / (k - 1) * g_range
    recon = recon_groups.ravel()[:n].reshape(tensor_2d.shape)

    orig_flat = tensor_2d.ravel().astype(np.float64)
    recon_flat = recon.ravel().astype(np.float64)
    dot = np.dot(orig_flat, recon_flat)
    na, nb = np.linalg.norm(orig_flat), np.linalg.norm(recon_flat)
    cos = float(dot / (na * nb)) if na > 1e-30 and nb > 1e-30 else 0.0

    err = np.abs(recon.ravel() - tensor_2d.ravel())

    bits_per_idx = np.log2(k)
    overhead_bpw = 2 * 16 / group_size
    bpw = bits_per_idx + overhead_bpw

    return recon, {
        "cosine": cos,
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
        "bpw": bpw,
    }


def excess_kurtosis(data):
    n = len(data)
    if n < 4:
        return 0.0
    m = data.mean()
    s = data.std()
    if s < 1e-12:
        return 0.0
    z = (data - m) / s
    return float(np.mean(z ** 4) - 3.0)


def find_safetensors(model_dir: Path) -> list[Path]:
    """Find safetensor shards, sorted by name."""
    shards = sorted(model_dir.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors in {model_dir}")
    return shards


def run_phase1():
    from safetensors import safe_open

    shards = find_safetensors(MODEL_DIR)
    print(f"\n  Model: {MODEL_DIR}")
    print(f"  Shards: {[s.name for s in shards]}")

    # Build tensor name list
    tensor_names = []
    for i in range(N_LAYERS):
        for ptype, pattern in HF_PATTERNS.items():
            tensor_names.append((pattern.format(i=i), i, ptype))

    print(f"  {len(tensor_names)} tensors x {len(STRATEGIES)} strategies")

    # Build shard index: which tensor lives in which shard
    shard_map: dict[str, Path] = {}
    for shard_path in shards:
        with safe_open(shard_path, framework="pt") as sf:
            for name in sf.keys():
                shard_map[name] = shard_path

    tensor_records = []
    t0 = time.time()

    # Process shard by shard to minimize file opens
    current_shard: Path | None = None
    sf_handle = None

    for idx, (tname, layer_idx, tensor_type) in enumerate(tensor_names):
        shard_path = shard_map.get(tname)
        if shard_path is None:
            print(f"  WARNING: {tname} not found in any shard, skipping")
            continue

        if shard_path != current_shard:
            if sf_handle is not None:
                del sf_handle
                gc.collect()
            sf_handle = safe_open(str(shard_path), framework="pt")
            current_shard = shard_path

        tensor = sf_handle.get_tensor(tname).float().numpy()
        if tensor.ndim == 1:
            continue

        flat = tensor.ravel().astype(np.float32)
        kurt = excess_kurtosis(flat)
        std_val = float(flat.std())
        shape = list(tensor.shape)

        record = {
            "name": tname,
            "layer": layer_idx,
            "type": tensor_type,
            "shape": shape,
            "n_params": int(np.prod(shape)),
            "kurtosis": round(kurt, 3),
            "std": round(std_val, 6),
            "strategies": {},
        }

        for sname, k, gs in STRATEGIES:
            _, stats = encode_affine_group(tensor, k=k, group_size=gs)
            record["strategies"][sname] = {
                "cosine": round(stats["cosine"], 6),
                "max_abs_error": round(stats["max_abs_error"], 6),
                "mean_abs_error": round(stats["mean_abs_error"], 6),
                "bpw": round(stats["bpw"], 2),
            }

        tensor_records.append(record)
        del tensor, flat
        gc.collect()

        if (idx + 1) % 36 == 0:
            elapsed = time.time() - t0
            print(f"    {idx + 1}/{len(tensor_names)} tensors, {elapsed:.1f}s")

    if sf_handle is not None:
        del sf_handle

    phase1_time = time.time() - t0
    print(f"\n  Phase 1 complete in {phase1_time:.1f}s ({len(tensor_records)} tensors)")

    # Summary
    print(f"\n  {'Strategy':<16} {'mean_cos':>10} {'min_cos':>10} {'p5_cos':>10} {'bpw':>6}")
    print("  " + "-" * 56)
    for sname, k, gs in STRATEGIES:
        cosines = [r["strategies"][sname]["cosine"] for r in tensor_records]
        ca = np.array(cosines)
        bpw = tensor_records[0]["strategies"][sname]["bpw"]
        print(f"  {sname:<16} {ca.mean():>10.6f} {ca.min():>10.6f} "
              f"{np.percentile(ca, 5):>10.6f} {bpw:>6.2f}")

    return tensor_records, phase1_time


def main():
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    cpu_start = time.process_time()

    print("=" * 60)
    print("HXQ AF6 Probe: Qwen2.5-Coder-3B-Instruct")
    print("Work Order: GAP-HXQ-009 Phase 1 (probe only)")
    print("=" * 60)

    tensor_records, phase1_time = run_phase1()

    # Cost block
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

    # Collect config hash
    import hashlib
    config_path = MODEL_DIR / "config.json"
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else "MISSING"

    receipt = {
        "experiment": "hxq_af6_probe_qwen3b",
        "work_order": "GAP-HXQ-009",
        "phase": "probe_only",
        "model": "Qwen2.5-Coder-3B-Instruct (dense BF16)",
        "model_dir": str(MODEL_DIR),
        "config_sha256": config_sha,
        "model_type": "qwen2",
        "n_layers": N_LAYERS,
        "n_tensors_probed": len(tensor_records),
        "strategies_probed": [s[0] for s in STRATEGIES],
        "phase1": {
            "tensor_records": tensor_records,
            "time_s": round(phase1_time, 3),
        },
        "cost": cost,
    }

    out_dir = Path(__file__).parent.parent / "receipts" / "hxq_af6_probe_qwen3b"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"af6_probe_qwen3b_{ts}.json"
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    receipt_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"\n  Receipt: {out_path}")
    print(f"  SHA256:  {receipt_sha}")
    print(f"  Cost:    {cost['wall_time_s']:.1f}s wall, {cost['peak_memory_mb']:.0f} MB peak")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
