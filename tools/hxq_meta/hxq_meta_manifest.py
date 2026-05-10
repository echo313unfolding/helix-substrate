#!/usr/bin/env python3
"""GO HXQ_META_CODEC_MODEL_MANIFEST_V0 — Full model routing manifest.

Runs the HXQ meta-router over ALL tensors in Qwen2.5-Coder-3B and produces
a routing manifest before full GGUF emit.

Per tensor: name, shape, role, tier, ggml_type, bpw, cosine, hash, reason.
Summary: by role, by tier, total size estimate.

Does NOT emit GGUF. Does NOT run MMLU. Does NOT add kernels.
"""
import json
import hashlib
import os
import platform
import resource
import struct
import sys
import time

import numpy as np

try:
    import safetensors.torch
except ImportError:
    print("ERROR: safetensors required")
    sys.exit(1)

sys.stdout.reconfigure(line_buffering=True)

# ── Constants ────────────────────────────────────────────────────

QK_K = 256
N_SUB = 8
SG = 32
NMAX_Q5 = 31
NMAX_AF6 = 63
GS_AF6 = 128

MODEL_DIR = os.path.expanduser("~/models/qwen2.5-coder-3b-instruct/")

# Cosine validation threshold
COSINE_GATE = 0.998


# ── Role classifier ─────────────────────────────────────────────

def classify_role(name):
    """Classify tensor role from name."""
    nl = name.lower()
    if "embed" in nl:
        return "embed"
    if "norm" in nl or "layernorm" in nl or "rmsnorm" in nl:
        return "norm"
    if "lm_head" in nl or "output" in nl and "proj" not in nl:
        return "output"
    if "self_attn" in nl or "attention" in nl:
        if "q_proj" in nl:
            return "attn_q"
        elif "k_proj" in nl:
            return "attn_k"
        elif "v_proj" in nl:
            return "attn_v"
        elif "o_proj" in nl:
            return "attn_o"
        return "attn_other"
    if "mlp" in nl:
        if "gate" in nl:
            return "mlp_gate"
        elif "up" in nl:
            return "mlp_up"
        elif "down" in nl:
            return "mlp_down"
        return "mlp_other"
    return "other"


def route_tensor(name, role, shape):
    """Route tensor to tier based on role and shape.

    Returns (tier, reason_code).
    """
    n_elements = int(np.prod(shape))

    # Norms: keep at full precision (too small to quantize meaningfully)
    if role == "norm":
        return "skip", "norm_too_small"

    # Embeddings: quality tier (high sensitivity, shared across tokens)
    if role == "embed":
        return "quality", "embed_high_sensitivity"

    # Output head: quality tier (final projection, high impact)
    if role == "output":
        return "quality", "output_high_impact"

    # Very small tensors (< 1024 elements): skip quantization
    if n_elements < 1024:
        return "skip", "too_small"

    # Attention and MLP: middle tier
    if role.startswith("attn"):
        return "middle", "attention_standard"
    if role.startswith("mlp"):
        return "middle", "mlp_standard"

    # Fallback
    return "middle", "default_middle"


# ── Vectorized encoders (for validation) ─────────────────────────

def validate_q5k(w_flat):
    """Quick vectorized Q5_K encode/decode for cosine validation."""
    n = len(w_flat)
    pad = (QK_K - n % QK_K) % QK_K
    if pad > 0:
        w_padded = np.concatenate([w_flat, np.zeros(pad)])
    else:
        w_padded = w_flat.copy()

    w_sg = w_padded.reshape(-1, SG)
    n_blocks = len(w_padded) // QK_K

    sg_min = np.minimum(w_sg.min(axis=1), 0.0)
    sg_max = w_sg.max(axis=1)
    rng = sg_max - sg_min
    scales = np.where(rng > 1e-10, rng / NMAX_Q5, 0.0)
    mins_pos = -sg_min

    scales_sb = scales.reshape(n_blocks, N_SUB)
    mins_sb = mins_pos.reshape(n_blocks, N_SUB)

    max_scale = scales_sb.max(axis=1, keepdims=True)
    max_min = mins_sb.max(axis=1, keepdims=True)
    inv_scale = np.where(max_scale > 0, 63.0 / max_scale, 0.0)
    inv_min = np.where(max_min > 0, 63.0 / max_min, 0.0)
    q_scales = np.clip(np.round(inv_scale * scales_sb), 0, 63)
    q_mins = np.clip(np.round(inv_min * mins_sb), 0, 63)
    d_fp16 = np.where(max_scale > 0, max_scale / 63.0, 0.0).astype(np.float16).astype(np.float64)
    dmin_fp16 = np.where(max_min > 0, max_min / 63.0, 0.0).astype(np.float16).astype(np.float64)

    d_recon = (d_fp16 * q_scales).reshape(-1)[:, None]
    m_recon = (dmin_fp16 * q_mins).reshape(-1)[:, None]

    L = np.where(d_recon > 0, np.clip(np.round((w_sg + m_recon) / np.maximum(d_recon, 1e-30)), 0, NMAX_Q5), 0.0)
    decoded = (d_recon * L - m_recon).reshape(-1)[:n]

    return decoded


def validate_af6(w_flat):
    """Quick vectorized AF6 encode/decode for cosine validation."""
    n = len(w_flat)
    pad = (GS_AF6 - n % GS_AF6) % GS_AF6
    if pad > 0:
        w_padded = np.concatenate([w_flat, np.zeros(pad)])
    else:
        w_padded = w_flat.copy()

    w_g = w_padded.reshape(-1, GS_AF6)
    wmin = w_g.min(axis=1, keepdims=True)
    wmax = w_g.max(axis=1, keepdims=True)
    scale = np.maximum(wmax - wmin, 1e-10) / NMAX_AF6

    # FP16 truncation
    scale_fp16 = scale.astype(np.float16).astype(np.float64)
    min_fp16 = wmin.astype(np.float16).astype(np.float64)

    idx = np.where(scale_fp16 > 0,
                   np.clip(np.round((w_g - min_fp16) / np.maximum(scale_fp16, 1e-30)), 0, NMAX_AF6),
                   0.0)
    decoded = (idx * scale_fp16 + min_fp16).reshape(-1)[:n]
    return decoded


def compute_cosine(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))


# ── BPW estimates ────────────────────────────────────────────────

def estimate_bpw(tier, n_elements):
    """Estimate bpw for a tier."""
    if tier == "middle":
        # Q5_K: 176 bytes / 256 elements
        n_blocks = (n_elements + QK_K - 1) // QK_K
        total_bytes = n_blocks * 176
        return total_bytes * 8 / (n_blocks * QK_K)  # exactly 5.5
    elif tier == "quality":
        # AF6: 100 bytes / 128 elements
        n_groups = (n_elements + GS_AF6 - 1) // GS_AF6
        total_bytes = n_groups * 100
        return total_bytes * 8 / (n_groups * GS_AF6)  # exactly 6.25
    elif tier == "skip":
        return 16.0  # kept at FP16
    else:
        return 4.5  # Q4_K_M approximate


def estimate_bytes(tier, n_elements):
    """Estimate stored bytes for a tier."""
    if tier == "middle":
        n_blocks = (n_elements + QK_K - 1) // QK_K
        return n_blocks * 176
    elif tier == "quality":
        n_groups = (n_elements + GS_AF6 - 1) // GS_AF6
        return n_groups * 100
    elif tier == "skip":
        return n_elements * 2  # FP16
    else:
        return int(n_elements * 4.5 / 8)


# ── Main ─────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    print("=" * 76)
    print("GO HXQ_META_CODEC_MODEL_MANIFEST_V0 — Qwen2.5-Coder-3B")
    print("=" * 76)

    # Find all safetensors shards
    shards = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")])
    print(f"\n  Model: {MODEL_DIR}")
    print(f"  Shards: {len(shards)}")

    manifest = []
    role_stats = {}
    tier_stats = {"skip": {"count": 0, "elements": 0, "bytes": 0},
                  "middle": {"count": 0, "elements": 0, "bytes": 0},
                  "quality": {"count": 0, "elements": 0, "bytes": 0}}
    total_elements = 0
    total_bytes_quantized = 0
    total_bytes_fp16 = 0
    cosine_failures = []
    n_validated = 0

    for shard_name in shards:
        shard_path = os.path.join(MODEL_DIR, shard_name)
        print(f"\n  Loading {shard_name}...")
        tensors = safetensors.torch.load_file(shard_path)
        print(f"    {len(tensors)} tensors")

        for name in sorted(tensors.keys()):
            t = tensors[name]
            shape = tuple(t.shape)
            n_elements = int(np.prod(shape))
            role = classify_role(name)
            tier, reason = route_tensor(name, role, shape)

            bpw = estimate_bpw(tier, n_elements)
            est_bytes = estimate_bytes(tier, n_elements)

            # Validation: compute cosine on a subset of tensors
            # (validate all non-skip tensors with > 256 elements)
            cos_val = None
            if tier != "skip" and n_elements >= QK_K:
                w_flat = t.float().numpy().astype(np.float64).ravel()
                if tier == "middle":
                    decoded = validate_q5k(w_flat)
                elif tier == "quality":
                    decoded = validate_af6(w_flat)
                else:
                    decoded = w_flat

                cos_val = compute_cosine(w_flat, decoded)
                n_validated += 1

                if cos_val < COSINE_GATE:
                    cosine_failures.append({
                        "name": name, "role": role, "tier": tier,
                        "cosine": cos_val, "n_elements": n_elements
                    })

            # Content hash (of raw tensor bytes for traceability)
            raw_hash = hashlib.sha256(t.float().numpy().tobytes()).hexdigest()[:16]

            entry = {
                "name": name,
                "shape": list(shape),
                "n_elements": n_elements,
                "role": role,
                "tier": tier,
                "ggml_type": {"skip": "FP16", "middle": "Q5_K", "quality": "HXQ_AF6"}.get(tier, "?"),
                "bpw": round(bpw, 3),
                "estimated_bytes": est_bytes,
                "validation_cosine": round(cos_val, 6) if cos_val is not None else None,
                "content_hash": raw_hash,
                "reason_code": reason,
            }
            manifest.append(entry)

            # Accumulate stats
            total_elements += n_elements
            total_bytes_quantized += est_bytes
            total_bytes_fp16 += n_elements * 2

            tier_stats[tier]["count"] += 1
            tier_stats[tier]["elements"] += n_elements
            tier_stats[tier]["bytes"] += est_bytes

            if role not in role_stats:
                role_stats[role] = {"count": 0, "elements": 0, "tiers": {}}
            role_stats[role]["count"] += 1
            role_stats[role]["elements"] += n_elements
            role_stats[role]["tiers"][tier] = role_stats[role]["tiers"].get(tier, 0) + 1

            # Print compact line
            cos_str = f"{cos_val:.4f}" if cos_val is not None else "  n/a "
            gate = "OK" if (cos_val is None or cos_val >= COSINE_GATE) else "!!"
            print(f"    [{gate}] {name:<55} {role:<10} → {tier:<7} {bpw:.2f}bpw cos={cos_str}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("ROUTING SUMMARY")
    print("=" * 76)

    print(f"\n  Total tensors: {len(manifest)}")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Validated: {n_validated} (cosine computed)")
    print(f"  Cosine failures (< {COSINE_GATE}): {len(cosine_failures)}")

    print(f"\n  {'Tier':<10} {'Count':>6} {'Elements':>14} {'Bytes':>14} {'%Model':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*14} {'-'*14} {'-'*8}")
    for tier in ["skip", "middle", "quality"]:
        s = tier_stats[tier]
        pct = 100.0 * s["elements"] / total_elements if total_elements > 0 else 0
        print(f"  {tier:<10} {s['count']:>6} {s['elements']:>14,} {s['bytes']:>14,} {pct:>7.1f}%")

    effective_bpw = total_bytes_quantized * 8 / total_elements if total_elements > 0 else 0
    compression = total_bytes_fp16 / total_bytes_quantized if total_bytes_quantized > 0 else 0

    print(f"\n  Model size estimates:")
    print(f"    FP16:       {total_bytes_fp16 / 1e9:.2f} GB")
    print(f"    HXQ mixed:  {total_bytes_quantized / 1e9:.2f} GB")
    print(f"    Effective:  {effective_bpw:.3f} bpw")
    print(f"    Ratio:      {compression:.2f}x")

    print(f"\n  {'Role':<12} {'Count':>5} {'Elements':>12} {'Tiers':>30}")
    print(f"  {'-'*12} {'-'*5} {'-'*12} {'-'*30}")
    for role in sorted(role_stats.keys()):
        rs = role_stats[role]
        tier_str = ", ".join(f"{t}:{c}" for t, c in sorted(rs["tiers"].items()))
        print(f"  {role:<12} {rs['count']:>5} {rs['elements']:>12,} {tier_str:>30}")

    if cosine_failures:
        print(f"\n  COSINE FAILURES:")
        for f in cosine_failures:
            print(f"    {f['name']}: cos={f['cosine']:.6f} (tier={f['tier']}, role={f['role']})")

    # ── Decision ─────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("DECISION")
    print("=" * 76)

    all_pass = len(cosine_failures) == 0
    if all_pass:
        decision = "MANIFEST_VALIDATED"
        print(f"\n  ALL validated tensors pass cosine gate ({COSINE_GATE}).")
        print(f"  Router choices are sane. Ready for full GGUF emit.")
    else:
        decision = "COSINE_FAILURES"
        print(f"\n  {len(cosine_failures)} tensors below cosine gate.")
        print(f"  Review failures above. May need tier adjustment.")

    print(f"\n  DECISION: {decision}")

    # ── Write manifest & receipt ─────────────────────────────────
    end_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": end_iso,
    }

    # Manifest
    manifest_dir = os.path.expanduser("~/manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "hxq_meta_qwen3b_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n  Manifest: {manifest_path}")

    # Receipt
    receipt = {
        "receipt_id": f"hxq_meta_qwen3b_manifest_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        "title": "GO HXQ_META_CODEC_MODEL_MANIFEST_V0 — Qwen2.5-Coder-3B routing manifest",
        "status": decision,
        "model": "qwen2.5-coder-3b-instruct",
        "n_tensors": len(manifest),
        "n_validated": n_validated,
        "n_cosine_failures": len(cosine_failures),
        "cosine_gate": COSINE_GATE,
        "total_elements": total_elements,
        "total_bytes_fp16": total_bytes_fp16,
        "total_bytes_quantized": total_bytes_quantized,
        "effective_bpw": round(effective_bpw, 3),
        "compression_ratio": round(compression, 3),
        "tier_stats": tier_stats,
        "role_stats": {k: {"count": v["count"], "elements": v["elements"], "tiers": v["tiers"]}
                      for k, v in role_stats.items()},
        "cosine_failures": cosine_failures,
        "cost": cost,
    }

    receipt_path = os.path.expanduser(
        f"~/receipts/hxq_meta_qwen3b_manifest_{time.strftime('%Y%m%d')}.json"
    )
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, default=str)
    print(f"  Receipt: {receipt_path}")
    print(f"  Cost: {cost['wall_time_s']:.1f}s wall, {cost['peak_memory_mb']:.0f} MB peak")
    print(f"\n  DONE — {decision}")


if __name__ == "__main__":
    main()
