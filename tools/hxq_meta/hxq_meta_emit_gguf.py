#!/usr/bin/env python3
"""GO HXQ_META_GGUF_EMIT_V0 — Emit Qwen2.5-Coder-3B GGUF via HXQ meta-codec.

Variant A (runtime-safe):
  - All weight tensors → Q5_K (middle tier)
  - Embedding → F16 (for stock llama.cpp compatibility)
  - Norms/biases → F16

Uses the validated HXQ_Q5_H encoder (byte-compatible with Q5_K) and the
gguf Python package to emit a standard GGUF loadable by any llama.cpp build.

Does NOT require custom kernels or GGUF types.
"""
import hashlib
import json
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
    print("ERROR: safetensors required"); sys.exit(1)

try:
    import gguf
    from gguf import GGMLQuantizationType
except ImportError:
    print("ERROR: gguf package required (pip install gguf)"); sys.exit(1)

sys.stdout.reconfigure(line_buffering=True)

# ── Tensor name mapping (HF → GGUF) ─────────────────────────────

sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
from gguf.tensor_mapping import TensorNameMap
from gguf.constants import MODEL_ARCH

_TMAP = None

def hf_to_gguf_name(hf_name, n_layers=36):
    """Convert HuggingFace tensor name to GGUF standard name."""
    global _TMAP
    if _TMAP is None:
        _TMAP = TensorNameMap(MODEL_ARCH.QWEN2, n_layers)

    # Strip suffix (.weight / .bias)
    suffix = ""
    if hf_name.endswith(".weight"):
        suffix = ".weight"
        stem = hf_name[:-7]
    elif hf_name.endswith(".bias"):
        suffix = ".bias"
        stem = hf_name[:-5]
    else:
        stem = hf_name

    mapped = _TMAP.get_name(stem)
    if mapped is None:
        return None  # Unknown tensor
    return mapped + suffix


# ── Constants ────────────────────────────────────────────────────

QK_K = 256
N_SUB = 8
SG = 32
NMAX = 31
BLOCK_SIZE = 176

MODEL_DIR = os.path.expanduser("~/models/qwen2.5-coder-3b-instruct/")
OUTPUT_PATH = os.path.expanduser("~/models/qwen2.5-coder-3b-instruct-hxq-q5k.gguf")


# ── Q5_K encoder (vectorized, from proven audit) ─────────────────

def encode_tensor_q5k(w_flat, out_dim, in_dim):
    """Encode a 2D weight tensor to Q5_K blocks.

    Args:
        w_flat: float64 array (out_dim * in_dim elements)
        out_dim: number of rows
        in_dim: number of columns (must be multiple of 256)

    Returns:
        uint8 array of shape (out_dim, n_blocks_per_row * 176)
    """
    assert in_dim % QK_K == 0, f"in_dim {in_dim} not multiple of {QK_K}"
    n_blocks_per_row = in_dim // QK_K
    bytes_per_row = n_blocks_per_row * BLOCK_SIZE

    w = w_flat.reshape(out_dim, in_dim)
    result = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)

    for row_idx in range(out_dim):
        row = w[row_idx]
        for blk_idx in range(n_blocks_per_row):
            x_block = row[QK_K * blk_idx: QK_K * (blk_idx + 1)]
            block_bytes = encode_single_q5k_block(x_block)
            offset = blk_idx * BLOCK_SIZE
            result[row_idx, offset:offset + BLOCK_SIZE] = np.frombuffer(block_bytes, dtype=np.uint8)

    return result


def encode_single_q5k_block(x_block):
    """Encode 256 floats to 176 Q5_K bytes."""
    scales = np.empty(N_SUB, dtype=np.float64)
    mins = np.empty(N_SUB, dtype=np.float64)

    for j in range(N_SUB):
        sg = x_block[SG * j: SG * (j + 1)]
        xmin = min(float(sg.min()), 0.0)
        xmax = float(sg.max())
        rng = xmax - xmin
        if rng < 1e-10:
            scales[j] = 0.0
            mins[j] = -xmin
        else:
            scales[j] = rng / NMAX
            mins[j] = -xmin

    max_scale = float(scales.max()) if scales.max() > 0 else 0.0
    max_min = float(mins.max()) if mins.max() > 0 else 0.0
    inv_scale = 63.0 / max_scale if max_scale > 0 else 0.0
    inv_min = 63.0 / max_min if max_min > 0 else 0.0

    q_scales = np.clip(np.round(inv_scale * scales), 0, 63).astype(np.uint8)
    q_mins = np.clip(np.round(inv_min * mins), 0, 63).astype(np.uint8)
    d_fp16 = np.float16(max_scale / 63.0) if max_scale > 0 else np.float16(0.0)
    dmin_fp16 = np.float16(max_min / 63.0) if max_min > 0 else np.float16(0.0)

    # Re-quantize
    L = np.zeros(QK_K, dtype=np.uint8)
    for j in range(N_SUB):
        d_recon = float(d_fp16) * float(q_scales[j])
        m_recon = float(dmin_fp16) * float(q_mins[j])
        sg = x_block[SG * j: SG * (j + 1)]
        if d_recon > 0:
            L[SG * j: SG * (j + 1)] = np.clip(np.round((sg + m_recon) / d_recon), 0, NMAX).astype(np.uint8)

    # Pack
    block = bytearray(BLOCK_SIZE)
    struct.pack_into('<e', block, 0, float(d_fp16))
    struct.pack_into('<e', block, 2, float(dmin_fp16))

    # Scale packing
    scale_bytes = np.zeros(12, dtype=np.uint8)
    for j in range(4):
        scale_bytes[j] = q_scales[j]
        scale_bytes[j + 4] = q_mins[j]
    for j in range(4, 8):
        scale_bytes[j + 4] = (q_scales[j] & 0x0F) | ((q_mins[j] & 0x0F) << 4)
        scale_bytes[j - 4] |= ((q_scales[j] >> 4) << 6)
        scale_bytes[j] |= ((q_mins[j] >> 4) << 6)
    block[4:16] = scale_bytes.tobytes()

    # High bits
    qh = np.zeros(32, dtype=np.uint8)
    for p in range(4):
        for l in range(32):
            if L[64 * p + l] > 15:
                qh[l] |= (1 << (2 * p))
            if L[64 * p + 32 + l] > 15:
                qh[l] |= (1 << (2 * p + 1))
    block[16:48] = qh.tobytes()

    # Low nibbles
    qs = np.zeros(128, dtype=np.uint8)
    for p in range(4):
        for l in range(32):
            qs[32 * p + l] = (L[64 * p + l] & 0x0F) | ((L[64 * p + 32 + l] & 0x0F) << 4)
    block[48:176] = qs.tobytes()

    return bytes(block)


# ── Routing policy (Variant A: runtime-safe) ─────────────────────

def should_quantize(name, shape):
    """Decide if tensor should be Q5_K quantized or kept F16.

    Returns: ("Q5_K", reason) or ("F16", reason)
    """
    n_elements = int(np.prod(shape))
    nl = name.lower()

    # Norms: always F16 (tiny, high sensitivity)
    if "norm" in nl or "layernorm" in nl:
        return "F16", "norm"

    # Biases: always F16 (tiny)
    if "bias" in nl:
        return "F16", "bias"

    # Embedding: F16 for runtime-safe variant (stock llama.cpp)
    if "embed" in nl:
        return "F16", "embed_runtime_safe"

    # lm_head / output: F16 for safety
    if "lm_head" in nl:
        return "F16", "output_head"

    # All other weight matrices: Q5_K if dimensions align
    if len(shape) == 2 and shape[1] % QK_K == 0:
        return "Q5_K", "weight_matrix"

    # 1D or non-aligned: F16
    if len(shape) == 1 or (len(shape) == 2 and shape[1] % QK_K != 0):
        return "F16", "non_aligned"

    return "F16", "fallback"


# ── Main ─────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    print("=" * 76)
    print("GO HXQ_META_GGUF_EMIT_V0 — Qwen2.5-Coder-3B (Variant A: runtime-safe)")
    print("=" * 76)
    print(f"\n  Output: {OUTPUT_PATH}")
    print(f"  Policy: Q5_K for weight matrices, F16 for embed/norm/bias")

    # Load config
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"\n  Model: {config.get('model_type', 'unknown')}")
    print(f"  Layers: {config.get('num_hidden_layers', '?')}")
    print(f"  Hidden: {config.get('hidden_size', '?')}")
    print(f"  Vocab: {config.get('vocab_size', '?')}")

    # Initialize GGUF writer
    writer = gguf.GGUFWriter(OUTPUT_PATH, "qwen2")

    # Write model metadata
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_embedding_length(config["hidden_size"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config.get("num_key_value_heads", config["num_attention_heads"]))
    writer.add_context_length(config.get("max_position_embeddings", 32768))
    writer.add_layer_norm_rms_eps(config.get("rms_norm_eps", 1e-6))
    writer.add_rope_freq_base(config.get("rope_theta", 1000000.0))
    writer.add_vocab_size(config["vocab_size"])
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q5_K_M)

    # Write tokenizer
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        # Use merges.txt + vocab.json approach for BPE
        merges_path = os.path.join(MODEL_DIR, "merges.txt")
        vocab_path = os.path.join(MODEL_DIR, "vocab.json")

        if os.path.exists(vocab_path) and os.path.exists(merges_path):
            with open(vocab_path) as f:
                vocab = json.load(f)
            with open(merges_path) as f:
                merges_lines = f.read().strip().split('\n')
                # Skip header if present
                if merges_lines and merges_lines[0].startswith('#'):
                    merges_lines = merges_lines[1:]

            # Pad token list to full vocab_size (model may have padding tokens)
            vocab_size = config["vocab_size"]
            tokens = [b""] * vocab_size
            scores = [0.0] * vocab_size
            toktypes = [gguf.TokenType.NORMAL] * vocab_size

            for tok, idx in vocab.items():
                if idx < vocab_size:
                    tokens[idx] = tok.encode("utf-8", errors="replace")
                    scores[idx] = -float(idx)  # BPE: rank as negative index

            # Add special tokens from tokenizer_config
            tokenizer_config_path = os.path.join(MODEL_DIR, "tokenizer_config.json")
            tok_config = {}
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path) as f:
                    tok_config = json.load(f)
                added = tok_config.get("added_tokens_decoder", {})
                for idx_str, info in added.items():
                    idx = int(idx_str)
                    if idx < vocab_size:
                        content = info.get("content", "")
                        tokens[idx] = content.encode("utf-8", errors="replace")
                        if info.get("special", False):
                            toktypes[idx] = 3  # CONTROL

            # Fill remaining empty slots
            for i in range(vocab_size):
                if tokens[i] == b"":
                    tokens[i] = f"<|unused{i}|>".encode("utf-8")

            writer.add_tokenizer_model("gpt2")
            writer.add_token_list(tokens)
            writer.add_token_scores(scores)
            writer.add_token_types(toktypes)
            writer.add_token_merges(merges_lines)

            # Special tokens
            if "bos_token" in tok_config:
                bos = tok_config["bos_token"]
                if isinstance(bos, dict):
                    bos = bos.get("content", "")
                if bos in vocab:
                    writer.add_bos_token_id(vocab[bos])
            if "eos_token" in tok_config:
                eos = tok_config["eos_token"]
                if isinstance(eos, dict):
                    eos = eos.get("content", "")
                if eos in vocab:
                    writer.add_eos_token_id(vocab[eos])

            print(f"  Tokenizer: {vocab_size} slots ({len(vocab)} base + padding), {len(merges_lines)} merges")

    # Process tensor shards
    shards = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")])
    print(f"\n  Processing {len(shards)} shards...")

    stats = {"Q5_K": 0, "F16": 0, "Q5_K_bytes": 0, "F16_bytes": 0, "total_elements": 0}
    tensor_log = []

    for shard_name in shards:
        shard_path = os.path.join(MODEL_DIR, shard_name)
        print(f"\n  [{shard_name}]")
        tensors = safetensors.torch.load_file(shard_path)

        for name in sorted(tensors.keys()):
            t = tensors[name]
            shape = tuple(t.shape)
            n_elements = int(np.prod(shape))
            quant_type, reason = should_quantize(name, shape)

            # Map HF name → GGUF name
            gguf_name = hf_to_gguf_name(name, config["num_hidden_layers"])
            if gguf_name is None:
                print(f"    SKIP (no mapping): {name}")
                continue

            if quant_type == "Q5_K":
                # Encode as Q5_K blocks
                w = t.float().numpy().astype(np.float64).ravel()
                out_dim, in_dim = shape[0], shape[1]
                q5k_bytes = encode_tensor_q5k(w, out_dim, in_dim)
                writer.add_tensor(gguf_name, q5k_bytes, raw_dtype=GGMLQuantizationType.Q5_K)
                n_bytes = q5k_bytes.nbytes
                stats["Q5_K"] += 1
                stats["Q5_K_bytes"] += n_bytes
            elif reason in ("bias", "norm"):
                # Biases and norms MUST be F32 for llama.cpp binary ops
                f32_data = t.float().numpy()
                writer.add_tensor(gguf_name, f32_data)
                n_bytes = f32_data.nbytes
                stats["F16"] += 1
                stats["F16_bytes"] += n_bytes
            else:
                # Embeddings: F16
                f16_data = t.half().numpy()
                writer.add_tensor(gguf_name, f16_data)
                n_bytes = f16_data.nbytes
                stats["F16"] += 1
                stats["F16_bytes"] += n_bytes

            stats["total_elements"] += n_elements
            tensor_log.append({
                "name": name, "gguf_name": gguf_name, "shape": list(shape),
                "type": quant_type, "reason": reason, "bytes": n_bytes
            })

            # Progress (every 36 tensors = 1 layer)
            total_done = stats["Q5_K"] + stats["F16"]
            if total_done % 36 == 0 or total_done <= 5:
                print(f"    {gguf_name:<55} → {quant_type} ({reason})")

    # Finalize
    print(f"\n  Writing GGUF...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(OUTPUT_PATH)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("EMIT SUMMARY")
    print("=" * 76)

    total_bytes = stats["Q5_K_bytes"] + stats["F16_bytes"]
    effective_bpw = total_bytes * 8 / stats["total_elements"] if stats["total_elements"] > 0 else 0

    print(f"\n  File: {OUTPUT_PATH}")
    print(f"  Size: {file_size / 1e9:.3f} GB ({file_size:,} bytes)")
    print(f"  Q5_K tensors: {stats['Q5_K']} ({stats['Q5_K_bytes'] / 1e9:.3f} GB)")
    print(f"  F16 tensors:  {stats['F16']} ({stats['F16_bytes'] / 1e9:.3f} GB)")
    print(f"  Effective bpw: {effective_bpw:.3f}")
    print(f"  Total elements: {stats['total_elements']:,}")

    # Verify file
    file_hash = hashlib.sha256(open(OUTPUT_PATH, 'rb').read(4096)).hexdigest()[:16]
    print(f"  Header hash: {file_hash}")
    print(f"\n  STATUS: GGUF_EMITTED")

    # ── Receipt ──────────────────────────────────────────────────
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

    receipt = {
        "receipt_id": f"hxq_meta_gguf_emit_v0_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        "title": "GO HXQ_META_GGUF_EMIT_V0 — Qwen2.5-Coder-3B runtime-safe GGUF",
        "status": "GGUF_EMITTED",
        "variant": "A_runtime_safe",
        "output_path": OUTPUT_PATH,
        "file_size_bytes": file_size,
        "file_size_gb": round(file_size / 1e9, 3),
        "effective_bpw": round(effective_bpw, 3),
        "stats": stats,
        "header_hash": file_hash,
        "model": "qwen2.5-coder-3b-instruct",
        "architecture": "qwen2",
        "n_tensors_q5k": stats["Q5_K"],
        "n_tensors_f16": stats["F16"],
        "cost": cost,
    }

    receipt_path = os.path.expanduser(
        f"~/receipts/hxq_meta_gguf_emit_v0_{time.strftime('%Y%m%d')}.json"
    )
    os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\n  Receipt: {receipt_path}")
    print(f"  Cost: {cost['wall_time_s']:.1f}s wall, {cost['peak_memory_mb']:.0f} MB peak")
    print(f"\n  DONE — GGUF emitted. Test with:")
    print(f"    llama-cli -m {OUTPUT_PATH} -p 'Hello' -n 32")


if __name__ == "__main__":
    main()
