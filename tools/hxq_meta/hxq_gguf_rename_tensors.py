#!/usr/bin/env python3
"""Rename tensors in existing HXQ GGUF from HF names to GGUF standard names.

Also optionally quantizes the embedding tensor to Q5_K for size parity.
Uses gguf library's GGUFReader + GGUFWriter.
"""
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.tensor_mapping import TensorNameMap
from gguf.constants import MODEL_ARCH

INPUT_PATH = os.path.expanduser("~/models/qwen2.5-coder-3b-instruct-hxq-q5k.gguf")
OUTPUT_PATH = os.path.expanduser("~/models/qwen2.5-coder-3b-instruct-hxq-q5k-v3.gguf")
MODEL_DIR = os.path.expanduser("~/models/qwen2.5-coder-3b-instruct/")

# Set True to quantize embedding to Q5_K (size parity with stock Q5_K_M)
# NOTE: Q5_K embedding crashes CUDA binbcast; keep F16 for runtime safety
QUANTIZE_EMBEDDING = False

# ── Q5_K encoder (from hxq_meta_emit_gguf.py) ────────────────────
QK_K = 256
N_SUB = 8
SG = 32
NMAX = 31
BLOCK_SIZE = 176


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
            L[SG * j: SG * (j + 1)] = np.clip(
                np.round((sg + m_recon) / d_recon), 0, NMAX).astype(np.uint8)

    # Pack block
    block = bytearray(BLOCK_SIZE)
    struct.pack_into('<e', block, 0, float(d_fp16))
    struct.pack_into('<e', block, 2, float(dmin_fp16))

    # Scale packing (get_scale_min_k4 scheme)
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


def encode_tensor_q5k(w_float, out_dim, in_dim):
    """Encode float array to Q5_K blocks. Returns uint8 array."""
    assert in_dim % QK_K == 0
    n_blocks_per_row = in_dim // QK_K
    bytes_per_row = n_blocks_per_row * BLOCK_SIZE
    w = w_float.reshape(out_dim, in_dim)
    result = np.zeros((out_dim, bytes_per_row), dtype=np.uint8)

    for row_idx in range(out_dim):
        row = w[row_idx]
        for blk_idx in range(n_blocks_per_row):
            x_block = row[QK_K * blk_idx: QK_K * (blk_idx + 1)]
            block_bytes = encode_single_q5k_block(x_block)
            offset = blk_idx * BLOCK_SIZE
            result[row_idx, offset:offset + BLOCK_SIZE] = np.frombuffer(block_bytes, dtype=np.uint8)
        if row_idx % 10000 == 0 and row_idx > 0:
            print(f"    embed encode: {row_idx}/{out_dim} rows...")

    return result

N_LAYERS = 36
tmap = TensorNameMap(MODEL_ARCH.QWEN2, N_LAYERS)


def hf_to_gguf_name(hf_name):
    suffix = ""
    if hf_name.endswith(".weight"):
        suffix = ".weight"
        stem = hf_name[:-7]
    elif hf_name.endswith(".bias"):
        suffix = ".bias"
        stem = hf_name[:-5]
    else:
        stem = hf_name
    mapped = tmap.get_name(stem)
    if mapped is None:
        return None
    return mapped + suffix


def main():
    t_start = time.time()
    print(f"Reading: {INPUT_PATH}")
    reader = GGUFReader(INPUT_PATH)

    # Read config for metadata
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"Found {len(reader.tensors)} tensors")

    # Build name mapping
    renames = {}
    skipped = []
    for tensor_info in reader.tensors:
        old_name = tensor_info.name
        new_name = hf_to_gguf_name(old_name)
        if new_name is None:
            skipped.append(old_name)
        else:
            renames[old_name] = new_name

    if skipped:
        print(f"WARNING: {len(skipped)} tensors have no mapping: {skipped[:5]}")

    print(f"Renaming {len(renames)} tensors...")

    # Create new GGUF with correct names
    writer = GGUFWriter(OUTPUT_PATH, "qwen2")

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
    writer.add_file_type(15)  # MOSTLY_Q5_K_M
    writer.add_quantization_version(2)
    writer.add_name("qwen2.5-coder-3b-instruct-hxq-q5k")

    # Write tokenizer
    vocab_size = config["vocab_size"]  # 151936 — must match embedding tensor dim
    vocab_path = os.path.join(MODEL_DIR, "vocab.json")
    merges_path = os.path.join(MODEL_DIR, "merges.txt")
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        with open(merges_path) as f:
            merges_lines = f.read().strip().split('\n')
            if merges_lines and merges_lines[0].startswith('#'):
                merges_lines = merges_lines[1:]

        # Pad to full vocab_size (model may have extra padding tokens)
        tokens = [b""] * vocab_size
        scores = [0.0] * vocab_size
        toktypes = [1] * vocab_size  # 1 = NORMAL for BPE tokenizers

        for tok, idx in vocab.items():
            if idx < vocab_size:
                tokens[idx] = tok.encode("utf-8", errors="replace")
                scores[idx] = -float(idx)

        # Add special tokens from tokenizer_config
        tokenizer_config_path = os.path.join(MODEL_DIR, "tokenizer_config.json")
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

        # Fill any remaining empty slots with placeholder (type=4 UNUSED)
        for i in range(vocab_size):
            if tokens[i] == b"":
                tokens[i] = f"[PAD{i}]".encode("utf-8")
                toktypes[i] = 4  # UNUSED

        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        writer.add_token_types(toktypes)
        writer.add_token_merges(merges_lines)

        # Special tokens and metadata
        if os.path.exists(tokenizer_config_path):
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
            # Padding token (same as BOS for Qwen2)
            if "bos_token" in tok_config:
                bos = tok_config["bos_token"]
                if isinstance(bos, dict):
                    bos = bos.get("content", "")
                if bos in vocab:
                    writer.add_pad_token_id(vocab[bos])
            # Chat template
            chat_template = tok_config.get("chat_template")
            if chat_template:
                writer.add_chat_template(chat_template)
            # Add BOS behavior and pre-tokenizer type
            writer.add_add_bos_token(False)
            writer.add_add_eos_token(False)

        # Pre-tokenizer type for Qwen2
        writer.add_string("tokenizer.ggml.pre", "qwen2")

        print(f"  Tokenizer: {vocab_size} slots ({len(vocab)} base + added + padding), {len(merges_lines)} merges")

    # Copy tensors with new names
    # Rule: Q5_K stays Q5_K, biases/norms → F32, embedding → Q5_K if QUANTIZE_EMBEDDING else F16
    n_q5k = 0
    n_f32 = 0
    n_f16 = 0
    for tensor_info in reader.tensors:
        old_name = tensor_info.name
        new_name = renames.get(old_name)
        if new_name is None:
            continue

        data = tensor_info.data
        tensor_type = tensor_info.tensor_type
        qtype = GGMLQuantizationType(tensor_type)

        if qtype == GGMLQuantizationType.Q5_K:
            writer.add_tensor(new_name, data, raw_dtype=GGMLQuantizationType.Q5_K)
            n_q5k += 1
        elif new_name.endswith(".bias") or "norm" in new_name:
            # Biases and norms MUST be F32 for llama.cpp binary ops
            if qtype == GGMLQuantizationType.F16:
                f16_arr = np.frombuffer(data, dtype=np.float16)
                f32_arr = f16_arr.astype(np.float32)
                writer.add_tensor(new_name, f32_arr, raw_dtype=GGMLQuantizationType.F32)
            else:
                writer.add_tensor(new_name, data, raw_dtype=GGMLQuantizationType.F32)
            n_f32 += 1
        elif QUANTIZE_EMBEDDING and "embd" in new_name and qtype == GGMLQuantizationType.F16:
            # Quantize embedding to Q5_K for size parity
            f16_arr = np.frombuffer(data, dtype=np.float16)
            f32_arr = f16_arr.astype(np.float64)
            # Embedding shape: (vocab_size, hidden_size)
            vocab_size = config["vocab_size"]
            hidden_size = config["hidden_size"]
            print(f"  Encoding embedding ({vocab_size} x {hidden_size}) to Q5_K...")
            q5k_bytes = encode_tensor_q5k(f32_arr, vocab_size, hidden_size)
            writer.add_tensor(new_name, q5k_bytes, raw_dtype=GGMLQuantizationType.Q5_K)
            n_q5k += 1
        else:
            # F16 passthrough
            writer.add_tensor(new_name, data, raw_dtype=qtype)
            n_f16 += 1

        total = n_q5k + n_f32 + n_f16
        if total % 72 == 0 or total <= 3:
            out_type = "Q5_K" if (qtype == GGMLQuantizationType.Q5_K or (QUANTIZE_EMBEDDING and "embd" in new_name)) else ("F32" if (new_name.endswith(".bias") or "norm" in new_name) else "F16")
            print(f"  [{total:3d}/{len(renames)}] {new_name} ({out_type})")

    print(f"\nWriting GGUF...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(OUTPUT_PATH)
    elapsed = time.time() - t_start

    print(f"\nDONE in {elapsed:.1f}s")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Size: {file_size / 1e9:.3f} GB")
    print(f"  Q5_K: {n_q5k}, F32 (bias/norm): {n_f32}, F16 (embed): {n_f16}")
    print(f"\n  Test with:")
    print(f"    llama-cli -m {OUTPUT_PATH} -p 'Hello' -n 32")


if __name__ == "__main__":
    main()
