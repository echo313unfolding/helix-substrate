"""
WO-BORN-HYBRID-01: Data preparation for born-compressed hybrid training.

Downloads code data, tokenizes with Qwen tokenizer, saves as memory-mapped
tensors for fast DataLoader access.

Usage:
    python -m echo_hybrid.prepare_data --output data/code_tokens --max-tokens 4000000000
    python -m echo_hybrid.prepare_data --output data/code_tokens --dataset the-stack-smol --max-tokens 500000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def tokenize_the_stack(
    output_dir: str,
    max_tokens: int = 4_000_000_000,
    seq_len: int = 2048,
    languages: list = None,
    dataset_name: str = "bigcode/the-stack-v2-train-smol-ids",
    split: str = "train",
    seed: int = 42,
):
    """Tokenize code from The Stack v2 with Qwen tokenizer.

    Saves:
        {output_dir}/train.bin  — memory-mapped uint32 token IDs
        {output_dir}/val.bin    — held-out validation split
        {output_dir}/meta.json  — metadata (n_tokens, vocab_size, etc.)
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if languages is None:
        languages = ["python", "javascript", "go", "rust", "typescript", "java", "c", "cpp"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: Qwen/Qwen2.5-Coder-1.5B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    eos_id = tokenizer.eos_token_id or 151645
    print(f"  vocab_size={vocab_size}, eos_id={eos_id}")

    # Estimate tokens needed (10% holdout for val)
    train_target = int(max_tokens * 0.9)
    val_target = int(max_tokens * 0.1)

    # Try loading dataset
    print(f"Loading dataset: {dataset_name} (split={split})")
    print(f"  Languages: {languages}")
    print(f"  Target: {max_tokens:,} tokens ({train_target:,} train + {val_target:,} val)")

    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"  Failed to load {dataset_name}: {e}")
        print(f"  Falling back to bigcode/starcoderdata...")
        dataset_name = "bigcode/starcoderdata"
        ds = load_dataset(dataset_name, split="train", streaming=True, data_dir=languages[0])

    # Tokenize in streaming mode, write to memmap
    train_path = out / "train.bin"
    val_path = out / "val.bin"

    # Pre-allocate memmaps
    train_tokens = np.memmap(str(train_path), dtype=np.uint32, mode="w+",
                             shape=(train_target,))
    val_tokens = np.memmap(str(val_path), dtype=np.uint32, mode="w+",
                           shape=(val_target,))

    train_offset = 0
    val_offset = 0
    n_docs = 0
    t0 = time.time()

    for example in ds:
        # Get text content
        text = example.get("content", example.get("text", ""))
        if not text or len(text) < 50:
            continue

        # Filter by language if field exists
        lang = example.get("lang", example.get("language", ""))
        if lang and languages and lang.lower() not in languages:
            continue

        # Tokenize
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)  # document boundary
        ids_arr = np.array(ids, dtype=np.uint32)

        # Route to train or val (every 10th doc goes to val)
        if n_docs % 10 == 0 and val_offset + len(ids_arr) <= val_target:
            end = min(val_offset + len(ids_arr), val_target)
            n_copy = end - val_offset
            val_tokens[val_offset:end] = ids_arr[:n_copy]
            val_offset = end
        elif train_offset + len(ids_arr) <= train_target:
            end = min(train_offset + len(ids_arr), train_target)
            n_copy = end - train_offset
            train_tokens[train_offset:end] = ids_arr[:n_copy]
            train_offset = end

        n_docs += 1

        if n_docs % 10000 == 0:
            elapsed = time.time() - t0
            tps = (train_offset + val_offset) / max(elapsed, 1)
            print(f"  {n_docs:,} docs | train: {train_offset:,}/{train_target:,} | "
                  f"val: {val_offset:,}/{val_target:,} | {tps:,.0f} tok/s")

        if train_offset >= train_target and val_offset >= val_target:
            break

    # Truncate memmaps to actual size
    del train_tokens, val_tokens
    _truncate_memmap(train_path, train_offset, np.uint32)
    _truncate_memmap(val_path, val_offset, np.uint32)

    # Chunk into sequences
    n_train_seqs = train_offset // seq_len
    n_val_seqs = val_offset // seq_len

    meta = {
        "dataset": dataset_name,
        "languages": languages,
        "tokenizer": "Qwen/Qwen2.5-Coder-1.5B",
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "seq_len": seq_len,
        "n_train_tokens": train_offset,
        "n_val_tokens": val_offset,
        "n_train_seqs": n_train_seqs,
        "n_val_seqs": n_val_seqs,
        "n_docs": n_docs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Train: {train_offset:,} tokens ({n_train_seqs:,} seqs of {seq_len})")
    print(f"  Val:   {val_offset:,} tokens ({n_val_seqs:,} seqs of {seq_len})")
    print(f"  Saved to {out}/")
    return meta


def _truncate_memmap(path: Path, actual_size: int, dtype):
    """Truncate a memmap file to actual used size."""
    itemsize = np.dtype(dtype).itemsize
    target_bytes = actual_size * itemsize
    current_bytes = path.stat().st_size
    if target_bytes < current_bytes:
        with open(path, "r+b") as f:
            f.truncate(target_bytes)


class TokenDataset:
    """Memory-mapped token dataset for training.

    Returns chunks of seq_len tokens from a .bin file.
    Loads only the needed chunk, never the full file.
    """

    def __init__(self, bin_path: str, seq_len: int = 2048):
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.n_tokens = len(self.data)
        self.n_seqs = self.n_tokens // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        import torch
        start = idx * self.seq_len
        end = start + self.seq_len
        tokens = self.data[start:end].astype(np.int64)
        return torch.from_numpy(tokens)

    def get_batch(self, batch_size: int, device: str = "cpu"):
        """Random batch of sequences."""
        import torch
        indices = np.random.randint(0, self.n_seqs, size=batch_size)
        batch = np.stack([
            self.data[i * self.seq_len:(i + 1) * self.seq_len].astype(np.int64)
            for i in indices
        ])
        return torch.from_numpy(batch).to(device)


def main():
    parser = argparse.ArgumentParser(description="Prepare tokenized code data")
    parser.add_argument("--output", type=str, default="data/code_tokens",
                        help="Output directory for token files")
    parser.add_argument("--max-tokens", type=int, default=4_000_000_000,
                        help="Max tokens to tokenize (default: 4B = ~3.3B train)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dataset", type=str, default="bigcode/the-stack-v2-train-smol-ids",
                        help="HuggingFace dataset name")
    parser.add_argument("--languages", type=str, default="python,javascript,go,rust,typescript,java,c,cpp",
                        help="Comma-separated language filter")
    args = parser.parse_args()

    languages = [l.strip() for l in args.languages.split(",")]
    tokenize_the_stack(
        output_dir=args.output,
        max_tokens=args.max_tokens,
        seq_len=args.seq_len,
        languages=languages,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
