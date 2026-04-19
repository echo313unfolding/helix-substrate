#!/usr/bin/env python3
"""Fix model card ratios to be honest from BF16 source.

For BF16-source models: ratio ~2x (8-bit VQ vs 16-bit BF16)
For FP32-source models (TinyLlama, Mamba-130m): keep ~4x

Also fixes:
- "Dense (FP32)" → "Dense (BF16)" where source is BF16
- Companion table ratios across all cards
- Subtitle ratio claims
"""

import re
from pathlib import Path

MODELS_DIR = Path.home() / "models"

# Ground truth: model name → (source_dtype, bf16_size_gb, compressed_size_gb, honest_ratio_str)
# honest_ratio = bf16_size / compressed_size, rounded
MODEL_DATA = {
    "tinyllama-1.1b-helix": ("FP32", "4.4 GB", "1.03 GB", "4.0x"),
    "mamba-130m-helix": ("FP32", "489 MB", "128 MB", "3.8x"),
    "qwen2.5-coder-1.5b-helix": ("BF16", "3.1 GB", "2.1 GB", "1.5x"),
    "qwen2.5-3b-instruct-helix": ("BF16", "6.0 GB", "3.8 GB", "1.6x"),
    "qwen2.5-coder-3b-helix": ("BF16", "6.2 GB", "3.8 GB", "1.6x"),
    "qwen2.5-7b-helix": ("BF16", "14.2 GB", "6.5 GB", "2.2x"),
    "qwen2.5-14b-helix": ("BF16", "28.8 GB", "8.4 GB", "3.4x"),
    "zamba2-1.2b-helix": ("BF16", "2.3 GB", "1.35 GB", "1.7x"),
    "zamba2-2.7b-instruct-helix": ("BF16", "5.1 GB", "2.8 GB", "1.8x"),
    "mamba2-1.3b-helix": ("BF16", "2.9 GB", "1.4 GB", "2.1x"),
}

# New companion table (consistent across all cards)
# Each card will show all models except itself
COMPANION_ENTRIES = [
    ('qwen2.5-14b-instruct-helix', 'Transformer', '3.4x', 'pending'),
    ('qwen2.5-7b-instruct-helix', 'Transformer', '2.2x', '+6.34%'),
    ('qwen2.5-3b-instruct-helix', 'Transformer', '1.6x', '+0.69%'),
    ('qwen2.5-coder-3b-helix', 'Transformer (code)', '1.6x', '+1.92%'),
    ('qwen2.5-coder-1.5b-helix', 'Transformer (code)', '1.5x', '+1.73%'),
    ('tinyllama-1.1b-helix', 'Transformer', '4.0x', '+0.78%'),
    ('zamba2-2.7b-instruct-helix', 'Hybrid (Mamba2+Transformer)', '1.8x', '+6.59%'),
    ('zamba2-1.2b-helix', 'Hybrid (Mamba2+Transformer)', '1.7x', '+2.90%'),
    ('mamba2-1.3b-helix', 'Pure SSM (Mamba2)', '2.1x', '+8.0%'),
    ('mamba-130m-helix', 'Pure SSM', '3.8x', '+18.4%'),
]

RATIO_NOTE = "\n> **Note:** Ratios are file size from BF16 source (or FP32 where noted). Compressed files currently include FP32 exact tensors — ratios will improve after FP16 downcast.\n"


def build_companion_table(exclude_model: str) -> str:
    """Build companion table excluding the current model."""
    lines = [
        "| Model | Architecture | Ratio | PPL Delta |",
        "|-------|-------------|-------|-----------|",
    ]
    for name, arch, ratio, ppl in COMPANION_ENTRIES:
        if name in exclude_model or exclude_model in name:
            continue
        lines.append(
            f"| [{name}](https://huggingface.co/EchoLabs33/{name}) "
            f"| {arch} | {ratio} | {ppl} |"
        )
    return "\n".join(lines)


def fix_companion_table(text: str, model_name: str) -> str:
    """Replace existing companion table with corrected ratios."""
    new_table = build_companion_table(model_name)

    # Find companion table section
    pattern = r"(\| Model \| Architecture \| Ratio \| PPL Delta \|.*?)(\n\n|\n##|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = text[:match.start(1)] + new_table + text[match.start(2):]
    return text


def fix_benchmark_table_ratio(text: str, old_ratio: str, new_ratio: str) -> str:
    """Fix ratio in main benchmark table."""
    text = text.replace(f"**{old_ratio}**", f"**{new_ratio}**")
    return text


def fix_dense_column(text: str, from_dtype: str, to_dtype: str, from_size: str, to_size: str) -> str:
    """Fix Dense column header and size."""
    text = text.replace(f"Dense ({from_dtype})", f"Dense ({to_dtype})")
    if from_size != to_size:
        text = text.replace(from_size, to_size)
    return text


def process_model(model_name: str, dry_run: bool = False) -> bool:
    """Process a single model card."""
    readme_path = MODELS_DIR / model_name / "README.md"
    if not readme_path.exists():
        print(f"  [{model_name}] README.md not found, skipping")
        return False

    data = MODEL_DATA.get(model_name)
    if not data:
        print(f"  [{model_name}] No data configured, skipping")
        return False

    source_dtype, bf16_size, compressed_size, honest_ratio = data
    original = readme_path.read_text()
    text = original

    changes = []

    # 1. Fix companion table
    text_before = text
    text = fix_companion_table(text, model_name)
    if text != text_before:
        changes.append("companion table ratios updated")

    # 2. Fix main ratio in benchmark table (only for BF16 models with wrong ratios)
    if source_dtype == "BF16":
        # Find any inflated ratio claims and replace with honest ratio
        # Common patterns: **4.0x**, **4.44x**, **4.7x**
        for old_r in ["4.0x", "4.44x", "4.7x", "5.61x"]:
            if f"**{old_r}**" in text and old_r != honest_ratio:
                text = text.replace(f"**{old_r}**", f"**{honest_ratio}**")
                changes.append(f"ratio {old_r} → {honest_ratio}")

        # Fix "Dense (FP32)" → "Dense (BF16)" where applicable
        if "Dense (FP32)" in text:
            text = text.replace("Dense (FP32)", "Dense (BF16)")
            changes.append("Dense (FP32) → Dense (BF16)")

        # Fix table size values that show FP32 size instead of BF16
        # Map: FP32 size shown → BF16 actual size
        FP32_TO_BF16_SIZES = {
            "6.2 GB": "3.1 GB",   # Qwen-Coder-1.5B: 6.2 GB FP32 → 3.1 GB BF16
            "4.6 GB": "2.3 GB",   # Zamba2-1.2B: 4.6 GB FP32 → 2.3 GB BF16
        }
        for fp32_sz, bf16_sz in FP32_TO_BF16_SIZES.items():
            if model_name in ("qwen2.5-coder-1.5b-helix", "zamba2-1.2b-helix"):
                # Only fix in table row: "| **Size** | X GB |"
                old_row = f"| **Size** | {fp32_sz} |"
                new_row = f"| **Size** | {bf16_sz} |"
                if old_row in text:
                    text = text.replace(old_row, new_row)
                    changes.append(f"table size: {fp32_sz} → {bf16_sz}")

    # 3. Fix subtitle ratio claims
    for old_r in ["4.0x smaller", "4.44x smaller", "4.7x smaller", "5.61x smaller"]:
        if old_r in text and source_dtype == "BF16":
            new_claim = f"{honest_ratio} smaller"
            text = text.replace(old_r, new_claim)
            changes.append(f"subtitle: {old_r} → {new_claim}")

    # 4. Fix "compressed from X GB (FP32)" subtitle patterns for BF16 models
    if source_dtype == "BF16":
        # Pattern: "from X GB (FP32) to Y GB"
        fp32_pattern = re.search(r"from (\d+\.?\d*) GB \(FP32\) to", text)
        if fp32_pattern:
            text = text.replace(
                f"from {fp32_pattern.group(1)} GB (FP32) to",
                f"from {bf16_size} (BF16) to"
            )
            changes.append(f"subtitle: FP32 size → BF16 {bf16_size}")

        # Pattern: "from X GB to Y GB" (no dtype specified, in subtitle)
        # Pattern: "compressed from 4.6 GB to 1.35 GB"
        # Only fix if we know the size is wrong

    # 5. Fix receipt ratio
    for old_r in ["4.0x", "4.44x", "4.7x", "5.61x"]:
        receipt_old = f"Compression ratio:   {old_r}"
        receipt_new = f"Compression ratio:   {honest_ratio}"
        if receipt_old in text and source_dtype == "BF16":
            text = text.replace(receipt_old, receipt_new)
            changes.append(f"receipt ratio: {old_r} → {honest_ratio}")

        weight_old = f"Weight ratio:        {old_r}"
        weight_new = f"Weight ratio:        {honest_ratio}"
        if weight_old in text and source_dtype == "BF16":
            text = text.replace(weight_old, weight_new)
            changes.append(f"receipt weight ratio: {old_r} → {honest_ratio}")

    # 6. Fix receipt "Dense size: X GB (FP32)"
    if source_dtype == "BF16":
        fp32_receipt = re.search(r"Dense size:\s+(\d+\.?\d*) GB \(FP32\)", text)
        if fp32_receipt:
            text = text.replace(
                f"Dense size:          {fp32_receipt.group(1)} GB (FP32)",
                f"Dense size:          {bf16_size} (BF16)"
            )
            changes.append(f"receipt: Dense FP32 → BF16 {bf16_size}")

    # 7. Fix "Helix (4.0x)" etc in benchmark section headers
    for old_r in ["4.0x", "4.44x", "4.7x"]:
        header_old = f"Helix ({old_r})"
        header_new = f"Helix ({honest_ratio})"
        if header_old in text and source_dtype == "BF16":
            text = text.replace(header_old, header_new)
            changes.append(f"benchmark header: ({old_r}) → ({honest_ratio})")

    # 8. Fix "after 4x compression" type prose
    if source_dtype == "BF16":
        for old_r in ["4x compression", "4.44x compression"]:
            if old_r in text:
                text = text.replace(old_r, f"{honest_ratio} compression")
                changes.append(f"prose: {old_r} → {honest_ratio}")

    # Add ratio note if not present and changes were made
    if changes and RATIO_NOTE.strip() not in text:
        # Insert after companion table intro line
        insert_marker = "Same codec, same `pip install`"
        if insert_marker in text:
            # Don't add note — keep it clean. The ratios speak for themselves.
            pass

    if text == original:
        print(f"  [{model_name}] no changes needed")
        return False

    if dry_run:
        print(f"  [{model_name}] would make {len(changes)} changes:")
        for c in changes:
            print(f"    - {c}")
        return True

    readme_path.write_text(text)
    print(f"  [{model_name}] {len(changes)} changes applied:")
    for c in changes:
        print(f"    - {c}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix model card ratios")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_DATA.keys())
    changed = 0
    for m in models:
        if process_model(m, args.dry_run):
            changed += 1

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"\n{mode}: {changed}/{len(models)} cards changed")


if __name__ == "__main__":
    main()
