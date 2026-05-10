#!/usr/bin/env python3
"""
Rebrand existing HF repos from "helix" to "HXQ/HelixCode".

For each local *-helix model dir:
  1. Patches config.json: quant_method "helix" → "hxq"
  2. Patches README.md: targeted string replacements (preserves hand-crafted content)
  3. Adds Verification Status section if missing
  4. Pushes updated config.json + README.md to existing HF repo

Does NOT rename repos (that would break download URLs for 4400+ users).
Does NOT re-upload model.safetensors (unchanged).

Usage:
    # Dry run — show what would change
    python3 tools/rebrand_hxq.py --dry-run

    # Apply locally only (no HF push)
    python3 tools/rebrand_hxq.py --local-only

    # Full rebrand (local patch + HF push)
    python3 tools/rebrand_hxq.py

    # Single model
    python3 tools/rebrand_hxq.py --model qwen2.5-coder-1.5b-helix
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path


MODELS_DIR = Path.home() / "models"
HF_ORG = "EchoLabs33"

# Local dir name → actual HF repo name (when they differ)
REPO_NAME_OVERRIDES = {
    "qwen2.5-14b-helix": "qwen2.5-14b-instruct-helix",
    "qwen2.5-7b-helix": "qwen2.5-7b-instruct-helix",
}

# Targeted string replacements for README body text.
# Order matters: more specific patterns first to avoid double-replacing.
# Rules: preserve "HelixLinear", "HelixCode", "helix-substrate" (package name).
#         Replace "helix quantizer" → "HXQ quantizer", quant_method values, etc.
README_REPLACEMENTS = [
    # ── CDNA v3 → HXQ/HelixCode (missed by first rebrand) ──────────
    ('quant_method = "cdna_v3"', 'quant_method = "hxq"'),
    ('"quant_method": "cdna_v3"', '"quant_method": "hxq"'),
    ("quant_method: cdna_v3", "quant_method: hxq"),
    ("registers cdna_v3 quantizer", "registers the HXQ quantizer"),
    ("registers the cdna_v3 quantizer", "registers the HXQ quantizer"),
    ("registers the `cdna_v3` quantizer", "registers the `hxq` quantizer"),
    ("registers the `cdna_v3` quantizer with HuggingFace",
     "registers the `hxq` quantizer with HuggingFace"),
    ("import helix_substrate.hf_quantizer", "import helix_substrate"),
    ('pip install helix-substrate transformers torch',
     'pip install "helix-substrate[hf]"'),
    ("## What is CDNA v3?", "## What is HelixCode?"),
    ("CDNA v3 is a universal", "HelixCode is a universal"),
    ("compressed 4.0x with CDNA v3", "compressed with HelixCode (HXQ)"),
    ("via CDNA v3", "via HelixCode"),
    ("with CDNA v3", "with HelixCode"),
    # ── helix → HXQ (original patterns) ────────────────────────────
    ('quant_method = "helix"', 'quant_method = "hxq"'),
    ("quant_method: helix", "quant_method: hxq"),
    ('"quant_method": "helix"', '"quant_method": "hxq"'),
    ("registers the helix quantizer", "registers the HXQ quantizer"),
    ("registers the `helix` quantizer", "registers the `hxq` quantizer"),
    ("registers the `helix` quantizer with HuggingFace",
     "registers the `hxq` quantizer with HuggingFace"),
    # ── Table headers ──────────────────────────────────────────────
    ("| Dense (FP32) | Helix (HXQ) |", "| Dense (FP32) | HXQ |"),
    ("| Helix PPL", "| HXQ PPL"),
]

# Tags to ensure are present in YAML frontmatter
REQUIRED_TAGS = {"hxq", "helixcode"}
# Tags to remove from YAML frontmatter
REMOVE_TAGS = {"cdna-v3"}  # old tag from pre-rebrand

VERIFICATION_SECTION = """## Verification Status

- **Compression receipt:** see stats above
- **Conversion receipt:** not available (pre-Gate 1)
- **GPU eval receipt:** Awaiting GPU verification — compression receipt only
"""


def find_helix_models(single: str | None = None) -> list[dict]:
    """Find all *-helix model dirs with config.json."""
    models = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("-helix"):
            continue
        if single and d.name != single:
            continue

        config_path = d / "config.json"
        if not config_path.exists():
            continue

        config = json.loads(config_path.read_text())
        qc = config.get("quantization_config", {})
        qm = qc.get("quant_method", "")

        has_safetensors = (d / "model.safetensors").exists()
        has_readme = (d / "README.md").exists()

        hf_name = REPO_NAME_OVERRIDES.get(d.name, d.name)
        models.append({
            "dir": d,
            "name": d.name,
            "config": config,
            "quant_method": qm,
            "has_safetensors": has_safetensors,
            "has_readme": has_readme,
            "repo_id": f"{HF_ORG}/{hf_name}",
        })

    return models


def patch_config(model: dict, dry_run: bool) -> bool:
    """Patch quant_method to 'hxq' in config.json. Returns True if changed."""
    config = model["config"]
    qc = config.get("quantization_config", {})

    if qc.get("quant_method") == "hxq":
        print(f"  config.json: already hxq")
        return False

    old_method = qc.get("quant_method", "???")
    qc["quant_method"] = "hxq"
    config["quantization_config"] = qc

    if dry_run:
        print(f"  config.json: would change quant_method '{old_method}' → 'hxq'")
        return True

    config_path = model["dir"] / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  config.json: quant_method '{old_method}' → 'hxq'")
    return True


def patch_readme(model: dict, dry_run: bool) -> bool:
    """Patch README.md with targeted replacements. Returns True if changed."""
    readme_path = model["dir"] / "README.md"
    if not readme_path.exists():
        print(f"  README.md: not found, skipping")
        return False

    original = readme_path.read_text()
    text = original

    # 1. Targeted string replacements in body
    replacements_made = []
    for old, new in README_REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
            replacements_made.append(f"'{old[:40]}...' → '{new[:40]}...'")

    # 2. Ensure required tags in YAML frontmatter
    tags_added = _patch_yaml_tags(text)
    if tags_added is not None:
        text = tags_added[0]
        for tag in tags_added[1]:
            replacements_made.append(f"added tag: {tag}")

    # 3. Add Verification Status section if missing
    if "## Verification Status" not in text:
        # Insert after first ## section heading that isn't the title
        # Try after "## Benchmark" or "## Compression Stats" or before "## Architecture Notes"
        insert_points = [
            "## Architecture Notes",
            "## Good to Know",
            "## How It Works",
            "## Usage",
        ]
        inserted = False
        for marker in insert_points:
            if marker in text:
                text = text.replace(marker, VERIFICATION_SECTION + "\n" + marker)
                replacements_made.append("added Verification Status section")
                inserted = True
                break
        if not inserted and "## " in text:
            # Fallback: insert before the last ## section
            replacements_made.append("Verification Status: could not find insertion point")

    if text == original:
        print(f"  README.md: no changes needed")
        return False

    if dry_run:
        print(f"  README.md: would make {len(replacements_made)} changes:")
        for r in replacements_made:
            print(f"    - {r}")
        return True

    readme_path.write_text(text)
    print(f"  README.md: {len(replacements_made)} changes applied")
    for r in replacements_made:
        print(f"    - {r}")
    return True


def _patch_yaml_tags(text: str) -> tuple[str, list[str]] | None:
    """Ensure required tags exist in YAML frontmatter. Returns (new_text, added_tags) or None."""
    # Find YAML frontmatter
    if not text.startswith("---"):
        return None

    end = text.find("---", 3)
    if end == -1:
        return None

    frontmatter = text[3:end]

    # Find existing tags
    tags_match = re.search(r"^tags:\s*\n((?:\s+-\s+.*\n)*)", frontmatter, re.MULTILINE)
    if not tags_match:
        return None

    existing_tags = set()
    tag_lines = []
    for line in tags_match.group(1).split("\n"):
        stripped = line.strip()
        if stripped.startswith("- "):
            tag = stripped[2:].strip()
            existing_tags.add(tag)
            tag_lines.append((line, tag))

    missing = REQUIRED_TAGS - existing_tags
    to_remove = REMOVE_TAGS & existing_tags
    if not missing and not to_remove:
        return None

    changes = sorted(missing) + [f"removed: {t}" for t in sorted(to_remove)]

    # Rebuild tags block: remove unwanted, add missing
    tags_block = tags_match.group(0)
    new_tag_section = "tags:\n"
    for orig_line, tag in tag_lines:
        if tag not in to_remove:
            new_tag_section += orig_line + "\n"
    for tag in sorted(missing):
        new_tag_section += f"  - {tag}\n"

    new_text = text[:3] + frontmatter.replace(tags_block, new_tag_section) + text[end:]
    return (new_text, changes)


def push_to_hf(model: dict, dry_run: bool) -> bool:
    """Push updated config.json + README.md to existing HF repo."""
    repo_id = model["repo_id"]

    if dry_run:
        print(f"  HF push: would upload config.json + README.md to {repo_id}")
        return True

    from huggingface_hub import HfApi

    api = HfApi()
    d = model["dir"]

    files_to_push = []
    if (d / "config.json").exists():
        files_to_push.append(("config.json", str(d / "config.json")))
    if (d / "README.md").exists():
        files_to_push.append(("README.md", str(d / "README.md")))

    if not files_to_push:
        print(f"  HF push: nothing to push")
        return False

    # Try direct push first, fall back to PR if 403
    for create_pr in (False, True):
        try:
            for remote_name, local_path in files_to_push:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_name,
                    repo_id=repo_id,
                    repo_type="model",
                    create_pr=create_pr,
                )
            mode = "via PR" if create_pr else "direct"
            print(f"  HF push: uploaded {len(files_to_push)} files to {repo_id} ({mode})")
            return True
        except Exception as e:
            err = str(e)
            if "403" in err and not create_pr:
                print(f"  HF push: direct push 403, retrying via PR...", file=sys.stderr)
                continue
            print(f"  HF push: FAILED — {e}", file=sys.stderr)
            return False
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Rebrand existing HF repos from 'helix' to 'HXQ/HelixCode'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change, don't modify anything")
    parser.add_argument("--local-only", action="store_true",
                        help="Patch local files only, don't push to HF")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model dir name (e.g. qwen2.5-coder-1.5b-helix)")
    parser.add_argument("--push-only", action="store_true",
                        help="Push existing local files to HF (skip patch step)")
    args = parser.parse_args()

    models = find_helix_models(args.model)
    if not models:
        print("No *-helix model dirs found with config.json")
        sys.exit(1)

    print(f"\nFound {len(models)} helix model(s) to rebrand:\n")

    pushed = 0
    patched = 0
    for model in models:
        print(f"[{model['name']}]  quant_method={model['quant_method']}  "
              f"safetensors={'YES' if model['has_safetensors'] else 'NO'}  "
              f"repo={model['repo_id']}")

        if args.push_only:
            config_changed = False
            readme_changed = False
        else:
            config_changed = patch_config(model, args.dry_run)
            readme_changed = patch_readme(model, args.dry_run)

        if config_changed or readme_changed:
            patched += 1

        should_push = (config_changed or readme_changed or args.push_only)
        if not args.dry_run and not args.local_only and should_push:
            if push_to_hf(model, args.dry_run):
                pushed += 1
            time.sleep(1)  # Rate limit

        print()

    mode = "DRY RUN" if args.dry_run else ("LOCAL ONLY" if args.local_only else "COMPLETE")
    print(f"Done ({mode}): {patched} patched, {pushed} pushed to HF")


if __name__ == "__main__":
    main()
