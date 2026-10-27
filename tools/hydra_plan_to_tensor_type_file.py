#!/usr/bin/env python3
"""
Hydra CompressionPlan -> llama.cpp --tensor-type-file bridge.

Scope: GAP-HXQ-009 Phase 1.
Supported Hydra heads:
    affine6 -> hxq_affine_6
    exact   -> configurable exact type (default: f16)

All other heads fail closed. Tensor name translation is delegated to the
existing HXQ GGUF mapper rather than duplicated here.

Example:
    python tools/hydra_plan_to_tensor_type_file.py \
        --plan receipts/qwen_hydra_plan.json \
        --config /home/voidstr3m33/models/qwen2.5-coder-3b-instruct-dense/config.json \
        --mapper /home/voidstr3m33/hxq-native/tools/convert_hxq_to_gguf.py \
        --output /tmp/qwen_hydra_af6.tensor-types.txt
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

HEAD_TO_GGML_TYPE = {
    "affine6": "hxq_affine_6",
    "exact": "f16",
}
SUPPORTED_HEADS = frozenset(HEAD_TO_GGML_TYPE)


class BridgeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise BridgeError(f"{path}: expected a JSON object")
    return obj


def load_name_mapper(mapper_path: Path) -> Callable[[str, dict], Optional[str]]:
    if not mapper_path.is_file():
        raise BridgeError(f"GGUF mapper does not exist: {mapper_path}")

    spec = importlib.util.spec_from_file_location("hxq_gguf_name_mapper", mapper_path)
    if spec is None or spec.loader is None:
        raise BridgeError(f"Could not import mapper: {mapper_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, "map_tensor_name_to_gguf", None)
    if not callable(fn):
        raise BridgeError(
            f"{mapper_path} does not export callable map_tensor_name_to_gguf(name, config)"
        )
    return fn


def normalize_plan(plan: dict) -> list[dict]:
    tensors = plan.get("plan")
    if not isinstance(tensors, list):
        raise BridgeError("CompressionPlan JSON must contain a list field named 'plan'")

    normalized: list[dict] = []
    for i, item in enumerate(tensors):
        if not isinstance(item, dict):
            raise BridgeError(f"plan[{i}] is not an object")

        name = item.get("tensor_name")
        head = item.get("head")
        if not isinstance(name, str) or not name:
            raise BridgeError(f"plan[{i}] has invalid tensor_name")
        if not isinstance(head, str) or not head:
            raise BridgeError(f"plan[{i}] has invalid head")

        head = head.lower()
        if head not in SUPPORTED_HEADS:
            raise BridgeError(
                f"Unsupported Hydra head {head!r} for tensor {name!r}. "
                "This bridge is intentionally limited to EXACT + AFFINE6."
            )

        normalized.append({"tensor_name": name, "head": head})
    return normalized


def build_tensor_type_entries(
    plan: dict,
    config: dict,
    mapper: Callable[[str, dict], Optional[str]],
    exact_type: str = "f16",
) -> tuple[list[tuple[str, str]], dict]:
    if exact_type not in {"f16", "f32"}:
        raise BridgeError("--exact-type must be f16 or f32")

    entries: list[tuple[str, str]] = []
    mapped_from: dict[str, str] = {}
    skipped: list[str] = []
    head_counts: Counter[str] = Counter()

    for item in normalize_plan(plan):
        hf_name = item["tensor_name"]
        head = item["head"]

        gguf_name = mapper(hf_name, config)
        if gguf_name is None:
            skipped.append(hf_name)
            continue
        if not isinstance(gguf_name, str) or not gguf_name:
            raise BridgeError(
                f"Mapper returned invalid name for {hf_name!r}: {gguf_name!r}"
            )

        if gguf_name in mapped_from:
            raise BridgeError(
                "Two Hydra tensors map to the same GGUF tensor: "
                f"{mapped_from[gguf_name]!r} and {hf_name!r} -> {gguf_name!r}"
            )
        mapped_from[gguf_name] = hf_name

        ggml_type = exact_type if head == "exact" else HEAD_TO_GGML_TYPE[head]
        entries.append((gguf_name, ggml_type))
        head_counts[head] += 1

    # Stable output makes the artifact hash deterministic for a fixed input plan/config/mapper.
    entries.sort(key=lambda x: x[0])

    if not entries:
        raise BridgeError("No tensor-type entries were produced")

    meta = {
        "input_tensor_count": len(plan["plan"]),
        "output_tensor_count": len(entries),
        "skipped_mapper_none_count": len(skipped),
        "skipped_mapper_none": skipped,
        "head_counts": dict(sorted(head_counts.items())),
        "exact_type": exact_type,
    }
    return entries, meta


def write_tensor_type_file(entries: list[tuple[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{name}={ggml_type}\n" for name, ggml_type in entries)
    output.write_text(text, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert Hydra CompressionPlan JSON to llama.cpp tensor-type-file"
    )
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--mapper",
        type=Path,
        default=Path.home() / "hxq-native/tools/convert_hxq_to_gguf.py",
        help="Path to existing convert_hxq_to_gguf.py mapper",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exact-type", choices=("f16", "f32"), default="f16")
    p.add_argument(
        "--receipt",
        type=Path,
        help="Optional JSON receipt path. Defaults to <output>.receipt.json",
    )
    args = p.parse_args()

    plan = load_json(args.plan)
    config = load_json(args.config)
    mapper = load_name_mapper(args.mapper)

    entries, meta = build_tensor_type_entries(
        plan=plan,
        config=config,
        mapper=mapper,
        exact_type=args.exact_type,
    )
    write_tensor_type_file(entries, args.output)

    receipt_path = args.receipt or args.output.with_suffix(
        args.output.suffix + ".receipt.json"
    )
    receipt = {
        "work_order": "GAP-HXQ-009",
        "phase": "bridge_generation",
        "status": "GENERATED_NOT_QUANTIZED",
        "plan": {
            "path": str(args.plan),
            "sha256": sha256_file(args.plan),
        },
        "config": {
            "path": str(args.config),
            "sha256": sha256_file(args.config),
            "model_type": config.get("model_type"),
        },
        "mapper": {
            "path": str(args.mapper),
            "sha256": sha256_file(args.mapper),
            "function": "map_tensor_name_to_gguf",
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
        },
        "mapping": {
            "affine6": "hxq_affine_6",
            "exact": args.exact_type,
        },
        **meta,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(entries)} tensor overrides: {args.output}")
    print(f"SHA256: {receipt['output']['sha256']}")
    print(f"Receipt: {receipt_path}")
    print(f"Heads: {meta['head_counts']}")
    if meta["skipped_mapper_none_count"]:
        print(f"Mapper-skipped tensors: {meta['skipped_mapper_none_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
