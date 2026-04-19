#!/usr/bin/env python3
"""
WO-35: Multi-Model Cross-Architecture MoE Demo.

Loads 3 compressed models onto one GPU simultaneously, routes a mixed workload
to the best expert, measures per-query latency and VRAM per model, emits a receipt.

Models:
    - Qwen2.5-3B-Instruct (reasoning/code) — Transformer with GQA
    - Zamba2-1.2B (state tracking) — Hybrid Mamba2+Transformer
    - CLIP ViT-B/32 (vision) — Dense FP16 (~300MB, tiny)

VRAM estimate (RTX 4090, 24GB):
    Qwen ~2.5-3GB + Zamba ~1-1.5GB + CLIP ~0.3GB + CUDA context ~0.5GB ≈ 4.5-5.5GB total.

CDNA v3 — how one codec covers all three architectures:
    Universal rule: 2D weights → VQ-256 + sidecar. 1D/embeddings/lm_head/conv1d → exact.
    - Qwen (Transformer): All Q/K/V/O + FFN projections compressed. Tied embeddings stored once.
    - Zamba2 (Hybrid): 136 HelixLinear (Mamba in_proj/out_proj + shared transformer + LoRA).
      conv1d exact (kurtosis ~48.6).
    - CLIP (ViT): Proven 3.98x, cos=0.997/0.999. Loaded dense here since it's tiny.
    Forward pass: W = codebook[indices] + sidecar, then X@W via Triton kernel.
    Compressed form IS the executable — no decompression step.

Usage:
    # Dry run (no GPU needed)
    python3 tools/multi_model_demo.py --dry-run

    # Single model
    python3 tools/multi_model_demo.py --models qwen

    # Full demo on GPU
    python3 tools/multi_model_demo.py --models qwen zamba clip

    # Custom workload
    python3 tools/multi_model_demo.py --workload queries.json

Work Order: WO-MULTI-MODEL-DEMO-01
"""

import argparse
import gc
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from helix_substrate.device_utils import (
    device_info,
    empty_cache,
    memory_allocated,
    resolve_device,
    synchronize_device,
)
from helix_substrate.helix_linear import swap_summary


# ---------------------------------------------------------------------------
# Router — simple 3-way keyword classifier (standalone, not extending production code)
# ---------------------------------------------------------------------------

ROUTING_KEYWORDS = {
    "vision": {
        "keywords": [
            "image", "picture", "photo", "visual", "describe image",
            "classify image", "scene", "object detection", "what is in",
            "look at", "see in", "photograph", "camera", "pixel",
        ],
        "weight": 0.5,  # highest — vision keywords are most specific
    },
    "state_tracking": {
        "keywords": [
            "sequence", "next", "previous", "track", "remember",
            "running total", "maintain", "update state", "accumulate",
            "history", "keep count", "step by step state", "recurrence",
            "memory state", "rolling",
        ],
        "weight": 0.3,
    },
    "reasoning": {
        "keywords": [
            "explain", "code", "implement", "python", "calculate",
            "compare", "analyze", "write", "function", "algorithm",
            "debug", "optimize", "refactor", "reason", "logic",
            "why", "how does", "what is", "define", "solve",
        ],
        "weight": 0.2,
    },
}


def route_query(query: str) -> str:
    """Route a query to the best expert via keyword matching. Default: reasoning."""
    query_lower = query.lower()
    scores = {}
    for expert, cfg in ROUTING_KEYWORDS.items():
        score = sum(cfg["weight"] for kw in cfg["keywords"] if kw in query_lower)
        scores[expert] = score
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "reasoning"  # default fallback
    return best


EXPERT_TO_MODEL = {
    "reasoning": "qwen",
    "state_tracking": "zamba",
    "vision": "clip",
}


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "qwen": {
        "hf_repo": "EchoLabs33/qwen2.5-3b-instruct-helix",
        "local_fallback": Path.home() / "models" / "qwen2.5-3b-instruct-helix",
        "type": "causal_lm",
        "trust_remote_code": True,
        "arch": "transformer",
        "description": "Qwen2.5-3B-Instruct (Transformer, GQA) — reasoning/code expert",
    },
    "zamba": {
        "hf_repo": "EchoLabs33/zamba2-1.2b-helix",
        "local_fallback": Path.home() / "models" / "zamba2-1.2b-helix",
        "type": "causal_lm",
        "trust_remote_code": True,
        "arch": "hybrid_ssm",
        "description": "Zamba2-1.2B (Hybrid Mamba2+Transformer) — state tracking expert",
    },
    "clip": {
        "hf_repo": "openai/clip-vit-base-patch32",
        "local_fallback": None,
        "type": "clip",
        "trust_remote_code": False,
        "arch": "vision_transformer",
        "description": "CLIP ViT-B/32 (Dense FP16) — vision expert",
    },
}


# ---------------------------------------------------------------------------
# Built-in workload
# ---------------------------------------------------------------------------

BUILTIN_WORKLOAD = [
    # Reasoning (5)
    {"query": "Explain how Python's GIL affects multithreading performance", "expected": "reasoning"},
    {"query": "Write a function to check if a binary tree is balanced", "expected": "reasoning"},
    {"query": "Compare quicksort and mergesort time complexity", "expected": "reasoning"},
    {"query": "Implement a simple LRU cache in Python", "expected": "reasoning"},
    {"query": "Analyze the trade-offs between microservices and monoliths", "expected": "reasoning"},
    # State tracking (5)
    {"query": "Track the running total of this sequence: 3, 7, 2, 8, 1", "expected": "state_tracking"},
    {"query": "Remember this sequence and tell me the next element: 1, 1, 2, 3, 5", "expected": "state_tracking"},
    {"query": "Maintain a rolling average over these values: 10, 20, 30, 40", "expected": "state_tracking"},
    {"query": "Update state: counter was 5, increment by 3, then decrement by 1", "expected": "state_tracking"},
    {"query": "Keep count of previous operations and report accumulated history", "expected": "state_tracking"},
    # Vision (5)
    {"query": "Describe the image: what objects are visible?", "expected": "vision"},
    {"query": "Classify this image into one of: cat, dog, car, house", "expected": "vision"},
    {"query": "What scene is shown in this photograph?", "expected": "vision"},
    {"query": "Detect objects in the picture and list them", "expected": "vision"},
    {"query": "Look at this photo and describe the visual elements", "expected": "vision"},
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_model_path(name: str, config: dict) -> str:
    """Return the best available model path (local dir or HF repo)."""
    local = config.get("local_fallback")
    if local and local.exists() and (local / "config.json").exists():
        return str(local)
    return config["hf_repo"]


def load_model_causal(name: str, config: dict, device: str, dry_run: bool) -> dict:
    """Load a CausalLM model (Qwen or Zamba) and return model+tokenizer+metadata."""
    if dry_run:
        return {
            "model": None, "tokenizer": None, "name": name,
            "vram_mb": 2500.0 if name == "qwen" else 1200.0,
            "load_time_s": 0.0, "helix_summary": {"helix_modules": 0, "dry_run": True},
        }

    import helix_substrate  # noqa: F401 — registers HF auto-quantizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = _resolve_model_path(name, config)
    print(f"  Loading {name} from {model_path}...", file=sys.stderr, flush=True)

    vram_before = memory_allocated(device)
    t0 = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=config["trust_remote_code"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device if device != "cpu" else None,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=config["trust_remote_code"],
        ).eval()
    except Exception as e:
        err_str = str(e)
        if "mamba" in err_str.lower() or "ssm" in err_str.lower():
            print(f"  WARNING: {name} load failed (likely mamba-ssm dependency): {err_str}",
                  file=sys.stderr, flush=True)
            return {
                "model": None, "tokenizer": None, "name": name,
                "error": err_str, "vram_mb": 0.0, "load_time_s": 0.0,
                "helix_summary": None,
            }
        raise

    if device != "cpu":
        synchronize_device(device)
    load_time = round(time.time() - t0, 2)
    vram_after = memory_allocated(device)
    vram_delta = round(vram_after - vram_before, 1)

    summary = swap_summary(model)
    print(f"  {name}: {summary['helix_modules']} HelixLinear, "
          f"{vram_delta:.0f} MB VRAM, {load_time}s",
          file=sys.stderr, flush=True)

    return {
        "model": model, "tokenizer": tokenizer, "name": name,
        "vram_mb": vram_delta, "load_time_s": load_time,
        "helix_summary": summary,
    }


def load_model_clip(config: dict, device: str, dry_run: bool) -> dict:
    """Load CLIP model and return model+processor+metadata."""
    if dry_run:
        return {
            "model": None, "processor": None, "name": "clip",
            "vram_mb": 300.0, "load_time_s": 0.0,
            "helix_summary": {"note": "dense FP16, no HelixLinear"},
        }

    from transformers import CLIPModel, CLIPProcessor

    model_path = config["hf_repo"]
    print(f"  Loading clip from {model_path}...", file=sys.stderr, flush=True)

    vram_before = memory_allocated(device)
    t0 = time.time()

    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device).eval()

    if device != "cpu":
        synchronize_device(device)
    load_time = round(time.time() - t0, 2)
    vram_after = memory_allocated(device)
    vram_delta = round(vram_after - vram_before, 1)

    print(f"  clip: dense FP16, {vram_delta:.0f} MB VRAM, {load_time}s",
          file=sys.stderr, flush=True)

    return {
        "model": model, "processor": processor, "name": "clip",
        "vram_mb": vram_delta, "load_time_s": load_time,
        "helix_summary": {"note": "dense FP16, no HelixLinear"},
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_causal(loaded: dict, query: str, max_tokens: int, device: str,
                 dry_run: bool) -> dict:
    """Run CausalLM inference. Returns response text + latency."""
    if dry_run or loaded.get("model") is None:
        return {"response": f"[DRY RUN] {loaded['name']}: {query[:40]}...",
                "latency_ms": 0.0}

    model, tokenizer = loaded["model"], loaded["tokenizer"]
    messages = [{"role": "user", "content": query}]

    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback for models without chat template
        input_text = f"User: {query}\nAssistant:"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs.input_ids.to(device if device != "cpu" else "cpu")

    if device != "cpu":
        synchronize_device(device)
    t0 = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device != "cpu":
        synchronize_device(device)
    latency = round((time.time() - t0) * 1000, 1)

    new_tokens = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {"response": response.strip(), "latency_ms": latency}


def infer_clip(loaded: dict, query: str, device: str, dry_run: bool,
               image_dir: Path | None = None) -> dict:
    """Run CLIP zero-shot classification. Returns similarity scores + latency."""
    if dry_run or loaded.get("model") is None:
        return {"response": "[DRY RUN] clip similarity scores", "latency_ms": 0.0}

    from PIL import Image as PILImage

    model, processor = loaded["model"], loaded["processor"]

    # Get or create image
    if image_dir and image_dir.exists():
        # Use first image found
        img_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        if img_files:
            image = PILImage.open(img_files[0]).convert("RGB")
        else:
            image = _synthetic_image()
    else:
        image = _synthetic_image()

    # Extract labels from query or use defaults
    labels = ["cat", "dog", "car", "house", "landscape", "person", "food", "text"]

    inputs = processor(text=labels, images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if device != "cpu":
        synchronize_device(device)
    t0 = time.time()

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)

    if device != "cpu":
        synchronize_device(device)
    latency = round((time.time() - t0) * 1000, 1)

    scores = {label: round(float(probs[0][i]), 4) for i, label in enumerate(labels)}
    top = max(scores, key=scores.get)
    response = f"Top: {top} ({scores[top]:.3f}). All: {json.dumps(scores)}"

    return {"response": response, "latency_ms": latency}


def _synthetic_image():
    """Deterministic synthetic 224x224 image for demo."""
    from PIL import Image as PILImage
    rng = np.random.RandomState(42)
    pixels = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return PILImage.fromarray(pixels)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(args):
    """Load models, route workload, measure everything, emit receipt."""
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    device = resolve_device(args.device) if not args.dry_run else "cpu"
    print(f"\n{'='*70}")
    print(f"  MULTI-MODEL DEMO — WO-35")
    print(f"  Device: {device}")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*70}\n")

    # ── Load models ──
    loaded_models = {}
    vram_log = []
    model_receipt = {}

    for model_name in args.models:
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            print(f"  WARNING: Unknown model '{model_name}', skipping", file=sys.stderr)
            continue

        vram_before = memory_allocated(device)

        if config["type"] == "clip":
            loaded = load_model_clip(config, device, args.dry_run)
        else:
            loaded = load_model_causal(model_name, config, device, args.dry_run)

        if loaded.get("error"):
            model_receipt[model_name] = {
                "status": "failed",
                "error": loaded["error"],
                "description": config["description"],
            }
            print(f"  SKIPPED {model_name}: {loaded['error'][:100]}", file=sys.stderr)
            continue

        loaded_models[model_name] = loaded
        cumulative_vram = memory_allocated(device)
        vram_log.append({
            "after_loading": model_name,
            "cumulative_vram_mb": round(cumulative_vram, 1),
        })

        model_receipt[model_name] = {
            "status": "loaded",
            "vram_delta_mb": loaded["vram_mb"],
            "load_time_s": loaded["load_time_s"],
            "helix_summary": loaded["helix_summary"],
            "description": config["description"],
            "arch": config["arch"],
        }

    available_experts = set()
    for mname in loaded_models:
        for expert, mapped_model in EXPERT_TO_MODEL.items():
            if mapped_model == mname:
                available_experts.add(expert)

    print(f"\n  Loaded {len(loaded_models)}/{len(args.models)} models. "
          f"Available experts: {sorted(available_experts)}\n",
          file=sys.stderr, flush=True)

    # ── Load workload ──
    if args.workload:
        workload = json.loads(Path(args.workload).read_text())
    else:
        workload = BUILTIN_WORKLOAD

    # ── Route and execute ──
    results = []
    for i, item in enumerate(workload):
        query = item["query"]
        expected = item.get("expected", None)

        # Route
        routed_expert = route_query(query)
        routed_model = EXPERT_TO_MODEL.get(routed_expert, "qwen")

        # Fallback if routed model isn't loaded
        if routed_model not in loaded_models:
            # Try any available causal model
            for fallback in ["qwen", "zamba"]:
                if fallback in loaded_models:
                    routed_model = fallback
                    break
            else:
                results.append({
                    "query": query, "routed_to": routed_expert,
                    "expected": expected, "correct": expected == routed_expert,
                    "latency_ms": 0.0, "response": "[NO MODEL AVAILABLE]",
                })
                continue

        loaded = loaded_models[routed_model]

        # Execute
        if MODEL_CONFIGS[routed_model]["type"] == "clip":
            out = infer_clip(loaded, query, device, args.dry_run,
                             getattr(args, 'image_dir', None))
        else:
            out = infer_causal(loaded, query, args.max_tokens, device, args.dry_run)

        correct = (expected == routed_expert) if expected else None
        tag = "OK" if correct else ("MISS" if correct is False else "?")

        results.append({
            "query": query,
            "routed_to": routed_expert,
            "model": routed_model,
            "expected": expected,
            "correct": correct,
            "latency_ms": out["latency_ms"],
            "response": out["response"][:200],
        })

        print(f"  [{i+1:2d}/{len(workload)}] {tag} → {routed_expert}/{routed_model} "
              f"({out['latency_ms']:.0f}ms): {query[:50]}...",
              file=sys.stderr, flush=True)

    # ── Compute stats ──
    correct_count = sum(1 for r in results if r.get("correct") is True)
    total_with_expected = sum(1 for r in results if r.get("expected") is not None)
    routing_accuracy = {
        "correct": correct_count,
        "total": total_with_expected,
        "pct": round(correct_count / total_with_expected * 100, 1) if total_with_expected > 0 else 0,
    }

    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    latency_summary = {}
    if latencies:
        latencies_sorted = sorted(latencies)
        latency_summary = {
            "median_ms": round(float(np.median(latencies_sorted)), 1),
            "p95_ms": round(float(np.percentile(latencies_sorted, 95)), 1),
            "min_ms": round(min(latencies_sorted), 1),
            "max_ms": round(max(latencies_sorted), 1),
            "count": len(latencies_sorted),
        }

    # ── Receipt ──
    wall_time = round(time.time() - t_start, 3)
    cpu_time = round(time.process_time() - cpu_start, 3)
    peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

    receipt = {
        "work_order": "WO-MULTI-MODEL-DEMO-01",
        "device": device_info(device) if not args.dry_run else {"device": "cpu", "dry_run": True},
        "models": model_receipt,
        "vram_log": vram_log,
        "workload": results,
        "routing_accuracy": routing_accuracy,
        "latency_summary": latency_summary,
        "cost": {
            "wall_time_s": wall_time,
            "cpu_time_s": cpu_time,
            "peak_memory_mb": peak_mb,
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    # Save receipt
    output_path = Path(args.output) if args.output else (
        REPO_DIR / "receipts" / "multi_model" / f"demo_{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(receipt, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"  DEMO COMPLETE")
    print(f"  Models loaded: {len(loaded_models)}/{len(args.models)}")
    if vram_log:
        print(f"  Total VRAM: {vram_log[-1]['cumulative_vram_mb']:.0f} MB")
    print(f"  Routing accuracy: {routing_accuracy['correct']}/{routing_accuracy['total']} "
          f"({routing_accuracy['pct']}%)")
    if latency_summary:
        print(f"  Latency: median={latency_summary['median_ms']:.0f}ms, "
              f"p95={latency_summary['p95_ms']:.0f}ms")
    print(f"  Wall time: {wall_time}s")
    print(f"  Receipt: {output_path}")
    print(f"{'='*70}\n")

    # JSON to stdout for scripting
    print(json.dumps(receipt))

    # Cleanup
    for loaded in loaded_models.values():
        if loaded.get("model") is not None:
            del loaded["model"]
    gc.collect()
    empty_cache(device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model cross-architecture MoE demo. "
                    "Loads compressed models on one GPU, routes queries to best expert.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 tools/multi_model_demo.py --dry-run\n"
               "  python3 tools/multi_model_demo.py --models qwen\n"
               "  python3 tools/multi_model_demo.py --models qwen zamba clip\n"
               "  python3 tools/multi_model_demo.py --workload queries.json\n",
    )

    parser.add_argument("--dry-run", action="store_true",
                        help="Mock all loads, validate routing + receipt format")
    parser.add_argument("--models", nargs="+", default=["qwen", "zamba", "clip"],
                        choices=["qwen", "zamba", "clip"],
                        help="Models to load (default: all three)")
    parser.add_argument("--workload", type=str, default=None,
                        help="JSON file with custom workload (default: built-in 15 queries)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output receipt path (default: receipts/multi_model/demo_TIMESTAMP.json)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens for CausalLM generation (default: 64)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (default: auto)")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Directory with images for CLIP (default: synthetic)")

    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
