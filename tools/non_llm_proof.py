#!/usr/bin/env python3
"""
Non-LLM HelixLinear Proof — Can CDNA v3 compressed execution work beyond LLMs?

Tests HelixLinear drop-in replacement on:
  1. Sentence embedding model (all-MiniLM-L6-v2)  — pure transformer encoder
  2. CLIP ViT vision encoder (openai/clip-vit-base-patch32) — vision transformer
  3. ResNet-18 (torchvision) — CNN with linear classifier head

For each model:
  - Load dense model
  - Extract all nn.Linear weights
  - Compress each to CDNA v3 (VQ codebook + sidecar)
  - Create HelixLinear replacements
  - Run same input through dense vs compressed
  - Compare output cosine similarity
  - Report memory savings

Receipt-grade: real models, real weights, real inputs, cosine fidelity measured.
"""

import json
import platform
import resource
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.helix_linear import HelixLinear, swap_summary
from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.tensor_policy import TensorPolicy, TensorClass


def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ))


def compress_and_replace_linears(model, model_name, tmp_dir):
    """
    Compress all nn.Linear modules in a model to HelixLinear.

    Returns: (compressed_model, stats_dict)
    """
    # Default policy for non-LLM linears
    policy = TensorPolicy(
        tensor_class=TensorClass.UNKNOWN,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        percentile=99.9,
        use_kmeans=True,
        sidecar_enabled=True,
        block_rows=32,
        max_corrections=256,
    )

    cdna_dir = Path(tmp_dir) / f"{model_name}_cdnav3"
    cdna_dir.mkdir(parents=True, exist_ok=True)
    writer = CDNAv3Writer(cdna_dir)

    # Collect all nn.Linear modules
    linear_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules[name] = module

    n_compressed = 0
    n_skipped = 0
    total_dense_bytes = 0
    total_compressed_bytes = 0
    per_layer = []

    for name, module in linear_modules.items():
        weight = module.weight.data.float().cpu().numpy()
        shape = weight.shape

        # Skip tiny layers (< 64 elements — compression overhead exceeds savings)
        if weight.size < 64:
            n_skipped += 1
            continue

        # Ensure 2D
        if weight.ndim == 1:
            weight = weight.reshape(1, -1)

        tensor_name = f"{name}.weight"
        safe_name = tensor_name.replace("/", "_").replace(".", "_")

        try:
            stats = writer.write_tensor(weight, tensor_name, policy=policy)
        except Exception as e:
            print(f"    SKIP {name}: {e}")
            n_skipped += 1
            continue

        # Load back as HelixLinear
        tensor_dir = cdna_dir / f"{safe_name}.cdnav3"
        if not tensor_dir.exists():
            # Try the writer's naming convention
            for d in cdna_dir.glob("*.cdnav3"):
                meta_path = d / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if meta.get("tensor_name") == tensor_name:
                        tensor_dir = d
                        break

        if not (tensor_dir / "codebook.npy").exists():
            n_skipped += 1
            continue

        from helix_substrate.helix_linear import load_helix_linear_from_cdnav3
        bias = module.bias.data.cpu().clone() if module.bias is not None else None
        helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)

        # Verify shape match
        if helix_mod.in_features != module.in_features or helix_mod.out_features != module.out_features:
            n_skipped += 1
            continue

        # Replace in model
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        if parts[-1].isdigit():
            parent[int(parts[-1])] = helix_mod
        else:
            setattr(parent, parts[-1], helix_mod)

        savings = helix_mod.memory_savings()
        dense_bytes = shape[0] * shape[1] * 4
        total_dense_bytes += dense_bytes
        total_compressed_bytes += savings["compressed_bytes"]

        per_layer.append({
            "name": name,
            "shape": list(shape),
            "ratio": savings["ratio"],
            "cosine": stats.get("cosine", stats.get("cosine_fidelity", 0)),
        })
        n_compressed += 1

    return model, {
        "n_compressed": n_compressed,
        "n_skipped": n_skipped,
        "total_dense_bytes": total_dense_bytes,
        "total_compressed_bytes": total_compressed_bytes,
        "overall_ratio": round(total_dense_bytes / max(1, total_compressed_bytes), 2),
        "per_layer": per_layer,
    }


def test_sentence_transformer():
    """Test 1: all-MiniLM-L6-v2 — sentence embedding model."""
    print("\n" + "=" * 70)
    print("  TEST 1: Sentence Transformer (all-MiniLM-L6-v2)")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer

    t0 = time.time()

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Get dense embeddings first
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Compressed neural networks can run on edge devices.",
        "VQ codebooks enable memory-efficient inference.",
        "Shalimar Florida is near Eglin Air Force Base.",
    ]
    dense_embeddings = model.encode(test_sentences, convert_to_tensor=True)
    print(f"  Dense embeddings shape: {dense_embeddings.shape}")

    # Extract the transformer model for compression
    transformer = model[0].auto_model  # The actual transformer

    # Count linears before
    n_linear_before = sum(1 for _, m in transformer.named_modules() if isinstance(m, nn.Linear))
    print(f"  Linear layers: {n_linear_before}")

    # Compress
    with tempfile.TemporaryDirectory() as tmp_dir:
        t_compress = time.time()
        transformer, comp_stats = compress_and_replace_linears(transformer, "minilm", tmp_dir)
        compress_time = time.time() - t_compress
        print(f"  Compressed {comp_stats['n_compressed']}/{n_linear_before} layers in {compress_time:.1f}s")
        print(f"  Compression ratio: {comp_stats['overall_ratio']}x")
        print(f"  Dense: {comp_stats['total_dense_bytes'] / 1e6:.1f} MB → "
              f"Compressed: {comp_stats['total_compressed_bytes'] / 1e6:.1f} MB")

        # Get compressed embeddings
        compressed_embeddings = model.encode(test_sentences, convert_to_tensor=True)

        # Compare
        cosines = []
        for i in range(len(test_sentences)):
            cos = cosine_sim(dense_embeddings[i], compressed_embeddings[i])
            cosines.append(cos)
            print(f"  Sentence {i+1} cosine: {cos:.6f}")

        avg_cosine = sum(cosines) / len(cosines)
        print(f"  Average cosine: {avg_cosine:.6f}")

        # Functional test: do nearest-neighbor rankings change?
        dense_sims = torch.nn.functional.cosine_similarity(
            dense_embeddings.unsqueeze(1), dense_embeddings.unsqueeze(0), dim=2
        )
        comp_sims = torch.nn.functional.cosine_similarity(
            compressed_embeddings.unsqueeze(1), compressed_embeddings.unsqueeze(0), dim=2
        )
        # Check if ranking order is preserved
        dense_ranks = dense_sims.argsort(dim=1, descending=True)
        comp_ranks = comp_sims.argsort(dim=1, descending=True)
        rank_match = (dense_ranks == comp_ranks).float().mean().item()

    return {
        "model": "all-MiniLM-L6-v2",
        "type": "sentence_embedding",
        "n_linear": n_linear_before,
        "n_compressed": comp_stats["n_compressed"],
        "compression_ratio": comp_stats["overall_ratio"],
        "dense_mb": round(comp_stats["total_dense_bytes"] / 1e6, 1),
        "compressed_mb": round(comp_stats["total_compressed_bytes"] / 1e6, 1),
        "avg_cosine": round(avg_cosine, 6),
        "min_cosine": round(min(cosines), 6),
        "rank_preservation": round(rank_match, 4),
        "compress_time_s": round(compress_time, 1),
    }


def test_clip_vit():
    """Test 2: CLIP ViT-B/32 — vision transformer."""
    print("\n" + "=" * 70)
    print("  TEST 2: CLIP Vision Encoder (ViT-B/32)")
    print("=" * 70)

    from transformers import CLIPModel, CLIPProcessor

    t0 = time.time()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    model.eval()

    # Create a test image (random but deterministic)
    torch.manual_seed(42)
    # CLIP expects 224x224 RGB images
    from PIL import Image
    test_images = []
    for i in range(4):
        np.random.seed(i + 100)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_images.append(Image.fromarray(img_array))

    test_texts = [
        "a photo of a cat",
        "a red car on the road",
        "compressed neural network inference",
        "construction site in Florida",
    ]

    # Get dense outputs
    inputs = processor(text=test_texts, images=test_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        dense_outputs = model(**inputs)
        dense_image_embeds = dense_outputs.image_embeds.clone()
        dense_text_embeds = dense_outputs.text_embeds.clone()
        dense_logits = dense_outputs.logits_per_image.clone()

    print(f"  Image embeds: {dense_image_embeds.shape}")
    print(f"  Text embeds: {dense_text_embeds.shape}")

    # Count linears
    vision_linears = sum(1 for n, m in model.vision_model.named_modules() if isinstance(m, nn.Linear))
    text_linears = sum(1 for n, m in model.text_model.named_modules() if isinstance(m, nn.Linear))
    proj_linears = sum(1 for n, m in model.named_modules()
                       if isinstance(m, nn.Linear) and "projection" in n)
    print(f"  Vision linears: {vision_linears}, Text linears: {text_linears}, Projections: {proj_linears}")

    # Compress BOTH vision and text models
    with tempfile.TemporaryDirectory() as tmp_dir:
        t_compress = time.time()

        # Compress vision model
        model.vision_model, vis_stats = compress_and_replace_linears(
            model.vision_model, "clip_vision", tmp_dir)
        # Compress text model
        model.text_model, txt_stats = compress_and_replace_linears(
            model.text_model, "clip_text", tmp_dir)
        # Compress projection layers
        model, proj_stats = compress_and_replace_linears(
            model, "clip_proj", tmp_dir)

        compress_time = time.time() - t_compress

        total_compressed = (vis_stats["n_compressed"] + txt_stats["n_compressed"]
                          + proj_stats["n_compressed"])
        total_dense = (vis_stats["total_dense_bytes"] + txt_stats["total_dense_bytes"]
                      + proj_stats["total_dense_bytes"])
        total_comp = (vis_stats["total_compressed_bytes"] + txt_stats["total_compressed_bytes"]
                     + proj_stats["total_compressed_bytes"])

        print(f"  Compressed {total_compressed} layers in {compress_time:.1f}s")
        print(f"  Vision: {vis_stats['n_compressed']} layers, {vis_stats['overall_ratio']}x")
        print(f"  Text: {txt_stats['n_compressed']} layers, {txt_stats['overall_ratio']}x")
        ratio = round(total_dense / max(1, total_comp), 2)
        print(f"  Overall: {ratio}x ({total_dense/1e6:.1f} MB → {total_comp/1e6:.1f} MB)")

        # Get compressed outputs
        with torch.no_grad():
            comp_outputs = model(**inputs)
            comp_image_embeds = comp_outputs.image_embeds
            comp_text_embeds = comp_outputs.text_embeds
            comp_logits = comp_outputs.logits_per_image

        # Compare embeddings
        img_cosines = [cosine_sim(dense_image_embeds[i], comp_image_embeds[i]) for i in range(4)]
        txt_cosines = [cosine_sim(dense_text_embeds[i], comp_text_embeds[i]) for i in range(4)]
        logit_cosine = cosine_sim(dense_logits, comp_logits)

        print(f"  Image embed cosines: {[round(c, 5) for c in img_cosines]}")
        print(f"  Text embed cosines: {[round(c, 5) for c in txt_cosines]}")
        print(f"  Logits cosine: {logit_cosine:.6f}")

        # Check if image-text matching ranking is preserved
        dense_ranking = dense_logits.argmax(dim=1).tolist()
        comp_ranking = comp_logits.argmax(dim=1).tolist()
        ranking_match = sum(1 for a, b in zip(dense_ranking, comp_ranking) if a == b) / len(dense_ranking)

    return {
        "model": "clip-vit-base-patch32",
        "type": "vision_language",
        "n_linear": vision_linears + text_linears,
        "n_compressed": total_compressed,
        "compression_ratio": ratio,
        "dense_mb": round(total_dense / 1e6, 1),
        "compressed_mb": round(total_comp / 1e6, 1),
        "avg_image_cosine": round(sum(img_cosines) / len(img_cosines), 6),
        "avg_text_cosine": round(sum(txt_cosines) / len(txt_cosines), 6),
        "logits_cosine": round(logit_cosine, 6),
        "ranking_preserved": round(ranking_match, 4),
        "compress_time_s": round(compress_time, 1),
    }


def test_resnet():
    """Test 3: ResNet-18 — CNN (only the FC classifier head is nn.Linear)."""
    print("\n" + "=" * 70)
    print("  TEST 3: ResNet-18 (CNN — linear classifier head)")
    print("=" * 70)

    import torchvision.models as models

    t0 = time.time()
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # ResNet has one nn.Linear: the fc (fully connected) classifier
    n_linear = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    print(f"  Linear layers: {n_linear}")
    print(f"  FC layer: {model.fc.in_features} → {model.fc.out_features}")

    # Create test inputs (random images)
    torch.manual_seed(42)
    test_input = torch.randn(8, 3, 224, 224)

    with torch.no_grad():
        dense_output = model(test_input).clone()
        dense_pred = dense_output.argmax(dim=1)
    print(f"  Dense predictions: {dense_pred.tolist()}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        t_compress = time.time()
        model, comp_stats = compress_and_replace_linears(model, "resnet18", tmp_dir)
        compress_time = time.time() - t_compress

        print(f"  Compressed {comp_stats['n_compressed']} layers in {compress_time:.1f}s")
        if comp_stats['total_compressed_bytes'] > 0:
            print(f"  Ratio: {comp_stats['overall_ratio']}x")

        with torch.no_grad():
            comp_output = model(test_input)
            comp_pred = comp_output.argmax(dim=1)

        output_cosine = cosine_sim(dense_output, comp_output)
        pred_match = (dense_pred == comp_pred).float().mean().item()

        print(f"  Compressed predictions: {comp_pred.tolist()}")
        print(f"  Output cosine: {output_cosine:.6f}")
        print(f"  Prediction match: {pred_match * 100:.0f}%")

    return {
        "model": "ResNet-18",
        "type": "cnn_classifier",
        "n_linear": n_linear,
        "n_compressed": comp_stats["n_compressed"],
        "compression_ratio": comp_stats["overall_ratio"],
        "dense_mb": round(comp_stats["total_dense_bytes"] / 1e6, 1),
        "compressed_mb": round(comp_stats["total_compressed_bytes"] / 1e6, 1),
        "output_cosine": round(output_cosine, 6),
        "prediction_match": round(pred_match, 4),
        "compress_time_s": round(compress_time, 1),
    }


def main():
    t_global = time.time()
    cpu_start = time.process_time()
    start_iso = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("  Non-LLM HelixLinear Proof")
    print(f"  {start_iso}")
    print("=" * 70)

    results = {}

    # Test 1: Sentence Transformer (cached, no download needed)
    try:
        results["sentence_transformer"] = test_sentence_transformer()
        print(f"\n  RESULT: {results['sentence_transformer']['avg_cosine']:.6f} cosine, "
              f"{results['sentence_transformer']['compression_ratio']}x compression")
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback; traceback.print_exc()
        results["sentence_transformer"] = {"error": str(e)}

    # Test 2: CLIP (may need download — ~600MB)
    try:
        results["clip_vit"] = test_clip_vit()
        print(f"\n  RESULT: image={results['clip_vit']['avg_image_cosine']:.6f}, "
              f"text={results['clip_vit']['avg_text_cosine']:.6f}, "
              f"{results['clip_vit']['compression_ratio']}x compression")
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback; traceback.print_exc()
        results["clip_vit"] = {"error": str(e)}

    # Test 3: ResNet (torchvision, usually cached or small download)
    try:
        results["resnet18"] = test_resnet()
        print(f"\n  RESULT: {results['resnet18']['output_cosine']:.6f} cosine, "
              f"{results['resnet18']['prediction_match'] * 100:.0f}% predictions preserved")
    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback; traceback.print_exc()
        results["resnet18"] = {"error": str(e)}

    # Summary
    wall_time = time.time() - t_global
    cpu_time = time.process_time() - cpu_start

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for name, r in results.items():
        if "error" in r:
            print(f"  {name}: FAILED — {r['error']}")
        else:
            cosine_key = "avg_cosine" if "avg_cosine" in r else "output_cosine"
            if "avg_image_cosine" in r:
                cosine_key = "avg_image_cosine"
            print(f"  {name}: cosine={r.get(cosine_key, 'N/A')}, "
                  f"ratio={r.get('compression_ratio', 'N/A')}x, "
                  f"{r.get('dense_mb', '?')} MB → {r.get('compressed_mb', '?')} MB")

    # Receipt
    receipt = {
        "experiment": "non_llm_helix_linear_proof",
        "timestamp": start_iso,
        "results": results,
        "cost": {
            "wall_time_s": round(wall_time, 3),
            "cpu_time_s": round(cpu_time, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
        },
        "verdict": {},
    }

    # Verdicts
    for name, r in results.items():
        if "error" in r:
            receipt["verdict"][name] = "FAILED"
        else:
            cosine_key = "avg_cosine" if "avg_cosine" in r else "output_cosine"
            if "avg_image_cosine" in r:
                cosine_key = "avg_image_cosine"
            cos = r.get(cosine_key, 0)
            if cos >= 0.999:
                receipt["verdict"][name] = "PROVEN"
            elif cos >= 0.99:
                receipt["verdict"][name] = "STRONG"
            elif cos >= 0.95:
                receipt["verdict"][name] = "USABLE"
            else:
                receipt["verdict"][name] = "WEAK"

    receipt_dir = Path(__file__).parent.parent / "receipts" / "non_llm_proof"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"non_llm_proof_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time:.1f}s")

    # Print verdicts
    print(f"\n  VERDICTS:")
    for name, verdict in receipt["verdict"].items():
        print(f"    {name}: {verdict}")

    print("=" * 70)


if __name__ == "__main__":
    main()
