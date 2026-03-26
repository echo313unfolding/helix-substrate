#!/usr/bin/env python3
"""Domain 5: Neural Video/Audio Codec Weight Compression.
Compress CLIP ViT (a perceptual codec) and ResNet-18 with CDNA v3.
Uses transformers.CLIPModel (not openai/clip package)."""

import torch, json, tempfile, numpy as np
from pathlib import Path
from _common import *

def compress_and_replace_linears(model_module, model_name, tmp_dir, policy):
    """Compress all nn.Linear in a module, replace with HelixLinear. Returns stats dict."""
    import torch.nn as nn
    from helix_substrate.helix_linear import load_helix_linear_from_cdnav3

    cdna_dir = Path(tmp_dir) / f"{model_name}_cdnav3"
    cdna_dir.mkdir(parents=True, exist_ok=True)
    writer = CDNAv3Writer(cdna_dir)

    linear_modules = {name: mod for name, mod in model_module.named_modules()
                      if isinstance(mod, nn.Linear)}

    n_compressed = 0
    total_dense_bytes = 0
    total_comp_bytes = 0
    per_layer = []

    for name, module in linear_modules.items():
        weight = module.weight.data.float().cpu().numpy()
        if weight.size < 64:
            continue
        if weight.ndim == 1:
            weight = weight.reshape(1, -1)

        tensor_name = f"{model_name}.{name}.weight"
        safe = tensor_name.replace(".", "_").replace("/", "_")

        try:
            stats = writer.write_tensor(weight, tensor_name, policy=policy)
        except Exception as e:
            print(f"    SKIP {name}: {e}")
            continue

        tensor_dir = cdna_dir / f"{safe}.cdnav3"
        if not tensor_dir.exists():
            # Try to find the directory by scanning
            for d in cdna_dir.glob("*.cdnav3"):
                meta_path = d / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if meta.get("tensor_name") == tensor_name:
                        tensor_dir = d
                        break

        if not (tensor_dir / "codebook.npy").exists():
            continue

        bias = module.bias.data.cpu().clone() if module.bias is not None else None
        helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)

        # Replace in model
        parts = name.split(".")
        parent = model_module
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, parts[-1], helix_mod)

        dense_bytes = weight.shape[0] * weight.shape[1] * 4
        savings = helix_mod.memory_savings()
        total_dense_bytes += dense_bytes
        total_comp_bytes += savings["compressed_bytes"]
        per_layer.append({"name": name, "ratio": savings["ratio"],
                          "cosine": stats.get("cosine", stats.get("cosine_fidelity", 0))})
        n_compressed += 1

    return {
        "n_compressed": n_compressed,
        "total_dense_bytes": total_dense_bytes,
        "total_compressed_bytes": total_comp_bytes,
        "overall_ratio": round(total_dense_bytes / max(1, total_comp_bytes), 2),
        "per_layer": per_layer,
    }

def test_clip():
    """Test CLIP ViT-B/32 text encoder via transformers.CLIPModel."""
    print("\n  --- CLIP ViT-B/32 (text encoder) ---")
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = model.float().eval()

    texts = ["a photo of a cat", "a photo of a dog", "a sunset over ocean",
             "a city skyline at night", "a person playing guitar"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        dense_features = model.get_text_features(**inputs)
        dense_features = dense_features / dense_features.norm(dim=-1, keepdim=True)

    policy = policy_vq(k=256, sidecar=True)
    text_encoder = model.text_model

    with tempfile.TemporaryDirectory() as tmp_dir:
        comp_stats = compress_and_replace_linears(text_encoder, "clip_text", tmp_dir, policy)
        print(f"    Compressed {comp_stats['n_compressed']} layers, ratio={comp_stats['overall_ratio']}x")

        with torch.no_grad():
            comp_features = model.get_text_features(**inputs)
            comp_features = comp_features / comp_features.norm(dim=-1, keepdim=True)

    cosines = []
    for i, text in enumerate(texts):
        cos = cosine_sim(dense_features[i].numpy(), comp_features[i].numpy())
        cosines.append(cos)
        print(f"    '{text}': cosine={cos:.6f}")

    return {
        "model": "CLIP ViT-B/32 text encoder (transformers)",
        "n_compressed": comp_stats["n_compressed"],
        "overall_ratio": comp_stats["overall_ratio"],
        "dense_mb": round(comp_stats["total_dense_bytes"] / 1e6, 1),
        "compressed_mb": round(comp_stats["total_compressed_bytes"] / 1e6, 1),
        "embedding_cosines": [round(c, 6) for c in cosines],
        "avg_cosine": round(float(np.mean(cosines)), 6),
        "min_cosine": round(min(cosines), 6),
    }

def test_resnet():
    """Test ResNet-18 classifier head."""
    print("\n  --- ResNet-18 (classifier head) ---")
    import torchvision.models as models

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).float().eval()

    # Random image input
    torch.manual_seed(42)
    x = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        dense_out = model(x)

    policy = policy_vq(k=256, sidecar=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Only compress the fc layer (512 x 1000)
        fc_weight = model.fc.weight.data.float().cpu().numpy()
        writer = CDNAv3Writer(Path(tmp_dir))
        stats = writer.write_tensor(fc_weight, "resnet_fc", policy=policy)

        from helix_substrate.helix_linear import load_helix_linear_from_cdnav3
        tensor_dir = Path(tmp_dir) / "resnet_fc.cdnav3"
        bias = model.fc.bias.data.cpu().clone()
        helix_fc = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
        model.fc = helix_fc

        with torch.no_grad():
            comp_out = model(x)

    cos = cosine_sim(dense_out.numpy(), comp_out.numpy())
    pred_match = float((dense_out.argmax(dim=1) == comp_out.argmax(dim=1)).float().mean())

    print(f"    Output cosine: {cos:.6f}")
    print(f"    Prediction match: {pred_match:.2%}")

    return {
        "model": "ResNet-18 (torchvision)",
        "fc_shape": list(fc_weight.shape),
        "output_cosine": round(cos, 6),
        "prediction_match": round(pred_match, 4),
        "compression_ratio": stats.get("compression_ratio", 1.0),
    }

def main():
    t_start, cpu_start, start_iso = start_cost()
    print("=" * 72)
    print("DOMAIN 5: Neural Codec Weight Compression (Compress the Compressor)")
    print("=" * 72)

    clip_result = test_clip()
    resnet_result = test_resnet()

    all_cosines = clip_result["embedding_cosines"] + [resnet_result["output_cosine"]]
    v, worst = verdict(all_cosines)
    print(f"\n  Overall verdict: {v} (worst cosine={worst:.6f})")

    cost = finish_cost(t_start, cpu_start, start_iso)
    write_receipt("tensor_infra_domain_5", "codec_weight_compress", {
        "clip": clip_result,
        "resnet": resnet_result,
        "verdict": v,
        "data_source": "REAL — CLIP ViT-B/32 + ResNet-18 pre-trained weights",
    }, cost)

if __name__ == "__main__":
    main()
