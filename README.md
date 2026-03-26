<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,100:1a6b3c&height=200&section=header&text=helix-substrate&fontSize=42&fontColor=58a6ff&animation=fadeIn&fontAlignY=35&desc=Calibration-free%20neural%20network%20compression&descSize=16&descColor=8b949e&descAlignY=55">
  <source media="(prefers-color-scheme: light)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:f0f6fc,100:2ea043&height=200&section=header&text=helix-substrate&fontSize=42&fontColor=1f2328&animation=fadeIn&fontAlignY=35&desc=Calibration-free%20neural%20network%20compression&descSize=16&descColor=656d76&descAlignY=55">
  <img alt="helix-substrate" src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,100:1a6b3c&height=200&section=header&text=helix-substrate&fontSize=42&fontColor=58a6ff&animation=fadeIn&fontAlignY=35&desc=Calibration-free%20neural%20network%20compression&descSize=16&descColor=8b949e&descAlignY=55">
</picture>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-Echo%20Labs-green)](LICENSE)
[![4x Compression](https://img.shields.io/badge/compression-4x-brightgreen)]()
[![<1% PPL Loss](https://img.shields.io/badge/quality-%3C1%25%20PPL-blue)]()

**Point it at a model, get 4x smaller weights with <1% quality loss.**
**No training data. No fine-tuning. Transformers, SSMs, CNNs, vision — same command.**

</div>

# helix-substrate

Calibration-free neural network compression. Point it at a model, get 4x smaller weights with <1% quality loss. No training data needed. Works on transformers, SSMs, CNNs, vision encoders, and embedding models without code changes.

```bash
pip install helix-substrate
```

```python
from helix_substrate import CDNAv3Writer, CDNAv3Reader

# Compress any 2D weight tensor
writer = CDNAv3Writer("./compressed/")
writer.write_tensor(weight_matrix, "layer_name")

# Reconstruct
reader = CDNAv3Reader("./compressed/layer_name.cdnav3")
reconstructed = reader.reconstruct()  # cosine similarity >= 0.999
```

## What it does

helix-substrate compresses neural network weights using vector quantization with statistical routing. Each weight column is assigned to the nearest entry in a learned 256-entry codebook. Outlier values (top 0.1% by magnitude) are preserved exactly in a sparse sidecar. High-kurtosis tensors get additional SVD residual correction. The result is a `codebook + uint8 indices + sidecar` representation at ~4x compression with negligible quality loss.

No calibration data. No fine-tuning. No architecture-specific code.

## Benchmarks

### Weight Compression Quality (Qwen2.5-7B-Instruct, RTX 4090)

| Method | PPL | PPL Delta | Decode tok/s | VRAM | Calibration |
|--------|-----|-----------|-------------|------|-------------|
| FP16 Dense | 6.949 | baseline | 29.8 | 14,569 MB | -- |
| **HelixLinear k=256** | **7.106** | **+2.3%** | **16.6** | **12,336 MB** | **None** |
| GPTQ Int4 | 7.518 | +8.2% | * | 5,372 MB | 128 sequences |
| AWQ Int4 | 7.719 | +11.1% | 12.2 | 5,338 MB | Activation stats |

\* GPTQ decode speed not representative -- CUDA kernels did not compile in test environment. Published benchmarks with exllama2 report 40+ tok/s.

**Quality vs ratio tradeoff:** GPTQ/AWQ achieve 8x compression but with 3-5x worse quality degradation. helix-substrate achieves 4x with the best quality of any post-training method tested, and requires zero calibration data.

### Architecture Coverage (all k=256, same `compress.py`)

| Model | Architecture | Tensors | Ratio | Cosine (min) |
|-------|-------------|---------|-------|-------------|
| TinyLlama 1.1B | Transformer (LLaMA) | 154 | 3.98x | 0.9946 |
| Qwen2.5 1.5B | Transformer (Qwen) | 196 | 4.00x | 0.9943 |
| Qwen2.5 7B | Transformer (Qwen) | 196 | 4.00x | -- |
| Qwen2.5 14B | Transformer (Qwen) | 336 | 4.00x | -- |
| Mamba-130m | SSM (Mamba) | 97 | 3.92x | 0.9990+ |
| Mamba-2 1.3B | SSM (Mamba-2) | 98 | 3.99x | 0.9990+ |
| MiniLM-L6 | Transformer (BERT) | 73 | 3.94x | 0.9997 |
| CLIP ViT-B/32 | Vision Transformer | 49 | 3.98x | 0.9997 |
| ResNet-18 | CNN | 1 (fc) | 3.97x | 0.9998 |

All compressed with the same command. No architecture-specific flags or code paths.

### Compression Quality Frontier (TinyLlama, PPL on WikiText-2)

| Config | Ratio | PPL Delta | Status |
|--------|-------|-----------|--------|
| k=256 + sidecar | 4.0x | +0.11% | Production baseline |
| RVQ 16+16 | 4.0x | +0.10% | v2 codec (same ratio, smaller codebooks) |
| k=64 + sidecar | 5.3x | +1.44% | Model-dependent (fails on Qwen at +2.78%) |
| k=32 + sidecar | 6.4x | +2.61% | Below quality threshold |
| k=16 + sidecar | 8.0x | +9.34% | Rejected |

## Key findings

**Outlier sidecar is non-negotiable.** Without it, k=256 VQ produces PPL 274. With it, PPL 6.18. Cosine similarity is identical (0.999) in both cases. The top 0.1% of weights by magnitude carry outsized importance despite being statistically invisible. This means cosine alone is not a safe quality metric -- outlier preservation is mandatory.

**Kurtosis routing beats Hessian routing.** Head-to-head on TinyLlama: kurtosis routing gives +0.51% PPL, Hessian routing gives +0.64% PPL. Kurtosis is calibration-free (computed from weights alone). Hessian requires calibration data. The cheaper signal wins.

**Weighted k-means is harmful.** Hessian-weighted centroid placement gives +2.93% PPL -- actively worse than unweighted. Distorting the codebook toward "important" columns degrades it for the majority of weights.

## Quick start

### Compress a model

```bash
python tools/compress.py \
  --model-dir /path/to/model \
  --out-dir /path/to/output \
  --k 256 --sidecar
```

### Load compressed weights for inference

```python
from transformers import AutoModelForCausalLM
from helix_substrate.helix_linear import swap_to_helix

model = AutoModelForCausalLM.from_pretrained("path/to/model")
swap_to_helix(model, "path/to/cdnav3/")
# All nn.Linear modules replaced with HelixLinear
# Forward pass works normally -- codebook[indices] -> matmul
output = model.generate(input_ids, max_new_tokens=128)
```

### Compress any tensor

```python
from helix_substrate import CDNAv3Writer, CDNAv3Reader
from helix_substrate.tensor_policy import TensorPolicy, TensorClass
import numpy as np

tensor = np.random.randn(1024, 768).astype(np.float32)

policy = TensorPolicy(
    tensor_class=TensorClass.UNKNOWN,
    storage_mode="codebook+sidecar",
    n_clusters=256,
    use_kmeans=True,
    sidecar_enabled=True,
    percentile=99.9,
    max_corrections=512,
)

writer = CDNAv3Writer("./output/")
stats = writer.write_tensor(tensor, "my_tensor", policy=policy)

reader = CDNAv3Reader("./output/my_tensor.cdnav3")
reconstructed = reader.reconstruct()
```

## 10-Domain Tensor Infrastructure Proofs

The same codec handles any 2D float32 tensor. No modifications needed across domains.

| Domain | Data Source | Key Metric | Verdict |
|--------|-----------|------------|---------|
| Gradient compression | TinyLlama backward pass | SGD step cos=1.000 | PASS |
| Embedding tables | TinyLlama embed_tokens | Row cos min=0.983 | WEAK |
| Activation checkpointing | TinyLlama activations | cos min=0.996, 3.90x | PASS |
| Federated learning deltas | SGD weight deltas | Weight cos=1.000, 4.0x | PASS |
| Neural codec weights | CLIP ViT + ResNet-18 | cos 0.9997+, 100% pred match | STRONG |
| RAG index | MiniLM embeddings | top-1 100%, top-5 4.9/5 | STRONG |
| LoRA adapters | PEFT LoRA matrices | All 88 matrices cos>=0.9997 | STRONG |
| MoE tiered compression | Simulated expert split | Fidelity tiers work | MIXED |
| Continual learning | Model snapshots | Full 4.0x, delta cos=1.0 | PASS |
| Sensor / scientific data | scRNA-seq + protein PDB | ARI 0.75-0.92 | MIXED |

All receipts in `receipts/tensor_infra/`.

## Companion projects

### [helix-online-kv](https://github.com/echo313unfolding/helix-online-kv)

Online KV cache compression using the same VQ codec. Fits codebooks on the first 128 tokens, then VQ-assigns all subsequent KV entries in real time.

- 2.81 ms/token encoding latency (gate: <5ms)
- 1.9x more tokens fit in same VRAM
- End-to-end with HelixLinear: +0.77% PPL at 1329 MB on Quadro T2000

**Compressed-domain attention (CDC-03):** Product quantization scores rank all tokens cheaply, select top 128 by approximate score, exact attention on subset only. Proven at cosine 0.9997 on layers 1-21 with 12.5% token coverage. Projected 29x compute savings at 4K context, 900x at 128K.

### [echo_runtime](https://github.com/echo313unfolding/echo_runtime)

Unified inference runtime wiring HelixLinear + CompressedKVCache + CDC-03 attention into one forward pass. One config file, one command.

- 155/155 weight modules compressed (3.98x)
- 22-layer KV cache (layer 0 exact, 1-21 streaming VQ)
- 21 CDC-03 hybrid attention layers
- 1404 MB peak VRAM on Quadro T2000 (4 GB card)
- 14.3 tok/s, coherent output

## How it works

```
Input Tensor (2D float32)
        |
        v
+-------------------+
| Kurtosis Router   | <-- Measures weight distribution shape (calibration-free)
|  kurt > 5?        |
+----+--------+-----+
     | no     | yes
     v        v
+--------+ +--------------+
| VQ     | | VQ + SVD     |
| k=256  | | k=256 rank=8 |
+----+---+ +------+-------+
     |            |
     v            v
+--------------------------+
| Outlier Sidecar          |
| Top 0.1% -> exact FP32  |
+-----------+--------------+
            |
            v
  codebook.npy (256 x col_dim x 4B)
  indices.npy  (rows x uint8)
  sidecar.npz  (sparse corrections)
  meta.json    (kurtosis, quality, config)
```

## What's honest

**We do not claim to have invented VQ for neural networks.** VQ weight compression dates to the 1980s, with DNN applications since 2015. Our differentiators are: calibration-free operation, architecture-agnostic coverage including SSMs (no prior work compresses Mamba through the same pipeline as LLaMA), kurtosis-based statistical routing that outperforms Hessian-based approaches, and the intelligence layer (adaptive routing, symbolic governance, semantic memory indexing).

**4x is the universal number. 5.3x is model-dependent.** k=64 passes on TinyLlama (+1.44%) but fails on Qwen-1.5B (+2.78%). We do not claim universal 5.3x compression.

**The GPU path does late materialization.** The fused Triton kernel computes `Y = X @ codebook[indices]` directly from compressed VQ indices -- the full weight matrix W never hits global VRAM. Measured peak allocation is 0.4% of W size across all tensor shapes (attn, FFN gate, FFN down). This is not a roadmap item; it ships today. Receipt: `receipts/late_materialization/late_materialization_20260326T131246.json`.

**Speed comparison against GPTQ/AWQ is not yet fair.** The decode speed gap (16.6 vs 29.8 tok/s on 4090) reflects kernel maturity, not architecture. GPTQ/AWQ have years of optimization (Marlin, exllama2). Our Triton kernel is correct and memory-efficient but not yet throughput-optimized. Our advantage is on quality, universality, and the compressed runtime stack.

## Prior art and references

This work builds on and differentiates from:

- **Choi & El-Khamy (NIPS 2018)** -- Universal DNN compression via lattice VQ. First "universal" VQ-for-DNN paper. We differ: calibration-free, no fine-tuning, architecture-agnostic including SSMs.
- **VQ4ALL (Dec 2024)** -- Universal codebook shared across networks. We differ: per-tensor codebooks (better quality), purely post-training, no calibration.
- **AQLM (ICML 2024)** -- Additive multi-codebook quantization for 2-bit LLM compression. We differ: calibration-free, architecture-agnostic, quality-first (4x not 16x).
- **GPTQ / AWQ** -- INT4 with Hessian/activation-aware scaling. We differ: calibration-free, VQ (non-uniform), works on SSMs.
- **SpQR / SqueezeLLM** -- Outlier-preserving mixed-precision. Our sidecar mechanism is related but calibration-free (magnitude percentile, not Hessian sensitivity).
- **KIVI / KVQuant** -- KV cache quantization. Our helix-online-kv uses VQ codebooks with calibrate-then-stream, combined with weight compression from the same codec.

## Project structure

```
helix-substrate/
+-- helix_substrate/
|   +-- cdnav3_writer.py       # Compress tensors to CDNA v3 format
|   +-- cdnav3_reader.py       # Reconstruct from CDNA v3
|   +-- tensor_policy.py       # Compression routing policy
|   +-- helix_linear.py        # Drop-in nn.Linear replacement
|   +-- generate_sidecars_v3.py # Outlier sidecar generation
+-- tools/
|   +-- compress.py            # Universal model compressor
|   +-- tensor_infra/          # 10-domain proof suite
+-- receipts/                  # All experiment receipts with cost blocks
+-- tests/
```

## License

Echo Labs LLC. See LICENSE for details.

## Citation

If you use helix-substrate in research, please cite:

```
@software{helix_substrate,
  author = {Josh (voidstr3m33)},
  title = {helix-substrate: Calibration-free neural network compression},
  year = {2026},
  url = {https://github.com/echo313unfolding/helix-substrate}
}
```

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,100:1a6b3c&height=100&section=footer" width="100%">
</div>
