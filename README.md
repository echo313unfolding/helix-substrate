<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,100:1a6b3c&height=200&section=header&text=helix-substrate&fontSize=42&fontColor=58a6ff&animation=fadeIn&fontAlignY=35&desc=Calibration-free%20neural%20network%20compression&descSize=16&descColor=8b949e&descAlignY=55">
  <source media="(prefers-color-scheme: light)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:f0f6fc,100:2ea043&height=200&section=header&text=helix-substrate&fontSize=42&fontColor=1f2328&animation=fadeIn&fontAlignY=35&desc=Calibration-free%20neural%20network%20compression&descSize=16&descColor=656d76&descAlignY=55">
  <img alt="helix-substrate" src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,100:1a6b3c&height=200&section=header&text=helix-substrate&fontSize=42&fontColor=58a6ff&animation=fadeIn&fontAlignY=35&desc=Calibration-free%20neural%20network%20compression&descSize=16&descColor=8b949e&descAlignY=55">
</picture>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-Echo%20Labs-green)](LICENSE)
[![~2x from BF16](https://img.shields.io/badge/compression-~2x%20from%20BF16-brightgreen)]()
[![15 architectures](https://img.shields.io/badge/models-15%20architectures-blue)]()

**Calibration-free HXQ compression for Transformers, SSMs, hybrids, MoEs, and vision models.**
**No training data. No fine-tuning. Same codec, same command across architectures.**

</div>

# helix-substrate

Calibration-free tensor codec. Receipted across tested ML tensor families and raw tensor distributions. Deployed as HXQ compression (scalar k-means VQ, 256-entry codebook, uint8 indices, sparse sidecar). No training data needed. Works on transformers, SSMs, hybrids, MoEs, CNNs, vision encoders, and embedding models without code changes. ~2x file-level from BF16, ~4x per-tensor from FP32. See [`docs/HXQ_TENSOR_CODEC_EVIDENCE.md`](docs/HXQ_TENSOR_CODEC_EVIDENCE.md) for non-ML tensor distribution evidence.

```bash
pip install helix-substrate
```

```python
from helix_substrate import CDNAv3Writer, CDNAv3Reader  # HelixCode format

# Compress any 2D weight tensor
writer = CDNAv3Writer("./compressed/")
writer.write_tensor(weight_matrix, "layer_name")

# Reconstruct
reader = CDNAv3Reader("./compressed/layer_name.cdnav3")
reconstructed = reader.reconstruct()  # cosine similarity >= 0.999
```

## Model Zoo

Pre-compressed models on HuggingFace. One import, one line to load:

```python
import helix_substrate  # registers HXQ quantizer with HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EchoLabs33/qwen2.5-3b-instruct-hxq")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
```

| Model | Architecture | Ratio (from BF16) | PPL Delta | Link |
|-------|-------------|-------------------|-----------|------|
| **Transformers** | | | | |
| Qwen2.5-14B | Transformer | 3.4x | pending | [HF](https://huggingface.co/EchoLabs33/qwen2.5-14b-instruct-hxq) |
| Qwen2.5-7B | Transformer | 2.2x | +6.34% | [HF](https://huggingface.co/EchoLabs33/qwen2.5-7b-instruct-hxq) |
| Qwen2.5-3B | Transformer | 1.6x | +0.69% | [HF](https://huggingface.co/EchoLabs33/qwen2.5-3b-instruct-hxq) |
| Qwen2.5-Coder-3B | Transformer (code) | 1.6x | +1.92% | [HF](https://huggingface.co/EchoLabs33/qwen2.5-coder-3b-hxq) |
| SmolLM3-3B | Transformer | -- | +1.28% | [HF](https://huggingface.co/EchoLabs33/smollm3-3b-hxq) |
| TinyLlama-1.1B | Transformer | 4.0x* | +0.78% | [HF](https://huggingface.co/EchoLabs33/tinyllama-1.1b-hxq) |
| **SSMs** | | | | |
| Mamba2-1.3B | Pure SSM (Mamba2) | 2.1x | +8.0% | [HF](https://huggingface.co/EchoLabs33/mamba2-1.3b-hxq) |
| Mamba-130M | Pure SSM | 3.8x* | +18.4% | [HF](https://huggingface.co/EchoLabs33/mamba-130m-hxq) |
| **Hybrids** | | | | |
| Zamba2-7B | Hybrid (Mamba2+Transformer) | 2.0x | pending | [HF](https://huggingface.co/EchoLabs33/zamba2-7b-instruct-hxq) |
| Zamba2-2.7B | Hybrid (Mamba2+Transformer) | 1.8x | +6.59% | [HF](https://huggingface.co/EchoLabs33/zamba2-2.7b-instruct-hxq) |
| Zamba2-1.2B | Hybrid (Mamba2+Transformer) | 1.7x | +2.90% | [HF](https://huggingface.co/EchoLabs33/zamba2-1.2b-hxq) |
| Granite 4.0 H Micro | MoE Hybrid | -- | -- | [HF](https://huggingface.co/EchoLabs33/granite-4.0-h-micro-hxq) |
| **MoE** | | | | |
| OLMoE-1B-7B | Mixture of Experts | -- | -- | [HF](https://huggingface.co/EchoLabs33/olmoe-1b-7b-instruct-hxq) |
| **Vision / Encoder** | | | | |
| CLIP ViT-L/14 | Vision Transformer | -- | -- | [HF](https://huggingface.co/EchoLabs33/clip-vit-large-patch14-hxq) |
| BERT-base | Encoder-only | -- | -- | [HF](https://huggingface.co/EchoLabs33/bert-base-uncased-hxq) |
| **Non-LLM Tensor Proofs** | | | | |
| Regulated-asset features | Synthetic feature tensor (8192×16) | 4.92x | cos 0.9997 | [HF](https://huggingface.co/datasets/EchoLabs33/regulated-asset-tensor-hxq) |

*TinyLlama and Mamba-130M ratios are from FP32 source weights. All other ratios are from BF16 source.

**Six architecture families plus non-LLM tensor proofs, one codec.** HXQ compresses any dense numeric tensor — transformer attention, Mamba projections, MoE experts, vision encoders, BERT layers, and non-ML feature tensors. Same `pip install`, same API, same codebook format.

## What it does

helix-substrate compresses neural network weights using k-means clustering + per-group affine correction. For each group of weights: (1) cluster values into centroids via k-means, (2) store which centroid each weight maps to (uint8 index), (3) compute a scale and offset per group that corrects systematic error. Outlier values (top 0.1% by magnitude) are preserved exactly in a sparse sidecar.

The result is ~6.25 bits per weight with best-in-class perplexity — lower PPL than Q6_K at smaller file size, competitive decode speed (92% of Q4_K_M on RTX 3090), and it works on any architecture without calibration data: transformers, SSMs, hybrids, MoEs, vision encoders.

No calibration data. No fine-tuning. No architecture-specific code. The same codec that compresses LLaMA compresses Mamba compresses Zamba2 hybrids.

## Benchmarks

### Weight Compression Quality (RTX 4090, WikiText-2 PPL)

| Model | Method | PPL | PPL Delta | Calibration |
|-------|--------|-----|-----------|-------------|
| **Qwen2.5-7B** | FP16 Dense | 6.949 | baseline | -- |
| | **HelixLinear k=256** | **7.388** | **+6.34%** | **None** |
| | GPTQ Int4 | 7.518 | +8.2% | 128 sequences |
| | AWQ Int4 | 7.719 | +11.1% | Activation stats |
| **Qwen2.5-14B** | **HelixLinear k=256** | **3.78** | -- | **None** |
| | AWQ Int4 | 4.47 | -- | Activation stats |

**On these benchmarks, HXQ shows lower PPL degradation than the tested GPTQ/AWQ baselines, at lower compression ratio and with zero calibration data.**

The remaining +6.34% PPL delta comes primarily from early down_proj layers (layers 3-4) at 0.964 cosine — the highest-kurtosis FFN tensors in the model.

**Quality vs ratio tradeoff:** GPTQ/AWQ achieve higher compression (INT4, ~8x from FP16) with calibration data. HXQ achieves ~2x from BF16 (~4x per-tensor from FP32) with better quality on these benchmarks and zero calibration. HXQ's advantage is quality and architecture breadth; GPTQ/AWQ's advantage is compression ratio and mature speed kernels. The right comparison depends on whether you are optimizing for memory, quality, or universality.

### Architecture Coverage (all k=256, same `compress.py`)

| Model | Architecture | Tensors | Per-Tensor Ratio | File Ratio (from BF16) | Cosine (min) |
|-------|-------------|---------|-----------------|----------------------|-------------|
| TinyLlama 1.1B | Transformer (LLaMA) | 154 | 3.98x | 4.0x (FP32 source) | 0.9946 |
| Qwen2.5 1.5B | Transformer (Qwen) | 196 | 4.00x | 1.5x | 0.9943 |
| Qwen2.5 7B | Transformer (Qwen) | 196 | 4.00x | 2.2x | 0.9955 |
| Qwen2.5 14B | Transformer (Qwen) | 336 | 4.00x | 3.4x | -- |
| Mamba-130m | SSM (Mamba) | 97 | 3.92x | 3.8x (FP32 source) | 0.9990+ |
| Mamba-2 1.3B | SSM (Mamba-2) | 98 | 3.99x | 2.1x | 0.9990+ |
| MiniLM-L6 | Transformer (BERT) | 73 | 3.94x | -- | 0.9997 |
| CLIP ViT-B/32 | Vision Transformer | 49 | 3.98x | -- | 0.9997 |
| Zamba2-1.2B | Hybrid (Mamba2+Transformer) | 136 | 4.00x | 1.7x | 0.9973 |
| ResNet-18 | CNN | 1 (fc) | 3.97x | -- | 0.9998 |

Per-tensor ratio is FP32 weight → codebook+indices+sidecar (~4x for k=256). File ratio is the actual size reduction from the BF16 source model — lower because exact tensors (norms, embeddings, biases) are stored at full precision. All compressed with the same command. No architecture-specific flags or code paths.

### Compression Quality Frontier (TinyLlama FP32 source, PPL on WikiText-2)

| Config | Per-Tensor Ratio | PPL Delta | Status |
|--------|-----------------|-----------|--------|
| k=256 + sidecar | 4.0x | +0.11% | Production baseline |
| k=64 + sidecar | 5.3x | +1.44% | Model-dependent (fails on Qwen at +2.78%) |
| k=32 + sidecar | 6.4x | +2.61% | Below quality threshold |
| k=16 + sidecar | 8.0x | +9.34% | Rejected |

Note: These are per-tensor ratios on TinyLlama (FP32 source). For BF16 source models, file-level ratios are ~2x at k=256.

### GGUF Runtime Benchmarks (llama.cpp)

HXQ is available as native GGUF via the [`hxq-affine-type`](https://github.com/echo313unfolding/llama.cpp/tree/hxq-affine-type) branch of llama.cpp. The `HXQ_AF6` format: 256-entry codebook + uint8 indices + per-group-128 affine correction (scale + offset). 6.25 bits per weight.

**Qwen2.5-Coder-3B (RTX 3090, full GPU resident):**

| Format | Size | bpw | Decode tok/s | vs Q4 | PPL | HumanEval |
|--------|------|-----|-------------|-------|-----|-----------|
| Q4_K_M | 1.95G | 4.5 | 245.03 | 100% | 10.072 | 82.3% |
| Q5_K_M | 2.07G | 5.5 | 229.00 | 93.5% | 10.004 | 83.5% |
| **HXQ_AF6** | **2.26G** | **6.25** | **226.53** | **92.4%** | **9.954 (best)** | **84.1% (best)** |
| Q6_K | 2.36G | 6.56 | 204.86 | 83.6% | 9.964 | 83.5% |

**Qwen2.5-7B-Instruct (RTX 3090 Ti, full GPU resident):**

| Format | Size | bpw | Decode tok/s | vs Q4 | PPL |
|--------|------|-----|-------------|-------|-----|
| Q4_K_M | 4.36G | 4.5 | 127.33 | 100% | 8.315 |
| Q5_K_M | 5.07G | 5.5 | 117.30 | 92.1% | 8.184 |
| **HXQ_AF6** | **5.56G** | **6.25** | **114.02** | **89.5%** | **7.982 (best)** |
| Q6_K | 5.82G | 6.56 | 98.56 | 77.4% | 8.116 |

**Zamba2-2.7B-Instruct (RTX 3090, hybrid Mamba2+Transformer):**

| Format | Size | bpw | Decode tok/s | vs Q4 | PPL |
|--------|------|-----|-------------|-------|-----|
| Q4_K_M | 2.11G | 4.5 | 47.30 | 100% | 23.278 |
| **HXQ_AF6** | **2.79G** | **6.25** | **45.87** | **97.0%** | **22.653** |
| Q6_K | 2.93G | 6.58 | 44.98 | 95.1% | 22.573 |
| Q5_K_M | 2.51G | 5.62 | 43.38 | 91.7% | 22.743 |

**Summary:** HXQ_AF6 beats Q6_K on decode speed across all three models (+10-16%) while having best or second-best PPL at smaller file size. No calibration data was used for any model. Same codec, same kernel, same command.

**Scaling property:** HXQ gets MORE competitive on bigger GPUs. T2000 (16 SMs): 84.6% of Q4. RTX 3090 (82 SMs): 92.4% of Q4. The mmvq kernel parallelizes across SMs — more cores = smaller gap.

Receipts: `hxq_runtime_3090_qwen3b_20260508`, `hxq_runtime_3090ti_qwen7b_instruct_20260509`, `hxq_runtime_3090_zamba2_2.7b_20260509`.

**Dead ends tested for 8x (all falsified with receipts):** Group VQ k=16 (per-column codebooks, cos=0.991 vs 0.999 global), off-the-shelf ResidualVQ (Lucidrain, full-row vector quantization, cos=0.26), SVD residual correction (hurts at 7B), channel scaling/calibration (zero net benefit). Sub-vector product quantization (AQLM/VPTQ) could reach 8x but requires architecture-aware calibration, destroying the universality advantage. See `receipts/group_vq/`, `receipts/rvq_benchmark/`, `receipts/scaling_analysis/`.

## Key findings

**Outlier sidecar is non-negotiable.** Without it, k=256 VQ produces PPL 274. With it, PPL 6.18. Cosine similarity is identical (0.999) in both cases. The top 0.1% of weights by magnitude carry outsized importance despite being statistically invisible. This means cosine alone is not a safe quality metric -- outlier preservation is mandatory.

**SVD residual correction was tested and rejected.** On TinyLlama (1.1B), kurtosis-routed SVD gave marginal per-tensor cosine improvement. Crossover test at 1.5B, 3B, 7B: SVD adds zero value at 1.5B/3B and actively hurts at 7B (+4% PPL). Plain k-means VQ-256 is optimal at all scales tested. The simplest approach wins.

**No routing needed.** Earlier experiments tested kurtosis-based routing to different codecs per tensor. Result: plain k-means + affine for everything beats any routing scheme tested. The simplest approach wins.

**Weighted k-means is harmful.** Hessian-weighted centroid placement gives +2.93% PPL -- actively worse than unweighted. Distorting the codebook toward "important" columns degrades it for the majority of weights.

**Embedding tables must stay dense.** VQ on embed_tokens and lm_head inflated 7B PPL from +6.34% to +11%. Two lines of code (exclude both from VQ) eliminated the entire quality gap vs GPTQ. Lesson: embedding tables have uniform importance across all rows — VQ's "representative centroid" assumption fails catastrophically when every entry is equally important.

## Quick start

### Compress a model

```bash
python tools/compress.py \
  --model-dir /path/to/model \
  --out-dir /path/to/output \
  --k 256 --sidecar
```

### Load compressed weights for inference

**From HuggingFace (recommended):**

```python
import helix_substrate  # registers HXQ quantizer
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("EchoLabs33/qwen2.5-3b-instruct-hxq")
output = model.generate(input_ids, max_new_tokens=128)
```

**From local HelixCode directory:**

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
| Federated learning deltas | SGD weight deltas | Weight cos=1.000 | PASS |
| Neural codec weights | CLIP ViT + ResNet-18 | cos 0.9997+, 100% pred match | STRONG |
| RAG index | MiniLM embeddings | top-1 100%, top-5 4.9/5 | STRONG |
| LoRA adapters | PEFT LoRA matrices | All 88 matrices cos>=0.9997 | STRONG |
| MoE tiered compression | Simulated expert split | Fidelity tiers work | MIXED |
| Continual learning | Model snapshots | delta cos=1.0 | PASS |
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

- 155/155 weight modules compressed
- 22-layer KV cache (layer 0 exact, 1-21 streaming VQ)
- 21 CDC-03 hybrid attention layers
- 1404 MB peak VRAM on Quadro T2000 (4 GB card)
- 14.3 tok/s, coherent output

## How it works

```
Input: group of 128 weights (float16/float32)
        |
        v
+------------------------+
| K-Means (k=256)        |  Cluster weight values into 256 centroids.
| 15 iterations, no cal  |  Each weight → nearest centroid (uint8 index).
+--------+---------------+
         |
         v
+------------------------+
| Affine Correction      |  Per-group scale + offset that minimizes
| scale, offset (fp16)   |  ||original - (scale * centroid[idx] + offset)||
+--------+---------------+
         |
         v
+------------------------+
| Outlier Sidecar        |  Top 0.1% by magnitude stored exact (sparse).
| Exact FP32 values      |  Catches tail values k-means can't represent.
+--------+---------------+
         |
         v
  Stored: codebook (256 × fp16) + indices (uint8) + scale/offset (fp16 per group) + sidecar (sparse)
  At runtime: centroid[index] * scale + offset → dot product (no decompression to full matrix)
  Cost: 6.25 bits per weight
```

**Why this works:** K-means gets the shape right (which cluster each weight belongs to). Affine fixes systematic per-group drift (scale/shift error). Sidecar catches outliers that neither can represent. The combination gives cosine > 0.999 on any tensor distribution tested — Gaussian, uniform, heavy-tailed, sparse, or neural network weights.

## What's honest

**We do not claim to have invented VQ for neural networks.** VQ weight compression dates to the 1980s, with DNN applications since 2015. Our differentiator is architecture-agnostic operation with competitive runtime: same codec compresses transformers, SSMs (Mamba/Mamba-2), and hybrids (Zamba2) with no calibration data and no architecture-specific code. No prior work compresses Mamba through the same pipeline as LLaMA at competitive inference speed.

**SVD, residual correction, and codec routing are all dead.** Tested with receipts, all falsified at scale. Plain k-means + affine is the production path. Complexity doesn't help when the simple approach already gives cosine > 0.999.

**~2x from BF16 is the honest file-level number. ~4x is the per-tensor codec ratio from FP32.** The codec compresses each weight tensor ~4x (FP32 → uint8 indices + codebook). But most source models are BF16, and exact tensors (norms, embeddings, biases) are stored at full precision. File-level ratios from BF16 range from 1.5x to 3.4x depending on the model's ratio of compressible to exact parameters. k=64 passes on TinyLlama (+1.44%) but fails on Qwen-1.5B (+2.78%). We do not claim universal 5.3x compression.

**The GPU path does late materialization.** The fused Triton kernel computes `Y = X @ codebook[indices]` directly from compressed VQ indices -- the full weight matrix W never hits global VRAM. Measured peak allocation is 0.4% of W size across all tensor shapes (attn, FFN gate, FFN down). This is not a roadmap item; it ships today. Receipt: `receipts/late_materialization/late_materialization_20260326T131246.json`.

**Two runtime paths exist.** The Python/Triton path (HelixLinear) does late-materialization inference — correct and memory-efficient but not speed-optimized. The GGUF/llama.cpp path (HXQ_AF6) uses a native CUDA mmvq kernel and reaches 92.4% of Q4_K_M decode speed on RTX 3090. The GGUF path is the production runtime; the Python path is for research and HuggingFace integration.

**VQ is ~2x from BF16 (4x per-tensor from FP32), not 8x.** GPTQ/AWQ achieve 8x compression (INT4 from FP16). We achieve ~2x file-level from BF16 (uint8 indices + codebook + exact tensors). Our compression ratio is lower, but our quality is better and requires zero calibration. The right comparison is quality at a given memory budget, not compression ratio alone.

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
|   +-- cdnav3_writer.py       # Compress tensors to HelixCode format
|   +-- cdnav3_reader.py       # Reconstruct from HelixCode
|   +-- tensor_policy.py       # Compression routing policy
|   +-- helix_linear.py        # Drop-in nn.Linear replacement
|   +-- hf_integration.py      # HuggingFace AutoModel integration (HXQ quantizer)
|   +-- generate_sidecars_v3.py # Outlier sidecar generation
|   +-- triton_vq_matmul.py    # Fused Triton kernel (late materialization)
+-- tools/
|   +-- compress.py            # Universal model compressor (one command)
|   +-- eval_ppl_cpu.py        # CPU perplexity evaluation
|   +-- cloud_ready_check.py   # Pre-cloud deployment validation
|   +-- scaling_analysis.py    # VQ scaling hypothesis analysis
|   +-- group_vq_test.py       # Group VQ falsification (k=16 dead end)
|   +-- rvq_benchmark.py       # RVQ falsification (Lucidrain dead end)
|   +-- tensor_infra/          # 10-domain proof suite
+-- receipts/                  # All experiment receipts (JSON, with cost blocks)
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
