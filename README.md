# helix-substrate

Calibration-free weight compression for neural networks. Scalar k-means VQ with sparse outlier sidecar. No training data, no fine-tuning, no architecture-specific code.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

```bash
pip install helix-substrate
```

## What this is

**HXQ-VQ** compresses dense weight tensors using:

1. **K-means clustering** (k=256 centroids, 15 iterations, no calibration data)
2. **uint8 index assignment** (each weight maps to its nearest centroid)
3. **Sparse outlier sidecar** (top 0.1% by magnitude stored exactly)

Per-tensor compression ratio: ~4x from FP32 (256-entry codebook + uint8 indices + sidecar).
File-level ratio from BF16 source: ~1.5x to 3.4x (exact tensors — norms, embeddings, biases — are stored at full precision).

This codec is architecture-agnostic: the same code compresses transformers, SSMs (Mamba/Mamba-2), hybrids (Zamba2), MoEs, vision encoders, and BERT-family models.

## Quick start

### Compress and reconstruct a tensor

```python
from helix_substrate import CDNAv3Writer, CDNAv3Reader
import numpy as np

tensor = np.random.randn(1024, 768).astype(np.float32)

writer = CDNAv3Writer("./compressed/")
stats = writer.write_tensor(tensor, "my_layer")

reader = CDNAv3Reader("./compressed/my_layer.cdnav3")
reconstructed = reader.reconstruct()
# Typical cosine similarity >= 0.999
```

### Compress a HuggingFace model

```bash
python tools/compress.py \
  --model-dir /path/to/model \
  --out-dir /path/to/output \
  --k 256 --sidecar
```

### Load compressed weights

```python
import helix_substrate  # registers HXQ quantizer with HuggingFace
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("EchoLabs33/qwen2.5-3b-instruct-hxq")
```

Or from a local compressed directory:

```python
from transformers import AutoModelForCausalLM
from helix_substrate.helix_linear import swap_to_helix

model = AutoModelForCausalLM.from_pretrained("path/to/model")
swap_to_helix(model, "path/to/cdnav3/")
# All nn.Linear modules replaced with HelixLinear (codebook[indices] matmul)
```

## Model zoo

Pre-compressed models on HuggingFace, all produced by the same `compress.py` command:

| Model | Architecture | File Ratio (from BF16) | PPL Delta | Link |
|-------|-------------|----------------------|-----------|------|
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

*TinyLlama and Mamba-130M ratios are from FP32 source weights. All other ratios are from BF16 source.

## HXQ-VQ vs HXQ-Affine

This repository implements **HXQ-VQ**: scalar k-means VQ with sparse sidecar. There is a separate, distinct format called **HXQ-Affine** (HXQ_AF6) implemented in a [llama.cpp fork](https://github.com/echo313unfolding/llama.cpp/tree/hxq-affine-type). They are not the same codec.

| | HXQ-VQ (this repo) | HXQ-Affine (llama.cpp fork) |
|--|---------------------|----------------------------|
| **Codec** | k-means VQ (256 centroids) + sparse sidecar | 256-entry codebook + uint8 indices + per-group-128 affine scale+offset |
| **Bits per weight** | ~8 bpw (uint8 indices + codebook overhead + sidecar) | 6.25 bpw |
| **Runtime** | Python/Triton (HelixLinear, late materialization) | Native CUDA mmvq kernel via llama.cpp |
| **Use case** | Research, HuggingFace integration, model zoo | Production inference via GGUF |
| **Interop tested** | No shared golden-vector test exists yet | No shared golden-vector test exists yet |

Benchmarks and speed numbers for HXQ_AF6 (decode tok/s, PPL comparisons against Q4_K_M/Q6_K) belong to the llama.cpp fork, not to this repository. See the [hxq-affine-type branch](https://github.com/echo313unfolding/llama.cpp/tree/hxq-affine-type) for those results.

## Benchmarks (HXQ-VQ, this repo)

### Weight compression quality (WikiText-2 PPL, CPU evaluation)

| Model | BF16 PPL | HXQ-VQ PPL | PPL Delta | Calibration |
|-------|----------|-----------|-----------|-------------|
| Qwen2.5-7B | 6.949 | 7.388 | +6.34% | None |
| Qwen2.5-3B | -- | -- | +0.69% | None |
| TinyLlama-1.1B | -- | -- | +0.78% | None |

PPL measurements from `tools/eval_ppl_cpu.py` against WikiText-2 test split.

### Per-tensor codec quality (k=256, same compress.py)

| Model | Architecture | Tensors | Per-Tensor Ratio | Cosine (min) |
|-------|-------------|---------|-----------------|-------------|
| TinyLlama 1.1B | Transformer (LLaMA) | 154 | 3.98x | 0.9946 |
| Qwen2.5 7B | Transformer (Qwen) | 196 | 4.00x | 0.9955 |
| Mamba-130m | SSM (Mamba) | 97 | 3.92x | 0.9990+ |
| Mamba-2 1.3B | SSM (Mamba-2) | 98 | 3.99x | 0.9990+ |
| Zamba2-1.2B | Hybrid | 136 | 4.00x | 0.9973 |
| MiniLM-L6 | BERT | 73 | 3.94x | 0.9997 |
| CLIP ViT-B/32 | Vision Transformer | 49 | 3.98x | 0.9997 |
| ResNet-18 | CNN | 1 (fc) | 3.97x | 0.9998 |

Per-tensor ratio is FP32 weight to codebook+indices+sidecar. All compressed with the same command. No architecture-specific flags.

### Compression quality frontier (TinyLlama, FP32 source)

| Config | Per-Tensor Ratio | PPL Delta |
|--------|-----------------|-----------|
| k=256 + sidecar | 4.0x | +0.11% |
| k=64 + sidecar | 5.3x | +1.44% (fails on Qwen at +2.78%) |
| k=32 + sidecar | 6.4x | +2.61% |
| k=16 + sidecar | 8.0x | +9.34% (rejected) |

## Key findings

**Outlier sidecar is non-negotiable.** Without it, k=256 VQ produces PPL 274. With it, PPL 6.18. Cosine similarity is identical (0.999) in both cases. The top 0.1% of weights by magnitude carry outsized importance despite being statistically invisible. Cosine alone is not a safe quality metric.

**Embedding tables must stay dense.** VQ on embed_tokens and lm_head inflated 7B PPL from +6.34% to +11%. Two lines of code (exclude both from VQ) eliminated the entire quality gap.

**SVD residual correction rejected.** Tested at 1.1B, 1.5B, 3B, 7B. Zero value at 1.5B/3B, actively hurts at 7B (+4% PPL). Plain k-means VQ-256 is optimal at all scales tested.

**Weighted k-means is harmful.** Hessian-weighted centroid placement gives +2.93% PPL vs unweighted.

**No routing needed.** Kurtosis-based routing to different codecs per tensor was tested and beaten by plain k-means everywhere.

Receipts for all negative results in `receipts/`.

## How it works

```
Input: weight tensor (float32 or float16)
        |
        v
+------------------------+
| K-Means (k=256)        |  Cluster all weight values into 256 centroids.
| 15 iterations, no cal  |  Each weight -> nearest centroid (uint8 index).
+--------+---------------+
         |
         v
+------------------------+
| Outlier Sidecar        |  Top 0.1% by magnitude stored exactly (sparse).
| Exact FP32 values      |  Catches tail values k-means can't represent.
+--------+---------------+
         |
         v
  Stored: codebook (256 x fp32) + indices (uint8) + sidecar (sparse)
  At runtime: codebook[indices] -> matmul (HelixLinear, no full W in memory)
```

## Prior art

VQ for neural network compression dates to the 1980s, with DNN applications since 2015. This work does not claim to have invented the technique. The differentiator is calibration-free, architecture-agnostic operation across transformers, SSMs, and hybrids using the same codec and command. Related work:

- **Choi & El-Khamy (NIPS 2018)** -- Universal DNN compression via lattice VQ. Requires calibration.
- **VQ4ALL (Dec 2024)** -- Universal codebook shared across networks. We use per-tensor codebooks.
- **AQLM (ICML 2024)** -- Additive multi-codebook quantization. Requires calibration, achieves higher compression.
- **GPTQ / AWQ** -- INT4 with Hessian/activation-aware scaling. Higher compression ratio (~8x from FP16), requires calibration data.
- **SpQR / SqueezeLLM** -- Outlier-preserving mixed-precision. Our sidecar mechanism is related but calibration-free.

**Tradeoff:** GPTQ/AWQ achieve higher compression (INT4, ~8x) with calibration. HXQ-VQ achieves ~4x per-tensor with zero calibration. The right comparison depends on whether you optimize for ratio, quality, or universality.

## Companion projects

- [helix-online-kv](https://github.com/echo313unfolding/helix-online-kv) -- Online KV cache compression using the same VQ codec
- [helix-codec](https://github.com/echo313unfolding/helix-codec) -- C implementation of the HXQ codec
- [hxq-native](https://github.com/echo313unfolding/hxq-native) -- Native C/CUDA HXQ library
- [llama.cpp fork (hxq-affine-type)](https://github.com/echo313unfolding/llama.cpp/tree/hxq-affine-type) -- HXQ-Affine GGUF runtime (separate codec, separate benchmarks)

## Project structure

```
helix-substrate/
+-- helix_substrate/
|   +-- hxq_writer_v3.py        # Compress tensors (CDNAv3Writer)
|   +-- hxq_reader_v3.py        # Reconstruct from compressed format (CDNAv3Reader)
|   +-- hxq_encoder.py          # Low-level k-means encoder
|   +-- hxq_reader.py           # CDNAv2 block-indexed reader
|   +-- tensor_policy.py        # Compression policy per tensor class
|   +-- helix_linear.py         # Drop-in nn.Linear replacement (HelixLinear)
|   +-- hf_quantizer.py         # HuggingFace AutoModel integration
|   +-- generate_sidecars_v3.py # Outlier sidecar generation
|   +-- triton_vq_matmul.py     # Fused Triton kernel (late materialization)
|   +-- se.py                   # Structural entropy computation
|   +-- receipt.py              # Tamper-evident operation receipts
+-- tools/
|   +-- compress.py             # Universal model compressor
|   +-- eval_ppl_cpu.py         # CPU perplexity evaluation
+-- receipts/                   # All experiment receipts (JSON, with cost blocks)
+-- tests/                      # 1200+ tests
```

## License

MIT. See [LICENSE](LICENSE) for details.

## Citation

```
@software{helix_substrate,
  author = {McElroy, Josh},
  title = {helix-substrate: Calibration-free neural network weight compression},
  year = {2026},
  url = {https://github.com/echo313unfolding/helix-substrate}
}
```
