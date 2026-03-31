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
[![Beats GPTQ](https://img.shields.io/badge/quality-beats%20GPTQ-blue)]()

**Calibration-free VQ compression. Beats GPTQ quality at 7B and AWQ at 14B.**
**No training data. No fine-tuning. Transformers, SSMs, CNNs, vision — same command.**

</div>

# helix-substrate

Calibration-free neural network compression. Beats GPTQ quality at 7B (+6.3% vs +8.2% PPL) and AWQ at 14B by 15.4%. No training data needed. Works on transformers, SSMs, CNNs, vision encoders, and embedding models without code changes.

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

model = AutoModelForCausalLM.from_pretrained("EchoLabs33/zamba2-1.2b-helix")
tokenizer = AutoTokenizer.from_pretrained("EchoLabs33/zamba2-1.2b-helix")
```

| Model | Architecture | Ratio (from BF16) | PPL Delta | Size | Link |
|-------|-------------|-------------------|-----------|------|------|
| Qwen2.5-14B | Transformer | 3.4x | pending | 8.4 GB | [HF](https://huggingface.co/EchoLabs33/qwen2.5-14b-instruct-helix) |
| Qwen2.5-7B | Transformer | 2.2x | +6.34% | 6.5 GB | [HF](https://huggingface.co/EchoLabs33/qwen2.5-7b-instruct-helix) |
| Zamba2-7B | Hybrid (Mamba2+Transformer) | 2.0x | pending | 7.5 GB | [HF](https://huggingface.co/EchoLabs33/zamba2-7b-instruct-hxq) |
| Qwen2.5-3B | Transformer | 1.6x | +0.69% | 3.8 GB | [HF](https://huggingface.co/EchoLabs33/qwen2.5-3b-instruct-helix) |
| Qwen2.5-Coder-3B | Transformer (code) | 1.6x | +1.92% | 3.8 GB | [HF](https://huggingface.co/EchoLabs33/qwen2.5-coder-3b-helix) |
| Qwen2.5-Coder-1.5B | Transformer (code) | 1.5x | +1.73% | 2.1 GB | [HF](https://huggingface.co/EchoLabs33/qwen2.5-coder-1.5b-helix) |
| Zamba2-2.7B | Hybrid (Mamba2+Transformer) | 1.8x | +6.59% | 2.8 GB | [HF](https://huggingface.co/EchoLabs33/zamba2-2.7b-instruct-helix) |
| Zamba2-1.2B | Hybrid (Mamba2+Transformer) | 1.7x | +2.90% | 1.35 GB | [HF](https://huggingface.co/EchoLabs33/zamba2-1.2b-helix) |
| TinyLlama-1.1B | Transformer | 4.0x* | +0.78% | 1.03 GB | [HF](https://huggingface.co/EchoLabs33/tinyllama-1.1b-helix) |
| Mamba2-1.3B | Pure SSM (Mamba2) | 2.1x | +8.0% | 1.4 GB | [HF](https://huggingface.co/EchoLabs33/mamba2-1.3b-helix) |
| Mamba-130M | Pure SSM | 3.8x* | +18.4% | 128 MB | [HF](https://huggingface.co/EchoLabs33/mamba-130m-helix) |

*TinyLlama and Mamba-130M ratios are from FP32 source weights. All other ratios are from BF16 source.

**Four architectures, one codec.** HelixCode (HXQ) compresses any `nn.Linear` — transformer attention, Mamba projections, hybrid layers. Same `pip install`, same API, same codebook format.

## What it does

helix-substrate compresses neural network weights using scalar k-means vector quantization. Each weight value is assigned to the nearest entry in a learned 256-entry codebook. Outlier values (top 0.1% by magnitude) are preserved exactly in a sparse sidecar. The result is a `codebook + uint8 indices + sidecar` representation at ~2x file-level compression from BF16 sources (~4x per-tensor from FP32) with negligible quality loss.

No calibration data. No fine-tuning. No architecture-specific code.

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

**Helix beats GPTQ by 23% less degradation at 7B, and beats AWQ by 15.4% at 14B. With zero calibration data.**

The remaining +6.34% PPL delta comes primarily from early down_proj layers (layers 3-4) at 0.964 cosine. These are the highest-kurtosis FFN tensors in the model. Rank-32 SVD on those specific layers is expected to push this below +4%.

**Quality vs ratio tradeoff:** GPTQ/AWQ achieve 8x compression with worse quality. helix-substrate achieves ~2x from BF16 (~4x per-tensor from FP32) with the best quality of any post-training method tested, and requires zero calibration data. VQ degrades more gracefully than INT4 at scale — the quality gap widens as model size increases.

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

**Dead ends tested for 8x (all falsified with receipts):** Group VQ k=16 (per-column codebooks, cos=0.991 vs 0.999 global), off-the-shelf ResidualVQ (Lucidrain, full-row vector quantization, cos=0.26), SVD residual correction (hurts at 7B), channel scaling/calibration (zero net benefit). Sub-vector product quantization (AQLM/VPTQ) could reach 8x but requires architecture-aware calibration, destroying the universality advantage. See `receipts/group_vq/`, `receipts/rvq_benchmark/`, `receipts/scaling_analysis/`.

## Key findings

**Outlier sidecar is non-negotiable.** Without it, k=256 VQ produces PPL 274. With it, PPL 6.18. Cosine similarity is identical (0.999) in both cases. The top 0.1% of weights by magnitude carry outsized importance despite being statistically invisible. This means cosine alone is not a safe quality metric -- outlier preservation is mandatory.

**SVD residual correction was tested and rejected.** On TinyLlama (1.1B), kurtosis-routed SVD gave marginal per-tensor cosine improvement. Crossover test at 1.5B, 3B, 7B: SVD adds zero value at 1.5B/3B and actively hurts at 7B (+4% PPL). Plain k-means VQ-256 is optimal at all scales tested. The simplest approach wins.

**Kurtosis routing beats Hessian routing on TinyLlama** but the routing target (SVD) is dead at scale. The finding stands: calibration-free signals (kurtosis from weights alone) outperform calibration-dependent signals (Hessian). But the routing itself is disabled — plain VQ for everything.

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

model = AutoModelForCausalLM.from_pretrained("EchoLabs33/zamba2-1.2b-helix")
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
Input Tensor (2D float32)
        |
        v
+-------------------+
| K-Means VQ k=256  | <-- 256 centroids, 15 iterations, no calibration
+--------+----------+
         |
         v
+--------------------------+
| Outlier Sidecar          |
| Top 0.1% -> exact FP32  |
+-----------+--------------+
            |
            v
  codebook.npy (256 x 4B)
  indices.npy  (rows x cols x uint8)
  sidecar.npz  (sparse corrections)
  meta.json    (kurtosis, quality, config)
```

## What's honest

**We do not claim to have invented VQ for neural networks.** VQ weight compression dates to the 1980s, with DNN applications since 2015. Our differentiators are: calibration-free operation, architecture-agnostic coverage including SSMs (no prior work compresses Mamba through the same pipeline as LLaMA), kurtosis-based statistical routing that outperforms Hessian-based approaches, and the intelligence layer (adaptive routing, symbolic governance, semantic memory indexing).

**~2x from BF16 is the honest file-level number. ~4x is the per-tensor codec ratio from FP32.** The codec compresses each weight tensor ~4x (FP32 → uint8 indices + codebook). But most source models are BF16, and exact tensors (norms, embeddings, biases) are stored at full precision. File-level ratios from BF16 range from 1.5x to 3.4x depending on the model's ratio of compressible to exact parameters. k=64 passes on TinyLlama (+1.44%) but fails on Qwen-1.5B (+2.78%). We do not claim universal 5.3x compression.

**The GPU path does late materialization.** The fused Triton kernel computes `Y = X @ codebook[indices]` directly from compressed VQ indices -- the full weight matrix W never hits global VRAM. Measured peak allocation is 0.4% of W size across all tensor shapes (attn, FFN gate, FFN down). This is not a roadmap item; it ships today. Receipt: `receipts/late_materialization/late_materialization_20260326T131246.json`.

**Speed comparison against GPTQ/AWQ is not yet fair.** The decode speed gap reflects kernel maturity, not architecture. GPTQ/AWQ have years of optimization (Marlin, exllama2). Our Triton kernel is correct and memory-efficient but not yet throughput-optimized. Our advantage is on quality, universality, and the compressed runtime stack.

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
