# helix-substrate

Model weight compression and streaming decode library. Compress neural network weights into a compact format (CDNA), then run matrix operations directly from the compressed representation — without ever loading the full weight matrix into memory.

## What it does

1. **CDNA Format** — Quantize model weights into a 256-entry codebook + uint8 indices, with per-block brotli compression and SHA256 verification.

2. **Streaming Block Decode** — Compute `Y = X @ W` where W is stored in CDNA format. W is never fully loaded. Instead, blocks of rows are decompressed one at a time, multiplied against the corresponding slice of X, and accumulated.

3. **Structural Entropy (Se) Routing** — Measure tensor complexity via `Se = H × U × D` (entropy × unstructuredness × rank depth). The Se score maps to a compute routing decision: simple tensors → CPU, parallel tensors → GPU, complex unstructured tensors → QPU.

4. **Receipts** — Every operation produces a tamper-evident receipt with SHA256 input/output hashes, timing, memory usage, and fidelity metrics. If you can't verify it, it didn't happen.

## Benchmarks

Measured peak memory for `Y = X @ W` with streaming vs loading full W:

| Matrix Size | Block Rows | Standard | Streaming | Ratio |
|-------------|------------|----------|-----------|-------|
| 64 MB       | 64         | 64 MB    | 18 MB     | **3.5x** |
| 256 MB      | 64         | 256 MB   | 68 MB     | **3.8x** |
| 1 GB        | 64         | 1024 MB  | 137 MB    | **7.5x** |
| 1 GB        | 32         | 1024 MB  | 69 MB     | **14.9x** |

Streaming overhead is ~constant (~68 MB). Ratio improves with matrix size. At LLM scale (1GB+ weight matrices), expect **7-15x memory reduction** depending on block size.

Correctness: cosine similarity = 1.000000 (exact match to full-matrix computation).

Run the benchmark yourself:
```bash
python tools/bench_memory.py --rows 16384 --cols 16384 --block-rows 32
```

## Install

```bash
pip install helix-substrate
```

For compression support:
```bash
pip install helix-substrate[brotli]
```

Required: `numpy>=1.24`. Optional: `brotli` (for CDNAv2 block compression), `zstandard` (alternative codec).

## Quick start

### Compress a tensor

```python
import numpy as np
from helix_substrate import encode_tensor_to_cdna, decode_cdna_to_tensor

# Compress
W = np.random.randn(4096, 4096).astype(np.float32)
encode_tensor_to_cdna(W, "weight.cdna", tensor_name="my_layer")

# Decompress
W_decoded = decode_cdna_to_tensor("weight.cdna")
print(f"Cosine similarity: {np.dot(W.flat, W_decoded.flat) / (np.linalg.norm(W) * np.linalg.norm(W_decoded)):.6f}")
```

### Measure tensor complexity

```python
from helix_substrate import compute_tensor_se

result = compute_tensor_se(W)
print(f"Se={result['Se']:.3f} → route to {result['routing_hint']}")
# Se=0.42 → route to gpu
```

### Streaming decode (CDNAv2)

```python
from helix_substrate.stream_matmul import stream_xw_from_cdna

# Y = X @ W, but W is never fully loaded
X = np.random.randn(1, 256, 4096).astype(np.float32)
Y, receipt = stream_xw_from_cdna(X, "weight.cdna2.hxz")
print(f"Memory savings: {receipt.savings_factor:.1f}x")
```

## Package structure

```
helix_substrate/
├── __init__.py          # Public API
├── cdna_encoder.py      # CDNA v1 encode/decode (k-means quantization)
├── cdna_reader.py       # CDNA v2 reader (block-indexed, brotli, SHA256)
├── sidecar.py           # HXZO outlier sidecar (high-precision corrections)
├── stream_matmul.py     # Core: Y = X @ W from CDNA (streaming, never loads W)
├── stream_attention.py  # Full attention layer (Q,K,V,O all streamed)
├── stream_ffn.py        # Full FFN layer (gate,up,down all streamed)
├── stream_block.py      # Full transformer block (attention + FFN + norms)
├── rope.py              # Rotary Position Embeddings
├── se.py                # Structural Entropy estimator and routing
└── receipt.py           # Tamper-evident execution receipts
```

## The Se formula

Structural Entropy decomposes tensor complexity into three independent factors:

| Component | Measures | High means |
|-----------|----------|------------|
| **H** (entropy) | Singular value spread | Energy spread across many directions |
| **U** (unstructuredness) | Neighbor coherence | No spatial correlation between adjacent rows |
| **D** (depth) | Effective rank ratio | Many dimensions matter |

`Se = H × U × D` produces a 0-1 score. The 2D routing policy uses `(Se, C_struct)` jointly:

- **Zone 1**: Se < 0.30, structured → CPU
- **Zone 2**: 0.30 ≤ Se < 0.70 → GPU
- **Zone 3**: Se ≥ 0.70, unstructured → QPU
- **Zone 4**: Se ≥ 0.70, structured → GPU

## Inspiration

The mathematical patterns in this library draw from nature — Fibonacci sequences in the block structure, golden-ratio-inspired codebook initialization, and structural entropy as a measure of order vs chaos in weight matrices. The thesis: nature already solved the compression math we need, because it's the world's largest dataset.

## License

MIT
