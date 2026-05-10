# HXQ Meta-Codec

HXQ is a meta-codec: it selects, routes, and emits the correct runtime
representation per tensor. It does not replace proven formats — it adopts them.

## Architecture

```
Input (FP16/BF16 safetensors)
  │
  ├─ Router (per-tensor policy)
  │   │
  │   ├─ speed  → emit Q4_K_M (GGML_TYPE_Q4_K, type 12)
  │   ├─ middle → emit Q5_K   (GGML_TYPE_Q5_K, type 13)  ← HXQ_Q5_H encoder
  │   └─ quality → emit AF6   (GGML_TYPE_HXQ_AF6)         ← HXQ custom
  │
  ├─ Encoder (per-block quantization)
  │   │
  │   ├─ Q5_K:  256-block, 8×g32, 6-bit scales, re-quantize
  │   └─ AF6:   128-block, flat, 64 levels, FP16 scale/min
  │
  ├─ Validator (per-tensor cosine gate ≥ 0.998)
  │
  └─ Receipt (tier, bpw, hash, cosine, cost)
```

## What HXQ contributes

HXQ does NOT:
- Replace Q4_K_M at speed tier (use upstream llama.cpp quantize)
- Replace Q5_K at middle tier (emit standard Q5_K blocks, use native runtime)
- Invent new GGUF types where existing ones work

HXQ DOES:
- Route tensors to optimal tier based on role/sensitivity
- Emit byte-compatible Q5_K blocks from its own encoder (proven 2026-05-04)
- Provide a quality tier (AF6) for maximum fidelity use cases
- Validate every conversion with receipts and cosine gates
- Enable mixed-precision models (attention@quality, MLP@middle)

## Routing policy v0

| Tensor role | Tier | Format | bpw |
|-------------|------|--------|-----|
| Embedding | quality | AF6 | 6.25 |
| LayerNorm/RMSNorm | quality | AF6 | 6.25 |
| lm_head | quality | AF6 | 6.25 |
| Attention Q/K/V/O | middle | Q5_K | 5.50 |
| MLP gate/up/down | middle | Q5_K | 5.50 |

## Byte-level compatibility (proven)

HXQ_Q5_H encoder emits standard `block_q5_K` bytes:
- 176 bytes per 256 elements
- FP16 d + FP16 dmin header
- 12 bytes packed 6-bit scales/mins (get_scale_min_k4 scheme)
- 32 bytes high-bit plane (interleaved pairs)
- 128 bytes low-nibble plane (paired sub-groups)

Audit result: 6/6 tests PASS, 0.00 max decode error.
Receipt: `receipts/hxq_q5h_runtime_alias_audit_20260504.json`

## Usage

```bash
# Convert with balanced routing
python3 tools/hxq_meta_convert.py

# Force all tensors to middle tier
python3 tools/hxq_meta_convert.py --tier middle

# Convert specific tensor
python3 tools/hxq_meta_convert.py --tensor model.layers.0.mlp.down_proj.weight
```

## Validation gate

Every conversion must pass:
- Tensor cosine ≥ 0.998
- Output cosine ≥ 0.998 (random activation proxy)
- Content hash recorded in receipt
- Cost block (wall time, CPU time, peak memory)

## Proof chain

| Date | Proof | Result |
|------|-------|--------|
| 2026-05-04 | Q5 hierarchy works | PASS — structure is key |
| 2026-05-04 | AF6 hierarchy marginal | KEEP FLAT |
| 2026-05-04 | Q5_K runtime alias | CONFIRMED — byte-compatible |
| 2026-05-04 | Meta-codec converter | VALIDATED |

## What remains

1. **Full-model GGUF emit** — write complete GGUF file with mixed types
2. **Q4_K_M integration** — call llama.cpp quantize for speed-tier tensors
3. **Hydra router upgrade** — replace simple role policy with kurtosis/sensitivity routing
4. **llama-server smoke test** — load emitted GGUF, verify inference
