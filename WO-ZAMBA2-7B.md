# WO-ZAMBA2-7B: Compress Zamba2-7B-Instruct for HuggingFace Zoo

**Target:** 2026-03-30 (cloud session)
**Priority:** High — 7B hybrid is the marquee proof point

## Model

- **HF ID:** `Zyphra/Zamba2-7B-Instruct`
- **Architecture:** Hybrid Mamba2 + shared Transformer (same as Zamba2-2.7B, scaled up)
- **Layers:** 81 total (68 Mamba2 + 13 hybrid at layers 6,11,17,23,29,35,41,47,53,59,65,71,77)
- **Hidden:** 3584, attention_hidden_size=7168, intermediate=14336
- **Shared blocks:** 2 (ABAB pattern), adapter_rank=128
- **BF16 size:** 15.1 GB (4 safetensors shards)
- **Vocab:** 32000
- **License:** Apache 2.0

## Resource Estimate

| Stage | Time (est) | GPU | RAM | Disk |
|-------|-----------|-----|-----|------|
| Download | ~5 min | — | — | 15.1 GB |
| Baseline PPL (GPU BF16) | ~30 min | 15 GB VRAM | 20 GB | — |
| Compress (CPU) | ~15 min | — | 30 GB | ~4 GB cdnav3 |
| Helix PPL (GPU BF16) | ~30 min | 8-10 GB | 20 GB | — |
| Convert to HF | ~1 min | — | 10 GB | ~5 GB |
| Upload | ~5 min | — | — | — |
| **Total** | **~90 min** | **15 GB peak** | **30 GB peak** | **~25 GB** |

Cloud box: RTX 4090 (24 GB), 64 GB RAM, 390 GB free. Fits comfortably.

## Expected Compression

Based on Zamba2-2.7B results:

| | Zamba2-2.7B | Zamba2-7B (est) |
|---|---|---|
| Dense size | 5.1 GB | 15.1 GB |
| Helix size | 2.8 GB | ~5-7 GB |
| Ratio | 1.83x | ~2.2-3.0x |
| PPL delta | +6.59% | target <+10% |

Ratio should be BETTER than 2.7B because:
- 81 layers → more in_proj/out_proj weight mass (compressible)
- Only 2 shared transformer blocks (same as 2.7B) → shared exact mass doesn't scale
- Net: compressible fraction of total params increases with model size

## Pipeline Command

```bash
# On cloud (user@38.224.253.150)
python3 ssm_compress_pipeline.py \
    --model Zyphra/Zamba2-7B-Instruct \
    --hf-org EchoLabs33 \
    --skip-upload
```

Then after verification:
```bash
# GPU BF16 eval (use batch_gpu_eval.py)
python3 batch_gpu_eval.py --model zamba2-7b --receipt-dir receipts/gpu_eval

# Upload after card is written
python3 -m huggingface_hub.commands.huggingface_cli upload \
    EchoLabs33/zamba2-7b-instruct-helix . . --repo-type model
```

## Pre-flight Checklist

- [x] Pipeline handles Zamba2 architecture (proven on 1.2B and 2.7B)
- [x] compress.py handles shared transformer blocks + LoRA adapters
- [x] convert_to_hf.py handles hybrid models
- [ ] **BUG:** Verify convert_to_hf.py doesn't introduce precision loss (Coder-3B/Qwen-7B failure)
- [ ] **BUG:** Verify HF post-processing handles multi-shard safetensors (7B output may be >2GB → 2 shards)
- [ ] Ensure GPU BF16 eval is used (not CPU FP32 — WO-CPU-FORWARD-BUG)
- [ ] Add `zamba2-7b` entry to batch_gpu_eval.py MODELS dict

## Files to SCP to Cloud

Already on cloud from Zamba2-2.7B session:
- `ssm_compress_pipeline.py`
- `compress.py`
- `convert_to_hf.py`
- `eval_ppl_cpu.py` (for pipeline stage 3, but should switch to GPU eval)
- `batch_gpu_eval.py`

Need to update after bug fixes:
- `helix_substrate/` package (hf_integration.py fixes)
- `batch_gpu_eval.py` (add zamba2-7b entry)

## Bug Fixes Required Before Run

### 1. convert_to_hf.py precision verification
The Coder-3B model shows per-layer activation cosine degrading from 0.98 to 0.55 across 36 layers,
while 3B-Instruct maintains 0.999+. Individual weights are cosine 0.998+.
Need to verify convert_to_hf.py isn't introducing precision loss during the numpy→safetensors conversion.

### 2. Multi-shard post-processing
Zamba2-7B helix output may exceed 2 GB (est. 5-7 GB) → safetensors will shard.
`hf_integration.py` `_process_model_after_weight_loading` currently only reads `model.safetensors`.
Need to handle `model.safetensors.index.json` + multiple shard files.

### 3. Pipeline GPU eval
`ssm_compress_pipeline.py` stage 3 (helix_ppl) uses `eval_ppl_cpu.py` which is CPU FP32.
Should add a `--gpu-eval` flag or replace with GPU BF16 eval for authoritative numbers.

## Model Card Template

Same structure as Zamba2-2.7B card. Key changes:
- 81 layers, 68 Mamba2 + 13 hybrid
- Hidden 3584
- Updated compression receipt
- GPU BF16 PPL numbers
- Companion table with all 11 models

## Success Criteria

1. Compression completes without errors
2. GPU BF16 PPL delta < +10%
3. Generation quality is coherent on 3 sanity prompts
4. Per-layer activation cosine stays > 0.99 (lesson from Coder-3B failure)
5. Model card has receipted numbers only
6. Uploaded to EchoLabs33/zamba2-7b-instruct-helix
