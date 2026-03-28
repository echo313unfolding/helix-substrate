# CDNA v3 as Tensor Infrastructure Layer: 10-Domain Proof Suite

## Thesis

The same codec — `CDNAv3Writer.write_tensor()` + `CDNAv3Reader.reconstruct()` — handles any 2D float32 tensor under resource constraints. No codec modifications needed across domains.

**Architecture:** VQ k-means → outlier sidecar → optional SVD residual → kurtosis routing

```
Input Tensor (2D float32)
    │
    ├─ kurtosis > 5.0 ──→ VQ k-means (k=256) + SVD residual (rank=8) + sidecar
    │
    └─ kurtosis ≤ 5.0 ──→ VQ k-means (k=256) + sidecar
    │
    └─→ CDNA v3 archive (.cdnav3/)
         ├── codebook.npy     (256 centroids × 1 float32 = 1 KB)
         ├── indices.npy      (N values × 1 byte)
         ├── sidecar.npz      (outlier corrections)
         ├── svd_residual.npz (optional low-rank correction)
         └── meta.json        (provenance, stats)
```

## Domain Table

| # | Domain | Script | Data Source | Key Metric | Gate |
|---|--------|--------|-------------|------------|------|
| 1 | Gradient Compression | `bench_gradient_compress.py` | REAL — TinyLlama backward on WikiText-2 | gradient cosine, SGD step cosine | cos ≥ 0.99 |
| 2 | Embedding Table | `bench_embedding_table.py` | REAL — TinyLlama embed_tokens + lm_head | row cosine, recall@10 | recall ≥ 0.95 |
| 3 | Activation Checkpointing | `bench_activation_checkpoint.py` | REAL — 44 calibration activations | cosine per tensor | cos ≥ 0.999 |
| 4 | Federated Learning | `bench_federated_deltas.py` | PARTIAL — real weights + WikiText-2 SGD | weight cosine, bandwidth ratio | w_cos ≥ 0.9999 |
| 5 | Neural Codec Weights | `bench_codec_weight_compress.py` | REAL — CLIP ViT-B/32 + ResNet-18 | embedding cosine, pred match | cos ≥ 0.99 |
| 6 | RAG Index | `bench_rag_index.py` | REAL — WikiText-2 docs via MiniLM | top-1 match, top-5 overlap | top1 ≥ 75% |
| 7 | LoRA Adapters | `bench_lora_compress.py` | PARTIAL — PEFT LoRA, 5 steps | matrix cosine, merged cosine | cos ≥ 0.99 |
| 8 | MoE Tiered | `bench_moe_tiered.py` | PARTIAL — real FFN weights, sim. experts | hot/cold cosine, savings % | hot ≥ 0.999, save ≥ 15% |
| 9 | Continual Learning | `bench_continual_snapshots.py` | PARTIAL — base + 3 SGD snapshots | restored cosine, delta ratio | cos ≥ 0.999 |
| 10 | Sensor Data | `bench_sensor_timeseries.py` | REAL — PBMC3K scRNA-seq + PDB coords | ARI, PCA var diff, RMSD | ARI ≥ 0.90 |

## Run Instructions

```bash
cd tools/tensor_infra

# Run individual domain
python3 bench_activation_checkpoint.py

# Run all sequentially
for f in bench_*.py; do echo "=== $f ===" && python3 "$f"; done
```

## Receipt Locations

All receipts go to `helix-substrate/receipts/tensor_infra/{domain_name}/` with:
- Full results per tensor
- `cost` block (WO-RECEIPT-COST-01)
- `data_source` field with REAL/PARTIAL flag

## Data Provenance

- **REAL**: Uses existing data on disk or pre-trained model weights. No synthetic generation.
- **PARTIAL**: Uses real model weights but applies simplified training (e.g., single SGD step instead of full fine-tuning). The tensor distributions are realistic but the training dynamics are simplified.
