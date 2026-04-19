# Born-Compressed Training: Status

## Result (2026-04-12)

| Config | d | Update | Final Loss | Gap vs Dense |
|---|---|---|---|---|
| Dense | — | — | 6.12 | — |
| Scalar d=1 + TopK | 1 | k-means/20 steps | 7.29 | +1.17 |
| Scalar d=1 + reassign | 1 | Lloyd's/step | 6.14 | +0.02 |
| Grouped d=2 + reassign | 2 | Lloyd's/step | 6.18 | +0.06 |
| Grouped d=4 + TopK | 4 | k-means/25 steps | 7.13 | +1.01 |
| Grouped d=4 + reassign | 4 | Lloyd's/step | 6.22 | +0.10 |

Model: EchoHybrid 79M (9 blocks, 7 SSM + 2 ATTN, 768d). Data: WikiText-2 train. 500 steps, batch=2, seq=64, lr=1e-4, 256 centroids. CPU.

### Geometry scaling (reassign only)

| d | Final Loss | Gap | Wall Time |
|---|---|---|---|
| 1 | 6.14 | +0.02 | ~25 min |
| 2 | 6.18 | +0.06 | 2.8 hr |
| 4 | 6.22 | +0.10 | 5.3 hr |

The gap scales smoothly with vector dimension. Each doubling of d adds ~0.04 loss points. The tradeoff is geometric cost, not algorithmic failure.

## What changed the result

Per-step Lloyd's reassignment — one iteration of: reassign every weight to its nearest centroid, recompute centroids as mean of new assignments. No EMA, no schedule tuning, no threshold logic. This single algorithm change closed the training gap from +1.17 to +0.02 (scalar) and from +1.01 to +0.10 (grouped d=4). The codebook update algorithm dominates the codec geometry.

## Held-Out Perplexity (WO-ECHO-HYBRID-06, 2026-04-12)

Trained each config independently for 500 steps on WikiText-2 train, evaluated on WikiText-2 validation.

| Config | d | Train Loss | Val PPL | PPL Gap |
|---|---|---|---|---|
| Dense | — | 6.0927 | 1176.04 | — |
| d=1 reassign | 1 | 6.1391 | 1177.98 | +1.94 (0.16%) |
| d=2 reassign | 2 | 6.1367 | 1189.18 | +13.14 (1.1%) |
| d=4 reassign | 4 | 6.2254 | 1205.03 | +28.99 (2.5%) |

Held-out PPL confirms training loss. d=1 reassign is indistinguishable from dense (0.16%). Geometric cost is smooth: ~1% PPL per doubling of d.

## What is still open

- **Wall time**: grouped reassign is expensive on CPU (14-15s per step for vstep alone). GPU would parallelize the distance matrix trivially.
- **Scale**: all results are on a 79M toy hybrid. Scaling behavior at 1B+ is unknown.
- **Downstream eval**: task accuracy not yet tested (held-out PPL now validated).

## Receipts

- `receipts/echo_hybrid/wo_echo_hybrid_05b_vstep.json` — scalar d=1 + reassign
- `receipts/echo_hybrid/wo_echo_hybrid_01b.json` — grouped d=4 + TopK
- `receipts/echo_hybrid/wo_echo_hybrid_01b_d2_vstep_reassign.json` — grouped d=2 + reassign
- `receipts/echo_hybrid/wo_echo_hybrid_01b_d4_vstep_reassign.json` — grouped d=4 + reassign
- `receipts/echo_hybrid/wo_echo_hybrid_06_summary.json` — held-out PPL comparison (all 4 configs)
- `receipts/echo_hybrid/wo_echo_hybrid_06_dense.json` — dense PPL
- `receipts/echo_hybrid/wo_echo_hybrid_06_d1_reassign.json` — d=1 reassign PPL
- `receipts/echo_hybrid/wo_echo_hybrid_06_d2_reassign.json` — d=2 reassign PPL
- `receipts/echo_hybrid/wo_echo_hybrid_06_d4_reassign.json` — d=4 reassign PPL
