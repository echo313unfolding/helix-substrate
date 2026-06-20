# Related Work: Compression-Induced Routing Signals

This document positions HXQ's "compression creates routing signals" contribution
against current literature. The goal is accurate placement, not overclaiming.

## 1. Reconstruction Error as Routing Signal

**Tang, "Compression is Routing: Reconstruction Error as an Intrinsic Signal
for Modular Language Models," arXiv:2512.16963, Dec 2025.**

Closest direct phrase and concept. Tang trains an 87M-parameter Transformer
autoencoder (512 tokens → 8 latent vectors) and observes that reconstruction
error varies systematically across domains (code 99.47%, Wikipedia 47.76%,
random 0.57%). This variance becomes an "Intrinsic Distribution Fingerprint"
that routes language-domain expert modules without an explicit gating network.

**Distinction:** Tang's signal is scalar (reconstruction accuracy per domain)
and routes *language domains* to *expert modules*. HXQ routes *compression
backends* per *tensor* using the *geometric structure* of the residual
(kurtosis, spectral ratio, SVD rank, ACF) and *encoded-body features*
(Ghost: transition entropy, markov order, autocorrelation) extracted from
compressed bytes without decompression. Different level of the stack,
different signal dimensionality.

## 2. Routing as Control vs Content

**Ye, Yuan, Sharkey, "Polysemantic Experts, Monosemantic Paths: Routing as
Control in MoEs," arXiv:2604.17837, Apr 2026.**

Introduces a parameter-free decomposition splitting each MoE layer's hidden
state into a *control signal* that causally drives routing and an orthogonal
*content channel* invisible to the router. Key finding: expert paths (not
individual experts) are the natural unit of interpretability.

**Distinction:** Ye et al. *analyze* an existing routing mechanism in trained
MoEs. HXQ *constructs* a new control signal — Ghost features (te, tr, mo, ac)
are properties of the *encoded representation*, not the original FP16 weights.
The act of compressing creates a signal space that did not exist before
compression. This is a constructive claim, not an analytical decomposition.

The vocabulary is directly applicable: HXQ separates content (weights, data,
verdicts) from control (Ghost features, residual damage, correction traces).

## 3. Residual Quantization and MoE Codebook Adaptation

**Zhong et al., "RQ-MoE: Residual Quantization via Mixture of Experts for
Efficient Input-Dependent Vector Compression," arXiv:2605.14359, May 2026.
ICML 2026.**

Two-level MoE with dual-stream quantization enabling input-dependent codebook
selection. Shows that standard RQ and QINCo are constrained special cases.
Achieves 6-14x faster decoding than prior VQ methods.

**Distinction:** RQ-MoE adapts codebooks *within* a single residual VQ method.
HXQ routes *across* heterogeneous compression backends (affine at multiple bit
widths, VQ, RVQ, sidecar correction, exact passthrough). The routing problem
is cross-codec selection, not within-codec codebook adaptation.

## 4. Quantization Preserving Routing Consistency

**Park et al., "Value-and-Structure Alignment for Routing-Consistent
Quantization of Mixture-of-Experts Models," arXiv:2606.05688, Jun 2026.**

Asks how quantization *affects* existing MoE routing stability and proposes
alignment techniques to preserve expert selection behavior under PTQ.

**Distinction:** VSRAQ preserves an existing router under the stress of
quantization. HXQ asks the adjacent inverse question: can the structural
artifacts of compression *create* routing signals for codec selection? VSRAQ
treats quantization-routing interaction as a problem to mitigate. HXQ treats
it as a signal to exploit.

## 5. Sensitivity-Guided and Mixed-Precision Quantization

**HAWQ (Dong et al., arXiv:1905.03696), HAWQ-V2, HAQ/DNAS-style mixed
precision (Wu et al., arXiv:1812.00090), constrained optimization approaches
(Yao et al., arXiv:2110.06554).**

Established prior art for profiling tensor/layer sensitivity and choosing
per-tensor or per-layer bit widths. HAWQ uses Hessian eigenvalues to rank
sensitivity. DNAS formulates bit-width selection as differentiable architecture
search. These methods profile the input *before* compression to choose the
compression policy.

**Distinction:** The mixed-precision family profiles sensitivity before
compression. HXQ *also* profiles damage *after* compression (Residual Contract:
12 structural features of E = W - W_hat) and *before full decompression*
(Ghost Bridge: 4 encoded-body features from compressed bytes). These are
different temporal positions in the compression pipeline:

- Pre-compression sensitivity (HAWQ/AWQ) → choose codec
- In-compression encoded structure (Ghost) → skip/probe decision
- Post-compression residual geometry (Residual Contract) → accept/correct/fallback

## 6. Outlier-Aware LLM Quantization

**AWQ (Lin et al., arXiv:2306.00978), SmoothQuant (Xiao et al.,
arXiv:2211.10438), SpQR (Dettmers et al., arXiv:2306.03078), GPTQ
(Frantar et al., arXiv:2210.17323).**

Strong prior art establishing that quantization error is structured: certain
channels and weights dominate loss. AWQ identifies salient weights from
activation statistics. SmoothQuant migrates quantization difficulty between
activations and weights. SpQR isolates outlier weights in sparse high-precision
storage. GPTQ uses approximate second-order information for one-shot quantization.

**Distinction:** These methods use error structure to *protect weights* or
*choose precision* during compression. HXQ turns the residual structure itself
into a *post-compression routing signal*: the damage pattern classifies as
DISTRIBUTED (accept), CONCENTRATED (outlier sidecar), LOW_RANK (low-rank
correction), or STRUCTURED (fallback to safer head). The residual is not just
measured — its geometry drives downstream routing decisions.

## 7. Compressed-Domain Computation

**JPEG/transform-domain CNN inference (Gueguen et al., arXiv:1812.11690),
compressed-domain learning (Wang et al., WACV 2022), MPEG Neural Network
Compression and Representation standard.**

Prior art that compressed representations can support downstream computation
without full reconstruction. Transform-domain inference operates on DCT
coefficients rather than pixel values. MPEG NNR standardizes compressed NN
bitstreams.

**Distinction:** Crystal Vault's original thesis ("seed IS decoder," Level 4)
was too strong and did not fully prove out. What proved out was Level 2 (zero
materialization with hardware translation boundary) plus the discovery that
compressed representations create useful *control-plane signals* for routing.
The narrower, defensible claim is not "compressed artifacts self-execute" but
"compressed artifacts generate routing information invisible to the
uncompressed system."

## 8. HXQ Contribution

HXQ does not claim to invent compression-aware routing. The contribution is a
cross-codec routing stack that combines encoded-body pre-routing with
residual-damage post-routing:

**a. Ghost Bridge** (`ghost_bridge.py`): Extracts 4 structural features
(transition entropy, transition rank, markov order, index autocorrelation)
from encoded bytes WITHOUT decompression. Architecture-aware linear model
predicts tensor fragility. Decides SKIP_PROBE (route directly) or
PROBE_REQUIRED (run full codec probing). 53.8% of tensors cleared at
precision=0.955, recall=0.904.

**b. Residual Contract** (`residual_contract.py`): Common damage language
across heterogeneous codecs. Profiles 12 structural features of E = W - W_hat
including spectral ratio, ACF, SVD rank, kurtosis, channel concentration.
Classifies damage as DISTRIBUTED / CONCENTRATED / LOW_RANK / STRUCTURED.

**c. Residual Router** (`residual_router.py`): Turns residual profiles into
routing decisions: accept, suggest outlier sidecar, suggest low-rank correction,
fall back to safer head, or request further probing. Conservative by design:
low-confidence signals do not override Hydra's probe-based routing.

**d. Hydra Router** (`hydra_router.py`): Multi-head codec routing with 4
policies and 7 heads. Three entry points: `route()` (probe-based),
`route_with_ghost()` (pre-screening), `route_with_residuals()` (post-verification).

**The control loop:**

```
tensor / encoded shard
  → Ghost pre-route (encoded-body features, no decompression)
  → Hydra route (probe-based head selection)
  → codec candidate reconstruction
  → Residual Contract profiles E = W - W_hat
  → Residual Router: accept / correct / fallback
  → receipt
```

**The constructive claim:** compression does not merely degrade information —
it creates a new signal space. Ghost features are properties of the encoded
artifact, not the original weights. Residual geometry is a property of the
damage pattern, not the output quality. These compression-induced signals
form a control plane for cross-codec routing that is invisible to systems
operating on uncompressed representations.

## Comparison Table

| Work | Signal used | Routes what | Temporal position | HXQ distinction |
|---|---|---|---|---|
| Tang 2025 | Reconstruction error magnitude | Language-domain experts | Post-reconstruction | HXQ uses residual *geometry* (12 features), not magnitude; routes codecs, not domains |
| Ye et al. 2026 | Control/content decomposition | Existing MoE experts | Within trained model | HXQ *constructs* new control signals via compression; Ye et al. *analyze* existing ones |
| RQ-MoE 2026 | Input features | Codebook experts within RVQ | During quantization | HXQ routes *across* heterogeneous codecs, not within one VQ family |
| VSRAQ 2026 | Routing logit alignment | Preserves existing MoE routing | During quantization | HXQ asks the inverse: can quantization *create* routing signals? |
| HAWQ/AWQ/SpQR | Sensitivity, activations, outliers | Bit-width or outlier protection | Pre-compression | HXQ adds post-compression (residual) and pre-decompression (Ghost) positions |
| **HXQ** | **Ghost (encoded-body) + Residual (damage geometry)** | **Cross-codec backends** | **Pre-decompression + post-reconstruction** | Compression-induced routing signal across heterogeneous codecs |

## Positioning Statement

> HXQ extends compression-aware routing from sequence/domain routing and
> mixed-precision quantization into cross-codec tensor routing, where
> compressed-domain structure and residual-damage geometry act as conservative
> control signals for backend selection, correction, and fallback.

## Evidence

All components are implemented, tested, and tagged:

- Ghost Bridge: 25/25 tests, `v0.4.0` (Phase 0.17b receipt: 53.8% cleared)
- Residual Contract: 26/26 tests, `v0.4.1-residual-contract`
- Residual Router: 25/25 tests, `v0.4.2-residual-router`
- Hydra Router: 19/19 tests
- Combined: 95/95 tests pass

Convergence with MorphSAT (safety governance) and Crystal Vault (compressed
runtime) documented in `morphsat/docs/PROFILE_ROUTE_RECEIPT_PATTERN.md`.
