# HXQ On-Chain Asset Spec

**Status:** Spec only. No implementation. No chain deployed.

This document maps existing HXQ infrastructure to a future on-chain asset transfer architecture. It is a design artifact for pitching and scoping, not a product claim.

---

## Architecture: Off-Chain Tensor, On-Chain State

```
+---------------------------+       +---------------------------+
|     OFF-CHAIN (local)     |       |    ON-CHAIN (Solana)      |
|                           |       |                           |
|  HXQ compressed tensor    |       |  Token account:           |
|  (GGUF, .cdnav3, .npy)   |       |    - content_hash (32B)   |
|                           |       |    - owner pubkey         |
|  Fidelity receipt (JSON)  |       |    - codec metadata       |
|    - sha256_original      |------>|    - promotion_status     |
|    - sha256_compressed    |       |    - transfer_count       |
|    - cosine_min           |       |                           |
|    - ppl_delta            |       |  Transfer Hook:           |
|    - gate PASS/FAIL       |       |    - validate receipt     |
|                           |       |    - enforce policy       |
|  Risk policy result       |       |    - emit event log       |
|    - decision             |       |                           |
|    - reason_codes         |       |  USDC payment leg:        |
|    - risk_score           |       |    - atomic with transfer |
+---------------------------+       +---------------------------+
```

Heavy tensor payloads stay off-chain. The chain holds identity, hashes, ownership, transfer rules, and settlement.

---

## Mapping: Existing Code to On-Chain Roles

| Existing component | Location | On-chain role |
|---|---|---|
| `HXQAssetReceipt` | `cell-runtime/src/cell/hxq_asset.py` | Receipt schema for Transfer Hook validation. Hash fields (`sha256_original`, `sha256_compressed`, `cosine_min`) become the on-chain verification payload. |
| `validate_hxq_asset()` | `cell-runtime/src/cell/hxq_asset.py` | Transfer Hook logic: block transfer if tensor fidelity < 0.998 or behavioral eval missing. Maps to Solana Token-2022 Transfer Hook. |
| `can_promote()` | `cell-runtime/src/cell/hxq_asset.py` | Promotion gate: candidate → active only with receipts. On-chain: update account state only when both tensor + behavioral receipts are submitted. |
| `evaluate_risk_policy()` | `cell-runtime/src/cell/regulated_asset_adapter.py` | Compliance layer: deterministic risk scoring (sanctions, KYC, velocity, structuring). Runs off-chain, result hash posted on-chain as attestation. |
| `TransferEvent` | `cell-runtime/src/cell/regulated_asset_adapter.py` | Event schema for regulated transfers. Fields map to on-chain event log. |
| `affine_quantize_g128_6bit()` | `helix-substrate/tools/test_raw_distributions.py` | The codec itself. Encodes any tensor at 6.25 bpw. Content hash of compressed output is the on-chain anchor. |
| Receipt JSON (all experiments) | `receipts/` | Off-chain proof store. On-chain record points to receipt hash. Verifier can fetch receipt, recompute hash, confirm match. |

---

## Transfer Flow (v1 Design)

```
1. PACK    Seller compresses model/tensor with HXQ
           Receipt generated: sha256, cosine, gate status

2. REGISTER  Content hash + metadata posted to Solana account
             Status: "candidate"

3. PROMOTE   Seller submits fidelity + behavioral receipt hashes
             validate_hxq_asset() logic runs (off-chain or in Hook)
             Status: "candidate" → "active"

4. LIST      Active asset is discoverable
             Metadata: codec, shape, cosine, bpw, model family

5. TRANSFER  Buyer initiates transfer
             Transfer Hook fires:
               - Checks asset is "active" (not quarantined)
               - Checks USDC payment amount matches listing
               - Checks buyer is not sanctioned (risk policy hash)
             If all pass: atomic transfer of token + USDC

6. DELIVER   Off-chain: buyer receives download link / IPFS CID
             Buyer verifies: recompute sha256, confirm match
```

---

## Solana Account Layout (Sketch)

```rust
// Not implemented. Design sketch only.

pub struct HxqAssetAccount {
    pub owner: Pubkey,
    pub content_hash: [u8; 32],       // SHA256 of compressed tensor
    pub original_hash: [u8; 32],      // SHA256 of original tensor
    pub codec: u8,                    // 0=hxq_affine_6, 1=hxq_affine_g128
    pub group_size: u16,              // 128
    pub bits_per_weight: u8,          // 6
    pub cosine_min: f32,              // from receipt
    pub ppl_delta_pct: f32,           // from receipt
    pub status: u8,                   // 0=candidate, 1=active, 2=quarantined
    pub fidelity_receipt_hash: [u8; 32],
    pub behavioral_receipt_hash: [u8; 32],
    pub risk_attestation_hash: [u8; 32],
    pub transfer_count: u32,
    pub created_at: i64,
    pub updated_at: i64,
}
```

---

## What Exists Today vs What Would Be Built

| Layer | Exists | Would build |
|---|---|---|
| Tensor codec (HXQ) | YES — 6 arch families, 5 raw distributions, receipted | — |
| Receipt schema | YES — `HXQAssetReceipt`, cost blocks, SHA256 | — |
| Fidelity validation | YES — `validate_hxq_asset()`, cosine gate | — |
| Promotion gate | YES — `can_promote()`, candidate → active | — |
| Risk policy engine | YES — `evaluate_risk_policy()`, 14 reason codes | — |
| Content hashing | YES — SHA256 on all tensors and receipts | — |
| Solana program (Anchor) | NO | Token-2022 mint + Transfer Hook |
| On-chain account state | NO | `HxqAssetAccount` PDA |
| Transfer Hook validator | NO | Port `validate_hxq_asset` logic |
| USDC payment integration | NO | SPL token transfer in Hook |
| Off-chain asset registry | NO | IPFS/S3 + content-addressed index |
| Frontend / marketplace | NO | Discovery + purchase flow |

---

## Chain Selection Rationale

**Solana first** because Token-2022 Transfer Hooks are the native mechanism for "run custom logic on every token transfer." No need for proxy contracts or wrapper tokens.

**Not Ethereum** because programmable transfer hooks require ERC-1155 or custom proxy patterns that add complexity. Solana's model is simpler for v1.

**Cross-chain (future):** USDC CCTP burn-and-mint for canonical stablecoin movement. Not in v1.

---

## What This Spec Does NOT Claim

- No code is deployed on any chain
- No stablecoin integration exists
- No wallet or custody system exists
- The regulated_asset_adapter is a risk scorer, not a settlement system
- HXQ is a tensor codec, not a blockchain product
- This spec maps existing infrastructure to a possible future product

---

## Prerequisites Before Building

1. HXQ llama.cpp PR lands (establishes codec credibility)
2. Zamba2 PR merges (establishes contributor standing)
3. Anchor/Rust learning (new skill for this box)
4. Solana devnet testing
5. Legal review of token classification
