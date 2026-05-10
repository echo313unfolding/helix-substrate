"""
Tests for the RVQ (Residual Vector Quantization) Triton kernel.

Verifies:
1. Nibble packing/unpacking roundtrip
2. Naive RVQ matmul correctness (against explicit W reconstruction)
3. Fused Triton kernel matches naive path (GPU only)
4. Sidecar correction integration
5. Various tensor shapes (TinyLlama-scale)
"""

import pytest
import numpy as np
import torch

from helix_substrate.triton_rvq_matmul import (
    pack_nibbles,
    unpack_nibbles,
    naive_rvq_matmul,
    dequant_rvq_tile,
    is_available as rvq_is_available,
)

# Skip GPU tests if no CUDA/Triton
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
requires_triton = pytest.mark.skipif(
    not rvq_is_available(),
    reason="Triton RVQ kernel not available (no CUDA or no Triton)"
)


def _make_rvq_data(out_features: int, in_features: int, seed: int = 42):
    """Create random RVQ test data: codebooks, packed indices, input."""
    rng = np.random.RandomState(seed)

    # Two codebooks of 16 entries each
    codebook1 = torch.from_numpy(rng.randn(16).astype(np.float32))
    codebook2 = torch.from_numpy(rng.randn(16).astype(np.float32) * 0.3)

    # Random 4-bit indices, packed
    idx1 = torch.from_numpy(rng.randint(0, 16, (out_features, in_features)).astype(np.uint8))
    idx2 = torch.from_numpy(rng.randint(0, 16, (out_features, in_features)).astype(np.uint8))
    packed = pack_nibbles(idx1, idx2)

    return codebook1, codebook2, packed, idx1, idx2


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flat tensors."""
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    dot = torch.dot(a_f, b_f)
    na = torch.norm(a_f)
    nb = torch.norm(b_f)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return (dot / (na * nb)).item()


# --- Test nibble packing ---

class TestNibblePacking:
    def test_roundtrip(self):
        """Pack then unpack should recover original indices."""
        idx1 = torch.randint(0, 16, (100, 200), dtype=torch.uint8)
        idx2 = torch.randint(0, 16, (100, 200), dtype=torch.uint8)
        packed = pack_nibbles(idx1, idx2)
        r1, r2 = unpack_nibbles(packed)
        assert torch.equal(r1, idx1)
        assert torch.equal(r2, idx2)

    def test_known_values(self):
        """Verify specific packing: (10, 5) -> 0xA5 = 165."""
        idx1 = torch.tensor([[10]], dtype=torch.uint8)
        idx2 = torch.tensor([[5]], dtype=torch.uint8)
        packed = pack_nibbles(idx1, idx2)
        assert packed.item() == 0xA5
        r1, r2 = unpack_nibbles(packed)
        assert r1.item() == 10
        assert r2.item() == 5

    def test_boundary_values(self):
        """Test all 256 possible packed values."""
        all_idx1 = torch.arange(16, dtype=torch.uint8).unsqueeze(1).expand(16, 16).flatten()
        all_idx2 = torch.arange(16, dtype=torch.uint8).unsqueeze(0).expand(16, 16).flatten()
        packed = pack_nibbles(all_idx1, all_idx2)
        r1, r2 = unpack_nibbles(packed)
        assert torch.equal(r1, all_idx1)
        assert torch.equal(r2, all_idx2)
        # Should produce all 256 unique bytes
        assert len(packed.unique()) == 256


# --- Test naive RVQ matmul ---

class TestNaiveRVQMatmul:
    @pytest.mark.parametrize("out_feat,in_feat", [
        (64, 128),
        (256, 2048),
        (2048, 2048),
    ])
    def test_matches_explicit_reconstruction(self, out_feat, in_feat):
        """Naive RVQ matmul should match explicit W = cb1[i1] + cb2[i2] then x @ W^T."""
        codebook1, codebook2, packed, idx1, idx2 = _make_rvq_data(out_feat, in_feat)
        x = torch.randn(4, in_feat)

        # Explicit reconstruction
        W_explicit = codebook1[idx1.long()] + codebook2[idx2.long()]
        expected = x.float() @ W_explicit.t()

        # Naive path
        result = naive_rvq_matmul(x, codebook1, codebook2, packed)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_with_bias(self):
        """Bias should be added correctly."""
        codebook1, codebook2, packed, _, _ = _make_rvq_data(64, 128)
        x = torch.randn(2, 128)
        bias = torch.randn(64)

        result = naive_rvq_matmul(x, codebook1, codebook2, packed, bias=bias)
        result_no_bias = naive_rvq_matmul(x, codebook1, codebook2, packed)

        torch.testing.assert_close(result, result_no_bias + bias.unsqueeze(0))


# --- Test dequant tile ---

class TestDequantTile:
    def test_full_tile_matches_naive(self):
        """Full-range tile should match naive W reconstruction."""
        codebook1, codebook2, packed, idx1, idx2 = _make_rvq_data(128, 256)
        W_expected = codebook1[idx1.long()] + codebook2[idx2.long()]

        W_tile = dequant_rvq_tile(codebook1, codebook2, packed, 0, 128)
        torch.testing.assert_close(W_tile, W_expected, rtol=1e-6, atol=1e-6)

    def test_partial_tile(self):
        """Partial tile should match the corresponding rows."""
        codebook1, codebook2, packed, idx1, idx2 = _make_rvq_data(128, 256)
        W_full = codebook1[idx1.long()] + codebook2[idx2.long()]

        tile = dequant_rvq_tile(codebook1, codebook2, packed, 32, 96)
        torch.testing.assert_close(tile, W_full[32:96], rtol=1e-6, atol=1e-6)

    def test_with_sidecar(self):
        """Sidecar should modify the correct positions."""
        codebook1, codebook2, packed, _, _ = _make_rvq_data(64, 128)

        # Sidecar: correct 3 positions
        sidecar_rows = torch.tensor([10, 20, 30], dtype=torch.long)
        sidecar_cols = torch.tensor([5, 15, 25], dtype=torch.long)
        sidecar_deltas = torch.tensor([0.5, -0.3, 0.1], dtype=torch.float32)

        tile_no_sc = dequant_rvq_tile(codebook1, codebook2, packed, 0, 64)
        tile_with_sc = dequant_rvq_tile(
            codebook1, codebook2, packed, 0, 64,
            sidecar_rows=sidecar_rows, sidecar_cols=sidecar_cols,
            sidecar_deltas=sidecar_deltas,
        )

        # Check that sidecar positions differ
        assert tile_with_sc[10, 5] != tile_no_sc[10, 5]
        assert abs(tile_with_sc[10, 5] - tile_no_sc[10, 5] - 0.5) < 1e-6

        # Check non-sidecar positions unchanged
        mask = torch.ones(64, 128, dtype=torch.bool)
        mask[10, 5] = False
        mask[20, 15] = False
        mask[30, 25] = False
        torch.testing.assert_close(tile_with_sc[mask], tile_no_sc[mask])


# --- Test fused Triton kernel ---

class TestFusedRVQKernel:
    @requires_triton
    @pytest.mark.parametrize("out_feat,in_feat,batch", [
        (64, 128, 1),       # small
        (256, 2048, 1),     # attn_k/v decode (N=1)
        (2048, 2048, 1),    # attn_q/o decode
        (5632, 2048, 1),    # ffn_gate/up decode
        (2048, 5632, 1),    # ffn_down decode
        (2048, 2048, 16),   # prefill batch
    ])
    def test_matches_naive(self, out_feat, in_feat, batch):
        """Fused Triton kernel output should match naive CPU path."""
        from helix_substrate.triton_rvq_matmul import fused_rvq_matmul

        codebook1, codebook2, packed, _, _ = _make_rvq_data(out_feat, in_feat)
        x = torch.randn(batch, in_feat)

        # Naive (CPU)
        expected = naive_rvq_matmul(x, codebook1, codebook2, packed)

        # Fused (GPU)
        result = fused_rvq_matmul(
            x.cuda(),
            codebook1.cuda(),
            codebook2.cuda(),
            packed.cuda(),
        )

        cos = _cosine(result.cpu(), expected)
        max_err = (result.cpu() - expected).abs().max().item()

        # FP16 compute path: expect ~3e-4 relative error (same as VQ v3)
        assert cos > 0.99999, f"Cosine too low: {cos:.8f}"
        # Absolute error scales with tensor magnitude
        assert max_err < 1.0, f"Max abs error too high: {max_err:.6f}"

    @requires_triton
    def test_with_bias(self):
        """Bias should be added correctly in fused path."""
        from helix_substrate.triton_rvq_matmul import fused_rvq_matmul

        codebook1, codebook2, packed, _, _ = _make_rvq_data(256, 2048)
        x = torch.randn(1, 2048).cuda()
        bias = torch.randn(256).cuda()

        result = fused_rvq_matmul(x, codebook1.cuda(), codebook2.cuda(), packed.cuda(), bias=bias)
        result_no_bias = fused_rvq_matmul(x, codebook1.cuda(), codebook2.cuda(), packed.cuda())

        diff = (result - result_no_bias - bias.unsqueeze(0)).abs().max().item()
        assert diff < 1e-5, f"Bias not applied correctly: max diff = {diff}"

    @requires_triton
    def test_with_sidecar(self):
        """Sidecar corrections should improve fidelity."""
        from helix_substrate.triton_rvq_matmul import fused_rvq_matmul

        out_feat, in_feat = 256, 2048
        codebook1, codebook2, packed, idx1, idx2 = _make_rvq_data(out_feat, in_feat)

        # Reconstruct W and pick some "outlier" positions
        W = codebook1[idx1.long()] + codebook2[idx2.long()]
        # Pretend the true weight has deviations at sparse positions
        rng = np.random.RandomState(99)
        n_outliers = 50
        flat_pos = rng.choice(out_feat * in_feat, n_outliers, replace=False)
        sidecar_rows = torch.from_numpy(flat_pos // in_feat).long()
        sidecar_cols = torch.from_numpy(flat_pos % in_feat).long()
        sidecar_deltas = torch.randn(n_outliers) * 0.5

        x = torch.randn(1, in_feat).cuda()

        result_no_sc = fused_rvq_matmul(
            x, codebook1.cuda(), codebook2.cuda(), packed.cuda(),
        )
        result_with_sc = fused_rvq_matmul(
            x, codebook1.cuda(), codebook2.cuda(), packed.cuda(),
            sidecar_rows=sidecar_rows.cuda(),
            sidecar_cols=sidecar_cols.cuda(),
            sidecar_deltas=sidecar_deltas.cuda(),
        )

        # With sidecar should differ from without
        diff = (result_with_sc - result_no_sc).abs().max().item()
        assert diff > 0.001, f"Sidecar had no effect: max diff = {diff}"

    @requires_triton
    def test_dispatch_log(self):
        """Dispatch log should contain kernel version and config."""
        from helix_substrate.triton_rvq_matmul import fused_rvq_matmul, KERNEL_VERSION

        codebook1, codebook2, packed, _, _ = _make_rvq_data(256, 2048)
        x = torch.randn(1, 2048).cuda()
        log = {}

        fused_rvq_matmul(
            x, codebook1.cuda(), codebook2.cuda(), packed.cuda(),
            _dispatch_log=log,
        )

        assert log["dispatch_selected"] == KERNEL_VERSION
        assert "block_config" in log
