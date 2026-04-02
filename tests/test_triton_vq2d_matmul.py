"""
Tests for the 2D Vector Quantization Triton kernel.

Verifies:
1. Naive 2D VQ matmul correctness (against explicit W reconstruction)
2. Dequant tile correctness (with and without sidecar)
3. Fused Triton kernel matches naive path (GPU only)
4. Sidecar correction integration
5. Various tensor shapes (TinyLlama-scale)

2D VQ: codebook[K, 2] encodes PAIRS of adjacent weights.
  W[n, 2j:2j+2] = codebook[indices[n, j]]
  indices: [OUT, IN//2] uint16
"""

import pytest
import numpy as np
import torch

from helix_substrate.triton_vq2d_matmul import (
    naive_vq2d_matmul,
    dequant_vq2d_tile,
    is_available as vq2d_is_available,
)

# Skip GPU tests if no CUDA/Triton
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
requires_triton = pytest.mark.skipif(
    not vq2d_is_available(),
    reason="Triton 2D VQ kernel not available (no CUDA or no Triton)"
)


def _make_vq2d_data(out_features: int, in_features: int, k: int = 4096, seed: int = 42):
    """Create random 2D VQ test data: codebook, indices, input.

    Args:
        out_features: OUT dimension
        in_features: IN dimension (must be even)
        k: codebook size (number of 2D entries)
        seed: random seed

    Returns:
        codebook: [K, 2] float32
        indices: [OUT, IN//2] int16
        x: [4, IN] float32 input
    """
    assert in_features % 2 == 0, f"in_features must be even, got {in_features}"
    rng = np.random.RandomState(seed)

    codebook = torch.from_numpy(rng.randn(k, 2).astype(np.float32))
    indices = torch.from_numpy(
        rng.randint(0, k, (out_features, in_features // 2)).astype(np.int16)
    )

    return codebook, indices


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


def _reconstruct_W(codebook, indices, out_features, in_features):
    """Explicit W reconstruction for reference.

    W[n, 2j:2j+2] = codebook[indices[n, j]]
    """
    # codebook[indices.long()] -> [OUT, IN//2, 2]
    W_pairs = codebook[indices.long()]  # [OUT, IN//2, 2]
    W = W_pairs.reshape(out_features, in_features)  # [OUT, IN]
    return W


# --- Test naive 2D VQ matmul ---

class TestNaiveVQ2DMatmul:
    @pytest.mark.parametrize("out_feat,in_feat", [
        (64, 128),
        (256, 2048),
        (2048, 2048),
    ])
    def test_matches_explicit_reconstruction(self, out_feat, in_feat):
        """Naive 2D VQ matmul should match explicit W reconstruction then x @ W^T."""
        codebook, indices = _make_vq2d_data(out_feat, in_feat)
        x = torch.randn(4, in_feat)

        # Explicit reconstruction
        W = _reconstruct_W(codebook, indices, out_feat, in_feat)
        expected = x.float() @ W.t()

        # Naive path
        result = naive_vq2d_matmul(x, codebook, indices)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_with_bias(self):
        """Bias should be added correctly."""
        codebook, indices = _make_vq2d_data(64, 128)
        x = torch.randn(2, 128)
        bias = torch.randn(64)

        result = naive_vq2d_matmul(x, codebook, indices, bias=bias)
        result_no_bias = naive_vq2d_matmul(x, codebook, indices)

        torch.testing.assert_close(result, result_no_bias + bias.unsqueeze(0))

    def test_single_batch(self):
        """Single-sample batch (decode-time)."""
        codebook, indices = _make_vq2d_data(256, 2048)
        x = torch.randn(1, 2048)

        W = _reconstruct_W(codebook, indices, 256, 2048)
        expected = x.float() @ W.t()
        result = naive_vq2d_matmul(x, codebook, indices)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_small_codebook(self):
        """Works with small codebook (k=16, uint8-range)."""
        codebook, indices = _make_vq2d_data(64, 128, k=16)
        x = torch.randn(2, 128)

        W = _reconstruct_W(codebook, indices, 64, 128)
        expected = x.float() @ W.t()
        result = naive_vq2d_matmul(x, codebook, indices)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


# --- Test dequant tile ---

class TestDequantTile:
    def test_full_tile_matches_reconstruction(self):
        """Full-range tile should match explicit W reconstruction."""
        codebook, indices = _make_vq2d_data(128, 256)
        W_expected = _reconstruct_W(codebook, indices, 128, 256)

        W_tile = dequant_vq2d_tile(codebook, indices, 0, 128)
        torch.testing.assert_close(W_tile, W_expected, rtol=1e-6, atol=1e-6)

    def test_partial_tile(self):
        """Partial tile should match the corresponding rows."""
        codebook, indices = _make_vq2d_data(128, 256)
        W_full = _reconstruct_W(codebook, indices, 128, 256)

        tile = dequant_vq2d_tile(codebook, indices, 32, 96)
        torch.testing.assert_close(tile, W_full[32:96], rtol=1e-6, atol=1e-6)

    def test_with_sidecar(self):
        """Sidecar should modify the correct positions in dequantized tile."""
        codebook, indices = _make_vq2d_data(64, 128)

        # Sidecar: correct 3 positions (in full IN dimension, not IN//2)
        sidecar_rows = torch.tensor([10, 20, 30], dtype=torch.long)
        sidecar_cols = torch.tensor([5, 15, 25], dtype=torch.long)
        sidecar_deltas = torch.tensor([0.5, -0.3, 0.1], dtype=torch.float32)

        tile_no_sc = dequant_vq2d_tile(codebook, indices, 0, 64)
        tile_with_sc = dequant_vq2d_tile(
            codebook, indices, 0, 64,
            sidecar_rows=sidecar_rows, sidecar_cols=sidecar_cols,
            sidecar_deltas=sidecar_deltas,
        )

        # Check that sidecar positions differ by expected amount
        assert abs(tile_with_sc[10, 5].item() - tile_no_sc[10, 5].item() - 0.5) < 1e-6
        assert abs(tile_with_sc[20, 15].item() - tile_no_sc[20, 15].item() - (-0.3)) < 1e-6
        assert abs(tile_with_sc[30, 25].item() - tile_no_sc[30, 25].item() - 0.1) < 1e-6

        # Non-sidecar positions unchanged
        mask = torch.ones(64, 128, dtype=torch.bool)
        mask[10, 5] = False
        mask[20, 15] = False
        mask[30, 25] = False
        torch.testing.assert_close(tile_with_sc[mask], tile_no_sc[mask])

    def test_sidecar_partial_range(self):
        """Sidecar corrections outside tile range should be ignored."""
        codebook, indices = _make_vq2d_data(128, 256)

        # Sidecar rows span full range but tile is partial
        sidecar_rows = torch.tensor([10, 50, 100], dtype=torch.long)
        sidecar_cols = torch.tensor([5, 15, 25], dtype=torch.long)
        sidecar_deltas = torch.tensor([0.5, -0.3, 0.1], dtype=torch.float32)

        # Tile covers rows 0-64, so row 100 sidecar should be ignored
        tile = dequant_vq2d_tile(
            codebook, indices, 0, 64,
            sidecar_rows=sidecar_rows, sidecar_cols=sidecar_cols,
            sidecar_deltas=sidecar_deltas,
        )
        tile_clean = dequant_vq2d_tile(codebook, indices, 0, 64)

        # Rows 10 and 50 should be corrected, row 100 (out of range) ignored
        assert abs(tile[10, 5].item() - tile_clean[10, 5].item() - 0.5) < 1e-6
        assert abs(tile[50, 15].item() - tile_clean[50, 15].item() - (-0.3)) < 1e-6


# --- Test fused Triton kernel ---

class TestFusedVQ2DKernel:
    @requires_triton
    @pytest.mark.parametrize("out_feat,in_feat,batch", [
        (64, 128, 1),        # small
        (256, 2048, 1),      # attn_k/v decode (N=1)
        (2048, 2048, 1),     # attn_q/o decode
        (5632, 2048, 1),     # ffn_gate/up decode
        (2048, 5632, 1),     # ffn_down decode
        (2048, 2048, 16),    # prefill batch
    ])
    def test_matches_naive(self, out_feat, in_feat, batch):
        """Fused Triton kernel output should match naive CPU path."""
        from helix_substrate.triton_vq2d_matmul import fused_vq2d_matmul

        codebook, indices = _make_vq2d_data(out_feat, in_feat)
        x = torch.randn(batch, in_feat)

        # Naive (CPU)
        expected = naive_vq2d_matmul(x, codebook, indices)

        # Fused (GPU)
        result = fused_vq2d_matmul(
            x.cuda(),
            codebook.cuda(),
            indices.cuda(),
        )

        cos = _cosine(result.cpu(), expected)
        max_err = (result.cpu() - expected).abs().max().item()

        # FP16 compute path: expect ~3e-4 relative error (same as scalar VQ)
        assert cos > 0.99999, f"Cosine too low: {cos:.8f}"
        assert max_err < 1.0, f"Max abs error too high: {max_err:.6f}"

    @requires_triton
    def test_with_bias(self):
        """Bias should be added correctly in fused path."""
        from helix_substrate.triton_vq2d_matmul import fused_vq2d_matmul

        codebook, indices = _make_vq2d_data(256, 2048)
        x = torch.randn(1, 2048).cuda()
        bias = torch.randn(256).cuda()

        result = fused_vq2d_matmul(
            x, codebook.cuda(), indices.cuda(), bias=bias,
        )
        result_no_bias = fused_vq2d_matmul(
            x, codebook.cuda(), indices.cuda(),
        )

        diff = (result - result_no_bias - bias.unsqueeze(0)).abs().max().item()
        assert diff < 1e-5, f"Bias not applied correctly: max diff = {diff}"

    @requires_triton
    def test_with_sidecar_scatter(self):
        """Sidecar corrections via scatter_add (N > 16)."""
        from helix_substrate.triton_vq2d_matmul import fused_vq2d_matmul

        out_feat, in_feat = 256, 2048
        codebook, indices = _make_vq2d_data(out_feat, in_feat)

        # Create sidecar
        rng = np.random.RandomState(99)
        n_outliers = 50
        sidecar_rows = torch.from_numpy(
            rng.randint(0, out_feat, n_outliers).astype(np.int64)
        ).cuda()
        sidecar_cols = torch.from_numpy(
            rng.randint(0, in_feat, n_outliers).astype(np.int64)
        ).cuda()
        sidecar_deltas = torch.randn(n_outliers).cuda()

        # N=32 > threshold=16, so scatter path
        x = torch.randn(32, in_feat).cuda()

        result_no_sc = fused_vq2d_matmul(
            x, codebook.cuda(), indices.cuda(),
        )
        result_with_sc = fused_vq2d_matmul(
            x, codebook.cuda(), indices.cuda(),
            sidecar_rows=sidecar_rows,
            sidecar_cols=sidecar_cols,
            sidecar_deltas=sidecar_deltas,
            sidecar_phase="scatter",
        )

        diff = (result_with_sc - result_no_sc).abs().max().item()
        assert diff > 0.001, f"Sidecar had no effect: max diff = {diff}"

    @requires_triton
    def test_with_sidecar_fused(self):
        """Sidecar corrections via fused Triton kernel (N <= 16)."""
        from helix_substrate.triton_vq2d_matmul import fused_vq2d_matmul

        out_feat, in_feat = 256, 2048
        codebook, indices = _make_vq2d_data(out_feat, in_feat)

        rng = np.random.RandomState(99)
        n_outliers = 50
        sidecar_rows = torch.from_numpy(
            rng.randint(0, out_feat, n_outliers).astype(np.int64)
        ).cuda()
        sidecar_cols = torch.from_numpy(
            rng.randint(0, in_feat, n_outliers).astype(np.int64)
        ).cuda()
        sidecar_deltas = torch.randn(n_outliers).cuda()

        # N=1, fused Triton sidecar path
        x = torch.randn(1, in_feat).cuda()

        result_no_sc = fused_vq2d_matmul(
            x, codebook.cuda(), indices.cuda(),
        )
        result_with_sc = fused_vq2d_matmul(
            x, codebook.cuda(), indices.cuda(),
            sidecar_rows=sidecar_rows,
            sidecar_cols=sidecar_cols,
            sidecar_deltas=sidecar_deltas,
            sidecar_phase="fused",
        )

        diff = (result_with_sc - result_no_sc).abs().max().item()
        assert diff > 0.001, f"Sidecar had no effect: max diff = {diff}"

    @requires_triton
    def test_dispatch_log(self):
        """Dispatch log should contain kernel version and config."""
        from helix_substrate.triton_vq2d_matmul import fused_vq2d_matmul, KERNEL_VERSION

        codebook, indices = _make_vq2d_data(256, 2048)
        x = torch.randn(1, 2048).cuda()
        log = {}

        fused_vq2d_matmul(
            x, codebook.cuda(), indices.cuda(),
            _dispatch_log=log,
        )

        assert log["dispatch_selected"] == KERNEL_VERSION
        assert "block_config" in log
        assert "KP" in log["block_config"]  # 2D VQ uses BLOCK_KP

    @requires_triton
    def test_k4096_quality(self):
        """k=4096 should produce high-fidelity output (the production config)."""
        from helix_substrate.triton_vq2d_matmul import fused_vq2d_matmul

        # Simulate a realistic TinyLlama attn layer
        out_feat, in_feat = 2048, 2048
        codebook, indices = _make_vq2d_data(out_feat, in_feat, k=4096)
        x = torch.randn(1, in_feat)

        expected = naive_vq2d_matmul(x, codebook, indices)
        result = fused_vq2d_matmul(
            x.cuda(), codebook.cuda(), indices.cuda(),
        )

        cos = _cosine(result.cpu(), expected)
        assert cos > 0.99999, f"k=4096 cosine too low: {cos:.8f}"

    @requires_triton
    def test_odd_in_features_rejected(self):
        """Should reject odd IN dimensions."""
        from helix_substrate.triton_vq2d_matmul import fused_vq2d_matmul

        codebook = torch.randn(16, 2).cuda()
        indices = torch.zeros(64, 64, dtype=torch.int16).cuda()  # IN//2=64 -> IN=128
        x = torch.randn(1, 129).cuda()  # Odd IN — mismatch with indices

        with pytest.raises(AssertionError):
            fused_vq2d_matmul(x, codebook, indices)
