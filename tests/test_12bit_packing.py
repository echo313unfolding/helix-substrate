"""Tests for 12-bit packed index storage in HelixLinear."""
import pytest
import torch
import numpy as np

from helix_substrate.index_packing import pack_12bit, unpack_12bit, unpack_12bit_rows
from helix_substrate.helix_linear import HelixLinear


class TestPackUnpack:
    """Core packing/unpacking correctness."""

    def test_roundtrip_small(self):
        orig = torch.tensor([0, 4095, 1, 2048, 100, 3999], dtype=torch.int16)
        packed = pack_12bit(orig)
        assert packed.dtype == torch.uint8
        assert packed.shape[0] == 9  # 6 * 3 // 2
        unpacked = unpack_12bit(packed, 6)
        assert torch.equal(orig, unpacked)

    def test_roundtrip_large(self):
        orig = torch.randint(0, 4096, (100000,), dtype=torch.int16)
        packed = pack_12bit(orig)
        assert packed.shape[0] == 150000
        unpacked = unpack_12bit(packed, 100000)
        assert torch.equal(orig, unpacked)

    def test_size_reduction(self):
        orig = torch.randint(0, 4096, (10000,), dtype=torch.int16)
        packed = pack_12bit(orig)
        assert packed.nbytes == orig.nbytes * 3 // 4  # 75% of original

    def test_row_unpack(self):
        rows, cols = 100, 200
        orig = torch.randint(0, 4096, (rows, cols), dtype=torch.int16)
        packed = pack_12bit(orig)
        # Unpack rows 10-20
        unpacked = unpack_12bit_rows(packed, 10, 20, cols)
        assert torch.equal(orig[10:20], unpacked)

    def test_row_unpack_first_last(self):
        rows, cols = 50, 100
        orig = torch.randint(0, 4096, (rows, cols), dtype=torch.int16)
        packed = pack_12bit(orig)
        # First row
        assert torch.equal(orig[0:1], unpack_12bit_rows(packed, 0, 1, cols))
        # Last row
        assert torch.equal(orig[49:50], unpack_12bit_rows(packed, 49, 50, cols))
        # All rows
        assert torch.equal(orig, unpack_12bit_rows(packed, 0, 50, cols))

    def test_odd_count_raises(self):
        with pytest.raises(ValueError, match="even"):
            pack_12bit(torch.tensor([1, 2, 3], dtype=torch.int16))

    def test_overflow_raises(self):
        with pytest.raises(ValueError, match="4096"):
            pack_12bit(torch.tensor([4096, 0], dtype=torch.int16))


class TestHelixLinearPacked:
    """HelixLinear with 12-bit packed indices."""

    def _make_packed_helix(self, out=64, inp=128, k=4096, vd=2):
        """Create a HelixLinear with packed 12-bit indices."""
        idx_cols = inp // vd
        codebook = torch.randn(k, vd, dtype=torch.float32)
        indices_raw = torch.randint(0, k, (out, idx_cols), dtype=torch.int16)
        packed = pack_12bit(indices_raw)

        hl = HelixLinear(
            in_features=inp,
            out_features=out,
            codebook=codebook,
            indices=indices_raw,  # will be stored as int16 by __init__
            vector_dim=vd,
        )
        # Manually swap to packed format
        hl.register_buffer("indices", packed)
        hl.index_packing = "12bit"
        hl._idx_rows = out
        hl._idx_cols = idx_cols
        return hl, indices_raw, codebook

    def test_dequant_tile_packed(self):
        """_dequant_tile with packed indices matches unpacked."""
        hl_packed, indices_raw, codebook = self._make_packed_helix()

        # Create unpacked version for reference
        hl_unpacked = HelixLinear(
            in_features=128, out_features=64,
            codebook=codebook, indices=indices_raw, vector_dim=2,
        )

        for start in [0, 10, 32]:
            end = min(start + 16, 64)
            tile_packed = hl_packed._dequant_tile(start, end)
            tile_unpacked = hl_unpacked._dequant_tile(start, end)
            assert torch.equal(tile_packed, tile_unpacked), \
                f"Mismatch at rows [{start}:{end}]"

    def test_forward_naive_packed(self):
        """Naive forward with packed indices matches unpacked."""
        hl_packed, indices_raw, codebook = self._make_packed_helix()

        hl_unpacked = HelixLinear(
            in_features=128, out_features=64,
            codebook=codebook, indices=indices_raw, vector_dim=2,
        )

        x = torch.randn(2, 128)
        out_packed = hl_packed._forward_naive(x)
        out_unpacked = hl_unpacked._forward_naive(x)
        assert torch.allclose(out_packed, out_unpacked, atol=1e-6), \
            f"Max diff: {(out_packed - out_unpacked).abs().max()}"

    def test_from_quantized_config_packed(self):
        """Shell creation with index_packing='12bit'."""
        shell = HelixLinear.from_quantized_config(
            in_features=128, out_features=64,
            n_clusters=4096, vector_dim=2,
            index_packing="12bit",
        )
        assert shell.index_packing == "12bit"
        assert shell.indices.dtype == torch.uint8
        assert shell.indices.ndim == 1
        # 64 * 64 indices (128/2=64 cols), packed = 64*64*3//2 = 6144 bytes
        assert shell.indices.shape[0] == 64 * 64 * 3 // 2

    def test_from_quantized_config_unpacked(self):
        """Shell creation without packing (backward compat)."""
        shell = HelixLinear.from_quantized_config(
            in_features=128, out_features=64,
            n_clusters=4096, vector_dim=2,
        )
        assert shell.index_packing is None
        assert shell.indices.dtype == torch.int16
        assert shell.indices.shape == (64, 64)

    def test_recompute_derived_detects_packed(self):
        """_recompute_derived auto-detects 12-bit packed format."""
        shell = HelixLinear.from_quantized_config(
            in_features=128, out_features=64,
            n_clusters=4096, vector_dim=2,
            index_packing="12bit",
        )
        # Simulate safetensors loading packed data
        packed_data = pack_12bit(torch.randint(0, 4096, (64 * 64,), dtype=torch.int16))
        shell.register_buffer("indices", packed_data)
        shell.register_buffer("codebook", torch.randn(4096, 2, dtype=torch.float32))

        shell._recompute_derived()
        assert shell.index_packing == "12bit"
        assert shell._idx_rows == 64
        assert shell._idx_cols == 64

    def test_vram_savings(self):
        """Verify 25% storage reduction."""
        out, inp, k, vd = 3584, 7168, 4096, 2
        idx_cols = inp // vd

        unpacked_bytes = out * idx_cols * 2  # int16
        packed_bytes = out * idx_cols * 3 // 2  # 12-bit

        ratio = packed_bytes / unpacked_bytes
        assert abs(ratio - 0.75) < 0.01, f"Expected 75% ratio, got {ratio:.2%}"
