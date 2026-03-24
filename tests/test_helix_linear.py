#!/usr/bin/env python3
"""
Tests for HelixLinear — CDNA v3 drop-in nn.Linear replacement.

Tests:
    1. HelixLinear output matches CDNAv3Reader.reconstruct() @ x
    2. VQ-only mode (no sidecar, no SVD)
    3. VQ + sidecar mode
    4. VQ + sidecar + SVD mode
    5. Memory savings are real (compressed < dense)
    6. swap_to_helix correctly replaces nn.Linear modules
    7. Gradient-free (buffers, not parameters)
    8. Device movement (.cuda() if available)

Work Order: WO-HELIX-LINEAR-01
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.helix_linear import (
    HelixLinear,
    load_helix_linear_from_cdnav3,
    load_cdna_factors,
    swap_to_helix,
    swap_summary,
)
from helix_substrate.cdnav3_reader import CDNAv3Reader
from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.tensor_policy import TensorPolicy, TensorClass


def _make_test_tensor(rows=64, cols=32, seed=42):
    """Create a reproducible test tensor."""
    rng = np.random.RandomState(seed)
    return rng.randn(rows, cols).astype(np.float32)


def _write_cdnav3(tmpdir, tensor, name, policy):
    """Write a tensor to CDNA v3 format and return the directory path."""
    writer = CDNAv3Writer(tmpdir)
    writer.write_tensor(tensor, name, policy=policy)
    safe = name.replace("/", "_").replace(".", "_")
    return tmpdir / f"{safe}.cdnav3"


# ============================================================================
# Test 1: VQ-only (no sidecar, no SVD)
# ============================================================================

def test_vq_only():
    """HelixLinear with VQ-only matches CDNAv3Reader reconstruction."""
    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook",
        n_clusters=256,
        sidecar_enabled=False,
        svd_residual_rank=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "test.vq_only.weight", policy)

        # Reference: CDNAv3Reader
        reader = CDNAv3Reader(tensor_dir)
        W_ref = reader.reconstruct()

        # HelixLinear
        helix = load_helix_linear_from_cdnav3(tensor_dir)

        assert helix.in_features == 32
        assert helix.out_features == 64
        assert not helix.has_svd
        assert helix.sidecar_positions is None

        # Forward pass
        x = torch.randn(2, 32)
        output = helix(x)
        expected = x @ torch.from_numpy(W_ref).float().t()

        assert output.shape == (2, 64)
        assert torch.allclose(output, expected, atol=1e-5), (
            f"Max diff: {(output - expected).abs().max().item()}"
        )

    print("PASS: test_vq_only")


# ============================================================================
# Test 2: VQ + sidecar
# ============================================================================

def test_vq_sidecar():
    """HelixLinear with VQ + sidecar matches CDNAv3Reader reconstruction."""
    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.ATTENTION_QK,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        sidecar_enabled=True,
        percentile=99.0,  # Keep more outliers for test visibility
        svd_residual_rank=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "blk.1.attn_q.weight", policy)

        reader = CDNAv3Reader(tensor_dir)
        W_ref = reader.reconstruct()

        helix = load_helix_linear_from_cdnav3(tensor_dir)

        assert helix.sidecar_positions is not None
        n_outliers = helix.sidecar_positions.numel()
        print(f"  Sidecar: {n_outliers} outliers patched")

        x = torch.randn(4, 32)
        output = helix(x)
        expected = x @ torch.from_numpy(W_ref).float().t()

        assert torch.allclose(output, expected, atol=1e-5), (
            f"Max diff: {(output - expected).abs().max().item()}"
        )

    print("PASS: test_vq_sidecar")


# ============================================================================
# Test 3: VQ + sidecar + SVD
# ============================================================================

def test_vq_sidecar_svd():
    """HelixLinear with full VQ + sidecar + SVD matches CDNAv3Reader."""
    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.ATTENTION_QK,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        sidecar_enabled=True,
        percentile=99.0,
        svd_residual_rank=8,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "blk.0.attn_q.weight", policy)

        reader = CDNAv3Reader(tensor_dir)
        W_ref = reader.reconstruct()

        helix = load_helix_linear_from_cdnav3(tensor_dir)

        assert helix.has_svd
        assert helix.rank == 8
        print(f"  SVD rank: {helix.rank}")
        print(f"  Sidecar outliers: {helix.sidecar_positions.numel()}")

        x = torch.randn(4, 32)
        output = helix(x)
        expected = x @ torch.from_numpy(W_ref).float().t()

        assert torch.allclose(output, expected, atol=1e-4), (
            f"Max diff: {(output - expected).abs().max().item()}"
        )

    print("PASS: test_vq_sidecar_svd")


# ============================================================================
# Test 4: decode_weight matches CDNAv3Reader.reconstruct
# ============================================================================

def test_decode_weight_matches_reader():
    """HelixLinear.decode_weight() produces identical tensor to CDNAv3Reader."""
    tensor = _make_test_tensor(128, 64)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        sidecar_enabled=True,
        percentile=99.5,
        svd_residual_rank=4,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "blk.0.ffn_gate.weight", policy)

        reader = CDNAv3Reader(tensor_dir)
        W_reader = torch.from_numpy(reader.reconstruct()).float()

        helix = load_helix_linear_from_cdnav3(tensor_dir)
        W_helix = helix.decode_weight()

        assert torch.allclose(W_reader, W_helix, atol=1e-5), (
            f"Max diff: {(W_reader - W_helix).abs().max().item()}"
        )

    print("PASS: test_decode_weight_matches_reader")


# ============================================================================
# Test 5: Memory savings
# ============================================================================

def test_memory_savings():
    """HelixLinear uses less memory than nn.Linear."""
    tensor = _make_test_tensor(256, 128)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook",
        n_clusters=256,
        sidecar_enabled=False,
        svd_residual_rank=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "test.memory.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir)

        savings = helix.memory_savings()
        print(f"  Dense: {savings['dense_bytes']:,} bytes")
        print(f"  Compressed: {savings['compressed_bytes']:,} bytes")
        print(f"  Ratio: {savings['ratio']}x")
        print(f"  Savings: {savings['savings_pct']}%")

        assert savings["ratio"] > 3.0, f"Expected >3x ratio, got {savings['ratio']}x"
        assert savings["compressed_bytes"] < savings["dense_bytes"]

    print("PASS: test_memory_savings")


# ============================================================================
# Test 6: swap_to_helix replaces nn.Linear
# ============================================================================

def test_swap_to_helix():
    """swap_to_helix correctly replaces nn.Linear modules in a model."""
    # Build a simple model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(32, 64, bias=False)
            self.layer2 = nn.Linear(64, 16, bias=True)

        def forward(self, x):
            return self.layer2(torch.relu(self.layer1(x)))

    model = TinyModel()

    # Create CDNA v3 for layer1
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook",
        n_clusters=256,
        sidecar_enabled=False,
        svd_residual_rank=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write layer1 weights
        w1 = model.layer1.weight.data.numpy()
        writer = CDNAv3Writer(tmpdir)
        writer.write_tensor(w1, "layer1.weight", policy=policy)
        safe = "layer1_weight"
        tensor_dir = tmpdir / f"{safe}.cdnav3"

        helix_mod = load_helix_linear_from_cdnav3(tensor_dir)
        helix_modules = {"layer1": helix_mod}

        # Swap
        model = swap_to_helix(model, helix_modules)

        # Verify replacement
        assert isinstance(model.layer1, HelixLinear)
        assert isinstance(model.layer2, nn.Linear)  # Not replaced

        # Forward still works
        x = torch.randn(2, 32)
        output = model(x)
        assert output.shape == (2, 16)

        # Summary
        summary = swap_summary(model)
        assert summary["helix_modules"] == 1
        assert summary["linear_modules"] == 1
        print(f"  Summary: {summary}")

    print("PASS: test_swap_to_helix")


# ============================================================================
# Test 7: No trainable parameters (all buffers)
# ============================================================================

def test_no_parameters():
    """HelixLinear has no trainable parameters (inference-only)."""
    tensor = _make_test_tensor(32, 16)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook",
        n_clusters=256,
        sidecar_enabled=False,
        svd_residual_rank=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "test.params.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir)

        params = list(helix.parameters())
        assert len(params) == 0, f"Expected 0 parameters, got {len(params)}"

        buffers = list(helix.buffers())
        assert len(buffers) > 0, "Expected at least codebook + indices as buffers"

    print("PASS: test_no_parameters")


# ============================================================================
# Test 8: Batch dimensions work
# ============================================================================

def test_batch_dimensions():
    """HelixLinear handles various input shapes correctly."""
    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        sidecar_enabled=True,
        percentile=99.0,
        svd_residual_rank=4,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "blk.0.ffn_up.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir)

        # 2D: [batch, in]
        x_2d = torch.randn(8, 32)
        out_2d = helix(x_2d)
        assert out_2d.shape == (8, 64)

        # 3D: [batch, seq, in]
        x_3d = torch.randn(2, 16, 32)
        out_3d = helix(x_3d)
        assert out_3d.shape == (2, 16, 64)

        # 1D: [in]
        x_1d = torch.randn(32)
        out_1d = helix(x_1d)
        assert out_1d.shape == (64,)

    print("PASS: test_batch_dimensions")


# ============================================================================
# Test 9: CUDA device movement (if available)
# ============================================================================

def test_cuda_if_available():
    """HelixLinear moves to CUDA correctly and produces same results."""
    if not torch.cuda.is_available():
        print("SKIP: test_cuda_if_available (no CUDA)")
        return

    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        sidecar_enabled=True,
        percentile=99.0,
        svd_residual_rank=4,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "blk.0.ffn_gate.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir)

        # CPU forward
        x_cpu = torch.randn(4, 32)
        out_cpu = helix(x_cpu)

        # CUDA forward
        helix_cuda = helix.cuda()
        x_cuda = x_cpu.cuda()
        out_cuda = helix_cuda(x_cuda)

        # Results should match within FP16 tl.dot precision
        # (v3 kernel uses FP16 internally, so CPU FP32 vs GPU FP16 has ~0.5% rel error)
        max_diff = (out_cpu - out_cuda.cpu()).abs().max().item()
        ref_max = out_cpu.abs().max().item() + 1e-8
        rel_err = max_diff / ref_max
        assert rel_err < 1e-3, (
            f"CPU vs CUDA rel_err: {rel_err:.2e} (max_diff={max_diff:.2e})"
        )

    print("PASS: test_cuda_if_available")


# ============================================================================
# Test 10: repr string
# ============================================================================

def test_repr():
    """HelixLinear has informative repr."""
    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.ATTENTION_QK,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        sidecar_enabled=True,
        percentile=99.0,
        svd_residual_rank=8,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "blk.0.attn_q.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir)

        repr_str = repr(helix)
        print(f"  repr: {repr_str}")
        assert "in_features=32" in repr_str
        assert "out_features=64" in repr_str
        assert "svd_rank=8" in repr_str
        assert "compression=" in repr_str

    print("PASS: test_repr")


# ============================================================================
# Main
# ============================================================================

# ============================================================================
# Test 11: Chunked naive matches original decode_weight
# ============================================================================

def test_chunked_naive_matches_decode():
    """Chunked naive path produces identical output to decode_weight @ x."""
    tensor = _make_test_tensor(256, 128)
    policy = TensorPolicy(
        tensor_class=TensorClass.ATTENTION_QK,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        sidecar_enabled=True,
        percentile=99.0,
        svd_residual_rank=8,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "blk.0.attn_q.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir)

        x = torch.randn(4, 128)
        # Naive chunked output
        output = helix._forward_naive(x)
        # Reference via decode_weight
        W = helix.decode_weight()
        expected = x @ W.t()
        if helix.bias is not None:
            expected += helix.bias.unsqueeze(0)

        assert torch.allclose(output, expected, atol=1e-4), (
            f"Chunked naive max diff: {(output - expected).abs().max().item()}"
        )

    print("PASS: test_chunked_naive_matches_decode")


# ============================================================================
# Test 12: compute_dtype FP16
# ============================================================================

def test_compute_dtype_fp16():
    """HelixLinear with compute_dtype=float16 produces reasonable output."""
    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        sidecar_enabled=True,
        percentile=99.0,
        svd_residual_rank=4,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "blk.0.ffn_gate.weight", policy)

        helix_fp32 = load_helix_linear_from_cdnav3(tensor_dir, compute_dtype=torch.float32)
        helix_fp16 = load_helix_linear_from_cdnav3(tensor_dir, compute_dtype=torch.float16)

        assert helix_fp16.codebook_f16 is not None
        assert helix_fp16.codebook_f16.dtype == torch.float16

        x = torch.randn(4, 32)
        out_fp32 = helix_fp32(x)
        out_fp16 = helix_fp16(x)

        # FP16 should be close to FP32 (within ~1% relative error)
        rel_err = (out_fp32 - out_fp16).abs().max().item() / (out_fp32.abs().max().item() + 1e-8)
        assert rel_err < 0.02, f"FP16 vs FP32 rel_err: {rel_err:.4f}"

    print(f"PASS: test_compute_dtype_fp16 (rel_err={rel_err:.4f})")


# ============================================================================
# Test 13: set_compute_dtype
# ============================================================================

def test_set_compute_dtype():
    """set_compute_dtype creates/removes FP16 codebook correctly."""
    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook",
        n_clusters=256,
        sidecar_enabled=False,
        svd_residual_rank=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "test.dtype.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir)

        assert helix.codebook_f16 is None
        helix.set_compute_dtype(torch.float16)
        assert helix.codebook_f16 is not None
        assert helix.codebook_f16.dtype == torch.float16
        helix.set_compute_dtype(torch.float32)
        assert helix.codebook_f16 is None

    print("PASS: test_set_compute_dtype")


# ============================================================================
# Test 14: Sidecar precomputation correctness
# ============================================================================

def test_sidecar_precomputation():
    """Precomputed sidecar rows/cols/deltas match original positions/values."""
    tensor = _make_test_tensor(128, 64)
    policy = TensorPolicy(
        tensor_class=TensorClass.ATTENTION_QK,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        sidecar_enabled=True,
        percentile=99.0,
        svd_residual_rank=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "blk.0.attn_k.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir)

        if helix.sidecar_positions is not None:
            # Verify rows/cols reconstruct positions
            expected_rows = helix.sidecar_positions // 64
            expected_cols = helix.sidecar_positions % 64
            assert torch.equal(helix._sidecar_rows, expected_rows.long())
            assert torch.equal(helix._sidecar_cols, expected_cols.long())

            # Verify deltas = values - VQ
            expected_deltas = helix.sidecar_values - helix._sidecar_vq_vals
            assert torch.allclose(helix._sidecar_deltas, expected_deltas, atol=1e-6)

    print("PASS: test_sidecar_precomputation")


# ============================================================================
# Test 15: Output dtype matches input (FP32, CPU)
# ============================================================================

def test_output_dtype_fp32():
    """HelixLinear output dtype matches fp32 input on CPU."""
    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook",
        n_clusters=256,
        sidecar_enabled=False,
        svd_residual_rank=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "test.dtype_fp32.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir)

        x = torch.randn(2, 32, dtype=torch.float32)
        output = helix(x)
        assert output.dtype == torch.float32, f"Expected fp32 output, got {output.dtype}"

    print("PASS: test_output_dtype_fp32")


# ============================================================================
# Test 16: Output dtype matches input (FP16, GPU fused path)
# ============================================================================

def test_output_dtype_fp16_gpu():
    """HelixLinear output dtype matches fp16 input on GPU (fused path)."""
    if not torch.cuda.is_available():
        print("SKIP: test_output_dtype_fp16_gpu (no CUDA)")
        return

    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook",
        n_clusters=256,
        sidecar_enabled=False,
        svd_residual_rank=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "test.dtype_fp16.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir).cuda()

        x = torch.randn(2, 32, dtype=torch.float16, device="cuda")
        output = helix(x)
        assert output.dtype == torch.float16, f"Expected fp16 output, got {output.dtype}"

    print("PASS: test_output_dtype_fp16_gpu")


# ============================================================================
# Test 17: Output dtype matches input (BF16, GPU fused path)
# ============================================================================

def test_output_dtype_bf16_gpu():
    """HelixLinear output dtype matches bf16 input on GPU (fused path)."""
    if not torch.cuda.is_available():
        print("SKIP: test_output_dtype_bf16_gpu (no CUDA)")
        return

    tensor = _make_test_tensor(64, 32)
    policy = TensorPolicy(
        tensor_class=TensorClass.FFN,
        storage_mode="codebook",
        n_clusters=256,
        sidecar_enabled=False,
        svd_residual_rank=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tensor_dir = _write_cdnav3(tmpdir, tensor, "test.dtype_bf16.weight", policy)
        helix = load_helix_linear_from_cdnav3(tensor_dir).cuda()

        x = torch.randn(2, 32, dtype=torch.bfloat16, device="cuda")
        output = helix(x)
        assert output.dtype == torch.bfloat16, f"Expected bf16 output, got {output.dtype}"

    print("PASS: test_output_dtype_bf16_gpu")


if __name__ == "__main__":
    tests = [
        test_vq_only,
        test_vq_sidecar,
        test_vq_sidecar_svd,
        test_decode_weight_matches_reader,
        test_memory_savings,
        test_swap_to_helix,
        test_no_parameters,
        test_batch_dimensions,
        test_cuda_if_available,
        test_repr,
        test_chunked_naive_matches_decode,
        test_compute_dtype_fp16,
        test_set_compute_dtype,
        test_sidecar_precomputation,
        test_output_dtype_fp32,
        test_output_dtype_fp16_gpu,
        test_output_dtype_bf16_gpu,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        sys.exit(1)
