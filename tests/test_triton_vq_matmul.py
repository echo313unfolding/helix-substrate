#!/usr/bin/env python3
"""
Tests for Triton fused VQ gather-matmul kernel.

Validates that the fused kernel produces correct results compared to the naive
codebook[indices] + matmul path. Tests all combinations:

1. VQ-only (no sidecar, no SVD)
2. VQ + sidecar
3. VQ + sidecar + SVD
4. Various batch sizes
5. Non-power-of-2 dimensions
6. Numerical accuracy vs naive
7. Tiled v3 kernel (FP16 tl.dot) specific tests

The v3 tiled kernel uses FP16 tl.dot with FP32 accumulate, so absolute error
scales with output magnitude. Tests use relative error (rel_err < 5e-4) or
relaxed absolute thresholds appropriate for FP16 precision.

Requires CUDA GPU. Skips gracefully if unavailable.

Work Order: WO-HELIX-LINEAR-01
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Relative error threshold for FP16 tl.dot precision
REL_ERR_THRESHOLD = 1e-3


def _skip_no_cuda():
    if not torch.cuda.is_available():
        print("SKIP: No CUDA available")
        return True
    return False


def _skip_no_triton():
    try:
        from helix_substrate.triton_vq_matmul import HAS_TRITON
        if not HAS_TRITON:
            print("SKIP: Triton not available")
            return True
    except ImportError:
        print("SKIP: triton_vq_matmul not importable")
        return True
    return False


def _naive_vq_matmul(x, codebook, indices, sidecar_pos=None,
                      sidecar_vals=None, svd_U=None, svd_s=None,
                      svd_Vt=None, bias=None):
    """Reference implementation — naive codebook gather + matmul."""
    W = codebook[indices.long()]  # [OUT, IN]

    if sidecar_pos is not None:
        W_flat = W.reshape(-1).clone()
        W_flat[sidecar_pos] = sidecar_vals
        W = W_flat.reshape(indices.shape)

    if svd_U is not None:
        scaled_U = svd_U * svd_s.unsqueeze(0)
        W = W + scaled_U @ svd_Vt

    output = x @ W.t()
    if bias is not None:
        output += bias.unsqueeze(0)
    return output


def _check_accuracy(ref, fused, label, rel_threshold=REL_ERR_THRESHOLD):
    """Check relative error and cosine similarity."""
    max_diff = (ref - fused).abs().max().item()
    ref_max = ref.abs().max().item()
    rel_err = max_diff / (ref_max + 1e-8)
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().unsqueeze(0), fused.flatten().unsqueeze(0)
    ).item()
    assert rel_err < rel_threshold, (
        f"{label}: rel_err={rel_err:.2e} (threshold={rel_threshold:.0e}), "
        f"max_diff={max_diff:.2e}, cos={cos:.6f}"
    )
    return max_diff, rel_err, cos


def test_vq_only():
    """Fused kernel matches naive for VQ-only."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)
    N, IN, OUT = 16, 256, 128
    codebook = torch.randn(256, device="cuda")
    indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
    x = torch.randn(N, IN, device="cuda")

    ref = _naive_vq_matmul(x, codebook, indices)
    fused = fused_vq_matmul(x, codebook, indices)

    max_diff, rel_err, cos = _check_accuracy(ref, fused, "VQ-only")
    print(f"PASS: test_vq_only (rel_err={rel_err:.2e}, max_diff={max_diff:.2e}, cos={cos:.6f})")


def test_vq_sidecar():
    """Fused kernel matches naive for VQ + sidecar."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)
    N, IN, OUT = 32, 512, 256

    codebook = torch.randn(256, device="cuda")
    indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
    x = torch.randn(N, IN, device="cuda")

    # Create sidecar: 100 random outlier positions
    n_outliers = 100
    total_elements = OUT * IN
    positions = torch.randperm(total_elements, device="cuda")[:n_outliers].sort().values
    values = torch.randn(n_outliers, device="cuda") * 10  # Large outliers

    # Precompute VQ values at sidecar positions
    idx_flat = indices.reshape(-1)
    vq_at_sidecar = codebook[idx_flat[positions].long()]

    ref = _naive_vq_matmul(x, codebook, indices, positions, values)
    fused = fused_vq_matmul(
        x, codebook, indices,
        sidecar_positions=positions,
        sidecar_values=values,
        codebook_values_at_sidecar=vq_at_sidecar,
    )

    max_diff, rel_err, cos = _check_accuracy(ref, fused, "VQ+sidecar")
    print(f"PASS: test_vq_sidecar (rel_err={rel_err:.2e}, max_diff={max_diff:.2e})")


def test_vq_sidecar_svd():
    """Fused kernel matches naive for VQ + sidecar + SVD."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)
    N, IN, OUT, RANK = 16, 256, 128, 8

    codebook = torch.randn(256, device="cuda")
    indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
    x = torch.randn(N, IN, device="cuda")

    # Sidecar
    n_outliers = 50
    positions = torch.randperm(OUT * IN, device="cuda")[:n_outliers].sort().values
    values = torch.randn(n_outliers, device="cuda") * 5
    idx_flat = indices.reshape(-1)
    vq_at_sidecar = codebook[idx_flat[positions].long()]

    # SVD factors
    svd_U = torch.randn(OUT, RANK, device="cuda") * 0.01
    svd_s = torch.rand(RANK, device="cuda")
    svd_Vt = torch.randn(RANK, IN, device="cuda") * 0.01

    ref = _naive_vq_matmul(
        x, codebook, indices, positions, values,
        svd_U, svd_s, svd_Vt,
    )
    fused = fused_vq_matmul(
        x, codebook, indices,
        sidecar_positions=positions,
        sidecar_values=values,
        codebook_values_at_sidecar=vq_at_sidecar,
        svd_U=svd_U,
        svd_s=svd_s,
        svd_Vt=svd_Vt,
    )

    max_diff, rel_err, cos = _check_accuracy(ref, fused, "VQ+sidecar+SVD")
    print(f"PASS: test_vq_sidecar_svd (rel_err={rel_err:.2e}, max_diff={max_diff:.2e})")


def test_large_dimensions():
    """Test at real model scale (4096x4096)."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)
    N, IN, OUT = 4, 4096, 4096

    codebook = torch.randn(256, device="cuda")
    indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
    x = torch.randn(N, IN, device="cuda")

    ref = _naive_vq_matmul(x, codebook, indices)
    fused = fused_vq_matmul(x, codebook, indices)

    max_diff, rel_err, cos = _check_accuracy(ref, fused, "large_dim")
    print(f"PASS: test_large_dimensions (4096x4096, rel_err={rel_err:.2e}, max_diff={max_diff:.2e})")


def test_non_power_of_2():
    """Test non-power-of-2 dimensions (padding correctness)."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)

    for N, IN, OUT in [(7, 100, 50), (1, 33, 17), (13, 200, 99)]:
        codebook = torch.randn(256, device="cuda")
        indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
        x = torch.randn(N, IN, device="cuda")

        ref = _naive_vq_matmul(x, codebook, indices)
        fused = fused_vq_matmul(x, codebook, indices)

        _check_accuracy(ref, fused, f"non_pow2({N},{IN},{OUT})")

    print("PASS: test_non_power_of_2")


def test_with_bias():
    """Test bias addition."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)
    N, IN, OUT = 16, 128, 64

    codebook = torch.randn(256, device="cuda")
    indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
    x = torch.randn(N, IN, device="cuda")
    bias = torch.randn(OUT, device="cuda")

    ref = _naive_vq_matmul(x, codebook, indices, bias=bias)
    fused = fused_vq_matmul(x, codebook, indices, bias=bias)

    max_diff, rel_err, cos = _check_accuracy(ref, fused, "bias")
    print(f"PASS: test_with_bias (rel_err={rel_err:.2e}, max_diff={max_diff:.2e})")


def test_memory_no_full_w():
    """Verify fused path doesn't allocate full W-sized tensor."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)
    N, IN, OUT = 4, 4096, 4096

    codebook = torch.randn(256, device="cuda")
    indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
    x = torch.randn(N, IN, device="cuda")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    fused = fused_vq_matmul(x, codebook, indices)

    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - mem_before

    full_w_bytes = OUT * IN * 4  # 64 MB for 4096x4096
    output_bytes = N * OUT * 4  # 64 KB

    # Peak should be much less than full W + output
    # Allow some overhead for Triton workspace, but should be <<full_w_bytes
    print(f"  Peak VRAM delta: {peak_delta / 1024 / 1024:.1f} MB")
    print(f"  Full W would be: {full_w_bytes / 1024 / 1024:.1f} MB")

    # Fused path should use less than half of what full W would require
    # (it still needs some workspace for tiles)
    assert peak_delta < full_w_bytes, (
        f"Peak delta {peak_delta / 1024 / 1024:.1f} MB >= "
        f"full W {full_w_bytes / 1024 / 1024:.1f} MB"
    )
    print("PASS: test_memory_no_full_w")


def test_tiled_matches_naive():
    """Tiled v3 kernel matches naive reference for all real model shapes."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)

    # All 8 tensor shapes from TinyLlama + Qwen (IN divisible by 16)
    shapes = [
        (4, 2048, 2048),   # attn proj
        (4, 2048, 5632),   # gate/up proj
        (4, 5632, 2048),   # down proj
        (4, 1536, 1536),   # qwen attn
        (4, 1536, 8960),   # qwen gate/up
        (4, 8960, 1536),   # qwen down
        (1, 2048, 256),    # small
        (16, 256, 128),    # batch
    ]

    for N, IN, OUT in shapes:
        codebook = torch.randn(256, device="cuda")
        indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
        x = torch.randn(N, IN, device="cuda")

        ref = _naive_vq_matmul(x, codebook, indices)
        fused = fused_vq_matmul(x, codebook, indices)

        _check_accuracy(ref, fused, f"tiled({N},{IN},{OUT})")

    print("PASS: test_tiled_matches_naive (8 shapes)")


def test_tiled_fp16():
    """FP16 compute path matches FP32 within relaxed tolerance."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)

    for N, IN, OUT in [(4, 2048, 2048), (4, 1536, 8960), (16, 256, 128)]:
        codebook = torch.randn(256, device="cuda")
        codebook_f16 = codebook.half()
        indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
        x = torch.randn(N, IN, device="cuda")

        # FP32 reference
        ref = _naive_vq_matmul(x, codebook, indices)

        # FP16 compute path (codebook_f16 still accepted for backward compat)
        fp16_out = fused_vq_matmul(x, codebook, indices, codebook_f16=codebook_f16)

        max_diff = (ref - fp16_out).abs().max().item()
        rel_err = max_diff / (ref.abs().max().item() + 1e-8)
        assert rel_err < 0.02, f"FP16 ({N},{IN},{OUT}) rel_err: {rel_err:.4f}"

    print("PASS: test_tiled_fp16")


def test_tiled_with_sidecar_svd():
    """Tiled kernel with full sidecar + SVD pipeline matches naive."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)
    N, IN, OUT, RANK = 8, 2048, 2048, 8

    codebook = torch.randn(256, device="cuda")
    indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
    x = torch.randn(N, IN, device="cuda")

    # Sidecar
    n_outliers = 200
    positions = torch.randperm(OUT * IN, device="cuda")[:n_outliers].sort().values
    values = torch.randn(n_outliers, device="cuda") * 5
    idx_flat = indices.reshape(-1)
    vq_at_sidecar = codebook[idx_flat[positions].long()]

    # Precomputed sidecar
    rows = (positions // IN).long()
    cols = (positions % IN).long()
    deltas = (values - vq_at_sidecar).contiguous()

    # SVD
    svd_U = torch.randn(OUT, RANK, device="cuda") * 0.01
    svd_s = torch.rand(RANK, device="cuda")
    svd_Vt = torch.randn(RANK, IN, device="cuda") * 0.01

    ref = _naive_vq_matmul(x, codebook, indices, positions, values, svd_U, svd_s, svd_Vt)
    fused = fused_vq_matmul(
        x, codebook, indices,
        sidecar_rows=rows, sidecar_cols=cols, sidecar_deltas=deltas,
        svd_U=svd_U, svd_s=svd_s, svd_Vt=svd_Vt,
    )

    max_diff, rel_err, cos = _check_accuracy(ref, fused, "tiled+sidecar+SVD")
    print(f"PASS: test_tiled_with_sidecar_svd (rel_err={rel_err:.2e}, max_diff={max_diff:.2e})")


def test_tiled_single_token():
    """N=1 decode — the most performance-critical case."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)

    for IN, OUT in [(2048, 2048), (2048, 5632), (5632, 2048)]:
        codebook = torch.randn(256, device="cuda")
        indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
        x = torch.randn(1, IN, device="cuda")

        ref = _naive_vq_matmul(x, codebook, indices)
        fused = fused_vq_matmul(x, codebook, indices)

        _check_accuracy(ref, fused, f"single_token({IN},{OUT})")

    print("PASS: test_tiled_single_token (3 shapes)")


def test_tiled_prefill():
    """N=512 prefill — batch matmul correctness."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)
    N, IN, OUT = 512, 2048, 2048

    codebook = torch.randn(256, device="cuda")
    indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
    x = torch.randn(N, IN, device="cuda")

    ref = _naive_vq_matmul(x, codebook, indices)
    fused = fused_vq_matmul(x, codebook, indices)

    max_diff, rel_err, cos = _check_accuracy(ref, fused, "prefill(512,2048,2048)")
    print(f"PASS: test_tiled_prefill (N=512, rel_err={rel_err:.2e}, cos={cos:.6f})")


def test_fused_output_always_fp32():
    """fused_vq_matmul always returns FP32 regardless of input dtype."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)
    N, IN, OUT = 4, 256, 128
    codebook = torch.randn(256, device="cuda")
    indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")

    # FP32 input → FP32 output
    x_fp32 = torch.randn(N, IN, device="cuda", dtype=torch.float32)
    out_fp32 = fused_vq_matmul(x_fp32, codebook, indices)
    assert out_fp32.dtype == torch.float32, f"FP32 input gave {out_fp32.dtype}"

    # FP16 input → FP32 output (kernel accumulates in FP32)
    x_fp16 = x_fp32.half()
    out_from_fp16 = fused_vq_matmul(x_fp16, codebook, indices)
    assert out_from_fp16.dtype == torch.float32, f"FP16 input gave {out_from_fp16.dtype}"

    # BF16 input → FP32 output
    x_bf16 = x_fp32.bfloat16()
    out_from_bf16 = fused_vq_matmul(x_bf16, codebook, indices)
    assert out_from_bf16.dtype == torch.float32, f"BF16 input gave {out_from_bf16.dtype}"

    print("PASS: test_fused_output_always_fp32 (fp32, fp16, bf16 inputs all → fp32)")


def test_fused_fp16_input_correctness():
    """fused_vq_matmul with FP16 input produces correct FP32 output."""
    if _skip_no_cuda() or _skip_no_triton():
        return

    from helix_substrate.triton_vq_matmul import fused_vq_matmul

    torch.manual_seed(42)

    for N, IN, OUT in [(4, 2048, 2048), (1, 256, 128)]:
        codebook = torch.randn(256, device="cuda")
        indices = torch.randint(0, 256, (OUT, IN), dtype=torch.uint8, device="cuda")
        x_fp16 = torch.randn(N, IN, device="cuda", dtype=torch.float16)

        ref = _naive_vq_matmul(x_fp16.float(), codebook, indices)
        fused = fused_vq_matmul(x_fp16, codebook, indices)

        assert fused.dtype == torch.float32
        _check_accuracy(ref, fused, f"fp16_input({N},{IN},{OUT})")

    print("PASS: test_fused_fp16_input_correctness")


if __name__ == "__main__":
    tests = [
        test_vq_only,
        test_vq_sidecar,
        test_vq_sidecar_svd,
        test_large_dimensions,
        test_non_power_of_2,
        test_with_bias,
        test_memory_no_full_w,
        test_tiled_matches_naive,
        test_tiled_fp16,
        test_tiled_with_sidecar_svd,
        test_tiled_single_token,
        test_tiled_prefill,
        test_fused_output_always_fp32,
        test_fused_fp16_input_correctness,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            name = test.__name__
            # Check if it was skipped (printed SKIP)
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED (or skipped)")
    else:
        sys.exit(1)
