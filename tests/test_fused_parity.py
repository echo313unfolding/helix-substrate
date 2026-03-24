#!/usr/bin/env python3
"""
Fused matmul parity harness for HelixLinear.

Compares dense reference (decode_weight @ x.T) vs each forward path:
  - CPU tiled (_forward_naive via _dequant_tile)
  - GPU fused (_forward_fused via Triton kernel)

Reports: cosine similarity, max abs diff, mean abs diff.

Work Order: WO-HELIX-FUSED-MATMUL-AUDIT
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.helix_linear import HelixLinear, load_helix_linear_from_cdnav3
from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.tensor_policy import TensorPolicy, TensorClass


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).double()
    b_flat = b.reshape(-1).double()
    return float(torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item())


def _metrics(output: torch.Tensor, reference: torch.Tensor) -> dict:
    diff = (output - reference).abs()
    return {
        "cosine": _cosine(output, reference),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
    }


def _make_helix(rows, cols, sidecar, svd_rank, seed=42):
    """Create a HelixLinear from a random tensor via CDNA v3 roundtrip."""
    rng = np.random.RandomState(seed)
    tensor = rng.randn(rows, cols).astype(np.float32)

    storage = "codebook+sidecar" if sidecar else "codebook"
    policy = TensorPolicy(
        tensor_class=TensorClass.ATTENTION_QK if sidecar else TensorClass.FFN,
        storage_mode=storage,
        n_clusters=256,
        sidecar_enabled=sidecar,
        percentile=99.0 if sidecar else 100.0,
        svd_residual_rank=svd_rank,
    )

    tmpdir = tempfile.mkdtemp()
    tmpdir = Path(tmpdir)
    writer = CDNAv3Writer(tmpdir)
    writer.write_tensor(tensor, "test.weight", policy=policy)
    safe = "test_weight"
    tensor_dir = tmpdir / f"{safe}.cdnav3"
    return load_helix_linear_from_cdnav3(tensor_dir)


# ============================================================================
# Test: _dequant_tile matches decode_weight
# ============================================================================

def test_dequant_tile_matches_decode():
    """_dequant_tile over full range equals decode_weight (they share the path)."""
    configs = [
        ("vq_only",     128, 64, False, 0),
        ("vq_sidecar",  128, 64, True,  0),
        ("vq_svd",      128, 64, True,  8),
        ("large",       512, 256, True,  4),
    ]

    for name, rows, cols, sidecar, svd_rank in configs:
        helix = _make_helix(rows, cols, sidecar, svd_rank)
        W_decode = helix.decode_weight()
        W_tile = helix._dequant_tile(0, rows)
        assert torch.equal(W_decode, W_tile), f"{name}: decode_weight != _dequant_tile(0, all)"
        print(f"  {name}: decode_weight == _dequant_tile (exact)")

    print("PASS: test_dequant_tile_matches_decode")


# ============================================================================
# Test: _dequant_tile subtiles are consistent
# ============================================================================

def test_dequant_tile_subtiles():
    """Concatenating subtiles equals one big tile."""
    helix = _make_helix(256, 128, sidecar=True, svd_rank=8)

    W_full = helix._dequant_tile(0, 256)
    tiles = []
    for start in range(0, 256, 64):
        tiles.append(helix._dequant_tile(start, start + 64))
    W_tiled = torch.cat(tiles, dim=0)

    assert torch.allclose(W_full, W_tiled, atol=1e-6), (
        f"Subtile max diff: {(W_full - W_tiled).abs().max().item()}"
    )
    print("PASS: test_dequant_tile_subtiles")


# ============================================================================
# Test: CPU tiled forward matches dense reference
# ============================================================================

def test_cpu_tiled_vs_dense():
    """CPU _forward_naive matches decode_weight @ x.T."""
    configs = [
        ("vq_only",     128, 64,  False, 0),
        ("vq_sidecar",  128, 64,  True,  0),
        ("vq_svd",      128, 64,  True,  8),
        ("large",       512, 256, True,  4),
    ]

    results = []
    for name, rows, cols, sidecar, svd_rank in configs:
        helix = _make_helix(rows, cols, sidecar, svd_rank)
        x = torch.randn(8, cols)

        # Reference: decode_weight @ x.T
        W = helix.decode_weight()
        ref = x @ W.t()

        # Tiled forward
        output = helix._forward_naive(x)

        m = _metrics(output, ref)
        results.append({"config": name, "path": "cpu_tiled", **m})
        print(f"  {name} (cpu_tiled): cos={m['cosine']:.8f}  max={m['max_abs_diff']:.2e}  mean={m['mean_abs_diff']:.2e}")

        assert m["cosine"] > 0.999999, f"{name}: cosine {m['cosine']} too low"
        assert m["max_abs_diff"] < 1e-4, f"{name}: max_abs_diff {m['max_abs_diff']} too high"

    print("PASS: test_cpu_tiled_vs_dense")
    return results


# ============================================================================
# Test: GPU fused forward matches dense reference
# ============================================================================

def test_gpu_fused_vs_dense():
    """GPU _forward_fused matches decode_weight @ x.T."""
    if not torch.cuda.is_available():
        print("SKIP: test_gpu_fused_vs_dense (no CUDA)")
        return []

    try:
        from helix_substrate.triton_vq_matmul import is_available
        if not is_available():
            print("SKIP: test_gpu_fused_vs_dense (Triton not available)")
            return []
    except ImportError:
        print("SKIP: test_gpu_fused_vs_dense (Triton not installed)")
        return []

    configs = [
        ("vq_only",     128, 64,  False, 0),
        ("vq_sidecar",  128, 64,  True,  0),
        ("vq_svd",      128, 64,  True,  8),
        ("large",       512, 256, True,  4),
    ]

    results = []
    for name, rows, cols, sidecar, svd_rank in configs:
        helix = _make_helix(rows, cols, sidecar, svd_rank).cuda()
        x = torch.randn(8, cols, device="cuda")

        # Reference: decode_weight @ x.T (computed on GPU for precision match)
        W = helix.decode_weight()
        ref = x @ W.t()

        # Fused forward
        output = helix._forward_fused(x)

        m = _metrics(output, ref)
        results.append({"config": name, "path": "gpu_fused", **m})
        print(f"  {name} (gpu_fused): cos={m['cosine']:.8f}  max={m['max_abs_diff']:.2e}  mean={m['mean_abs_diff']:.2e}")

        # Fused kernel uses outer-product accumulation (not cuBLAS) — accumulation
        # order differs, so expect ~1e-2 max diff at 512x256 scale.
        # Cosine is the reliable metric here.
        assert m["cosine"] > 0.99999, f"{name}: cosine {m['cosine']} too low"
        assert m["max_abs_diff"] < 0.05, f"{name}: max_abs_diff {m['max_abs_diff']} too high"

    print("PASS: test_gpu_fused_vs_dense")
    return results


# ============================================================================
# Test: GPU fused matches CPU tiled (cross-path)
# ============================================================================

def test_gpu_vs_cpu_cross():
    """GPU fused and CPU tiled produce similar outputs on same input."""
    if not torch.cuda.is_available():
        print("SKIP: test_gpu_vs_cpu_cross (no CUDA)")
        return []

    try:
        from helix_substrate.triton_vq_matmul import is_available
        if not is_available():
            print("SKIP: test_gpu_vs_cpu_cross (Triton not available)")
            return []
    except ImportError:
        print("SKIP: test_gpu_vs_cpu_cross (Triton not installed)")
        return []

    helix_cpu = _make_helix(256, 128, sidecar=True, svd_rank=8)
    helix_gpu = _make_helix(256, 128, sidecar=True, svd_rank=8).cuda()
    x_cpu = torch.randn(8, 128)
    x_gpu = x_cpu.cuda()

    out_cpu = helix_cpu._forward_naive(x_cpu)
    out_gpu = helix_gpu._forward_fused(x_gpu).cpu()

    m = _metrics(out_gpu, out_cpu)
    print(f"  gpu_fused vs cpu_tiled: cos={m['cosine']:.8f}  max={m['max_abs_diff']:.2e}")

    assert m["cosine"] > 0.9999, f"Cross-path cosine {m['cosine']} too low"
    print("PASS: test_gpu_vs_cpu_cross")
    return [{"config": "cross_path", "path": "gpu_vs_cpu", **m}]


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    t0 = time.time()

    tests = [
        test_dequant_tile_matches_decode,
        test_dequant_tile_subtiles,
        test_cpu_tiled_vs_dense,
        test_gpu_fused_vs_dense,
        test_gpu_vs_cpu_cross,
    ]

    passed = 0
    failed = 0
    all_results = []

    for test in tests:
        try:
            result = test()
            if isinstance(result, list):
                all_results.extend(result)
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    wall = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Fused Parity Harness: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"Wall time: {wall:.1f}s")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        sys.exit(1)
