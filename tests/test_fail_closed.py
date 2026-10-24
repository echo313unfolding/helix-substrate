"""
Forced-failure E2E: prove the system fails closed when Triton is unavailable.

Scenarios tested:
    1. CUDA input + Triton unavailable → buffered fallback (not naive)
       StrictBenchmarkMode accepts buffered and does not raise.
    2. CUDA input + Triton unavailable + no strict mode → buffered path, no warning
       (warning was for naive fallback; buffered is acceptable CUDA execution)
    3. dispatch_metadata() correctly reports buffered path
    4. build_dispatch_receipt() records dispatch path
    5. StrictBenchmarkMode restores original forward on exception

The naive path (41x slower Python loop) is structurally unreachable for CUDA
inputs. The dispatch chain routes CUDA → buffered (cuBLAS) as the fallback,
which is near-dense speed. StrictBenchmarkMode only prohibits naive on CUDA.

Work Order: WO-INSTRUMENTATION-HARDEN-01
"""

import warnings
from unittest.mock import patch, PropertyMock

import pytest
import torch

from helix_substrate.helix_linear import HelixLinear


def _make_cuda_helix():
    """Create a minimal HelixLinear on CUDA."""
    codebook = torch.randn(256, device="cuda")
    indices = torch.randint(0, 256, (32, 16), dtype=torch.uint8, device="cuda")
    helix = HelixLinear(16, 32, codebook, indices, tensor_name="test_fail_closed")
    return helix.cuda()


# ---------------------------------------------------------------------------
# Test 1: StrictBenchmarkMode fails closed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_strict_mode_allows_buffered():
    """CUDA input + Triton unavailable → buffered path. StrictBenchmarkMode accepts this.

    The dispatch chain routes CUDA inputs to buffered (cuBLAS matmul) when fused
    Triton is unavailable. This is acceptable CUDA execution — not the 41x-slower
    naive path. StrictBenchmarkMode only prohibits naive on CUDA.
    """
    from helix_substrate.basin_runtime import StrictBenchmarkMode

    helix = _make_cuda_helix()
    x = torch.randn(1, 16, device="cuda")

    # Force _use_fused to return False (simulates Triton unavailable)
    with patch.object(
        type(helix), '_use_fused',
        new_callable=lambda: property(lambda self: False)
    ):
        with StrictBenchmarkMode():
            out = helix(x)  # Should NOT raise — buffered is acceptable

    assert out.shape == (1, 32)
    assert helix._last_dispatch_path == "buffered"


# ---------------------------------------------------------------------------
# Test 2: Warning emitted without strict mode
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_buffered_no_warning():
    """CUDA input + Triton unavailable + no strict mode → buffered, no naive warning.

    The buffered path (cuBLAS matmul) is acceptable CUDA execution. The naive
    warning was for the 41x-slower Python loop which is no longer reachable on CUDA.
    """
    helix = _make_cuda_helix()
    x = torch.randn(1, 16, device="cuda")

    with patch.object(
        type(helix), '_use_fused',
        new_callable=lambda: property(lambda self: False)
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = helix(x)

    # Output must still be produced
    assert output.shape == (1, 32)

    # No naive-path warning should fire — we're on buffered, not naive
    warn_messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    naive_warnings = [msg for msg in warn_messages if "41x slower" in msg]
    assert len(naive_warnings) == 0, (
        f"Unexpected naive-path warning on buffered dispatch: {naive_warnings}"
    )

    # Dispatch path must be buffered
    assert helix._last_dispatch_path == "buffered"


# ---------------------------------------------------------------------------
# Test 3: dispatch_metadata reports correctly on fallback
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dispatch_metadata_on_buffered_fallback():
    """dispatch_metadata() correctly reports buffered path when Triton forced off."""
    helix = _make_cuda_helix()
    x = torch.randn(1, 16, device="cuda")

    with patch.object(
        type(helix), '_use_fused',
        new_callable=lambda: property(lambda self: False)
    ):
        helix(x)

    dm = helix.dispatch_metadata()
    assert dm["dispatch_path"] == "buffered"
    assert dm["is_cuda"] is True
    assert dm["compute_dtype"] == "torch.float32"


# ---------------------------------------------------------------------------
# Test 4: dispatch_receipt captures fallback
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dispatch_receipt_captures_buffered():
    """build_dispatch_receipt records buffered dispatch when fused unavailable."""
    from helix_substrate.basin_runtime import build_dispatch_receipt

    helix = _make_cuda_helix()
    x = torch.randn(1, 16, device="cuda")

    # Build a minimal model-like container
    class FakeModel(torch.nn.Module):
        def __init__(self, helix_module):
            super().__init__()
            self.layer = helix_module

    fake_model = FakeModel(helix)

    # Force buffered fallback (fused unavailable)
    with patch.object(
        type(helix), '_use_fused',
        new_callable=lambda: property(lambda self: False)
    ):
        helix(x)

    assert helix._last_dispatch_path == "buffered"

    receipt = build_dispatch_receipt(
        fake_model,
        timing={
            "prompt_tokens": 5, "generated_tokens": 1,
            "prefill_ms": 10.0, "decode_ms": 50.0,
            "total_ms": 60.0, "tok_s": 20.0, "prefill_tok_s": 500.0,
        },
        prompt_info={"prompt": "test", "max_tokens": 1},
        load_metadata={"manifest_model": "test", "manifest_sha256": "abc"},
    )

    ds = receipt["dispatch_summary"]
    # Buffered is not fused, not naive — it's the cuBLAS materialized path
    assert ds["naive_count"] == 0
    assert ds["all_fused"] is False


# ---------------------------------------------------------------------------
# Test 5: StrictBenchmarkMode restores forward on exception
# ---------------------------------------------------------------------------

def test_strict_mode_restores_forward():
    """Original forward is restored even when StrictBenchmarkMode raises."""
    from helix_substrate.basin_runtime import StrictBenchmarkMode

    original_forward = HelixLinear.forward

    try:
        with StrictBenchmarkMode():
            # Verify forward was replaced
            assert HelixLinear.forward is not original_forward
            raise ValueError("simulated error")
    except ValueError:
        pass

    # Forward must be restored
    assert HelixLinear.forward is original_forward


# ---------------------------------------------------------------------------
# Test 6: CPU input with naive path — no warning (expected behavior)
# ---------------------------------------------------------------------------

def test_cpu_naive_no_warning():
    """CPU input on naive path should NOT emit a warning — this is expected."""
    codebook = torch.randn(256)
    indices = torch.randint(0, 256, (32, 16), dtype=torch.uint8)
    helix = HelixLinear(16, 32, codebook, indices, tensor_name="test_cpu")

    x = torch.randn(1, 16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = helix(x)

    assert output.shape == (1, 32)
    assert helix._last_dispatch_path == "naive"
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0, (
        f"CPU naive should not warn, got: {[str(w.message) for w in runtime_warnings]}"
    )


# ---------------------------------------------------------------------------
# Test 7: GPU default path — no warning, dispatch_path="buffered"
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_default_no_warning():
    """GPU default path (buffered/cuBLAS) should NOT emit a warning."""
    helix = _make_cuda_helix()
    x = torch.randn(1, 16, device="cuda")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = helix(x)

    assert output.shape == (1, 32)
    # Default _forward_mode is "buffered" — cuBLAS matmul with materialized weights
    assert helix._last_dispatch_path == "buffered"
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0, (
        f"GPU buffered should not warn, got: {[str(w.message) for w in runtime_warnings]}"
    )
