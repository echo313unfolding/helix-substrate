"""
Forced-failure E2E: prove the system fails closed when Triton is unavailable.

Scenarios tested:
    1. CUDA input + Triton unavailable + StrictBenchmarkMode → RuntimeError (fails closed)
    2. CUDA input + Triton unavailable + no strict mode → RuntimeWarning + naive path
    3. dispatch_metadata() correctly reports naive + fallback reason
    4. build_dispatch_receipt() captures fallback_reason when naive
    5. StrictBenchmarkMode restores original forward on exception

This is the "break glass" test: it proves that silent fallback to 41x-slower
naive path CANNOT happen undetected in any configuration.

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
def test_strict_mode_fails_closed():
    """CUDA input + Triton unavailable + StrictBenchmarkMode → RuntimeError.

    This is the critical test: proves silent fallback CANNOT happen under
    strict mode. The system must raise, not silently degrade.
    """
    from helix_substrate.basin_runtime import StrictBenchmarkMode

    helix = _make_cuda_helix()
    x = torch.randn(1, 16, device="cuda")

    # Force _use_fused to return False (simulates Triton unavailable)
    with patch.object(
        type(helix), '_use_fused',
        new_callable=lambda: property(lambda self: False)
    ):
        with pytest.raises(RuntimeError, match="StrictBenchmarkMode"):
            with StrictBenchmarkMode():
                helix(x)

    # Verify dispatch path was set to naive
    assert helix._last_dispatch_path == "naive"


# ---------------------------------------------------------------------------
# Test 2: Warning emitted without strict mode
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_naive_emits_warning():
    """CUDA input + Triton unavailable + no strict mode → RuntimeWarning.

    Without strict mode, the system MUST warn loudly. The warning contains
    the tensor name and '41x slower' to make it impossible to miss.
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

    # Output must still be produced (graceful degradation)
    assert output.shape == (1, 32)

    # Warning must have fired
    assert len(caught) >= 1, "No RuntimeWarning emitted on CUDA→naive fallback"
    warn_messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("41x slower" in msg for msg in warn_messages), (
        f"Warning missing '41x slower': {warn_messages}"
    )
    assert any("test_fail_closed" in msg for msg in warn_messages), (
        f"Warning missing tensor name: {warn_messages}"
    )

    # Dispatch path must be naive
    assert helix._last_dispatch_path == "naive"


# ---------------------------------------------------------------------------
# Test 3: dispatch_metadata reports correctly on fallback
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dispatch_metadata_on_fallback():
    """dispatch_metadata() correctly reports naive path when Triton forced off."""
    helix = _make_cuda_helix()
    x = torch.randn(1, 16, device="cuda")

    with patch.object(
        type(helix), '_use_fused',
        new_callable=lambda: property(lambda self: False)
    ):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            helix(x)

    dm = helix.dispatch_metadata()
    assert dm["dispatch_path"] == "naive"
    assert dm["is_cuda"] is True
    # triton_available reflects live state (True on this box), not the
    # patched state during forward(). The key signal is dispatch_path="naive".
    assert dm["compute_dtype"] == "torch.float32"
    # kernel_metadata should be None (no fused call happened)
    assert dm["kernel_metadata"] is None


# ---------------------------------------------------------------------------
# Test 4: dispatch_receipt captures fallback
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dispatch_receipt_captures_fallback():
    """build_dispatch_receipt correctly records naive_count and fallback_reason."""
    import helix_substrate.triton_vq_matmul as tvq
    from helix_substrate.basin_runtime import build_dispatch_receipt

    helix = _make_cuda_helix()
    x = torch.randn(1, 16, device="cuda")

    # Build a minimal model-like container
    class FakeModel(torch.nn.Module):
        def __init__(self, helix_module):
            super().__init__()
            self.layer = helix_module

    fake_model = FakeModel(helix)

    # Force naive path
    with patch.object(
        type(helix), '_use_fused',
        new_callable=lambda: property(lambda self: False)
    ):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            helix(x)

    # Build receipt — patch is_available for fallback_reason detection
    with patch.object(tvq, 'is_available', return_value=False):
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
    assert ds["naive_count"] == 1
    assert ds["fused_count"] == 0
    assert ds["all_fused"] is False
    assert receipt["fallback_reason"] is not None
    assert "Triton" in receipt["fallback_reason"] or "not available" in receipt["fallback_reason"]


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
# Test 7: GPU fused path — no warning, dispatch_path="fused"
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_fused_no_warning():
    """GPU fused path should NOT emit a warning — this is the happy path."""
    helix = _make_cuda_helix()
    x = torch.randn(1, 16, device="cuda")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = helix(x)

    assert output.shape == (1, 32)
    assert helix._last_dispatch_path == "fused"
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0, (
        f"GPU fused should not warn, got: {[str(w.message) for w in runtime_warnings]}"
    )
