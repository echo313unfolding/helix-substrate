"""Tests for device_utils — multi-backend device abstraction."""

import torch
import pytest

from helix_substrate.device_utils import (
    resolve_device,
    synchronize_device,
    empty_cache,
    memory_allocated,
    reset_peak_memory,
    get_device_name,
    device_info,
)


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------

def test_resolve_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_explicit_cuda():
    assert resolve_device("cuda:0") == "cuda:0"


def test_resolve_explicit_mps():
    assert resolve_device("mps") == "mps"


def test_resolve_auto_returns_valid():
    d = resolve_device("auto")
    assert d in ("cuda:0", "mps", "cpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA available")
@pytest.mark.skipif(
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    reason="MPS available",
)
def test_resolve_auto_cpu_fallback():
    assert resolve_device("auto") == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resolve_auto_picks_cuda():
    assert resolve_device("auto") == "cuda:0"


# ---------------------------------------------------------------------------
# No-crash on CPU (these should be no-ops, not errors)
# ---------------------------------------------------------------------------

def test_synchronize_cpu():
    synchronize_device("cpu")


def test_empty_cache_cpu():
    empty_cache("cpu")


def test_reset_peak_memory_cpu():
    reset_peak_memory("cpu")


# ---------------------------------------------------------------------------
# memory_allocated
# ---------------------------------------------------------------------------

def test_memory_allocated_cpu():
    assert memory_allocated("cpu") == 0.0


def test_memory_allocated_returns_float():
    assert isinstance(memory_allocated("cpu"), float)


# ---------------------------------------------------------------------------
# get_device_name
# ---------------------------------------------------------------------------

def test_device_name_cpu():
    assert get_device_name("cpu") == "CPU"


def test_device_name_mps():
    assert get_device_name("mps") == "Apple Silicon (MPS)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_device_name_cuda():
    name = get_device_name("cuda:0")
    assert isinstance(name, str) and len(name) > 0


# ---------------------------------------------------------------------------
# device_info
# ---------------------------------------------------------------------------

def test_device_info_structure():
    info = device_info("cpu")
    assert "device" in info
    assert "device_name" in info
    assert "cuda_available" in info
    assert "mps_available" in info
    assert "torch_version" in info
    assert info["device"] == "cpu"
    assert info["device_name"] == "CPU"
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["mps_available"], bool)
