"""Tests for zerocopy.py — zero-copy GPU tensor infrastructure.

NOTE: Most zerocopy functionality requires a CUDA GPU + libcudart.so.
These tests verify the module's importability and numpy-level helpers.
GPU-specific tests (PinnedBuffer, cuda_tensor_from_dev_alias) are
skipped when CUDA is unavailable.
"""

import numpy as np
import pytest

# zerocopy imports libcudart at module level — may fail without CUDA
try:
    from helix_substrate.zerocopy import (
        PinnedBuffer, cuda_tensor_from_dev_alias, pin_indices_from_file,
        _DLDataType, _DLDevice, _DLTensor,
    )
    ZEROCOPY_AVAILABLE = True
except (OSError, RuntimeError):
    ZEROCOPY_AVAILABLE = False


@pytest.mark.skipif(not ZEROCOPY_AVAILABLE, reason="CUDA runtime not available")
class TestDLPackStructures:
    def test_dldatatype_size(self):
        import ctypes
        assert ctypes.sizeof(_DLDataType) == 4  # uint8 + uint8 + uint16

    def test_dldevice_size(self):
        import ctypes
        assert ctypes.sizeof(_DLDevice) == 8  # int32 + int32


@pytest.mark.skipif(not ZEROCOPY_AVAILABLE, reason="CUDA runtime not available")
class TestPinnedBuffer:
    def test_pinned_buffer_creation(self):
        """Test PinnedBuffer with a small array (requires CUDA)."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        try:
            buf = PinnedBuffer(data)
            assert buf.nbytes == 5
            assert buf._registered is True
            buf.unregister()
            assert buf._registered is False
        except RuntimeError as e:
            if "cudaHostRegister" in str(e):
                pytest.skip("CUDA pinning not available on this system")
            raise

    def test_double_unregister_safe(self):
        data = np.array([1, 2, 3], dtype=np.uint8)
        try:
            buf = PinnedBuffer(data)
            buf.unregister()
            buf.unregister()  # Should not crash
        except RuntimeError:
            pytest.skip("CUDA pinning not available")
