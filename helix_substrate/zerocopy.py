"""
Zero-copy GPU tensor infrastructure — pinned host memory + DLPack bridge.

Creates PyTorch CUDA tensors backed by cudaHostRegister'd host RAM.
GPU reads via PCIe BAR1 — zero VRAM allocation for the tensor data.

Primary use: HelixLinear indices (uint8, ~97% of compressed model size)
stay in host memory while GPU reads them during Triton kernel execution.

Note: cudaHostRegister on mmap'd memory fails on Turing/CUDA 12.0.
This module uses heap allocation (np.fromfile) + cudaHostRegister instead.
"""

import ctypes
from pathlib import Path

import numpy as np
import torch


# ── CUDA runtime ──

try:
    _cuda_rt = ctypes.CDLL('libcudart.so')
except OSError:
    _cuda_rt = ctypes.CDLL('libcudart.so.12')
_cuda_rt.cudaGetErrorString.restype = ctypes.c_char_p


# ── PinnedBuffer: page-aligned heap + cudaHostRegister ──

class PinnedBuffer:
    """Page-aligned heap buffer registered with CUDA for zero-copy GPU access.

    Holds data in host RAM. GPU reads via BAR1 using the dev_alias pointer.
    """

    def __init__(self, data: np.ndarray):
        page = 4096
        nbytes = data.nbytes

        # Allocate page-aligned buffer
        self._raw = np.zeros(nbytes + page, dtype=np.uint8)
        base = self._raw.ctypes.data
        self._aligned_offset = ((base + page - 1) // page) * page - base
        self._nbytes = nbytes

        # Copy data into aligned region
        self._raw[self._aligned_offset:self._aligned_offset + nbytes] = data.view(np.uint8).ravel()
        self.host_ptr = self._raw.ctypes.data + self._aligned_offset

        # Register with CUDA (Portable | Mapped)
        rc = _cuda_rt.cudaHostRegister(
            ctypes.c_void_p(self.host_ptr),
            ctypes.c_size_t(nbytes),
            ctypes.c_uint(3),
        )
        if rc != 0:
            err = _cuda_rt.cudaGetErrorString(rc).decode()
            raise RuntimeError(f"cudaHostRegister failed: {err} (code {rc})")

        # Get GPU-visible pointer
        self._dev_ptr = ctypes.c_void_p()
        rc2 = _cuda_rt.cudaHostGetDevicePointer(
            ctypes.byref(self._dev_ptr),
            ctypes.c_void_p(self.host_ptr),
            ctypes.c_uint(0),
        )
        if rc2 != 0:
            _cuda_rt.cudaHostUnregister(ctypes.c_void_p(self.host_ptr))
            err = _cuda_rt.cudaGetErrorString(rc2).decode()
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")

        self.dev_alias = self._dev_ptr.value
        self._registered = True

    @property
    def nbytes(self):
        return self._nbytes

    def unregister(self):
        if self._registered:
            _cuda_rt.cudaHostUnregister(ctypes.c_void_p(self.host_ptr))
            self._registered = False

    def __del__(self):
        self.unregister()


# ── DLPack bridge: CUDA tensor from dev_alias pointer ──

class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]

class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]

class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

class _DLManagedTensor(ctypes.Structure):
    pass

_DELETER_TYPE = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))

_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DELETER_TYPE),
]

_dlpack_keepalive = []

@_DELETER_TYPE
def _noop_deleter(ptr):
    pass

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def cuda_tensor_from_dev_alias(dev_alias, shape, dtype_code, dtype_bits):
    """Create a PyTorch CUDA tensor from a cudaHostGetDevicePointer dev_alias.

    GPU reads this data via PCIe BAR1 — zero VRAM allocation.

    Args:
        dev_alias: GPU-visible pointer from cudaHostGetDevicePointer.
        shape: Tensor shape tuple.
        dtype_code: DLPack dtype code (0=int, 1=uint, 2=float).
        dtype_bits: Bits per element (8=uint8, 32=float32).

    Returns:
        torch.Tensor on CUDA, backed by host memory.
    """
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)
    strides_arr = (ctypes.c_int64 * ndim)()
    stride = 1
    for i in range(ndim - 1, -1, -1):
        strides_arr[i] = stride
        stride *= shape[i]

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(dev_alias)
    managed.dl_tensor.device.device_type = 2  # kDLCUDA
    managed.dl_tensor.device.device_id = 0
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype.code = dtype_code
    managed.dl_tensor.dtype.bits = dtype_bits
    managed.dl_tensor.dtype.lanes = 1
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = strides_arr
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    managed.deleter = _noop_deleter

    refs = (managed, shape_arr, strides_arr)
    _dlpack_keepalive.append(refs)

    capsule = _PyCapsule_New(ctypes.addressof(managed), b"dltensor", None)
    return torch.from_dlpack(capsule)


def pin_indices_from_file(indices_path, shape):
    """Load indices.bin, pin in host memory, return (CUDA tensor, PinnedBuffer).

    Args:
        indices_path: Path to raw uint8 indices file.
        shape: (rows, cols) tuple.

    Returns:
        (cuda_tensor, pinned_buffer) — keep pinned_buffer alive!
    """
    raw = np.fromfile(str(indices_path), dtype=np.uint8)
    pinned = PinnedBuffer(raw)
    tensor = cuda_tensor_from_dev_alias(
        pinned.dev_alias, shape,
        dtype_code=1, dtype_bits=8,
    )
    return tensor, pinned
