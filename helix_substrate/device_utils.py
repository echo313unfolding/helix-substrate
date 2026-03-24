"""
Device abstraction for multi-backend support (CUDA, MPS, CPU).

Thin utility module — free functions, no class hierarchy.
Enables HelixLinear and ModelManager to run on Apple Silicon (MPS)
without touching the Triton fused kernel (stays NVIDIA-only).

Work Order: WO-DEVICE-UTILS-01
"""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> str:
    """Resolve device string. 'auto' picks best available: CUDA > MPS > CPU."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize_device(device: str) -> None:
    """Synchronize device for timing accuracy."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def empty_cache(device: str) -> None:
    """Free cached memory on device."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def memory_allocated(device: str) -> float:
    """Return allocated memory in MB. 0.0 if not trackable."""
    if device.startswith("cuda"):
        return torch.cuda.memory_allocated() / 1024 / 1024
    if device == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1024 / 1024
        except AttributeError:
            return 0.0
    return 0.0


def reset_peak_memory(device: str) -> None:
    """Reset peak memory tracking."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def get_device_name(device: str) -> str:
    """Human-readable device name for receipts."""
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def device_info(device: str) -> dict:
    """Full device info dict for receipt embedding."""
    return {
        "device": device,
        "device_name": get_device_name(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available(),
        "torch_version": torch.__version__,
    }
