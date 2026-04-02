"""
HelixLinear — drop-in nn.Linear replacement backed by CDNA v3 compressed storage.

The compressed form IS the executable. Weights are never fully materialized
in persistent GPU memory. Storage is VQ-compressed (codebook + uint8 indices),
with optional sparse sidecar corrections and SVD residual factors.

Memory footprint: ~1/4 of nn.Linear (uint8 indices vs float32 weights).
Compute: codebook gather + matmul per forward (slightly more compute, much less memory).

Usage:
    from helix_substrate.helix_linear import HelixLinear, swap_to_helix, load_cdna_factors

    # Load factors from CDNA v3 directory
    factors = load_cdna_factors("/path/to/cdna_output/")

    # Replace all nn.Linear modules with HelixLinear
    model = swap_to_helix(model, factors)
    model = model.cuda().eval()

Work Order: WO-HELIX-LINEAR-01
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class HelixLinearStats:
    """Compression statistics for a HelixLinear layer."""
    tensor_name: str
    in_features: int
    out_features: int
    rank: int  # SVD rank (0 if VQ-only)
    n_outliers: int
    compression_ratio: float
    cosine_fidelity: float  # best available cosine from stats.json
    storage_bytes: int  # total bytes on disk
    dense_bytes: int  # what nn.Linear would use (out * in * 4)


class HelixLinear(nn.Module):
    """
    Drop-in nn.Linear replacement that stores weights in CDNA v3 format.

    Internal representation:
        W ≈ codebook[indices] + sidecar_deltas + (U * s) @ Vt

    Where:
        codebook: [256] float32 cluster centers

    Instrumentation:
        _last_dispatch_path: "fused" | "naive" | "unknown" — set on every forward() call.
        dispatch_metadata(): returns dict with dispatch path, device, kernel info.
        Emits RuntimeWarning if CUDA input falls to naive path (Triton unavailable).
        indices:  [out, in] uint8 cluster assignments  (4x smaller than float32)
        sidecar:  sparse outlier corrections (precomputed rows/cols/deltas)
        U, s, Vt: optional SVD residual factors

    Forward paths:
        GPU (Triton fused):  Codebook gathered in registers, W never in global memory.
        CPU (tiled naive):   256-row tiles via _dequant_tile(), peak ~8.75 MB temporary.

    Memory: codebook(1KB) + indices(out*in bytes) + sidecar(~few KB) + SVD(small)
    vs nn.Linear: out*in*4 bytes

    Full W is never materialized during forward(). Only bounded tiles exist transiently.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        codebook: torch.Tensor,
        indices: torch.Tensor,
        sidecar_positions: Optional[torch.Tensor] = None,
        sidecar_values: Optional[torch.Tensor] = None,
        svd_U: Optional[torch.Tensor] = None,
        svd_s: Optional[torch.Tensor] = None,
        svd_Vt: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        tensor_name: str = "",
        compute_dtype: torch.dtype = torch.float32,
        channel_scales: Optional[torch.Tensor] = None,
        vector_dim: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tensor_name = tensor_name
        self.compute_dtype = compute_dtype
        # Multi-dimensional VQ: 1=scalar, 2=2D pairs, 4=4D quads.
        # When d>1, codebook is [k, d] and indices are [out, in/d].
        self.vector_dim = vector_dim
        self._last_dispatch_path: str = "unknown"
        # Cached dispatch decision — frozen at init, updated on .to()/.cuda()
        # Avoids per-forward property evaluation (154 calls/token).
        self._fused_available: bool = self._check_fused_available()
        # Cell-driven SVD skip: when True, SVD correction is bypassed in forward().
        # Set externally by codec_bridge.apply_cell_signal_to_model() when a cell's
        # codebook health indicates SVD is unnecessary (saving compute).
        # Default False = always apply SVD if factors are present.
        self._cell_skip_svd: bool = False
        # Kurtosis gate: runtime SVD gating for Class 2 modules.
        # Attached via attach_kurtosis_gate() or kurtosis_gate.attach_kurtosis_gates().
        # When present, forward() computes input kurtosis and lets the gate
        # decide _cell_skip_svd before the main computation. Zero overhead
        # when None (the common case for 147 of 154 modules).
        self._kurtosis_gate = None
        # Sidecar phase: "fused" | "scatter" | None (auto). Frozen at request start.
        self._sidecar_phase: Optional[str] = None
        # CUDA graph mode: when True, forward() avoids GPU→CPU syncs (.item()).
        # Kurtosis gate returns cached decision; evaluation happens out-of-band
        # via pre_step_kurtosis(). Toggle with set_cuda_graph_mode().
        self._cuda_graph_mode: bool = False

        # Weight stub for compatibility with model code that accesses
        # .weight.device (e.g. Mamba2 fast-path check, tie_weights) and
        # transformers' get_parameter/get_parameter_or_buffer() which checks
        # _parameters/_buffers dicts.  Registered as a non-trainable Parameter
        # so that both get_parameter() and get_parameter_or_buffer() succeed
        # (transformers 5.x's mark_tied_weights_as_initialized requires this).
        # The actual computation goes through forward(), never through .weight.
        self.weight = nn.Parameter(torch.empty(0, dtype=compute_dtype), requires_grad=False)

        # 12-bit packed index storage: when True, indices buffer is uint8 1D
        # containing pairs of 12-bit values packed into 3-byte groups.
        # Saves 25% VRAM on index storage for k>256 (6 bits/weight vs 8).
        self.index_packing: Optional[str] = None
        # Original index matrix dimensions for unpacking
        self._idx_rows: int = out_features
        self._idx_cols: int = in_features // vector_dim if vector_dim > 1 else in_features
        # Buffered forward: materialize W into reusable buffer, then cuBLAS.
        # "buffered" (default on CUDA) | "fused" (Triton, VRAM-constrained) | "naive" (CPU)
        self._forward_mode: str = "buffered"
        self._weight_buffer: Optional[torch.Tensor] = None

        # VQ components (read-only buffers, not parameters)
        self.register_buffer("codebook", codebook.contiguous())  # [k] (256 or 512+)
        # Store indices as uint8 (k<=256) or uint16 (k>256) to save memory.
        # Convert to long only during forward() for the gather operation.
        if indices.dtype not in (torch.uint8, torch.int16):
            # For k<=256 use uint8, for k>256 use int16 (PyTorch lacks uint16)
            if codebook.shape[0] > 256:
                indices = indices.to(torch.int16)
            else:
                indices = indices.to(torch.uint8)
        self.register_buffer("indices", indices.contiguous())  # [out, in] uint8 or int16

        # FP16 codebook for mixed-precision compute path
        if compute_dtype == torch.float16:
            self.register_buffer("codebook_f16", codebook.half().contiguous())
        else:
            self.register_buffer("codebook_f16", None)

        # Sidecar (sparse outlier corrections)
        if sidecar_positions is not None and sidecar_values is not None:
            self.register_buffer("sidecar_positions", sidecar_positions.contiguous())
            self.register_buffer("sidecar_values", sidecar_values.contiguous())
            # Precompute VQ values at sidecar positions for fused kernel
            # (avoids re-gather during forward)
            if vector_dim > 1:
                rows = sidecar_positions // in_features
                cols = sidecar_positions % in_features
                idx_2d = indices.long()[rows, cols // vector_dim]
                vq_at_sidecar = codebook[idx_2d, cols % vector_dim]
            else:
                idx_flat = indices.reshape(-1).long()
                vq_at_sidecar = codebook[idx_flat[sidecar_positions]]
            self.register_buffer("_sidecar_vq_vals", vq_at_sidecar.contiguous())
            # Precompute row/col/delta for chunked naive + fused paths (Phase 4)
            self.register_buffer("_sidecar_rows", (sidecar_positions // in_features).long())
            self.register_buffer("_sidecar_cols", (sidecar_positions % in_features).long())
            self.register_buffer("_sidecar_deltas",
                                 (sidecar_values - vq_at_sidecar).contiguous())
        else:
            self.register_buffer("sidecar_positions", None)
            self.register_buffer("sidecar_values", None)
            self.register_buffer("_sidecar_vq_vals", None)
            self.register_buffer("_sidecar_rows", None)
            self.register_buffer("_sidecar_cols", None)
            self.register_buffer("_sidecar_deltas", None)

        # SVD residual factors
        self.has_svd = svd_U is not None
        if self.has_svd:
            self.register_buffer("svd_U", svd_U.contiguous())  # [out, rank]
            self.register_buffer("svd_s", svd_s.contiguous())  # [rank]
            self.register_buffer("svd_Vt", svd_Vt.contiguous())  # [rank, in]
            self.rank = svd_U.shape[1]
        else:
            self.register_buffer("svd_U", None)
            self.register_buffer("svd_s", None)
            self.register_buffer("svd_Vt", None)
            self.rank = 0

        # Bias
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.register_buffer("bias", None)

        # AWQ-style channel scales: when present, input is pre-scaled by 1/scales
        # before matmul. Codebook/sidecar/SVD are in scaled space. The division
        # undoes the scaling applied at compression time.
        # Math: y = W_original @ x = W_scaled @ (x / scales)
        self.has_channel_scales = channel_scales is not None
        if self.has_channel_scales:
            self.register_buffer("channel_scales", channel_scales.contiguous())
        else:
            self.register_buffer("channel_scales", None)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Override to handle buffer size changes during HF safetensors loading.

        When loading from a converted safetensors file, optional buffers
        (sidecar_positions, svd_U, etc.) may change from empty [0]-shape
        placeholders to real data. Standard load_state_dict rejects size
        mismatches even with assign=True. We intercept and replace directly.
        """
        # Buffers that may change size during loading
        _resizable = {
            "codebook", "indices",
            "sidecar_positions", "sidecar_values",
            "svd_U", "svd_s", "svd_Vt",
            "bias", "channel_scales", "weight",
            # Derived buffers won't be in the file, but handle if present
            "codebook_f16", "_sidecar_vq_vals",
            "_sidecar_rows", "_sidecar_cols", "_sidecar_deltas",
        }

        for local_name in list(_resizable):
            key = prefix + local_name
            if key in state_dict and hasattr(self, local_name):
                new_tensor = state_dict[key]
                # Replace the buffer directly, bypassing size check
                self.register_buffer(local_name, new_tensor)
                # Remove from state_dict so parent doesn't try to load it again
                del state_dict[key]

        # Let parent handle any remaining keys (exact tensors, etc.)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _apply(self, fn, recurse=True):
        """Override nn.Module._apply to refresh dispatch cache after .to()/.cuda()/.cpu()."""
        result = super()._apply(fn, recurse)
        self._refresh_dispatch_cache()
        # Invalidate cached codebook dtype casts on device move
        for attr in list(vars(self)):
            if attr.startswith("_cb_"):
                delattr(self, attr)
        return result

    def set_compute_dtype(self, dtype: torch.dtype) -> None:
        """Set compute dtype, creating FP16 codebook buffer if needed."""
        self.compute_dtype = dtype
        if dtype == torch.float16 and self.codebook is not None:
            self.register_buffer("codebook_f16", self.codebook.half().contiguous())
        else:
            self.register_buffer("codebook_f16", None)

    def __setattr__(self, name, value):
        """Override to intercept 'weight' assignments from tie_weights().

        HF's tie_weights() does ``self.lm_head.weight = self.embed_tokens.weight``.
        We accept any Tensor and store it as a non-trainable Parameter so both
        get_parameter() and get_parameter_or_buffer() work across transformers
        versions (4.x uses buffers, 5.x requires parameters).
        """
        if name == "weight":
            # Handle legacy buffer-based weight (transformers 4.x models)
            if hasattr(self, "_buffers") and "weight" in self._buffers:
                self._buffers["weight"] = value if isinstance(value, torch.Tensor) else torch.empty(0)
                return
            # Handle parameter-based weight (transformers 5.x compat)
            if hasattr(self, "_parameters") and "weight" in self._parameters:
                if isinstance(value, nn.Parameter):
                    self._parameters["weight"] = value
                elif isinstance(value, torch.Tensor):
                    self._parameters["weight"] = nn.Parameter(value, requires_grad=False)
                else:
                    self._parameters["weight"] = nn.Parameter(torch.empty(0), requires_grad=False)
                return
        super().__setattr__(name, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute output = x @ W^T + bias.

        Dispatch order (CUDA):
        1. "buffered" (default): Materialize W into reusable buffer → cuBLAS matmul.
           Near-dense speed, costs one layer buffer in VRAM (~50-100 MB).
        2. "fused": Triton VQ gather-matmul (W never in global memory). VRAM-minimal
           but memory-bound (~5x slower). Use for extreme VRAM constraints.
        3. "naive": CPU tiled path (256-row tiles). Slowest.

        Set self._forward_mode = "fused" to force Triton kernel path.
        """
        # Kurtosis gate: Class 2 modules decide SVD enable/skip per forward call.
        if self._kurtosis_gate is not None and self.has_svd:
            with torch.no_grad():
                enable_svd = self._kurtosis_gate.step(
                    x, cuda_graph_safe=self._cuda_graph_mode
                )
                self._cell_skip_svd = not enable_svd

        input_dtype = x.dtype
        mode = getattr(self, "_forward_mode", "buffered")

        if x.is_cuda and mode == "buffered":
            self._last_dispatch_path = "buffered"
            out = self._forward_buffered(x)
        elif self._fused_available and x.is_cuda and mode == "fused":
            self._last_dispatch_path = "fused"
            out = self._forward_fused(x)
        elif x.is_cuda:
            # CUDA but mode is unknown or fused unavailable — use buffered
            self._last_dispatch_path = "buffered"
            out = self._forward_buffered(x)
        else:
            self._last_dispatch_path = "naive"
            out = self._forward_naive(x)

        if out.dtype != input_dtype:
            out = out.to(input_dtype)
        return out

    def _check_fused_available(self) -> bool:
        """One-time check if fused Triton kernel is available."""
        d = getattr(self, "vector_dim", 1)
        if d == 2:
            # 2D VQ: use dedicated vq2d kernel
            try:
                from helix_substrate.triton_vq2d_matmul import is_available
                return is_available()
            except (ImportError, RuntimeError):
                return False
        elif d > 2:
            # VQ_DIM > 2: not yet supported
            return False
        try:
            from helix_substrate.triton_vq_matmul import is_available
            return is_available()
        except (ImportError, RuntimeError):
            return False

    @property
    def _use_fused(self) -> bool:
        """Cached dispatch decision. Updated on device moves via _refresh_dispatch_cache()."""
        return self._fused_available

    def _refresh_dispatch_cache(self) -> None:
        """Refresh cached dispatch decision after device change."""
        self._fused_available = self._check_fused_available() and self.codebook.is_cuda

    def attach_kurtosis_gate(self, threshold: float, hysteresis_n: int = 2,
                              ema_decay: float = 0.5) -> None:
        """Attach a kurtosis gate for runtime SVD gating (Class 2 modules).

        Once attached, forward() computes input kurtosis and lets the gate
        decide whether to enable SVD correction. Only useful on modules
        with has_svd=True.

        Args:
            threshold: Kurtosis above this enables SVD. Calibrate per-module.
            hysteresis_n: Consecutive windows before switching.
            ema_decay: Kurtosis EMA smoothing (lower = more responsive).
        """
        from helix_substrate.kurtosis_gate import KurtosisGate
        self._kurtosis_gate = KurtosisGate(
            threshold=threshold,
            switch_to_svd_after=hysteresis_n,
            recover_to_skip_after=hysteresis_n,
            ema_decay=ema_decay,
        )

    def detach_kurtosis_gate(self) -> None:
        """Remove kurtosis gate, restoring default SVD behavior."""
        self._kurtosis_gate = None
        self._cell_skip_svd = False

    def set_cuda_graph_mode(self, enabled: bool) -> None:
        """Toggle CUDA graph compatibility mode.

        When enabled:
        - forward() avoids GPU→CPU syncs (.item() in kurtosis gate)
        - Kurtosis gate returns cached decision from last pre_step_kurtosis() call
        - Safe for torch.cuda.CUDAGraph.capture() / replay()

        When disabled:
        - Normal behavior: kurtosis computed inline with GPU→CPU sync
        - Required for accurate gate tracking on varying inputs

        Switch freely between modes at any time.
        """
        self._cuda_graph_mode = enabled

    def pre_step_kurtosis(self, x: torch.Tensor) -> Optional[bool]:
        """Evaluate kurtosis gate OUTSIDE a CUDA graph capture region.

        Call this before graph replay to update the gate decision.
        During graph replay, forward() uses the cached decision.

        Returns:
            True if SVD is enabled, False if skipped, None if no gate attached.
        """
        if self._kurtosis_gate is None or not self.has_svd:
            return None
        with torch.no_grad():
            enable_svd = self._kurtosis_gate.pre_step(x)
            self._cell_skip_svd = not enable_svd
            return enable_svd

    def dispatch_metadata(self) -> dict:
        """Return instrumentation dict for receipt embedding.

        Call after forward() to capture which path was taken.
        """
        return {
            "dispatch_path": self._last_dispatch_path,
            "device": str(self.codebook.device),
            "is_cuda": self.codebook.is_cuda,
            "triton_available": self._fused_available,
            "compute_dtype": str(self.compute_dtype),
            "kernel_metadata": getattr(self, "_last_dispatch", None),
            "cell_skip_svd": self._cell_skip_svd,
            "svd_active": self.has_svd and not self._cell_skip_svd,
            "kurtosis_gate": self._kurtosis_gate.summary() if self._kurtosis_gate else None,
            "cuda_graph_mode": self._cuda_graph_mode,
        }

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused VQ gather-matmul via Triton. W never in global memory."""
        if getattr(self, "vector_dim", 1) == 2:
            return self._forward_fused_vq2d(x)

        from helix_substrate.triton_vq_matmul import fused_vq_matmul

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        # AWQ channel scaling: pre-divide input so codebook operates in scaled space
        if self.has_channel_scales:
            x_2d = x_2d / self.channel_scales.unsqueeze(0)

        dispatch_log = {}
        # Cell-driven SVD gating: skip SVD correction when cell signals it's unnecessary
        skip_svd = self._cell_skip_svd and self.has_svd
        output = fused_vq_matmul(
            x=x_2d,
            codebook=self.codebook,
            indices=self.indices,
            sidecar_rows=self._sidecar_rows,
            sidecar_cols=self._sidecar_cols,
            sidecar_deltas=self._sidecar_deltas,
            svd_U=None if skip_svd else self.svd_U,
            svd_s=None if skip_svd else self.svd_s,
            svd_Vt=None if skip_svd else self.svd_Vt,
            bias=self.bias,
            codebook_f16=self.codebook_f16,
            _dispatch_log=dispatch_log,
            sidecar_phase=self._sidecar_phase,
        )
        self._last_dispatch = dispatch_log

        return output.reshape(*orig_shape[:-1], self.out_features)

    def _forward_fused_vq2d(self, x: torch.Tensor) -> torch.Tensor:
        """Fused 2D VQ gather-matmul via dedicated Triton kernel.

        Codebook [K, 2] stays in L1 (32KB for K=4096). Two loads per index
        instead of one. Same tl.dot path as scalar, paired reduction.
        """
        from helix_substrate.triton_vq2d_matmul import fused_vq2d_matmul

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        if self.has_channel_scales:
            x_2d = x_2d / self.channel_scales.unsqueeze(0)

        # Unpack 12-bit indices for Triton kernel (kernel reads int16 2D)
        if self.index_packing == "12bit":
            from helix_substrate.index_packing import unpack_12bit
            indices_for_kernel = unpack_12bit(
                self.indices, self._idx_rows * self._idx_cols
            ).reshape(self._idx_rows, self._idx_cols)
        else:
            indices_for_kernel = self.indices

        dispatch_log = {}
        output = fused_vq2d_matmul(
            x=x_2d,
            codebook=self.codebook,
            indices=indices_for_kernel,
            sidecar_rows=self._sidecar_rows,
            sidecar_cols=self._sidecar_cols,
            sidecar_deltas=self._sidecar_deltas,
            bias=self.bias,
            _dispatch_log=dispatch_log,
            sidecar_phase=self._sidecar_phase,
        )
        self._last_dispatch = dispatch_log

        # SVD residual (if present and not gated off)
        if self.has_svd and not self._cell_skip_svd:
            x_f32 = x_2d.float()
            down = x_f32 @ self.svd_Vt.t()
            scaled = down * self.svd_s.unsqueeze(0)
            output += scaled @ self.svd_U.t()

        return output.reshape(*orig_shape[:-1], self.out_features)

    # Class-level shared buffer: ONE buffer reused by ALL HelixLinear modules.
    # Allocated lazily on first forward, sized to the largest layer's W matrix.
    # VRAM cost: max(out_features * in_features) * 2 bytes ≈ 50-100 MB for 7B models.
    _shared_buffer: Optional[torch.Tensor] = None
    _shared_buffer_size: int = 0  # numel of current shared buffer

    def _forward_buffered(self, x: torch.Tensor) -> torch.Tensor:
        """Buffered forward: materialize full W into shared buffer, then cuBLAS.

        Per-forward cost: one codebook gather (fast parallel GPU op) + one cuBLAS
        matmul at full tensor core speed. Shared buffer reused across all layers.

        VRAM cost: ONE buffer sized to the largest layer ≈ 50-100 MB total.
        Optimization: indices unpacked once and cached; codebook pre-cast to compute dtype.
        """
        import torch.nn.functional as F

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        # AWQ channel scaling
        if self.has_channel_scales:
            x_2d = x_2d / self.channel_scales.unsqueeze(0)

        # Determine compute dtype — use bfloat16 for tensor cores if input is bf16/f16
        compute_dt = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

        # Shared buffer: allocate/grow if needed, reused across ALL modules
        needed = self.out_features * self.in_features
        buf_ref = HelixLinear._shared_buffer
        if (buf_ref is None or buf_ref.device != x.device
                or buf_ref.dtype != compute_dt or HelixLinear._shared_buffer_size < needed):
            HelixLinear._shared_buffer = torch.empty(
                needed, dtype=compute_dt, device=x.device,
            )
            HelixLinear._shared_buffer_size = needed
        buf = HelixLinear._shared_buffer[:needed].reshape(self.out_features, self.in_features)

        # Steps 1+2: Fused unpack + codebook gather into buffer
        # Try Triton fused kernel first (one launch, zero intermediates).
        # Falls back to PyTorch index_select if Triton unavailable.
        cb = self._get_codebook_for_dtype(compute_dt)
        n_idx = self._idx_rows * self._idx_cols

        if self.vector_dim == 2 and self._triton_gather_available():
            from helix_substrate.triton_gather_12bit import (
                fused_gather_12bit_vq2d, fused_gather_unpacked_vq2d,
            )
            buf_flat = buf.reshape(-1)
            if self.index_packing == "12bit":
                fused_gather_12bit_vq2d(self.indices, cb, buf_flat, n_idx)
            else:
                fused_gather_unpacked_vq2d(
                    self.indices.reshape(-1), cb, buf_flat, n_idx,
                )
        else:
            # Fallback: PyTorch unpack + int32 index_select
            if self.index_packing == "12bit":
                from helix_substrate.index_packing import unpack_12bit
                idx_flat = unpack_12bit(self.indices, n_idx).int()
            else:
                idx_flat = self.indices.reshape(-1).int()

            if self.vector_dim > 1:
                gathered = torch.index_select(cb, 0, idx_flat)
                buf[:] = gathered.reshape(self.out_features, self.in_features)
            else:
                buf.reshape(-1)[:] = torch.index_select(cb, 0, idx_flat).reshape(-1)

        # Step 3: Apply sidecar corrections in-place
        if self._sidecar_rows is not None:
            buf[self._sidecar_rows, self._sidecar_cols] += self._sidecar_deltas.to(compute_dt)

        # Step 4: SVD residual correction
        if self.has_svd and not self._cell_skip_svd:
            scaled_U = self.svd_U * self.svd_s.unsqueeze(0)  # [out, rank]
            buf += (scaled_U @ self.svd_Vt).to(compute_dt)   # [out, in]

        # Step 5: cuBLAS matmul at full tensor core speed
        x_compute = x_2d.to(compute_dt)
        out = F.linear(x_compute, buf, self.bias)

        return out.reshape(*orig_shape[:-1], self.out_features)

    def _triton_gather_available(self) -> bool:
        """Check if the fused Triton gather kernel is available."""
        cached = getattr(self, "_triton_gather_ok", None)
        if cached is not None:
            return cached
        try:
            from helix_substrate.triton_gather_12bit import is_available
            ok = is_available()
        except (ImportError, RuntimeError):
            ok = False
        self._triton_gather_ok = ok
        return ok

    def _get_codebook_for_dtype(self, dt: torch.dtype) -> torch.Tensor:
        """Return codebook pre-cast to target dtype, cached."""
        attr = f"_cb_{dt}".replace(".", "_")
        cached = getattr(self, attr, None)
        if cached is not None:
            return cached
        cb = self.codebook.to(dt)
        setattr(self, attr, cb)
        return cb

    def _dequant_tile(self, start_row: int, end_row: int) -> torch.Tensor:
        """
        Dequantize a tile of weight rows from compressed representation.

        This is the single interface for bounded weight materialization.
        Both the CPU tiled forward path and decode_weight() use this.

        Args:
            start_row: First output row (inclusive)
            end_row:   Last output row (exclusive)

        Returns:
            [end_row - start_row, in_features] float32 tensor with VQ + sidecar + SVD applied.
        """
        # VQ gather for this tile
        if self.index_packing == "12bit":
            from helix_substrate.index_packing import unpack_12bit_rows
            idx_slice = unpack_12bit_rows(
                self.indices, start_row, end_row, self._idx_cols
            ).long()
        else:
            idx_slice = self.indices[start_row:end_row].long()
        if self.vector_dim > 1:
            # Vector VQ: codebook [k, d], indices [rows, in/d]
            # Gather: [rows, in/d, d] → reshape to [rows, in]
            raw = self.codebook[idx_slice]  # [rows, in/d, d]
            tile = raw.reshape(end_row - start_row, self.in_features)
        else:
            tile = self.codebook[idx_slice]  # [rows, in]

        # Sidecar correction (precomputed rows/cols/deltas)
        if self._sidecar_rows is not None:
            mask = (self._sidecar_rows >= start_row) & (self._sidecar_rows < end_row)
            if mask.any():
                tile = tile.clone()
                local_rows = self._sidecar_rows[mask] - start_row
                local_cols = self._sidecar_cols[mask]
                tile[local_rows, local_cols] += self._sidecar_deltas[mask]

        # SVD residual correction (gated by cell signal)
        if self.has_svd and not self._cell_skip_svd:
            scaled_U = self.svd_U[start_row:end_row] * self.svd_s.unsqueeze(0)
            tile = tile + scaled_U @ self.svd_Vt

        return tile

    def _forward_naive(self, x: torch.Tensor) -> torch.Tensor:
        """Tiled CPU path: process 256 output rows at a time via _dequant_tile().

        Peak temporary: 256 * max_in * 4 bytes (~8.75 MB worst case)
        vs full W:      out * in * 4 bytes (~52.5 MB worst case).
        """
        CHUNK = 256
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).float()

        # AWQ channel scaling: pre-divide input so codebook operates in scaled space
        if self.has_channel_scales:
            x_2d = x_2d / self.channel_scales.unsqueeze(0)
        N = x_2d.shape[0]
        output = torch.zeros(N, self.out_features, device=x.device, dtype=torch.float32)

        # Optional FP16 compute (matmul in FP16, accumulate in FP32)
        use_fp16 = self.compute_dtype == torch.float16 and x.is_cuda
        x_compute = x_2d.half() if use_fp16 else x_2d

        for i in range(0, self.out_features, CHUNK):
            end = min(i + CHUNK, self.out_features)
            W_tile = self._dequant_tile(i, end)

            if use_fp16:
                output[:, i:end] = (x_compute @ W_tile.half().t()).float()
            else:
                output[:, i:end] = x_compute @ W_tile.float().t()

        # Bias
        if self.bias is not None:
            output += self.bias.unsqueeze(0)

        return output.reshape(*orig_shape[:-1], self.out_features)

    def decode_weight(self) -> torch.Tensor:
        """Reconstruct the full weight tensor (for debugging/validation).

        Uses _dequant_tile() over the full row range — same code path as forward,
        just without the tiled loop. Only call this for validation, never in forward().
        """
        with torch.no_grad():
            return self._dequant_tile(0, self.out_features)

    def memory_savings(self) -> dict:
        """Report memory usage vs equivalent nn.Linear."""
        dense_bytes = self.out_features * self.in_features * 4  # float32
        index_bytes_each = 2 if self.indices.dtype == torch.int16 else 1
        compressed = (
            self.codebook.numel() * 4  # codebook: k * 4
            + self.indices.numel() * index_bytes_each  # uint8 or int16 indices
        )
        if self.codebook_f16 is not None:
            compressed += self.codebook_f16.numel() * 2  # float16
        if self.sidecar_positions is not None:
            compressed += self.sidecar_positions.numel() * 8  # int64
            compressed += self.sidecar_values.numel() * 4  # float32
        if self._sidecar_rows is not None:
            compressed += self._sidecar_rows.numel() * 8   # int64
            compressed += self._sidecar_cols.numel() * 8   # int64
            compressed += self._sidecar_deltas.numel() * 4  # float32
        if self.has_svd:
            compressed += self.svd_U.numel() * 4
            compressed += self.svd_s.numel() * 4
            compressed += self.svd_Vt.numel() * 4
        if self.has_channel_scales:
            compressed += self.channel_scales.numel() * 4  # float32
        return {
            "dense_bytes": dense_bytes,
            "compressed_bytes": compressed,
            "ratio": round(dense_bytes / max(1, compressed), 2),
            "savings_pct": round(100 * (1 - compressed / dense_bytes), 1),
        }

    def _recompute_derived(self) -> None:
        """Recompute derived buffers from primary data after safetensors load.

        Called by the HF quantizer in _process_model_after_weight_loading.
        At this point, safetensors has loaded real data into primary buffers
        (codebook, indices) and optional buffers (sidecar_*, svd_*, bias,
        channel_scales). Empty [0]-shaped tensors indicate absent optional data
        and are normalized back to None here.
        """
        # Infer vector_dim from codebook shape: [k] = scalar, [k, d] = vector
        if self.codebook is not None and self.codebook.ndim == 2:
            self.vector_dim = self.codebook.shape[1]
        elif not hasattr(self, "vector_dim"):
            self.vector_dim = 1

        # Detect 12-bit packed indices: uint8 dtype with k > 256
        # (normal uint8 indices only used for k <= 256)
        if not hasattr(self, "index_packing"):
            self.index_packing = None
        k = self.codebook.shape[0] if self.codebook is not None else 0
        if (self.indices is not None and self.indices.dtype == torch.uint8
                and self.indices.ndim == 1 and k > 256):
            self.index_packing = "12bit"
        # Set index matrix dimensions for unpacking
        if not hasattr(self, "_idx_rows"):
            idx_cols = self.in_features // self.vector_dim if self.vector_dim > 1 else self.in_features
            self._idx_rows = self.out_features
            self._idx_cols = idx_cols

        # Normalize empty tensors → None for optional buffers
        _optional = [
            "sidecar_positions", "sidecar_values",
            "svd_U", "svd_s", "svd_Vt",
            "bias", "channel_scales",
        ]
        for name in _optional:
            buf = getattr(self, name, None)
            if buf is not None and buf.numel() == 0:
                self.register_buffer(name, None)

        # Sidecar derived buffers
        # For 12-bit packed indices, temporarily unpack for sidecar computation
        if self.sidecar_positions is not None and self.sidecar_values is not None:
            if self.index_packing == "12bit":
                from helix_substrate.index_packing import unpack_12bit
                _indices_2d = unpack_12bit(
                    self.indices, self._idx_rows * self._idx_cols
                ).reshape(self._idx_rows, self._idx_cols)
            else:
                _indices_2d = self.indices

            d = getattr(self, "vector_dim", 1)
            if d > 1:
                # Vector VQ: position p in flat [out*in] space maps to
                # index at p//d in flat indices, sub-element p%d in codebook vector
                rows = self.sidecar_positions // self.in_features
                cols = self.sidecar_positions % self.in_features
                idx_rows = rows
                idx_cols = cols // d
                sub_elem = cols % d
                idx_2d = _indices_2d.long()[idx_rows, idx_cols]
                vq_at_sidecar = self.codebook[idx_2d, sub_elem]
            else:
                idx_flat = _indices_2d.reshape(-1).long()
                vq_at_sidecar = self.codebook[idx_flat[self.sidecar_positions]]
            self.register_buffer("_sidecar_vq_vals", vq_at_sidecar.contiguous())
            self.register_buffer(
                "_sidecar_rows",
                (self.sidecar_positions // self.in_features).long(),
            )
            self.register_buffer(
                "_sidecar_cols",
                (self.sidecar_positions % self.in_features).long(),
            )
            self.register_buffer(
                "_sidecar_deltas",
                (self.sidecar_values - vq_at_sidecar).contiguous(),
            )
        else:
            self.register_buffer("_sidecar_vq_vals", None)
            self.register_buffer("_sidecar_rows", None)
            self.register_buffer("_sidecar_cols", None)
            self.register_buffer("_sidecar_deltas", None)

        # SVD flag
        self.has_svd = self.svd_U is not None
        self.rank = self.svd_U.shape[1] if self.has_svd else 0

        # Channel scales flag
        self.has_channel_scales = self.channel_scales is not None

        # FP16 codebook
        if self.compute_dtype == torch.float16 and self.codebook is not None:
            self.register_buffer("codebook_f16", self.codebook.half().contiguous())
        else:
            self.register_buffer("codebook_f16", None)

        # Refresh dispatch cache
        self._fused_available = self._check_fused_available()

    @classmethod
    def from_quantized_config(
        cls,
        in_features: int,
        out_features: int,
        tensor_name: str = "",
        compute_dtype: torch.dtype = torch.float32,
        n_clusters: int = 256,
        vector_dim: int = 1,
        index_packing: Optional[str] = None,
    ) -> "HelixLinear":
        """Create a HelixLinear shell for HF quantizer loading.

        Registers all buffers as real tensors (empty [0]-shape for optional
        buffers) so safetensors can load data into them. After loading,
        call _recompute_derived() to normalize empty→None and compute
        derived buffers.

        Args:
            index_packing: None for standard int16/uint8, "12bit" for packed
                           12-bit indices (saves 25% VRAM for k>256).
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        instance.in_features = in_features
        instance.out_features = out_features
        instance.tensor_name = tensor_name
        instance.compute_dtype = compute_dtype
        instance.vector_dim = vector_dim
        instance._last_dispatch_path = "unknown"
        instance._fused_available = False
        instance._cell_skip_svd = False
        instance._kurtosis_gate = None
        instance._sidecar_phase = None
        instance._forward_mode = "buffered"
        instance._weight_buffer = None
        instance._cuda_graph_mode = False
        instance.has_svd = False
        instance.rank = 0
        instance.has_channel_scales = False

        # Weight stub for get_parameter/get_parameter_or_buffer/tie_weights compat
        instance.weight = nn.Parameter(torch.empty(0, dtype=compute_dtype), requires_grad=False)

        # Primary buffers — will be overwritten by safetensors load
        # Codebook: [k] for scalar, [k, d] for vector VQ
        if vector_dim > 1:
            instance.register_buffer("codebook", torch.zeros(n_clusters, vector_dim, dtype=torch.float32))
        else:
            instance.register_buffer("codebook", torch.zeros(n_clusters, dtype=torch.float32))

        # Indices buffer — shape depends on packing mode
        idx_cols = in_features // vector_dim if vector_dim > 1 else in_features
        instance._idx_rows = out_features
        instance._idx_cols = idx_cols

        if index_packing == "12bit" and n_clusters > 256:
            # 12-bit packed: uint8 1D, 3 bytes per 2 indices
            n_indices = out_features * idx_cols
            packed_bytes = n_indices * 3 // 2
            instance.register_buffer(
                "indices",
                torch.zeros(packed_bytes, dtype=torch.uint8),
            )
            instance.index_packing = "12bit"
        else:
            # Standard: int16 for k>256, uint8 for k<=256
            index_dtype = torch.int16 if n_clusters > 256 else torch.uint8
            instance.register_buffer(
                "indices",
                torch.zeros(out_features, idx_cols, dtype=index_dtype),
            )
            instance.index_packing = None

        # Optional primary buffers — empty tensors so safetensors can load into them.
        # _recompute_derived() normalizes empty→None after loading.
        instance.register_buffer("sidecar_positions", torch.zeros(0, dtype=torch.int64))
        instance.register_buffer("sidecar_values", torch.zeros(0, dtype=torch.float32))
        instance.register_buffer("svd_U", torch.zeros(0, dtype=torch.float32))
        instance.register_buffer("svd_s", torch.zeros(0, dtype=torch.float32))
        instance.register_buffer("svd_Vt", torch.zeros(0, dtype=torch.float32))
        instance.register_buffer("bias", torch.zeros(0, dtype=torch.float32))
        instance.register_buffer("channel_scales", torch.zeros(0, dtype=torch.float32))

        # Derived buffers — will be computed by _recompute_derived()
        instance.register_buffer("codebook_f16", torch.zeros(0, dtype=torch.float16))
        instance.register_buffer("_sidecar_vq_vals", torch.zeros(0, dtype=torch.float32))
        instance.register_buffer("_sidecar_rows", torch.zeros(0, dtype=torch.int64))
        instance.register_buffer("_sidecar_cols", torch.zeros(0, dtype=torch.int64))
        instance.register_buffer("_sidecar_deltas", torch.zeros(0, dtype=torch.float32))

        return instance

    def extra_repr(self) -> str:
        parts = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
        ]
        if self.has_svd:
            parts.append(f"svd_rank={self.rank}")
        if self.sidecar_positions is not None:
            parts.append(f"n_outliers={self.sidecar_positions.numel()}")
        if self.bias is not None:
            parts.append("bias=True")
        savings = self.memory_savings()
        parts.append(f"compression={savings['ratio']}x")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# CDNA v3 Loader
# ---------------------------------------------------------------------------

def load_helix_linear_from_cdnav3(
    tensor_dir: Path,
    bias: Optional[torch.Tensor] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> HelixLinear:
    """
    Load a single HelixLinear from a .cdnav3 directory.

    Args:
        tensor_dir: Path to the {name}.cdnav3/ directory
        bias: Optional bias tensor (from original nn.Linear)

    Returns:
        HelixLinear module ready for .cuda() or .eval()
    """
    tensor_dir = Path(tensor_dir)
    meta = json.loads((tensor_dir / "meta.json").read_text())
    rows, cols = meta["shape"]
    vector_dim = meta.get("vector_dim", 1)

    # Codebook: [k] float32 for scalar, [k, d] for vector VQ
    codebook = torch.from_numpy(
        np.load(tensor_dir / "codebook.npy").astype(np.float32)
    )

    # Indices: [rows, cols] for scalar, [rows, cols/d] for vector VQ
    index_dtype_str = meta.get("index_dtype", "uint8")
    np_index_dtype = np.uint16 if index_dtype_str == "uint16" else np.uint8
    raw_indices = np.fromfile(tensor_dir / "indices.bin", dtype=np_index_dtype)
    idx_cols = cols // vector_dim if vector_dim > 1 else cols
    indices = torch.from_numpy(raw_indices.reshape(rows, idx_cols).copy())

    # Sidecar: optional outlier corrections
    sidecar_positions = None
    sidecar_values = None
    sidecar_path = tensor_dir / "sidecar.npz"
    if sidecar_path.exists():
        sidecar_data = np.load(sidecar_path)
        sidecar_positions = torch.from_numpy(
            sidecar_data["positions"].astype(np.int64).copy()
        )
        sidecar_values = torch.from_numpy(
            sidecar_data["values"].astype(np.float32).copy()
        )

    # SVD residual factors: optional
    svd_U = svd_s = svd_Vt = None
    if (tensor_dir / "svd_U.npy").exists():
        svd_U = torch.from_numpy(
            np.load(tensor_dir / "svd_U.npy").astype(np.float32).copy()
        )
        svd_s = torch.from_numpy(
            np.load(tensor_dir / "svd_s.npy").astype(np.float32).copy()
        )
        svd_Vt = torch.from_numpy(
            np.load(tensor_dir / "svd_Vt.npy").astype(np.float32).copy()
        )

    # AWQ channel scales: optional (present when compressed with --scale-file)
    channel_scales = None
    scales_path = tensor_dir / "channel_scales.npy"
    if scales_path.exists():
        channel_scales = torch.from_numpy(
            np.load(scales_path).astype(np.float32).copy()
        )

    return HelixLinear(
        in_features=cols,
        out_features=rows,
        codebook=codebook,
        indices=indices,
        sidecar_positions=sidecar_positions,
        sidecar_values=sidecar_values,
        svd_U=svd_U,
        svd_s=svd_s,
        svd_Vt=svd_Vt,
        bias=bias,
        tensor_name=meta.get("tensor_name", ""),
        compute_dtype=compute_dtype,
        channel_scales=channel_scales,
        vector_dim=vector_dim,
    )


def load_cdna_factors(
    cdna_dir: Path,
    model: Optional[nn.Module] = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Dict[str, HelixLinear]:
    """
    Load all CDNA v3 tensors from a directory into HelixLinear modules.

    Scans for all .cdnav3/ subdirectories, loads each as HelixLinear,
    and maps them to HuggingFace-style tensor names.

    Args:
        cdna_dir: Path containing .cdnav3/ subdirectories
        model: Optional model to extract biases from original nn.Linear modules

    Returns:
        Dict mapping HF tensor names → HelixLinear modules
        e.g. {"model.layers.0.self_attn.q_proj": HelixLinear(...), ...}
    """
    cdna_dir = Path(cdna_dir)
    result: Dict[str, HelixLinear] = {}

    # Collect biases: first from saved .npy files, then from model if provided
    biases: Dict[str, torch.Tensor] = {}

    # Load bias .npy files (saved by compress.py for models with attention biases)
    for meta_path in sorted(cdna_dir.glob("*.npy.meta.json")):
        meta = json.loads(meta_path.read_text())
        tensor_name = meta.get("tensor_name", "")
        if tensor_name.endswith(".bias"):
            npy_path = meta_path.parent / meta_path.name.replace(".meta.json", "")
            if npy_path.exists():
                module_path = tensor_name[:-5]  # strip ".bias"
                biases[module_path] = torch.from_numpy(np.load(npy_path))

    # Override with model biases if model is provided (authoritative)
    if model is not None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                biases[name] = module.bias.data.clone()

    # Scan for .cdnav3 directories
    for tensor_path in sorted(cdna_dir.glob("*.cdnav3")):
        if not tensor_path.is_dir():
            continue

        meta_path = tensor_path / "meta.json"
        if not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        tensor_name = meta["tensor_name"]

        # Skip non-2D (norms stored as .npy, not .cdnav3)
        if meta.get("storage_mode") == "exact":
            continue

        # Convert tensor name to module path
        # "model.layers.0.self_attn.q_proj.weight" → "model.layers.0.self_attn.q_proj"
        module_name = _tensor_name_to_module_path(tensor_name)

        bias = biases.get(module_name)
        helix_mod = load_helix_linear_from_cdnav3(tensor_path, bias=bias, compute_dtype=compute_dtype)
        result[module_name] = helix_mod

    return result


def swap_to_helix(
    model: nn.Module,
    helix_modules: Dict[str, HelixLinear],
    compute_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Replace nn.Linear modules in a model with HelixLinear equivalents.

    This is a one-shot surgery: walks model.named_modules(), replaces any
    nn.Linear whose path matches a key in helix_modules.

    Args:
        model: PyTorch model (e.g., AutoModelForCausalLM)
        helix_modules: Dict from load_cdna_factors()

    Returns:
        Modified model (same object, modules replaced in-place)

    Example:
        factors = load_cdna_factors("/path/to/cdna/", model)
        model = swap_to_helix(model, factors)
        model = model.cuda().eval()
    """
    replaced = 0
    skipped = 0

    for name, module in list(model.named_modules()):
        if name in helix_modules:
            # Only replace nn.Linear modules — skip Embedding, Conv1d, etc.
            if not isinstance(module, nn.Linear):
                skipped += 1
                continue

            new_mod = helix_modules[name]
            if compute_dtype != torch.float32:
                new_mod.set_compute_dtype(compute_dtype)

            # Verify shape compatibility
            assert new_mod.in_features == module.in_features, (
                f"{name}: in_features mismatch "
                f"({new_mod.in_features} vs {module.in_features})"
            )
            assert new_mod.out_features == module.out_features, (
                f"{name}: out_features mismatch "
                f"({new_mod.out_features} vs {module.out_features})"
            )

            # Replace module in parent
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_mod)
            replaced += 1

    return model


def set_cuda_graph_mode(model: nn.Module, enabled: bool) -> int:
    """Toggle CUDA graph mode across all HelixLinear modules.

    When enabled, forward() avoids GPU→CPU syncs (safe for graph capture/replay).
    When disabled, normal kurtosis evaluation with inline .item() calls.

    Call pre_step_kurtosis_all() before graph replay to update gate decisions.

    Args:
        model: Model containing HelixLinear modules.
        enabled: True to enable graph-safe mode, False for normal mode.

    Returns:
        Number of HelixLinear modules updated.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, HelixLinear):
            module.set_cuda_graph_mode(enabled)
            count += 1
    return count


def pre_step_kurtosis_all(model: nn.Module, inputs: Optional[Dict[str, torch.Tensor]] = None) -> int:
    """Evaluate kurtosis gates for all gated modules OUTSIDE graph capture.

    Call this before CUDA graph replay to update cached gate decisions.
    In graph mode, forward() uses these cached decisions with zero sync.

    Args:
        model: Model containing HelixLinear modules.
        inputs: Optional {tensor_name: activation_tensor} for per-module input.
                If None, gates keep their current cached decision.

    Returns:
        Number of gates evaluated.
    """
    count = 0
    if inputs is None:
        return count
    for name, module in model.named_modules():
        if isinstance(module, HelixLinear) and module._kurtosis_gate is not None:
            tensor_key = module.tensor_name
            if tensor_key in inputs:
                module.pre_step_kurtosis(inputs[tensor_key])
                count += 1
    return count


def freeze_sidecar_phase(model: nn.Module, phase: Optional[str]) -> int:
    """Freeze sidecar routing across all HelixLinear modules.

    Args:
        model: Model containing HelixLinear modules.
        phase: "fused" (decode, N<=16), "scatter" (prefill, N>16), or None (auto).

    Returns:
        Number of HelixLinear modules updated.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, HelixLinear):
            module._sidecar_phase = phase
            count += 1
    return count


def swap_summary(model: nn.Module) -> dict:
    """Report how many modules are HelixLinear vs nn.Linear."""
    helix_count = 0
    linear_count = 0
    helix_bytes = 0
    dense_bytes = 0

    for name, module in model.named_modules():
        if isinstance(module, HelixLinear):
            helix_count += 1
            savings = module.memory_savings()
            helix_bytes += savings["compressed_bytes"]
            dense_bytes += savings["dense_bytes"]
        elif isinstance(module, nn.Linear):
            linear_count += 1
            dense_bytes += module.weight.numel() * 4

    return {
        "helix_modules": helix_count,
        "linear_modules": linear_count,
        "total_linear": helix_count + linear_count,
        "compressed_bytes": helix_bytes,
        "dense_equivalent_bytes": dense_bytes,
        "overall_ratio": round(dense_bytes / max(1, helix_bytes), 2),
    }


# ---------------------------------------------------------------------------
# Tensor name mapping helpers
# ---------------------------------------------------------------------------

# HuggingFace → module path mapping
_HF_WEIGHT_SUFFIX = ".weight"

# GGUF → HF name mapping (for cross-format compatibility)
_GGUF_TO_HF = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def _tensor_name_to_module_path(tensor_name: str) -> str:
    """
    Convert a tensor name (from meta.json) to a PyTorch module path.

    Handles both HuggingFace and GGUF naming conventions:
        "model.layers.0.self_attn.q_proj.weight" → "model.layers.0.self_attn.q_proj"
        "blk.0.attn_q.weight" → "model.layers.0.self_attn.q_proj"
    """
    # HuggingFace format: strip .weight suffix
    if tensor_name.startswith("model.layers."):
        if tensor_name.endswith(_HF_WEIGHT_SUFFIX):
            return tensor_name[: -len(_HF_WEIGHT_SUFFIX)]
        return tensor_name

    # GGUF format: blk.N.module.weight
    if tensor_name.startswith("blk."):
        parts = tensor_name.split(".")
        # blk.N.module.weight → layer_idx=N, module=parts[2]
        layer_idx = int(parts[1])
        module_key = parts[2]
        if len(parts) > 3 and parts[3] != "weight":
            module_key = f"{parts[2]}_{parts[3]}"

        hf_module = _GGUF_TO_HF.get(module_key, module_key)
        return f"model.layers.{layer_idx}.{hf_module}"

    # Special tensors
    if tensor_name in ("token_embd.weight", "model.embed_tokens.weight"):
        return "model.embed_tokens"
    if tensor_name in ("output.weight", "lm_head.weight"):
        return "lm_head"

    # Fallback: strip .weight if present
    if tensor_name.endswith(_HF_WEIGHT_SUFFIX):
        return tensor_name[: -len(_HF_WEIGHT_SUFFIX)]
    return tensor_name
