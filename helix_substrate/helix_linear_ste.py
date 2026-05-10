"""
HelixLinearSTE — Trainable compressed linear layer with straight-through estimator.

Stores weights in CDNA v3 format (codebook + uint8 indices + sidecar + SVD)
but makes codebook entries, sidecar values, and SVD factors into nn.Parameters
so gradients flow through during training.

The non-differentiable step (argmin index assignment) uses a straight-through
estimator: forward quantizes, backward pretends it didn't happen.

Periodically, indices are re-assigned via k-means to track the evolving codebook.

This is a RESEARCH module for compressed-domain training experiments.
For inference, use HelixLinear (helix_linear.py) which is production-hardened.

Work Order: WO-COMPRESSED-NATIVE-TRAINING-01
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class VQStraightThrough(torch.autograd.Function):
    """Straight-through estimator for VQ codebook gather.

    Supports both scalar (codebook [k]) and grouped (codebook [k, d]) VQ.

    Forward: W_vq = codebook[indices]  →  [N] or [N, d]
    Backward: gradient flows to codebook entries via scatter_add.
              For grouped VQ, each sub-element gradient is scattered independently.
    """

    @staticmethod
    def forward(ctx, codebook: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.codebook_shape = codebook.shape
        return codebook[indices.long()]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices, = ctx.saved_tensors
        k = ctx.codebook_shape[0]

        if len(ctx.codebook_shape) == 1:
            # Scalar VQ: codebook [k], grad_output [...] → flatten and scatter
            grad_codebook = torch.zeros(k, device=grad_output.device,
                                        dtype=grad_output.dtype)
            grad_codebook.scatter_add_(0, indices.long().reshape(-1),
                                       grad_output.reshape(-1))
        else:
            # Grouped VQ: codebook [k, d], grad_output [..., d]
            d = ctx.codebook_shape[1]
            flat_idx = indices.long().reshape(-1)             # [N]
            flat_grad = grad_output.reshape(-1, d)            # [N, d]
            grad_codebook = torch.zeros(k, d, device=grad_output.device,
                                        dtype=grad_output.dtype)
            # Scatter per sub-element: expand index to [N, d] for dim=0 scatter
            idx_expand = flat_idx.unsqueeze(1).expand(-1, d)  # [N, d]
            grad_codebook.scatter_add_(0, idx_expand, flat_grad)

        return grad_codebook, None  # no gradient for indices


def _torch_kmeans(
    data: torch.Tensor,
    n_clusters: int = 256,
    max_iters: int = 10,
    rtol: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-friendly scalar k-means. Mirrors cdna_encoder._simple_kmeans.

    Args:
        data: 1D tensor of values
        n_clusters: Number of clusters (max 256 for uint8 indices)
        max_iters: Maximum iterations
        rtol: Relative convergence tolerance

    Returns:
        (centroids [n_clusters], assignments [N] uint8)
    """
    n_unique = len(torch.unique(data))
    n_clusters = min(n_clusters, n_unique)

    # Percentile initialization
    percentiles = torch.linspace(0, 1, n_clusters, device=data.device)
    sorted_data = torch.sort(data)[0]
    idx = (percentiles * (len(sorted_data) - 1)).long()
    centroids = sorted_data[idx].float()

    cb_range = (centroids[-1] - centroids[0]).item()
    if cb_range < 1e-30:
        cb_range = 1.0
    abs_tol = rtol * cb_range

    CHUNK = 1 << 20  # 1M elements per chunk to avoid OOM on [N, 256]

    for _ in range(max_iters):
        # Chunked assignment
        assignments = torch.empty(data.shape[0], device=data.device, dtype=torch.uint8)
        for start in range(0, data.shape[0], CHUNK):
            end = min(start + CHUNK, data.shape[0])
            dists = (data[start:end, None] - centroids[None, :]).abs()
            assignments[start:end] = dists.argmin(dim=1).to(torch.uint8)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_clusters):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = data[mask].mean()
            else:
                new_centroids[i] = centroids[i]

        max_delta = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if max_delta < abs_tol:
            break

    return centroids, assignments


def _torch_vector_kmeans(
    data: torch.Tensor,
    n_clusters: int = 256,
    max_iters: int = 10,
    rtol: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-friendly d-dimensional k-means for grouped VQ.

    Args:
        data: [N, d] tensor of d-dimensional vectors
        n_clusters: Number of clusters (max 256 for uint8 indices)
        max_iters: Maximum iterations
        rtol: Relative convergence tolerance

    Returns:
        (centroids [n_clusters, d], assignments [N] uint8)
    """
    n, d = data.shape
    n_clusters = min(n_clusters, n)

    # Random subset initialization (k-means++ is overkill for training — SGD refines)
    perm = torch.randperm(n, device=data.device)[:n_clusters]
    centroids = data[perm].float().clone()

    cb_range = float((centroids.max() - centroids.min()).item())
    if cb_range < 1e-30:
        cb_range = 1.0
    abs_tol = rtol * cb_range

    CHUNK = 500000  # vectors per chunk

    for _ in range(max_iters):
        # Chunked assignment via cdist
        assignments = torch.empty(n, device=data.device, dtype=torch.uint8)
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            dists = torch.cdist(data[start:end].float(), centroids)  # [chunk, k]
            assignments[start:end] = dists.argmin(dim=1).to(torch.uint8)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_clusters):
            mask = assignments == i
            if mask.any():
                new_centroids[i] = data[mask].float().mean(dim=0)
            else:
                new_centroids[i] = centroids[i]

        max_delta = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if max_delta < abs_tol:
            break

    return centroids, assignments


class HelixLinearSTE(nn.Module):
    """Trainable compressed linear layer with VQ straight-through estimator.

    Weights are stored as codebook[indices] + sidecar + SVD, same as HelixLinear.
    But codebook, sidecar_values, and SVD factors are nn.Parameters.

    Forward materializes the full W = codebook[indices] + corrections, then
    computes x @ W.T. STE lets gradients flow through the VQ gather.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        codebook: torch.Tensor,
        indices: torch.Tensor,
        vector_dim: int = 1,
        svd_rank: int = 0,
        svd_U: Optional[torch.Tensor] = None,
        svd_s: Optional[torch.Tensor] = None,
        svd_Vt: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.vector_dim = vector_dim

        assert in_features % vector_dim == 0, \
            f"in_features ({in_features}) must be divisible by vector_dim ({vector_dim})"

        # Learnable VQ codebook: [k] for scalar, [k, d] for grouped
        self.codebook = nn.Parameter(codebook.float().contiguous())
        # Discrete indices — NOT learnable, updated by reassign_indices()
        # For grouped VQ: indices shape = [out, in/d]
        self.register_buffer("indices", indices.to(torch.uint8).contiguous())

        # Learnable SVD residual
        self.has_svd = svd_U is not None
        if self.has_svd:
            self.svd_U = nn.Parameter(svd_U.float().contiguous())
            self.svd_s = nn.Parameter(svd_s.float().contiguous())
            self.svd_Vt = nn.Parameter(svd_Vt.float().contiguous())
        else:
            self.svd_U = None
            self.svd_s = None
            self.svd_Vt = None

        # Optional bias
        if bias is not None:
            self.bias = nn.Parameter(bias.float().contiguous())
        else:
            self.bias = None

        # LoRA adapter (initialized by enable_lora())
        self.has_lora = False
        self.lora_rank = 0
        self.lora_alpha = 1.0
        self.lora_A: Optional[nn.Parameter] = None
        self.lora_B: Optional[nn.Parameter] = None

        # Monitoring
        self._reassign_count = 0
        self._last_indices_changed = 0

    def enable_lora(self, rank: int = 8, alpha: float = 1.0) -> None:
        """Attach a LoRA adapter to this layer.

        Creates A [rank, in_features] (Kaiming) and B [out_features, rank] (zeros).
        At step 0, output is identical to base because B=0 → LoRA contribution = 0.

        Args:
            rank: LoRA rank (bottleneck dimension).
            alpha: LoRA scaling factor. Output scaled by alpha/rank.
        """
        self.lora_rank = rank
        self.lora_alpha = alpha
        self.has_lora = True

        # A: [rank, in_features] — Kaiming init
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features,
                                               device=self.codebook.device,
                                               dtype=self.codebook.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        # B: [out_features, rank] — zeros (output = base at step 0)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank,
                                               device=self.codebook.device,
                                               dtype=self.codebook.dtype))

    def disable_lora(self) -> None:
        """Remove LoRA adapter and free memory."""
        self.has_lora = False
        self.lora_rank = 0
        self.lora_alpha = 1.0
        self.lora_A = None
        self.lora_B = None

    def freeze_base(self, freeze_svd: bool = True) -> None:
        """Freeze all base parameters. Only LoRA params remain trainable.

        Args:
            freeze_svd: If True (default), freeze SVD factors too. Set False to
                        allow SVD to adapt alongside LoRA (experimental).
        """
        self.codebook.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        if self.has_svd and freeze_svd:
            self.svd_U.requires_grad_(False)
            self.svd_s.requires_grad_(False)
            self.svd_Vt.requires_grad_(False)

    def unfreeze_base(self) -> None:
        """Unfreeze all base parameters (return to born-compressed training mode)."""
        self.codebook.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)
        if self.has_svd:
            self.svd_U.requires_grad_(True)
            self.svd_s.requires_grad_(True)
            self.svd_Vt.requires_grad_(True)

    def merge_lora(self) -> None:
        """Permanently bake LoRA delta into the base codebook for checkpoint export.

        Reconstructs W_effective = W_base + B @ A * (alpha/rank), then runs
        k-means to produce a new codebook+indices that absorbs the LoRA delta.
        After merge, LoRA is disabled and the layer returns to base-only mode.

        WARNING: This is expensive (full k-means per layer) and non-deterministic.
        Use only for shipping a new base checkpoint, NOT during adapter-swap inference.
        For multi-adapter serving, keep base frozen and swap LoRA A/B tensors instead.
        """
        if not self.has_lora:
            return
        with torch.no_grad():
            # Get base W
            W_base = self._reconstruct_full()
            # Add LoRA delta
            lora_delta = (self.lora_B @ self.lora_A) * (self.lora_alpha / self.lora_rank)
            W_merged = W_base + lora_delta

            # Re-quantize via k-means
            if self.vector_dim == 1:
                flat = W_merged.reshape(-1)
                new_cb, new_idx = _torch_kmeans(flat, n_clusters=self.codebook.shape[0])
                self.codebook.data.copy_(new_cb)
                self.indices.copy_(new_idx.reshape(self.out_features, self.in_features))
            else:
                d = self.vector_dim
                vectors = W_merged.reshape(-1, d)
                new_cb, new_idx = _torch_vector_kmeans(vectors, n_clusters=self.codebook.shape[0])
                self.codebook.data.copy_(new_cb)
                self.indices.copy_(new_idx.reshape(self.out_features, self.in_features // d))

            # Re-fit SVD on new quantization residual
            if self.has_svd:
                rank = self.svd_U.shape[1]
                W_vq = self.codebook[self.indices.long()]
                if self.vector_dim > 1:
                    W_vq = W_vq.reshape(self.out_features, self.in_features)
                residual = W_merged - W_vq
                U, s, Vt = torch.linalg.svd(residual, full_matrices=False)
                self.svd_U.data.copy_(U[:, :rank])
                self.svd_s.data.copy_(s[:rank])
                self.svd_Vt.data.copy_(Vt[:rank, :])

        self.disable_lora()
        self.unfreeze_base()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. VQ reconstruction — STE when codebook is trainable, plain gather when frozen
        if self.codebook.requires_grad:
            W_vq = VQStraightThrough.apply(self.codebook, self.indices)
        else:
            W_vq = self.codebook[self.indices.long()]

        # For grouped VQ: codebook[indices] → [out, in/d, d], reshape to [out, in]
        if self.vector_dim > 1:
            W = W_vq.reshape(self.out_features, self.in_features)
        else:
            W = W_vq

        # 2. SVD residual correction (direct gradient when unfrozen)
        if self.has_svd:
            W = W + (self.svd_U * self.svd_s.unsqueeze(0)) @ self.svd_Vt

        # 3. Linear operation
        out = x @ W.t()

        # 4. LoRA addition: y += (x @ A^T) @ B^T * (alpha / rank)
        if self.has_lora:
            out = out + (x @ self.lora_A.t()) @ self.lora_B.t() * (self.lora_alpha / self.lora_rank)

        if self.bias is not None:
            out = out + self.bias
        return out

    @torch.no_grad()
    def reassign_indices(self):
        """Re-run k-means on effective weights to recompute codebook + indices.

        For born-compressed training, the naive approach (reconstruct W from
        codebook[indices] then find nearest) is a tautological no-op because
        W[i] = codebook[indices[i]] by definition.

        This method runs a full k-means pass on the current effective weight
        matrix (including SVD residual if present), producing a fresh codebook
        and index assignment. This allows the codebook to reorganize — merging
        collapsed entries, splitting overloaded ones, and fixing crossings from
        STE gradient updates.
        """
        W = self._reconstruct_full()  # [out, in]

        if self.vector_dim == 1:
            flat = W.reshape(-1)
            new_codebook, new_indices_flat = _torch_kmeans(
                flat, n_clusters=self.codebook.shape[0], max_iters=5)
            new_indices = new_indices_flat.reshape(
                self.out_features, self.in_features)
            self.codebook.data.copy_(new_codebook)
        else:
            d = self.vector_dim
            vectors = W.reshape(-1, d)
            new_codebook, new_indices_flat = _torch_vector_kmeans(
                vectors, n_clusters=self.codebook.shape[0], max_iters=5)
            new_indices = new_indices_flat.reshape(
                self.out_features, self.in_features // d)
            self.codebook.data.copy_(new_codebook)

        # Track how many indices changed
        changed = (new_indices != self.indices).sum().item()
        total = self.indices.numel()
        self._last_indices_changed = changed
        self._reassign_count += 1

        self.indices.copy_(new_indices)
        return changed, total

    @torch.no_grad()
    def _reconstruct_full(self) -> torch.Tensor:
        """Reconstruct full W matrix (for reassignment and diagnostics)."""
        W_vq = self.codebook[self.indices.long()]
        if self.vector_dim > 1:
            W = W_vq.reshape(self.out_features, self.in_features)
        else:
            W = W_vq
        if self.has_svd:
            W = W + (self.svd_U * self.svd_s.unsqueeze(0)) @ self.svd_Vt
        return W

    @torch.no_grad()
    def weight_kurtosis(self) -> float:
        """Compute excess kurtosis of the effective weight distribution."""
        W = self._reconstruct_full()
        flat = W.reshape(-1).float()
        mean = flat.mean()
        var = ((flat - mean) ** 2).mean()
        if var < 1e-12:
            return 0.0
        kurt = ((flat - mean) ** 4).mean() / (var ** 2) - 3.0
        return kurt.item()

    @torch.no_grad()
    def codebook_utilization(self) -> float:
        """Fraction of codebook entries that are actually used."""
        used = len(torch.unique(self.indices))
        return used / self.codebook.shape[0]

    @classmethod
    def from_scratch(
        cls,
        in_features: int,
        out_features: int,
        bias: bool = False,
        svd_rank: int = 0,
        n_clusters: int = 256,
        vector_dim: int = 1,
        device: Optional[torch.device] = None,
    ) -> "HelixLinearSTE":
        """Initialize with Kaiming-normal weights, quantized to CDNA v3 format.

        Args:
            vector_dim: VQ group size. 1=scalar (legacy), 2/4/8=grouped (production).
                        in_features must be divisible by vector_dim.
        """
        assert in_features % vector_dim == 0, \
            f"in_features ({in_features}) must be divisible by vector_dim ({vector_dim})"

        W = torch.empty(out_features, in_features, device=device or "cpu")
        nn.init.kaiming_normal_(W, mode="fan_out", nonlinearity="linear")

        if vector_dim == 1:
            # Scalar path: percentile init + searchsorted (existing fast path)
            flat = W.reshape(-1)
            sorted_vals = torch.sort(flat)[0]
            pct_idx = torch.linspace(0, len(sorted_vals) - 1, n_clusters).long()
            codebook = sorted_vals[pct_idx].float()  # [k]

            bucket = torch.searchsorted(codebook, flat).clamp(0, n_clusters - 1)
            left = (bucket - 1).clamp(0)
            dist_right = (flat - codebook[bucket]).abs()
            dist_left = (flat - codebook[left]).abs()
            indices = torch.where(dist_left < dist_right, left, bucket).to(torch.uint8)
            indices = indices.reshape(out_features, in_features)  # [out, in]
        else:
            # Grouped path: d-dimensional k-means
            d = vector_dim
            vectors = W.reshape(-1, d)  # [out * in/d, d]
            codebook, indices = _torch_vector_kmeans(vectors, n_clusters, max_iters=10)
            # codebook: [k, d], indices: [out * in/d] uint8
            indices = indices.reshape(out_features, in_features // d)  # [out, in/d]

        # Optional SVD of quantization residual
        svd_U = svd_s = svd_Vt = None
        if svd_rank > 0:
            W_vq = codebook[indices.long()]
            if vector_dim > 1:
                W_vq = W_vq.reshape(out_features, in_features)
            residual = W - W_vq
            U, s, Vt = torch.linalg.svd(residual, full_matrices=False)
            svd_U = U[:, :svd_rank]
            svd_s = s[:svd_rank]
            svd_Vt = Vt[:svd_rank, :]

        bias_tensor = torch.zeros(out_features, device=device or "cpu") if bias else None

        return cls(
            in_features=in_features,
            out_features=out_features,
            codebook=codebook,
            indices=indices,
            vector_dim=vector_dim,
            svd_rank=svd_rank,
            svd_U=svd_U,
            svd_s=svd_s,
            svd_Vt=svd_Vt,
            bias=bias_tensor,
        )

    def extra_repr(self) -> str:
        parts = [f"in_features={self.in_features}", f"out_features={self.out_features}"]
        parts.append(f"codebook={self.codebook.shape[0]}")
        if self.vector_dim > 1:
            parts.append(f"vector_dim={self.vector_dim}")
        if self.has_svd:
            parts.append(f"svd_rank={self.svd_U.shape[1]}")
        if self.has_lora:
            parts.append(f"lora_rank={self.lora_rank}")
            parts.append(f"lora_alpha={self.lora_alpha}")
        if self.bias is not None:
            parts.append("bias=True")
        return ", ".join(parts)
