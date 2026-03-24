"""
CDNA v3 tensor reader.

Reads per-tensor directories written by CDNAv3Writer and reconstructs tensors.
Supports full reconstruction and block-level partial reconstruction for streaming.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from helix_substrate.generate_sidecars_v3 import read_sidecar_v3
from helix_substrate.morpho_codec import morpho_decode, morpho_decode_block


class CDNAv3Reader:
    """Read a v3 tensor directory and reconstruct the tensor."""

    def __init__(self, tensor_dir: Path):
        self.tensor_dir = Path(tensor_dir)
        if not self.tensor_dir.exists():
            raise FileNotFoundError(f"Tensor directory not found: {tensor_dir}")

        meta_path = self.tensor_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {tensor_dir}")

        self._meta = json.loads(meta_path.read_text())
        self._codebook = None
        self._indices = None
        self._sidecar = None

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._meta["shape"])

    @property
    def tensor_name(self) -> str:
        return self._meta["tensor_name"]

    @property
    def has_sidecar(self) -> bool:
        return (self.tensor_dir / "sidecar.npz").exists()

    def _load_codebook(self) -> np.ndarray:
        if self._codebook is None:
            self._codebook = np.load(self.tensor_dir / "codebook.npy")
        return self._codebook

    def _load_indices(self) -> np.ndarray:
        if self._indices is None:
            rows, cols = self.shape
            raw = np.fromfile(self.tensor_dir / "indices.bin", dtype=np.uint8)
            self._indices = raw.reshape(rows, cols)
        return self._indices

    def _load_sidecar(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._sidecar is None:
            sidecar_path = self.tensor_dir / "sidecar.npz"
            if sidecar_path.exists():
                self._sidecar = read_sidecar_v3(sidecar_path)
            else:
                return None
        return self._sidecar

    @property
    def is_morpho(self) -> bool:
        return self._meta.get("storage_mode") == "morpho"

    @property
    def has_svd_residual(self) -> bool:
        return (self.tensor_dir / "svd_U.npy").exists()

    def _load_svd_factors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load SVD residual factors (U_r, s_r, Vt_r)."""
        U = np.load(self.tensor_dir / "svd_U.npy")
        s = np.load(self.tensor_dir / "svd_s.npy")
        Vt = np.load(self.tensor_dir / "svd_Vt.npy")
        return U, s, Vt

    def reconstruct(self) -> np.ndarray:
        """
        Fully reconstruct the tensor.

        For morpho: re-grows from seed via wave dynamics.
        For codebook: loads codebook, maps indices, patches sidecar.

        Returns:
            Reconstructed tensor as float32
        """
        if self.is_morpho:
            return morpho_decode(self.tensor_dir)

        codebook = self._load_codebook()
        indices = self._load_indices()
        tensor = codebook[indices]

        sidecar = self._load_sidecar()
        if sidecar is not None:
            positions, values = sidecar
            flat = tensor.ravel().copy()
            flat[positions] = values
            tensor = flat.reshape(self.shape)

        # SVD residual correction
        if self.has_svd_residual:
            U, s, Vt = self._load_svd_factors()
            tensor = tensor + (U * s[None, :]) @ Vt

        return tensor

    def reconstruct_block(self, start_row: int, end_row: int) -> np.ndarray:
        """
        Reconstruct a block of rows (for streaming compatibility).

        Args:
            start_row: First row (inclusive)
            end_row: Last row (exclusive)

        Returns:
            Tensor block [end_row - start_row, cols] as float32
        """
        if self.is_morpho:
            return morpho_decode_block(self.tensor_dir, start_row, end_row)

        codebook = self._load_codebook()
        indices = self._load_indices()
        block = codebook[indices[start_row:end_row]]

        sidecar = self._load_sidecar()
        if sidecar is not None:
            positions, values = sidecar
            rows, cols = self.shape
            # Filter sidecar entries to this row range
            flat_start = start_row * cols
            flat_end = end_row * cols
            mask = (positions >= flat_start) & (positions < flat_end)
            if np.any(mask):
                local_pos = positions[mask] - flat_start
                local_vals = values[mask]
                flat_block = block.ravel().copy()
                flat_block[local_pos] = local_vals
                block = flat_block.reshape(end_row - start_row, cols)

        # SVD residual correction (row-sliced)
        if self.has_svd_residual:
            U, s, Vt = self._load_svd_factors()
            block = block + (U[start_row:end_row] * s[None, :]) @ Vt

        return block

    def get_stats(self) -> dict:
        """Read and return stats.json."""
        stats_path = self.tensor_dir / "stats.json"
        if not stats_path.exists():
            return {}
        return json.loads(stats_path.read_text())

    def verify_codebook(self) -> bool:
        """Verify codebook SHA256 against meta.json (or config hash for morpho)."""
        if self.is_morpho:
            config_path = self.tensor_dir / "morpho_config.json"
            if not config_path.exists():
                return False
            config = json.loads(config_path.read_text())
            stored_sha = config.get("config_sha256", "")
            # Re-derive: remove the hash itself, recompute
            check = {k: v for k, v in config.items() if k != "config_sha256"}
            actual = hashlib.sha256(
                json.dumps(check, sort_keys=True).encode()
            ).hexdigest()
            return actual == stored_sha

        codebook = self._load_codebook()
        actual = hashlib.sha256(codebook.tobytes()).hexdigest()
        expected = self._meta.get("codebook_sha256", "")
        return actual == expected
