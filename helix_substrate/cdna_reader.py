#!/usr/bin/env python3
"""
cdna_stream_v2.py - CDNAv2 Format with Block Indexing

CDNAv2 adds:
  - Per-block brotli compression (32 rows default)
  - Block index for random access (offset, size, SHA256)
  - Parallel decompression support
  - Per-block verification

Format (.cdna2.hxz):
  [MAGIC: "HXZ2"]              4 bytes
  [VERSION: 1]                 2 bytes
  [HEADER_LEN]                 2 bytes
  [JSON HEADER]                variable (tensor_name, shape, block_rows)
  [CODEBOOK: 256 × f32]        1024 bytes
  [NUM_BLOCKS]                 4 bytes (uint32)
  [BLOCK_INDEX]                48 bytes × num_blocks
  [BLOCK_0: brotli(rows 0-31)]
  [BLOCK_1: brotli(rows 32-63)]
  ...

Block Index Entry (48 bytes):
  offset: u64       (8 bytes) - position in file
  comp_len: u32     (4 bytes) - compressed size
  decomp_len: u32   (4 bytes) - decompressed size
  sha256: bytes32   (32 bytes) - verification anchor

Comparison to v1:
  - v1: zlib whole-file, no random access
  - v2: brotli per-block, random access, SHA256 verification
  - Size: ~0.5-0.8% smaller due to brotli (marginal)
  - Main benefit: Random access for partial tensor loads

Block Rows Sweep (2026-01-24):
  - Tested: 16, 32, 64, 128 on output.weight (122M), ffn_gate (56M), attn_k (4M)
  - Winner: block_rows=16 (lowest latency, +0.24% file overhead vs 128)
  - Verify: 0.62ms avg (9.9x faster than BR=128)
  - Matmul: 0.20ms avg per block
  - Receipt: receipts/blockrows_sweep/sweep_summary.json

Work Order: WO-CDNA-V2-BLOCK-INDEX
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


# --- Codec Registry ---
# Allows swapping brotli for faster codecs (zstd/lz4)

def _brotli_compress(data: bytes, level: int) -> bytes:
    if not HAS_BROTLI:
        raise ImportError("brotli not available")
    return brotli.compress(data, quality=level)

def _brotli_decompress(data: bytes) -> bytes:
    if not HAS_BROTLI:
        raise ImportError("brotli not available")
    return brotli.decompress(data)

def _zstd_compress(data: bytes, level: int) -> bytes:
    if not HAS_ZSTD:
        raise ImportError("zstandard not available. Install: pip install zstandard")
    return zstandard.ZstdCompressor(level=level).compress(data)

def _zstd_decompress(data: bytes) -> bytes:
    if not HAS_ZSTD:
        raise ImportError("zstandard not available. Install: pip install zstandard")
    return zstandard.ZstdDecompressor().decompress(data)

def _lz4_compress(data: bytes, level: int) -> bytes:
    if not HAS_LZ4:
        raise ImportError("lz4 not available. Install: pip install lz4")
    return lz4.compress(data, compression_level=level)

def _lz4_decompress(data: bytes) -> bytes:
    if not HAS_LZ4:
        raise ImportError("lz4 not available. Install: pip install lz4")
    return lz4.decompress(data)


CODEC_COMPRESSORS = {
    "brotli": _brotli_compress,
    "zstd": _zstd_compress,
    "lz4": _lz4_compress,
}

CODEC_DECOMPRESSORS = {
    "brotli": _brotli_decompress,
    "zstd": _zstd_decompress,
    "lz4": _lz4_decompress,
}

CODEC_DEFAULTS = {
    "brotli": 11,  # Best compression (slow)
    "zstd": 3,     # Fast default
    "lz4": 0,      # Default (fastest)
}

# Codec profiles for different use cases (WO-STREAMING-PERF-05)
# Profiles encode both codec settings and block structure
CODEC_PROFILES = {
    "fast": {
        "codec": "zstd",
        "quality": 3,
        "block_rows": 256,
        "description": "Optimized for streaming speed (10% faster than archive)",
    },
    "archive": {
        "codec": "brotli",
        "quality": 11,
        "block_rows": 16,
        "description": "Optimized for smallest file size",
    },
    "balanced": {
        "codec": "zstd",
        "quality": 3,
        "block_rows": 128,
        "description": "Good balance of speed and compatibility",
    },
}


# --- WO-CDNA2-FIDELITY-ROUTER-03: Unified Routing Helpers ---

def _resolve_min_cosine(decode_mode: str, min_cosine: Optional[float]) -> float:
    """Resolve effective min_cosine from mode or explicit value."""
    if min_cosine is not None:
        return float(min_cosine)
    if decode_mode == "accurate":
        return 0.999
    elif decode_mode == "fast":
        return 0.998
    return 0.999  # Default to court-grade


def _should_apply_sidecar(
    fidelity_stats: Dict[str, Any],
    effective_min_cosine: float,
    decode_mode: str,
    min_cosine: Optional[float],
) -> Tuple[bool, Optional[str]]:
    """
    Determine if sidecar should be applied based on truth or fallback heuristic.

    Args:
        fidelity_stats: Dict from header containing cosine_no_sidecar, sidecar_required, etc.
        effective_min_cosine: Resolved min_cosine threshold
        decode_mode: "accurate" or "fast"
        min_cosine: Explicit min_cosine if provided by caller

    Returns:
        (apply_sidecar: bool, skip_reason: str or None)
        skip_reason is one of: "meets_min_cosine", "legacy_heuristic_no", None
    """
    measured_cosine = fidelity_stats.get("cosine_no_sidecar")

    if measured_cosine is not None:
        # Truth-based routing
        if measured_cosine < effective_min_cosine:
            return True, None
        else:
            return False, "meets_min_cosine"
    else:
        # Fallback to v01 heuristic (backward compatibility)
        if (decode_mode == "accurate") or (min_cosine is not None and min_cosine >= 0.999):
            return True, None
        else:
            if fidelity_stats.get("sidecar_required", False):
                return True, None
            else:
                return False, "legacy_heuristic_no"


# CDNAv2 format constants
CDNA2_MAGIC = b"HXZ2"
CDNA2_VERSION = 1
CODEBOOK_SIZE = 256
BLOCK_INDEX_ENTRY_SIZE = 48  # 8 + 4 + 4 + 32 = 48 bytes


class BlockIndexEntry:
    """Single entry in the block index."""

    __slots__ = ("offset", "comp_len", "decomp_len", "sha256")

    def __init__(
        self,
        offset: int,
        comp_len: int,
        decomp_len: int,
        sha256: bytes,
    ):
        self.offset = offset
        self.comp_len = comp_len
        self.decomp_len = decomp_len
        self.sha256 = sha256

    def to_bytes(self) -> bytes:
        """Serialize to 48 bytes."""
        return (
            struct.pack("<Q", self.offset)       # 8 bytes
            + struct.pack("<I", self.comp_len)   # 4 bytes
            + struct.pack("<I", self.decomp_len) # 4 bytes
            + self.sha256                        # 32 bytes
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "BlockIndexEntry":
        """Deserialize from 48 bytes."""
        assert len(data) == BLOCK_INDEX_ENTRY_SIZE
        offset = struct.unpack("<Q", data[0:8])[0]
        comp_len = struct.unpack("<I", data[8:12])[0]
        decomp_len = struct.unpack("<I", data[12:16])[0]
        sha256 = data[16:48]
        return cls(offset, comp_len, decomp_len, sha256)


class CDNAv2Writer:
    """
    Write CDNAv2 format with block indexing.

    Usage:
        writer = CDNAv2Writer()
        receipt = writer.write(
            indices=indices,    # [M, K] uint8
            codebook=codebook,  # [256] float32
            output_path="tensor.cdna2.hxz",
            tensor_name="blk.0.attn_k.weight",
            block_rows=32,
        )
    """

    def __init__(self, compression_quality: int = None, codec: str = "brotli", block_rows: int = 16):
        """
        Initialize writer.

        Args:
            compression_quality: Compression level (default: codec-specific).
                                 brotli: 0-11 (default 11), zstd: 1-22 (default 3), lz4: 0-16 (default 0)
            codec: Compression codec ("brotli", "zstd", "lz4")
            block_rows: Rows per block (default 16 per blockrows sweep)
        """
        if codec not in CODEC_COMPRESSORS:
            raise ValueError(f"Unknown codec: {codec}. Available: {list(CODEC_COMPRESSORS.keys())}")
        self.codec = codec
        self.block_rows = block_rows
        if compression_quality is None:
            compression_quality = CODEC_DEFAULTS.get(codec, 11)
        self.compression_quality = compression_quality

    def write(
        self,
        indices: np.ndarray,
        codebook: np.ndarray,
        output_path: Path,
        tensor_name: str = "weight",
        block_rows: int = None,  # Default from constructor
        extra_meta: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Write CDNAv2 format with block indexing.

        Args:
            indices: [M, K] uint8 codebook indices
            codebook: [256] float32 codebook values
            output_path: Where to write .cdna2.hxz file
            tensor_name: Name of the tensor
            block_rows: Rows per block (default from constructor, typically 16)
            extra_meta: Additional metadata for header

        Returns:
            Receipt with write stats
        """
        # Validate codec availability
        compress_fn = CODEC_COMPRESSORS.get(self.codec)
        if compress_fn is None:
            raise ValueError(f"Unknown codec: {self.codec}")

        output_path = Path(output_path)
        indices = np.asarray(indices, dtype=np.uint8)
        codebook = np.asarray(codebook, dtype=np.float32)

        if indices.ndim == 1:
            # Treat 1D as single row
            indices = indices.reshape(1, -1)

        M, K = indices.shape
        assert len(codebook) == CODEBOOK_SIZE, f"Codebook must be 256 entries, got {len(codebook)}"

        # Use block_rows from constructor if not overridden
        if block_rows is None:
            block_rows = self.block_rows

        # Calculate number of blocks
        num_blocks = (M + block_rows - 1) // block_rows

        # Build header
        header = {
            "schema": "cdna2_hxz_v1",
            "tensor_name": tensor_name,
            "shape": [M, K],
            "dtype": "uint8",
            "codebook_size": CODEBOOK_SIZE,
            "block_rows": block_rows,
            "num_blocks": num_blocks,
            "encoding": self.codec,  # Dynamic codec from writer
            "created_utc": datetime.utcnow().isoformat() + "Z",
        }
        if extra_meta:
            header.update(extra_meta)

        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

        # First pass: compress blocks and build index
        blocks: List[bytes] = []
        block_index: List[BlockIndexEntry] = []

        for block_idx in range(num_blocks):
            start_row = block_idx * block_rows
            end_row = min(start_row + block_rows, M)
            block_indices = indices[start_row:end_row]
            raw_bytes = block_indices.tobytes()

            # Compress with selected codec
            compressed = compress_fn(raw_bytes, self.compression_quality)

            # SHA256 of raw (decompressed) bytes
            sha256 = hashlib.sha256(raw_bytes).digest()

            blocks.append(compressed)
            block_index.append(BlockIndexEntry(
                offset=0,  # Will be filled in second pass
                comp_len=len(compressed),
                decomp_len=len(raw_bytes),
                sha256=sha256,
            ))

        # Calculate offsets
        # File structure:
        #   4 (magic) + 2 (version) + 2 (header_len) + header_len + 1024 (codebook)
        #   + 4 (num_blocks) + 48 * num_blocks (index) + blocks...
        header_overhead = 4 + 2 + 2 + len(header_bytes) + 1024 + 4 + (BLOCK_INDEX_ENTRY_SIZE * num_blocks)
        current_offset = header_overhead

        for i, entry in enumerate(block_index):
            entry.offset = current_offset
            current_offset += entry.comp_len

        # Build file
        output = bytearray()

        # Magic
        output.extend(CDNA2_MAGIC)

        # Version
        output.extend(struct.pack("<H", CDNA2_VERSION))

        # Header length
        output.extend(struct.pack("<H", len(header_bytes)))

        # Header JSON
        output.extend(header_bytes)

        # Codebook (256 × f32 = 1024 bytes)
        output.extend(codebook.tobytes())

        # Number of blocks
        output.extend(struct.pack("<I", num_blocks))

        # Block index
        for entry in block_index:
            output.extend(entry.to_bytes())

        # Blocks
        for block in blocks:
            output.extend(block)

        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bytes(output))

        # Calculate stats
        raw_indices_bytes = M * K
        compressed_total = sum(len(b) for b in blocks)

        receipt = {
            "schema": "create_cdna2_hxz_v1",
            "tensor_name": tensor_name,
            "shape": [M, K],
            "block_rows": block_rows,
            "num_blocks": num_blocks,
            "indices_raw_bytes": raw_indices_bytes,
            "indices_compressed_bytes": compressed_total,
            "codebook_bytes": CODEBOOK_SIZE * 4,
            "index_bytes": BLOCK_INDEX_ENTRY_SIZE * num_blocks,
            "total_bytes": len(output),
            "compression_ratio": raw_indices_bytes / compressed_total if compressed_total > 0 else 0,
            "output_path": str(output_path),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        return receipt


class CDNAv2Reader:
    """
    Streaming reader for CDNAv2 format.

    Supports:
      - Header-only parsing (O(1), no decompression)
      - Block streaming (decompress on demand)
      - Random access to specific rows
      - Per-block SHA256 verification

    Usage:
        reader = CDNAv2Reader("tensor.cdna2.hxz")
        print(reader.shape)  # (4096, 4096)

        # Stream all blocks
        for indices_block, start_row in reader.stream_blocks():
            process(indices_block)

        # Random access
        rows_100_200 = reader.get_rows(100, 200)

        # Verify block
        assert reader.verify_block(5)
    """

    def __init__(self, path: Union[str, Path], cache_size: int = 128):
        """
        Initialize reader.

        Parses header and block index but does NOT decompress any blocks.

        Args:
            path: Path to .cdna2.hxz file
            cache_size: Number of decompressed blocks to cache (default 128).
                        Set to 0 to disable caching.
        """
        self.path = Path(path)
        self._parse_header()

        # Block cache for avoiding double decompression
        # Key: block_idx, Value: (raw_bytes, indices_array)
        # Using OrderedDict for O(1) LRU eviction (Phase 4 of WO-STREAMING-PERF-01)
        self._cache_size = cache_size
        self._block_cache: OrderedDict[int, Tuple[bytes, np.ndarray]] = OrderedDict()
        self._verified_blocks: set = set()  # Track which blocks passed SHA256

    def _parse_header(self) -> None:
        """Parse header, codebook, and block index."""
        with open(self.path, "rb") as f:
            # Magic
            magic = f.read(4)
            if magic != CDNA2_MAGIC:
                raise ValueError(f"Not a CDNAv2 file (magic: {magic!r}, expected {CDNA2_MAGIC!r})")

            # Version
            version = struct.unpack("<H", f.read(2))[0]
            if version != CDNA2_VERSION:
                raise ValueError(f"Unsupported CDNAv2 version: {version}")
            self.version = version

            # Header length
            header_len = struct.unpack("<H", f.read(2))[0]

            # Header JSON
            header_json = f.read(header_len)
            self.header = json.loads(header_json.decode("utf-8"))

            # Extract metadata
            self.tensor_name = self.header.get("tensor_name", "weight")
            self.shape = tuple(self.header["shape"])
            self.rows, self.cols = self.shape
            self.block_rows = self.header.get("block_rows", 32)
            self.num_blocks = self.header.get("num_blocks", 1)
            # Codec (default brotli for backward compatibility with older files)
            self.codec = self.header.get("encoding", "brotli")

            # Codebook (256 × f32 = 1024 bytes)
            codebook_bytes = f.read(CODEBOOK_SIZE * 4)
            self.codebook = np.frombuffer(codebook_bytes, dtype=np.float32).copy()

            # Number of blocks (should match header)
            num_blocks_actual = struct.unpack("<I", f.read(4))[0]
            assert num_blocks_actual == self.num_blocks, (
                f"Block count mismatch: header={self.num_blocks}, actual={num_blocks_actual}"
            )

            # Block index
            self.block_index: List[BlockIndexEntry] = []
            for _ in range(self.num_blocks):
                entry_bytes = f.read(BLOCK_INDEX_ENTRY_SIZE)
                entry = BlockIndexEntry.from_bytes(entry_bytes)
                self.block_index.append(entry)

    def _fetch_block_cached(
        self,
        block_idx: int,
        file_handle: Optional[Any] = None,
        verify: bool = False,
    ) -> Tuple[bytes, np.ndarray]:
        """
        Fetch a single block with LRU caching.

        Args:
            block_idx: Block index to fetch
            file_handle: Optional open file handle (for batched reads)
            verify: Whether to verify SHA256 (only done on first access)

        Returns:
            (raw_bytes, indices_array) - both cached

        Raises:
            ValueError: If verify=True and SHA256 check fails
        """
        # Check cache
        if self._cache_size > 0 and block_idx in self._block_cache:
            # Move to end (most recently used) - O(1) with OrderedDict
            self._block_cache.move_to_end(block_idx)
            return self._block_cache[block_idx]

        # Not in cache - fetch and decompress
        entry = self.block_index[block_idx]

        if file_handle is not None:
            file_handle.seek(entry.offset)
            compressed = file_handle.read(entry.comp_len)
        else:
            with open(self.path, "rb") as f:
                f.seek(entry.offset)
                compressed = f.read(entry.comp_len)

        # Decompress using codec from header
        decompress_fn = CODEC_DECOMPRESSORS.get(self.codec)
        if decompress_fn is None:
            raise ValueError(f"Unknown codec: {self.codec}. Available: {list(CODEC_DECOMPRESSORS.keys())}")
        raw_bytes = decompress_fn(compressed)
        assert len(raw_bytes) == entry.decomp_len

        # Verify if requested and not already verified
        if verify and block_idx not in self._verified_blocks:
            computed_sha = hashlib.sha256(raw_bytes).digest()
            if computed_sha != entry.sha256:
                raise ValueError(f"Block {block_idx} failed SHA256 verification")
            self._verified_blocks.add(block_idx)

        # Convert to array - no .copy() needed because raw_bytes is cached
        # alongside indices (line 516), keeping the buffer alive. The
        # decompressors return immutable bytes objects.
        actual_rows = entry.decomp_len // self.cols
        indices = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(actual_rows, self.cols)

        # Cache if enabled
        if self._cache_size > 0:
            # Evict oldest if at capacity - O(1) with OrderedDict.popitem()
            while len(self._block_cache) >= self._cache_size:
                self._block_cache.popitem(last=False)  # Remove oldest (first) entry

            self._block_cache[block_idx] = (raw_bytes, indices)

        return raw_bytes, indices

    def clear_cache(self) -> None:
        """Clear the block cache."""
        self._block_cache.clear()

    def stream_blocks(
        self,
        start_block: int = 0,
        end_block: Optional[int] = None,
        verify: bool = False,
        file_handle: Optional[Any] = None,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Stream blocks of indices with optional verification and caching.

        Args:
            start_block: First block to stream (default 0)
            end_block: Last block (exclusive, default all)
            verify: Whether to verify SHA256 on first access (default False)
            file_handle: Optional open file handle to reuse (Phase 2 of WO-STREAMING-PERF-01)

        Yields:
            (indices_block [B, K] uint8, start_row_idx)
        """
        # Verify codec is available
        if self.codec not in CODEC_DECOMPRESSORS:
            raise ValueError(f"Unknown codec: {self.codec}")

        if end_block is None:
            end_block = self.num_blocks

        # Phase 2: Reuse file handle if provided, otherwise open new one
        if file_handle is not None:
            for block_idx in range(start_block, end_block):
                _, indices = self._fetch_block_cached(block_idx, file_handle, verify=verify)
                start_row = block_idx * self.block_rows
                yield indices, start_row
        else:
            with open(self.path, "rb") as f:
                for block_idx in range(start_block, end_block):
                    # Use cached fetch
                    _, indices = self._fetch_block_cached(block_idx, f, verify=verify)

                    start_row = block_idx * self.block_rows
                    yield indices, start_row

    def get_rows(
        self,
        start_row: int,
        end_row: int,
    ) -> np.ndarray:
        """
        Random access to specific rows.

        Only decompresses blocks that contain the requested rows.

        Args:
            start_row: First row (inclusive)
            end_row: Last row (exclusive)

        Returns:
            indices [end_row - start_row, K] uint8
        """
        if start_row < 0 or end_row > self.rows or start_row >= end_row:
            raise ValueError(f"Invalid row range [{start_row}, {end_row}) for tensor with {self.rows} rows")

        # Calculate which blocks we need
        start_block = start_row // self.block_rows
        end_block = (end_row - 1) // self.block_rows + 1

        # Collect rows from blocks
        result_rows = []

        for indices_block, block_start_row in self.stream_blocks(start_block, end_block):
            block_end_row = block_start_row + indices_block.shape[0]

            # Calculate overlap
            local_start = max(0, start_row - block_start_row)
            local_end = min(indices_block.shape[0], end_row - block_start_row)

            if local_start < local_end:
                result_rows.append(indices_block[local_start:local_end])

        return np.vstack(result_rows)

    def get_block_compressed(
        self,
        block_idx: int,
        file_handle: Optional[Any] = None,
    ) -> Tuple[bytes, int, int, bytes]:
        """
        Get raw compressed block without decompression.

        For C++ kernel integration (WO-CDNA-CPP-KERNEL-02):
        Returns compressed bytes for fused decode in C++.

        Args:
            block_idx: Block index to fetch
            file_handle: Optional open file handle (for batched reads)

        Returns:
            (compressed_bytes, rows_in_block, cols, expected_sha256)
        """
        entry = self.block_index[block_idx]

        if file_handle is not None:
            file_handle.seek(entry.offset)
            compressed = file_handle.read(entry.comp_len)
        else:
            with open(self.path, "rb") as f:
                f.seek(entry.offset)
                compressed = f.read(entry.comp_len)

        # Calculate actual rows (last block may have fewer)
        if block_idx == self.num_blocks - 1:
            rows_in_block = self.rows - (block_idx * self.block_rows)
        else:
            rows_in_block = self.block_rows

        return compressed, rows_in_block, self.cols, entry.sha256

    def verify_block(self, block_idx: int) -> bool:
        """
        Verify a single block via SHA256.

        Uses cache if available to avoid re-decompression.

        Args:
            block_idx: Block index to verify

        Returns:
            True if SHA256 matches, False otherwise
        """
        # Verify codec is available
        if self.codec not in CODEC_DECOMPRESSORS:
            raise ValueError(f"Unknown codec: {self.codec}")

        if block_idx < 0 or block_idx >= self.num_blocks:
            raise ValueError(f"Block index {block_idx} out of range [0, {self.num_blocks})")

        # If already verified, return True
        if block_idx in self._verified_blocks:
            return True

        # Use cached fetch with verification
        try:
            self._fetch_block_cached(block_idx, verify=True)
            return True
        except ValueError:
            return False

    def verify_all_blocks(self) -> Tuple[int, int, List[int]]:
        """
        Verify all blocks.

        Returns:
            (ok_count, fail_count, failed_block_indices)
        """
        ok_count = 0
        fail_count = 0
        failed = []

        for i in range(self.num_blocks):
            if self.verify_block(i):
                ok_count += 1
            else:
                fail_count += 1
                failed.append(i)

        return ok_count, fail_count, failed

    def get_all_indices(self) -> np.ndarray:
        """
        Load all indices (for compatibility with v1).

        Note: This defeats the streaming purpose. Use stream_blocks() for large tensors.

        Returns:
            indices [M, K] uint8
        """
        return self.get_rows(0, self.rows)

    @property
    def memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        full_tensor_bytes = self.rows * self.cols * 4  # float32
        codebook_bytes = CODEBOOK_SIZE * 4
        index_bytes = BLOCK_INDEX_ENTRY_SIZE * self.num_blocks

        return {
            "shape": list(self.shape),
            "rows": self.rows,
            "cols": self.cols,
            "block_rows": self.block_rows,
            "num_blocks": self.num_blocks,
            "full_tensor_bytes": full_tensor_bytes,
            "codebook_bytes": codebook_bytes,
            "index_bytes": index_bytes,
            "codebook_entries": CODEBOOK_SIZE,
        }

    def stream_rows(
        self,
        start_row: int,
        end_row: int,
        emit_receipt: bool = True,
        verify: bool = False,
        file_handle: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Stream a range of rows with optional receipt emission.

        Only decompresses blocks that contain the requested rows.
        Returns codebook indices (uint8), not dequantized values.

        Args:
            start_row: First row (inclusive)
            end_row: Last row (exclusive)
            emit_receipt: Whether to generate access receipt
            verify: Whether to verify SHA256 on first access (cached)
            file_handle: Optional open file handle to reuse (Phase 2 of WO-STREAMING-PERF-01)

        Returns:
            (indices [end_row - start_row, K] uint8, receipt or None)

        Receipt schema:
        {
            "schema": "cdna2_row_access_v1",
            "tensor_name": "blk.0.attn_k.weight",
            "tensor_shape": [4096, 4096],
            "row_range": [100, 200],
            "rows_loaded": 100,
            "rows_total": 4096,
            "blocks_touched": [3, 4, 5, 6],
            "bytes_decompressed": 51200,
            "bytes_if_full_load": 67108864,
            "savings_factor": 40.96,
            "timestamp_utc": "2026-01-25T..."
        }
        """
        if start_row < 0 or end_row > self.rows or start_row >= end_row:
            raise ValueError(f"Invalid row range [{start_row}, {end_row}) for tensor with {self.rows} rows")

        # Calculate which blocks we need
        start_block = start_row // self.block_rows
        end_block = (end_row - 1) // self.block_rows + 1

        # Track blocks touched and bytes decompressed
        blocks_touched = list(range(start_block, end_block))
        bytes_decompressed = 0

        # Collect rows from blocks
        result_rows = []

        for indices_block, block_start_row in self.stream_blocks(start_block, end_block, verify=verify, file_handle=file_handle):
            bytes_decompressed += indices_block.nbytes

            # Calculate overlap
            local_start = max(0, start_row - block_start_row)
            local_end = min(indices_block.shape[0], end_row - block_start_row)

            if local_start < local_end:
                result_rows.append(indices_block[local_start:local_end])

        indices = np.vstack(result_rows)

        # Build receipt
        receipt = None
        if emit_receipt:
            bytes_if_full = self.rows * self.cols  # Full tensor as uint8
            receipt = {
                "schema": "cdna2_row_access_v1",
                "tensor_name": self.tensor_name,
                "tensor_shape": list(self.shape),
                "row_range": [start_row, end_row],
                "rows_loaded": end_row - start_row,
                "rows_total": self.rows,
                "blocks_touched": blocks_touched,
                "bytes_decompressed": bytes_decompressed,
                "bytes_if_full_load": bytes_if_full,
                "savings_factor": bytes_if_full / bytes_decompressed if bytes_decompressed > 0 else 0,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            }

        return indices, receipt

    def stream_rows_dequant(
        self,
        start_row: int,
        end_row: int,
        sidecar_path: Optional[Path] = None,
        decode_mode: str = "accurate",  # WO-CDNA2-FIDELITY-ROUTER-01
        min_cosine: Optional[float] = None,  # WO-CDNA2-FIDELITY-ROUTER-02
        emit_receipt: bool = True,
        verify: bool = False,
        file_handle: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Stream rows and dequantize to float32, applying sidecar based on quality contract.

        Args:
            start_row: First row (inclusive)
            end_row: Last row (exclusive)
            sidecar_path: Optional path to HXZO sidecar for outlier correction
            decode_mode: "accurate" or "fast" (convenience preset for min_cosine)
                - "accurate": min_cosine default = 0.999 (court-grade)
                - "fast": min_cosine default = 0.998 (speed-leaning)
            min_cosine: Explicit quality contract (WO-CDNA2-FIDELITY-ROUTER-02)
                - If provided, overrides decode_mode
                - Sidecar applied if header.cosine_no_sidecar < min_cosine
            emit_receipt: Whether to generate access receipt
            verify: Whether to verify SHA256 on first access (cached)
            file_handle: Optional open file handle to reuse (Phase 2 of WO-STREAMING-PERF-01)

        Returns:
            (weights [end_row - start_row, K] float32, receipt or None)

        Note: Receipt always contains timing info (t_indices_ms, t_dequant_ms)
        even when emit_receipt=False, to support Phase 0.5 timing split.
        """
        # Phase 0.5: Time I/O + decompress (stream_rows call)
        t_indices_start = time.perf_counter()
        indices, receipt = self.stream_rows(start_row, end_row, emit_receipt=emit_receipt, verify=verify, file_handle=file_handle)
        t_indices_ms = (time.perf_counter() - t_indices_start) * 1000

        # Phase 0.5: Time dequantization (codebook gather)
        t_dequant_start = time.perf_counter()
        weights = self.codebook[indices]
        t_dequant_ms = (time.perf_counter() - t_dequant_start) * 1000

        # WO-CDNA2-FIDELITY-ROUTER-03: Use unified routing helpers
        sidecar_applied = False
        sidecar_corrections = 0
        sidecar_skip_reason = None
        fidelity = self.header.get("fidelity_stats", {})
        effective_min_cosine = _resolve_min_cosine(decode_mode, min_cosine)

        apply_sidecar_now = False
        if sidecar_path is not None:
            apply_sidecar_now, sidecar_skip_reason = _should_apply_sidecar(
                fidelity, effective_min_cosine, decode_mode, min_cosine
            )

        if apply_sidecar_now and sidecar_path is not None:
            from helix_cdc.regrow.hxzo_sidecar import read_outlier_sidecar

            positions, values, meta = read_outlier_sidecar(str(sidecar_path))

            # Convert flat positions to row indices within our range
            # positions are flat indices into the full tensor
            for i in range(len(positions)):
                pos = positions[i]
                row = pos // self.cols
                col = pos % self.cols

                # Check if this position is within our loaded row range
                if start_row <= row < end_row:
                    local_row = row - start_row
                    weights[local_row, col] = float(values[i])
                    sidecar_corrections += 1

            sidecar_applied = True

        # Phase 0.5: Always return timing info, even when emit_receipt=False
        # This allows aggregation in stream_xw_matmul without full receipt overhead
        if receipt is None:
            receipt = {}

        # Always add timing (minimal overhead)
        receipt["t_indices_ms"] = round(t_indices_ms, 3)
        receipt["t_dequant_ms"] = round(t_dequant_ms, 3)

        # Full receipt fields only when emit_receipt=True
        if emit_receipt:
            receipt["dequantized"] = True
            receipt["output_dtype"] = "float32"
            receipt["output_bytes"] = weights.nbytes
            receipt["sidecar_applied"] = sidecar_applied
            # WO-CDNA2-FIDELITY-ROUTER-02: Include min_cosine contract and routing info
            receipt["decode_mode"] = decode_mode
            receipt["min_cosine"] = effective_min_cosine
            receipt["fidelity_stats"] = self.header.get("fidelity_stats", {})
            if sidecar_applied:
                receipt["sidecar_path"] = str(sidecar_path)
                receipt["sidecar_corrections"] = sidecar_corrections
            # WO-CDNA2-FIDELITY-ROUTER-03: Use sidecar_skip_reason instead of sidecar_skipped_by_mode
            if sidecar_skip_reason:
                receipt["sidecar_skip_reason"] = sidecar_skip_reason
            # WO-CDNA2-FIDELITY-ROUTER-03: Add routing_basis for forensics
            if fidelity.get("cosine_no_sidecar") is not None:
                receipt["routing_basis"] = "truth"
            else:
                receipt["routing_basis"] = "legacy_heuristic"

        return weights, receipt


    def stream_rows_matmul(
        self,
        x: np.ndarray,
        start_row: int,
        end_row: int,
        sidecar_path: Optional[Path] = None,
        decode_mode: str = "accurate",  # WO-CDNA2-FIDELITY-ROUTER-01
        min_cosine: Optional[float] = None,  # WO-CDNA2-FIDELITY-ROUTER-02
        verify: bool = True,
        emit_receipt: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Compute W[start_row:end_row] @ x using only needed blocks.

        This is the core "compute-while-selective" primitive. Only decompresses
        blocks that contain the requested rows, then performs matmul using the
        CDNA dot product trick (bincount + codebook matmul).

        Args:
            x: Input vector [K] float32
            start_row: First row (inclusive)
            end_row: Last row (exclusive)
            sidecar_path: Optional path to HXZO sidecar for outlier correction
            decode_mode: "accurate" or "fast" (convenience preset for min_cosine)
                - "accurate": min_cosine default = 0.999 (court-grade)
                - "fast": min_cosine default = 0.998 (speed-leaning)
            min_cosine: Explicit quality contract (WO-CDNA2-FIDELITY-ROUTER-02)
                - If provided, overrides decode_mode
                - Sidecar applied if header.cosine_no_sidecar < min_cosine
            verify: If True, verify SHA256 of each touched block
            emit_receipt: Whether to generate receipt

        Returns:
            (y_partial [end_row - start_row], receipt or None)

        Receipt schema: cdna2_row_range_matmul_v1

        Example:
            >>> reader = CDNAv2Reader("ffn_down.cdna2.hxz")
            >>> x = np.random.randn(14336).astype(np.float32)
            >>> y_partial, receipt = reader.stream_rows_matmul(x, 100, 200)
            >>> print(y_partial.shape)  # (100,)
            >>> print(receipt["savings_factor"])  # e.g., 40.96x
        """
        import time as _time

        t0 = _time.perf_counter()

        if start_row < 0 or end_row > self.rows or start_row >= end_row:
            raise ValueError(
                f"Invalid row range [{start_row}, {end_row}) for tensor with {self.rows} rows"
            )

        x = np.asarray(x, dtype=np.float32).ravel()
        if len(x) != self.cols:
            raise ValueError(
                f"Input dimension {len(x)} != tensor columns {self.cols}"
            )

        # Calculate which blocks we need
        start_block = start_row // self.block_rows
        end_block = (end_row - 1) // self.block_rows + 1

        # Track blocks touched and verification
        blocks_touched = list(range(start_block, end_block))
        blocks_verified = 0
        verify_details = {}
        bytes_decompressed = 0

        # Collect partial results
        y_chunks = []

        # WO-CDNA2-FIDELITY-ROUTER-03: Use unified routing helpers
        sidecar_positions = None
        sidecar_values = None
        sidecar_corrections = 0
        sidecar_skip_reason = None
        fidelity = self.header.get("fidelity_stats", {})
        effective_min_cosine = _resolve_min_cosine(decode_mode, min_cosine)

        if sidecar_path is not None:
            apply_sidecar_now, sidecar_skip_reason = _should_apply_sidecar(
                fidelity, effective_min_cosine, decode_mode, min_cosine
            )

            if apply_sidecar_now:
                from helix_cdc.regrow.hxzo_sidecar import read_outlier_sidecar
                sidecar_positions, sidecar_values, _ = read_outlier_sidecar(str(sidecar_path))

        # Process each block
        with open(self.path, "rb") as f:
            for block_idx in range(start_block, end_block):
                entry = self.block_index[block_idx]

                # Seek to block and read
                f.seek(entry.offset)
                compressed = f.read(entry.comp_len)

                # Decompress using codec from header
                decompress_fn = CODEC_DECOMPRESSORS.get(self.codec)
                if decompress_fn is None:
                    raise ValueError(f"Unknown codec: {self.codec}")
                raw_bytes = decompress_fn(compressed)

                bytes_decompressed += len(raw_bytes)

                # Verify if requested
                if verify:
                    computed_sha = hashlib.sha256(raw_bytes).digest()
                    stored_sha = entry.sha256
                    match = computed_sha == stored_sha

                    verify_details[str(block_idx)] = {
                        "stored": stored_sha.hex()[:16] + "...",
                        "computed": computed_sha.hex()[:16] + "...",
                        "match": match,
                    }

                    if not match:
                        raise ValueError(f"Block {block_idx} failed SHA256 verification")

                    blocks_verified += 1

                # Convert to indices
                block_start_row = block_idx * self.block_rows
                actual_rows = entry.decomp_len // self.cols
                indices = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
                indices = indices.reshape(actual_rows, self.cols)

                # Calculate which rows from this block we need
                local_start = max(0, start_row - block_start_row)
                local_end = min(actual_rows, end_row - block_start_row)

                if local_start >= local_end:
                    continue

                # Get the needed slice of indices
                row_indices = indices[local_start:local_end]

                # Apply sidecar corrections if available
                # We need to compute y = W @ x where W has corrections
                # Dequantize first, apply corrections, then matmul
                weights_slice = self.codebook[row_indices]

                if sidecar_positions is not None:
                    global_start_row = block_start_row + local_start
                    global_end_row = block_start_row + local_end

                    for i in range(len(sidecar_positions)):
                        pos = sidecar_positions[i]
                        row = pos // self.cols
                        col = pos % self.cols

                        if global_start_row <= row < global_end_row:
                            local_row = row - global_start_row
                            weights_slice[local_row, col] = float(sidecar_values[i])
                            sidecar_corrections += 1

                # Matmul: y_chunk = W_slice @ x
                y_chunk = weights_slice @ x
                y_chunks.append(y_chunk)

        # Concatenate results
        y = np.concatenate(y_chunks)

        t1 = _time.perf_counter()
        duration_ms = (t1 - t0) * 1000

        # Build receipt
        receipt = None
        if emit_receipt:
            # Compute hashes for proof
            input_x_sha256 = hashlib.sha256(x.tobytes()).hexdigest()
            output_y_sha256 = hashlib.sha256(y.tobytes()).hexdigest()

            # Memory comparison
            bytes_if_full = self.rows * self.cols  # Full tensor as uint8
            full_tensor_float32 = self.rows * self.cols * 4  # As float32
            row_range_float32 = (end_row - start_row) * self.cols * 4

            receipt = {
                "schema": "cdna2_row_range_matmul_v1",
                "tensor_name": self.tensor_name,
                "tensor_shape": list(self.shape),
                "row_range": [start_row, end_row],
                "rows_computed": end_row - start_row,
                "rows_total": self.rows,
                "blocks_touched": blocks_touched,
                "blocks_verified": blocks_verified if verify else 0,
                "verify_details": verify_details if verify else {},
                "compute_proof": {
                    "input_x_sha256": input_x_sha256,
                    "output_y_sha256": output_y_sha256,
                },
                "sidecar_applied": sidecar_positions is not None,
                "sidecar_corrections": sidecar_corrections,
                # WO-CDNA2-FIDELITY-ROUTER-02: Include min_cosine contract and routing info
                "decode_mode": decode_mode,
                "min_cosine": effective_min_cosine,
                "fidelity_stats": self.header.get("fidelity_stats", {}),
                "memory": {
                    "bytes_decompressed": bytes_decompressed,
                    "bytes_if_full_load": bytes_if_full,
                    "full_tensor_float32_bytes": full_tensor_float32,
                    "row_range_float32_bytes": row_range_float32,
                    "savings_factor": round(bytes_if_full / bytes_decompressed, 2) if bytes_decompressed > 0 else 0,
                },
                "timing": {
                    "total_ms": round(duration_ms, 3),
                },
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            }

            if sidecar_positions is not None:
                receipt["sidecar_path"] = str(sidecar_path)
            # WO-CDNA2-FIDELITY-ROUTER-03: Use sidecar_skip_reason instead of sidecar_skipped_by_mode
            if sidecar_skip_reason:
                receipt["sidecar_skip_reason"] = sidecar_skip_reason
            # WO-CDNA2-FIDELITY-ROUTER-03: Add routing_basis for forensics
            if fidelity.get("cosine_no_sidecar") is not None:
                receipt["routing_basis"] = "truth"
            else:
                receipt["routing_basis"] = "legacy_heuristic"

        return y, receipt


def _load_cdna_auto_uncached(path: Path) -> Union["CDNAv1Reader", CDNAv2Reader]:
    """
    Auto-detect CDNA format and return appropriate reader (uncached).

    Internal implementation - use load_cdna_auto() for cached access.
    """
    from helix_cdc.regrow.cdna_stream import CDNAStreamReader as CDNAv1Reader

    with open(path, "rb") as f:
        magic = f.read(4)

    if magic == CDNA2_MAGIC:
        return CDNAv2Reader(path)
    elif magic == b"HXZC":
        return CDNAv1Reader(str(path))
    else:
        raise ValueError(f"Unknown CDNA format: magic={magic!r}")


@lru_cache(maxsize=256)
def _get_cached_reader(path_str: str) -> Union["CDNAv1Reader", CDNAv2Reader]:
    """
    Cached reader factory (WO-CDNA2-KERNEL-01).

    Avoids re-parsing headers and block indices for the same file.
    Key is string path for hashability.
    """
    return _load_cdna_auto_uncached(Path(path_str))


def load_cdna_auto(path: Union[str, Path]) -> Union["CDNAv1Reader", CDNAv2Reader]:
    """
    Auto-detect CDNA format and return cached reader.

    Args:
        path: Path to .hxz file

    Returns:
        CDNAv1Reader or CDNAv2Reader (cached instance)

    Note:
        Readers are cached by path string. Use clear_reader_cache() to clear.
        Cache stats available via _get_cached_reader.cache_info().
    """
    return _get_cached_reader(str(Path(path).resolve()))


def clear_reader_cache() -> None:
    """Clear the CDNA reader cache."""
    _get_cached_reader.cache_clear()


def get_reader_cache_info() -> Dict[str, Any]:
    """Get reader cache statistics for receipts."""
    info = _get_cached_reader.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
    }


def convert_v1_to_v2(
    v1_path: Path,
    v2_path: Path,
    block_rows: int = 16,  # Default 16 per blockrows sweep (2026-01-24)
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Convert CDNAv1 file to CDNAv2 format.

    Args:
        v1_path: Path to CDNAv1 .hxz file
        v2_path: Path for output CDNAv2 .cdna2.hxz file
        block_rows: Rows per block
        verify: Verify conversion by comparing indices

    Returns:
        Conversion receipt
    """
    from helix_cdc.regrow.cdna_stream import CDNAStreamReader as CDNAv1Reader

    # Read v1
    v1_reader = CDNAv1Reader(str(v1_path))
    v1_indices_blob = v1_reader._ensure_indices_blob()
    v1_indices = np.frombuffer(v1_indices_blob, dtype=np.uint8).reshape(v1_reader.rows, v1_reader.cols)
    v1_codebook = v1_reader.codebook.copy()

    # Write v2
    writer = CDNAv2Writer()
    write_receipt = writer.write(
        indices=v1_indices,
        codebook=v1_codebook,
        output_path=v2_path,
        tensor_name=v1_reader.tensor_name,
        block_rows=block_rows,
    )

    # Verify
    verification = {"verified": False, "match": None}
    if verify:
        v2_reader = CDNAv2Reader(v2_path)
        v2_indices = v2_reader.get_all_indices()

        indices_match = np.array_equal(v1_indices, v2_indices)
        codebook_match = np.allclose(v1_codebook, v2_reader.codebook)

        verification = {
            "verified": True,
            "indices_match": bool(indices_match),
            "codebook_match": bool(codebook_match),
            "match": indices_match and codebook_match,
        }

    receipt = {
        "schema": "cdna_v1_to_v2_conversion",
        "v1_path": str(v1_path),
        "v2_path": str(v2_path),
        "v1_bytes": v1_path.stat().st_size,
        "v2_bytes": v2_path.stat().st_size,
        "size_diff_bytes": v2_path.stat().st_size - v1_path.stat().st_size,
        "block_rows": block_rows,
        "write_receipt": write_receipt,
        "verification": verification,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    return receipt


# Convenience function for CDNA matmul using v2 format
def cdna2_matmul_row(
    indices_row: np.ndarray,
    codebook: np.ndarray,
    x: np.ndarray,
) -> float:
    """Compute one row of y = W @ x using CDNA (no decoding)."""
    n_bins = len(codebook)
    weighted_x = np.bincount(
        indices_row,
        weights=x,
        minlength=n_bins
    ).astype(np.float32)
    return float(np.dot(codebook, weighted_x))


def cdna2_matmul_block(
    indices_block: np.ndarray,
    codebook: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Compute block of y = W @ x using CDNA (no decoding).

    Uses per-row np.bincount which is already highly optimized in NumPy.
    The bincount approach avoids materializing the full weight matrix.
    """
    B = indices_block.shape[0]
    y_block = np.zeros(B, dtype=np.float32)
    for i in range(B):
        y_block[i] = cdna2_matmul_row(indices_block[i], codebook, x)
    return y_block


def stream_cdna2_matmul(
    hxz_path: Union[str, Path],
    x: np.ndarray,
    block_rows: int = 32,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute y = W @ x where W is CDNAv2-compressed.

    Args:
        hxz_path: Path to CDNAv2 .cdna2.hxz file
        x: Input vector [K]
        block_rows: Block size (should match file's block_rows for efficiency)

    Returns:
        (y [M], stats)
    """
    reader = CDNAv2Reader(hxz_path)

    x = np.asarray(x, dtype=np.float32).ravel()
    if len(x) != reader.cols:
        raise ValueError(
            f"Input dimension mismatch: W is [{reader.rows}, {reader.cols}], "
            f"x has {len(x)} elements"
        )

    y_chunks = []
    blocks_processed = 0
    max_block_bytes = 0

    for indices_block, start_idx in reader.stream_blocks():
        y_chunk = cdna2_matmul_block(indices_block, reader.codebook, x)
        y_chunks.append(y_chunk)
        blocks_processed += 1
        max_block_bytes = max(max_block_bytes, indices_block.nbytes)

    y = np.concatenate(y_chunks)

    float32_touched = CODEBOOK_SIZE * 4 + len(x) * 4 + len(y) * 4
    float32_dense = reader.rows * reader.cols * 4

    stats = {
        "tensor_name": reader.tensor_name,
        "tensor_shape": list(reader.shape),
        "block_rows": reader.block_rows,
        "blocks_processed": blocks_processed,
        "max_block_bytes_uint8": max_block_bytes,
        "codebook_bytes": CODEBOOK_SIZE * 4,
        "full_tensor_bytes_if_decoded": reader.rows * reader.cols * 4,
        "float32_touched_cdna": float32_touched,
        "float32_touched_dense": float32_dense,
        "memory_reduction_factor": float32_dense / float32_touched,
        "w_decoded_to_float32": False,
    }

    return y, stats


if __name__ == "__main__":
    print("=== CDNAv2 Stream Test ===")
    print()
    print("CDNAv2 adds to CDNA:")
    print("  - Per-block brotli compression (32 rows default)")
    print("  - Block index for random access")
    print("  - Per-block SHA256 verification")
    print()
    print("Usage:")
    print("  from helix_cdc.regrow.cdna_stream_v2 import CDNAv2Reader, stream_cdna2_matmul")
    print("  reader = CDNAv2Reader('weight.cdna2.hxz')")
    print("  y, stats = stream_cdna2_matmul('weight.cdna2.hxz', x)")
    print()
