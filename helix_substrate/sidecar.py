#!/usr/bin/env python3
"""
hxzo_sidecar.py - HXZO Outlier Sidecar Format

Stores outlier positions and values from hybrid CDNA quantization.
Outliers are weights that fall outside the percentile threshold and
need exact fp16 representation to maintain fidelity.

File Format (HXZO v1):
  [0:4]   Magic: b"HXZO"
  [4:6]   Version: uint16 BE (1)
  [6:8]   Header length: uint16 BE
  [8:N]   Header JSON with threshold_policy, tensor_name, shape, num_outliers
  [N:...] zlib-compressed payload:
          - delta-varint encoded positions
          - fp16 values (raw bytes)

Delta-varint encoding:
  Positions are sorted, then delta-encoded (store difference from previous).
  Each delta is encoded as variable-length integer (7 bits per byte, high bit = continuation).
  This typically achieves 2-3x compression on sparse outlier positions.

Work Order: WO-HYBRID-CDNA-SIDECAR
"""

from __future__ import annotations

import json
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# HXZO format constants
HXZO_MAGIC = b"HXZO"
HXZO_VERSION = 1


# --- Sidecar Cache (WO-CDNA2-KERNEL-01) ---
# Cache decompressed (positions, values, metadata) to avoid repeated I/O

_sidecar_cache: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = {}
_sidecar_cache_stats = {"hits": 0, "misses": 0}


def get_sidecar_cache_stats() -> Dict[str, int]:
    """Return cache hit/miss stats for receipts."""
    return dict(_sidecar_cache_stats)


def clear_sidecar_cache() -> None:
    """Clear cache between sessions."""
    global _sidecar_cache, _sidecar_cache_stats
    _sidecar_cache.clear()
    _sidecar_cache_stats["hits"] = 0
    _sidecar_cache_stats["misses"] = 0


def _encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as a variable-length integer (7 bits per byte)."""
    if value < 0:
        raise ValueError(f"varint requires non-negative value, got {value}")
    parts = []
    while value >= 0x80:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value)
    return bytes(parts)


def _decode_varint_scalar(data: bytes, offset: int) -> Tuple[int, int]:
    """Decode a varint from data at offset. Returns (value, new_offset).

    Scalar fallback for rare long varints (>3 bytes).
    """
    result = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("Truncated varint")
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
        if shift > 63:
            raise ValueError("Varint too large")
    return result, offset


# Alias for backward compatibility
_decode_varint = _decode_varint_scalar


def delta_encode_positions(positions: np.ndarray) -> bytes:
    """
    Delta-encode sorted positions to bytes using varint encoding.

    Args:
        positions: 1D array of sorted position indices (int32/int64)

    Returns:
        Compact byte representation
    """
    if len(positions) == 0:
        return b""

    positions = np.asarray(positions).flatten()
    if len(positions) > 1:
        # Ensure sorted
        if not np.all(positions[:-1] <= positions[1:]):
            positions = np.sort(positions)

    # Delta encode
    deltas = np.zeros(len(positions), dtype=np.int64)
    deltas[0] = positions[0]
    if len(positions) > 1:
        deltas[1:] = np.diff(positions)

    # Varint encode each delta
    parts = []
    for d in deltas:
        parts.append(_encode_varint(int(d)))

    return b"".join(parts)


def delta_decode_positions(data: bytes, count: int) -> np.ndarray:
    """
    Decode delta-varint encoded positions (vectorized).

    Uses NumPy batch operations for 1-3 byte varints (common case),
    with scalar fallback for longer varints (rare).

    Args:
        data: Varint-encoded delta bytes
        count: Number of positions to decode

    Returns:
        1D array of original positions (int64)

    Performance:
        ~50-100x faster than scalar loop for typical outlier distributions.
        Bottleneck was 13.6M scalar calls taking 48s; now <1s.
    """
    if count == 0:
        return np.array([], dtype=np.int64)

    # Convert bytes to numpy array (zero-copy view)
    arr = np.frombuffer(data, dtype=np.uint8)

    # Find terminal bytes (high bit = 0, marks end of each varint)
    is_terminal = (arr & 0x80) == 0
    terminal_indices = np.nonzero(is_terminal)[0]

    if len(terminal_indices) < count:
        raise ValueError(
            f"Not enough varints in data: found {len(terminal_indices)}, expected {count}"
        )

    # Only process the varints we need
    terminal_indices = terminal_indices[:count]

    # Compute start position of each varint
    # First varint starts at 0, subsequent start after previous terminal
    starts = np.empty(count, dtype=np.int64)
    starts[0] = 0
    if count > 1:
        starts[1:] = terminal_indices[:-1] + 1

    # Length of each varint
    lengths = terminal_indices - starts + 1

    # Decode deltas based on varint length
    deltas = np.zeros(count, dtype=np.int64)

    # === Single-byte varints (value 0-127) - most common ===
    mask1 = lengths == 1
    if np.any(mask1):
        deltas[mask1] = arr[terminal_indices[mask1]]

    # === Two-byte varints (value 128-16383) ===
    mask2 = lengths == 2
    if np.any(mask2):
        idx = terminal_indices[mask2]
        # Byte order: [low7 | 0x80] [high7]
        # Value = low7 | (high7 << 7)
        deltas[mask2] = (arr[idx - 1] & 0x7F).astype(np.int64) | (
            arr[idx].astype(np.int64) << 7
        )

    # === Three-byte varints (value 16384-2097151) ===
    mask3 = lengths == 3
    if np.any(mask3):
        idx = terminal_indices[mask3]
        deltas[mask3] = (
            (arr[idx - 2] & 0x7F).astype(np.int64)
            | ((arr[idx - 1] & 0x7F).astype(np.int64) << 7)
            | (arr[idx].astype(np.int64) << 14)
        )

    # === Four-byte varints (value 2097152-268435455) ===
    mask4 = lengths == 4
    if np.any(mask4):
        idx = terminal_indices[mask4]
        deltas[mask4] = (
            (arr[idx - 3] & 0x7F).astype(np.int64)
            | ((arr[idx - 2] & 0x7F).astype(np.int64) << 7)
            | ((arr[idx - 1] & 0x7F).astype(np.int64) << 14)
            | (arr[idx].astype(np.int64) << 21)
        )

    # === Five-byte varints (value 268435456-34359738367) ===
    mask5 = lengths == 5
    if np.any(mask5):
        idx = terminal_indices[mask5]
        deltas[mask5] = (
            (arr[idx - 4] & 0x7F).astype(np.int64)
            | ((arr[idx - 3] & 0x7F).astype(np.int64) << 7)
            | ((arr[idx - 2] & 0x7F).astype(np.int64) << 14)
            | ((arr[idx - 1] & 0x7F).astype(np.int64) << 21)
            | (arr[idx].astype(np.int64) << 28)
        )

    # === Longer varints (rare) - scalar fallback ===
    mask_long = lengths > 5
    if np.any(mask_long):
        long_indices = np.nonzero(mask_long)[0]
        for i in long_indices:
            delta, _ = _decode_varint_scalar(data, int(starts[i]))
            deltas[i] = delta

    # Cumulative sum of deltas gives original positions
    return np.cumsum(deltas)


def write_outlier_sidecar(
    positions: np.ndarray,
    values: np.ndarray,
    tensor_name: str,
    threshold_policy: Dict[str, Any],
    shape: Tuple[int, ...],
    output_path: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write outlier sidecar file in HXZO format.

    Args:
        positions: Flat indices of outlier positions (will be sorted)
        values: Outlier values (will be cast to fp16)
        tensor_name: Name of the tensor
        threshold_policy: Dict with method, percentile, clip_min, clip_max
        shape: Original tensor shape
        output_path: Where to write the .hxzo file
        extra_meta: Optional additional metadata

    Returns:
        Receipt with creation stats
    """
    positions = np.asarray(positions).flatten().astype(np.int64)
    values = np.asarray(values).flatten().astype(np.float16)

    if len(positions) != len(values):
        raise ValueError(f"positions ({len(positions)}) and values ({len(values)}) must have same length")

    # Sort by position
    sort_idx = np.argsort(positions)
    positions = positions[sort_idx]
    values = values[sort_idx]

    # Build header
    header = {
        "schema": "hxzo_outlier_sidecar_v1",
        "tensor_name": tensor_name,
        "shape": list(shape),
        "num_outliers": int(len(positions)),
        "threshold_policy": threshold_policy,
        "value_dtype": "float16",
        "position_encoding": "delta_varint",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if extra_meta:
        header.update(extra_meta)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # Encode positions with delta-varint
    pos_encoded = delta_encode_positions(positions)

    # Values as raw fp16 bytes
    values_bytes = values.tobytes()

    # Combine and compress
    payload = struct.pack(">I", len(pos_encoded)) + pos_encoded + values_bytes
    compressed = zlib.compress(payload, level=9)

    # Build file
    output = bytearray()
    output.extend(HXZO_MAGIC)
    output.extend(struct.pack(">H", HXZO_VERSION))
    output.extend(struct.pack(">H", len(header_bytes)))
    output.extend(header_bytes)
    output.extend(compressed)

    # Write
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(output))

    # Receipt
    receipt = {
        "schema": "write_hxzo_sidecar_v1",
        "tensor_name": tensor_name,
        "shape": list(shape),
        "num_outliers": int(len(positions)),
        "threshold_policy": threshold_policy,
        "pos_encoded_bytes": len(pos_encoded),
        "values_bytes": len(values_bytes),
        "payload_bytes": len(payload),
        "compressed_bytes": len(compressed),
        "total_bytes": len(output),
        "compression_ratio": len(payload) / len(compressed) if len(compressed) > 0 else 0,
        "output_path": str(out_path),
    }

    return receipt


def read_outlier_sidecar(
    sidecar_path: str,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Read outlier sidecar from HXZO file with optional caching.

    Args:
        sidecar_path: Path to .hxzo file
        use_cache: If True, cache and reuse decompressed data (default: True)

    Returns:
        (positions [N] int64, values [N] float16, metadata dict)

    Note (WO-CDNA2-KERNEL-01):
        Caching avoids repeated zlib decompression and varint decoding.
        Use get_sidecar_cache_stats() to check hit/miss rates.
        Use clear_sidecar_cache() to clear between sessions.
    """
    global _sidecar_cache_stats

    path_key = str(Path(sidecar_path).resolve())

    # Check cache first
    if use_cache and path_key in _sidecar_cache:
        _sidecar_cache_stats["hits"] += 1
        return _sidecar_cache[path_key]

    _sidecar_cache_stats["misses"] += 1

    # Read and parse
    path = Path(sidecar_path)
    data = path.read_bytes()

    # Parse header
    if data[:4] != HXZO_MAGIC:
        raise ValueError(f"Not an HXZO file (magic: {data[:4]!r})")

    version = struct.unpack(">H", data[4:6])[0]
    if version != HXZO_VERSION:
        raise ValueError(f"Unsupported HXZO version: {version}")

    header_len = struct.unpack(">H", data[6:8])[0]
    header = json.loads(data[8:8 + header_len].decode("utf-8"))

    # Decompress payload
    compressed = data[8 + header_len:]
    payload = zlib.decompress(compressed)

    # Parse payload: pos_encoded_len (4 bytes) + pos_encoded + values
    pos_encoded_len = struct.unpack(">I", payload[:4])[0]
    pos_encoded = payload[4:4 + pos_encoded_len]
    values_bytes = payload[4 + pos_encoded_len:]

    num_outliers = header.get("num_outliers", 0)

    # Decode positions
    positions = delta_decode_positions(pos_encoded, num_outliers)

    # Decode values
    values = np.frombuffer(values_bytes, dtype=np.float16).copy()  # .copy() for cache safety

    if len(positions) != len(values):
        raise ValueError(f"Position/value count mismatch: {len(positions)} vs {len(values)}")

    metadata = {
        "tensor_name": header.get("tensor_name", "unknown"),
        "shape": tuple(header.get("shape", [])),
        "num_outliers": num_outliers,
        "threshold_policy": header.get("threshold_policy", {}),
        "header": header,
    }

    # Cache if enabled
    if use_cache:
        _sidecar_cache[path_key] = (positions, values, metadata)

    return positions, values, metadata


def inspect_hxzo_header(sidecar_path: str) -> Dict[str, Any]:
    """
    Inspect HXZO header without decompressing payload.

    Args:
        sidecar_path: Path to .hxzo file

    Returns:
        Header metadata
    """
    path = Path(sidecar_path)
    file_size = path.stat().st_size

    with path.open("rb") as f:
        magic = f.read(4)
        if magic != HXZO_MAGIC:
            return {"valid": False, "error": f"Invalid magic: {magic!r}"}

        version = struct.unpack(">H", f.read(2))[0]
        header_len = struct.unpack(">H", f.read(2))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))

    return {
        "valid": True,
        "version": version,
        "file_size_bytes": file_size,
        "compressed_payload_bytes": file_size - 8 - header_len,
        **header,
    }


# Unit test for roundtrip
if __name__ == "__main__":
    import tempfile

    print("=== HXZO Sidecar Format Test ===")
    print()

    # Create test data
    n_elements = 1_000_000
    n_outliers = 1234  # ~0.12%

    # Random positions (sparse)
    rng = np.random.default_rng(42)
    positions = np.sort(rng.choice(n_elements, n_outliers, replace=False)).astype(np.int64)
    values = rng.standard_normal(n_outliers).astype(np.float32)  # Will be cast to fp16

    threshold_policy = {
        "method": "percentile",
        "percentile": 99.9,
        "clip_min": -0.04,
        "clip_max": 0.05,
    }

    with tempfile.NamedTemporaryFile(suffix=".hxzo", delete=False) as f:
        tmp_path = f.name

    print(f"Test: {n_outliers} outliers out of {n_elements} elements ({100*n_outliers/n_elements:.3f}%)")
    print()

    # Write
    receipt = write_outlier_sidecar(
        positions=positions,
        values=values,
        tensor_name="test_tensor",
        threshold_policy=threshold_policy,
        shape=(1000, 1000),
        output_path=tmp_path,
    )

    print("Write receipt:")
    for k, v in receipt.items():
        if k != "output_path":
            print(f"  {k}: {v}")
    print()

    # Read back
    pos_read, val_read, meta = read_outlier_sidecar(tmp_path)

    # Verify positions (exact)
    pos_match = np.array_equal(positions, pos_read)
    print(f"Positions match: {pos_match}")

    # Verify values (fp16 precision)
    values_fp16 = values.astype(np.float16)
    val_match = np.allclose(values_fp16, val_read, rtol=0, atol=0)
    print(f"Values match (fp16): {val_match}")

    # Size analysis
    naive_bytes = n_outliers * (8 + 4)  # int64 + float32
    hxzo_bytes = receipt["total_bytes"]
    print()
    print(f"Naive storage: {naive_bytes:,} bytes ({naive_bytes/1024:.1f} KB)")
    print(f"HXZO storage:  {hxzo_bytes:,} bytes ({hxzo_bytes/1024:.1f} KB)")
    print(f"Savings: {100*(1 - hxzo_bytes/naive_bytes):.1f}%")

    # Cleanup
    Path(tmp_path).unlink()

    print()
    if pos_match and val_match:
        print("PASS: HXZO sidecar roundtrip test")
    else:
        print("FAIL: HXZO sidecar roundtrip test")
