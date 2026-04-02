"""12-bit index packing for k>256 codebooks.

When k=4096 (2D VQ), each index needs log2(4096)=12 bits, but uint16 uses 16.
Packing pairs of 12-bit values into 3 bytes saves 25% on index storage.

Storage: 6 bits/weight (2D VQ) vs 8 bits/weight (uint16), closing gap to bnb 4-bit.

Packing scheme (two 12-bit values a,b → 3 bytes):
    byte0 = a[7:0]
    byte1 = a[11:8] | b[3:0]<<4
    byte2 = b[11:4]

Work Order: WO-12BIT-PACK-01
"""

from __future__ import annotations

import torch


def pack_12bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 12-bit index values into 3-byte pairs.

    Args:
        indices: int16 or int32 tensor, values in [0, 4095], total elements must be even.
                 Can be any shape — flattened internally.

    Returns:
        uint8 tensor of length N*3//2 (1D)
    """
    flat = indices.reshape(-1).to(torch.int32)
    n = flat.shape[0]
    if n % 2 != 0:
        raise ValueError(f"Total index count must be even for 12-bit packing, got {n}")
    if n > 0 and flat.max().item() >= 4096:
        raise ValueError(f"Values must be < 4096 for 12-bit packing, got max {flat.max().item()}")

    a = flat[0::2]
    b = flat[1::2]

    packed = torch.empty(n * 3 // 2, dtype=torch.uint8, device=indices.device)
    packed[0::3] = (a & 0xFF).to(torch.uint8)
    packed[1::3] = (((a >> 8) & 0x0F) | ((b & 0x0F) << 4)).to(torch.uint8)
    packed[2::3] = ((b >> 4) & 0xFF).to(torch.uint8)

    return packed


def unpack_12bit(packed: torch.Tensor, n_values: int) -> torch.Tensor:
    """Unpack 3-byte pairs back to 12-bit values.

    Args:
        packed: uint8 tensor of length n_values*3//2
        n_values: number of original indices (must be even)

    Returns:
        int16 tensor of length n_values
    """
    if n_values % 2 != 0:
        raise ValueError(f"n_values must be even, got {n_values}")

    byte0 = packed[0::3].to(torch.int32)
    byte1 = packed[1::3].to(torch.int32)
    byte2 = packed[2::3].to(torch.int32)

    a = byte0 | ((byte1 & 0x0F) << 8)
    b = ((byte1 >> 4) & 0x0F) | (byte2 << 4)

    result = torch.empty(n_values, dtype=torch.int16, device=packed.device)
    result[0::2] = a.to(torch.int16)
    result[1::2] = b.to(torch.int16)
    return result


def unpack_12bit_rows(packed: torch.Tensor, start_row: int, end_row: int,
                      cols: int) -> torch.Tensor:
    """Unpack specific rows from a packed 2D index matrix.

    The packed tensor represents a [total_rows, cols] matrix flattened and
    packed. This extracts rows [start_row:end_row] without unpacking everything.

    Args:
        packed: uint8 1D tensor (full packed matrix)
        start_row: first row (inclusive)
        end_row: last row (exclusive)
        cols: number of columns per row (must be even)

    Returns:
        int16 tensor of shape [end_row - start_row, cols]
    """
    bytes_per_row = cols * 3 // 2
    byte_start = start_row * bytes_per_row
    byte_end = end_row * bytes_per_row
    n_values = (end_row - start_row) * cols
    return unpack_12bit(packed[byte_start:byte_end], n_values).reshape(end_row - start_row, cols)
