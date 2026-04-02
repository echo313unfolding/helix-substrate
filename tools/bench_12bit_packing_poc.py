#!/usr/bin/env python3
"""POC: 12-bit packed index storage for k=4096 2D VQ.

Current: uint16 (16 bits) per index → 8 bits/weight for 2D VQ
Packed:  12 bits per index → 6 bits/weight for 2D VQ

Packing scheme: pairs of 12-bit values into 3 bytes (24 bits).
  byte0 = a[7:0]
  byte1 = a[11:8] | b[3:0]<<4
  byte2 = b[11:4]

Savings: 25% on index storage (1.5 bytes vs 2 bytes per index).

Receipt: WO-12BIT-PACK-01
"""
import time
import resource
import platform
import numpy as np
import torch

t_start = time.time()
cpu_start = time.process_time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')


# ---------------------------------------------------------------------------
# Core packing/unpacking
# ---------------------------------------------------------------------------

def pack_12bit(indices: np.ndarray) -> np.ndarray:
    """Pack array of 12-bit values (0..4095) into 3-byte pairs.

    Input:  uint16 array of length N (must be even, values in [0, 4095])
    Output: uint8 array of length N * 3 // 2
    """
    assert indices.dtype == np.uint16
    assert len(indices) % 2 == 0, f"Length must be even, got {len(indices)}"
    assert indices.max() < 4096, f"Values must be < 4096, got max {indices.max()}"

    n_pairs = len(indices) // 2
    a = indices[0::2]  # even indices
    b = indices[1::2]  # odd indices

    packed = np.empty(n_pairs * 3, dtype=np.uint8)
    packed[0::3] = (a & 0xFF).astype(np.uint8)
    packed[1::3] = (((a >> 8) & 0x0F) | ((b & 0x0F) << 4)).astype(np.uint8)
    packed[2::3] = ((b >> 4) & 0xFF).astype(np.uint8)

    return packed


def unpack_12bit(packed: np.ndarray, n_values: int) -> np.ndarray:
    """Unpack 3-byte pairs back to 12-bit values.

    Input:  uint8 array of length n_values * 3 // 2
    Output: uint16 array of length n_values
    """
    assert packed.dtype == np.uint8
    assert n_values % 2 == 0

    byte0 = packed[0::3].astype(np.uint16)
    byte1 = packed[1::3].astype(np.uint16)
    byte2 = packed[2::3].astype(np.uint16)

    a = byte0 | ((byte1 & 0x0F) << 8)
    b = ((byte1 >> 4) & 0x0F) | (byte2 << 4)

    result = np.empty(n_values, dtype=np.uint16)
    result[0::2] = a
    result[1::2] = b
    return result


def pack_12bit_torch(indices: torch.Tensor) -> torch.Tensor:
    """Pack 12-bit indices using PyTorch ops (GPU-compatible).

    Input:  int16 or int32 tensor, values in [0, 4095], length must be even
    Output: uint8 tensor of length N * 3 // 2
    """
    flat = indices.reshape(-1).to(torch.int32)
    n = flat.shape[0]
    assert n % 2 == 0

    a = flat[0::2]
    b = flat[1::2]

    packed = torch.empty(n * 3 // 2, dtype=torch.uint8, device=indices.device)
    packed[0::3] = (a & 0xFF).to(torch.uint8)
    packed[1::3] = (((a >> 8) & 0x0F) | ((b & 0x0F) << 4)).to(torch.uint8)
    packed[2::3] = ((b >> 4) & 0xFF).to(torch.uint8)

    return packed


def unpack_12bit_torch(packed: torch.Tensor, n_values: int) -> torch.Tensor:
    """Unpack 3-byte pairs back to 12-bit values using PyTorch ops.

    Input:  uint8 tensor
    Output: int16 tensor of length n_values
    """
    byte0 = packed[0::3].to(torch.int32)
    byte1 = packed[1::3].to(torch.int32)
    byte2 = packed[2::3].to(torch.int32)

    a = byte0 | ((byte1 & 0x0F) << 8)
    b = ((byte1 >> 4) & 0x0F) | (byte2 << 4)

    result = torch.empty(n_values, dtype=torch.int16, device=packed.device)
    result[0::2] = a.to(torch.int16)
    result[1::2] = b.to(torch.int16)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

print("=" * 60)
print("12-bit Index Packing POC")
print("=" * 60)

# Test 1: Round-trip correctness
print("\n--- Test 1: Round-trip correctness ---")
rng = np.random.default_rng(42)

for n in [2, 10, 100, 1000, 1_000_000]:
    orig = rng.integers(0, 4096, size=n, dtype=np.uint16)
    packed = pack_12bit(orig)
    unpacked = unpack_12bit(packed, n)
    match = np.array_equal(orig, unpacked)
    print(f"  N={n:>10d}: packed {orig.nbytes} -> {packed.nbytes} bytes "
          f"({packed.nbytes/orig.nbytes:.2%}), roundtrip={'PASS' if match else 'FAIL'}")
    assert match, f"Round-trip FAILED for N={n}"

# Test 2: Edge cases
print("\n--- Test 2: Edge cases ---")
edges = np.array([0, 4095, 0, 4095, 2048, 1, 4095, 0], dtype=np.uint16)
packed = pack_12bit(edges)
unpacked = unpack_12bit(packed, len(edges))
assert np.array_equal(edges, unpacked), f"Edge case FAILED: {edges} != {unpacked}"
print(f"  Boundary values (0, 4095, 2048, 1): PASS")

# All same value
for v in [0, 1, 2047, 2048, 4095]:
    arr = np.full(100, v, dtype=np.uint16)
    assert np.array_equal(arr, unpack_12bit(pack_12bit(arr), 100))
print(f"  Constant arrays (0, 1, 2047, 2048, 4095): PASS")

# Test 3: PyTorch round-trip
print("\n--- Test 3: PyTorch round-trip ---")
for n in [100, 10000, 1_000_000]:
    orig_t = torch.randint(0, 4096, (n,), dtype=torch.int16)
    packed_t = pack_12bit_torch(orig_t)
    unpacked_t = unpack_12bit_torch(packed_t, n)
    match = torch.equal(orig_t, unpacked_t)
    print(f"  N={n:>10d}: packed {orig_t.nbytes} -> {packed_t.nbytes} bytes "
          f"({packed_t.nbytes/orig_t.nbytes:.2%}), roundtrip={'PASS' if match else 'FAIL'}")
    assert match

# Test 4: 2D tensor round-trip (simulating [out_features, in_features // vector_dim])
print("\n--- Test 4: 2D matrix round-trip ---")
for shape in [(3584, 1792), (7168, 1792), (896, 1792)]:
    out, in_div2 = shape
    n = out * in_div2
    # Ensure even
    if n % 2 != 0:
        continue
    orig = torch.randint(0, 4096, (out, in_div2), dtype=torch.int16)
    flat = orig.reshape(-1)
    packed = pack_12bit_torch(flat)
    unpacked = unpack_12bit_torch(packed, n).reshape(out, in_div2)
    match = torch.equal(orig, unpacked)
    print(f"  [{out}x{in_div2}]: {flat.nbytes/1024/1024:.2f} MB -> "
          f"{packed.nbytes/1024/1024:.2f} MB, roundtrip={'PASS' if match else 'FAIL'}")
    assert match

# Test 5: Speed benchmark
print("\n--- Test 5: Speed benchmark ---")
n_large = 10_000_000  # ~size of a large layer's indices
orig_np = rng.integers(0, 4096, size=n_large, dtype=np.uint16)
orig_pt = torch.from_numpy(orig_np.astype(np.int16))

# NumPy pack speed
t0 = time.time()
for _ in range(10):
    p = pack_12bit(orig_np)
np_pack_ms = (time.time() - t0) / 10 * 1000

# NumPy unpack speed
t0 = time.time()
for _ in range(10):
    u = unpack_12bit(p, n_large)
np_unpack_ms = (time.time() - t0) / 10 * 1000

# PyTorch pack speed
t0 = time.time()
for _ in range(10):
    p_t = pack_12bit_torch(orig_pt)
pt_pack_ms = (time.time() - t0) / 10 * 1000

# PyTorch unpack speed
t0 = time.time()
for _ in range(10):
    u_t = unpack_12bit_torch(p_t, n_large)
pt_unpack_ms = (time.time() - t0) / 10 * 1000

print(f"  N={n_large:,}")
print(f"  NumPy  pack: {np_pack_ms:.1f} ms, unpack: {np_unpack_ms:.1f} ms")
print(f"  PyTorch pack: {pt_pack_ms:.1f} ms, unpack: {pt_unpack_ms:.1f} ms")

# ---------------------------------------------------------------------------
# Projected savings for Zamba2-7B
# ---------------------------------------------------------------------------
print("\n--- Projected VRAM savings for Zamba2-7B ---")

# From the benchmark: 213 HelixLinear modules, 7365 MB total VRAM
# Index storage dominates. Let's estimate.
# Zamba2-7B has ~7B params. Most are in linear layers.
# With 2D VQ, indices are [out, in//2] int16 = 2 bytes per index = 1 byte per weight.
# Total index bytes ≈ 7B * 1 byte = ~7 GB (most of the 7.365 GB VRAM)

# More precise: count from config
# 213 modules. Typical Zamba2-7B shapes:
# Hidden: 3584, MLP intermediate: 14336 (or 7168 gate+up merged)
# 54 Mamba layers (in_proj: 14336x3584, out_proj: 3584x14336) → but Mamba in_proj is larger
# Actually let's just compute from the known VRAM numbers.

# VRAM breakdown:
# - Codebooks: 213 modules * [4096, 2] * 4 bytes = 213 * 32KB = 6.8 MB (negligible)
# - Indices: the bulk. int16, 2 bytes each
# - Sidecars, norms, embeddings, etc.: maybe 200-400 MB

# From benchmark: VRAM at load = 7365 MB
# Dense model ≈ 14 GB (BF16) → 7B params * 2 bytes = 14 GB
# Index storage = 7B / 2 (vector_dim=2) * 2 bytes (int16) = 7 GB ≈ matches
# Non-index overhead ≈ 7365 - 7000 = ~365 MB

total_params_approx = 7_000_000_000
index_count = total_params_approx // 2  # 2D VQ: one index per 2 weights
current_index_bytes = index_count * 2  # int16 = 2 bytes
packed_index_bytes = index_count * 3 // 2  # 12-bit = 1.5 bytes

overhead_mb = 365  # codebooks + sidecars + norms + embedding
current_vram = current_index_bytes / 1024**2 + overhead_mb
packed_vram = packed_index_bytes / 1024**2 + overhead_mb

print(f"  Total params (approx): {total_params_approx:,}")
print(f"  Index count (2D VQ):   {index_count:,}")
print(f"  Current (int16):       {current_index_bytes/1024**3:.2f} GB indices → {current_vram:.0f} MB total")
print(f"  Packed (12-bit):       {packed_index_bytes/1024**3:.2f} GB indices → {packed_vram:.0f} MB total")
print(f"  Savings:               {(current_index_bytes - packed_index_bytes)/1024**2:.0f} MB ({1 - packed_index_bytes/current_index_bytes:.0%})")
print(f"  Bits per weight:       current={16/2} → packed={12/2}")

# Compare to bnb
print(f"\n  Comparison:")
print(f"    Dense BF16:        ~14,000 MB  (16 bits/weight)")
print(f"    bnb 8-bit:          7,831 MB  ( 8 bits/weight)")
print(f"    HXQ 2D VQ current:  7,365 MB  ( 8 bits/weight)")
print(f"    HXQ 2D VQ 12-bit:  ~{packed_vram:.0f} MB  ( 6 bits/weight)")
print(f"    bnb 4-bit NF4:      5,131 MB  ( 4 bits/weight)")

# ---------------------------------------------------------------------------
# Cost block
# ---------------------------------------------------------------------------
cost = {
    'wall_time_s': round(time.time() - t_start, 3),
    'cpu_time_s': round(time.process_time() - cpu_start, 3),
    'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    'python_version': platform.python_version(),
    'hostname': platform.node(),
    'timestamp_start': start_iso,
    'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
}

print(f"\n--- Cost ---")
for k, v in cost.items():
    print(f"  {k}: {v}")

print("\nAll tests PASSED. 12-bit packing is correct and ready for integration.")
