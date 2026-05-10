"""
helix_substrate/tile_scheduler.py
=================================

Tile-level matmul scheduler with FibPi3D firing order.

Maps HelixLinear's 256-row output tiles to CPU cores using low-discrepancy
spatial sequencing (golden ratio Kronecker sequence on S³). Self-contained —
no helix-cdc dependency.

The FibPi3D sequence ensures maximum spatial separation between consecutively
fired cores, spreading cache pressure evenly and avoiding adjacent-core
contention (same principle as V8 engine piston firing order).

Usage:
    from helix_substrate.tile_scheduler import tile_schedule

    schedule = tile_schedule(out_features=2048, chunk_size=256, n_cores=8)
    # Returns: [(0, 256, 3), (256, 512, 7), (512, 768, 1), ...]
    #           tile_start, tile_end, core_id
"""

import math
import os
import hashlib
import random
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Inline FibPi3D point generation (same algorithm as helix_cdc/scheduler/fibpi_nd.py)
# Self-contained so helix-substrate has zero dependency on helix-cdc.
# ---------------------------------------------------------------------------

_GOLD = (1 + 5 ** 0.5) / 2
_G = (math.sqrt(5) - 1) / 2  # golden ratio conjugate

_ALPHAS = [
    1 / _GOLD,          # ~0.618
    1 / _GOLD ** 2,     # ~0.381
    2 ** 0.5 - 1,       # ~0.414
    3 ** 0.5 - 1,       # ~0.732
]


def _frac(x: float) -> float:
    return x - math.floor(x)


def _weyl_hash(k: int) -> int:
    """Weyl hash to break k ~ square resonances."""
    return (k * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)


def _phi_inv(u: float) -> float:
    """Inverse CDF of standard normal (Acklam's approximation)."""
    u = min(max(u, 1e-12), 1 - 1e-12)
    a1, a2, a3 = -39.69683028665376, 220.9460984245205, -275.9285104469687
    a4, a5, a6 = 138.3577518672690, -30.66479806614716, 2.506628277459239
    b1, b2, b3 = -54.47609879822406, 161.5858368580409, -155.6989798598866
    b4, b5 = 66.80131188771972, -13.28068155288572
    c1, c2, c3 = -0.007784894002430293, -0.3223964580411365, -2.400758277161838
    c4, c5, c6 = -2.549732539343734, 4.374664141464968, 2.938163982698783
    d1, d2, d3 = 0.007784695709041462, 0.3224671290700398, 2.445134137142996
    d4 = 3.754408661907416
    pl, ph = 0.02425, 1 - 0.02425
    if u < pl:
        q = math.sqrt(-2 * math.log(u))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4))
    if u > ph:
        q = math.sqrt(-2 * math.log(1 - u))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4))
    q = u - 0.5
    r = q * q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)


def _fibpi_points_4d(n: int, seed: str = "helix_tile") -> List[List[float]]:
    """Generate n low-discrepancy points on S³ via Kronecker torus → Gaussian → normalize."""
    N = 4
    h = hashlib.sha256(("fibpi_nd:" + seed).encode()).digest()
    offs = [(int.from_bytes(h[d * 4:(d + 1) * 4], "big") / 2 ** 32) for d in range(N)]
    alphas = _ALPHAS[:N]

    rng = random.Random(h)
    cp_shift = [rng.random() for _ in range(N)]

    pts = []
    for i in range(n):
        kk = _weyl_hash(i)
        u = [_frac(offs[d] + cp_shift[d] + (kk / (1 << 64)) * alphas[d]) for d in range(N)]
        z = [_phi_inv(uu) for uu in u]
        norm = math.sqrt(sum(v * v for v in z)) or 1.0
        pts.append([v / norm for v in z])

    return pts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cpu_count() -> int:
    """Get number of CPU cores to use for wave tile dispatch.

    Priority:
    1. HELIX_WAVE_CORES env var (explicit override)
    2. OMP_NUM_THREADS env var (user's stated parallelism preference)
    3. os.sched_getaffinity (respects cgroups/taskset)
    4. os.cpu_count() fallback
    """
    env = os.environ.get("HELIX_WAVE_CORES")
    if env:
        return int(env)
    omp = os.environ.get("OMP_NUM_THREADS")
    if omp and int(omp) > 1:
        return int(omp)
    try:
        n = len(os.sched_getaffinity(0))
        if n > 1:
            return n
    except Exception:
        pass
    return os.cpu_count() or 1


def fibonacci_firing_order(n_cores: int, seed: str = "helix_tile") -> List[int]:
    """Return core IDs in FibPi3D angular firing order.

    Maximizes spatial separation between consecutively fired cores.
    Same algorithm as WaveEngine's engine_firing_order() but self-contained.
    """
    coords = _fibpi_points_4d(n_cores, seed=seed)
    angles = [(i, math.atan2(coord[1], coord[0])) for i, coord in enumerate(coords)]
    angles.sort(key=lambda a: a[1])
    return [core_id for core_id, _ in angles]


def tile_schedule(
    out_features: int,
    chunk_size: int = 256,
    n_cores: Optional[int] = None,
    seed: str = "helix_tile",
) -> List[Tuple[int, int, int]]:
    """Map HelixLinear output tiles to CPU cores in FibPi3D firing order.

    Args:
        out_features: Total output rows of the weight matrix.
        chunk_size: Tile height (default 256, matches HelixLinear CHUNK).
        n_cores: Number of CPU cores (auto-detect if None).
        seed: FibPi3D seed for deterministic scheduling.

    Returns:
        List of (tile_start, tile_end, core_id) tuples ordered by
        FibPi3D firing sequence for maximum spatial separation.
    """
    if n_cores is None:
        n_cores = get_cpu_count()

    # Build tile boundaries
    tiles = []
    for i in range(0, out_features, chunk_size):
        tiles.append((i, min(i + chunk_size, out_features)))

    # Get firing order
    firing = fibonacci_firing_order(n_cores, seed=seed)

    # Assign tiles round-robin through firing order
    return [(start, end, firing[idx % len(firing)]) for idx, (start, end) in enumerate(tiles)]
