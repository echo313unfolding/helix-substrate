"""
Morpho codec for CDNA v3 -- grow/fit weights via wave dynamics on graphs.

Two modes:
  1. GENERATIVE (seed-only): derive codons from seed hash, grow weights.
     Extreme compression, noise-level fidelity on real tensors.
  2. TARGET-FITTING (new): optimize codon parameters (freq, phase, amp)
     to minimize distance to a target tensor. Stores optimized params.

The wave PDE: u_next = 2u - u_prev + c^2*(L@u) - gamma*(u - u_prev) + force
where L is a sparse graph Laplacian (FibPi3D or ring).

This codec is opt-in. Default CDNA v3 routing does NOT use morpho.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse


_PHI = 1.618033988749895
_GOLDEN_ANGLE = 2.39996322972865332


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_fibpi3d_graph(n_nodes: int, k_neighbors: int = 4) -> dict:
    """Build a FibPi3D graph for wave simulation."""
    positions = np.zeros((n_nodes, 3), dtype=np.float64)
    for i in range(n_nodes):
        theta = i * _GOLDEN_ANGLE
        z = 1.0 - (2.0 * i / max(1, n_nodes - 1))
        r = math.sqrt(max(0, 1.0 - z * z))
        positions[i] = [r * math.cos(theta), r * math.sin(theta), z]

    neighbors = []
    weights = []
    for i in range(n_nodes):
        diffs = positions - positions[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists[i] = np.inf
        nn_idx = np.argsort(dists)[:k_neighbors]
        nn_dists = dists[nn_idx]
        nn_weights = 1.0 / (nn_dists + 1e-10)
        nn_weights /= nn_weights.sum()
        neighbors.append(nn_idx.tolist())
        weights.append(nn_weights.tolist())

    return {"n_nodes": n_nodes, "neighbors": neighbors, "weights": weights}


def _build_laplacian(graph: dict) -> sparse.csr_matrix:
    """Build sparse graph Laplacian from neighbor lists.

    L[i,j] = w_ij  for j in neighbors(i)
    L[i,i] = -sum(w_ij)

    So L @ u gives the discrete Laplacian: sum_j w_ij*(u_j - u_i).
    """
    n = graph["n_nodes"]
    rows, cols, vals = [], [], []
    for i in range(n):
        total_w = 0.0
        for j, w in zip(graph["neighbors"][i], graph["weights"][i]):
            rows.append(i)
            cols.append(j)
            vals.append(w)
            total_w += w
        rows.append(i)
        cols.append(i)
        vals.append(-total_w)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# Codon helpers
# ---------------------------------------------------------------------------

def _derive_codons(seed: bytes, n_codons: int = 32) -> list:
    """Derive forcing codons from seed via SHAKE256."""
    h = hashlib.shake_256(seed)
    raw = h.digest(n_codons * 4)
    codons = []
    for i in range(n_codons):
        chunk = raw[i * 4:(i + 1) * 4]
        val = int.from_bytes(chunk, "big")
        freq = 0.5 + (val % 1000) / 100.0
        phase = (val >> 10) % 628 / 100.0
        amp = 0.1 + ((val >> 20) % 100) / 200.0
        codons.append({"freq": freq, "phase": phase, "amp": amp})
    return codons


def _codons_to_array(codons: list) -> np.ndarray:
    """Pack codons into flat array [freq0, phase0, amp0, freq1, ...]."""
    arr = np.empty(len(codons) * 3, dtype=np.float64)
    for i, c in enumerate(codons):
        arr[i * 3] = c["freq"]
        arr[i * 3 + 1] = c["phase"]
        arr[i * 3 + 2] = c["amp"]
    return arr


def _array_to_codons(arr: np.ndarray) -> list:
    """Unpack flat array back to codon list."""
    n = len(arr) // 3
    codons = []
    for i in range(n):
        codons.append({
            "freq": float(arr[i * 3]),
            "phase": float(arr[i * 3 + 1]),
            "amp": float(arr[i * 3 + 2]),
        })
    return codons


# ---------------------------------------------------------------------------
# Vectorized wave simulation
# ---------------------------------------------------------------------------

def _precompute_node_phases(n: int) -> np.ndarray:
    """Per-node golden angle offsets for forcing."""
    return np.arange(n, dtype=np.float64) * _GOLDEN_ANGLE


def _grow_from_codons(
    codon_arr: np.ndarray,
    c: float,
    gamma: float,
    L: sparse.csr_matrix,
    u_init: np.ndarray,
    n: int,
    steps: int,
    target_size: int,
    node_phases: np.ndarray,
    perturb_vec: np.ndarray,
    out_dim: int,
    in_dim: int,
    scale: float = 0.0,
    shift: float = 0.0,
) -> np.ndarray:
    """
    Vectorized wave simulation from explicit codon parameters.

    Args:
        codon_arr: flat array [freq0, phase0, amp0, ...] (n_codons*3,)
        c, gamma: wave speed and damping
        L: sparse Laplacian (n, n)
        u_init: initial field (n,)
        n: number of nodes
        steps: simulation steps
        target_size: out_dim * in_dim
        node_phases: precomputed per-node phase offsets (n,)
        perturb_vec: position-dependent perturbation vector (target_size,)
        out_dim, in_dim: output shape
        scale: post-normalization scale (0.0 = use Xavier default)
        shift: post-normalization shift

    Returns:
        Weight matrix (out_dim, in_dim) float64
    """
    n_codons = len(codon_arr) // 3
    codon_window = max(1, steps // n_codons)
    sample_interval = max(1, steps // max(64, target_size // n + 1))

    c2 = c * c
    u_curr = u_init.copy()
    u_prev = np.zeros(n, dtype=np.float64)
    history = []

    for t in range(steps):
        if t % sample_interval == 0:
            history.append(u_curr.copy())

        ci = min(t // codon_window, n_codons - 1)
        freq = codon_arr[ci * 3]
        phase = codon_arr[ci * 3 + 1]
        amp = codon_arr[ci * 3 + 2]

        force = amp * np.sin(freq * t * 0.01 + phase + node_phases)
        lap = L.dot(u_curr)
        vel = u_curr - u_prev
        u_next = 2.0 * u_curr - u_prev + c2 * lap - gamma * vel + force

        u_prev = u_curr
        u_curr = u_next

    if history:
        attractor = np.concatenate(history)
    else:
        attractor = u_curr

    if len(attractor) >= target_size:
        indices = np.linspace(0, len(attractor) - 1, target_size).astype(int)
        weights = attractor[indices]
    else:
        repeats = int(math.ceil(target_size / len(attractor)))
        tiled = np.tile(attractor, repeats)[:target_size]
        field_std = np.std(tiled) + 1e-10
        position_weights = np.arange(target_size, dtype=np.float64) / target_size
        weights = tiled + perturb_vec * 0.2 * field_std * (0.5 + position_weights)

    W = weights.reshape(out_dim, in_dim)
    W = W - np.mean(W)
    current_std = np.std(W) + 1e-10

    if scale == 0.0:
        # Default: Xavier-like normalization
        std_target = 1.0 / math.sqrt(in_dim)
        W = W * (std_target / current_std)
    else:
        # Optimizer-controlled: normalize to unit std, then apply scale/shift
        W = W / current_std * scale + shift

    return W


# ---------------------------------------------------------------------------
# Public API: grow_weights (seed-based, backward compatible)
# ---------------------------------------------------------------------------

def grow_weights(
    seed: bytes,
    shape: tuple[int, int],
    steps: int = 1000,
    target_se: float = 1.5,
    c: float = 0.35,
    gamma: float = 0.02,
    geometry: str = "fibpi3d",
    k_neighbors: int = 4,
) -> np.ndarray:
    """
    Grow a weight matrix from seed using wave dynamics on a graph.

    Deterministic: same seed + config -> identical output.

    Returns:
        np.ndarray of shape (out_dim, in_dim), float32
    """
    out_dim, in_dim = shape
    target_size = out_dim * in_dim
    field_size = min(512, max(64, int(math.sqrt(target_size))))

    if geometry == "fibpi3d":
        graph = _build_fibpi3d_graph(field_size, k_neighbors)
    else:
        graph = {
            "n_nodes": field_size,
            "neighbors": [[(i - 1) % field_size, (i + 1) % field_size]
                          for i in range(field_size)],
            "weights": [[0.5, 0.5] for _ in range(field_size)],
        }

    n = graph["n_nodes"]
    L = _build_laplacian(graph)
    node_phases = _precompute_node_phases(n)

    init_seed = hashlib.sha256(seed + b"init").digest()
    rng = np.random.RandomState(int.from_bytes(init_seed[:4], "big"))
    u_init = rng.randn(n).astype(np.float64) * 0.01

    perturb_seed = hashlib.sha256(seed + b"position_diversity").digest()
    prng = np.random.RandomState(int.from_bytes(perturb_seed[:4], "big"))
    perturb_vec = prng.randn(target_size).astype(np.float64)

    codons = _derive_codons(seed)
    codon_arr = _codons_to_array(codons)

    W = _grow_from_codons(
        codon_arr, c, gamma, L, u_init, n,
        steps, target_size, node_phases, perturb_vec, out_dim, in_dim,
    )
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Target fitting: optimize codon params to match a target tensor
# ---------------------------------------------------------------------------

def fit_to_target(
    target: np.ndarray,
    n_codons: int = 32,
    steps: int = 500,
    c: float = 0.35,
    gamma: float = 0.02,
    geometry: str = "fibpi3d",
    k_neighbors: int = 4,
    seed: Optional[bytes] = None,
    max_iter: int = 200,
    verbose: bool = False,
    n_restarts: int = 1,
) -> dict:
    """
    Optimize codon parameters to grow weights matching a target tensor.

    This is the target-fitting mechanism that turns morpho from a generator
    into an encoder. It optimizes n_codons*3 + 4 parameters via L-BFGS-B
    to maximize cosine similarity with the target.

    Args:
        target: Target weight tensor (2D float32/64)
        n_codons: Number of forcing codons
        steps: Wave simulation steps per evaluation
        c: Initial wave speed
        gamma: Initial damping
        geometry: Graph geometry
        k_neighbors: Graph connectivity
        seed: Seed for initial conditions (random if None)
        max_iter: Maximum optimization iterations
        verbose: Print progress
        n_restarts: Number of random restarts (best result kept)

    Returns:
        Dict with keys:
            codons: list of {freq, phase, amp}
            c, gamma: optimized globals
            cosine: final cosine similarity
            seed_hex: seed used for initial conditions
            n_evals: number of wave simulations run
    """
    from scipy.optimize import minimize

    target = target.astype(np.float64)
    out_dim, in_dim = target.shape
    target_size = out_dim * in_dim
    target_flat = target.ravel()
    target_norm = np.linalg.norm(target_flat)
    if target_norm == 0:
        raise ValueError("Target tensor is all zeros")

    field_size = min(512, max(64, int(math.sqrt(target_size))))

    # Build graph + Laplacian (once, shared across restarts)
    if geometry == "fibpi3d":
        graph = _build_fibpi3d_graph(field_size, k_neighbors)
    else:
        graph = {
            "n_nodes": field_size,
            "neighbors": [[(i - 1) % field_size, (i + 1) % field_size]
                          for i in range(field_size)],
            "weights": [[0.5, 0.5] for _ in range(field_size)],
        }

    n = graph["n_nodes"]
    L = _build_laplacian(graph)
    node_phases = _precompute_node_phases(n)

    # Seed-based initial conditions
    if seed is None:
        seed = np.random.bytes(32)
    init_seed = hashlib.sha256(seed + b"init").digest()
    rng = np.random.RandomState(int.from_bytes(init_seed[:4], "big"))
    u_init = rng.randn(n).astype(np.float64) * 0.01

    perturb_seed = hashlib.sha256(seed + b"position_diversity").digest()
    prng = np.random.RandomState(int.from_bytes(perturb_seed[:4], "big"))
    perturb_vec = prng.randn(target_size).astype(np.float64)

    # Target statistics for scale/shift init
    target_std = float(np.std(target))
    target_mean = float(np.mean(target))
    init_scale = target_std if target_std > 0 else 0.01
    init_shift = target_mean

    # Bounds (same for all restarts)
    codon_bounds = []
    for _ in range(n_codons):
        codon_bounds.append((0.1, 20.0))   # freq
        codon_bounds.append((0.0, 6.284))  # phase
        codon_bounds.append((0.01, 2.0))   # amp
    codon_bounds.append((0.05, 1.0))       # c
    codon_bounds.append((0.001, 0.5))      # gamma
    codon_bounds.append((1e-6, 10.0))      # scale
    codon_bounds.append((-5.0, 5.0))       # shift

    total_evals = [0]
    best_result = None
    best_cosine_overall = -1.0

    for restart in range(n_restarts):
        if restart == 0:
            # First restart: use seed-derived codons (deterministic)
            initial_codons = _derive_codons(seed, n_codons)
            x0_codons = _codons_to_array(initial_codons)
        else:
            # Subsequent restarts: random initialization
            restart_rng = np.random.RandomState(
                int.from_bytes(
                    hashlib.sha256(seed + f"restart_{restart}".encode()).digest()[:4],
                    "big"
                )
            )
            x0_codons = np.empty(n_codons * 3, dtype=np.float64)
            for i in range(n_codons):
                x0_codons[i * 3] = restart_rng.uniform(0.5, 15.0)      # freq
                x0_codons[i * 3 + 1] = restart_rng.uniform(0.0, 6.28)  # phase
                x0_codons[i * 3 + 2] = restart_rng.uniform(0.05, 1.5)  # amp

        x0 = np.concatenate([x0_codons, [c, gamma, init_scale, init_shift]])

        eval_count = [0]
        best_cosine = [-1.0]

        def objective(x):
            codon_arr = x[:-4]
            c_val = x[-4]
            gamma_val = x[-3]
            scale_val = x[-2]
            shift_val = x[-1]

            W = _grow_from_codons(
                codon_arr, c_val, gamma_val, L, u_init, n,
                steps, target_size, node_phases, perturb_vec, out_dim, in_dim,
                scale=scale_val, shift=shift_val,
            )

            w_flat = W.ravel()
            w_norm = np.linalg.norm(w_flat)
            if w_norm == 0:
                return 1.0

            cosine = np.dot(target_flat, w_flat) / (target_norm * w_norm)
            loss = -cosine

            eval_count[0] += 1
            if cosine > best_cosine[0]:
                best_cosine[0] = cosine
                if verbose:
                    print(f"  [restart {restart}] eval={eval_count[0]:4d}  "
                          f"cosine={cosine:.6f}  c={c_val:.3f}  "
                          f"gamma={gamma_val:.4f}  scale={scale_val:.4f}  "
                          f"shift={shift_val:.4f}")

            return loss

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=codon_bounds,
            options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-7},
        )

        total_evals[0] += eval_count[0]
        restart_cosine = -result.fun

        if verbose:
            print(f"  restart {restart}: cosine={restart_cosine:.6f} "
                  f"({eval_count[0]} evals)")

        if restart_cosine > best_cosine_overall:
            best_cosine_overall = restart_cosine
            best_result = result

            # Early stopping: skip remaining restarts if already excellent
            if best_cosine_overall > 0.995:
                if verbose:
                    print(f"  Early stop: cosine {best_cosine_overall:.6f} > 0.995")
                break

    # Extract final params from best restart
    final_codons = _array_to_codons(best_result.x[:-4])
    final_c = float(best_result.x[-4])
    final_gamma = float(best_result.x[-3])
    final_scale = float(best_result.x[-2])
    final_shift = float(best_result.x[-1])
    final_cosine = best_cosine_overall

    if verbose:
        print(f"  DONE: cosine={final_cosine:.6f} after {total_evals[0]} evals "
              f"({n_restarts} restarts)")

    return {
        "codons": final_codons,
        "c": final_c,
        "gamma": final_gamma,
        "scale": final_scale,
        "shift": final_shift,
        "cosine": float(final_cosine),
        "seed_hex": seed.hex(),
        "n_evals": total_evals[0],
        "n_restarts": n_restarts,
        "steps": steps,
        "geometry": geometry,
        "k_neighbors": k_neighbors,
        "field_size": field_size,
    }


def grow_from_fit_result(
    fit_result: dict,
    shape: tuple[int, int],
) -> np.ndarray:
    """Decode weights from a fit_to_target result."""
    seed = bytes.fromhex(fit_result["seed_hex"])
    out_dim, in_dim = shape
    target_size = out_dim * in_dim
    field_size = fit_result["field_size"]
    geometry = fit_result["geometry"]
    k_neighbors = fit_result.get("k_neighbors", 4)

    if geometry == "fibpi3d":
        graph = _build_fibpi3d_graph(field_size, k_neighbors)
    else:
        graph = {
            "n_nodes": field_size,
            "neighbors": [[(i - 1) % field_size, (i + 1) % field_size]
                          for i in range(field_size)],
            "weights": [[0.5, 0.5] for _ in range(field_size)],
        }

    n = graph["n_nodes"]
    L = _build_laplacian(graph)
    node_phases = _precompute_node_phases(n)

    init_seed = hashlib.sha256(seed + b"init").digest()
    rng = np.random.RandomState(int.from_bytes(init_seed[:4], "big"))
    u_init = rng.randn(n).astype(np.float64) * 0.01

    perturb_seed = hashlib.sha256(seed + b"position_diversity").digest()
    prng = np.random.RandomState(int.from_bytes(perturb_seed[:4], "big"))
    perturb_vec = prng.randn(target_size).astype(np.float64)

    codon_arr = _codons_to_array(fit_result["codons"])

    W = _grow_from_codons(
        codon_arr, fit_result["c"], fit_result["gamma"],
        L, u_init, n, fit_result["steps"],
        target_size, node_phases, perturb_vec, out_dim, in_dim,
        scale=fit_result.get("scale", 0.0),
        shift=fit_result.get("shift", 0.0),
    )
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Graph spectral basis (morpho_v3_spectral)
# ---------------------------------------------------------------------------

_HARMONICS_CACHE: dict = {}


def _compute_graph_harmonics(
    L: sparse.csr_matrix,
    k: int,
    cache_key: tuple | None = None,
) -> tuple:
    """
    Compute first k eigenvectors of L (graph harmonics).

    Returns:
        eigenvalues: (k,) sorted ascending
        eigenvectors: (n, k) columns are harmonics
    """
    if cache_key and cache_key in _HARMONICS_CACHE:
        return _HARMONICS_CACHE[cache_key]

    from scipy.sparse.linalg import eigsh

    # eigsh with 'SM' finds smallest magnitude eigenvalues
    # For a Laplacian, smallest eigenvalue is 0 (constant mode)
    vals, vecs = eigsh(L, k=min(k, L.shape[0] - 2), which='SM')
    order = np.argsort(vals)
    result = (vals[order], vecs[:, order])

    if cache_key:
        _HARMONICS_CACHE[cache_key] = result
    return result


def _spectral_synthesis(
    codon_arr: np.ndarray,
    eigenvectors: np.ndarray,
    target_size: int,
    out_dim: int,
    in_dim: int,
    scale: float = 0.0,
    shift: float = 0.0,
    perturb_vec: np.ndarray | None = None,
) -> np.ndarray:
    """
    Direct spectral synthesis: field = sum_k amp_k * rotated_harmonic_k.

    Each codon specifies (harmonic_index, amp, phase). The phase rotates
    between two adjacent harmonics: cos(phase)*phi_k + sin(phase)*phi_{k+1}.

    Args:
        codon_arr: flat [harmonic_idx0, amp0, phase0, ...]  (n_codons*3,)
        eigenvectors: (n_nodes, n_harmonics) graph harmonic columns
        target_size: out_dim * in_dim
        out_dim, in_dim: output shape
        scale: post-normalization scale (0.0 = Xavier default)
        shift: post-normalization shift
        perturb_vec: position-dependent perturbation (target_size,)

    Returns:
        Weight matrix (out_dim, in_dim) float64
    """
    n_nodes, n_harmonics = eigenvectors.shape
    n_codons = len(codon_arr) // 3

    # Build field on graph nodes
    field = np.zeros(n_nodes, dtype=np.float64)
    for k in range(n_codons):
        hi = int(codon_arr[k * 3]) % n_harmonics
        amp = codon_arr[k * 3 + 1]
        phase = codon_arr[k * 3 + 2]
        hi_next = (hi + 1) % n_harmonics
        field += amp * (
            math.cos(phase) * eigenvectors[:, hi]
            + math.sin(phase) * eigenvectors[:, hi_next]
        )

    # Expand field to target_size by tiling + perturbation
    if n_nodes >= target_size:
        indices = np.linspace(0, n_nodes - 1, target_size).astype(int)
        weights = field[indices]
    else:
        repeats = int(math.ceil(target_size / n_nodes))
        tiled = np.tile(field, repeats)[:target_size]
        if perturb_vec is not None:
            field_std = np.std(tiled) + 1e-10
            position_weights = np.arange(target_size, dtype=np.float64) / target_size
            weights = tiled + perturb_vec * 0.2 * field_std * (0.5 + position_weights)
        else:
            weights = tiled

    W = weights.reshape(out_dim, in_dim)
    W = W - np.mean(W)
    current_std = np.std(W) + 1e-10

    if scale == 0.0:
        std_target = 1.0 / math.sqrt(in_dim)
        W = W * (std_target / current_std)
    else:
        W = W / current_std * scale + shift

    return W


def fit_to_target_spectral(
    target: np.ndarray,
    n_codons: int = 32,
    n_harmonics: int = 64,
    c: float = 0.35,
    gamma: float = 0.02,
    geometry: str = "fibpi3d",
    k_neighbors: int = 4,
    seed: bytes | None = None,
    max_iter: int = 200,
    verbose: bool = False,
    n_restarts: int = 1,
) -> dict:
    """
    Fit target tensor using graph spectral basis (Laplacian eigenvectors).

    Instead of running a wave PDE, this directly synthesizes weights as a
    linear combination of graph harmonics. Each codon specifies which harmonic
    to activate and with what amplitude/phase rotation.

    Same parameter budget as sinusoidal fitting: n_codons*3 + 2 (scale, shift).
    """
    from scipy.optimize import minimize

    target = target.astype(np.float64)
    out_dim, in_dim = target.shape
    target_size = out_dim * in_dim
    target_flat = target.ravel()
    target_norm = np.linalg.norm(target_flat)
    if target_norm == 0:
        raise ValueError("Target tensor is all zeros")

    # For 1D tensors, use a ring graph matching target size — its eigenvectors
    # are exact Fourier modes, no tiling artifacts.  For 2D, use sqrt heuristic.
    is_1d = (out_dim == 1)
    if is_1d:
        field_size = min(target_size, 2048)
    else:
        field_size = min(512, max(64, int(math.sqrt(target_size))))

    # Build graph + Laplacian + harmonics
    # 1D targets get a ring graph (Fourier-exact); 2D targets get FibPi3D
    if is_1d or geometry == "ring":
        graph = {
            "n_nodes": field_size,
            "neighbors": [[(i - 1) % field_size, (i + 1) % field_size]
                          for i in range(field_size)],
            "weights": [[0.5, 0.5] for _ in range(field_size)],
        }
    elif geometry == "fibpi3d":
        graph = _build_fibpi3d_graph(field_size, k_neighbors)
    else:
        graph = {
            "n_nodes": field_size,
            "neighbors": [[(i - 1) % field_size, (i + 1) % field_size]
                          for i in range(field_size)],
            "weights": [[0.5, 0.5] for _ in range(field_size)],
        }

    n = graph["n_nodes"]
    L = _build_laplacian(graph)
    cache_key = (n, k_neighbors, n_harmonics)
    eigenvalues, eigenvectors = _compute_graph_harmonics(L, n_harmonics, cache_key)

    if verbose:
        print(f"  Graph: {n} nodes, {eigenvectors.shape[1]} harmonics computed")

    # Seed for perturbation vector
    if seed is None:
        seed = np.random.bytes(32)
    perturb_seed = hashlib.sha256(seed + b"position_diversity").digest()
    prng = np.random.RandomState(int.from_bytes(perturb_seed[:4], "big"))
    perturb_vec = prng.randn(target_size).astype(np.float64)

    # Target statistics for scale/shift init
    target_std = float(np.std(target))
    target_mean = float(np.mean(target))
    init_scale = target_std if target_std > 0 else 0.01
    init_shift = target_mean

    # Bounds: [harmonic_idx, amp, phase] * n_codons + [scale, shift]
    bounds = []
    for _ in range(n_codons):
        bounds.append((0.0, float(eigenvectors.shape[1] - 1)))  # harmonic index
        bounds.append((0.01, 5.0))    # amp
        bounds.append((0.0, 6.284))   # phase
    bounds.append((1e-6, 10.0))       # scale
    bounds.append((-5.0, 5.0))        # shift

    total_evals = [0]
    best_result = None
    best_cosine_overall = -1.0

    # FFT-initialized starting point: project target onto eigenvectors,
    # pick the n_codons modes with highest projection magnitude.
    projections = eigenvectors.T @ np.interp(
        np.linspace(0, 1, eigenvectors.shape[0]),
        np.linspace(0, 1, target_size),
        target_flat,
    )  # (n_harmonics,)
    proj_mag = np.abs(projections)
    top_k = np.argsort(proj_mag)[::-1][:n_codons]

    for restart in range(n_restarts):
        if restart == 0:
            # Initialize from top-k spectral projections
            x0_codons = np.empty(n_codons * 3, dtype=np.float64)
            n_h = eigenvectors.shape[1]
            for i, hi in enumerate(top_k):
                x0_codons[i * 3] = float(hi)
                x0_codons[i * 3 + 1] = max(0.01, min(5.0, float(proj_mag[hi])))
                x0_codons[i * 3 + 2] = float(np.angle(projections[hi]) % (2 * math.pi)) if np.iscomplexobj(projections) else 0.0
        else:
            restart_rng = np.random.RandomState(
                int.from_bytes(
                    hashlib.sha256(seed + f"spectral_restart_{restart}".encode()).digest()[:4],
                    "big"
                )
            )
            x0_codons = np.empty(n_codons * 3, dtype=np.float64)
            n_h = eigenvectors.shape[1]
            for i in range(n_codons):
                x0_codons[i * 3] = restart_rng.uniform(0, n_h - 1)
                x0_codons[i * 3 + 1] = restart_rng.uniform(0.1, 3.0)
                x0_codons[i * 3 + 2] = restart_rng.uniform(0.0, 6.28)

        x0 = np.concatenate([x0_codons, [init_scale, init_shift]])

        eval_count = [0]
        best_cosine = [-1.0]

        def objective(x):
            codon_arr = x[:-2]
            scale_val = x[-2]
            shift_val = x[-1]

            W = _spectral_synthesis(
                codon_arr, eigenvectors, target_size,
                out_dim, in_dim,
                scale=scale_val, shift=shift_val,
                perturb_vec=perturb_vec,
            )

            w_flat = W.ravel()
            w_norm = np.linalg.norm(w_flat)
            if w_norm == 0:
                return 1.0

            cosine = np.dot(target_flat, w_flat) / (target_norm * w_norm)
            loss = -cosine

            eval_count[0] += 1
            if cosine > best_cosine[0]:
                best_cosine[0] = cosine
                if verbose:
                    print(f"  [spectral r{restart}] eval={eval_count[0]:4d}  "
                          f"cosine={cosine:.6f}  scale={scale_val:.4f}  "
                          f"shift={shift_val:.4f}")

            return loss

        result = minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-7},
        )

        total_evals[0] += eval_count[0]
        restart_cosine = -result.fun

        if verbose:
            print(f"  restart {restart}: cosine={restart_cosine:.6f} "
                  f"({eval_count[0]} evals)")

        if restart_cosine > best_cosine_overall:
            best_cosine_overall = restart_cosine
            best_result = result

            if best_cosine_overall > 0.995:
                if verbose:
                    print(f"  Early stop: cosine {best_cosine_overall:.6f} > 0.995")
                break

    # Extract final params
    final_codons = []
    for i in range(n_codons):
        final_codons.append({
            "harmonic": float(best_result.x[i * 3]),
            "amp": float(best_result.x[i * 3 + 1]),
            "phase": float(best_result.x[i * 3 + 2]),
        })
    final_scale = float(best_result.x[-2])
    final_shift = float(best_result.x[-1])

    return {
        "codons": final_codons,
        "scale": final_scale,
        "shift": final_shift,
        "cosine": float(best_cosine_overall),
        "seed_hex": seed.hex(),
        "n_evals": total_evals[0],
        "n_restarts": n_restarts,
        "n_harmonics": eigenvectors.shape[1],
        "geometry": geometry,
        "k_neighbors": k_neighbors,
        "field_size": field_size,
    }


def grow_from_spectral_result(
    fit_result: dict,
    shape: tuple[int, int],
) -> np.ndarray:
    """Decode weights from a fit_to_target_spectral result."""
    seed = bytes.fromhex(fit_result["seed_hex"])
    out_dim, in_dim = shape
    target_size = out_dim * in_dim
    field_size = fit_result["field_size"]
    geometry = fit_result["geometry"]
    k_neighbors = fit_result.get("k_neighbors", 4)
    n_harmonics = fit_result["n_harmonics"]

    if geometry == "fibpi3d":
        graph = _build_fibpi3d_graph(field_size, k_neighbors)
    else:
        graph = {
            "n_nodes": field_size,
            "neighbors": [[(i - 1) % field_size, (i + 1) % field_size]
                          for i in range(field_size)],
            "weights": [[0.5, 0.5] for _ in range(field_size)],
        }

    n = graph["n_nodes"]
    L = _build_laplacian(graph)
    cache_key = (n, k_neighbors, n_harmonics)
    _, eigenvectors = _compute_graph_harmonics(L, n_harmonics, cache_key)

    perturb_seed = hashlib.sha256(seed + b"position_diversity").digest()
    prng = np.random.RandomState(int.from_bytes(perturb_seed[:4], "big"))
    perturb_vec = prng.randn(target_size).astype(np.float64)

    # Pack codons into flat array
    codons = fit_result["codons"]
    codon_arr = np.empty(len(codons) * 3, dtype=np.float64)
    for i, c in enumerate(codons):
        codon_arr[i * 3] = c["harmonic"]
        codon_arr[i * 3 + 1] = c["amp"]
        codon_arr[i * 3 + 2] = c["phase"]

    W = _spectral_synthesis(
        codon_arr, eigenvectors, target_size,
        out_dim, in_dim,
        scale=fit_result.get("scale", 0.0),
        shift=fit_result.get("shift", 0.0),
        perturb_vec=perturb_vec,
    )
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Direct FFT projection (morpho_v3_fft) — optimal for 1D broadband signals
# ---------------------------------------------------------------------------

def fit_to_target_fft(
    target: np.ndarray,
    n_coeffs: int = 32,
    seed: bytes | None = None,
) -> dict:
    """
    Encode a 1D target using top-k FFT coefficients (direct projection).

    This is provably optimal: for a given number of Fourier coefficients,
    keeping the highest-power modes minimizes reconstruction error.

    Parameters: 2 per coefficient (real, imag) + DC = 2*n_coeffs + 1 total.
    """
    target = target.astype(np.float64).ravel()
    n = len(target)
    target_norm = np.linalg.norm(target)
    if target_norm == 0:
        raise ValueError("Target tensor is all zeros")

    fft_coeffs = np.fft.rfft(target)
    power = np.abs(fft_coeffs) ** 2

    # Keep DC (index 0) always, then top n_coeffs-1 by power
    top_idx = np.argsort(power[1:])[::-1][: n_coeffs - 1] + 1
    keep_idx = np.concatenate([[0], np.sort(top_idx)])

    # Reconstruct
    recon_coeffs = np.zeros_like(fft_coeffs)
    recon_coeffs[keep_idx] = fft_coeffs[keep_idx]
    recon = np.fft.irfft(recon_coeffs, n=n)
    cosine = float(np.dot(target, recon) / (target_norm * np.linalg.norm(recon)))

    # Store as list of (index, real, imag) triples
    coefficients = []
    for idx in keep_idx:
        coefficients.append({
            "freq_bin": int(idx),
            "real": float(fft_coeffs[idx].real),
            "imag": float(fft_coeffs[idx].imag),
        })

    if seed is None:
        seed = np.random.bytes(32)

    return {
        "coefficients": coefficients,
        "n_original": n,
        "cosine": cosine,
        "seed_hex": seed.hex(),
        "n_evals": 1,  # Direct projection, no optimization
        "n_coeffs": n_coeffs,
    }


def grow_from_fft_result(
    fit_result: dict,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Decode weights from an FFT projection result."""
    n = fit_result["n_original"]
    n_fft = n // 2 + 1
    recon_coeffs = np.zeros(n_fft, dtype=np.complex128)
    for c in fit_result["coefficients"]:
        recon_coeffs[c["freq_bin"]] = complex(c["real"], c["imag"])
    recon = np.fft.irfft(recon_coeffs, n=n)
    return recon.reshape(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Codec API
# ---------------------------------------------------------------------------

def morpho_encode(
    tensor: np.ndarray,
    tensor_name: str,
    output_dir: Path,
    steps: int = 1000,
    target_se: float = 1.5,
    c: float = 0.35,
    gamma: float = 0.02,
    geometry: str = "fibpi3d",
    fit: bool = False,
    n_codons: int = 32,
    max_iter: int = 200,
    verbose: bool = False,
    spectral: bool = False,
    n_harmonics: int = 64,
) -> dict:
    """
    Encode a tensor via morpho codec.

    Two modes:
      fit=False: Derive codons from seed hash (fast, noise-level fidelity)
      fit=True:  Optimize codons to match target (slower, real fidelity)

    Returns:
        Stats dict with encoding metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor = tensor.astype(np.float32)

    is_1d = tensor.ndim == 1
    original_shape = tensor.shape
    if is_1d:
        tensor = tensor.reshape(1, -1)
    if tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])

    rows, cols = tensor.shape
    seed = hashlib.sha256(tensor_name.encode()).digest()

    if spectral and fit and is_1d:
        # 1D tensors: use direct FFT projection (provably optimal)
        # Adaptive coefficient count: broadband signals get more coefficients
        target_f64 = tensor.astype(np.float64)
        fft_full = np.fft.rfft(target_f64.ravel())
        power = np.abs(fft_full) ** 2
        total_power = power.sum() + 1e-30
        energy_low = power[:65].sum() / total_power
        if energy_low < 0.3:
            # Broadband signal — bump coefficient count
            adaptive_n = min(256, max(n_codons, 128))
        else:
            adaptive_n = n_codons
        fit_result = fit_to_target_fft(
            target=target_f64,
            n_coeffs=adaptive_n,
            seed=seed,
        )
        config = {
            "format_version": "morpho_v3_fft",
            "tensor_name": tensor_name,
            "shape": list(original_shape),
            "coefficients": fit_result["coefficients"],
            "n_original": fit_result["n_original"],
            "n_coeffs": fit_result["n_coeffs"],
            "cosine_similarity": round(fit_result["cosine"], 6),
            "seed_hex": fit_result["seed_hex"],
            "n_evals": fit_result["n_evals"],
        }
        cosine = fit_result["cosine"]
    elif spectral and fit:
        # 2D: Spectral fitting mode: graph Laplacian eigenvector basis
        fit_result = fit_to_target_spectral(
            target=tensor.astype(np.float64),
            n_codons=n_codons,
            n_harmonics=n_harmonics,
            c=c,
            gamma=gamma,
            geometry=geometry,
            seed=seed,
            max_iter=max_iter,
            verbose=verbose,
        )

        config = {
            "format_version": "morpho_v3_spectral",
            "tensor_name": tensor_name,
            "shape": list(original_shape),
            "growth_shape": [rows, cols],
            "seed_hex": fit_result["seed_hex"],
            "scale": fit_result["scale"],
            "shift": fit_result["shift"],
            "geometry": fit_result["geometry"],
            "k_neighbors": fit_result.get("k_neighbors", 4),
            "field_size": fit_result["field_size"],
            "n_harmonics": fit_result["n_harmonics"],
            "codons": fit_result["codons"],
            "cosine_similarity": round(fit_result["cosine"], 6),
            "n_evals": fit_result["n_evals"],
        }
        cosine = fit_result["cosine"]
    elif fit:
        # Target-fitting mode: optimize codons (wave PDE)
        fit_result = fit_to_target(
            target=tensor.astype(np.float64),
            n_codons=n_codons,
            steps=steps,
            c=c,
            gamma=gamma,
            geometry=geometry,
            seed=seed,
            max_iter=max_iter,
            verbose=verbose,
        )

        config = {
            "format_version": "morpho_v2_fitted",
            "tensor_name": tensor_name,
            "shape": list(original_shape),
            "growth_shape": [rows, cols],
            "seed_hex": fit_result["seed_hex"],
            "steps": fit_result["steps"],
            "c": fit_result["c"],
            "gamma": fit_result["gamma"],
            "scale": fit_result["scale"],
            "shift": fit_result["shift"],
            "geometry": fit_result["geometry"],
            "k_neighbors": fit_result.get("k_neighbors", 4),
            "field_size": fit_result["field_size"],
            "codons": fit_result["codons"],
            "cosine_similarity": round(fit_result["cosine"], 6),
            "n_evals": fit_result["n_evals"],
        }
        cosine = fit_result["cosine"]
    else:
        # Seed-only mode (original behavior)
        grown = grow_weights(
            seed=seed, shape=(rows, cols), steps=steps,
            target_se=target_se, c=c, gamma=gamma, geometry=geometry,
        )
        flat_orig = tensor.ravel()
        flat_grown = grown.ravel()
        norm_a = np.linalg.norm(flat_orig)
        norm_b = np.linalg.norm(flat_grown)
        cosine = float(np.dot(flat_orig, flat_grown) / (norm_a * norm_b)) \
            if (norm_a > 0 and norm_b > 0) else 0.0

        config = {
            "format_version": "morpho_v1",
            "tensor_name": tensor_name,
            "shape": list(original_shape),
            "growth_shape": [rows, cols],
            "seed_hex": seed.hex(),
            "steps": steps,
            "target_se": target_se,
            "c": c,
            "gamma": gamma,
            "geometry": geometry,
            "k_neighbors": 4,
            "cosine_similarity": round(cosine, 6),
        }

    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    config["config_sha256"] = config_sha

    config_path = output_dir / "morpho_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    meta = {
        "format_version": "cdna_v3",
        "tensor_name": tensor_name,
        "shape": [rows, cols],
        "original_shape": list(original_shape),
        "dtype": "float32",
        "storage_mode": "morpho",
        "tensor_class": "morpho",
        "config_sha256": config_sha,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    config_bytes = len(config_path.read_bytes())
    stats = {
        "tensor_name": tensor_name,
        "shape": list(original_shape),
        "original_bytes": tensor.nbytes,
        "compressed_bytes": config_bytes,
        "compression_ratio": round(tensor.nbytes / max(1, config_bytes), 2),
        "cosine": round(cosine, 6),
        "storage_mode": "morpho",
        "growth_steps": steps,
        "fitted": fit,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    return stats


def morpho_decode(tensor_dir: Path) -> np.ndarray:
    """
    Decode a tensor from morpho config.

    Handles both v1 (seed-only) and v2 (fitted codons).
    """
    tensor_dir = Path(tensor_dir)
    config_path = tensor_dir / "morpho_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"morpho_config.json not found in {tensor_dir}")

    config = json.loads(config_path.read_text())

    if config["format_version"] == "morpho_v3_fft":
        # Direct FFT projection (1D tensors)
        original_shape = tuple(config["shape"])
        grown = grow_from_fft_result(config, original_shape)
        return grown

    rows, cols = config["growth_shape"]

    if config["format_version"] == "morpho_v3_spectral":
        # Spectral basis: graph Laplacian eigenvector synthesis
        grown = grow_from_spectral_result(config, (rows, cols))
    elif config["format_version"] == "morpho_v2_fitted":
        # Fitted mode: use stored codons (wave PDE)
        grown = grow_from_fit_result(config, (rows, cols))
    else:
        # Legacy seed-only mode
        seed = bytes.fromhex(config["seed_hex"])
        grown = grow_weights(
            seed=seed, shape=(rows, cols),
            steps=config["steps"],
            target_se=config.get("target_se", 1.5),
            c=config["c"], gamma=config["gamma"],
            geometry=config["geometry"],
            k_neighbors=config.get("k_neighbors", 4),
        )

    original_shape = config.get("original_shape", config["shape"])
    if len(original_shape) == 1:
        grown = grown.ravel()[:original_shape[0]]

    return grown


def morpho_decode_block(
    tensor_dir: Path, start_row: int, end_row: int
) -> np.ndarray:
    """Decode a block of rows from morpho config."""
    full = morpho_decode(tensor_dir)
    if full.ndim == 1:
        return full[start_row:end_row]
    return full[start_row:end_row]
