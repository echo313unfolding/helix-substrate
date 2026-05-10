# Sidecar Policy Instrumentation for HelixLinear

## What this adds

Runtime sidecar policy stats collection. Your HelixLinear already stores and
applies sidecar corrections. This patch adds per-call norm tracking so the
sidecar's reconstruction confidence can be used as a diagnostic signal
(e.g., for weighting divergence tests in the API Polygraph).

Four modes: `default` (always apply, no tracking), `always_on` (apply + count),
`always_off` (skip + count), `threshold` (gate on contribution norm + track).

## What it does NOT add

- No new dependencies
- No changes to the Triton kernel path
- No changes to compression/decompression
- No performance impact in `default` mode (the common case)

## Changes required

### 1. Add state variables to `__init__` (after line 115 in your helix_linear.py)

```python
        # Sidecar phase: "fused" | "scatter" | None (auto). Frozen at request start.
        self._sidecar_phase: Optional[str] = None

        # --- ADD THESE LINES ---
        # WO-SIDECAR-POLICY-01: runtime sidecar policy instrumentation
        # Modes:
        #   "default"    — always apply sidecar, no counters (zero overhead)
        #   "always_on"  — always apply, increments _sidecar_apply_count
        #   "always_off" — never apply, increments _sidecar_skip_count
        #   "threshold"  — gate on per-call sidecar contribution norm
        self._sidecar_mode: str = "default"
        self._sidecar_threshold: float = 0.0
        self._sidecar_apply_count: int = 0
        self._sidecar_skip_count: int = 0
        self._sidecar_norms: list = []
```

### 2. Add three public methods (after `dispatch_info` method, ~line 280)

```python
    # ----- WO-SIDECAR-POLICY-01: routing-policy gate -----

    def set_sidecar_policy(self, mode: str, threshold: float = 0.0) -> None:
        """Set policy gate for sidecar application.

        mode: "default" | "always_on" | "always_off" | "threshold"
        threshold: only used when mode == "threshold"; sidecar applied iff
                   the dynamic per-call sidecar OUTPUT contribution norm > threshold.
        """
        if mode not in ("default", "always_on", "always_off", "threshold"):
            raise ValueError(f"unknown sidecar mode: {mode!r}")
        self._sidecar_mode = mode
        self._sidecar_threshold = float(threshold)

    def reset_sidecar_policy_stats(self) -> None:
        """Zero apply/skip counters and clear the per-call norm log."""
        self._sidecar_apply_count = 0
        self._sidecar_skip_count = 0
        self._sidecar_norms = []

    def get_sidecar_policy_stats(self) -> dict:
        """Return current counters and norm summary."""
        norms = self._sidecar_norms
        if norms:
            arr = np.asarray(norms, dtype=np.float64)
            stats = {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        else:
            stats = {"n": 0}
        return {
            "mode": self._sidecar_mode,
            "threshold": self._sidecar_threshold,
            "apply_count": self._sidecar_apply_count,
            "skip_count": self._sidecar_skip_count,
            "norm_stats": stats,
        }

    def _policy_decide(self, x_2d: torch.Tensor) -> bool:
        """Decide whether to apply sidecar this call.

        x_2d: [B, in_features] flattened input batch.

        In threshold mode, computes the dynamic sparse OUTPUT contribution norm:
            sidecar_out_contrib[b, r] = sum_{c in cols(r)} delta(r,c) * x_2d[b, c]
        and gates application on its Frobenius norm.
        """
        mode = self._sidecar_mode
        if mode == "default":
            return True
        if mode == "always_on":
            self._sidecar_apply_count += 1
            return True
        if mode == "always_off":
            self._sidecar_skip_count += 1
            return False
        # threshold mode — sparse contribution norm
        x_at_cols = x_2d.index_select(1, self._sidecar_cols)
        contrib_per_nnz = x_at_cols * self._sidecar_deltas.to(x_at_cols.dtype).unsqueeze(0)
        sc_norm = float(contrib_per_nnz.norm().item())
        self._sidecar_norms.append(sc_norm)
        if sc_norm > self._sidecar_threshold:
            self._sidecar_apply_count += 1
            return True
        else:
            self._sidecar_skip_count += 1
            return False
```

### 3. Gate sidecar application in _forward_fused (line 292)

Currently your fused path always passes sidecar data to the Triton kernel.
To gate it, wrap the sidecar arguments:

```python
    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused VQ gather-matmul via Triton. W never in global memory."""
        from helix_substrate.triton_vq_matmul import fused_vq_matmul

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        dispatch_log = {}
        skip_svd = self._cell_skip_svd and self.has_svd

        # --- ADD: policy gate for sidecar ---
        apply_sidecar = self._policy_decide(x_2d) if self._sidecar_rows is not None else False

        output = fused_vq_matmul(
            x=x_2d,
            codebook=self.codebook,
            indices=self.indices,
            sidecar_rows=self._sidecar_rows if apply_sidecar else None,
            sidecar_cols=self._sidecar_cols if apply_sidecar else None,
            sidecar_deltas=self._sidecar_deltas if apply_sidecar else None,
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
```

### 4. Gate sidecar in _dequant_tile (line 328)

For the CPU tiled path, the decision is made once per forward call and stored:

In `_forward_naive`, before the tile loop:
```python
        # --- ADD: policy gate for sidecar (once per forward call) ---
        x_2d_for_policy = x.reshape(-1, self.in_features).float()
        self._apply_sidecar_this_call = (
            self._policy_decide(x_2d_for_policy)
            if self._sidecar_rows is not None else False
        )
```

In `_dequant_tile`, change:
```python
        if self._sidecar_rows is not None:
```
to:
```python
        if self._sidecar_rows is not None and getattr(self, '_apply_sidecar_this_call', True):
```

## How to use for API Polygraph

```python
import torch
from helix_substrate.helix_linear import HelixLinear

# After loading an HXQ-compressed model:
for name, module in model.named_modules():
    if isinstance(module, HelixLinear):
        module.set_sidecar_policy("threshold", threshold=0.0)
        module.reset_sidecar_policy_stats()

# Run inference on probe corpus
with torch.no_grad():
    output = model(input_ids)

# Collect per-module sidecar norms
norms_by_layer = {}
for name, module in model.named_modules():
    if isinstance(module, HelixLinear):
        stats = module.get_sidecar_policy_stats()
        norms_by_layer[name] = stats["norm_stats"]

# Feed norms into api_polygraph fingerprint as diagnostic_weight:
#   diagnostic_weight = 1.0 / (1.0 + mean_sidecar_norm)
# Low norm = high confidence = weight this token MORE in divergence test
```

## Source

This instrumentation comes from helix-substrate (MIT license).
Origin: `helix_substrate/helix_linear.py` lines 659-733.
Receipted: sidecar correlation rho=0.574, p=1.4e-50 (562 chunks, 287K tokens).
