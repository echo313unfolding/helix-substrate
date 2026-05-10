# WO-STE-BACKWARD-C: Custom STE Backward for Born-Compressed Training

## Motivation

At 79M (8s/step), backward is 3.85s (48%) and vstep is 3.4s (43%).
The backward is ~18x the forward (0.21s), far exceeding the expected 2x ratio.

Two causes:
1. **Mamba selective scan backward**: sequential O(seq_len) per SSM state, many small matmuls
2. **Autograd overhead**: graph construction, intermediate tensor allocation, the STE detach trick

## Current STE Implementation

```python
# apply_quantized_weights():
shadow = weight.data.clone()           # 0.15s total across 40 layers
W_q = codebook[indices]                # reconstruct
weight.data = W_shadow + (W_q - W_shadow).detach()  # STE trick

# After forward + backward:
weight.data = shadow                   # restore
```

The `.detach()` trick creates extra graph nodes. Autograd traverses them during backward
even though the gradient through quantization is mathematically identity.

## Architecture

### Phase 1: torch.autograd.Function (no C needed)

Replace the clone+detach+restore pattern with a custom autograd function:

```python
class STELinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, shadow_weight, codebook, indices, sidecar_pos, sidecar_val, vd, bias=None):
        # Reconstruct W_q from codebook
        W_q = codebook[indices.long()]
        if vd > 1:
            W_q = W_q.reshape(shadow_weight.shape)
        # Apply sidecar corrections
        W_q_flat = W_q.reshape(-1)
        W_q_flat[sidecar_pos] += sidecar_val
        W_q = W_q_flat.reshape(shadow_weight.shape)

        ctx.save_for_backward(input, W_q)
        ctx.has_bias = bias is not None

        return F.linear(input, W_q, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, W_q = ctx.saved_tensors
        # grad_input = grad_output @ W_q (for chain rule propagation)
        grad_input = grad_output.matmul(W_q)
        # grad_shadow_weight = grad_output^T @ input (STE: identity through VQ)
        # For batched: [B, seq, out]^T @ [B, seq, in] -> [out, in]
        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).T @ input.reshape(-1, input.shape[-1])
        grad_bias = grad_output.sum(dim=(0, 1)) if ctx.has_bias else None
        # No gradient for codebook, indices, sidecar_pos, sidecar_val, vd
        return grad_input, grad_weight, None, None, None, None, None, grad_bias
```

**Savings**: eliminates clone (0.15s), restore (0.004s), and detach graph nodes.
**Risk**: moderate — same matmuls, less overhead.
**Expected speedup**: ~10-20% on backward (0.4-0.8s saved per step).

### Phase 2: C backward for linear layers (extend hxq_lloyd.c)

```c
// hxq_ste_backward.h

/**
 * Fused STE linear backward: compute weight gradient and input gradient
 * directly from codebook-reconstructed weights.
 *
 * Forward was: output = input @ W_q.T + bias
 *   where W_q = codebook[indices].reshape(out_dim, in_dim)
 *
 * Backward computes:
 *   grad_weight = grad_output.T @ input      [out_dim, in_dim]
 *   grad_input  = grad_output @ W_q          [batch*seq, in_dim]
 *
 * W_q is reconstructed on-the-fly from codebook+indices (never materialized
 * as a full matrix). This halves memory traffic vs PyTorch's approach.
 *
 * @param grad_output   [N, out_dim] float32 (N = batch * seq_len)
 * @param input         [N, in_dim] float32
 * @param codebook      [k, d] float32  (d = vector_dim)
 * @param indices       [out_dim, in_dim/d] uint8
 * @param out_dim       output dimension
 * @param in_dim        input dimension
 * @param vector_dim    VQ group size (1, 2, 4, 8)
 * @param N             batch * seq_len
 * @param grad_weight   [out_dim, in_dim] float32 OUTPUT — accumulated to shadow
 * @param grad_input    [N, in_dim] float32 OUTPUT
 *
 * @return 0 on success
 */
int hxq_ste_backward(
    const float *grad_output,
    const float *input,
    const float *codebook,
    const uint8_t *indices,
    int out_dim,
    int in_dim,
    int vector_dim,
    int N,
    float *grad_weight,
    float *grad_input
);
```

**Key optimization**: W_q is never materialized as a contiguous [out_dim, in_dim] matrix.
Instead, during the `grad_input = grad_output @ W_q` computation, each row of W_q is
reconstructed from codebook+indices on-the-fly. This means:
- No malloc for the full W_q matrix
- Better cache behavior (codebook stays in L1/L2)
- OpenMP parallelism over the N dimension

**Savings**: eliminates W_q materialization (~2.4MB per 768x768 layer), reduces memory
bandwidth, enables SIMD on the inner loop.
**Risk**: high — must exactly match PyTorch's gradient computation.
**Expected speedup**: 2-5x on the linear layer backward portion.

### Phase 3: C backward for Mamba selective scan (future)

The Mamba scan backward is the hardest target. It involves:
- Sequential gradient propagation through SSM states (can't parallelize across time)
- Many small matrix operations (d_state x d_inner per time step)
- Complex gradient flow through gates (dt, B, C, D)

This is NOT a week of work — it's a research project. The selective scan backward in
Mamba's reference implementation (mamba_ssm) uses a custom CUDA kernel. A CPU C version
would need to replicate that logic.

**Recommendation**: Skip Phase 3 for now. The GPU path (when rented) uses mamba_ssm's
CUDA kernel which is already fast. The C backward is only needed for CPU training,
and Phases 1+2 get enough speedup to make the Dell useful.

## Test Plan

### Phase 1 tests (torch.autograd.Function):
1. **Gradient correctness**: `torch.autograd.gradcheck` on STELinear with small inputs
2. **Numerical match**: Run 10 steps with old STE and new STELinear, compare loss curves
   within float32 tolerance (max diff < 0.01)
3. **No shadow leakage**: Verify shadow weights are updated by optimizer, not by forward
4. **Sidecar correctness**: Verify sidecar corrections are applied in forward

### Phase 2 tests (C backward):
5. **C vs PyTorch grad match**: Compare grad_weight and grad_input against PyTorch reference
   for random inputs, max elementwise diff < 1e-5
6. **Grouped VQ**: Test d=1, d=2, d=4 vector dims
7. **Training loop match**: 10 steps C backward vs PyTorch, loss diff < 0.01
8. **Memory check**: Verify no leaks (valgrind)
9. **Benchmark**: Measure per-layer backward time, compare C vs PyTorch

## Projected Impact

| Phase | Backward savings | Step time (79M) | 2000 steps |
|-------|-----------------|-----------------|------------|
| Current | — | 8.0s | 4.4 hr |
| Phase 1 (autograd.Function) | ~15% | 7.2s | 4.0 hr |
| Phase 2 (C linear backward) | ~40% | 5.5s | 3.1 hr |
| Phase 2 + batched vstep | ~40% + vstep 50% | 3.8s | 2.1 hr |

Phase 1 is a few hours of work. Phase 2 is 2-3 days. Together they cut step time roughly in half.

## Build

Phase 2 extends the existing hxq-native library:
```bash
# Add to hxq-native/src/hxq_ste.c, hxq-native/include/hxq_ste.h
gcc -O3 -march=native -fopenmp -shared -fPIC \
    -o lib/libhxq_ste.so src/hxq_ste.c -lm

# Or add to existing libhxq_lloyd.so build
```

## Dependencies

- hxq-native/lib/libhxq_lloyd.so (already built)
- Phase 1: pure Python, no new dependencies
- Phase 2: extends hxq-native Makefile
