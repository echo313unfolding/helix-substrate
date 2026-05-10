"""Gate 1 tests: LoRA on HelixLinearSTE.

Tests:
1. Output at step 0 matches base (B=zeros)
2. Parameter hash: base params byte-identical after optimizer step on LoRA-only
3. LoRA params changed after optimizer step
4. STE bypass when codebook frozen (no VQStraightThrough in graph)
5. merge_lora produces valid compressed layer
6. Gradients flow only to LoRA when base frozen
7. Scalar VQ (d=1) and grouped VQ (d=4) both work

Work Order: WO-GATE1-LORA-STE-01
"""

import hashlib
import torch
import torch.nn as nn

from helix_substrate.helix_linear_ste import HelixLinearSTE


def _tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw bytes of a tensor, handling bfloat16 (no numpy support for bf16)."""
    t = t.data.cpu().contiguous()
    if t.dtype == torch.bfloat16:
        # bf16→fp32 is lossless; every bf16 value maps to exactly one fp32 value
        t = t.float()
    return t.numpy().tobytes()


def _param_hash(p: torch.Tensor) -> str:
    """SHA256 of parameter bytes — detects any mutation."""
    return hashlib.sha256(_tensor_bytes(p)).hexdigest()


def _buffer_hash(b: torch.Tensor) -> str:
    """SHA256 of buffer bytes."""
    return hashlib.sha256(_tensor_bytes(b)).hexdigest()


class TestLoRABasic:
    """Test LoRA enable/disable and step-0 output identity."""

    def test_step0_output_identical_scalar(self):
        """With B=zeros, LoRA output must exactly equal base output."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=1)
        x = torch.randn(4, 64)

        base_out = layer(x).clone()

        layer.enable_lora(rank=8, alpha=1.0)
        lora_out = layer(x)

        assert torch.equal(base_out, lora_out), \
            f"Step-0 output differs: max delta = {(base_out - lora_out).abs().max().item()}"

    def test_step0_output_identical_grouped(self):
        """Same test for grouped VQ (vector_dim=4)."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=4)
        x = torch.randn(4, 64)

        base_out = layer(x).clone()

        layer.enable_lora(rank=8, alpha=2.0)
        lora_out = layer(x)

        assert torch.equal(base_out, lora_out), \
            f"Step-0 output differs: max delta = {(base_out - lora_out).abs().max().item()}"

    def test_step0_output_identical_with_svd(self):
        """With SVD residual + LoRA, step-0 output still matches base."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=1, svd_rank=4)
        x = torch.randn(4, 64)

        base_out = layer(x).clone()

        layer.enable_lora(rank=8)
        lora_out = layer(x)

        assert torch.equal(base_out, lora_out), \
            f"Step-0 output differs: max delta = {(base_out - lora_out).abs().max().item()}"

    def test_disable_lora(self):
        """After disable_lora(), output reverts to base."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=1)
        x = torch.randn(4, 64)

        base_out = layer(x).clone()

        layer.enable_lora(rank=8)
        # Mutate B so LoRA has non-zero effect
        layer.lora_B.data.fill_(1.0)
        lora_out = layer(x)
        assert not torch.equal(base_out, lora_out), "LoRA should change output"

        layer.disable_lora()
        restored_out = layer(x)
        assert torch.equal(base_out, restored_out), "After disable, output should match base"


class TestParameterIsolation:
    """THE critical test: base params frozen, LoRA trains, base unchanged."""

    def test_param_hash_frozen_after_step(self):
        """After one optimizer step on LoRA-only, base params are byte-identical."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=4, svd_rank=4, bias=True)

        # Snapshot all base param/buffer hashes BEFORE
        hashes_before = {
            "codebook": _param_hash(layer.codebook),
            "indices": _buffer_hash(layer.indices),
            "svd_U": _param_hash(layer.svd_U),
            "svd_s": _param_hash(layer.svd_s),
            "svd_Vt": _param_hash(layer.svd_Vt),
            "bias": _param_hash(layer.bias),
        }

        # Enable LoRA, freeze base
        layer.enable_lora(rank=8, alpha=1.0)
        layer.freeze_base()

        # Snapshot LoRA hashes BEFORE
        lora_A_before = _param_hash(layer.lora_A)
        lora_B_before = _param_hash(layer.lora_B)

        # One optimizer step
        optimizer = torch.optim.SGD(
            [p for p in layer.parameters() if p.requires_grad], lr=0.01
        )
        x = torch.randn(4, 64)
        target = torch.randn(4, 32)
        loss = nn.functional.mse_loss(layer(x), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Verify base params are BYTE-IDENTICAL
        hashes_after = {
            "codebook": _param_hash(layer.codebook),
            "indices": _buffer_hash(layer.indices),
            "svd_U": _param_hash(layer.svd_U),
            "svd_s": _param_hash(layer.svd_s),
            "svd_Vt": _param_hash(layer.svd_Vt),
            "bias": _param_hash(layer.bias),
        }

        for name, h_before in hashes_before.items():
            h_after = hashes_after[name]
            assert h_before == h_after, \
                f"BASE PARAM MUTATED: {name} changed after LoRA step!"

        # Verify LoRA params DID change
        # B starts as zeros — after one grad step with non-zero grad, it must change
        # A also gets gradient through the chain
        lora_A_after = _param_hash(layer.lora_A)
        # Note: lora_B starts as zeros and gets gradient, so it WILL change
        lora_B_after = _param_hash(layer.lora_B)
        assert lora_B_after != lora_B_before, \
            "lora_B should have changed after optimizer step"

    def test_param_hash_frozen_scalar_vq(self):
        """Same test for scalar VQ (d=1)."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=1, bias=True)

        hashes_before = {
            "codebook": _param_hash(layer.codebook),
            "indices": _buffer_hash(layer.indices),
            "bias": _param_hash(layer.bias),
        }

        layer.enable_lora(rank=4, alpha=1.0)
        layer.freeze_base()

        optimizer = torch.optim.SGD(
            [p for p in layer.parameters() if p.requires_grad], lr=0.01
        )
        x = torch.randn(4, 64)
        target = torch.randn(4, 32)
        loss = nn.functional.mse_loss(layer(x), target)
        loss.backward()
        optimizer.step()

        hashes_after = {
            "codebook": _param_hash(layer.codebook),
            "indices": _buffer_hash(layer.indices),
            "bias": _param_hash(layer.bias),
        }

        for name in hashes_before:
            assert hashes_before[name] == hashes_after[name], \
                f"BASE PARAM MUTATED: {name}"

    def test_only_lora_params_have_grad(self):
        """When base is frozen, only lora_A and lora_B have requires_grad=True."""
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=4, svd_rank=4, bias=True)
        layer.enable_lora(rank=8)
        layer.freeze_base()

        trainable = {n for n, p in layer.named_parameters() if p.requires_grad}
        assert trainable == {"lora_A", "lora_B"}, \
            f"Expected only lora_A, lora_B trainable, got {trainable}"

    def test_gradient_flows_to_lora_only(self):
        """After backward, only LoRA params have non-None grad."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=4, svd_rank=2, bias=True)
        layer.enable_lora(rank=8)
        layer.freeze_base()

        x = torch.randn(4, 64)
        target = torch.randn(4, 32)
        loss = nn.functional.mse_loss(layer(x), target)
        loss.backward()

        # LoRA params MUST have gradients
        assert layer.lora_A.grad is not None, "lora_A should have gradient"
        assert layer.lora_B.grad is not None, "lora_B should have gradient"

        # Base params must NOT have gradients
        assert layer.codebook.grad is None, "codebook should have no gradient when frozen"
        assert layer.svd_U.grad is None, "svd_U should have no gradient when frozen"
        assert layer.svd_s.grad is None, "svd_s should have no gradient when frozen"
        assert layer.svd_Vt.grad is None, "svd_Vt should have no gradient when frozen"
        assert layer.bias.grad is None, "bias should have no gradient when frozen"


class TestSTEBypass:
    """Verify STE is bypassed when codebook is frozen."""

    def test_no_ste_in_frozen_mode(self):
        """When codebook.requires_grad=False, forward uses plain gather."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=1)
        layer.enable_lora(rank=4)
        layer.freeze_base()

        x = torch.randn(4, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        # x should have gradient (from LoRA path)
        assert x.grad is not None, "x should have gradient through LoRA"
        # codebook should NOT be in the grad graph at all
        assert layer.codebook.grad is None, "codebook should not accumulate grad"

    def test_ste_active_when_unfrozen(self):
        """When codebook.requires_grad=True, STE is used and codebook gets grad."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=1)
        layer.enable_lora(rank=4)
        # Don't freeze — codebook should get STE grad

        x = torch.randn(4, 64)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.codebook.grad is not None, "codebook should get STE gradient when unfrozen"
        assert layer.lora_B.grad is not None, "lora_B should also get gradient"


class TestMergeLoRA:
    """Test merge_lora absorbs LoRA delta into codebook."""

    def test_merge_preserves_effective_weights(self):
        """After merge, reconstructed W ≈ W_base + LoRA delta."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=4, svd_rank=2)
        layer.enable_lora(rank=8, alpha=1.0)

        # Set non-trivial LoRA weights
        layer.lora_A.data.normal_(0, 0.01)
        layer.lora_B.data.normal_(0, 0.01)

        # Compute effective W before merge
        x = torch.randn(8, 64)
        out_before = layer(x).detach().clone()

        # Merge
        layer.merge_lora()

        # After merge: no LoRA, but output should be close
        assert not layer.has_lora, "LoRA should be disabled after merge"
        out_after = layer(x)

        # Tolerance: k-means re-quantization introduces some error
        cos = nn.functional.cosine_similarity(
            out_before.reshape(1, -1), out_after.reshape(1, -1)
        ).item()
        assert cos > 0.99, f"Merge quality too low: cosine = {cos:.6f}"

    def test_merge_scalar_vq(self):
        """Merge works for scalar VQ (d=1)."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=1)
        layer.enable_lora(rank=4, alpha=1.0)
        layer.lora_A.data.normal_(0, 0.01)
        layer.lora_B.data.normal_(0, 0.01)

        x = torch.randn(8, 64)
        out_before = layer(x).detach().clone()

        layer.merge_lora()

        out_after = layer(x)
        cos = nn.functional.cosine_similarity(
            out_before.reshape(1, -1), out_after.reshape(1, -1)
        ).item()
        assert cos > 0.99, f"Merge quality too low: cosine = {cos:.6f}"


class TestUnfreezeAndRetrain:
    """Test the full cycle: train base → freeze → LoRA → merge → retrain."""

    def test_full_cycle(self):
        """Full lifecycle: create → LoRA → freeze → step → merge → unfreeze → step."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=4)
        x = torch.randn(4, 64)
        target = torch.randn(4, 32)

        # Phase 1: Born-compressed training (codebook trainable)
        opt1 = torch.optim.SGD(layer.parameters(), lr=0.001)
        loss1 = nn.functional.mse_loss(layer(x), target)
        loss1.backward()
        opt1.step()
        opt1.zero_grad()

        # Phase 2: Add LoRA, freeze base
        layer.enable_lora(rank=4)
        layer.freeze_base()
        opt2 = torch.optim.SGD(
            [p for p in layer.parameters() if p.requires_grad], lr=0.01
        )
        loss2 = nn.functional.mse_loss(layer(x), target)
        loss2.backward()
        opt2.step()
        opt2.zero_grad()

        # Phase 3: Merge LoRA back
        layer.merge_lora()

        # Phase 4: Resume born-compressed training
        assert layer.codebook.requires_grad, "Codebook should be trainable after merge"
        opt3 = torch.optim.SGD(layer.parameters(), lr=0.001)
        loss3 = nn.functional.mse_loss(layer(x), target)
        loss3.backward()
        opt3.step()

        # Verify codebook got gradient in phase 4
        assert layer.codebook.grad is not None, "Codebook should get STE grad after unfreeze"


class TestBFloat16Precision:
    """Cross-precision tests: base in bfloat16, LoRA init and training still work.

    Production HXQ models load as bfloat16. LoRA must work at that precision.
    """

    def _make_bf16_layer(self, vector_dim=4, svd_rank=0, bias=False):
        """Create a HelixLinearSTE with codebook in bfloat16."""
        torch.manual_seed(42)
        layer = HelixLinearSTE.from_scratch(64, 32, vector_dim=vector_dim,
                                            svd_rank=svd_rank, bias=bias)
        # Cast codebook (and SVD if present) to bfloat16, simulating HXQ model load
        layer.codebook.data = layer.codebook.data.to(torch.bfloat16)
        if layer.has_svd:
            layer.svd_U.data = layer.svd_U.data.to(torch.bfloat16)
            layer.svd_s.data = layer.svd_s.data.to(torch.bfloat16)
            layer.svd_Vt.data = layer.svd_Vt.data.to(torch.bfloat16)
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.to(torch.bfloat16)
        return layer

    def test_lora_init_bf16_std_sane(self):
        """Kaiming init on bfloat16 layer produces sane std for A."""
        layer = self._make_bf16_layer()
        layer.enable_lora(rank=8)

        # A should be bfloat16 (inherits from codebook.dtype)
        assert layer.lora_A.dtype == torch.bfloat16, \
            f"lora_A dtype should be bfloat16, got {layer.lora_A.dtype}"
        assert layer.lora_B.dtype == torch.bfloat16

        # Kaiming std for fan_in=64: sqrt(2/64) ≈ 0.177 (with a=sqrt(5), bound ≈ 0.306)
        # In bf16 this should still be within a reasonable range
        std = layer.lora_A.float().std().item()
        assert 0.05 < std < 1.0, f"lora_A init std is {std}, expected ~0.18"

        # B should be exactly zero
        assert layer.lora_B.abs().max().item() == 0.0, "lora_B should be zeros"

    def test_step0_identity_bf16(self):
        """Output at step 0 matches base in bfloat16."""
        layer = self._make_bf16_layer(svd_rank=2)
        x = torch.randn(4, 64, dtype=torch.bfloat16)

        base_out = layer(x).clone()
        layer.enable_lora(rank=8)
        lora_out = layer(x)

        assert torch.equal(base_out, lora_out), \
            f"BF16 step-0 output differs: max delta = {(base_out - lora_out).abs().max().item()}"

    def test_param_hash_bf16(self):
        """Parameter hash test works in bfloat16."""
        layer = self._make_bf16_layer(svd_rank=2, bias=True)

        hashes_before = {
            "codebook": _param_hash(layer.codebook),
            "svd_U": _param_hash(layer.svd_U),
            "svd_s": _param_hash(layer.svd_s),
            "svd_Vt": _param_hash(layer.svd_Vt),
            "bias": _param_hash(layer.bias),
        }

        layer.enable_lora(rank=4)
        layer.freeze_base()

        optimizer = torch.optim.SGD(
            [p for p in layer.parameters() if p.requires_grad], lr=0.01
        )
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        target = torch.randn(4, 32, dtype=torch.bfloat16)
        loss = nn.functional.mse_loss(layer(x), target)
        loss.backward()
        optimizer.step()

        hashes_after = {
            "codebook": _param_hash(layer.codebook),
            "svd_U": _param_hash(layer.svd_U),
            "svd_s": _param_hash(layer.svd_s),
            "svd_Vt": _param_hash(layer.svd_Vt),
            "bias": _param_hash(layer.bias),
        }

        for name in hashes_before:
            assert hashes_before[name] == hashes_after[name], \
                f"BF16 BASE PARAM MUTATED: {name}"

    def test_gradient_flows_bf16(self):
        """Gradients flow to LoRA params in bfloat16."""
        layer = self._make_bf16_layer()
        layer.enable_lora(rank=4)
        layer.freeze_base()

        x = torch.randn(4, 64, dtype=torch.bfloat16)
        target = torch.randn(4, 32, dtype=torch.bfloat16)
        loss = nn.functional.mse_loss(layer(x), target)
        loss.backward()

        assert layer.lora_A.grad is not None, "lora_A should have gradient in bf16"
        assert layer.lora_B.grad is not None, "lora_B should have gradient in bf16"
        assert layer.lora_A.grad.dtype == torch.bfloat16
        assert layer.lora_B.grad.dtype == torch.bfloat16

    def test_training_step_converges_bf16(self):
        """3 training steps in bf16 reduce loss (not diverging due to precision)."""
        layer = self._make_bf16_layer()
        layer.enable_lora(rank=8, alpha=1.0)
        layer.freeze_base()

        x = torch.randn(16, 64, dtype=torch.bfloat16)
        target = torch.randn(16, 32, dtype=torch.bfloat16)

        optimizer = torch.optim.Adam(
            [p for p in layer.parameters() if p.requires_grad], lr=0.01
        )

        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(layer(x), target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0], \
            f"BF16 loss not decreasing: {losses[0]:.4f} → {losses[-1]:.4f}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
