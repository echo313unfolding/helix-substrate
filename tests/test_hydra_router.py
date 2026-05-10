"""Tests for HXQ Hydra Router.

Verifies routing rules from docs/HXQ_HYDRA_ROUTER.md:
- High-kurtosis early attention routes to affine6
- Easy MLP tensor routes to affine5 under edge_balanced
- affine3 never selected unless gate passes
- size_target prefers lower bit but respects fragile fallback
- sidecar selected only when residual budget improves quality estimate
- exact tensors (embed, lm_head, norm) always route to exact
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.hydra_router import (
    Head,
    HydraRouter,
    TensorProfile,
    CompressionPlan,
    POLICIES,
    profiles_from_probe_receipt,
)


# ── Fixtures ──

def _make_profile(
    name="model.layers.5.mlp.gate_proj.weight",
    tensor_type="gate_proj",
    layer=5,
    shape=(4096, 2048),
    kurtosis=3.0,
    affine6_cos=0.9995,
    affine5_cos=0.999,
    affine4_cos=0.995,
    affine3_cos=0.97,
) -> TensorProfile:
    return TensorProfile(
        tensor_name=name,
        shape=shape,
        layer_index=layer,
        tensor_type=tensor_type,
        n_params=shape[0] * shape[1],
        kurtosis=kurtosis,
        std=0.01,
        affine6_cosine=affine6_cos,
        affine5_cosine=affine5_cos,
        affine4_cosine=affine4_cos,
        affine3_cosine=affine3_cos,
        max_abs_error=0.01,
        mean_abs_error=0.001,
    )


def _easy_mlp():
    """MLP tensor with good cosine at all bit levels."""
    return _make_profile(
        name="model.layers.10.mlp.gate_proj.weight",
        tensor_type="gate_proj",
        layer=10,
        kurtosis=2.5,
        affine5_cos=0.9992,
        affine4_cos=0.9995,
        affine3_cos=0.985,
    )


def _fragile_early_qkv():
    """Early-layer attention with high kurtosis."""
    return _make_profile(
        name="model.layers.0.self_attn.q_proj.weight",
        tensor_type="q_proj",
        layer=0,
        kurtosis=142.0,
        affine6_cos=0.9992,
        affine5_cos=0.9968,
        affine4_cos=0.987,
        affine3_cos=0.944,
    )


def _embed_tensor():
    return _make_profile(
        name="model.embed_tokens.weight",
        tensor_type="embed_tokens",
        layer=0,
        shape=(32000, 2048),
    )


def _norm_tensor():
    return _make_profile(
        name="model.layers.5.input_layernorm.weight",
        tensor_type="input_layernorm",
        layer=5,
        shape=(2048, 1),
    )


def _barely_fails_affine5():
    """Tensor that fails affine5 cosine gate — falls to affine6."""
    return _make_profile(
        name="model.layers.3.self_attn.v_proj.weight",
        tensor_type="v_proj",
        layer=3,
        kurtosis=8.0,
        affine5_cos=0.994,  # below 0.998 gate
        affine4_cos=0.990,
    )


def _almost_passes_affine5():
    """Tensor barely below affine5 gate — still falls to affine6, no sidecar."""
    return _make_profile(
        name="model.layers.4.self_attn.o_proj.weight",
        tensor_type="o_proj",
        layer=4,
        kurtosis=10.0,
        affine5_cos=0.9965,  # below 0.998 gate — no sidecar for affine
    )


# ── Tests ──

def test_all_policies_exist():
    assert len(POLICIES) == 4
    for p in POLICIES:
        router = HydraRouter(policy=p)
        assert router.policy == p


def test_invalid_policy_raises():
    try:
        HydraRouter(policy="nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_exact_tensors_always_exact():
    """Embed, lm_head, norm tensors must route to exact under all policies."""
    embed = _embed_tensor()
    norm = _norm_tensor()

    for policy in POLICIES:
        router = HydraRouter(policy=policy)
        plan = router.route([embed, norm], model_name="test")
        for tp in plan.tensors:
            assert tp.head == Head.EXACT, (
                f"Policy {policy}: {tp.tensor_name} routed to {tp.head}, expected exact"
            )


def test_high_kurtosis_early_attention_routes_affine6():
    """High-kurtosis early q_proj must route to affine6 under edge_balanced."""
    fragile = _fragile_early_qkv()
    router = HydraRouter(policy="edge_balanced")
    plan = router.route([fragile])
    tp = plan.tensors[0]
    assert tp.head == Head.AFFINE6, f"Expected affine6, got {tp.head}"
    assert any("early_attention" in r or "high_kurtosis" in r for r in tp.reason)


def test_easy_mlp_routes_affine5_edge_balanced():
    """Easy MLP tensor should route to affine5 under edge_balanced."""
    easy = _easy_mlp()
    router = HydraRouter(policy="edge_balanced")
    plan = router.route([easy])
    tp = plan.tensors[0]
    assert tp.head == Head.AFFINE5, f"Expected affine5, got {tp.head}"
    assert "cosine_pass" in tp.reason


def test_quality_first_prefers_affine6():
    """quality_first should use affine6 for normal tensors, not affine5."""
    easy = _easy_mlp()
    router = HydraRouter(policy="quality_first")
    plan = router.route([easy])
    tp = plan.tensors[0]
    assert tp.head == Head.AFFINE6, f"Expected affine6, got {tp.head}"


def test_affine3_never_default():
    """affine3 must never be selected unless the cosine gate explicitly passes."""
    # Tensor with terrible affine3 cosine
    p = _make_profile(affine3_cos=0.95, kurtosis=2.0)
    for policy in POLICIES:
        router = HydraRouter(policy=policy)
        plan = router.route([p])
        tp = plan.tensors[0]
        assert tp.head != Head.AFFINE3, (
            f"Policy {policy}: affine3 selected with cosine 0.95"
        )


def test_affine3_selected_when_gate_passes():
    """experimental_lowbit should select affine3 when cosine AND kurtosis pass."""
    p = _make_profile(
        kurtosis=2.0,        # low kurtosis (< 5.0)
        affine3_cos=0.999,   # above 0.998 gate
        affine4_cos=0.9995,
    )
    router = HydraRouter(policy="experimental_lowbit")
    plan = router.route([p])
    tp = plan.tensors[0]
    assert tp.head == Head.AFFINE3, f"Expected affine3, got {tp.head}"


def test_affine4_only_experimental():
    """affine4 should only appear under experimental_lowbit."""
    p = _make_profile(
        kurtosis=3.0,
        affine4_cos=0.9995,
        affine5_cos=0.999,
    )
    # Non-experimental policies should not use affine4
    for policy in ("quality_first", "edge_balanced", "size_target"):
        router = HydraRouter(policy=policy)
        plan = router.route([p])
        tp = plan.tensors[0]
        assert tp.head != Head.AFFINE4, (
            f"Policy {policy}: affine4 selected unexpectedly"
        )

    # Experimental should use it
    router = HydraRouter(policy="experimental_lowbit")
    plan = router.route([p])
    tp = plan.tensors[0]
    assert tp.head == Head.AFFINE4, f"Expected affine4 under experimental, got {tp.head}"


def test_size_target_prefers_lower_bit():
    """size_target should prefer affine5 over affine6 when cosine passes."""
    easy = _easy_mlp()
    router = HydraRouter(policy="size_target")
    plan = router.route([easy])
    tp = plan.tensors[0]
    assert tp.head == Head.AFFINE5, f"Expected affine5, got {tp.head}"


def test_size_target_respects_fragile():
    """size_target should fall back to affine6 when cosine fails."""
    fragile = _barely_fails_affine5()
    router = HydraRouter(policy="size_target")
    plan = router.route([fragile])
    tp = plan.tensors[0]
    assert tp.head == Head.AFFINE6, f"Expected affine6 for fragile, got {tp.head}"


def test_no_sidecar_for_affine_tensors():
    """Sidecar must NOT be assigned for uniform affine heads.

    Uniform affine error is distributed (not sparse outliers), so sidecar
    is useless. Tensor that barely fails affine5 should fall to affine6
    with zero sidecar budget.
    Receipt: receipts/hxq_mixed_lowbit_sidecar/sidecar_20260430T192842.json
    """
    almost = _almost_passes_affine5()
    for policy in POLICIES:
        router = HydraRouter(policy=policy)
        plan = router.route([almost])
        tp = plan.tensors[0]
        assert tp.sidecar_budget == 0, (
            f"Policy {policy}: sidecar_budget={tp.sidecar_budget} on affine tensor"
        )
        assert "sidecar" not in " ".join(tp.reason).lower(), (
            f"Policy {policy}: sidecar mentioned in reason for affine tensor"
        )


def test_avg_bpw_calculation():
    """avg_bpw should be a weighted average by param count."""
    profiles = [_easy_mlp(), _fragile_early_qkv(), _embed_tensor()]
    router = HydraRouter(policy="edge_balanced")
    plan = router.route(profiles, model_name="test")
    bpw = plan.avg_bpw
    # Should be between affine5 (5.25) and exact (16.0)
    assert 5.0 < bpw < 16.0, f"avg_bpw {bpw} out of expected range"


def test_compression_plan_summary():
    """CompressionPlan.summary() should return correct structure."""
    profiles = [_easy_mlp(), _fragile_early_qkv()]
    router = HydraRouter(policy="edge_balanced")
    plan = router.route(profiles, model_name="test")
    s = plan.summary()
    assert s["model"] == "test"
    assert s["policy"] == "edge_balanced"
    assert s["n_tensors"] == 2
    assert "head_distribution" in s
    assert isinstance(s["avg_bpw"], float)


def test_compression_plan_to_json():
    """to_json should produce valid JSON with required fields."""
    profiles = [_easy_mlp()]
    router = HydraRouter(policy="edge_balanced")
    plan = router.route(profiles, model_name="test")
    text = plan.to_json()
    data = json.loads(text)
    assert "plan" in data
    assert len(data["plan"]) == 1
    assert data["plan"][0]["head"] == "affine5"
    assert "reason" in data["plan"][0]


def test_fallback_head_always_set():
    """Every tensor plan should have a valid fallback_head."""
    profiles = [_easy_mlp(), _fragile_early_qkv(), _embed_tensor()]
    for policy in POLICIES:
        router = HydraRouter(policy=policy)
        plan = router.route(profiles)
        for tp in plan.tensors:
            assert isinstance(tp.fallback_head, Head), (
                f"Missing fallback_head on {tp.tensor_name}"
            )


def test_head_bpw_values():
    """Head.bpw should return known values."""
    assert Head.EXACT.bpw == 16.0
    assert Head.AFFINE6.bpw == 6.25
    assert Head.AFFINE5.bpw == 5.25
    assert Head.AFFINE4.bpw == 4.25
    assert Head.AFFINE3.bpw == 3.25


def test_profiles_from_receipt():
    """profiles_from_probe_receipt should load from actual receipt if available."""
    receipt_path = (Path.home() / "helix-substrate" / "receipts" /
                    "hxq_mixed_lowbit_probe" / "mixed_lowbit_20260430T190238.json")
    if not receipt_path.exists():
        return  # skip if receipt not available

    profiles = profiles_from_probe_receipt(receipt_path)
    assert len(profiles) == 154, f"Expected 154 tensors, got {len(profiles)}"
    assert all(p.tensor_name for p in profiles)
    assert all(p.n_params > 0 for p in profiles)

    # Route them
    router = HydraRouter(policy="edge_balanced")
    plan = router.route(profiles, model_name="TinyLlama-1.1B")
    assert plan.avg_bpw > 0
    s = plan.summary()
    assert s["n_tensors"] == 154


def test_mixed_model_routing():
    """Route a mix of tensor types and verify distribution makes sense."""
    profiles = [
        _embed_tensor(),
        _norm_tensor(),
        _fragile_early_qkv(),
        _easy_mlp(),
        _barely_fails_affine5(),
    ]
    router = HydraRouter(policy="edge_balanced")
    plan = router.route(profiles, model_name="test")
    s = plan.summary()

    # Should have exact (embed + norm), affine6 (fragile + barely_fails), affine5 (easy)
    dist = s["head_distribution"]
    assert dist.get("exact", 0) >= 2, "embed + norm should be exact"
    assert "affine6" in dist, "Fragile tensors should be affine6"
    assert "affine5" in dist, "Easy MLP should be affine5"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")
    if failed:
        sys.exit(1)
