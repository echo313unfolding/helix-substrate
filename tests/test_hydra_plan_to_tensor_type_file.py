import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "hydra_plan_to_tensor_type_file.py"
spec = importlib.util.spec_from_file_location("bridge", MODULE_PATH)
bridge = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(bridge)


def qwen_mapper(name, config):
    assert config["model_type"] == "qwen2"
    table = {
        "model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight",
        "model.layers.0.mlp.down_proj.weight": "blk.0.ffn_down.weight",
        "model.norm.weight": "output_norm.weight",
        "duplicate.alias": "blk.0.attn_q.weight",
        "skip.me": None,
    }
    return table[name]


def plan(items):
    return {"model": "qwen-test", "policy": "edge_balanced", "plan": items}


def test_affine6_and_exact_mapping():
    p = plan([
        {"tensor_name": "model.layers.0.self_attn.q_proj.weight", "head": "affine6"},
        {"tensor_name": "model.norm.weight", "head": "exact"},
    ])
    entries, meta = bridge.build_tensor_type_entries(
        p, {"model_type": "qwen2"}, qwen_mapper
    )
    assert entries == [
        ("blk.0.attn_q.weight", "hxq_affine_6"),
        ("output_norm.weight", "f16"),
    ]
    assert meta["head_counts"] == {"affine6": 1, "exact": 1}


def test_exact_can_be_f32():
    p = plan([{"tensor_name": "model.norm.weight", "head": "exact"}])
    entries, _ = bridge.build_tensor_type_entries(
        p, {"model_type": "qwen2"}, qwen_mapper, exact_type="f32"
    )
    assert entries == [("output_norm.weight", "f32")]


def test_unsupported_head_fails_closed():
    p = plan([
        {"tensor_name": "model.layers.0.mlp.down_proj.weight", "head": "affine5"},
    ])
    with pytest.raises(bridge.BridgeError, match="intentionally limited"):
        bridge.build_tensor_type_entries(p, {"model_type": "qwen2"}, qwen_mapper)


def test_mapper_none_is_skipped_and_receipted():
    p = plan([
        {"tensor_name": "skip.me", "head": "exact"},
        {"tensor_name": "model.norm.weight", "head": "exact"},
    ])
    entries, meta = bridge.build_tensor_type_entries(
        p, {"model_type": "qwen2"}, qwen_mapper
    )
    assert entries == [("output_norm.weight", "f16")]
    assert meta["skipped_mapper_none"] == ["skip.me"]


def test_duplicate_gguf_names_fail():
    p = plan([
        {"tensor_name": "model.layers.0.self_attn.q_proj.weight", "head": "affine6"},
        {"tensor_name": "duplicate.alias", "head": "exact"},
    ])
    with pytest.raises(bridge.BridgeError, match="same GGUF tensor"):
        bridge.build_tensor_type_entries(p, {"model_type": "qwen2"}, qwen_mapper)


def test_output_is_deterministically_sorted():
    p = plan([
        {"tensor_name": "model.norm.weight", "head": "exact"},
        {"tensor_name": "model.layers.0.self_attn.q_proj.weight", "head": "affine6"},
        {"tensor_name": "model.layers.0.mlp.down_proj.weight", "head": "affine6"},
    ])
    entries, _ = bridge.build_tensor_type_entries(
        p, {"model_type": "qwen2"}, qwen_mapper
    )
    assert entries == sorted(entries)
