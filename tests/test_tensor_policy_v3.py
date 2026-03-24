"""Tests for CDNA v3 tensor classification and policy assignment."""

from helix_substrate.tensor_policy import (
    TensorClass,
    TensorPolicy,
    classify_tensor,
    get_default_policy,
    parse_tensor_name,
)


class TestClassifyGGUFNames:
    def test_attn_q(self):
        assert classify_tensor("blk.0.attn_q.weight") == TensorClass.ATTENTION_QK

    def test_attn_k(self):
        assert classify_tensor("blk.5.attn_k.weight") == TensorClass.ATTENTION_QK

    def test_attn_v(self):
        assert classify_tensor("blk.0.attn_v.weight") == TensorClass.ATTENTION_VO

    def test_attn_output(self):
        assert classify_tensor("blk.31.attn_output.weight") == TensorClass.ATTENTION_VO

    def test_ffn_gate(self):
        assert classify_tensor("blk.0.ffn_gate.weight") == TensorClass.FFN

    def test_ffn_up(self):
        assert classify_tensor("blk.0.ffn_up.weight") == TensorClass.FFN

    def test_ffn_down(self):
        assert classify_tensor("blk.2.ffn_down.weight") == TensorClass.FFN

    def test_embedding(self):
        assert classify_tensor("token_embd.weight") == TensorClass.EMBEDDING

    def test_lm_head(self):
        assert classify_tensor("output.weight") == TensorClass.LM_HEAD

    def test_output_norm(self):
        assert classify_tensor("output_norm.weight") == TensorClass.NORM

    def test_layer_norm(self):
        assert classify_tensor("blk.0.attn_norm.weight") == TensorClass.NORM


class TestClassifyHFNames:
    def test_q_proj(self):
        assert classify_tensor("model.layers.0.self_attn.q_proj.weight") == TensorClass.ATTENTION_QK

    def test_k_proj(self):
        assert classify_tensor("model.layers.0.self_attn.k_proj.weight") == TensorClass.ATTENTION_QK

    def test_v_proj(self):
        assert classify_tensor("model.layers.0.self_attn.v_proj.weight") == TensorClass.ATTENTION_VO

    def test_o_proj(self):
        assert classify_tensor("model.layers.0.self_attn.o_proj.weight") == TensorClass.ATTENTION_VO

    def test_gate_proj(self):
        assert classify_tensor("model.layers.0.mlp.gate_proj.weight") == TensorClass.FFN

    def test_up_proj(self):
        assert classify_tensor("model.layers.0.mlp.up_proj.weight") == TensorClass.FFN

    def test_down_proj(self):
        assert classify_tensor("model.layers.0.mlp.down_proj.weight") == TensorClass.FFN

    def test_embed_tokens(self):
        assert classify_tensor("model.embed_tokens.weight") == TensorClass.EMBEDDING

    def test_lm_head(self):
        assert classify_tensor("lm_head.weight") == TensorClass.LM_HEAD


class TestClassify1DTensors:
    def test_1d_always_norm(self):
        assert classify_tensor("blk.0.ffn_down.weight", shape=(2048,)) == TensorClass.NORM

    def test_1d_unknown_name_still_norm(self):
        assert classify_tensor("some_random_bias", shape=(512,)) == TensorClass.NORM

    def test_2d_uses_name(self):
        assert classify_tensor("blk.0.ffn_down.weight", shape=(2048, 5632)) == TensorClass.FFN


class TestUnknownFallback:
    def test_unrecognized_name(self):
        assert classify_tensor("weird_tensor_name") == TensorClass.UNKNOWN

    def test_partial_match_fails(self):
        assert classify_tensor("not_a_blk.attn_q.weight") == TensorClass.UNKNOWN


class TestDefaultPolicies:
    def test_norm_is_exact(self):
        p = get_default_policy(TensorClass.NORM)
        assert p.storage_mode == "exact"
        assert p.sidecar_enabled is False

    def test_ffn_has_sidecar(self):
        p = get_default_policy(TensorClass.FFN)
        assert p.storage_mode == "codebook+sidecar"
        assert p.sidecar_enabled is True
        assert p.n_clusters == 256

    def test_lm_head_has_sidecar(self):
        p = get_default_policy(TensorClass.LM_HEAD)
        assert p.sidecar_enabled is True

    def test_embedding_no_sidecar(self):
        p = get_default_policy(TensorClass.EMBEDDING)
        assert p.sidecar_enabled is False
        assert p.use_kmeans is False

    def test_all_classes_have_policy(self):
        for tc in TensorClass:
            p = get_default_policy(tc)
            assert isinstance(p, TensorPolicy)
            assert p.tensor_class == tc


class TestParseTensorName:
    def test_gguf_ffn(self):
        info = parse_tensor_name("blk.2.ffn_down.weight")
        assert info["layer_idx"] == 2
        assert info["module_family"] == "ffn"
        assert info["projection"] == "down"

    def test_gguf_attn(self):
        info = parse_tensor_name("blk.0.attn_q.weight")
        assert info["layer_idx"] == 0
        assert info["module_family"] == "attention"
        assert info["projection"] == "q"

    def test_hf_attn(self):
        info = parse_tensor_name("model.layers.5.self_attn.q_proj.weight")
        assert info["layer_idx"] == 5
        assert info["module_family"] == "attention"

    def test_hf_ffn(self):
        info = parse_tensor_name("model.layers.3.mlp.down_proj.weight")
        assert info["layer_idx"] == 3
        assert info["module_family"] == "ffn"

    def test_embedding(self):
        info = parse_tensor_name("token_embd.weight")
        assert info["module_family"] == "embedding"

    def test_unknown(self):
        info = parse_tensor_name("weird_thing")
        assert info["module_family"] == "unknown"
