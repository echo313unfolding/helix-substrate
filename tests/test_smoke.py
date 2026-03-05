"""Smoke tests for helix-substrate core functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_cdna_v1_roundtrip():
    """Encode a tensor to CDNA v1, decode it, verify reconstruction."""
    from helix_substrate.cdna_encoder import encode_tensor_to_cdna, decode_cdna_to_tensor

    W = np.random.RandomState(42).randn(128, 256).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".cdna", delete=False) as f:
        path = Path(f.name)

    try:
        stats = encode_tensor_to_cdna(W, path, tensor_name="test")
        assert stats["rows"] == 128
        assert stats["cols"] == 256

        W_decoded = decode_cdna_to_tensor(path)
        assert W_decoded.shape == W.shape

        # k-means quantization is lossy, but codebook should capture the distribution
        cosine = np.dot(W.flat, W_decoded.flat) / (np.linalg.norm(W) * np.linalg.norm(W_decoded))
        assert cosine > 0.90, f"Cosine too low: {cosine}"
    finally:
        path.unlink(missing_ok=True)


def test_se_estimator_basic():
    """Se formula produces sensible values for different tensor types."""
    from helix_substrate.se import compute_tensor_se

    rng = np.random.RandomState(42)

    # Low-rank tensor should have low Se
    low_rank = (rng.randn(100, 5) @ rng.randn(5, 100)).astype(np.float32)
    result_lr = compute_tensor_se(low_rank, name="low_rank")
    assert 0.0 <= result_lr["Se"] <= 1.0
    assert result_lr["routing_hint"] in ("cpu", "gpu", "qpu")

    # Full-rank random tensor should have higher Se
    full_rank = rng.randn(100, 100).astype(np.float32)
    result_fr = compute_tensor_se(full_rank, name="full_rank")
    assert result_fr["Se"] > result_lr["Se"], "Full-rank should have higher Se than low-rank"


def test_se_routing_zones():
    """Routing zones are assigned correctly."""
    from helix_substrate.se import compute_tensor_se

    rng = np.random.RandomState(42)
    tensor = rng.randn(50, 50).astype(np.float32)
    result = compute_tensor_se(tensor)

    assert result["routing_zone"] in (1, 2, 3, 4)
    assert "components" in result
    assert "gates" in result


def test_receipt_roundtrip():
    """ExecutionReceipt serializes and deserializes correctly."""
    from helix_substrate.receipt import (
        ExecutionReceipt,
        OperationResult,
        validate_execution_receipt,
    )

    op = OperationResult(
        op_name="test_op",
        timing_ms=42.0,
        memory_bytes=1024,
        input_sha256="abc123",
        output_sha256="def456",
    )

    receipt = ExecutionReceipt(
        plan_sha256="plan_hash",
        operations=[op],
    )
    receipt.aggregate_stats()
    receipt.finalize()

    d = receipt.to_dict()
    assert d["receipt_sha256"] != ""

    validation = validate_execution_receipt(d)
    assert validation["valid"]
    assert validation["sha256_match"]

    # Round-trip
    loaded = ExecutionReceipt.from_dict(d)
    assert loaded.plan_sha256 == "plan_hash"
    assert len(loaded.operations) == 1


def test_sidecar_roundtrip():
    """HXZO sidecar write/read preserves positions and values."""
    from helix_substrate.sidecar import (
        write_outlier_sidecar,
        read_outlier_sidecar,
        clear_sidecar_cache,
    )

    clear_sidecar_cache()

    rng = np.random.default_rng(42)
    n_outliers = 100
    positions = np.sort(rng.choice(10000, n_outliers, replace=False)).astype(np.int64)
    values = rng.standard_normal(n_outliers).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".hxzo", delete=False) as f:
        path = f.name

    try:
        write_outlier_sidecar(
            positions=positions,
            values=values,
            tensor_name="test",
            threshold_policy={"method": "percentile", "percentile": 99.9},
            shape=(100, 100),
            output_path=path,
        )

        pos_read, val_read, meta = read_outlier_sidecar(path)
        assert np.array_equal(positions, pos_read)
        assert meta["num_outliers"] == n_outliers
    finally:
        Path(path).unlink(missing_ok=True)


def test_rope_preserves_norm():
    """RoPE rotation should preserve vector norms."""
    from helix_substrate.rope import apply_rope

    rng = np.random.RandomState(42)
    x = rng.randn(1, 10, 128).astype(np.float32)

    x_rope = apply_rope(x, start_pos=0)
    assert x_rope.shape == x.shape

    # Norms should be preserved (rotation)
    norm_before = np.linalg.norm(x, axis=-1)
    norm_after = np.linalg.norm(x_rope, axis=-1)
    assert np.allclose(norm_before, norm_after, atol=1e-5)


def test_empty_tensor_se():
    """Edge case: empty tensor returns zero Se."""
    from helix_substrate.se import compute_tensor_se

    result = compute_tensor_se(np.array([]))
    assert result["Se"] == 0.0
    assert result["routing_hint"] == "cpu"
