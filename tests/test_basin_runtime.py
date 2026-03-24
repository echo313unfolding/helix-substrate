"""
Tests for basin_runtime: capability check, strict benchmark mode, unsupported manifest,
receipt validation (structural + semantic), startup receipt, multi-model capability.

Work Order: WO-BASIN-HELIX-FUSED-01
Work Order: WO-BASIN-HARDENING-01
Work Order: WO-BASIN-RECEIPT-SEMANTICS-01
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from helix_substrate.basin_runtime import (
    check_cdnav3_capability,
    StrictBenchmarkMode,
    validate_receipt,
    validate_startup_receipt,
    validate_hardening_summary,
    build_startup_receipt,
    build_hardening_summary,
    RUNTIME_PATH_FUSED,
    RUNTIME_PATH_NAIVE,
    RUNTIME_PATH_MIXED,
    RUNTIME_PATH_UNSUPPORTED,
    RUNTIME_PATH_STARTUP_FAIL,
    RUNTIME_PATH_RECEIPT_FAIL,
    SCHEMA_REQUEST_RECEIPT,
    SCHEMA_STARTUP_RECEIPT,
    SCHEMA_HARDENING_SUMMARY,
)

# Real CDNA v3 paths
TINYLLAMA_CDNA = Path.home() / "models" / "tinyllama_fp32" / "cdnav3"
QWEN_CDNA = Path.home() / "models" / "qwen2.5-coder-1.5b-instruct" / "cdnav3"


# ---------------------------------------------------------------------------
# check_cdnav3_capability tests
# ---------------------------------------------------------------------------

class TestCheckCapability:

    def test_valid_tinyllama(self):
        """Real TinyLlama manifest -> valid=True, 154 tensors, no issues."""
        if not TINYLLAMA_CDNA.exists():
            pytest.skip("TinyLlama CDNA v3 not present")

        cap = check_cdnav3_capability(TINYLLAMA_CDNA)

        # CUDA/Triton might not be available in CI, so only check data fields
        data_issues = [i for i in cap["issues"]
                       if "CUDA" not in i and "Triton" not in i]
        assert len(data_issues) == 0, f"Unexpected issues: {data_issues}"
        assert cap["model_name"] == "TinyLlama-1.1B"
        assert cap["n_tensors"] == 154
        assert cap["n_blocks"] == 22
        assert cap["compression_ratio"] == 3.99
        assert cap["manifest_sha256"] is not None
        assert cap["tensor_dirs_found"] == 154

    def test_missing_dir(self):
        """Nonexistent dir -> valid=False, 'not found' in issues."""
        cap = check_cdnav3_capability("/nonexistent/path/cdnav3")

        assert cap["valid"] is False
        assert any("not found" in i for i in cap["issues"])
        assert cap["model_name"] is None

    def test_unsupported_manifest(self):
        """Manifest missing required fields -> valid=False with explicit issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cdna_dir = Path(tmpdir)

            # Write incomplete manifest (missing n_tensors, n_blocks)
            manifest = {
                "model": "TestModel",
                "compression_ratio": 3.5,
                # missing: n_tensors, n_blocks
            }
            (cdna_dir / "manifest.json").write_text(json.dumps(manifest))

            cap = check_cdnav3_capability(cdna_dir)

            assert cap["valid"] is False
            assert any("n_tensors" in i for i in cap["issues"]), \
                f"Expected n_tensors issue, got: {cap['issues']}"
            assert any("n_blocks" in i for i in cap["issues"]), \
                f"Expected n_blocks issue, got: {cap['issues']}"

    def test_no_manifest(self):
        """Empty dir with no manifest.json -> valid=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cap = check_cdnav3_capability(tmpdir)

            assert cap["valid"] is False
            assert any("manifest.json not found" in i for i in cap["issues"])

    def test_bad_tensor_count(self):
        """Manifest claims 10 tensors but dir has 0 -> issue reported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cdna_dir = Path(tmpdir)
            manifest = {
                "model": "TestModel",
                "n_blocks": 2,
                "n_tensors": 10,
                "compression_ratio": 3.0,
            }
            (cdna_dir / "manifest.json").write_text(json.dumps(manifest))

            cap = check_cdnav3_capability(cdna_dir)

            assert cap["valid"] is False
            assert any("Expected 10" in i for i in cap["issues"])


# ---------------------------------------------------------------------------
# StrictBenchmarkMode tests
# ---------------------------------------------------------------------------

class TestStrictBenchmarkMode:

    def test_cpu_naive_allowed(self):
        """CPU input on naive path does NOT raise -- expected behavior."""
        from helix_substrate.helix_linear import HelixLinear

        codebook = torch.randn(256)
        indices = torch.randint(0, 256, (32, 16), dtype=torch.uint8)
        helix = HelixLinear(16, 32, codebook, indices)

        x = torch.randn(1, 16)  # CPU input

        with StrictBenchmarkMode():
            out = helix(x)  # Should NOT raise

        assert out.shape == (1, 32)
        assert helix._last_dispatch_path == "naive"  # Expected for CPU

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_naive_raises(self):
        """CUDA input dispatched to naive (Triton unavailable) -> raises RuntimeError."""
        from helix_substrate.helix_linear import HelixLinear

        codebook = torch.randn(256, device="cuda")
        indices = torch.randint(0, 256, (32, 16), dtype=torch.uint8, device="cuda")
        helix = HelixLinear(16, 32, codebook, indices)
        helix = helix.cuda()

        x = torch.randn(1, 16, device="cuda")

        # Mock _use_fused to return False, forcing naive path on CUDA
        with patch.object(type(helix), '_use_fused', new_callable=lambda: property(lambda self: False)):
            with pytest.raises(RuntimeError, match="StrictBenchmarkMode"):
                with StrictBenchmarkMode():
                    helix(x)

    def test_forward_restored_after_exit(self):
        """Original forward is restored after context manager exits."""
        from helix_substrate.helix_linear import HelixLinear

        original_forward = HelixLinear.forward

        with StrictBenchmarkMode():
            assert HelixLinear.forward is not original_forward

        assert HelixLinear.forward is original_forward


# ---------------------------------------------------------------------------
# Receipt test helpers
# ---------------------------------------------------------------------------

def _make_valid_receipt(**overrides):
    """Build a minimal valid receipt for testing. Use overrides to tweak fields."""
    r = {
        "schema": SCHEMA_REQUEST_RECEIPT,
        "model_id": "TinyLlama-1.1B",
        "manifest_sha256": "abc123",
        "prompt_info": {
            "prompt_sha256": "def456",
            "prompt_length": 10,
            "max_tokens": 32,
            "seed": 42,
        },
        "timing": {
            "prompt_tokens": 5,
            "generated_tokens": 32,
            "prefill_ms": 100.0,
            "decode_ms": 500.0,
            "total_ms": 600.0,
            "tok_s": 15.0,
            "prefill_tok_s": 10.0,
            "backend": "helix_fused",
        },
        "dispatch_summary": {
            "total_helix_modules": 154,
            "fused_count": 154,
            "naive_count": 0,
            "unknown_count": 0,
            "all_fused": True,
            "runtime_path": RUNTIME_PATH_FUSED,
        },
        "kernel_metadata": {"kernel_version": "v3_tiled_fp16dot"},
        "fallback_reason": None,
        "load_metadata": {
            "helix_modules": 154,
            "compression_ratio": 3.99,
        },
        "cost": {
            "wall_time_s": 1.5,
            "cpu_time_s": 1.4,
            "peak_memory_mb": 2000.0,
            "python_version": "3.10.12",
            "hostname": "Echo",
            "timestamp_start": "2026-03-13T00:00:00+00:00",
            "timestamp_end": "2026-03-13T00:00:01+00:00",
        },
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in r and isinstance(r[k], dict):
            r[k].update(v)
        else:
            r[k] = v
    return r


def _make_naive_receipt():
    """Build a valid receipt for a fully naive dispatch."""
    return _make_valid_receipt(
        dispatch_summary={
            "total_helix_modules": 154,
            "fused_count": 0,
            "naive_count": 154,
            "unknown_count": 0,
            "all_fused": False,
            "runtime_path": RUNTIME_PATH_NAIVE,
        },
        kernel_metadata=None,
        fallback_reason="Triton fused kernel not available",
    )


# ---------------------------------------------------------------------------
# validate_receipt: structural tests (WO-BASIN-HARDENING-01)
# ---------------------------------------------------------------------------

class TestValidateReceiptStructural:

    def test_valid_receipt(self):
        """Fully valid receipt -> no issues."""
        issues = validate_receipt(_make_valid_receipt())
        assert issues == [], f"Unexpected issues: {issues}"

    def test_valid_naive_receipt(self):
        """Fully valid naive receipt -> no issues."""
        issues = validate_receipt(_make_naive_receipt())
        assert issues == [], f"Unexpected issues: {issues}"

    def test_missing_top_level_field(self):
        """Missing schema -> detected."""
        r = _make_valid_receipt()
        del r["schema"]
        issues = validate_receipt(r)
        assert any("schema" in i for i in issues)

    def test_missing_dispatch_field(self):
        """Missing runtime_path -> detected."""
        r = _make_valid_receipt()
        del r["dispatch_summary"]["runtime_path"]
        issues = validate_receipt(r)
        assert any("runtime_path" in i for i in issues)

    def test_invalid_runtime_path(self):
        """Invalid runtime_path value -> detected."""
        r = _make_valid_receipt()
        r["dispatch_summary"]["runtime_path"] = "magic"
        issues = validate_receipt(r)
        assert any("invalid runtime_path" in i for i in issues)

    def test_old_runtime_path_rejected(self):
        """Old-style 'fused' label is no longer valid."""
        r = _make_valid_receipt()
        r["dispatch_summary"]["runtime_path"] = "fused"
        issues = validate_receipt(r)
        assert any("invalid runtime_path" in i for i in issues)

    def test_missing_timing_field(self):
        """Missing backend in timing -> detected."""
        r = _make_valid_receipt()
        del r["timing"]["backend"]
        issues = validate_receipt(r)
        assert any("backend" in i for i in issues)

    def test_missing_cost_field(self):
        """Missing wall_time_s -> detected."""
        r = _make_valid_receipt()
        del r["cost"]["wall_time_s"]
        issues = validate_receipt(r)
        assert any("wall_time_s" in i for i in issues)

    def test_kernel_metadata_none_ok_for_naive(self):
        """kernel_metadata=None is valid on naive path."""
        issues = validate_receipt(_make_naive_receipt())
        assert issues == []

    def test_fallback_reason_none_ok_for_fused(self):
        """fallback_reason=None is valid when all fused."""
        r = _make_valid_receipt()
        r["fallback_reason"] = None
        issues = validate_receipt(r)
        assert issues == []

    def test_fallback_reason_string_ok_for_naive(self):
        """fallback_reason as string is valid on naive path."""
        issues = validate_receipt(_make_naive_receipt())
        assert issues == []

    def test_multiple_missing_fields(self):
        """Multiple missing fields all detected in one pass."""
        r = _make_valid_receipt()
        del r["schema"]
        del r["cost"]
        issues = validate_receipt(r)
        assert len(issues) >= 2
        assert any("schema" in i for i in issues)
        assert any("cost" in i for i in issues)

    def test_not_a_dict(self):
        """Non-dict input -> detected."""
        issues = validate_receipt("not a dict")
        assert issues == ["receipt is not a dict"]


# ---------------------------------------------------------------------------
# validate_receipt: semantic invariant tests (WO-BASIN-RECEIPT-SEMANTICS-01)
# ---------------------------------------------------------------------------

class TestValidateReceiptSemantic:

    def test_s1_fused_requires_kernel_metadata(self):
        """S1: fused_triton_v3 with kernel_metadata=None -> semantic error."""
        r = _make_valid_receipt(kernel_metadata=None)
        issues = validate_receipt(r)
        assert any("kernel_metadata is None" in i for i in issues), \
            f"Expected kernel_metadata issue, got: {issues}"

    def test_s2_all_fused_requires_zero_naive(self):
        """S2: all_fused=True but naive_count>0 -> semantic error."""
        r = _make_valid_receipt()
        r["dispatch_summary"]["naive_count"] = 3
        r["fallback_reason"] = "test forced"
        issues = validate_receipt(r)
        assert any("all_fused=True but naive_count=3" in i for i in issues), \
            f"Expected all_fused/naive_count issue, got: {issues}"

    def test_s3_null_fallback_with_naive_dispatch(self):
        """S3: fallback_reason=None but naive_count>0 -> semantic error (no silent pass-through)."""
        r = _make_valid_receipt()
        r["dispatch_summary"]["all_fused"] = False
        r["dispatch_summary"]["fused_count"] = 100
        r["dispatch_summary"]["naive_count"] = 54
        r["dispatch_summary"]["runtime_path"] = RUNTIME_PATH_MIXED
        r["fallback_reason"] = None  # Missing explanation!
        issues = validate_receipt(r)
        assert any("fallback_reason is None but naive_count=54" in i for i in issues), \
            f"Expected fallback explanation issue, got: {issues}"

    def test_s5_all_fused_true_but_fused_count_zero(self):
        """S5: all_fused=True but fused_count=0 -> semantic error."""
        r = _make_valid_receipt()
        r["dispatch_summary"]["all_fused"] = True
        r["dispatch_summary"]["fused_count"] = 0
        r["dispatch_summary"]["runtime_path"] = RUNTIME_PATH_UNSUPPORTED
        r["kernel_metadata"] = None
        issues = validate_receipt(r)
        assert any("all_fused=True but fused_count=0" in i for i in issues), \
            f"Expected fused_count issue, got: {issues}"

    def test_s5_all_fused_false_but_everything_fused(self):
        """S5: all_fused=False but all modules ran fused -> semantic error."""
        r = _make_valid_receipt()
        r["dispatch_summary"]["all_fused"] = False  # wrong!
        # fused_count=154, naive_count=0, total=154
        issues = validate_receipt(r)
        assert any("all_fused=False" in i for i in issues), \
            f"Expected all_fused inconsistency, got: {issues}"

    def test_schema_version_wrong(self):
        """Wrong schema version -> detected."""
        r = _make_valid_receipt(schema="basin_helix_fused:v0")
        issues = validate_receipt(r)
        assert any("schema mismatch" in i for i in issues), \
            f"Expected schema issue, got: {issues}"

    def test_valid_fused_passes_all_semantic(self):
        """A correct fused receipt passes ALL semantic checks."""
        issues = validate_receipt(_make_valid_receipt())
        assert issues == []

    def test_valid_naive_passes_all_semantic(self):
        """A correct naive receipt passes ALL semantic checks."""
        issues = validate_receipt(_make_naive_receipt())
        assert issues == []

    def test_all_canonical_paths_accepted(self):
        """Every canonical runtime_path is accepted by the validator."""
        for rp in (RUNTIME_PATH_FUSED, RUNTIME_PATH_NAIVE, RUNTIME_PATH_MIXED,
                   RUNTIME_PATH_UNSUPPORTED, RUNTIME_PATH_STARTUP_FAIL,
                   RUNTIME_PATH_RECEIPT_FAIL):
            r = _make_valid_receipt()
            r["dispatch_summary"]["runtime_path"] = rp
            # Suppress semantic cross-check failures for non-fused paths
            if rp != RUNTIME_PATH_FUSED:
                r["kernel_metadata"] = None
                r["dispatch_summary"]["all_fused"] = False
                r["dispatch_summary"]["fused_count"] = 0
                r["dispatch_summary"]["naive_count"] = 0
                r["dispatch_summary"]["total_helix_modules"] = 0
            issues = validate_receipt(r)
            path_issues = [i for i in issues if "invalid runtime_path" in i]
            assert path_issues == [], f"runtime_path {rp!r} rejected: {path_issues}"


# ---------------------------------------------------------------------------
# validate_startup_receipt tests (WO-BASIN-RECEIPT-SEMANTICS-01)
# ---------------------------------------------------------------------------

class TestValidateStartupReceipt:

    def _make_valid_startup(self):
        cap = {
            "valid": True, "issues": [],
            "model_name": "TinyLlama-1.1B", "n_tensors": 154, "n_blocks": 22,
            "compression_ratio": 3.99, "manifest_sha256": "abc",
            "triton_available": True, "cuda_available": True,
        }
        return build_startup_receipt(cap, "tinyllama", "/m/tinyllama", False)

    def test_valid_startup(self):
        """Valid startup receipt -> no issues."""
        issues = validate_startup_receipt(self._make_valid_startup())
        assert issues == [], f"Unexpected: {issues}"

    def test_wrong_schema(self):
        """Wrong schema -> detected."""
        r = self._make_valid_startup()
        r["schema"] = "wrong:v2"
        issues = validate_startup_receipt(r)
        assert any("schema mismatch" in i for i in issues)

    def test_missing_field(self):
        """Missing model_id -> detected."""
        r = self._make_valid_startup()
        del r["model_id"]
        issues = validate_startup_receipt(r)
        assert any("model_id" in i for i in issues)

    def test_capability_invalid_but_no_issues(self):
        """capability_valid=False with empty issues -> semantic error."""
        r = self._make_valid_startup()
        r["capability_valid"] = False
        r["capability_issues"] = []
        issues = validate_startup_receipt(r)
        assert any("capability_valid=False but capability_issues is empty" in i
                    for i in issues)


# ---------------------------------------------------------------------------
# validate_hardening_summary tests (WO-BASIN-RECEIPT-SEMANTICS-01)
# ---------------------------------------------------------------------------

class TestValidateHardeningSummary:

    def _make_valid_summary(self):
        startup = {"backend": "helix_fused", "strict_mode": False}
        receipts = [
            {"dispatch_summary": {"runtime_path": RUNTIME_PATH_FUSED},
             "timing": {"tok_s": 15.0, "prefill_tok_s": 10.0}},
            {"dispatch_summary": {"runtime_path": RUNTIME_PATH_FUSED},
             "timing": {"tok_s": 16.0, "prefill_tok_s": 11.0}},
        ]
        return build_hardening_summary(startup, receipts, [], ["tinyllama"], 5.0)

    def test_valid_summary(self):
        """Valid summary -> no issues."""
        issues = validate_hardening_summary(self._make_valid_summary())
        assert issues == [], f"Unexpected: {issues}"

    def test_wrong_schema(self):
        """Wrong schema -> detected."""
        s = self._make_valid_summary()
        s["schema"] = "wrong:v99"
        issues = validate_hardening_summary(s)
        assert any("schema mismatch" in i for i in issues)

    def test_count_mismatch(self):
        """request_count != success_count + len(failure_reasons) -> semantic error."""
        s = self._make_valid_summary()
        s["request_count"] = 10  # but success=2, failures=0
        issues = validate_hardening_summary(s)
        assert any("request_count(10)" in i for i in issues)

    def test_strict_with_naive(self):
        """strict_mode=True with naive_count>0 -> semantic error."""
        s = self._make_valid_summary()
        s["strict_mode"] = True
        s["naive_count"] = 1
        issues = validate_hardening_summary(s)
        assert any("strict_mode=True but naive_count=1" in i for i in issues)

    def test_fused_plus_naive_exceeds_success(self):
        """fused+naive > success_count -> semantic error."""
        s = self._make_valid_summary()
        s["fused_count"] = 5
        s["naive_count"] = 3
        # success_count=2 but fused+naive=8
        issues = validate_hardening_summary(s)
        assert any("fused_count(5) + naive_count(3) > success_count(2)" in i
                    for i in issues)

    def test_missing_field(self):
        """Missing request_count -> detected."""
        s = self._make_valid_summary()
        del s["request_count"]
        issues = validate_hardening_summary(s)
        assert any("request_count" in i for i in issues)


# ---------------------------------------------------------------------------
# build_startup_receipt tests (WO-BASIN-HARDENING-01)
# ---------------------------------------------------------------------------

class TestStartupReceipt:

    def test_structure(self):
        """Startup receipt has all required fields."""
        cap = {
            "valid": True,
            "issues": [],
            "model_name": "TinyLlama-1.1B",
            "n_tensors": 154,
            "n_blocks": 22,
            "compression_ratio": 3.99,
            "manifest_sha256": "abc123",
            "triton_available": True,
            "cuda_available": True,
        }
        r = build_startup_receipt(cap, "tinyllama", "/models/tinyllama", False)

        assert r["schema"] == SCHEMA_STARTUP_RECEIPT
        assert r["backend"] == "helix_fused"
        assert r["model_id"] == "tinyllama"
        assert r["strict_mode"] is False
        assert r["capability_valid"] is True
        assert r["capability_issues"] == []
        assert r["triton_available"] is True
        assert r["cuda_available"] is True
        assert r["manifest_sha256"] == "abc123"
        assert r["model_name"] == "TinyLlama-1.1B"
        assert r["n_tensors"] == 154
        assert r["timestamp"] is not None
        assert r["hostname"] is not None

    def test_strict_mode_captured(self):
        """Strict mode flag is captured."""
        cap = {"valid": True, "issues": []}
        r = build_startup_receipt(cap, "tinyllama", "/x", True)
        assert r["strict_mode"] is True

    def test_issues_captured(self):
        """Capability issues are propagated."""
        cap = {"valid": False, "issues": ["Triton unavailable", "CUDA not found"]}
        r = build_startup_receipt(cap, "tinyllama", "/x", False)
        assert r["capability_valid"] is False
        assert len(r["capability_issues"]) == 2


# ---------------------------------------------------------------------------
# build_hardening_summary tests (WO-BASIN-HARDENING-01)
# ---------------------------------------------------------------------------

class TestHardeningSummary:

    def test_basic_summary(self):
        """Summary aggregates correctly."""
        startup = {"backend": "helix_fused", "strict_mode": False}
        receipts = [
            {
                "dispatch_summary": {"runtime_path": RUNTIME_PATH_FUSED},
                "timing": {"tok_s": 15.0, "prefill_tok_s": 10.0},
            },
            {
                "dispatch_summary": {"runtime_path": RUNTIME_PATH_FUSED},
                "timing": {"tok_s": 16.0, "prefill_tok_s": 11.0},
            },
        ]
        s = build_hardening_summary(startup, receipts, [], ["tinyllama"], 5.0)

        assert s["schema"] == SCHEMA_HARDENING_SUMMARY
        assert s["request_count"] == 2
        assert s["success_count"] == 2
        assert s["fused_count"] == 2
        assert s["naive_count"] == 0
        assert s["mean_decode_tok_s"] == 15.5
        assert s["mean_prefill_tok_s"] == 10.5
        assert s["failure_reasons"] == []
        assert "tinyllama" in s["models_tested"]

    def test_with_failures(self):
        """Summary includes failure reasons."""
        startup = {"backend": "helix_fused", "strict_mode": True}
        failures = [{"request_index": 2, "reason": "strict mode violation"}]
        s = build_hardening_summary(startup, [], failures, ["tinyllama"], 1.0)

        assert s["request_count"] == 1
        assert s["success_count"] == 0
        assert s["failure_reasons"] == ["strict mode violation"]


# ---------------------------------------------------------------------------
# Multi-model capability tests (WO-BASIN-HARDENING-01)
# ---------------------------------------------------------------------------

class TestMultiModelCapability:

    def test_tinyllama_capability(self):
        """TinyLlama CDNA v3 passes capability check."""
        if not TINYLLAMA_CDNA.exists():
            pytest.skip("TinyLlama CDNA v3 not present")

        cap = check_cdnav3_capability(TINYLLAMA_CDNA)
        data_issues = [i for i in cap["issues"]
                       if "CUDA" not in i and "Triton" not in i]
        assert len(data_issues) == 0, f"Issues: {data_issues}"
        assert cap["model_name"] == "TinyLlama-1.1B"
        assert cap["n_tensors"] == 154
        assert cap["n_blocks"] == 22

    def test_qwen_capability(self):
        """Qwen CDNA v3 passes capability check."""
        if not QWEN_CDNA.exists():
            pytest.skip("Qwen CDNA v3 not present")

        cap = check_cdnav3_capability(QWEN_CDNA)
        data_issues = [i for i in cap["issues"]
                       if "CUDA" not in i and "Triton" not in i]
        assert len(data_issues) == 0, f"Issues: {data_issues}"
        assert cap["model_name"] == "Qwen2.5-Coder-1.5B"
        assert cap["n_tensors"] == 196
        assert cap["n_blocks"] == 28

    def test_both_manifests_have_sha256(self):
        """Both models produce distinct manifest SHA256s."""
        if not TINYLLAMA_CDNA.exists() or not QWEN_CDNA.exists():
            pytest.skip("Both models required")

        cap_tl = check_cdnav3_capability(TINYLLAMA_CDNA)
        cap_qw = check_cdnav3_capability(QWEN_CDNA)
        assert cap_tl["manifest_sha256"] is not None
        assert cap_qw["manifest_sha256"] is not None
        assert cap_tl["manifest_sha256"] != cap_qw["manifest_sha256"]


# ---------------------------------------------------------------------------
# Negative-path tests (WO-BASIN-HARDENING-01)
# ---------------------------------------------------------------------------

class TestNegativePaths:

    def test_corrupt_manifest_json(self):
        """Corrupt JSON -> valid=False with parse error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "manifest.json").write_text("{broken json")
            cap = check_cdnav3_capability(tmpdir)
            assert cap["valid"] is False
            assert any("Failed to read" in i for i in cap["issues"])

    def test_empty_manifest(self):
        """Empty JSON object -> missing all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "manifest.json").write_text("{}")
            cap = check_cdnav3_capability(tmpdir)
            assert cap["valid"] is False
            for field in ("model", "n_blocks", "n_tensors", "compression_ratio"):
                assert any(field in i for i in cap["issues"]), \
                    f"Expected issue for {field}, got: {cap['issues']}"

    def test_wrong_format_version(self):
        """Tensor with wrong format_version -> detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cdna_dir = Path(tmpdir)
            manifest = {
                "model": "TestModel",
                "n_blocks": 1,
                "n_tensors": 1,
                "compression_ratio": 2.0,
            }
            (cdna_dir / "manifest.json").write_text(json.dumps(manifest))

            # Create one tensor dir with wrong format
            tensor_dir = cdna_dir / "test_tensor.cdnav3"
            tensor_dir.mkdir()
            (tensor_dir / "meta.json").write_text(
                json.dumps({"format_version": "cdna_v2"})
            )

            cap = check_cdnav3_capability(cdna_dir)
            assert cap["valid"] is False
            assert any("cdna_v3" in i for i in cap["issues"])

    def test_file_not_directory(self):
        """Path is a file, not a directory -> detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "fake_cdna"
            file_path.write_text("not a dir")
            cap = check_cdnav3_capability(file_path)
            assert cap["valid"] is False
            assert any("not a directory" in i for i in cap["issues"])

    def test_tensor_dir_missing_meta(self):
        """Tensor dir exists but meta.json missing -> detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cdna_dir = Path(tmpdir)
            manifest = {
                "model": "TestModel",
                "n_blocks": 1,
                "n_tensors": 1,
                "compression_ratio": 2.0,
            }
            (cdna_dir / "manifest.json").write_text(json.dumps(manifest))
            (cdna_dir / "test_tensor.cdnav3").mkdir()

            cap = check_cdnav3_capability(cdna_dir)
            assert cap["valid"] is False
            assert any("meta.json missing" in i for i in cap["issues"])
