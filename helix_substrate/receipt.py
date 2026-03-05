"""
helix_cdc/substrate/execution_receipt.py
========================================

ExecutionReceipt schema for substrate operations.

This module defines the tamper-evident receipt format that proves
compute-while-compressed execution actually happened.

Architecture:
    SubstratePlan → ExecutionOrchestrator → ExecutionReceipt
                           ↓
                    [cdna_decode, sidecar_apply, ...]
                           ↓
                    BehavioralGate validation
                           ↓
                    Tamper-evident proof bundle

Schema Version: execution_receipt:v1
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OperationResult:
    """
    Result of a single substrate operation.

    This provides receipts for individual ops like cdna_decode, sidecar_apply,
    stream_xw_matmul, stream_attention_forward, etc.
    """
    op_name: str              # "cdna_decode", "sidecar_apply", "stream_xw_matmul", etc.
    timing_ms: float          # Execution time in milliseconds
    memory_bytes: int         # Peak memory used for this operation

    # Determinism proof (compute proof)
    input_sha256: str         # SHA256 of input (shard/sidecar file OR activation tensor)
    output_sha256: str        # SHA256 of output tensor bytes

    # Fidelity metrics (optional, operation-dependent)
    mse_vs_baseline: Optional[float] = None      # MSE if baseline available
    max_error: Optional[float] = None            # Max absolute error
    cosine_similarity: Optional[float] = None    # Cosine similarity to baseline

    # Operation-specific stats
    elements_processed: int = 0
    outliers_applied: int = 0  # For sidecar_apply
    tensors_decoded: int = 0   # For cdna_decode

    # Status
    status: str = "OK"        # "OK" | "WARN" | "FAIL"
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    # Claim hygiene fields (for stream_xw_matmul, stream_attention_forward)
    # These fields document which codec/streaming mode was used
    tensor_name: Optional[str] = None
    codec_version: Optional[str] = None       # "cdna_v1" | "cdna_v2"
    streaming_mode: Optional[str] = None      # "true_block_streaming" | "full_load_fallback"

    # For stream_attention_forward
    block_index: Optional[int] = None
    gqa_handling: Optional[str] = None        # "test_shim_slice_pad" | "no_adjustment_needed"
    codec_versions_used: List[str] = field(default_factory=list)
    streaming_modes_used: List[str] = field(default_factory=list)
    all_true_block_streaming: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = {
            "op_name": self.op_name,
            "timing_ms": round(self.timing_ms, 3),
            "memory_bytes": self.memory_bytes,
            # Compute proof: input → output hash chain
            "compute_proof": {
                "input_sha256": self.input_sha256,
                "output_sha256": self.output_sha256,
            },
            "mse_vs_baseline": self.mse_vs_baseline,
            "max_error": self.max_error,
            "cosine_similarity": self.cosine_similarity,
            "elements_processed": self.elements_processed,
            "outliers_applied": self.outliers_applied,
            "tensors_decoded": self.tensors_decoded,
            "status": self.status,
            "warnings": self.warnings,
            "error_message": self.error_message,
        }

        # Add claim hygiene fields if present (for compute ops)
        if self.tensor_name is not None:
            d["tensor_name"] = self.tensor_name
        if self.codec_version is not None:
            d["claim_hygiene"] = d.get("claim_hygiene", {})
            d["claim_hygiene"]["codec_version"] = self.codec_version
        if self.streaming_mode is not None:
            d["claim_hygiene"] = d.get("claim_hygiene", {})
            d["claim_hygiene"]["streaming_mode"] = self.streaming_mode

        # For stream_attention_forward
        if self.block_index is not None:
            d["block_index"] = self.block_index
        if self.gqa_handling is not None:
            d["claim_hygiene"] = d.get("claim_hygiene", {})
            d["claim_hygiene"]["gqa_handling"] = self.gqa_handling
        if self.codec_versions_used:
            d["claim_hygiene"] = d.get("claim_hygiene", {})
            d["claim_hygiene"]["codec_versions_used"] = self.codec_versions_used
        if self.streaming_modes_used:
            d["claim_hygiene"] = d.get("claim_hygiene", {})
            d["claim_hygiene"]["streaming_modes_used"] = self.streaming_modes_used
        if self.all_true_block_streaming is not None:
            d["claim_hygiene"] = d.get("claim_hygiene", {})
            d["claim_hygiene"]["all_true_block_streaming"] = self.all_true_block_streaming

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OperationResult":
        """Create from dict."""
        # Handle both old format (input_sha256 at top level) and new format (compute_proof)
        compute_proof = d.get("compute_proof", {})
        input_sha256 = compute_proof.get("input_sha256", d.get("input_sha256", ""))
        output_sha256 = compute_proof.get("output_sha256", d.get("output_sha256", ""))

        # Extract claim hygiene fields
        claim_hygiene = d.get("claim_hygiene", {})

        return cls(
            op_name=d["op_name"],
            timing_ms=d["timing_ms"],
            memory_bytes=d["memory_bytes"],
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            mse_vs_baseline=d.get("mse_vs_baseline"),
            max_error=d.get("max_error"),
            cosine_similarity=d.get("cosine_similarity"),
            elements_processed=d.get("elements_processed", 0),
            outliers_applied=d.get("outliers_applied", 0),
            tensors_decoded=d.get("tensors_decoded", 0),
            status=d.get("status", "OK"),
            warnings=d.get("warnings", []),
            error_message=d.get("error_message"),
            # Claim hygiene fields
            tensor_name=d.get("tensor_name"),
            codec_version=claim_hygiene.get("codec_version"),
            streaming_mode=claim_hygiene.get("streaming_mode"),
            block_index=d.get("block_index"),
            gqa_handling=claim_hygiene.get("gqa_handling"),
            codec_versions_used=claim_hygiene.get("codec_versions_used", []),
            streaming_modes_used=claim_hygiene.get("streaming_modes_used", []),
            all_true_block_streaming=claim_hygiene.get("all_true_block_streaming"),
        )


@dataclass
class ExecutionReceipt:
    """
    Complete execution receipt for a SubstratePlan.

    This is the tamper-evident proof that compute-while-compressed happened.
    Links to plan, operations, and behavioral gate validation.

    Schema: execution_receipt:v1
    """
    schema: str = "execution_receipt:v1"

    # Link to plan
    plan_sha256: str = ""
    brainstem_band: str = "alpha"

    # Operations executed
    operations: List[OperationResult] = field(default_factory=list)
    total_timing_ms: float = 0.0
    total_memory_bytes: int = 0

    # Validation
    behavioral_gate_verdict: str = "NOT_CHECKED"  # "PASS" | "ACCEPTABLE_WITH_TAIL_RISK" | "FAIL" | "NOT_CHECKED"
    gates_checked: Dict[str, bool] = field(default_factory=dict)
    gate_receipt_path: Optional[str] = None

    # Aggregate fidelity metrics
    avg_cosine_similarity: Optional[float] = None
    max_mse: Optional[float] = None
    tensors_total: int = 0
    tensors_ok: int = 0
    tensors_warn: int = 0
    tensors_fail: int = 0

    # Provenance
    timestamp_utc: str = ""
    git_commit: str = "unknown"
    manifest_path: Optional[str] = None
    manifest_sha256: Optional[str] = None
    output_gguf_path: Optional[str] = None
    output_gguf_sha256: Optional[str] = None

    # Tamper-evident hash
    receipt_sha256: str = ""

    # Kernel-mode syscall receipts (NEW for kernel ABI integration)
    syscall_receipts: List[Dict] = field(default_factory=list)
    total_it_tokens_spent: float = 0.0

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def compute_hash(self) -> str:
        """Compute deterministic hash of the receipt (excluding receipt_sha256)."""
        canonical = self.to_dict()
        canonical.pop("receipt_sha256", None)
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def finalize(self) -> "ExecutionReceipt":
        """Finalize the receipt by computing the tamper-evident hash."""
        self.receipt_sha256 = self.compute_hash()
        return self

    def aggregate_stats(self) -> None:
        """Aggregate statistics from operations."""
        if not self.operations:
            return

        self.total_timing_ms = sum(op.timing_ms for op in self.operations)
        self.total_memory_bytes = max(op.memory_bytes for op in self.operations) if self.operations else 0

        # Count status
        self.tensors_ok = sum(1 for op in self.operations if op.status == "OK")
        self.tensors_warn = sum(1 for op in self.operations if op.status == "WARN")
        self.tensors_fail = sum(1 for op in self.operations if op.status == "FAIL")
        self.tensors_total = len(self.operations)

        # Aggregate fidelity
        cosines = [op.cosine_similarity for op in self.operations if op.cosine_similarity is not None]
        if cosines:
            self.avg_cosine_similarity = sum(cosines) / len(cosines)

        mses = [op.mse_vs_baseline for op in self.operations if op.mse_vs_baseline is not None]
        if mses:
            self.max_mse = max(mses)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "schema": self.schema,
            "plan_sha256": self.plan_sha256,
            "brainstem_band": self.brainstem_band,
            "operations": [op.to_dict() for op in self.operations],
            "total_timing_ms": round(self.total_timing_ms, 3),
            "total_memory_bytes": self.total_memory_bytes,
            "behavioral_gate_verdict": self.behavioral_gate_verdict,
            "gates_checked": self.gates_checked,
            "gate_receipt_path": self.gate_receipt_path,
            "avg_cosine_similarity": self.avg_cosine_similarity,
            "max_mse": self.max_mse,
            "tensors_total": self.tensors_total,
            "tensors_ok": self.tensors_ok,
            "tensors_warn": self.tensors_warn,
            "tensors_fail": self.tensors_fail,
            "timestamp_utc": self.timestamp_utc,
            "git_commit": self.git_commit,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "output_gguf_path": self.output_gguf_path,
            "output_gguf_sha256": self.output_gguf_sha256,
            "receipt_sha256": self.receipt_sha256,
            # Kernel-mode syscall receipts
            "syscall_receipts": self.syscall_receipts,
            "total_it_tokens_spent": round(self.total_it_tokens_spent, 6),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionReceipt":
        """Create from dict."""
        return cls(
            schema=d.get("schema", "execution_receipt:v1"),
            plan_sha256=d.get("plan_sha256", ""),
            brainstem_band=d.get("brainstem_band", "alpha"),
            operations=[OperationResult.from_dict(op) for op in d.get("operations", [])],
            total_timing_ms=d.get("total_timing_ms", 0.0),
            total_memory_bytes=d.get("total_memory_bytes", 0),
            behavioral_gate_verdict=d.get("behavioral_gate_verdict", "NOT_CHECKED"),
            gates_checked=d.get("gates_checked", {}),
            gate_receipt_path=d.get("gate_receipt_path"),
            avg_cosine_similarity=d.get("avg_cosine_similarity"),
            max_mse=d.get("max_mse"),
            tensors_total=d.get("tensors_total", 0),
            tensors_ok=d.get("tensors_ok", 0),
            tensors_warn=d.get("tensors_warn", 0),
            tensors_fail=d.get("tensors_fail", 0),
            timestamp_utc=d.get("timestamp_utc", ""),
            git_commit=d.get("git_commit", "unknown"),
            manifest_path=d.get("manifest_path"),
            manifest_sha256=d.get("manifest_sha256"),
            output_gguf_path=d.get("output_gguf_path"),
            output_gguf_sha256=d.get("output_gguf_sha256"),
            receipt_sha256=d.get("receipt_sha256", ""),
            # Kernel-mode syscall receipts
            syscall_receipts=d.get("syscall_receipts", []),
            total_it_tokens_spent=d.get("total_it_tokens_spent", 0.0),
        )


def validate_execution_receipt(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an execution receipt.

    Args:
        receipt: Receipt dict to validate

    Returns:
        Validation report: {"valid": bool, "errors": [...], "sha256_match": bool}
    """
    errors = []

    # Check required fields
    required = [
        "schema", "plan_sha256", "operations",
        "behavioral_gate_verdict", "timestamp_utc"
    ]
    for field in required:
        if field not in receipt:
            errors.append(f"Missing required field: {field}")

    # Check schema
    if not receipt.get("schema", "").startswith("execution_receipt:"):
        errors.append(f"Invalid schema: {receipt.get('schema')}")

    # Check operations structure
    operations = receipt.get("operations", [])
    if not isinstance(operations, list):
        errors.append("operations must be a list")
    else:
        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                errors.append(f"operations[{i}] must be a dict")
            elif "op_name" not in op:
                errors.append(f"operations[{i}] missing op_name")

    # Verify receipt_sha256
    sha256_match = None
    if "receipt_sha256" in receipt:
        receipt_clean = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        canonical = json.dumps(receipt_clean, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        actual = receipt["receipt_sha256"]
        sha256_match = (expected == actual)
        if not sha256_match:
            errors.append(f"receipt_sha256 mismatch: expected {expected[:16]}..., got {actual[:16]}...")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "sha256_match": sha256_match,
    }


def save_execution_receipt(
    receipt: ExecutionReceipt,
    path: Path,
    finalize: bool = True,
) -> Path:
    """
    Save execution receipt to file.

    Args:
        receipt: ExecutionReceipt to save
        path: Output path
        finalize: If True, finalize receipt before saving

    Returns:
        Path to saved receipt
    """
    if finalize:
        receipt.finalize()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt.to_dict(), indent=2))

    return path


def load_execution_receipt(path: Path) -> ExecutionReceipt:
    """Load execution receipt from file."""
    return ExecutionReceipt.from_dict(json.loads(Path(path).read_text()))
