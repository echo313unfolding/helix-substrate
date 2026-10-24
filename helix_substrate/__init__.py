"""
helix-substrate: Calibration-free weight compression (HXQ-VQ).

Core components:
    - HXQ-VQ codec: scalar k-means VQ (k=256) with sparse outlier sidecar
    - CDNAv3Writer/Reader: per-tensor directory format
    - HelixLinear: drop-in nn.Linear replacement (codebook[indices] matmul)
    - Receipts: tamper-evident verification of all operations
"""

__version__ = "0.4.0"

from helix_substrate.hxq_encoder import encode_tensor_to_cdna, decode_cdna_to_tensor
from helix_substrate.se import compute_tensor_se, compute_routing_decision
from helix_substrate.convert import convert_huggingface_model
from helix_substrate.convert_gguf import convert_gguf_model
from helix_substrate.receipt import (
    OperationResult,
    ExecutionReceipt,
    validate_execution_receipt,
    save_execution_receipt,
    load_execution_receipt,
)
from helix_substrate.tensor_policy import classify_tensor, TensorPolicy, TensorClass, MORPHO_FFT_POLICY
from helix_substrate.hxq_writer_v3 import CDNAv3Writer
from helix_substrate.hxq_reader_v3 import CDNAv3Reader

# Auto-register CDNA v3 / Helix quantizer with HuggingFace Transformers.
# hf_quantizer is the canonical implementation — handles HelixLinear, shared modules,
# nn.Embedding, and leverages HF's native loading infrastructure.
try:
    from helix_substrate.hf_quantizer import CDNAv3Config, HelixHfQuantizer
except ImportError:
    pass

# Layer-chain receipt (load-time SHAKE256 integrity gate). Kept separate from
# helix_substrate.receipt (per-operation execution receipts) for semantic clarity.
try:
    from helix_substrate.layer_chain_receipt import (
        LayerReceipt,
        ReceiptManifest,
        VerificationResult,
        generate_receipt_manifest,
        verify_receipt_manifest,
        save_manifest,
        load_manifest,
        assert_receipt_valid,
    )
except ImportError:
    pass

__all__ = [
    "encode_tensor_to_cdna",
    "decode_cdna_to_tensor",
    "compute_tensor_se",
    "compute_routing_decision",
    "convert_huggingface_model",
    "convert_gguf_model",
    "OperationResult",
    "ExecutionReceipt",
    "validate_execution_receipt",
    "save_execution_receipt",
    "load_execution_receipt",
    "classify_tensor",
    "TensorPolicy",
    "TensorClass",
    "CDNAv3Writer",
    "CDNAv3Reader",
    "MORPHO_FFT_POLICY",
    # Layer-chain receipt (load-time integrity)
    "LayerReceipt",
    "ReceiptManifest",
    "VerificationResult",
    "generate_receipt_manifest",
    "verify_receipt_manifest",
    "save_manifest",
    "load_manifest",
    "assert_receipt_valid",
]
