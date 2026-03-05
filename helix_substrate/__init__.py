"""
helix-substrate: Nature-inspired model compression and streaming decode.

Core components:
    - CDNA format: k-means quantized model weight compression
    - Streaming decode: block-by-block Y = X @ W without loading full weights
    - Se routing: structural entropy for tensor-level compute routing
    - Receipts: tamper-evident verification of all operations
"""

__version__ = "0.1.0"

from helix_substrate.cdna_encoder import encode_tensor_to_cdna, decode_cdna_to_tensor
from helix_substrate.se import compute_tensor_se, compute_routing_decision
from helix_substrate.receipt import (
    OperationResult,
    ExecutionReceipt,
    validate_execution_receipt,
    save_execution_receipt,
    load_execution_receipt,
)

__all__ = [
    "encode_tensor_to_cdna",
    "decode_cdna_to_tensor",
    "compute_tensor_se",
    "compute_routing_decision",
    "OperationResult",
    "ExecutionReceipt",
    "validate_execution_receipt",
    "save_execution_receipt",
    "load_execution_receipt",
]
