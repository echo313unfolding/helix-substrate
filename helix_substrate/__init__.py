"""
helix-substrate: Nature-inspired model compression and streaming decode.

Core components:
    - CDNA format: k-means quantized model weight compression
    - Streaming decode: block-by-block Y = X @ W without loading full weights
    - Se routing: structural entropy for tensor-level compute routing
      (legacy H×U×D formula; revised four-component model in se_final_form.md)
    - Receipts: tamper-evident verification of all operations
"""

__version__ = "0.2.6"

from helix_substrate.cdna_encoder import encode_tensor_to_cdna, decode_cdna_to_tensor
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
from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.cdnav3_reader import CDNAv3Reader

# Step 5: Auto-register with HuggingFace Transformers (if available).
# Importing hf_integration triggers @register_quantization_config("helix")
# and @register_quantizer("helix"), so from_pretrained() works on Helix checkpoints.
try:
    from helix_substrate.hf_integration import HelixQuantizationConfig, HelixHfQuantizer
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
]
