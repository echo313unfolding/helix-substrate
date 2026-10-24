"""Backwards-compatibility shim: cdna_encoder → hxq_encoder (renamed in 0.4.0)."""
from helix_substrate.hxq_encoder import *  # noqa: F401,F403
from helix_substrate.hxq_encoder import encode_tensor_to_cdna, decode_cdna_to_tensor
