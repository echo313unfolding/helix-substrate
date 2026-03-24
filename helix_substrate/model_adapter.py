"""
Model Adapter Manifest — WO-AI-OS-CONTROLLER-01 Task 4.

Standardized contract for plugging models into the governed AI OS.
Every model that participates in the runtime must provide an adapter
manifest declaring its capabilities, resource requirements, and
preferred routing role.

A model without a manifest cannot be loaded by the governor.

Usage:
    manifest = load_model_manifest("~/models/tinyllama_fp32")
    if manifest.fits_vram(available_vram_mb=3899):
        ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


SCHEMA = "model_adapter_manifest:v1"


@dataclass
class CompressionInfo:
    """Compression details for models using CDNA v3 or similar codecs."""
    codec: str                  # "cdna_v3", "none" (dense)
    ratio: float                # e.g., 4.26
    n_helix_linear: int         # number of HelixLinear layers
    n_dense_linear: int         # remaining nn.Linear / _BF16Linear layers


@dataclass
class ResourceRequirements:
    """Hardware resources this model needs."""
    weight_vram_mb: float       # VRAM for weights only
    kv_per_token_mb: float      # KV cache cost per token
    max_safe_context_solo: int  # max tokens in solo mode
    max_safe_context_coresident: int  # max tokens when sharing GPU
    min_compute_capability: float     # minimum CC (e.g., 7.5)
    load_time_s: float          # typical load time from compressed


@dataclass
class ModelAdapterManifest:
    """The contract a model must fulfill to plug into the AI OS."""
    schema: str = SCHEMA
    model_id: str = ""                # unique ID: "tinyllama-1.1b-helix"
    model_name: str = ""              # display name: "TinyLlama 1.1B"
    model_family: str = ""            # "llama", "qwen", etc.
    model_dir: str = ""               # path to model directory
    capabilities: List[str] = field(default_factory=list)  # ["chat", "planning", "reasoning"]
    preferred_role: str = ""          # "controller", "coder", "responder"
    fallback_policy: str = "deny"     # "deny", "downgrade", "queue"
    compression: Optional[CompressionInfo] = None
    resources: Optional[ResourceRequirements] = None
    receipt_schema: str = ""          # receipt format this model's outputs conform to
    tokenizer_id: str = ""            # HF tokenizer ID or path
    prompt_format: str = ""           # "chatml", "llama2", "plain"
    max_output_tokens: int = 256      # default generation cap

    def fits_vram(self, available_vram_mb: float) -> bool:
        """Check if this model fits in available VRAM."""
        if self.resources is None:
            return False
        return self.resources.weight_vram_mb < available_vram_mb

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        d = asdict(self)
        return d

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ModelAdapterManifest":
        """Load manifest from JSON file."""
        data = json.loads(path.read_text())
        compression = None
        if data.get("compression"):
            compression = CompressionInfo(**data["compression"])
        resources = None
        if data.get("resources"):
            resources = ResourceRequirements(**data["resources"])
        return cls(
            schema=data.get("schema", SCHEMA),
            model_id=data.get("model_id", ""),
            model_name=data.get("model_name", ""),
            model_family=data.get("model_family", ""),
            model_dir=data.get("model_dir", ""),
            capabilities=data.get("capabilities", []),
            preferred_role=data.get("preferred_role", ""),
            fallback_policy=data.get("fallback_policy", "deny"),
            compression=compression,
            resources=resources,
            receipt_schema=data.get("receipt_schema", ""),
            tokenizer_id=data.get("tokenizer_id", ""),
            prompt_format=data.get("prompt_format", ""),
            max_output_tokens=data.get("max_output_tokens", 256),
        )


# ─── Built-in manifests for proven models ─────────────────────────────

TINYLLAMA_MANIFEST = ModelAdapterManifest(
    model_id="tinyllama-1.1b-helix",
    model_name="TinyLlama 1.1B Chat v1.0",
    model_family="llama",
    model_dir=str(Path.home() / "models" / "tinyllama_fp32"),
    capabilities=["chat", "planning", "reasoning", "verification", "summarization"],
    preferred_role="controller",
    fallback_policy="deny",
    compression=CompressionInfo(
        codec="cdna_v3",
        ratio=4.26,
        n_helix_linear=154,
        n_dense_linear=1,  # lm_head (BF16)
    ),
    resources=ResourceRequirements(
        weight_vram_mb=1433,
        kv_per_token_mb=0.20,
        max_safe_context_solo=2048,
        max_safe_context_coresident=128,
        min_compute_capability=7.5,
        load_time_s=0.9,
    ),
    receipt_schema="stabilization_receipt_v1",
    tokenizer_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    prompt_format="llama2",
    max_output_tokens=256,
)

QWEN_CODER_MANIFEST = ModelAdapterManifest(
    model_id="qwen2.5-coder-1.5b-helix",
    model_name="Qwen2.5 Coder 1.5B Instruct",
    model_family="qwen",
    model_dir=str(Path.home() / "models" / "qwen2.5-coder-1.5b-instruct"),
    capabilities=["code_generation", "code_review", "structured_output", "debugging"],
    preferred_role="coder",
    fallback_policy="deny",
    compression=CompressionInfo(
        codec="cdna_v3",
        ratio=4.7,
        n_helix_linear=196,
        n_dense_linear=0,  # tied embeddings, no separate lm_head
    ),
    resources=ResourceRequirements(
        weight_vram_mb=2247,
        kv_per_token_mb=0.19,
        max_safe_context_solo=2048,
        max_safe_context_coresident=512,
        min_compute_capability=7.5,
        load_time_s=4.2,
    ),
    receipt_schema="stabilization_receipt_v1",
    tokenizer_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    prompt_format="chatml",
    max_output_tokens=256,
)

# Registry of all known model adapters
MODEL_REGISTRY = {
    "tinyllama-1.1b-helix": TINYLLAMA_MANIFEST,
    "qwen2.5-coder-1.5b-helix": QWEN_CODER_MANIFEST,
}


def list_models() -> list:
    """List all registered model adapters."""
    return [
        {"model_id": m.model_id, "role": m.preferred_role,
         "vram_mb": m.resources.weight_vram_mb if m.resources else 0,
         "capabilities": m.capabilities}
        for m in MODEL_REGISTRY.values()
    ]


def get_manifest(model_id: str) -> Optional[ModelAdapterManifest]:
    """Get manifest by model ID."""
    return MODEL_REGISTRY.get(model_id)
