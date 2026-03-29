"""
HuggingFace Transformers integration for Helix VQ compression.

Registers HelixQuantizationConfig and HelixHfQuantizer so that
`AutoModelForCausalLM.from_pretrained()` works on Helix-compressed checkpoints.

Usage:
    import helix_substrate  # triggers registration

    model = AutoModelForCausalLM.from_pretrained(
        "path/to/helix-checkpoint",
        trust_remote_code=True,
    )
    # All compressed layers are HelixLinear modules.
    # Forward pass works normally.

Steps 3-4 of HF integration (WO-HF-INTEGRATION-01).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 3: HelixQuantizationConfig
# ---------------------------------------------------------------------------

try:
    from transformers.quantizers.auto import (
        register_quantization_config,
        register_quantizer,
    )
    from transformers.quantizers.base import HfQuantizer
    from transformers.utils.quantization_config import QuantizationConfigMixin

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


if HF_AVAILABLE:

    @register_quantization_config("helix")
    @dataclass
    class HelixQuantizationConfig(QuantizationConfigMixin):
        """
        Configuration for Helix VQ-256 compressed models.

        Stored in config.json as:
            "quantization_config": {
                "quant_method": "helix",
                "codebook_size": 256,
                "sidecar_enabled": true,
                "exact_patterns": ["embed_tokens", "lm_head", ...],
                "compressed_modules": ["model.layers.0.self_attn.q_proj", ...]
            }
        """

        def __init__(
            self,
            codebook_size: int = 256,
            sidecar_enabled: bool = True,
            exact_patterns: Optional[list[str]] = None,
            compressed_modules: Optional[list[str]] = None,
            modules_to_not_convert: Optional[list[str]] = None,
            **kwargs,
        ):
            self.quant_method = "helix"
            self.codebook_size = codebook_size
            self.sidecar_enabled = sidecar_enabled
            self.exact_patterns = exact_patterns or [
                "embed_tokens", "embed_positions", "wte", "wpe",
                "lm_head", "layernorm", "layer_norm", "norm",
                "backbone.embedding",
            ]
            self.compressed_modules = compressed_modules or []
            self.modules_to_not_convert = modules_to_not_convert
            self.post_init()

        def post_init(self):
            """Validate config."""
            if self.codebook_size < 2 or self.codebook_size > 65536:
                raise ValueError(f"codebook_size must be 2-65536, got {self.codebook_size}")

    # ---------------------------------------------------------------------------
    # Step 4: HelixHfQuantizer
    # ---------------------------------------------------------------------------

    @register_quantizer("helix")
    class HelixHfQuantizer(HfQuantizer):
        """
        HuggingFace quantizer for Helix VQ-256 compressed models.

        Handles the two-phase loading:
        1. Before weights: replace nn.Linear with empty HelixLinear shells
        2. After weights: HelixLinear buffers populated from safetensors
        """

        requires_calibration = False  # Helix is calibration-free
        required_packages = ["helix_substrate"]

        def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
            super().__init__(quantization_config, **kwargs)
            self.quantization_config = quantization_config

        def validate_environment(self, *args, **kwargs):
            try:
                import helix_substrate
            except ImportError:
                raise ImportError(
                    "Using `helix` quantization requires helix-substrate: "
                    "`pip install helix-substrate`"
                )

        def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
            if dtype is None:
                dtype = torch.float32
                logger.info("Helix: defaulting to float32 compute dtype")
            return dtype

        def _process_model_before_weight_loading(
            self,
            model: "PreTrainedModel",
            keep_in_fp32_modules: Optional[list[str]] = None,
            **kwargs,
        ):
            """Replace nn.Linear modules with HelixLinear shells before weights load."""
            from helix_substrate.helix_linear import HelixLinear

            compressed_set = set(self.quantization_config.compressed_modules)
            if not compressed_set:
                logger.warning("Helix: no compressed_modules in quantization_config, nothing to replace")
                return

            modules_to_skip = set(self.quantization_config.modules_to_not_convert or [])

            replaced = 0
            self._compressed_embeddings: set[str] = set()

            for name, module in list(model.named_modules()):
                if name not in compressed_set:
                    continue
                if name in modules_to_skip:
                    continue

                # Handle nn.Embedding: register codebook/indices buffers so
                # safetensors can load them, then reconstruct dense weight
                # in _process_model_after_weight_loading.
                if isinstance(module, nn.Embedding):
                    k = self.quantization_config.codebook_size
                    module.register_buffer(
                        "codebook", torch.zeros(k, dtype=torch.float32)
                    )
                    module.register_buffer(
                        "indices",
                        torch.zeros(
                            module.num_embeddings,
                            module.embedding_dim,
                            dtype=torch.uint8,
                        ),
                    )
                    self._compressed_embeddings.add(name)
                    replaced += 1
                    continue

                if not isinstance(module, nn.Linear):
                    continue

                # Create empty HelixLinear shell — buffers will be loaded from safetensors
                in_f = module.in_features
                out_f = module.out_features
                k = self.quantization_config.codebook_size

                # Create placeholder tensors (will be overwritten by safetensors loader)
                empty_codebook = torch.zeros(k, dtype=torch.float32)
                empty_indices = torch.zeros(out_f, in_f, dtype=torch.uint8)

                helix_mod = HelixLinear(
                    in_features=in_f,
                    out_features=out_f,
                    codebook=empty_codebook,
                    indices=empty_indices,
                    tensor_name=name,
                )
                helix_mod.requires_grad_(False)

                # Replace in parent
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], helix_mod)
                replaced += 1

            logger.info(f"Helix: replaced {replaced} modules ({len(self._compressed_embeddings)} embeddings)")
            model.config.quantization_config = self.quantization_config

        def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
            """
            After safetensors loads codebook/indices into HelixLinear buffers,
            load sidecar data directly from the safetensors file.

            Sidecar keys (sidecar_indices, sidecar_values) are not registered as
            buffers in the initial shell, so HF's default loader skips them.
            We load them here and wire up the derived sidecar buffers.
            """
            from helix_substrate.helix_linear import HelixLinear
            from safetensors import safe_open

            # Find the safetensors file to load sidecar data from
            model_dir = getattr(model.config, "_name_or_path", None)
            if model_dir is None:
                return model

            safetensors_path = Path(model_dir) / "model.safetensors"
            if not safetensors_path.exists():
                # _name_or_path may be a HF hub repo ID — resolve via cache
                try:
                    from huggingface_hub import hf_hub_download
                    safetensors_path = Path(hf_hub_download(model_dir, "model.safetensors"))
                except Exception:
                    logger.warning("Helix: could not locate model.safetensors for sidecar loading")
                    return model

            # Load sidecar data directly from safetensors
            # Handle both key conventions: .sidecar_positions (HelixLinear native)
            # and .sidecar_indices (convert_to_hf.py format)
            sidecar_data = {}
            with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if ".sidecar_positions" in key or ".sidecar_indices" in key or ".sidecar_values" in key:
                        sidecar_data[key] = f.get_tensor(key)

            # Wire sidecar data into HelixLinear modules
            for name, module in model.named_modules():
                if not isinstance(module, HelixLinear):
                    continue

                # Try both key conventions
                si_key = f"{name}.sidecar_positions"
                if si_key not in sidecar_data:
                    si_key = f"{name}.sidecar_indices"
                sv_key = f"{name}.sidecar_values"

                if si_key in sidecar_data and sv_key in sidecar_data:
                    positions = sidecar_data[si_key].long()
                    values = sidecar_data[sv_key].float()

                    device = module.codebook.device
                    positions = positions.to(device)
                    values = values.to(device)

                    module.register_buffer("sidecar_positions", positions.contiguous())
                    module.register_buffer("sidecar_values", values.contiguous())

                    # Recompute derived sidecar buffers
                    in_f = module.in_features
                    idx_flat = module.indices.reshape(-1)
                    vq_at_sidecar = module.codebook[idx_flat[positions].long()]

                    module.register_buffer("_sidecar_vq_vals", vq_at_sidecar.contiguous())
                    module.register_buffer("_sidecar_rows", (positions // in_f).long())
                    module.register_buffer("_sidecar_cols", (positions % in_f).long())
                    module.register_buffer("_sidecar_deltas",
                                           (values - vq_at_sidecar).contiguous())

                # Handle SVD factors if loaded
                if module.svd_U is not None and module.svd_U.numel() > 0:
                    module.has_svd = True
                    module.rank = module.svd_U.shape[1] if module.svd_U.dim() > 1 else 0

            sidecar_count = sum(1 for n, m in model.named_modules()
                                if isinstance(m, HelixLinear)
                                and m.sidecar_positions is not None
                                and m.sidecar_positions.numel() > 0)
            logger.info(f"Helix: loaded sidecar corrections for {sidecar_count} modules")

            # Reconstruct dense weights for compressed nn.Embedding modules.
            # The codebook/indices buffers were registered in _process_model_before_weight_loading
            # and populated from safetensors. Now reconstruct: weight = codebook[indices].
            compressed_embeddings = getattr(self, "_compressed_embeddings", set())
            embed_count = 0
            for name, module in model.named_modules():
                if name not in compressed_embeddings:
                    continue
                if not isinstance(module, nn.Embedding):
                    continue
                codebook = getattr(module, "codebook", None)
                indices = getattr(module, "indices", None)
                if codebook is None or indices is None:
                    logger.warning(f"Helix: {name} marked as compressed embedding but missing codebook/indices")
                    continue

                # Reconstruct dense embedding weight from VQ representation
                with torch.no_grad():
                    dense_weight = codebook[indices.long()]
                    module.weight.copy_(dense_weight)

                # Also apply sidecar corrections if present in safetensors
                sp_key = f"{name}.sidecar_positions"
                if sp_key not in sidecar_data:
                    sp_key = f"{name}.sidecar_indices"
                sv_key = f"{name}.sidecar_values"
                if sp_key in sidecar_data and sv_key in sidecar_data:
                    positions = sidecar_data[sp_key].long()
                    values = sidecar_data[sv_key].float()
                    in_f = module.embedding_dim
                    rows = positions // in_f
                    cols = positions % in_f
                    module.weight.data[rows, cols] = values.to(module.weight.device)
                    logger.info(f"Helix: {name} embedding reconstructed with {len(positions)} sidecar corrections")
                else:
                    logger.info(f"Helix: {name} embedding reconstructed (no sidecar)")

                # Clean up temporary buffers — not needed at runtime
                del module.codebook
                del module.indices
                embed_count += 1

            if embed_count:
                logger.info(f"Helix: reconstructed {embed_count} compressed embedding(s)")

            return model

        @property
        def is_trainable(self) -> bool:
            return False

        def is_serializable(self, safe_serialization=None):
            return True

        def check_quantized_param(
            self,
            model: "PreTrainedModel",
            param_value: "torch.Tensor",
            param_name: str,
            state_dict: dict[str, Any],
            **kwargs,
        ) -> bool:
            """
            Check if a parameter should be loaded into a HelixLinear module.

            This is called during weight loading. We need to handle:
            - .codebook, .indices → load into existing buffers
            - .sidecar_indices, .sidecar_values → need special registration
            - .svd_U, .svd_s, .svd_Vt → load into existing buffers
            """
            # Extract module name from param_name (e.g., "model.layers.0.self_attn.q_proj.codebook")
            helix_suffixes = {".codebook", ".indices", ".sidecar_indices", ".sidecar_positions",
                              ".sidecar_values", ".svd_U", ".svd_s", ".svd_Vt"}

            for suffix in helix_suffixes:
                if param_name.endswith(suffix):
                    module_name = param_name[:-len(suffix)]
                    try:
                        module = model
                        for part in module_name.split("."):
                            module = getattr(module, part)
                        from helix_substrate.helix_linear import HelixLinear
                        # Accept both HelixLinear and compressed Embedding modules
                        if isinstance(module, HelixLinear):
                            return True
                        compressed_embeddings = getattr(self, "_compressed_embeddings", set())
                        if isinstance(module, nn.Embedding) and module_name in compressed_embeddings:
                            return True
                        return False
                    except AttributeError:
                        return False

            return False

        def create_quantized_param(
            self,
            model: "PreTrainedModel",
            param_value: "torch.Tensor",
            param_name: str,
            target_device: "torch.device",
            state_dict: dict[str, Any],
            unexpected_keys: Optional[list[str]] = None,
        ):
            """
            Load a quantized parameter into the correct HelixLinear buffer.
            """
            from helix_substrate.helix_linear import HelixLinear

            helix_suffixes = [".codebook", ".indices", ".sidecar_indices", ".sidecar_positions",
                              ".sidecar_values", ".svd_U", ".svd_s", ".svd_Vt"]

            for suffix in helix_suffixes:
                if param_name.endswith(suffix):
                    module_name = param_name[:-len(suffix)]
                    attr_name = suffix[1:]  # strip leading dot
                    break
            else:
                return

            # Navigate to the module
            module = model
            for part in module_name.split("."):
                module = getattr(module, part)

            # Handle compressed nn.Embedding — load codebook/indices into registered buffers
            compressed_embeddings = getattr(self, "_compressed_embeddings", set())
            if isinstance(module, nn.Embedding) and module_name in compressed_embeddings:
                param_value = param_value.to(target_device)
                if attr_name == "codebook":
                    module.codebook.copy_(param_value.float())
                elif attr_name == "indices":
                    if param_value.dtype != torch.uint8:
                        param_value = param_value.to(torch.uint8)
                    module.indices.copy_(param_value)
                if unexpected_keys is not None and param_name in unexpected_keys:
                    unexpected_keys.remove(param_name)
                return

            if not isinstance(module, HelixLinear):
                return

            # Register or update the buffer
            param_value = param_value.to(target_device)

            if attr_name in ("sidecar_indices", "sidecar_positions"):
                # Register as buffer — both names map to sidecar_positions
                module.register_buffer("sidecar_positions", param_value.long().contiguous())
            elif attr_name == "indices":
                # Ensure uint8
                if param_value.dtype != torch.uint8:
                    param_value = param_value.to(torch.uint8)
                module.indices = param_value.contiguous()
            elif attr_name == "codebook":
                module.codebook = param_value.float().contiguous()
            elif attr_name == "sidecar_values":
                module.sidecar_values = param_value.float().contiguous()
            elif attr_name in ("svd_U", "svd_s", "svd_Vt"):
                setattr(module, attr_name, param_value.float().contiguous())
                if attr_name == "svd_U":
                    module.has_svd = True
                    module.rank = param_value.shape[1] if param_value.dim() > 1 else 0
            else:
                setattr(module, attr_name, param_value.contiguous())

            # Remove from unexpected keys if present
            if unexpected_keys is not None and param_name in unexpected_keys:
                unexpected_keys.remove(param_name)


else:
    # transformers not available — provide stub classes so import doesn't fail
    class HelixQuantizationConfig:
        def __init__(self, **kwargs):
            raise ImportError("transformers is required for HF integration")

    class HelixHfQuantizer:
        def __init__(self, **kwargs):
            raise ImportError("transformers is required for HF integration")
