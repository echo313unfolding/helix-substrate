"""HuggingFace Quantizer integration for CDNA v3 compressed models.

Enables loading CDNA v3 compressed models via the standard HF API:

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("echolabs/zamba2-1.2b-helix")

The quantizer handles:
    1. Replacing nn.Linear with HelixLinear shells before weight loading
    2. Letting safetensors populate the compressed buffers (codebook, indices, etc.)
    3. Computing derived buffers (sidecar deltas, etc.) after loading

Registration:
    import helix_substrate.hf_quantizer  # registers cdna_v3 quantizer
    # Now AutoModelForCausalLM.from_pretrained() recognizes quantization_config.quant_method="cdna_v3"

Work Order: WO-HF-DISTRIBUTION-01
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quantization Config
# ---------------------------------------------------------------------------

try:
    from transformers.utils.quantization_config import QuantizationConfigMixin
    from transformers.quantizers.auto import (
        register_quantization_config,
        register_quantizer,
    )
    from transformers.quantizers.base import HfQuantizer

    HAS_HF_QUANTIZER_API = True
except ImportError:
    HAS_HF_QUANTIZER_API = False
    # Stubs so the module can still be imported for offline use
    QuantizationConfigMixin = object
    HfQuantizer = object

    def register_quantization_config(name):
        def decorator(cls):
            return cls
        return decorator

    def register_quantizer(name):
        def decorator(cls):
            return cls
        return decorator


class _QuantMethodShim:
    """Enum-like wrapper for quant_method string.

    transformers >= 4.49.0 expects quantization_config.quant_method to have a
    .value attribute (like an enum). Older versions treat it as a plain string.
    This shim satisfies both: str() returns the name, .value returns it too,
    and equality/hashing work against both strings and other shims.
    """

    def __init__(self, name: str):
        self.value = name

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"_QuantMethodShim({self.value!r})"

    def __eq__(self, other: object) -> bool:
        return self.value == getattr(other, "value", other)

    def __hash__(self) -> int:
        return hash(self.value)


@register_quantization_config("cdna_v3")
class CDNAv3Config(QuantizationConfigMixin):
    """Configuration for CDNA v3 quantized models.

    Stored in config.json under "quantization_config":
    {
        "quant_method": "cdna_v3",
        "bits": 8,
        "n_clusters": 256,
        "compressed_modules": ["model.layers.0.mamba.in_proj", ...],
        "compression_ratio": 4.0
    }
    """

    def __init__(
        self,
        bits: int = 8,
        n_clusters: int = 256,
        compressed_modules: Optional[List[str]] = None,
        compression_ratio: float = 4.0,
        n_svd_routed: int = 0,
        modules_to_not_convert: Optional[List[str]] = None,
        module_k_map: Optional[Dict[str, int]] = None,
        # Multi-dimensional VQ
        vector_dim: int = 1,
        module_vector_dim_map: Optional[Dict[str, int]] = None,
        # 12-bit packed index storage (saves 25% VRAM for k>256)
        index_packing: Optional[str] = None,
        # Backward compat with hf_integration.py config format
        codebook_size: Optional[int] = None,
        sidecar_enabled: bool = True,
        exact_patterns: Optional[List[str]] = None,
        **kwargs,
    ):
        # Use _QuantMethodShim so quant_method has .value (transformers >= 4.49.0)
        # while still comparing equal to plain "cdna_v3" strings (older versions).
        self.quant_method = _QuantMethodShim("cdna_v3")
        self.bits = bits
        # Accept codebook_size (old format) as n_clusters
        self.n_clusters = codebook_size if codebook_size is not None else n_clusters
        self.compressed_modules = compressed_modules or []
        self.compression_ratio = compression_ratio
        self.n_svd_routed = n_svd_routed
        self.modules_to_not_convert = modules_to_not_convert or []
        self.sidecar_enabled = sidecar_enabled
        self.exact_patterns = exact_patterns or []
        # Per-module codebook size overrides (module_path -> k)
        self.module_k_map = module_k_map or {}
        # Multi-dimensional VQ: 1=scalar (default), 2=2D pairs, 4=4D quads
        self.vector_dim = vector_dim
        # Per-module vector_dim overrides (module_path -> d)
        self.module_vector_dim_map = module_vector_dim_map or {}
        # 12-bit packed index storage: "12bit" or None
        self.index_packing = index_packing

    def to_dict(self) -> dict:
        d = {
            "quant_method": "cdna_v3",
            "bits": self.bits,
            "n_clusters": self.n_clusters,
            "compressed_modules": self.compressed_modules,
            "compression_ratio": self.compression_ratio,
            "n_svd_routed": self.n_svd_routed,
        }
        if self.module_k_map:
            d["module_k_map"] = self.module_k_map
        if self.vector_dim > 1:
            d["vector_dim"] = self.vector_dim
        if self.module_vector_dim_map:
            d["module_vector_dim_map"] = self.module_vector_dim_map
        return d

    @classmethod
    def from_dict(
        cls, config_dict: Dict[str, Any], return_unused_kwargs: bool = False, **kwargs
    ) -> Union["CDNAv3Config", tuple]:
        """Deserialize from config.json dict.

        Required by transformers >= 4.49.0 for quantization config loading.
        Strips 'quant_method' (set in __init__) and passes remaining fields.
        """
        d = dict(config_dict)
        d.pop("quant_method", None)
        d.update(kwargs)
        obj = cls(**d)
        return (obj, {}) if return_unused_kwargs else obj


# ---------------------------------------------------------------------------
# Module replacement
# ---------------------------------------------------------------------------

def _get_helix_linear_cls():
    """Lazy import to avoid circular dependencies."""
    from helix_substrate.helix_linear import HelixLinear
    return HelixLinear


def _replace_with_helix_linear(
    model: nn.Module,
    compressed_modules: set,
    compute_dtype: torch.dtype = torch.float32,
    module_k_map: Optional[Dict[str, int]] = None,
    n_clusters_default: int = 256,
    vector_dim_default: int = 1,
    module_vector_dim_map: Optional[Dict[str, int]] = None,
    index_packing: Optional[str] = None,
) -> tuple:
    """Replace nn.Linear modules with HelixLinear shells.

    Handles shared modules (e.g., Zamba2's shared_transformer) by tracking
    already-replaced objects to avoid double replacement.

    Also handles nn.Embedding: registers codebook/indices buffers so
    safetensors can load them, then reconstructs dense weight in
    _process_model_after_weight_loading.

    Args:
        model: HF model (on meta device during loading)
        compressed_modules: Set of module paths to replace
        compute_dtype: Compute precision for HelixLinear
        module_k_map: Optional per-module codebook size overrides
        n_clusters_default: Default codebook size
        vector_dim_default: Default vector dimension (1=scalar)
        module_vector_dim_map: Optional per-module vector_dim overrides

    Returns:
        (n_replaced, compressed_embeddings_set)
    """
    HelixLinear = _get_helix_linear_cls()
    replaced = 0
    replaced_ids = set()  # Track object ids to handle shared modules
    compressed_embeddings = set()
    k_map = module_k_map or {}
    vd_map = module_vector_dim_map or {}

    for name, module in list(model.named_modules()):
        if name not in compressed_modules:
            continue

        # Skip if this exact Python object was already replaced (shared modules)
        if id(module) in replaced_ids:
            continue

        # Handle nn.Embedding: register codebook/indices buffers for safetensors
        if isinstance(module, nn.Embedding):
            k = k_map.get(name, n_clusters_default)
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
            compressed_embeddings.add(name)
            replaced_ids.add(id(module))
            replaced += 1
            continue

        if not isinstance(module, nn.Linear):
            continue

        # Per-module codebook size (default 256 for backward compat)
        n_clusters = k_map.get(name, n_clusters_default)
        vd = vd_map.get(name, vector_dim_default)

        # Create shell with matching dimensions
        helix_shell = HelixLinear.from_quantized_config(
            in_features=module.in_features,
            out_features=module.out_features,
            tensor_name=f"{name}.weight",
            compute_dtype=compute_dtype,
            n_clusters=n_clusters,
            vector_dim=vd,
            index_packing=index_packing if n_clusters > 256 else None,
        )

        replaced_ids.add(id(module))

        # Replace in parent
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], helix_shell)
        replaced += 1

    return replaced, compressed_embeddings


# ---------------------------------------------------------------------------
# HF Quantizer
# ---------------------------------------------------------------------------

@register_quantizer("cdna_v3")
class HelixHfQuantizer(HfQuantizer if HAS_HF_QUANTIZER_API else object):
    """HuggingFace quantizer for CDNA v3 compressed models.

    Lifecycle:
        1. HF creates model on meta device
        2. preprocess_model() → _process_model_before_weight_loading()
           Replaces target nn.Linear with HelixLinear shells
        3. HF loads safetensors into the shell buffers
        4. postprocess_model() → _process_model_after_weight_loading()
           Calls _recompute_derived() on each HelixLinear
    """

    requires_calibration = False
    required_packages = ["helix_substrate"]
    # Must be True so transformers calls check_quantized_param/create_quantized_param
    # for HelixLinear's variable-shape buffers (codebook, indices, sidecar, SVD).
    # Without this, set_module_tensor_to_device rejects shape mismatches between
    # HelixLinear's empty placeholder buffers and actual safetensors data.
    requires_parameters_quantization = True

    def __init__(self, quantization_config: CDNAv3Config, **kwargs):
        if HAS_HF_QUANTIZER_API:
            super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        """Check helix_substrate is available."""
        try:
            import helix_substrate.helix_linear
        except ImportError:
            raise ImportError(
                "CDNA v3 quantization requires helix-substrate: "
                "pip install helix-substrate"
            )

    def update_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """CDNA v3 works with any dtype — codebook is always float32."""
        if dtype is None:
            return torch.float32
        return dtype

    # transformers >= 4.49.0 calls update_torch_dtype instead of update_dtype.
    def update_torch_dtype(self, torch_dtype: torch.dtype) -> torch.dtype:
        """Same as update_dtype — alias for transformers >= 4.49.0 compat."""
        return self.update_dtype(torch_dtype)

    def check_quantized_param(
        self,
        model: nn.Module,
        param_value: torch.Tensor,
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """Return True for HelixLinear buffer parameters.

        When requires_parameters_quantization is True, transformers calls this
        for each parameter. Returning True routes the parameter through
        create_quantized_param instead of set_module_tensor_to_device,
        bypassing shape-mismatch errors for HelixLinear's variable-shape buffers.
        """
        HelixLinear = _get_helix_linear_cls()
        _helix_suffixes = (
            ".codebook", ".indices", ".sidecar_positions", ".sidecar_values",
            ".svd_U", ".svd_s", ".svd_Vt", ".channel_scales",
            ".codebook_f16", ".weight", ".bias",
        )
        if any(param_name.endswith(s) for s in _helix_suffixes):
            parts = param_name.rsplit(".", 1)
            if len(parts) == 2:
                try:
                    mod = model.get_submodule(parts[0])
                    if isinstance(mod, HelixLinear):
                        return True
                except (AttributeError, ValueError):
                    pass
        return False

    def create_quantized_param(
        self,
        model: nn.Module,
        param_value: torch.Tensor,
        param_name: str,
        target_device: torch.device,
        *args,
        **kwargs,
    ) -> None:
        """Register a HelixLinear buffer with the correct shape from safetensors.

        Called instead of set_module_tensor_to_device when check_quantized_param
        returns True. Deletes the old placeholder buffer and registers the actual
        tensor with its real shape.
        """
        parts = param_name.rsplit(".", 1)
        if len(parts) != 2:
            return
        mod = model.get_submodule(parts[0])
        buf_name = parts[1]
        new_val = param_value.to(device=target_device)
        if hasattr(mod, buf_name):
            delattr(mod, buf_name)
        mod.register_buffer(buf_name, new_val)

    def _process_model_before_weight_loading(
        self, model: nn.Module, **kwargs
    ) -> None:
        """Replace target nn.Linear with HelixLinear shells."""
        compressed = set(self.quantization_config.compressed_modules)
        if not compressed:
            logger.warning(
                "CDNAv3Config.compressed_modules is empty — "
                "no modules will be replaced with HelixLinear"
            )
            return

        # Expand compressed_modules with aliased paths from tied weights.
        # In transformers 5.x, shared modules (e.g., Zamba2's shared_transformer)
        # are separate Python objects, not aliases. The config only lists canonical
        # paths (e.g., model.layers.5.shared_transformer.*.weight), but layers
        # 11/17/23/29/35 have their own copies that also need replacement.
        tied_keys = getattr(model, "all_tied_weights_keys", {})
        k_map = dict(self.quantization_config.module_k_map or {})
        vd_map = dict(getattr(self.quantization_config, "module_vector_dim_map", None) or {})
        n_expanded = 0
        for target_key, source_key in tied_keys.items():
            # Strip .weight suffix to get module path
            if source_key.endswith(".weight"):
                source_mod = source_key[:-7]
                if source_mod in compressed and target_key.endswith(".weight"):
                    target_mod = target_key[:-7]
                    if target_mod not in compressed:
                        compressed.add(target_mod)
                        # Propagate k_map and vd_map from canonical to alias
                        if source_mod in k_map and target_mod not in k_map:
                            k_map[target_mod] = k_map[source_mod]
                        if source_mod in vd_map and target_mod not in vd_map:
                            vd_map[target_mod] = vd_map[source_mod]
                        n_expanded += 1
        if n_expanded:
            logger.info(
                f"CDNA v3: expanded compressed_modules with {n_expanded} "
                f"aliased paths from tied weights"
            )

        n, self._compressed_embeddings = _replace_with_helix_linear(
            model, compressed,
            module_k_map=k_map,
            n_clusters_default=self.quantization_config.n_clusters,
            vector_dim_default=getattr(self.quantization_config, "vector_dim", 1),
            module_vector_dim_map=vd_map,
            index_packing=getattr(self.quantization_config, "index_packing", None),
        )
        logger.info(f"CDNA v3: replaced {n} nn.Linear → HelixLinear"
                     f" ({len(self._compressed_embeddings)} embeddings)")

        # Detect aliased module paths (e.g., Zamba2's shared_transformer).
        # When the same module object appears under multiple named paths,
        # only the first path is "canonical" — the others are aliases.
        # We track aliased prefixes to filter them from expected/missing keys.
        #
        # Use _modules tree traversal (not named_modules, which deduplicates)
        # to find ALL paths including aliased ones.
        self._aliased_prefixes: set[str] = set()
        seen_ids: dict[int, str] = {}  # id(module) → canonical path

        def _walk_modules(mod: nn.Module, prefix: str = ""):
            for child_name, child in mod._modules.items():
                if child is None:
                    continue
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                mid = id(child)
                if mid in seen_ids:
                    # This is an aliased path — all keys under it are duplicates
                    self._aliased_prefixes.add(full_name)
                else:
                    seen_ids[mid] = full_name
                    _walk_modules(child, full_name)

        _walk_modules(model)
        if self._aliased_prefixes:
            logger.info(
                f"CDNA v3: detected {len(self._aliased_prefixes)} aliased module "
                f"prefix(es) (shared modules)"
            )

    def _process_model_after_weight_loading(
        self, model: nn.Module, **kwargs
    ) -> None:
        """Compute derived buffers for all HelixLinear modules + reconstruct embeddings."""
        HelixLinear = _get_helix_linear_cls()

        # Copy compressed buffers from canonical HelixLinears to aliased ones.
        # In transformers 5.x, shared modules are separate objects. The safetensors
        # only loads into the canonical path (e.g., layer 5). Aliased paths
        # (layers 11/17/23/29/35) have empty shells that need their buffers copied.
        tied_keys = getattr(model, "all_tied_weights_keys", {})
        _copy_bufs = ("codebook", "indices", "sidecar_positions", "sidecar_values",
                       "svd_U", "svd_s", "svd_Vt", "bias", "channel_scales",
                       "codebook_f16")
        copied_count = 0
        for target_key, source_key in tied_keys.items():
            if not source_key.endswith(".weight") or not target_key.endswith(".weight"):
                continue
            source_path = source_key[:-7]
            target_path = target_key[:-7]
            try:
                src_mod = model.get_submodule(source_path)
                tgt_mod = model.get_submodule(target_path)
            except (AttributeError, ValueError):
                continue
            if not isinstance(src_mod, HelixLinear) or not isinstance(tgt_mod, HelixLinear):
                continue
            if src_mod is tgt_mod:
                continue  # Already shared (transformers 4.x)
            # Copy buffers from canonical to alias if alias has empty codebook
            tgt_cb = getattr(tgt_mod, "codebook", None)
            if tgt_cb is not None and tgt_cb.numel() > 0:
                continue  # Already populated
            for buf_name in _copy_bufs:
                src_buf = getattr(src_mod, buf_name, None)
                if src_buf is not None:
                    tgt_mod.register_buffer(buf_name, src_buf)
            # Copy non-buffer attributes
            tgt_mod.has_svd = src_mod.has_svd
            tgt_mod.rank = src_mod.rank
            tgt_mod.has_channel_scales = src_mod.has_channel_scales
            tgt_mod.vector_dim = src_mod.vector_dim
            copied_count += 1
        if copied_count:
            logger.info(f"CDNA v3: copied buffers to {copied_count} aliased HelixLinear modules")

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, HelixLinear):
                module._recompute_derived()
                count += 1
        logger.info(f"CDNA v3: finalized {count} HelixLinear modules")

        # Recompute rotary embedding inv_freq if present.
        # When model is created on meta device, inv_freq (computed from config
        # in __init__) becomes a meta tensor. If not stored in safetensors,
        # it gets materialized as torch.empty() garbage after weight loading.
        config = getattr(model, "config", None)
        for name, module in model.named_modules():
            inv = getattr(module, "inv_freq", None)
            if inv is not None and isinstance(inv, torch.Tensor) and inv.numel() > 0:
                dim = inv.shape[0] * 2
                base = getattr(config, "rope_theta", 10000.0) if config else 10000.0
                new_inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                module.inv_freq = new_inv.to(inv.device)
                logger.info(f"CDNA v3: recomputed inv_freq for {name}")

        # Reconstruct compressed nn.Embedding modules.
        # codebook/indices buffers were registered in before_weight_loading
        # and populated from safetensors. Now: weight = codebook[indices].
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
                logger.warning(f"CDNA v3: {name} missing codebook/indices")
                continue

            with torch.no_grad():
                dense_weight = codebook[indices.long()]
                module.weight.copy_(dense_weight)

            # Clean up temporary buffers
            del module.codebook
            del module.indices
            embed_count += 1

        if embed_count:
            logger.info(f"CDNA v3: reconstructed {embed_count} compressed embedding(s)")

    def _is_under_alias(self, key: str) -> bool:
        """Check if a state_dict key belongs to an aliased (non-canonical) module path.

        Handles architectures with shared modules (e.g., Zamba2's shared_transformer
        referenced by layers 5, 11, 17, 23, 29, 35 but only instantiated once).
        """
        aliased = getattr(self, "_aliased_prefixes", set())
        for prefix in aliased:
            if key.startswith(prefix + "."):
                return True
        return False

    def update_missing_keys(
        self, model: nn.Module, missing_keys: list, prefix: str
    ) -> list:
        """Remove HelixLinear buffer keys and aliased module keys from missing_keys."""
        _helix_suffixes = {
            ".codebook_f16",
            "._sidecar_vq_vals",
            "._sidecar_rows",
            "._sidecar_cols",
            "._sidecar_deltas",
            ".svd_U", ".svd_s", ".svd_Vt",
            ".bias", ".channel_scales",
            ".weight",
        }
        compressed = set(self.quantization_config.compressed_modules)
        filtered = []
        for k in missing_keys:
            # Drop ALL keys under aliased module paths (data loaded via canonical path)
            if self._is_under_alias(k):
                continue
            # Drop derived/optional HelixLinear buffers
            is_helix_key = any(k.endswith(s) for s in _helix_suffixes)
            module_path = k.rsplit(".", 1)[0] if "." in k else k
            if is_helix_key and module_path in compressed:
                continue
            filtered.append(k)
        return filtered

    def update_expected_keys(
        self, model: nn.Module, expected_keys: list, loaded_keys: list
    ) -> list:
        """Filter expected keys for aliased paths and HelixLinear optional buffers."""
        _helix_suffixes = {
            ".codebook_f16",
            "._sidecar_vq_vals",
            "._sidecar_rows",
            "._sidecar_cols",
            "._sidecar_deltas",
            ".svd_U", ".svd_s", ".svd_Vt",
            ".bias", ".channel_scales",
            ".weight",
        }
        loaded = set(loaded_keys)
        compressed = set(self.quantization_config.compressed_modules)
        filtered = []
        for k in expected_keys:
            # Drop ALL keys under aliased module paths
            if self._is_under_alias(k):
                continue
            # Drop HelixLinear optional buffers not in the file
            is_helix_key = any(k.endswith(s) for s in _helix_suffixes)
            module_path = k.rsplit(".", 1)[0] if "." in k else k
            if is_helix_key and module_path in compressed and k not in loaded:
                continue
            filtered.append(k)
        return filtered

    @property
    def is_trainable(self) -> bool:
        return False

    def is_serializable(self, safe_serialization=None) -> bool:
        return True


# ---------------------------------------------------------------------------
# Aliases: register under all known quant_method names for backward compat.
# Models on HF may have "helix", "cdna_v3", or "hxq" in their config.json.
#
# Two paths:
#   1. Decorator API (transformers < 4.49.0): register_quantization_config/register_quantizer
#   2. Direct mapping (transformers >= 4.49.0): AUTO_QUANTIZER_MAPPING / AUTO_QUANTIZATION_CONFIG_MAPPING
# ---------------------------------------------------------------------------
if HAS_HF_QUANTIZER_API:
    for _alias in ("helix", "hxq"):
        try:
            register_quantization_config(_alias)(CDNAv3Config)
            register_quantizer(_alias)(HelixHfQuantizer)
        except Exception:
            pass  # Already registered or API changed

# Direct mapping fallback for transformers >= 4.49.0 where decorator API may not exist
# but AUTO_QUANTIZER_MAPPING is available.
try:
    from transformers.quantizers.auto import (
        AUTO_QUANTIZER_MAPPING,
        AUTO_QUANTIZATION_CONFIG_MAPPING,
    )
    for _alias in ("cdna_v3", "helix", "hxq"):
        AUTO_QUANTIZATION_CONFIG_MAPPING[_alias] = CDNAv3Config
        AUTO_QUANTIZER_MAPPING[_alias] = HelixHfQuantizer
except ImportError:
    pass  # Older transformers without these dicts
