"""
Multi-model manager for constrained devices.

Four modes:
1. ModelManager: One model on GPU at a time. Clean swap (unload → load). ~5s per swap.
   Works for models that fit in VRAM (TinyLlama, Qwen 1.5B).
2. ZeroSwapModelManager: Two models co-resident on GPU. HelixLinear indices in pinned
   host memory (GPU reads via BAR1). Swap = pointer switch (~0 ms).
   Uses pin-before-move for 3B+ models that don't fit temporarily in VRAM.
3. SplitRuntimeManager: TinyLlama on CPU (control plane), Qwen on GPU (execution plane).
   No model swaps. GPU stays 100% dedicated to the coder.
4. Coder3BManager: Qwen 3B only, zero-copy, no controller model. Fastest coding path.

Usage:
    # Classic swap mode (1.5B only):
    mgr = ModelManager()
    model, tokenizer = mgr.ensure_model(ModelTarget.TINYLLAMA)

    # Zero-swap with 3B coder:
    mgr = ZeroSwapModelManager(coder_target=ModelTarget.QWEN_CODER_3B)
    model, tokenizer = mgr.ensure_model(ModelTarget.QWEN_CODER_3B)  # ~0 ms

    # Coder-only 3B (fastest):
    mgr = Coder3BManager()
    model, tokenizer = mgr.ensure_model(ModelTarget.QWEN_CODER_3B)
"""

import gc
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from .device_utils import resolve_device, empty_cache, memory_allocated, reset_peak_memory
from .helix_linear import HelixLinear, load_helix_linear_from_cdnav3, swap_to_helix, swap_summary
from .query_classifier import ModelTarget


HF_PATTERNS = {
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
}

TENSOR_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

MODEL_CONFIGS = {
    ModelTarget.TINYLLAMA: {
        "name": "TinyLlama-1.1B",
        "model_dir": Path.home() / "models" / "tinyllama_fp32",
        "n_blocks": 22,
        "classic_ok": True,   # fits in VRAM for classic mode
    },
    ModelTarget.QWEN_CODER: {
        "name": "Qwen2.5-Coder-1.5B",
        "model_dir": Path.home() / "models" / "qwen2.5-coder-1.5b-instruct",
        "n_blocks": 28,
        "classic_ok": True,
    },
    ModelTarget.QWEN_CODER_3B: {
        "name": "Qwen2.5-Coder-3B",
        "model_dir": Path.home() / "models" / "qwen2.5-coder-3b-instruct",
        "n_blocks": 36,
        "classic_ok": False,  # too big for T2000 VRAM even compressed
    },
    ModelTarget.QWEN_INSTRUCT_3B: {
        "name": "Qwen2.5-3B-Instruct",
        "model_dir": Path.home() / "models" / "qwen2.5-3b-instruct",
        "n_blocks": 36,
        "classic_ok": False,  # same architecture as coder-3B
    },
}

# Coder targets — any of these are valid coder models
CODER_TARGETS = {ModelTarget.QWEN_CODER, ModelTarget.QWEN_CODER_3B}

# Instruct targets — general-purpose extraction/reasoning (FGIP, domain tasks)
INSTRUCT_TARGETS = {ModelTarget.QWEN_INSTRUCT_3B}


class ModelManager:
    """Manages dual-model lifecycle on a constrained device."""

    def __init__(self, device: str = "auto"):
        self.device = resolve_device(device)
        self.active_target: Optional[ModelTarget] = None
        self.model = None
        self.tokenizer = None
        self._tokenizers = {}  # Pre-loaded on CPU
        self._swap_count = 0
        self._swap_history = []

    def _get_tokenizer(self, target: ModelTarget):
        """Get or load tokenizer (cached on CPU)."""
        if target not in self._tokenizers:
            from transformers import AutoTokenizer
            cfg = MODEL_CONFIGS[target]
            self._tokenizers[target] = AutoTokenizer.from_pretrained(str(cfg["model_dir"]))
        return self._tokenizers[target]

    def _unload(self):
        """Unload current model from device. Verifies memory is freed."""
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            self.tokenizer = None
            empty_cache(self.device)
            gc.collect()

            mem_after = memory_allocated(self.device)
            if mem_after > 50:
                # Force cleanup
                empty_cache(self.device)
                gc.collect()

        self.active_target = None

    def _load(self, target: ModelTarget) -> Tuple[torch.nn.Module, object]:
        """Load model from persistent CDNA v3 dir onto GPU."""
        from transformers import AutoModelForCausalLM

        cfg = MODEL_CONFIGS[target]
        cdna_dir = cfg["model_dir"] / "cdnav3"

        if not cdna_dir.exists():
            raise FileNotFoundError(
                f"Pre-compressed CDNA dir not found: {cdna_dir}\n"
                f"Run: python tools/precompress_models.py"
            )

        t0 = time.time()

        # Load base model on CPU
        model = AutoModelForCausalLM.from_pretrained(
            str(cfg["model_dir"]), dtype=torch.float32
        )
        model.eval()

        # Load HelixLinear modules from persistent CDNA v3
        helix_modules = {}
        for block_idx in range(cfg["n_blocks"]):
            for tensor_type in TENSOR_TYPES:
                hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                safe_name = hf_name.replace("/", "_").replace(".", "_")
                tensor_dir = cdna_dir / f"{safe_name}.cdnav3"

                if not tensor_dir.exists():
                    continue

                module_path = hf_name.replace(".weight", "")

                # Get bias from base model
                parts = module_path.split(".")
                mod = model
                for p in parts:
                    mod = getattr(mod, p)
                bias = mod.bias.data.clone() if mod.bias is not None else None

                helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
                helix_modules[module_path] = helix_mod

        # Swap
        model = swap_to_helix(model, helix_modules)
        summary = swap_summary(model)

        # Move to device
        reset_peak_memory(self.device)
        model = model.to(self.device)
        model.eval()

        vram_mb = memory_allocated(self.device)
        load_time = time.time() - t0

        tokenizer = self._get_tokenizer(target)

        self.model = model
        self.tokenizer = tokenizer
        self.active_target = target

        swap_info = {
            "target": target.value,
            "load_time_s": round(load_time, 1),
            "vram_mb": round(vram_mb),
            "helix_modules": summary["helix_modules"],
            "linear_modules": summary["linear_modules"],
            "compression_ratio": summary["overall_ratio"],
        }
        self._swap_history.append(swap_info)
        self._swap_count += 1

        return model, tokenizer

    def ensure_model(self, target: ModelTarget) -> Tuple[torch.nn.Module, object]:
        """
        Ensure the target model is loaded on GPU.

        If already loaded, returns immediately (no-op).
        If different model is loaded, swaps cleanly.

        Returns:
            (model, tokenizer)
        """
        if self.active_target == target and self.model is not None:
            return self.model, self.tokenizer

        if self.active_target is not None:
            self._unload()

        model, tokenizer = self._load(target)
        return model, tokenizer

    def status(self) -> dict:
        """Return current manager state."""
        vram = memory_allocated(self.device)
        return {
            "active_model": self.active_target.value if self.active_target else None,
            "vram_mb": round(vram),
            "swap_count": self._swap_count,
            "last_swap": self._swap_history[-1] if self._swap_history else None,
        }


class ZeroSwapModelManager:
    """Co-resident multi-model manager. All models always loaded. Swap = pointer switch.

    HelixLinear indices live in pinned host RAM. GPU reads via PCIe BAR1 — zero VRAM
    for index data. Uses pin-before-move: indices are pinned on CPU before model.to(cuda),
    so models of any size work without temporary VRAM spike.

    Swap cost: ~0 ms (pointer switch) vs ~5100 ms (old load/unload cycle).
    Per-tensor latency: ~20-38% slower depending on matrix size (PCIe 3.0 x4).

    Args:
        device: GPU device string or "auto".
        coder_target: Which coder model to load. Default QWEN_CODER (1.5B).
            Use QWEN_CODER_3B for the 3B model (4.44x, 3.6 tok/s, 1254 MB VRAM).
        default_target: Which model is active at init.
        models: Explicit list of ModelTargets to load. Overrides coder_target if set.
    """

    def __init__(self, device: str = "auto",
                 coder_target: ModelTarget = ModelTarget.QWEN_CODER,
                 default_target: Optional[ModelTarget] = None,
                 models: Optional[list] = None):
        from .zerocopy import PinnedBuffer, pin_indices_from_file

        self.device = resolve_device(device)
        self._models: dict = {}
        self._tokenizers: dict = {}
        self._pinned_buffers: list = []
        self._load_times: dict = {}
        self.active_target: Optional[ModelTarget] = None
        self.model = None
        self.tokenizer = None
        self._swap_count = 0
        self._swap_history = []

        # Determine which models to load
        if models is not None:
            targets = models
        else:
            targets = [ModelTarget.TINYLLAMA, coder_target]

        for target in targets:
            t0 = time.time()
            model, tok, n_pinned = self._load_zc(target)
            load_s = round(time.time() - t0, 1)
            self._models[target] = model
            self._tokenizers[target] = tok
            self._load_times[target] = load_s
            vram = memory_allocated(self.device)
            print(f"  [{MODEL_CONFIGS[target]['name']}] loaded: "
                  f"{load_s}s, {n_pinned} ZC indices, VRAM={vram:.0f} MB")

        # Set default active model
        if default_target is None:
            default_target = targets[0]
        self.active_target = default_target
        self.model = self._models[default_target]
        self.tokenizer = self._tokenizers[default_target]

    def _load_zc(self, target: ModelTarget) -> Tuple:
        """Load model with zero-copy HelixLinear indices.

        Uses pin-before-move: indices are pinned to host on CPU, then model.to(cuda)
        only moves non-index data to GPU. This works for models of any size (no
        temporary VRAM spike from indices).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .zerocopy import pin_indices_from_file

        cfg = MODEL_CONFIGS[target]
        cdna_dir = cfg["model_dir"] / "cdnav3"

        if not cdna_dir.exists():
            raise FileNotFoundError(f"CDNA dir not found: {cdna_dir}")

        model = AutoModelForCausalLM.from_pretrained(
            str(cfg["model_dir"]), dtype=torch.float32
        )
        model.eval()

        # Load HelixLinear modules from CDNA v3
        helix_modules = {}
        for block_idx in range(cfg["n_blocks"]):
            for tensor_type in TENSOR_TYPES:
                hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                safe_name = hf_name.replace("/", "_").replace(".", "_")
                tensor_dir = cdna_dir / f"{safe_name}.cdnav3"
                if not tensor_dir.exists():
                    continue
                module_path = hf_name.replace(".weight", "")
                parts = module_path.split(".")
                mod = model
                for p in parts:
                    mod = getattr(mod, p)
                bias = mod.bias.data.clone() if mod.bias is not None else None
                helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
                helix_modules[module_path] = helix_mod

        model = swap_to_helix(model, helix_modules)

        # Pin-before-move: pin indices to host while model is still on CPU.
        # Pinned tensors appear as CUDA to PyTorch, so model.to(cuda) is a no-op on them.
        n_pinned = 0
        for name, module in model.named_modules():
            if not isinstance(module, HelixLinear):
                continue
            tensor_name = name + ".weight"
            safe_name = tensor_name.replace("/", "_").replace(".", "_")
            tensor_dir = cdna_dir / f"{safe_name}.cdnav3"
            indices_path = tensor_dir / "indices.bin"
            if not indices_path.exists():
                continue
            shape = tuple(module.indices.shape)
            zc_indices, pinned = pin_indices_from_file(indices_path, shape)
            self._pinned_buffers.append(pinned)
            module._buffers["indices"] = zc_indices
            n_pinned += 1

        # Now move to GPU — indices are already "CUDA" (pinned host), rest moves normally
        reset_peak_memory(self.device)
        model = model.to(self.device)
        model.eval()

        gc.collect()
        empty_cache(self.device)

        tokenizer = AutoTokenizer.from_pretrained(str(cfg["model_dir"]))
        return model, tokenizer, n_pinned

    def ensure_model(self, target: ModelTarget) -> Tuple[torch.nn.Module, object]:
        """Switch to target model. Cost: ~0 ms (pointer switch).

        Also accepts QWEN_CODER as alias for whatever coder is loaded.
        """
        # Remap QWEN_CODER to whatever coder is actually loaded
        if target == ModelTarget.QWEN_CODER and target not in self._models:
            for t in self._models:
                if t in CODER_TARGETS:
                    target = t
                    break

        if self.active_target == target and self.model is not None:
            return self.model, self.tokenizer

        if target not in self._models:
            raise ValueError(f"Model {target.value} not loaded. "
                             f"Available: {[t.value for t in self._models]}")

        t0 = time.perf_counter()
        self.model = self._models[target]
        self.tokenizer = self._tokenizers[target]
        self.active_target = target
        switch_us = (time.perf_counter() - t0) * 1e6

        self._swap_count += 1
        self._swap_history.append({
            "target": target.value,
            "switch_us": round(switch_us, 1),
        })

        return self.model, self.tokenizer

    def status(self) -> dict:
        """Return current manager state."""
        vram = memory_allocated(self.device)
        return {
            "active_model": self.active_target.value if self.active_target else None,
            "vram_mb": round(vram),
            "swap_count": self._swap_count,
            "mode": "zero_swap",
            "models_loaded": [t.value for t in self._models],
            "pinned_buffers": len(self._pinned_buffers),
            "pinned_mb": round(sum(b.nbytes for b in self._pinned_buffers) / 1024 / 1024),
            "load_times": {t.value: s for t, s in self._load_times.items()},
            "last_swap": self._swap_history[-1] if self._swap_history else None,
        }

    def cleanup(self):
        """Unregister all pinned buffers and free models."""
        for buf in self._pinned_buffers:
            buf.unregister()
        self._pinned_buffers.clear()
        for target in list(self._models):
            del self._models[target]
        self._models.clear()
        gc.collect()
        empty_cache(self.device)


class Coder3BManager:
    """Single-model manager: Qwen 3B only, zero-copy. Fastest coding path.

    No TinyLlama, no model swaps. GPU stays 100% dedicated to coding.
    Uses pin-before-move for the 3B model.

    Measured on T2000:
        VRAM: 1254 MB (31%)
        Speed: 3.6 tok/s
        Headroom: 2738 MB
    """

    def __init__(self, device: str = "auto"):
        self._zs = ZeroSwapModelManager(
            device=device,
            models=[ModelTarget.QWEN_CODER_3B],
            default_target=ModelTarget.QWEN_CODER_3B,
        )
        self.device = self._zs.device
        self.model = self._zs.model
        self.tokenizer = self._zs.tokenizer
        self.active_target = ModelTarget.QWEN_CODER_3B

    def ensure_model(self, target: ModelTarget = None) -> Tuple[torch.nn.Module, object]:
        """Always returns Qwen 3B. Ignores target."""
        return self.model, self.tokenizer

    def get_device(self, target=None) -> str:
        return self.device

    def status(self) -> dict:
        s = self._zs.status()
        s["mode"] = "coder_3b_only"
        return s

    def cleanup(self):
        self._zs.cleanup()


class SplitRuntimeManager:
    """Split-runtime: TinyLlama on CPU (control plane), Qwen on GPU (execution plane).

    No model swaps. GPU stays 100% dedicated to the coding model.
    CPU handles classification, routing, and short controller outputs.

    VRAM budget on T2000 (4 GB):
        CUDA context:        ~200 MB (no TinyLlama on GPU)
        Qwen non-HelixLinear: ~892 MB (FP32)
        Qwen HelixLinear:    ~1313 MB (compressed, in VRAM — no BAR1 penalty)
        Headroom:            ~1594 MB (for KV cache + activations)

    CPU budget:
        TinyLlama FP32:      ~4.2 GB RAM (model weights)
        HelixLinear on CPU:   naive tiled path, ~0.3-0.5 tok/s
        Controller tasks:     short outputs only (8-16 tokens)
    """

    def __init__(self, gpu_device: str = "auto"):
        self.gpu_device = resolve_device(gpu_device)
        self.cpu_device = "cpu"
        self._models: dict = {}
        self._tokenizers: dict = {}
        self._load_times: dict = {}
        self._devices: dict = {}
        self.active_target: Optional[ModelTarget] = None
        self.model = None
        self.tokenizer = None
        self._swap_count = 0
        self._swap_history = []

        # Load TinyLlama on CPU
        t0 = time.time()
        model_cpu, tok_cpu = self._load_model(
            ModelTarget.TINYLLAMA, device=self.cpu_device)
        cpu_time = round(time.time() - t0, 1)
        self._models[ModelTarget.TINYLLAMA] = model_cpu
        self._tokenizers[ModelTarget.TINYLLAMA] = tok_cpu
        self._devices[ModelTarget.TINYLLAMA] = self.cpu_device
        self._load_times[ModelTarget.TINYLLAMA] = cpu_time
        print(f"  [TinyLlama] CPU: {cpu_time}s")

        # Load Qwen on GPU
        t0 = time.time()
        model_gpu, tok_gpu = self._load_model(
            ModelTarget.QWEN_CODER, device=self.gpu_device)
        gpu_time = round(time.time() - t0, 1)
        self._models[ModelTarget.QWEN_CODER] = model_gpu
        self._tokenizers[ModelTarget.QWEN_CODER] = tok_gpu
        self._devices[ModelTarget.QWEN_CODER] = self.gpu_device
        self._load_times[ModelTarget.QWEN_CODER] = gpu_time
        vram = memory_allocated(self.gpu_device)
        print(f"  [Qwen] GPU: {gpu_time}s, VRAM={vram:.0f} MB")

        # Default to Qwen (GPU coder)
        self.active_target = ModelTarget.QWEN_CODER
        self.model = self._models[ModelTarget.QWEN_CODER]
        self.tokenizer = self._tokenizers[ModelTarget.QWEN_CODER]

    def _load_model(self, target: ModelTarget, device: str):
        """Load a model with HelixLinear onto the specified device."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = MODEL_CONFIGS[target]
        cdna_dir = cfg["model_dir"] / "cdnav3"

        if not cdna_dir.exists():
            raise FileNotFoundError(f"CDNA dir not found: {cdna_dir}")

        model = AutoModelForCausalLM.from_pretrained(
            str(cfg["model_dir"]), dtype=torch.float32
        )
        model.eval()

        # Load HelixLinear modules
        helix_modules = {}
        for block_idx in range(cfg["n_blocks"]):
            for tensor_type in TENSOR_TYPES:
                hf_name = HF_PATTERNS[tensor_type].format(i=block_idx)
                safe_name = hf_name.replace("/", "_").replace(".", "_")
                tensor_dir = cdna_dir / f"{safe_name}.cdnav3"
                if not tensor_dir.exists():
                    continue
                module_path = hf_name.replace(".weight", "")
                parts = module_path.split(".")
                mod = model
                for p in parts:
                    mod = getattr(mod, p)
                bias = mod.bias.data.clone() if mod.bias is not None else None
                helix_mod = load_helix_linear_from_cdnav3(tensor_dir, bias=bias)
                helix_modules[module_path] = helix_mod

        model = swap_to_helix(model, helix_modules)

        # Move to target device
        if device != "cpu":
            reset_peak_memory(device)
        model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(str(cfg["model_dir"]))
        return model, tokenizer

    def ensure_model(self, target: ModelTarget) -> Tuple[torch.nn.Module, object]:
        """Switch to target model. No GPU swap — just pointer switch."""
        if self.active_target == target and self.model is not None:
            return self.model, self.tokenizer

        t0 = time.perf_counter()
        self.model = self._models[target]
        self.tokenizer = self._tokenizers[target]
        self.active_target = target
        switch_us = (time.perf_counter() - t0) * 1e6

        self._swap_count += 1
        self._swap_history.append({
            "target": target.value,
            "device": self._devices[target],
            "switch_us": round(switch_us, 1),
        })

        return self.model, self.tokenizer

    def get_device(self, target: ModelTarget = None) -> str:
        """Return the device for a model target (or current active)."""
        if target is None:
            target = self.active_target
        return self._devices.get(target, self.cpu_device)

    def status(self) -> dict:
        vram = memory_allocated(self.gpu_device)
        return {
            "active_model": self.active_target.value if self.active_target else None,
            "vram_mb": round(vram),
            "swap_count": self._swap_count,
            "mode": "split_runtime",
            "models_loaded": {
                t.value: self._devices[t] for t in self._models
            },
            "load_times": {t.value: s for t, s in self._load_times.items()},
            "last_swap": self._swap_history[-1] if self._swap_history else None,
        }

    def cleanup(self):
        """Free all models."""
        for target in list(self._models):
            del self._models[target]
        self._models.clear()
        gc.collect()
        empty_cache(self.gpu_device)
