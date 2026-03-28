#!/usr/bin/env python3
"""
Universal CDNA v3 compressor.

Point at any model directory. It reads config.json, discovers all weight
tensors from safetensors, classifies each one, routes by kurtosis, and
encodes via CDNAv3Writer. No hardcoded block counts, no per-model scripts,
no architecture-specific tensor patterns.

The model is the environment. The pipeline is the organism. The config is
the genome. Everything discovered, nothing hardcoded.

Usage:
    python3 tools/compress.py ~/models/tinyllama_fp32
    python3 tools/compress.py ~/models/qwen2.5-7b-instruct
    python3 tools/compress.py ~/models/mamba-130m-hf
    python3 tools/compress.py ~/models/anything-new --dry-run

Handles:
  - Single-file safetensors (model.safetensors)
  - Sharded safetensors (model.safetensors.index.json)
  - pytorch_model.bin (auto-converts to numpy)
  - Any architecture: transformer, SSM, CNN, encoder-only, decoder-only
  - Resume: skips already-compressed tensors
  - Tied weights: detects lm_head == embed_tokens, compresses once

Output:
  {model_dir}/cdnav3/         .cdnav3/ subdirectories
  {model_dir}/cdnav3/manifest.json
  receipts/compress/{model_name}_{timestamp}.json
"""

import argparse
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.tensor_policy import (
    classify_tensor,
    get_policy,
    TensorClass,
    get_default_policy,
)


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def read_model_config(model_dir: Path) -> dict:
    """Read config.json and extract everything we need."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        raw = json.load(f)

    # Extract fields with fallback keys (different architectures use different names)
    n_layers = (
        raw.get("num_hidden_layers")
        or raw.get("n_layer")
        or raw.get("num_layers")
        or raw.get("n_layers")
    )

    model_type = raw.get("model_type", "unknown")
    architectures = raw.get("architectures", [])

    # Model name: prefer _name_or_path, fall back to directory name
    model_name = raw.get("_name_or_path", model_dir.name)
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    return {
        "n_layers": n_layers,
        "model_type": model_type,
        "architectures": architectures,
        "model_name": model_name,
        "vocab_size": raw.get("vocab_size"),
        "hidden_size": raw.get("hidden_size") or raw.get("d_model"),
        "tie_word_embeddings": raw.get("tie_word_embeddings", False),
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Weight source abstraction
# ---------------------------------------------------------------------------

class WeightSource:
    """Unified interface to read tensors from any model format."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._mode = None
        self._sf_handles = {}
        self._weight_map = {}
        self._pt_state = None
        self._discover()

    def _discover(self):
        """Discover which weight format is available."""
        # Priority 1: single safetensors
        single = self.model_dir / "model.safetensors"
        if single.exists():
            self._mode = "single_sf"
            return

        # Priority 2: sharded safetensors
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            self._weight_map = data.get("weight_map", {})
            # Verify all shards exist
            for shard_file in set(self._weight_map.values()):
                if not (self.model_dir / shard_file).exists():
                    raise FileNotFoundError(f"Missing shard: {shard_file}")
            self._mode = "sharded_sf"
            return

        # Priority 3: pytorch_model.bin
        pt_path = self.model_dir / "pytorch_model.bin"
        if pt_path.exists():
            import torch
            self._pt_state = torch.load(pt_path, map_location="cpu", weights_only=True)
            self._mode = "pytorch"
            return

        raise FileNotFoundError(
            f"No model weights found in {self.model_dir}. "
            f"Expected model.safetensors, sharded safetensors, or pytorch_model.bin"
        )

    def tensor_names(self) -> list[str]:
        """List all tensor names in the model."""
        from safetensors import safe_open

        if self._mode == "single_sf":
            sf = self._get_sf_handle("model.safetensors")
            return list(sf.keys())
        elif self._mode == "sharded_sf":
            return list(self._weight_map.keys())
        elif self._mode == "pytorch":
            return list(self._pt_state.keys())
        return []

    def get_tensor(self, name: str) -> np.ndarray:
        """Load a single tensor as float32 numpy array."""
        if self._mode == "single_sf":
            sf = self._get_sf_handle("model.safetensors")
            return sf.get_tensor(name).float().numpy()
        elif self._mode == "sharded_sf":
            shard_file = self._weight_map[name]
            sf = self._get_sf_handle(shard_file)
            return sf.get_tensor(name).float().numpy()
        elif self._mode == "pytorch":
            return self._pt_state[name].float().numpy()
        raise RuntimeError(f"Unknown mode: {self._mode}")

    def get_shape(self, name: str) -> tuple:
        """Get tensor shape without loading full data."""
        if self._mode == "pytorch":
            return tuple(self._pt_state[name].shape)
        # Use torch framework for safetensors (handles bf16/fp16 natively)
        if self._mode == "single_sf":
            sf = self._get_sf_handle("model.safetensors")
        else:
            shard_file = self._weight_map[name]
            sf = self._get_sf_handle(shard_file)
        t = sf.get_tensor(name)
        return tuple(t.shape)

    def _get_sf_handle(self, filename: str):
        from safetensors import safe_open
        if filename not in self._sf_handles:
            # Use PyTorch framework to handle bf16/fp16 (numpy lacks bf16)
            self._sf_handles[filename] = safe_open(
                str(self.model_dir / filename), framework="pt"
            )
        return self._sf_handles[filename]

    def close(self):
        self._sf_handles.clear()
        self._pt_state = None

    @property
    def mode(self) -> str:
        return self._mode


# ---------------------------------------------------------------------------
# Kurtosis computation (no scipy dependency)
# ---------------------------------------------------------------------------

def kurtosis_1d(data: np.ndarray) -> float:
    """Fisher excess kurtosis. Matches scipy.stats.kurtosis(fisher=True)."""
    mu = data.mean()
    var = ((data - mu) ** 2).mean()
    if var < 1e-30:
        return 0.0
    m4 = ((data - mu) ** 4).mean()
    return float(m4 / (var ** 2) - 3.0)


# ---------------------------------------------------------------------------
# Tensor filtering
# ---------------------------------------------------------------------------

def should_compress(name: str, shape: tuple, config: dict) -> str:
    """
    Decide what to do with a tensor.

    Returns:
        "compress" — VQ encode this tensor
        "exact"    — store as-is (1D norms, tiny tensors)
        "skip"     — don't store (tied duplicate, non-weight)
    """
    # Biases: save exact (critical for models like Qwen2.5 with attention biases)
    if name.endswith(".bias"):
        if len(shape) == 1:
            return "exact"
        return "skip"

    # Skip other non-weight tensors
    if not name.endswith(".weight") and not name.endswith("_weight"):
        # Some models have A_log, D, etc. (Mamba state params)
        # Compress if 2D, skip if 1D
        if len(shape) == 1:
            return "skip"
        if len(shape) == 2:
            return "compress"
        return "skip"

    # 1D tensors: norms, biases → store exact
    if len(shape) == 1:
        return "exact"

    # Embedding tables and output heads: store exact, not VQ.
    # Embeddings are lookup tables (nn.Embedding), not linear projections —
    # VQ with k=256 on 150K+ vocab rows destroys token identity.
    # Output heads (lm_head) project to vocab-sized logits — VQ distortion
    # here corrupts every token probability.
    # Cost: ~1-2 GB at FP16 for 150K vocab — negligible vs PPL impact.
    if "embed_tokens" in name or "embed_positions" in name or "wte" in name or "wpe" in name or "backbone.embedding" in name:
        return "exact"
    if "lm_head" in name:
        return "exact"

    # 2D tensors: the main event
    if len(shape) == 2:
        # Tiny tensors (< 256 elements): not worth quantizing
        if shape[0] * shape[1] < 256:
            return "exact"
        return "compress"

    # conv1d weights: tiny 3D kernels (e.g., 4352×1×4 in Mamba).
    # VQ-256 is too coarse for these high-kurtosis tensors (~48.6).
    # Cost of storing exact: ~650 KB total. PPL impact of compressing: +~1% delta.
    if "conv1d" in name and len(shape) == 3:
        return "exact"

    # 3D+ tensors: rare, but reshape to 2D and compress
    if len(shape) >= 2:
        return "compress"

    return "skip"


def detect_tied_weights(names: list[str], source: WeightSource) -> set[str]:
    """
    Detect tied weights (e.g., lm_head.weight == model.embed_tokens.weight).

    Returns set of tensor names that are duplicates (should be skipped).
    """
    # Common tied pairs
    embed_names = [n for n in names if "embed" in n.lower() and n.endswith(".weight")]
    head_names = [n for n in names if "lm_head" in n.lower() and n.endswith(".weight")]

    tied = set()
    if embed_names and head_names:
        # Check if they share the same data
        for ename in embed_names:
            for hname in head_names:
                try:
                    e_shape = source.get_shape(ename)
                    h_shape = source.get_shape(hname)
                    if e_shape == h_shape:
                        # Same shape — likely tied. Keep embedding, skip lm_head.
                        tied.add(hname)
                except Exception:
                    pass
    return tied


# ---------------------------------------------------------------------------
# Main compression pipeline
# ---------------------------------------------------------------------------

def compress_model(model_dir: Path, dry_run: bool = False, force: bool = False,
                   k_override: int = None, adaptive: bool = False,
                   quality_target: float = 0.998,
                   scale_file: Path = None, policy_file: Path = None,
                   force_svd_layers: set = None,
                   force_svd_rank: int = 8,
                   force_k_layers: dict = None,
                   k_map_file: Path = None):
    """
    Compress a model to CDNA v3 format.

    Modes:
      --k N:        Fixed codebook size (output: cdnav3_k{N}/)
      --adaptive:   Per-tensor quality-driven k selection (output: cdnav3_adaptive/)
                    Starts at k=64, escalates to 128→256→256+SVD until
                    cosine >= quality_target. Logs final k per tensor.
      (default):    k=256, standard mode (output: cdnav3/)
    """
    import re
    from dataclasses import replace as dc_replace
    import shutil

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    model_dir = Path(model_dir).expanduser().resolve()
    assert model_dir.is_dir(), f"Not a directory: {model_dir}"

    # ── Phase 1: Discover environment ──
    config = read_model_config(model_dir)
    model_name = config.get("model_name", model_dir.name)
    n_layers = config.get("n_layers")
    model_type = config.get("model_type", "unknown")
    tied = config.get("tie_word_embeddings", False)

    # Adaptive k escalation ladder
    K_LADDER = [64, 128, 256, 512]

    print("=" * 70)
    print(f"  CDNA v3 Universal Compressor")
    print("=" * 70)
    print(f"  Model:       {model_name}")
    print(f"  Type:        {model_type}")
    print(f"  Directory:   {model_dir}")
    print(f"  Layers:      {n_layers or '(not in config)'}")
    print(f"  Tied embeds: {tied}")
    # Load activation scale factors if provided
    act_scales = {}
    if scale_file is not None:
        scale_file = Path(scale_file).expanduser().resolve()
        assert scale_file.exists(), f"Scale file not found: {scale_file}"
        loaded = np.load(scale_file)
        act_scales = {k: loaded[k] for k in loaded.files}
        print(f"  Scaling:     AWQ-style channel scaling ({len(act_scales)} tensors)")
        print(f"  Scale file:  {scale_file}")

    # Load per-tensor dynamic policy if provided
    dynamic_policies = {}
    if policy_file is not None:
        policy_file = Path(policy_file).expanduser().resolve()
        assert policy_file.exists(), f"Policy file not found: {policy_file}"
        dynamic_policies = json.loads(policy_file.read_text())
        print(f"  Policy:      Dynamic per-tensor ({len(dynamic_policies)} tensors)")
        print(f"  Policy file: {policy_file}")

    # Load per-tensor k-map if provided (from k_allocator.py)
    k_map = {}  # tensor_name -> k_value
    if k_map_file is not None:
        k_map_file = Path(k_map_file).expanduser().resolve()
        assert k_map_file.exists(), f"k-map file not found: {k_map_file}"
        k_map_data = json.loads(k_map_file.read_text())
        k_map = k_map_data.get("overrides", {})
        k_map_default = k_map_data.get("k_default", 256)
        print(f"  k-map:       {len(k_map)} tensor overrides (default k={k_map_default})")
        print(f"  k-map file:  {k_map_file}")

    if adaptive:
        print(f"  Mode:        ADAPTIVE (quality target: {quality_target})")
        print(f"  k ladder:    {K_LADDER} → +SVD if needed")
    elif k_override:
        print(f"  Mode:        FIXED k={k_override} (info-theoretic {32.0 / np.log2(k_override):.1f}x)")
    else:
        print(f"  Mode:        STANDARD k=256")
    if force_svd_layers:
        print(f"  Force SVD:   layers {sorted(force_svd_layers)} → rank={force_svd_rank}")
    if force_k_layers:
        layer_ids = sorted(force_k_layers.keys())
        k_val = next(iter(force_k_layers.values()))
        print(f"  Force k:     layers {layer_ids} → k={k_val}")

    # Open weight source
    source = WeightSource(model_dir)
    print(f"  Weight fmt:  {source.mode}")

    # Discover all tensors
    all_names = source.tensor_names()
    print(f"  Total keys:  {len(all_names)}")

    # Detect tied weights
    tied_names = detect_tied_weights(all_names, source) if tied else set()
    if tied_names:
        print(f"  Tied (skip): {', '.join(sorted(tied_names))}")

    # Classify and filter
    plan = []  # List of (name, shape, action)
    for name in sorted(all_names):
        if name in tied_names:
            continue
        shape = source.get_shape(name)
        action = should_compress(name, shape, config)
        if action != "skip":
            plan.append((name, shape, action))

    n_compress = sum(1 for _, _, a in plan if a == "compress")
    n_exact = sum(1 for _, _, a in plan if a == "exact")

    print(f"\n  Compression plan:")
    print(f"    Compress (VQ):  {n_compress} tensors")
    print(f"    Exact (store):  {n_exact} tensors")
    print(f"    Skip (tied/1D): {len(all_names) - len(plan) - len(tied_names)} tensors")

    if dry_run:
        print(f"\n  [DRY RUN] Would compress {n_compress} tensors. Exiting.")
        for name, shape, action in plan:
            tc = classify_tensor(name, shape=shape)
            print(f"    {action:8s}  {tc.value:15s}  {str(shape):>20s}  {name}")
        return

    # ── Phase 2: Encode ──
    if adaptive:
        cdna_dir = model_dir / "cdnav3_adaptive"
    elif k_override and k_override != 256:
        cdna_dir = model_dir / f"cdnav3_k{k_override}"
    else:
        cdna_dir = model_dir / "cdnav3"
    cdna_dir.mkdir(parents=True, exist_ok=True)
    writer = CDNAv3Writer(cdna_dir)

    print(f"\n  Output: {cdna_dir}")
    print(f"  Encoding {n_compress} tensors...\n")

    stats_all = []
    k_distribution = {}  # k_value → count (adaptive mode tracking)
    n_done = 0
    n_cached = 0
    total_dense = 0
    total_compressed = 0
    n_svd = 0
    n_escalated = 0  # tensors that needed k > 64 in adaptive mode

    for name, shape, action in plan:
        if action == "exact":
            safe_name = name.replace("/", "_").replace(".", "_")
            exact_path = cdna_dir / f"{safe_name}.npy"
            if not exact_path.exists() or force:
                tensor_np = source.get_tensor(name)
                np.save(exact_path, tensor_np)
                del tensor_np
            # Write meta.json companion (needed for standalone bias loading)
            meta_path = cdna_dir / f"{safe_name}.npy.meta.json"
            if not meta_path.exists() or force:
                meta_path.write_text(json.dumps({
                    "tensor_name": name,
                    "action": "exact",
                    "format": "npy",
                }))
            continue

        # action == "compress"
        safe_name = name.replace("/", "_").replace(".", "_")
        tensor_dir = cdna_dir / f"{safe_name}.cdnav3"

        # Parse block index from name early (needed for recompression check + policy routing)
        block_match = re.search(r'layers?\.(\d+)', name)
        block_idx = int(block_match.group(1)) if block_match else None

        # Resume: skip if already compressed (unless force or layer override needs recompression)
        needs_recompress = False
        if force_k_layers and block_idx is not None and block_idx in force_k_layers:
            # Check if existing compression used a different k
            stats_path = tensor_dir / "stats.json"
            if stats_path.exists():
                s = json.loads(stats_path.read_text())
                existing_k = s.get("chosen_k", 256)
                if isinstance(existing_k, str):
                    existing_k = int(existing_k.split("+")[0])
                if existing_k != force_k_layers[block_idx]:
                    needs_recompress = True
            else:
                needs_recompress = True
        if force_svd_layers and block_idx is not None and block_idx in force_svd_layers:
            # Check if existing compression used a different SVD rank
            stats_path = tensor_dir / "stats.json"
            if stats_path.exists():
                s = json.loads(stats_path.read_text())
                existing_svd = s.get("svd_residual_rank", 0)
                if existing_svd != force_svd_rank:
                    needs_recompress = True
        # k-map override: recompress if tensor has a different k than requested
        k_map_k = k_map.get(name)
        if k_map_k is not None:
            meta_path = tensor_dir / "meta.json"
            if meta_path.exists():
                m = json.loads(meta_path.read_text())
                if m.get("n_clusters", 256) != k_map_k:
                    needs_recompress = True
            else:
                needs_recompress = True

        if tensor_dir.exists() and (tensor_dir / "codebook.npy").exists() and not force and not needs_recompress:
            n_cached += 1
            n_done += 1
            stats_path = tensor_dir / "stats.json"
            if stats_path.exists():
                s = json.loads(stats_path.read_text())
                total_dense += s.get("original_bytes", 0)
                total_compressed += s.get("compressed_bytes", 0)
                # Track k for cached tensors too
                cached_k = s.get("chosen_k", k_override or 256)
                k_distribution[cached_k] = k_distribution.get(cached_k, 0) + 1
            continue

        # Clean stale files from previous compression runs (e.g., SVD files from
        # a run with channel scaling, or channel_scales from a run with --scale-file).
        # Without this, leftover files from a different compression mode corrupt
        # inference: HelixLinear loads SVD/scales that don't match the current codebook.
        if tensor_dir.exists():
            stale_files = ["svd_U.npy", "svd_s.npy", "svd_Vt.npy", "channel_scales.npy"]
            for sf in stale_files:
                stale_path = tensor_dir / sf
                if stale_path.exists():
                    stale_path.unlink()

        # Load tensor
        tensor_np = source.get_tensor(name)

        # Ensure 2D for cdnav3_writer
        orig_shape = tensor_np.shape
        if tensor_np.ndim == 1:
            tensor_np = tensor_np.reshape(1, -1)
        elif tensor_np.ndim > 2:
            tensor_np = tensor_np.reshape(-1, tensor_np.shape[-1])

        # AWQ-style channel scaling: scale columns by activation magnitude
        # before k-means. Codebook fits the scaled distribution, giving more
        # resolution to important channels. Scale factors stored alongside
        # codebook; inference pre-scales input x by 1/scale.
        channel_scales = None
        if name in act_scales:
            channel_scales = act_scales[name]
            # Ensure shape matches tensor columns
            if channel_scales.shape[0] == tensor_np.shape[1]:
                tensor_np = tensor_np * channel_scales[np.newaxis, :]
            else:
                print(f"    WARNING: scale shape mismatch for {name}: "
                      f"{channel_scales.shape} vs cols={tensor_np.shape[1]}, skipping scaling")
                channel_scales = None

        # Dynamic policy channel scaling (overrides --scale-file for this tensor)
        dp_entry = dynamic_policies.get(name)
        if dp_entry is not None and channel_scales is None and "act_scales" in dp_entry:
            cal_scales = np.array(dp_entry["act_scales"], dtype=np.float32)
            if cal_scales.shape[0] == tensor_np.shape[1]:
                channel_scales = cal_scales
                tensor_np = tensor_np * channel_scales[np.newaxis, :]
            else:
                print(f"    WARNING: policy act_scales shape mismatch for {name}: "
                      f"{cal_scales.shape} vs cols={tensor_np.shape[1]}")

        # Classify + route
        tc = classify_tensor(name, shape=orig_shape)
        kurt = kurtosis_1d(tensor_np.ravel())

        base_policy = get_policy(
            name, orig_shape,
            block_idx=block_idx,
            kurtosis=kurt,
            n_blocks=n_layers,
        )

        # Dynamic policy overrides (per-tensor k, SVD, sidecar from --policy-file)
        if dp_entry is not None:
            dp_pol = dp_entry.get("policy", {})
            overrides = {}
            if "k" in dp_pol:
                overrides["n_clusters"] = dp_pol["k"]
            if dp_pol.get("svd", False):
                overrides["svd_residual_rank"] = dp_pol.get("svd_rank", 8)
            elif "svd" in dp_pol and not dp_pol["svd"]:
                overrides["svd_residual_rank"] = 0
            if "sidecar_top_k" in dp_pol:
                overrides["max_corrections"] = dp_pol["sidecar_top_k"]
            if overrides:
                base_policy = dc_replace(base_policy, **overrides)

        # Force SVD on specified layers (--force-svd-layers + --force-svd-rank)
        if force_svd_layers and block_idx is not None and block_idx in force_svd_layers:
            base_policy = dc_replace(base_policy, svd_residual_rank=force_svd_rank)

        # Force higher k on specified layers (--force-k-layers)
        layer_k_override = None
        if force_k_layers and block_idx is not None and block_idx in force_k_layers:
            layer_k_override = force_k_layers[block_idx]
            base_policy = dc_replace(base_policy, n_clusters=layer_k_override)

        if adaptive:
            # ── Adaptive k escalation ──
            # Try each k in the ladder. Accept the first that meets quality_target.
            # If none meet it, try k=256 + SVD as final fallback.
            #
            # NOTE: SVD routing disabled (2026-03-28). base_policy.svd_residual_rank
            # is always 0 now. Crossover test proved SVD adds zero value at all scales.
            chosen_k = None
            stats = None
            base_svd_rank = base_policy.svd_residual_rank

            for try_k in K_LADDER:
                policy = dc_replace(base_policy, n_clusters=try_k)

                # Clean previous attempt's output
                if tensor_dir.exists():
                    shutil.rmtree(tensor_dir)

                stats = writer.write_tensor(tensor_np, name, policy=policy)

                # Check quality — use best metric available
                cos = stats.get("cosine_with_svd",
                      stats.get("cosine_with_sidecar",
                      stats.get("cosine_no_sidecar", 0)))

                if cos >= quality_target:
                    chosen_k = try_k
                    if base_svd_rank > 0:
                        chosen_k = f"{try_k}+SVD"
                    break

            # Final fallback: k=256 + SVD (force SVD even if base didn't have it)
            if chosen_k is None:
                policy = dc_replace(base_policy, n_clusters=256, svd_residual_rank=max(8, base_svd_rank))
                if tensor_dir.exists():
                    shutil.rmtree(tensor_dir)
                stats = writer.write_tensor(tensor_np, name, policy=policy)
                chosen_k = "256+SVD"

            # Record chosen k in stats.json for resume tracking
            stats["chosen_k"] = chosen_k
            stats_json_path = tensor_dir / "stats.json"
            if stats_json_path.exists():
                s = json.loads(stats_json_path.read_text())
                s["chosen_k"] = chosen_k
                stats_json_path.write_text(json.dumps(s, indent=2))

            k_distribution[chosen_k] = k_distribution.get(chosen_k, 0) + 1
            if chosen_k != K_LADDER[0]:
                n_escalated += 1

            # Save channel scale factors (adaptive path)
            if channel_scales is not None:
                scale_path = tensor_dir / "channel_scales.npy"
                np.save(scale_path, channel_scales.astype(np.float32))

        else:
            # ── Fixed k mode ──
            policy = base_policy

            # k-map override: per-tensor k from k_allocator.py output
            k_map_k = k_map.get(name)
            if k_map_k is not None:
                policy = dc_replace(policy, n_clusters=k_map_k)
            elif layer_k_override is None and k_override and dp_entry is None:
                policy = dc_replace(policy, n_clusters=k_override)

            stats = writer.write_tensor(tensor_np, name, policy=policy)

            # Hessian sidecar post-processing (replaces percentile sidecar)
            if (dp_entry is not None
                    and dp_entry.get("policy", {}).get("sidecar_mode") == "hessian"
                    and "hessian_diag" in dp_entry):
                _replace_sidecar_hessian(
                    tensor_dir, tensor_np,
                    np.array(dp_entry["hessian_diag"], dtype=np.float32),
                    top_k=dp_entry.get("policy", {}).get("sidecar_top_k"),
                )
                # Reload stats after sidecar replacement
                stats_path = tensor_dir / "stats.json"
                if stats_path.exists():
                    stats = json.loads(stats_path.read_text())

            chosen_k = k_map_k or layer_k_override or k_override or 256

        # Save channel scale factors alongside the tensor artifacts
        if channel_scales is not None:
            scale_path = tensor_dir / "channel_scales.npy"
            np.save(scale_path, channel_scales.astype(np.float32))

        stats_all.append(stats)

        n_done += 1
        total_dense += stats.get("original_bytes", 0)
        total_compressed += stats.get("compressed_bytes", 0)
        if stats.get("svd_bytes", 0) > 0:
            n_svd += 1

        del tensor_np

        # Progress
        cos_val = stats.get("cosine_with_svd", stats.get("cosine_with_sidecar", 0))
        ratio_so_far = total_dense / max(1, total_compressed)
        k_label = f"k={chosen_k}" if not adaptive else f"k={chosen_k:>7s}" if isinstance(chosen_k, str) else f"k={chosen_k:>3d}    "
        if n_done % 5 == 0 or n_done == n_compress or (adaptive and chosen_k != K_LADDER[0]):
            print(f"    {n_done:4d}/{n_compress}  "
                  f"cos={cos_val:.4f}  "
                  f"kurt={kurt:6.1f}  "
                  f"{k_label}  "
                  f"ratio={ratio_so_far:.2f}x  "
                  f"{tc.value:15s}  {name}",
                  flush=True)

    source.close()

    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start
    ratio = round(total_dense / max(1, total_compressed), 2)

    # ── Phase 3: Manifest + Receipt ──
    print(f"\n{'=' * 70}")
    print(f"  Complete: {n_done} tensors ({n_cached} cached)")
    print(f"  SVD sidecars: {n_svd} tensors (kurtosis-routed)")
    if adaptive:
        print(f"  Adaptive k distribution:")
        for k_val in sorted(k_distribution.keys(), key=lambda x: str(x)):
            count = k_distribution[k_val]
            pct = 100 * count / max(1, n_done)
            print(f"    k={k_val}: {count} tensors ({pct:.0f}%)")
        print(f"  Escalated: {n_escalated}/{n_done} tensors needed k > {K_LADDER[0]}")
    if total_dense > 0:
        print(f"  Dense: {total_dense / 1e9:.3f} GB -> Compressed: {total_compressed / 1e9:.3f} GB ({ratio}x)")
    print(f"  Time: {wall:.0f}s wall, {cpu:.0f}s CPU")
    print(f"{'=' * 70}")

    manifest = {
        "model": model_name,
        "model_type": model_type,
        "architectures": config.get("architectures", []),
        "n_layers": n_layers,
        "mode": "adaptive" if adaptive else "fixed",
        "k": "adaptive" if adaptive else (k_override or 256),
        "quality_target": quality_target if adaptive else None,
        "k_distribution": {str(k): v for k, v in k_distribution.items()} if adaptive else None,
        "n_tensors_compressed": n_done,
        "n_tensors_cached": n_cached,
        "n_tensors_exact": n_exact,
        "n_svd_routed": n_svd,
        "n_escalated": n_escalated if adaptive else 0,
        "total_dense_bytes": int(total_dense),
        "total_compressed_bytes": int(total_compressed),
        "compression_ratio": ratio,
        "compressor": "compress.py (universal, adaptive)" if adaptive else "compress.py (universal)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = cdna_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest: {manifest_path}")

    # Receipt
    receipt = {
        "work_order": f"compress-{model_name}" + ("-adaptive" if adaptive else ""),
        "question": f"Does {model_name} compress through CDNA v3"
                    + (f" with adaptive k (target cos>={quality_target})?" if adaptive
                       else "?"),
        "verdict": "PASS" if n_done > 0 else "EMPTY",
        "manifest": manifest,
        "cost": {
            "wall_time_s": round(wall, 3),
            "cpu_time_s": round(cpu, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipts_dir = Path(__file__).parent.parent / "receipts" / "compress"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_").replace(" ", "_").lower()
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"{safe_model}_{ts}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"  Receipt:  {receipt_path}")


# ---------------------------------------------------------------------------
# Hessian sidecar post-processing
# ---------------------------------------------------------------------------

def _replace_sidecar_hessian(tensor_dir: Path, original: np.ndarray,
                              hessian_diag: np.ndarray, top_k: int = None):
    """Replace percentile sidecar with hessian-sensitivity-weighted sidecar."""
    from helix_substrate.generate_sidecars_v3 import find_outliers_hessian, write_sidecar_npz

    codebook = np.load(tensor_dir / "codebook.npy")
    # Read index dtype from meta.json (uint16 for k>256, default uint8)
    meta_path = tensor_dir / "meta.json"
    idx_dtype = np.uint8
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("index_dtype") == "uint16":
            idx_dtype = np.uint16
    indices = np.fromfile(tensor_dir / "indices.bin", dtype=idx_dtype)
    quantized = codebook[indices].reshape(original.shape)

    positions, values = find_outliers_hessian(original, quantized, hessian_diag, top_k=top_k)

    if len(positions) > 0:
        sidecar_path = tensor_dir / "sidecar.npz"
        write_sidecar_npz(positions, values, sidecar_path)

        # Update stats.json
        stats_path = tensor_dir / "stats.json"
        if stats_path.exists():
            s = json.loads(stats_path.read_text())
            s["num_outliers"] = len(positions)
            s["sidecar_bytes"] = int(sidecar_path.stat().st_size)
            s["sidecar_mode"] = "hessian"
            # Recompute cosine_with_sidecar
            patched = codebook[indices].copy()
            patched[positions] = values
            flat_orig = original.ravel().astype(np.float32)
            patched = patched.astype(np.float32)
            dot = np.dot(flat_orig, patched)
            norm_a = np.linalg.norm(flat_orig)
            norm_b = np.linalg.norm(patched)
            cos = float(dot / max(norm_a * norm_b, 1e-12))
            s["cosine_with_sidecar"] = round(cos, 6)
            stats_path.write_text(json.dumps(s, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Universal CDNA v3 compressor. Point at any model directory.",
        epilog="Examples:\n"
               "  python3 tools/compress.py ~/models/tinyllama_fp32\n"
               "  python3 tools/compress.py ~/models/qwen2.5-7b-instruct --adaptive\n"
               "  python3 tools/compress.py ~/models/qwen2.5-7b-instruct --k 64\n"
               "  python3 tools/compress.py ~/models/anything --adaptive --quality-target 0.999\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_dir", type=Path, help="Path to model directory (must contain config.json + weights)")
    parser.add_argument("--dry-run", action="store_true", help="Show compression plan without encoding")
    parser.add_argument("--force", action="store_true", help="Re-compress even if cached")
    parser.add_argument("--k", type=int, default=None,
                        help="Fixed codebook size (e.g. --k 64 for 6-bit). "
                             "Mutually exclusive with --adaptive. Output: cdnav3_k{K}/")
    parser.add_argument("--adaptive", action="store_true",
                        help="Adaptive k selection per tensor. Starts at k=64, escalates "
                             "to 128/256/256+SVD until cosine >= quality-target. "
                             "Output: cdnav3_adaptive/")
    parser.add_argument("--quality-target", type=float, default=0.998,
                        help="Cosine similarity threshold for adaptive mode (default: 0.998)")
    parser.add_argument("--scale-file", type=Path, default=None,
                        help="Path to act_scales.npz from calibrate.py. "
                             "Enables AWQ-style channel scaling before k-means.")
    parser.add_argument("--force-svd-layers", type=str, default=None,
                        help="Force SVD on these layer indices (e.g. '23-27' or '23,24,25,26,27'). "
                             "Overrides kurtosis routing for specified layers.")
    parser.add_argument("--force-svd-rank", type=int, default=8,
                        help="SVD rank to use with --force-svd-layers (default: 8). "
                             "Higher ranks (16, 32) capture more residual error.")
    parser.add_argument("--force-k-layers", type=str, default=None,
                        help="Force higher k on specific layer indices. "
                             "Format: 'LAYERS:K' e.g. '23-27:512' or '23,24,25,26,27:1024'. "
                             "NOTE: k>256 requires uint16 indices (not yet supported). "
                             "Overrides default k=256 for specified layers only.")
    parser.add_argument("--policy-file", type=Path, default=None,
                        help="Path to dynamic_policy.json from calibrate_dynamic.py. "
                             "Per-tensor compression recipes (k, SVD, sidecar mode).")
    parser.add_argument("--k-map", type=Path, default=None, dest="k_map",
                        help="Path to k_map.json from k_allocator.py. "
                             "Per-tensor codebook size assignments (supports k>256 via uint16 indices). "
                             "Format: {\"overrides\": {\"tensor.name.weight\": 512, ...}, \"k_default\": 256}")
    args = parser.parse_args()

    if args.adaptive and args.k:
        parser.error("--adaptive and --k are mutually exclusive")

    # Parse --force-svd-layers
    force_svd_layers = set()
    if args.force_svd_layers:
        for part in args.force_svd_layers.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                force_svd_layers.update(range(int(lo), int(hi) + 1))
            else:
                force_svd_layers.add(int(part))

    # Parse --force-k-layers (format: "LAYERS:K" e.g. "23-27:512")
    force_k_layers = {}  # {block_idx: k_value}
    if args.force_k_layers:
        layers_part, k_part = args.force_k_layers.rsplit(":", 1)
        k_val = int(k_part)
        for part in layers_part.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                for idx in range(int(lo), int(hi) + 1):
                    force_k_layers[idx] = k_val
            else:
                force_k_layers[int(part)] = k_val

    compress_model(args.model_dir, dry_run=args.dry_run, force=args.force,
                   k_override=args.k, adaptive=args.adaptive,
                   quality_target=args.quality_target,
                   scale_file=args.scale_file,
                   policy_file=args.policy_file,
                   force_svd_layers=force_svd_layers,
                   force_svd_rank=args.force_svd_rank,
                   force_k_layers=force_k_layers,
                   k_map_file=args.k_map)


if __name__ == "__main__":
    main()
