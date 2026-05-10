#!/usr/bin/env python3
"""
Fastest Falsifier — Live Sidecar Norm Signal (WO-FALSIFIER-LIVE-SIDECAR-NORM-01)

Confirms that the sidecar L2 norm signal measured by probe_v3b_aggregate.py
(ρ = 0.574 on static weights) still holds under *live* HXQ decoding.

Math MUST match probe_v3b_aggregate.py exactly:
    - post-forward hook
    - per-token: sc_out = scatter_add(x[:, cols] * deltas over rows); sc_norm = ||sc_out||
    - per-chunk: mean of per-token sc_norms over all tokens in the chunk
    - per-layer: array of per-chunk means
    - headline:  spearman(mean-across-top-layers, chunk_losses)

Branch gate:
    GREEN        0.4 ≤ ρ ≤ 0.6      → proceed to Layer 1 integration
    AMBER_LOW    0.3 ≤ ρ < 0.4      → borderline; investigate
    AMBER_HIGH   ρ > 0.6            → too good; audit for leakage
    RED          ρ < 0.3            → runtime signal dead; negative receipt
    ERROR        pipeline failure

Critical preflight:
    T2000 has 4 GB. Probe needs ~1.2–1.5 GB. Script exits 2 if free < 2000 MB.

Reference:
    /home/voidstr3m33/mamba-scan-lite/tools/probe_v3b_aggregate.py
    Reference result: ρ = 0.574, p = 1.42e-50, n ≈ 562 chunks (WikiText-2 val)
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# Ensure helix-substrate importable (script lives in helix-substrate/tools/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WO = "WO-FALSIFIER-LIVE-SIDECAR-NORM-01"
SCHEMA = "falsifier_receipt:v1"

MODEL_DIR = Path("/home/voidstr3m33/models/zamba2-1.2b-helix")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_SIZE = 512
N_CHUNKS_REQUESTED = 500

MIN_FREE_VRAM_MB = 2000

# Top-6 layers from probe v3 (reference ρ-per-layer values)
TOP_LAYERS = [
    "model.layers.25.mamba.in_proj",                                         # ρ=0.395
    "model.layers.26.mamba.in_proj",                                         # ρ=0.371
    "model.layers.24.mamba.in_proj",                                         # ρ=0.365
    "model.layers.5.shared_transformer.self_attn.linear_q_adapter_list.2.0", # ρ=0.351
    "model.layers.10.mamba.in_proj",                                         # ρ=0.328
    "model.layers.5.shared_transformer.feed_forward.gate_up_proj_adapter_list.4.0",  # ρ=0.316
]

REFERENCE = {
    "probe": "mamba-scan-lite/tools/probe_v3b_aggregate.py",
    "rho_reference": 0.574,
    "rho_p_reference": 1.42e-50,
    "n_chunks_reference": 562,
}

RECEIPT_DIR = Path(__file__).resolve().parent.parent / "receipts" / "falsifier"


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def vram_preflight() -> Dict[str, float]:
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available — running on CPU, probe will be slow but OK")
        return {"free_mb": None, "total_mb": None}

    free_b, total_b = torch.cuda.mem_get_info()
    free_mb = free_b / 1e6
    total_mb = total_b / 1e6
    print(f"  VRAM free: {free_mb:.0f} MB / {total_mb:.0f} MB total")

    if free_mb < MIN_FREE_VRAM_MB:
        print()
        print(f"[FATAL] Only {free_mb:.0f} MB VRAM free (need ≥ {MIN_FREE_VRAM_MB}).")
        print("        Free VRAM and re-run. Typical recipe:")
        print("          systemctl --user stop basin-server.service")
        print("          pkill -f 'python.*chat.py' || true")
        print("          pkill -f 'python.*echo' || true")
        print("          nvidia-smi   # confirm")
        sys.exit(2)

    return {"free_mb": free_mb, "total_mb": total_mb}


# ---------------------------------------------------------------------------
# Model load (MUST match reference: HF quantizer path)
# ---------------------------------------------------------------------------

def load_model():
    # Apply mamba-scan-lite patch (T2000 requires sequential scan)
    try:
        from mamba_scan_lite import patch  # noqa: F401
        patch.apply_patch()
        print("  [mamba_scan_lite patch applied]")
    except ImportError:
        pass

    # Register HF quantizer
    import helix_substrate.hf_quantizer  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Loading tokenizer from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"  Loading model on {DEVICE} (bfloat16)")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / 1e6
        print(f"  Post-load VRAM allocated: {allocated_mb:.0f} MB")

    return tokenizer, model


# ---------------------------------------------------------------------------
# Layer enumeration (HelixLinear with non-empty sidecar)
# ---------------------------------------------------------------------------

def enumerate_hooked_layers(model) -> List[tuple]:
    """Return [(name, module)] for all HelixLinear modules with sidecar present."""
    from helix_substrate.helix_linear import HelixLinear

    hooked = []
    for name, mod in model.named_modules():
        if not isinstance(mod, HelixLinear):
            continue
        rows = getattr(mod, "_sidecar_rows", None)
        if rows is None or rows.numel() == 0:
            continue
        mod._fls_layer_name = name  # namespaced attribute to avoid collisions
        hooked.append((name, mod))
    return hooked


def static_norm(mod) -> float:
    """Single scalar per layer: sqrt(mean(deltas^2)). Input-independent."""
    d = mod._sidecar_deltas
    if d is None or d.numel() == 0:
        return 0.0
    return float(d.float().pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Probe (reference math — identical to probe_v3b_aggregate.py lines 83-119)
# ---------------------------------------------------------------------------

class LiveSidecarProbe:
    def __init__(self, hooked_layers):
        self.hooks = []
        self.chunk_data: Dict[str, List[float]] = {}
        self._current_chunk: Dict[str, list] = {}

        for name, mod in hooked_layers:
            self.chunk_data[name] = []
            self._current_chunk[name] = []
            h = mod.register_forward_hook(self._make_hook(name))
            self.hooks.append(h)

        print(f"  Hooked {len(hooked_layers)} HelixLinear layers with sidecars")

    def _make_hook(self, layer_name: str):
        def fn(module, inp, out):
            with torch.no_grad():
                x = inp[0]
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                B, S, D_in = x.shape

                # Defensive: ensure sidecar buffers are on input device
                rows = module._sidecar_rows.to(x.device)
                cols = module._sidecar_cols.to(x.device)
                deltas = module._sidecar_deltas.to(x.device).to(x.dtype)

                x_2d = x.reshape(B * S, D_in)
                weighted = x_2d[:, cols] * deltas.unsqueeze(0)

                sc_out = torch.zeros(
                    B * S, module.out_features,
                    device=x.device, dtype=x.dtype,
                )
                sc_out.scatter_add_(
                    1, rows.unsqueeze(0).expand(B * S, -1), weighted,
                )
                # Per-token sidecar L2 norm (cast to float32 before norm)
                sc_norms = sc_out.float().norm(dim=-1)  # [B*S]
                self._current_chunk[layer_name].append(sc_norms.cpu())
        return fn

    def end_chunk(self):
        """Flush per-token norms into a single per-chunk mean."""
        for name in self._current_chunk:
            vals = self._current_chunk[name]
            if vals:
                all_norms = torch.cat(vals)
                self.chunk_data[name].append(float(all_norms.mean()))
            else:
                self.chunk_data[name].append(0.0)
            self._current_chunk[name] = []

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_arrays(self) -> Dict[str, np.ndarray]:
        return {name: np.array(vals) for name, vals in self.chunk_data.items()}


# ---------------------------------------------------------------------------
# WikiText-2 loader (match reference)
# ---------------------------------------------------------------------------

def load_wikitext2(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join([t for t in ds["text"] if t.strip()])
    tokens = tokenizer.encode(text)
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    return tokens_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    cpu_start = time.process_time()
    ts_start = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    print("=" * 64)
    print(f"  {WO}")
    print("  Live sidecar-norm falsifier — match probe_v3b_aggregate.py math")
    print("=" * 64)

    # 1. Preflight
    vram_pre = vram_preflight()

    # 2. Load model
    tokenizer, model = load_model()

    # 3. Enumerate hooked layers
    hooked = enumerate_hooked_layers(model)
    n_hooked = len(hooked)
    print(f"  Total HelixLinear layers with sidecar: {n_hooked}")

    if n_hooked == 0:
        print("[FATAL] No HelixLinear layers with sidecar found. Aborting.")
        return _write_error_receipt(
            "no_hooked_layers", t_start, cpu_start, ts_start, vram_pre
        )

    # 4. Static norms
    static_norms = {name: static_norm(mod) for name, mod in hooked}

    # 5. Install probe
    probe = LiveSidecarProbe(hooked)

    # Report which TOP_LAYERS were found
    hooked_names = {name for name, _ in hooked}
    layers_used = [n for n in TOP_LAYERS if n in hooked_names]
    layers_missing = [n for n in TOP_LAYERS if n not in hooked_names]
    print(f"  TOP_LAYERS present in hooked set: {len(layers_used)}/{len(TOP_LAYERS)}")
    if layers_missing:
        print("  MISSING from TOP_LAYERS:")
        for n in layers_missing:
            print(f"    - {n}")

    # 6. Load WikiText-2 val
    print("  Loading WikiText-2 validation…")
    try:
        tokens = load_wikitext2(tokenizer)
    except Exception as e:
        print(f"[FATAL] Dataset load failed: {e}")
        probe.clear()
        return _write_error_receipt(
            f"dataset_load_failed:{e}", t_start, cpu_start, ts_start, vram_pre
        )

    total_tokens = len(tokens)
    max_chunks = (total_tokens - 1) // CHUNK_SIZE
    n_chunks = min(N_CHUNKS_REQUESTED, max_chunks)
    print(f"  Tokens: {total_tokens}  max_chunks: {max_chunks}  running: {n_chunks}")

    # 7. Stream chunks
    chunk_losses: List[float] = []
    print(f"\n  Processing {n_chunks} chunks (chunk_size={CHUNK_SIZE})…")

    try:
        with torch.no_grad():
            for i in range(n_chunks):
                start = i * CHUNK_SIZE
                end = start + CHUNK_SIZE + 1
                if end > total_tokens:
                    break

                ids = tokens[start:end].unsqueeze(0).to(DEVICE)
                logits = model(ids).logits

                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    ids[:, 1:].reshape(-1),
                    reduction="none",
                )
                chunk_losses.append(float(loss.float().mean()))
                probe.end_chunk()

                if (i + 1) % 50 == 0 or i == 0:
                    elapsed = time.time() - t_start
                    eta = elapsed / (i + 1) * n_chunks - elapsed
                    print(
                        f"    [{i + 1:4d}/{n_chunks}] loss={chunk_losses[-1]:.3f} "
                        f"({elapsed:.0f}s, ~{eta:.0f}s left)"
                    )
    except Exception as e:
        print(f"[FATAL] Forward loop crashed at chunk {len(chunk_losses)}: {e}")
        probe.clear()
        return _write_error_receipt(
            f"forward_crash:{e}", t_start, cpu_start, ts_start, vram_pre
        )

    probe.clear()
    n_actual = len(chunk_losses)
    losses_arr = np.array(chunk_losses)
    layer_arrays = probe.get_arrays()

    print(f"\n  Chunks processed: {n_actual}")
    print(f"  Mean chunk CE: {losses_arr.mean():.4f}")

    # 8. Stats
    from scipy.stats import spearmanr, pearsonr

    # Per-layer live ρ (over ALL hooked layers, not just TOP_LAYERS)
    per_layer_rho: Dict[str, Dict[str, float]] = {}
    for name, arr in layer_arrays.items():
        a = np.asarray(arr[:n_actual], dtype=float)
        if a.std() < 1e-12:
            per_layer_rho[name] = {"rho": 0.0, "p": 1.0}
            continue
        rho, p = spearmanr(a, losses_arr)
        per_layer_rho[name] = {
            "rho": float(rho) if not np.isnan(rho) else 0.0,
            "p": float(p) if not np.isnan(p) else 1.0,
        }

    # Headline: mean across TOP_LAYERS that were actually hooked
    if layers_used:
        stacked = np.stack(
            [np.asarray(layer_arrays[n][:n_actual], dtype=float) for n in layers_used]
        )
        mean_live = stacked.mean(axis=0)
        rho_head, p_head = spearmanr(mean_live, losses_arr)
        r_head, p_pearson = pearsonr(mean_live, losses_arr)
        headline = {
            "n_layers": len(layers_used),
            "layers_used": layers_used,
            "layers_missing": layers_missing,
            "spearman_rho": float(rho_head) if not np.isnan(rho_head) else 0.0,
            "spearman_p": float(p_head) if not np.isnan(p_head) else 1.0,
            "pearson_r": float(r_head) if not np.isnan(r_head) else 0.0,
            "pearson_p": float(p_pearson) if not np.isnan(p_pearson) else 1.0,
        }
    else:
        headline = {
            "n_layers": 0,
            "layers_used": [],
            "layers_missing": layers_missing,
            "spearman_rho": None,
            "spearman_p": None,
            "pearson_r": None,
            "pearson_p": None,
        }

    # Cross-layer: does static_norm predict live ρ?
    hooked_names_list = list(layer_arrays.keys())
    static_vec = np.array(
        [static_norms[n] for n in hooked_names_list], dtype=float
    )
    live_rho_vec = np.array(
        [per_layer_rho[n]["rho"] for n in hooked_names_list], dtype=float
    )
    if (
        len(hooked_names_list) >= 3
        and static_vec.std() > 1e-12
        and live_rho_vec.std() > 1e-12
    ):
        cross_rho, _ = spearmanr(static_vec, live_rho_vec)
        cross_rho = float(cross_rho) if not np.isnan(cross_rho) else None
    else:
        cross_rho = None

    # 9. Branch
    head_rho = headline["spearman_rho"]
    if head_rho is None:
        branch = "ERROR"
    elif head_rho < 0.3:
        branch = "RED"
    elif head_rho < 0.4:
        branch = "AMBER_LOW"
    elif head_rho <= 0.6:
        branch = "GREEN"
    else:
        branch = "AMBER_HIGH"

    # 10. Cost block
    wall_time_s = round(time.time() - t_start, 3)
    cpu_time_s = round(time.process_time() - cpu_start, 3)
    peak_rss_mb = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
    )
    if torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    else:
        peak_vram_mb = None

    cost = {
        "wall_time_s": wall_time_s,
        "cpu_time_s": cpu_time_s,
        "peak_memory_mb": peak_rss_mb,
        "peak_vram_mb": peak_vram_mb,
        "free_vram_pre_mb": vram_pre.get("free_mb"),
        "total_vram_mb": vram_pre.get("total_mb"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "hostname": platform.node(),
        "timestamp_start": ts_start,
        "timestamp_end": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
    }

    receipt = {
        "wo": WO,
        "schema": SCHEMA,
        "branch": branch,
        "model": {
            "path": str(MODEL_DIR),
            "architecture": "Zamba2",
            "compression": "HXQ (CDNA v3)",
        },
        "device": DEVICE,
        "eval": {
            "dataset": "wikitext-2-raw-v1",
            "split": "validation",
            "chunk_size": CHUNK_SIZE,
            "n_chunks_requested": N_CHUNKS_REQUESTED,
            "n_chunks_actual": n_actual,
            "mean_ce_loss": round(float(losses_arr.mean()), 4),
        },
        "target_layers": TOP_LAYERS,
        "headline": {
            **headline,
            "spearman_rho": (
                round(headline["spearman_rho"], 4)
                if headline["spearman_rho"] is not None
                else None
            ),
            "spearman_p": headline["spearman_p"],
            "pearson_r": (
                round(headline["pearson_r"], 4)
                if headline["pearson_r"] is not None
                else None
            ),
            "pearson_p": headline["pearson_p"],
        },
        "per_layer_live_rho": {
            name: {"rho": round(v["rho"], 4), "p": v["p"]}
            for name, v in per_layer_rho.items()
        },
        "per_layer_static_norm": {
            name: round(v, 6) for name, v in static_norms.items()
        },
        "cross_layer_static_vs_live_rho_spearman": (
            round(cross_rho, 4) if cross_rho is not None else None
        ),
        "n_hooked_layers": n_hooked,
        "n_chunks_delta_from_reference": (
            n_actual - REFERENCE["n_chunks_reference"]
        ),
        "reference": REFERENCE,
        "cost": cost,
    }

    # 11. Write receipt
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RECEIPT_DIR / f"sidecar_live_norm_{ts_file}.json"
    out_path.write_text(json.dumps(receipt, indent=2))

    # 12. Print branch and exit
    print()
    print("=" * 64)
    print(f"  BRANCH: {branch}")
    if head_rho is not None:
        print(f"  Headline Spearman ρ = {head_rho:.4f}  (reference = {REFERENCE['rho_reference']})")
    else:
        print("  Headline Spearman ρ = None (ERROR)")
    if cross_rho is not None:
        print(f"  Cross-layer static_norm vs live ρ: {cross_rho:.4f}")
    print(f"  Receipt: {out_path}")
    print("=" * 64)

    if branch == "GREEN":
        msg = "  Next: Layer 1 baseline registry + drift detector (separate plan)"
    elif branch == "AMBER_LOW":
        msg = "  Next: re-run with n=300 and n=700 for stability check"
    elif branch == "AMBER_HIGH":
        msg = "  Next: audit for leakage BEFORE trusting signal. Re-run at 300/700."
    elif branch == "RED":
        msg = "  Next: write post-mortem; close runtime-gate direction"
    else:
        msg = "  Next: diagnose pipeline failure"
    print(msg)

    if branch == "GREEN":
        return 0
    if branch in ("AMBER_LOW", "AMBER_HIGH"):
        return 1
    if branch == "RED":
        return 1
    return 2  # ERROR


def _write_error_receipt(reason, t_start, cpu_start, ts_start, vram_pre):
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RECEIPT_DIR / f"sidecar_live_norm_{ts_file}_ERROR.json"
    cost = {
        "wall_time_s": round(time.time() - t_start, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated() / 1e6, 1)
            if torch.cuda.is_available()
            else None
        ),
        "free_vram_pre_mb": vram_pre.get("free_mb") if vram_pre else None,
        "total_vram_mb": vram_pre.get("total_mb") if vram_pre else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "hostname": platform.node(),
        "timestamp_start": ts_start,
        "timestamp_end": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
    }
    receipt = {
        "wo": WO,
        "schema": SCHEMA,
        "branch": "ERROR",
        "error": reason,
        "reference": REFERENCE,
        "cost": cost,
    }
    out_path.write_text(json.dumps(receipt, indent=2))
    print(f"  Error receipt: {out_path}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
