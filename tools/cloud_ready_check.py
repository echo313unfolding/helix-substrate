#!/usr/bin/env python3
"""
Pre-flight check for cloud GPU sessions.

Run this BEFORE you're on the clock. It catches every broken import,
missing file, package version mismatch, and silent failure that would
otherwise burn $2/hr of cloud time debugging.

Usage:
    # Quick check (imports + files + dummy forward, ~30s)
    python3 tools/cloud_ready_check.py --model-dir ~/models/tinyllama_fp32

    # Full check (includes PPL eval on 512 tokens, ~5min on CPU)
    python3 tools/cloud_ready_check.py --model-dir ~/models/tinyllama_fp32 --full

    # GPU check (CUDA + Triton kernel + GPU forward)
    python3 tools/cloud_ready_check.py --model-dir ~/models/tinyllama_fp32 --gpu

    # Everything
    python3 tools/cloud_ready_check.py --model-dir ~/models/tinyllama_fp32 --full --gpu

Exit codes:
    0 = all checks passed
    1 = critical failure (will break cloud session)
    2 = warning (non-critical, but note it)
"""

import argparse
import json
import os
import platform
import resource
import sys
import tempfile
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pass_count = 0
_warn_count = 0
_fail_count = 0
_results = []


def _record(phase, name, status, detail=""):
    global _pass_count, _warn_count, _fail_count
    if status == "PASS":
        _pass_count += 1
        icon = "\033[32m✓\033[0m"
    elif status == "WARN":
        _warn_count += 1
        icon = "\033[33m⚠\033[0m"
    else:
        _fail_count += 1
        icon = "\033[31m✗\033[0m"
    line = f"  {icon} [{phase}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line, flush=True)
    _results.append({"phase": phase, "name": name, "status": status, "detail": detail})


def check(phase, name, condition, detail="", fail_detail=""):
    if condition:
        _record(phase, name, "PASS", detail)
        return True
    else:
        _record(phase, name, "FAIL", fail_detail or detail)
        return False


def warn(phase, name, condition, detail="", warn_detail=""):
    if condition:
        _record(phase, name, "PASS", detail)
        return True
    else:
        _record(phase, name, "WARN", warn_detail or detail)
        return False


# ---------------------------------------------------------------------------
# Phase 0: Environment
# ---------------------------------------------------------------------------

def phase0_environment():
    print("\n── Phase 0: Environment ──", flush=True)

    # Python version
    v = sys.version_info
    check("env", "Python >= 3.10",
          v.major == 3 and v.minor >= 10,
          f"{v.major}.{v.minor}.{v.micro}",
          f"Python {v.major}.{v.minor} — need 3.10+")

    # Core packages
    for pkg_name, min_ver in [
        ("torch", "2.0"),
        ("transformers", "4.38"),
        ("numpy", "1.24"),
        ("safetensors", "0.3"),
    ]:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "unknown")
            check("env", f"{pkg_name} installed", True, f"v{ver}")
        except ImportError:
            check("env", f"{pkg_name} installed", False,
                  fail_detail=f"MISSING — pip install {pkg_name}")

    # Optional but important
    for pkg_name, purpose in [
        ("datasets", "WikiText-2 PPL eval"),
        ("scipy", "k-means in cdnav3_writer"),
        ("yaml", "echo_runtime config"),
    ]:
        try:
            __import__(pkg_name)
            _record("env", f"{pkg_name} (optional)", "PASS", purpose)
        except ImportError:
            _record("env", f"{pkg_name} (optional)", "WARN",
                    f"missing — needed for {purpose}")

    # System resources
    import torch
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    check("env", f"RAM >= 16 GB", ram_gb >= 16, f"{ram_gb:.1f} GB")
    check("env", f"CPU cores", os.cpu_count() is not None, f"{os.cpu_count()} cores")


# ---------------------------------------------------------------------------
# Phase 1: Imports
# ---------------------------------------------------------------------------

def phase1_imports():
    print("\n── Phase 1: helix_substrate imports ──", flush=True)

    # Ensure helix_substrate is importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Core module
    imports_ok = True
    try:
        import helix_substrate
        check("import", "helix_substrate", True, f"v{helix_substrate.__version__}")
    except Exception as e:
        check("import", "helix_substrate", False, fail_detail=str(e))
        imports_ok = False

    # Critical imports for the product pipeline
    critical_imports = [
        ("helix_substrate.helix_linear",
         ["HelixLinear", "load_cdna_factors", "swap_to_helix", "swap_summary",
          "load_helix_linear_from_cdnav3"]),
        ("helix_substrate.cdnav3_writer", ["CDNAv3Writer"]),
        ("helix_substrate.cdnav3_reader", ["CDNAv3Reader"]),
        ("helix_substrate.tensor_policy",
         ["classify_tensor", "TensorPolicy", "TensorClass", "get_policy",
          "get_default_policy"]),
    ]

    for module_name, attrs in critical_imports:
        try:
            mod = __import__(module_name, fromlist=attrs)
            missing = [a for a in attrs if not hasattr(mod, a)]
            if missing:
                check("import", module_name, False,
                      fail_detail=f"missing attrs: {missing}")
                imports_ok = False
            else:
                check("import", module_name, True, f"{len(attrs)} attrs OK")
        except Exception as e:
            check("import", module_name, False, fail_detail=str(e))
            imports_ok = False

    # Optional: Triton kernel (crashes without CUDA, so catch all exceptions)
    try:
        from helix_substrate.triton_vq_matmul import (
            fused_vq_matmul, is_available, get_kernel_metadata
        )
        _record("import", "triton_vq_matmul", "PASS", "fused kernel importable")
    except Exception as e:
        _record("import", "triton_vq_matmul (optional)", "WARN",
                f"{type(e).__name__}: {str(e)[:100]}")

    # compress.py importability (used as a library in the optimize CLI)
    compress_path = Path(__file__).resolve().parent / "compress.py"
    check("import", "tools/compress.py exists", compress_path.exists())

    # calibrate.py
    calibrate_path = Path(__file__).resolve().parent / "calibrate.py"
    warn("import", "tools/calibrate.py exists", calibrate_path.exists(),
         warn_detail="AWQ calibration script missing")

    # eval_ppl_cpu.py
    eval_path = Path(__file__).resolve().parent / "eval_ppl_cpu.py"
    check("import", "tools/eval_ppl_cpu.py exists", eval_path.exists())

    return imports_ok


# ---------------------------------------------------------------------------
# Phase 2: Model files
# ---------------------------------------------------------------------------

def phase2_model_files(model_dir: Path):
    print("\n── Phase 2: Model files ──", flush=True)

    model_dir = model_dir.expanduser().resolve()
    check("files", "model_dir exists", model_dir.is_dir(), str(model_dir))

    # Config
    config_path = model_dir / "config.json"
    has_config = config_path.exists()
    check("files", "config.json", has_config)

    if has_config:
        try:
            cfg = json.loads(config_path.read_text())
            arch = cfg.get("architectures", ["unknown"])[0]
            hidden = cfg.get("hidden_size", "?")
            layers = cfg.get("num_hidden_layers", "?")
            vocab = cfg.get("vocab_size", "?")
            check("files", "config.json readable", True,
                  f"{arch}, {layers}L, h={hidden}, vocab={vocab}")
        except Exception as e:
            check("files", "config.json readable", False, fail_detail=str(e))

    # Weights file
    has_weights = False
    for pattern in ["model.safetensors", "model-00001-of-*.safetensors",
                    "pytorch_model.bin"]:
        matches = list(model_dir.glob(pattern))
        if matches:
            has_weights = True
            total_mb = sum(f.stat().st_size for f in matches) / (1024**2)
            check("files", "weight files", True,
                  f"{len(matches)} file(s), {total_mb:.0f} MB")
            break
    if not has_weights:
        check("files", "weight files", False,
              fail_detail="no .safetensors or .bin found")

    # Tokenizer
    tokenizer_files = list(model_dir.glob("tokenizer*")) + list(model_dir.glob("special_tokens_map*"))
    check("files", "tokenizer files", len(tokenizer_files) > 0,
          f"{len(tokenizer_files)} tokenizer files")

    # CDNA v3 artifacts
    cdna_dir = model_dir / "cdnav3"
    has_cdna = cdna_dir.is_dir()
    check("files", "cdnav3/ directory", has_cdna)

    if has_cdna:
        manifest_path = cdna_dir / "manifest.json"
        check("files", "cdnav3/manifest.json", manifest_path.exists())

        cdna_dirs = list(cdna_dir.glob("*.cdnav3"))
        check("files", f"compressed tensors", len(cdna_dirs) > 0,
              f"{len(cdna_dirs)} .cdnav3 directories")

        # Spot-check one tensor directory
        if cdna_dirs:
            sample = cdna_dirs[0]
            meta = sample / "meta.json"
            codebook = sample / "codebook.npy"
            has_indices = (sample / "indices.npy").exists() or (sample / "indices.bin").exists()
            check("files", f"sample tensor complete ({sample.name})",
                  meta.exists() and codebook.exists() and has_indices,
                  fail_detail=f"missing: meta={meta.exists()}, cb={codebook.exists()}, idx={has_indices}")

    return has_config and has_weights and has_cdna


# ---------------------------------------------------------------------------
# Phase 3: Load and forward pass
# ---------------------------------------------------------------------------

def phase3_load_and_forward(model_dir: Path):
    print("\n── Phase 3: Load + forward pass ──", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from helix_substrate.helix_linear import load_cdna_factors, swap_to_helix, swap_summary

    model_dir = model_dir.expanduser().resolve()
    cdna_dir = model_dir / "cdnav3"

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        check("load", "tokenizer", True, f"vocab={tokenizer.vocab_size}")
    except Exception as e:
        check("load", "tokenizer", False, fail_detail=str(e)[:200])
        return False

    # Load CDNA factors
    try:
        t0 = time.time()
        factors = load_cdna_factors(cdna_dir)
        t_factors = time.time() - t0
        check("load", "cdna factors", len(factors) > 0,
              f"{len(factors)} tensors in {t_factors:.1f}s")
    except Exception as e:
        check("load", "cdna factors", False, fail_detail=str(e)[:200])
        return False

    # Load model and swap
    try:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.float32,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        model = swap_to_helix(model, factors)
        del factors
        model.eval()
        t_load = time.time() - t0
        summary = swap_summary(model)
        n_helix = summary.get("helix_modules", 0)
        n_linear = summary.get("linear_modules_remaining",
                               summary.get("linear_remaining", 0))
        check("load", "swap_to_helix", n_helix > 0,
              f"{n_helix} HelixLinear, {n_linear} dense, {t_load:.1f}s")
    except Exception as e:
        check("load", "swap_to_helix", False, fail_detail=str(e)[:200])
        return False

    # Forward pass on dummy input
    try:
        dummy = tokenizer("The quick brown fox", return_tensors="pt")
        input_ids = dummy["input_ids"]
        with torch.no_grad():
            t0 = time.time()
            out = model(input_ids=input_ids)
            t_fwd = time.time() - t0

        logits = out.logits
        check("forward", "output shape",
              logits.ndim == 3 and logits.shape[0] == 1,
              f"shape={list(logits.shape)}, {t_fwd:.2f}s")

        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        check("forward", "no NaN/Inf", not has_nan and not has_inf,
              fail_detail=f"NaN={has_nan}, Inf={has_inf}")

        # Sanity: top token should be a real word, not garbage
        top_id = logits[0, -1].argmax().item()
        top_token = tokenizer.decode([top_id])
        check("forward", "top prediction is real token", len(top_token.strip()) > 0,
              f"top='{top_token.strip()}'")
    except Exception as e:
        check("forward", "dummy forward pass", False, fail_detail=str(e)[:200])
        return False

    # Generation
    try:
        t0 = time.time()
        gen_ids = model.generate(
            input_ids, max_new_tokens=16, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        t_gen = time.time() - t0
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        n_new = gen_ids.shape[1] - input_ids.shape[1]
        check("forward", "model.generate()", n_new > 0,
              f"{n_new} tokens in {t_gen:.1f}s: '{gen_text[:60]}...'")
    except Exception as e:
        check("forward", "model.generate()", False, fail_detail=str(e)[:200])

    # Free memory
    del model
    import gc
    gc.collect()
    return True


# ---------------------------------------------------------------------------
# Phase 4: Compression roundtrip
# ---------------------------------------------------------------------------

def phase4_compression_roundtrip():
    print("\n── Phase 4: Compression roundtrip ──", flush=True)

    import numpy as np
    from helix_substrate.cdnav3_writer import CDNAv3Writer
    from helix_substrate.cdnav3_reader import CDNAv3Reader
    from helix_substrate.tensor_policy import TensorPolicy, TensorClass

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test tensor
        rng = np.random.RandomState(42)
        tensor = rng.randn(128, 64).astype(np.float32)

        # Write
        try:
            writer = CDNAv3Writer(tmpdir)
            policy = TensorPolicy(
                tensor_class=TensorClass.UNKNOWN,
                storage_mode="codebook+sidecar",
                n_clusters=256,
                use_kmeans=True,
                sidecar_enabled=True,
                percentile=99.9,
                max_corrections=512,
            )
            stats = writer.write_tensor(tensor, "test_tensor", policy=policy)
            check("roundtrip", "CDNAv3Writer.write_tensor()", True,
                  f"ratio={stats.get('compression_ratio', '?')}x")
        except Exception as e:
            check("roundtrip", "CDNAv3Writer.write_tensor()", False,
                  fail_detail=str(e)[:200])
            return False

        # Read
        try:
            reader = CDNAv3Reader(Path(tmpdir) / "test_tensor.cdnav3")
            recon = reader.reconstruct()
            check("roundtrip", "CDNAv3Reader.reconstruct()", True,
                  f"shape={recon.shape}")
        except Exception as e:
            check("roundtrip", "CDNAv3Reader.reconstruct()", False,
                  fail_detail=str(e)[:200])
            return False

        # Quality
        cosine = float(np.dot(tensor.ravel(), recon.ravel()) /
                       (np.linalg.norm(tensor) * np.linalg.norm(recon)))
        check("roundtrip", "cosine >= 0.99", cosine >= 0.99, f"cos={cosine:.6f}")

    return True


# ---------------------------------------------------------------------------
# Phase 5: PPL evaluation (optional, slow)
# ---------------------------------------------------------------------------

def phase5_ppl_eval(model_dir: Path):
    print("\n── Phase 5: PPL evaluation (small) ──", flush=True)

    # Check datasets is available
    try:
        from datasets import load_dataset
        check("ppl", "datasets.load_dataset importable", True)
    except ImportError:
        check("ppl", "datasets.load_dataset importable", False,
              fail_detail="pip install datasets")
        return False

    # Check WikiText-2 loads
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if t.strip()]
        check("ppl", "WikiText-2 loaded", len(texts) > 100,
              f"{len(texts)} paragraphs")
    except Exception as e:
        check("ppl", "WikiText-2 loaded", False, fail_detail=str(e)[:200])
        return False

    # Run mini PPL eval (512 tokens only — just proves the pipeline works)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from helix_substrate.helix_linear import load_cdna_factors, swap_to_helix

    model_dir = model_dir.expanduser().resolve()

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        factors = load_cdna_factors(model_dir / "cdnav3")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.float32,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        model = swap_to_helix(model, factors)
        del factors
        model.eval()
    except Exception as e:
        check("ppl", "model load for PPL", False, fail_detail=str(e)[:200])
        return False

    try:
        text = "\n\n".join(texts[:50])
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=513)
        ids = enc.input_ids[:, :513]

        with torch.no_grad():
            t0 = time.time()
            out = model(input_ids=ids[:, :-1])
            logits = out.logits.float()
            labels = ids[:, 1:]
            import torch.nn.functional as F
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            ppl = float(torch.exp(loss).item())
            t_eval = time.time() - t0

        # TinyLlama should be < 20 PPL, larger models < 30
        check("ppl", "PPL is finite", ppl < 1000, f"PPL={ppl:.2f}, {t_eval:.1f}s")
        check("ppl", "PPL is reasonable", ppl < 50,
              f"PPL={ppl:.2f}",
              fail_detail=f"PPL={ppl:.2f} — suspiciously high, check model integrity")
    except Exception as e:
        check("ppl", "PPL eval", False, fail_detail=str(e)[:200])
        return False
    finally:
        del model
        import gc
        gc.collect()

    return True


# ---------------------------------------------------------------------------
# Phase 6: GPU checks (optional)
# ---------------------------------------------------------------------------

def phase6_gpu():
    print("\n── Phase 6: GPU ──", flush=True)

    import torch

    has_cuda = torch.cuda.is_available()
    check("gpu", "CUDA available", has_cuda,
          fail_detail="no CUDA — GPU checks skipped")
    if not has_cuda:
        return False

    # Device info
    dev = torch.cuda.get_device_properties(0)
    vram_mb = dev.total_memory / (1024**2)
    cc = f"{dev.major}.{dev.minor}"
    check("gpu", "GPU detected", True,
          f"{dev.name}, {vram_mb:.0f} MB VRAM, CC {cc}")

    # VRAM availability
    free_mb = (torch.cuda.mem_get_info(0)[0]) / (1024**2)
    check("gpu", "VRAM free", free_mb > 500,
          f"{free_mb:.0f} MB free",
          fail_detail=f"only {free_mb:.0f} MB free — kill other GPU processes")

    # Triton kernel
    try:
        from helix_substrate.triton_vq_matmul import (
            fused_vq_matmul, is_available, get_kernel_metadata
        )
        if is_available():
            meta = get_kernel_metadata()
            check("gpu", "Triton kernel available", True,
                  f"{meta.get('kernel_version', '?')}, CC {meta.get('compute_capability', '?')}")

            # Actually run the kernel on a small test
            codebook = torch.randn(256, device="cuda", dtype=torch.float32)
            indices = torch.randint(0, 256, (64, 32), device="cuda", dtype=torch.uint8)
            x = torch.randn(1, 32, device="cuda", dtype=torch.float32)
            try:
                y = fused_vq_matmul(x, codebook, indices)
                check("gpu", "Triton kernel executes", y.shape == (1, 64),
                      f"output shape={list(y.shape)}")
            except Exception as e:
                check("gpu", "Triton kernel executes", False,
                      fail_detail=str(e)[:200])
        else:
            _record("gpu", "Triton kernel", "WARN", "not available on this device")
    except ImportError:
        _record("gpu", "Triton kernel (optional)", "WARN", "triton not installed")

    # GPU matmul sanity
    try:
        a = torch.randn(128, 128, device="cuda")
        b = torch.randn(128, 128, device="cuda")
        c = a @ b
        check("gpu", "GPU matmul", c.shape == (128, 128) and not torch.isnan(c).any(),
              "128x128 matmul OK")
    except Exception as e:
        check("gpu", "GPU matmul", False, fail_detail=str(e)[:200])

    return True


# ---------------------------------------------------------------------------
# Phase 7: Receipt format
# ---------------------------------------------------------------------------

def phase7_receipt_format():
    print("\n── Phase 7: Receipt system ──", flush=True)

    # Check receipts directory exists
    receipts_dir = Path(__file__).resolve().parent.parent / "receipts"
    check("receipt", "receipts/ directory", receipts_dir.is_dir())

    if not receipts_dir.is_dir():
        return False

    # Find any receipt and validate it has a cost block
    receipt_files = list(receipts_dir.rglob("*.json"))
    check("receipt", "receipt files exist", len(receipt_files) > 0,
          f"{len(receipt_files)} receipts found")

    if receipt_files:
        # Check a sample receipt for cost block
        sample = receipt_files[0]
        try:
            data = json.loads(sample.read_text())
            has_cost = "cost" in data
            if has_cost:
                cost = data["cost"]
                has_wall = "wall_time_s" in cost
                has_ts = "timestamp_start" in cost or "timestamp_end" in cost
                check("receipt", "cost block present", True,
                      f"wall_time={cost.get('wall_time_s', '?')}s")
                check("receipt", "cost block complete", has_wall and has_ts,
                      fail_detail="missing wall_time_s or timestamps")
            else:
                check("receipt", "cost block present", False,
                      fail_detail=f"no 'cost' key in {sample.name}")
        except Exception as e:
            check("receipt", "receipt parseable", False,
                  fail_detail=f"{sample.name}: {str(e)[:100]}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight check for cloud GPU sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Path to model directory with cdnav3/ subdirectory")
    parser.add_argument("--full", action="store_true",
                        help="Include PPL evaluation (slow, ~5min on CPU)")
    parser.add_argument("--gpu", action="store_true",
                        help="Include GPU/Triton checks")
    parser.add_argument("--receipt", type=Path, default=None,
                        help="Save check receipt to this path")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()

    print("=" * 70)
    print("  CLOUD READY CHECK — helix-substrate")
    print("=" * 70)
    print(f"  Model:  {args.model_dir}")
    print(f"  Full:   {args.full}")
    print(f"  GPU:    {args.gpu}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Host:   {platform.node()}")

    # Run phases
    phase0_environment()
    imports_ok = phase1_imports()
    files_ok = phase2_model_files(args.model_dir) if imports_ok else False
    forward_ok = phase3_load_and_forward(args.model_dir) if files_ok else False
    roundtrip_ok = phase4_compression_roundtrip() if imports_ok else False

    if args.full and forward_ok:
        phase5_ppl_eval(args.model_dir)

    if args.gpu:
        phase6_gpu()

    phase7_receipt_format()

    # Summary
    wall = time.time() - t_start
    cpu = time.process_time() - cpu_start

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: \033[32m{_pass_count} passed\033[0m, "
          f"\033[33m{_warn_count} warnings\033[0m, "
          f"\033[31m{_fail_count} failed\033[0m")
    print(f"  Time: {wall:.1f}s wall, {cpu:.1f}s CPU")

    if _fail_count == 0 and _warn_count == 0:
        print(f"\n  \033[32m★ ALL CLEAR — ready for cloud session\033[0m")
    elif _fail_count == 0:
        print(f"\n  \033[33m⚠ WARNINGS — check optional deps before cloud\033[0m")
    else:
        print(f"\n  \033[31m✗ FAILURES — fix before spending cloud $$$\033[0m")

    print(f"{'=' * 70}")

    # Save receipt
    receipt_path = args.receipt
    if receipt_path is None:
        receipt_dir = Path(__file__).resolve().parent.parent / "receipts" / "cloud_checks"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        receipt_path = receipt_dir / f"cloud_check_{ts}.json"

    receipt = {
        "schema": "cloud_ready_check:v1",
        "model_dir": str(args.model_dir.expanduser().resolve()),
        "full_check": args.full,
        "gpu_check": args.gpu,
        "passed": _pass_count,
        "warnings": _warn_count,
        "failed": _fail_count,
        "verdict": "PASS" if _fail_count == 0 else "FAIL",
        "checks": _results,
        "cost": {
            "wall_time_s": round(wall, 3),
            "cpu_time_s": round(cpu, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S",
                                             time.localtime(t_start)),
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"\n  Receipt: {receipt_path}")

    sys.exit(1 if _fail_count > 0 else (2 if _warn_count > 0 else 0))


if __name__ == "__main__":
    main()
