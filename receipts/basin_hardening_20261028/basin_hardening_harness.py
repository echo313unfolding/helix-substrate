#!/usr/bin/env python3
"""
Basin helix_fused operational hardening test.

Starts Basin with helix_fused backend, runs all hardening checks, produces
a summary receipt. Exercises: startup receipt, strict mode, repeated runs,
multi-model capability, negative paths, and receipt completeness.

Usage:
    python3 tools/basin_hardening.py [--strict] [--port 8099]

Work Order: WO-BASIN-HARDENING-01
Work Order: WO-BASIN-RECEIPT-SEMANTICS-01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Ensure imports work
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASIN_REQUEST = {
    "messages": [{"role": "user", "content": "What is the meaning of life?"}],
    "max_tokens": 32,
    "temperature": 0.0,
    "top_p": 1.0,
    "quant": {"phi0_deg": 0.0, "cardinality": 2, "control_consent": True},
}


def _post_chat(port: int, prompt: str = None, max_tokens: int = 32) -> dict:
    """Send a chat completion request. Returns parsed JSON response."""
    import urllib.request
    import urllib.error

    body = dict(BASIN_REQUEST)
    if prompt:
        body["messages"] = [{"role": "user", "content": prompt}]
    body["max_tokens"] = max_tokens

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": json.loads(e.read().decode()), "status": e.code}


def _wait_healthy(port: int, timeout: float = 60) -> bool:
    """Wait for Basin server to be healthy."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def _extract_receipt(response: dict) -> dict | None:
    """Extract the helix_fused receipt from a Basin response."""
    receipts = response.get("streaming_receipts", [])
    if receipts:
        return receipts[0]
    return None


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_startup_receipt(port: int) -> tuple[bool, str, dict | None]:
    """Task 1: Verify startup receipt was persisted and semantically valid."""
    from helix_substrate.basin_runtime import validate_startup_receipt

    # Startup receipt is written to receipts/api_chat/<date>/receipt_startup_helix_fused.json
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    receipt_path = PROJECT_ROOT / "receipts" / "api_chat" / date_str / "receipt_startup_helix_fused.json"

    if not receipt_path.exists():
        return False, f"startup receipt not found: {receipt_path}", None

    receipt = json.loads(receipt_path.read_text())

    issues = validate_startup_receipt(receipt)
    if issues:
        return False, f"startup receipt validation failed: {issues}", receipt

    return True, "startup receipt structurally + semantically valid", receipt


def test_single_request(port: int) -> tuple[bool, str, dict | None]:
    """Task 2: Single request with receipt validation."""
    from helix_substrate.basin_runtime import validate_receipt

    resp = _post_chat(port)
    if "error" in resp:
        return False, f"request failed: {resp['error']}", None

    receipt = _extract_receipt(resp)
    if receipt is None:
        return False, "no receipt in response", None

    issues = validate_receipt(receipt)
    if issues:
        return False, f"receipt validation failed: {issues}", receipt

    # Verify receipt persistence
    persist_path = receipt.get("persist_path")
    if not persist_path:
        return False, "no persist_path in receipt", receipt

    full_path = PROJECT_ROOT / persist_path
    if not full_path.exists():
        return False, f"receipt file not on disk: {full_path}", receipt

    return True, "single request + receipt valid", receipt


def test_strict_mode_success(port: int) -> tuple[bool, str, dict | None]:
    """Task 3a: Strict mode request succeeds when Triton is available."""
    from helix_substrate.basin_runtime import validate_receipt, RUNTIME_PATH_FUSED

    resp = _post_chat(port)
    if "error" in resp:
        return False, f"strict request failed: {resp['error']}", None

    receipt = _extract_receipt(resp)
    if receipt is None:
        return False, "no receipt in strict response", None

    # Semantic validation
    issues = validate_receipt(receipt)
    if issues:
        return False, f"strict receipt semantic fail: {issues}", receipt

    ds = receipt.get("dispatch_summary", {})
    if ds.get("runtime_path") != RUNTIME_PATH_FUSED:
        return False, f"strict mode but runtime_path={ds.get('runtime_path')!r}", receipt

    if ds.get("naive_count", -1) != 0:
        return False, f"strict mode but naive_count={ds.get('naive_count')}", receipt

    return True, f"strict mode: runtime_path={RUNTIME_PATH_FUSED}, zero naive", receipt


def test_repeated_runs(port: int, n: int = 5) -> tuple[bool, str, list]:
    """Task 4: N sequential requests — no drift, no naive, latency spread, semantic validation."""
    from helix_substrate.basin_runtime import validate_receipt

    prompts = [
        "What is the meaning of life?",
        "Explain recursion in one sentence.",
        "What is the capital of France?",
        "How does a hash table work?",
        "Name the planets in our solar system.",
    ]

    receipts = []
    for i in range(n):
        resp = _post_chat(port, prompt=prompts[i % len(prompts)])
        if "error" in resp:
            return False, f"request {i} failed: {resp['error']}", receipts

        receipt = _extract_receipt(resp)
        if receipt is None:
            return False, f"request {i}: no receipt", receipts

        # Structural + semantic validation
        issues = validate_receipt(receipt)
        if issues:
            return False, f"request {i} semantic fail: {issues}", receipts

        receipts.append(receipt)

    # Check no drift
    paths = [r["dispatch_summary"]["runtime_path"] for r in receipts]
    unique_paths = set(paths)
    if len(unique_paths) != 1:
        return False, f"runtime_path drift: {paths}", receipts

    # Check no naive
    naive_counts = [r["dispatch_summary"]["naive_count"] for r in receipts]
    if any(nc > 0 for nc in naive_counts):
        return False, f"naive dispatch appeared: {naive_counts}", receipts

    # Latency spread
    tok_s_values = [r["timing"]["tok_s"] for r in receipts]
    spread = max(tok_s_values) - min(tok_s_values)

    return True, (
        f"{n} runs: path={paths[0]}, "
        f"tok/s range=[{min(tok_s_values):.1f}, {max(tok_s_values):.1f}], "
        f"spread={spread:.1f}"
    ), receipts


def test_multi_model_capability() -> tuple[bool, str, list]:
    """Task 5: Capability check for both models (no HTTP needed)."""
    from helix_substrate.basin_runtime import check_cdnav3_capability

    models = {
        "falcon_h1": Path.home() / "models" / "falcon-h1-0.5b-base" / "cdnav3",
        "jamba_tiny": Path.home() / "models" / "jamba-tiny-dev" / "cdnav3",
        "qwen3b": Path.home() / "models" / "qwen2.5-coder-3b-instruct" / "cdnav3",
        "granite_micro": Path.home() / "models" / "granite-4.0-h-micro" / "cdnav3",
        "zamba2": Path.home() / "models" / "zamba2-2.7b-instruct-dense" / "cdnav3",
    }

    results = []
    tested = []
    for model_id, cdna_dir in models.items():
        if not cdna_dir.exists():
            results.append(f"{model_id}: SKIP (not present)")
            continue

        cap = check_cdnav3_capability(cdna_dir)
        # Only check data issues, not CUDA/Triton (those are runtime)
        data_issues = [i for i in cap["issues"]
                       if "CUDA" not in i and "Triton" not in i]

        if data_issues:
            return False, f"{model_id}: data issues: {data_issues}", results

        results.append(
            f"{model_id}: valid={cap['valid']}, "
            f"model={cap['model_name']}, "
            f"tensors={cap['tensor_dirs_found']}/{cap['n_tensors']}, "
            f"sha256={cap['manifest_sha256'][:12]}..."
        )
        tested.append(model_id)

    if not tested:
        return False, "no models found to test", results

    return True, f"models tested: {tested}", results


def test_negative_paths() -> tuple[bool, str, list]:
    """Task 6: Negative paths — broken manifest, missing dir, unsupported config."""
    from helix_substrate.basin_runtime import check_cdnav3_capability
    import tempfile

    results = []

    # 6a: nonexistent dir
    cap = check_cdnav3_capability("/nonexistent/no/such/path")
    if cap["valid"]:
        return False, "nonexistent dir passed as valid", results
    results.append(f"nonexistent_dir: valid=False, issues={cap['issues']}")

    # 6b: broken manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "manifest.json").write_text("{broken")
        cap = check_cdnav3_capability(tmpdir)
        if cap["valid"]:
            return False, "broken manifest passed as valid", results
        results.append(f"broken_manifest: valid=False, issues={cap['issues']}")

    # 6c: empty manifest (missing all fields)
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "manifest.json").write_text("{}")
        cap = check_cdnav3_capability(tmpdir)
        if cap["valid"]:
            return False, "empty manifest passed as valid", results
        # Must detect all 4 missing fields
        for field in ("model", "n_blocks", "n_tensors", "compression_ratio"):
            if not any(field in i for i in cap["issues"]):
                return False, f"empty manifest: missing field '{field}' not detected", results
        results.append(f"empty_manifest: valid=False, {len(cap['issues'])} issues detected")

    # 6d: tensor count mismatch
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = {"model": "X", "n_blocks": 1, "n_tensors": 99, "compression_ratio": 2.0}
        (Path(tmpdir) / "manifest.json").write_text(json.dumps(manifest))
        cap = check_cdnav3_capability(tmpdir)
        if cap["valid"]:
            return False, "tensor count mismatch passed as valid", results
        if not any("Expected 99" in i for i in cap["issues"]):
            return False, f"tensor count mismatch not detected: {cap['issues']}", results
        results.append(f"tensor_count_mismatch: valid=False, issues={cap['issues']}")

    # 6e: wrong format_version
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = {"model": "X", "n_blocks": 1, "n_tensors": 1, "compression_ratio": 2.0}
        (Path(tmpdir) / "manifest.json").write_text(json.dumps(manifest))
        tdir = Path(tmpdir) / "test.cdnav3"
        tdir.mkdir()
        (tdir / "meta.json").write_text(json.dumps({"format_version": "cdna_v1"}))
        cap = check_cdnav3_capability(tmpdir)
        if cap["valid"]:
            return False, "wrong format_version passed as valid", results
        results.append(f"wrong_format_version: valid=False, issues={cap['issues']}")

    return True, f"{len(results)} negative paths checked", results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Basin helix_fused hardening")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--strict", action="store_true",
                        help="Run with HELIX_STRICT_BENCHMARK=1")
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 70)
    print("BASIN HELIX_FUSED OPERATIONAL HARDENING")
    print(f"Port: {args.port} | Strict: {args.strict}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Task 5 + 6: Offline tests (no server needed)
    # -----------------------------------------------------------------------
    print("\n--- Task 5: Multi-model capability ---")
    ok5, msg5, detail5 = test_multi_model_capability()
    print(f"  {'PASS' if ok5 else 'FAIL'}: {msg5}")
    for d in detail5:
        print(f"    {d}")

    print("\n--- Task 6: Negative paths ---")
    ok6, msg6, detail6 = test_negative_paths()
    print(f"  {'PASS' if ok6 else 'FAIL'}: {msg6}")
    for d in detail6:
        print(f"    {d}")

    if not ok5 or not ok6:
        print("\nOffline checks failed. Aborting server tests.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Start Basin server
    # -----------------------------------------------------------------------
    print(f"\n--- Starting Basin on port {args.port} ---")
    env = os.environ.copy()
    env["BASIN_BACKEND"] = "helix_fused"
    env["HELIX_MODEL_DIR"] = str(Path.home() / "models" / "falcon-h1-0.5b-base")
    env["HELIX_MODEL_ID"] = "falcon_h1"
    if args.strict:
        env["HELIX_STRICT_BENCHMARK"] = "1"

    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.basin_server:app",
         "--port", str(args.port), "--host", "127.0.0.1"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        print(f"  Server PID: {server_proc.pid}")
        if not _wait_healthy(args.port, timeout=60):
            stdout = server_proc.stdout.read().decode() if server_proc.stdout else ""
            print(f"  FAIL: server did not become healthy\n{stdout[-2000:]}")
            sys.exit(1)
        print("  Server healthy")

        all_receipts = []
        all_failures = []
        models_tested = ["falcon_h1"]

        # -------------------------------------------------------------------
        # Task 1: Startup receipt
        # -------------------------------------------------------------------
        print("\n--- Task 1: Startup receipt ---")
        ok1, msg1, startup_receipt = test_startup_receipt(args.port)
        print(f"  {'PASS' if ok1 else 'FAIL'}: {msg1}")

        # -------------------------------------------------------------------
        # Task 2: Single request + receipt validation
        # -------------------------------------------------------------------
        print("\n--- Task 2: Per-request receipt persistence ---")
        ok2, msg2, single_receipt = test_single_request(args.port)
        print(f"  {'PASS' if ok2 else 'FAIL'}: {msg2}")
        if single_receipt:
            all_receipts.append(single_receipt)

        # -------------------------------------------------------------------
        # Task 3: Strict mode E2E (only if --strict)
        # -------------------------------------------------------------------
        print("\n--- Task 3: Strict mode E2E ---")
        if args.strict:
            ok3, msg3, strict_receipt = test_strict_mode_success(args.port)
            print(f"  {'PASS' if ok3 else 'FAIL'}: {msg3}")
            if strict_receipt:
                all_receipts.append(strict_receipt)
        else:
            ok3 = True
            msg3 = "SKIP (run with --strict to test)"
            print(f"  {msg3}")

        # -------------------------------------------------------------------
        # Task 4: Repeated runs
        # -------------------------------------------------------------------
        print("\n--- Task 4: Repeated-run smoke (5 requests) ---")
        ok4, msg4, repeated_receipts = test_repeated_runs(args.port, n=5)
        print(f"  {'PASS' if ok4 else 'FAIL'}: {msg4}")
        all_receipts.extend(repeated_receipts)

    finally:
        # Kill the server
        print(f"\n--- Stopping Basin (PID {server_proc.pid}) ---")
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("  Server stopped")

    # -----------------------------------------------------------------------
    # Task 7: Summary receipt
    # -----------------------------------------------------------------------
    wall_time = time.time() - t_start
    all_ok = all([ok1, ok2, ok3, ok4, ok5, ok6])

    # Build failures list from failed tests
    test_results = [
        (1, ok1, msg1), (2, ok2, msg2), (3, ok3, msg3),
        (4, ok4, msg4), (5, ok5, msg5), (6, ok6, msg6),
    ]
    for task_num, ok, msg in test_results:
        if not ok:
            all_failures.append({"request_index": task_num, "reason": msg})

    from helix_substrate.basin_runtime import (
        build_hardening_summary,
        validate_hardening_summary,
    )
    summary = build_hardening_summary(
        startup_receipt=startup_receipt or {"backend": "helix_fused", "strict_mode": args.strict},
        request_receipts=all_receipts,
        failures=all_failures,
        models_tested=models_tested,
        wall_time_s=wall_time,
    )

    # WO-BASIN-RECEIPT-SEMANTICS-01: Validate the summary itself
    summary_issues = validate_hardening_summary(summary)
    if summary_issues:
        print(f"\n  SUMMARY SEMANTIC FAIL: {summary_issues}")
        all_ok = False

    # Persist summary
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    receipt_dir = PROJECT_ROOT / "receipts" / "hardening" / date_str
    receipt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = receipt_dir / "hardening_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("HARDENING SUMMARY")
    print("=" * 70)
    print(f"  Result:          {'ALL PASS' if all_ok else 'FAILURES DETECTED'}")
    print(f"  Requests:        {summary['request_count']}")
    print(f"  Successes:       {summary['success_count']}")
    print(f"  Fused:           {summary['fused_count']}")
    print(f"  Naive:           {summary['naive_count']}")
    print(f"  Mean decode:     {summary['mean_decode_tok_s']} tok/s")
    print(f"  Mean prefill:    {summary['mean_prefill_tok_s']} tok/s")
    print(f"  Latency spread:  {summary['latency_spread']}")
    print(f"  Failures:        {summary['failure_reasons'] or 'none'}")
    print(f"  Models tested:   {summary['models_tested']}")
    print(f"  Wall time:       {summary['cost']['wall_time_s']}s")
    print(f"  Receipt:         {summary_path}")
    if summary_issues:
        print(f"  Semantic issues: {summary_issues}")
    print("=" * 70)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
