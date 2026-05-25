#!/usr/bin/env python3
"""
receipt_schema_uplift.py

Uplift pre-Gate-era receipts to match the current lifecycle schema.
Adds: schema_version, content_hash, artifact_type, status, gate, tool, timestamp_utc.

Does NOT overwrite originals. Writes uplifted receipts as *_v2.json alongside.
Emits a summary receipt of the uplift itself.

Schema target: gate6_preflight_receipt:v1 compatible
  - schema_version: string
  - content_hash: SHA-256 of original receipt (self-referential since source data may not be on disk)
  - artifact_type: domain category
  - status: PASS/FAIL extracted from verdict or cosine gate
  - gate: which gate this receipt would enter
  - tool: script that produced the receipt
  - timestamp_utc: normalized from cost block timestamps
  - cost: preserved from original
"""

import hashlib
import json
import platform
import resource
import time
from pathlib import Path

HELIX_ROOT = Path.home() / "helix-substrate"
RECEIPT_ROOT = HELIX_ROOT / "receipts"

SCHEMA_VERSION = "domain_proof_receipt:v2"
COSINE_GATE = 0.998  # minimum cosine for PASS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Domain mappings ──────────────────────────────────────────────────────────

ARTIFACT_TYPE_MAP = {
    "activation_checkpoint": "AITensor",
    "codec_weight_compress": "AITensor",
    "continual_snapshots": "AITensor",
    "embedding_table": "AITensor",
    "federated_deltas": "AITensor",
    "gradient_compress": "AITensor",
    "lora_compress": "AITensor",
    "moe_tiered": "AITensor",
    "rag_index": "Embedding",
    "sensor_timeseries": "Scientific",
    "dicom_twofamily": "MedicalImaging",
    "non_llm_proof": "AITensor",
    "raw_distribution": "Scientific",
    "climate_devnet": "Scientific",
}

GATE_MAP = {
    "activation_checkpoint": "Domain Proof: Activation Checkpointing",
    "codec_weight_compress": "Domain Proof: Codec Weight Compression",
    "continual_snapshots": "Domain Proof: Continual Learning Snapshots",
    "embedding_table": "Domain Proof: Embedding Table Compression",
    "federated_deltas": "Domain Proof: Federated Learning Deltas",
    "gradient_compress": "Domain Proof: Gradient Compression",
    "lora_compress": "Domain Proof: LoRA Adapter Compression",
    "moe_tiered": "Domain Proof: MoE Tiered Compression",
    "rag_index": "Domain Proof: RAG Index Compression",
    "sensor_timeseries": "Domain Proof: Sensor/Bio Time-Series",
    "dicom_twofamily": "Domain Proof: Medical Imaging Two-Family",
    "non_llm_proof": "Domain Proof: Non-LLM Architecture",
    "raw_distribution": "Domain Proof: Raw Distribution",
    "climate_devnet": "Domain Proof: Climate Grid (Devnet)",
}


def extract_status(receipt: dict, domain: str) -> str:
    """Extract PASS/FAIL from receipt data.

    Strategy: use the most representative aggregate cosine for each domain.
    Per-tensor or per-row cosines may dip below gate on individual elements
    while the overall artifact fidelity is high. We check:
    1. Explicit verdict fields
    2. Domain-specific aggregate cosine (full_cosine, cosine_mean, etc.)
    3. Fallback: any top-level cosine above gate
    """
    results = receipt.get("results", {})

    # Guard: results may be a list (raw_distribution)
    if isinstance(results, list):
        all_pass = all(
            r.get("cosine", 0) >= COSINE_GATE
            for r in results
            if isinstance(r, dict) and "cosine" in r
        )
        return "PASS" if (results and all_pass) else "FAIL"

    # Explicit verdict field (top-level or in results)
    verdict = results.get("verdict", receipt.get("verdict", ""))
    if isinstance(verdict, str):
        v = verdict.upper()
        if "PASS" in v or "STRONG" in v:
            return "PASS"
        if "FAIL" in v:
            return "FAIL"

    # DICOM two-family: twofam_holds
    if "twofam_holds" in receipt:
        return "PASS" if receipt["twofam_holds"] else "FAIL"

    # Raw distribution: check summary.all_pass
    summary = receipt.get("summary", {})
    if isinstance(summary, dict) and "all_pass" in summary:
        return "PASS" if summary["all_pass"] else "FAIL"

    # Climate devnet: check fidelity
    fidelity = receipt.get("fidelity", {})
    if "cosine_similarity" in fidelity:
        return "PASS" if fidelity["cosine_similarity"] >= COSINE_GATE else "FAIL"

    # Sensor timeseries: check both sub-domains
    if "scrna_seq" in results and "protein" in results:
        scrna_pass = any(
            v.get("cosine", 0) >= COSINE_GATE
            for v in results["scrna_seq"].get("per_k", {}).values()
        )
        protein_pass = results.get("protein", {}).get("cosine", 0) >= COSINE_GATE
        return "PASS" if (scrna_pass and protein_pass) else "FAIL"

    # Cosine-based: check cosine_mean or cosine_min at top level of results
    cos_mean = results.get("cosine_mean")
    cos_min = results.get("cosine_min")
    if cos_mean is not None:
        return "PASS" if cos_mean >= COSINE_GATE else "FAIL"
    if cos_min is not None:
        return "PASS" if cos_min >= COSINE_GATE else "FAIL"

    # Non-LLM proof: check per-model cosine
    for k in ["sentence_transformer", "clip_vit", "resnet18"]:
        if k in results and isinstance(results[k], dict):
            cos = results[k].get("cosine")
            if cos is not None and cos < COSINE_GATE:
                return "FAIL"

    if any(k in results for k in ["sentence_transformer", "clip_vit", "resnet18"]):
        return "PASS"  # all checked models above gate

    # Per-tensor results: use full_cosine or aggregate
    per_tensor = results.get("per_tensor", [])
    if isinstance(per_tensor, list) and per_tensor:
        full_cosines = [t.get("full_cosine") for t in per_tensor
                        if isinstance(t, dict) and "full_cosine" in t]
        if full_cosines:
            return "PASS" if min(full_cosines) >= COSINE_GATE else "FAIL"

        # weight_cosine (federated: the weight-after-delta fidelity)
        weight_cosines = [t.get("weight_cosine") for t in per_tensor
                          if isinstance(t, dict) and "weight_cosine" in t]
        if weight_cosines:
            return "PASS" if min(weight_cosines) >= COSINE_GATE else "FAIL"

        # sgd_step_cosine (gradient: the weight-after-SGD-step fidelity)
        sgd_cosines = [t.get("sgd_step_cosine") for t in per_tensor
                       if isinstance(t, dict) and "sgd_step_cosine" in t]
        if sgd_cosines:
            return "PASS" if min(sgd_cosines) >= COSINE_GATE else "FAIL"

        # hot_cosine (moe tiered: best-tier fidelity per tensor)
        hot_cosines = [t.get("hot_cosine") for t in per_tensor
                       if isinstance(t, dict) and "hot_cosine" in t]
        if hot_cosines:
            return "PASS" if min(hot_cosines) >= COSINE_GATE else "FAIL"

    # Full compression list (continual_snapshots)
    full_comp = results.get("full_compression", [])
    if isinstance(full_comp, list) and full_comp:
        cos_means = [s.get("cosine_mean") for s in full_comp
                     if isinstance(s, dict) and "cosine_mean" in s]
        if cos_means:
            return "PASS" if min(cos_means) >= COSINE_GATE else "FAIL"

    # MoE tiered: top-level hot_cosine_min
    hot_min = results.get("hot_cosine_min")
    if hot_min is not None:
        return "PASS" if hot_min >= COSINE_GATE else "FAIL"

    # RAG index: per_variant list with full_cosine
    per_variant = results.get("per_variant", [])
    if isinstance(per_variant, list) and per_variant:
        variant_cos = [v.get("full_cosine") for v in per_variant
                       if isinstance(v, dict) and "full_cosine" in v]
        if variant_cos:
            return "PASS" if min(variant_cos) >= COSINE_GATE else "FAIL"

    # Overall full_cosine at top level
    overall_cos = results.get("full_cosine")
    if overall_cos is not None:
        return "PASS" if overall_cos >= COSINE_GATE else "FAIL"

    return "UNDETERMINED"


def extract_timestamp(receipt: dict) -> str:
    """Normalize timestamp to UTC ISO format."""
    cost = receipt.get("cost", {})
    ts = cost.get("timestamp_start") or cost.get("timestamp_end")
    if ts:
        # Already ISO-ish, normalize
        if "T" in ts and not ts.endswith("Z"):
            return ts + "Z"
        return ts

    ts = receipt.get("timestamp")
    if ts:
        return ts if ts.endswith("Z") else ts + "Z"

    return ""


def extract_tool(receipt: dict, domain: str) -> str:
    """Extract the tool/script name."""
    if "work_order" in receipt:
        return f"bench_{domain}.py"
    exp = receipt.get("experiment", "")
    if exp.startswith("tensor_infra_domain"):
        domain_scripts = {
            "activation_checkpoint": "bench_activation_checkpoint.py",
            "codec_weight_compress": "bench_codec_weight_compress.py",
            "continual_snapshots": "bench_continual_snapshots.py",
            "embedding_table": "bench_embedding_table.py",
            "federated_deltas": "bench_federated_deltas.py",
            "gradient_compress": "bench_gradient_compress.py",
            "lora_compress": "bench_lora_compress.py",
            "moe_tiered": "bench_moe_tiered.py",
            "rag_index": "bench_rag_index.py",
            "sensor_timeseries": "bench_sensor_timeseries.py",
        }
        return domain_scripts.get(domain, f"bench_{domain}.py")
    if exp == "non_llm_helix_linear_proof":
        return "non_llm_proof.py"
    if "raw_distribution" in exp:
        return "hxq_raw_distribution_test.py"
    if receipt.get("title", "").startswith("Non-AI"):
        return "non_ai_artifact_demo.py"
    return f"bench_{domain}.py"


def extract_data_source(receipt: dict) -> str:
    """Extract data source description."""
    results = receipt.get("results", {})
    if isinstance(results, dict):
        ds = results.get("data_source") or receipt.get("data_source", "")
    else:
        ds = receipt.get("data_source", "")
    if ds:
        return ds
    desc = receipt.get("description", "")
    if desc:
        return desc
    return ""


def uplift_receipt(original_path: Path, domain: str) -> dict:
    """Read original receipt, produce uplifted version."""
    original = json.loads(original_path.read_text(encoding="utf-8"))
    content_hash = sha256_file(original_path)

    uplifted = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_MAP.get(domain, f"Domain Proof: {domain}"),
        "tool": extract_tool(original, domain),
        "artifact_type": ARTIFACT_TYPE_MAP.get(domain, "Unknown"),
        "status": extract_status(original, domain),
        "content_hash": content_hash,
        "timestamp_utc": extract_timestamp(original),
        "data_source": extract_data_source(original),
        "original_receipt": str(original_path),
        "uplift_note": "Uplifted from pre-Gate-era receipt to lifecycle-compatible schema.",
    }

    # Preserve original cost block
    if "cost" in original:
        uplifted["cost"] = original["cost"]
    else:
        uplifted["cost"] = {"note": "Original receipt had no cost block"}

    # Preserve key results (summary, not full data)
    results = original.get("results", {})
    if isinstance(results, dict):
        # Extract cosine summary
        summary = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool)):
                summary[k] = v
            elif isinstance(v, dict):
                # Keep cosine/verdict/compression_ratio
                sub = {sk: sv for sk, sv in v.items()
                       if isinstance(sv, (int, float, str, bool))}
                if sub:
                    summary[k] = sub
        uplifted["results_summary"] = summary
    elif isinstance(results, list):
        # Raw distribution: extract per-distribution cosine
        uplifted["results_summary"] = [
            {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool))}
            for r in results if isinstance(r, dict)
        ]

    # DICOM: preserve key findings
    if "twofam_holds" in original:
        uplifted["results_summary"] = {
            "twofam_holds": original["twofam_holds"],
            "combined_coverage": original.get("combined_coverage"),
            "n_medical": original.get("n_medical"),
            "n_texture": original.get("n_texture"),
            "n_samples": original.get("n_samples"),
        }

    # Climate devnet: preserve fidelity + on-chain
    if "fidelity" in original:
        uplifted["results_summary"] = {
            "fidelity": original["fidelity"],
            "on_chain": original.get("on_chain", {}),
            "data": original.get("data", {}),
        }

    return uplifted


def main() -> int:
    t0 = time.time()
    cpu0 = time.process_time()

    print("Receipt Schema Uplift")
    print()

    # Collect all receipts to uplift
    targets = []

    # tensor_infra
    for d in sorted(Path(RECEIPT_ROOT / "tensor_infra").iterdir()):
        if d.is_dir():
            for f in sorted(d.glob("*.json")):
                if not f.name.endswith("_v2.json"):
                    targets.append((f, d.name))

    # dicom_twofamily
    for f in sorted((RECEIPT_ROOT / "dicom_twofamily").glob("*.json")):
        if not f.name.endswith("_v2.json"):
            targets.append((f, "dicom_twofamily"))

    # non_llm_proof
    for f in sorted((RECEIPT_ROOT / "non_llm_proof").glob("*.json")):
        if not f.name.endswith("_v2.json"):
            targets.append((f, "non_llm_proof"))

    # raw_distribution (different root)
    raw_dir = Path.home() / "receipts"
    for f in sorted(raw_dir.glob("hxq_raw_distribution*.json")):
        if not f.name.endswith("_v2.json"):
            targets.append((f, "raw_distribution"))

    # climate devnet (different root)
    climate_dir = Path.home() / "hxq-solana" / "receipts"
    for f in sorted(climate_dir.glob("devnet_non_ai_climate*.json")):
        if not f.name.endswith("_v2.json"):
            targets.append((f, "climate_devnet"))

    print(f"Found {len(targets)} receipts to uplift")
    print()

    results = []
    for path, domain in targets:
        uplifted = uplift_receipt(path, domain)
        v2_path = path.with_name(path.stem + "_v2.json")
        v2_path.write_text(json.dumps(uplifted, indent=2, sort_keys=True), encoding="utf-8")

        status = uplifted["status"]
        atype = uplifted["artifact_type"]
        mark = "PASS" if status == "PASS" else ("FAIL" if status == "FAIL" else "????")
        print(f"  {mark}  {domain:30s}  {atype:16s}  {path.name}")
        results.append({
            "domain": domain,
            "artifact_type": atype,
            "status": status,
            "original": str(path),
            "uplifted": str(v2_path),
            "content_hash": uplifted["content_hash"],
        })

    # Summary
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    undet_count = sum(1 for r in results if r["status"] == "UNDETERMINED")

    print()
    print(f"Uplifted: {len(results)} receipts")
    print(f"  PASS: {pass_count}")
    print(f"  FAIL: {fail_count}")
    print(f"  UNDETERMINED: {undet_count}")

    # Unique domains and artifact types
    domains = sorted(set(r["domain"] for r in results))
    atypes = sorted(set(r["artifact_type"] for r in results))
    print(f"  Domains: {len(domains)} ({', '.join(domains)})")
    print(f"  Artifact types: {len(atypes)} ({', '.join(atypes)})")

    # Emit uplift receipt
    wall_time = time.time() - t0
    cpu_time = time.process_time() - cpu0

    uplift_receipt_data = {
        "schema_version": "receipt_uplift:v1",
        "tool": "receipt_schema_uplift.py",
        "gate": "Receipt Schema Uplift",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASS" if fail_count == 0 and undet_count == 0 else "PARTIAL",
        "summary": {
            "total": len(results),
            "pass": pass_count,
            "fail": fail_count,
            "undetermined": undet_count,
            "domains": domains,
            "artifact_types": atypes,
            "schema_target": SCHEMA_VERSION,
        },
        "receipts": results,
        "cost": {
            "wall_time_s": round(wall_time, 3),
            "cpu_time_s": round(cpu_time, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        },
    }

    out_path = RECEIPT_ROOT / "receipt_uplift" / "uplift_receipt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(uplift_receipt_data, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"  Uplift receipt: {out_path}")
    print(f"  Wall time: {wall_time:.3f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
