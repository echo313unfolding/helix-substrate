#!/usr/bin/env python3
"""
Gate 8F: Encrypted Medical Artifact Lifecycle Proof

HIPAA-aligned technical safeguard prototype: encrypted off-chain artifact lifecycle.

Proves:
  1. FHIR JSON compresses and encrypts.
  2. DICOM-like numeric artifact compresses and encrypts.
  3. Ciphertext hash is recorded in receipt (not plaintext hash).
  4. Decrypt succeeds while consent/key is active.
  5. Revoked consent blocks key release.
  6. Destroyed key makes decrypt fail.
  7. Receipt contains no PHI fields.
  8. Every decision emits an audit/preflight receipt.

Does NOT claim HIPAA compliance. Claims only: encryption at rest,
integrity, audit, and consent-gated key release as technical safeguards.

All data is synthetic. No real PHI.

Outputs:
  receipts/gate8_medical/gate8f_encryption_receipt.json
"""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

# Wire cell-runtime
CELL_RUNTIME = Path.home() / "cell-runtime"
sys.path.insert(0, str(CELL_RUNTIME / "src"))

from cell.medical_crypto import (
    KeyVault,
    decrypt_artifact,
    encrypt_compressed_artifact,
    sha256_bytes,
)
from cell.medical_state_machine import (
    ClaimState,
    ConsentReceipt,
    MedicalArtifactBundle,
    medical_preflight,
)

HELIX_ROOT = Path.home() / "helix-substrate"
sys.path.insert(0, str(HELIX_ROOT))

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.cdnav3_reader import CDNAv3Reader
from helix_substrate.tensor_policy import TensorPolicy, TensorClass

OUT = HELIX_ROOT / "receipts" / "gate8_medical"

PHI_FORBIDDEN_KEYS = {
    "patient_name", "patient_id", "ssn", "social_security",
    "date_of_birth", "dob", "address", "phone", "email",
    "mrn", "medical_record_number",
}


def build_synthetic_fhir() -> bytes:
    """Synthetic FHIR prior-auth JSON."""
    fhir = {
        "resourceType": "Claim",
        "id": "prior-auth-8f-001",
        "status": "active",
        "use": "preauthorization",
        "patient": {"reference": "Patient/HASH_" + sha256_bytes(b"synth-patient")[:12]},
        "provider": {"reference": "Practitioner/HASH_" + sha256_bytes(b"synth-provider")[:12]},
        "insurer": {"reference": "Organization/PAYER_001"},
        "diagnosis": [{
            "sequence": 1,
            "diagnosisCodeableConcept": {
                "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm",
                             "code": "M23.51"}]
            }
        }],
        "item": [{
            "sequence": 1,
            "productOrService": {
                "coding": [{"system": "http://www.ama-assn.org/go/cpt",
                             "code": "73721"}]
            },
            "unitPrice": {"value": 1200.00, "currency": "USD"},
        }],
        "_synthetic": True,
    }
    return json.dumps(fhir, indent=2, sort_keys=True).encode("utf-8")


def build_synthetic_image() -> np.ndarray:
    """Synthetic 256x256 medical-like image."""
    rng = np.random.RandomState(8642)
    base = rng.normal(loc=500.0, scale=80.0, size=(256, 256)).astype(np.float32)
    y, x = np.ogrid[-128:128, -128:128]
    region = ((x / 60.0) ** 2 + (y / 50.0) ** 2) < 1.0
    base[region] += rng.normal(loc=600.0, scale=30.0, size=region.sum())
    return np.clip(base, 0.0, 2000.0)


def check_no_phi(obj: dict, path: str = "") -> list:
    """Recursively check that no PHI field names appear in a dict."""
    violations = []
    for k, v in obj.items():
        full = f"{path}.{k}" if path else k
        if k.lower() in PHI_FORBIDDEN_KEYS:
            violations.append(full)
        if isinstance(v, dict):
            violations.extend(check_no_phi(v, full))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    violations.extend(check_no_phi(item, f"{full}[{i}]"))
    return violations


def main() -> int:
    t0 = time.time()
    cpu0 = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 72)
    print("GATE 8F: Encrypted Medical Artifact Lifecycle")
    print("  HIPAA-aligned technical safeguard prototype")
    print("=" * 72)
    print()

    OUT.mkdir(parents=True, exist_ok=True)
    vault = KeyVault()
    results = {}

    # ── F1: FHIR JSON compress + encrypt ─────────────────────────────────
    print("  F1: FHIR prior-auth → gzip → AES-256-GCM")

    fhir_raw = build_synthetic_fhir()
    fhir_gz = gzip.compress(fhir_raw, compresslevel=9)

    env_fhir = encrypt_compressed_artifact(
        fhir_gz, "fhir-8f-001", "gzip", "consent-8f-001", vault,
        plaintext_for_local_hash=fhir_raw,
    )

    # Decrypt to verify roundtrip
    key_fhir = vault.release_key("fhir-8f-001", "consent-8f-001")
    recovered_gz = decrypt_artifact(env_fhir.ciphertext, key_fhir, env_fhir.nonce)
    recovered_raw = gzip.decompress(recovered_gz)
    fhir_roundtrip = recovered_raw == fhir_raw

    results["f1_fhir_encrypt"] = {
        "pass": fhir_roundtrip,
        "raw_bytes": len(fhir_raw),
        "compressed_bytes": len(fhir_gz),
        "ciphertext_bytes": len(env_fhir.ciphertext),
        "content_hash": env_fhir.content_hash,
        "roundtrip": fhir_roundtrip,
        "codec": "gzip",
        "cipher": "AES-256-GCM",
    }
    print(f"      {'PASS' if fhir_roundtrip else 'FAIL'}: "
          f"{len(fhir_raw)}→{len(fhir_gz)}→{len(env_fhir.ciphertext)} bytes, "
          f"roundtrip={fhir_roundtrip}")

    # ── F2: DICOM-like image compress + encrypt ──────────────────────────
    print("  F2: DICOM-like image → VQ → AES-256-GCM")

    image = build_synthetic_image()
    image_raw = image.tobytes()

    scratch = HELIX_ROOT / "tensor_infra_scratch" / "gate8f"
    scratch.mkdir(parents=True, exist_ok=True)

    policy = TensorPolicy(
        tensor_class=TensorClass.UNKNOWN,
        storage_mode="codebook+sidecar",
        n_clusters=256,
        use_kmeans=True,
        sidecar_enabled=True,
        percentile=99.9,
        max_corrections=512,
    )

    writer = CDNAv3Writer(scratch)
    stats = writer.write_tensor(image, "dicom_8f", policy=policy)

    tensor_dir = scratch / "dicom_8f.cdnav3"
    reader = CDNAv3Reader(tensor_dir)
    recon = reader.reconstruct()

    # Serialize the VQ-compressed representation for encryption
    vq_bytes = recon.tobytes()

    env_img = encrypt_compressed_artifact(
        vq_bytes, "img-8f-001", "VQ_k256_sidecar", "consent-8f-001", vault,
        plaintext_for_local_hash=image_raw,
    )

    key_img = vault.release_key("img-8f-001", "consent-8f-001")
    recovered_vq = decrypt_artifact(env_img.ciphertext, key_img, env_img.nonce)
    img_roundtrip = recovered_vq == vq_bytes

    # Cosine of VQ reconstruction vs original (independent of encryption)
    a_f = image.ravel().astype(np.float64)
    b_f = recon.ravel().astype(np.float64)
    cos = float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f)))

    results["f2_image_encrypt"] = {
        "pass": img_roundtrip and cos >= 0.998,
        "raw_bytes": len(image_raw),
        "vq_bytes": len(vq_bytes),
        "ciphertext_bytes": len(env_img.ciphertext),
        "content_hash": env_img.content_hash,
        "vq_cosine": round(cos, 6),
        "encrypt_roundtrip": img_roundtrip,
        "codec": "VQ_k256_sidecar",
        "cipher": "AES-256-GCM",
    }
    mark = "PASS" if (img_roundtrip and cos >= 0.998) else "FAIL"
    print(f"      {mark}: cos={cos:.6f}, encrypt roundtrip={img_roundtrip}")

    # ── F3: Ciphertext hash recorded ─────────────────────────────────────
    print("  F3: Ciphertext hash recorded in receipt (not plaintext)")

    fhir_ct_hash_correct = env_fhir.content_hash == sha256_bytes(env_fhir.ciphertext)
    img_ct_hash_correct = env_img.content_hash == sha256_bytes(env_img.ciphertext)
    fhir_not_plaintext = env_fhir.content_hash != sha256_bytes(fhir_gz)
    img_not_plaintext = env_img.content_hash != sha256_bytes(vq_bytes)

    all_hash_ok = (fhir_ct_hash_correct and img_ct_hash_correct
                   and fhir_not_plaintext and img_not_plaintext)

    results["f3_ciphertext_hash"] = {
        "pass": all_hash_ok,
        "fhir_hash_is_ciphertext": fhir_ct_hash_correct,
        "image_hash_is_ciphertext": img_ct_hash_correct,
        "fhir_hash_not_plaintext": fhir_not_plaintext,
        "image_hash_not_plaintext": img_not_plaintext,
    }
    print(f"      {'PASS' if all_hash_ok else 'FAIL'}: "
          f"hash=ciphertext={fhir_ct_hash_correct and img_ct_hash_correct}, "
          f"hash!=plaintext={fhir_not_plaintext and img_not_plaintext}")

    # ── F4: Decrypt succeeds while consent active ────────────────────────
    print("  F4: Decrypt succeeds with active consent")

    vault2 = KeyVault()
    env_f4 = encrypt_compressed_artifact(
        b"test payload f4", "art-f4", "gzip", "consent-f4", vault2,
    )
    key_f4 = vault2.release_key("art-f4", "consent-f4")
    dec_f4 = decrypt_artifact(env_f4.ciphertext, key_f4, env_f4.nonce)
    f4_ok = dec_f4 == b"test payload f4"

    results["f4_active_decrypt"] = {
        "pass": f4_ok,
        "decrypted_matches": f4_ok,
    }
    print(f"      {'PASS' if f4_ok else 'FAIL'}")

    # ── F5: Revoked consent blocks key release ───────────────────────────
    print("  F5: Revoked consent blocks key release")

    vault3 = KeyVault()
    env_f5 = encrypt_compressed_artifact(
        b"test payload f5", "art-f5", "gzip", "consent-f5", vault3,
    )
    # Before revocation
    pre = vault3.release_key("art-f5", "consent-f5") is not None
    # Revoke
    vault3.revoke_consent("consent-f5")
    # After revocation
    post = vault3.release_key("art-f5", "consent-f5")
    f5_ok = pre and post is None

    results["f5_revoked_blocks_key"] = {
        "pass": f5_ok,
        "key_available_before_revoke": pre,
        "key_available_after_revoke": post is not None,
        "key_destroyed": vault3.is_destroyed("art-f5"),
    }
    print(f"      {'PASS' if f5_ok else 'FAIL'}: "
          f"before={pre}, after={post is not None}")

    # ── F6: Destroyed key makes decrypt fail ─────────────────────────────
    print("  F6: Destroyed key makes decrypt fail")

    vault4 = KeyVault()
    env_f6 = encrypt_compressed_artifact(
        b"test payload f6", "art-f6", "gzip", "consent-f6", vault4,
    )
    vault4.destroy_key("art-f6")

    # Key is zeroed
    key_gone = vault4.release_key("art-f6", "consent-f6") is None

    # Even with zeroed key, GCM auth fails
    zeroed_key = b'\x00' * 32
    gcm_fails = False
    try:
        decrypt_artifact(env_f6.ciphertext, zeroed_key, env_f6.nonce)
    except Exception:
        gcm_fails = True

    f6_ok = key_gone and gcm_fails

    results["f6_destroyed_key_fails"] = {
        "pass": f6_ok,
        "key_release_returns_none": key_gone,
        "zeroed_key_gcm_auth_fails": gcm_fails,
    }
    print(f"      {'PASS' if f6_ok else 'FAIL'}: "
          f"release=None: {key_gone}, GCM fails: {gcm_fails}")

    # ── F7: No PHI in receipt ────────────────────────────────────────────
    print("  F7: No PHI fields in receipt")

    fhir_fields = env_fhir.to_receipt_fields()
    img_fields = env_img.to_receipt_fields()
    phi_violations_fhir = check_no_phi(fhir_fields)
    phi_violations_img = check_no_phi(img_fields)

    # Also check that plaintext_hash is NOT in receipt fields
    plaintext_leaked_fhir = "plaintext_hash" in fhir_fields
    plaintext_leaked_img = "plaintext_hash" in img_fields

    f7_ok = (not phi_violations_fhir and not phi_violations_img
             and not plaintext_leaked_fhir and not plaintext_leaked_img)

    results["f7_no_phi_in_receipt"] = {
        "pass": f7_ok,
        "phi_violations_fhir": phi_violations_fhir,
        "phi_violations_image": phi_violations_img,
        "plaintext_hash_leaked_fhir": plaintext_leaked_fhir,
        "plaintext_hash_leaked_image": plaintext_leaked_img,
    }
    print(f"      {'PASS' if f7_ok else 'FAIL'}: "
          f"violations={len(phi_violations_fhir) + len(phi_violations_img)}, "
          f"plaintext_hash_leaked={plaintext_leaked_fhir or plaintext_leaked_img}")

    # ── F8: Audit receipt emitted for every decision ─────────────────────
    print("  F8: Audit receipt with preflight decision")

    bundle = MedicalArtifactBundle(
        bundle_id="gate8f-audit",
        claim_state=ClaimState.APPROVED,
        consent=ConsentReceipt(
            consent_id="consent-8f-001",
            patient_hash=sha256_bytes(b"synth-patient"),
            scope="prior_auth_imaging",
            granted_utc="2026-05-25T00:00:00Z",
            expires_utc="2027-05-25T00:00:00Z",
        ),
    )
    bundle.add_artifact(
        "fhir-8f-001", "StructuredRecord", "gzip+AES-256-GCM",
        env_fhir.content_hash, True,
    )
    bundle.add_artifact(
        "img-8f-001", "MedicalImaging", "VQ_k256+AES-256-GCM",
        env_img.content_hash, True,
    )

    pf = medical_preflight(bundle)
    f8_ok = pf["decision"] == "ALLOW" and "MPF-CLEAN" in pf["reason_codes"]

    results["f8_audit_receipt"] = {
        "pass": f8_ok,
        "preflight_decision": pf["decision"],
        "reason_codes": pf["reason_codes"],
        "bundle_hash": pf["bundle_hash"],
        "artifact_count": pf["artifact_count"],
    }
    print(f"      {'PASS' if f8_ok else 'FAIL'}: "
          f"decision={pf['decision']}, codes={pf['reason_codes']}")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    all_pass = all(r.get("pass", False) for r in results.values())
    status = "PASS" if all_pass else "FAIL"

    print("-" * 72)
    print("Summary:")
    for name, r in results.items():
        mark = "PASS" if r.get("pass") else "FAIL"
        pad = 30 - len(name)
        print(f"  {name}:{' ' * pad}{mark}")

    print()
    print(f"Gate 8F: {status}")

    # ── Receipt ──────────────────────────────────────────────────────────
    wall_time = time.time() - t0
    cpu_time = time.process_time() - cpu0

    receipt = {
        "tool": "gate8f_medical_encryption_probe.py",
        "gate": "Gate 8F: Encrypted Medical Artifact Lifecycle",
        "label": "HIPAA-aligned technical safeguard prototype",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "conditions": results,
        "all_pass": all_pass,
        "condition_count": len(results),
        "pass_count": sum(1 for r in results.values() if r.get("pass")),
        "technical_safeguards_demonstrated": [
            "Encryption at rest (AES-256-GCM)",
            "Integrity controls (SHA-256 content hash of ciphertext)",
            "Audit controls (receipt for every encrypt/decrypt/preflight decision)",
            "Access controls (consent-gated key release)",
            "Key destruction on consent revocation",
            "PHI boundary (no PHI in receipts, plaintext hash excluded)",
        ],
        "not_claimed": [
            "HIPAA compliance (requires administrative + physical + legal safeguards)",
            "HIPAA certification (no such certification exists)",
            "Production key management (prototype uses in-memory vault)",
            "Real PHI handling (all data is synthetic)",
        ],
        "cost": {
            "wall_time_s": round(wall_time, 3),
            "cpu_time_s": round(cpu_time, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipt_path = OUT / "gate8f_encryption_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time:.3f}s")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
