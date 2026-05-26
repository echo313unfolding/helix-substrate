#!/usr/bin/env python3
"""
Gate 9: Personal Encryption Agent — Device-Bound Key Proof

Proves:
  9A: Local encrypt/decrypt with device-bound two-factor key
  9B: User-verifiable receipt (ciphertext hash, no PHI, codebook AAD binding)
  9C: Consent-scoped key release (grant/deny access)
  9D: Revoke consent -> decrypt blocked (wrapped DEK destroyed)
  9E: Audit log + human-readable explanation
  9F: Wrong device (simulated different hardware IDs) -> unwrap fails
  9G: Wrong passphrase -> unwrap fails
  9H: Tampered codebook hash (wrong AAD) -> GCM auth fails
  9I: Receipt contains no raw serials (hardware_fingerprint_hash only)

Does NOT claim HIPAA compliance. Claims only: user-controlled local encryption
with device-bound keys, envelope encryption, consent-gated key release,
artifact binding via codebook hash as authenticated data.

All data is synthetic. No real PHI.

Outputs:
  receipts/gate9_personal_agent/gate9_receipt.json
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from pathlib import Path

# Wire cell-runtime
CELL_RUNTIME = Path.home() / "cell-runtime"
sys.path.insert(0, str(CELL_RUNTIME / "src"))

from cell.personal_agent import PersonalEncryptionAgent
from cell.medical_crypto import generate_key, sha256_bytes
from cryptography.exceptions import InvalidTag

HELIX_ROOT = Path.home() / "helix-substrate"
OUT = HELIX_ROOT / "receipts" / "gate9_personal_agent"

# Fixed test hardware IDs
HW_DEVICE_A = {"machine_id": "gate9-device-A", "cpu_model": "test-cpu", "board_serial": "SN-A-001"}
HW_DEVICE_B = {"machine_id": "gate9-device-B", "cpu_model": "test-cpu", "board_serial": "SN-B-002"}
PASSPHRASE = "gate9-test-passphrase"

PHI_FORBIDDEN_KEYS = {
    "patient_name", "patient_id", "ssn", "social_security",
    "date_of_birth", "dob", "address", "phone", "email",
    "mrn", "medical_record_number", "board_serial",
    "product_uuid", "product_serial",
}


def check_no_secrets(obj: dict) -> list:
    """Check that no raw secrets or PHI fields appear in receipt."""
    violations = []
    for k, v in obj.items():
        if k.lower() in PHI_FORBIDDEN_KEYS:
            violations.append(k)
        if isinstance(v, dict):
            violations.extend(check_no_secrets(v))
        elif isinstance(v, str):
            # Check for raw hardware values
            for raw in HW_DEVICE_A.values():
                if raw in v and k != "hardware_fingerprint_hash":
                    violations.append(f"{k} contains raw hardware ID: {raw}")
    return violations


def main() -> int:
    t0 = time.time()
    cpu0 = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 72)
    print("GATE 9: Personal Encryption Agent — Device-Bound Key Proof")
    print("  User-controlled envelope encryption with three-property binding")
    print("=" * 72)
    print()

    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── 9A: Local encrypt/decrypt with device-bound key ──────────────────
    print("  9A: Local encrypt/decrypt with device-bound two-factor key")

    agent_a = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
    test_data = b"Synthetic FHIR prior-auth: CPT 73721, ICD-10 M23.51, $1200.00"
    codebook_hash = sha256_bytes(b"synthetic-codebook-content")

    envelope = agent_a.encrypt_artifact(
        test_data, "gate9-art-001", codec="gzip",
        consent_id="consent-gate9-001", codebook_hash=codebook_hash,
    )
    recovered = agent_a.decrypt_artifact(envelope, consent_id="consent-gate9-001")
    a_ok = recovered == test_data

    results["9a_encrypt_decrypt"] = {
        "pass": a_ok,
        "plaintext_bytes": len(test_data),
        "ciphertext_bytes": len(envelope.ciphertext),
        "roundtrip": a_ok,
        "codec": "gzip",
        "cipher": "AES-256-GCM",
        "device_bound": True,
        "two_factor": True,
    }
    print(f"      {'PASS' if a_ok else 'FAIL'}: {len(test_data)}→{len(envelope.ciphertext)} bytes, roundtrip={a_ok}")

    # ── 9B: Receipt safety — no PHI, codebook AAD binding ────────────────
    print("  9B: User-verifiable receipt (no PHI, codebook AAD binding)")

    receipt_fields = envelope.to_receipt_fields()
    phi_violations = check_no_secrets(receipt_fields)
    has_content_hash = "content_hash" in receipt_fields
    has_codebook_aad = receipt_fields.get("codebook_hash_aad") == codebook_hash
    has_hw_hash = len(receipt_fields.get("hardware_fingerprint_hash", "")) == 64
    no_plaintext_hash = "plaintext_hash" not in receipt_fields
    no_wrapped_dek = "wrapped_dek" not in receipt_fields
    no_salt = "salt" not in receipt_fields

    b_ok = (not phi_violations and has_content_hash and has_codebook_aad
            and has_hw_hash and no_plaintext_hash and no_wrapped_dek and no_salt)

    results["9b_receipt_safety"] = {
        "pass": b_ok,
        "phi_violations": phi_violations,
        "has_content_hash": has_content_hash,
        "codebook_aad_matches": has_codebook_aad,
        "has_hardware_fingerprint_hash": has_hw_hash,
        "no_plaintext_hash": no_plaintext_hash,
        "no_wrapped_dek": no_wrapped_dek,
        "no_salt_in_receipt": no_salt,
    }
    print(f"      {'PASS' if b_ok else 'FAIL'}: violations={len(phi_violations)}, "
          f"AAD={has_codebook_aad}, hw_hash={has_hw_hash}")

    # ── 9C: Consent-scoped key release (grant/deny) ──────────────────────
    print("  9C: Consent-scoped key release (grant/deny)")

    agent_c = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
    env_c = agent_c.encrypt_artifact(
        b"shared artifact", "gate9-art-c", codec="none", consent_id="consent-c",
    )

    requester_key = generate_key()
    grant = agent_c.grant_access("gate9-art-c", "dr-example", "read", requester_key)
    grant_ok = (grant.requester_id == "dr-example"
                and grant.scope == "read"
                and not grant.revoked
                and len(grant.wrapped_dek) > 0)

    # Deny: wrong consent
    deny_ok = False
    try:
        agent_c.decrypt_artifact(env_c, consent_id="wrong-consent")
    except PermissionError:
        deny_ok = True

    c_ok = grant_ok and deny_ok

    results["9c_consent_scope"] = {
        "pass": c_ok,
        "grant_created": grant_ok,
        "wrong_consent_denied": deny_ok,
    }
    print(f"      {'PASS' if c_ok else 'FAIL'}: grant={grant_ok}, deny={deny_ok}")

    # ── 9D: Revoke consent → decrypt blocked ─────────────────────────────
    print("  9D: Revoke consent -> decrypt blocked (DEK destroyed)")

    agent_d = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
    env_d = agent_d.encrypt_artifact(
        b"revocable data", "gate9-art-d", codec="none", consent_id="consent-d",
    )

    # Before revocation: can decrypt
    pre_decrypt = agent_d.decrypt_artifact(env_d, consent_id="consent-d")
    pre_ok = pre_decrypt == b"revocable data"

    # Revoke
    agent_d.revoke_consent("consent-d")

    # After revocation: wrapped DEK zeroed
    dek_zeroed = env_d.wrapped_dek == b'\x00' * len(env_d.wrapped_dek)

    # After revocation: decrypt blocked
    post_blocked = False
    try:
        agent_d.decrypt_artifact(env_d, consent_id="consent-d")
    except PermissionError:
        post_blocked = True

    d_ok = pre_ok and dek_zeroed and post_blocked

    results["9d_revoke_consent"] = {
        "pass": d_ok,
        "pre_revocation_decrypt": pre_ok,
        "wrapped_dek_zeroed": dek_zeroed,
        "post_revocation_blocked": post_blocked,
    }
    print(f"      {'PASS' if d_ok else 'FAIL'}: pre={pre_ok}, zeroed={dek_zeroed}, blocked={post_blocked}")

    # ── 9E: Audit log + explain ──────────────────────────────────────────
    print("  9E: Audit log + human-readable explanation")

    agent_e = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
    agent_e.encrypt_artifact(b"audit data", "gate9-art-e", codec="gzip", consent_id="consent-e")
    env_e = agent_e._envelopes["gate9-art-e"]
    agent_e.decrypt_artifact(env_e, consent_id="consent-e")
    agent_e.grant_access("gate9-art-e", "auditor", "read", generate_key())
    agent_e.revoke_access("gate9-art-e", "auditor")

    log = agent_e.audit_log()
    actions = [e["action"] for e in log]
    log_ok = ("encrypt" in actions and "decrypt" in actions
              and "grant" in actions and "revoke" in actions)

    explanation = agent_e.explain("gate9-art-e")
    explain_ok = ("gate9-art-e" in explanation and "AES-256-GCM" in explanation
                  and "ACTIVE" in explanation)

    e_ok = log_ok and explain_ok and len(log) >= 4

    results["9e_audit_explain"] = {
        "pass": e_ok,
        "log_entry_count": len(log),
        "actions_logged": actions,
        "explanation_contains_id": "gate9-art-e" in explanation,
        "explanation_contains_cipher": "AES-256-GCM" in explanation,
    }
    print(f"      {'PASS' if e_ok else 'FAIL'}: {len(log)} log entries, actions={actions}")

    # ── 9F: Wrong device → unwrap fails ──────────────────────────────────
    print("  9F: Wrong device (different hardware IDs) -> unwrap fails")

    agent_f_real = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
    env_f = agent_f_real.encrypt_artifact(b"device-bound", "gate9-art-f", codec="none")

    agent_f_wrong = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_B)
    f_blocked = False
    try:
        agent_f_wrong.decrypt_artifact(env_f)
    except InvalidTag:
        f_blocked = True

    results["9f_wrong_device"] = {
        "pass": f_blocked,
        "wrong_device_raises_InvalidTag": f_blocked,
    }
    print(f"      {'PASS' if f_blocked else 'FAIL'}: InvalidTag raised={f_blocked}")

    # ── 9G: Wrong passphrase → unwrap fails ──────────────────────────────
    print("  9G: Wrong passphrase -> unwrap fails")

    agent_g_real = PersonalEncryptionAgent("correct-pass", hw_ids=HW_DEVICE_A)
    env_g = agent_g_real.encrypt_artifact(b"passphrase-bound", "gate9-art-g", codec="none")

    agent_g_wrong = PersonalEncryptionAgent("wrong-pass", hw_ids=HW_DEVICE_A)
    g_blocked = False
    try:
        agent_g_wrong.decrypt_artifact(env_g)
    except InvalidTag:
        g_blocked = True

    results["9g_wrong_passphrase"] = {
        "pass": g_blocked,
        "wrong_passphrase_raises_InvalidTag": g_blocked,
    }
    print(f"      {'PASS' if g_blocked else 'FAIL'}: InvalidTag raised={g_blocked}")

    # ── 9H: Tampered codebook hash (wrong AAD) → GCM auth fails ─────────
    print("  9H: Tampered codebook hash (wrong AAD) -> GCM auth fails")

    agent_h = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
    env_h = agent_h.encrypt_artifact(
        b"aad-bound", "gate9-art-h", codec="none",
        codebook_hash="original_codebook_hash_12345",
    )
    # Tamper the AAD
    env_h.codebook_hash_aad = "tampered_codebook_hash_99999"

    h_blocked = False
    try:
        agent_h.decrypt_artifact(env_h)
    except InvalidTag:
        h_blocked = True

    results["9h_tampered_aad"] = {
        "pass": h_blocked,
        "tampered_aad_raises_InvalidTag": h_blocked,
    }
    print(f"      {'PASS' if h_blocked else 'FAIL'}: InvalidTag raised={h_blocked}")

    # ── 9I: Receipt contains no raw serials ──────────────────────────────
    print("  9I: Receipt contains no raw serials (fingerprint hash only)")

    agent_i = PersonalEncryptionAgent(PASSPHRASE, hw_ids=HW_DEVICE_A)
    env_i = agent_i.encrypt_artifact(b"receipt safety", "gate9-art-i", codec="none")
    fields_i = env_i.to_receipt_fields()
    secrets_found = check_no_secrets(fields_i)

    # Also verify no key material in fields
    field_str = json.dumps(fields_i)
    no_key_material = ("wrapped_dek" not in field_str
                       and "device_key" not in field_str
                       and "passphrase" not in field_str)

    i_ok = not secrets_found and no_key_material

    results["9i_no_raw_serials"] = {
        "pass": i_ok,
        "secrets_found": secrets_found,
        "no_key_material_leaked": no_key_material,
        "receipt_fields_present": list(fields_i.keys()),
    }
    print(f"      {'PASS' if i_ok else 'FAIL'}: secrets={len(secrets_found)}, "
          f"key_material_leaked={not no_key_material}")

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

    total = len(results)
    passed = sum(1 for r in results.values() if r.get("pass"))
    print()
    print(f"Gate 9: {status} ({passed}/{total})")

    # ── Receipt ──────────────────────────────────────────────────────────
    wall_time = time.time() - t0
    cpu_time = time.process_time() - cpu0

    receipt = {
        "tool": "gate9_personal_agent_probe.py",
        "gate": "Gate 9: Personal Encryption Agent — Device-Bound Key Proof",
        "label": "User-controlled envelope encryption with three-property binding",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "conditions": results,
        "all_pass": all_pass,
        "condition_count": total,
        "pass_count": passed,
        "technical_safeguards_demonstrated": [
            "Device-bound key derivation (passphrase + hardware fingerprint via HKDF-SHA256)",
            "Envelope encryption (DEK wrapped by device master key, AES-256-GCM)",
            "Artifact binding (codebook hash as AAD in AES-256-GCM)",
            "Consent-gated key release with audit trail",
            "Wrapped DEK destruction on consent revocation",
            "Receipt contains fingerprint hash only (no raw serials, no key material)",
        ],
        "three_property_model": {
            "something_you_know": "passphrase (scrypt KDF)",
            "something_you_have": "device hardware fingerprint (HKDF binding factor)",
            "something_it_is": "artifact codebook hash (AAD in AES-GCM)",
        },
        "not_claimed": [
            "HIPAA compliance (requires administrative + physical + legal safeguards)",
            "Biometric key (hardware fingerprint != biometric)",
            "DNA-based encryption (brand is biological, math is standard crypto)",
            "Unbreakable (standard crypto, standard limitations)",
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

    receipt_path = OUT / "gate9_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time:.3f}s")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
