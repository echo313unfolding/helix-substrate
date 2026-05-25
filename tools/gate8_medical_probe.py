#!/usr/bin/env python3
"""
Gate 8: Medical/Insurance Artifact Lifecycle Proof

Proves the receipt-gated lifecycle works for mixed medical artifact bundles.
Uses synthetic FHIR prior-auth + synthetic DICOM-like imaging arrays.

Sub-gates:
  8A — FHIR prior-auth JSON → gzip → hash → receipt
  8B — DICOM-like numeric pixel array → VQ/HXQ → hash → receipt
  8C — Mixed bundle: FHIR + imaging in one bundle with composite hash
  8D — 14-state claim state machine: valid + invalid transitions receipted
  8E — Preflight: consent gate, PHI boundary, allow/hold/reject decisions

Does NOT use real PHI. All data is synthetic.
Does NOT claim HIPAA compliance. Proves the lifecycle shape fits.

Outputs:
  receipts/gate8_medical/gate8_receipt.json
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

# Wire cell-runtime into path
CELL_RUNTIME = Path.home() / "cell-runtime"
sys.path.insert(0, str(CELL_RUNTIME / "src"))

from cell.medical_state_machine import (
    ClaimState,
    ConsentReceipt,
    MedicalArtifactBundle,
    TransitionResult,
    VALID_TRANSITIONS,
    claim_to_artifact_state,
    medical_preflight,
)

HELIX_ROOT = Path.home() / "helix-substrate"
sys.path.insert(0, str(HELIX_ROOT))

from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.cdnav3_reader import CDNAv3Reader
from helix_substrate.tensor_policy import TensorPolicy, TensorClass

OUT = HELIX_ROOT / "receipts" / "gate8_medical"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    dot = np.dot(a_f, b_f)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0


# ── 8A: Synthetic FHIR Prior-Auth ────────────────────────────────────────────

def build_synthetic_fhir_prior_auth() -> dict:
    """Create a structurally representative FHIR PriorAuthorizationRequest."""
    return {
        "resourceType": "Claim",
        "id": "prior-auth-mri-knee-001",
        "status": "active",
        "type": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/claim-type",
                "code": "institutional",
            }]
        },
        "use": "preauthorization",
        "patient": {"reference": "Patient/HASH_a1b2c3d4"},
        "created": "2026-05-25",
        "provider": {"reference": "Practitioner/HASH_e5f6g7h8"},
        "insurer": {"reference": "Organization/PAYER_001"},
        "priority": {"coding": [{"code": "normal"}]},
        "diagnosis": [{
            "sequence": 1,
            "diagnosisCodeableConcept": {
                "coding": [{
                    "system": "http://hl7.org/fhir/sid/icd-10-cm",
                    "code": "M23.51",
                    "display": "Internal derangement of knee, right",
                }]
            }
        }],
        "procedure": [{
            "sequence": 1,
            "procedureCodeableConcept": {
                "coding": [{
                    "system": "http://www.ama-assn.org/go/cpt",
                    "code": "73721",
                    "display": "MRI knee without contrast",
                }]
            }
        }],
        "insurance": [{
            "sequence": 1,
            "focal": True,
            "coverage": {"reference": "Coverage/HASH_i9j0k1l2"},
        }],
        "item": [{
            "sequence": 1,
            "productOrService": {
                "coding": [{
                    "system": "http://www.ama-assn.org/go/cpt",
                    "code": "73721",
                }]
            },
            "quantity": {"value": 1},
            "unitPrice": {"value": 1200.00, "currency": "USD"},
        }],
        "_synthetic": True,
        "_phi_note": "All patient/provider references are SHA-256 hashes, not identifiers",
    }


def gate_8a() -> dict:
    """8A: FHIR prior-auth → gzip → hash → receipt."""
    print("  8A: FHIR prior-auth JSON → gzip → hash")

    fhir = build_synthetic_fhir_prior_auth()
    fhir_json = json.dumps(fhir, indent=2, sort_keys=True).encode("utf-8")
    fhir_gz = gzip.compress(fhir_json, compresslevel=9)

    raw_hash = sha256_bytes(fhir_json)
    gz_hash = sha256_bytes(fhir_gz)

    ratio = len(fhir_json) / len(fhir_gz)

    # Verify roundtrip
    rt = gzip.decompress(fhir_gz)
    rt_hash = sha256_bytes(rt)
    roundtrip_ok = rt_hash == raw_hash

    result = {
        "pass": roundtrip_ok,
        "artifact_type": "StructuredRecord",
        "codec": "gzip",
        "raw_bytes": len(fhir_json),
        "compressed_bytes": len(fhir_gz),
        "compression_ratio": round(ratio, 2),
        "content_hash_raw": raw_hash,
        "content_hash_gz": gz_hash,
        "roundtrip_hash_match": roundtrip_ok,
        "fhir_resource_type": fhir["resourceType"],
        "fhir_use": fhir["use"],
        "phi_boundary": "All patient/provider refs are SHA-256 hashes",
    }
    mark = "PASS" if roundtrip_ok else "FAIL"
    print(f"      {mark}: ratio={ratio:.2f}x, roundtrip={roundtrip_ok}")
    return result


# ── 8B: Synthetic DICOM-like Imaging ─────────────────────────────────────────

def build_synthetic_medical_image() -> np.ndarray:
    """Create a synthetic DICOM-like 512x512 pixel array.

    Uses structured noise that resembles medical imaging characteristics:
    soft tissue background + higher-intensity regions + noise.
    NOT a real medical image. Structurally representative for codec testing.
    """
    rng = np.random.RandomState(42)

    # Soft tissue background (Gaussian, centered, moderate range)
    base = rng.normal(loc=500.0, scale=80.0, size=(512, 512)).astype(np.float32)

    # Anatomical structure (elliptical regions with different intensities)
    y, x = np.ogrid[-256:256, -256:256]

    # Bone-like region (high intensity ellipse)
    bone = ((x / 100.0) ** 2 + (y / 80.0) ** 2) < 1.0
    base[bone] += rng.normal(loc=800.0, scale=40.0, size=bone.sum())

    # Fluid-like region (low intensity circle)
    fluid = ((x + 60) ** 2 + (y - 40) ** 2) < 30 ** 2
    base[fluid] = rng.normal(loc=200.0, scale=20.0, size=fluid.sum())

    # Imaging noise (Poisson-like)
    base += rng.normal(loc=0, scale=15.0, size=base.shape).astype(np.float32)

    # Clip to valid range
    base = np.clip(base, 0.0, 2000.0)

    return base


def gate_8b() -> dict:
    """8B: DICOM-like pixel array → VQ → hash → receipt."""
    print("  8B: DICOM-like image → VQ codec → hash")

    image = build_synthetic_medical_image()
    raw_hash = sha256_bytes(image.tobytes())

    # Compress with VQ (k=256, same as domain proofs)
    scratch = HELIX_ROOT / "tensor_infra_scratch" / "gate8_medical"
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
    stats = writer.write_tensor(image, "dicom_like_mri", policy=policy)

    tensor_dir = scratch / "dicom_like_mri.cdnav3"
    reader = CDNAv3Reader(tensor_dir)
    recon = reader.reconstruct()

    cos = cosine_sim(image, recon)
    compressed_hash = sha256_bytes(recon.tobytes())

    gate = cos >= 0.998

    result = {
        "pass": gate,
        "artifact_type": "MedicalImaging",
        "codec": "VQ_k256_sidecar",
        "shape": list(image.shape),
        "raw_hash": raw_hash,
        "compressed_hash": compressed_hash,
        "cosine": round(cos, 6),
        "cosine_gate": 0.998,
        "compression_ratio": stats.get("compression_ratio", 0),
        "synthetic": True,
        "note": "Structured noise proxy, NOT real DICOM",
    }
    mark = "PASS" if gate else "FAIL"
    print(f"      {mark}: cos={cos:.6f}, gate=0.998")
    return result


# ── 8C: Mixed Bundle ─────────────────────────────────────────────────────────

def gate_8c(fhir_hash: str, image_hash: str) -> dict:
    """8C: Mixed bundle with FHIR + imaging, composite hash."""
    print("  8C: Mixed artifact bundle (FHIR + imaging)")

    bundle = MedicalArtifactBundle(
        bundle_id="gate8-mixed-bundle-001",
        consent=ConsentReceipt(
            consent_id="consent-001",
            patient_hash=sha256_bytes(b"synthetic-patient-id-001"),
            scope="prior_auth_imaging",
            granted_utc="2026-05-25T00:00:00Z",
            expires_utc="2027-05-25T00:00:00Z",
        ),
    )

    bundle.add_artifact(
        artifact_id="fhir-prior-auth-001",
        artifact_type="StructuredRecord",
        codec="gzip",
        content_hash=fhir_hash,
        phi_fields_hashed=True,
    )

    bundle.add_artifact(
        artifact_id="mri-knee-001",
        artifact_type="MedicalImaging",
        codec="VQ_k256_sidecar",
        content_hash=image_hash,
        phi_fields_hashed=True,
    )

    composite = bundle.bundle_hash()
    has_both = len(bundle.artifacts) == 2
    codecs = {a["codec"] for a in bundle.artifacts.values()}
    mixed_codec = len(codecs) == 2

    result = {
        "pass": has_both and mixed_codec,
        "bundle_id": bundle.bundle_id,
        "artifact_count": len(bundle.artifacts),
        "codecs": sorted(codecs),
        "mixed_codec": mixed_codec,
        "composite_hash": composite,
        "consent_valid": bundle.consent.is_valid(),
        "bundle_state": bundle.to_dict(),
    }
    mark = "PASS" if result["pass"] else "FAIL"
    print(f"      {mark}: {len(bundle.artifacts)} artifacts, codecs={sorted(codecs)}")
    return result


# ── 8D: Claim State Machine ─────────────────────────────────────────────────

def gate_8d() -> dict:
    """8D: 14-state claim state machine with valid + invalid transitions."""
    print("  8D: Claim state machine (14 states)")

    results = {}

    # ── D1: Happy path — Draft → Submitted → Acknowledged → In_Review → Approved
    bundle = MedicalArtifactBundle(
        bundle_id="gate8d-happy-path",
        consent=ConsentReceipt(
            consent_id="consent-d1",
            patient_hash=sha256_bytes(b"patient-d1"),
            scope="claim_review",
            granted_utc="2026-05-25T00:00:00Z",
        ),
    )
    bundle.add_artifact("a1", "StructuredRecord", "gzip",
                        sha256_bytes(b"fhir-d1"), True)

    happy_path = [
        (ClaimState.SUBMITTED, "Provider submitted"),
        (ClaimState.ACKNOWLEDGED, "Payer acknowledged"),
        (ClaimState.IN_REVIEW, "Clinical review started"),
        (ClaimState.APPROVED, "Claim approved"),
    ]
    all_ok = True
    transitions = []
    for target, reason in happy_path:
        r = bundle.transition(target, reason=reason, actor="test")
        transitions.append(r.to_dict())
        if not r.allowed:
            all_ok = False

    final_artifact_state = claim_to_artifact_state(bundle.claim_state)
    results["happy_path"] = {
        "pass": all_ok and final_artifact_state == "Active",
        "final_claim_state": bundle.claim_state.value,
        "final_artifact_state": final_artifact_state,
        "transition_count": len(transitions),
        "transitions": transitions,
    }

    # ── D2: Denial + appeal path
    bundle2 = MedicalArtifactBundle(
        bundle_id="gate8d-appeal-path",
        consent=ConsentReceipt(
            consent_id="consent-d2",
            patient_hash=sha256_bytes(b"patient-d2"),
            scope="claim_review",
            granted_utc="2026-05-25T00:00:00Z",
        ),
    )
    bundle2.add_artifact("a2", "StructuredRecord", "gzip",
                         sha256_bytes(b"fhir-d2"), True)

    appeal_path = [
        (ClaimState.SUBMITTED, "Submitted"),
        (ClaimState.ACKNOWLEDGED, "Acknowledged"),
        (ClaimState.IN_REVIEW, "Review"),
        (ClaimState.DENIED, "Denied — medical necessity not met"),
        (ClaimState.APPEALED, "Provider appealed"),
        (ClaimState.APPEAL_IN_REVIEW, "Appeal under review"),
        (ClaimState.APPEAL_DECIDED, "Appeal upheld — approved"),
    ]
    all_ok2 = True
    transitions2 = []
    for target, reason in appeal_path:
        r = bundle2.transition(target, reason=reason, actor="test")
        transitions2.append(r.to_dict())
        if not r.allowed:
            all_ok2 = False

    results["appeal_path"] = {
        "pass": all_ok2 and claim_to_artifact_state(bundle2.claim_state) == "Active",
        "final_claim_state": bundle2.claim_state.value,
        "final_artifact_state": claim_to_artifact_state(bundle2.claim_state),
        "transition_count": len(transitions2),
    }

    # ── D3: Info request loop
    bundle3 = MedicalArtifactBundle(
        bundle_id="gate8d-info-loop",
        consent=ConsentReceipt(
            consent_id="consent-d3",
            patient_hash=sha256_bytes(b"patient-d3"),
            scope="claim_review",
            granted_utc="2026-05-25T00:00:00Z",
        ),
    )
    bundle3.add_artifact("a3", "StructuredRecord", "gzip",
                         sha256_bytes(b"fhir-d3"), True)

    info_path = [
        (ClaimState.SUBMITTED, "Submitted"),
        (ClaimState.ACKNOWLEDGED, "Acknowledged"),
        (ClaimState.PENDED, "Pended for clinical review"),
        (ClaimState.INFO_REQUESTED, "Additional imaging requested"),
        (ClaimState.INFO_RECEIVED, "Provider sent MRI report"),
        (ClaimState.IN_REVIEW, "Review with new info"),
        (ClaimState.PRE_APPROVED, "Conditionally approved"),
        (ClaimState.APPROVED, "Final approval"),
    ]
    all_ok3 = True
    transitions3 = []
    for target, reason in info_path:
        r = bundle3.transition(target, reason=reason, actor="test")
        transitions3.append(r.to_dict())
        if not r.allowed:
            all_ok3 = False

    results["info_request_loop"] = {
        "pass": all_ok3 and claim_to_artifact_state(bundle3.claim_state) == "Active",
        "final_claim_state": bundle3.claim_state.value,
        "final_artifact_state": claim_to_artifact_state(bundle3.claim_state),
        "transition_count": len(transitions3),
    }

    # ── D4: Invalid transition — Draft → Approved (skip steps)
    bundle4 = MedicalArtifactBundle(bundle_id="gate8d-invalid")
    bundle4.add_artifact("a4", "StructuredRecord", "gzip",
                         sha256_bytes(b"fhir-d4"), True)

    r_invalid = bundle4.transition(ClaimState.APPROVED, reason="skip", actor="test")
    results["invalid_transition_rejected"] = {
        "pass": not r_invalid.allowed and "MS-INVALID-TRANSITION" in r_invalid.reason_codes,
        "from_state": r_invalid.from_state,
        "to_state": r_invalid.to_state,
        "reason_codes": r_invalid.reason_codes,
    }

    # ── D5: Quarantine is absorbing
    bundle5 = MedicalArtifactBundle(bundle_id="gate8d-quarantine-absorb")
    bundle5.add_artifact("a5", "StructuredRecord", "gzip",
                         sha256_bytes(b"fhir-d5"), True)
    bundle5.transition(ClaimState.SUBMITTED, reason="submit")
    bundle5.transition(ClaimState.QUARANTINED, reason="fraud detected", actor="compliance")
    r_escape = bundle5.transition(ClaimState.ACKNOWLEDGED, reason="try to escape")

    results["quarantine_absorbing"] = {
        "pass": not r_escape.allowed and "MS-INVALID-TRANSITION" in r_escape.reason_codes,
        "claim_state": bundle5.claim_state.value,
        "artifact_state": claim_to_artifact_state(bundle5.claim_state),
        "reason_codes": r_escape.reason_codes,
    }

    # ── D6: Transition coverage — every state reachable
    reachable = set()
    to_visit = [ClaimState.DRAFT]
    while to_visit:
        s = to_visit.pop()
        if s in reachable:
            continue
        reachable.add(s)
        for target in VALID_TRANSITIONS.get(s, set()):
            if target not in reachable:
                to_visit.append(target)

    all_states = set(ClaimState)
    results["all_states_reachable"] = {
        "pass": reachable == all_states,
        "reachable_count": len(reachable),
        "total_states": len(all_states),
        "unreachable": [s.value for s in all_states - reachable],
    }

    all_pass = all(r.get("pass", False) for r in results.values())
    print(f"      {'PASS' if all_pass else 'FAIL'}: {sum(1 for r in results.values() if r.get('pass'))}/{len(results)} conditions")

    return {"pass": all_pass, "conditions": results}


# ── 8E: Preflight (consent + PHI + decisions) ───────────────────────────────

def gate_8e() -> dict:
    """8E: Medical preflight — consent gate, PHI boundary, decisions."""
    print("  8E: Medical preflight (consent + PHI + decisions)")

    results = {}

    # ── E1: Active bundle with valid consent → ALLOW
    bundle_ok = MedicalArtifactBundle(
        bundle_id="gate8e-allow",
        claim_state=ClaimState.APPROVED,
        consent=ConsentReceipt(
            consent_id="consent-e1",
            patient_hash=sha256_bytes(b"patient-e1"),
            scope="claim_review",
            granted_utc="2026-05-25T00:00:00Z",
            expires_utc="2027-05-25T00:00:00Z",
        ),
    )
    bundle_ok.add_artifact("fhir-e1", "StructuredRecord", "gzip",
                           sha256_bytes(b"fhir-e1"), True)
    bundle_ok.add_artifact("img-e1", "MedicalImaging", "VQ_k256",
                           sha256_bytes(b"img-e1"), True)

    pf = medical_preflight(bundle_ok)
    results["active_consent_allow"] = {
        "pass": pf["decision"] == "ALLOW" and "MPF-CLEAN" in pf["reason_codes"],
        "decision": pf["decision"],
        "reason_codes": pf["reason_codes"],
    }

    # ── E2: No consent → REJECT
    bundle_no_consent = MedicalArtifactBundle(
        bundle_id="gate8e-no-consent",
        claim_state=ClaimState.APPROVED,
    )
    bundle_no_consent.add_artifact("fhir-e2", "StructuredRecord", "gzip",
                                   sha256_bytes(b"fhir-e2"), True)

    pf2 = medical_preflight(bundle_no_consent)
    results["no_consent_reject"] = {
        "pass": pf2["decision"] == "REJECT" and "MPF-NO-CONSENT" in pf2["reason_codes"],
        "decision": pf2["decision"],
        "reason_codes": pf2["reason_codes"],
    }

    # ── E3: Revoked consent → REJECT
    bundle_revoked = MedicalArtifactBundle(
        bundle_id="gate8e-revoked",
        claim_state=ClaimState.APPROVED,
        consent=ConsentReceipt(
            consent_id="consent-e3",
            patient_hash=sha256_bytes(b"patient-e3"),
            scope="claim_review",
            granted_utc="2026-05-25T00:00:00Z",
            revoked=True,
            revoked_utc="2026-05-25T12:00:00Z",
        ),
    )
    bundle_revoked.add_artifact("fhir-e3", "StructuredRecord", "gzip",
                                sha256_bytes(b"fhir-e3"), True)

    pf3 = medical_preflight(bundle_revoked)
    results["revoked_consent_reject"] = {
        "pass": pf3["decision"] == "REJECT" and "MPF-CONSENT-INVALID" in pf3["reason_codes"],
        "decision": pf3["decision"],
        "reason_codes": pf3["reason_codes"],
    }

    # ── E4: Candidate claim (not yet approved) → HOLD
    bundle_candidate = MedicalArtifactBundle(
        bundle_id="gate8e-candidate",
        claim_state=ClaimState.IN_REVIEW,
        consent=ConsentReceipt(
            consent_id="consent-e4",
            patient_hash=sha256_bytes(b"patient-e4"),
            scope="claim_review",
            granted_utc="2026-05-25T00:00:00Z",
        ),
    )
    bundle_candidate.add_artifact("fhir-e4", "StructuredRecord", "gzip",
                                  sha256_bytes(b"fhir-e4"), True)

    pf4 = medical_preflight(bundle_candidate)
    results["candidate_hold"] = {
        "pass": pf4["decision"] == "HOLD" and "MPF-NOT-ACTIVE" in pf4["reason_codes"],
        "decision": pf4["decision"],
        "reason_codes": pf4["reason_codes"],
    }

    # ── E5: Quarantined bundle → REJECT
    bundle_q = MedicalArtifactBundle(
        bundle_id="gate8e-quarantined",
        claim_state=ClaimState.QUARANTINED,
        consent=ConsentReceipt(
            consent_id="consent-e5",
            patient_hash=sha256_bytes(b"patient-e5"),
            scope="claim_review",
            granted_utc="2026-05-25T00:00:00Z",
        ),
    )
    bundle_q.add_artifact("fhir-e5", "StructuredRecord", "gzip",
                          sha256_bytes(b"fhir-e5"), True)

    pf5 = medical_preflight(bundle_q)
    results["quarantined_reject"] = {
        "pass": pf5["decision"] == "REJECT" and "MPF-QUARANTINED" in pf5["reason_codes"],
        "decision": pf5["decision"],
        "reason_codes": pf5["reason_codes"],
    }

    # ── E6: Empty bundle → REJECT
    bundle_empty = MedicalArtifactBundle(
        bundle_id="gate8e-empty",
        claim_state=ClaimState.APPROVED,
        consent=ConsentReceipt(
            consent_id="consent-e6",
            patient_hash=sha256_bytes(b"patient-e6"),
            scope="claim_review",
            granted_utc="2026-05-25T00:00:00Z",
        ),
    )

    pf6 = medical_preflight(bundle_empty)
    results["empty_bundle_reject"] = {
        "pass": pf6["decision"] == "REJECT" and "MPF-EMPTY-BUNDLE" in pf6["reason_codes"],
        "decision": pf6["decision"],
        "reason_codes": pf6["reason_codes"],
    }

    # ── E7: PHI boundary violation → REJECT
    bundle_phi = MedicalArtifactBundle(
        bundle_id="gate8e-phi-violation",
        claim_state=ClaimState.APPROVED,
        consent=ConsentReceipt(
            consent_id="consent-e7",
            patient_hash=sha256_bytes(b"patient-e7"),
            scope="claim_review",
            granted_utc="2026-05-25T00:00:00Z",
        ),
    )
    bundle_phi.add_artifact("fhir-e7", "StructuredRecord", "gzip",
                            sha256_bytes(b"fhir-e7"),
                            phi_fields_hashed=False)  # violation

    pf7 = medical_preflight(bundle_phi)
    results["phi_boundary_reject"] = {
        "pass": pf7["decision"] == "REJECT" and "MPF-PHI-BOUNDARY" in pf7["reason_codes"],
        "decision": pf7["decision"],
        "reason_codes": pf7["reason_codes"],
    }

    # ── E8: Consent blocks transition
    bundle_consent_block = MedicalArtifactBundle(
        bundle_id="gate8e-consent-blocks-transition",
        consent=ConsentReceipt(
            consent_id="consent-e8",
            patient_hash=sha256_bytes(b"patient-e8"),
            scope="claim_review",
            granted_utc="2026-05-25T00:00:00Z",
            revoked=True,
            revoked_utc="2026-05-25T06:00:00Z",
        ),
    )
    bundle_consent_block.add_artifact("fhir-e8", "StructuredRecord", "gzip",
                                      sha256_bytes(b"fhir-e8"), True)

    r_blocked = bundle_consent_block.transition(ClaimState.SUBMITTED, reason="try submit")
    results["consent_blocks_transition"] = {
        "pass": not r_blocked.allowed and "MS-CONSENT-REVOKED" in r_blocked.reason_codes,
        "allowed": r_blocked.allowed,
        "reason_codes": r_blocked.reason_codes,
    }

    all_pass = all(r.get("pass", False) for r in results.values())
    print(f"      {'PASS' if all_pass else 'FAIL'}: {sum(1 for r in results.values() if r.get('pass'))}/{len(results)} conditions")

    return {"pass": all_pass, "conditions": results}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    t0 = time.time()
    cpu0 = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 72)
    print("GATE 8: Medical/Insurance Artifact Lifecycle Proof")
    print("=" * 72)
    print()

    OUT.mkdir(parents=True, exist_ok=True)

    # Run sub-gates
    r_8a = gate_8a()
    print()
    r_8b = gate_8b()
    print()

    # 8C needs hashes from 8A and 8B
    fhir_hash = r_8a["content_hash_gz"]
    image_hash = r_8b["compressed_hash"]
    r_8c = gate_8c(fhir_hash, image_hash)
    print()

    r_8d = gate_8d()
    print()
    r_8e = gate_8e()
    print()

    # ── Summary ──────────────────────────────────────────────────────────
    gates = {
        "8A_fhir_prior_auth": r_8a,
        "8B_dicom_like_imaging": r_8b,
        "8C_mixed_bundle": r_8c,
        "8D_claim_state_machine": r_8d,
        "8E_medical_preflight": r_8e,
    }

    all_pass = all(g.get("pass", False) for g in gates.values())
    status = "PASS" if all_pass else "FAIL"

    print("-" * 72)
    print("Summary:")
    for name, g in gates.items():
        mark = "PASS" if g.get("pass") else "FAIL"
        pad = 30 - len(name)
        print(f"  {name}:{' ' * pad}{mark}")

    print()
    print(f"Gate 8: {status}")

    # ── Receipt ──────────────────────────────────────────────────────────
    wall_time = time.time() - t0
    cpu_time = time.process_time() - cpu0

    receipt = {
        "tool": "gate8_medical_probe.py",
        "gate": "Gate 8: Medical/Insurance Artifact Lifecycle",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "sub_gates": gates,
        "all_pass": all_pass,
        "gate_count": len(gates),
        "pass_count": sum(1 for g in gates.values() if g.get("pass")),
        "architecture": {
            "state_machine": "14-state claim lifecycle (medical_state_machine.py)",
            "base_lifecycle": "3-state artifact (Candidate/Active/Quarantined)",
            "codec_families": {
                "structured_text": "gzip (FHIR JSON)",
                "numeric_arrays": "VQ/HXQ (DICOM-like imaging)",
            },
            "consent_gate": "ConsentReceipt with scope/expiry/revocation",
            "phi_boundary": "Only SHA-256 hashes of identifiers; no PHI in receipts or on-chain",
            "preflight_decisions": "ALLOW / HOLD / REJECT with MPF-* reason codes",
        },
        "limitations": [
            "All data is synthetic — no real PHI used",
            "FHIR structure is representative, not fully R4-compliant",
            "Imaging is structured noise, not real DICOM pixel data",
            "HIPAA compliance is a legal question, not proven here",
            "14-state model is a simplification of real payer workflows",
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

    receipt_path = OUT / "gate8_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time:.3f}s")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
