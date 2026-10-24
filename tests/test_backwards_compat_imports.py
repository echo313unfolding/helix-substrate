"""
Regression tests: 0.3.3 import patterns must keep working in 0.4.0.

These tests use the EXACT import paths that published code (including
downstream users like Pat Valbusa's integration) relied on in 0.3.3.
If any of these break, the 0.4.0 upgrade is not backwards-compatible.

Work Order: WO-040-RELEASE-STEP3
"""

import pytest


# ---------------------------------------------------------------------------
# Test 1: cdna_encoder (renamed to hxq_encoder in 0.4.0)
# ---------------------------------------------------------------------------

def test_import_cdna_encoder_module():
    """from helix_substrate.cdna_encoder import ... must still work."""
    from helix_substrate.cdna_encoder import encode_tensor_to_cdna, decode_cdna_to_tensor
    assert callable(encode_tensor_to_cdna)
    assert callable(decode_cdna_to_tensor)


def test_cdna_encoder_is_same_as_hxq():
    """Shim must re-export the exact same objects, not copies."""
    from helix_substrate.cdna_encoder import encode_tensor_to_cdna as old_enc
    from helix_substrate.hxq_encoder import encode_tensor_to_cdna as new_enc
    assert old_enc is new_enc


# ---------------------------------------------------------------------------
# Test 2: cdnav3_writer (renamed to hxq_writer_v3 in 0.4.0)
# ---------------------------------------------------------------------------

def test_import_cdnav3_writer_module():
    """from helix_substrate.cdnav3_writer import CDNAv3Writer must still work."""
    from helix_substrate.cdnav3_writer import CDNAv3Writer
    assert CDNAv3Writer is not None


def test_cdnav3_writer_is_same_as_hxq():
    """Shim must re-export the exact same class."""
    from helix_substrate.cdnav3_writer import CDNAv3Writer as old
    from helix_substrate.hxq_writer_v3 import CDNAv3Writer as new
    assert old is new


# ---------------------------------------------------------------------------
# Test 3: cdnav3_reader (renamed to hxq_reader_v3 in 0.4.0)
# ---------------------------------------------------------------------------

def test_import_cdnav3_reader_module():
    """from helix_substrate.cdnav3_reader import CDNAv3Reader must still work."""
    from helix_substrate.cdnav3_reader import CDNAv3Reader
    assert CDNAv3Reader is not None


def test_cdnav3_reader_is_same_as_hxq():
    """Shim must re-export the exact same class."""
    from helix_substrate.cdnav3_reader import CDNAv3Reader as old
    from helix_substrate.hxq_reader_v3 import CDNAv3Reader as new
    assert old is new


# ---------------------------------------------------------------------------
# Test 4: cdna_reader (renamed to hxq_reader in 0.4.0)
# ---------------------------------------------------------------------------

def test_import_cdna_reader_module():
    """from helix_substrate.cdna_reader import CDNAv2Reader must still work."""
    from helix_substrate.cdna_reader import CDNAv2Reader
    assert CDNAv2Reader is not None


def test_cdna_reader_is_same_as_hxq():
    """Shim must re-export the exact same class."""
    from helix_substrate.cdna_reader import CDNAv2Reader as old
    from helix_substrate.hxq_reader import CDNAv2Reader as new
    assert old is new


# ---------------------------------------------------------------------------
# Test 5: Top-level package imports (unchanged but verify)
# ---------------------------------------------------------------------------

def test_toplevel_imports():
    """helix_substrate top-level exports must include CDNAv3Writer/Reader."""
    from helix_substrate import CDNAv3Writer, CDNAv3Reader
    from helix_substrate import encode_tensor_to_cdna, decode_cdna_to_tensor
    assert all(callable(f) for f in [encode_tensor_to_cdna, decode_cdna_to_tensor])
    assert CDNAv3Writer is not None
    assert CDNAv3Reader is not None
