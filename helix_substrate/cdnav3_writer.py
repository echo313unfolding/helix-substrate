"""Backwards-compatibility shim: cdnav3_writer → hxq_writer_v3 (renamed in 0.4.0)."""
from helix_substrate.hxq_writer_v3 import *  # noqa: F401,F403
from helix_substrate.hxq_writer_v3 import CDNAv3Writer
