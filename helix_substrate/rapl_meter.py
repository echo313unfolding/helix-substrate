"""
CPU energy measurement via Intel RAPL (Running Average Power Limit).

Usage as context manager::

    from helix_substrate.rapl_meter import RaplMeter

    with RaplMeter() as meter:
        # ... work ...
        pass

    if meter.available:
        receipt["cost"]["energy_joules"] = meter.joules

If RAPL is unavailable (no Intel CPU, no permissions), ``meter.available``
is False and ``meter.joules`` is None -- callers never need to guard imports.

Ported from echo-box/tools/rapl_meter.py (2025-11).
"""

from pathlib import Path


class RaplMeter:
    """Context manager for measuring CPU energy via RAPL."""

    def __init__(self):
        self.available = False
        self.joules = None
        self._energy_file = None
        self._start_uj = None

        rapl_base = Path("/sys/class/powercap/intel-rapl")
        if rapl_base.exists():
            for pkg in rapl_base.glob("intel-rapl:*/"):
                name_file = pkg / "name"
                if name_file.exists() and name_file.read_text().strip() == "package-0":
                    energy_file = pkg / "energy_uj"
                    if energy_file.exists():
                        self._energy_file = energy_file
                        self.available = True
                        break

    def __enter__(self):
        if self.available:
            try:
                self._start_uj = int(self._energy_file.read_text())
            except Exception:
                self.available = False
        return self

    def __exit__(self, *args):
        if self.available:
            try:
                end_uj = int(self._energy_file.read_text())
                # Handle 48-bit counter wraparound
                if end_uj < self._start_uj:
                    end_uj += 1 << 48
                self.joules = (end_uj - self._start_uj) / 1e6
            except Exception:
                self.available = False
                self.joules = None
