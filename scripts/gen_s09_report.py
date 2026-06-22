#!/usr/bin/env python
"""Regenerate the s09 non-MD validation HTML report for a FINISHED run from its
on-disk provenance -- WITHOUT re-running s09 (no docking, no ThermoMPNN, no GPU).

Thin wrapper around ``evoliez.io.validation_report.write_validation_report`` -- the
SAME builder the s09 stage now calls on completion, so the standalone report is
byte-identical to the auto-generated one.

Usage:  python scripts/gen_s09_report.py [RUN_DIR] [OUT_HTML]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evoliez.io.validation_report import write_validation_report  # noqa: E402

RD = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/runs/fdh_5track")
OUT = sys.argv[2] if len(sys.argv) > 2 else None
out = write_validation_report(RD, OUT)
print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
