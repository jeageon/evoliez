#!/usr/bin/env python
"""Regenerate the s08 family-specific reranker HTML report for a FINISHED run from
its on-disk provenance -- WITHOUT re-running s08.

Thin wrapper around ``evoliez.io.rerank_report.write_rerank_report`` -- the SAME
builder the s08/s09 stages call on completion. The "folded" / "reach MD" cross-refs
are filled in once s09 has run (validated_candidates.json); before that the report
still shows the full ranking + scores + features.

Usage:  python scripts/gen_s08_report.py [RUN_DIR] [OUT_HTML]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evoliez.io.rerank_report import write_rerank_report  # noqa: E402

RD = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/runs/fdh_5track")
OUT = sys.argv[2] if len(sys.argv) > 2 else None
out = write_rerank_report(RD, OUT)
print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
