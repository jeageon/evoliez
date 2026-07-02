#!/usr/bin/env python3
"""Write the V4 guided-Boltz feasibility report.

The default invocation is conservative and returns NO_GO until a validated
atom map plus differentiable sampler probe are supplied by a real integration.
"""

from __future__ import annotations

from pathlib import Path

from evoliez.guided.feasibility import render_feasibility_report, run_feasibility_gate


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = run_feasibility_gate()
    out = ROOT / "reports" / "v4_guided_boltz_feasibility.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_feasibility_report(result))


if __name__ == "__main__":
    main()
