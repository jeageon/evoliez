"""Render v4 soft reaction-geometry summaries."""

from __future__ import annotations

from typing import Dict


def render_geometry_report(candidate_scores: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "# V4 soft reaction-geometry report",
        "",
        "Low reaction-geometry support is reported as uncertainty or low support, not as inactivity.",
        "",
        "| Candidate | Mutation | Soft geometry support | Pose uncertainty |",
        "|---|---|---:|---:|",
    ]
    for cid, rec in sorted(candidate_scores.items()):
        lines.append(
            f"| {cid} | `{rec.get('mutation', '')}` | "
            f"{rec.get('reaction_geometry_accommodation', 0.0):.3f} | "
            f"{rec.get('pose_uncertainty', 0.0):.3f} |"
        )
    return "\n".join(lines) + "\n"
