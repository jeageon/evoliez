"""Render v4 candidate cards without over-claiming."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from evoliez.ranking.evidence_card_v4 import EvidenceCardV4


def render_candidate_cards(cards: Iterable[EvidenceCardV4]) -> str:
    lines = [
        "# V4 candidate evidence cards",
        "",
        "Claim level: L0 uncalibrated. These cards support assay prioritization and uncertainty review.",
        "",
    ]
    for card in cards:
        lines.extend(
            [
                f"## {card.candidate_id} `{card.mutation}`",
                "",
                f"- role: {card.role}",
                f"- summary: {card.summary_verdict}",
                f"- reaction-geometry support: {card.reaction_geometry_accommodation.score:.3f} "
                f"({card.reaction_geometry_accommodation.confidence})",
                f"- pose uncertainty risk: {card.pose_uncertainty.score:.3f} "
                f"({card.pose_uncertainty.confidence})",
                f"- calibration: {card.experimental_calibration.reason}",
                "",
            ]
        )
    return "\n".join(lines)


def write_candidate_cards(cards: Iterable[EvidenceCardV4], markdown_path: Path, json_path: Path) -> None:
    card_list = list(cards)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_candidate_cards(card_list))
    json_path.write_text(json.dumps([c.model_dump() for c in card_list], indent=2, sort_keys=True))
