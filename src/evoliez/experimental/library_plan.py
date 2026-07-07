"""V4 first-round experimental library planner."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from evoliez.experimental.deconvolution import deconvolution_set
from evoliez.experimental.seed_manifest import SeedManifest
from evoliez.ranking.candidate_accommodation import CandidateAccommodationScore
from evoliez.ranking.evidence_card_v4 import EvidenceCardV4

from .controls import (
    BASELINE_VARIANTS,
    LOW_ML_UNCERTAINTY_CONTROLS,
    NO_GAIN_CONTROLS,
    SCALAR_CONTROLS,
)


CSV_FIELDS = [
    "plate",
    "well",
    "variant_id",
    "candidate_id",
    "mutation",
    "lane",
    "all_lanes",
    "why_include",
    "risk",
    "expected_information_gain",
    "assay_priority",
    "codon_design_note",
]


@dataclass
class LibraryVariant:
    variant_id: str
    candidate_id: str
    mutation: str
    lane: str
    all_lanes: List[str]
    why_include: str
    risk: str
    expected_information_gain: float
    assay_priority: int
    plate: str = "P1"
    well: str = ""
    codon_design_note: str = "codon optimization pending"

    def as_row(self) -> Dict[str, str]:
        return {
            "plate": self.plate,
            "well": self.well,
            "variant_id": self.variant_id,
            "candidate_id": self.candidate_id,
            "mutation": self.mutation,
            "lane": self.lane,
            "all_lanes": ";".join(self.all_lanes),
            "why_include": self.why_include,
            "risk": self.risk,
            "expected_information_gain": f"{self.expected_information_gain:.3f}",
            "assay_priority": str(self.assay_priority),
            "codon_design_note": self.codon_design_note,
        }


def _well(index: int) -> str:
    row = chr(ord("A") + (index // 12))
    col = index % 12 + 1
    return f"{row}{col}"


def _candidate_for_mutation(manifest: SeedManifest, mutation: str):
    return manifest.find_by_mutation(mutation)


def reject_final_score_only(rows: Sequence[Dict[str, object]]) -> None:
    for row in rows:
        keys = {k for k, v in row.items() if v not in (None, "", [])}
        if keys - {"candidate_id", "mutation", "final_score"}:
            return
    raise ValueError("v4 library planning rejects final_score-only candidate input")


def _add_variant(
    rows: List[LibraryVariant],
    seen: Dict[str, LibraryVariant],
    *,
    mutation: str,
    lane: str,
    why: str,
    risk: str,
    info_gain: float,
    priority: int,
    candidate_id: Optional[str] = None,
) -> None:
    key = mutation
    if key in seen:
        if lane not in seen[key].all_lanes:
            seen[key].all_lanes.append(lane)
        return
    cid = candidate_id or mutation.replace(";", "_").replace(" ", "_")
    variant = LibraryVariant(
        variant_id=f"v4_{len(rows) + 1:03d}",
        candidate_id=cid,
        mutation=mutation,
        lane=lane,
        all_lanes=[lane],
        why_include=why,
        risk=risk,
        expected_information_gain=info_gain,
        assay_priority=priority,
    )
    rows.append(variant)
    seen[key] = variant


def build_first_round_library(
    manifest: SeedManifest,
    cards: Optional[Dict[str, EvidenceCardV4]] = None,
    accommodation_scores: Optional[Dict[str, CandidateAccommodationScore]] = None,
    *,
    size: int = 32,
) -> List[LibraryVariant]:
    if size not in (20, 32, 96):
        raise ValueError("first-round v4 library size must be 20, 32, or 96")
    rows: List[LibraryVariant] = []
    seen: Dict[str, LibraryVariant] = {}

    for mutation in BASELINE_VARIANTS:
        _add_variant(
            rows,
            seen,
            mutation=mutation,
            lane="baseline",
            why="assay baseline and normalization anchor",
            risk="baseline control",
            info_gain=1.0,
            priority=1,
            candidate_id="WT",
        )

    lead = manifest.find_by_mutation("I208T;R207K;R228P")
    if lead is None:
        raise ValueError("manifest missing I208T;R207K;R228P lead")
    _add_variant(
        rows,
        seen,
        mutation=lead.mutation,
        lane="tier_A_lead",
        why="v2/v3 supported catalytic hypothesis seed",
        risk="uncalibrated reaction-geometry confidence",
        info_gain=0.98,
        priority=1,
        candidate_id=lead.candidate_id,
    )
    for mutation in deconvolution_set(lead.mutation):
        seed = _candidate_for_mutation(manifest, mutation)
        _add_variant(
            rows,
            seen,
            mutation=mutation,
            lane="lead_deconvolution",
            why="maps epistasis in the multi-point lead",
            risk="may lose combination-dependent support",
            info_gain=0.90,
            priority=2,
            candidate_id=seed.candidate_id if seed else mutation,
        )

    for c in manifest.candidates:
        if c.role == "v2_positive_probe":
            _add_variant(
                rows,
                seen,
                mutation=c.mutation,
                lane="v2_positive_probe",
                why="tests whether v2 functional-state signal recurs",
                risk="not confirmed by v3",
                info_gain=0.78,
                priority=3,
                candidate_id=c.candidate_id,
            )

    for mutation in SCALAR_CONTROLS:
        seed = _candidate_for_mutation(manifest, mutation)
        _add_variant(
            rows,
            seen,
            mutation=mutation,
            lane="scalar_rank_control",
            why="contrasts scalar structural ranking against catalytic accommodation",
            risk="should not be promoted by final_score alone",
            info_gain=0.74,
            priority=4,
            candidate_id=seed.candidate_id if seed else mutation,
        )

    for mutation in NO_GAIN_CONTROLS:
        seed = _candidate_for_mutation(manifest, mutation)
        _add_variant(
            rows,
            seen,
            mutation=mutation,
            lane="no_gain_control",
            why="binding-stable or historical no-gain control lane",
            risk="control, not a catalytic hypothesis",
            info_gain=0.70,
            priority=5,
            candidate_id=seed.candidate_id if seed else mutation,
        )

    for mutation in LOW_ML_UNCERTAINTY_CONTROLS:
        seed = _candidate_for_mutation(manifest, mutation)
        _add_variant(
            rows,
            seen,
            mutation=mutation,
            lane="uncertainty_probe",
            why="preserves low-ML and pose-uncertainty false-negative lane",
            risk="high uncertainty by design",
            info_gain=0.68,
            priority=6,
            candidate_id=seed.candidate_id if seed else mutation,
        )

    filler = [
        ("M261L", "diversity_probe"),
        ("Q382S", "scalar_neighborhood_probe"),
        ("Q382Y", "scalar_neighborhood_probe"),
        ("R228K", "cofactor_contact_probe"),
        ("Q382N", "scalar_neighborhood_probe"),
        ("M261V", "diversity_probe"),
        ("R207Q", "cofactor_contact_probe"),
        ("M261C", "low_ml_uncertainty_probe"),
        ("N288S", "diversity_probe"),
        ("P262Y", "diversity_probe"),
        ("Q382E", "scalar_neighborhood_probe"),
        ("R228H", "cofactor_contact_probe"),
        ("N288V", "diversity_probe"),
        ("R207A", "cofactor_contact_probe"),
    ]
    for mutation, lane in filler:
        if len(rows) >= size:
            break
        _add_variant(
            rows,
            seen,
            mutation=mutation,
            lane=lane,
            why="fills plate while preserving diversity and uncertainty coverage",
            risk="screening-level support only",
            info_gain=0.45,
            priority=7,
            candidate_id=mutation,
        )

    if len(rows) < size:
        raise ValueError(f"unable to build {size}-variant library; only {len(rows)} variants")
    rows = rows[:size]
    for i, row in enumerate(rows):
        row.well = _well(i)
    required = set(manifest.selection_policy.required_lanes)
    present = {r.lane for r in rows} | {lane for r in rows for lane in r.all_lanes}
    missing = required - present
    if missing:
        raise ValueError("v4 library missing required lanes: " + ", ".join(sorted(missing)))
    return rows


def write_library_csv(rows: Iterable[LibraryVariant], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())
    return path


def render_library_markdown(rows: Iterable[LibraryVariant]) -> str:
    lines = [
        "# V4 first-round FDH library",
        "",
        "Claim level: L0 uncalibrated. This library prioritizes variants for assay and preserves controls.",
        "",
        "| Well | Mutation | Lane | Rationale | Risk |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.well} | `{row.mutation}` | {row.lane} | "
            f"{row.why_include} | {row.risk} |"
        )
    return "\n".join(lines) + "\n"


def write_library_markdown(rows: Iterable[LibraryVariant], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_library_markdown(rows))
    return path
