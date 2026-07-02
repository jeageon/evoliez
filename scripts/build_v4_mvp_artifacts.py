#!/usr/bin/env python3
"""Build deterministic V4 MVP artifacts.

This command is intentionally light: it does not run docking, MD, or guided
Boltz. It freezes v1-v3 evidence, builds ReferenceEnsemble v0 summaries,
renders candidate cards, and writes the first-round library plus L0 calibration
report.
"""

from __future__ import annotations

import json
from pathlib import Path

from evoliez.experimental.calibration import calibrate_assay_results, write_calibration_report
from evoliez.experimental.library_plan import (
    build_first_round_library,
    write_library_csv,
    write_library_markdown,
)
from evoliez.experimental.seed_manifest import freeze_record, load_seed_manifest
from evoliez.guided.ensemble_builder import build_reference_ensemble_v0
from evoliez.guided.ensemble_readout import score_manifest_candidates
from evoliez.guided.feasibility import render_feasibility_report, run_feasibility_gate
from evoliez.provenance.preflight_report import write_preflight_report
from evoliez.ranking.candidate_accommodation import compute_candidate_accommodation
from evoliez.ranking.evidence_card_v4 import build_evidence_card_v4
from evoliez.reports.v4_benchmark_smoke import write_benchmark_smoke_report
from evoliez.reports.v4_candidate_card import write_candidate_cards
from evoliez.reports.v4_geometry_report import render_geometry_report


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    write_preflight_report(ROOT)
    manifest = load_seed_manifest()
    ensemble = build_reference_ensemble_v0(manifest)
    scores = score_manifest_candidates(manifest, ensemble)
    cards = {
        c.candidate_id: build_evidence_card_v4(c, scores[c.candidate_id])
        for c in manifest.candidates
    }
    accommodations = {
        cid: compute_candidate_accommodation(card)
        for cid, card in cards.items()
    }

    reports = ROOT / "reports"
    provenance = reports / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    (provenance / "v4_seed_manifest_freeze.json").write_text(
        json.dumps(freeze_record(), indent=2, sort_keys=True)
    )
    (provenance / "v4_reference_ensemble.json").write_text(
        ensemble.model_dump_json(indent=2)
    )
    (provenance / "v4_accommodation_scores.json").write_text(
        json.dumps({k: v.model_dump() for k, v in accommodations.items()}, indent=2, sort_keys=True)
    )
    (reports / "v4_geometry_report.md").write_text(render_geometry_report(scores))
    write_candidate_cards(
        cards.values(),
        reports / "v4_candidate_cards.md",
        reports / "v4_candidate_cards.json",
    )
    library20 = build_first_round_library(manifest, cards, accommodations, size=20)
    library32 = build_first_round_library(manifest, cards, accommodations, size=32)
    write_library_csv(library20, reports / "v4_first_round_library_20.csv")
    write_library_csv(library32, reports / "v4_first_round_library_32.csv")
    write_library_csv(library32, reports / "v4_first_round_library.csv")
    write_library_markdown(library32, reports / "v4_first_round_library.md")
    calibration = calibrate_assay_results(None)
    write_calibration_report(
        calibration,
        reports / "v4_calibration_report.md",
        reports / "v4_calibration_report.json",
    )
    (reports / "v4_guided_boltz_feasibility.md").write_text(
        render_feasibility_report(run_feasibility_gate())
    )
    write_benchmark_smoke_report(ROOT)


if __name__ == "__main__":
    main()
