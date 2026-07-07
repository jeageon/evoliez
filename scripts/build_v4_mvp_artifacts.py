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

# The V4 MVP scores are hardcoded design fixtures, not computed data.
_SYNTHETIC = {
    "data_provenance": "synthetic_fixture",
    "computed": False,
    "warning": ("V4 MVP artifact — per-mutation geometry/accommodation scores are "
                "HARDCODED fixtures (ensemble_readout._PROFILES), NOT computed from "
                "docking/MD/Boltz/wet-lab. Do NOT present as computed experimental data."),
}
_SYNTHETIC_MD = (
    "> ⚠ **SYNTHETIC — NOT COMPUTED DATA.** The scores in this V4 MVP artifact are "
    "hardcoded design fixtures (`ensemble_readout._PROFILES`), not derived from "
    "docking/MD/Boltz or wet-lab. Do **not** present these numbers as computed "
    "experimental evidence in a paper.\n\n"
)


def _mark_synthetic(paths) -> None:
    """Stamp each generated V4 artifact with an explicit data-provenance marker."""
    for p in paths:
        if not p.exists():
            continue
        try:
            if p.suffix == ".json":
                obj = json.loads(p.read_text())
                wrapped = ({**obj, "_data_provenance": _SYNTHETIC}
                           if isinstance(obj, dict)
                           else {"_data_provenance": _SYNTHETIC, "items": obj})
                p.write_text(json.dumps(wrapped, indent=2, sort_keys=True))
            elif p.suffix == ".md":
                p.write_text(_SYNTHETIC_MD + p.read_text())
            elif p.suffix == ".csv":
                # a leading comment line keeps the marker with the data; readers that
                # feed this to a plate should pass comment='#'. Honest by construction.
                p.write_text("# data_provenance=synthetic_fixture — hardcoded V4 MVP "
                             "fixtures, NOT computed data\n" + p.read_text())
        except Exception:  # noqa: BLE001 — marking must never abort artifact generation
            pass


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

    # ROADMAP_V3 review — the per-mutation geometry/accommodation scores below are
    # HARDCODED fixtures (ensemble_readout._PROFILES), NOT computed from docking/MD/Boltz.
    # Mark every reader-facing V4 artifact with an explicit data-provenance so it can never
    # be mistaken for computed experimental data in a paper.
    _mark_synthetic([
        provenance / "v4_reference_ensemble.json",
        provenance / "v4_accommodation_scores.json",
        reports / "v4_geometry_report.md",
        reports / "v4_candidate_cards.md",
        reports / "v4_candidate_cards.json",
        reports / "v4_first_round_library.csv",
        reports / "v4_first_round_library_20.csv",
        reports / "v4_first_round_library_32.csv",
        reports / "v4_first_round_library.md",
    ])


if __name__ == "__main__":
    main()
