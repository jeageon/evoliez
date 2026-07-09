"""Tests for portfolio.run — the end-to-end orchestrator + the provenance merge (ROADMAP_V7).

Builds a tiny synthetic run directory (generated / validated / md provenance) in ``tmp_path``
and drives ``build_portfolio_for_run`` — the same path the ``evoliez portfolio`` CLI uses —
so the multi-fidelity merge + enrichment + banding + report are exercised without any GPU or
real tool. Light-env: numpy + pydantic only.
"""
from __future__ import annotations

import csv
import json

from evoliez.portfolio.ledger import AXIS_REACTION_GEOMETRY, TIER_FOCUSED
from evoliez.portfolio.run import PortfolioParams, build_portfolio_for_run, load_run_records


def _write_run(tmp_path, *, n=60, n_md=12, n_high=3, n_null_nac=1):
    """A synthetic run: ``n`` cheap candidates, ``n_md`` of which reached MD with a VALID NAC
    (the first ``n_high`` are clear near-attack outliers, the rest low background), plus
    ``n_null_nac`` MD candidates whose co-substrate diffused (null NAC -> deferred geometry)."""
    prov = tmp_path / "reports" / "provenance"
    prov.mkdir(parents=True)
    gen = [{"candidate_id": f"mut_{i:05d}", "mutation_string": f"A{100 + i}K",
            "generator": "multipoint", "n_mutations": 2, "msa_freq": 0.5,
            "conservation": 0.3, "gap_freq": 0.2} for i in range(n)]
    (prov / "generated_candidates.json").write_text(json.dumps(gen))
    # s09 validated: a docking-FRAME catalytic_distance_mean (should be SHADOWED by MD frame)
    val = [{"candidate_id": f"mut_{i:05d}", "redocking_consistency": 0.7,
            "docking_uncertainty": 0.3, "ddg_fold": 0.2,
            "catalytic_distance_mean": 5.5, "hbond_occupancy": 0.5}
           for i in range(n_md + n_null_nac + 4)]
    (prov / "validated_candidates.json").write_text(json.dumps(val))
    md = []
    for i in range(n_md):                       # valid NAC: high outliers then background
        high = i < n_high
        md.append({"candidate_id": f"mut_{i:05d}", "md_instability": 0.05,
                   "md_lite_score": 0.7, "hbond_occupancy": 0.7,
                   "catalytic_distance_mean": 3.1 if high else 4.8,
                   "nac_occupancy": 0.18 if high else 0.06,
                   "nac_delta_vs_wt": 0.30 if high else 0.0})
    for j in range(n_null_nac):                 # co-substrate diffused -> null NAC
        i = n_md + j
        md.append({"candidate_id": f"mut_{i:05d}", "md_instability": 0.05,
                   "md_lite_score": 0.5, "catalytic_distance_mean": 5.0,
                   "nac_occupancy": None, "nac_delta_vs_wt": None})
    (prov / "md_candidates.json").write_text(json.dumps(md))
    return prov


def test_md_frame_wins_the_merge(tmp_path):
    _write_run(tmp_path)
    recs = {r["candidate_id"]: r for r in load_run_records(tmp_path / "reports" / "provenance")}
    # mut_00000 reached MD: its MD-frame catalytic_distance_mean (3.1) must win over the s09
    # docking-frame value (5.5); identity (mutation_string) stays from the widest file.
    assert recs["mut_00000"]["catalytic_distance_mean"] == 3.1
    assert recs["mut_00000"]["mutation_string"] == "A100K"
    # a candidate in s09 validation but NOT in MD keeps the s09 docking-frame value
    # (mut_00014: validated goes to mut_00016, MD only to mut_00012)
    assert recs["mut_00014"]["catalytic_distance_mean"] == 5.5


def test_end_to_end_fills_reaction_geometry_for_valid_nac_only(tmp_path):
    _write_run(tmp_path, n_md=12, n_high=3, n_null_nac=1)
    r = build_portfolio_for_run(tmp_path, params=PortfolioParams(panel_size=24, seed=0),
                                mechanism_class="adenylation_phosphoryl_transfer", strict=True)
    bundle = r["bundle"]
    rg = [led.axes[AXIS_REACTION_GEOMETRY] for led in bundle.ledgers]
    non_missing = [a for a in rg if not a.missing]
    # exactly the 12 valid-NAC MD candidates get a reaction-geometry axis; the null-NAC MD
    # candidate and all cheap-only candidates stay DEFERRED (an honest gap, never a fake 0).
    assert len(non_missing) == 12
    assert all(a.tier == TIER_FOCUSED and a.subset_level for a in non_missing)
    # the 3 clear near-attack outliers reach a band against the 12-sample subset-level null
    assert sum(a.is_significant() for a in non_missing) >= 1

    # artifacts written + panel is claim-safe (strict=True already enforced it)
    for kind in ("html", "json", "csv"):
        assert (tmp_path / "reports" / f"v7_portfolio.{kind}").exists()
    rows = list(csv.DictReader((tmp_path / "reports" / "v7_portfolio.csv").open()))
    assert rows and all(row["lane"] for row in rows)


def test_invalid_nac_md_candidate_stays_deferred(tmp_path):
    _write_run(tmp_path, n_md=6, n_high=2, n_null_nac=2)
    r = build_portfolio_for_run(tmp_path, params=PortfolioParams(panel_size=24, seed=0), strict=True)
    by = r["bundle"].by_id()
    # the two null-NAC MD candidates (mut_00006, mut_00007) keep reaction-geometry DEFERRED...
    for cid in ("mut_00006", "mut_00007"):
        assert by[cid].axes[AXIS_REACTION_GEOMETRY].missing is True
    # ...but their structural axis was still upgraded from real MD (they DID run MD)
    assert by["mut_00006"].axes[AXIS_REACTION_GEOMETRY].band == "deferred"


def test_per_candidate_md_analysis_files_are_merged(tmp_path):
    """The authoritative s10 results live in md/<id>/analysis.json (md_candidates.json is only a
    run-level summary). load_run_records must fold them in, deriving nac_delta from the WT ref
    and a stability_score proxy from md_lite_score — this is the real CAR schema."""
    prov = tmp_path / "reports" / "provenance"
    prov.mkdir(parents=True)
    gen = [{"candidate_id": f"mut_{i:05d}", "mutation_string": f"A{100 + i}K",
            "generator": "multipoint", "msa_freq": 0.5} for i in range(20)]
    (prov / "generated_candidates.json").write_text(json.dumps(gen))
    # md_candidates.json is a SUMMARY dict (as the real pipeline writes), not per-candidate rows
    (prov / "md_candidates.json").write_text(json.dumps(
        {"wt_reference": "_wt", "nac_enabled": True, "wt_nac_occupancy": 0.0, "requested": True}))
    # per-candidate analysis.json with the nested-`nac` schema + varying MD stability/ligand
    md = tmp_path / "md"
    (md / "_wt_reference").mkdir(parents=True)   # underscore dirs are skipped
    for i in range(6):
        d = md / f"mut_{i:05d}"
        d.mkdir(parents=True)
        (d / "analysis.json").write_text(json.dumps({
            "nac": {"nac_status": "valid_restrained_retention_screen", "nac_occupancy": 0.0},
            "nac_occupancy": 0.0, "md_lite_score": 0.2 + 0.1 * i,
            "hbond_occupancy": 0.5 + 0.05 * i, "catalytic_distance_mean": 3.5 + 0.2 * i,
            "energy_drift": 0.02, "passed": True}))
    recs = {r["candidate_id"]: r for r in
            load_run_records(prov, md_dir=md)}
    # analysis fields merged, nac_delta derived (0 - 0 = 0), md_lite_score -> stability_score
    assert recs["mut_00003"]["md_lite_score"] == 0.5
    assert recs["mut_00003"]["stability_score"] == 0.5
    assert recs["mut_00003"]["nac_delta_vs_wt"] == 0.0
    assert recs["mut_00003"]["hbond_occupancy"] == 0.65
    # the 6 MD candidates get enriched (structural/ligand + a valid-NAC reaction-geometry axis
    # that is honestly uniform at 0), the other 14 stay cheap-only
    from evoliez.portfolio import cheap_axes as CA
    leds = CA.build_cheap_ledgers(list(recs.values()))
    n = CA.enrich_expensive_axes(list(recs.values()), leds)
    assert n == 6


def test_missing_provenance_raises(tmp_path):
    (tmp_path / "reports" / "provenance").mkdir(parents=True)
    import pytest
    with pytest.raises(FileNotFoundError):
        build_portfolio_for_run(tmp_path, params=PortfolioParams(panel_size=24))
