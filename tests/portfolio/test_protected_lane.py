"""Tests for the Layer-1 mechanism-protected lane + two-layer panel (reviewer breakthrough).

When a hard target produces NO statistical evidence band, the panel must still carry the
expert-curated hypotheses (forced in regardless of bands), synthesise a deferred probe for any
hypothesis the universe did not generate, expose the two-layer split, and stamp the no-band
caveat into the CSV. All claim-safe.
"""
from __future__ import annotations

import csv

from evoliez.portfolio.builder import build_portfolio
from evoliez.portfolio.ledger import (
    AXIS_STRUCTURAL, BAND_UNRESOLVED, LANE_PROTECTED, PANEL_LAYER_PROTECTED,
    PANEL_LAYER_STATISTICAL, AxisEvidenceV7, EvidenceLedgerV7, LedgerBundle, panel_layer,
)
from evoliez.portfolio.run import PortfolioParams, build_portfolio_for_run
from evoliez.ranking.claim_guard import assert_clean


def _uniform_bundle(n=30, present=("S433F", "P438N", "G407K")):
    leds = []
    for i in range(n):
        mut = present[i] if i < len(present) else f"A{100 + i}K"
        leds.append(EvidenceLedgerV7(
            variant_id=f"mut_{i:05d}", mutation=mut, overall_band=BAND_UNRESOLVED,
            axes={AXIS_STRUCTURAL: AxisEvidenceV7(axis=AXIS_STRUCTURAL, score=0.5,
                                                  band=BAND_UNRESOLVED)}))
    return LedgerBundle(ledgers=leds, n_candidates=n)


def test_protected_hypotheses_forced_in_regardless_of_bands():
    bundle = _uniform_bundle()
    protected = ["G430R;S433F;G407K", "P438N", "G430K"]
    pf = build_portfolio(bundle, panel_size=48, protected=protected,
                         protected_deconvolution=True, seed=0)
    prot = {v.mutation for v in pf.variants if v.lane == LANE_PROTECTED}
    # the core multipoint (NOT in the universe) is synthesised and forced in...
    assert "G430R;S433F;G407K" in prot
    # ...with its full single/pairwise deconvolution...
    assert {"G430R", "S433F", "G407K", "G430R;S433F", "G430R;G407K", "S433F;G407K"} <= prot
    # ...and the standalone probes.
    assert {"P438N", "G430K"} <= prot
    # a synthesised (un-generated) protected variant carries a protected:: id; a generated one
    # (S433F) reuses its real ledger id.
    by_mut = {v.mutation: v for v in pf.variants if v.lane == LANE_PROTECTED}
    assert by_mut["G430R;S433F;G407K"].variant_id.startswith("protected::")
    assert not by_mut["S433F"].variant_id.startswith("protected::")


def test_protected_reasons_are_claim_safe():
    pf = build_portfolio(_uniform_bundle(), panel_size=48,
                         protected=["G430R;S433F;G407K"], seed=0)
    assert_clean(" ".join(v.reason_to_test for v in pf.variants))


def test_panel_layer_classification():
    assert panel_layer(LANE_PROTECTED) == PANEL_LAYER_PROTECTED
    from evoliez.portfolio.ledger import LANE_STRONG
    assert panel_layer(LANE_STRONG) == PANEL_LAYER_STATISTICAL


def test_no_band_caveat_written_into_csv_and_report(tmp_path):
    prov = tmp_path / "reports" / "provenance"
    prov.mkdir(parents=True)
    import json
    gen = [{"candidate_id": f"mut_{i:05d}", "mutation_string": ("S433F" if i == 0 else f"A{100 + i}K"),
            "generator": "multipoint", "msa_freq": 0.5, "conservation": 0.5, "n_mutations": 1}
           for i in range(60)]
    (prov / "generated_candidates.json").write_text(json.dumps(gen))
    r = build_portfolio_for_run(
        tmp_path, params=PortfolioParams(panel_size=48,
                                         protected_hypotheses=["G430R;S433F;G407K"], seed=0),
        strict=True)
    assert r["portfolio"].lane_counts.get(LANE_PROTECTED, 0) >= 1

    lines = (tmp_path / "reports" / "v7_portfolio.csv").read_text().splitlines()
    assert lines[0].startswith("#")                       # no-band caveat is in the CSV itself
    assert "lead set" in lines[0]
    # the CSV still parses cleanly once the #-comment is skipped, and carries panel_layer
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    assert rows and "panel_layer" in rows[0]
    assert any(row["panel_layer"] == PANEL_LAYER_PROTECTED for row in rows)

    html = (tmp_path / "reports" / "v7_portfolio.html").read_text()
    assert "this layer is empty" in html                  # Layer A explicitly empty, not hidden
