"""v3 integration: triage report over a provenance dir, claim-clean (ROADMAP_V3 §7)."""
import json

from evoliez.ranking import claim_guard as cg
from evoliez.ranking.triage_report import build_triage, render_triage_text


def _write_prov(tmp_path):
    md = [
        {"candidate_id": "mut_A", "stability_score": 1.5, "nac_occupancy": 0.7,
         "hbond_occupancy": 0.8, "catalytic_distance_mean": 3.1, "docking_uncertainty": 0.2,
         "msa_permissiveness": 0.6},
        {"candidate_id": "mut_B", "stability_score": -1.0, "nac_occupancy": 0.1,
         "docking_uncertainty": 0.9},
    ]
    (tmp_path / "md_candidates.json").write_text(json.dumps(md))
    return str(tmp_path)


def test_build_triage_and_render_clean(tmp_path):
    prov_dir = _write_prov(tmp_path)
    prov = cg.ClaimProvenance()                       # conservative: nothing unlocked
    triage = build_triage(prov_dir, claim_provenance=prov)
    assert triage["n_candidates"] == 2
    assert triage["claim_ceiling"] == "uncalibrated"  # no controls in conservative provenance
    # a strong candidate is recommended, a weak one deprioritized
    recs = {r["variant_id"]: r["recommendation"] for r in triage["recommendations"]}
    assert recs["mut_A"] in ("test_in_round1", "secondary_candidate")
    assert recs["mut_B"] == "deprioritized"
    # the rendered report is claim-clean by construction (assert inside)
    text = render_triage_text(triage, claim_provenance=prov)
    assert "does not directly predict kcat" in text
    # and double-check: no prohibited claim leaked
    assert cg.lint_report(text, prov) == []


def test_render_would_fail_on_injected_overclaim(tmp_path):
    prov_dir = _write_prov(tmp_path)
    prov = cg.ClaimProvenance()
    triage = build_triage(prov_dir, claim_provenance=prov)
    # inject an over-claim into a recommendation -> render must raise
    triage["recommendations"][0]["primary_reason"].append("activity improved")
    import pytest
    with pytest.raises(AssertionError):
        render_triage_text(triage, claim_provenance=prov)


def test_empty_provenance_dir_is_safe(tmp_path):
    triage = build_triage(str(tmp_path), claim_provenance=cg.ClaimProvenance())
    assert triage["n_candidates"] == 0
    text = render_triage_text(triage, claim_provenance=cg.ClaimProvenance())
    assert "candidates evaluated: 0" in text
