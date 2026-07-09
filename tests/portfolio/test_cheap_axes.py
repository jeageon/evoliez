"""Tests for the Tier-0 all-candidate cheap ledger (ROADMAP_V7 Phase V7-1).

Depends only on the module under test + the frozen ledger contract + ml.learnability +
numpy/pydantic + pytest (no sibling portfolio module is imported).
"""
from __future__ import annotations

import numpy as np

from evoliez.ml import learnability
from evoliez.mechanism.cards import EvidenceAxis, EvidenceCard
from evoliez.portfolio.cheap_axes import build_cheap_ledgers, ledger_from_evidence_card
from evoliez.portfolio.ledger import (
    AXIS_EVOLUTIONARY, AXIS_LIGAND, AXIS_MECHANISM, AXIS_PORTFOLIO,
    AXIS_REACTION_GEOMETRY, AXIS_STRUCTURAL, AXIS_UNCERTAINTY, BAND_CONTROL,
    BAND_DEFERRED, BAND_UNRESOLVED, CHEAP_AXES, SEVEN_AXES, TIER_CHEAP,
)


class _Res:
    """Minimal duck-typed catalytic-residue for a generic MechanismSpec stand-in."""
    def __init__(self, residue):
        self.residue = residue


class _Mech:
    def __init__(self, residues):
        self.catalytic_residues = [_Res(r) for r in residues]


def _rows():
    return [
        # 0: rich MSA/structural single-point, close to ligand
        {"candidate_id": "mut_00000", "mutation_string": "L55I", "generator": "chemistry_rules",
         "n_mutations": 1, "conservation": 0.2, "msa_permissiveness": 0.7, "esm_permissive": 0.6,
         "subfamily_specific": 0.4, "gap_freq": 0.05, "n_contacts": 4,
         "distance_to_design_ligand": 1.2, "distance_to_nearest_catalytic": 3.0,
         "ddg_fold": -1.0, "stability_score": 0.8},
        # 1: destabilizing, far from ligand, sparse MSA
        {"candidate_id": "mut_00001", "mutation_string": "L55V", "generator": "chemistry_rules",
         "n_mutations": 1, "conservation": 0.9, "msa_permissiveness": 0.05, "gap_freq": 0.6,
         "n_contacts": 1, "distance_to_design_ligand": 7.5, "ddg_fold": 3.0},
        # 2: multipoint active-site design touching a catalytic residue (290)
        {"candidate_id": "mut_00002", "mutation_string": "R290K;F58I;L55M", "generator": "multipoint",
         "n_mutations": 3, "multipoint_order": 3, "gap_freq": 0.1,
         "distance_to_design_ligand": 0.4, "n_contacts": 6},
        # 3: role-tagged mutation via ligand_roles, no distances at all -> ligand DEFERRED
        {"candidate_id": "mut_00003", "mutation_string": "H338A", "generator": "ligandmpnn",
         "n_mutations": 1, "ligand_roles": ["catalytic"], "ligandmpnn_logp": -1.2,
         "conservation": 0.5},
        # 4: NO cheap structural / evolutionary / ligand signal at all (weak vs deferred split)
        {"candidate_id": "mut_00004", "mutation_string": "G127S", "generator": "msa",
         "n_mutations": 1},
        # 5: deconvolution probe of a multipoint parent
        {"candidate_id": "mut_00005", "mutation_string": "F58I", "generator": "multipoint",
         "n_mutations": 1, "multipoint_order": 1, "deconvolution_of": "mut_00002",
         "distance_to_design_ligand": 2.0, "n_contacts": 3, "gap_freq": 0.2},
    ]


def test_all_seven_axes_present_cheap_scored_geometry_and_portfolio_deferred():
    mech = _Mech(["ARG290", "HIS338"])
    ledgers = build_cheap_ledgers(_rows(), mechanism=mech, seed=0)
    assert len(ledgers) == 6
    for led in ledgers:
        # every one of the seven axes exists (ledger auto-fill contract)
        assert set(led.axes) == set(SEVEN_AXES)
        # the two panel-level / expensive axes stay deferred at tier-0
        for ax in (AXIS_REACTION_GEOMETRY, AXIS_PORTFOLIO):
            assert led.axis(ax).missing is True
            assert led.axis(ax).band == BAND_DEFERRED
        # cheap axes are scored (tier0) and in range
        for ax in CHEAP_AXES:
            a = led.axis(ax)
            assert a.tier == TIER_CHEAP
            assert 0.0 <= a.score <= 1.0
            assert a.confidence in ("high", "medium", "low_to_medium", "low")


def test_ligand_deferred_when_no_proximity_but_present_when_distances_exist():
    ledgers = {l.variant_id: l for l in build_cheap_ledgers(_rows())}
    # row 3 & 4 carry no distance_to_* and no n_contacts -> ligand DEFERRED, not a zero score
    for cid in ("mut_00003", "mut_00004"):
        lig = ledgers[cid].axis(AXIS_LIGAND)
        assert lig.missing is True and lig.band == BAND_DEFERRED
    # row 0 has both distances + contacts -> scored, not missing
    lig0 = ledgers["mut_00000"].axis(AXIS_LIGAND)
    assert lig0.missing is False and lig0.band == BAND_UNRESOLVED and lig0.score > 0.0


def test_weak_not_missing_when_expected_cheap_prior_absent():
    # row 4 has no ddg/stability and no MSA permissiveness -> structural & evolutionary are
    # present-but-weak (scored 0, unresolved), NOT deferred (those cheap priors are expected).
    led = {l.variant_id: l for l in build_cheap_ledgers(_rows())}["mut_00004"]
    for ax, reason in ((AXIS_STRUCTURAL, "no_cheap_structural_signal"),
                       (AXIS_EVOLUTIONARY, "no_cheap_evolutionary_signal")):
        a = led.axis(ax)
        assert a.missing is False
        assert a.band == BAND_UNRESOLVED
        assert a.score == 0.0
        assert reason in a.provenance


def test_mechanism_axis_rewards_multipoint_role_and_deconvolution():
    mech = _Mech(["ARG290"])
    led = {l.variant_id: l for l in build_cheap_ledgers(_rows(), mechanism=mech)}
    # multipoint active-site design touching catalytic 290 scores higher than a lone msa pick
    strong = led["mut_00002"].axis(AXIS_MECHANISM).score
    weak = led["mut_00004"].axis(AXIS_MECHANISM).score
    assert strong > weak
    # catalytic-residue match recorded generically (no enzyme name hardcoded)
    assert "catalytic_residue_match" in led["mut_00002"].axis(AXIS_MECHANISM).provenance
    # role tag via ligand_roles recorded on row 3
    assert "ligand_roles" in led["mut_00003"].axis(AXIS_MECHANISM).provenance
    # deconvolution membership surfaced on row 5
    assert led["mut_00005"].deconvolution_of == "mut_00002"
    assert "deconvolution_of" in led["mut_00005"].axis(AXIS_MECHANISM).provenance


def test_uncertainty_is_higher_worse_for_sparse_high_load_rows():
    led = {l.variant_id: l for l in build_cheap_ledgers(_rows())}
    # row 1 (gap_freq 0.6) is more uncertain than row 0 (gap_freq 0.05)
    assert led["mut_00001"].axis(AXIS_UNCERTAINTY).score > led["mut_00000"].axis(AXIS_UNCERTAINTY).score
    # more simultaneous mutations -> more uncertain
    assert led["mut_00002"].axis(AXIS_UNCERTAINTY).score > led["mut_00000"].axis(AXIS_UNCERTAINTY).score


def test_deterministic_same_seed_identical_dump():
    a = build_cheap_ledgers(_rows(), seed=7)
    b = build_cheap_ledgers(_rows(), seed=7)
    assert [l.model_dump() for l in a] == [l.model_dump() for l in b]


def test_no_expensive_field_leaks_into_a_cheap_score():
    ledgers = build_cheap_ledgers(_rows(), mechanism=_Mech(["ARG290"]))
    all_prov = [p for led in ledgers for ax in CHEAP_AXES for p in led.axis(ax).provenance]
    # nothing expensive ever appears in any cheap-axis provenance
    assert not (set(all_prov) & learnability.EXPENSIVE_FIELDS)
    # the learnability-tracked feature keys used pass the leakage guard mechanically
    cheap_keys = sorted({p for p in all_prov if learnability.is_cheap(p)})
    learnability.assert_no_leakage(cheap_keys)  # must not raise
    assert cheap_keys  # at least some cheap features were actually consumed


def test_leakage_guard_trips_if_an_expensive_field_were_scored():
    # sanity: the guard the module relies on actually rejects a label source as a feature
    import pytest
    with pytest.raises(ValueError):
        learnability.assert_no_leakage(["msa_freq", "nac_delta_vs_wt"])


def test_ledger_from_evidence_card_maps_six_axes_onto_seven():
    card = EvidenceCard(
        variant_id="mut_00002",
        structural_viability=EvidenceAxis(score=0.7, confidence="high", evidence=["thermompnn"]),
        evolutionary_tolerance=EvidenceAxis(score=0.6, confidence="medium"),
        ligand_cofactor_competence=EvidenceAxis(score=0.5, confidence="medium"),
        substrate_positioning=EvidenceAxis(score=0.9, confidence="high", evidence=["dist"]),
        reaction_geometry_accommodation=EvidenceAxis(score=0.3, confidence="low", evidence=["angle"]),
        uncertainty=EvidenceAxis(score=0.2, confidence="medium"),
    )
    rec = {"mutation_string": "R290K;F58I;L55M", "generator": "multipoint",
           "n_mutations": 3, "multipoint_order": 3}
    led = ledger_from_evidence_card(card, rec=rec)
    assert set(led.axes) == set(SEVEN_AXES)
    # substrate_positioning + reaction_geometry_accommodation FOLD into reaction_geometry_access,
    # kept SEPARATE in detail, axis score conservatively the MIN (short distance != NAC).
    rg = led.axis(AXIS_REACTION_GEOMETRY)
    assert rg.missing is False
    # no absolute nac_occupancy in rec -> angle falls back to the card's (delta-based) score;
    # distance and angle stay SEPARATE, ΔNAC kept as its own observable.
    assert rg.detail == {"distance": 0.9, "angle_nac_access": 0.3, "nac_delta_vs_wt_score": 0.3}
    assert abs(rg.score - 0.3) < 1e-9
    # ligand_cofactor_competence -> ligand axis
    assert abs(led.axis(AXIS_LIGAND).score - 0.5) < 1e-9
    # mechanism axis derived from rec, portfolio deferred (non-control)
    assert led.axis(AXIS_MECHANISM).missing is False
    assert led.axis(AXIS_PORTFOLIO).missing is True


def test_ledger_from_evidence_card_control_sets_portfolio_and_band():
    card = EvidenceCard(variant_id="wt")
    led = ledger_from_evidence_card(
        card, rec={"mutation_string": "", "control_type": "wt_parental"},
        is_control=True)
    assert led.is_control is True
    assert led.control_role == "wt_parental"
    port = led.axis(AXIS_PORTFOLIO)
    assert port.band == BAND_CONTROL and port.missing is False
    assert led.overall_band == BAND_CONTROL


def test_seed_param_is_accepted_and_numpy_importable():
    # module + test env sanity: numpy usable, seed accepted without RNG divergence
    _ = np.array([1.0, 2.0])
    assert build_cheap_ledgers([], seed=123) == []


# --- V7-4/5 expensive-tier enrichment --------------------------------------------------
def test_enrich_expensive_axes_fills_reaction_geometry_for_md_subset():
    """The deferred reaction-geometry axis becomes real (subset-level, focused-MD tier) only
    for candidates carrying MD/NAC evidence; s09-only candidates upgrade structural/ligand but
    keep reaction-geometry deferred; cheap-only candidates are untouched."""
    from evoliez.portfolio.cheap_axes import enrich_expensive_axes
    from evoliez.portfolio.ledger import (
        AXIS_REACTION_GEOMETRY, AXIS_STRUCTURAL, TIER_CHEAP, TIER_FOCUSED, TIER_GPU_BROAD)

    recs = [{"candidate_id": f"mut_{i:05d}", "mutation_string": f"A{i}K",
             "generator": "multipoint", "msa_freq": 0.5, "n_mutations": 2} for i in range(12)]
    # MD subset (real NAC): fills reaction-geometry
    for i in range(3):
        recs[i].update({"nac_occupancy": 0.2, "nac_delta_vs_wt": 0.05, "md_instability": 0.1,
                        "catalytic_distance_mean": 3.2, "hbond_occupancy": 0.6})
    # s09-only subset (no NAC): upgrades structural/ligand only
    for i in range(3, 6):
        recs[i].update({"redocking_consistency": 0.7, "docking_uncertainty": 0.3,
                        "catalytic_distance_mean": 4.0, "hbond_occupancy": 0.5})

    ledgers = build_cheap_ledgers(recs)
    n = enrich_expensive_axes(recs, ledgers)
    assert n == 6
    by = {led.variant_id: led for led in ledgers}

    md_rg = by["mut_00000"].axes[AXIS_REACTION_GEOMETRY]
    assert md_rg.missing is False and md_rg.tier == TIER_FOCUSED and md_rg.subset_level is True
    # distance and angle observables stay SEPARATE (a short distance never == NAC)
    assert "distance" in md_rg.detail and "angle_nac_access" in md_rg.detail
    assert by["mut_00000"].tier_reached == TIER_FOCUSED

    s09_rg = by["mut_00004"].axes[AXIS_REACTION_GEOMETRY]
    assert s09_rg.missing is True                      # no NAC -> still deferred
    assert by["mut_00004"].axes[AXIS_STRUCTURAL].tier == TIER_GPU_BROAD
    assert by["mut_00004"].tier_reached == TIER_GPU_BROAD

    cheap = by["mut_00010"]
    assert cheap.axes[AXIS_REACTION_GEOMETRY].missing is True
    assert cheap.tier_reached == TIER_CHEAP
