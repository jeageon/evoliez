"""s08_reranker: known_binding_site prior (cheap-run recall fix).

Cheap-run PseFDH discovery: the heuristic conservation penalty
(`- 0.8 * (cons - 0.55)`) was pushing the Tishkov D222S/N/T/Q family
below the `top_for_redocking` cut because D222 is a well-conserved
NAD-binding-loop residue. The fix adds `at_binding_site` to the
feature row and a `+0.6 * at_binding_site` bonus to the heuristic so
the conservation penalty doesn't bury the very mutations the user is
asking the pipeline to evaluate.

These tests are unit-level on `_features` + `_score_heuristic` so they
don't need the heavy RunContext / Boltz stack.
"""

from __future__ import annotations

import inspect

from evoliez.stages.s08_reranker import (
    _FEATURE_KEYS, RerankerStage,
)
from evoliez.types import Mutation


def test_at_binding_site_in_feature_keys():
    """xgboost path reads features via _FEATURE_KEYS; if at_binding_site
    isn't in the list, the heuristic and xgboost paths disagree."""
    assert "at_binding_site" in _FEATURE_KEYS


def test_features_at_binding_site_single_mutant():
    stage = RerankerStage()
    # Minimal stand-ins: _features just reads .position from each mutation.
    cand = type("C", (), {})()
    cand.mutations = [Mutation("D", 222, "S")]
    cand.details = {}

    f_in = stage._features(
        cand, res_by_pos={}, pf_by_pos={}, nearest={}, atom_by_id={},
        binding_site={222, 148, 333},
    )
    assert f_in["at_binding_site"] == 1.0

    f_out = stage._features(
        cand, res_by_pos={}, pf_by_pos={}, nearest={}, atom_by_id={},
        binding_site={148, 333},   # D222 NOT included
    )
    assert f_out["at_binding_site"] == 0.0


def test_features_at_binding_site_multi_mutant_average():
    stage = RerankerStage()
    cand = type("C", (), {})()
    cand.mutations = [
        Mutation("D", 222, "S"),
        Mutation("X", 999, "Y"),
    ]
    cand.details = {}
    f = stage._features(
        cand, res_by_pos={}, pf_by_pos={}, nearest={}, atom_by_id={},
        binding_site={222},
    )
    # 1 of 2 mutations at binding site → 0.5
    assert f["at_binding_site"] == 0.5


def test_heuristic_bonus_lifts_binding_site_mutant_above_non():
    """The whole point: a conserved binding-site mutation should rank
    higher than a non-binding-site mutation with otherwise identical
    features."""
    stage = RerankerStage()

    # Two synthetic candidates with the SAME features except at_binding_site
    # AND conservation (binding-site residue is well-conserved, the off-site
    # one is mid-conservation). This mirrors PseFDH D222 (cons~0.85) vs a
    # random surface designable position (cons~0.55).
    bs_cand = type("C", (), {})()
    bs_cand.details = {"features": {
        "family_interaction_score": 0.5,
        "interaction_gain": 0.0,
        "msa_permissiveness": 0.0,
        "at_binding_site": 1.0,
        "conservation": 0.85,
        "n_mutations": 1,
        "dist_to_ligand": 4.0,
    }}
    bs_cand.scores = {}

    non_cand = type("C", (), {})()
    non_cand.details = {"features": {
        "family_interaction_score": 0.5,
        "interaction_gain": 0.0,
        "msa_permissiveness": 0.0,
        "at_binding_site": 0.0,
        "conservation": 0.55,   # below the penalty threshold
        "n_mutations": 1,
        "dist_to_ligand": 4.0,
    }}
    non_cand.scores = {}

    stage._score_heuristic([bs_cand, non_cand])
    # Binding-site candidate should beat the conserved-only candidate even
    # though its conservation is higher (worse under the penalty term).
    assert bs_cand.scores["ml_score"] > non_cand.scores["ml_score"], (
        f"binding-site mutation lost to non-binding-site: "
        f"bs={bs_cand.scores['ml_score']} non={non_cand.scores['ml_score']}"
    )


def test_heuristic_bonus_doesnt_overwhelm_strong_negative_features():
    """The +0.6 bonus is a NUDGE, not a free pass. A binding-site
    candidate with terrible everything else should still rank below a
    non-binding-site candidate with strong features."""
    stage = RerankerStage()

    bad_bs = type("C", (), {})()
    bad_bs.details = {"features": {
        "family_interaction_score": 0.0,
        "interaction_gain": 0.0,
        "msa_permissiveness": 0.0,
        "at_binding_site": 1.0,
        "conservation": 0.95,        # very conserved → big penalty
        "n_mutations": 3,            # 3 mutations → -0.3 already
        "dist_to_ligand": 12.0,       # far from ligand
    }}
    bad_bs.scores = {}

    good_non = type("C", (), {})()
    good_non.details = {"features": {
        "family_interaction_score": 0.9,   # high family signal
        "interaction_gain": 1.0,           # strong ligand interaction
        "msa_permissiveness": 0.8,         # frequent in MSA
        "at_binding_site": 0.0,
        "conservation": 0.50,
        "n_mutations": 1,
        "dist_to_ligand": 4.5,
    }}
    good_non.scores = {}

    stage._score_heuristic([bad_bs, good_non])
    assert good_non.scores["ml_score"] > bad_bs.scores["ml_score"]


def test_source_carries_at_binding_site_branch():
    """Source-level guard: a future refactor mustn't silently drop the
    feature or the heuristic term (matches the pattern in
    test_ligandmpnn_preflight)."""
    src = inspect.getsource(RerankerStage._features)
    assert "binding_site" in src
    assert "at_binding_site" in src
    src2 = inspect.getsource(RerankerStage._score_heuristic)
    assert "at_binding_site" in src2
