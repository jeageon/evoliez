"""Regression tests for the reference-card claim clamps (Fable safety review, 2026-07-09).

An explicitly-supplied claim_strength / claim_ceiling must never EXCEED the evidence-derived
grade — it may only lower it (a manual downgrade). Previously an explicit value was accepted
verbatim, letting a weak reference carry an inflated claim into downstream confidence.
"""
from __future__ import annotations

from evoliez.mechanism.cards import (
    ActiveStateReferenceEnsemble, EnsembleDisagreement, ReferenceConfidenceCard, ReferenceMember,
)
from evoliez.mechanism.vocab import HYPOTHESIS_GRADE, UNCALIBRATED


def test_explicit_strength_cannot_exceed_derived_grade():
    c = ReferenceConfidenceCard(
        tier="D", claim_strength="strong_screening",
        substrate_is_real=False, functional_atom_mapping_verified=False)
    assert c.claim_strength == UNCALIBRATED == c.grade()


def test_explicit_strength_may_downgrade():
    # a manual downgrade below the derived grade is honored (clamp only tightens).
    c = ReferenceConfidenceCard(
        tier="A", claim_strength=HYPOTHESIS_GRADE,
        known_active_controls_available=True, functional_atom_mapping_verified=True)
    assert c.claim_strength == HYPOTHESIS_GRADE


def test_explicit_ensemble_ceiling_cannot_exceed_derived():
    e = ActiveStateReferenceEnsemble(
        ensemble_id="e", references=[ReferenceMember(reference_id="r", tier="D")],
        claim_ceiling="strong_screening")
    assert e.claim_ceiling == HYPOTHESIS_GRADE   # tier-D derived cap, not the supplied strong


def test_pose_and_contact_disagreement_cap_the_ceiling():
    # a tier-A ensemble that agrees on backbone geometry but disagrees on ligand pose /
    # catalytic contact must NOT still claim strong (all three variance axes count now).
    e = ActiveStateReferenceEnsemble(
        ensemble_id="e2", references=[ReferenceMember(reference_id="rA", tier="A")],
        disagreement=EnsembleDisagreement(
            geometry_variance=0.0, ligand_pose_variance=0.99, catalytic_contact_variance=0.99))
    assert e.claim_ceiling == HYPOTHESIS_GRADE
