"""OpenMM ligand parameterization + skip semantics (expert review P1)."""

import inspect

from evoliez.adapters import openmm_engine
from evoliez.adapters.openmm_engine import MDResult
from evoliez.config import ScoreWeights
from evoliez.md.analysis import analyse


def test_real_md_registers_ligand_forcefield():
    # Module-level introspection (robust to internal factoring: the
    # SystemGenerator now lives in the _ligand_system_generator probe
    # helper, not inline in _run_real).
    mod = inspect.getsource(openmm_engine)
    assert "SystemGenerator" in mod
    assert "small_molecule_forcefield" in mod           # GAFF/OpenFF template
    run = inspect.getsource(openmm_engine._run_real)
    assert "skipped_parameterization" in run             # explicit skip status
    # ligand-specific (not whole-system) RMSD
    assert "lig_idx" in run and "pkt_idx" in run


def test_skipped_parameterization_is_not_a_failure():
    r = MDResult(
        candidate_id="c", status="skipped_parameterization",
        protocol_level=1, solvent_mode="implicit", simulation_time_ns=0.0,
        failure_reason="no GAFF template for metal cofactor",
    )
    m = analyse(r, ScoreWeights())
    assert m.passed is True               # NOT auto-rejected
    assert m.md_lite_score == 0.0          # neutral, not penalised
    assert any("skipped" in s for s in m.failure_reasons)


def test_failed_md_is_still_penalised():
    r = MDResult(
        candidate_id="c", status="failed", protocol_level=1,
        solvent_mode="implicit", simulation_time_ns=0.0,
        integration_failed=True, failure_reason="NaN",
    )
    m = analyse(r, ScoreWeights())
    assert m.passed is False
    assert m.md_lite_score < 0


def test_structure_skip_is_neutral_not_penalised():
    """A structure-skip (no mutant / no full-atom structure) means real MD
    never RAN for this candidate - a pipeline COVERAGE gap, not evidence the
    mutant is unstable. It must be NEUTRAL for scoring (md_instability 0), the
    same as a candidate never sent to MD; otherwise a structure-skipped
    candidate ranks BELOW an unevaluated one (a ~1 unit non-biological swing).
    Honesty is still surfaced via passed=False + failure_reasons."""
    for status in ("skipped_no_mutant_structure", "skipped_no_full_atom_structure"):
        r = MDResult(
            candidate_id="c", status=status, protocol_level=1,
            solvent_mode="implicit", simulation_time_ns=0.0,
            failure_reason="no structure available",
        )
        m = analyse(r, ScoreWeights())
        assert m.md_lite_score == 0.0                     # neutral md-lite
        # s10 derives md_instability = 0 if health_ok else 1; a skip must NOT
        # fire the instability penalty.
        assert m.simulation_health_ok is True, (
            f"{status} sets health False -> -md_instability penalty"
        )
        assert (0.0 if m.simulation_health_ok else 1.0) == 0.0
        # honesty preserved
        assert m.passed is False
        assert any("skipped" in s for s in m.failure_reasons)


def test_structure_skip_md_score_equals_unevaluated():
    """End-to-end: the MD contribution of a structure-skip candidate equals
    that of a candidate never sent to MD (s11 setdefault 0.0)."""
    from evoliez.ranking.score import compute_final_score
    from evoliez.types import Candidate

    w = ScoreWeights()
    r = MDResult(
        candidate_id="skip", status="skipped_no_mutant_structure",
        protocol_level=1, solvent_mode="implicit", simulation_time_ns=0.0,
    )
    m = analyse(r, w)
    skip = Candidate(
        candidate_id="skip", mutations=[], generator="g",
        scores={"md_lite_score": m.md_lite_score,
                "md_instability": 0.0 if m.simulation_health_ok else 1.0},
    )
    uneval = Candidate(
        candidate_id="uneval", mutations=[], generator="g",
        scores={"md_lite_score": 0.0, "md_instability": 0.0},
    )
    bs = compute_final_score(skip, w)
    bu = compute_final_score(uneval, w)
    assert bs.penalties["md_instability"] == 0.0
    assert bs.total == bu.total
