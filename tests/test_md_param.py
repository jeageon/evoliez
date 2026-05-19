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
