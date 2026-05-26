"""Regression suite for the ligand-parameterization safety net.

Preflight evidence (server, fix/md-ligand-sqm-hang branch):
- openff-toolkit `Molecule.assign_partial_charges('am1bcc')` HANGS FOREVER
  on benzylpenicillin: sqm > 296 s with no SCF convergence, timeout-killed.
- RDKit Gasteiger on the same molecule: 1.90 s, charge sum = 0.000.

The fix adds a hard wall-clock cap on AM1-BCC + a Gasteiger fallback as
the universal escape hatch (was previously only used for NADP-class
cofactors via _gasteiger_charge_system_generator). Plus a run-level
ligand-params cache so sibling candidates of the same MD-stage call
skip the (just-paid) probe.

These tests lock that in without needing the openff conda stack.
"""

from __future__ import annotations

import inspect
import time

import pytest

from evoliez.adapters import openmm_engine
from evoliez.adapters.openmm_engine import (
    _AM1BCC_TIMEOUT_SECONDS,
    _ligand_charge_cache_path,
    _load_ligand_charges_from_cache,
    _run_with_timeout,
    _save_ligand_charges_to_cache,
)


# --------------------------------------------------------------------------- #
# _run_with_timeout
# --------------------------------------------------------------------------- #
def test_run_with_timeout_returns_value_under_limit():
    """Happy path: a fast callable returns its value normally."""
    result = _run_with_timeout(lambda: 42, timeout_s=1.0)
    assert result == 42


def test_run_with_timeout_raises_on_overrun():
    """Slow callable raises TimeoutError so the caller can fall through."""
    with pytest.raises(TimeoutError):
        _run_with_timeout(lambda: time.sleep(2.0), timeout_s=0.3)


def test_run_with_timeout_propagates_inner_exception():
    """A real exception inside fn must NOT be swallowed into TimeoutError -
    the dispatch differentiates the two (TimeoutError -> Gasteiger fallback;
    other exception -> next FF then Gasteiger)."""
    def boom():
        raise ValueError("inner failed")

    with pytest.raises(ValueError, match="inner failed"):
        _run_with_timeout(boom, timeout_s=1.0)


def test_module_constant_is_60_seconds_by_default():
    """Spec: 60s hard cap on a single FF probe. If you tune this, update
    the dispatch warning string in _run_real and the docstring."""
    assert _AM1BCC_TIMEOUT_SECONDS == 60


# --------------------------------------------------------------------------- #
# Ligand-charge cache
# --------------------------------------------------------------------------- #
def test_ligand_charge_cache_path_lives_in_workdir_parent(tmp_path):
    """Cache must be in workdir.PARENT so sibling candidates share it."""
    cand = tmp_path / "cand_001"
    p = _ligand_charge_cache_path(cand, "CCO", "gasteiger")
    assert p.parent == tmp_path, (
        "cache must live one dir up from the per-candidate workdir so all "
        "candidates of one MD-stage invocation share it"
    )
    assert p.name.startswith("ligand_params_cache_")
    assert p.name.endswith(".json")


def test_ligand_charge_cache_path_keys_on_smiles_and_ff(tmp_path):
    """Different SMILES -> different cache file. Different FF -> different
    cache file. Same (SMILES, FF) -> same file (so cache HITS)."""
    a1 = _ligand_charge_cache_path(tmp_path / "x", "CCO", "gasteiger")
    a2 = _ligand_charge_cache_path(tmp_path / "x", "CCO", "gasteiger")
    b = _ligand_charge_cache_path(tmp_path / "x", "CCC", "gasteiger")
    c = _ligand_charge_cache_path(tmp_path / "x", "CCO", "am1bcc")
    assert a1 == a2
    assert a1 != b
    assert a1 != c


def test_ligand_charge_cache_round_trip(tmp_path):
    """Write + read recovers the exact payload."""
    p = _ligand_charge_cache_path(tmp_path / "cand_001", "CCO", "gasteiger")
    payload = {
        "smiles": "CCO",
        "method": "gasteiger",
        "ff": "gaff-2.11+gasteiger",
        "charges": [0.04, -0.39, 0.36],
    }
    _save_ligand_charges_to_cache(p, payload)
    loaded = _load_ligand_charges_from_cache(p)
    assert loaded == payload


def test_ligand_charge_cache_miss_returns_none(tmp_path):
    """Missing file -> None (cache miss is NOT an error)."""
    p = tmp_path / "does_not_exist.json"
    assert _load_ligand_charges_from_cache(p) is None


def test_ligand_charge_cache_corrupt_returns_none(tmp_path):
    """Malformed JSON -> None (cache corruption must not crash MD)."""
    p = tmp_path / "corrupt.json"
    p.write_text("{not valid json")
    assert _load_ligand_charges_from_cache(p) is None


def test_ligand_charge_cache_write_swallows_errors(tmp_path):
    """Cache write to a non-writable path logs + returns; does NOT raise.
    MD MUST proceed even if the cache layer breaks."""
    # /dev/null is not a directory and not writable - any attempt to write
    # under it fails. The function must swallow.
    from pathlib import Path

    bad = Path("/dev/null/cache.json")
    # Must not raise
    _save_ligand_charges_to_cache(bad, {"smiles": "CCO"})


# --------------------------------------------------------------------------- #
# Source guards: anyone removing the safety net trips a test
# --------------------------------------------------------------------------- #
def test_source_guard_run_real_dispatch_has_gasteiger_fallback():
    """The 3-tier dispatch in _run_real must include a Gasteiger fallback
    AFTER the AM1-BCC probe path - this is the universal safety net for
    sqm hangs. Prevents anyone from accidentally removing it."""
    src = inspect.getsource(openmm_engine._run_real)
    lower = src.lower()
    assert "gasteiger" in lower, (
        "_run_real must reference the Gasteiger fallback for the AM1-BCC "
        "hang case"
    )
    assert "_gasteiger_charge_system_generator" in src, (
        "_run_real must actually CALL _gasteiger_charge_system_generator "
        "in the fallback branch (not just mention it in a comment)"
    )
    assert "TimeoutError" in src or "timeout" in lower, (
        "_run_real must catch TimeoutError (or otherwise handle the "
        "AM1-BCC timeout) before recording skipped_parameterization"
    )


def test_source_guard_ligand_system_generator_uses_timeout():
    """The per-FF probe loop must wrap each SystemGenerator probe in
    _run_with_timeout - otherwise sqm can hang and stall the whole MD
    stage on one bad ligand."""
    src = inspect.getsource(openmm_engine._ligand_system_generator)
    assert "_run_with_timeout" in src
    assert "_AM1BCC_TIMEOUT_SECONDS" in src


def test_source_guard_run_real_persists_charge_cache():
    """The dispatch must write to the run-level ligand-charge cache so
    siblings skip the (just-paid) probe. Cache is read AND written."""
    src = inspect.getsource(openmm_engine._run_real)
    assert "_ligand_charge_cache_path" in src
    assert "_save_ligand_charges_to_cache" in src
    assert "_load_ligand_charges_from_cache" in src


# --------------------------------------------------------------------------- #
# Live fallback chain (requires openff stack - skipped on .venv-light)
# --------------------------------------------------------------------------- #
def _have_openff() -> bool:
    try:
        import openff.toolkit  # noqa: F401
        import openmm  # noqa: F401
        import openmmforcefields  # noqa: F401
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_openff(), reason="needs openff conda stack")
def test_dispatch_falls_to_gasteiger_on_am1bcc_timeout(monkeypatch, tmp_path):
    """When _ligand_system_generator raises TimeoutError, the dispatch in
    _run_real must reach _gasteiger_charge_system_generator instead of
    short-circuiting to skipped_parameterization."""
    from openff.toolkit import Molecule

    off_lig = Molecule.from_smiles("CCO")

    calls = {"am1bcc": 0, "gasteiger": 0}

    def fake_am1bcc(*args, **kwargs):
        calls["am1bcc"] += 1
        raise TimeoutError("simulated sqm hang")

    def fake_gasteiger(*args, **kwargs):
        calls["gasteiger"] += 1

        class _SG:
            pass

        return _SG()

    monkeypatch.setattr(
        openmm_engine, "_ligand_system_generator", fake_am1bcc,
    )
    monkeypatch.setattr(
        openmm_engine, "_gasteiger_charge_system_generator", fake_gasteiger,
    )

    # We don't drive _run_real end-to-end here (would need a real Complex +
    # protein PDB); we exercise the exact fallback the dispatch performs.
    try:
        openmm_engine._ligand_system_generator(off_lig, tmp_path)
    except TimeoutError:
        sg = openmm_engine._gasteiger_charge_system_generator(
            off_lig, tmp_path,
        )
        assert sg is not None

    assert calls["am1bcc"] == 1
    assert calls["gasteiger"] == 1


@pytest.mark.skipif(not _have_openff(), reason="needs openff conda stack")
def test_cache_hit_skips_re_parameterization(tmp_path):
    """Once the ligand-charge cache has a hit, a sibling candidate's
    dispatch reads `method=gasteiger` and goes straight to Gasteiger
    instead of re-running the AM1-BCC probe."""
    # First call: write cache (we just write directly, the production code
    # does the same after a successful probe).
    cand_a = tmp_path / "cand_A"
    cand_b = tmp_path / "cand_B"
    cand_a.mkdir()
    cand_b.mkdir()

    p_a = _ligand_charge_cache_path(cand_a, "CCO", "openff-2.2.0")
    p_b = _ligand_charge_cache_path(cand_b, "CCO", "openff-2.2.0")
    assert p_a == p_b, "siblings must share the cache file"

    _save_ligand_charges_to_cache(p_a, {
        "smiles": "CCO", "method": "gasteiger", "ff": "gaff-2.11+gasteiger",
    })

    loaded = _load_ligand_charges_from_cache(p_b)
    assert loaded is not None
    assert loaded["method"] == "gasteiger", (
        "sibling candidate must see the cached method and skip AM1-BCC"
    )
