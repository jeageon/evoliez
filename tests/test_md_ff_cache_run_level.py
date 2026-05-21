"""Expert audit follow-up: per-candidate FF cache forced openmmforcefields
to re-parameterize the SAME ligand (NADP+) once per candidate. On
production scale (top-30 candidates × 3 final-tier replicas) that's ~30
redundant antechamber/Gasteiger passes. Run-level cache shares one
cache JSON across candidates (openmmforcefields indexes by SMILES
internally, so co-locating is correct).

These tests lock in:
- `_ff_cache_path` puts the cache in workdir.PARENT (one dir up from
  the per-candidate workdir), so siblings share it.
- Filename is sanitized (no slashes/special chars from FF name).
- Source guard: both _gasteiger_charge_system_generator and
  _ligand_system_generator call the helper.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from evoliez.adapters import openmm_engine
from evoliez.adapters.openmm_engine import _ff_cache_path


def test_ff_cache_lives_one_dir_up_from_candidate_workdir(tmp_path):
    cand_dir = tmp_path / "md" / "cand_001"
    cand_dir.mkdir(parents=True, exist_ok=True)
    path = _ff_cache_path(cand_dir, "gaff-2.11")
    # Cache file is under md/ (parent), not md/cand_001/ (candidate).
    assert path.parent == cand_dir.parent
    assert path.name == "ff_cache_gaff-2.11.json"


def test_ff_cache_filename_sanitizes_special_chars(tmp_path):
    cand_dir = tmp_path / "md" / "cand_001"
    cand_dir.mkdir(parents=True, exist_ok=True)
    # FF name with a "+" (Gasteiger probe variant) and a slash-ish path.
    path = _ff_cache_path(cand_dir, "gaff-2.11+gasteiger")
    assert "ff_cache_gaff-2.11+gasteiger.json" == path.name
    # No reserved characters that would break the JSON path.
    assert "/" not in path.name
    assert "\\" not in path.name


def test_two_candidates_share_the_same_cache_path(tmp_path):
    md_dir = tmp_path / "md"
    a = md_dir / "cand_A"; a.mkdir(parents=True)
    b = md_dir / "cand_B"; b.mkdir(parents=True)
    pa = _ff_cache_path(a, "gaff-2.11")
    pb = _ff_cache_path(b, "gaff-2.11")
    assert pa == pb, (
        "FF cache must be shared across candidates so the same ligand "
        "isn't re-parameterized per candidate"
    )


def test_both_system_generators_use_the_helper():
    g = inspect.getsource(openmm_engine._gasteiger_charge_system_generator)
    a = inspect.getsource(openmm_engine._ligand_system_generator)
    assert "_ff_cache_path(workdir" in g
    assert "_ff_cache_path(workdir" in a
    # Old per-candidate pattern must not return.
    assert 'workdir / "ff_cache' not in g
    assert 'workdir / f"ff_cache' not in a
