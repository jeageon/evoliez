"""Reproducibility regression: real LigandMPNN is invoked with an explicit --seed so
the ligandmpnn candidate fraction is deterministic (run.py self-randomizes on --seed 0).
Captures the subprocess argv without running the tool.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evoliez.adapters import ligandmpnn
from evoliez.adapters.ligandmpnn import design_sequences
from evoliez.config import Backend, MutationGenConfig


class _Stop(Exception):
    pass


def _fake_cx():
    residues = [SimpleNamespace(index=i) for i in range(1, 11)]
    return SimpleNamespace(
        structure=SimpleNamespace(residues=residues, pdb_path=None),
        ligand=SimpleNamespace(atoms=[]))


def _patch(monkeypatch, captured):
    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        raise _Stop()
    monkeypatch.setattr(ligandmpnn, "run", fake_run)
    monkeypatch.setattr(ligandmpnn, "require", lambda *a, **k: None)
    monkeypatch.setattr(ligandmpnn, "apply_gpu_selection", lambda *a, **k: None)
    monkeypatch.setattr(ligandmpnn, "full_atom_receptor_pdb", lambda *a, **k: False)
    monkeypatch.setattr(ligandmpnn, "write_min_pdb", lambda *a, **k: None)
    monkeypatch.setattr(ligandmpnn, "tool_env", lambda *a, **k: {})


def test_seed_passed_to_ligandmpnn_cmd(monkeypatch, tmp_path):
    captured = {}
    _patch(monkeypatch, captured)
    with pytest.raises(_Stop):
        design_sequences(_fake_cx(), [3, 4, 5], MutationGenConfig(), tmp_path,
                         backend=Backend.real, dry_run=False, seed=42)
    cmd = captured["cmd"]
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "42"


def test_no_seed_omits_flag(monkeypatch, tmp_path):
    captured = {}
    _patch(monkeypatch, captured)
    with pytest.raises(_Stop):
        design_sequences(_fake_cx(), [3, 4, 5], MutationGenConfig(), tmp_path,
                         backend=Backend.real, dry_run=False, seed=None)
    assert "--seed" not in captured["cmd"]


def test_zero_seed_becomes_nonzero(monkeypatch, tmp_path):
    # run.py treats --seed 0 as "randomize", so a 0 must be bumped to a fixed non-zero
    captured = {}
    _patch(monkeypatch, captured)
    with pytest.raises(_Stop):
        design_sequences(_fake_cx(), [3, 4, 5], MutationGenConfig(), tmp_path,
                         backend=Backend.real, dry_run=False, seed=0)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--seed") + 1] == "1"
