"""P0 #1: real docking/stability must use the FULL-ATOM Boltz structure, not a
CA-only trace.

write_min_pdb() emits CA-only protein records; real Vina/GNINA/DiffDock would
dock into a sidechain-less pocket and FoldX/Rosetta cannot build residue
templates. full_atom_receptor_pdb() resolves structure.pdb_path (protein only)
when it is full-atom; the 5 adapters degrade HONESTLY otherwise (mock dock /
unavailable ddG) instead of running the tool on a backbone-only receptor.
"""

from pathlib import Path

import pytest

from evoliez.adapters import diffdock, foldx, gnina, rosetta, vina
from evoliez.adapters.base import (
    RealToolError,
    full_atom_receptor_pdb,
    is_full_atom_pdb,
    write_min_pdb,
)
from evoliez.config import Backend, DockingConfig, StabilityConfig
from evoliez.types import LigandAtom, Mutation, ProteinStructure, Residue

_FULL_ATOM = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O\n"
    "ATOM      5  CB  ALA A   1       1.000   1.000   0.000  1.00  0.00           C\n"
    "ATOM      6  N   GLY A   2       4.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      7  CA  GLY A   2       5.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      8  C   GLY A   2       6.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      9  O   GLY A   2       7.000   0.000   0.000  1.00  0.00           O\n"
    "HETATM   10  C1  LIG L   1       2.000   2.000   2.000  1.00  0.00           C\n"
    "END\n"
)


def _structure(pdb_path=None):
    res = [Residue(index=1, aa="A", ca=(1.0, 0.0, 0.0)),
           Residue(index=2, aa="G", ca=(5.0, 0.0, 0.0))]
    return ProteinStructure(sequence="AG", residues=res, pdb_path=pdb_path)


def _ref():
    return [LigandAtom(id="C1", element="C", coord=(2.0, 2.0, 2.0))]


def _full_atom_structure(tmp_path):
    src = tmp_path / "complex.pdb"
    src.write_text(_FULL_ATOM)
    return _structure(str(src))


# --------------------------- the shared helper ------------------------------
def test_helper_extracts_full_atom_protein_only(tmp_path):
    out = tmp_path / "rec.pdb"
    assert full_atom_receptor_pdb(_full_atom_structure(tmp_path), out) is True
    body = out.read_text()
    assert is_full_atom_pdb(out)            # a non-CA atom (CB) survived
    assert "CB  ALA" in body                # full sidechain kept
    assert "HETATM" not in body             # ligand stripped from the receptor


def test_helper_false_on_ca_only_or_missing(tmp_path):
    ca = tmp_path / "ca.pdb"
    write_min_pdb(ca, _structure())         # CA-only trace
    assert is_full_atom_pdb(ca) is False
    assert full_atom_receptor_pdb(_structure(str(ca)), tmp_path / "a.pdb") is False
    assert full_atom_receptor_pdb(_structure(None), tmp_path / "b.pdb") is False
    assert not (tmp_path / "a.pdb").exists()  # False never writes an output


# require() tolerates missing tools under EVOLIEZ_DRY_RUN, so we can reach the
# receptor guard on the REAL path without vina/foldx installed.
@pytest.fixture
def _toolless(monkeypatch):
    monkeypatch.setenv("EVOLIEZ_DRY_RUN", "1")


# ------------------- dockers: full-atom receptor is used --------------------
def test_dockers_use_full_atom_receptor(tmp_path, _toolless):
    st = _full_atom_structure(tmp_path)
    cfg = DockingConfig()
    vina.redock("c", st, _ref(), cfg, tmp_path / "v", instability=0.1,
                backend=Backend.real, dry_run=True)
    rec = tmp_path / "v" / "c_rec.pdb"
    assert rec.exists() and is_full_atom_pdb(rec)   # came from pdb_path
    assert "HETATM" not in rec.read_text()

    gnina.redock("c", st, _ref(), cfg, tmp_path / "g", instability=0.1,
                 backend=Backend.real, dry_run=True)
    assert is_full_atom_pdb(tmp_path / "g" / "c_rec.pdb")

    diffdock.redock("c", st, _ref(), cfg, tmp_path / "d", instability=0.1,
                    smiles="CCO", backend=Backend.real, dry_run=True)
    assert is_full_atom_pdb(tmp_path / "d" / "c_rec.pdb")


# -------- dockers: CA-only under real -> HARD-FAIL (audit P0 #3) -------------
def test_dockers_hardfail_without_full_atom(tmp_path, _toolless):
    # strict default (no allow_mock_fallback): a real dock with no full-atom
    # structure raises rather than silently scoring a mock pose.
    st = _structure(None)
    with pytest.raises(RealToolError):
        vina.redock("c", st, _ref(), DockingConfig(), tmp_path / "v",
                    instability=0.2, backend=Backend.real, dry_run=False)
    assert not (tmp_path / "v" / "c_rec.pdb").exists()   # no receptor prepped


def test_dockers_mock_fallback_when_explicitly_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLIEZ_DRY_RUN", "1")           # require() tolerant
    monkeypatch.setenv("EVOLIEZ_ALLOW_MOCK_FALLBACK", "1")
    pose = vina.redock("c", _structure(None), _ref(), DockingConfig(),
                       tmp_path / "v", instability=0.2, backend=Backend.real,
                       dry_run=False)
    assert pose is not None                              # degraded, no raise


# ---------------- stability: CA-only under real -> ddG unavailable -----------
def test_stability_unavailable_without_full_atom(tmp_path, _toolless):
    st = _structure(None)
    muts = [Mutation(wt="A", position=1, mut="V")]
    fres = foldx.estimate_stability("c", st, muts, StabilityConfig(),
                                    tmp_path / "fx", backend=Backend.real,
                                    dry_run=False)
    assert fres["ddg_fold"] is None
    assert not (tmp_path / "fx" / "c.pdb").exists()

    rres = rosetta.estimate_stability("c", st, muts, tmp_path / "rs",
                                      backend=Backend.real, dry_run=False)
    assert rres["ddg_fold"] is None


# ----------------------------- MD parity (no regression) --------------------
def test_stability_full_atom_proceeds_past_guard(tmp_path, _toolless):
    # full-atom present -> guard passes, the WT receptor pdb is written from
    # pdb_path (FoldX then builds the mutant from it). dry_run -> mock ddG.
    st = _full_atom_structure(tmp_path)
    foldx.estimate_stability("c", st, [Mutation(wt="A", position=1, mut="V")],
                             StabilityConfig(), tmp_path / "fx",
                             backend=Backend.real, dry_run=True)
    assert is_full_atom_pdb(tmp_path / "fx" / "c.pdb")
