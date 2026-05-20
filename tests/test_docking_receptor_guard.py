"""P0.1: full-atom receptor guard for real docking + pose-quality persistence.

The previous behaviour wrote `structure` through `write_min_pdb` even on the
real path, handing Vina/GNINA/DiffDock a CA-only receptor and getting back
plausible-looking but scientifically invalid scores. These tests lock in
the new behaviour:

- ``resolve_real_receptor_pdb`` returns ``structure.pdb_path`` when it's a
  full-atom PDB, raises :class:`NotFullAtomReceptor` otherwise.
- The three real-backend dockers emit ``Pose(skipped="skipped_no_full_atom_
  structure")`` instead of fabricating a score.
- ``Pose`` carries the new physical-validity surface (status, reasons,
  receptor_clashes, plif_recovery, CNN features) for the reranker /
  evidence-class layer downstream.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from evoliez.adapters import diffdock, gnina, vina
from evoliez.adapters.receptor_io import (
    NotFullAtomReceptor, is_full_atom_pdb, resolve_real_receptor_pdb,
)
from evoliez.config import Backend, DockingConfig
from evoliez.types import LigandAtom, Pose, ProteinStructure, Residue


# --------------------------------------------------------------------------- #
# resolve_real_receptor_pdb
# --------------------------------------------------------------------------- #
def test_full_atom_check_distinguishes_ca_only_from_real(tmp_path):
    ca_only = tmp_path / "ca.pdb"
    ca_only.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00\n"
        "ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 80.00\n"
        "END\n"
    )
    full = tmp_path / "full.pdb"
    full.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00\n"
        "ATOM      2  CA  ALA A   1       0.500   0.500   0.500  1.00 80.00\n"
        "ATOM      3  C   ALA A   1       1.000   1.000   1.000  1.00 80.00\n"
        "END\n"
    )
    assert not is_full_atom_pdb(ca_only)
    assert is_full_atom_pdb(full)


def test_resolve_real_receptor_pdb_returns_full_atom_path(tmp_path):
    full = tmp_path / "full.pdb"
    full.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000\n"
        "ATOM      2  CA  ALA A   1       0.500   0.500   0.500\n"
        "END\n"
    )
    s = ProteinStructure(sequence="A", residues=[
        Residue(index=1, aa="A", ca=(0.5, 0.5, 0.5))
    ], pdb_path=str(full))
    assert resolve_real_receptor_pdb(s, candidate_id="c") == full


def test_resolve_real_receptor_pdb_refuses_ca_only(tmp_path):
    ca_only = tmp_path / "ca.pdb"
    ca_only.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0\nEND\n")
    s = ProteinStructure(sequence="A", residues=[
        Residue(index=1, aa="A", ca=(0, 0, 0))
    ], pdb_path=str(ca_only))
    with pytest.raises(NotFullAtomReceptor, match="CA-only"):
        resolve_real_receptor_pdb(s, candidate_id="c")


def test_resolve_real_receptor_pdb_refuses_missing_path():
    s = ProteinStructure(sequence="A", residues=[
        Residue(index=1, aa="A", ca=(0, 0, 0))
    ])  # no pdb_path
    with pytest.raises(NotFullAtomReceptor):
        resolve_real_receptor_pdb(s, candidate_id="c")


# --------------------------------------------------------------------------- #
# Source-level: all three real adapters now call the resolver
# (no inline write_min_pdb on the receptor for real backends)
# --------------------------------------------------------------------------- #
def test_real_adapters_route_through_resolver():
    for mod in (vina, gnina, diffdock):
        src = inspect.getsource(mod._redock_real)
        assert "resolve_real_receptor_pdb" in src, (
            f"{mod.__name__}: real path no longer uses the receptor resolver"
        )
        # The mock-style write_min_pdb is allowed for ligand temp files,
        # NOT for the receptor.
        assert "write_min_pdb(rec" not in src, (
            f"{mod.__name__}: real path still write_min_pdb's the receptor"
        )


def test_pose_has_new_validity_surface():
    p = Pose(candidate_id="c", method="vina", score=-7.0)
    # New fields exist with safe defaults so older callers don't break.
    assert p.skipped is None
    assert p.pose_validity_status is None
    assert p.pose_validity_reasons == []
    assert p.receptor_clashes == 0
    assert p.plif_recovery is None
    assert p.cnn_score is None and p.cnn_vs is None and p.cnn_affinity is None


# --------------------------------------------------------------------------- #
# End-to-end-light: vina.redock(real, CA-only receptor) returns skipped Pose
# instead of running vina. (Doesn't actually invoke the vina binary.)
# --------------------------------------------------------------------------- #
def test_vina_redock_real_skips_on_ca_only(tmp_path):
    ca = tmp_path / "ca.pdb"
    ca.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0\nEND\n")
    s = ProteinStructure(sequence="A", residues=[
        Residue(index=1, aa="A", ca=(0, 0, 0))
    ], pdb_path=str(ca))
    cfg = DockingConfig()
    ref = [LigandAtom(id="C0", element="C", coord=(0, 0, 0))]
    pose = vina.redock(
        "c", s, ref, cfg, tmp_path / "work",
        instability=0.0, backend=Backend.real, dry_run=False,
    )
    # Skipped honestly - no vina invocation, no fabricated score.
    assert pose.skipped == "skipped_no_full_atom_structure"
    assert pose.pose_validity_status == "unknown"
    assert pose.score == 0.0
