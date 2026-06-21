"""Real gnina/diffdock poses must carry docked COORDINATES + RMSD-to-reference.

Previously gnina/diffdock returned ligand_atoms=reference and rmsd=None, so s09
read a falsely 'perfect' redocking_consistency (1.0) and never flagged a ligand
that left the pocket. The shared SDF parser + lock helper now adopt the docked
coordinates so pose-escape / redocking-consistency are real for all three
dockers, matching what Vina already did.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evoliez.adapters.base import (
    lock_pose_to_reference,
    parse_sdf_all_poses,
    parse_sdf_first_pose,
)
from evoliez.types import LigandAtom

FX = Path(__file__).parent / "fixtures" / "tool_outputs"

# gnina writes its modes with a BLANK molblock title (RDKit-style: blank title,
# program, comment, then the V2000 counts line). The legacy parse_sdf_all_poses
# stripped ALL leading blank lines then read index 3 — landing on an ATOM line —
# and dropped EVERY record, so all gnina modes silently lost their coordinates
# and lock_pose_to_reference substituted the reference (a false RMSD of 0).
_BLANK_TITLE_SDF = (
    "\n\n\n"
    "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
    "    1.0000    2.0000    3.0000 O   0  0\n"
    "    4.0000    5.0000    6.0000 P   0  0\n"
    "  1  2  1  0\nM  END\n>  <minimizedAffinity>\n-7.85\n\n$$$$\n"
    "\n\n\n"
    "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
    "    7.0000    8.0000    9.0000 O   0  0\n"
    "   10.0000   11.0000   12.0000 P   0  0\n"
    "  1  2  1  0\nM  END\n>  <minimizedAffinity>\n-6.20\n\n$$$$\n"
)


def test_parse_all_poses_handles_blank_molblock_title(tmp_path):
    """A gnina-style multi-mode SDF with BLANK molblock titles must parse to ALL
    modes with their real coordinates (regression: it parsed to ZERO)."""
    p = tmp_path / "blank.sdf"
    p.write_text(_BLANK_TITLE_SDF)
    poses = parse_sdf_all_poses(p)
    assert len(poses) == 2
    assert [a.element for a in poses[0]] == ["O", "P"]
    assert poses[0][0].coord == (1.0, 2.0, 3.0)
    assert poses[1][1].coord == (10.0, 11.0, 12.0)


def test_parse_all_poses_titled_sdf_unchanged():
    """The titled-SDF fixture (non-blank title) still parses to its two modes —
    the counts-line anchor is byte-equivalent for a well-titled record."""
    poses = parse_sdf_all_poses(FX / "gnina" / "out.sdf")
    assert len(poses) == 2
    assert poses[0][0].coord == (8.0, 8.7, 9.6)


def test_sdf_first_pose_coords():
    atoms = parse_sdf_first_pose(FX / "gnina" / "out.sdf")
    assert [a.element for a in atoms] == ["O", "P"]
    assert atoms[0].coord == (8.0, 8.7, 9.6)
    # only the FIRST molecule (best pose), not the second in the multi-mol SDF
    assert len(atoms) == 2


def test_lock_computes_real_rmsd_for_displaced_reference():
    parsed = parse_sdf_first_pose(FX / "diffdock" / "rank1_confidence-0.42.sdf")
    # reference displaced by (1,1,1) from each docked atom -> rmsd = sqrt(3)
    ref = [
        LigandAtom(id="O0", element="O", coord=(9.0, 9.7, 10.6)),
        LigandAtom(id="P1", element="P", coord=(8.6, 9.4, 10.3)),
    ]
    locked, rmsd = lock_pose_to_reference(parsed, ref, method="diffdock")
    assert [a.id for a in locked] == ["O0", "P1"]      # canonical ids kept
    assert locked[0].coord == (8.0, 8.7, 9.6)          # docked xyz adopted
    assert rmsd is not None and abs(rmsd - 1.732) < 1e-3


def test_unparseable_sdf_falls_back_to_reference():
    ref = [LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0))]
    locked, rmsd = lock_pose_to_reference([], ref, method="gnina")
    assert locked == ref and rmsd is None


def test_template_lock_recovers_docked_coords_with_canonical_ids():
    """The SMILES-template lock adopts the REAL docked coordinates onto the
    canonical reference ids (and a true RMSD), even when no LigandAtom list is
    given — the path that recovers a flexible cofactor's pose where the
    coordinate-graph lock falls back to the reference (RMSD 0)."""
    rdkit = pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Ethanol (C C O) — a small asymmetric molecule with a clean template match.
    smiles = "CCO"
    tmpl = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7)
    m = Chem.RemoveHs(m)
    conf = m.GetConformer()
    docked = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
               conf.GetAtomPosition(i).z) for i in range(m.GetNumAtoms())]
    pose_text = Chem.MolToMolBlock(m)
    # A reference at a KNOWN offset (+10 in x) from the docked frame -> rmsd 10.
    ref = [LigandAtom(id=f"{a.GetSymbol()}{i}", element=a.GetSymbol(),
                      coord=(docked[i][0] + 10.0, docked[i][1], docked[i][2]))
           for i, a in enumerate(m.GetAtoms())]
    locked, rmsd = lock_pose_to_reference(
        [], ref, method="gnina", pose_text=pose_text, pose_fmt="sdf",
        smiles=smiles)
    # canonical ids kept, DOCKED xyz adopted (not the reference), rmsd ~ 10.
    assert [a.id for a in locked] == [a.id for a in ref]
    assert abs(locked[0].coord[0] - docked[0][0]) < 1e-3
    assert rmsd is not None and abs(rmsd - 10.0) < 1e-2
