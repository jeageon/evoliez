"""Real gnina/diffdock poses must carry docked COORDINATES + RMSD-to-reference.

Previously gnina/diffdock returned ligand_atoms=reference and rmsd=None, so s09
read a falsely 'perfect' redocking_consistency (1.0) and never flagged a ligand
that left the pocket. The shared SDF parser + lock helper now adopt the docked
coordinates so pose-escape / redocking-consistency are real for all three
dockers, matching what Vina already did.
"""

from __future__ import annotations

from pathlib import Path

from evoliez.adapters.base import lock_pose_to_reference, parse_sdf_first_pose
from evoliez.types import LigandAtom

FX = Path(__file__).parent / "fixtures" / "tool_outputs"


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
