"""Vina contract (expert review P1): ligand pdbqt prep + out.pdbqt parse."""

import inspect

from evoliez.adapters import vina
from evoliez.adapters.vina import _parse_vina
from evoliez.types import LigandAtom


def test_real_vina_prepares_ligand_file():
    """Regression: _redock_real must create the ligand pdbqt (obabel) before
    invoking vina --ligand (previously the path was never written)."""
    src = inspect.getsource(vina._redock_real)
    # ligand structure is written and converted to pdbqt
    assert "write_min_pdb(lig_pdb" in src
    assert 'obabel", str(lig_pdb), "-O", str(lig_q)' in src
    # the converted ligand file is what vina receives
    assert '"--ligand", str(lig_q)' in src


def test_parse_vina_out_pdbqt(tmp_path):
    out = tmp_path / "vina_out.pdbqt"
    out.write_text(
        "MODEL 1\n"
        "REMARK VINA RESULT:    -8.7      0.000      0.000\n"
        "ATOM      1  C   LIG     1       0.0   0.0   0.0\n"
        "ENDMDL\n"
        "MODEL 2\n"
        "REMARK VINA RESULT:    -7.1      1.234      2.345\n"
        "ENDMDL\n"
    )
    ref = [LigandAtom(id="C0", element="C", coord=(0, 0, 0))]
    pose = _parse_vina("cand", out, ref)
    assert pose.method == "vina"
    assert abs(pose.score - (-8.7)) < 1e-9   # best (first) mode
    assert pose.ligand_atoms == ref
