"""Locally-verifiable slices of the REAL MD prep recipe.

The full real-MD path needs the conda-only openff/ambertools stack (server
only). But the bug-prone, frequently-iterated pieces - CA-only detection,
the mutant-structure sequence guard, and the RDKit ligand build at the
Boltz pose - are pure-Python / RDKit and MUST be regression-locked so we
stop discovering their breakage one server round-trip at a time.

rdkit-gated tests run in .venv-md (rdkit installed); they skip in the
fallback .venv. The non-rdkit tests run everywhere.
"""

import textwrap
from pathlib import Path

import pytest

from evoliez.adapters.openmm_engine import (
    _is_full_atom_pdb,
    _pdb_one_letter_seq,
)

_CA_ONLY = textwrap.dedent("""\
    ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00  0.00           C
    ATOM      2  CA  HIS A   2       3.800   0.000   0.000  1.00  0.00           C
    END
""")

_FULL_ATOM = textwrap.dedent("""\
    ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  MET A   1       1.460   0.000   0.000  1.00  0.00           C
    ATOM      3  C   MET A   1       2.000   1.420   0.000  1.00  0.00           C
    ATOM      4  O   MET A   1       3.220   1.560   0.000  1.00  0.00           O
    END
""")


def test_full_atom_vs_ca_only_detection(tmp_path):
    ca = tmp_path / "ca.pdb"
    ca.write_text(_CA_ONLY)
    fa = tmp_path / "fa.pdb"
    fa.write_text(_FULL_ATOM)
    assert _is_full_atom_pdb(ca) is False        # CA trace -> unusable for MD
    assert _is_full_atom_pdb(fa) is True         # has N/C/O -> real structure


def test_pdb_one_letter_seq(tmp_path):
    p = tmp_path / "x.pdb"
    p.write_text(_CA_ONLY)
    assert _pdb_one_letter_seq(p) == "MH"        # MET, HIS -> drives #2 guard


# --- RDKit ligand build at the Boltz pose (the bug-prone new code) -------- #
# rdkit-gated PER TEST (module-level importorskip would also skip the
# pure-Python tests above in the fallback .venv).

_LIG_PDB = textwrap.dedent("""\
    HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C
    HETATM    2  C2  LIG L   1       1.500   0.000   0.000  1.00  0.00           C
    HETATM    3  O1  LIG L   1       2.200   1.200   0.000  1.00  0.00           O
    CONECT    1    2
    CONECT    2    1    3
    CONECT    3    2
    END
""")


def test_ligand_rdkit_at_pose_uses_conect_and_template(tmp_path):
    pytest.importorskip("rdkit")
    from evoliez.adapters.openmm_engine import _ligand_rdkit_at_pose

    p = tmp_path / "complex.pdb"
    # protein noise must be ignored; only HETATM+CONECT define the ligand
    p.write_text(_FULL_ATOM.replace("END\n", "") + _LIG_PDB)

    mol = _ligand_rdkit_at_pose(p, "CCO")        # ethanol template
    heavy = [a for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    assert len(heavy) == 3                        # C,C,O from HETATM
    assert mol.GetNumAtoms() == 9                 # +6 explicit H (AddHs)
    assert mol.GetNumConformers() == 1            # pose carried
    c0 = mol.GetConformer().GetAtomPosition(0)
    assert (round(c0.x, 3), round(c0.y, 3), round(c0.z, 3)) == (0.0, 0.0, 0.0)


def test_ligand_rdkit_at_pose_raises_without_hetatm(tmp_path):
    pytest.importorskip("rdkit")
    from evoliez.adapters.openmm_engine import _ligand_rdkit_at_pose

    p = tmp_path / "noligand.pdb"
    p.write_text(_FULL_ATOM)
    with pytest.raises(ValueError):
        _ligand_rdkit_at_pose(p, "CCO")
