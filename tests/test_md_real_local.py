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
    # MUST be returned SANITIZED - Molecule.from_rdkit requires it, and an
    # unsanitized mol is exactly what broke openmmforcefields residue
    # matching ("No template for residue LIG").
    assert not mol.NeedsUpdatePropertyCache()
    assert mol.GetRingInfo().NumRings() >= 0  # ring perception ran


_BENZENE_PDB = textwrap.dedent("""\
    HETATM    1  C1  LIG L   1       1.390   0.000   0.000  1.00  0.00           C
    HETATM    2  C2  LIG L   1       0.695   1.204   0.000  1.00  0.00           C
    HETATM    3  C3  LIG L   1      -0.695   1.204   0.000  1.00  0.00           C
    HETATM    4  C4  LIG L   1      -1.390   0.000   0.000  1.00  0.00           C
    HETATM    5  C5  LIG L   1      -0.695  -1.204   0.000  1.00  0.00           C
    HETATM    6  C6  LIG L   1       0.695  -1.204   0.000  1.00  0.00           C
    CONECT    1    2    6
    CONECT    2    1    3
    CONECT    3    2    4
    CONECT    4    3    5
    CONECT    5    4    6
    CONECT    6    5    1
    END
""")


def test_ligand_rdkit_at_pose_returns_sanitized_aromatic(tmp_path):
    """The returned mol must be SANITIZED with aromaticity perceived - the
    actual root cause of the NADP 'No template for residue LIG' failure
    (unsanitized mol -> bad OpenFF graph -> openmmforcefields can't match).
    Aromaticity is only set if SanitizeMol ran."""
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from evoliez.adapters.openmm_engine import _ligand_rdkit_at_pose

    p = tmp_path / "benz.pdb"
    p.write_text(_BENZENE_PDB)
    mol = _ligand_rdkit_at_pose(p, "c1ccccc1")
    assert not mol.NeedsUpdatePropertyCache()              # sanitized
    assert mol.GetRingInfo().NumRings() >= 0  # ring perception ran
    arom = [a for a in mol.GetAtoms() if a.GetIsAromatic()]
    assert len(arom) == 6                                  # ring perceived
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.CanonSmiles(
        "c1ccccc1"
    )


def test_ligand_rdkit_at_pose_raises_without_hetatm(tmp_path):
    pytest.importorskip("rdkit")
    from evoliez.adapters.openmm_engine import _ligand_rdkit_at_pose

    p = tmp_path / "noligand.pdb"
    p.write_text(_FULL_ATOM)
    with pytest.raises(ValueError):
        _ligand_rdkit_at_pose(p, "CCO")


# --- protein-only prep must DROP the ligand (pdbfixer; no openff) -------- #
_PROT_PLUS_LIG = textwrap.dedent("""\
    ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
    ATOM      4  O   ALA A   1       3.228   1.563   0.000  1.00  0.00           O
    ATOM      5  CB  ALA A   1       1.988  -0.773  -1.199  1.00  0.00           C
    HETATM    6  C1  LIG B   1      10.000  10.000  10.000  1.00  0.00           C
    CONECT    6
    END
""")


def test_protein_only_pdbfixed_strips_heterogen(tmp_path):
    """Candidate #1 of the real-MD blocker: the Boltz H-less LIG must NOT
    survive structure prep into create_system. removeHeterogens(False) must
    yield a PROTEIN-ONLY topology - locked locally so this can't regress."""
    pytest.importorskip("pdbfixer")
    pytest.importorskip("openmm")
    from evoliez.adapters.openmm_engine import (
        _STD_RES,
        _protein_only_pdbfixed,
    )

    p = tmp_path / "complex.pdb"
    p.write_text(_PROT_PLUS_LIG)
    topo, pos = _protein_only_pdbfixed(p)
    assert topo is not None                       # pdbfixer present here
    names = {r.name for r in topo.residues()}
    assert "LIG" not in names                      # heterogen removed
    assert names and names <= _STD_RES             # protein-only invariant
