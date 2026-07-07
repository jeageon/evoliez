"""Multi-residue ligand splitting + per-molecule template matching for the
OpenMM catalytic-screen path (evoliez.adapters.openmm_engine).

These are the pure RDKit pieces of the "NADP + formate as SEPARATE residues"
index map: split a complex PDB's HETATM into per-molecule groups and match each
group to its SMILES (no residue-name reliance). RDKit-only, so they run in the
full venv and skip where RDKit is absent. The OpenMM/OpenFF assembly itself is
server-validated.
"""
import pytest

pytest.importorskip("rdkit")

from evoliez.adapters.openmm_engine import (_hetatm_groups, _ligands_at_pose,
                                            _rdkit_from_het_lines)


def _two_ligand_pdb(tmp_path):
    """A complex PDB with TWO distinct small molecules as separate HETATM
    residues (formate + acetate), with CONECT, written by RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def prep(smi, resname, resnum):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=7)
        for atom in m.GetAtoms():
            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName(resname.ljust(3)[:3])
            info.SetResidueNumber(resnum)
            info.SetChainId("L")
            info.SetIsHeteroAtom(True)
            atom.SetMonomerInfo(info)
        return m

    combined = Chem.CombineMols(prep("[O-]C=O", "FMT", 1),
                                prep("CC(=O)[O-]", "ACT", 2))
    p = tmp_path / "two_ligand.pdb"
    p.write_text(Chem.MolToPDBBlock(combined))
    return p


def test_hetatm_groups_separates_residues(tmp_path):
    p = _two_ligand_pdb(tmp_path)
    groups = _hetatm_groups(p.read_text())
    assert len(groups) == 2                      # formate and acetate split apart
    # each group carries its own CONECT records (bonds for that molecule)
    for _key, lines in groups:
        assert any(ln.startswith("HETATM") for ln in lines)
        assert any(ln.startswith("CONECT") for ln in lines)


def test_ligands_at_pose_matches_each_to_its_smiles(tmp_path):
    p = _two_ligand_pdb(tmp_path)
    out = _ligands_at_pose(p, [("formate", "[O-]C=O"), ("acetate", "CC(=O)[O-]")])
    assert [i for i, _ in out] == ["formate", "acetate"]
    heavy = {i: sum(1 for a in rd.GetAtoms() if a.GetAtomicNum() > 1)
             for i, rd in out}
    assert heavy["formate"] == 3 and heavy["acetate"] == 4   # right molecule each


def test_heavy_count_gate_rejects_wrong_template(tmp_path):
    """A smaller template (formate) must NOT partial-match a larger group
    (acetate) - the heavy-atom-count gate forces a whole-molecule match."""
    p = _two_ligand_pdb(tmp_path)
    groups = _hetatm_groups(p.read_text())
    # the acetate group (4 heavy) vs the formate template (3 heavy) -> no match
    acetate_lines = max(groups, key=lambda g: sum(
        1 for ln in g[1] if ln.startswith("HETATM")))[1]
    assert _rdkit_from_het_lines(acetate_lines, "[O-]C=O") is None
    assert _rdkit_from_het_lines(acetate_lines, "CC(=O)[O-]") is not None


def test_ligands_at_pose_reports_missing_partner(tmp_path):
    """Only the design ligand present -> the unmatched extra is dropped (and
    logged), the design is still returned."""
    p = _two_ligand_pdb(tmp_path)
    out = _ligands_at_pose(p, [("formate", "[O-]C=O"), ("absent", "c1ccccc1")])
    assert [i for i, _ in out] == ["formate"]    # benzene absent -> not returned
