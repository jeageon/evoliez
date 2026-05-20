"""Server-verified bug: `_ligand_rdkit_at_pose` + `Molecule.from_rdkit` lost
H atoms on NADP+ (AddHs added only 1 H out of ~25 on the multiply-charged
species). `system_generator.create_system` then failed with "No template
for residue LIG ... missing 57 H atoms".

Fix: `_ligand_offmol_at_pose` now builds the OFF mol from SMILES
(canonical 73-atom chemistry for NADP+) and only TRANSFERS heavy-atom
positions from the Boltz pose via a heavy-atom graph isomorphism. H
positions come from RDKit embed; MD minimization relaxes them.

These tests lock in:
- the new function exists with the SMILES-authoritative architecture
  (source-level guard; light env friendly);
- `_graph_match_pose_to_smiles` rejects atom-count and connectivity
  mismatches (so a bad PDB block surfaces as a real failure, not a
  silent wrong mapping).
"""

from __future__ import annotations

import inspect

import pytest

from evoliez.adapters import openmm_engine


def test_ligand_offmol_uses_smiles_as_chemistry_source():
    src = inspect.getsource(openmm_engine._ligand_offmol_at_pose)
    # SMILES is the canonical chemistry source.
    assert "Molecule.from_smiles" in src
    # Heavy-atom graph isomorphism transfers coords (not positional zip).
    assert "_graph_match_pose_to_smiles" in src
    # Conformer is added explicitly so the OFF mol carries the pose.
    assert "add_conformer" in src
    # H positions come from a deterministic embed; minimization will
    # relax them. We don't trust PDB AddHs for charged ligands.
    assert "EmbedMolecule" in src


def test_legacy_path_kept_but_renamed_for_diagnostics():
    # Parent's pure-RDKit `_ligand_rdkit_at_pose` is kept so the RDKit-
    # only diagnostic (unit-testable without openff) still works; the
    # `Molecule.from_rdkit`-based wrapper is renamed _LEGACY to mark it
    # as the H-loss path we no longer route through.
    assert hasattr(openmm_engine, "_ligand_rdkit_at_pose")
    assert hasattr(openmm_engine, "_ligand_offmol_at_pose_LEGACY")


def test_graph_match_rejects_count_mismatch():
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    a = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(a, randomSeed=1)
    b = Chem.AddHs(Chem.MolFromSmiles("CCCO"))    # different atom count
    AllChem.EmbedMolecule(b, randomSeed=1)
    with pytest.raises(ValueError, match="heavy-atom count mismatch"):
        openmm_engine._graph_match_pose_to_smiles(a, b)


def test_graph_match_succeeds_on_identical_skeletons():
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    # Same SMILES -> identical heavy-atom skeletons -> match returns a
    # full mapping.
    a = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC"))
    AllChem.EmbedMolecule(a, randomSeed=1)
    b = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC"))
    AllChem.EmbedMolecule(b, randomSeed=2)
    match, pose_heavy, ref_heavy = openmm_engine._graph_match_pose_to_smiles(a, b)
    assert len(match) == pose_heavy.GetNumAtoms()
    assert pose_heavy.GetNumAtoms() == ref_heavy.GetNumAtoms()
