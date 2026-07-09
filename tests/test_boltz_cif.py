"""Boltz output contract (expert review P1): mmCIF parse + --output_format."""

import inspect

from evoliez.adapters import boltz
from evoliez.adapters.boltz import _parse_cif_atoms

_CIF = """data_pred
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 N N ALA A 1 1.000 2.000 3.000
ATOM 2 C CA ALA A 1 1.500 2.500 3.500
ATOM 3 C CA GLY A 2 4.000 5.000 6.000
HETATM 4 O O1 LIG B 1 7.000 8.000 9.000
HETATM 5 P P1 LIG B 1 7.500 8.500 9.500
#
"""


def test_boltz_forces_pdb_output_format():
    # the cmd (incl. --output_format pdb) is built in the shared
    # _build_predict_cmd, used by both the per-rep and the GPU-batched paths.
    src = inspect.getsource(boltz._build_predict_cmd)
    assert '"--output_format", "pdb"' in src


def test_parse_cif_atom_site_loop(tmp_path):
    f = tmp_path / "pred.cif"
    f.write_text(_CIF)
    residues, lig, _extra = _parse_cif_atoms(f)   # 3-tuple: (+ per-chain extra ligand atoms)
    assert [r.index for r in residues] == [1, 2]          # CA only
    assert residues[0].ca == (1.5, 2.5, 3.5)
    assert len(lig) == 2 and lig[0].element == "O"        # HETATM ligand
    assert lig[1].coord == (7.5, 8.5, 9.5)


def test_parse_real_structure_dispatches_cif(tmp_path):
    from evoliez.config import LigandInput
    from evoliez.features.ligand import parse_ligand

    f = tmp_path / "model_0.cif"
    f.write_text(_CIF)
    lig = parse_ligand(LigandInput(id="L", type="smiles", value="OP(O)=O"))
    cx = boltz._parse_real_structure(f, "AG", lig)
    assert len(cx.structure.residues) == 2
    assert cx.structure.residues[0].aa == "A"   # sequence applied
    assert len(cx.ligand.atoms) >= 1
