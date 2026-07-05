"""ROADMAP_V5 V5-2 — co-fold the physiological metal ion (e.g. Mg2+) in Boltz so the anionic
substrate + cofactor are pre-organized around it, instead of electrostatically repelling to a
non-productive starting pose. The ion is co-folded ONLY: it is DROPPED from the parsed ligand
set (MD re-places it deterministically as an Amber ion), so s09/s10 see exactly the organic
ligand set they saw before — only the substrate coordinates change."""
import tempfile
from pathlib import Path

from evoliez.adapters.boltz import _METAL_ION_ELEMENTS, _build_spec, _parse_pdb_atoms
from evoliez.config import ComplexPredictionConfig
from evoliez.mechanism.spec import metal_ion_ccd
from evoliez.types import Ligand


class _RS:
    def __init__(self, ms):
        self.metal_state = ms


class _Mech:
    def __init__(self, ms):
        self.reaction_state = _RS(ms)


def test_metal_ion_ccd_parses_leading_element_not_suffix():
    assert metal_ion_ccd(_Mech("Mg2+_bridged")) == "MG"
    # "_coordinated" must NOT false-match Co/Cu — the LEADING symbol wins
    assert metal_ion_ccd(_Mech("Zn2+_coordinated")) == "ZN"
    assert metal_ion_ccd(_Mech("Mn2+")) == "MN"
    assert metal_ion_ccd(_Mech(None)) is None
    assert metal_ion_ccd(_Mech("")) is None
    assert metal_ion_ccd(None) is None


def _lig():
    return Ligand(id="3HP", smiles="OC(=O)CCO")


def test_build_spec_emits_metal_as_ccd_ion_after_extra_chains():
    atp = Ligand(id="ATP", smiles="Nc1ncnc2n(cnc12)C1OC(CO)C(O)C1O")
    spec, _ = _build_spec("wt", "MKV", _lig(), ComplexPredictionConfig(),
                          Path(tempfile.gettempdir()), None,
                          extra_ligands=[atp], metal_ccd="MG")
    ligs = [s["ligand"] for s in spec["sequences"] if "ligand" in s]
    # protein A is not a ligand; B=design(3HP, smiles), C=ATP(smiles), D=MG(ccd)
    mg = [l for l in ligs if l.get("ccd") == "MG"]
    assert len(mg) == 1
    assert mg[0]["id"] == "D"                      # next free chain after B(design)+C(extra)
    assert "smiles" not in mg[0]                    # ccd XOR smiles (Boltz schema)


def test_build_spec_no_metal_when_none_is_backward_identical():
    atp = Ligand(id="ATP", smiles="Nc1ncnc2n(cnc12)C1OC(CO)C(O)C1O")
    a, _ = _build_spec("wt", "MKV", _lig(), ComplexPredictionConfig(),
                       Path(tempfile.gettempdir()), None, extra_ligands=[atp])
    b, _ = _build_spec("wt", "MKV", _lig(), ComplexPredictionConfig(),
                       Path(tempfile.gettempdir()), None, extra_ligands=[atp],
                       metal_ccd=None)
    assert a == b
    assert not any(s.get("ligand", {}).get("ccd") for s in a["sequences"])


def test_parser_drops_cofolded_metal_but_keeps_organic_ligands():
    pdb = (
        "ATOM      1  CA  MET A   1      10.000  10.000  10.000  1.00  0.00           C\n"
        "HETATM    2  C1  3HP B   1      20.000  20.000  20.000  1.00  0.00           C\n"
        "HETATM    3  O1  3HP B   1      21.200  20.000  20.000  1.00  0.00           O\n"
        "HETATM    4  PA  ATP C   1      24.000  20.000  20.000  1.00  0.00           P\n"
        "HETATM    5 MG    MG D   1      22.000  20.000  20.000  1.00  0.00          MG\n"
        "END\n"
    )
    f = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    f.write(pdb)
    f.close()
    _res, lig, extra = _parse_pdb_atoms(Path(f.name))
    all_el = {a.element.upper() for a in lig}
    all_el |= {a.element.upper() for ch in extra.values() for a in ch}
    assert "MG" not in all_el                       # co-folded metal DROPPED from ligand set
    assert "P" in all_el                            # ATP (phosphate) survives as an extra chain
    assert "MG" in _METAL_ION_ELEMENTS              # guard the drop-set membership
