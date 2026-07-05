"""ROADMAP_V5 V5-2 — co-fold the physiological metal ion (e.g. Mg2+) in Boltz so the anionic
substrate + cofactor are pre-organized around it, instead of electrostatically repelling to a
non-productive starting pose. The ion is co-folded ONLY: it is DROPPED from the parsed ligand
set (MD re-places it deterministically as an Amber ion), so s09/s10 see exactly the organic
ligand set they saw before — only the substrate coordinates change.

Scope discipline (commercial-generality): the drop is keyed on the CALLER-SUPPLIED CCD of the
ion EvoLiEZ injected into THIS spec (`fold_only_metal`), and only fires for a true MONATOMIC ion
(resname == element == that CCD). A target's OWN metal — heme Fe, a catalytic Zn, a metal-centred
organic ligand — is NEVER dropped (fold_only_metal is None off the co-fold path, and resname/element
never match a polyatomic ligand). These tests lock that boundary."""
import tempfile
from pathlib import Path

from evoliez.adapters.boltz import _KNOWN_METAL_CCDS, _build_spec, _parse_pdb_atoms
from evoliez.config import ComplexPredictionConfig
from evoliez.mechanism.spec import metal_ion_ccd
from evoliez.types import Ligand


class _RS:
    def __init__(self, ms):
        self.metal_state = ms


class _Mech:
    def __init__(self, ms):
        self.reaction_state = _RS(ms)


def _elements(path, fold_only_metal=None):
    """All element symbols across the primary + extra ligand chains a parse yields."""
    _res, lig, extra = _parse_pdb_atoms(Path(path), fold_only_metal=fold_only_metal)
    els = {a.element.upper() for a in lig}
    els |= {a.element.upper() for ch in extra.values() for a in ch}
    return els


def _write(pdb):
    f = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    f.write(pdb)
    f.close()
    return f.name


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
    assert mg[0]["id"] == "D"                       # next free chain after B(design)+C(extra)
    assert "smiles" not in mg[0]                     # ccd XOR smiles (Boltz schema)


def test_build_spec_no_metal_when_none_is_backward_identical():
    atp = Ligand(id="ATP", smiles="Nc1ncnc2n(cnc12)C1OC(CO)C(O)C1O")
    a, _ = _build_spec("wt", "MKV", _lig(), ComplexPredictionConfig(),
                       Path(tempfile.gettempdir()), None, extra_ligands=[atp])
    b, _ = _build_spec("wt", "MKV", _lig(), ComplexPredictionConfig(),
                       Path(tempfile.gettempdir()), None, extra_ligands=[atp],
                       metal_ccd=None)
    assert a == b
    assert not any(s.get("ligand", {}).get("ccd") for s in a["sequences"])


# --- reviewer-requested parser-scope regression tests -----------------------------------------

_PDB_MG = (
    "ATOM      1  CA  MET A   1      10.000  10.000  10.000  1.00  0.00           C\n"
    "HETATM    2  C1  3HP B   1      20.000  20.000  20.000  1.00  0.00           C\n"
    "HETATM    3  O1  3HP B   1      21.200  20.000  20.000  1.00  0.00           O\n"
    "HETATM    4  PA  ATP C   1      24.000  20.000  20.000  1.00  0.00           P\n"
    "HETATM    5 MG    MG D   1      22.000  20.000  20.000  1.00  0.00          MG\n"
    "END\n"
)


def test_parser_drops_explicit_fold_only_mg_chain():
    """When EvoLiEZ states it co-folded MG, that monatomic ion is removed from the ligand set;
    the organic ligands (3HP design, ATP cofactor) are untouched."""
    name = _write(_PDB_MG)
    els = _elements(name, fold_only_metal="MG")
    assert "MG" not in els                           # the injected metal is dropped
    assert "P" in els and "O" in els                 # ATP + 3HP survive
    # and it only ever drops a RECOGNIZED metal CCD (allowlist guards a stray value)
    assert "MG" in _KNOWN_METAL_CCDS


def test_parser_keeps_metal_when_not_declared_fold_only():
    """Off the co-fold path (fold_only_metal=None) NOTHING is dropped — the parser stays
    general-purpose; a caller that never injected a metal keeps every atom."""
    name = _write(_PDB_MG)
    assert "MG" in _elements(name, fold_only_metal=None)


def test_parser_keeps_heme_or_metal_cofactor_fe_atom():
    """A target's OWN catalytic metal (heme Fe) must survive even while we co-fold Mg: the Fe
    sits in a HEM residue (resname != injected CCD), so it is never dropped."""
    pdb = (
        "ATOM      1  CA  MET A   1      10.000  10.000  10.000  1.00  0.00           C\n"
        "HETATM    2  C1  3HP B   1      20.000  20.000  20.000  1.00  0.00           C\n"
        "HETATM    3 FE   HEM C   1      30.000  30.000  30.000  1.00  0.00          FE\n"
        "HETATM    4  NA  HEM C   1      31.000  30.000  30.000  1.00  0.00           N\n"
        "HETATM    5 MG    MG D   1      22.000  20.000  20.000  1.00  0.00          MG\n"
        "END\n"
    )
    name = _write(pdb)
    els = _elements(name, fold_only_metal="MG")       # we injected Mg, NOT Fe
    assert "FE" in els                                # heme iron kept
    assert "MG" not in els                            # our co-folded Mg dropped
    # even if a caller (wrongly) named FE as fold-only, the heme Fe lives in resname HEM != FE
    assert "FE" in _elements(name, fold_only_metal="FE")


def test_parser_keeps_organic_ligand_with_metal_center():
    """A metal that is part of a polyatomic ligand (resname LIG, a metal-centred cofactor) is
    kept: the element-match guard fires only for a monatomic ion whose resname==element==CCD."""
    pdb = (
        "ATOM      1  CA  MET A   1      10.000  10.000  10.000  1.00  0.00           C\n"
        "HETATM    2  C1  LIG B   1      20.000  20.000  20.000  1.00  0.00           C\n"
        "HETATM    3 ZN   LIG B   1      21.000  20.000  20.000  1.00  0.00          ZN\n"
        "HETATM    4  N1  LIG B   1      22.000  20.000  20.000  1.00  0.00           N\n"
        "HETATM    5 MG    MG C   1      25.000  20.000  20.000  1.00  0.00          MG\n"
        "END\n"
    )
    name = _write(pdb)
    # inject MG; also probe with ZN named fold-only to prove resname LIG != ZN protects the center
    for fom in ("MG", "ZN"):
        els = _elements(name, fold_only_metal=fom)
        assert "ZN" in els                            # ligand's own metal centre kept
        assert "C" in els and "N" in els              # rest of the organic ligand kept
    assert "MG" not in _elements(name, fold_only_metal="MG")   # only the monatomic co-fold drops
