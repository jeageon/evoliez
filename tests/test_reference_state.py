"""v2 Phase A — curated reference-state hard gates."""
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from evoliez.reference_state import load_reference_complex, validate_reference


@dataclass
class _R:
    index: int


_RES = [_R(i) for i in range(1, 101)]            # 100 residues, numbered 1..100
_PRIMARY = list(range(48))                        # 48-atom primary ligand
_EXTRA = {"C": [1], "D": [2]}                     # 2 extra ligand chains


def _g(**kw):
    base = dict(target_sequence="A" * 100, catalytic_positions=[10, 20],
                design_ligand_natoms=48, expected_extra_ligands=2,
                has_charge_provenance=True)
    base.update(kw)
    return validate_reference(_RES, _PRIMARY, _EXTRA, **base)


def test_all_gates_pass():
    assert _g().passed


def test_catalytic_numbering_mismatch_fails():
    g = _g(catalytic_positions=[10, 999])         # 999 not in the structure
    assert not g.passed and not g.checks["catalytic_residues_present"]


def test_ligand_atom_map_mismatch_fails():
    g = validate_reference(_RES, list(range(10)), _EXTRA, target_sequence="A" * 100,
                           catalytic_positions=[10], design_ligand_natoms=48,
                           expected_extra_ligands=2, has_charge_provenance=True)
    assert not g.passed and not g.checks["reference_atom_map_verified"]


def test_missing_extra_chain_and_charge_fail():
    g = validate_reference(_RES, _PRIMARY, {"C": [1]}, target_sequence="A" * 100,
                           catalytic_positions=[10], design_ligand_natoms=48,
                           expected_extra_ligands=2, has_charge_provenance=False)
    assert not g.passed
    assert not g.checks["ligand_role_chain_map"]
    assert not g.checks["charge_protonation_provenance"]


def test_load_reference_complex_smoke():
    def L(rec, ser, name, resn, ch, rs):
        return (f"{rec:<6}{ser:>5} {name:<4} {resn:>3} {ch}{rs:>4}    "
                f"{1.0 * ser:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")
    pdb = "\n".join(
        [L("ATOM", i, "CA", "ALA", "A", i) for i in range(1, 6)]
        + [L("HETATM", 10, "C1", "LIG", "B", 900),
           L("HETATM", 11, "ZN", "ZN", "C", 901)])
    p = Path(tempfile.mktemp(suffix=".pdb"))
    p.write_text(pdb)
    lig = SimpleNamespace(id="lig", smiles="C", atoms=[1], formal_charge=0,
                          charges_mol2=None, allow_am1bcc=True)
    cx, gates = load_reference_complex(
        str(p), target_sequence="AAAAA", ligand=lig,
        extra_ligands=[SimpleNamespace(id="zn")], catalytic_positions=[3])
    assert cx.method == "curated_pdb"
    assert "C" in cx.extra_ligand_atoms              # the metal chain is carried
    assert gates.checks["catalytic_residues_present"]  # residue 3 present
