"""v2 multi-ligand: the Boltz structure parser separates the primary ligand chain from the
co-modelled extra ligand chains (cofactor/substrate/metal), so the full functional state is
available to the s06 design mask — not just one main ligand."""
import tempfile
from pathlib import Path

from evoliez.adapters.boltz import _parse_structure_atoms


def _line(rec, ser, name, resn, ch, rs):
    return (f"{rec:<6}{ser:>5} {name:<4} {resn:>3} {ch}{rs:>4}    "
            f"{1.0 * ser:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")


def test_parser_separates_primary_and_extra_chains():
    pdb = "\n".join([
        _line("ATOM", 1, "CA", "ALA", "A", 1),       # protein residue
        _line("HETATM", 2, "C1", "LIG", "B", 900),   # primary ligand (chain B)
        _line("HETATM", 3, "C2", "LIG", "B", 900),
        _line("HETATM", 4, "ZN", "ZN", "C", 901),    # extra: metal (chain C)
        _line("HETATM", 5, "O", "HOH", "D", 902),    # extra: substrate water (chain D)
    ])
    p = Path(tempfile.mktemp(suffix=".pdb"))
    p.write_text(pdb)
    residues, primary, extra = _parse_structure_atoms(p)
    assert len(residues) == 1                         # protein CA
    assert len(primary) == 2                          # primary ligand kept intact (chain B)
    assert set(extra) == {"C", "D"}                   # extra ligand chains collected
    assert len(extra["C"]) == 1 and len(extra["D"]) == 1


def test_single_ligand_has_no_extra():
    pdb = "\n".join([
        _line("ATOM", 1, "CA", "ALA", "A", 1),
        _line("HETATM", 2, "C1", "LIG", "B", 900),
    ])
    p = Path(tempfile.mktemp(suffix=".pdb"))
    p.write_text(pdb)
    _res, primary, extra = _parse_structure_atoms(p)
    assert len(primary) == 1 and extra == {}          # back-compat: single ligand, no extras
