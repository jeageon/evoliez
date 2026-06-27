"""Unit tests for the WT-anchored mutant builder's pure helpers (no OpenMM).

The full PDBFixer build path is verified against the real WT FDH complex
(scripts run) — these cover the parsing/normalisation logic that has no engine dep.
"""
from evoliez.md.anchored_build import (
    _AA1TO3, _atom_chain, _norm_mut, _split_protein_ligand,
)


def _line(rec, ser, name, resn, ch, rs):
    return (f"{rec:<6}{ser:>5} {name:<4} {resn:>3} {ch}{rs:>4}    "
            f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C")


def test_split_protein_ligand():
    txt = "\n".join([
        _line("ATOM", 1, "CA", "ALA", "A", 1),
        _line("ATOM", 2, "CA", "GLY", "A", 2),
        _line("HETATM", 3, "C1", "UNK", "A", 900),
        "CONECT    3    4",
        "TER",
        "END",
    ])
    prot, het = _split_protein_ligand(txt)
    assert len(prot) == 2 and len(het) == 1
    assert all(p.startswith("ATOM") for p in prot)
    assert het[0].startswith("HETATM")


def test_atom_chain_majority():
    lines = [_line("ATOM", i, "CA", "ALA", "A", i) for i in range(5)]
    lines += [_line("ATOM", 99, "CA", "ALA", "B", 99)]
    assert _atom_chain(lines) == "A"


def test_norm_mut_object_and_tuple():
    class M:
        wt, position, mut = "S", 340, "G"
    assert _norm_mut(M()) == ("S", 340, "G")
    assert _norm_mut(("n", 260, "h")) == ("N", 260, "H")
    assert _norm_mut(["P", 262, "R"]) == ("P", 262, "R")
    assert _norm_mut("garbage") is None


def test_aa_map_complete():
    assert _AA1TO3["S"] == "SER" and _AA1TO3["G"] == "GLY"
    assert _AA1TO3["H"] == "HIS" and _AA1TO3["V"] == "VAL"
    assert len(_AA1TO3) == 20
