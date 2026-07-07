"""s05 multi-ligand co-docking: dock every input ligand, each with the others
kept as fixed receptor context."""

from __future__ import annotations

from pathlib import Path

from evoliez.adapters.base import (
    full_atom_receptor_pdb,
    het_chains_in_pdb,
    parse_pdb_het_chain,
)
from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.db.schema import DockingPose
from evoliez.pipeline import Pipeline
from evoliez.types import ProteinStructure
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]

_COMPLEX = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       3.200   1.600   0.000  1.00  0.00           O\n"
    "HETATM    5  C1  LIG B 999       5.000   0.000   0.000  1.00  0.00           C\n"
    "HETATM    6  N1  LIG B 999       6.200   0.300   0.000  1.00  0.00           N\n"
    "HETATM    7  C2  FMT C 999       8.000   0.000   0.000  1.00  0.00           C\n"
    "HETATM    8  O2  FMT C 999       9.000   0.500   0.000  1.00  0.00           O\n"
)


def test_receptor_keeps_only_context_het_chains(tmp_path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(_COMPLEX)
    struct = ProteinStructure(sequence="A", residues=[], pdb_path=str(pdb))
    out = tmp_path / "rec.pdb"
    # docking the design ligand (chain B) -> keep chain C (formate) as context
    assert full_atom_receptor_pdb(struct, out, keep_het_chains={"C"})
    txt = out.read_text()
    assert "ATOM" in txt
    assert "FMT C 999" in txt           # context ligand kept
    assert "LIG B 999" not in txt       # the ligand being docked is dropped
    # default (no context) drops ALL hetatm
    assert full_atom_receptor_pdb(struct, out)
    assert "HETATM" not in out.read_text()


def test_het_chain_helpers(tmp_path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(_COMPLEX)
    assert het_chains_in_pdb(str(pdb)) == ["B", "C"]
    atoms = parse_pdb_het_chain(str(pdb), "C")
    assert len(atoms) == 2 and {a.element for a in atoms} == {"C", "O"}
    assert atoms[0].coord[0] == 8.0


def test_s05_docks_every_input_ligand(tmp_path):
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "input.extra_ligands": [
                {"id": "formate", "type": "smiles", "value": "[O-]C=O"}],
            "validation": {"redocking": {"methods": ["gnina", "diffdock"]}},
            "mutation_generation": {"methods": ["chemistry_rules"], "max_candidates": 10},
            "gnn": {"build_dataset": False},
        },
    )
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s05_docking")
    with ctx.store.session() as s:
        rows = s.query(DockingPose).all()
        cands = {r.candidate_id for r in rows}
        methods_per = {}
        for r in rows:
            methods_per.setdefault(r.candidate_id, set()).add(r.method)
    # BOTH the design ligand and the co-modelled formate were docked
    assert "wt" in cands and "wt__formate" in cands
    # each by both configured methods
    assert methods_per["wt"] == {"gnina", "diffdock"}
    assert methods_per["wt__formate"] == {"gnina", "diffdock"}
