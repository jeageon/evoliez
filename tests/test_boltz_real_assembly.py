"""Regression guards for the real-Boltz assembly path (ultra-review G1-2/G1-3).

Two production-only defects this pins:
  * per-residue pLDDT was parsed but never written onto Residue.plddt, so every
    per-residue confidence feature silently read the 0.0 dataclass default on
    real runs (the mock path set it, so CI stayed green);
  * the diffusion ensemble reused ONE ligand pose for every sample, so the
    priority-#1 ensemble contact-frequency feature was degenerate on real runs.

These exercise the real-output assembly without the Boltz binary by building a
finished-run output directory on disk and parsing it the way _predict_real does.
"""

import json
from pathlib import Path

import numpy as np

from evoliez.adapters.boltz import (
    _assemble_real_complex,
    _assign_residue_plddt,
    _load_plddt,
)
from evoliez.types import Ligand, LigandAtom, Residue

FX = Path(__file__).parent / "fixtures" / "tool_outputs"


# --------------------------------------------------------------------------- #
# G1-2: per-residue pLDDT actually reaches Residue.plddt
# --------------------------------------------------------------------------- #
def test_assign_residue_plddt_maps_protein_tokens_and_normalizes():
    residues = [Residue(index=i + 1, aa="A") for i in range(3)]
    # 5 tokens: 3 protein + 2 ligand (0-1 scale). Ligand tokens excluded.
    _assign_residue_plddt(residues, [0.85, 0.78, 0.60, 0.10, 0.20])
    assert [round(r.plddt, 1) for r in residues] == [85.0, 78.0, 60.0]
    assert all(r.plddt > 0 for r in residues)  # the 0.0 production default is gone


def test_assign_residue_plddt_passthrough_0_100_scale():
    residues = [Residue(index=i + 1, aa="A") for i in range(2)]
    _assign_residue_plddt(residues, [85.0, 78.0, 60.0])
    assert [r.plddt for r in residues] == [85.0, 78.0]


def test_assign_residue_plddt_noop_on_missing_array():
    residues = [Residue(index=1, aa="A")]
    _assign_residue_plddt(residues, [])
    assert residues[0].plddt == 0.0


def test_real_fixture_plddt_reaches_protein_residues_only():
    plddt = _load_plddt(FX / "boltz_real", 0)  # 445 = 401 protein + 44 ligand
    residues = [Residue(index=i + 1, aa="A") for i in range(401)]
    _assign_residue_plddt(residues, plddt)
    assert len(plddt) == 445 and len(residues) == 401  # ligand tokens dropped
    assert all(r.plddt > 0 for r in residues)          # no silent-zero residue


# --------------------------------------------------------------------------- #
# G1-2 + G1-3: full real-assembly path wires pLDDT AND distinct per-sample poses
# --------------------------------------------------------------------------- #
def _write_model_pdb(path: Path, lig_xyz):
    """3-residue CA backbone + a 3-atom O/P/C ligand at the given coords.

    CA lines copied verbatim from the committed boltz/model_0.pdb fixture;
    HETATM lines are padded to put coords at cols 31-54 and the element at
    cols 77-78 (PDB-strict, what _parse_pdb_atoms reads)."""
    lines = [
        "ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00 85.00           C",
        "ATOM      4  CA  GLY A   2       4.000   5.000   6.000  1.00 78.00           C",
        "ATOM      5  CA  SER A   3       7.000   8.000   9.000  1.00 60.00           C",
    ]
    for serial, (el, (x, y, z)) in zip((6, 7, 8), zip("OPC", lig_xyz)):
        rec = (f"HETATM{serial:>5d}  {el}1  LIG L   1    "
               f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
        lines.append(f"{rec:<76}{el:>2s}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def test_real_assembly_wires_plddt_and_gives_distinct_poses(tmp_path):
    outdir = tmp_path / "out"
    preds = outdir / "boltz_results_x" / "predictions" / "pred"
    preds.mkdir(parents=True)

    # two diffusion samples with DIFFERENT ligand coordinates (same backbone)
    pose0 = [(7.20, 8.10, 9.05), (7.60, 8.40, 9.30), (8.00, 8.70, 9.60)]
    pose1 = [(6.20, 7.10, 8.05), (6.60, 7.40, 8.30), (7.00, 7.70, 8.60)]
    m0 = preds / "pred_model_0.pdb"
    m1 = preds / "pred_model_1.pdb"
    _write_model_pdb(m0, pose0)
    _write_model_pdb(m1, pose1)
    # model_0 is the better sample (higher confidence)
    (preds / "confidence_pred_model_0.json").write_text(
        json.dumps({"confidence_score": 0.90, "complex_plddt": 0.86, "iptm": 0.7})
    )
    (preds / "confidence_pred_model_1.json").write_text(
        json.dumps({"confidence_score": 0.55, "complex_plddt": 0.50, "iptm": 0.4})
    )
    # per-residue pLDDT (3 protein tokens, 0-1 scale)
    np.savez(preds / "plddt_pred_model_0.npz", plddt=np.array([0.85, 0.78, 0.60]))
    np.savez(preds / "plddt_pred_model_1.npz", plddt=np.array([0.40, 0.42, 0.35]))

    ligand = Ligand(
        id="L", smiles="OPC",
        atoms=[LigandAtom(id="O0", element="O", coord=(0.0, 0.0, 0.0)),
               LigandAtom(id="P1", element="P", coord=(0.5, 0.0, 0.0)),
               LigandAtom(id="C2", element="C", coord=(1.0, 0.0, 0.0))],
    )
    found = sorted(preds.rglob("*.pdb"))
    cx = _assemble_real_complex(outdir, found, "AGS", ligand, "boltz2")

    # G1-2: structure residues carry real per-residue pLDDT, not the 0.0 default
    assert cx.structure.residues, "no residues parsed"
    assert any(r.plddt > 0 for r in cx.structure.residues)
    assert all(r.plddt == 0.0 for r in cx.structure.residues) is False

    # G1-3: the two samples have DISTINCT ligand poses (not one shared pose)
    assert len(cx.samples) == 2
    coords = [tuple(a.coord for a in s.ligand_atoms) for s in cx.samples]
    assert coords[0] != coords[1], "samples share one ligand pose (G1-3 bug)"
