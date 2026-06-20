"""s05 docking report: GNINA score parsing, ligand normalization, symmetry-
corrected heavy-atom RMSD, and self-contained HTML render."""

from __future__ import annotations

import sqlite3

import pytest

from evoliez.io.docking_report import (
    _no_align_rmsd,
    _pose_mol,
    _sdf_props,
    _template,
    build_docking_report_html,
    compute_docking_stats,
)

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

GNINA_PROPS = ("\n>  <minimizedAffinity>\n-13.20717\n\n>  <CNNscore>\n0.97326\n"
               "\n>  <CNNaffinity>\n6.50568\n\n$$$$\n")


def _embed(smiles, seed=1, dx=0.0):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    m = Chem.RemoveHs(m)
    if dx:
        c = m.GetConformer()
        for i in range(m.GetNumAtoms()):
            p = c.GetAtomPosition(i)
            c.SetAtomPosition(i, Point3D(p.x + dx, p.y, p.z))
    return m


# --- focused unit tests (the new logic) ----------------------------------- #
def test_sdf_props_parses_gnina_fields():
    p = _sdf_props("mol\n\n\n  0  0\nM  END\n" + GNINA_PROPS)
    assert p["minimizedAffinity"] == -13.20717
    assert p["CNNaffinity"] == 6.50568 and p["CNNscore"] == 0.97326


def test_no_align_rmsd_is_unaligned():
    a = _embed("CCO", seed=3)
    b = _embed("CCO", seed=3, dx=1.0)        # rigid 1 A translation
    # WITHOUT re-alignment a pure 1 A shift must read as 1.0 A (not ~0)
    assert abs(_no_align_rmsd(a, b) - 1.0) < 0.02


def test_pose_mol_drops_smaller_fragment():
    combo = Chem.AddHs(Chem.CombineMols(Chem.MolFromSmiles("CCCO"),
                                        Chem.MolFromSmiles("C(=O)O")))
    AllChem.EmbedMolecule(combo, randomSeed=4)
    sdf = Chem.MolToMolBlock(Chem.RemoveHs(combo))
    mol, qc = _pose_mol(sdf, "sdf", _template("CCCO"))
    assert qc["frags"] == 2 and qc["heavy"] == 4   # propanol kept, formate dropped
    assert mol is not None


# --- full compute + render -------------------------------------------------- #
def _build_run(tmp, smiles="CCO"):
    dock = tmp / "docking"
    (dock / "gnina").mkdir(parents=True)
    (dock / "diffdock" / "wt_dd_out" / "wt").mkdir(parents=True)
    (tmp / "reports").mkdir()

    (dock / "gnina" / "wt_rec.pdb").write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    (dock / "gnina" / "wt_gnina_out.sdf").write_text(
        Chem.MolToMolBlock(_embed(smiles, 1)).rstrip("$\n") + GNINA_PROPS)
    # reference as PDB (HETATM), shifted ~0.1 A from gnina
    Chem.MolToPDBFile(_embed(smiles, 1, dx=0.1), str(dock / "gnina" / "wt_ref_lig.pdb"),
                      flavor=4)
    for n, c in [(1, -0.53), (2, -0.55), (3, -0.70), (10, -3.40)]:
        Chem.MolToMolFile(_embed(smiles, 1, dx=0.4),
                          str(dock / "diffdock" / "wt_dd_out" / "wt"
                              / f"rank{n}_confidence{c}.sdf"))

    db = tmp / "evoliez.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table docking_pose (candidate_id text, method text, "
                "score real, ligand_rmsd_to_reference real)")
    con.executemany("insert into docking_pose values (?,?,?,?)",
                    [("wt", "gnina", -13.21, 0.0), ("wt", "diffdock", -0.53, 0.0)])
    con.commit()
    con.close()
    return db


def test_compute_stats_full(tmp_path):
    db = _build_run(tmp_path)
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    assert s["scores"]["gnina"]["score"] == -13.21
    # GNINA real sub-scores split out
    assert s["gnina_props"]["minimizedAffinity"] == -13.20717
    assert s["gnina_props"]["CNNaffinity"] == 6.50568
    assert (1, -0.53) in s["landscape"] and (10, -3.4) in s["landscape"]
    # RMSD pairs computed (symmetry-corrected, unaligned)
    pairs = {(a, b): r for a, b, r, _ in s["rmsd_pairs"]}
    assert ("gnina", "diffdock") in pairs
    gd = pairs[("gnina", "diffdock")]
    assert gd is not None and abs(gd - 0.4) < 0.05      # the 0.4 A shift, unaligned
    # normalization QC present
    assert s["ligand_qc"]["gnina"]["heavy"] == 3 and s["rdkit_ok"]


def test_build_html_no_leftover_tokens(tmp_path):
    db = _build_run(tmp_path)
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    # GENERIC target/ligand -> the template itself must not hardcode the fdh case
    html = build_docking_report_html(
        target_id="myprot", stats=s, ligand_name="myligand",
        generated="2026-06-20 09:00", conditions=[("methods", "gnina, diffdock")])
    assert "%%" not in html
    assert html.lstrip().startswith("<!DOCTYPE html>")
    for mid in ("m_rec", "m_gnina", "m_diffdock", "m_reference"):
        assert f'id="{mid}"' in html
    # the honesty fixes are present
    assert "minimizedAffinity" in html and "CNNaffinity" in html
    assert "heavy-atom RMSD" in html and "coarse proxy" in html
    assert "not experimental validation" in html or "not an experimental" in html
    assert "QC checklist" in html
    # generality: no FDH/NADP/formate hardcoded in the template text
    for term in ("NADP", "formate", "FDH", "fdh", "phosphate-2"):
        assert term not in html, f"hardcoded {term!r} leaked into the report"
