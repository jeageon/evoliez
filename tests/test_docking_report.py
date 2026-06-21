"""s05 docking report: GNINA score parsing, ligand normalization, symmetry-
corrected heavy-atom RMSD, and self-contained HTML render."""

from __future__ import annotations

import sqlite3

import pytest

from evoliez.io.docking_report import (
    _candidate_label,
    _no_align_rmsd,
    _pose_mol,
    _sdf_props,
    _split_candidate,
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
def _write_candidate_artifacts(dock, cand, smiles, dx):
    """Docked pose / receptor / reference-ligand artifacts for ONE candidate id
    (a base like ``'wt'`` or a context variant like ``'wt__formate'``). ``dx``
    rigidly translates this candidate's poses so distinct candidates land at
    distinct, comparable RMSDs."""
    (dock / "gnina").mkdir(parents=True, exist_ok=True)
    ddc = dock / "diffdock" / f"{cand}_dd_out" / cand
    ddc.mkdir(parents=True, exist_ok=True)
    (dock / "gnina" / f"{cand}_rec.pdb").write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    (dock / "gnina" / f"{cand}_gnina_out.sdf").write_text(
        Chem.MolToMolBlock(_embed(smiles, 1, dx=dx)).rstrip("$\n") + GNINA_PROPS)
    # reference as PDB (HETATM), shifted ~0.1 A from this candidate's gnina pose
    Chem.MolToPDBFile(_embed(smiles, 1, dx=dx + 0.1),
                      str(dock / "gnina" / f"{cand}_ref_lig.pdb"), flavor=4)
    for n, c in [(1, -0.53), (2, -0.55), (3, -0.70), (10, -3.40)]:
        Chem.MolToMolFile(_embed(smiles, 1, dx=dx + 0.4),
                          str(ddc / f"rank{n}_confidence{c}.sdf"))


def _build_run(tmp, smiles="CCO", candidates=(("wt", 0.0),)):
    """Synthetic finished-s05 run dir. ``candidates`` is a list of
    ``(candidate_id, dx)``: the default single ``'wt'`` reproduces the original
    base-only fixture; pass extra ``'<base>__<cofactor>'`` entries to exercise the
    multi-ligand context-variant path."""
    dock = tmp / "docking"
    (tmp / "reports").mkdir(exist_ok=True)
    rows = []
    for cand, dx in candidates:
        _write_candidate_artifacts(dock, cand, smiles, dx)
        rows += [(cand, "gnina", -13.21, round(dx, 2)),
                 (cand, "diffdock", -0.53, round(0.4 + dx, 2))]

    db = tmp / "evoliez.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table docking_pose (candidate_id text, method text, "
                "score real, ligand_rmsd_to_reference real)")
    con.executemany("insert into docking_pose values (?,?,?,?)", rows)
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


# --- multi-ligand co-modelling (the visibility fix) ------------------------- #
def test_split_candidate_parses_context_suffix():
    # GENERIC: the '__' suffix is parsed at runtime, never a hardcoded cofactor.
    assert _split_candidate("wt") == ("wt", None)
    assert _split_candidate("wt__formate") == ("wt", "formate")
    # a cofactor id may itself contain '__'/'_': split on the FIRST '__'
    assert _split_candidate("d1__metal_ion") == ("d1", "metal_ion")
    assert _split_candidate("d1__a__b") == ("d1", "a__b")
    assert (_candidate_label("wt__formate")
            == "wt + formate as receptor context")


def test_multi_ligand_variants_are_discovered_and_labeled(tmp_path):
    # base 'wt' (design ligand alone) + context variant 'wt__formate'
    db = _build_run(tmp_path, candidates=(("wt", 0.0), ("wt__formate", 0.25)))
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    # both candidates discovered from docking_pose (candidate_id NOT dropped)
    cand_ids = {c["candidate"] for c in s["candidate_summaries"]}
    assert cand_ids == {"wt", "wt__formate"}
    # base vs variant are distinguished, cofactor parsed at runtime
    by_id = {c["candidate"]: c for c in s["candidate_summaries"]}
    assert by_id["wt"]["is_variant"] is False and by_id["wt"]["cofactor"] is None
    assert by_id["wt__formate"]["is_variant"] is True
    assert by_id["wt__formate"]["cofactor"] == "formate"
    # the multi-ligand effect block pairs base<->variant per method
    ml = s["multi_ligand"]
    assert ml is not None
    assert ml["cofactors"] == ["formate"]
    methods_paired = {(r[0], r[3]) for r in ml["rows"]}  # (base, method)
    assert ("wt", "gnina") in methods_paired
    assert ("wt", "diffdock") in methods_paired
    # base vs variant RMSD-to-reference surfaced side by side (the EFFECT)
    g_row = next(r for r in ml["rows"] if r[3] == "gnina")
    _base, cof, var_id, _m, b_ref, v_ref, shift = g_row
    assert cof == "formate" and var_id == "wt__formate"
    assert b_ref is not None and v_ref is not None  # both RMSD-to-ref present


def test_multi_ligand_render_makes_setup_visible(tmp_path):
    db = _build_run(tmp_path, candidates=(("wt", 0.0), ("wt__formate", 0.25)))
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    html = build_docking_report_html(
        target_id="myprot", stats=s, ligand_name="myligand",
        generated="2026-06-21 09:00", conditions=[("methods", "gnina, diffdock")])
    assert "%%" not in html
    # the multi-ligand setup is now VISIBLE and the cofactor is NAMED
    assert "Multi-ligand co-modelling" in html
    assert "as receptor context" in html
    assert "formate" in html                       # the parsed cofactor surfaces
    # both candidates appear as distinct rows
    assert "wt__formate" in html or "wt + formate" in html
    # the effect (alone vs with-cofactor RMSD-to-ref) is surfaced
    assert "with cofactor" in html and "alone" in html


def test_single_candidate_run_has_no_multi_ligand_section(tmp_path):
    # only a base candidate -> render exactly as before, NO multi-ligand section
    db = _build_run(tmp_path, candidates=(("wt", 0.0),))
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    assert s["multi_ligand"] is None
    html = build_docking_report_html(
        target_id="myprot", stats=s, ligand_name="myligand",
        generated="2026-06-21 09:00", conditions=[("methods", "gnina, diffdock")])
    assert "Multi-ligand co-modelling" not in html
    assert "as receptor context" not in html
    assert "%%" not in html
