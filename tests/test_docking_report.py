"""s05 docking report: GNINA score parsing, ligand normalization, symmetry-
corrected heavy-atom RMSD, and self-contained HTML render."""

from __future__ import annotations

import sqlite3

import pytest

from evoliez.io.docking_report import (
    _COFCOL,
    _candidate_label,
    _docked_ligand_id,
    _no_align_rmsd,
    _pdb_het_block,
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


# The co-modelled cofactor as fixed receptor context: HETATM records in the
# docking receptor PDB (the protein is ATOM). Mirrors production, where the gnina
# receptor for a co-modelled run carries the OTHER ligand's atoms held fixed.
_COF_HETATM = (
    "HETATM 3033  O2  LIG C   1       0.216  -1.314  -1.187  1.00 78.13           O\n"
    "HETATM 3034  C4  LIG C   1      -0.131  -0.787  -0.296  1.00 85.19           C\n"
    "HETATM 3035  O3  LIG C   1      -0.505  -0.223   0.600  1.00 76.57           O\n")


# --- full compute + render -------------------------------------------------- #
def _write_candidate_artifacts(dock, cand, smiles, dx, cofactor=False):
    """Docked pose / receptor / reference-ligand artifacts for ONE candidate id
    (a base like ``'wt'`` or a context variant like ``'wt__formate'``). ``dx``
    rigidly translates this candidate's poses so distinct candidates land at
    distinct, comparable RMSDs. When ``cofactor`` is set, the receptor PDB also
    carries a co-modelled cofactor as fixed-context HETATM (as in a real
    co-modelled run), so the overlay's cofactor block can be exercised."""
    (dock / "gnina").mkdir(parents=True, exist_ok=True)
    ddc = dock / "diffdock" / f"{cand}_dd_out" / cand
    ddc.mkdir(parents=True, exist_ok=True)
    rec = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    if cofactor:
        rec += _COF_HETATM
    (dock / "gnina" / f"{cand}_rec.pdb").write_text(rec)
    (dock / "gnina" / f"{cand}_gnina_out.sdf").write_text(
        Chem.MolToMolBlock(_embed(smiles, 1, dx=dx)).rstrip("$\n") + GNINA_PROPS)
    # reference as PDB (HETATM), shifted ~0.1 A from this candidate's gnina pose
    Chem.MolToPDBFile(_embed(smiles, 1, dx=dx + 0.1),
                      str(dock / "gnina" / f"{cand}_ref_lig.pdb"), flavor=4)
    for n, c in [(1, -0.53), (2, -0.55), (3, -0.70), (10, -3.40)]:
        Chem.MolToMolFile(_embed(smiles, 1, dx=dx + 0.4),
                          str(ddc / f"rank{n}_confidence{c}.sdf"))


def _build_run(tmp, smiles="CCO", candidates=(("wt", 0.0),), cofactor=False):
    """Synthetic finished-s05 run dir. ``candidates`` is a list of
    ``(candidate_id, dx)``: the default single ``'wt'`` reproduces the original
    base-only fixture; pass extra ``'<base>__<cofactor>'`` entries to exercise the
    multi-ligand context-variant path. ``cofactor`` writes the co-modelled
    cofactor as fixed-context HETATM into the design-ligand (non-variant)
    receptor(s) — the overlay the user actually looks at."""
    dock = tmp / "docking"
    (tmp / "reports").mkdir(exist_ok=True)
    rows = []
    for cand, dx in candidates:
        # the design ligand (bare candidate) is docked WITH the co-modelled
        # ligand(s) as fixed context; a '<target>__<lig>' candidate docks that
        # OTHER ligand itself, so its own receptor would instead carry the design
        # ligand — keep the fixture's fixed-context HETATM on the design-ligand
        # candidate, which is what the primary overlay shows.
        is_comodelled = "__" in cand
        _write_candidate_artifacts(dock, cand, smiles, dx,
                                   cofactor=cofactor and not is_comodelled)
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
def test_split_candidate_parses_docked_ligand_suffix():
    # GENERIC: the '__' suffix is the DOCKED ligand's id, parsed at runtime, never
    # a hardcoded ligand. s05 docks every input ligand; the candidate id says which.
    assert _split_candidate("wt") == ("wt", None)
    assert _split_candidate("wt__formate") == ("wt", "formate")
    # a ligand id may itself contain '__'/'_': split on the FIRST '__'
    assert _split_candidate("d1__metal_ion") == ("d1", "metal_ion")
    assert _split_candidate("d1__a__b") == ("d1", "a__b")
    # the docked-ligand id: bare candidate -> the candidate id (design ligand);
    # a '<target>__<lig>' candidate -> the named co-modelled ligand it docks.
    assert _docked_ligand_id("wt") == "wt"
    assert _docked_ligand_id("wt__formate") == "formate"
    # the candidate label names the docked ligand (design vs co-modelled), NOT a
    # "receptor context" of the design ligand (the candidates are different mols).
    assert _candidate_label("wt") == "wt (design ligand)"
    assert _candidate_label("wt__formate") == "formate"


def test_multi_ligand_per_ligand_rows_are_discovered_and_labeled(tmp_path):
    # design-ligand candidate 'wt' + a co-modelled-ligand candidate 'wt__formate'
    # (each docks a DIFFERENT molecule against its OWN reference)
    db = _build_run(tmp_path, candidates=(("wt", 0.0), ("wt__formate", 0.25)))
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    # both candidates discovered from docking_pose (candidate_id NOT dropped)
    cand_ids = {c["candidate"] for c in s["candidate_summaries"]}
    assert cand_ids == {"wt", "wt__formate"}
    # design vs co-modelled distinguished; the docked-ligand id parsed at runtime
    by_id = {c["candidate"]: c for c in s["candidate_summaries"]}
    assert by_id["wt"]["is_variant"] is False
    assert by_id["wt__formate"]["is_variant"] is True
    assert by_id["wt__formate"]["cofactor"] == "formate"
    # the multi-ligand block is now ONE row per (docked ligand, method) — NOT a
    # base-vs-variant effect. Each candidate is a distinct docked molecule.
    ml = s["multi_ligand"]
    assert ml is not None
    assert ml["ligands"] == ["wt", "formate"]          # the design + co-modelled
    # rows: (candidate_id, ligand_id, is_design, method, score_info, rmsd)
    by_key = {(r[0], r[3]): r for r in ml["rows"]}
    assert ("wt", "gnina") in by_key and ("wt", "diffdock") in by_key
    assert ("wt__formate", "gnina") in by_key
    assert ("wt__formate", "diffdock") in by_key
    # the design-ligand row is flagged as design; its ligand id is the candidate id
    cand, lig, is_design, _m, score_info, rmsd = by_key[("wt", "gnina")]
    assert lig == "wt" and is_design is True
    # the co-modelled row docks 'formate' and is NOT flagged design
    cand2, lig2, is_design2, _m2, sinfo2, rmsd2 = by_key[("wt__formate", "gnina")]
    assert lig2 == "formate" and is_design2 is False
    # each row carries that ligand's OWN labelled score + its OWN pose RMSD
    assert score_info["type"] == "minimizedAffinity"
    assert rmsd is not None and rmsd2 is not None
    # NO base-vs-context effect: rows are per-ligand, not (base, variant) pairs
    assert all(len(r) == 6 for r in ml["rows"])
    assert "cofactors" not in ml and "variants" not in ml


def test_multi_ligand_render_is_per_ligand_not_effect(tmp_path):
    db = _build_run(tmp_path, candidates=(("wt", 0.0), ("wt__formate", 0.25)))
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    html = build_docking_report_html(
        target_id="myprot", stats=s, ligand_name="myligand",
        generated="2026-06-21 09:00", conditions=[("methods", "gnina, diffdock")])
    assert "%%" not in html
    # the section is the PER-LIGAND reference-docking table
    assert "Per-ligand reference docking" in html
    # each docked ligand is named (design ligand id + the co-modelled ligand id)
    assert "formate" in html                       # the parsed co-modelled ligand
    assert "design ligand" in html and "co-modelled ligand" in html
    assert "wt__formate" in html                   # the candidate id is shown
    # the fixed-receptor-context note is present (every ligand docked with others)
    assert "fixed receptor context" in html
    assert "pose accuracy" in html
    # the WRONG base-vs-context "effect" framing is GONE
    assert "Multi-ligand co-modelling" not in html
    assert "design ligand alone" not in html
    assert "as receptor context" not in html
    assert "Effect of cofactor context" not in html
    assert "Δ RMSD" not in html and "pose shift" not in html


def test_multi_ligand_table_uses_recomputed_rmsd_not_db_placeholder(tmp_path):
    # FIX 1 (paper-grade QC): the multi-ligand variant table must show the SAME
    # recomputed symmetry-corrected RMSD-to-reference as the "Pose accuracy"
    # section — NOT the DB ligand_rmsd_to_reference column (which gnina leaves at
    # the 0.0 placeholder, contradicting the recomputed value). We seed the DB
    # rmsd column with a deliberately WRONG sentinel and assert it is ignored.
    dock = tmp_path / "docking"
    (tmp_path / "reports").mkdir(exist_ok=True)
    for cand, dx in (("wt", 0.0), ("wt__formate", 0.25)):
        _write_candidate_artifacts(dock, cand, "CCO", dx, cofactor="__" not in cand)
    db = tmp_path / "evoliez.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table docking_pose (candidate_id text, method text, "
                "score real, ligand_rmsd_to_reference real)")
    # DB rmsd column = the 0.0 PLACEHOLDER for gnina (exactly the production bug),
    # and a bogus 9.99 for diffdock — neither must reach the rendered table.
    con.executemany("insert into docking_pose values (?,?,?,?)", [
        ("wt", "gnina", -13.21, 0.0), ("wt", "diffdock", -0.53, 9.99),
        ("wt__formate", "gnina", 0.897, 0.0),
        ("wt__formate", "diffdock", -0.40, 9.99)])
    con.commit()
    con.close()

    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    by_id = {c["candidate"]: c for c in s["candidate_summaries"]}
    # per-candidate RECOMPUTED RMSD-to-ref map is present and is NOT the 0.0/9.99
    # DB placeholders (the gnina pose sits ~0.1 A from its shifted reference).
    rr = by_id["wt"]["recomputed_ref_rmsd"]
    assert "gnina" in rr and rr["gnina"] is not None
    assert rr["gnina"] != 0.0 and abs(rr["gnina"] - 0.1) < 0.1
    # the per-ligand rows carry the recomputed RMSD (the row's pose-accuracy cell),
    # not the DB placeholder. Row = (candidate, ligand, is_design, method, score, rmsd)
    gnina_rows = [r for r in s["multi_ligand"]["rows"] if r[3] == "gnina"]
    assert gnina_rows, "expected per-ligand gnina rows"
    for _cand, _lig, _isd, _m, _score, rmsd in gnina_rows:
        assert rmsd is not None and rmsd != 0.0   # NOT the 0.0 DB placeholder

    html = build_docking_report_html(
        target_id="myprot", stats=s, ligand_name="myligand",
        generated="2026-06-21 09:00", conditions=[("methods", "gnina, diffdock")])
    # the recomputed value the Pose-accuracy table shows for gnina vs reference is
    # the SAME number the multi-ligand candidate table shows (consistency), and
    # the bogus DB sentinels never appear as a gnina RMSD cell.
    gnina_ref = [r for a, b, r, _ in s["rmsd_pairs"]
                 if a == "gnina" and b == "reference" and r is not None]
    assert gnina_ref, "expected a recomputed gnina↔reference RMSD"
    assert f"{gnina_ref[0]:.2f} A" in html
    assert "9.99 A" not in html                 # the bogus diffdock DB rmsd is gone
    assert "%%" not in html


def test_multi_ligand_score_is_labelled_with_type(tmp_path):
    # FIX 2 (paper-grade QC): every score in the multi-ligand table must be LABELLED
    # with its type so a GNINA Vina energy (minimizedAffinity) near 0 for a
    # weak/strained co-modelled fragment can't be conflated with a 0–1 CNNscore.
    db = _build_run(tmp_path, candidates=(("wt", 0.0), ("wt__formate", 0.25)))
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    by_id = {c["candidate"]: c for c in s["candidate_summaries"]}
    # the per-candidate labelled-score provenance: gnina score is the SDF
    # minimizedAffinity (re-read from disk), explicitly typed, lower=better. For
    # the co-modelled ligand this is the GENUINE minimizedAffinity of THAT ligand.
    st = by_id["wt__formate"]["score_types"]["gnina"]
    assert st["type"] == "minimizedAffinity" and st["lower_better"] is True
    assert st["value"] == -13.20717        # the SDF tag, never the CNNscore 0.973
    html = build_docking_report_html(
        target_id="myprot", stats=s, ligand_name="myligand",
        generated="2026-06-21 09:00", conditions=[("methods", "gnina, diffdock")])
    # the score type is named next to the value in the per-ligand table, and the
    # column header advertises it
    assert "minimizedAffinity" in html
    assert "docking score (type)" in html
    assert "CNNscore" in html               # the note warns against conflation
    assert "%%" not in html


def test_single_candidate_run_has_no_multi_ligand_section(tmp_path):
    # only the bare design-ligand candidate -> render exactly as before, with NO
    # per-ligand section and NO "variant" language (there is only one molecule).
    db = _build_run(tmp_path, candidates=(("wt", 0.0),))
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    assert s["multi_ligand"] is None
    html = build_docking_report_html(
        target_id="myprot", stats=s, ligand_name="myligand",
        generated="2026-06-21 09:00", conditions=[("methods", "gnina, diffdock")])
    assert "Per-ligand reference docking" not in html
    assert "Multi-ligand co-modelling" not in html
    assert "co-modelled ligand" not in html
    assert "as receptor context" not in html
    assert "%%" not in html


# --- co-modelled cofactor in the 3D overlay (the "why isn't formate shown?" fix) #
def test_pdb_het_block_extracts_only_hetatm():
    # the co-modelled cofactor = the HETATM in the docking receptor; the protein
    # (ATOM) and TER/END records are NOT pulled into the cofactor model.
    rec = ("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
           "TER\n" + _COF_HETATM + "END\n")
    block, resns = _pdb_het_block(rec)
    assert block.count("HETATM") == 3 and "ATOM " not in block
    assert "TER" not in block and "END" not in block
    assert resns == ["LIG"]
    # no HETATM -> empty (a stripped receptor, e.g. DiffDock's)
    assert _pdb_het_block("ATOM      1  CA  ALA A   1       0.0 0.0 0.0\n") == ("", [])
    assert _pdb_het_block(None) == ("", [])


def test_overlay_includes_comodelled_cofactor_block(tmp_path):
    # design ligand 'wt' co-modelled WITH a cofactor context ('wt__formate' docks
    # the cofactor); the primary 'wt' receptor carries the cofactor as fixed HETATM
    db = _build_run(tmp_path, candidates=(("wt", 0.0), ("wt__formate", 0.25)),
                    cofactor=True)
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    # the cofactor's fixed-context coords are surfaced from the receptor PDB
    assert s["cofactor_pdb"].count("HETATM") == 3
    # generic: the cofactor id is parsed at runtime from the '__<cof>' variant
    assert s["cofactor_label"] == "formate (cofactor — fixed context)"

    html = build_docking_report_html(
        target_id="myprot", stats=s, ligand_name="myligand",
        generated="2026-06-21 09:00", conditions=[("methods", "gnina, diffdock")])
    assert "%%" not in html
    # a DISTINCT cofactor model + toggle is emitted in the overlay
    assert 'id="m_cofactor"' in html
    assert "toggleCof" in html and "styleCof" in html
    # its label is shown and names the cofactor as fixed context
    assert "formate (cofactor — fixed context)" in html
    # the cofactor colour is DISTINCT from every design-ligand pose / ref colour
    assert _COFCOL in html
    assert _COFCOL not in ("#1D9E75", "#BA7517", "#3b7dd8")
    # the cofactor HETATM coords actually reached the embedded model block
    assert "HETATM 3033  O2  LIG" in html


def test_overlay_unchanged_when_no_comodelled_cofactor(tmp_path):
    # no co-modelled cofactor -> the overlay is exactly as before (no cofactor
    # model / toggle / label leaks in)
    db = _build_run(tmp_path, candidates=(("wt", 0.0),))  # cofactor=False
    s = compute_docking_stats(str(tmp_path), str(db), ["gnina", "diffdock"],
                              ligand_smiles="CCO")
    assert s["cofactor_pdb"] == "" and s["cofactor_label"] == ""
    html = build_docking_report_html(
        target_id="myprot", stats=s, ligand_name="myligand",
        generated="2026-06-21 09:00", conditions=[("methods", "gnina, diffdock")])
    assert 'id="m_cofactor"' not in html
    assert "cofactor — fixed context" not in html
    # the cofactor control group is absent (the design-ligand overlay is intact)
    assert '<span class="cgl">cofactor</span>' not in html
    assert "%%" not in html


# --- overlay embeds the GEOMETRY-SELECTED gnina mode, not the SDF's model 0 --- #
def _aspirin(seed=7, dx=0.0):
    smi = "CC(=O)Oc1ccccc1C(=O)O"          # asymmetric -> a flip really differs
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    m = Chem.RemoveHs(m)
    if dx:
        c = m.GetConformer()
        for i in range(m.GetNumAtoms()):
            p = c.GetAtomPosition(i)
            c.SetAtomPosition(i, Point3D(p.x + dx, p.y, p.z))
    return m


def _gnina_rec(mol, aff):
    return (Chem.MolToMolBlock(mol).rstrip("$\n")
            + f"\n>  <minimizedAffinity>\n{aff}\n\n$$$$\n")


def test_overlay_embeds_geometry_selected_gnina_mode(tmp_path):
    # gnina SDF: mode 0 (CNN-rank-1) is FLIPPED 6 A off the reference with the BEST
    # affinity; mode 2 sits ON the reference with the WORST affinity. The overlay +
    # score + RMSD must reflect mode 2 (geometry), not the SDF's first model.
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    dock = tmp_path / "docking"
    (dock / "gnina").mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    (dock / "gnina" / "wt_rec.pdb").write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    # multi-mode gnina output: rank-1 flipped (best aff), rank-3 on-reference (worst)
    (dock / "gnina" / "wt_gnina_out.sdf").write_text(
        _gnina_rec(_aspirin(7, 6.0), -12.182)
        + _gnina_rec(_aspirin(7, 2.0), -9.0)
        + _gnina_rec(_aspirin(7, 0.05), -7.5))
    # reference ~ on mode 2 (a HETATM PDB, like the real ref_lig.pdb)
    Chem.MolToPDBFile(_aspirin(7, 0.0), str(dock / "gnina" / "wt_ref_lig.pdb"),
                      flavor=4)

    db = tmp_path / "evoliez.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table docking_pose (candidate_id text, method text, "
                "score real, ligand_rmsd_to_reference real)")
    con.execute("insert into docking_pose values ('wt','gnina',-7.5,0.0)")
    con.commit()
    con.close()

    s = compute_docking_stats(str(tmp_path), str(db), ["gnina"], ligand_smiles=smi)
    # the gnina pose surfaced to the overlay is a SINGLE mode (one $$$$ record)
    assert s["poses"]["gnina"].count("$$$$") == 1
    # its score tag is the SELECTED (on-reference) mode's minimizedAffinity, NOT
    # the flipped rank-1's -12.182
    assert s["gnina_props"]["minimizedAffinity"] == -7.5
    # the recomputed RMSD-to-reference for gnina is SMALL (the on-reference mode),
    # not the ~6 A of the flipped rank-1 it would have shown before
    gref = [r for a, b, r, _ in s["rmsd_pairs"]
            if a == "gnina" and b == "reference" and r is not None]
    assert gref and gref[0] < 1.0

    html = build_docking_report_html(
        target_id="myprot", stats=s, ligand_name="aspirin",
        generated="2026-06-21 09:00", conditions=[("methods", "gnina")])
    assert "%%" not in html
    # exactly one gnina model is embedded, and it is the selected single mode
    assert 'id="m_gnina"' in html
    # the SELECTED mode's minimizedAffinity (-7.5) is the value shown, never the
    # flipped rank-1's -12.182
    assert "-7.5" in html and "-12.182" not in html
