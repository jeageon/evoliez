"""s07 mutation-library report.

Builds the self-contained HTML report from a minimal SYNTHETIC generated-candidate
pool (a few candidates across all four generators + the three budget tiers) + a
tiny WT-complex PDB + the design-context inputs (designable / catalytic / fixed
positions, the contact list, per-position MSA features) — pure-python data layer,
no LigandMPNN / torch — and checks the things a paper-quality s07 report must
guarantee: no leftover template tokens, the summary cards + generator/tier
break-downs render, the per-position design-space table/heatmap (with the
PROTECTED core visually distinct) appears, the per-mutation provenance table
(generator + distances + MSA conservation + risk flag + forbidden-check + the
dynamic extra-ligand distance column) appears, the budget tiers render, the
3Dmol viewer block + the design/protected residue highlight are wired, and the
template carries NO hardcoded protein/ligand identity (generic for any target).
"""

from __future__ import annotations

from types import SimpleNamespace

from evoliez.io.s07_report import (
    build_mutation_report_html,
    compute_mutation_stats,
    find_wt_complex_pdb,
    load_generated_provenance,
    write_mutation_report,
)


# --------------------------------------------------------------------------- #
# synthetic, generic fixtures (an invented protein "xyz1" + ligand "L0")
# --------------------------------------------------------------------------- #
# Generated-candidate provenance rows: the SAME shape s07 writes to
# reports/provenance/generated_candidates.json — one per candidate, with the
# generator, n_mutations, the rationale features, the structural distances + the
# safety flags + the dynamic extra-ligand distance column (distance_to_cof).
def _gen_rows():
    return [
        # chemistry_rules single, close to the ligand, clean
        {"candidate_id": "mut_00000", "mutation_string": "A40K", "generator":
         "chemistry_rules", "n_mutations": 1, "ligand_roles": "anion;acceptor",
         "n_contacts": 3, "distance_to_design_ligand": 3.4,
         "distance_to_cof": 9.1, "distance_to_nearest_catalytic": 6.8,
         "conservation": 0.21, "msa_permissiveness": 0.71, "msa_freq": "",
         "ligandmpnn_logp": "", "forbidden_check": "ok", "risk_flag": False,
         "disallowed_reason": "", "buried_polar_note": "deferred_to_s09_s11"},
        # msa_sampler single, family-permissive
        {"candidate_id": "mut_00001", "mutation_string": "S55T", "generator":
         "msa_sampler", "n_mutations": 1, "msa_freq": 0.42, "gap_freq": 0.05,
         "conservation": 0.30, "msa_permissiveness": 0.62,
         "distance_to_design_ligand": 5.1, "distance_to_cof": 4.2,
         "distance_to_nearest_catalytic": 7.9, "ligandmpnn_logp": "",
         "forbidden_check": "ok", "risk_flag": False, "disallowed_reason": "",
         "buried_polar_note": "deferred_to_s09_s11"},
        # ligandmpnn single, carries a log-probability
        {"candidate_id": "mut_00002", "mutation_string": "V70L", "generator":
         "ligandmpnn", "n_mutations": 1, "ligandmpnn_logp": -1.23,
         "dropped_invalid": 0, "conservation": 0.18, "msa_permissiveness": 0.80,
         "distance_to_design_ligand": 4.0, "distance_to_cof": 8.0,
         "distance_to_nearest_catalytic": 5.5, "msa_freq": "",
         "forbidden_check": "ok", "risk_flag": False, "disallowed_reason": "",
         "buried_polar_note": "deferred_to_s09_s11"},
        # multipoint (2 substitutions) -> a per-position distance list column
        {"candidate_id": "mut_00003", "mutation_string": "A40K;V70L",
         "generator": "multipoint", "n_mutations": 2, "multipoint_order": 2,
         "distance_to_design_ligand": 3.4,
         "distance_to_design_ligand_per_pos": "3.4;4.0",
         "distance_to_cof": 8.0, "distance_to_nearest_catalytic": 5.5,
         "conservation": 0.20, "msa_permissiveness": 0.75, "msa_freq": "",
         "ligandmpnn_logp": "", "forbidden_check": "ok", "risk_flag": False,
         "disallowed_reason": "", "buried_polar_note": "deferred_to_s09_s11"},
        # RISKY: introduces a proline near a catalytic residue -> flagged
        {"candidate_id": "mut_00004", "mutation_string": "L71P", "generator":
         "chemistry_rules", "n_mutations": 1, "n_contacts": 2,
         "distance_to_design_ligand": 4.6, "distance_to_cof": 7.0,
         "distance_to_nearest_catalytic": 3.1, "conservation": 0.40,
         "msa_permissiveness": 0.55, "msa_freq": "", "ligandmpnn_logp": "",
         "forbidden_check": "ok", "risk_flag": True,
         "disallowed_reason": "introduces_P@71;radical_2shell_L71P(seq)",
         "buried_polar_note": "deferred_to_s09_s11"},
    ]


# Budget-tier rows (subset views: single / multipoint / risky). Same shape as
# generated_candidates_{single,multipoint,risky}.json.
def _tier_rows():
    rows = {r["candidate_id"]: r for r in _gen_rows()}
    return {
        "single": [rows["mut_00000"], rows["mut_00001"], rows["mut_00002"]],
        "multipoint": [rows["mut_00003"]],
        "risky": [rows["mut_00004"]],
    }


# Design context: designable positions (the library edits), the PROTECTED core
# (catalytic + fixed), the residue↔ligand contacts, and per-position MSA features.
_DESIGNABLE = [40, 55, 70, 71]
_CATALYTIC = [120]
_FIXED = [15, 33, 120]          # includes the catalytic overlap (subtracted)


def _contacts():
    # objects exposing residue_index / distance / contact_probability (the
    # in-memory ResidueLigandContact shape); the report reads by attr-or-key.
    def c(ri, d, p):
        return SimpleNamespace(residue_index=ri, distance=d,
                               contact_probability=p, ligand_atom_id="LX1")
    return [c(40, 3.4, 0.95), c(40, 3.9, 0.80), c(55, 5.1, 0.40),
            c(70, 4.0, 0.66), c(71, 4.6, 0.30), c(120, 3.0, 0.99)]


def _features():
    def f(pos, cons, gap, aa):
        return SimpleNamespace(target_position=pos, conservation_score=cons,
                               gap_frequency=gap, wt=aa)
    return [f(40, 0.21, 0.02, "A"), f(55, 0.30, 0.05, "S"),
            f(70, 0.18, 0.01, "V"), f(71, 0.40, 0.03, "L"),
            f(120, 0.97, 0.0, "H"), f(15, 0.90, 0.0, "C"),
            f(33, 0.88, 0.0, "G")]


# a tiny PDB: a few protein residues (incl. the designable + catalytic ones) plus
# a 2-atom ligand (ATOM = protein, HETATM = ligand).
_PDB = (
    "ATOM      1  CA  ALA A  40      10.000  10.000  10.000  1.00 88.00           C\n"
    "ATOM      2  CA  SER A  55      13.000  12.000  10.000  1.00 90.00           C\n"
    "ATOM      3  CA  VAL A  70      11.000  13.000  11.000  1.00 86.00           C\n"
    "ATOM      4  CA  LEU A  71      11.500  13.500  11.500  1.00 85.00           C\n"
    "ATOM      5  CA  HIS A 120      12.000  11.000   9.000  1.00 92.00           C\n"
    "HETATM    6  C1  LIG B   1      12.000  11.000  10.500  1.00  0.00           C\n"
    "HETATM    7  O1  LIG B   1      12.500  11.500  11.000  1.00  0.00           O\n"
    "END\n"
)


def _build(**over):
    stats = compute_mutation_stats(
        over.pop("rows", _gen_rows()),
        tier_rows=over.pop("tier_rows", _tier_rows()),
        designable_positions=over.pop("designable", _DESIGNABLE),
        catalytic_positions=over.pop("catalytic", _CATALYTIC),
        fixed_positions=over.pop("fixed", _FIXED),
        contacts=over.pop("contacts", _contacts()),
        position_features=over.pop("features", _features()),
        wt_pdb_path=over.pop("wt_pdb", _PDB),
        ligand_ids=over.pop("ligand_ids", ["L0", "cof"]),
    )
    return build_mutation_report_html(
        target_id=over.pop("target_id", "xyz1"),
        stats=stats,
        conditions=[("generators", "chemistry_rules, msa_sampler, ligandmpnn"),
                    ("backend", "real")],
        generated="2026-06-21 12:00")


# --------------------------------------------------------------------------- #
# DATA layer
# --------------------------------------------------------------------------- #
def test_stats_compute_shapes():
    s = compute_mutation_stats(
        _gen_rows(), tier_rows=_tier_rows(),
        designable_positions=_DESIGNABLE, catalytic_positions=_CATALYTIC,
        fixed_positions=_FIXED, contacts=_contacts(),
        position_features=_features(), wt_pdb_path=_PDB,
        ligand_ids=["L0", "cof"])
    # library-level counts
    assert s["n_candidates"] == 5
    assert s["n_single"] == 4 and s["n_multi"] == 1
    assert s["n_risky"] == 1
    # generator break-down: all four generators present, counted right
    gen = s["gen"]
    assert gen["generators"] == ["chemistry_rules", "msa_sampler",
                                 "ligandmpnn", "multipoint"]
    # chemistry_rules has 2 (mut_00000 + the risky mut_00004), the rest 1 each
    by = dict(zip(gen["generators"], gen["counts"]))
    assert by == {"chemistry_rules": 2, "msa_sampler": 1, "ligandmpnn": 1,
                  "multipoint": 1}
    assert gen["total"] == 5
    # tier break-down
    tier = s["tier"]
    assert tier["tiers"] == ["single", "multipoint", "risky"]
    assert dict(zip(tier["tiers"], tier["counts"])) == {
        "single": 3, "multipoint": 1, "risky": 1}
    # design space: 4 designable, protected = 1 catalytic + 2 fixed (120 dedup'd)
    sp = s["space"]
    assert sp["n_designable"] == 4
    assert sp["n_catalytic"] == 1 and sp["n_fixed"] == 2
    assert sp["n_protected"] == 3
    # every designable position is touched by >= 1 candidate here
    assert sp["n_touched"] == 4
    # the per-position table carries the protection class + candidate touch count
    tbl = {d["position"]: d for d in sp["table"]}
    assert tbl[40]["klass"] == "designable" and tbl[40]["n_candidates"] == 2
    assert tbl[120]["klass"] == "catalytic"
    assert tbl[15]["klass"] == "fixed" and tbl[33]["klass"] == "fixed"
    # dynamic extra-ligand distance column discovered from the rows
    assert s["extra_cols"] == ["distance_to_cof"]
    # 3D highlight sets: designed positions (40,55,70,71), catalytic, fixed
    assert s["designed_resis"] == [40, 55, 70, 71]
    assert s["catalytic_resis"] == [120]
    assert s["fixed_resis"] == [15, 33]            # 120 removed (catalytic)
    assert s["pdb"] and "HETATM" in s["pdb"]


def test_provenance_rows_sorted_risky_last():
    s = compute_mutation_stats(_gen_rows(), tier_rows=_tier_rows())
    prov = s["provenance"]
    assert len(prov) == 5
    # clean rows first, the risk-flagged one last
    assert prov[-1]["candidate_id"] == "mut_00004" and prov[-1]["risk"] is True
    assert all(not r["risk"] for r in prov[:-1])


# --------------------------------------------------------------------------- #
# render: leftover tokens + summary + charts
# --------------------------------------------------------------------------- #
def test_no_leftover_tokens_and_summary_render():
    html = _build()
    assert "%%" not in html                          # every token substituted
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "xyz1" in html                            # the target id rendered
    # summary cards: total, single/multi, risk, designable, protected
    assert "candidates" in html and "designable" in html and "protected" in html
    assert "5" in html                               # total candidates
    # the four generator names appear as cards / legend
    assert "chemistry rules" in html and "MSA sampler" in html
    assert "LigandMPNN" in html and "multipoint" in html


def test_charts_present_and_wired():
    html = _build()
    # Chart.js library + a real data blob the charts read (var R={...})
    assert "cdnjs.cloudflare.com/ajax/libs/Chart.js" in html
    assert "var R=" in html
    assert '"gen"' in html and '"tier"' in html
    assert 'id="genChart"' in html and 'id="tierChart"' in html
    # the generator + tier counts are in the blob
    assert '"counts"' in html


def test_design_space_section_and_heatmap():
    html = _build()
    assert "Design space" in html
    # protected core visually distinct: catalytic + fixed pills, tinted rows
    assert "catalytic" in html and "designable" in html
    assert 'class="pill"' in html
    assert "tr.catalytic td,tr.fixed td" in html     # the tint CSS rule
    # the per-position table carries the protected positions + a designable one
    assert "A40" in html and "H120" in html          # wt+pos labels
    # the inline-SVG design heatmap (no external JS lib for it)
    assert "<svg" in html and "viewBox" in html
    assert "MSA permissive" in html and "ligand closeness" in html


def test_provenance_table_renders():
    html = _build()
    assert "Per-mutation provenance" in html
    # the per-mutation table: candidate ids + mutation strings + the dynamic
    # extra-ligand distance column header (generic id from the rows)
    assert "mut_00000" in html and "A40K" in html
    assert "mut_00003" in html and "A40K;V70L" in html     # a multipoint row
    assert "d&rarr;cof (&Aring;)" in html or "d→cof (Å)" in html
    # the risk-flagged row surfaces its flag + reason, forbidden-check ok
    assert "mut_00004" in html
    assert "introduces_P@71" in html or "radical_2shell" in html
    assert "flag" in html
    # LigandMPNN log-probability rendered for the mpnn candidate
    assert "-1.230" in html or "-1.23" in html


def test_budget_tiers_render():
    html = _build()
    assert "budget tier" in html.lower()
    # the three tiers + their counts + meanings
    for tier in ("single", "multipoint", "risky"):
        assert tier in html
    assert "exploratory" in html                      # risky tier meaning


def test_3dmol_block_and_highlight_wired():
    html = _build()
    # the 3Dmol viewer script tag + the embedded PDB payload
    assert "3Dmol-min.js" in html or "3dmol" in html
    assert 'id="pdbdata"' in html and "HETATM" in html
    # the three residue-highlight sets are wired into the blob for the viewer
    assert '"designed_resis":[40,55,70,71]' in html
    assert '"catalytic_resis":[120]' in html
    assert '"fixed_resis":[15,33]' in html
    # the design-vs-core legend + the viewer controls
    assert "design region vs protected core" in html.lower()
    assert "toggleDesign" in html and "toggleCore" in html


def test_methods_and_references_present():
    html = _build()
    assert "Methods" in html and "References" in html
    # the multi-strategy generation is described + the protection policy + tiers
    assert "contact-ensemble chemistry rules" in html
    assert "gated MSA sampling" in html
    assert "LigandMPNN" in html and "FuncLib" in html
    # the grounded references
    assert "Dauparas" in html and "Khersonsky" in html
    assert "Nat. Methods" in html                     # LigandMPNN venue
    assert "ProteinMPNN" in html
    assert "Rego" in html and "3Dmol.js" in html
    # the cofactor-switching literature is cited (generic NAD<->NADP design)
    assert "cofactor" in html.lower()


def test_limitations_qc_block_present():
    html = _build()
    assert "Limitations &amp; QC" in html
    # the honest caveats: no score yet; buried-polar deferred; epistasis; risk
    assert "not a ranked list of validated mutants" in html
    assert "deferred_to_s09_s11" in html
    assert "epistasis" in html.lower()
    assert "combinatorial" in html.lower()
    assert "risk filter" in html.lower() or "risk flags" in html.lower()


# --------------------------------------------------------------------------- #
# genericity + graceful degradation
# --------------------------------------------------------------------------- #
def test_generic_no_hardcoded_identity():
    # the TEMPLATE must not bake in any specific protein/ligand identity — the
    # report has to work for any target (report code must not be FDH-specific).
    html = _build(target_id="some_dehydrogenase", ligand_ids=["LIG", "EXTRA"])
    low = html.lower()
    for banned in ("fdh", "nadp", "nadph", "formate", "formic"):
        assert banned not in low, f"hardcoded {banned!r} leaked into the report"
    assert "some_dehydrogenase" in html


def test_degrades_without_pdb():
    # no WT PDB on disk -> the report still builds; the 3D section explains the
    # omission and the rest is intact (report never fails the stage).
    html = _build(wt_pdb=None)
    assert "%%" not in html
    assert "3D design-space view is omitted" in html
    assert 'id="pdbdata"' not in html                # no PDB payload embedded
    # sections 1-3 unaffected
    assert "Per-mutation provenance" in html and "Design space" in html


def test_degrades_without_tiers():
    # no budget-tier files -> the tier section still renders (says none on disk),
    # the rest is intact.
    html = _build(tier_rows={})
    assert "%%" not in html
    assert "no tier files on disk" in html
    assert "Per-mutation provenance" in html


def test_empty_library_does_not_crash():
    # a degenerate run with zero candidates must still render a clean report
    # (no leftover tokens, the empty-state messages appear).
    html = _build(rows=[], tier_rows={})
    assert "%%" not in html
    assert "no candidates generated" in html


# --------------------------------------------------------------------------- #
# standalone loader
# --------------------------------------------------------------------------- #
def test_load_generated_provenance(tmp_path):
    import json

    prov = tmp_path / "reports" / "provenance"
    prov.mkdir(parents=True)
    (prov / "generated_candidates.json").write_text(json.dumps(_gen_rows()))
    tiers = _tier_rows()
    for t, rows in tiers.items():
        (prov / f"generated_candidates_{t}.json").write_text(json.dumps(rows))
    loaded = load_generated_provenance(tmp_path)
    assert len(loaded["generated"]) == 5
    assert set(loaded["tiers"]) == {"single", "multipoint", "risky"}
    assert len(loaded["tiers"]["single"]) == 3
    # nothing on disk -> empty (the caller errors, but the loader does not raise)
    empty = load_generated_provenance(tmp_path / "nope")
    assert empty["generated"] == [] and empty["tiers"] == {}


def test_find_wt_complex_pdb(tmp_path):
    base = (tmp_path / "complexes" / "boltz"
            / "boltz_results_wt_boltz_input" / "predictions" / "wt_boltz_input")
    base.mkdir(parents=True)
    (base / "wt_boltz_input_model_0.pdb").write_text(_PDB)
    found = find_wt_complex_pdb(tmp_path / "complexes")
    assert found is not None and found.endswith("wt_boltz_input_model_0.pdb")
    assert find_wt_complex_pdb(tmp_path / "nope") is None


def test_write_report_to_disk(tmp_path):
    out = tmp_path / "reports" / "s07_mutation_report.html"
    stats = compute_mutation_stats(
        _gen_rows(), tier_rows=_tier_rows(),
        designable_positions=_DESIGNABLE, catalytic_positions=_CATALYTIC,
        fixed_positions=_FIXED, contacts=_contacts(),
        position_features=_features(), wt_pdb_path=_PDB)
    write_mutation_report(
        out, target_id="xyz1", stats=stats,
        conditions=[("generators", "chemistry_rules")],
        generated="2026-06-21 12:00")
    assert out.exists()
    txt = out.read_text()
    assert "%%" not in txt and "xyz1" in txt
