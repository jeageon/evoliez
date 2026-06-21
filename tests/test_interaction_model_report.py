"""s06b family interaction-geometry report.

Builds the self-contained HTML report from a minimal SYNTHETIC meta + artifacts +
model + a tiny WT-complex PDB (no Boltz, no torch — pure-python data layer), and
checks the things a paper-quality report must guarantee: no leftover template
tokens, the headline holdout AUROC + the self-supervised composition counts render,
the Chart.js data blob is present, the 3Dmol viewer script is included, and the
template carries NO hardcoded protein/ligand identity (it must be generic for any
target — a standing constraint that the report code not be FDH-specific).
"""

from __future__ import annotations

from evoliez.io.interaction_model_report import (
    build_interaction_report_html,
    compute_interaction_stats,
    compute_multi_engine_stats,
    find_wt_complex_pdb,
    write_interaction_report,
)

# ---- synthetic, generic fixtures (an invented protein "xyz1" + ligand atoms) -
_META = {
    "representatives": 12,
    "representative_selection":
        "auto: 30/45 largest clusters cover 90% of 600 homologs "
        "-> clamped to 12 [10,40]",
    "poses_total": 36,
    "train_rows": 60,
    "n_consensus": 30,
    "n_alternative": 4,
    "n_outlier": 6,
    "n_decoy": 14,
    "n_hard_decoy": 6,
    "subfamily_holdout_auroc": 0.9123,
    "model_kind": "xgboost",
    "fp_dim": 26,
}


def _artifacts():
    # ensemble_contacts: a few (residue, ligand-atom) pairs, two are consensus
    # (freq >= 0.5), spanning two ligand atoms — generic atom ids.
    econ = [
        {"residue_index": 40, "ligand_atom_id": "LX1", "contact_frequency": 0.92,
         "mean_distance": 2.7, "confidence_weighted_score": 0.81},
        {"residue_index": 55, "ligand_atom_id": "LX2", "contact_frequency": 0.61,
         "mean_distance": 3.4, "confidence_weighted_score": 0.55},
        {"residue_index": 70, "ligand_atom_id": "LX1", "contact_frequency": 0.30,
         "mean_distance": 4.8, "confidence_weighted_score": 0.20},
        {"residue_index": 70, "ligand_atom_id": "LX2", "contact_frequency": 0.10,
         "mean_distance": 5.6, "confidence_weighted_score": 0.06},
    ]
    # pose_dataset: 6 diffusion samples across 2 groups, with confidence metrics.
    poses = []
    for gid in ("hom_000", "hom_001"):
        for k in range(3):
            poses.append({
                "group_id": gid, "sample_idx": k,
                "confidence_score": 0.70 + 0.05 * k,
                "ligand_iptm": 0.60 + 0.04 * k,
                "complex_plddt": 0.80 + 0.02 * k,
                "affinity_pred_value": -1.2 + 0.3 * k,
                "ptm": 0.75, "iptm": 0.66,
            })
    edges = [{"residue_index": c["residue_index"],
              "ligand_atom_id": c["ligand_atom_id"],
              "mean_distance": c["mean_distance"],
              "contact_frequency": c["contact_frequency"],
              "confidence_weighted_score": c["confidence_weighted_score"],
              "weak_contact": int(c["contact_frequency"] >= 0.5)} for c in econ]
    return {"ensemble_contacts": econ, "pose_dataset": poses,
            "edge_dataset": edges}


# fp_dim 26 = hist(8) + types(6) + summary(4) + kshell(8); a plausible vector.
_MODEL = {
    "kind": "xgboost", "fp_dim": 26,
    "consensus": ([0.1] * 8 + [0.4, 0.0, 0.0, 0.2, 0.3, 0.1]
                  + [3.1, 1.2, 2.6, 0.7] + [3.0] * 8),
    "thr": 2.77, "scale": 0.83, "cutoff": 6.0, "k_nearest": 8, "n_bins": 8,
}

# a tiny 2-residue + 2-atom-ligand PDB (ATOM = protein, HETATM = ligand)
_PDB = (
    "ATOM      1  CA  ALA A  40      10.000  10.000  10.000  1.00 88.00           C\n"
    "ATOM      2  CB  ALA A  40      11.200  10.300  10.100  1.00 86.00           C\n"
    "ATOM      3  CA  ASP A  55      13.000  12.000  10.000  1.00 90.00           C\n"
    "HETATM    4  C1  LIG B   1      12.000  11.000  10.500  1.00  0.00           C\n"
    "HETATM    5  O1  LIG B   1      12.500  11.500  11.000  1.00  0.00           O\n"
    "END\n"
)


# ---- synthetic multi-engine docking audit (generic engines/roles/contexts) --
# Two engines that are COMPLEMENTARY: "alpha" mostly corroborates the consensus
# (weak_positive) while "beta" supplies the hard-negatives; each engine handles
# the cofactor context differently (retained vs dropped). Shape mirrors the on-
# disk multi_engine_docking.json the s06b stage writes.
def _me_audit():
    rows = []

    def row(source, candidate, role, rmsd, fp, score, ctx, rank, score_type):
        return {"candidate_id": candidate, "source": source, "role": role,
                "rmsd_to_consensus": rmsd, "fp_overlap": fp, "clash": False,
                "key_contacts_ok": role != "strong_negative", "score": score,
                "sample_weight": 0.5, "score_gate_pass": role == "hard_negative",
                "rank": rank, "score_type": score_type, "cnn_score": None,
                "cnn_affinity": None, "context_mode": ctx,
                "ligand_id": "L0", "ligand_role": "primary"}

    # alpha: corroborates consensus (weak_positive), context retained
    rows += [row("alpha", "wt", "weak_positive", 1.1, 0.82, -8.1,
                 "context_retained", 1, "cnn_affinity"),
             row("alpha", "rep_007", "weak_positive", 1.4, 0.77, -7.6,
                 "context_retained", 1, "cnn_affinity"),
             row("alpha", "rep_012", "excluded", 9.0, 0.10, -2.0,
                 "context_retained", 5, "cnn_affinity")]
    # beta: hard-negatives (plausible score, discordant geometry), context dropped
    rows += [row("beta", "wt", "hard_negative", 6.4, 0.21, 0.94,
                 "context_dropped", 1, "confidence"),
             row("beta", "rep_007", "hard_negative", 5.8, 0.18, 0.88,
                 "context_dropped", 2, "confidence"),
             row("beta", "rep_021", "strong_negative", 8.9, 0.05, 0.30,
                 "context_dropped", 7, "confidence")]
    return {"methods": ["alpha", "beta"], "n_targets": 3, "n_kept": 4,
            "n_classified": len(rows), "status": [], "rows": rows}


def _build(**over):
    stats = compute_interaction_stats(
        _META, _artifacts(), _MODEL, over.pop("wt_pdb", _PDB),
        multi_engine_audit=over.pop("multi_engine_audit", None))
    return build_interaction_report_html(
        target_id=over.pop("target_id", "xyz1"),
        stats=stats,
        conditions=[("model", "xgboost"), ("backend", "real")],
        generated="2026-06-20 12:00",
    )


# --------------------------------------------------------------------------- #
def test_stats_compute_from_disk_shapes():
    s = compute_interaction_stats(_META, _artifacts(), _MODEL, _PDB)
    # representative-selection funnel parsed from the human note
    fn = s["funnel"]
    assert fn["mode"] == "auto"
    assert fn["n_total"] == 600 and fn["n_clusters"] == 45
    assert fn["coverage"] == 0.90 and fn["n_reps"] == 12
    assert fn["steps"][0]["value"] == 600          # homolog pool first
    assert fn["steps"][-1]["value"] == 12           # representatives last
    # WT-ensemble contact map: the two >=0.5 contacts are the high-freq set
    assert s["n_contacts"] == 4 and s["n_strong"] == 2
    assert {d["residue_index"] for d in s["consensus_contacts"]} == {40, 55}
    assert s["conserved_resis"] == [40, 55]         # for the 3D highlight
    # clash-range physical QC: none of these contacts is < 1.5 A (min is 2.7)
    assert s["n_clash_contacts"] == 0
    assert s["min_contact_distance"] == 2.7
    # per-residue map + per-atom summary populated
    assert any(d["max_freq"] >= 0.9 for d in s["res_map"])
    assert {d["ligand_atom_id"] for d in s["atom_summary"]} == {"LX1", "LX2"}
    # pose-reliability histograms for the populated metrics
    assert "confidence_score" in s["pose_hists"]
    assert s["pose_hists"]["confidence_score"]["stats"]["n"] == 6
    assert len(s["per_group"]) == 2                 # one row per representative
    # decoded consensus vector (honest: dominant type + distance summary)
    assert s["consensus_vec"]["dominant_type"] == "hbond"
    assert s["consensus_vec"]["summary"]["mean_min_dist"] == 3.1
    assert s["pdb"] and "HETATM" in s["pdb"]


def test_no_leftover_tokens_and_headline_render():
    html = _build()
    assert "%%" not in html                          # every token substituted
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "xyz1" in html                            # the target id rendered
    # headline holdout AUROC present (card + section + gauge value)
    assert "0.912" in html                           # 0.9123 rounded in cards
    assert "0.9123" in html                          # exact in section 5
    assert "subfamily-holdout AUROC" in html


def test_composition_counts_render():
    html = _build()
    # the self-supervised pose-selection counts must all appear
    for lab in ("consensus", "alternative", "outlier", "decoy", "hard_decoy"):
        assert lab in html
    assert ">30<" in html                            # n_consensus
    assert ">14<" in html                            # n_decoy
    # positives vs negatives headline (6 outlier + 14 decoy + 6 hard = 26)
    assert "26" in html


def test_chartjs_blob_and_3dmol_present():
    html = _build()
    # Chart.js library + a real data blob the charts read (var R={...})
    assert "cdnjs.cloudflare.com/ajax/libs/Chart.js" in html
    assert "var R=" in html
    assert '"freq_hist"' in html and '"res_map"' in html
    assert '"composition"' in html
    # the 3Dmol viewer script tag + the embedded PDB payload
    assert "3Dmol-min.js" in html or "3dmol" in html
    assert 'id="pdbdata"' in html and "HETATM" in html
    assert '"conserved_resis":[40,55]' in html       # 3D highlight wired


def test_section3_wording_not_overclaiming():
    # the domain-expert fix: section-3 (the WT-ensemble contact table/map) must
    # NOT be framed as homolog conservation. The old overclaiming phrases are gone
    # and the honest WT-ensemble framing is present.
    html = _build()
    # the ORIGINAL affirmative overclaims (the contact table called itself a
    # family consensus / conserved binding site). Honest negations like "...not a
    # family-conserved binding site" are allowed, so we ban the affirmative forms.
    for banned in ("Family-consensus contacts",
                   "the family-conserved binding site",
                   "define the conserved binding mode",
                   "the pharmacophore the family conserves"):
        assert banned not in html, f"overclaiming phrase still present: {banned!r}"
    # the corrected section-3 heading + WT-ensemble framing
    assert "WT Boltz-ensemble contact frequency" in html
    assert "WT-ensemble contact frequency" in html
    assert "WT diffusion-sample reproducibility" in html
    # the section-3 contacts table is renamed to the honest, non-conservation title
    assert "High-frequency WT-ensemble contacts" in html


# --------------------------------------------------------------------------- #
# Multi-engine docking augmentation section
# --------------------------------------------------------------------------- #
def test_multi_engine_stats_crosstab():
    # the DATA layer builds the role x engine cross-tab, per-engine context-mode
    # breakdown and the top hard-negatives directly from the audit rows.
    me = compute_multi_engine_stats(_me_audit())
    assert me is not None
    assert me["engines"] == ["alpha", "beta"]          # declared method order
    # roles present, canonical order first
    assert me["roles"][:3] == ["weak_positive", "hard_negative",
                               "strong_negative"]
    # role x engine counts: alpha=2 weak_positive + 1 excluded; beta=2 hard_neg
    assert me["role_by_engine"]["alpha"]["weak_positive"] == 2
    assert me["role_by_engine"]["alpha"]["excluded"] == 1
    assert me["role_by_engine"]["alpha"]["total"] == 3
    assert me["role_by_engine"]["beta"]["hard_negative"] == 2
    assert me["role_by_engine"]["beta"]["strong_negative"] == 1
    # hard-negatives come ONLY from beta here -> complementarity surfaced
    assert me["role_totals"]["hard_negative"] == 2
    assert me["role_by_engine"]["alpha"].get("hard_negative", 0) == 0
    # per-engine context-mode: alpha retained vs beta dropped (honest)
    a_ctx = {m["mode"]: m["count"] for m in me["context_by_engine"]["alpha"]}
    b_ctx = {m["mode"]: m["count"] for m in me["context_by_engine"]["beta"]}
    assert a_ctx == {"context_retained": 3}
    assert b_ctx == {"context_dropped": 3}
    # top hard-negative is beta/wt (highest score 0.94), with its geometry
    assert me["hard_negatives"][0]["source"] == "beta"
    assert me["hard_negatives"][0]["candidate_id"] == "wt"
    assert me["hard_negatives"][0]["score"] == 0.94
    # chart matrix is engine x role aligned to the role order
    assert me["chart"]["engines"] == ["alpha", "beta"]
    assert me["chart"]["matrix"][1][me["chart"]["roles"].index(
        "hard_negative")] == 2


def test_multi_engine_section_renders():
    # the section renders: heading, role x engine table, per-engine counts, the
    # context-mode breakdown, an inline-SVG chart, and a hard-negative example.
    html = _build(multi_engine_audit=_me_audit())
    assert "%%" not in html
    assert "Multi-engine docking augmentation" in html      # the section heading
    assert "Pose role &times; engine" in html or "Pose role × engine" in html
    # both engines named + the complementary roles in the cross-tab
    assert "alpha" in html and "beta" in html
    assert "weak positive" in html and "hard negative" in html
    # per-engine context-mode handling (retained vs dropped) is shown
    assert "context_retained" in html and "context_dropped" in html
    # a hard-negative example row: top is beta / wt with score 0.94
    assert "0.940" in html                                  # score rendered (3dp)
    assert "rep_007" in html                                # a per-rep target id
    # the inline-SVG role-distribution chart (no external JS lib for it)
    assert "<svg" in html and "viewBox" in html
    assert "plausible engine score, discordant family geometry" in html \
        or "discordant family geometry" in html
    # methods trail: classification inputs + the audit-trail file name
    assert "RMSD-to-its-own-target-pose" in html
    assert "multi_engine_docking.json" in html


def test_multi_engine_section_absent_when_no_audit():
    # multi_engine off (no audit) -> the section is skipped CLEANLY: no heading,
    # no empty section, and the rest of the report is unaffected.
    html = _build()                                          # no audit passed
    assert "%%" not in html
    assert "Multi-engine docking augmentation" not in html
    assert "Pose role" not in html
    # the rest still renders (sections renumbered: 3D viewer is section 7)
    assert "7 · 3D structure" in html
    assert "0.9123" in html

    # None / empty-rows audit also skips the section (no error)
    assert compute_multi_engine_stats(None) is None
    assert compute_multi_engine_stats({"methods": ["a"], "rows": []}) is None
    html_empty = _build(multi_engine_audit={"methods": ["a"], "rows": []})
    assert "Multi-engine docking augmentation" not in html_empty


def test_limitations_qc_box_present():
    # a prominent Limitations & QC box must state the four honest caveats and frame
    # the report as a computational prior, not a validated binding mode.
    html = _build()
    assert "Limitations &amp; QC" in html
    assert "COMPUTATIONAL PRIOR for" in html
    assert "not a measured or validated binding mode" in html
    # (a) WT reproducibility, not a per-representative conservation matrix
    assert "WT-ensemble" in html
    assert "per-representative contact" in html
    assert "residue × ligand-atom ×" in html        # the missing true-conservation
    assert "not yet computed" in html
    # (b) co-modelled extra ligands (cofactor + substrate), not yet annotated
    assert "extra ligands" in html
    assert "cofactor + substrate" in html
    assert "not yet" in html and "annotated" in html
    # (c) clash-range physical QC needs PoseBusters / PLIP
    assert "clash range" in html and "PoseBusters / PLIP" in html
    # (d) holdout AUROC is internal CV, not experimental validation
    assert "internal\ncross-validation, not experimental validation" in html
    assert "a measured binding affinity" in html     # "...not</b> a measured ..."


def test_clash_qc_renders_when_short_contacts_exist():
    # a HIGH-frequency (>= 0.5) clash-range contact is NOT sporadic: it must be
    # flagged as needing resolution, not dismissed as a diffusion artifact.
    arts = _artifacts()
    arts["ensemble_contacts"] = arts["ensemble_contacts"] + [
        {"residue_index": 41, "ligand_atom_id": "LX3", "contact_frequency": 0.88,
         "mean_distance": 1.1, "confidence_weighted_score": 0.70},
    ]
    s = compute_interaction_stats(_META, arts, _MODEL, _PDB)
    assert s["n_clash_contacts"] == 1
    assert s["min_contact_distance"] == 1.1
    assert s["n_clash_highfreq"] == 1 and s["clash_all_sporadic"] is False
    html = build_interaction_report_html(
        target_id="xyz1", stats=s,
        conditions=[("model", "xgboost")], generated="2026-06-20 12:00")
    assert "%%" not in html
    # the dynamic, run-specific Physical-QC warning (distinct from the static
    # limitations caveat that always mentions "clash range")
    assert "<b>Physical QC:" in html
    assert "1 ensemble contact" in html          # singular, count rendered
    assert "high-frequency" in html and "NOT sporadic" in html
    assert "PoseBusters" in html

    # control: with no short contacts the dynamic clash warning is absent (the
    # static limitations caveat about clash range may still be present)
    s0 = compute_interaction_stats(_META, _artifacts(), _MODEL, _PDB)
    assert s0["n_clash_contacts"] == 0
    html0 = build_interaction_report_html(
        target_id="xyz1", stats=s0,
        conditions=[("model", "xgboost")], generated="2026-06-20 12:00")
    assert "<b>Physical QC:" not in html0
    assert "sporadic diffusion clashes" not in html0


def test_sporadic_clash_contacts_flagged_and_excluded():
    # FIX 3 (the production scenario): several LOW-frequency clash-range contacts
    # (none reaching the >= 0.5 consensus set) must be flagged as sporadic
    # Boltz-diffusion artifacts, their (low) frequency shown, EXCLUDED from any
    # binding-mode statement, and the high-frequency pocket noted as clash-free.
    arts = _artifacts()
    # mirror fdh_5track: a clutch of <1.5 Å contacts at frequency 0.025–0.075,
    # closest 0.67 Å, while the bulk pocket median sits ~4.8 Å and the >=0.5
    # contacts (from _artifacts) are all at >= 2.7 Å (clash-free).
    arts["ensemble_contacts"] = arts["ensemble_contacts"] + [
        {"residue_index": 355, "ligand_atom_id": "O42", "contact_frequency": 0.025,
         "mean_distance": 0.67, "confidence_weighted_score": 0.02},
        {"residue_index": 356, "ligand_atom_id": "O43", "contact_frequency": 0.05,
         "mean_distance": 1.2, "confidence_weighted_score": 0.03},
        {"residue_index": 357, "ligand_atom_id": "O44", "contact_frequency": 0.075,
         "mean_distance": 1.4, "confidence_weighted_score": 0.04},
    ]
    s = compute_interaction_stats(_META, arts, _MODEL, _PDB)
    assert s["n_clash_contacts"] == 3
    assert s["n_clash_highfreq"] == 0            # ZERO high-frequency clashes
    assert s["clash_all_sporadic"] is True
    assert s["min_contact_distance"] == 0.67
    assert s["clash_freq_max"] == 0.075 and s["clash_freq_min"] == 0.025
    # the sporadic clashes do NOT enter the high-frequency contact / 3D-highlight
    # set (binding-mode statement is unaffected)
    assert 355 not in s["conserved_resis"]
    assert all(d["residue_index"] not in (355, 356, 357)
               for d in s["consensus_contacts"])

    html = build_interaction_report_html(
        target_id="xyz1", stats=s,
        conditions=[("model", "xgboost")], generated="2026-06-20 12:00")
    assert "%%" not in html
    # the sporadic-artifact framing, the low frequency, the explicit exclusion,
    # and the clash-free median are all surfaced
    assert "sporadic diffusion clashes" in html
    assert "3 ensemble contacts" in html
    assert "0.025" in html and "0.075" in html            # the low frequency band
    assert "0 high-frequency" in html
    assert "EXCLUDED from any binding-mode statement" in html
    assert "bulk pocket is clash-free" in html
    assert "median contact distance" in html
    # the clashing residues are NOT highlighted in 3D
    assert '"conserved_resis":[40,55]' in html


def test_generic_no_hardcoded_identity():
    # the TEMPLATE must not bake in any specific protein/ligand identity — the
    # report has to work for any target (report code must not be FDH-specific).
    html = _build(target_id="some_dehydrogenase")
    low = html.lower()
    for banned in ("fdh", "nadp", "nadph", "formate", "formic"):
        assert banned not in low, f"hardcoded {banned!r} leaked into the report"
    assert "some_dehydrogenase" in html


def test_degrades_without_pdb():
    # no WT PDB on disk -> the report still builds; the 3D section explains the
    # omission and the rest is intact (report never fails the stage).
    html = _build(wt_pdb=None)
    assert "%%" not in html
    assert "3D pocket view is omitted" in html
    assert "0.9123" in html                          # sections 1-5 unaffected
    assert 'id="pdbdata"' not in html                # no PDB payload embedded


def test_find_wt_complex_pdb(tmp_path):
    # the on-disk locator globs the Boltz predictions tree and prefers the wt one
    base = (tmp_path / "complexes" / "boltz"
            / "boltz_results_wt_boltz_input" / "predictions" / "wt_boltz_input")
    base.mkdir(parents=True)
    p = base / "wt_boltz_input_model_0.pdb"
    p.write_text(_PDB)
    found = find_wt_complex_pdb(tmp_path / "complexes")
    assert found is not None and found.endswith("wt_boltz_input_model_0.pdb")
    # nothing on disk -> None (not an error)
    assert find_wt_complex_pdb(tmp_path / "nope") is None


def test_write_report_to_disk(tmp_path):
    out = tmp_path / "reports" / "interaction_model_report.html"
    write_interaction_report(
        out, target_id="xyz1",
        stats=compute_interaction_stats(_META, _artifacts(), _MODEL, _PDB),
        conditions=[("model", "xgboost")], generated="2026-06-20 12:00")
    assert out.exists()
    txt = out.read_text()
    assert "%%" not in txt and "xyz1" in txt
