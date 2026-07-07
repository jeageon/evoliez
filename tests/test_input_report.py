"""s01 input report: provenance + RDKit ligand chemistry, generic for any target."""

from __future__ import annotations

import pytest

from evoliez.io.input_report import build_input_report_html, compute_input_stats

pytest.importorskip("rdkit")


def _stats(**over):
    base = dict(
        target_id="myprot", sequence="MKVLAAACDEFGHIKLMNPQ",
        sequence_source="inline target_sequence",
        ligand=("lig1", "smiles", "CC(=O)[O-]"),          # acetate, charge -1
        extra_ligands=[("formate", "smiles", "[O-]C=O")],
        residues={"catalytic": ["H100"], "fixed": [], "binding": ["D55"]},
        organism="E. coli", ec_number="1.2.1.2", target_ph=7.4,
        ligand_atom_ids=["C1", "C2", "O1", "O2"],
    )
    base.update(over)
    return compute_input_stats(**base)


def test_chemistry_and_composition():
    s = _stats()
    assert s["length"] == 20
    assert s["ligand"]["charge"] == -1           # acetate anion
    assert s["ligand"]["formula"] and s["ligand"]["heavy"] == 4
    assert s["ligand"]["n_atom_ids"] == 4
    assert s["composition"]["A"] == 3            # 3 alanines in the test seq
    assert len(s["input_hash"]) == 16
    assert len(s["extra_ligands"]) == 1 and s["extra_ligands"][0]["id"] == "formate"


def test_build_no_leftover_tokens():
    html = build_input_report_html(stats=_stats(), generated="2026-06-20 10:00")
    assert "%%" not in html
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "compChart" in html and "input hash" in html
    assert "not provided" in html              # honest provenance gaps surfaced


def test_generic_for_other_proteins():
    # a totally different protein + a neutral aromatic ligand, no residues set
    s = _stats(target_id="some_kinase", sequence="GSAGSAGSAGSA" * 4,
               ligand=("benz", "smiles", "c1ccccc1"), extra_ligands=[],
               residues={}, organism=None, ec_number=None)
    html = build_input_report_html(stats=s, generated="x")
    assert "%%" not in html and "some_kinase" in html
    assert s["ligand"]["charge"] == 0 and s["ligand"]["rings"] == 1
    assert s["extra_ligands"] == []


def test_unassigned_stereo_flagged():
    # a chiral SMILES with UNspecified stereo -> report must flag it
    s = _stats(ligand=("chir", "smiles", "CC(O)C(=O)O"))  # lactic acid, no @/@@
    assert s["ligand"]["stereo_unassigned"] >= 1
    html = build_input_report_html(stats=s, generated="x")
    assert "unassigned stereocentres" in html


# --- FIX 4: structured accession metadata + stereocentre COUNT warning ------- #
def test_accession_metadata_structured_from_config():
    # UniProt + PDB accession + numbering scheme are surfaced as STRUCTURED
    # provenance (not the generic "not provided" warning) when config supplies them.
    s = _stats(accession="Q9S7E4", pdb_id="3JTM",
               numbering_scheme="PDB 3JTM author numbering")
    assert s["accession"] == "Q9S7E4" and s["pdb_id"] == "3JTM"
    assert s["numbering_scheme"] == "PDB 3JTM author numbering"
    html = build_input_report_html(stats=s, generated="x")
    assert "%%" not in html
    assert "UniProt Q9S7E4 · PDB 3JTM" in html       # combined accession cell
    assert "PDB 3JTM author numbering" in html        # numbering convention
    assert "residue-numbering convention" in html
    # the as-provided / not-provided warnings for these two rows are GONE
    assert "as-provided — verify vs reference DB" not in html
    assert "not provided — add for provenance" not in html


def test_accession_partial_and_absent():
    # only one of the two ids -> still structured; neither -> keep the warning
    only_uni = build_input_report_html(stats=_stats(accession="P12345"),
                                       generated="x")
    assert "UniProt P12345" in only_uni and "PDB" not in only_uni.split(
        "UniProt P12345")[1][:8]
    only_pdb = build_input_report_html(stats=_stats(pdb_id="1ABC"), generated="x")
    assert "PDB 1ABC" in only_pdb
    none_html = build_input_report_html(stats=_stats(), generated="x")
    assert "not provided — add for provenance" in none_html
    assert "as-provided — verify vs reference DB" in none_html


def test_stereo_count_warning_covers_extra_ligands():
    # an EXTRA ligand (cofactor) with unassigned stereocentres must be caught too,
    # with its COUNT shown — mirrors NADP-as-cofactor with floating stereo.
    s = _stats(ligand=("acet", "smiles", "CC(=O)[O-]"),       # no stereocentres
               extra_ligands=[("cof", "smiles", "CC(O)C(O)C(=O)O")])  # 2 chiral C
    assert s["ligand_stereo_unassigned"] == 0
    assert s["extra_ligands"][0]["stereo_unassigned"] >= 1
    html = build_input_report_html(stats=s, generated="x")
    assert "%%" not in html
    assert "unassigned stereocentres" in html
    # the offending ligand is NAMED with its count, e.g. "cof (2 unassigned)"
    n = s["extra_ligands"][0]["stereo_unassigned"]
    assert f"cof ({n} unassigned)" in html


def test_stereo_note_clean_when_all_assigned():
    s = _stats(ligand=("acet", "smiles", "CC(=O)[O-]"), extra_ligands=[])
    assert s["ligand_stereo_unassigned"] == 0
    html = build_input_report_html(stats=s, generated="x")
    assert "All stereocentres assigned." in html
    assert "unassigned stereocentres" not in html
