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
