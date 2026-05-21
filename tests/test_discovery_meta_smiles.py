"""Polish-pass regression tests for discovery + builder defaults.

After wave 2 manual smoke we found:
  1. discovery only read SMILES from inputs/ligand.smi - missed the
     pipeline-meta path (s01_input writes meta.ligand_smiles).
  2. discovery only saw target.fasta under inputs/ - some setups drop
     it at run_dir root.
  3. build_report didn't write a zip when output_zip was None - so the
     "portable package" was a directory, not an archive.

This file pins each of those fixes.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from evoliez.figures.discovery import discover


def test_ligand_smiles_from_state_meta(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "_state.json").write_text(json.dumps({
        "completed_stages": ["s01_input"],
        "meta": {"ligand_smiles": "CCO", "sequence_length": 10},
    }))
    arts = discover(run)
    assert arts.ligand_smiles == "CCO"


def test_ligand_smiles_inputs_file_wins_over_meta(tmp_path):
    """If inputs/ligand.smi exists, prefer it (it's the canonical source).
    meta is the fallback."""
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    (run / "inputs" / "ligand.smi").write_text("c1ccccc1 benzene\n")  # canonical
    (run / "_state.json").write_text(json.dumps({"meta": {"ligand_smiles": "CCO"}}))
    arts = discover(run)
    assert arts.ligand_smiles == "c1ccccc1"


def test_ligand_smiles_from_provenance_last_resort(tmp_path):
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    (run / "reports" / "provenance.json").write_text(
        json.dumps({"ligand_smiles": "O=C=O"})
    )
    arts = discover(run)
    assert arts.ligand_smiles == "O=C=O"


def test_target_fasta_found_at_run_root(tmp_path):
    """Some setups don't put it under inputs/ - discovery should still
    find it at the run_dir root."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "target.fasta").write_text(">x\nMAKVL\n")
    arts = discover(run)
    assert arts.target_fasta is not None
    assert arts.target_fasta.name == "target.fasta"


def test_target_fasta_inputs_wins_over_root(tmp_path):
    """If both exist, inputs/ is canonical."""
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    (run / "inputs" / "target.fasta").write_text(">canon\nM\n")
    (run / "target.fasta").write_text(">root\nM\n")
    arts = discover(run)
    assert arts.target_fasta is not None
    assert "inputs" in str(arts.target_fasta)


def test_corrupt_state_json_no_crash(tmp_path):
    """A truncated state.json must not crash discovery - we just lose
    the meta fallback for SMILES."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "_state.json").write_text("{not valid json")
    arts = discover(run)
    # No SMILES (none in inputs/, meta unreadable), but discovery succeeds.
    assert arts.ligand_smiles is None
    assert arts.run_dir == run


# --------------------------------------------------------------------- #
# build_report default zip behavior
# --------------------------------------------------------------------- #
def test_build_report_writes_default_zip(tmp_path):
    """When output_zip is None, build_report should still create a
    portable zip at <run_dir>/report_package.zip (the WHOLE POINT of
    `evoliez figures` is portability - silently producing only a
    directory was a UX gap)."""
    pytest.importorskip("jinja2")
    pytest.importorskip("matplotlib")
    from evoliez.figures.html.builder import build_report

    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    (run / "_state.json").write_text(json.dumps({"meta": {"sequence_length": 5}}))

    result = build_report(run, style="presentation", html_mode="linked")
    assert result["zip_path"] is not None
    assert result["zip_path"].exists()
    assert result["zip_path"].name == "report_package.zip"
    # Zip should contain the entry HTML
    with zipfile.ZipFile(result["zip_path"]) as zf:
        names = zf.namelist()
        assert any("visual_report.html" in n for n in names)
