"""Foundation tests for the ``evoliez.figures`` package.

Covers the wave-1 surface: package imports, style helpers, manifest writer,
artifact discovery, and the CLI stub registration.  Heavier deps
(matplotlib, jinja2, pillow) are skipped gracefully when not installed so
the suite stays green on the laptop dev box.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Imports / public API
# ---------------------------------------------------------------------------


def test_public_api_imports() -> None:
    """All advertised symbols are importable from the package root."""
    from evoliez.figures import (  # noqa: F401
        EVIDENCE_COLORS,
        JOURNAL_STYLES,
        WONG_PALETTE,
        FigureSpec,
        ManifestBuilder,
        ReportArtifacts,
        apply_mpl_style,
        discover,
        evidence_color,
        is_paper_style,
    )


# ---------------------------------------------------------------------------
# style.py
# ---------------------------------------------------------------------------


def test_evidence_color_known_and_unknown() -> None:
    from evoliez.figures.style import evidence_color

    assert evidence_color("Strong") == "#009E73"
    assert evidence_color("Promising") == "#56B4E9"
    assert evidence_color("Uncertain") == "#E69F00"
    assert evidence_color("Reject") == "#999999"
    # Unknown classes fall back to neutral grey.
    assert evidence_color("Bogus") == "#cccccc"
    assert evidence_color("") == "#cccccc"


def test_is_paper_style() -> None:
    from evoliez.figures.style import is_paper_style

    assert is_paper_style("paper") is True
    assert is_paper_style("presentation") is False
    assert is_paper_style("poster") is False


def test_journal_styles_complete() -> None:
    from evoliez.figures.style import JOURNAL_STYLES

    assert set(JOURNAL_STYLES) == {"paper", "presentation", "poster"}
    for name, params in JOURNAL_STYLES.items():
        assert "dpi" in params, f"{name} missing dpi"
        assert "figure.figsize" in params, f"{name} missing figure.figsize"


def test_wong_palette_shape() -> None:
    from evoliez.figures.style import WONG_PALETTE

    assert len(WONG_PALETTE) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in WONG_PALETTE)


def test_apply_mpl_style_sets_color_cycle() -> None:
    """If matplotlib is installed, apply_mpl_style updates rcParams + cycler."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cycler")
    import matplotlib as mpl

    from evoliez.figures.style import WONG_PALETTE, apply_mpl_style

    apply_mpl_style("presentation")
    # font.size from the presentation preset.
    assert mpl.rcParams["font.size"] == 12
    cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle == WONG_PALETTE


def test_apply_mpl_style_rejects_unknown() -> None:
    from evoliez.figures.style import apply_mpl_style

    with pytest.raises(KeyError):
        apply_mpl_style("not-a-style")


# ---------------------------------------------------------------------------
# manifest.py
# ---------------------------------------------------------------------------


def test_manifest_add_and_write(tmp_path: Path) -> None:
    from evoliez.figures import FigureSpec, ManifestBuilder

    builder = ManifestBuilder(run_dir=tmp_path, style="presentation", mock_backend=True)
    builder.add_figure(
        FigureSpec(
            figure_id="02_conservation_heatmap",
            section="msa",
            title="Per-residue conservation",
            description="Heatmap of conservation scores",
            path=Path("figures/02_conservation_heatmap.png"),
            source_files=[Path("msa/conservation.json")],
            renderer="matplotlib",
            params={"palette": "viridis"},
        )
    )
    builder.add_warning("ligand SMILES missing - skipped figure 04")

    out = builder.write(tmp_path / "reports" / "visual_manifest.json")
    assert out.exists()

    data = json.loads(out.read_text())
    assert data["schema_version"] == "1.0"
    assert data["style"] == "presentation"
    assert data["mock_backend"] is True
    assert data["warnings"] == ["ligand SMILES missing - skipped figure 04"]
    assert data["generated_at"].endswith("Z")
    assert len(data["figures"]) == 1

    fig = data["figures"][0]
    assert fig["figure_id"] == "02_conservation_heatmap"
    # Paths are posix-style strings in the JSON form.
    assert fig["path"] == "figures/02_conservation_heatmap.png"
    assert fig["source_files"] == ["msa/conservation.json"]
    assert fig["generated_at"] is not None  # auto-stamped on add


def test_manifest_git_sha_graceful(tmp_path: Path) -> None:
    """``git rev-parse`` outside a repo (or with no git binary) -> 'unknown'."""
    from evoliez.figures import ManifestBuilder

    builder = ManifestBuilder(run_dir=tmp_path, style="paper")
    # Either we're inside the repo (short sha) or we got the graceful fallback.
    sha = builder.git_sha
    assert isinstance(sha, str) and sha != ""
    # Short SHAs are 4-40 hex chars; "unknown" is the documented fallback.
    assert sha == "unknown" or all(c in "0123456789abcdef" for c in sha.lower())


def test_manifest_config_sha1_from_state(tmp_path: Path) -> None:
    from evoliez.figures import ManifestBuilder

    (tmp_path / "_state.json").write_text('{"stages": {}}')
    builder = ManifestBuilder(run_dir=tmp_path, style="paper")
    assert builder.config_sha1 != "unknown"
    assert len(builder.config_sha1) == 8


def test_manifest_config_sha1_missing(tmp_path: Path) -> None:
    from evoliez.figures import ManifestBuilder

    builder = ManifestBuilder(run_dir=tmp_path, style="paper")
    assert builder.config_sha1 == "unknown"


# ---------------------------------------------------------------------------
# discovery.py
# ---------------------------------------------------------------------------


def test_discover_empty_dir(tmp_path: Path) -> None:
    from evoliez.figures import discover

    arts = discover(tmp_path)
    assert arts.run_dir == tmp_path
    assert arts.target_fasta is None
    assert arts.ligand_smiles is None
    assert arts.alignment_fasta is None
    assert arts.conservation_json is None
    assert arts.wt_complex_pdb is None
    assert arts.mutant_complex_pdbs == {}
    assert arts.homolog_complex_dirs == []
    assert arts.final_candidates_csv is None
    assert arts.focused_library_csv is None
    assert arts.benchmark_json is None
    assert arts.benchmark_csv is None
    assert arts.provenance_json is None
    assert arts.state_json is None
    assert arts.md_dirs == {}
    assert arts.interaction_model_json is None
    assert arts.ml_datasets == {}


def test_discover_missing_run_dir(tmp_path: Path) -> None:
    """discover() on a path that doesn't exist returns an all-None record."""
    from evoliez.figures import discover

    ghost = tmp_path / "no-such-run"
    arts = discover(ghost)
    assert arts.run_dir == ghost
    assert arts.final_candidates_csv is None


def test_discover_populated(tmp_path: Path) -> None:
    """Lay out a minimal pipeline tree and verify each artifact is found."""
    from evoliez.figures import discover

    # state + reports
    (tmp_path / "_state.json").write_text("{}")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "final_candidates.csv").write_text("mutation,score\nA1G,0.9\n")
    (reports / "focused_library.csv").write_text("seq\n")
    (reports / "benchmark.json").write_text("{}")
    (reports / "provenance.json").write_text("{}")

    # inputs
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "target.fasta").write_text(">t\nMVLS\n")
    (inputs / "ligand.smi").write_text("CCO ethanol\n")

    # msa
    msa = tmp_path / "msa"
    msa.mkdir()
    (msa / "alignment.fasta").write_text(">a\nMVLS\n")
    (msa / "conservation.json").write_text("[]")

    # complexes - WT
    wt_pred = (
        tmp_path
        / "complexes"
        / "boltz_results_wt_boltz_input"
        / "predictions"
        / "wt_boltz_input"
    )
    wt_pred.mkdir(parents=True)
    wt_pdb = wt_pred / "wt_boltz_input_model_0.pdb"
    wt_pdb.write_text("REMARK wt\n")

    # complexes - mutants
    mut_pred = (
        tmp_path
        / "complexes"
        / "mutant_boltz"
        / "boltz_results_mut_cand_001_boltz_input"
        / "predictions"
        / "mut_cand_001_boltz_input"
    )
    mut_pred.mkdir(parents=True)
    mut_pdb = mut_pred / "mut_cand_001_boltz_input_model_0.pdb"
    mut_pdb.write_text("REMARK mut\n")

    # homolog representatives
    homolog_dir = tmp_path / "complexes" / "representatives" / "rep_A"
    homolog_dir.mkdir(parents=True)

    # md
    md_dir = tmp_path / "md" / "cand_001"
    md_dir.mkdir(parents=True)
    (md_dir / "analysis.json").write_text("{}")

    # interaction graphs
    igraphs = tmp_path / "interaction_graphs"
    igraphs.mkdir()
    (igraphs / "interaction_model.json").write_text("{}")

    # ml datasets
    ml = tmp_path / "ml_datasets"
    ml.mkdir()
    (ml / "pose_level.csv").write_text("a,b\n1,2\n")
    (ml / "mutation_level.csv").write_text("a,b\n1,2\n")

    arts = discover(tmp_path)
    assert arts.state_json == tmp_path / "_state.json"
    assert arts.final_candidates_csv == reports / "final_candidates.csv"
    assert arts.focused_library_csv == reports / "focused_library.csv"
    assert arts.benchmark_json == reports / "benchmark.json"
    assert arts.provenance_json == reports / "provenance.json"
    assert arts.target_fasta == inputs / "target.fasta"
    assert arts.ligand_smiles == "CCO"
    assert arts.alignment_fasta == msa / "alignment.fasta"
    assert arts.conservation_json == msa / "conservation.json"
    assert arts.wt_complex_pdb == wt_pdb
    assert arts.mutant_complex_pdbs == {"cand_001": mut_pdb}
    assert arts.homolog_complex_dirs == [homolog_dir]
    assert arts.md_dirs == {"cand_001": md_dir}
    assert arts.interaction_model_json == igraphs / "interaction_model.json"
    assert arts.ml_datasets == {
        "pose_level": ml / "pose_level.csv",
        "mutation_level": ml / "mutation_level.csv",
    }


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def test_cli_figures_help() -> None:
    """`evoliez figures --help` exits cleanly and mentions the key options."""
    from typer.testing import CliRunner

    from evoliez.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["figures", "--help"])
    assert result.exit_code == 0, result.output
    assert "--style" in result.output
    assert "--html" in result.output
    assert "--run-dir" in result.output


def test_cli_figures_rejects_bad_style(tmp_path: Path) -> None:
    """Wave-2: CLI validates --style/--html before touching the config."""
    pytest.importorskip("jinja2")
    from typer.testing import CliRunner

    from evoliez.cli import app

    runner = CliRunner()
    cfg = tmp_path / "config.yaml"
    cfg.write_text("project: {}\n")
    # Bogus style flag is rejected with exit code 2 before any config
    # validation runs.
    result = runner.invoke(app, ["figures", "-c", str(cfg), "--style", "bogus"])
    assert result.exit_code == 2, result.output
    assert "--style" in result.output
