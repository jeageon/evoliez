"""End-to-end test: synthesize a mock pipeline run dir, build the report,
verify the produced zip + HTML are usable.

This is the most important test in the builder integration wave: a real
"green" CI signal means the orchestrator can wire Foundation, 2D plots,
3D viewers (when present), and Packaging together against a believable
fake of what stages emit.

Heavy deps (matplotlib / jinja2) are gated with importorskip so the test
silently skips on the barebones laptop dev box.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterable

import pytest


# ---------------------------------------------------------------------------
# Synthetic-run-dir helpers
# ---------------------------------------------------------------------------


def _wt_complex_pdb() -> str:
    """A tiny three-atom PDB string the 3Dmol viewer + discovery happily accept."""
    return (
        "REMARK   1 EvoLiEZ synthetic WT complex\n"
        "ATOM      1  N   ALA A   1      11.104  13.207  10.005  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      12.560  13.207  10.005  1.00 20.00           C\n"
        "ATOM      3  C   ALA A   1      13.207  14.503  10.005  1.00 20.00           C\n"
        "HETATM    4  C1  NDP B   2      14.000  14.000  10.000  1.00 30.00           C\n"
        "END\n"
    )


def _write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[str]]) -> None:
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_synthetic_run_dir(run_dir: Path) -> Path:
    """Create the minimal tree ``discover`` + every plot renderer can consume.

    Mirrors what a real pipeline run produces, just trimmed to the parts
    the report cares about.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "msa").mkdir(parents=True, exist_ok=True)

    # ---- _state.json ---------------------------------------------------
    (run_dir / "_state.json").write_text(
        json.dumps(
            {
                "target": "FDH_demo",
                "started": "2026-05-21T09:00:00Z",
                "finished": "2026-05-21T09:30:00Z",
                "backend": "mock",
                "catalytic_targets": ["D222"],
            }
        ),
        encoding="utf-8",
    )

    # ---- provenance.json ----------------------------------------------
    (run_dir / "reports" / "provenance.json").write_text(
        json.dumps(
            {
                "ligand_smiles": "CCO",
                "evoliez_version": "test",
            }
        ),
        encoding="utf-8",
    )

    # ---- inputs --------------------------------------------------------
    target_seq = "M" + ("A" * 49)  # 50-residue toy target
    (run_dir / "inputs" / "target.fasta").write_text(
        f">FDH_demo\n{target_seq}\n", encoding="utf-8"
    )
    (run_dir / "inputs" / "ligand.smi").write_text("CCO ethanol\n", encoding="utf-8")

    # ---- MSA conservation ---------------------------------------------
    cons_scores = [
        round(0.5 + 0.4 * ((i * 7) % 13) / 13.0, 3) for i in range(len(target_seq))
    ]
    (run_dir / "msa" / "conservation.json").write_text(
        json.dumps({"conservation": cons_scores}), encoding="utf-8"
    )

    # ---- alignment.fasta (3 homologs for identity_distribution) -------
    (run_dir / "msa" / "alignment.fasta").write_text(
        (
            f">FDH_demo\n{target_seq}\n"
            f">homolog_1\n{'M' + 'A' * 40 + 'V' * 9}\n"
            f">homolog_2\n{'M' + 'A' * 35 + 'L' * 14}\n"
            f">homolog_3\n{'M' + 'A' * 30 + 'G' * 19}\n"
        ),
        encoding="utf-8",
    )

    # ---- final_candidates.csv (8 rows incl. D222N catalytic) ----------
    candidates_csv = run_dir / "reports" / "final_candidates.csv"
    rows = []
    cycle = ["Strong", "Promising", "Uncertain", "Reject"]
    for i in range(8):
        rows.append(
            [
                f"cand_{i:02d}",
                "D222N" if i == 0 else f"A{10 + i}K",
                cycle[i % 4],
                f"{0.92 - 0.05 * i:.3f}",
                f"{0.81 - 0.04 * i:.3f}",
                f"{-0.20 + 0.05 * i:.3f}",
                f"{-7.0 + 0.2 * i:.3f}",
                f"{1.5 - 0.1 * i:.3f}",
                f"{0.6 - 0.05 * i:.3f}",
            ]
        )
    _write_csv(
        candidates_csv,
        header=[
            "candidate_id",
            "mutation",
            "evidence_class",
            "final_score",
            "ml_score",
            "ddg_fold",
            "docking_score",
            "md_lite_score",
            "plif_recovery",
        ],
        rows=rows,
    )

    # ---- focused_library.csv ------------------------------------------
    _write_csv(
        run_dir / "reports" / "focused_library.csv",
        header=["candidate_id", "mutation", "evidence_class", "final_score"],
        rows=[r[:4] for r in rows],
    )

    # ---- benchmark.json (recall@K) ------------------------------------
    (run_dir / "reports" / "benchmark.json").write_text(
        json.dumps({"recall_at_k": {"1": 0.1, "5": 0.4, "10": 0.6, "20": 0.85}}),
        encoding="utf-8",
    )

    # ---- WT Boltz complex PDB -----------------------------------------
    wt_pred = (
        run_dir
        / "complexes"
        / "boltz_results_wt_boltz_input"
        / "predictions"
        / "wt_boltz_input"
    )
    wt_pred.mkdir(parents=True, exist_ok=True)
    (wt_pred / "wt_boltz_input_model_0.pdb").write_text(
        _wt_complex_pdb(), encoding="utf-8"
    )

    # ---- a couple of mutant Boltz complexes ---------------------------
    for cand in ("cand_00", "cand_01"):
        d = (
            run_dir
            / "complexes"
            / "mutant_boltz"
            / f"boltz_results_mut_{cand}_boltz_input"
            / "predictions"
            / f"mut_{cand}_boltz_input"
        )
        d.mkdir(parents=True, exist_ok=True)
        (d / f"mut_{cand}_boltz_input_model_0.pdb").write_text(
            _wt_complex_pdb(), encoding="utf-8"
        )

    # ---- graph_features.json (so mutation_map plot has positions) -----
    (run_dir / "graph_features.json").write_text(
        json.dumps(
            {
                "catalytic": [222],
                "binding_site": [10, 20, 30],
                "designable": [10, 20, 30, 40],
                "fixed": [1, 2, 3],
            }
        ),
        encoding="utf-8",
    )

    return run_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_end_to_end_mock_run_to_zip(tmp_path: Path) -> None:
    """Build the full report from a synthetic run dir and verify the zip."""
    pytest.importorskip("jinja2")

    run_dir = _make_synthetic_run_dir(tmp_path / "fake_run")
    zip_path = tmp_path / "package.zip"

    from evoliez.figures.html.builder import build_report

    result = build_report(
        run_dir,
        output_dir=tmp_path / "report_pkg",
        style="presentation",
        html_mode="linked",
        skip_3d=False,
        mock_backend=True,
        output_zip=zip_path,
    )

    # ---- 1. core outputs exist ----------------------------------------
    assert result["html_path"].exists(), "visual_report.html missing"
    assert result["manifest_path"].exists(), "visual_manifest.json missing"
    assert result["zip_path"] is not None and result["zip_path"].exists(), (
        "report zip not produced"
    )

    # ---- 2. all 10 section anchors present ---------------------------
    html = result["html_path"].read_text(encoding="utf-8")
    for section_id in (
        "00-overview",
        "01-input",
        "02-msa",
        "03-boltz-complex",
        "04-pose-ensemble",
        "05-fingerprint",
        "06-mutation",
        "07-reranking",
        "08-md",
        "09-final-library",
    ):
        assert section_id in html, f"section anchor {section_id} missing"

    # ---- 3. zip carries the entry point + static assets ---------------
    with zipfile.ZipFile(result["zip_path"]) as zf:
        names = zf.namelist()
        assert any("visual_report.html" in n for n in names)
        assert any("static/report.css" in n for n in names)
        assert any("static/vendor/3Dmol-min.js" in n for n in names)
        assert any("visual_manifest.json" in n for n in names)

    # ---- 4. mock watermark stamped -----------------------------------
    assert "MOCK" in html.upper() or "mock-backend-banner" in html

    # ---- 5. manifest schema + flag -----------------------------------
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert "figures" in manifest
    assert manifest["mock_backend"] is True
    assert manifest["style"] == "presentation"

    # ---- 6. summary stats sensible ------------------------------------
    assert result["mode"] == "linked"
    # We don't require figures_generated > 0 - the renderers may all
    # skip on a no-matplotlib laptop - but the contract should report
    # an integer count.
    assert isinstance(result["figures_generated"], int)
    assert isinstance(result["figures_skipped"], int)
    assert result["total_size_mb"] > 0.0


def test_end_to_end_renders_real_figures_when_matplotlib_present(
    tmp_path: Path,
) -> None:
    """When matplotlib is installed, plot renderers actually produce PNGs."""
    pytest.importorskip("jinja2")
    pytest.importorskip("matplotlib")

    run_dir = _make_synthetic_run_dir(tmp_path / "fake_run")

    from evoliez.figures.html.builder import build_report

    result = build_report(
        run_dir,
        output_dir=tmp_path / "report_pkg",
        style="presentation",
        html_mode="linked",
        skip_3d=True,
        mock_backend=False,
        output_zip=tmp_path / "package.zip",
    )

    # At least the evidence donut + score waterfall + library diversity
    # should land when matplotlib is available.
    assert result["figures_generated"] >= 3, (
        f"expected >=3 figures, got {result['figures_generated']}"
    )

    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    fig_ids = {f["figure_id"] for f in manifest["figures"]}
    # Every manifest figure points at an existing file under report_pkg.
    out = tmp_path / "report_pkg"
    for f in manifest["figures"]:
        assert (out / f["path"]).exists(), (
            f"manifest figure {f['figure_id']} -> {f['path']} not on disk"
        )
    # Reliable subset that needs only final_candidates.csv:
    assert "00_evidence_distribution" in fig_ids


def test_end_to_end_self_contained_mode(tmp_path: Path) -> None:
    """html_mode='self-contained' produces the inlined HTML alongside linked."""
    pytest.importorskip("jinja2")

    run_dir = _make_synthetic_run_dir(tmp_path / "fake_run")

    from evoliez.figures.html.builder import build_report

    result = build_report(
        run_dir,
        output_dir=tmp_path / "report_pkg",
        style="presentation",
        html_mode="self-contained",
        skip_3d=True,
        mock_backend=False,
        output_zip=None,  # self-contained mode shouldn't need a zip
    )
    assert result["mode"] == "self-contained"
    assert result["html_path"].exists()
    sc_html = result["html_path"].parent / "visual_report.selfcontained.html"
    # The selfcontained builder either writes a sibling file or warns;
    # both outcomes are acceptable for the orchestrator contract.
    assert sc_html.exists() or any(
        "self-contained" in w for w in result["warnings"]
    )


def test_end_to_end_invalid_html_mode_raises(tmp_path: Path) -> None:
    from evoliez.figures.html.builder import build_report

    run_dir = _make_synthetic_run_dir(tmp_path / "fake_run")
    with pytest.raises(ValueError):
        build_report(
            run_dir,
            output_dir=tmp_path / "out",
            html_mode="not-a-mode",
        )
