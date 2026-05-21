"""Unit tests for ``evoliez.figures.html.builder``.

These cover the orchestration layer in isolation:

* defensive imports work when sibling packages are partially broken
* ``build_report`` can run end-to-end against a minimal synthetic
  ``run_dir`` and produces ``visual_report.html`` + ``visual_manifest.json``
* the manifest lists the figures actually generated on disk
* every section ID expected by ``components/toc.html`` appears in the
  rendered HTML

Heavy dependencies (jinja2 / matplotlib) are gated with
``pytest.importorskip`` so the suite stays green on the laptop dev box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


pytest.importorskip("evoliez.figures.html")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_run_dir(run_dir: Path) -> Path:
    """Smallest possible tree that ``discover`` can chew on without crashing."""
    run_dir = Path(run_dir)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "msa").mkdir(parents=True, exist_ok=True)
    (run_dir / "complexes").mkdir(parents=True, exist_ok=True)

    # A tiny target FASTA so section 01 has something to mention.
    (run_dir / "inputs" / "target.fasta").write_text(
        ">target\nMKLKVLGAGAGAGAGAGAGAGAGA\n", encoding="utf-8"
    )
    (run_dir / "inputs" / "ligand.smi").write_text(
        "CCO ethanol\n", encoding="utf-8"
    )

    # A minimal _state.json so manifest fingerprint is stable.
    (run_dir / "_state.json").write_text(
        json.dumps(
            {
                "target": "demo_target",
                "started": "2026-05-21T10:00:00Z",
                "finished": "2026-05-21T10:05:00Z",
                "backend": "mock",
                "catalytic_targets": ["D222"],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _make_candidates_csv(run_dir: Path, *, rows: int = 5) -> Path:
    cols = [
        "candidate_id",
        "mutation",
        "evidence_class",
        "final_score",
        "ml_score",
        "ddg_fold",
        "docking_score",
        "md_lite_score",
        "plif_recovery",
    ]
    evidence_cycle = ["Strong", "Promising", "Uncertain", "Reject"]
    lines = [",".join(cols)]
    for i in range(rows):
        ec = evidence_cycle[i % len(evidence_cycle)]
        mut = "D222N" if i == 0 else f"A{10 + i}K"
        lines.append(
            ",".join(
                [
                    f"cand_{i:02d}",
                    mut,
                    ec,
                    f"{0.9 - 0.05 * i:.3f}",
                    f"{0.8 - 0.04 * i:.3f}",
                    f"{-0.2 + 0.05 * i:.3f}",
                    f"{-7.0 + 0.2 * i:.3f}",
                    f"{1.5 - 0.1 * i:.3f}",
                    f"{0.6 - 0.05 * i:.3f}",
                ]
            )
        )
    path = run_dir / "reports" / "final_candidates.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_builder_imports() -> None:
    """The builder module always imports - even without sibling agents."""
    from evoliez.figures.html import builder

    assert callable(builder.build_report)
    # Defensive fallbacks are present.
    assert callable(builder.build_zip)
    # 3D helpers are at least defined (no-op stubs if sibling absent).
    assert callable(builder.pymol_available)
    assert builder.pymol_available() in (True, False)


def test_build_report_minimal_run_dir(tmp_path: Path) -> None:
    """Empty-ish run_dir produces a valid report skeleton (no exceptions)."""
    pytest.importorskip("jinja2")

    run_dir = _make_minimal_run_dir(tmp_path / "run")

    from evoliez.figures.html.builder import build_report

    result = build_report(
        run_dir,
        output_dir=tmp_path / "out",
        style="presentation",
        html_mode="linked",
        skip_3d=True,
        mock_backend=True,
        output_zip=None,  # skip zip for this test - we cover it elsewhere
    )

    html_path = result["html_path"]
    manifest_path = result["manifest_path"]
    assert html_path.exists(), "visual_report.html not written"
    assert manifest_path.exists(), "visual_manifest.json not written"
    assert result["mode"] == "linked"
    assert result["total_size_mb"] >= 0.0


def test_build_report_html_has_all_sections(tmp_path: Path) -> None:
    """visual_report.html includes every section id the TOC references."""
    pytest.importorskip("jinja2")

    run_dir = _make_minimal_run_dir(tmp_path / "run")

    from evoliez.figures.html.builder import build_report

    result = build_report(
        run_dir,
        output_dir=tmp_path / "out",
        style="presentation",
        html_mode="linked",
        skip_3d=True,
        mock_backend=False,
        output_zip=None,
    )
    html = result["html_path"].read_text(encoding="utf-8")

    for sec_id in (
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
        assert sec_id in html, f"missing section id {sec_id}"


def test_build_report_with_candidates_emits_figures(tmp_path: Path) -> None:
    """Adding final_candidates.csv unlocks the evidence-distribution figure."""
    pytest.importorskip("jinja2")
    pytest.importorskip("matplotlib")

    run_dir = _make_minimal_run_dir(tmp_path / "run")
    _make_candidates_csv(run_dir, rows=8)

    from evoliez.figures.html.builder import build_report

    result = build_report(
        run_dir,
        output_dir=tmp_path / "out",
        style="presentation",
        html_mode="linked",
        skip_3d=True,
        mock_backend=False,
        output_zip=None,
    )

    # At least the evidence donut + score waterfall + library diversity
    # should have been generated (mutation_map gracefully skips when the
    # target sequence + graph features are absent).
    assert result["figures_generated"] >= 1
    manifest = json.loads(result["manifest_path"].read_text())
    assert manifest["schema_version"] == "1.0"
    figs: list[Dict[str, Any]] = manifest["figures"]
    fig_ids = {f["figure_id"] for f in figs}
    # Every figure listed must point at an existing file under output_dir.
    out_dir = tmp_path / "out"
    for f in figs:
        assert (out_dir / f["path"]).exists(), (
            f"manifest references missing figure {f['path']}"
        )
    # The evidence-class donut is the most reliably-generated figure
    # from final_candidates.csv alone.
    assert "00_evidence_distribution" in fig_ids


def test_build_report_mock_watermark(tmp_path: Path) -> None:
    """mock_backend=True stamps the report with a visible MOCK banner."""
    pytest.importorskip("jinja2")

    run_dir = _make_minimal_run_dir(tmp_path / "run")

    from evoliez.figures.html.builder import build_report

    result = build_report(
        run_dir,
        output_dir=tmp_path / "out",
        style="presentation",
        html_mode="linked",
        skip_3d=True,
        mock_backend=True,
        output_zip=None,
    )
    html = result["html_path"].read_text(encoding="utf-8")
    # The base template's own banner uses ``mock-backend-banner`` and the
    # fallback watermark uses the same class; either is acceptable.
    assert ("mock-backend-banner" in html) or ("MOCK" in html.upper())
    # Manifest reflects the flag.
    manifest = json.loads(result["manifest_path"].read_text())
    assert manifest["mock_backend"] is True


def test_build_report_rejects_bad_html_mode(tmp_path: Path) -> None:
    from evoliez.figures.html.builder import build_report

    run_dir = _make_minimal_run_dir(tmp_path / "run")
    with pytest.raises(ValueError):
        build_report(
            run_dir,
            output_dir=tmp_path / "out",
            html_mode="bogus",
        )


def test_build_report_copies_static_assets(tmp_path: Path) -> None:
    """The package's CSS/JS/vendor JS land in output_dir/static/."""
    pytest.importorskip("jinja2")

    run_dir = _make_minimal_run_dir(tmp_path / "run")
    out = tmp_path / "out"

    from evoliez.figures.html.builder import build_report

    build_report(
        run_dir,
        output_dir=out,
        style="presentation",
        html_mode="linked",
        skip_3d=True,
        mock_backend=False,
        output_zip=None,
    )
    assert (out / "static" / "report.css").is_file()
    assert (out / "static" / "report.js").is_file()
    assert (out / "static" / "vendor" / "3Dmol-min.js").is_file()
