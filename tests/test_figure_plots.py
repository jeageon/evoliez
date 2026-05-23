"""Synthetic-fixture tests for the 2D plot renderers.

All tests use ``pytest.importorskip("matplotlib")`` (and rdkit / sklearn
where applicable) so the suite passes on a barebones laptop where the
``[figures]`` extra hasn't been pulled in.  Fixtures are written to
``tmp_path`` - we never touch the real production output.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

# Skip the entire module if the foundation layer hasn't landed yet.
pytest.importorskip("evoliez.figures.style")

from evoliez.figures.types import FigureSpec, ReportArtifacts  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _png_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def _make_candidates_csv(path: Path, rows: int = 5) -> Path:
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
        lines.append(
            ",".join(
                [
                    f"cand_{i:02d}",
                    f"A{10+i}K",
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
    path.write_text("\n".join(lines) + "\n")
    return path


def _empty_artifacts(run_dir: Path) -> ReportArtifacts:
    run_dir.mkdir(parents=True, exist_ok=True)
    return ReportArtifacts(run_dir=run_dir)


# ---------------------------------------------------------------------------
# ligand_2d
# ---------------------------------------------------------------------------


def test_ligand_2d_renders_simple_smiles(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("rdkit.Chem.Draw")
    from evoliez.figures.plots import ligand_2d

    art = _empty_artifacts(tmp_path / "run")
    art.ligand_smiles = "CCO"
    out = tmp_path / "ligand.png"
    spec = ligand_2d.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "01_ligand_2d"
    assert spec.section == "input"
    assert _png_ok(out)


def test_ligand_2d_returns_none_without_smiles(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import ligand_2d

    art = _empty_artifacts(tmp_path / "run")
    out = tmp_path / "ligand.png"
    assert ligand_2d.render(art, out) is None
    assert not out.exists()


# ---------------------------------------------------------------------------
# conservation
# ---------------------------------------------------------------------------


def test_conservation_heatmap(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import conservation

    cons = tmp_path / "conservation.json"
    cons.write_text(json.dumps({"conservation": [i / 20 for i in range(20)]}))

    art = _empty_artifacts(tmp_path / "run")
    art.conservation_json = cons
    out = tmp_path / "cons.png"
    spec = conservation.render(art, out, catalytic=[3, 12])

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "02_conservation_heatmap"
    assert spec.section == "msa"
    assert _png_ok(out)


def test_conservation_returns_none_when_missing(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import conservation

    art = _empty_artifacts(tmp_path / "run")
    out = tmp_path / "cons.png"
    assert conservation.render(art, out) is None


# ---------------------------------------------------------------------------
# identity_distribution
# ---------------------------------------------------------------------------


def test_identity_distribution_from_fasta(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import identity_distribution

    fasta = tmp_path / "aln.fasta"
    # WT + four homologs at varying identities.
    fasta.write_text(
        ">wt\nMAAAAAAAAA\n"
        ">h1\nMAAAAAAAAA\n"
        ">h2\nMAAAAAGGGG\n"
        ">h3\nMAAAGGGGGG\n"
        ">h4\nMGGGGGGGGG\n"
    )
    art = _empty_artifacts(tmp_path / "run")
    art.alignment_fasta = fasta
    out = tmp_path / "id.png"
    spec = identity_distribution.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "02_identity_distribution"
    assert _png_ok(out)


# ---------------------------------------------------------------------------
# score_waterfall
# ---------------------------------------------------------------------------


def test_score_waterfall_top5(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import score_waterfall

    csv_path = _make_candidates_csv(tmp_path / "final_candidates.csv", rows=5)
    art = _empty_artifacts(tmp_path / "run")
    art.final_candidates_csv = csv_path
    out = tmp_path / "waterfall.png"
    spec = score_waterfall.render(art, out, top_k=5)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "07_score_waterfall"
    assert spec.section == "reranking"
    assert _png_ok(out)


def test_score_waterfall_missing_csv(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import score_waterfall

    art = _empty_artifacts(tmp_path / "run")
    out = tmp_path / "waterfall.png"
    assert score_waterfall.render(art, out) is None


# ---------------------------------------------------------------------------
# benchmark_recovery
# ---------------------------------------------------------------------------


def test_benchmark_recovery_valid(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import benchmark_recovery

    bench_json = tmp_path / "benchmark.json"
    bench_csv = tmp_path / "benchmark.csv"
    bench_csv.write_text("mutation,activity\nA1K,1\nA2L,0\n")
    bench_json.write_text(
        json.dumps(
            {
                "valid": True,
                "n_total": 100,
                "n_positives": 10,
                "recall_at_k": {"1": 0.1, "5": 0.4, "10": 0.6, "30": 0.85, "100": 1.0},
            }
        )
    )

    art = _empty_artifacts(tmp_path / "run")
    art.benchmark_json = bench_json
    art.benchmark_csv = bench_csv
    out = tmp_path / "bench.png"
    spec = benchmark_recovery.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "00_benchmark_recovery"
    assert spec.section == "overview"
    assert _png_ok(out)


def test_benchmark_recovery_invalid_returns_none(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import benchmark_recovery

    bench_json = tmp_path / "benchmark.json"
    bench_csv = tmp_path / "benchmark.csv"
    bench_csv.write_text("mutation,activity\n")
    bench_json.write_text(json.dumps({"valid": False, "recall_at_k": {"1": 0.1}}))

    art = _empty_artifacts(tmp_path / "run")
    art.benchmark_json = bench_json
    art.benchmark_csv = bench_csv
    out = tmp_path / "bench.png"
    assert benchmark_recovery.render(art, out) is None
    assert not out.exists()


def test_benchmark_recovery_missing_inputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import benchmark_recovery

    art = _empty_artifacts(tmp_path / "run")
    out = tmp_path / "bench.png"
    assert benchmark_recovery.render(art, out) is None


# ---------------------------------------------------------------------------
# md_rmsd
# ---------------------------------------------------------------------------


def test_md_rmsd_with_series(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import md_rmsd

    run = tmp_path / "run"
    run.mkdir()
    md_a = run / "md" / "cand_00"
    md_a.mkdir(parents=True)
    (md_a / "analysis.json").write_text(
        json.dumps(
            {
                "time_ps": [0, 10, 20, 30, 40],
                "ligand_rmsd_series": [0.5, 1.0, 1.5, 1.8, 2.0],
                "pocket_rmsd_series": [0.4, 0.6, 0.9, 1.1, 1.3],
            }
        )
    )
    md_b = run / "md" / "cand_01"
    md_b.mkdir(parents=True)
    (md_b / "analysis.json").write_text(
        json.dumps(
            {
                "time_ps": [0, 10, 20, 30, 40],
                "ligand_rmsd_series": [0.6, 1.2, 2.1, 2.9, 3.2],
                "pocket_rmsd_series": [0.5, 0.8, 1.0, 1.4, 1.6],
            }
        )
    )

    art = _empty_artifacts(run)
    art.md_dirs = {"cand_00": md_a, "cand_01": md_b}
    csv_path = _make_candidates_csv(run / "final_candidates.csv", rows=2)
    art.final_candidates_csv = csv_path

    out = tmp_path / "rmsd.png"
    spec = md_rmsd.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "08_md_rmsd_timeseries"
    assert spec.section == "md"
    assert _png_ok(out)


def test_md_rmsd_summary_only_renders_bar_fallback(tmp_path: Path) -> None:
    """Production schema (md/analysis.json) carries only mean+final RMSD,
    not the full time series. Renderer must fall back to a horizontal
    bar chart rather than skipping the whole section. Without this,
    s10_md's actual output produces an empty section 8."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import md_rmsd

    run = tmp_path / "run"
    run.mkdir()
    md_a = run / "md" / "cand_00"
    md_a.mkdir(parents=True)
    # Production-shape analysis.json: only mean + final, no series.
    (md_a / "analysis.json").write_text(json.dumps({
        "ligand_rmsd_mean": 1.2, "ligand_rmsd_final": 2.1,
        "pocket_rmsd_mean": 0.8, "pocket_rmsd_final": 1.0,
        "passed": True, "md_lite_score": 0.5,
    }))
    art = _empty_artifacts(run)
    art.md_dirs = {"cand_00": md_a}
    out = tmp_path / "rmsd.png"
    spec = md_rmsd.render(art, out)
    assert spec is not None
    assert spec.figure_id == "08_md_rmsd_timeseries"
    assert spec.params.get("mode") == "summary_bars"
    assert _png_ok(out)


def test_md_rmsd_returns_none_when_no_md_dirs(tmp_path: Path) -> None:
    """Sanity floor: with NO md_dirs at all, render must still skip."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import md_rmsd

    art = _empty_artifacts(tmp_path / "run")
    art.md_dirs = {}
    assert md_rmsd.render(art, tmp_path / "out.png") is None


# ---------------------------------------------------------------------------
# evidence_distribution
# ---------------------------------------------------------------------------


def test_evidence_distribution_donut(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import evidence_distribution

    csv_path = _make_candidates_csv(tmp_path / "final_candidates.csv", rows=4)
    art = _empty_artifacts(tmp_path / "run")
    art.final_candidates_csv = csv_path
    out = tmp_path / "donut.png"
    spec = evidence_distribution.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "00_evidence_distribution"
    assert spec.section == "overview"
    assert _png_ok(out)
    assert spec.params["total"] == 4


def test_evidence_distribution_missing_csv(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import evidence_distribution

    art = _empty_artifacts(tmp_path / "run")
    out = tmp_path / "donut.png"
    assert evidence_distribution.render(art, out) is None


# ---------------------------------------------------------------------------
# library_diversity
# ---------------------------------------------------------------------------


def test_library_diversity_scatter(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import library_diversity

    csv_path = _make_candidates_csv(tmp_path / "focused_library.csv", rows=8)
    art = _empty_artifacts(tmp_path / "run")
    art.focused_library_csv = csv_path
    out = tmp_path / "lib.png"
    spec = library_diversity.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "09_library_diversity"
    assert spec.section == "final_library"
    assert _png_ok(out)


# ---------------------------------------------------------------------------
# mutation_map
# ---------------------------------------------------------------------------


def test_mutation_map_track(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import mutation_map

    run = tmp_path / "run"
    run.mkdir()
    fasta = run / "target.fasta"
    fasta.write_text(">wt\n" + ("A" * 60) + "\n")
    (run / "graph_features.json").write_text(
        json.dumps(
            {
                "catalytic": [10, 25],
                "binding_site": [12, 14, 18, 22],
                "designable": [5, 11, 19, 28, 35, 40],
                "fixed": [1, 2, 3, 60],
            }
        )
    )

    art = _empty_artifacts(run)
    art.target_fasta = fasta
    out = tmp_path / "mut.png"
    spec = mutation_map.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "06_mutation_design_space"
    assert spec.section == "mutation"
    assert _png_ok(out)
    assert spec.params["sequence_length"] == 60


def test_mutation_map_missing_inputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import mutation_map

    art = _empty_artifacts(tmp_path / "run")
    out = tmp_path / "mut.png"
    assert mutation_map.render(art, out) is None
