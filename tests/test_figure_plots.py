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


def test_ligand_2d_functional_groups_match_nadp(tmp_path: Path) -> None:
    """NADP+ should light up the phosphate / pyridinium / amide / purine /
    sugar / hydroxyl groups - that's the whole point of the catalogue."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("rdkit.Chem.Draw")
    from evoliez.figures.plots import ligand_2d

    nadp = ("NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)"
            "C(OP(=O)(O)O)C3O)C(O)C2O)c1")
    art = _empty_artifacts(tmp_path / "run")
    art.ligand_smiles = nadp
    out = tmp_path / "nadp.png"
    spec = ligand_2d.render(art, out)

    assert spec is not None
    assert spec.params["name"] == "NADP+"
    fg = set(spec.params["functional_groups"])
    # All five "cofactor signature" groups must be present.
    for required in ("phosphate", "pyridinium", "amide", "purine",
                     "sugar (furanose)", "hydroxyl"):
        assert required in fg, f"missing {required!r}; got {fg}"
    # At least a non-trivial chunk of atoms got colored.
    assert spec.params["n_atoms_highlighted"] >= 25
    assert _png_ok(out)


def test_ligand_2d_catalytic_overlay_separate_from_fg(tmp_path: Path) -> None:
    """Explicit catalytic_atoms list overrides the functional-group color
    on those atoms (deep red), so the catalytic centre always reads."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("rdkit.Chem.Draw")
    from evoliez.figures.plots import ligand_2d

    art = _empty_artifacts(tmp_path / "run")
    art.ligand_smiles = "CCO"  # ethanol: O at idx 2 is hydroxyl
    out = tmp_path / "etoh.png"
    spec = ligand_2d.render(art, out, catalytic_atoms=[2])
    assert spec is not None
    assert spec.params["catalytic_atoms"] == [2]
    # Hydroxyl still detected on the catalogue scan.
    assert "hydroxyl" in spec.params["functional_groups"]
    assert _png_ok(out)


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


def _write_benchmark_csv(path: Path, rows: list) -> Path:
    """Helper: write a ``mutation,label,activity,source`` benchmark CSV."""
    lines = ["mutation,label,activity,source"]
    for mut, lab, act in rows:
        lines.append(f"{mut},{lab},{act},test-fixture")
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_candidates_with_blocked(path: Path, accepted: list, blocked: list) -> Path:
    """Helper: write a final_candidates.csv that includes ``is_blocked`` so
    benchmark_recovery's accepted-only ranking actually filters."""
    cols = [
        "rank", "candidate_id", "mutations", "evidence_class",
        "final_score", "is_blocked", "block_reason",
    ]
    lines = [",".join(cols)]
    rank = 1
    for mut, ev, score in accepted:
        lines.append(f"{rank},cand_{rank:03d},{mut},{ev},{score:.3f},0,")
        rank += 1
    for mut, ev, score in blocked:
        lines.append(f"{rank},cand_{rank:03d},{mut},{ev},{score:.3f},1,evidence_reject")
        rank += 1
    path.write_text("\n".join(lines) + "\n")
    return path


def test_benchmark_recovery_valid(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import benchmark_recovery

    # 4-panel render needs: benchmark.csv (mutation,label,activity) and
    # final_candidates.csv (with the P0a is_blocked gate column).
    bench_csv = _write_benchmark_csv(
        tmp_path / "benchmark.csv",
        rows=[
            ("D222S", "beneficial", 1.8),
            ("D222H", "beneficial", 1.5),
            ("R285E", "deleterious", 0.05),
            ("T221N", "neutral", 1.0),
        ],
    )
    final_csv = _make_candidates_with_blocked(
        tmp_path / "final_candidates.csv",
        # accepted: top picks include one of the beneficial mutations
        accepted=[
            ("D222S", "Strong", 1.20),
            ("D222H", "Strong", 1.10),
            ("X999Y", "Promising", 0.90),
            ("T221N", "Uncertain", 0.50),
        ],
        # blocked: deleterious R285E correctly gated out
        blocked=[("R285E", "Reject", 0.10)],
    )

    art = _empty_artifacts(tmp_path / "run")
    art.benchmark_csv = bench_csv
    art.final_candidates_csv = final_csv
    out = tmp_path / "bench.png"
    spec = benchmark_recovery.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "00_benchmark_recovery"
    assert spec.section == "overview"
    # Universal-tool guarantees: panel counts + recovered count derived
    # from the inputs (not hard-coded), and the figure uses both CSVs.
    assert spec.params["n_benchmark"] == 4
    assert spec.params["n_beneficial"] == 2
    assert spec.params["n_deleterious"] == 1
    assert spec.params["n_candidates"] == 5
    # D222S + D222H both in accepted ranks 1-2 → both beneficial recovered.
    assert spec.params["n_recovered"] >= 2
    assert _png_ok(out)


def test_benchmark_recovery_with_ablation_panel(tmp_path: Path) -> None:
    """Panel D fires when benchmark.json carries an ablation_study dict."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import benchmark_recovery

    bench_csv = _write_benchmark_csv(
        tmp_path / "benchmark.csv",
        rows=[("D222S", "beneficial", 1.8), ("R285E", "deleterious", 0.05)],
    )
    final_csv = _make_candidates_with_blocked(
        tmp_path / "final_candidates.csv",
        accepted=[("D222S", "Strong", 1.2), ("X1Y", "Promising", 0.9)],
        blocked=[("R285E", "Reject", 0.1)],
    )
    bench_json = tmp_path / "benchmark.json"
    bench_json.write_text(json.dumps({
        "ablation_study": {
            "full": {"auroc_beneficial": 0.82},
            "no_md": {"auroc_beneficial": 0.71},
            "no_boltz": {"auroc_beneficial": 0.55},
        }
    }))

    art = _empty_artifacts(tmp_path / "run")
    art.benchmark_csv = bench_csv
    art.final_candidates_csv = final_csv
    art.benchmark_json = bench_json
    out = tmp_path / "bench_with_ablation.png"
    spec = benchmark_recovery.render(art, out)
    assert spec is not None
    assert spec.params["ablation_panel"] is True
    assert _png_ok(out)


def test_benchmark_recovery_empty_benchmark_returns_none(tmp_path: Path) -> None:
    """Header-only benchmark CSV → no rows → skip the figure."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import benchmark_recovery

    bench_csv = tmp_path / "benchmark.csv"
    bench_csv.write_text("mutation,label,activity,source\n")
    final_csv = _make_candidates_with_blocked(
        tmp_path / "final_candidates.csv",
        accepted=[("A1K", "Strong", 1.0)],
        blocked=[],
    )

    art = _empty_artifacts(tmp_path / "run")
    art.benchmark_csv = bench_csv
    art.final_candidates_csv = final_csv
    out = tmp_path / "bench.png"
    assert benchmark_recovery.render(art, out) is None
    assert not out.exists()


def test_benchmark_recovery_missing_inputs(tmp_path: Path) -> None:
    """No artifacts at all → return None (figure is opportunistic)."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import benchmark_recovery

    art = _empty_artifacts(tmp_path / "run")
    out = tmp_path / "bench.png"
    assert benchmark_recovery.render(art, out) is None


def test_benchmark_recovery_missing_final_csv_returns_none(tmp_path: Path) -> None:
    """benchmark.csv alone is not enough — we need the pipeline output too."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import benchmark_recovery

    bench_csv = _write_benchmark_csv(
        tmp_path / "benchmark.csv",
        rows=[("D222S", "beneficial", 1.8)],
    )
    art = _empty_artifacts(tmp_path / "run")
    art.benchmark_csv = bench_csv
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
