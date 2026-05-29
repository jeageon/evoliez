"""Regression tests for the ultra-review P0/P1 figures-layer fixes.

Each test pins one verified bug so a future refactor can't silently
re-break the report's data wiring:

1. ``score_waterfall`` must read the production ``stability_ddg`` column
   (the CSV header) -- not only the legacy ``ddg_fold`` alias.
2. ``discovery`` must find s06b representatives under
   ``structures/representatives`` (the path s06b actually writes).
3. ``pose_ensemble`` must return ``None`` when fewer than 2 distinct
   model files exist (a mock / single-sample run is not an "ensemble").
4. ``pose_ensemble`` must find poses under the
   ``complexes/boltz/...`` layout s04 writes.
5. ``build_report`` must treat a run whose ``_state.json`` says
   ``backend == "mock"`` as mock even if the caller passed
   ``mock_backend=False`` -- a mock run is never stampable as real.

Heavy deps (matplotlib / jinja2 / pillow / rdkit) are gated with
``pytest.importorskip`` so the suite stays green on a barebones box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

# Skip the whole module if the figures foundation hasn't landed.
pytest.importorskip("evoliez.figures.types")

from evoliez.figures.types import ReportArtifacts  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _wt_complex_pdb() -> str:
    """A tiny PDB the 3Dmol viewer + discovery accept (protein + ligand)."""
    return (
        "REMARK   1 EvoLiEZ synthetic WT complex\n"
        "ATOM      1  N   ALA A   1      11.104  13.207  10.005  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      12.560  13.207  10.005  1.00 20.00           C\n"
        "ATOM      3  C   ALA A   1      13.207  14.503  10.005  1.00 20.00           C\n"
        "HETATM    4  C1  NDP B   2      14.000  14.000  10.000  1.00 30.00           C\n"
        "END\n"
    )


def _empty_artifacts(run_dir: Path) -> ReportArtifacts:
    run_dir.mkdir(parents=True, exist_ok=True)
    return ReportArtifacts(run_dir=run_dir)


# ===========================================================================
# Bug 1 - score_waterfall reads the production ``stability_ddg`` column
# ===========================================================================


def test_waterfall_reads_stability_ddg_column(tmp_path: Path) -> None:
    """A CSV that uses the production ``stability_ddg`` header must still
    surface a ddG component in the waterfall (not be silently dropped)."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import score_waterfall

    csv_path = tmp_path / "final_candidates.csv"
    # Production header: ``stability_ddg`` (NOT ``ddg_fold``).
    csv_path.write_text(
        "candidate_id,evidence_class,final_score,ml_score,stability_ddg,"
        "docking_score,md_lite_score,plif_recovery\n"
        "cand_00,Strong,0.90,0.80,-0.25,-7.0,1.5,0.60\n"
        "cand_01,Promising,0.80,0.70,0.30,-6.8,1.4,0.55\n",
        encoding="utf-8",
    )

    art = _empty_artifacts(tmp_path / "run")
    art.final_candidates_csv = csv_path
    out = tmp_path / "waterfall.png"
    spec = score_waterfall.render(art, out, top_k=5)

    assert spec is not None, "waterfall returned None for a valid CSV"
    # The canonical ddG component must appear among the plotted components.
    assert "stability_ddg" in spec.params["components"], (
        f"stability_ddg dropped; components={spec.params['components']}"
    )
    # And the legacy alias must NOT leak through as a separate component.
    assert "ddg_fold" not in spec.params["components"]
    assert out.exists() and out.stat().st_size > 0


def test_waterfall_legacy_ddg_fold_still_maps(tmp_path: Path) -> None:
    """Old CSVs that wrote ``ddg_fold`` must still plot, normalized onto
    the canonical ``stability_ddg`` component (backward compatibility)."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import score_waterfall

    csv_path = tmp_path / "final_candidates.csv"
    csv_path.write_text(
        "candidate_id,evidence_class,final_score,ml_score,ddg_fold,"
        "docking_score,md_lite_score,plif_recovery\n"
        "cand_00,Strong,0.90,0.80,-0.25,-7.0,1.5,0.60\n"
        "cand_01,Promising,0.80,0.70,0.30,-6.8,1.4,0.55\n",
        encoding="utf-8",
    )

    art = _empty_artifacts(tmp_path / "run")
    art.final_candidates_csv = csv_path
    out = tmp_path / "waterfall.png"
    spec = score_waterfall.render(art, out, top_k=5)

    assert spec is not None
    # Legacy column normalizes to the canonical name.
    assert "stability_ddg" in spec.params["components"]
    assert "ddg_fold" not in spec.params["components"]


def test_waterfall_row_breakdown_normalizes_alias() -> None:
    """Unit-level: a row carrying ``ddg_fold`` is read as ``stability_ddg``."""
    from evoliez.figures.plots.score_waterfall import _row_breakdown

    br = _row_breakdown({"ml_score": "0.5", "ddg_fold": "-0.3"})
    assert "stability_ddg" in br
    assert br["stability_ddg"] == pytest.approx(-0.3)
    assert "ddg_fold" not in br


# ===========================================================================
# Bug 3 - discovery finds s06b representatives under structures/
# ===========================================================================


def test_discovery_finds_structures_representatives(tmp_path: Path) -> None:
    """s06b writes ``structures/representatives/<hom>/`` -- discovery must
    catalog those dirs, not only the legacy ``complexes/representatives``."""
    from evoliez.figures.discovery import discover

    run_dir = tmp_path / "run"
    reps = run_dir / "structures" / "representatives"
    (reps / "hom_000").mkdir(parents=True, exist_ok=True)
    (reps / "hom_001").mkdir(parents=True, exist_ok=True)
    # A stray file at the representatives root must be ignored (dirs only).
    (reps / "notes.txt").write_text("x", encoding="utf-8")

    arts = discover(run_dir)
    names = sorted(p.name for p in arts.homolog_complex_dirs)
    assert names == ["hom_000", "hom_001"], names


def test_discovery_legacy_complexes_representatives(tmp_path: Path) -> None:
    """Old layout (``complexes/representatives``) is still picked up when
    ``structures/representatives`` is absent."""
    from evoliez.figures.discovery import discover

    run_dir = tmp_path / "run"
    reps = run_dir / "complexes" / "representatives"
    (reps / "hom_000").mkdir(parents=True, exist_ok=True)

    arts = discover(run_dir)
    names = sorted(p.name for p in arts.homolog_complex_dirs)
    assert names == ["hom_000"], names


def test_discovery_structures_preferred_over_complexes(tmp_path: Path) -> None:
    """When both exist, the canonical ``structures/`` location wins."""
    from evoliez.figures.discovery import discover

    run_dir = tmp_path / "run"
    (run_dir / "structures" / "representatives" / "from_structures").mkdir(
        parents=True, exist_ok=True
    )
    (run_dir / "complexes" / "representatives" / "from_complexes").mkdir(
        parents=True, exist_ok=True
    )

    arts = discover(run_dir)
    names = sorted(p.name for p in arts.homolog_complex_dirs)
    assert names == ["from_structures"], names


# ===========================================================================
# Bug 4 - pose_ensemble path discovery + ensemble honesty guard
# ===========================================================================


def _make_wt_pose_dir(run_dir: Path, *, under_boltz: bool, model_indices: List[int]) -> Path:
    """Write WT Boltz pose PDBs under either layout.

    ``under_boltz=True`` mimics s04 (``complexes/boltz/...``);
    ``under_boltz=False`` mimics the legacy direct ``complexes/...``.
    """
    base = run_dir / "complexes"
    if under_boltz:
        base = base / "boltz"
    pred = base / "boltz_results_wt_boltz_input" / "predictions" / "wt_boltz_input"
    pred.mkdir(parents=True, exist_ok=True)
    for idx in model_indices:
        (pred / f"wt_boltz_input_model_{idx}.pdb").write_text(
            _wt_complex_pdb(), encoding="utf-8"
        )
    return pred


def test_pose_ensemble_finds_poses_under_boltz_layout(tmp_path: Path) -> None:
    """``_find_pose_pdbs`` must walk the complexes/ tree and locate poses
    written under the production ``complexes/boltz/...`` layout."""
    from evoliez.figures.three_d.pose_ensemble import _find_pose_pdbs

    run_dir = tmp_path / "run"
    _make_wt_pose_dir(run_dir, under_boltz=True, model_indices=[0, 1, 2])

    art = _empty_artifacts(run_dir)
    poses = _find_pose_pdbs(art)
    assert len(poses) == 3, [p.name for p in poses]
    # model_0 sorts first (best confidence by Boltz convention).
    assert poses[0].name.endswith("_model_0.pdb")
    # And they really came from the boltz/ subtree.
    assert all("boltz" in str(p.parent) for p in poses)


def test_pose_ensemble_finds_poses_legacy_layout(tmp_path: Path) -> None:
    """Legacy direct ``complexes/boltz_results_wt_boltz_input`` still works."""
    from evoliez.figures.three_d.pose_ensemble import _find_pose_pdbs

    run_dir = tmp_path / "run"
    _make_wt_pose_dir(run_dir, under_boltz=False, model_indices=[0, 1])

    art = _empty_artifacts(run_dir)
    poses = _find_pose_pdbs(art)
    assert len(poses) == 2, [p.name for p in poses]


def test_pose_ensemble_returns_none_with_single_model(tmp_path: Path) -> None:
    """A mock / single-sample run (one model_0) is not an ensemble: render
    must return None so the builder's 3Dmol single-pose fallback takes over."""
    from evoliez.figures.three_d import pose_ensemble

    run_dir = tmp_path / "run"
    _make_wt_pose_dir(run_dir, under_boltz=True, model_indices=[0])

    art = _empty_artifacts(run_dir)
    # Give it a WT complex so the earlier guard passes and we reach the
    # distinct-model check.
    art.wt_complex_pdb = next(
        (run_dir / "complexes").rglob("*_model_0.pdb")
    )

    # Pretend PyMOL is available so we exercise the ensemble guard rather
    # than short-circuiting on a missing binary.
    orig = pose_ensemble.pymol_available
    pose_ensemble.pymol_available = lambda: True  # type: ignore[assignment]
    try:
        spec = pose_ensemble.render(art, run_dir / "ens.png")
    finally:
        pose_ensemble.pymol_available = orig  # type: ignore[assignment]

    assert spec is None, "single-model pose set must not render as an ensemble"


def test_pose_ensemble_distinct_model_count(tmp_path: Path) -> None:
    """Two copies of model_0 count as ONE distinct model, not two."""
    from evoliez.figures.three_d.pose_ensemble import _distinct_model_count

    a = tmp_path / "x_model_0.pdb"
    b = tmp_path / "sub" / "y_model_0.pdb"
    b.parent.mkdir(parents=True, exist_ok=True)
    for p in (a, b):
        p.write_text("x", encoding="utf-8")
    assert _distinct_model_count([a, b]) == 1

    c = tmp_path / "z_model_1.pdb"
    c.write_text("x", encoding="utf-8")
    assert _distinct_model_count([a, b, c]) == 2


# ===========================================================================
# Bug 5 - mock watermark forced from run-state even if caller said False
# ===========================================================================


def _minimal_run_for_build(run_dir: Path, *, backend: str) -> Path:
    """Smallest tree ``build_report`` accepts, with a chosen backend tag."""
    run_dir = Path(run_dir)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "msa").mkdir(parents=True, exist_ok=True)
    (run_dir / "complexes").mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs" / "target.fasta").write_text(
        ">target\nMKLKVLGAGAGAGAGAGAGA\n", encoding="utf-8"
    )
    (run_dir / "_state.json").write_text(
        json.dumps(
            {
                "target": "demo",
                "started": "2026-05-21T10:00:00Z",
                "finished": "2026-05-21T10:05:00Z",
                "backend": backend,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_mock_from_state_overrides_caller_false(tmp_path: Path) -> None:
    """A run whose _state.json says backend=mock must be treated as mock
    even when the caller passes mock_backend=False."""
    pytest.importorskip("jinja2")
    from evoliez.figures.html.builder import build_report

    run_dir = _minimal_run_for_build(tmp_path / "run", backend="mock")
    result = build_report(
        run_dir,
        output_dir=tmp_path / "out",
        style="presentation",
        html_mode="linked",
        skip_3d=True,
        mock_backend=False,  # caller LIES (or just didn't know)
        output_zip=None,
    )

    html = result["html_path"].read_text(encoding="utf-8")
    assert ("mock-backend-banner" in html) or ("MOCK" in html.upper()), (
        "mock run not watermarked despite _state.json backend=mock"
    )
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["mock_backend"] is True


def test_mock_from_state_homolog_source(tmp_path: Path) -> None:
    """meta.homolog_source == 'mock' also forces the mock treatment."""
    pytest.importorskip("jinja2")
    from evoliez.figures.html.builder import build_report

    run_dir = tmp_path / "run"
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "msa").mkdir(parents=True, exist_ok=True)
    (run_dir / "complexes").mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs" / "target.fasta").write_text(
        ">target\nMKLK\n", encoding="utf-8"
    )
    (run_dir / "_state.json").write_text(
        json.dumps({"target": "demo", "backend": "real",
                    "meta": {"homolog_source": "mock"}}),
        encoding="utf-8",
    )

    result = build_report(
        run_dir,
        output_dir=tmp_path / "out",
        style="presentation",
        html_mode="linked",
        skip_3d=True,
        mock_backend=False,
        output_zip=None,
    )
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["mock_backend"] is True


def test_real_run_stays_real(tmp_path: Path) -> None:
    """A genuine real run (backend=real, no mock meta) must NOT be forced
    to mock -- the fix only ADDS the mock signal, never invents one."""
    pytest.importorskip("jinja2")
    from evoliez.figures.html.builder import build_report

    run_dir = _minimal_run_for_build(tmp_path / "run", backend="real")
    result = build_report(
        run_dir,
        output_dir=tmp_path / "out",
        style="presentation",
        html_mode="linked",
        skip_3d=True,
        mock_backend=False,
        output_zip=None,
    )
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    assert manifest["mock_backend"] is False


def test_run_is_mock_defensive_on_corrupt_state(tmp_path: Path) -> None:
    """A corrupt / missing _state.json must not crash and must not add the
    mock signal (defensive: unknown provenance != mock)."""
    from evoliez.figures.html.builder import _run_is_mock

    # Corrupt JSON.
    bad = tmp_path / "bad_state.json"
    bad.write_text("{not json", encoding="utf-8")
    art = ReportArtifacts(run_dir=tmp_path)
    art.state_json = bad
    assert _run_is_mock(art) is False

    # Missing file.
    art.state_json = tmp_path / "does_not_exist.json"
    assert _run_is_mock(art) is False

    # No state_json at all.
    art.state_json = None
    assert _run_is_mock(art) is False
