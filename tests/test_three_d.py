"""Tests for the 3D rendering layer (``evoliez.figures.three_d``).

The PyMOL subprocess is always mocked, so this suite passes on the laptop
dev box that has no PyMOL CLI installed.  We exercise:

* ``pymol_available`` returns a bool no matter what's on PATH.
* ``make_viewer_context`` populates ``pdb_text`` for a tiny synthetic PDB,
  returns a ``missing=True`` context for missing files, and strips ANISOU
  records when the file is over the 1.5 MB trim threshold.
* All four ``render_*`` helpers return ``None`` when the WT artifact is
  missing or when PyMOL is unavailable (mocked).
* When PyMOL *is* available (subprocess mocked) and the PNG appears on
  disk, ``render_pocket_view`` returns a well-formed ``FigureSpec``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest

# Skip the entire module if the foundation isn't importable.
pytest.importorskip("evoliez.figures.style")

from evoliez.figures.types import FigureSpec, ReportArtifacts  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


_SYNTH_PDB = (
    "HEADER    TEST                                    01-JAN-26   TEST\n"
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)


def _make_pdb(path: Path, text: str = _SYNTH_PDB) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _make_candidates_csv(path: Path, mutations: List[str], evidence: List[str]) -> Path:
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["candidate_id", "mutations", "evidence_class", "final_score"])
        for i, (m, e) in enumerate(zip(mutations, evidence)):
            w.writerow([f"cand_{i:02d}", m, e, f"{0.9 - 0.05 * i:.3f}"])
    return path


def _artifacts(run_dir: Path) -> ReportArtifacts:
    run_dir.mkdir(parents=True, exist_ok=True)
    return ReportArtifacts(run_dir=run_dir)


# ---------------------------------------------------------------------------
# pymol_runner
# ---------------------------------------------------------------------------


def test_pymol_available_returns_bool() -> None:
    from evoliez.figures.three_d import pymol_available

    val = pymol_available()
    assert isinstance(val, bool)


def test_run_pymol_script_returns_tuple_when_missing() -> None:
    """With PyMOL not on PATH, run_pymol_script should report (False, msg)."""
    from evoliez.figures.three_d import pymol_runner

    with patch.object(pymol_runner, "_which_pymol", return_value=""):
        ok, msg = pymol_runner.run_pymol_script("print('hi')")
    assert ok is False
    assert "pymol" in msg.lower()


def test_run_pymol_script_invokes_subprocess() -> None:
    """When PyMOL is on PATH we should shell out via subprocess.run."""
    from evoliez.figures.three_d import pymol_runner

    class _FakeCompleted:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = "ok"
            self.stderr = ""

    with patch.object(pymol_runner, "_which_pymol", return_value="/usr/bin/pymol"):
        with patch.object(
            pymol_runner.subprocess, "run", return_value=_FakeCompleted()
        ) as run_mock:
            ok, out = pymol_runner.run_pymol_script("dummy\n")
    assert ok is True
    assert out == "ok"
    args, kwargs = run_mock.call_args
    assert args[0][0] == "/usr/bin/pymol"
    assert args[0][1] == "-cq"
    assert kwargs.get("input") == "dummy\n"


def test_run_pymol_script_reports_failure() -> None:
    from evoliez.figures.three_d import pymol_runner

    class _Bad:
        def __init__(self) -> None:
            self.returncode = 2
            self.stdout = ""
            self.stderr = "boom"

    with patch.object(pymol_runner, "_which_pymol", return_value="/usr/bin/pymol"):
        with patch.object(pymol_runner.subprocess, "run", return_value=_Bad()):
            ok, msg = pymol_runner.run_pymol_script("dummy")
    assert ok is False
    assert "boom" in msg


# ---------------------------------------------------------------------------
# pdb_inline
# ---------------------------------------------------------------------------


def test_make_viewer_context_populates_pdb_text(tmp_path: Path) -> None:
    from evoliez.figures.three_d import make_viewer_context

    pdb = _make_pdb(tmp_path / "wt.pdb")
    ctx = make_viewer_context(pdb, viewer_id="wt")
    assert ctx["missing"] is False
    assert ctx["id"] == "wt"
    assert ctx["height"] == 420
    assert "ATOM" in ctx["pdb_text"]
    # highlight_residues is JSON-serialised already so the template can
    # drop it straight into a JS array.
    assert json.loads(ctx["highlight_residues"]) == []


def test_make_viewer_context_sanitizes_viewer_id(tmp_path: Path) -> None:
    from evoliez.figures.three_d import make_viewer_context

    pdb = _make_pdb(tmp_path / "wt.pdb")
    ctx = make_viewer_context(pdb, viewer_id="wt$ complex/3D")
    assert ctx["id"] == "wt__complex_3D"


def test_make_viewer_context_highlight_residues_json(tmp_path: Path) -> None:
    from evoliez.figures.three_d import make_viewer_context

    pdb = _make_pdb(tmp_path / "wt.pdb")
    ctx = make_viewer_context(
        pdb, viewer_id="wt", highlight_residues=[12, 45, 78]
    )
    assert json.loads(ctx["highlight_residues"]) == [12, 45, 78]


def test_make_viewer_context_missing_file(tmp_path: Path) -> None:
    from evoliez.figures.three_d import make_viewer_context

    ctx = make_viewer_context(tmp_path / "nope.pdb", viewer_id="missing")
    assert ctx["missing"] is True
    assert ctx["pdb_text"] == ""
    assert ctx["id"] == "missing"


def test_make_viewer_context_strips_anisou_when_large(tmp_path: Path) -> None:
    from evoliez.figures.three_d import make_viewer_context

    # Build a > 1.5 MB PDB with lots of ANISOU lines so the trimmer is
    # forced to act.  Each ANISOU line is ~80 chars, so 25_000 lines is
    # plenty.
    anisou_block = "\n".join(
        "ANISOU    1  N   ALA A   1     1000   1000   1000      0      0      0       N"
        for _ in range(25_000)
    )
    big_pdb = (
        "HEADER    BIG TEST\n"
        + anisou_block
        + "\n"
        + "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        + "END\n"
    )
    p = _make_pdb(tmp_path / "big.pdb", big_pdb)
    assert p.stat().st_size > 1_500_000, "fixture must exceed the trim threshold"

    ctx = make_viewer_context(p, viewer_id="big")
    assert ctx["missing"] is False
    assert ctx.get("trimmed") is True
    assert "ANISOU" not in ctx["pdb_text"]
    # Coordinate record survives.
    assert "ATOM      1" in ctx["pdb_text"]


# ---------------------------------------------------------------------------
# render_pocket_view
# ---------------------------------------------------------------------------


def test_pocket_view_returns_none_without_wt(tmp_path: Path) -> None:
    from evoliez.figures.three_d import render_pocket_view

    art = _artifacts(tmp_path / "run")
    out = tmp_path / "pocket.png"
    # No wt_complex_pdb set.
    assert render_pocket_view(art, out) is None


def test_pocket_view_returns_none_without_pymol(tmp_path: Path) -> None:
    from evoliez.figures.three_d import pocket_view as mod

    art = _artifacts(tmp_path / "run")
    art.wt_complex_pdb = _make_pdb(tmp_path / "wt.pdb")
    out = tmp_path / "pocket.png"
    with patch.object(mod, "pymol_available", return_value=False):
        assert mod.render(art, out) is None


def test_pocket_view_returns_spec_when_pymol_ok(tmp_path: Path) -> None:
    """When PyMOL reports success and the PNG appears, build a FigureSpec."""
    from evoliez.figures.three_d import pocket_view as mod

    art = _artifacts(tmp_path / "run")
    art.wt_complex_pdb = _make_pdb(tmp_path / "wt.pdb")
    out = tmp_path / "pocket.png"

    def _fake_run(script: str, *, timeout: float = 120):  # noqa: ARG001
        # Simulate PyMOL writing the PNG.
        Path(out).write_bytes(b"\x89PNG\r\n\x1a\n")
        return (True, "rendered")

    with patch.object(mod, "pymol_available", return_value=True):
        with patch.object(mod, "run_pymol_script", side_effect=_fake_run):
            spec = mod.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "03_boltz_binding_pocket"
    assert spec.renderer == "pymol"
    assert spec.path == out
    assert spec.params["style"] == "presentation"


# ---------------------------------------------------------------------------
# render_pose_ensemble
# ---------------------------------------------------------------------------


def _make_pose_files(run_dir: Path, n: int = 3) -> List[Path]:
    pred_dir = (
        run_dir
        / "complexes"
        / "boltz_results_wt_boltz_input"
        / "predictions"
        / "wt_boltz_input"
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i in range(n):
        p = pred_dir / f"wt_boltz_input_model_{i}.pdb"
        p.write_text(_SYNTH_PDB)
        paths.append(p)
    return paths


def test_pose_ensemble_returns_none_without_wt(tmp_path: Path) -> None:
    from evoliez.figures.three_d import render_pose_ensemble

    art = _artifacts(tmp_path / "run")
    out = tmp_path / "ensemble.png"
    assert render_pose_ensemble(art, out) is None


def test_pose_ensemble_returns_none_without_pymol(tmp_path: Path) -> None:
    from evoliez.figures.three_d import pose_ensemble as mod

    run_dir = tmp_path / "run"
    art = _artifacts(run_dir)
    poses = _make_pose_files(run_dir, 3)
    art.wt_complex_pdb = poses[0]
    out = tmp_path / "ensemble.png"
    with patch.object(mod, "pymol_available", return_value=False):
        assert mod.render(art, out) is None


def test_pose_ensemble_returns_none_without_pose_pdbs(tmp_path: Path) -> None:
    """Even with PyMOL "available", missing pose PDBs -> None."""
    from evoliez.figures.three_d import pose_ensemble as mod

    art = _artifacts(tmp_path / "run")
    art.wt_complex_pdb = _make_pdb(tmp_path / "wt.pdb")
    out = tmp_path / "ensemble.png"
    with patch.object(mod, "pymol_available", return_value=True):
        with patch.object(mod, "run_pymol_script", return_value=(True, "")):
            assert mod.render(art, out) is None


def test_pose_ensemble_renders_when_all_present(tmp_path: Path) -> None:
    from evoliez.figures.three_d import pose_ensemble as mod

    run_dir = tmp_path / "run"
    art = _artifacts(run_dir)
    poses = _make_pose_files(run_dir, 4)
    art.wt_complex_pdb = poses[0]
    out = tmp_path / "ensemble.png"

    def _fake_run(script: str, *, timeout: float = 120):  # noqa: ARG001
        Path(out).write_bytes(b"\x89PNG\r\n\x1a\n")
        return (True, "rendered")

    with patch.object(mod, "pymol_available", return_value=True):
        with patch.object(mod, "run_pymol_script", side_effect=_fake_run):
            spec = mod.render(art, out, max_samples=3)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "04_boltz_pose_ensemble"
    assert spec.renderer == "pymol"
    # max_samples=3 caps the number of source poses included.
    assert spec.params["n_poses"] == 3


# ---------------------------------------------------------------------------
# render_mutation_overlay
# ---------------------------------------------------------------------------


def test_mutation_overlay_returns_none_without_wt(tmp_path: Path) -> None:
    from evoliez.figures.three_d import render_mutation_overlay

    art = _artifacts(tmp_path / "run")
    out = tmp_path / "muts.png"
    assert render_mutation_overlay(art, out) is None


def test_mutation_overlay_returns_none_without_csv(tmp_path: Path) -> None:
    from evoliez.figures.three_d import mutation_overlay as mod

    art = _artifacts(tmp_path / "run")
    art.wt_complex_pdb = _make_pdb(tmp_path / "wt.pdb")
    out = tmp_path / "muts.png"
    assert mod.render(art, out) is None


def test_mutation_overlay_returns_none_without_pymol(tmp_path: Path) -> None:
    from evoliez.figures.three_d import mutation_overlay as mod

    art = _artifacts(tmp_path / "run")
    art.wt_complex_pdb = _make_pdb(tmp_path / "wt.pdb")
    art.final_candidates_csv = _make_candidates_csv(
        tmp_path / "reports" / "final_candidates.csv",
        mutations=["D222N", "Y196F,A55G"],
        evidence=["Strong", "Promising"],
    )
    out = tmp_path / "muts.png"
    with patch.object(mod, "pymol_available", return_value=False):
        assert mod.render(art, out) is None


def test_mutation_overlay_parses_positions_and_renders(tmp_path: Path) -> None:
    from evoliez.figures.three_d import mutation_overlay as mod

    art = _artifacts(tmp_path / "run")
    art.wt_complex_pdb = _make_pdb(tmp_path / "wt.pdb")
    art.final_candidates_csv = _make_candidates_csv(
        tmp_path / "reports" / "final_candidates.csv",
        mutations=["D222N", "Y196F,A55G", "BADENTRY", "Q100H"],
        evidence=["Strong", "Promising", "Uncertain", "Strong"],
    )
    out = tmp_path / "muts.png"
    captured = {}

    def _fake_run(script: str, *, timeout: float = 120):  # noqa: ARG001
        captured["script"] = script
        Path(out).write_bytes(b"\x89PNG\r\n\x1a\n")
        return (True, "")

    with patch.object(mod, "pymol_available", return_value=True):
        with patch.object(mod, "run_pymol_script", side_effect=_fake_run):
            spec = mod.render(art, out, top_k=4)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "09_final_library_structure"
    assert spec.renderer == "pymol"
    # 222, 196, 55, 100 - "BADENTRY" filtered out by the regex.
    assert spec.params["n_positions"] == 4
    # The script should reference at least one of the parsed positions.
    for pos in ("222", "196", "55", "100"):
        assert pos in captured["script"]


def test_parse_positions_handles_edge_cases() -> None:
    from evoliez.figures.three_d.mutation_overlay import _parse_positions

    assert _parse_positions("") == []
    assert _parse_positions("D222N") == [222]
    assert _parse_positions("D222N,Y196F") == [222, 196]
    assert _parse_positions("D222N; Y196F") == [222, 196]
    assert _parse_positions("BADENTRY") == []
    # Stop codon style is accepted (the * mutation).
    assert _parse_positions("R100*") == [100]
