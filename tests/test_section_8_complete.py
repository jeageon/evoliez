"""Section 8 (MD Validation) renderer coverage.

Three deliverables beyond the existing ligand / pocket RMSD plots:

* ``plots.md_key_distances`` -- time-series key catalytic distances
  (preferred) or summary bars (fallback);
* ``three_d.md_frames`` -- 3Dmol viewer contexts for start / mid / end
  frames of the trajectory;
* ``movies.md_trajectory`` -- best-effort mp4 via PyMOL + ffmpeg.

Each test uses synthetic fixtures under ``tmp_path`` so it runs without
real MD output, without PyMOL, and without ffmpeg.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


pytest.importorskip("evoliez.figures.style")

from evoliez.figures.types import FigureSpec, ReportArtifacts  # noqa: E402


_MINIMAL_PDB = (
    "HEADER    TEST\n"
    "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N\n"
    "ATOM      2  CA  ALA A   1      11.804  13.907  10.500  1.00 20.00           C\n"
    "ATOM      3  C   ALA A   1      13.304  13.907  10.500  1.00 20.00           C\n"
    "ATOM      4  O   ALA A   1      13.904  14.907  10.500  1.00 20.00           O\n"
    "HETATM    5  C1  LIG A 999      15.000  15.000  15.000  1.00 20.00           C\n"
    "END\n"
)


def _empty_artifacts(run_dir: Path) -> ReportArtifacts:
    run_dir.mkdir(parents=True, exist_ok=True)
    return ReportArtifacts(run_dir=run_dir)


def _make_candidates_csv(path: Path, *, candidate_ids) -> Path:
    cols = ["candidate_id", "mutation", "evidence_class", "final_score"]
    lines = [",".join(cols)]
    cycle = ["Strong", "Promising", "Uncertain", "Reject"]
    for i, cid in enumerate(candidate_ids):
        lines.append(
            ",".join(
                [
                    str(cid),
                    f"A{10+i}K",
                    cycle[i % len(cycle)],
                    f"{0.9 - 0.05 * i:.3f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# md_key_distances
# ---------------------------------------------------------------------------


def test_md_key_distances_timeseries(tmp_path: Path) -> None:
    """key_distances dict -> multi-panel PNG."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import md_key_distances

    run = tmp_path / "run"
    run.mkdir()
    md_a = run / "md" / "cand_00"
    md_a.mkdir(parents=True)
    (md_a / "analysis.json").write_text(
        json.dumps(
            {
                "time_ps": [0, 10, 20, 30, 40],
                "key_distances": {
                    "D222-CA_LIG-C1": [3.5, 3.6, 3.4, 3.5, 3.6],
                    "Y196-OH_LIG-N1": [2.8, 2.7, 2.9, 2.8, 2.7],
                },
            }
        )
    )
    md_b = run / "md" / "cand_01"
    md_b.mkdir(parents=True)
    (md_b / "analysis.json").write_text(
        json.dumps(
            {
                "time_ps": [0, 10, 20, 30, 40],
                "key_distances": {
                    "D222-CA_LIG-C1": [3.8, 4.1, 4.5, 4.7, 5.0],
                    "Y196-OH_LIG-N1": [2.9, 3.0, 3.1, 3.2, 3.3],
                },
            }
        )
    )

    art = _empty_artifacts(run)
    art.md_dirs = {"cand_00": md_a, "cand_01": md_b}
    csv_path = _make_candidates_csv(
        run / "final_candidates.csv", candidate_ids=["cand_00", "cand_01"]
    )
    art.final_candidates_csv = csv_path

    out = tmp_path / "kdist.png"
    spec = md_key_distances.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "08_md_key_distances"
    assert spec.section == "md"
    assert out.exists() and out.stat().st_size > 0
    assert spec.params["mode"] == "timeseries"
    assert spec.params["n_distances"] == 2
    assert spec.params["n_candidates"] == 2


def test_md_key_distances_summary_bars(tmp_path: Path) -> None:
    """Only catalytic_distance_mean available -> horizontal-bar PNG."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import md_key_distances

    run = tmp_path / "run"
    run.mkdir()
    for i, cid in enumerate(["cand_00", "cand_01", "cand_02"]):
        d = run / "md" / cid
        d.mkdir(parents=True)
        (d / "analysis.json").write_text(
            json.dumps(
                {
                    "catalytic_distance_mean": 3.5 + 0.3 * i,
                    "catalytic_distance_std": 0.1 + 0.05 * i,
                }
            )
        )

    art = _empty_artifacts(run)
    art.md_dirs = {
        "cand_00": run / "md" / "cand_00",
        "cand_01": run / "md" / "cand_01",
        "cand_02": run / "md" / "cand_02",
    }
    csv_path = _make_candidates_csv(
        run / "final_candidates.csv",
        candidate_ids=["cand_00", "cand_01", "cand_02"],
    )
    art.final_candidates_csv = csv_path

    out = tmp_path / "kdist.png"
    spec = md_key_distances.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "08_md_key_distances"
    assert spec.params["mode"] == "summary_bars"
    assert spec.params["n_candidates"] == 3
    assert out.exists() and out.stat().st_size > 0


def test_md_key_distances_missing_returns_none(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import md_key_distances

    # No md_dirs at all.
    art = _empty_artifacts(tmp_path / "run")
    out = tmp_path / "k.png"
    assert md_key_distances.render(art, out) is None

    # md_dirs exist but analysis.json has neither series nor summary mean.
    run = tmp_path / "run2"
    run.mkdir()
    md_a = run / "md" / "cand_00"
    md_a.mkdir(parents=True)
    (md_a / "analysis.json").write_text(json.dumps({"ligand_rmsd_mean": 1.2}))
    art2 = _empty_artifacts(run)
    art2.md_dirs = {"cand_00": md_a}
    assert md_key_distances.render(art2, out) is None


# ---------------------------------------------------------------------------
# md_frames
# ---------------------------------------------------------------------------


def test_md_frames_start_only(tmp_path: Path) -> None:
    """Minimised PDB present, no trajectory -> 1 viewer (start only)."""
    from evoliez.figures.three_d import md_frames

    run = tmp_path / "run"
    md_dir = run / "md" / "cand_00"
    md_dir.mkdir(parents=True)
    minimised = md_dir / "cand_00_minimized.pdb"
    minimised.write_text(_MINIMAL_PDB)
    (md_dir / "analysis.json").write_text(
        json.dumps(
            {
                "topology_path": str(minimised),
                "trajectory_path": None,
            }
        )
    )

    art = _empty_artifacts(run)
    art.md_dirs = {"cand_00": md_dir}
    csv_path = _make_candidates_csv(
        run / "final_candidates.csv", candidate_ids=["cand_00"]
    )
    art.final_candidates_csv = csv_path

    contexts = md_frames.render(art)
    assert isinstance(contexts, list)
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["id"].endswith("_start")
    assert ctx["pdb_text"]
    assert ctx.get("missing") is False


def test_md_frames_picks_highest_scoring(tmp_path: Path) -> None:
    """With multiple md_dirs + a CSV, picks the top final_score candidate."""
    from evoliez.figures.three_d import md_frames

    run = tmp_path / "run"
    for cid in ("cand_00", "cand_01"):
        d = run / "md" / cid
        d.mkdir(parents=True)
        (d / f"{cid}_minimized.pdb").write_text(_MINIMAL_PDB)
        (d / "analysis.json").write_text(json.dumps({}))

    art = _empty_artifacts(run)
    art.md_dirs = {
        "cand_00": run / "md" / "cand_00",
        "cand_01": run / "md" / "cand_01",
    }
    # cand_01 has the higher score (0.95 vs 0.50).
    (run / "final_candidates.csv").write_text(
        "candidate_id,mutation,evidence_class,final_score\n"
        "cand_00,A10K,Promising,0.500\n"
        "cand_01,A11K,Strong,0.950\n"
    )
    art.final_candidates_csv = run / "final_candidates.csv"

    contexts = md_frames.render(art)
    assert contexts
    assert any("cand_01" in c["id"] for c in contexts)
    assert all("cand_00" not in c["id"] for c in contexts)


def test_md_frames_empty_md_dirs(tmp_path: Path) -> None:
    from evoliez.figures.three_d import md_frames

    art = _empty_artifacts(tmp_path / "run")
    assert md_frames.render(art) == []


def test_md_frames_missing_minimised_pdb(tmp_path: Path) -> None:
    """analysis.json points at a non-existent minimised PDB -> []"""
    from evoliez.figures.three_d import md_frames

    run = tmp_path / "run"
    md_dir = run / "md" / "cand_00"
    md_dir.mkdir(parents=True)
    (md_dir / "analysis.json").write_text(json.dumps({}))

    art = _empty_artifacts(run)
    art.md_dirs = {"cand_00": md_dir}
    assert md_frames.render(art) == []


def test_md_frames_explicit_candidate_id(tmp_path: Path) -> None:
    """Passing candidate_id overrides the CSV-based pick."""
    from evoliez.figures.three_d import md_frames

    run = tmp_path / "run"
    for cid in ("cand_00", "cand_01"):
        d = run / "md" / cid
        d.mkdir(parents=True)
        (d / f"{cid}_minimized.pdb").write_text(_MINIMAL_PDB)

    art = _empty_artifacts(run)
    art.md_dirs = {
        "cand_00": run / "md" / "cand_00",
        "cand_01": run / "md" / "cand_01",
    }
    contexts = md_frames.render(art, candidate_id="cand_00")
    assert contexts
    assert all("cand_00" in c["id"] for c in contexts)


# ---------------------------------------------------------------------------
# md_trajectory movie
# ---------------------------------------------------------------------------


def test_md_trajectory_returns_none_without_pymol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No PyMOL on PATH -> None, no exception."""
    from evoliez.figures.movies import md_trajectory

    monkeypatch.setattr(md_trajectory, "_which", lambda name: None)

    run = tmp_path / "run"
    md_dir = run / "md" / "cand_00"
    md_dir.mkdir(parents=True)
    (md_dir / "cand_00_minimized.pdb").write_text(_MINIMAL_PDB)
    (md_dir / "cand_00.dcd").write_bytes(b"fake-dcd")
    (md_dir / "analysis.json").write_text(
        json.dumps(
            {
                "trajectory_path": str(md_dir / "cand_00.dcd"),
                "topology_path": str(md_dir / "cand_00_minimized.pdb"),
            }
        )
    )

    art = _empty_artifacts(run)
    art.md_dirs = {"cand_00": md_dir}

    out = tmp_path / "movie.mp4"
    assert md_trajectory.render(art, out) is None
    assert not out.exists()


def test_md_trajectory_returns_none_without_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PyMOL present but ffmpeg missing -> None."""
    from evoliez.figures.movies import md_trajectory

    def _which(name: str) -> Optional[str]:
        return "/usr/bin/pymol" if name in ("pymol", "pymolcli", "PyMOL") else None

    monkeypatch.setattr(md_trajectory, "_which", _which)

    run = tmp_path / "run"
    md_dir = run / "md" / "cand_00"
    md_dir.mkdir(parents=True)
    (md_dir / "cand_00_minimized.pdb").write_text(_MINIMAL_PDB)
    (md_dir / "cand_00.dcd").write_bytes(b"fake-dcd")
    (md_dir / "analysis.json").write_text(
        json.dumps(
            {
                "trajectory_path": str(md_dir / "cand_00.dcd"),
                "topology_path": str(md_dir / "cand_00_minimized.pdb"),
            }
        )
    )

    art = _empty_artifacts(run)
    art.md_dirs = {"cand_00": md_dir}

    assert md_trajectory.render(art, tmp_path / "movie.mp4") is None


def test_md_trajectory_returns_none_without_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both binaries present but analysis.json has no trajectory_path -> None."""
    from evoliez.figures.movies import md_trajectory

    monkeypatch.setattr(md_trajectory, "_which", lambda name: f"/usr/bin/{name}")

    run = tmp_path / "run"
    md_dir = run / "md" / "cand_00"
    md_dir.mkdir(parents=True)
    (md_dir / "cand_00_minimized.pdb").write_text(_MINIMAL_PDB)
    (md_dir / "analysis.json").write_text(json.dumps({}))

    art = _empty_artifacts(run)
    art.md_dirs = {"cand_00": md_dir}

    assert md_trajectory.render(art, tmp_path / "movie.mp4") is None


def test_md_trajectory_returns_none_when_empty_md_dirs(tmp_path: Path) -> None:
    from evoliez.figures.movies import md_trajectory

    art = _empty_artifacts(tmp_path / "run")
    assert md_trajectory.render(art, tmp_path / "movie.mp4") is None


def test_md_trajectory_success_path_with_stubbed_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full success: stubbed pymol writes frame_*.png, stubbed ffmpeg writes mp4."""
    from evoliez.figures.movies import md_trajectory

    monkeypatch.setattr(md_trajectory, "_which", lambda name: f"/usr/bin/{name}")

    # Stub _run_pymol so it writes frames into the scratch dir that the
    # script's ``mpng`` line points at.  We can recover the scratch dir
    # by parsing the script.
    def fake_run_pymol(binary: str, script: str, **kwargs: Any) -> tuple:
        # The script writes ``mpng <scratch>/frame_,...`` -- extract.
        prefix = None
        for line in script.splitlines():
            if line.startswith("mpng "):
                prefix = line.split()[1].rstrip(",")
                break
        assert prefix is not None
        scratch = Path(prefix).parent
        scratch.mkdir(parents=True, exist_ok=True)
        # Emit 3 PNG-ish files.
        for i in range(1, 4):
            (scratch / f"frame_{i:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return (True, "")

    def fake_run_ffmpeg(
        ffmpeg_bin: str,
        pattern: Path,
        out_path: Path,
        **kwargs: Any,
    ) -> tuple:
        out_path = Path(out_path)
        out_path.write_bytes(b"\x00\x00\x00 ftypisom" + b"\x00" * 100)
        return (True, "")

    monkeypatch.setattr(md_trajectory, "_run_pymol", fake_run_pymol)
    monkeypatch.setattr(md_trajectory, "_run_ffmpeg", fake_run_ffmpeg)

    run = tmp_path / "run"
    md_dir = run / "md" / "cand_00"
    md_dir.mkdir(parents=True)
    (md_dir / "cand_00_minimized.pdb").write_text(_MINIMAL_PDB)
    (md_dir / "cand_00.dcd").write_bytes(b"fake-dcd")
    (md_dir / "analysis.json").write_text(
        json.dumps(
            {
                "trajectory_path": str(md_dir / "cand_00.dcd"),
                "topology_path": str(md_dir / "cand_00_minimized.pdb"),
            }
        )
    )

    art = _empty_artifacts(run)
    art.md_dirs = {"cand_00": md_dir}

    out = tmp_path / "movies" / "08_md_trajectory.mp4"
    spec = md_trajectory.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "08_md_trajectory"
    assert spec.section == "md"
    assert spec.renderer == "pymol+ffmpeg"
    assert out.exists() and out.stat().st_size > 0
    assert spec.params["candidate_id"] == "cand_00"
    # Poster image written alongside the mp4.
    poster = Path(spec.params.get("poster") or "")
    assert poster.exists()


def test_md_trajectory_pymol_failure_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pymol subprocess fails -> render returns None gracefully."""
    from evoliez.figures.movies import md_trajectory

    monkeypatch.setattr(md_trajectory, "_which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        md_trajectory,
        "_run_pymol",
        lambda binary, script, **kw: (False, "pymol blew up"),
    )

    run = tmp_path / "run"
    md_dir = run / "md" / "cand_00"
    md_dir.mkdir(parents=True)
    (md_dir / "cand_00_minimized.pdb").write_text(_MINIMAL_PDB)
    (md_dir / "cand_00.dcd").write_bytes(b"fake-dcd")
    (md_dir / "analysis.json").write_text(
        json.dumps(
            {
                "trajectory_path": str(md_dir / "cand_00.dcd"),
                "topology_path": str(md_dir / "cand_00_minimized.pdb"),
            }
        )
    )

    art = _empty_artifacts(run)
    art.md_dirs = {"cand_00": md_dir}

    assert md_trajectory.render(art, tmp_path / "movie.mp4") is None


# ---------------------------------------------------------------------------
# Builder dispatch guard
# ---------------------------------------------------------------------------


def test_builder_dispatch_includes_md_key_distances() -> None:
    """Builder's plot dispatch wires the new key-distances renderer."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.html.builder import _plot_dispatch

    dispatch = _plot_dispatch()
    assert "08_md_key_distances" in dispatch
    section_key, fn = dispatch["08_md_key_distances"]
    assert section_key == "08_md"
    assert callable(fn)


def test_builder_source_references_md_frames_and_movie() -> None:
    """Builder source-code text references the section-8 viewer + movie hooks.

    A static guard so the dispatch wiring can't be silently removed --
    we only assert the references exist; behaviour is covered by the
    other tests in this module.
    """
    from evoliez.figures.html import builder

    src = Path(builder.__file__).read_text(encoding="utf-8")
    assert "md_frames" in src
    assert "md_trajectory" in src
    assert "08_md_trajectory" in src or "08_md_trajectory.mp4" in src
