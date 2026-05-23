"""Regression tests for the builder's 3Dmol inline fallback.

When PyMOL is unavailable, the three PyMOL-backed sections (03 boltz
complex, 04 pose ensemble, 09 final library 3D) used to render empty.
The fix in ``evoliez.figures.html.builder._build_3d_section_assets`` now
substitutes a 3Dmol inline viewer + records a placeholder ``FigureSpec``
with ``renderer="3dmol-inline"`` in the manifest.

These tests stub out the PyMOL renderers via ``monkeypatch`` so the
suite doesn't depend on a PyMOL binary, and verify:

* with PyMOL unavailable + a WT PDB on disk, every targeted section has
  a non-empty ``viewers`` list and the manifest gains a ``3dmol-inline``
  entry per section;
* with PyMOL available (renderers return real ``FigureSpec``s), the
  *fallback*-specific viewer/spec pair is NOT added (we don't
  double-stack viewers over a working PyMOL render);
* with NO WT PDB on disk at all, no fallback viewers/specs appear -
  the report degrades gracefully rather than fabricating empty viewers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest


pytest.importorskip("evoliez.figures.html")
pytest.importorskip("jinja2")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


_MINIMAL_PDB = (
    "HEADER    TEST\n"
    "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N\n"
    "ATOM      2  CA  ALA A   1      11.804  13.907  10.500  1.00 20.00           C\n"
    "ATOM      3  C   ALA A   1      13.304  13.907  10.500  1.00 20.00           C\n"
    "ATOM      4  O   ALA A   1      13.904  14.907  10.500  1.00 20.00           O\n"
    "HETATM    5  C1  LIG A 999      15.000  15.000  15.000  1.00 20.00           C\n"
    "END\n"
)


def _make_run_dir(run_dir: Path, *, with_wt: bool = True, with_csv: bool = True) -> Path:
    """Build the smallest run_dir that the builder will accept.

    ``with_wt`` controls whether a WT complex PDB is materialised at the
    location discovery expects (``complexes/.../wt_boltz_input/*model_0.pdb``).
    ``with_csv`` adds a final_candidates.csv with a top mutation so the
    section-09 fallback has highlight residues to surface.
    """
    run_dir = Path(run_dir)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "msa").mkdir(parents=True, exist_ok=True)
    (run_dir / "complexes").mkdir(parents=True, exist_ok=True)

    (run_dir / "inputs" / "target.fasta").write_text(
        ">target\nMKLKVLGAGAGAGAGA\n", encoding="utf-8"
    )
    (run_dir / "_state.json").write_text(
        json.dumps(
            {
                "target": "demo_target",
                "backend": "mock",
                "catalytic_targets": ["D222"],
            }
        ),
        encoding="utf-8",
    )

    if with_wt:
        # Layout matches evoliez.figures.discovery._find_wt_complex.
        wt_dir = (
            run_dir / "complexes" / "boltz_results_wt_boltz_input"
            / "predictions" / "wt_boltz_input"
        )
        wt_dir.mkdir(parents=True, exist_ok=True)
        (wt_dir / "wt_boltz_input_model_0.pdb").write_text(_MINIMAL_PDB, encoding="utf-8")

    if with_csv:
        csv_path = run_dir / "reports" / "final_candidates.csv"
        csv_path.write_text(
            "candidate_id,mutation,evidence_class,final_score\n"
            "cand_00,D222N,Strong,0.900\n"
            "cand_01,Y196F,Promising,0.850\n",
            encoding="utf-8",
        )

    return run_dir


def _patch_pymol(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool,
    return_spec: bool = False,
) -> None:
    """Stub the PyMOL availability check + renderers on the builder module.

    When ``return_spec=True``, every renderer returns a real ``FigureSpec``
    that points at an actual file under ``tmp_path`` (the builder requires
    ``Path(spec.path).exists()`` before adding the figure).
    """
    from evoliez.figures.html import builder
    from evoliez.figures.types import FigureSpec

    monkeypatch.setattr(builder, "pymol_available", lambda: available)

    if not return_spec:
        monkeypatch.setattr(
            builder, "render_pocket_view", lambda *a, **k: None
        )
        monkeypatch.setattr(
            builder, "render_pose_ensemble", lambda *a, **k: None
        )
        monkeypatch.setattr(
            builder, "render_mutation_overlay", lambda *a, **k: None
        )
        return

    # PyMOL-success path: write a tiny PNG-ish placeholder to satisfy
    # the ``Path(spec.path).exists()`` guard in build_report.
    def _spec_for(fid: str, section: str):
        def _render(artifacts, out_path: Path, *, style: str = "presentation", **kw):
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
            return FigureSpec(
                figure_id=fid,
                section=section,
                title=f"{fid} (pymol)",
                description="fake pymol render",
                path=out_path,
                source_files=[],
                renderer="pymol",
                params={},
            )

        return _render

    monkeypatch.setattr(
        builder,
        "render_pocket_view",
        _spec_for("03_boltz_binding_pocket", "boltz_complex"),
    )
    monkeypatch.setattr(
        builder,
        "render_pose_ensemble",
        _spec_for("04_boltz_pose_ensemble", "pose_ensemble"),
    )
    monkeypatch.setattr(
        builder,
        "render_mutation_overlay",
        _spec_for("09_final_library_structure", "final_library"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fallback_viewers_added_when_pymol_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PyMOL absent + WT PDB present -> 3Dmol fallback in 03 / 04 / 09."""
    run_dir = _make_run_dir(tmp_path / "run")
    _patch_pymol(monkeypatch, available=False)

    # Inspect the internal helper directly so we don't depend on the
    # partial templates iterating ``section.viewers``.
    from evoliez.figures.discovery import discover
    from evoliez.figures.html.builder import _build_3d_section_assets

    artifacts = discover(run_dir)
    assert artifacts.wt_complex_pdb is not None, "fixture must materialise WT PDB"

    viewers, pymol_specs, fallback_specs = _build_3d_section_assets(
        artifacts, tmp_path / "figs", style="presentation", skip_3d=False
    )

    # Section 03 - two fallback viewers: pLDDT-colored full structure
    # and an 8 A pocket close-up.  The legacy single ``wt_pocket`` id
    # was retired when section 03 grew the close-up companion.
    sec03_ids = [v.get("id") for v in viewers["03_boltz_complex"]]
    assert "wt_pocket_plddt" in sec03_ids
    assert "wt_pocket_closeup" in sec03_ids
    # Section 04 - pose-ensemble fallback viewer.
    sec04_ids = [v.get("id") for v in viewers["04_pose_ensemble"]]
    assert "pose_ensemble_inline" in sec04_ids
    # Section 09 - mutation overlay fallback viewer (requires CSV +
    # parseable mutation tokens, both supplied by the fixture).
    sec09_ids = [v.get("id") for v in viewers["09_final_library"]]
    assert "final_library_structure_inline" in sec09_ids

    # Each section also has corresponding manifest FigureSpec(s) with
    # renderer="3dmol-inline".  Section 03 contributes TWO (full +
    # close-up); sections 04 / 09 contribute one each.
    assert pymol_specs == []
    fallback_ids = {s.figure_id for s in fallback_specs}
    assert "03_boltz_complex_3dmol" in fallback_ids
    assert "03_boltz_complex_3dmol_closeup" in fallback_ids
    assert "04_pose_ensemble_3dmol" in fallback_ids
    assert "09_final_library_3dmol" in fallback_ids
    for spec in fallback_specs:
        assert spec.renderer == "3dmol-inline"
        # ``path`` is a ``Path(".")`` sentinel - the bundler validates
        # every manifest figure resolves to a real file, and "." -> the
        # report root which always exists.  ``params["inline"]`` is the
        # real marker the templates check.
        assert spec.params.get("inline") is True
        assert spec.params.get("viewer_id")
        assert "inline://" in spec.params.get("inline_uri", "")


def test_no_fallback_when_pymol_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PyMOL renders return real specs -> no inline-fallback viewer added."""
    run_dir = _make_run_dir(tmp_path / "run")
    _patch_pymol(monkeypatch, available=True, return_spec=True)

    from evoliez.figures.discovery import discover
    from evoliez.figures.html.builder import _build_3d_section_assets

    artifacts = discover(run_dir)
    viewers, pymol_specs, fallback_specs = _build_3d_section_assets(
        artifacts, tmp_path / "figs", style="presentation", skip_3d=False
    )

    # Three PyMOL specs from the patched renderers, zero fallbacks.
    pymol_ids = {s.figure_id for s in pymol_specs}
    assert pymol_ids == {
        "03_boltz_binding_pocket",
        "04_boltz_pose_ensemble",
        "09_final_library_structure",
    }
    assert fallback_specs == []

    # The fallback-specific viewer ids must NOT be present (the
    # legacy unconditional ``wt-complex`` / per-mutant viewers may
    # still appear - that's pre-existing behavior and intentional).
    all_viewer_ids = []
    for vs in viewers.values():
        all_viewer_ids.extend(v.get("id") for v in vs)
    for forbidden in (
        "wt_pocket_plddt",
        "wt_pocket_closeup",
        "pose_ensemble_inline",
        "final_library_structure_inline",
    ):
        assert forbidden not in all_viewer_ids, (
            f"fallback viewer {forbidden!r} should not be added when "
            "PyMOL succeeded"
        )


def test_no_fallback_when_no_wt_pdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No WT PDB on disk -> graceful skip, no fallback viewers/specs."""
    run_dir = _make_run_dir(tmp_path / "run", with_wt=False, with_csv=True)
    _patch_pymol(monkeypatch, available=False)

    from evoliez.figures.discovery import discover
    from evoliez.figures.html.builder import _build_3d_section_assets

    artifacts = discover(run_dir)
    assert artifacts.wt_complex_pdb is None, "fixture must not have WT PDB"

    viewers, pymol_specs, fallback_specs = _build_3d_section_assets(
        artifacts, tmp_path / "figs", style="presentation", skip_3d=False
    )

    # No PyMOL specs (we mocked them to None), no fallback specs (no
    # WT PDB to anchor the viewer to).
    assert pymol_specs == []
    assert fallback_specs == []
    # And the per-section viewer buckets stay empty for the three
    # PyMOL-related sections.
    assert viewers["03_boltz_complex"] == []
    assert viewers["04_pose_ensemble"] == []
    assert viewers["09_final_library"] == []


def test_manifest_records_3dmol_inline_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: visual_manifest.json contains the 3dmol-inline entries."""
    run_dir = _make_run_dir(tmp_path / "run")
    _patch_pymol(monkeypatch, available=False)

    from evoliez.figures.html.builder import build_report

    result = build_report(
        run_dir,
        output_dir=tmp_path / "out",
        style="presentation",
        html_mode="linked",
        skip_3d=False,
        mock_backend=False,
        output_zip=None,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    figures = manifest["figures"]
    inline_figs = [f for f in figures if f.get("renderer") == "3dmol-inline"]
    inline_ids = {f["figure_id"] for f in inline_figs}
    assert "03_boltz_complex_3dmol" in inline_ids
    assert "03_boltz_complex_3dmol_closeup" in inline_ids
    assert "04_pose_ensemble_3dmol" in inline_ids
    assert "09_final_library_3dmol" in inline_ids
    # Inline figures use a ``Path(".")`` sentinel path; the bundler's
    # manifest validator must resolve it to the report root (always
    # present) so packaging doesn't error out.
    for f in inline_figs:
        assert f["params"].get("inline") is True
        assert f["params"].get("viewer_id")


def test_skip_3d_disables_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``skip_3d=True`` short-circuits both PyMOL renders and fallbacks."""
    run_dir = _make_run_dir(tmp_path / "run")
    _patch_pymol(monkeypatch, available=False)

    from evoliez.figures.discovery import discover
    from evoliez.figures.html.builder import _build_3d_section_assets

    artifacts = discover(run_dir)
    viewers, pymol_specs, fallback_specs = _build_3d_section_assets(
        artifacts, tmp_path / "figs", style="presentation", skip_3d=True
    )
    # All three sections empty - no viewers, no specs.
    assert pymol_specs == []
    assert fallback_specs == []
    for sec in ("03_boltz_complex", "04_pose_ensemble", "09_final_library"):
        assert viewers[sec] == []
