"""Tests for the Section 3 (Boltz Complex) pLDDT + pocket close-up feature.

Expert plan requirements for section 3:

* predicted protein-ligand complex 3D structure;
* ligand binding pocket close-up;
* Boltz confidence (pLDDT) coloring.

This suite verifies the two pieces that ship the feature:

1. ``three_d/pdb_inline.make_viewer_context`` honors the new
   ``color_by="plddt"`` and ``pocket_zoom=8.0`` options by populating
   the returned dict with executable JS snippets (``color_js`` /
   ``zoom_js``) and provenance flags.  Other call sites that omit
   these kwargs keep the legacy default behavior (no extra JS) so
   sections 04 and 09 are unaffected.
2. ``html/builder._build_inline_fallback`` produces two viewer +
   FigureSpec pairs for section 3's ``03_boltz_binding_pocket`` figure
   id: the full pLDDT-colored structure and the 8 A close-up.  Both
   land in ``viewers["03_boltz_complex"]`` and the manifest
   ``FigureSpec`` list with ``renderer="3dmol-inline"``.

The PyMOL availability check is patched off so the suite runs on the
laptop dev box.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip("evoliez.figures.three_d")
pytest.importorskip("evoliez.figures.html")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_SYNTH_PDB = (
    "HEADER    SECTION 3 TEST\n"
    "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N\n"
    "ATOM      2  CA  ALA A   1      11.804  13.907  10.500  1.00 80.00           C\n"
    "ATOM      3  C   ALA A   1      13.304  13.907  10.500  1.00 95.00           C\n"
    "ATOM      4  O   ALA A   1      13.904  14.907  10.500  1.00 40.00           O\n"
    "HETATM    5  C1  LIG A 999      15.000  15.000  15.000  1.00 20.00           C\n"
    "END\n"
)


def _make_pdb(path: Path, text: str = _SYNTH_PDB) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _make_run_dir(run_dir: Path) -> Path:
    """Materialise the smallest run_dir the discovery layer accepts."""
    run_dir = Path(run_dir)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "msa").mkdir(parents=True, exist_ok=True)
    (run_dir / "complexes").mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs" / "target.fasta").write_text(
        ">target\nMKLKVLGAGAGAGAGA\n", encoding="utf-8"
    )
    (run_dir / "_state.json").write_text(
        json.dumps({"target": "demo", "backend": "mock"}),
        encoding="utf-8",
    )
    wt_dir = (
        run_dir / "complexes" / "boltz_results_wt_boltz_input"
        / "predictions" / "wt_boltz_input"
    )
    wt_dir.mkdir(parents=True, exist_ok=True)
    (wt_dir / "wt_boltz_input_model_0.pdb").write_text(_SYNTH_PDB, encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# make_viewer_context: pLDDT coloring option
# ---------------------------------------------------------------------------


def test_make_viewer_context_plddt_emits_colorfunc(tmp_path: Path) -> None:
    """``color_by='plddt'`` should populate ``color_js`` with a colorfunc."""
    from evoliez.figures.three_d import make_viewer_context

    pdb = _make_pdb(tmp_path / "wt.pdb")
    ctx = make_viewer_context(pdb, viewer_id="wt", color_by="plddt")

    assert ctx["missing"] is False
    assert ctx.get("color_by") == "plddt"
    # The JS must be a complete 3Dmol cartoon-style call that uses a
    # colorfunc reading from atom.b (B-factor / pLDDT column).
    color_js = ctx.get("color_js", "")
    assert isinstance(color_js, str) and color_js, "color_js must be populated"
    assert "colorfunc" in color_js
    assert "atom.b" in color_js
    # Confidence thresholds + Boltz/AlphaFold palette.
    assert "50" in color_js
    assert "70" in color_js
    assert "90" in color_js
    for color in ("red", "orange", "yellow", "blue"):
        assert color in color_js


def test_make_viewer_context_plddt_for_missing_file(tmp_path: Path) -> None:
    """Missing-file branch still annotates color_by / color_js for parity."""
    from evoliez.figures.three_d import make_viewer_context

    ctx = make_viewer_context(
        tmp_path / "nope.pdb", viewer_id="wt", color_by="plddt"
    )
    assert ctx["missing"] is True
    assert ctx.get("color_by") == "plddt"
    assert "colorfunc" in ctx.get("color_js", "")


# ---------------------------------------------------------------------------
# make_viewer_context: pocket_zoom option
# ---------------------------------------------------------------------------


def test_make_viewer_context_pocket_zoom_emits_zoom_js(tmp_path: Path) -> None:
    """``pocket_zoom=8.0`` should populate ``zoom_js`` with zoomTo+zoom calls."""
    from evoliez.figures.three_d import make_viewer_context

    pdb = _make_pdb(tmp_path / "wt.pdb")
    ctx = make_viewer_context(pdb, viewer_id="wt", pocket_zoom=8.0)

    assert ctx.get("pocket_zoom") == pytest.approx(8.0)
    zoom_js = ctx.get("zoom_js", "")
    assert isinstance(zoom_js, str) and zoom_js, "zoom_js must be populated"
    # Should center on the ligand resn list...
    assert "zoomTo" in zoom_js
    assert "resn" in zoom_js
    # ...and then close in (any positive zoom factor is fine; ~1.4 is the
    # plan's recommended target for an 8 A close-up).
    assert "v.zoom(" in zoom_js


def test_make_viewer_context_pocket_zoom_resn_list_contains_cofactors(
    tmp_path: Path,
) -> None:
    """The zoom-to selection must include the same cofactor resn list the
    threed_viewer template recognizes (NDP/NAP/LIG/NAI/NAD/SAH/SAM)."""
    from evoliez.figures.three_d import make_viewer_context

    pdb = _make_pdb(tmp_path / "wt.pdb")
    ctx = make_viewer_context(pdb, viewer_id="wt", pocket_zoom=8.0)
    zoom_js = ctx.get("zoom_js", "")
    for resn in ("NDP", "NAP", "LIG", "NAI", "NAD", "SAH", "SAM"):
        assert resn in zoom_js, f"cofactor {resn!r} missing from zoom JS"


# ---------------------------------------------------------------------------
# make_viewer_context: backwards compatibility for sections 04 / 09
# ---------------------------------------------------------------------------


def test_make_viewer_context_default_emits_no_extra_js(tmp_path: Path) -> None:
    """Default args (no color_by / pocket_zoom) must keep sections 04 / 09
    unaffected: no colorfunc / zoom JS leaks out."""
    from evoliez.figures.three_d import make_viewer_context

    pdb = _make_pdb(tmp_path / "wt.pdb")
    ctx = make_viewer_context(pdb, viewer_id="wt")

    assert ctx.get("color_by") == "spectrum"
    assert ctx.get("color_js", "") == ""
    assert ctx.get("pocket_zoom") is None
    assert ctx.get("zoom_js", "") == ""


def test_make_viewer_context_unknown_color_by_falls_back(tmp_path: Path) -> None:
    """An unknown ``color_by`` should warn-and-fallback to spectrum, not raise."""
    from evoliez.figures.three_d import make_viewer_context

    pdb = _make_pdb(tmp_path / "wt.pdb")
    ctx = make_viewer_context(pdb, viewer_id="wt", color_by="rainbow-unicorn")
    assert ctx.get("color_by") == "spectrum"
    assert ctx.get("color_js", "") == ""


# ---------------------------------------------------------------------------
# Builder source guard: section 3 ships two viewers
# ---------------------------------------------------------------------------


def test_builder_section_3_appends_two_viewers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the PyMOL render is unavailable, section 3 must fall back to
    BOTH the pLDDT-colored full viewer AND the binding-pocket close-up."""
    run_dir = _make_run_dir(tmp_path / "run")

    from evoliez.figures.discovery import discover
    from evoliez.figures.html import builder

    # Pretend PyMOL isn't installed and force the three PyMOL renderers
    # to return None so the fallback path runs.
    monkeypatch.setattr(builder, "pymol_available", lambda: False)
    monkeypatch.setattr(builder, "render_pocket_view", lambda *a, **k: None)
    monkeypatch.setattr(builder, "render_pose_ensemble", lambda *a, **k: None)
    monkeypatch.setattr(builder, "render_mutation_overlay", lambda *a, **k: None)

    artifacts = discover(run_dir)
    assert artifacts.wt_complex_pdb is not None
    viewers, pymol_specs, fallback_specs = builder._build_3d_section_assets(
        artifacts, tmp_path / "figs", style="presentation", skip_3d=False
    )

    sec03 = viewers["03_boltz_complex"]
    sec03_ids = [v.get("id") for v in sec03]
    # Two new viewer ids are present (legacy ``wt_pocket`` was retired).
    assert "wt_pocket_plddt" in sec03_ids
    assert "wt_pocket_closeup" in sec03_ids

    # The pLDDT viewer carries the colorfunc JS; the close-up viewer
    # carries the zoom JS.  This is what makes the section actually
    # render as the expert plan describes once a future template pass
    # consumes the new dict keys.
    by_id = {v["id"]: v for v in sec03}
    assert "colorfunc" in by_id["wt_pocket_plddt"].get("color_js", "")
    assert "v.zoom(" in by_id["wt_pocket_closeup"].get("zoom_js", "")

    # Manifest FigureSpecs: one per new viewer, both renderer="3dmol-inline".
    assert pymol_specs == []
    section_3_specs = [
        s for s in fallback_specs if s.section == "boltz_complex"
    ]
    section_3_ids = {s.figure_id for s in section_3_specs}
    assert "03_boltz_complex_3dmol" in section_3_ids
    assert "03_boltz_complex_3dmol_closeup" in section_3_ids
    for spec in section_3_specs:
        assert spec.renderer == "3dmol-inline"
        assert spec.params.get("inline") is True
        assert spec.params.get("viewer_id")
        assert "inline://" in spec.params.get("inline_uri", "")


def test_builder_section_3_other_sections_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dual-viewer change is scoped to section 3.  Section 04 still
    emits exactly one fallback viewer (``pose_ensemble_inline``) and
    section 09 still emits exactly one (``final_library_structure_inline``)."""
    run_dir = _make_run_dir(tmp_path / "run")
    # Add a final_candidates.csv so the section-09 fallback has highlight
    # residues to surface.
    (run_dir / "reports" / "final_candidates.csv").write_text(
        "candidate_id,mutation,evidence_class,final_score\n"
        "cand_00,D222N,Strong,0.900\n",
        encoding="utf-8",
    )

    from evoliez.figures.discovery import discover
    from evoliez.figures.html import builder

    monkeypatch.setattr(builder, "pymol_available", lambda: False)
    monkeypatch.setattr(builder, "render_pocket_view", lambda *a, **k: None)
    monkeypatch.setattr(builder, "render_pose_ensemble", lambda *a, **k: None)
    monkeypatch.setattr(builder, "render_mutation_overlay", lambda *a, **k: None)

    artifacts = discover(run_dir)
    viewers, _pymol_specs, _fallback_specs = builder._build_3d_section_assets(
        artifacts, tmp_path / "figs", style="presentation", skip_3d=False
    )

    sec04_ids = [v.get("id") for v in viewers["04_pose_ensemble"]]
    assert sec04_ids.count("pose_ensemble_inline") == 1
    sec09_ids = [v.get("id") for v in viewers["09_final_library"]]
    assert sec09_ids.count("final_library_structure_inline") == 1

    # And the section 04 / 09 viewers must still use spectrum coloring
    # (no pLDDT leak from the section-3 change).
    for v in viewers["04_pose_ensemble"]:
        if v.get("id") == "pose_ensemble_inline":
            assert v.get("color_by") == "spectrum"
            assert v.get("color_js", "") == ""
            assert v.get("zoom_js", "") == ""
    for v in viewers["09_final_library"]:
        if v.get("id") == "final_library_structure_inline":
            assert v.get("color_by") == "spectrum"
            assert v.get("color_js", "") == ""
            assert v.get("zoom_js", "") == ""
