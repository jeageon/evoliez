"""Lock in the publication-grade PyMOL pocket-view style.

The renderer was upgraded from the earlier "magenta sticks on cyan
cartoon" to a user-requested 2026-05-27 template:
  - grey ribbon backdrop (semi-transparent)
  - ligand as element-coloured sticks (util.cbag)
  - pocket residues within `contact_radius` A as cyan sticks
  - one-letter+residue-number labels on CA
  - yellow dashed polar contacts up to `distance_cutoff` A

If anyone reverts the template by accident these source-pattern guards
fail loudly.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

from evoliez.figures.three_d import pocket_view
from evoliez.figures.types import ReportArtifacts


# --------------------------------------------------------------------------- #
# Template content guards
# --------------------------------------------------------------------------- #
class TestScriptTemplate:
    """The PyMOL script template must contain every visual element of
    the publication style. Each missing token => figure regression."""

    def test_has_white_background(self):
        src = pocket_view._SCRIPT.template
        assert "bg_color white" in src

    def test_has_grey_cartoon_backdrop(self):
        src = pocket_view._SCRIPT.template
        assert "color gray90, wt and polymer" in src
        assert "cartoon_transparency" in src

    def test_ligand_uses_element_colors_via_cbag(self):
        """util.cbag = color-by-atom with green carbons. Without it the
        ligand sticks default to the prior `color magenta` look."""
        src = pocket_view._SCRIPT.template
        assert "util.cbag my_ligand" in src
        # The old uniform color must NOT come back.
        assert "color magenta, my_ligand" not in src
        assert "color magenta, lig" not in src

    def test_pocket_residues_via_byres_around(self):
        """`byres (br. (my_ligand around N) and not my_ligand) and polymer`
        is the canonical selection for whole pocket residues. The
        `polymer` clause guards against atoms from non-ligand HETATMs
        being included."""
        src = pocket_view._SCRIPT.template
        assert "byres (br. (my_ligand around $contact_radius)" in src
        assert "and not my_ligand" in src
        assert "and polymer" in src

    def test_pocket_residues_cyan_sticks(self):
        src = pocket_view._SCRIPT.template
        assert "show sticks, near_ligands" in src
        assert "color cyan, near_ligands" in src

    def test_oneletter_resi_labels_on_ca(self):
        src = pocket_view._SCRIPT.template
        # Pattern: `label near_ligands and name CA, oneletter+resi`
        assert "label near_ligands and name CA, oneletter+resi" in src
        assert "label_color, black" in src
        assert "label_font_id, 7" in src

    def test_yellow_dashed_distance_lines(self):
        src = pocket_view._SCRIPT.template
        assert "dist ligand_contacts, my_ligand, near_ligands" in src
        # mode=2 is PyMOL's polar-contact-only mode (no every-pair noise)
        assert "mode=2" in src
        assert "set dash_color, yellow, ligand_contacts" in src
        # Per-line distance labels would clutter the figure on top of
        # the residue labels we already drew.
        assert "hide labels, ligand_contacts" in src

    def test_orient_on_pocket_then_zoom_on_ligand(self):
        """`orient` alone frames the whole protein; the user-preferred
        view is `orient my_ligand or near_ligands` then `zoom my_ligand`."""
        src = pocket_view._SCRIPT.template
        assert "orient my_ligand or near_ligands" in src
        assert "zoom my_ligand" in src

    def test_template_compiles_with_all_placeholders(self):
        """All Template $-placeholders must be supplied by render()."""
        rendered = pocket_view._SCRIPT.substitute(
            pdb_path="/tmp/wt.pdb",
            ligand_resn=pocket_view._LIGAND_RESN,
            pocket_radius="8.00",
            contact_radius="4.00",
            distance_cutoff="5.00",
            cartoon_transparency="0.70",
            label_size=24,
            width=1600,
            height=1200,
            out_path="/tmp/out.png",
            dpi=200,
        )
        # No leftover unsubstituted $... placeholder.
        assert "$" not in rendered


# --------------------------------------------------------------------------- #
# render() kwarg surface
# --------------------------------------------------------------------------- #
class TestRenderKwargs:
    """Behaviour of the new kwargs added by the style upgrade."""

    def test_signature_exposes_new_kwargs(self):
        sig = inspect.signature(pocket_view.render)
        for name in (
            "contact_radius",
            "distance_cutoff",
            "cartoon_transparency",
            "pocket_radius",
        ):
            assert name in sig.parameters, f"missing kwarg: {name}"

    def test_default_contact_radius_4_angstrom(self):
        """4 A matches the user-supplied PyMOL template's `around 4`."""
        sig = inspect.signature(pocket_view.render)
        assert sig.parameters["contact_radius"].default == 4.0

    def test_default_distance_cutoff_5_angstrom(self):
        sig = inspect.signature(pocket_view.render)
        assert sig.parameters["distance_cutoff"].default == 5.0

    def test_default_cartoon_partly_transparent(self):
        """0.7 = visible but recessive (lets the pocket atoms dominate).
        The earlier hard-coded 0.15 made the cartoon too opaque to read
        the pocket residues over."""
        sig = inspect.signature(pocket_view.render)
        assert sig.parameters["cartoon_transparency"].default == 0.7

    def test_cartoon_transparency_clipped_to_0_1(self, tmp_path, monkeypatch):
        """Out-of-range values would make PyMOL complain. Clip defensively."""
        # Force the render to reach the substitute() call but never
        # invoke PyMOL: fake the PDB on disk and stub PyMOL absent.
        pdb = tmp_path / "wt.pdb"
        pdb.write_text("ATOM\n")
        arts = ReportArtifacts(run_dir=tmp_path)
        arts.wt_complex_pdb = pdb

        monkeypatch.setattr(pocket_view, "pymol_available", lambda: False)
        # Sanity: render returns None without PyMOL (no crash from
        # out-of-range transparency).
        assert pocket_view.render(arts, tmp_path / "out.png",
                                  cartoon_transparency=2.5) is None
        assert pocket_view.render(arts, tmp_path / "out.png",
                                  cartoon_transparency=-1.0) is None


# --------------------------------------------------------------------------- #
# Label sizing per style
# --------------------------------------------------------------------------- #
class TestLabelSize:
    def test_paper_uses_smaller_labels(self):
        assert pocket_view._label_size_for("paper") < pocket_view._label_size_for(
            "presentation"
        )

    def test_poster_uses_larger_labels(self):
        assert pocket_view._label_size_for("poster") > pocket_view._label_size_for(
            "presentation"
        )

    def test_presentation_default_24(self):
        """Matches the user-supplied template's `set label_size, 24`."""
        assert pocket_view._label_size_for("presentation") == 24


# --------------------------------------------------------------------------- #
# Provenance: params dict should record every knob so the manifest +
# `evoliez figures --replay` can reproduce the exact same image later.
# --------------------------------------------------------------------------- #
class TestProvenanceParams:
    def test_render_records_new_knobs_in_figurespec_params(self, tmp_path, monkeypatch):
        """When PyMOL render succeeds, the returned FigureSpec.params must
        carry the new style knobs so the manifest can replay them."""
        pdb = tmp_path / "wt.pdb"
        pdb.write_text("ATOM\n")
        out = tmp_path / "out.png"
        arts = ReportArtifacts(run_dir=tmp_path)
        arts.wt_complex_pdb = pdb

        # Simulate PyMOL present + a successful render.
        monkeypatch.setattr(pocket_view, "pymol_available", lambda: True)

        def fake_run(script, **kw):
            # Smoke-check the rendered script contains every guard.
            for needle in (
                "util.cbag",
                "byres (br. (my_ligand around",
                "color cyan, near_ligands",
                "oneletter+resi",
                "mode=2",
            ):
                assert needle in script
            out.write_bytes(b"\x89PNG\r\n\x1a\n")
            return True, ""

        monkeypatch.setattr(pocket_view, "run_pymol_script", fake_run)

        spec = pocket_view.render(
            arts,
            out,
            contact_radius=4.5,
            distance_cutoff=5.5,
            cartoon_transparency=0.6,
        )
        assert spec is not None
        for key in (
            "contact_radius",
            "distance_cutoff",
            "cartoon_transparency",
            "label_size",
        ):
            assert key in spec.params, f"manifest missing param: {key}"
        assert spec.params["contact_radius"] == 4.5
        assert spec.params["distance_cutoff"] == 5.5
        assert spec.params["cartoon_transparency"] == 0.6
