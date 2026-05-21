"""Smoke tests for the ``evoliez.figures.html`` template + static assets.

These tests cover the layout layer only (jinja2 syntax, expected anchors,
CSS palette codes, vendor JS marker).  They do not exercise the wave-2
``build_html`` entry point, which composes the templates against a real
discovered run.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Locate the package without importing it (so jinja2-less envs still work).
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "src" / "evoliez" / "figures" / "html"
TEMPLATES = PKG / "templates"
STATIC = PKG / "static"
VENDOR = STATIC / "vendor"


SECTION_PARTIALS = [
    "00_overview.html",
    "01_input.html",
    "02_msa.html",
    "03_boltz_complex.html",
    "04_pose_ensemble.html",
    "05_fingerprint.html",
    "06_mutation.html",
    "07_reranking.html",
    "08_md.html",
    "09_final_library.html",
]


# ---------------------------------------------------------------------------
# Structural checks that do not need jinja2.
# ---------------------------------------------------------------------------
def test_package_layout_exists():
    assert (PKG / "__init__.py").is_file()
    assert (TEMPLATES / "base.html").is_file()
    assert (TEMPLATES / "components").is_dir()
    assert (TEMPLATES / "partials").is_dir()
    assert (STATIC / "report.css").is_file()
    assert (STATIC / "report.js").is_file()
    assert (VENDOR / "README.md").is_file()


def test_all_ten_section_partials_present():
    found = sorted(p.name for p in (TEMPLATES / "partials").glob("*.html"))
    assert found == sorted(SECTION_PARTIALS), found


def test_components_present():
    for name in ("figure_card.html", "datatable.html", "threed_viewer.html", "toc.html"):
        assert (TEMPLATES / "components" / name).is_file(), name


def test_css_contains_wong_palette():
    css = (STATIC / "report.css").read_text()
    for hex_code in (
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ):
        assert hex_code in css, f"missing Wong color {hex_code}"


def test_css_print_rules_present():
    css = (STATIC / "report.css").read_text()
    assert "@media print" in css
    assert "page-break-inside" in css


def test_js_is_iife_and_has_datatable_hooks():
    js = (STATIC / "report.js").read_text()
    assert js.lstrip().startswith("/*") or js.lstrip().startswith("(")
    assert "IntersectionObserver" in js
    assert "datatable" in js.lower()
    # No ES module syntax, no jQuery
    assert "import " not in js
    assert "require(" not in js
    assert "$(" not in js


def test_toc_lists_all_sections():
    toc = (TEMPLATES / "components" / "toc.html").read_text()
    for anchor in (
        "#00-overview",
        "#01-input",
        "#02-msa",
        "#03-boltz-complex",
        "#04-pose-ensemble",
        "#05-fingerprint",
        "#06-mutation",
        "#07-reranking",
        "#08-md",
        "#09-final-library",
    ):
        assert anchor in toc, f"TOC missing {anchor}"


def test_partial_section_ids_present():
    """Each partial must declare its expected section anchor."""
    expected = {
        "00_overview.html": '"00-overview"',
        "01_input.html": '"01-input"',
        "02_msa.html": '"02-msa"',
        "03_boltz_complex.html": '"03-boltz-complex"',
        "04_pose_ensemble.html": '"04-pose-ensemble"',
        "05_fingerprint.html": '"05-fingerprint"',
        "06_mutation.html": '"06-mutation"',
        "07_reranking.html": '"07-reranking"',
        "08_md.html": '"08-md"',
        "09_final_library.html": '"09-final-library"',
    }
    for name, marker in expected.items():
        body = (TEMPLATES / "partials" / name).read_text()
        assert marker in body, f"{name} is missing anchor id {marker}"


def test_vendor_marker_or_real_bundle():
    """Vendor 3Dmol bundle is either a real download or the placeholder."""
    js = VENDOR / "3Dmol-min.js"
    if not js.exists():
        pytest.skip("3Dmol-min.js not present; placeholder mode")
    head = js.read_text(errors="ignore")[:400]
    if "3Dmol.csb.pitt.edu" in head and "download" in head.lower():
        # Placeholder header - tolerated.
        return
    # Real bundle: must mention 3Dmol or be a substantial JS file.
    assert js.stat().st_size > 10_000, "3Dmol-min.js looks truncated"


# ---------------------------------------------------------------------------
# jinja2-driven render tests (skip cleanly when jinja2 is unavailable).
# ---------------------------------------------------------------------------
def _env():
    jinja2 = pytest.importorskip("jinja2")
    # Two search roots: `templates/` for the report templates, and `static/`'s
    # parent (the package root) so `{% include 'static/report.css' %}` in
    # base.html can resolve from the same Environment.
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader([str(TEMPLATES), str(PKG)]),
        autoescape=jinja2.select_autoescape(["html"]),
    )


def test_jinja_env_parses_all_templates():
    env = _env()
    names = [
        "base.html",
        "components/figure_card.html",
        "components/datatable.html",
        "components/threed_viewer.html",
        "components/toc.html",
    ] + [f"partials/{p}" for p in SECTION_PARTIALS]
    for name in names:
        env.get_template(name)  # parses on load; raises on syntax error


def test_base_renders_minimum_context():
    env = _env()
    tmpl = env.get_template("base.html")
    html = tmpl.render(
        run={
            "title": "Smoke run",
            "subtitle": "unit test",
            "mock_backend": False,
            "evoliez_version": "0.0.0",
            "git_sha": "deadbeefcafe1234",
            "generated_at": "2026-01-01T00:00:00Z",
            "config_sha1": "abc",
            "target": "TgtX",
            "ligand": "C(=O)O",
            "backend": "boltz",
            "started": "2026-01-01T00:00:00Z",
            "finished": "2026-01-01T01:00:00Z",
            "catalytic_targets": ["D222"],
        },
        manifest={"warnings": []},
        section={"figures": [], "tables": [], "viewers": [], "notes": []},
    )
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert 'id="00-overview"' in html
    assert 'id="09-final-library"' in html
    assert "Smoke run" in html
    # Sidebar TOC is included
    assert 'class="toc"' in html
    # Footer provenance
    assert "deadbeefcafe" in html


def test_base_mock_backend_banner():
    env = _env()
    html = env.get_template("base.html").render(
        run={"title": "Mock run", "mock_backend": True},
        manifest={"warnings": []},
        section={"figures": [], "tables": [], "viewers": [], "notes": []},
    )
    assert "mock-backend-banner" in html
    assert "illustrative" in html.lower()


def test_figure_card_image_video_iframe():
    env = _env()
    tmpl = env.get_template("components/figure_card.html")
    png = tmpl.render(
        fig={
            "figure_id": "test_png",
            "title": "PNG figure",
            "description": "demo",
            "path": "figures/test.png",
            "source_files": ["data/raw.csv"],
            "renderer": "matplotlib",
            "generated_at": "2026-01-01T00:00:00Z",
            "params": {"dpi": 200},
        }
    )
    assert "<img" in png and "test.png" in png
    assert "<video" not in png

    mp4 = tmpl.render(
        fig={
            "figure_id": "test_mp4",
            "title": "MP4",
            "description": "demo",
            "path": "movies/test.mp4",
            "source_files": [],
            "renderer": "ffmpeg",
            "poster": "movies/test.png",
        }
    )
    assert "<video" in mp4 and "test.mp4" in mp4
    assert 'poster="movies/test.png"' in mp4

    html_frag = tmpl.render(
        fig={
            "figure_id": "interactive",
            "title": "Plotly",
            "description": "demo",
            "path": "interactive/plot.html",
            "source_files": [],
            "renderer": "plotly",
        }
    )
    assert "<iframe" in html_frag and "plot.html" in html_frag


def test_threed_viewer_embeds_pdb():
    env = _env()
    pdb_text = "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00  0.00           N\n"
    html = env.get_template("components/threed_viewer.html").render(
        viewer={
            "id": "wt",
            "pdb_text": pdb_text,
            "height": 360,
            "highlight_residues": [222, "100A"],
        }
    )
    assert 'id="viewer-wt"' in html
    assert 'id="pdb-wt"' in html
    assert 'type="text/plain"' in html
    # jinja2 autoescape replaces spaces? no -- but it escapes <, > etc.
    # The PDB text we embedded contains no markup chars, so it survives.
    assert "ALA A" in html
    assert "$3Dmol.createViewer" in html
    assert "[222, &#34;100A&#34;]" in html or "[222, \"100A\"]" in html


def test_datatable_renders_rows_and_evidence_class():
    env = _env()
    html = env.get_template("components/datatable.html").render(
        table={
            "id": "cands",
            "columns": ["mutant", "score"],
            "rows": [
                {"mutant": "A1B", "score": 0.9, "evidence_class": "strong"},
                {"mutant": "C2D", "score": 0.3, "evidence_class": "reject"},
            ],
        }
    )
    assert 'id="table-cands"' in html
    assert "evidence-strong" in html
    assert "evidence-reject" in html
    assert "A1B" in html and "C2D" in html


def test_overview_renders_with_warnings_and_callouts():
    env = _env()
    html = env.get_template("partials/00_overview.html").render(
        run={
            "target": "TgtX",
            "ligand": "C",
            "backend": "boltz",
            "git_sha": "abc",
            "catalytic_targets": ["D222", "H145"],
        },
        manifest={"warnings": ["msa shallow", "no benchmark"]},
        section={"figures": []},
    )
    assert "Catalytic target" in html
    assert "D222" in html and "H145" in html
    assert "msa shallow" in html
    assert "No curated benchmark for this target" in html
