"""Tests for ``evoliez.figures.packaging``.

Covers the three sub-modules:

* :mod:`evoliez.figures.packaging.bundler`        -- ZIP archive
* :mod:`evoliez.figures.packaging.selfcontained`  -- single-file HTML
* :mod:`evoliez.figures.packaging.watermark`      -- mock banner + PNG stamp

The PNG-stamp test uses Pillow when available; everything else runs on
the bare laptop venv with stdlib only.
"""

from __future__ import annotations

import base64
import json
import re
import struct
import sys
import types
import zipfile
import zlib
from pathlib import Path

import pytest

from evoliez.figures.packaging import (
    apply_mock_watermark,
    build_self_contained,
    build_zip,
    stamp_figures_with_mock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_real_png(path: Path, color=(255, 0, 0)) -> None:
    """Write a 1x1 PNG that Pillow can actually open."""
    # PNG signature + IHDR + IDAT + IEND, hand-rolled.
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1 RGB
    raw = b"\x00" + bytes(color)  # filter byte + one RGB pixel
    idat = zlib.compress(raw)
    path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def _make_report(tmp_path: Path, *, with_movie: bool = False) -> Path:
    """Synthesise a minimal report directory matching the builder layout."""
    report_dir = tmp_path / "reports"
    (report_dir / "figures").mkdir(parents=True)
    (report_dir / "static").mkdir()
    (report_dir / "pdb").mkdir()
    if with_movie:
        (report_dir / "movies").mkdir()
        (report_dir / "movies" / "rmsd.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)

    _make_real_png(report_dir / "figures" / "test.png")
    (report_dir / "static" / "report.css").write_text("body{font-family:sans-serif}", encoding="utf-8")
    (report_dir / "static" / "report.js").write_text("console.log('ok')", encoding="utf-8")
    (report_dir / "pdb" / "wt.pdb").write_text("HEADER  TEST\nEND\n", encoding="utf-8")

    html = (
        "<!doctype html><html><head>"
        "<link rel=\"stylesheet\" href=\"static/report.css\">"
        "</head><body>"
        "<h1>Run</h1>"
        "<img src=\"figures/test.png\" alt=\"t\">"
        "<script src=\"static/report.js\"></script>"
        "</body></html>"
    )
    if with_movie:
        html = html.replace(
            "<script",
            "<video src=\"movies/rmsd.mp4\" controls></video><script",
        )
    (report_dir / "visual_report.html").write_text(html, encoding="utf-8")
    (report_dir / "visual_manifest.json").write_text(
        json.dumps({"figures": [{"path": "figures/test.png", "figure_id": "test"}]}),
        encoding="utf-8",
    )
    return report_dir


# ---------------------------------------------------------------------------
# bundler.build_zip
# ---------------------------------------------------------------------------


def test_build_zip_creates_archive(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path)
    zip_path = build_zip(report_dir, tmp_path / "report.zip")
    assert zip_path.exists()
    assert zip_path == tmp_path / "report.zip"
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "visual_report.html" in names
    assert "figures/test.png" in names
    assert "static/report.css" in names
    assert "visual_manifest.json" in names


def test_build_zip_skips_movies_when_disabled(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path, with_movie=True)
    zip_path = build_zip(report_dir, tmp_path / "report.zip", include_movies=False)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert not any(n.startswith("movies/") for n in names), names
    # static / figures still present.
    assert "figures/test.png" in names


def test_build_zip_includes_movies_by_default(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path, with_movie=True)
    zip_path = build_zip(report_dir, tmp_path / "report.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "movies/rmsd.mp4" in names


def test_build_zip_skips_static_when_disabled(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path)
    zip_path = build_zip(report_dir, tmp_path / "report.zip", include_static=False)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert not any(n.startswith("static/") for n in names), names


def test_build_zip_missing_entry_html_raises(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        build_zip(report_dir, tmp_path / "report.zip")


def test_build_zip_rejects_manifest_missing_figure(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path)
    (report_dir / "visual_manifest.json").write_text(
        json.dumps({"figures": [{"path": "figures/does_not_exist.png"}]}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        build_zip(report_dir, tmp_path / "report.zip")


def test_build_zip_creates_parent_dirs(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path)
    out = tmp_path / "nested" / "deeper" / "report.zip"
    zip_path = build_zip(report_dir, out)
    assert zip_path.exists()


# ---------------------------------------------------------------------------
# selfcontained.build_self_contained
# ---------------------------------------------------------------------------


def test_self_contained_inlines_images(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path)
    out = tmp_path / "single.html"
    build_self_contained(report_dir, out)
    html = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    # The original relative href is gone.
    assert "figures/test.png" not in html


def test_self_contained_inlines_css_js(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path)
    out = tmp_path / "single.html"
    build_self_contained(report_dir, out)
    html = out.read_text(encoding="utf-8")

    # CSS contents inlined inside <style>.
    assert "<style>" in html
    assert "font-family:sans-serif" in html
    # The original <link> stylesheet ref is gone.
    assert "static/report.css" not in html

    # JS contents inlined inside <script>.
    assert "console.log('ok')" in html
    assert "static/report.js" not in html


def test_self_contained_inlines_video(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path, with_movie=True)
    out = tmp_path / "single.html"
    build_self_contained(report_dir, out)
    html = out.read_text(encoding="utf-8")
    assert "data:video/mp4;base64," in html
    assert "movies/rmsd.mp4" not in html


def test_self_contained_preserves_remote_assets(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path)
    # Add a CDN script that must NOT be inlined.
    entry = report_dir / "visual_report.html"
    html = entry.read_text(encoding="utf-8").replace(
        "</body>",
        "<script src=\"https://cdn.example.com/x.js\"></script></body>",
    )
    entry.write_text(html, encoding="utf-8")

    out = tmp_path / "single.html"
    build_self_contained(report_dir, out)
    text = out.read_text(encoding="utf-8")
    assert "https://cdn.example.com/x.js" in text


def test_self_contained_inlines_iframe(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path)
    (report_dir / "figures" / "frag.html").write_text(
        "<p>fragment</p>", encoding="utf-8"
    )
    entry = report_dir / "visual_report.html"
    html = entry.read_text(encoding="utf-8").replace(
        "</body>",
        "<iframe src=\"figures/frag.html\"></iframe></body>",
    )
    entry.write_text(html, encoding="utf-8")

    out = tmp_path / "single.html"
    build_self_contained(report_dir, out)
    text = out.read_text(encoding="utf-8")
    assert "data:text/html;base64," in text
    assert "figures/frag.html" not in text


def test_self_contained_missing_html_raises(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        build_self_contained(report_dir, tmp_path / "out.html")


def test_self_contained_missing_asset_left_intact(tmp_path: Path) -> None:
    report_dir = _make_report(tmp_path)
    # Reference a file that doesn't exist.
    entry = report_dir / "visual_report.html"
    html = entry.read_text(encoding="utf-8").replace(
        "<h1>Run</h1>",
        "<h1>Run</h1><img src=\"figures/missing.png\">",
    )
    entry.write_text(html, encoding="utf-8")

    out = tmp_path / "single.html"
    build_self_contained(report_dir, out)
    text = out.read_text(encoding="utf-8")
    # Existing PNG still inlined.
    assert "data:image/png;base64," in text
    # Missing reference left alone (warning logged).
    assert "figures/missing.png" in text


# ---------------------------------------------------------------------------
# watermark.apply_mock_watermark
# ---------------------------------------------------------------------------


def test_apply_mock_watermark_idempotent() -> None:
    html = "<html><body>...</body></html>"
    once = apply_mock_watermark(html)
    twice = apply_mock_watermark(once)
    assert once == twice
    assert "MOCK" in once.upper()


def test_apply_mock_watermark_adds_banner() -> None:
    html = "<html><body><h1>Run</h1></body></html>"
    out = apply_mock_watermark(html)
    assert "MOCK" in out.upper()
    assert "<h1>Run</h1>" in out
    # Banner sits between <body> and the first child.
    body_idx = out.lower().index("<body>")
    h1_idx = out.index("<h1>Run</h1>")
    mock_idx = out.upper().index("MOCK")
    assert body_idx < mock_idx < h1_idx


def test_apply_mock_watermark_no_body_prepended() -> None:
    fragment = "<div>just a fragment</div>"
    out = apply_mock_watermark(fragment)
    assert "MOCK" in out.upper()
    assert fragment in out


def test_apply_mock_watermark_body_with_attributes() -> None:
    html = "<html><body class=\"x\" data-y='1'><p>hi</p></body></html>"
    out = apply_mock_watermark(html)
    assert "MOCK" in out.upper()
    assert "<p>hi</p>" in out


# ---------------------------------------------------------------------------
# watermark.stamp_figures_with_mock
# ---------------------------------------------------------------------------


def test_stamp_figures_with_mock_pillow_missing(monkeypatch, tmp_path: Path) -> None:
    """When Pillow import fails we log + return; no exception raised."""
    figures = tmp_path / "figures"
    figures.mkdir()
    _make_real_png(figures / "x.png")

    from evoliez.figures.packaging import watermark as wm

    monkeypatch.setattr(wm, "_try_import_pillow", lambda: None)
    original_bytes = (figures / "x.png").read_bytes()

    # Must not raise.
    stamp_figures_with_mock(figures)

    # File untouched.
    assert (figures / "x.png").read_bytes() == original_bytes


def test_stamp_figures_with_mock_non_directory(tmp_path: Path) -> None:
    # Must not raise on a non-existent dir.
    stamp_figures_with_mock(tmp_path / "does_not_exist")


def test_stamp_figures_with_mock_real_pillow(tmp_path: Path) -> None:
    """Actual Pillow stamp -- skipped when Pillow isn't installed."""
    pytest.importorskip("PIL")

    figures = tmp_path / "figures"
    figures.mkdir()
    # Make a slightly larger PNG so the stamp has room.
    from PIL import Image  # noqa: WPS433 -- guarded by importorskip
    Image.new("RGB", (256, 256), (255, 255, 255)).save(figures / "big.png")
    original_size = (figures / "big.png").stat().st_size

    stamp_figures_with_mock(figures)

    # File modified (size changed) and contains our PNG metadata marker.
    new_size = (figures / "big.png").stat().st_size
    assert new_size != original_size

    with Image.open(figures / "big.png") as im:
        assert im.info.get("evoliez_mock_stamp") == "1"

    # Second call is idempotent: PNG bytes unchanged.
    before = (figures / "big.png").read_bytes()
    stamp_figures_with_mock(figures)
    after = (figures / "big.png").read_bytes()
    assert before == after
