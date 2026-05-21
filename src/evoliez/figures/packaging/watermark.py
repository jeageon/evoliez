"""Mock-backend safety watermarks.

When a report is generated from a mock backend (synthetic Boltz scores,
fake MD, etc.) it must not be mistaken for real data.  Two complementary
helpers add a clear warning:

* :func:`apply_mock_watermark` injects a red banner at the top of
  ``<body>`` in the report HTML.
* :func:`stamp_figures_with_mock` overlays a diagonal "MOCK -- NOT FOR
  PUBLICATION" string on every PNG in ``figures/`` (Pillow-based;
  gracefully no-ops when Pillow is missing).

Both are idempotent: calling them on already-stamped output produces no
further change.
"""

from __future__ import annotations

import re
from pathlib import Path

from evoliez.logging_utils import get_logger

_LOG = get_logger(__name__)

# Marker comment we use to detect "already watermarked".  Keep it stable
# and unique so future renames don't break idempotency.
_BANNER_MARKER = "<!-- evoliez:mock-banner -->"

_BANNER_HTML = (
    _BANNER_MARKER
    + "\n<div style=\""
    "background:#c0392b;color:#ffffff;"
    "padding:0.75em 1em;"
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
    "font-weight:700;font-size:1rem;"
    "text-align:center;letter-spacing:0.05em;"
    "border-bottom:3px solid #7b1d12;"
    "position:sticky;top:0;z-index:9999;\""
    " role=\"alert\" aria-label=\"Mock data warning\">"
    "MOCK DATA -- generated from mock backends. NOT FOR PUBLICATION."
    "</div>\n"
)

# PNG stamp parameters.
_STAMP_TEXT = "MOCK -- NOT FOR PUBLICATION"
_STAMP_OPACITY = int(255 * 0.30)  # 30 %
_STAMP_COLOR = (192, 57, 43, _STAMP_OPACITY)  # deep red, transparent
_STAMP_MARKER_KEY = "evoliez_mock_stamp"  # PNG metadata key for idempotency


# ---------------------------------------------------------------------------
# HTML banner
# ---------------------------------------------------------------------------


def apply_mock_watermark(html_text: str) -> str:
    """Inject a red MOCK DATA banner immediately after ``<body>``.

    Idempotent: re-running on output containing :data:`_BANNER_MARKER`
    returns the input unchanged.

    If no ``<body>`` tag is found we prepend the banner to the document
    (best-effort -- bare fragments still get the warning).
    """
    if _BANNER_MARKER in html_text:
        return html_text

    # Match <body ...> with or without attributes; case-insensitive.
    body_re = re.compile(r"<body\b[^>]*>", re.IGNORECASE)
    match = body_re.search(html_text)
    if match is None:
        return _BANNER_HTML + html_text
    insert_at = match.end()
    return html_text[:insert_at] + "\n" + _BANNER_HTML + html_text[insert_at:]


# ---------------------------------------------------------------------------
# PNG stamping
# ---------------------------------------------------------------------------


def _try_import_pillow():
    """Return ``(Image, ImageDraw, ImageFont, PngInfo)`` or ``None``.

    Deferred import so test environments without Pillow still load this
    module.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        from PIL.PngImagePlugin import PngInfo  # type: ignore
    except Exception:  # pragma: no cover - exercised via monkeypatch
        return None
    return Image, ImageDraw, ImageFont, PngInfo


def _stamp_one_png(path: Path, pillow) -> bool:
    """Overlay the MOCK watermark on a single PNG.

    Returns ``True`` if the file was modified, ``False`` if skipped
    (already stamped, unreadable, etc.).
    """
    Image, ImageDraw, ImageFont, PngInfo = pillow

    try:
        with Image.open(path) as im:
            existing = (im.info or {}).get(_STAMP_MARKER_KEY)
            if existing:
                return False
            base = im.convert("RGBA")
    except Exception as exc:
        _LOG.warning("Could not open %s for watermarking: %s", path, exc)
        return False

    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Pick a font size proportional to the shorter dimension; fall back
    # to default bitmap font when truetype isn't available.
    short = min(base.size)
    target_pt = max(12, short // 18)
    font = None
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
        "Helvetica-Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, target_pt)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Measure text -- support old + new Pillow APIs.
    try:
        bbox = draw.textbbox((0, 0), _STAMP_TEXT, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:  # pragma: no cover - very old Pillow
        text_w, text_h = draw.textsize(_STAMP_TEXT, font=font)

    # Render text on its own layer so we can rotate it before compositing.
    pad = max(8, target_pt // 2)
    text_layer = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)
    text_draw.text((pad, pad), _STAMP_TEXT, font=font, fill=_STAMP_COLOR)
    rotated = text_layer.rotate(30, resample=Image.BICUBIC, expand=True)

    # Centre the rotated stamp on the image.
    px = (base.size[0] - rotated.size[0]) // 2
    py = (base.size[1] - rotated.size[1]) // 2
    overlay.paste(rotated, (px, py), rotated)

    combined = Image.alpha_composite(base, overlay)

    # Preserve original mode when possible (matplotlib outputs RGBA so
    # keeping RGBA is fine).
    out = combined
    if path.suffix.lower() == ".png":
        meta = PngInfo()
        meta.add_text(_STAMP_MARKER_KEY, "1")
        out.save(path, format="PNG", pnginfo=meta)
    else:
        out.save(path)
    return True


def stamp_figures_with_mock(figures_dir: Path) -> None:
    """Overlay a MOCK watermark on every PNG in ``figures_dir``.

    Uses Pillow.  If Pillow is not installed we log a single warning
    and return without raising -- the HTML banner from
    :func:`apply_mock_watermark` is still in place, so the report stays
    safe.

    Idempotent: a PNG metadata marker (:data:`_STAMP_MARKER_KEY`) is
    written on first stamp and checked on subsequent calls.
    """
    figures_dir = Path(figures_dir)
    if not figures_dir.is_dir():
        _LOG.warning("stamp_figures_with_mock: not a directory: %s", figures_dir)
        return

    pillow = _try_import_pillow()
    if pillow is None:
        _LOG.warning(
            "Pillow is not installed; PNG watermarking skipped. "
            "The HTML banner from apply_mock_watermark still applies."
        )
        return

    stamped = 0
    skipped = 0
    for png in sorted(figures_dir.rglob("*.png")):
        if _stamp_one_png(png, pillow):
            stamped += 1
        else:
            skipped += 1
    _LOG.info(
        "stamp_figures_with_mock: %d stamped, %d skipped (already-stamped or unreadable)",
        stamped,
        skipped,
    )
