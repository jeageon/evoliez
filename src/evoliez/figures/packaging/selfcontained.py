"""Self-contained HTML packaging.

Reads ``visual_report.html`` from a linked-mode report directory and
inlines every external asset:

* ``<img src="figures/X.png">``           -> ``data:image/png;base64,...``
* ``<video src="movies/X.mp4">``          -> ``data:video/mp4;base64,...``
* ``<source src="movies/X.mp4">``         -> ``data:video/mp4;base64,...``
* ``<link rel="stylesheet" href="...">``  -> inline ``<style>...</style>``
* ``<script src="...">``                  -> inline ``<script>...</script>``
* ``<iframe src="...">``                  -> ``data:text/html;base64,...``

The result is a single HTML file that opens on any machine without
needing a sibling ``figures/`` or ``static/`` folder.

Implementation notes
--------------------
* No new dependency: we use regex rather than BeautifulSoup.  The HTML
  produced by the builder is well-formed and predictable so regex is
  safe here.
* Only assets referenced by *relative* paths are inlined.  Anything
  whose ``src``/``href`` starts with ``http``, ``https``, ``data:``,
  ``//`` or ``#`` is left untouched (third-party CDN / fragment links).
* If an asset file is missing on disk we leave the tag alone and emit
  a warning rather than failing the whole package -- a single missing
  movie shouldn't sink the report.
"""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Callable, Match, Optional

from evoliez.logging_utils import get_logger

_LOG = get_logger(__name__)

# Warn-only soft limit -- browsers handle larger files fine but email
# attachments / chat uploads typically refuse beyond this.
_LARGE_HTML_BYTES = 50 * 1024 * 1024  # 50 MB

# Schemes that must NOT be rewritten -- already inline or remote.
_SKIP_PREFIXES = ("http://", "https://", "data:", "//", "#", "mailto:", "javascript:")

# Default MIME types for assets matplotlib & 3Dmol typically produce.
_EXTRA_MIMETYPES = {
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".pdb": "chemical/x-pdb",
}


def _guess_mime(path: Path, default: str) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    return _EXTRA_MIMETYPES.get(path.suffix.lower(), default)


def _should_skip(src: str) -> bool:
    s = src.strip()
    if not s:
        return True
    low = s.lower()
    return any(low.startswith(p) for p in _SKIP_PREFIXES)


def _resolve(report_dir: Path, src: str) -> Optional[Path]:
    """Resolve a relative asset reference to a real file on disk.

    Returns ``None`` if the file does not exist (caller logs + skips).
    """
    # Strip query / fragment off the URL ("foo.png?v=2" -> "foo.png").
    clean = src.split("#", 1)[0].split("?", 1)[0]
    candidate = (report_dir / clean).resolve()
    try:
        candidate.relative_to(report_dir.resolve())
    except ValueError:
        # Refuse to inline anything outside the report dir.
        _LOG.warning("Skipping out-of-tree asset reference: %s", src)
        return None
    if not candidate.is_file():
        return None
    return candidate


def _b64_data_url(path: Path, mime_default: str) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = _guess_mime(path, mime_default)
    return f"data:{mime};base64,{payload}"


def _sub_attr(
    pattern: re.Pattern,
    html: str,
    report_dir: Path,
    transform: Callable[[Path, str], str],
) -> str:
    """Generic ``re.sub`` driver for tag/attr replacement.

    ``pattern`` must have a single capture group containing the asset URL.
    ``transform(path, original_src)`` returns the replacement *value*
    (without quotes); the wrapping ``attr="..."`` quoting is preserved.
    """

    def repl(m: Match) -> str:
        full = m.group(0)
        src = m.group(1)
        if _should_skip(src):
            return full
        resolved = _resolve(report_dir, src)
        if resolved is None:
            _LOG.warning("Asset not found on disk, leaving link intact: %s", src)
            return full
        try:
            new_val = transform(resolved, src)
        except OSError as exc:
            _LOG.warning("Failed to inline %s (%s)", src, exc)
            return full
        # Replace just the captured src portion within the match.
        start, end = m.start(1) - m.start(0), m.end(1) - m.start(0)
        return full[:start] + new_val + full[end:]

    return pattern.sub(repl, html)


# Tag patterns -------------------------------------------------------------
# We match ``src=`` / ``href=`` attributes with either single or double
# quotes.  ``[^"'>\s]+`` keeps the capture conservative.

_IMG_RE = re.compile(
    r"""<img\b[^>]*?\bsrc\s*=\s*['"]([^'"]+)['"][^>]*>""",
    re.IGNORECASE,
)
_VIDEO_RE = re.compile(
    r"""<video\b[^>]*?\bsrc\s*=\s*['"]([^'"]+)['"][^>]*>""",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(
    r"""<source\b[^>]*?\bsrc\s*=\s*['"]([^'"]+)['"][^>]*/?>""",
    re.IGNORECASE,
)
_IFRAME_RE = re.compile(
    r"""<iframe\b[^>]*?\bsrc\s*=\s*['"]([^'"]+)['"][^>]*>""",
    re.IGNORECASE,
)
_LINK_CSS_RE = re.compile(
    r"""<link\b[^>]*?\brel\s*=\s*['"]stylesheet['"][^>]*?\bhref\s*=\s*['"]([^'"]+)['"][^>]*/?>""",
    re.IGNORECASE,
)
_LINK_CSS_RE_REV = re.compile(
    r"""<link\b[^>]*?\bhref\s*=\s*['"]([^'"]+)['"][^>]*?\brel\s*=\s*['"]stylesheet['"][^>]*/?>""",
    re.IGNORECASE,
)
_SCRIPT_SRC_RE = re.compile(
    r"""<script\b[^>]*?\bsrc\s*=\s*['"]([^'"]+)['"][^>]*>\s*</script>""",
    re.IGNORECASE,
)


def _inline_link_css(html: str, report_dir: Path) -> str:
    def repl(m: Match) -> str:
        href = m.group(1)
        if _should_skip(href):
            return m.group(0)
        resolved = _resolve(report_dir, href)
        if resolved is None:
            _LOG.warning("Stylesheet not found, leaving link intact: %s", href)
            return m.group(0)
        try:
            css = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            _LOG.warning("Failed to read stylesheet %s (%s)", href, exc)
            return m.group(0)
        return f"<style>\n{css}\n</style>"

    html = _LINK_CSS_RE.sub(repl, html)
    html = _LINK_CSS_RE_REV.sub(repl, html)
    return html


def _inline_scripts(html: str, report_dir: Path) -> str:
    def repl(m: Match) -> str:
        src = m.group(1)
        if _should_skip(src):
            return m.group(0)
        resolved = _resolve(report_dir, src)
        if resolved is None:
            _LOG.warning("Script not found, leaving link intact: %s", src)
            return m.group(0)
        try:
            js = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            _LOG.warning("Failed to read script %s (%s)", src, exc)
            return m.group(0)
        # Keep any non-src attributes off the original tag (e.g. defer).
        # Simplest robust approach: drop attributes; inline scripts don't
        # need them, and we know what the builder writes.
        return f"<script>\n{js}\n</script>"

    return _SCRIPT_SRC_RE.sub(repl, html)


def _inline_images(html: str, report_dir: Path) -> str:
    return _sub_attr(
        _IMG_RE,
        html,
        report_dir,
        lambda p, _src: _b64_data_url(p, "image/png"),
    )


def _inline_videos(html: str, report_dir: Path) -> str:
    html = _sub_attr(
        _VIDEO_RE,
        html,
        report_dir,
        lambda p, _src: _b64_data_url(p, "video/mp4"),
    )
    html = _sub_attr(
        _SOURCE_RE,
        html,
        report_dir,
        lambda p, _src: _b64_data_url(p, "video/mp4"),
    )
    return html


def _inline_iframes(html: str, report_dir: Path) -> str:
    def transform(path: Path, _src: str) -> str:
        body = path.read_bytes()
        payload = base64.b64encode(body).decode("ascii")
        return f"data:text/html;base64,{payload}"

    return _sub_attr(_IFRAME_RE, html, report_dir, transform)


def build_self_contained(
    report_dir: Path,
    output_html: Path,
) -> Path:
    """Inline every asset in ``visual_report.html`` into a single file.

    Parameters
    ----------
    report_dir
        A linked-mode report directory (produced by the HTML builder).
        Must contain ``visual_report.html``.
    output_html
        Destination ``.html`` path.  Parent directories are created if
        needed; any existing file is overwritten.

    Returns
    -------
    Path
        ``output_html`` (for chaining).

    Warning
    -------
    Self-contained HTML can balloon to 50-200 MB if the run includes
    many figures and movies.  The caller should size-check the result
    before sharing via email.
    """
    report_dir = Path(report_dir)
    output_html = Path(output_html)

    entry = report_dir / "visual_report.html"
    if not entry.exists():
        raise FileNotFoundError(f"visual_report.html not found in {report_dir}")

    html = entry.read_text(encoding="utf-8")

    # Order matters only insofar as CSS / JS happen before the image
    # inliner so that we never accidentally inline a giant base64 blob
    # twice.  In practice the regexes are tag-disjoint so any order works.
    html = _inline_link_css(html, report_dir)
    html = _inline_scripts(html, report_dir)
    html = _inline_iframes(html, report_dir)  # before images: iframe srcs aren't image-y but be explicit
    html = _inline_images(html, report_dir)
    html = _inline_videos(html, report_dir)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")

    size = output_html.stat().st_size
    _LOG.info(
        "Wrote self-contained HTML %s (%.1f MB)",
        output_html,
        size / (1024 * 1024),
    )
    if size > _LARGE_HTML_BYTES:
        _LOG.warning(
            "Self-contained HTML is %.1f MB (>%d MB) -- prefer linked ZIP for sharing.",
            size / (1024 * 1024),
            _LARGE_HTML_BYTES // (1024 * 1024),
        )
    return output_html
