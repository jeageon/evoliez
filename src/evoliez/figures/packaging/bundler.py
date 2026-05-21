"""ZIP bundler for the linked-mode visual report.

The HTML builder produces a directory like::

    report_dir/
        visual_report.html           <- entry point
        visual_manifest.json
        figures/*.png|.svg
        movies/*.mp4
        static/report.css
        static/report.js
        static/vendor/3Dmol-min.js
        pdb/wt.pdb
        pdb/mut_*.pdb

:func:`build_zip` walks that tree and packs it into a single ``.zip``.
It verifies every file referenced by ``visual_manifest.json``'s
``figures`` list exists before writing the archive so we don't ship a
broken report.  Movies (often the dominant size) can be excluded with
``include_movies=False``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List, Optional

from evoliez.logging_utils import get_logger

_LOG = get_logger(__name__)

# Warn-only soft limit -- emails / Slack typically reject anything past this.
_LARGE_ZIP_BYTES = 200 * 1024 * 1024  # 200 MB


def _iter_report_files(
    report_dir: Path,
    *,
    include_movies: bool,
    include_static: bool,
) -> List[Path]:
    """Return every file inside ``report_dir`` that should land in the zip.

    Ordering is deterministic (sorted) so two zips built from the same
    directory hash identically.
    """
    out: List[Path] = []
    for path in sorted(report_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(report_dir)
        top = rel.parts[0] if rel.parts else ""
        if top == "movies" and not include_movies:
            continue
        if top == "static" and not include_static:
            continue
        out.append(path)
    return out


def _verify_manifest_figures(report_dir: Path) -> None:
    """Best-effort sanity check on ``visual_manifest.json``.

    Missing manifest -> no-op (the builder may not have written one in
    some test fixtures).  Present manifest with a missing referenced
    figure -> ``FileNotFoundError``.
    """
    manifest_path = report_dir / "visual_manifest.json"
    if not manifest_path.exists():
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"visual_manifest.json is not valid JSON: {exc}") from exc

    figures = data.get("figures") or []
    for entry in figures:
        # FigureSpec uses ``path`` (rel to report_dir).  Accept either key.
        rel = entry.get("path") if isinstance(entry, dict) else None
        if not rel:
            continue
        target = report_dir / rel
        if not target.exists():
            raise FileNotFoundError(
                f"visual_manifest.json references missing figure: {rel}"
            )


def build_zip(
    report_dir: Path,
    output_zip: Path,
    *,
    include_movies: bool = True,
    include_static: bool = True,
    compression: int = zipfile.ZIP_DEFLATED,
) -> Path:
    """Bundle a finished report directory into a single portable ZIP.

    Parameters
    ----------
    report_dir
        Directory produced by the HTML builder; must contain at least
        ``visual_report.html``.
    output_zip
        Destination ``.zip`` path.  Parent directories are created if
        needed and any pre-existing archive at this path is overwritten.
    include_movies
        When ``False``, skips the ``movies/`` subtree (saves megabytes
        when sharing via email / chat).
    include_static
        When ``False``, skips the ``static/`` subtree.  Generally leave
        ``True`` -- self-contained mode is the right answer when you
        want a static-free copy.
    compression
        :mod:`zipfile` compression constant; defaults to
        ``ZIP_DEFLATED``.

    Returns
    -------
    Path
        ``output_zip`` (for chaining).
    """
    report_dir = Path(report_dir)
    output_zip = Path(output_zip)

    if not report_dir.is_dir():
        raise NotADirectoryError(f"report_dir is not a directory: {report_dir}")
    entry = report_dir / "visual_report.html"
    if not entry.exists():
        raise FileNotFoundError(f"visual_report.html not found in {report_dir}")

    _verify_manifest_figures(report_dir)

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    files = _iter_report_files(
        report_dir,
        include_movies=include_movies,
        include_static=include_static,
    )
    total_in = sum(f.stat().st_size for f in files)

    with zipfile.ZipFile(output_zip, "w", compression=compression) as zf:
        for path in files:
            arcname = path.relative_to(report_dir).as_posix()
            zf.write(path, arcname=arcname)

    total_out = output_zip.stat().st_size
    ratio = (total_out / total_in) if total_in else 0.0
    _LOG.info(
        "Built report ZIP %s (%d files, %.1f MB -> %.1f MB, ratio %.2f)",
        output_zip,
        len(files),
        total_in / (1024 * 1024),
        total_out / (1024 * 1024),
        ratio,
    )
    if total_out > _LARGE_ZIP_BYTES:
        _LOG.warning(
            "Report ZIP is %.1f MB (>%d MB) -- consider include_movies=False "
            "or self-contained HTML for sharing.",
            total_out / (1024 * 1024),
            _LARGE_ZIP_BYTES // (1024 * 1024),
        )
    return output_zip
