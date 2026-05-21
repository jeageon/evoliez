"""Builder for ``reports/visual_manifest.json``.

The manifest is the single source of truth the HTML templates iterate over:
each :class:`~evoliez.figures.types.FigureSpec` becomes one entry under the
``"figures"`` array, alongside provenance metadata (evoliez version, git
SHA, config fingerprint) so a downloaded report bundle stays
self-describing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from evoliez.figures.types import FigureSpec

SCHEMA_VERSION = "1.0"


class ManifestBuilder:
    """Accumulate :class:`FigureSpec` records and emit ``visual_manifest.json``."""

    def __init__(
        self,
        run_dir: Path,
        style: str,
        mock_backend: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.style = style
        self.mock_backend = mock_backend
        self.figures: List[FigureSpec] = []
        self.warnings: List[str] = []
        self.git_sha: str = self._detect_git_sha()
        self.config_sha1: str = self._detect_config_sha1()
        self.generated_at: str = self._utc_now()

    # -- mutation ---------------------------------------------------------

    def add_figure(self, spec: FigureSpec) -> None:
        """Append a figure; auto-stamp ``generated_at`` if missing."""
        if spec.generated_at is None:
            spec.generated_at = self._utc_now()
        self.figures.append(spec)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    # -- output -----------------------------------------------------------

    def write(self, output_path: Path) -> Path:
        """Serialize the manifest to ``output_path`` and return that path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        output_path.write_text(json.dumps(payload, indent=2, default=str))
        return output_path

    def to_dict(self) -> Dict[str, Any]:
        """Return the manifest as a JSON-serializable dict."""
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "evoliez_version": _evoliez_version(),
            "git_sha": self.git_sha,
            "config_sha1": self.config_sha1,
            "style": self.style,
            "mock_backend": self.mock_backend,
            "warnings": list(self.warnings),
            "figures": [_figure_to_json(f) for f in self.figures],
        }

    # -- provenance helpers ----------------------------------------------

    @staticmethod
    def _detect_git_sha() -> str:
        """Short git SHA for the working tree, or ``"unknown"`` outside a repo."""
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return "unknown"
        if out.returncode != 0:
            return "unknown"
        return out.stdout.strip() or "unknown"

    def _detect_config_sha1(self) -> str:
        """SHA-1 of the run's ``_state.json`` if present, else ``"unknown"``.

        ``_state.json`` already embeds the resolved config + stage
        fingerprints, so hashing it gives us a stable identifier for "the
        exact run this report was built from" without re-serializing the
        config ourselves.
        """
        state = self.run_dir / "_state.json"
        if not state.exists():
            return "unknown"
        try:
            data = state.read_bytes()
        except OSError:
            return "unknown"
        return hashlib.sha1(data).hexdigest()[:8]

    @staticmethod
    def _utc_now() -> str:
        # Drop microseconds and use a trailing 'Z' for the standard
        # ISO-8601 UTC form templates expect.
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _evoliez_version() -> str:
    try:
        from evoliez import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _figure_to_json(spec: FigureSpec) -> Dict[str, Any]:
    """``FigureSpec`` -> JSON-safe dict (Paths become posix strings)."""
    raw = asdict(spec)
    raw["path"] = Path(raw["path"]).as_posix()
    raw["source_files"] = [Path(p).as_posix() for p in raw.get("source_files", [])]
    return raw
