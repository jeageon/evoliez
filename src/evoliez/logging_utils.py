"""Logging setup. Uses rich if available, falls back to stdlib."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_FILE_ATTACHED = False


def setup_logging(level: str = "INFO", logfile: Optional[Path] = None) -> None:
    """Configure console logging once; attach a file handler when a logfile is
    first supplied. cli.run() calls this early with no logfile, then
    RunContext.setup() calls it again WITH logs/pipeline.log - the second call
    must still attach the FileHandler (previously the _CONFIGURED short-circuit
    swallowed it, leaving pipeline.log empty for every server run)."""
    global _CONFIGURED, _FILE_ATTACHED
    if not _CONFIGURED:
        handlers: list[logging.Handler] = []
        try:
            from rich.logging import RichHandler

            handlers.append(RichHandler(rich_tracebacks=True, show_path=False))
            fmt = "%(message)s"
        except Exception:  # pragma: no cover - rich is a hard dep but stay safe
            handlers.append(logging.StreamHandler())
            fmt = "%(asctime)s %(levelname)s %(name)s | %(message)s"
        logging.basicConfig(level=level, format=fmt, datefmt="[%X]",
                            handlers=handlers)
        _CONFIGURED = True

    if logfile is not None and not _FILE_ATTACHED:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
        )
        logging.getLogger().addHandler(fh)
        _FILE_ATTACHED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
