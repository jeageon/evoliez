"""Logging setup. Uses rich if available, falls back to stdlib."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def setup_logging(level: str = "INFO", logfile: Optional[Path] = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handlers: list[logging.Handler] = []
    try:
        from rich.logging import RichHandler

        handlers.append(RichHandler(rich_tracebacks=True, show_path=False))
        fmt = "%(message)s"
    except Exception:  # pragma: no cover - rich is a hard dep but stay safe
        handlers.append(logging.StreamHandler())
        fmt = "%(asctime)s %(levelname)s %(name)s | %(message)s"

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
        )
        handlers.append(fh)

    logging.basicConfig(level=level, format=fmt, datefmt="[%X]", handlers=handlers)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
