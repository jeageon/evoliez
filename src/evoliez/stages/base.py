"""Stage contract.

A stage reads artifacts from the context, does its work, writes new artifacts
and (optionally) DB rows / files, and is idempotent: re-running a completed
stage with ``--resume`` is a no-op as long as it can ``load`` its outputs.
"""

from __future__ import annotations

import abc

from evoliez.context import RunContext
from evoliez.logging_utils import get_logger


class Stage(abc.ABC):
    name: str = "stage"

    def __init__(self) -> None:
        self.log = get_logger(f"evoliez.{self.name}")

    @abc.abstractmethod
    def run(self, ctx: RunContext) -> None:
        ...

    def load(self, ctx: RunContext) -> bool:
        """Reconstruct this stage's artifacts from disk on resume.

        Return True if the downstream-required artifacts are now available,
        False to force a re-run. Default: cannot reload -> re-run.
        """
        return False

    def backend(self, ctx: RunContext):
        return ctx.config.backend_for(self.name)
