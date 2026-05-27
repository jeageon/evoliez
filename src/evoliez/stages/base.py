"""Stage contract.

A stage reads artifacts from the context, does its work, writes new artifacts
and (optionally) DB rows / files, and is idempotent: re-running a completed
stage with ``--resume`` is a no-op as long as it can ``load`` its outputs.
"""

from __future__ import annotations

import abc

from evoliez.context import RunContext
from evoliez.logging_utils import get_logger


class PreflightBlocked(Exception):
    """Raised by s01 when ``MDConfig.strict_preflight=True`` AND the MD
    parameterisation preflight returns a blocking status (``unsupported``,
    ``timeout_preflight``, ``failed_preflight``).

    ``Pipeline.run`` catches this and exits cleanly (not as a crash) so a
    project where MD cannot run is recognised as "halted on purpose"
    rather than a stage failure. The Pipeline writes
    ``ctx.persist_meta("preflight_blocked", <reason>)`` and downstream
    stages are NOT executed.

    Distinct from generic RuntimeError so the CLI / harness can
    distinguish "preflight said don't bother" from "stage crashed".
    """

    def __init__(self, status: str, reason: str = "") -> None:
        super().__init__(f"preflight blocked ({status}): {reason}")
        self.status = status
        self.reason = reason


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
