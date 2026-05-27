"""Stage-DAG orchestrator with checkpoint/resume (spec section 18.1)."""

from __future__ import annotations

import os
import time
from typing import List, Optional

from evoliez.context import RunContext
from evoliez.logging_utils import get_logger
from evoliez.stages import ALL_STAGES
from evoliez.stages.base import PreflightBlocked, Stage

log = get_logger("evoliez.pipeline")


class Pipeline:
    def __init__(self, stages: Optional[List[type[Stage]]] = None):
        self.stages: List[Stage] = [s() for s in (stages or ALL_STAGES)]

    def stage_names(self) -> List[str]:
        return [s.name for s in self.stages]

    def run(
        self,
        ctx: RunContext,
        *,
        resume: bool = False,
        from_stage: Optional[str] = None,
        to_stage: Optional[str] = None,
    ) -> RunContext:
        prev_dry = os.environ.get("EVOLIEZ_DRY_RUN")
        if ctx.dry_run:
            os.environ["EVOLIEZ_DRY_RUN"] = "1"
        try:
            names = self.stage_names()
            start = names.index(from_stage) if from_stage else 0
            end = names.index(to_stage) + 1 if to_stage else len(self.stages)

            for stage in self.stages[start:end]:
                t0 = time.time()
                if resume and ctx.is_stage_done(stage.name) and stage.load(ctx):
                    log.info(
                        "[skip] %s (already complete, artifacts reloaded)",
                        stage.name,
                    )
                    continue
                log.info("[run ] %s", stage.name)
                try:
                    stage.run(ctx)
                except PreflightBlocked as pb:
                    # K (post-expert-audit) — clean halt, not a crash.
                    # s01 raised this because strict_preflight=True and
                    # the MD parameterisation preflight returned a
                    # blocking status. No downstream stage should run:
                    # Boltz / docking / ranking / MD all consume the
                    # ligand the preflight just declared MD-unusable.
                    # `ctx.meta["preflight_blocked"]` has already been
                    # set by s01 so the harness / report can see the
                    # reason.
                    log.warning(
                        "[halt] %s: preflight blocked (status=%s) — "
                        "pipeline stops cleanly. Reason: %s",
                        stage.name, pb.status, pb.reason or "n/a",
                    )
                    return ctx
                ctx.mark_stage_done(stage.name)
                log.info("[done] %s (%.1fs)", stage.name, time.time() - t0)
            return ctx
        finally:
            # Never leak dry-run permissiveness into a later real run in the
            # same process: subprocess_utils.require() tolerates missing
            # tools while EVOLIEZ_DRY_RUN is set.
            if prev_dry is None:
                os.environ.pop("EVOLIEZ_DRY_RUN", None)
            else:
                os.environ["EVOLIEZ_DRY_RUN"] = prev_dry


def run_pipeline(
    ctx: RunContext,
    *,
    resume: bool = False,
    from_stage: Optional[str] = None,
    to_stage: Optional[str] = None,
) -> RunContext:
    return Pipeline().run(
        ctx, resume=resume, from_stage=from_stage, to_stage=to_stage
    )
