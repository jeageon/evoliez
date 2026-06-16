"""Stage-DAG orchestrator with checkpoint/resume (spec section 18.1)."""

from __future__ import annotations

import os
import time
from typing import List, Optional

from evoliez.context import RunContext
from evoliez.logging_utils import get_logger
from evoliez.stages import ALL_STAGES
from evoliez.stages.base import Stage

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
        prev_amf = os.environ.get("EVOLIEZ_ALLOW_MOCK_FALLBACK")
        if ctx.dry_run:
            os.environ["EVOLIEZ_DRY_RUN"] = "1"
        # Whether a real-backend tool may silently degrade to mock (default no;
        # audit P0 #3). Adapters read this via base.mock_fallback_allowed().
        os.environ["EVOLIEZ_ALLOW_MOCK_FALLBACK"] = (
            "1" if ctx.config.allow_mock_fallback else "0"
        )
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
                stage.run(ctx)
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
            if prev_amf is None:
                os.environ.pop("EVOLIEZ_ALLOW_MOCK_FALLBACK", None)
            else:
                os.environ["EVOLIEZ_ALLOW_MOCK_FALLBACK"] = prev_amf


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
