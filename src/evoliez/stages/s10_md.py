"""Stage 10 - molecular-dynamics validation (spec section 15)."""

from __future__ import annotations

from typing import List

from evoliez.adapters.openmm_engine import run_md
from evoliez.context import RunContext
from evoliez.db.schema import MDSimulation
from evoliez.md.analysis import analyse, to_json
from evoliez.stages.base import Stage
from evoliez.stages.s09_nonmd_validation import _mutant_complex
from evoliez.types import Candidate


class MDStage(Stage):
    name = "s10_md"

    def run(self, ctx: RunContext) -> None:
        mdcfg = ctx.config.validation.md
        candidates: List[Candidate] = ctx.require("md_candidates")
        if not mdcfg.enabled:
            self.log.info("MD disabled in config; skipping")
            for c in candidates:
                c.scores.setdefault("md_lite_score", 0.0)
                c.scores.setdefault("md_instability", 0.0)
            ctx.put("md_candidates", candidates)
            return

        wt = ctx.require("wt_complex")
        catalytic = ctx.require("catalytic_positions")
        weights = ctx.config.scoring
        backend = ctx.config.backend_for(self.name)

        assert ctx.store is not None
        for cand in candidates:
            mc = _mutant_complex(wt, cand)
            inst = cand.scores.get("instability", 0.3)
            result = run_md(
                mc, cand.candidate_id, mdcfg,
                ctx.paths.md_candidate(cand.candidate_id),
                instability=inst, catalytic_positions=catalytic,
                backend=backend, dry_run=ctx.dry_run,
            )
            metrics = analyse(result, weights)
            cand.scores["md_lite_score"] = metrics.md_lite_score
            cand.scores["md_instability"] = round(
                0.0 if metrics.simulation_health_ok else 1.0, 3
            )
            cand.details["md_passed"] = metrics.passed
            if metrics.failure_reasons:
                cand.details["md_failure_reasons"] = "; ".join(
                    metrics.failure_reasons
                )

            aj = to_json(metrics)
            (ctx.paths.md_candidate(cand.candidate_id) / "analysis.json").write_text(
                __import__("json").dumps(aj, indent=2)
            )
            with ctx.store.session() as s:
                s.add(
                    MDSimulation(
                        project_id=ctx.project_id,
                        candidate_id=cand.candidate_id,
                        protocol_level=result.protocol_level,
                        solvent_mode=result.solvent_mode,
                        simulation_time_ns=result.simulation_time_ns,
                        replica_id=0,
                        status=result.status,
                        md_score=metrics.md_lite_score,
                        trajectory_path=result.trajectory_path,
                        analysis_json=aj,
                    )
                )

        n_pass = sum(1 for c in candidates if c.details.get("md_passed"))
        ctx.put("md_candidates", candidates)
        ctx.persist_meta("n_md_passed", n_pass)
        mode = ("dry-run preview" if ctx.dry_run
                else getattr(backend, "value", str(backend)))
        self.log.info(
            "MD (L%d, %s) [%s]: %d/%d passed",
            mdcfg.protocol_level, mdcfg.solvent, mode, n_pass, len(candidates),
        )
