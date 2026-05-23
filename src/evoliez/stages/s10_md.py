"""Stage 10 - molecular-dynamics validation (spec section 15)."""

from __future__ import annotations

import gc
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
        # Real per-mutant Boltz structures from s08b (if it ran): use them
        # so real MD runs the ACTUAL mutant, not the WT-derived proxy that
        # the openmm sequence guard correctly skips.
        mut_complexes = ctx.get("mutant_complexes", {}) or {}
        n_ran = n_skipped = n_failed = 0
        # Idempotency on resume / re-run: drop prior MDSimulation rows for
        # the candidates we're about to (re-)run so we don't accumulate
        # duplicate (DockingPose, MDSimulation) rows the way the
        # observability test caught (1 -> 2 -> 4 -> ...). Once s10 has a
        # proper load(), this no-op when the stage skips entirely.
        cand_ids = [c.candidate_id for c in candidates]
        if cand_ids:
            with ctx.store.session() as s:
                s.query(MDSimulation).filter(
                    MDSimulation.project_id == ctx.project_id,
                    MDSimulation.candidate_id.in_(cand_ids),
                ).delete(synchronize_session=False)
        # P0.6 final-tier replicas. md_candidates arrive ordered by upstream
        # rank (s09 sorts and keeps the top `top_candidates`). The first
        # ceil(top_candidates * 0.25) are treated as "final tier" and get
        # `final_tier_replicas` runs; the rest get `replicas`. Replicas
        # aggregate to median series + per-replica spread on the MDResult.
        from math import ceil
        n_final_tier = max(1, ceil(len(candidates) * 0.25))
        for cand_idx, cand in enumerate(candidates):
            mc = mut_complexes.get(cand.candidate_id) or _mutant_complex(
                wt, cand
            )
            inst = cand.scores.get("instability", 0.3)
            in_final_tier = cand_idx < n_final_tier
            target_replicas = max(
                1,
                (mdcfg.final_tier_replicas if in_final_tier
                 else getattr(mdcfg, "replicas", 1)),
            )
            replica_results = []
            for replica_id in range(target_replicas):
                workdir = ctx.paths.md_candidate(cand.candidate_id)
                if replica_id > 0:
                    workdir = workdir.parent / f"{workdir.name}_r{replica_id}"
                replica_results.append(run_md(
                    mc, cand.candidate_id, mdcfg, workdir,
                    instability=inst, catalytic_positions=catalytic,
                    backend=backend, dry_run=ctx.dry_run,
                ))
            # Pick the "primary" result for downstream metrics: the one with
            # the lowest final ligand RMSD across runs that didn't fail.
            usable = [r for r in replica_results
                      if not r.integration_failed and r.status != "failed"]
            if usable:
                result = min(
                    usable,
                    key=lambda r: (r.ligand_rmsd_series or [99.0])[-1],
                )
                # Aggregate per-replica RMSD series onto the chosen result so
                # the report can compute variance / confidence intervals.
                result.ligand_rmsd_replicas = [
                    list(r.ligand_rmsd_series or []) for r in replica_results
                ]
                result.pocket_rmsd_replicas = [
                    list(r.pocket_rmsd_series or []) for r in replica_results
                ]
                result.replicas_run = len(replica_results)
            else:
                # No usable replica - keep the first one for status reporting.
                result = replica_results[0]
                result.replicas_run = len(replica_results)
            _st = str(result.status)
            if result.integration_failed or _st == "failed":
                n_failed += 1
            elif _st.startswith("skipped"):
                n_skipped += 1
            else:
                n_ran += 1
            metrics = analyse(result, weights)
            cand.scores["md_lite_score"] = metrics.md_lite_score
            cand.scores["md_status"] = result.status   # P0.5: feeds evidence class
            # P0.6 honesty: distinguish "MD passed" from "MD never ran".
            # `skipped*` statuses mean the engine refused to start (neutral
            # for skipped_parameterization, failed for everything else);
            # the report uses this to label rows accordingly.
            md_did_run = not (result.integration_failed
                              or result.status == "failed"
                              or str(result.status).startswith("skipped"))
            cand.scores["md_did_run"] = int(md_did_run)
            cand.scores["md_instability"] = round(
                0.0 if metrics.simulation_health_ok else 1.0, 3
            )
            cand.details["md_passed"] = metrics.passed
            cand.details["md_did_run"] = md_did_run
            if metrics.failure_reasons:
                cand.details["md_failure_reasons"] = "; ".join(
                    metrics.failure_reasons
                )
            # P0.6: persist MD provenance per candidate so the final report
            # says HOW MD ran ("Sage, HMR off, 1 replica") instead of just
            # a pass/fail. Sage and HMR flags surface on cand.scores so the
            # report layer can include them in tables without spelunking.
            if mdcfg.persist_provenance:
                cand.scores["md_ligand_forcefield"] = (
                    result.ligand_forcefield or "?"
                )
                cand.scores["md_hmr_enabled"] = int(bool(result.hmr_enabled))
                cand.scores["md_timestep_fs"] = float(result.timestep_fs)
                cand.scores["md_replicas_run"] = int(result.replicas_run)

            aj = to_json(metrics)
            # Trajectory metadata (additive): when the OpenMM engine wrote a
            # real .dcd, the HTML report needs the path / topology / frame
            # count to generate movies. Mock backend runs leave these as
            # None. Wrapped in try/except so a metadata-write hiccup never
            # poisons the stage. See adapters/openmm_engine.py for the
            # DCDReporter convention (`<md_dir>/<cand>.dcd`, 50 frames per
            # production run; minimised PDB is the topology).
            try:
                _traj_str = result.trajectory_path
                _traj_p = __import__("pathlib").Path(_traj_str) if _traj_str else None
                _top_str = result.minimized_pdb
                _top_p = __import__("pathlib").Path(_top_str) if _top_str else None
                _real_md = (
                    _traj_p is not None
                    and _traj_p.exists()
                    and not result.integration_failed
                    and not str(result.status).startswith("skipped")
                )
                _n_frames = (
                    len(result.ligand_rmsd_series)
                    if (_real_md and result.ligand_rmsd_series) else None
                )
                # production_ns is total simulated time; per-frame dt = total / n_frames
                _dt_ps = None
                if _real_md and _n_frames:
                    _dt_ps = round(
                        float(result.simulation_time_ns) * 1000.0 / float(_n_frames),
                        4,
                    )
                aj["trajectory_path"] = (
                    str(_traj_p) if (_traj_p is not None and _traj_p.exists()) else None
                )
                aj["topology_path"] = (
                    str(_top_p) if (_top_p is not None and _top_p.exists()) else None
                )
                aj["n_frames"] = _n_frames
                aj["dt_ps"] = _dt_ps
                aj["trajectory_format"] = "dcd" if _real_md else None
            except Exception as exc:  # noqa: BLE001 - never break stage on metadata
                self.log.warning(
                    "trajectory metadata write skipped for %s: %s",
                    cand.candidate_id, exc,
                )
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

            # Production GPU-memory-leak fix (belt + suspenders).
            # `_run_real` now releases OpenMM resources in its finally
            # block, but the per-candidate loop also holds the list of
            # MDResult objects (replica_results) and the per-mutant
            # Complex (mc). Drop those references now and force a GC
            # cycle so any cyclic refs between OpenMM/OpenFF/RDKit
            # wrappers (which the C++ destructors only finalise when
            # the wrapper actually drops) are collected before the
            # next candidate's allocations begin. Without this, we saw
            # GPU memory grow ~1-2 GB per 10 candidates on the server.
            replica_results.clear()
            del result
            mc = None
            gc.collect()

        n_pass = sum(1 for c in candidates if c.details.get("md_passed"))
        n_replicated = sum(1 for c in candidates
                           if int(c.scores.get("md_replicas_run", 1)) > 1)
        ctx.put("md_candidates", candidates)
        ctx.persist_meta("n_md_passed", n_pass)
        ctx.persist_meta("n_md_real_ran", n_ran)
        ctx.persist_meta("n_md_skipped", n_skipped)
        ctx.persist_meta("n_md_failed", n_failed)
        ctx.persist_meta("n_md_replicated", n_replicated)
        ctx.persist_meta("n_md_final_tier", n_final_tier)
        mode = ("dry-run preview" if ctx.dry_run
                else getattr(backend, "value", str(backend)))
        self.log.info(
            "MD (L%d, %s) [%s]: %d/%d passed",
            mdcfg.protocol_level, mdcfg.solvent, mode, n_pass, len(candidates),
        )
        # Deterministic, greppable honesty line: how many candidates the MD
        # engine ACTUALLY ran (vs skipped/failed). server_smoke.sh asserts
        # this is >0 for the real MD rung so a degraded stage can't pass as
        # OK (skipped_parameterization is neutral-pass but is NOT "ran").
        self.log.info(
            "MD real-execution: %d/%d actually ran (skipped=%d, failed=%d)",
            n_ran, len(candidates), n_skipped, n_failed,
        )
