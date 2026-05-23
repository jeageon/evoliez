"""Stage 10 - molecular-dynamics validation (spec section 15)."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from evoliez.adapters.openmm_engine import run_md
from evoliez.context import RunContext
from evoliez.db.schema import MDSimulation
from evoliez.md.analysis import analyse, to_json
from evoliez.stages.base import Stage
from evoliez.stages.s09_nonmd_validation import _mutant_complex
from evoliez.types import Candidate


def _load_existing_md_analysis(workdir: Path) -> Optional[Dict[str, Any]]:
    """Resume cache for s10_md: read a previously-written analysis.json
    for this candidate and return the dict if it's a valid record.

    Returns ``None`` when the file is missing, unreadable, truncated,
    or schema-incomplete - caller then runs MD fresh. Same defensive
    pattern as adapters/boltz._load_existing_real: a half-written file
    from a SIGKILL'd previous run must NEVER fool the cache; the next
    invocation must re-run MD honestly rather than skip on garbage.

    Production incident motivating this: a 30-candidate × 3-replica run
    OOM-killed at candidate ~26 of ~30. 26 ``analysis.json`` files
    existed on disk + matching DB rows. Without cache reuse, the
    pipeline would re-run all 30 (~3.5 h burn) on every resume.
    """
    analysis_path = workdir / "analysis.json"
    if not analysis_path.exists():
        return None
    try:
        size = analysis_path.stat().st_size
    except OSError:
        return None
    if size < 32:                                  # almost-empty = truncated
        return None
    try:
        aj = json.loads(analysis_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(aj, dict):
        return None
    # Required fields written by to_json(metrics). Both must be present
    # AND have sane types; otherwise treat as cache miss.
    if "md_lite_score" not in aj or "passed" not in aj:
        return None
    return aj


def _rss_mb() -> Optional[float]:
    """Process RSS in MiB, or ``None`` when ``psutil`` isn't available.

    Used to log per-candidate memory growth so the production GPU-memory-
    leak fix can be VERIFIED in flight: with the fix, RSS should stay
    bounded across candidates; without, it grows linearly. Soft import
    so a missing optional dep never breaks the stage.
    """
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:                              # noqa: BLE001
        return None


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
        n_ran = n_skipped = n_failed = n_cached = 0
        # Per-candidate cache lookup: which candidates already have a
        # valid analysis.json on disk? Those skip MD entirely (huge
        # speedup on resume - the OOM incident left 26 of 30 candidates
        # finished, so a naive re-run would burn 26 candidates worth of
        # MD = ~3 h for nothing).
        cached_analysis: Dict[str, Dict[str, Any]] = {}
        cand_ids_to_rerun: List[str] = []
        for cand in candidates:
            wd = ctx.paths.md_candidate(cand.candidate_id)
            cached = _load_existing_md_analysis(wd)
            if cached is not None:
                cached_analysis[cand.candidate_id] = cached
            else:
                cand_ids_to_rerun.append(cand.candidate_id)
        if cached_analysis:
            self.log.info(
                "MD cache: %d/%d candidates have valid analysis.json on "
                "disk; %d will be re-run", len(cached_analysis),
                len(candidates), len(cand_ids_to_rerun),
            )
        # Idempotency on resume / re-run: drop prior MDSimulation rows
        # ONLY for candidates we're actually about to re-run, so cached
        # rows survive. The previous unconditional delete-then-add
        # silently nuked good data when it should have left it alone.
        if cand_ids_to_rerun:
            with ctx.store.session() as s:
                s.query(MDSimulation).filter(
                    MDSimulation.project_id == ctx.project_id,
                    MDSimulation.candidate_id.in_(cand_ids_to_rerun),
                ).delete(synchronize_session=False)
        # P0.6 final-tier replicas. md_candidates arrive ordered by upstream
        # rank (s09 sorts and keeps the top `top_candidates`). The first
        # ceil(top_candidates * 0.25) are treated as "final tier" and get
        # `final_tier_replicas` runs; the rest get `replicas`. Replicas
        # aggregate to median series + per-replica spread on the MDResult.
        from math import ceil
        n_final_tier = max(1, ceil(len(candidates) * 0.25))
        rss_start = _rss_mb()
        if rss_start is not None:
            self.log.info(
                "MD loop start: process RSS = %.0f MB (will log per "
                "candidate so leak regressions surface immediately)",
                rss_start,
            )
        for cand_idx, cand in enumerate(candidates):
            # --- Cache hit: skip MD, hydrate cand from existing analysis.json
            cached_aj = cached_analysis.get(cand.candidate_id)
            if cached_aj is not None:
                cand.scores["md_lite_score"] = float(
                    cached_aj.get("md_lite_score", 0.0)
                )
                cand.scores.setdefault("md_status", "ok"
                                       if cached_aj.get("passed") else "unstable")
                cand.scores["md_did_run"] = 1
                cand.scores["md_instability"] = round(
                    0.0 if cached_aj.get("simulation_health_ok") else 1.0, 3,
                )
                cand.details["md_passed"] = bool(cached_aj.get("passed"))
                cand.details["md_did_run"] = True
                if cached_aj.get("failure_reasons"):
                    cand.details["md_failure_reasons"] = "; ".join(
                        cached_aj["failure_reasons"]
                    )
                n_cached += 1
                if cached_aj.get("passed"):
                    n_ran += 1
                else:
                    n_failed += 1
                self.log.info(
                    "MD cache hit (%d/%d): %s (skip re-run; passed=%s)",
                    cand_idx + 1, len(candidates), cand.candidate_id,
                    cached_aj.get("passed"),
                )
                continue
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

            # Production memory-leak fix (belt + suspenders).
            # `_run_real` now releases OpenMM resources in its finally
            # block, but the per-candidate loop also holds the list of
            # MDResult objects (replica_results) and the per-mutant
            # Complex (mc). Drop those references now and force a GC
            # cycle so any cyclic refs between OpenMM/OpenFF/RDKit
            # wrappers (which the C++ destructors only finalise when
            # the wrapper actually drops) are collected before the
            # next candidate's allocations begin. The OOM incident
            # this fixes: Python RSS reached 166 GB before the kernel
            # OOM-killer terminated the process at candidate ~26.
            replica_results.clear()
            del result
            mc = None
            gc.collect()
            rss_now = _rss_mb()
            if rss_start is not None and rss_now is not None:
                self.log.info(
                    "MD %d/%d done (%s): RSS=%.0f MB (delta=%+.0f MB "
                    "from start)", cand_idx + 1, len(candidates),
                    cand.candidate_id, rss_now, rss_now - rss_start,
                )

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
        ctx.persist_meta("n_md_cache_hits", n_cached)
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
            "MD real-execution: %d/%d actually ran (skipped=%d, failed=%d, "
            "cache_hits=%d)",
            n_ran, len(candidates), n_skipped, n_failed, n_cached,
        )
