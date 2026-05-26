"""Stage 10 - molecular-dynamics validation (spec section 15)."""

from __future__ import annotations

import time
from typing import List, Optional

from evoliez.adapters.openmm_engine import run_md
from evoliez.adapters.openmm_subprocess import run_md_in_subprocess
from evoliez.config import Backend
from evoliez.context import RunContext
from evoliez.db.schema import MDSimulation
from evoliez.md.analysis import analyse, to_json
from evoliez.stages.base import Stage
from evoliez.stages.s09_nonmd_validation import _mutant_complex
from evoliez.types import Candidate

# Machine-parseable phase marker. Format:
#     [evoliez-md-phase] <name> <unix_ts_seconds>
# Anchored with the literal prefix so a regex r"^\[evoliez-md-phase\]" parses
# cleanly from `_md_subprocess.log` / pipeline.log. New phase names are
# free-form (snake_case) — operators grep by name; the prefix is the contract.
_PHASE_PREFIX = "[evoliez-md-phase]"


def _phase(name: str) -> None:
    """Emit a phase marker on stdout. MUST be stdout (not stderr) so
    subprocess.run(capture_output=True) and the worker's stdout->log
    redirection both pick it up."""
    print(f"{_PHASE_PREFIX} {name} {time.time():.3f}", flush=True)


class MDStage(Stage):
    name = "s10_md"

    # ---- Preflight ----------------------------------------------------- #
    # Runs ONCE before the per-candidate loop. The single biggest waste of
    # server time on this pipeline was 12 candidates each timing out at
    # subprocess_timeout_seconds because AM1-BCC/sqm hangs on the cofactor
    # for every single one. One 60 s probe up front tells us whether the
    # ligand can be parameterized AT ALL with the available FFs; if not,
    # every candidate gets honestly marked skipped_no_params instead of
    # burning 22 hours producing 22 timeouts.
    def _preflight_ligand_params(self, ctx: RunContext) -> Optional[str]:
        """One-shot ligand parameterization sanity check.

        Returns ``None`` on success or when the check doesn't apply
        (dry-run, mock backend). Returns a short reason string on
        failure; the caller treats that as "every candidate is
        skipped_no_params, do not enter the per-candidate loop".

        MUST NOT raise: a preflight failure is a routing decision, not
        a stage crash. We catch broadly on purpose.
        """
        if ctx.dry_run:
            return None
        backend = ctx.config.backend_for(self.name)
        if backend is not Backend.real:
            return None  # mock backend has no parameterization step
        wt = ctx.require("wt_complex")
        # Use the WT ligand as the probe; per-candidate mutants don't change
        # the ligand chemistry, so a successful preflight is sufficient.
        self.log.info(
            "MD preflight: probing ligand parameterization (60 s budget)"
        )
        _phase("preflight_start")
        t0 = time.time()
        try:
            # Heavy imports deferred so the preflight cost itself isn't
            # paid on mock/dry-run paths (and so test stubs can intercept).
            from evoliez.adapters.openmm_engine import (
                _ligand_offmol_at_pose,
                _ligand_system_generator,
            )
            # Build the OpenFF Molecule from the predicted complex PDB +
            # SMILES (same call path the real engine takes). Falls back
            # to a SMILES-only OpenFF Molecule if we have no full-atom PDB
            # to graph-match against (still exercises the FF probe).
            off_lig = None
            pdb_str = getattr(wt.structure, "pdb_path", None)
            if pdb_str:
                from pathlib import Path as _P
                pdb_path = _P(pdb_str)
                if pdb_path.exists():
                    try:
                        off_lig = _ligand_offmol_at_pose(
                            pdb_path, wt.ligand.smiles,
                        )
                    except Exception as exc:
                        # Fall through to SMILES-only probe below.
                        self.log.info(
                            "preflight: pose-based build skipped (%s); "
                            "probing FF on SMILES-only OpenFF Molecule",
                            exc,
                        )
            if off_lig is None:
                from openff.toolkit import Molecule
                off_lig = Molecule.from_smiles(
                    wt.ligand.smiles, allow_undefined_stereo=True,
                )
                off_lig.generate_conformers(n_conformers=1)
            workdir = ctx.paths.md / "_preflight"
            workdir.mkdir(parents=True, exist_ok=True)
            mdcfg = ctx.config.validation.md
            _ligand_system_generator(
                off_lig, workdir,
                prefer_ff=getattr(mdcfg, "ligand_forcefield", None),
            )
            elapsed = time.time() - t0
            self.log.info("MD preflight OK (%.1fs)", elapsed)
            _phase("preflight_done_ok")
            return None
        except Exception as exc:  # noqa: BLE001 — see docstring
            elapsed = time.time() - t0
            reason = (
                f"ligand_preflight_failed: {type(exc).__name__}: "
                f"{str(exc)[:300]} (after {elapsed:.1f}s)"
            )
            self.log.warning(reason)
            _phase("preflight_done_failed")
            return reason

    def _mark_all_skipped_no_params(
        self, candidates: List[Candidate], reason: str,
    ) -> None:
        """Apply skipped_no_params status to every candidate AND write an
        honest analysis.json saying so. Called when preflight returns a
        failure reason; the per-candidate loop must NOT be entered."""
        for c in candidates:
            c.scores["md_lite_score"] = 0.0
            c.scores["md_status"] = "skipped_no_params"
            c.scores["md_did_run"] = 0
            c.scores["md_instability"] = 0.0
            c.details["md_passed"] = False
            c.details["md_did_run"] = False
            c.details["md_failure_reasons"] = reason

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

        # Preflight: one shared ligand parameterization probe. If the
        # ligand can't be parameterized by ANY available small-molecule FF
        # (the symptom we hit at 22h/22-timeout scale), every candidate is
        # honestly recorded as skipped_no_params and the per-candidate loop
        # is skipped entirely. NEUTRAL skip (judged on other layers).
        preflight_reason = self._preflight_ligand_params(ctx)
        if preflight_reason is not None:
            self.log.warning(
                "MD preflight failed; skipping all %d candidates with "
                "reason: %s", len(candidates), preflight_reason,
            )
            self._mark_all_skipped_no_params(candidates, preflight_reason)
            ctx.put("md_candidates", candidates)
            ctx.persist_meta("n_md_passed", 0)
            ctx.persist_meta("n_md_real_ran", 0)
            ctx.persist_meta("n_md_skipped", len(candidates))
            ctx.persist_meta("n_md_failed", 0)
            ctx.persist_meta("n_md_failed_timeout", 0)
            ctx.persist_meta("n_md_skipped_no_params", len(candidates))
            return

        assert ctx.store is not None
        # Real per-mutant Boltz structures from s08b (if it ran): use them
        # so real MD runs the ACTUAL mutant, not the WT-derived proxy that
        # the openmm sequence guard correctly skips.
        mut_complexes = ctx.get("mutant_complexes", {}) or {}
        n_ran = n_skipped = n_failed = n_failed_timeout = 0
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
                # subprocess isolation (opt-in via mdcfg.subprocess_isolation)
                # contains OpenMM / CUDA state to a per-candidate process and
                # enforces a hard timeout. Falls back to in-process run_md
                # when the flag is off or the backend isn't real - mock MD
                # doesn't need the per-candidate process overhead.
                # Phase markers wrap the call on the s10_md side; the worker
                # subprocess emits its own markers around run_md (see
                # openmm_subprocess_worker.py).
                _phase(f"candidate_start {cand.candidate_id} r{replica_id}")
                _rep = run_md_in_subprocess(
                    mc, cand.candidate_id, mdcfg, workdir,
                    instability=inst, catalytic_positions=catalytic,
                    backend=backend, dry_run=ctx.dry_run,
                    timeout_seconds=int(getattr(
                        mdcfg, "subprocess_timeout_seconds", 1800,
                    )),
                )
                _phase(
                    f"candidate_done {cand.candidate_id} r{replica_id} "
                    f"{_rep.status}"
                )
                # Reclassify the generic 'failed' that the subprocess wrapper
                # returns on wall-clock timeout into the more specific
                # 'failed_timeout' so bench-summary can count timeout-vs-real
                # failure separately. The wrapper stamps the failure_reason
                # with "subprocess timeout" — that's the only signal we have
                # because Popen returns the same returncode (-9) for SIGKILL
                # whether we sent it or the OOM killer did.
                if (_rep.status == "failed"
                        and "subprocess timeout" in (_rep.failure_reason or "").lower()):
                    _rep.status = "failed_timeout"
                replica_results.append(_rep)
            # Pick the "primary" result for downstream metrics: the one with
            # the lowest final ligand RMSD across runs that didn't fail.
            usable = [r for r in replica_results
                      if not r.integration_failed
                      and r.status not in ("failed", "failed_timeout")]
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
            if _st == "failed_timeout":
                n_failed_timeout += 1
                n_failed += 1     # also counted under generic failed total
            elif result.integration_failed or _st == "failed":
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
                              or result.status in ("failed", "failed_timeout")
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

            # P0b: pass the MDResult so to_json persists per-frame RMSD
            # time series (ligand, pocket, key distances) along with the
            # summary stats. Without `result`, analysis.json is summary-
            # only and downstream consumers (HTML report, paper figures)
            # have to fall back to bar charts instead of real trajectories.
            aj = to_json(metrics, result)
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

        n_pass = sum(1 for c in candidates if c.details.get("md_passed"))
        n_replicated = sum(1 for c in candidates
                           if int(c.scores.get("md_replicas_run", 1)) > 1)
        ctx.put("md_candidates", candidates)
        ctx.persist_meta("n_md_passed", n_pass)
        ctx.persist_meta("n_md_real_ran", n_ran)
        ctx.persist_meta("n_md_skipped", n_skipped)
        ctx.persist_meta("n_md_failed", n_failed)
        # P0.7: split timeout from generic failed so bench-summary can
        # report timeout-vs-real-failure ratio. n_md_failed_timeout is a
        # SUBSET of n_md_failed (timeouts are still counted in the
        # failed total for backward compatibility).
        ctx.persist_meta("n_md_failed_timeout", n_failed_timeout)
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
            "MD real-execution: %d/%d actually ran (skipped=%d, failed=%d "
            "[of which timeout=%d])",
            n_ran, len(candidates), n_skipped, n_failed, n_failed_timeout,
        )
