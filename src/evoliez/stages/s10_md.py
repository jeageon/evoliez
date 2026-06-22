"""Stage 10 - molecular-dynamics validation (spec section 15)."""

from __future__ import annotations

from typing import List

from evoliez.adapters.openmm_engine import run_md
from evoliez.config import Backend
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
        # Resume-safe charge policy. A stage's load() can rebuild a complex without the
        # ligand's charges_mol2 / allow_am1bcc (serialization parity), so re-apply them
        # from the CONFIG (the static source of truth) onto EVERY complex the MD will
        # see. This routes the -3 NADP to its fixed-charge template instead of a
        # non-converging on-the-fly AM1-BCC. charges_mol2 is resolved to an absolute
        # path here (run CWD = repo root); the engine fails loudly if it is missing.
        from pathlib import Path as _Path
        _lig_cfg = ctx.config.input.ligand
        _cm = _lig_cfg.charges_mol2
        if _cm and not _Path(_cm).is_absolute():
            _cm = str((_Path.cwd() / _cm).resolve())

        def _apply_charge_policy(cx):
            if cx is None or getattr(cx, "ligand", None) is None:
                return
            if cx.ligand.id == _lig_cfg.id:
                cx.ligand.charges_mol2 = _cm
                cx.ligand.allow_am1bcc = _lig_cfg.allow_am1bcc
                if _lig_cfg.net_charge is not None:
                    cx.ligand.formal_charge = _lig_cfg.net_charge

        _apply_charge_policy(wt)
        for _mc in mut_complexes.values():
            _apply_charge_policy(_mc)
        # Shared, stable per-run ligand force-field cache. The ligand
        # (cofactor + substrate) is identical across every candidate, so the
        # slow AM1-BCC/antechamber charge derivation is cached HERE once and
        # reused, rather than re-derived under each per-candidate workdir
        # (~md.top_candidates times). Lives under the run root, so it survives
        # --resume; ligand-keyed inside the engine so it self-invalidates if
        # the ligand changes.
        ligand_cache_dir = ctx.paths.md / "_ligand_ff_cache"

        # Catalytic-power (NAC) layer: co-substrates/cofactors (e.g. formate
        # beside NADP) must ALSO be in the MD box so the reaction donor and
        # acceptor are both present. Pass them as (id, smiles); the engine uses
        # them only when md.reactive_geometry.enabled, so the default binding MD
        # is unchanged. type != smiles entries (path-based) are skipped.
        nac_cfg = getattr(mdcfg, "reactive_geometry", None)
        nac_enabled = bool(nac_cfg and getattr(nac_cfg, "enabled", False))
        # Prefer the resumed ctx artifact, fall back to the CONFIG: a stage's load()
        # can drop the ctx "extra_ligands" copy on --resume (s01 load-parity), which
        # silently strips NAC's co-substrate (formate). The design ligand would then
        # take the single-ligand build that reads ALL HETATM as one molecule and
        # mis-matches the NADP(48)+formate(3) block (atom-count gate) -> the whole MD
        # fails. The config is the static source of truth, so it is never lost.
        # Prefer the CONFIG (LigandInput: .value + .type) as the static source of truth.
        # The resumed ctx "extra_ligands" are types.Ligand (.smiles, NO .value/.type), so
        # the old code -- ctx-first, then read .value -- silently dropped formate (ctx
        # list is truthy so the config fallback never fired; .value is empty on a
        # types.Ligand). Without formate the design ligand takes the single-ligand build
        # and NAC loses its hydride donor. Accept EITHER shape.
        _extra_ligs = (ctx.config.input.extra_ligands
                       or ctx.get("extra_ligands", []) or [])

        def _extra_smiles(e):
            if getattr(e, "type", "smiles") not in ("smiles", None):
                return None                        # path-based extra (sdf/mol2) -> skip
            return getattr(e, "value", None) or getattr(e, "smiles", None)

        extra_specs = [(getattr(e, "id", f"extra{i}"), _extra_smiles(e))
                       for i, e in enumerate(_extra_ligs)]
        extra_specs = [(eid, smi) for eid, smi in extra_specs if smi]
        # WT reference NAC baseline: run the reaction-geometry screen ONCE on the
        # WT complex so each mutant's reactivity is reported as ΔNAC = mutant - WT
        # (mutant > WT == a geometrically MORE productive active site). Only when
        # NAC is enabled and real MD will actually run (mock/dry-run yield no NAC).
        wt_nac = None
        wt_record = None
        if nac_enabled and not ctx.dry_run and backend is Backend.real:
            try:
                wt_res = run_md(
                    wt, "_wt_reference", mdcfg,
                    ctx.paths.md_candidate("_wt_reference"),
                    instability=0.0, catalytic_positions=catalytic,
                    backend=backend, dry_run=ctx.dry_run,
                    ligand_cache_dir=ligand_cache_dir, extra_ligands=extra_specs,
                )
                wt_metrics = analyse(wt_res, weights)
                wt_nac = wt_metrics.nac_occupancy
                (ctx.paths.md_candidate("_wt_reference") / "analysis.json"
                 ).write_text(__import__("json").dumps(to_json(wt_metrics), indent=2))
                wt_record = {
                    "candidate_id": "_wt_reference", "mutation_string": "WT",
                    "status": wt_res.status, "solvent_mode": wt_res.solvent_mode,
                    "simulation_time_ns": wt_res.simulation_time_ns,
                    "protocol_level": wt_res.protocol_level,
                    "nac_occupancy": wt_metrics.nac_occupancy,
                    "nac": wt_metrics.nac, "binding_dg": wt_metrics.binding_dg,
                }
                self.log.info("WT reference NAC occupancy: %s (status=%s)",
                              wt_nac, wt_res.status)
            except Exception as exc:    # the reference is a diagnostic, never fatal
                self.log.warning("WT reference NAC run failed (%s); ΔNAC "
                                 "unavailable", exc)

        # Per-candidate MD provenance: a self-contained on-disk record (what
        # ACTUALLY ran -- solvent/ns/status -- plus every metric) so the s10
        # report regenerates from disk with no DB and no stage re-run.
        md_records: List[dict] = []

        n_ran = n_skipped = n_failed = 0
        for cand in candidates:
            mc = mut_complexes.get(cand.candidate_id) or _mutant_complex(
                wt, cand
            )
            inst = cand.scores.get("instability", 0.3)
            result = run_md(
                mc, cand.candidate_id, mdcfg,
                ctx.paths.md_candidate(cand.candidate_id),
                instability=inst, catalytic_positions=catalytic,
                backend=backend, dry_run=ctx.dry_run,
                ligand_cache_dir=ligand_cache_dir, extra_ligands=extra_specs,
            )
            _st = str(result.status)
            if result.integration_failed or _st == "failed":
                n_failed += 1
            elif _st.startswith("skipped"):
                n_skipped += 1
            else:
                n_ran += 1
            metrics = analyse(result, weights)
            cand.scores["md_lite_score"] = metrics.md_lite_score
            cand.scores["md_instability"] = round(
                0.0 if metrics.simulation_health_ok else 1.0, 3
            )
            # Catalytic-power: occupancy + ΔNAC vs WT (surfaced as a diagnostic;
            # NOT yet folded into the composite ranking score). Only when the NAC
            # layer actually produced a value for this candidate.
            if metrics.nac_occupancy is not None:
                cand.scores["nac_occupancy"] = metrics.nac_occupancy
                if wt_nac is not None:
                    cand.scores["nac_delta_vs_wt"] = round(
                        metrics.nac_occupancy - wt_nac, 4
                    )
                if metrics.nac:
                    cand.details["nac"] = metrics.nac
            cand.details["md_passed"] = metrics.passed
            if metrics.failure_reasons:
                cand.details["md_failure_reasons"] = "; ".join(
                    metrics.failure_reasons
                )

            aj = to_json(metrics)
            (ctx.paths.md_candidate(cand.candidate_id) / "analysis.json").write_text(
                __import__("json").dumps(aj, indent=2)
            )
            md_records.append({
                "candidate_id": cand.candidate_id,
                "mutation_string": cand.mutation_str,
                "status": result.status,
                "solvent_mode": result.solvent_mode,            # what ACTUALLY ran
                "simulation_time_ns": result.simulation_time_ns,
                "protocol_level": result.protocol_level,
                "md_lite_score": metrics.md_lite_score,
                "passed": metrics.passed,
                "md_instability": cand.scores.get("md_instability"),
                "ligand_rmsd_mean": metrics.ligand_rmsd_mean,
                "pocket_rmsd_mean": metrics.pocket_rmsd_mean,
                "hbond_occupancy": metrics.hbond_occupancy,
                "catalytic_distance_mean": metrics.catalytic_distance_mean,
                "energy_drift": metrics.energy_drift,
                "nac_occupancy": metrics.nac_occupancy,
                "nac_delta_vs_wt": cand.scores.get("nac_delta_vs_wt"),
                "nac": metrics.nac,
                "binding_dg": metrics.binding_dg,
                "failure_reasons": metrics.failure_reasons,
            })
            with ctx.store.session() as s:
                # idempotent: no load() guard, so a --resume re-runs MD; clear
                # this candidate's prior MD row(s) before inserting to avoid
                # duplicate rows (no unique constraint rejects them).
                s.query(MDSimulation).filter_by(
                    project_id=ctx.project_id,
                    candidate_id=cand.candidate_id,
                ).delete()
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

        # Relative binding free energy (ΔΔG_bind, mutant vs WT) — FINAL confirmatory
        # tier on the top-N MD candidates by md_lite_score. Opt-in (md.rbfe.enabled),
        # real backend only; softcore TI via Amber pmemd.cuda (adapters/amber_rbfe).
        # Non-fatal: a failed RBFE leaves the candidate's MD scores intact.
        rbcfg = getattr(mdcfg, "rbfe", None)
        if (rbcfg and getattr(rbcfg, "enabled", False)
                and backend is Backend.real and not ctx.dry_run):
            try:
                from pathlib import Path as _Path

                from evoliez.adapters.amber_engine import parameterize_ligand
                from evoliez.adapters.amber_rbfe import run_rbfe
                wt_pdb = getattr(wt.structure, "pdb_path", None)
                if wt_pdb and _Path(wt_pdb).exists():
                    wt_pdb_text = _Path(wt_pdb).read_text()
                    rbfe_root = ctx.paths.md / "_rbfe"
                    lig_mol2, lig_frcmod = parameterize_ligand(
                        wt, _Path(wt_pdb), rbfe_root / "_ligand")
                    ranked = sorted(
                        (c for c in candidates
                         if c.scores.get("md_lite_score") is not None),
                        key=lambda c: -c.scores["md_lite_score"])[: rbcfg.top_n]
                    self.log.info(
                        "RBFE (ΔΔG_bind, softcore TI): %d candidate(s)", len(ranked))
                    for cand in ranked:
                        res = run_rbfe(
                            wt_pdb_text, lig_mol2, lig_frcmod, cand.mutations,
                            rbfe_root / cand.candidate_id,
                            multipoint=rbcfg.multipoint, n_lambda=rbcfg.n_lambda,
                            min_cyc=rbcfg.min_cyc, heat_steps=rbcfg.heat_steps,
                            prod_steps=rbcfg.prod_steps)
                        cand.details["rbfe"] = res
                        if res.get("ddg_bind") is not None:
                            cand.scores["rbfe_ddg_bind"] = res["ddg_bind"]
                        self.log.info("  %s: ΔΔG_bind=%s kcal/mol (%s)",
                                      cand.candidate_id, res.get("ddg_bind"),
                                      res.get("mode") or res.get("skipped"))
                else:
                    self.log.warning("RBFE skipped: WT complex has no full-atom PDB")
            except Exception as exc:  # noqa: BLE001
                self.log.warning("RBFE stage failed (%s); MD scores intact", exc)

        n_pass = sum(1 for c in candidates if c.details.get("md_passed"))
        ctx.put("md_candidates", candidates)
        ctx.persist_meta("n_md_passed", n_pass)
        ctx.persist_meta("n_md_real_ran", n_ran)
        ctx.persist_meta("n_md_skipped", n_skipped)
        ctx.persist_meta("n_md_failed", n_failed)
        if nac_enabled:
            n_nac = sum(1 for c in candidates if "nac_occupancy" in c.scores)
            n_better = sum(1 for c in candidates
                           if c.scores.get("nac_delta_vs_wt", 0.0) > 0.0)
            ctx.persist_meta("n_md_nac", n_nac)
            ctx.persist_meta("wt_nac_occupancy",
                             wt_nac if wt_nac is not None else -1.0)
            self.log.info(
                "MD NAC (catalytic power): %d/%d candidates scored; "
                "WT occupancy=%s; %d candidate(s) beat WT (ΔNAC>0)",
                n_nac, len(candidates), wt_nac, n_better,
            )
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

        # Self-contained MD provenance dump + auto-generated s10 report (mirrors
        # s09): the report reads ONLY this JSON + meta, so it regenerates from
        # disk with no DB and no stage re-run.
        import json as _json
        _prov = ctx.paths.reports / "provenance"
        _prov.mkdir(parents=True, exist_ok=True)
        (_prov / "md_candidates.json").write_text(_json.dumps({
            "wt_reference": wt_record,
            "nac_enabled": nac_enabled,
            "wt_nac_occupancy": wt_nac,
            "requested": {"protocol_level": mdcfg.protocol_level,
                          "solvent": mdcfg.solvent,
                          "production_ns": mdcfg.production_ns,
                          "engine": getattr(mdcfg, "engine", "openmm")},
            "counts": {"n_real_ran": n_ran, "n_passed": n_pass,
                       "n_skipped": n_skipped, "n_failed": n_failed,
                       "n_total": len(candidates)},
            "candidates": md_records,
        }, indent=2, default=str))
        try:    # the report is a deliverable, never fatal to the pipeline
            from evoliez.io.md_report import write_md_report
            p = write_md_report(ctx.paths.root)
            self.log.info("s10 MD report: %s", p.name)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("s10 MD report generation skipped (%s)", exc)
