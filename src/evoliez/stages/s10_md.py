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

        # Functional-state anchored validation + reference-like pose gate.
        from evoliez.md.anchored_build import build_anchored_mutant_complex
        from evoliez.md.pose_gate import gate_from_pdb
        anchored_on = (bool(getattr(mdcfg, "anchored_validation", False))
                       and bool(getattr(wt.structure, "pdb_path", None)))
        pose_gate_on = bool(getattr(mdcfg, "pose_gate_enabled", False))
        _design_role = getattr(ctx.config.input.ligand, "role", None) or "cofactor"
        if anchored_on:
            self.log.info("s10 anchored validation ON: mutants built from the "
                          "reference complex (ligand poses kept); Boltz pose -> "
                          "alternative hypothesis")

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
        # ROADMAP_V5 step 4 — reaction-geometry source of truth (mechanism.geometry_terms >
        # legacy reactive_geometry > none). Recorded in provenance so a reader knows WHICH
        # geometry drove the NAC. A declared mechanism reports 'mechanism_spec_pending' until
        # the adenylation template + ReactiveSpec/GeometryTerm bridge land (V5-3/V5-6); the
        # trajectory engine currently consumes the legacy ReactiveSpec.
        from evoliez.mechanism.mode import resolve_geometry_source
        geometry_source = resolve_geometry_source(
            ctx.config, ctx.get("geometry_terms"), nac_enabled)
        # ROADMAP_V5 V5-2 — metal-setup CONTRACT (reviewer requirement B). Mg2+ is requested
        # when the mechanism declares a Mg metal_state; the actual structure-level MG HETATM
        # insertion (md.metal_placement.prepare_metal_setup) + Amber-ion verification happen in
        # the OpenMM build (server-verified-pending). Recorded here so provenance shows Mg was
        # requested and HOW it is handled (never via OpenFF; a missing/failed metal is a
        # classified setup state, not a low-NAC biological result).
        _mech = getattr(ctx.config, "mechanism", None)
        _metal_state = (getattr(getattr(_mech, "reaction_state", None), "metal_state", None)
                        if _mech is not None else None)
        _metal_requested = bool(_metal_state and "mg" in _metal_state.lower())
        metal_setup = {
            "requested": _metal_requested,
            "metal_state": _metal_state,
            "ion": "MG" if _metal_requested else None,
            "placement": "deterministic_bridge" if _metal_requested else None,
            "openff_parameterized": False,
            "amber_standard_ion": ("server_verified_pending" if _metal_requested else None),
            "note": ("Mg2+ requested by mechanism.metal_state; structure-level MG HETATM via "
                     "metal_placement.prepare_metal_setup in the OpenMM build "
                     "(server-verified-pending)" if _metal_requested else "no metal requested"),
        }
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

        def _charge_meta(e):
            # Pre-derived fixed-charge template + net charge + heavy-atom count, so the
            # MULTI-ligand NAC build in run_md can inject a high-charge cofactor's
            # charges (ATP/NADPH, -4) instead of running AM1-BCC/sqm on it (never
            # converges, cofactor drops from the FF set, MD fails). Accept EITHER a
            # config LigandInput (.charges_mol2, .net_charge) or a resumed types.Ligand
            # (.charges_mol2, .formal_charge, .n_heavy).
            cm = getattr(e, "charges_mol2", None)
            fc = getattr(e, "formal_charge", None)
            if fc is None:
                fc = getattr(e, "net_charge", 0)
            return cm, int(fc or 0), getattr(e, "n_heavy", None)

        extra_specs = [(getattr(e, "id", f"extra{i}"), _extra_smiles(e),
                        *_charge_meta(e))
                       for i, e in enumerate(_extra_ligs)]
        extra_specs = [t for t in extra_specs if t[1]]   # keep those with a SMILES
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
                    metal_requested=_metal_requested,
                )
                # WT + every mutant get the SAME deterministic Mg placement, so the WT run's
                # actual metal_setup outcome is authoritative for the run-level contract.
                if getattr(wt_res, "metal_setup", None):
                    metal_setup.update(wt_res.metal_setup)
                wt_metrics = analyse(wt_res, weights)
                wt_nac = wt_metrics.nac_occupancy
                (ctx.paths.md_candidate("_wt_reference") / "analysis.json"
                 ).write_text(__import__("json").dumps(to_json(wt_metrics), indent=2))
                wt_record = {
                    "candidate_id": "_wt_reference", "mutation_string": "WT",
                    "status": wt_res.status, "solvent_mode": wt_res.solvent_mode,
                    "simulation_time_ns": wt_res.simulation_time_ns,
                    "protocol_level": wt_res.protocol_level,
                    "nac_status": wt_metrics.nac_status,
                    "nac_occupancy": wt_metrics.nac_occupancy,
                    "nac": wt_metrics.nac, "binding_dg": wt_metrics.binding_dg or None,
                }
                self.log.info("WT reference NAC: status=%s occupancy=%s (md=%s)",
                              wt_metrics.nac_status, wt_nac, wt_res.status)
            except Exception as exc:    # the reference is a diagnostic, never fatal
                self.log.warning("WT reference NAC run failed (%s); ΔNAC "
                                 "unavailable", exc)

        # Reference structure for the pose gate: prefer the WT post-MD minimised
        # complex (same protocol as the candidates), else the WT input PDB.
        wt_ref_pdb = getattr(wt.structure, "pdb_path", None)
        _wt_min = (ctx.paths.md_candidate("_wt_reference")
                   / "_wt_reference_minimized.pdb")
        if _wt_min.exists():
            wt_ref_pdb = str(_wt_min)

        # Per-candidate MD provenance: a self-contained on-disk record (what
        # ACTUALLY ran -- solvent/ns/status -- plus every metric) so the s10
        # report regenerates from disk with no DB and no stage re-run.
        md_records: List[dict] = []

        n_ran = n_skipped = n_failed = 0
        # ROADMAP_V2 Phase H1 — fan the EXPENSIVE step (run_md) across the GPU pool.
        # Phase 0: build the (anchored) mutant complex per candidate (CPU, fast).
        _prebuilt: dict = {}
        for cand in candidates:
            boltz_mc = mut_complexes.get(cand.candidate_id)
            anchored_used = False
            if anchored_on:
                try:
                    _apdb = (ctx.paths.md_candidate(cand.candidate_id)
                             / f"{cand.candidate_id}_anchored.pdb")
                    _apdb.parent.mkdir(parents=True, exist_ok=True)
                    mc, _abuild = build_anchored_mutant_complex(
                        wt, cand.mutations, _apdb)
                    cand.details["anchored_build"] = {
                        "applied": _abuild.applied, "skipped": _abuild.skipped}
                    anchored_used = bool(_abuild.applied)
                    if not anchored_used:        # nothing applied -> not a mutant
                        mc = boltz_mc or _mutant_complex(wt, cand)
                except Exception as _aexc:       # noqa: BLE001
                    self.log.warning("anchored build failed for %s (%s); using "
                                     "Boltz/proxy", cand.candidate_id, _aexc)
                    mc = boltz_mc or _mutant_complex(wt, cand)
            else:
                mc = boltz_mc or _mutant_complex(wt, cand)
            _apply_charge_policy(mc)
            _prebuilt[cand.candidate_id] = (mc, boltz_mc, anchored_used)

        # Phase 1: run_md — fanned across compute.gpu_pool when >1 real GPU, else serial
        # (byte-identical to the previous per-candidate loop). MDResult is picklable.
        from evoliez.utils.compute import gpu_pool_list
        _gpu_pool = gpu_pool_list(getattr(ctx.config.compute, "gpu_pool", None))
        _md_tasks = [(c.candidate_id, _prebuilt[c.candidate_id][0],
                      str(ctx.paths.md_candidate(c.candidate_id)),
                      c.scores.get("instability", 0.3)) for c in candidates]
        _results: dict = {}
        _fail_loud = bool(getattr(ctx.config.compute, "fail_loud_on_cpu_md", False))
        if len(_gpu_pool) > 1 and backend is Backend.real and not ctx.dry_run:
            from evoliez.adapters.md_batch import run_md_batches
            self.log.info("s10 run_md FAN-OUT across GPUs %s (%d candidate(s))",
                          _gpu_pool, len(_md_tasks))
            _results = run_md_batches(
                _md_tasks, _gpu_pool, mdcfg,
                ligand_cache_dir=str(ligand_cache_dir) if ligand_cache_dir
                else None, extra_specs=extra_specs, catalytic=catalytic,
                fail_loud_on_cpu=_fail_loud)
        else:
            for _cid, _mc, _wd, _inst in _md_tasks:
                _results[_cid] = run_md(
                    _mc, _cid, mdcfg, ctx.paths.md_candidate(_cid),
                    instability=_inst, catalytic_positions=catalytic,
                    backend=backend, dry_run=ctx.dry_run,
                    ligand_cache_dir=ligand_cache_dir, extra_ligands=extra_specs,
                    fail_loud_on_cpu=_fail_loud, metal_requested=_metal_requested)

        # Phase 2: analyse + scores + pose gate + record + DB (serial, main process).
        for cand in candidates:
            mc, boltz_mc, anchored_used = _prebuilt[cand.candidate_id]
            result = _results[cand.candidate_id]
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

            # Reference-like pose gate: did the candidate keep the reference
            # cofactor/substrate pose through MD? Anchored mutants should stay
            # reference_like; the Boltz pose is recorded as an alternative hypothesis.
            pose_gate_json = None
            if pose_gate_on and wt_ref_pdb:
                _cpdb = (ctx.paths.md_candidate(cand.candidate_id)
                         / f"{cand.candidate_id}_minimized.pdb")
                if _cpdb.exists():
                    try:
                        pgd = gate_from_pdb(wt_ref_pdb, str(_cpdb), ligand_rank=0,
                                            role=_design_role,
                                            ligand_id="design_ligand")
                        pose_gate_json = {"design_ligand": pgd.to_json()}
                        co = gate_from_pdb(wt_ref_pdb, str(_cpdb), ligand_rank=1,
                                           role="substrate", ligand_id="cosubstrate")
                        if co.status != "skipped_no_correspondence":
                            pose_gate_json["cosubstrate"] = co.to_json()
                        cand.details["pose_gate"] = pose_gate_json
                        if pgd.pocket_pose_rmsd is not None:
                            cand.scores["design_pose_rmsd"] = round(
                                pgd.pocket_pose_rmsd, 3)
                        cand.scores["pose_reference_like"] = (
                            1.0 if pgd.reference_like else 0.0)
                    except Exception as _pexc:        # noqa: BLE001
                        self.log.warning("pose gate failed for %s (%s)",
                                         cand.candidate_id, _pexc)
                if (anchored_used and boltz_mc is not None
                        and getattr(boltz_mc.structure, "pdb_path", None)):
                    try:
                        _alt = gate_from_pdb(
                            wt_ref_pdb, boltz_mc.structure.pdb_path, ligand_rank=0,
                            role=_design_role, ligand_id="design_ligand")
                        cand.details["alternative_pose_boltz"] = _alt.to_json()
                    except Exception:                 # noqa: BLE001
                        pass
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
                "md_lite_status": "valid",
                "nac_status": metrics.nac_status,
                "nac_occupancy": metrics.nac_occupancy,
                "nac_delta_vs_wt": cand.scores.get("nac_delta_vs_wt"),
                "nac": metrics.nac,
                "binding_dg": metrics.binding_dg or None,
                "binding_dg_status": ("computed" if metrics.binding_dg
                                      else "not_calculated_openmm_screening"),
                "validation_structure": ("wt_anchored" if anchored_used
                                         else ("boltz" if boltz_mc is not None
                                               else "proxy")),
                "pose_gate": pose_gate_json,
                "alternative_pose_boltz": cand.details.get("alternative_pose_boltz"),
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

        # Endpoint binding free energy (MM-PB/GBSA) confirmation tier. OpenMM
        # remains the NAC-capable screening engine and leaves binding_dg empty;
        # this optional pass runs Amber explicit-solvent MMPBSA only on the top
        # MD candidates, then copies the ΔG values into per-candidate
        # analysis/provenance without replacing the screening MD status.
        bdcfg = getattr(mdcfg, "binding_dg", None)
        if (bdcfg and getattr(bdcfg, "enabled", False)
                and backend is Backend.real and not ctx.dry_run):
            try:
                import json as _json

                amber_cfg = mdcfg.model_copy(deep=True)
                amber_cfg.engine = "amber"
                amber_cfg.solvent = "explicit"
                if getattr(amber_cfg, "reactive_geometry", None):
                    amber_cfg.reactive_geometry.enabled = False
                record_by_id = {r["candidate_id"]: r for r in md_records}
                ranked = sorted(
                    (c for c in candidates
                     if c.scores.get("md_lite_score") is not None),
                    key=lambda c: -c.scores["md_lite_score"])[: bdcfg.top_n]
                self.log.info(
                    "Amber binding_dg (MM-PB/GBSA): %d candidate(s)", len(ranked))
                for cand in ranked:
                    mc = mut_complexes.get(cand.candidate_id) or _mutant_complex(
                        wt, cand
                    )
                    res = run_md(
                        mc, cand.candidate_id, amber_cfg,
                        ctx.paths.md / "_binding_dg" / cand.candidate_id,
                        instability=cand.scores.get("instability", 0.3),
                        catalytic_positions=catalytic,
                        backend=backend, dry_run=False,
                        ligand_cache_dir=ligand_cache_dir, extra_ligands=None,
                    )
                    binding = res.binding_dg or {}
                    status = "computed" if binding else "not_computed_amber_mmpbsa"
                    cand.details["binding_dg"] = binding or None
                    cand.details["binding_dg_status"] = status
                    if res.failure_reason and not binding:
                        cand.details["binding_dg_failure"] = res.failure_reason
                    rec = record_by_id.get(cand.candidate_id)
                    if rec is not None:
                        rec["binding_dg"] = binding or None
                        rec["binding_dg_status"] = status
                    aj_path = ctx.paths.md_candidate(cand.candidate_id) / "analysis.json"
                    if aj_path.exists():
                        aj = _json.loads(aj_path.read_text())
                        aj["binding_dg"] = binding or None
                        aj["binding_dg_status"] = status
                        aj_path.write_text(_json.dumps(aj, indent=2))
                        with ctx.store.session() as s:
                            row = (s.query(MDSimulation)
                                   .filter_by(project_id=ctx.project_id,
                                              candidate_id=cand.candidate_id)
                                   .order_by(MDSimulation.md_id.desc())
                                   .first())
                            if row is not None:
                                row.analysis_json = aj
                    self.log.info("  %s: binding_dg=%s (%s)",
                                  cand.candidate_id, binding or None, res.status)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("Amber binding_dg tier failed (%s); MD scores intact",
                                 exc)

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
                    rbfe_rec = {r["candidate_id"]: r for r in md_records}
                    for cand in ranked:
                        try:
                            res = run_rbfe(
                                wt_pdb_text, lig_mol2, lig_frcmod, cand.mutations,
                                rbfe_root / cand.candidate_id,
                                multipoint=rbcfg.multipoint, n_lambda=rbcfg.n_lambda,
                                min_cyc=rbcfg.min_cyc, heat_steps=rbcfg.heat_steps,
                                prod_steps=rbcfg.prod_steps)
                        except Exception as exc:  # one candidate must not abort the RBFE stage
                            self.log.warning("  %s: RBFE failed (%s); skipped, others continue",
                                             cand.candidate_id, exc)
                            continue
                        cand.details["rbfe"] = res
                        if res.get("ddg_bind") is not None:
                            cand.scores["rbfe_ddg_bind"] = res["ddg_bind"]
                        # Persist into the on-disk MD record so RBFE reaches
                        # md_candidates.json + the report (the binding_dg tier does
                        # the same; without this the ΔΔG only ever hit the log --
                        # the s11-merge data-loss trap, cf [[resume-load-parity]]).
                        rec = rbfe_rec.get(cand.candidate_id)
                        if rec is not None:
                            rec["rbfe_ddg_bind"] = res.get("ddg_bind")
                            rec["rbfe_mode"] = res.get("mode") or res.get("skipped")
                        self.log.info("  %s: ΔΔG_bind=%s kcal/mol (%s)",
                                      cand.candidate_id, res.get("ddg_bind"),
                                      res.get("mode") or res.get("skipped"))
                else:
                    self.log.warning("RBFE skipped: WT complex has no full-atom PDB")
            except Exception as exc:  # noqa: BLE001
                self.log.warning("RBFE stage failed (%s); MD scores intact", exc)

        # Gate-stack verdict per candidate — the honest final claim from the
        # accumulated gates (stability -> reference-pose -> functional geometry ->
        # energetic). Non-fatal; mirrors into the on-disk record AND the in-memory
        # candidate (for s11 / the report).
        try:
            from evoliez.md.gate_stack import evaluate_gate_stack
            _cby = {c.candidate_id: c for c in candidates}
            _vc: dict = {}
            for _rec in md_records:
                _gj = evaluate_gate_stack(_rec).to_json()
                _rec["gate_stack"] = _gj
                _vc[_gj["verdict"]] = _vc.get(_gj["verdict"], 0) + 1
                _c = _cby.get(_rec["candidate_id"])
                if _c is not None:
                    _c.details["gate_stack"] = _gj
            if _vc:
                self.log.info("gate-stack verdicts: %s", _vc)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("gate-stack pass failed (%s); records intact", exc)

        n_pass = sum(1 for c in candidates if c.details.get("md_passed"))
        ctx.put("md_candidates", candidates)
        ctx.persist_meta("n_md_passed", n_pass)
        ctx.persist_meta("n_md_real_ran", n_ran)
        ctx.persist_meta("n_md_skipped", n_skipped)
        ctx.persist_meta("n_md_failed", n_failed)
        ctx.persist_meta("geometry_source", geometry_source)
        ctx.persist_meta("metal_setup", metal_setup)
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
            "geometry_source": geometry_source,
            "metal_setup": metal_setup,
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
