"""Stage 09 - non-MD structural validation (spec section 14).

Cheaper filters before MD: stability (FoldX/Rosetta/mock), catalytic-geometry
proxy, and per-mutant redocking consistency vs the WT reference pose.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import List

from evoliez.adapters import foldx, rosetta
from evoliez.context import RunContext
from evoliez.features.geometry import catalytic_distances, rmsd
from evoliez.stages.base import Stage
from evoliez.stages.s05_docking import redock_with
from evoliez.types import Candidate, Complex


def _mutant_complex(wt: Complex, cand: Candidate) -> Complex:
    """Approximate mutant complex: WT geometry with substituted residue
    identities (coords unchanged). Cheap proxy for pre-MD filtering; MD then
    relaxes it. Real backend can re-predict per mutant if configured."""
    mc = copy.deepcopy(wt)
    by_pos = {r.index: r for r in mc.structure.residues}
    seq = list(mc.structure.sequence)
    for m in cand.mutations:
        r = by_pos.get(m.position)
        if r is not None:
            r.aa = m.mut
            if 0 < m.position <= len(seq):
                seq[m.position - 1] = m.mut
    mc.structure.sequence = "".join(seq)
    return mc


def _instability(cand: Candidate) -> float:
    """0..1 disruption proxy driving mock redock/MD behaviour."""
    ddg = cand.scores.get("ddg_fold", 0.0)
    clash = cand.scores.get("clash_score", 0.0)
    cons = cand.details.get("features", {}).get("conservation", 0.5)
    val = 0.12 * max(0.0, ddg) + 0.25 * clash + 0.4 * cons
    return float(max(0.0, min(1.0, val)))


def _redock_metrics(rmsd):
    """(redocking_consistency, docking_uncertainty, ligand_escape) from the
    pose's RMSD-to-reference. ``rmsd`` is None when the docked pose could NOT be
    graph-verified vs the reference: treat as UNKNOWN (neutral consistency,
    surfaced uncertainty), NEVER as a perfect (consistency 1.0) redock that also
    silently bypasses the ligand-escape filter."""
    if rmsd is None:
        return 0.5, 0.5, False
    c = max(0.0, 1.0 - rmsd / 4.0)
    return round(c, 4), round(1.0 - c, 4), rmsd > 4.5


class NonMDValidationStage(Stage):
    name = "s09_nonmd"

    def run(self, ctx: RunContext) -> None:
        wt = ctx.require("wt_complex")
        candidates: List[Candidate] = ctx.require("redock_candidates")
        catalytic = ctx.require("catalytic_positions")
        ref_atoms = ctx.require("reference_atoms")
        scfg = ctx.config.validation.stability
        dcfg = ctx.config.validation.redocking
        backend = ctx.config.backend_for(self.name)

        wt_cat = catalytic_distances(wt.structure, ref_atoms, catalytic)
        wt_mech = ctx.get("mechanism")
        pfeats = ctx.get("position_features", [])
        adv = ctx.config.advanced
        # Real per-mutant Boltz structures from s08b (top-N only). Redocking the
        # ACTUAL mutant pocket is more meaningful than the WT-coords proxy
        # (audit P1 #2). Stability stays on the proxy: FoldX/Rosetta build the
        # mutant from the WT structure + mutation list, so they want WT coords.
        mutant_complexes = ctx.get("mutant_complexes", {}) or {}

        # --- shared-server CPU bound for the fan-out below -------------------
        # The watchdog throttles us to 0.2 core if ~48 cores stay busy 10+ min,
        # and vina/gnina/FoldX each grab cores. So BOUND total concurrency:
        #   peak cores <= MAX_WORKERS x CPU_PER_DOCK = 4 x 4 = 16  (<= ~16-24).
        # MAX_WORKERS is a small hardcoded constant (no config field, by design).
        # CPU_PER_DOCK is pushed into vina via `--cpu` (DockingConfig.cpu): the
        # stock default is 0 = "all cores", which 4-wide would oversubscribe the
        # box, so we cap it. Each worker runs its dockers SEQUENTIALLY (the
        # methods dict-comp redocks one method at a time) and FoldX/Rosetta
        # before them, so a single worker draws at most CPU_PER_DOCK cores at
        # once -> 4 workers x 4 = 16 peak, safely under the 24/48 thresholds.
        # If the user already configured a SMALLER positive cpu cap, keep it
        # (min); never raise their cap.
        MAX_WORKERS = 4
        CPU_PER_DOCK = 4
        user_cpu = getattr(dcfg, "cpu", 0) or 0
        capped_cpu = (min(user_cpu, CPU_PER_DOCK) if user_cpu > 0 else CPU_PER_DOCK)
        dcfg = dcfg.model_copy(update={"cpu": capped_cpu})

        def _process_candidate(cand: Candidate) -> Candidate:
            """Validate ONE candidate (stability + redocking + geometry). Run by
            the ThreadPoolExecutor below. Thread-safe: each call mutates only its
            own ``cand.scores``/``cand.details`` (distinct objects) and writes to
            a candidate_id-scoped workdir, while reading ctx/wt/catalytic/wt_cat
            read-only. The heavy work (FoldX/Rosetta/gnina/vina) is in
            subprocesses, so this is I/O-bound (GIL released in subprocess.run) —
            a thread pool is the right tool, not a process pool. Returns ``cand``;
            the caller applies the kept-filter + MD selection deterministically."""
            mc = _mutant_complex(wt, cand)
            redock_cx = mutant_complexes.get(cand.candidate_id)
            redock_structure = (redock_cx.structure if redock_cx is not None
                                else mc.structure)
            cand.details["redock_structure_source"] = (
                "mutant_boltz" if redock_cx is not None else "wt_proxy")

            if scfg.method == "rosetta":
                stab = rosetta.estimate_stability(
                    cand.candidate_id, mc.structure, cand.mutations,
                    ctx.paths.validation / "rosetta", backend=backend,
                    dry_run=ctx.dry_run,
                )
            else:
                stab = foldx.estimate_stability(
                    cand.candidate_id, mc.structure, cand.mutations, scfg,
                    ctx.paths.validation / "foldx", backend=backend,
                    dry_run=ctx.dry_run,
                )
            ddg = stab.get("ddg_fold")
            cand.scores["clash_score"] = stab.get("clash_score", 0.0)
            # positive stability contribution (was dead: score.py reads
            # "stability_score" which nothing set). Favourable/neutral ddG
            # (<=0) -> 1.0; at the allowed cap -> 0.0.
            if ddg is None:
                # stability measurement FAILED (tool produced no parseable ddG).
                # Do NOT default to 0.0 - that is the BEST value (zero penalty,
                # stability_score 1.0, passes the ddG filter), so a tool failure
                # would masquerade as a maximally-stable mutant. Surface it, give
                # NO positive reward, and route to MD rather than reject.
                self.log.warning(
                    "stability ddG unavailable for %s; not scored as stable",
                    cand.candidate_id,
                )
                cand.details["stability_unavailable"] = True
                cand.scores["ddg_fold"] = 0.0       # neutral for penalty/filter
                cand.scores["stability_score"] = 0.0  # no positive reward (was 1.0)
            else:
                cand.scores["ddg_fold"] = ddg
                cap = scfg.max_ddg_allowed or 2.5
                cand.scores["stability_score"] = round(
                    max(0.0, 1.0 - max(0.0, ddg) / cap), 4
                )

            inst = _instability(cand)
            # Run ALL configured dockers (not just methods[0]) so multi-docker
            # AGREEMENT is real: a candidate counts as a consistent redock only
            # if EVERY method reproduces the reference pose. Per-method score +
            # RMSD are recorded for the validation table; the cross-method spread
            # surfaces disagreement.
            poses = {
                method: redock_with(
                    method, ctx, cand.candidate_id, redock_structure,
                    ref_atoms, dcfg, ctx.paths.validation / "redock", inst,
                    wt.ligand.smiles, stage_name=self.name,
                )
                for method in dcfg.methods
            }
            cand.details["redock"] = {
                m: {"score": round(p.score, 3),
                    "rmsd_to_reference": (round(p.rmsd_to_reference, 3)
                                          if p.rmsd_to_reference is not None else None)}
                for m, p in poses.items()
            }
            cand.scores["docking_score"] = poses[dcfg.methods[0]].score
            rmsds = [p.rmsd_to_reference for p in poses.values()]
            # worst (largest) RMSD across methods; an unverifiable pose (None)
            # propagates as unknown -> neutral consistency, never a free pass.
            worst = None if any(r is None for r in rmsds) else max(rmsds)
            consistency, uncertainty, ligand_escape = _redock_metrics(worst)
            cand.scores["redocking_consistency"] = consistency
            cand.scores["docking_uncertainty"] = uncertainty
            verified = [r for r in rmsds if r is not None]
            if len(verified) > 1:                    # multi-docker disagreement
                cand.scores["docking_method_spread"] = round(
                    max(verified) - min(verified), 3)

            # Catalytic-geometry penalty (audit P1 #5): measure it on the REAL
            # mutant pose when s08b produced one. ``_mutant_complex`` is just WT
            # coords + a residue-identity swap, so catalytic_distances on it ~=
            # WT -> penalty ~= 0 (it can't see backbone/pose change). The s08b
            # Boltz mutant has its OWN backbone AND its OWN ligand pose, so using
            # its structure + its ligand atoms makes the penalty actually capture
            # how the mutation moved the active site. Fall back to the proxy (WT
            # coords + WT reference ligand) when no real mutant complex exists.
            if redock_cx is not None:
                geom_structure = redock_cx.structure
                geom_ligand = redock_cx.ligand.atoms
            else:
                geom_structure = mc.structure
                geom_ligand = ref_atoms
            mut_cat = catalytic_distances(geom_structure, geom_ligand, catalytic)
            geom_pen = 0.0
            for k, v in mut_cat.items():
                geom_pen += abs(v - wt_cat.get(k, v)) / 4.0
            cand.scores["catalytic_geometry_penalty"] = round(geom_pen, 4)
            cand.details["catalytic_geometry_source"] = (
                "mutant_boltz" if redock_cx is not None else "wt_proxy")
            cand.scores["key_contact_preservation"] = round(consistency, 4)
            cand.scores["complex_confidence"] = wt.confidence
            cand.scores["instability"] = round(inst, 4)

            # mechanism-aware negative design (user §1, §4)
            if adv.negative_design and wt_mech is not None:
                from evoliez.features.mechanism import annotate
                from evoliez.ranking.negative_design import negative_penalties

                mut_mech = annotate(
                    mc, catalytic_positions=catalytic,
                    cofactor=ctx.config.input.cofactor,
                    annotation_file=adv.mechanism_annotation_file,
                )
                cand.scores["ts_geometry_score"] = mut_mech.ts_geometry_score
                negp = negative_penalties(
                    cand, wt_mech=wt_mech, mut_mech=mut_mech,
                    position_features=pfeats,
                    catalytic_positions=catalytic,
                    buried_fraction=cand.details.get("features", {}).get(
                        "buried_fraction", 0.5),
                    docking_score=cand.scores["docking_score"],
                    redocking_consistency=consistency,
                )
                cand.scores.update(negp)

            # filters (spec 14.3). Record the reject reason on the candidate; the
            # serial KEPT list is rebuilt deterministically after the fan-out so
            # completion order never leaks into the result.
            reasons = []
            if cand.scores["ddg_fold"] > scfg.max_ddg_allowed:
                reasons.append(f"ddG {cand.scores['ddg_fold']:.2f} > "
                               f"{scfg.max_ddg_allowed}")
            if ligand_escape:
                reasons.append("ligand displaced on redocking")
            cand.details.pop("nonmd_rejected", None)  # clear stale flag on resume
            if reasons:
                cand.details["nonmd_rejected"] = "; ".join(reasons)
            return cand

        # Fan the per-candidate work out across a BOUNDED thread pool. Each
        # candidate is independent (own scores/details + own scoped workdir) and
        # blocks in subprocesses, so threads overlap the I/O wait without GIL
        # contention. We do NOT consume results in completion order: we wait for
        # ALL of them, then rebuild `kept` by iterating `candidates` in their
        # original order, so the kept set + MD selection are byte-identical to the
        # serial version for a given input.
        workers = max(1, min(MAX_WORKERS, len(candidates)))
        if workers == 1:
            for cand in candidates:
                _process_candidate(cand)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # list() forces every future to complete (and re-raises any
                # worker exception) before we proceed to selection.
                list(pool.map(_process_candidate, candidates))

        kept: List[Candidate] = [
            c for c in candidates if "nonmd_rejected" not in c.details
        ]

        # advance the best survivors to MD
        kept.sort(
            key=lambda c: (
                c.scores.get("ml_score", 0.0)
                + c.scores.get("redocking_consistency", 0.0)
                - 0.2 * max(0.0, c.scores.get("ddg_fold", 0.0))
            ),
            reverse=True,
        )
        md_top = kept[: ctx.config.validation.md.top_candidates]
        ctx.put("candidates", candidates)
        ctx.put("validated_candidates", kept)
        ctx.put("md_candidates", md_top)
        ctx.persist_meta("n_after_nonmd", len(kept))
        ctx.persist_meta("n_for_md", len(md_top))
        mode = ("dry-run preview" if ctx.dry_run
                else getattr(backend, "value", str(backend)))
        self.log.info(
            "non-MD validation [%s]: %d/%d passed; %d advance to MD",
            mode, len(kept), len(candidates), len(md_top),
        )
        _ = rmsd  # geometry helper kept importable for real-backend extensions
