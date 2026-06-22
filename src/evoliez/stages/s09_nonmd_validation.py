"""Stage 09 - non-MD structural validation (spec section 14).

Cheaper filters before MD: stability (FoldX/Rosetta/mock), catalytic-geometry
proxy, and per-mutant redocking consistency vs the WT reference pose.
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from evoliez.adapters import foldx, rosetta, thermompnn
from evoliez.adapters.base import het_chains_in_pdb
from evoliez.context import RunContext
from evoliez.features.geometry import catalytic_distances, rmsd
from evoliez.stages.base import Stage
from evoliez.stages.s05_docking import redock_with
from evoliez.types import Candidate, Complex, Pose


def _context_chains_for(redock_structure) -> Optional[List[str]]:
    """The co-modelled context-ligand chains to keep as FIXED receptor context
    when redocking the design ligand (audit P1 #1).

    The s04/s08b complex PDB carries the design (primary) ligand at chain B and
    each co-modelled extra (cofactor / substrate, e.g. formate) at C/D/...
    (Boltz input order: protein A, primary ligand B, extras after). The design
    ligand is the one being REDOCKED, so it is removed from the receptor and
    re-added by the docker; the OTHERS stay as context. So the context chains are
    every HETATM chain EXCEPT the first (the design/primary ligand) — exactly the
    ``[c for c in all_chains if c != design_chain]`` derivation s05 uses.

    Returns ``None`` (no context) when the redock structure has no real PDB (the
    WT-coords proxy is CA-only -> ``full_atom_receptor_pdb`` degrades honestly
    anyway) or carries at most the design ligand. gnina/vina keep these chains;
    diffdock receives-but-ignores them (protein-only), matching ``redock_with``.
    """
    pdb = getattr(redock_structure, "pdb_path", None)
    if not pdb:
        return None
    chains = het_chains_in_pdb(pdb)
    if len(chains) <= 1:
        return None
    # chain[0] = the design/primary ligand being redocked; keep the rest.
    return [c for c in chains[1:]] or None


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


# --- Multi-signal failure detection for real-Boltz mutants (s09) -------------------
# A candidate WITH a real s08b Boltz mutant complex carries FOUR INDEPENDENT tests; no
# single one decides. DiffDock is NOT ground truth for a large, highly-charged,
# flexible cofactor like NADP (it is itself sensitive to ligand size/charge and the
# formate context), so a DiffDock escape ALONE is only "caution" -- a pose the blind
# global search would not re-select, worth re-verifying, not a confirmed failure. A
# failure is a CORROBORATED escape: DiffDock escape co-occurring with a degraded Boltz
# binding signal (d_ligand_iptm) and/or disrupted catalytic geometry. Thresholds are
# explicit and recorded in the validated_candidates provenance so they can be tuned.
_GNINA_RETAIN_A = 3.0   # gnina reference-local RMSD <= this -> pose retained near Boltz pose
_DD_ESCAPE_A = 4.0      # diffdock global RMSD > this        -> pose escaped under blind search
_IPTM_DROP = -0.05      # d_ligand_iptm < this               -> Boltz says target binding weakened
_CAT_DISRUPT = 2.5      # catalytic_geometry_penalty > this  -> active-site geometry disrupted


def _classify_real_mutant(cand: Candidate, gnina_pose, diffdock_pose):
    """Grade a real-Boltz mutant from the four independent tests. Returns
    (class, reason) with class in {pass, caution, penalty, reject}:
      pass    - local redock retained, no global escape, binding + mechanism intact
      caution - DiffDock-only escape (Boltz + catalytic intact) or a single off signal;
                KEPT for re-verification, NOT rejected (DiffDock is not ground truth)
      penalty - escape corroborated by ONE structural signal (binding or mechanism)
      reject  - escape corroborated by BOTH (a confirmed NADP-binding/geometry failure)
    Only ``reject`` gates the candidate out; the ranking demotes penalty/caution via
    the worst-of-both consistency and the real Boltz delta already in _md_key."""
    g = gnina_pose.rmsd_to_reference if gnina_pose is not None else None
    d = diffdock_pose.rmsd_to_reference if diffdock_pose is not None else None
    retain = g is not None and g <= _GNINA_RETAIN_A
    escape = d is None or d > _DD_ESCAPE_A
    binding_bad = cand.scores.get("d_ligand_iptm", 0.0) < _IPTM_DROP
    mech_bad = cand.scores.get("catalytic_geometry_penalty", 0.0) > _CAT_DISRUPT
    struct_bad = int(binding_bad) + int(mech_bad)
    corr = " + ".join(s for s, b in (("Boltz binding drop", binding_bad),
                                     ("catalytic disruption", mech_bad)) if b)
    if escape and struct_bad == 2:
        return "reject", "confirmed NADP-binding failure: DiffDock escape + " + corr
    if escape and struct_bad == 1:
        return "penalty", "DiffDock escape corroborated by " + corr
    if escape:
        return "caution", "DiffDock-only escape; Boltz binding + catalytic intact -- re-verify"
    if struct_bad == 2:
        return "penalty", "structural degradation without DiffDock escape: " + corr
    if struct_bad == 1 or not retain:
        return "caution", ("single structural signal off: " + corr) if corr else \
               "local redock did not retain the pose -- re-verify"
    return "pass", None


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

        # --- GPU-batched redocking (audit P1 #3) ----------------------------
        # The naive per-candidate loop runs each GPU docker (gnina/diffdock) once
        # PER CANDIDATE, so ~600 candidates -> ~600 DiffDock model loads + ~600
        # gnina launches. Instead, reuse the SAME GPU-amortising skeleton s06b's
        # per-rep augmentation uses (diffdock.redock_batch one-model-load-per-GPU
        # + a gnina per-GPU worker queue) to redock all candidates with ONE model
        # load per GPU. Gated EXACTLY like s06b's batched ensemble — real backend
        # + >1 pinned GPU — because the mock backend's batch vs per-target
        # base_instability differ (see diffdock.redock_batch), so mock / single-
        # GPU keep the per-candidate path (byte-identical to before). The poses
        # are numerically identical per candidate; only model loads drop.
        gpu_list = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
        batch_methods = {"gnina", "diffdock"} & set(dcfg.methods)
        use_batch = (backend.value == "real" and len(gpu_list) > 1
                     and not ctx.dry_run and bool(batch_methods))
        # {candidate_id: {method: Pose}} pre-docked across GPUs; read by
        # _process_candidate instead of a per-candidate redock_with for the
        # batched methods. Non-batched methods (e.g. vina, CPU) still run inline.
        batched_poses: Dict[str, Dict[str, Pose]] = {}

        def _redock_structure_for(cand: Candidate):
            """The receptor + design-ligand reference pose + context chains for ONE
            candidate's redock. Shared by the inline path and the batch precompute so
            both dock the IDENTICAL target against the IDENTICAL reference. Returns
            (redock_structure, reference_atoms, context_chains)."""
            redock_cx = mutant_complexes.get(cand.candidate_id)
            redock_structure = (redock_cx.structure if redock_cx is not None
                                else _mutant_complex(wt, cand).structure)
            # Reference pose for the redock RMSD. Redocking INTO a real s08b Boltz
            # mutant complex but scoring RMSD against the WT-frame reference yields a
            # 12-32 A CROSS-FRAME distance (Boltz predicts each complex in its own
            # frame) -> redocking_consistency collapses to 0 -> every real-Boltz
            # mutant spuriously fails s09. Use the complex's OWN ligand pose (same
            # frame as redock_structure): "do gnina/diffdock reproduce THIS pose?" —
            # the identical docker-vs-Boltz check the wt_proxy path applies to WT.
            ref_for_cand = (redock_cx.ligand.atoms if redock_cx is not None
                            else ref_atoms)
            cx_chains = _context_chains_for(redock_structure)
            return redock_structure, ref_for_cand, cx_chains

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
            # Redock RMSD reference: the mutant's OWN Boltz ligand pose (shares
            # redock_structure's frame) when a real complex exists, else the WT
            # reference — mirrors _redock_structure_for. Without it the redock into
            # the independently-framed Boltz mutant is scored against the WT-frame
            # pose (12-32 A) and consistency collapses to 0 for every real mutant.
            ref_for_cand = (redock_cx.ligand.atoms if redock_cx is not None
                            else ref_atoms)
            cand.details["redock_structure_source"] = (
                "mutant_boltz" if redock_cx is not None else "wt_proxy")
            # Co-modelled context-ligand chains kept as FIXED receptor context
            # when redocking the design ligand (audit P1 #1): without this the
            # full-atom receptor drops EVERY HETATM, so e.g. NADP is redocked into
            # a formate-LESS pocket and the catalytic-geometry consistency is
            # measured against the wrong context. gnina/vina keep these chains;
            # diffdock receives-but-ignores them (protein-only) — same contract as
            # redock_with / s05.
            cx_chains = _context_chains_for(redock_structure)
            cand.details["redock_context_chains"] = list(cx_chains or [])

            if scfg.method == "rosetta":
                stab = rosetta.estimate_stability(
                    cand.candidate_id, mc.structure, cand.mutations,
                    ctx.paths.validation / "rosetta", backend=backend,
                    dry_run=ctx.dry_run,
                )
            elif scfg.method == "thermompnn":
                stab = thermompnn.estimate_stability(
                    cand.candidate_id, mc.structure, cand.mutations, scfg,
                    ctx.paths.validation / "thermompnn", backend=backend,
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
            # surfaces disagreement. The batched methods (real multi-GPU) are
            # served from the pre-docked map; everything else runs inline — the
            # resulting Pose per (candidate, method) is identical either way.
            pre = batched_poses.get(cand.candidate_id, {})
            poses = {
                method: (pre[method] if method in pre else redock_with(
                    method, ctx, cand.candidate_id, redock_structure,
                    ref_for_cand, dcfg, ctx.paths.validation / "redock", inst,
                    wt.ligand.smiles, stage_name=self.name,
                    context_chains=cx_chains,
                ))
                for method in dcfg.methods
            }
            # A Pose.score is None when the engine produced a pose but NO
            # parseable score (DiffDock absent/sentinel confidence — diffdock.py).
            # Keep None in the provenance dict (genuinely unscored) and NEVER feed
            # it to round()/arithmetic; downstream stats must skip it, never read
            # it as 0 (audit P1 #3).
            cand.details["redock"] = {
                m: {"score": (round(p.score, 3) if p.score is not None else None),
                    "rmsd_to_reference": (round(p.rmsd_to_reference, 3)
                                          if p.rmsd_to_reference is not None else None)}
                for m, p in poses.items()
            }
            # docking_score = the primary method's pose score, or None when that
            # pose is unscored. Recorded as-is; the kept-filter / MD selection /
            # over-binding numerics below all treat None as UNKNOWN (never 0).
            primary_pose = poses[dcfg.methods[0]]
            cand.scores["docking_score"] = primary_pose.score
            rmsds = [p.rmsd_to_reference for p in poses.values()]
            # Consistency gate = worst (largest) RMSD across methods; an unverifiable
            # pose (None) propagates as unknown -> neutral, never a free pass.
            #
            # For a real-Boltz mutant we deliberately KEEP DiffDock in the gate (no
            # gnina-only shortcut). gnina's autobox sits on the mutant's OWN Boltz
            # pose, so its ~2 A reproduction is partly self-fulfilling; DiffDock
            # (blind, box-free) is the only INDEPENDENT check. Its large deviation on
            # the folded mutants is CORROBORATED, not docker noise: DiffDock's own
            # confidence collapses (median -2.8 vs -0.8 for WT) and the catalytic-
            # geometry penalty shows NADP shifted 2-6 A off the WT position -- both
            # consistent with the negative Boltz ligand-iptm delta. The honest reading
            # is that these designs disrupt NADP binding, so a low redocking_
            # consistency is CORRECT and must not be masked. gnina/DiffDock RMSDs +
            # docking_method_spread + the DiffDock confidence stay in the provenance.
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

                # Annotate the mechanism on the REAL s08b mutant complex when one
                # exists (audit P2 #6), consistent with the catalytic-geometry
                # path above: ``mc`` is only WT coords + a residue-identity swap,
                # so its TS geometry ~= WT and the mutation's mechanistic effect
                # is invisible. The Boltz mutant has its own backbone + pose, so
                # ts_geometry_score / catalytic_geometry_deviation are meaningful.
                # Fall back to the WT proxy when no real mutant complex exists.
                mech_cx = redock_cx if redock_cx is not None else mc
                mut_mech = annotate(
                    mech_cx, catalytic_positions=catalytic,
                    cofactor=ctx.config.input.cofactor,
                    annotation_file=adv.mechanism_annotation_file,
                )
                cand.details["mechanism_source"] = (
                    "mutant_boltz" if redock_cx is not None else "wt_proxy")
                cand.scores["ts_geometry_score"] = mut_mech.ts_geometry_score
                # docking_score may be None (unscored primary pose). Pass 0.0 to
                # the over-binding numeric ONLY as "no over-binding evidence" — we
                # cannot claim a candidate over-binds without a score (treating
                # None as a real, very-negative affinity would fabricate a
                # penalty). redocking_consistency is always a real float here.
                ds = cand.scores["docking_score"]
                negp = negative_penalties(
                    cand, wt_mech=wt_mech, mut_mech=mut_mech,
                    position_features=pfeats,
                    catalytic_positions=catalytic,
                    buried_fraction=cand.details.get("features", {}).get(
                        "buried_fraction", 0.5),
                    docking_score=(ds if ds is not None else 0.0),
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
            # Redock gate. For a candidate WITH a real s08b Boltz complex, do NOT
            # auto-reject on a DiffDock-only escape (the worst-of-both ligand_escape):
            # DiffDock is not ground truth for NADP. Run the 4-signal failure
            # detection and reject ONLY a corroborated failure; caution/penalty are
            # kept (the consistency + real Boltz delta in _md_key demote them). Proxy
            # candidates keep the same-frame escape filter -- there every method docks
            # the identical WT-frame reference, so an escape is a genuine displacement.
            cand.details.pop("failure_mode", None)
            if redock_cx is not None:
                vclass, why = _classify_real_mutant(
                    cand, poses.get("gnina"), poses.get("diffdock"))
                cand.details["failure_mode"] = vclass
                if vclass == "reject":
                    reasons.append(why)
            elif ligand_escape:
                reasons.append("ligand displaced on redocking")
            cand.details.pop("nonmd_rejected", None)  # clear stale flag on resume
            if reasons:
                cand.details["nonmd_rejected"] = "; ".join(reasons)
            return cand

        # GPU-BATCH PRECOMPUTE (audit P1 #3): on the real multi-GPU server, dock
        # the batched engines (gnina/diffdock) for ALL candidates up front with
        # ONE model load per GPU, reusing s06b's diffdock-batch + gnina-queue
        # schedulers (factored into adapters.redock_batch). _process_candidate
        # then reads each pose from ``batched_poses`` instead of redocking per
        # candidate. The pose per (candidate, method) is identical to the inline
        # path; only model loads drop (~600 -> n_GPUs). Wrapped so a batch failure
        # falls back to the per-candidate inline path rather than crashing the
        # stage (the inline redock_with still runs for any method not pre-docked).
        if use_batch:
            try:
                from evoliez.adapters import redock_batch as _rb

                tasks = []
                for cand in candidates:
                    struct, ref, cxc = _redock_structure_for(cand)
                    tasks.append((cand.candidate_id, struct, ref,
                                  wt.ligand.smiles, cxc))
                root = ctx.paths.validation / "redock"
                if "diffdock" in batch_methods:
                    dd = _rb.run_diffdock_batches(
                        tasks, gpu_list, dcfg, backend=backend, root=root,
                        log=self.log)
                    for cid, plist in dd.items():
                        if plist:  # best rank first == redock_with's [0]
                            batched_poses.setdefault(cid, {})["diffdock"] = plist[0]
                if "gnina" in batch_methods:
                    gq = _rb.run_gnina_queue(
                        tasks, gpu_list, dcfg, backend=backend, root=root,
                        log=self.log)
                    for cid, pose in gq.items():
                        batched_poses.setdefault(cid, {})["gnina"] = pose
                self.log.info(
                    "s09 GPU-batched redock: %d candidate(s) x %s over %d GPU(s) "
                    "-> %d model load(s)/engine (was per-candidate)",
                    len(candidates), sorted(batch_methods), len(gpu_list),
                    len(gpu_list))
            except Exception as exc:
                self.log.warning(
                    "s09 GPU-batched redock failed (%s); falling back to the "
                    "per-candidate inline redock path", exc)
                batched_poses.clear()

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

        # advance the best survivors to MD. Base score = ML rank + redocking
        # consistency - ddG penalty (the proxy-only formula). When s08b spent a
        # REAL mutant-Boltz re-prediction on a candidate (top-N; boltz_delta_source
        # == "real"), FOLD that real Δ into the selection so the MD budget is spent
        # on the candidates the real Boltz says actually improved/worsened, not on
        # what the cheap proxy guessed (audit P2 #4). d_ligand_iptm > 0 = the
        # mutant binds the design-target ligand better than WT (same sign as
        # ranking.score's mutant_boltz_gain); d_key_distance > 0 = catalytic
        # contacts drifted (penalised). Candidates WITHOUT a real Δ (proxy / mock /
        # dry-run / none) fall back to exactly the proxy-only base — never mixing a
        # proxy Δ into this selection. Weights match ranking.score's scale
        # (mutant_boltz_gain) so MD selection and final scoring agree in spirit.
        _W_REAL_DLIGAND = 2.0     # reward real ΔBoltz binding gain
        _W_REAL_DKEYDIST = 0.5    # penalise real catalytic-geometry drift

        def _md_key(c: Candidate) -> float:
            base = (
                c.scores.get("ml_score", 0.0)
                + c.scores.get("redocking_consistency", 0.0)
                - 0.2 * max(0.0, c.scores.get("ddg_fold", 0.0))
            )
            if c.details.get("boltz_delta_source") == "real":
                base += _W_REAL_DLIGAND * c.scores.get("d_ligand_iptm", 0.0)
                base -= _W_REAL_DKEYDIST * abs(c.scores.get("d_key_distance", 0.0))
            return base

        kept.sort(key=_md_key, reverse=True)
        _n_md = ctx.config.validation.md.top_candidates
        if getattr(ctx.config.validation.md, "require_real_structure", True):
            # Paper-grade MD set: only candidates with a REAL s08b Boltz mutant
            # complex (boltz_delta_source=="real"), so s10 never validates a WT-coords
            # identity-swap proxy. The proxy-top picks that were never folded are
            # excluded even when _md_key ranks them high — they carry no real Boltz Δ
            # penalty, an unfair advantage over the folded set. Fall back to the full
            # kept set ONLY when nothing was folded (configs without s08b).
            _real = [c for c in kept if c.details.get("boltz_delta_source") == "real"]
            md_top = (_real if _real else kept)[:_n_md]
            if _real and len(_real) < _n_md:
                self.log.info(
                    "MD shortlist: %d real-Boltz-structure candidate(s) (< budget "
                    "%d) — NOT padding with WT-proxy structures", len(_real), _n_md)
        else:
            md_top = kept[:_n_md]
        ctx.put("candidates", candidates)
        ctx.put("validated_candidates", kept)
        ctx.put("md_candidates", md_top)
        # Persist per-candidate validation scores (report + reproducibility + audit;
        # previously ctx.put in-memory only, so the s09 results were lost after the
        # run and could not be verified against stored data). The s09 report and the
        # anti-regression verification read this provenance.
        import json as _json
        _prov = ctx.paths.reports / "provenance"
        _prov.mkdir(parents=True, exist_ok=True)
        _kept_ids = {c.candidate_id for c in kept}
        _md_ids = {c.candidate_id for c in md_top}
        (_prov / "validated_candidates.json").write_text(_json.dumps([{
            "candidate_id": c.candidate_id,
            "mutation_string": ";".join(f"{m.wt}{m.position}{m.mut}" for m in c.mutations),
            "ddg_fold": c.scores.get("ddg_fold"),
            "stability_score": c.scores.get("stability_score"),
            "stability_unavailable": c.details.get("stability_unavailable", False),
            "redocking_consistency": c.scores.get("redocking_consistency"),
            "docking_uncertainty": c.scores.get("docking_uncertainty"),
            "catalytic_geometry_penalty": c.scores.get("catalytic_geometry_penalty"),
            "redock": c.details.get("redock"),
            "redock_context_chains": c.details.get("redock_context_chains", []),
            "mechanism_source": c.details.get("mechanism_source"),
            "catalytic_geometry_source": c.details.get("catalytic_geometry_source"),
            "boltz_delta_source": c.details.get("boltz_delta_source"),
            "failure_mode": c.details.get("failure_mode"),
            "d_ligand_iptm": c.scores.get("d_ligand_iptm"),
            "passed": c.candidate_id in _kept_ids,
            "for_md": c.candidate_id in _md_ids,
        } for c in candidates], indent=2, default=str))
        ctx.persist_meta("n_after_nonmd", len(kept))
        ctx.persist_meta("n_for_md", len(md_top))
        try:  # auto-generate the s09 report (+ refresh s08 with the fold cross-ref)
            from evoliez.io.rerank_report import write_rerank_report
            from evoliez.io.validation_report import write_validation_report
            write_validation_report(ctx.paths.root)
            write_rerank_report(ctx.paths.root)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("s09 report generation skipped (%s)", exc)
        mode = ("dry-run preview" if ctx.dry_run
                else getattr(backend, "value", str(backend)))
        self.log.info(
            "non-MD validation [%s]: %d/%d passed; %d advance to MD",
            mode, len(kept), len(candidates), len(md_top),
        )
        _ = rmsd  # geometry helper kept importable for real-backend extensions
