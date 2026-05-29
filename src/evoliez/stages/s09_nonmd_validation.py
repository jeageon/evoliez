"""Stage 09 - non-MD structural validation (spec section 14).

Cheaper filters before MD: stability (FoldX/Rosetta/mock), catalytic-geometry
proxy, and per-mutant redocking consistency vs the WT reference pose.
"""

from __future__ import annotations

import copy
from typing import List

from evoliez.adapters import foldx, rosetta
from evoliez.context import RunContext
from evoliez.features.geometry import catalytic_distances, dist, rmsd
from evoliez.ml.pose_validity import pose_sanity
from evoliez.stages.base import Stage
from evoliez.stages.s05_docking import redock_with
from evoliez.types import Candidate, Complex, LigandAtom, Pose, ProteinStructure


def _apply_binding_site_floor(
    ranked: List[Candidate],
    *,
    top_n: int,
    binding_site: set,
    n_reserve: int,
):
    """For each ``binding_site`` position with fewer than ``n_reserve``
    candidates in the current top-N, promote the best-ranked tail
    candidate touching that position, displacing the lowest-ranked
    top-N entry that is NOT itself covering a binding-site position.

    Same shape as
    :func:`evoliez.stages.s08_reranker._promote_binding_site_reservations`
    but operates on the s09→s10 transition (post nonmd_validation, pre
    real MD). Returns ``(new_top, n_promoted)``.
    """
    if n_reserve <= 0 or not binding_site or not ranked:
        return ranked[:top_n], 0

    def _positions(c):
        return {m.position for m in c.mutations}

    top = list(ranked[:top_n])
    tail = list(ranked[top_n:])

    # Count current coverage per binding-site position in top-N.
    coverage = {pos: 0 for pos in binding_site}
    for c in top:
        for p in _positions(c) & binding_site:
            coverage[p] += 1

    n_promoted = 0
    for pos in sorted(binding_site):
        need = n_reserve - coverage[pos]
        while need > 0:
            # Best-scoring tail candidate touching this position.
            cand_promote = next(
                (c for c in tail if pos in _positions(c)),
                None,
            )
            if cand_promote is None:
                break
            # Lowest-scoring top entry NOT covering ANY binding site →
            # displace it. NON-DESTRUCTIVE: if every top entry covers
            # SOME binding-site position, abort this position's
            # promotion rather than overwriting an already-promoted BS
            # candidate. This is the cheap-run-4 fix: the previous
            # fallback (`len(top)-1`) silently replaced earlier
            # promotions, so 16 "promoted" log lines turned into only
            # 1 net BS addition in top-12.
            displace_idx = next(
                (i for i in range(len(top) - 1, -1, -1)
                 if not (_positions(top[i]) & binding_site)),
                None,
            )
            if displace_idx is None:
                break  # honestly out of non-BS displacement targets
            top[displace_idx] = cand_promote
            tail.remove(cand_promote)
            coverage[pos] += 1
            need -= 1
            n_promoted += 1

    # Resort by ml_score so the new top stays rank-ordered.
    top.sort(
        key=lambda c: (
            c.scores.get("ml_score", 0.0)
            + c.scores.get("redocking_consistency", 0.0)
            - 0.2 * max(0.0, c.scores.get("ddg_fold", 0.0))
        ),
        reverse=True,
    )
    return top, n_promoted


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


_PLIF_CONTACT_RADIUS = 5.0       # heavy atom <-> CA contact threshold in Angstrom


def _contact_set(
    ligand_atoms, structure: ProteinStructure,
    radius: float = _PLIF_CONTACT_RADIUS,
) -> set:
    """Cheap PLIF stand-in: (residue_index, ligand_atom_id) pairs within
    ``radius`` Å. Used for `plif_recovery` = |pred ∩ ref| / |ref|, so a
    docking method that loses key cofactor/catalytic contacts can't quietly
    win on score alone."""
    out = set()
    for la in ligand_atoms or []:
        for r in structure.residues:
            if dist(la.coord, r.ca) <= radius:
                out.add((r.index, la.id))
    return out


def _plif_recovery(pose: Pose, ref_atoms, structure: ProteinStructure) -> float:
    ref = _contact_set(ref_atoms, structure)
    if not ref:
        return 0.0
    pred = _contact_set(pose.ligand_atoms or ref_atoms, structure)
    return round(len(ref & pred) / len(ref), 4)


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
        kept: List[Candidate] = []
        for cand in candidates:
            mc = _mutant_complex(wt, cand)

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
            ddg = stab.get("ddg_fold", 0.0)
            cand.scores["ddg_fold"] = ddg
            cand.scores["clash_score"] = stab.get("clash_score", 0.0)
            # positive stability contribution (was dead: score.py reads
            # "stability_score" which nothing set). Favourable/neutral ddG
            # (<=0) -> 1.0; at the allowed cap -> 0.0.
            cap = scfg.max_ddg_allowed or 2.5
            cand.scores["stability_score"] = round(
                max(0.0, 1.0 - max(0.0, ddg) / cap), 4
            )

            inst = _instability(cand)
            # P0.3: redock with EVERY configured method, not just the first.
            # Method-level disagreement on score / RMSD / PLIF is itself a
            # candidate-quality signal (Strong = all methods agree on a
            # physically valid pose; Uncertain = methods disagree on contacts).
            poses_by_method = {}
            for method in (dcfg.methods or ["vina"]):
                p = redock_with(
                    method, ctx, cand.candidate_id, mc.structure,
                    ref_atoms, dcfg, ctx.paths.validation / "redock", inst,
                    wt.ligand.smiles,
                )
                poses_by_method[method] = p
            # Primary pose = first configured method (preserves legacy
            # docking_score / redocking_consistency semantics).
            pose = poses_by_method[(dcfg.methods or ["vina"])[0]]

            # Honest skip if real docking refused (CA-only receptor etc.).
            # The candidate proceeds (other layers still inform it) but its
            # docking signal is marked unreliable - the final ranker treats
            # this as "proxy" evidence (see s11 evidence class).
            skipped_methods = [m for m, p in poses_by_method.items() if p.skipped]
            if skipped_methods:
                cand.details["docking_skipped"] = {
                    m: poses_by_method[m].skipped for m in skipped_methods
                }

            # P0/ultra-review: a SKIPPED primary pose (real docker refused -
            # e.g. CA-only receptor) carries score=0.0, rmsd_to_reference=None
            # and ligand_atoms=reference. The old defaults turned that into a
            # PERFECT redock (docking_score 0.0, consistency 1.0,
            # uncertainty 0.0) for a pose that was never actually docked. Mark
            # the docking signal NEUTRAL instead so a never-docked candidate
            # can't earn full redock marks. pose_validity_status stays
            # "unknown" (set below) which already blocks the Strong pose_clean
            # check; ligand_escape stays False (we don't know it escaped).
            if pose.skipped:
                cand.scores["docking_score"] = None
                consistency = 0.5
                cand.scores["redocking_consistency"] = 0.5
                cand.scores["docking_uncertainty"] = 0.5
                cand.details["docking_score_skipped"] = True
                ligand_escape = False
            else:
                cand.scores["docking_score"] = pose.score
                consistency = max(
                    0.0, 1.0 - (pose.rmsd_to_reference or 0.0) / 4.0
                )
                cand.scores["redocking_consistency"] = round(consistency, 4)
                cand.scores["docking_uncertainty"] = round(1.0 - consistency, 4)
                ligand_escape = (pose.rmsd_to_reference or 0.0) > 4.5

            # Pose-quality surface: physical-validity stand-in + PLIF recovery
            # vs the WT reference pose. Stored per pose; primary fields copied
            # onto cand.scores so the reranker / final report can use them.
            valid_count = 0
            plifs = []
            method_scores = []
            for m, p in poses_by_method.items():
                if p.skipped:                    # honest skip, no validity verdict
                    continue
                v = pose_sanity(p.ligand_atoms or ref_atoms, mc.structure)
                p.pose_validity_status = v["status"]
                p.pose_validity_reasons = list(v["reasons"])
                p.receptor_clashes = int(v["receptor_clashes"])
                p.plif_recovery = _plif_recovery(p, ref_atoms, mc.structure)
                plifs.append(p.plif_recovery)
                method_scores.append(p.score)
                if v["valid"]:
                    valid_count += 1
            cand.scores["pose_validity_valid_methods"] = valid_count
            cand.scores["pose_validity_total_methods"] = (
                len(poses_by_method) - len(skipped_methods)
            )
            cand.scores["plif_recovery_mean"] = (
                round(sum(plifs) / len(plifs), 4) if plifs else 0.0
            )
            cand.scores["plif_recovery_min"] = (
                round(min(plifs), 4) if plifs else 0.0
            )
            # Cross-method disagreement (>=2 methods); 0 when only one method.
            if len(method_scores) >= 2:
                mean = sum(method_scores) / len(method_scores)
                spread = (sum((s - mean) ** 2 for s in method_scores)
                          / len(method_scores)) ** 0.5
                cand.scores["docking_method_disagreement"] = round(spread, 4)
            # Primary pose's PLIF on cand.scores (legacy column for s11/score)
            if pose.plif_recovery is not None:
                cand.scores["plif_recovery"] = pose.plif_recovery
            cand.scores["pose_validity_status"] = (
                pose.pose_validity_status or "unknown"
            )
            if pose.pose_validity_reasons:
                cand.details["pose_validity_reasons"] = list(
                    pose.pose_validity_reasons
                )

            mut_cat = catalytic_distances(mc.structure, ref_atoms, catalytic)
            geom_pen = 0.0
            for k, v in mut_cat.items():
                geom_pen += abs(v - wt_cat.get(k, v)) / 4.0
            cand.scores["catalytic_geometry_penalty"] = round(geom_pen, 4)
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
                    docking_score=pose.score,
                    redocking_consistency=consistency,
                )
                cand.scores.update(negp)

            # filters (spec 14.3)
            reasons = []
            if cand.scores["ddg_fold"] > scfg.max_ddg_allowed:
                reasons.append(f"ddG {cand.scores['ddg_fold']:.2f} > "
                               f"{scfg.max_ddg_allowed}")
            if ligand_escape:
                reasons.append("ligand displaced on redocking")
                # ultra-review fix #1: propagate to scores so the s11
                # evidence_class REJECT branch (`scores.get("ligand_escape")`)
                # can actually fire. Previously this was a local only -> the
                # REJECT branch was dead (nothing ever wrote the key).
                cand.scores["ligand_escape"] = True
            if reasons:
                cand.details["nonmd_rejected"] = "; ".join(reasons)
            else:
                kept.append(cand)

        # advance the best survivors to MD
        kept.sort(
            key=lambda c: (
                c.scores.get("ml_score", 0.0)
                + c.scores.get("redocking_consistency", 0.0)
                - 0.2 * max(0.0, c.scores.get("ddg_fold", 0.0))
            ),
            reverse=True,
        )
        top_n = ctx.config.validation.md.top_candidates
        md_top = kept[: top_n]
        # Binding-site reserved-slots floor (mirrors s08_reranker.
        # _promote_binding_site_reservations). Guarantees at least N
        # candidates per known_binding_site position reach real MD even
        # when the s08 ml_score puts them below the top_candidates cut.
        # Without this, cheap-run cannot evaluate the literature D222
        # family with real Boltz+MD because chemistry_rules favourites
        # consistently outscore binding-site mutations at cheap scale.
        n_reserve_md = int(getattr(
            ctx.config.validation.md,
            "binding_site_reserved_per_position", 0,
        ))
        binding_site = set(ctx.get("known_binding_site", []) or [])
        if n_reserve_md > 0 and binding_site and kept:
            md_top, n_promoted = _apply_binding_site_floor(
                ranked=kept, top_n=top_n,
                binding_site=binding_site, n_reserve=n_reserve_md,
            )
            if n_promoted:
                self.log.info(
                    "MD binding-site reserved-slots: promoted %d "
                    "candidate(s) into top-%d (n_reserve=%d per position)",
                    n_promoted, top_n, n_reserve_md,
                )
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
