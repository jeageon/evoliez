"""Stage 09 - non-MD structural validation (spec section 14).

Cheaper filters before MD: stability (FoldX/Rosetta/mock), catalytic-geometry
proxy, and per-mutant redocking consistency vs the WT reference pose.
"""

from __future__ import annotations

import copy
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
            pose = redock_with(
                dcfg.methods[0], ctx, cand.candidate_id, mc.structure,
                ref_atoms, dcfg, ctx.paths.validation / "redock", inst,
                wt.ligand.smiles,
            )
            cand.scores["docking_score"] = pose.score
            consistency, uncertainty, ligand_escape = _redock_metrics(
                pose.rmsd_to_reference
            )
            cand.scores["redocking_consistency"] = consistency
            cand.scores["docking_uncertainty"] = uncertainty

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
