"""Stage 11 - final multi-objective ranking & focused library (spec section 16)."""

from __future__ import annotations

from typing import Dict, List

from evoliez.context import RunContext
from evoliez.db.schema import MutationCandidate
from evoliez.io.report import write_reports
from evoliez.ranking.score import ScoreBreakdown, compute_final_score
from evoliez.stages.base import Stage
from evoliez.types import Candidate


# P0.5: evidence classes for the final library. Mapping a candidate to one
# of these classes uses ONLY signals we can verify (structure source, MD
# status, docking method agreement, pose validity); the labels are not
# decided by `final_score` magnitude alone - the plan explicitly forbids
# a "proxy-only candidate silently entering the strongest class".
EVIDENCE_STRONG = "Strong"
EVIDENCE_PROMISING = "Promising"
EVIDENCE_UNCERTAIN = "Uncertain"
EVIDENCE_REJECT = "Reject"


def evidence_class(c: Candidate) -> str:
    """Assign one of {Strong, Promising, Uncertain, Reject} based on the
    structural / MD / docking signals stored on the candidate.

    Hard rules first (Reject), then anti-promotion (Uncertain), then
    promotion (Strong if all clean), default Promising.
    """
    scores = c.scores
    details = c.details

    # ---- Reject: hard structural / MD failure (was not filtered earlier)
    if details.get("nonmd_rejected"):
        return EVIDENCE_REJECT
    # ultra-review fix #3: default to "not_run" (NOT "ok"). A candidate that
    # never reached MD has no md_status key; treating that as "ok" let
    # phantom "MD passed" rows reach Strong. The caller (s11.run) now sets
    # md_status="not_run" explicitly for non-MD candidates, but defaulting
    # here too keeps evidence_class honest for any direct caller.
    md_status = scores.get("md_status", "not_run")
    if md_status == "failed":
        return EVIDENCE_REJECT
    # ultra-review fix #2: an "unstable" MD (openmm sets it when the ligand
    # RMSD final > 5 A) means the ligand left the pocket. That is neither
    # "failed" nor "skipped*", so it used to slip through to Promising and
    # get accepted by the ranking gate. The ligand leaving the pocket is a
    # hard correctness failure for a recommendable candidate -> Reject.
    if md_status == "unstable":
        return EVIDENCE_REJECT
    if scores.get("ligand_escape"):
        return EVIDENCE_REJECT
    # All non-skipped docking methods produced invalid poses -> reject.
    n_valid = scores.get("pose_validity_valid_methods")
    n_total = scores.get("pose_validity_total_methods")
    if (n_valid == 0 and n_total and n_total > 0):
        return EVIDENCE_REJECT

    # ---- Uncertain: proxy-only structure, skipped MD, high disagreement,
    # missing real Boltz Δ on a top candidate (set by caller).
    source = details.get("boltz_delta_source", "missing")
    if source not in ("real",):
        # proxy / mock / missing -> never Strong; can be Promising at best.
        # The promotion gate below would otherwise allow Strong on a
        # purely proxy-derived candidate, which the plan explicitly forbids.
        if details.get("requires_real_boltz_for_strong", False):
            return EVIDENCE_UNCERTAIN
    if md_status.startswith("skipped"):
        return EVIDENCE_UNCERTAIN
    disagreement = scores.get("docking_method_disagreement", 0.0)
    if disagreement and disagreement > 1.5:
        return EVIDENCE_UNCERTAIN
    plif_min = scores.get("plif_recovery_min")
    if plif_min is not None and plif_min < 0.3:
        return EVIDENCE_UNCERTAIN

    # ---- Strong: every clean signal lines up + real Boltz Δ.
    # md_passed is TRUE only for a real "ok" MD: "not_run" (phantom MD on a
    # candidate that never reached the MD stage), "skipped*", "unstable" and
    # "failed" all fail this check, so none of them can reach Strong.
    md_passed = (md_status == "ok") and scores.get("md_lite_score", 0.0) >= 0.0
    real_boltz = (source == "real")
    pose_clean = (scores.get("pose_validity_status", "unknown") == "valid")
    plif_ok = (plif_min is None) or (plif_min >= 0.5)
    if md_passed and real_boltz and pose_clean and plif_ok:
        return EVIDENCE_STRONG

    return EVIDENCE_PROMISING


class FinalRankingStage(Stage):
    name = "s11_final"

    def run(self, ctx: RunContext) -> None:
        weights = ctx.config.scoring
        # Distinguish "s09 did not run" from "s09 ran and rejected every
        # candidate". The old `ctx.get(...) or ctx.require(...)` treated
        # the empty-list case as truthy-false, silently re-ranking the
        # FULL unfiltered candidate set as if validation never happened -
        # a Strong-looking final report on a run where nothing actually
        # passed non-MD validation. Honest behaviour: if s09 published
        # the key, trust it (even if empty).
        validated = ctx.get("validated_candidates")
        if validated is None:
            validated = ctx.require("candidates")
        validated_list: List[Candidate] = list(validated)
        if ctx.get("validated_candidates") is not None and not validated_list:
            self.log.warning(
                "s09 published validated_candidates=[] (every candidate "
                "failed non-MD validation); s11 will write an EMPTY "
                "final library rather than silently ranking the full "
                "unfiltered candidate set"
            )
        validated = validated_list
        # md_candidates carry MD scores; merge them back by id
        md_by_id = {c.candidate_id: c for c in ctx.get("md_candidates", [])}
        for c in validated:
            if c.candidate_id in md_by_id:
                # P0.5: also merge md_status / ligand_escape so evidence_class
                # sees the REAL MD verdict for candidates that ran MD.
                c.scores.update(
                    {
                        k: md_by_id[c.candidate_id].scores[k]
                        for k in (
                            "md_lite_score", "md_instability",
                            "md_status", "ligand_escape",
                        )
                        if k in md_by_id[c.candidate_id].scores
                    }
                )
            else:
                # ultra-review fix #3: validated_candidates is the FULL s09
                # survivor list; many of them never reached the MD stage
                # (only md_candidates do). Without a md_status key,
                # evidence_class previously read the "ok" default and marked
                # them md_passed -> a phantom "MD ok" that could reach Strong.
                # Tag them honestly so they land Promising/Uncertain on the
                # OTHER signals, never Strong on an MD that never ran.
                c.scores["md_status"] = "not_run"
            c.scores.setdefault("md_lite_score", 0.0)
            c.scores.setdefault("md_instability", 0.0)

        breakdowns: Dict[str, ScoreBreakdown] = {}
        for c in validated:
            bd = compute_final_score(c, weights)
            c.scores["final_score"] = bd.total
            c.details["score_breakdown"] = {
                "contributions": bd.contributions,
                "penalties": bd.penalties,
            }
            breakdowns[c.candidate_id] = bd

        ranked = sorted(validated, key=lambda c: -c.scores["final_score"])
        n = len(ranked)
        adv = ctx.config.advanced
        if adv.calibration:
            from evoliez.ml.calibration import (
                candidate_uncertainty,
                recommendation,
            )

        # P0a: ranking gate. Production observed `S335T` at rank 1 with
        # evidence_class=Reject + pose_validity_status=invalid - a clearly
        # broken signal. Expert audit (correct): Reject / invalid pose /
        # failed MD candidates MUST NOT occupy top recommendation slots.
        # We compute the "blocked" flag PER CANDIDATE here so the
        # downstream code paths (final ranking, focused library, top-
        # candidate log) all see a consistent split. The gate is HARD
        # (not score-based) so a high ML score can never override a
        # structural / MD failure - those are correctness conditions,
        # not preferences.
        def _blocked_with_reason(c: Candidate) -> tuple:
            ec = c.scores.get("evidence_class") or c.details.get("evidence_class")
            if ec == "Reject":
                return True, "evidence_class=Reject"
            pose = c.scores.get("pose_validity_status") or c.details.get(
                "pose_validity_status"
            )
            if pose == "invalid":
                return True, "pose_validity_status=invalid"
            md_status = c.scores.get("md_status") or c.details.get("md_status")
            if md_status == "failed":
                return True, "md_status=failed"
            # ultra-review fix #2: "unstable" MD (ligand left the pocket) is a
            # hard correctness failure like "failed" - block it from top
            # recommendation slots. evidence_class already maps it to Reject
            # (which also blocks), but gating on md_status directly is
            # defence-in-depth in case the order of signals changes.
            if md_status == "unstable":
                return True, "md_status=unstable"
            return False, ""
        # P0.5: require a real per-mutant Boltz delta to earn Strong evidence
        # for top-ranked candidates. Below the rank threshold the requirement
        # is relaxed (a real Boltz delta would have been wasteful there).
        rank_threshold_for_real = max(1, int(0.25 * n))   # top 25% must be real
        for i, c in enumerate(ranked, 1):
            c.details["rank"] = i
            if adv.calibration:
                u = candidate_uncertainty(c)
                c.scores["uncertainty"] = u
                c.details["recommendation"] = recommendation(
                    c.scores["final_score"], u, i, n
                )
            # mark which candidates MUST have real Boltz Δ to be Strong
            c.details["requires_real_boltz_for_strong"] = (
                i <= rank_threshold_for_real
            )
            # surface structure source onto scores for the report layer
            c.scores["boltz_delta_source"] = c.details.get(
                "boltz_delta_source", "missing"
            )
            cls = evidence_class(c)
            c.details["evidence_class"] = cls
            c.scores["evidence_class"] = cls
        # P0a: apply the gate AFTER evidence_class is assigned so its
        # decision flows into the block flag. Split into accepted (top
        # of recommendation list) and blocked (still reported, but
        # demoted to the bottom with a clear block_reason).
        from collections import Counter
        accepted: List[Candidate] = []
        blocked: List[Candidate] = []
        for c in ranked:
            is_blocked, reason = _blocked_with_reason(c)
            c.details["is_blocked"] = is_blocked
            c.details["block_reason"] = reason
            c.scores["is_blocked"] = int(is_blocked)
            c.scores["block_reason"] = reason
            (blocked if is_blocked else accepted).append(c)

        block_counts = Counter(c.details["block_reason"] for c in blocked)
        self.log.info(
            "ranking gate: %d accepted, %d blocked %s",
            len(accepted), len(blocked),
            dict(block_counts) if blocked else "",
        )
        if not accepted and blocked:
            self.log.warning(
                "ranking gate: ALL %d candidates blocked - check "
                "production for systematic failure (pose/MD/evidence)",
                len(blocked),
            )

        # Re-sort within each group by final_score desc, then concat:
        # accepted occupy rank 1..M, blocked rank M+1..N. So the top of
        # the report is always trustworthy; blocked candidates remain
        # auditable at the bottom with their block_reason.
        accepted.sort(key=lambda c: -c.scores["final_score"])
        blocked.sort(key=lambda c: -c.scores["final_score"])
        ranked = accepted + blocked
        for i, c in enumerate(ranked, 1):
            c.details["rank"] = i

        # P0.5 (final library): exploit/explore split when uncertainty is
        # available (calibration on) - the strongest "exploit" pool is
        # capped to Strong/Promising; the "explore" pool brings in higher-
        # uncertainty candidates so the next DBTL round learns something
        # new. ONLY accepted candidates feed select_focused_library so
        # rejected/invalid never enter the experimental library.
        if adv.calibration:
            from evoliez.ml.active_learning import select_focused_library
            explore_n = min(
                max(4, n // 10),                  # 10% of the library, min 4
                ctx.config.output.final_library_size,
            )
            try:
                explore = set(
                    c.candidate_id for c in select_focused_library(
                        accepted, size=explore_n,   # accepted-only, was: ranked
                    )
                )
            except Exception:
                explore = set()
            for c in ranked:
                c.details["pool"] = (
                    "explore" if c.candidate_id in explore else "exploit"
                )
                c.scores["pool"] = c.details["pool"]

        assert ctx.store is not None
        with ctx.store.session() as s:
            for c in ranked:
                row = s.get(MutationCandidate, c.candidate_id)
                payload = dict(
                    project_id=ctx.project_id,
                    mutations=c.mutation_str,
                    generator=c.generator,
                    ml_score=c.scores.get("ml_score"),
                    stability_score=c.scores.get("ddg_fold"),
                    docking_score=c.scores.get("docking_score"),
                    md_score=c.scores.get("md_lite_score"),
                    final_score=c.scores.get("final_score"),
                    rank=c.details["rank"],
                    details={"scores": c.scores, "meta": c.details.get("features", {})},
                )
                if row is None:
                    s.add(MutationCandidate(candidate_id=c.candidate_id, **payload))
                else:
                    for k, v in payload.items():
                        setattr(row, k, v)

        written = write_reports(ctx.config, ctx.paths, ranked, breakdowns)

        # model-used transparency (expert review #5): make the fallback
        # explicit in the human report so results are never over-trusted.
        md_report = ctx.paths.reports / "final_report.md"
        if md_report.exists():
            gstat = ctx.meta("gnn_status", "disabled")
            ikind = (ctx.meta("interaction_model", {}) or {}).get(
                "model_kind", "n/a"
            )
            with md_report.open("a") as fh:
                fh.write(
                    f"\n## Model provenance\n\n"
                    f"- EvoLigand-GNN: **{gstat}**"
                    f"{' (heuristic family model used instead)' if gstat == 'heuristic_fallback' else ''}\n"
                    f"- family interaction model: `{ikind}`\n"
                    f"- backend: `{ctx.config.backend.value}`\n"
                )

        # Multi-level ML datasets (spec section 7). Boltz-derived columns are
        # tagged feature/weight/weak-label/filter in roles.json; the only
        # supervised-label column is experiment-sourced (ml/labels policy).
        from evoliez.ml.datasets import (
            mutation_rows,
            residue_rows,
            variant_rows,
            write_datasets,
        )

        ds = {
            "pose_level": ctx.get("pose_dataset", []),
            "edge_level": ctx.get("edge_dataset", []),
            "residue_level": residue_rows(
                ctx.require("wt_complex"), ctx.require("position_features")
            ),
            "mutation_level": mutation_rows(ranked),
            "variant_level": variant_rows(ranked),
        }
        ds_written = write_datasets(ctx.paths.ml_datasets, ds)
        ctx.persist_meta(
            "ml_datasets",
            {k: len(v) for k, v in ds.items()},
        )

        # Relative-vector graph dataset for EvoLigand-GNN (server training).
        if ctx.config.gnn.build_dataset:
            from evoliez.adapters.disorder import predict_disorder
            from evoliez.features.confidence import residue_confidence
            from evoliez.ml.graph_dataset import (
                build_graph_sample,
                save_graph_dataset,
            )
            from evoliez.stages.s08_reranker import _approx_mutant_complex

            wt = ctx.require("wt_complex")
            pfeats = ctx.require("position_features")
            econ = ctx.get("ensemble_contacts", [])
            cat = ctx.get("catalytic_positions", [])
            gcfg = ctx.config.gnn
            rconf = residue_confidence(wt.structure)
            dis = (
                predict_disorder(
                    wt.structure.sequence, ctx.paths.root / "datasets",
                    backend=ctx.config.backend_for("s06b_interaction"),
                    dry_run=ctx.dry_run,
                )
                if gcfg.use_disorder
                else None
            )
            li = float(wt.metrics.get("ligand_iptm", 1.0))
            ip = float(wt.metrics.get("complex_ipde", 2.0))

            def _g(cx_):
                return build_graph_sample(
                    cx_, pfeats, econ, radius_lr=gcfg.radius_lr,
                    radius_rr=gcfg.radius_rr, catalytic_positions=cat,
                    residue_confidence=rconf, disorder=dis,
                    ligand_iptm=li, complex_ipde=ip,
                    low_plddt_cutoff=gcfg.low_plddt_cutoff,
                    drop_far_low_plddt=gcfg.drop_far_low_plddt,
                )

            samples = []
            wt_s = _g(wt)
            if wt_s is not None:
                samples.append(wt_s)
            for c in ranked[:30]:
                gs = _g(_approx_mutant_complex(wt, c))
                if gs is not None:
                    samples.append(gs)
            if samples:
                save_graph_dataset(samples, ctx.paths.graph_dataset)
                ctx.persist_meta("graph_dataset_samples", len(samples))

        # active-learning diverse focused library (user §11/§15) - replaces
        # the raw top-N so the experimental plate spans positions /
        # chemistries / ligand-atom targets / subfamilies.
        if adv.active_learning and ranked:
            import csv as _csv

            from evoliez.ml.active_learning import select_focused_library

            lib = select_focused_library(
                ranked, ctx.config.output.final_library_size,
                beta=adv.al_beta, gamma=adv.al_gamma,
            )
            p = ctx.paths.reports / "focused_library.csv"
            with p.open("w", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["well", "candidate_id", "mutations",
                            "final_score", "acquisition_score",
                            "uncertainty", "recommendation"])
                for i, c in enumerate(lib):
                    well = f"{chr(65 + i // 12)}{i % 12 + 1}"
                    w.writerow([
                        well, c.candidate_id, c.mutation_str,
                        c.scores.get("final_score"),
                        c.scores.get("acquisition_score"),
                        c.scores.get("uncertainty"),
                        c.details.get("recommendation", "uncertain candidate"),
                    ])
            ctx.put("focused_library", lib)
            ctx.persist_meta("focused_library_size", len(lib))

        # provenance / reproducibility (user §16)
        if adv.provenance:
            from evoliez.io.provenance import (
                build_provenance,
                write_provenance,
            )

            prov = build_provenance(
                sequence=ctx.require("target_sequence"),
                ligand_smiles=ctx.require("ligand").smiles,
                config_dict=ctx.config.model_dump(mode="json"),
                seed=ctx.config.seed,
                backend=ctx.config.backend.value,
                gnn_ckpt=(str(ctx.paths.root / ctx.config.gnn.checkpoint)
                          if ctx.config.gnn.enabled else None),
            )
            prov["run_fingerprint"] = ctx.run_fingerprint()
            prov["resume_invalidated"] = ctx.invalidated
            prov["gnn_status"] = ctx.meta("gnn_status", "disabled")
            prov["interaction_model_kind"] = (
                ctx.meta("interaction_model", {}) or {}
            ).get("model_kind")
            prov["ligand_atom_ids"] = ctx.meta("ligand_atom_ids", [])
            write_provenance(ctx.paths.reports / "provenance.json", prov)
            for c in ranked:
                c.details["provenance_id"] = prov["run_fingerprint"][
                    "config_sha1"
                ]

        ctx.put("ranked_candidates", ranked)
        ctx.put("accepted_candidates", accepted)
        ctx.put("blocked_candidates", blocked)
        ctx.persist_meta("n_ranked", len(ranked))
        ctx.persist_meta("n_accepted", len(accepted))
        ctx.persist_meta("n_blocked", len(blocked))
        ctx.persist_meta(
            "block_reasons",
            dict(block_counts) if blocked else {},
        )
        # P0a: top_candidate now reflects the top ACCEPTED candidate, not
        # the top scoring one. Previously a Reject/invalid candidate with
        # high ml_score (e.g. mut_00053 S335T in PseFDH production) could
        # land here, misleading every downstream consumer.
        ctx.persist_meta(
            "top_candidate",
            {
                "mutations": accepted[0].mutation_str,
                "final_score": accepted[0].scores["final_score"],
                "evidence_class": accepted[0].details.get("evidence_class"),
            }
            if accepted
            else None,
        )
        self.log.info(
            "final ranking complete: %d total (%d accepted, %d blocked); "
            "reports: %s; ml_datasets: %s",
            len(ranked), len(accepted), len(blocked),
            [str(p) for p in written],
            [p.name for p in ds_written],
        )
        if accepted:
            self.log.info(
                "top ACCEPTED: %s (score %.3f, evidence=%s)",
                accepted[0].mutation_str,
                accepted[0].scores["final_score"],
                accepted[0].details.get("evidence_class"),
            )
        elif blocked:
            self.log.warning(
                "NO accepted candidates after gate - top of report is the "
                "highest-scoring BLOCKED candidate (%s, reason=%s). Check "
                "production for systematic failure (pose/MD/evidence).",
                ranked[0].mutation_str,
                ranked[0].details.get("block_reason"),
            )
