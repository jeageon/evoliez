"""Stage 11 - final multi-objective ranking & focused library (spec section 16)."""

from __future__ import annotations

from typing import Dict, List

from evoliez.context import RunContext
from evoliez.db.schema import MutationCandidate
from evoliez.io.report import write_reports
from evoliez.ranking.score import ScoreBreakdown, compute_final_score
from evoliez.stages.base import Stage
from evoliez.types import Candidate


def _ranked(candidates: List[Candidate]) -> List[Candidate]:
    """Final ranking order: descending final_score with candidate_id as a
    deterministic tie-breaker, so equal scores (final_score is rounded to 4
    decimals, ties are common) resolve identically regardless of upstream
    ordering. Rank numbers carry into focused-library well assignment."""
    return sorted(
        candidates,
        key=lambda c: (-c.scores["final_score"], c.candidate_id),
    )


class FinalRankingStage(Stage):
    name = "s11_final"

    def run(self, ctx: RunContext) -> None:
        weights = ctx.config.scoring
        validated: List[Candidate] = ctx.get("validated_candidates") or ctx.require(
            "candidates"
        )
        # md_candidates carry MD scores; merge them back by id. The MD subset is
        # a DIFFERENT object list than `validated`, so every MD-derived score a
        # downstream report/CSV needs must be listed here or it is silently lost
        # (this dropped nac_occupancy/nac_delta_vs_wt before). nac_* are NOT
        # setdefault-ed to 0.0: absent means "NAC not run for this candidate",
        # which is distinct from a real zero occupancy.
        md_by_id = {c.candidate_id: c for c in ctx.get("md_candidates", [])}
        for c in validated:
            if c.candidate_id in md_by_id:
                c.scores.update(
                    {
                        k: md_by_id[c.candidate_id].scores[k]
                        for k in ("md_lite_score", "md_instability",
                                  "nac_occupancy", "nac_delta_vs_wt")
                        if k in md_by_id[c.candidate_id].scores
                    }
                )
                if "nac" in md_by_id[c.candidate_id].details:
                    c.details["nac"] = md_by_id[c.candidate_id].details["nac"]
                # carry the anchored gate-stack verdict + pose gate into the final
                # candidate so the report/CSV + evidence library see the verdict (Phase E).
                for _k in ("gate_stack", "pose_gate", "alternative_pose_boltz"):
                    if _k in md_by_id[c.candidate_id].details:
                        c.details[_k] = md_by_id[c.candidate_id].details[_k]
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

        ranked = _ranked(validated)
        n = len(ranked)
        adv = ctx.config.advanced
        if adv.calibration:
            from evoliez.ml.calibration import (
                candidate_uncertainty,
                recommendation,
            )
        for i, c in enumerate(ranked, 1):
            c.details["rank"] = i
            if adv.calibration:
                u = candidate_uncertainty(c)
                c.scores["uncertainty"] = u
                c.details["recommendation"] = recommendation(
                    c.scores["final_score"], u, i, n
                )

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

        # Evidence-class library (ROADMAP_V2 Phase E): alongside the scalar ranking, group the
        # MD candidates by gate-stack verdict + a Pareto front + the anchored paper-grade gate,
        # so the final claim is an evidence class (not just a sorted position that silently
        # mixes binding/stability/catalysis). Reads the s10 provenance; non-fatal.
        try:
            import json as _json

            from evoliez.ranking.evidence import build_evidence_library
            _mcj = ctx.paths.reports / "provenance" / "md_candidates.json"
            if _mcj.exists():
                _recs = _json.loads(_mcj.read_text()).get("candidates", [])
                _lib = build_evidence_library(_recs)
                (ctx.paths.reports / "provenance" / "evidence_classes.json").write_text(
                    _json.dumps(_lib.to_json(), indent=2))
                ctx.persist_meta("evidence_counts", _lib.counts)
                ctx.persist_meta("n_paper_grade", len(_lib.paper_grade))
                self.log.info(
                    "evidence-class library: %s; paper-grade=%d; pareto=%d",
                    _lib.counts, len(_lib.paper_grade), len(_lib.pareto))
                # Phase F: re-evaluate ML on FUNCTIONAL-STATE preservation (not just MD-pass)
                # + count functional winners the ML cut would have dropped (control lane).
                from evoliez.ranking.ml_eval import ml_functional_eval
                _mle = ml_functional_eval(
                    _recs,
                    ml_score_by_id={c.candidate_id: c.scores.get("ml_score")
                                    for c in ranked},
                    lane_by_id={c.candidate_id: c.details.get("selection_lane")
                                for c in ranked})
                ctx.persist_meta("ml_functional_eval", _mle)
                self.log.info(
                    "ML functional eval: AUC(ml->functional)=%s vs AUC(ml->MD-pass)=%s;"
                    " low-ml-control functional winners=%s",
                    _mle.get("auc_ml_to_functional"), _mle.get("auc_ml_to_mdpass"),
                    _mle.get("functional_in_control"))
                # v2 evidence-class paper report (Phase E deliverable) — regenerates
                # from the provenance on disk; non-fatal.
                from evoliez.io.paper_report_v2 import write_paper_report_v2
                _rp = write_paper_report_v2(ctx.paths.root)
                self.log.info("v2 evidence-class report: %s", _rp.name)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("evidence-class library skipped (%s)", exc)

        # ROADMAP_V3 B8 — EvidenceCard (D4) as a CANONICAL s11 output. Build a per-candidate
        # EvidenceCard (score/confidence split per axis) for every ranked candidate and write
        # evidence_cards.json + the claim-clean triage_v3.{json,md}, so the card and its
        # triage recommendation are first-class run artifacts instead of an offline CLI only.
        try:
            import json as _json

            from evoliez.ranking.triage_report import write_triage_artifacts
            _prov = ctx.paths.reports / "provenance"
            if (_prov / "md_candidates.json").exists() or (
                    _prov / "validated_candidates.json").exists():
                write_triage_artifacts(str(_prov))          # triage_v3.{json,md} in reports/
                _tri = _json.loads((ctx.paths.reports / "triage_v3.json").read_text())
                _cards = _tri.get("evidence_cards", [])
                (_prov / "evidence_cards.json").write_text(
                    _json.dumps(_cards, indent=2, default=str))
                ctx.persist_meta("n_evidence_cards", len(_cards))
                self.log.info("evidence cards: %d written (evidence_cards.json + triage_v3)",
                              len(_cards))
        except Exception as exc:  # noqa: BLE001
            self.log.warning("evidence-card triage skipped (%s)", exc)

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
                beta=adv.al_beta,
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
        ctx.persist_meta("n_ranked", len(ranked))

        # Funnel-provenance report (paper methods): attrition through the
        # generate→rerank→Boltz→validate→MD funnel, per-generator survival, and a
        # per-candidate evidence-class audit. Reads the durable per-stage meta
        # counts + s07's generated-provenance table; generic for any target.
        from evoliez.io.provenance_report import (
            attach_generated_counts,
            build_funnel_report,
            write_funnel_report,
        )

        funnel = build_funnel_report(
            ranked=ranked,
            meta=ctx.meta,
            md_candidate_ids=[c.candidate_id for c in ctx.get("md_candidates", [])],
            top_n=ctx.config.output.final_library_size,
        )
        prov_dir = ctx.paths.reports / "provenance"
        attach_generated_counts(funnel, prov_dir / "generated_candidates.json")
        funnel_written = write_funnel_report(prov_dir, funnel)
        ctx.persist_meta("funnel_evidence_class_counts",
                         funnel["summary"]["evidence_class_counts"])
        self.log.info("funnel provenance: %s",
                      [p.name for p in funnel_written])

        ctx.persist_meta(
            "top_candidate",
            {
                "mutations": ranked[0].mutation_str,
                "final_score": ranked[0].scores["final_score"],
            }
            if ranked
            else None,
        )
        self.log.info(
            "final ranking complete: %d candidates; reports: %s; "
            "ml_datasets: %s",
            len(ranked), [str(p) for p in written],
            [p.name for p in ds_written],
        )
        if ranked:
            self.log.info(
                "top: %s (score %.3f)",
                ranked[0].mutation_str, ranked[0].scores["final_score"],
            )
