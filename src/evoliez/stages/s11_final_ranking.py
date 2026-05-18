"""Stage 11 - final multi-objective ranking & focused library (spec section 16)."""

from __future__ import annotations

from typing import Dict, List

from evoliez.context import RunContext
from evoliez.db.schema import MutationCandidate
from evoliez.io.report import write_reports
from evoliez.ranking.score import ScoreBreakdown, compute_final_score
from evoliez.stages.base import Stage
from evoliez.types import Candidate


class FinalRankingStage(Stage):
    name = "s11_final"

    def run(self, ctx: RunContext) -> None:
        weights = ctx.config.scoring
        validated: List[Candidate] = ctx.get("validated_candidates") or ctx.require(
            "candidates"
        )
        # md_candidates carry MD scores; merge them back by id
        md_by_id = {c.candidate_id: c for c in ctx.get("md_candidates", [])}
        for c in validated:
            if c.candidate_id in md_by_id:
                c.scores.update(
                    {
                        k: md_by_id[c.candidate_id].scores[k]
                        for k in ("md_lite_score", "md_instability")
                        if k in md_by_id[c.candidate_id].scores
                    }
                )
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
        for i, c in enumerate(ranked, 1):
            c.details["rank"] = i

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
        ds_written = write_datasets(ctx.paths.root / "ml_datasets", ds)
        ctx.persist_meta(
            "ml_datasets",
            {k: len(v) for k, v in ds.items()},
        )

        ctx.put("ranked_candidates", ranked)
        ctx.persist_meta("n_ranked", len(ranked))
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
