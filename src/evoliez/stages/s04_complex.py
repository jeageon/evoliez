"""Stage 04 - protein-ligand complex prediction (spec section 9)."""

from __future__ import annotations

from evoliez.adapters.boltz import predict_complex
from evoliez.context import RunContext
from evoliez.db.schema import ComplexPrediction, Structure
from evoliez.stages.base import Stage


class ComplexPredictionStage(Stage):
    name = "s04_complex"

    def run(self, ctx: RunContext) -> None:
        seq = ctx.require("target_sequence")
        ligand = ctx.require("ligand")
        msa_path = ctx.paths.msa / "alignment.fasta"

        cx = predict_complex(
            "wt",
            seq,
            ligand,
            ctx.config.complex_prediction,
            ctx.paths.complexes / "boltz",
            backend=self.backend(ctx),
            dry_run=ctx.dry_run,
            msa_path=msa_path if msa_path.exists() else None,
        )
        ctx.put("wt_complex", cx)
        ctx.persist_meta("complex_confidence", cx.confidence)
        ctx.persist_meta("complex_affinity", cx.affinity_score)

        assert ctx.store is not None
        with ctx.store.session() as s:
            st = Structure(
                project_id=ctx.project_id,
                method=cx.method,
                confidence_score=cx.confidence,
                pdb_path=cx.path or "",
                pocket_confidence=cx.confidence,
            )
            s.add(st)
            s.flush()
            s.add(
                ComplexPrediction(
                    project_id=ctx.project_id,
                    structure_id=st.structure_id,
                    ligand_id=ligand.id,
                    method=cx.method,
                    confidence=cx.confidence,
                    affinity_score=cx.affinity_score,
                    complex_path=cx.path or "",
                )
            )
        self.log.info(
            "WT complex: method=%s confidence=%.3f affinity=%s",
            cx.method, cx.confidence, cx.affinity_score,
        )
