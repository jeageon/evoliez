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
        extra_ligands = ctx.get("extra_ligands", [])
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
            seed=ctx.config.seed,
            extra_ligands=extra_ligands,
        )
        ctx.put("wt_complex", cx)
        ctx.persist_meta("complex_confidence", cx.confidence)
        ctx.persist_meta("complex_affinity", cx.affinity_score)

        assert ctx.store is not None
        with ctx.store.session() as s:
            # idempotent: a --resume re-run must not duplicate WT complex rows
            if (
                s.query(ComplexPrediction)
                .filter_by(project_id=ctx.project_id, method=cx.method)
                .first()
                is None
            ):
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

        # s04 complex-prediction report (interactive 3D viewer + pLDDT / PAE /
        # ipTM / affinity / diffusion-ensemble) — only when a REAL Boltz run
        # left per-model confidence outputs (mock / dry-run produce none).
        try:
            import glob
            from datetime import datetime

            from evoliez.io.complex_report import (compute_complex_stats,
                                                   write_complex_report)
            preds = [p for p in glob.glob(str(
                ctx.paths.complexes / "boltz" / "boltz_results_*"
                / "predictions" / "*"))
                if glob.glob(p + "/confidence_*model_*.json")]
            if preds:
                cp = ctx.config.complex_prediction
                write_complex_report(
                    ctx.paths.reports / "complex_report.html",
                    target_id=ctx.config.input.target_id,
                    stats=compute_complex_stats(preds[0]),
                    ligand_names=[ligand.id] + [e.id for e in extra_ligands],
                    generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    conditions=[
                        ("model", f"{cp.primary_method}, "
                                  f"{cp.diffusion_samples} diffusion samples"),
                        ("ligands", ", ".join(
                            [ligand.id] + [e.id for e in extra_ligands])),
                        ("selected", "model_0 (top confidence_score)"),
                        ("affinity",
                         "Boltz-2 head: log10(IC50/uM) + P(binder)"),
                        ("backend", self.backend(ctx).value),
                    ],
                )
                self.log.info("complex prediction report: %s",
                              ctx.paths.reports / "complex_report.html")
        except Exception as exc:  # report is secondary — never fail the stage
            self.log.warning("complex report failed: %s", exc)
