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

        # P0.2: when `complex_prediction.pocket_constraints: true`, feed Boltz
        # an actual `constraints` section pointing at the catalytic / known-
        # binding-site residues so the prediction is STEERED, not just
        # labelled. The flag was metadata-only before; now it's a real CLI
        # surface (the contract test asserts the YAML contains a pocket
        # block whenever the flag is set + residues are available).
        catalytic = ctx.get("catalytic_positions", []) or []
        binding_site = ctx.get("known_binding_site", []) or []
        pocket_residues = sorted(set(catalytic) | set(binding_site))

        cx = predict_complex(
            "wt",
            seq,
            ligand,
            ctx.config.complex_prediction,
            ctx.paths.complexes / "boltz",
            backend=self.backend(ctx),
            dry_run=ctx.dry_run,
            msa_path=msa_path if msa_path.exists() else None,
            pocket_residues=pocket_residues,
        )
        ctx.put("wt_complex", cx)
        ctx.persist_meta("complex_confidence", cx.confidence)
        ctx.persist_meta("complex_affinity", cx.affinity_score)
        # P0.2: ensemble disagreement reaches the report + the reranker so
        # a noisy/uncertain Boltz prediction can't quietly win on the
        # mean score alone.
        for k in ("ensemble_disagreement", "affinity_ensemble_std",
                  "affinity_ensemble_mean", "confidence_ensemble_std"):
            v = cx.metrics.get(k)
            if v is not None:
                ctx.persist_meta(f"complex_{k}", v)

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
