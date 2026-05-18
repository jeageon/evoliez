"""Stage 03 - MSA & evolutionary features (spec section 8)."""

from __future__ import annotations

import json

from evoliez.adapters.msa_tools import build_msa
from evoliez.adapters.remote_msa import fetch_msa
from evoliez.context import RunContext
from evoliez.db.schema import MSAPosition
from evoliez.features.evolutionary import compute_position_features
from evoliez.stages.base import Stage


class MSAStage(Stage):
    name = "s03_msa"

    def run(self, ctx: RunContext) -> None:
        seq = ctx.require("target_sequence")
        homologs = ctx.require("homologs")
        mcfg = ctx.config.msa

        msa = None
        if mcfg.remote_server and self.backend(ctx).value == "real":
            msa = fetch_msa(seq, ctx.paths.msa)
            if msa:
                self.log.info("using remote MSA (%d sequences)", len(msa))
        if msa is None:
            msa = build_msa(
                ctx.config.input.target_id, seq, homologs, mcfg,
                ctx.paths.msa, backend=self.backend(ctx), dry_run=ctx.dry_run,
            )

        (ctx.paths.msa / "alignment.fasta").write_text(
            "".join(f">{cid}\n{s}\n" for cid, s in msa)
        )

        feats = compute_position_features(msa)

        # subfamily-aware evolutionary prior (user §7)
        if ctx.config.advanced.subfamily_msa:
            from evoliez.features.subfamily import annotate_subfamilies

            cluster_of = {h.id: h.cluster_id for h in homologs}
            cluster_of[ctx.config.input.target_id] = -1
            annotate_subfamilies(msa, feats, cluster_of)
            ctx.persist_meta(
                "mean_specificity_divergence",
                round(sum(f.specificity_divergence for f in feats)
                      / max(1, len(feats)), 4),
            )

        # MSA QC (spec 8.2): effective sequence count.
        neff = len(msa)
        ctx.persist_meta("msa_depth", neff)
        ctx.persist_meta(
            "mean_conservation",
            round(sum(f.conservation_score for f in feats) / max(1, len(feats)), 4),
        )

        cons = {
            f.target_position: {
                "conservation": f.conservation_score,
                "entropy": f.entropy,
                "gap_frequency": f.gap_frequency,
                "allowed_aa": f.allowed_aa,
            }
            for f in feats
            if f.target_position is not None
        }
        (ctx.paths.msa / "conservation.json").write_text(json.dumps(cons, indent=2))

        ctx.put("msa", msa)
        ctx.put("position_features", feats)

        assert ctx.store is not None
        with ctx.store.session() as s:
            if not s.query(MSAPosition).first():
                for f in feats:
                    s.add(
                        MSAPosition(
                            project_id=ctx.project_id,
                            alignment_position=f.alignment_position,
                            target_position=f.target_position,
                            conservation_score=f.conservation_score,
                            entropy=f.entropy,
                            gap_frequency=f.gap_frequency,
                            amino_acid_frequencies=f.amino_acid_frequencies,
                            pssm_vector=f.pssm_vector,
                        )
                    )
        self.log.info(
            "MSA depth=%d, mean conservation=%.3f",
            neff, ctx.meta("mean_conservation"),
        )
