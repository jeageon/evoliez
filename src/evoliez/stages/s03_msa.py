"""Stage 03 - MSA & evolutionary features (spec section 8)."""

from __future__ import annotations

import json

from evoliez.adapters.msa_tools import build_msa
from evoliez.adapters.remote_msa import cached_fetch_msa
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
            if ctx.dry_run:
                # dry-run must NOT execute anything (incl. network). Preview
                # the remote-MSA call and fall through to the offline path.
                self.log.info(
                    "[dry-run] would POST the query to the remote MSA "
                    "server (ColabFold/MMseqs2 API) - skipped, no network"
                )
            else:
                msa = cached_fetch_msa(seq, ctx.paths.msa)
                if msa:
                    self.log.info(
                        "using remote MSA (%d sequences)", len(msa)
                    )
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

        # ESM2 single-sequence prior (roadmap P1.1): MSA-free per-position
        # substitution variability, attached alongside the MSA conservation.
        if mcfg.esm_enabled:
            from evoliez.adapters.esm import esm_position_priors

            priors = esm_position_priors(
                seq, model=mcfg.esm_model, backend=self.backend(ctx),
                dry_run=ctx.dry_run, workdir=ctx.paths.msa,
            )
            by_pos = {i + 1: v for i, v in enumerate(priors)}   # 1-based target
            for f in feats:
                if f.target_position in by_pos:
                    f.esm_variability = by_pos[f.target_position]
            ctx.persist_meta(
                "mean_esm_variability",
                round(sum(f.esm_variability for f in feats)
                      / max(1, len(feats)), 4),
            )
            self.log.info("ESM2 prior attached to %d positions (model=%s)",
                          len(priors), mcfg.esm_model)

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
                "esm_variability": f.esm_variability,
            }
            for f in feats
            if f.target_position is not None
        }
        (ctx.paths.msa / "conservation.json").write_text(json.dumps(cons, indent=2))

        ctx.put("msa", msa)
        ctx.put("position_features", feats)

        assert ctx.store is not None
        with ctx.store.session() as s:
            # refresh on re-run (delete-then-insert), so an MSA from new
            # sources / a code fix replaces stale rows instead of being skipped
            # (audit P0 #4).
            s.query(MSAPosition).filter_by(project_id=ctx.project_id).delete()
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
