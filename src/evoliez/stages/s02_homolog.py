"""Stage 02 - homolog retrieval & stratification (spec section 7)."""

from __future__ import annotations

from evoliez.adapters.msa_tools import search_homologs
from evoliez.context import RunContext
from evoliez.db.schema import Sequence
from evoliez.stages.base import Stage


class HomologStage(Stage):
    name = "s02_homolog"

    def run(self, ctx: RunContext) -> None:
        seq = ctx.require("target_sequence")
        homologs = search_homologs(
            seq,
            ctx.config.homologs,
            ctx.paths.homologs,
            backend=self.backend(ctx),
            dry_run=ctx.dry_run,
        )
        # Stratify (spec 7.3): core (40-90% id) vs diverse (20-40%).
        core = [h for h in homologs if 0.40 <= h.identity <= 0.90]
        diverse = [h for h in homologs if h.identity < 0.40]

        fasta = ctx.paths.homologs / "filtered_sequences.fasta"
        with fasta.open("w") as fh:
            for h in homologs:
                fh.write(f">{h.id} id={h.identity:.3f}\n{h.sequence}\n")

        ctx.put("homologs", homologs)
        ctx.persist_meta("n_homologs", len(homologs))
        ctx.persist_meta("n_core_homologs", len(core))
        ctx.persist_meta("n_diverse_homologs", len(diverse))

        assert ctx.store is not None
        with ctx.store.session() as s:
            for h in homologs:
                s.add(
                    Sequence(
                        project_id=ctx.project_id,
                        fasta=f">{h.id}\n{h.sequence}\n",
                        source="homolog",
                        identity_to_target=h.identity,
                        coverage=h.coverage,
                        annotation=h.annotation,
                        cluster_id=h.cluster_id,
                    )
                )
        self.log.info(
            "homologs=%d (core=%d, diverse=%d)",
            len(homologs), len(core), len(diverse),
        )
        if len(homologs) < 10:
            self.log.warning("few homologs - evolutionary signal will be weak")
