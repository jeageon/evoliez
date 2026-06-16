"""Stage 02 - homolog retrieval & stratification (spec section 7)."""

from __future__ import annotations

from collections import Counter

from evoliez.adapters.msa_tools import gather_homologs
from evoliez.context import RunContext
from evoliez.db.schema import Sequence
from evoliez.stages.base import Stage


class HomologStage(Stage):
    name = "s02_homolog"

    def run(self, ctx: RunContext) -> None:
        seq = ctx.require("target_sequence")
        # Integrated multi-source homology (sequence + structure), merged.
        homologs = gather_homologs(
            seq,
            ctx.config.homologs,
            ctx.paths.homologs,
            backend=self.backend(ctx),
            dry_run=ctx.dry_run,
            remote_server=ctx.config.msa.remote_server,
            msa_dir=ctx.paths.msa,          # share the ColabFold a3m with s03
        )
        by_source = dict(Counter(h.source for h in homologs))
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
        ctx.persist_meta("homolog_sources", by_source)

        assert ctx.store is not None
        with ctx.store.session() as s:
            # REFRESH (delete-then-insert, like s05): s02 has no load() so it
            # re-runs on --resume; a plain skip-if-exists left STALE rows when a
            # re-run produced different homologs (new sources / a code fix), so
            # the DB evidence disagreed with the in-memory set. Delete prior
            # homolog rows first — no duplicates, always current (audit P0 #4).
            s.query(Sequence).filter_by(
                project_id=ctx.project_id, source="homolog"
            ).delete()
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
            "homologs=%d (core=%d, diverse=%d) sources=%s",
            len(homologs), len(core), len(diverse), by_source,
        )
        if len(homologs) < 10:
            self.log.warning("few homologs - evolutionary signal will be weak")
