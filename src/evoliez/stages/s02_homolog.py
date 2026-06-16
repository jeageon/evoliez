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

        self._write_report(ctx, seq, homologs)

    def _write_report(self, ctx, seq, homologs) -> None:
        """Auto-generate the self-contained HTML homolog-analysis report. Every
        axis / table / colour is derived from the data, so it works for any
        target; the footer documents the exact analysis conditions."""
        from datetime import datetime

        from evoliez import __version__
        from evoliez.io.homolog_report import write_homolog_report

        h, m = ctx.config.homologs, ctx.config.msa
        a3m = ctx.paths.msa / "remote.a3m"
        depth = (sum(1 for ln in a3m.open() if ln.startswith(">"))
                 if a3m.exists() else None)
        srcs = list(h.sources) + (["structure"] if h.use_foldseek
                                  and "structure" not in h.sources else [])
        conditions = [
            ("backend", self.backend(ctx).value),
            ("homolog sources", ", ".join(srcs)),
            ("sequence search",
             "ColabFold remote MSA" if m.remote_server
             else f"{h.method}, DB={h.database or 'unset'}"),
            ("structural search",
             f"Foldseek (ProstT5), DB={h.foldseek_database}"
             if "structure" in srcs and h.foldseek_database
             else ("Foldseek (DB unset)" if "structure" in srcs else "off")),
            ("identity band", f"{h.identity_min:.2f} – {h.identity_max:.2f}"),
            ("subfamily clustering", f"k-mer Jaccard @ cluster_identity={h.cluster_identity:.2f}"),
            ("max sequences", f"{h.max_sequences:,}"),
            ("MSA alignment", "ColabFold remote" if m.remote_server else m.method),
            ("ESM2 prior", m.esm_model if m.esm_enabled else "off"),
            ("evoliez version", __version__),
        ]
        out = ctx.paths.reports / "homolog_report.html"
        write_homolog_report(
            out, target_id=ctx.config.input.target_id, target_len=len(seq),
            homologs=homologs, msa_depth=depth, conditions=conditions,
            generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        self.log.info("homolog analysis report: %s", out)
