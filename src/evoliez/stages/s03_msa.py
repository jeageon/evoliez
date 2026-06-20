"""Stage 03 - MSA & evolutionary features (spec section 8)."""

from __future__ import annotations

import json

from evoliez.adapters.msa_tools import build_msa, integrate_aligned_homologs
from evoliez.adapters.remote_msa import cached_fetch_msa
from evoliez.context import RunContext
from evoliez.db.schema import MSAPosition
from evoliez.features.evolutionary import PositionFeature, compute_position_features
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

        # Integrate the STRUCTURE track (Foldseek) and any independent LOCAL
        # sequence track (mmseqs) into the SAME target-column MSA, using each
        # hit's native (structural / sequence) alignment instead of re-aligning
        # (user §7). No-op unless those tracks produced target-anchored rows
        # (real run on a ColabFold a3m base); dedupes the ColabFold-mined rows.
        msa, n_integrated = integrate_aligned_homologs(msa, homologs, len(seq))
        if n_integrated:
            self.log.info(
                "integrated %d structure/local homolog rows into the MSA "
                "(target-anchored, native alignment)", n_integrated,
            )
        ctx.persist_meta("msa_integrated_homologs", n_integrated)

        (ctx.paths.msa / "alignment.fasta").write_text(
            "".join(f">{cid}\n{s}\n" for cid, s in msa)
        )

        # Refresh the s02 homolog report so its MSA-depth headline reflects the
        # FINAL integrated MSA (s02 only knew the ColabFold a3m base at its time;
        # the integrated 5-track depth is only known here, after integration).
        try:
            from evoliez.stages.s02_homolog import write_homolog_analysis_report

            write_homolog_analysis_report(
                ctx, self.backend(ctx).value, seq, homologs, msa_depth=len(msa)
            )
        except Exception as exc:  # report is secondary — never fail the MSA stage
            self.log.warning("homolog report refresh failed: %s", exc)

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
                # subfamily signal (set only when advanced.subfamily_msa) — persist
                # it so load() can restore it; downstream s07/s08 scoring depends
                # on it, so a resume that dropped it would change final scores.
                "specificity_divergence": f.specificity_divergence,
            }
            for f in feats
            if f.target_position is not None
        }
        (ctx.paths.msa / "conservation.json").write_text(json.dumps(cons, indent=2))

        # s03 MSA-analysis report (AF2/ColabFold-style coverage map, conservation
        # + information-content tracks, identity-by-track distribution, sequence
        # logo, methods) — the structural analogue of the s02 homolog report, for
        # the FINAL integrated MSA. Secondary: never fail the stage on a hiccup.
        try:
            from datetime import datetime

            from evoliez import __version__
            from evoliez.io.msa_report import (compute_msa_stats, effective_neff,
                                               write_msa_report)
            if len(msa) >= 2:
                aln_seqs = [a for _, a in msa]
                mstats = compute_msa_stats([cid for cid, _ in msa], aln_seqs,
                                           {str(k): v for k, v in cons.items()})
                mstats["neff80"] = effective_neff(aln_seqs)
                h = ctx.config.homologs
                write_msa_report(
                    ctx.paths.reports / "msa_report.html",
                    target_id=ctx.config.input.target_id, target_len=len(seq),
                    stats=mstats,
                    generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    conditions=[
                        ("retrieval tracks (independent, merged)",
                         ", ".join(h.sources)),
                        ("identity band",
                         f"{h.identity_min:.2f} – {h.identity_max:.2f}"),
                        ("subfamily clustering",
                         f"k-mer Jaccard @ cluster_identity={h.cluster_identity:.2f}"),
                        ("max sequences", f"{h.max_sequences:,}"),
                        ("conservation metric",
                         "per-column Shannon entropy (gaps excluded)"),
                        ("Neff (effective)",
                         "reweighted at 80% identity (AlphaFold2 convention)"),
                        ("alignment",
                         "target-anchored (native per-track alignment, de-duplicated)"),
                        ("evoliez version", __version__),
                    ],
                )
                self.log.info("MSA analysis report: %s",
                              ctx.paths.reports / "msa_report.html")
        except Exception as exc:  # report is secondary — never fail the MSA stage
            self.log.warning("MSA report failed: %s", exc)

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

    def load(self, ctx: RunContext) -> bool:
        """Resume without re-running the (expensive) MSA stage.

        s03 reloads ESM2-650M, re-aligns (MAFFT), recomputes per-position
        features and re-renders two HTML reports on every call — pure waste when
        the inputs are byte-identical. The two artifacts run() puts on the bus and
        that downstream stages consume are ``msa`` and ``position_features``;
        rebuild both from disk:

          - ``msa``                from the persisted ``alignment.fasta``
          - ``position_features``  from the ``MSAPosition`` DB rows, with the
            non-columnar fields (``allowed_aa``, ``esm_variability``,
            ``specificity_divergence``) restored from ``conservation.json``

        Return False (force a real re-run) if any required artifact is missing,
        so the pipeline never hands downstream stages partial / empty features.

        ``residue_class`` is intentionally left as persisted (None unless s06 has
        run): s06 recomputes it in place from the live structure/pocket geometry,
        so it never needs to be reconstructed here.
        """
        aln_path = ctx.paths.msa / "alignment.fasta"
        cons_path = ctx.paths.msa / "conservation.json"
        if not aln_path.exists() or not cons_path.exists():
            return False
        if ctx.store is None:
            return False

        # 1) rebuild the MSA from the persisted FASTA (id, aligned-seq) pairs
        msa = _read_fasta_pairs(aln_path)
        if not msa:
            return False

        # 2) per-position conservation extras keyed by 1-based target position
        try:
            cons = json.loads(cons_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        extras = {int(k): v for k, v in cons.items()}
        # A conservation.json written before specificity_divergence was persisted
        # lacks that field; loading it would silently zero the subfamily-
        # specificity signal that s07/s08 scoring uses. Force a real s03 re-run
        # for those (pre-fix) checkpoints rather than degrade the result.
        if extras and not any("specificity_divergence" in v for v in extras.values()):
            return False

        # 3) rebuild position_features from the DB rows (the columnar features),
        #    layering in allowed_aa / esm_variability from conservation.json.
        with ctx.store.session() as s:
            rows = (
                s.query(MSAPosition)
                .filter_by(project_id=ctx.project_id)
                .order_by(MSAPosition.alignment_position)
                .all()
            )
            if not rows:
                return False  # no persisted features -> real re-run
            feats = []
            for r in rows:
                ex = extras.get(r.target_position, {}) if r.target_position else {}
                feats.append(
                    PositionFeature(
                        alignment_position=r.alignment_position,
                        target_position=r.target_position,
                        conservation_score=r.conservation_score,
                        entropy=r.entropy,
                        gap_frequency=r.gap_frequency,
                        amino_acid_frequencies=dict(r.amino_acid_frequencies or {}),
                        pssm_vector=dict(r.pssm_vector or {}),
                        allowed_aa=list(ex.get("allowed_aa", [])),
                        residue_class=r.residue_class,
                        esm_variability=float(ex.get("esm_variability", 0.0)),
                        specificity_divergence=float(
                            ex.get("specificity_divergence", 0.0)
                        ),
                    )
                )

        ctx.put("msa", msa)
        ctx.put("position_features", feats)
        self.log.info(
            "restored MSA (depth=%d) + %d position features from disk",
            len(msa), len(feats),
        )
        return True


def _read_fasta_pairs(path) -> list[tuple[str, str]]:
    """Parse a simple (no-wrap) FASTA into [(id, sequence), ...]."""
    pairs: list[tuple[str, str]] = []
    cid: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cid is not None:
                pairs.append((cid, "".join(chunks)))
            cid = line[1:].strip()
            chunks = []
        elif cid is not None:
            chunks.append(line.strip())
    if cid is not None:
        pairs.append((cid, "".join(chunks)))
    return pairs
