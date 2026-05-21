"""Stage 08b - per-mutant Boltz re-evaluation (expert review #2).

The fast reranker (s08) uses a cheap *proxy* WT-vs-mutant Boltz Δ. Re-running
Boltz for every candidate is infeasible, so here we re-predict the **mutant
complex with Boltz for the top-N reranked candidates only** and replace the
proxy Δ with a real Δ. Each candidate records ``boltz_delta_source`` =
proxy | mock | real so downstream analysis never confuses the two.

Inserted between s08 (fast rerank) and s09 (non-MD validation).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from evoliez.adapters.boltz import predict_complex
from evoliez.context import RunContext
from evoliez.features.delta import WTDeltaCache, boltz_delta_features
from evoliez.stages.base import Stage
from evoliez.types import Candidate

_DELTA_KEYS = ("d_ligand_iptm", "d_complex_iplddt", "d_complex_ipde",
               "d_key_distance", "d_pocket_plddt")


def _mutant_sequence(wt_seq: str, cand: Candidate) -> str:
    seq = list(wt_seq)
    for m in cand.mutations:
        if 0 < m.position <= len(seq):
            seq[m.position - 1] = m.mut
    return "".join(seq)


class MutantBoltzStage(Stage):
    name = "s08b_mutant_boltz"

    def run(self, ctx: RunContext) -> None:
        rcfg = ctx.config.reranking
        # PIPELINE-ORDER fix: s08b now runs AFTER s09 (see stages/__init__.py
        # ALL_STAGES). Prefer `md_candidates` (set by s09) so real per-mutant
        # Boltz runs on the candidates that ACTUALLY reach MD - eliminates
        # the s08b-top-N vs s10-top-N set mismatch that caused 3/4 honest
        # skips on smoke. Falls back to `redock_candidates` for backwards
        # compatibility (e.g., when MD is disabled and s09 didn't run).
        candidates: List[Candidate] = (
            ctx.get("md_candidates")
            or ctx.get("redock_candidates", [])
            or []
        )
        if not rcfg.mutant_boltz_enabled or not candidates:
            self.log.info("mutant Boltz re-eval disabled; Δ stays proxy")
            return

        wt = ctx.require("wt_complex")
        seq = ctx.require("target_sequence")
        ligand = ctx.require("ligand")
        catalytic = ctx.get("catalytic_positions", [])
        backend = ctx.config.backend_for(self.name)
        cp_cfg = ctx.config.complex_prediction.model_copy(
            update={"diffusion_samples": rcfg.mutant_boltz_diffusion_samples}
        )
        # Reuse the WT MSA built by s03 for every mutant prediction. A point
        # mutation doesn't change the MSA - the homologs were retrieved
        # using WT as the query and are unchanged regardless of our 1-2
        # residue substitution. Skipping --use_msa_server eliminates a
        # ~15-18 min HTTP round-trip per mutant (server-observed: mutant
        # Boltz with --use_msa_server takes ~20 min for 3 samples;
        # pure inference is ~2 min - the rest is the MSA fetch). Standard
        # practice in MSA-conditioned structure prediction (e.g.
        # Tishkov-style mutant analyses).
        msa_path = ctx.paths.msa / "alignment.fasta"
        wt_msa: Optional[Path] = msa_path if msa_path.exists() else None
        if wt_msa is None:
            self.log.warning(
                "no WT MSA at %s; per-mutant Boltz will fall back to "
                "--use_msa_server (slow). Run s03 first.", msa_path,
            )
        # When the input is md_candidates (post-s09), top_for_md already
        # capped the size. mutant_boltz_top_n still caps it further if the
        # user wants Boltz on only a subset of MD candidates (config knob).
        top = candidates[: rcfg.mutant_boltz_top_n]
        outdir = ctx.paths.complexes / "mutant_boltz"

        n_done = 0
        # Per-mutant predicted complex -> consumed by s10_md so REAL MD runs
        # on the actual mutant structure (the sequence guard otherwise
        # skips, since _mutant_complex(WT) is not the mutant). In-memory
        # context map (Complex isn't JSON-persisted); MD-candidates not in
        # this top-N fall back to the WT-derived proxy and honestly skip.
        mut_complexes = ctx.get("mutant_complexes", {}) or {}
        # Reuse the WT-side delta cache built in s08 if it's still in
        # context; otherwise build one here. Identical numerical result
        # to the old per-candidate path because the cache is a pure
        # function of (wt, catalytic_positions, contact_cutoff).
        wt_delta_cache = ctx.get("wt_delta_cache")
        if wt_delta_cache is None:
            wt_delta_cache = WTDeltaCache.build(
                wt, catalytic_positions=catalytic,
            )
        for cand in top:
            mut_cx = predict_complex(
                cand.candidate_id, _mutant_sequence(seq, cand), ligand,
                cp_cfg, outdir, backend=backend, dry_run=ctx.dry_run,
                msa_path=wt_msa,           # reuse WT MSA: skip HTTP fetch
            )
            mut_complexes[cand.candidate_id] = mut_cx
            delta = boltz_delta_features(
                mut_cx, wt, catalytic_positions=catalytic,
                wt_cache=wt_delta_cache,
            )
            cand.details["delta"] = delta
            cand.details["boltz_delta_source"] = backend.value  # mock | real
            feat = cand.details.setdefault("features", {})
            for dk in _DELTA_KEYS:
                v = delta.get(dk, 0.0)
                feat[dk] = v
                cand.scores[dk] = v
            n_done += 1

        ctx.put("mutant_complexes", mut_complexes)
        # Don't overwrite redock_candidates with the (possibly smaller)
        # md_candidates list - keep the full s09 output intact for s11
        # final ranking, while md_candidates is updated below for s10.
        ctx.put("md_candidates", candidates)
        ctx.persist_meta("n_mutant_boltz_evaluated", n_done)
        ctx.persist_meta(
            "mutant_boltz_backend",
            backend.value if not ctx.dry_run else "dry-run",
        )
        self.log.info(
            "real ΔBoltz on top %d/%d candidates (backend=%s); "
            "remainder keep proxy Δ",
            n_done, len(candidates), backend.value,
        )
