"""Stage 08b - per-mutant Boltz re-evaluation (expert review #2).

The fast reranker (s08) uses a cheap *proxy* WT-vs-mutant Boltz Δ. Re-running
Boltz for every candidate is infeasible, so here we re-predict the **mutant
complex with Boltz for the top-N reranked candidates only** and replace the
proxy Δ with a real Δ. Each candidate records ``boltz_delta_source`` =
proxy | mock | real so downstream analysis never confuses the two.

Inserted between s08 (fast rerank) and s09 (non-MD validation).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from evoliez.adapters.boltz import predict_complex
from evoliez.context import RunContext
from evoliez.features.delta import boltz_delta_features
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


def _mutant_predict_worker(payload, gpu):
    """Predict ONE mutant complex with Boltz, pinned to ``gpu``. Module-level +
    picklable so it runs in a ProcessPool worker (GIL-free). Each mutant writes
    to its OWN outdir subdir so the result globbing is scoped per mutant. The WT
    MSA + co-modelled extra ligands are passed so the mutant is predicted under
    the SAME conditions as the WT (s04) -- otherwise the Δ confounds the mutation
    effect with an input-condition difference (no MSA / missing cofactor)."""
    (cand_id, mut_seq, ligand, cp_cfg, outdir, backend, seed, dry_run,
     msa_path, extra_ligands) = payload
    mut_outdir = Path(outdir) / cand_id
    mut_outdir.mkdir(parents=True, exist_ok=True)
    mut_cx = predict_complex(
        cand_id, mut_seq, ligand, cp_cfg, mut_outdir,
        backend=backend, dry_run=dry_run, seed=seed, gpu_device=gpu,
        msa_path=msa_path, extra_ligands=extra_ligands,
    )
    return cand_id, mut_cx


def _run_mutant_chunk(gpu, payloads):
    """Round-robin chunk of mutants on ONE pinned GPU (one process per GPU)."""
    return [_mutant_predict_worker(p, gpu) for p in payloads]


class MutantBoltzStage(Stage):
    name = "s08b_mutant_boltz"

    def run(self, ctx: RunContext) -> None:
        rcfg = ctx.config.reranking
        candidates: List[Candidate] = ctx.get("redock_candidates", [])
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
        top = candidates[: rcfg.mutant_boltz_top_n]
        outdir = ctx.paths.complexes / "mutant_boltz"

        # GPU fan-out: re-predicting N mutant complexes is the same independent-
        # per-item GPU work as the s06b ensemble. A serial loop pins all of it to
        # ONE GPU (leaving the others idle); instead use the same GIL-free
        # ProcessPool across the pinned GPUs (each mutant scoped to its own
        # outdir). The Δ computation is cheap CPU work, done serially afterward.
        gpu_list = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
        # SAME inputs as the WT (s04): the family MSA + co-modelled extra ligands,
        # so ΔBoltz reflects the mutation, not an input-condition difference.
        msa_file = ctx.paths.msa / "alignment.fasta"
        msa_path = str(msa_file) if msa_file.exists() else None
        extra_ligands = ctx.get("extra_ligands", []) or None
        payloads = [
            (c.candidate_id, _mutant_sequence(seq, c), ligand, cp_cfg, outdir,
             backend, ctx.config.seed, ctx.dry_run, msa_path, extra_ligands)
            for c in top
        ]
        predicted = {}  # candidate_id -> mutant Complex
        if len(gpu_list) > 1 and not ctx.dry_run:
            import multiprocessing as mp
            import sys
            from concurrent.futures import ProcessPoolExecutor, as_completed
            chunks = [(g, payloads[i::len(gpu_list)]) for i, g in enumerate(gpu_list)]
            mpctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
            self.log.info("mutant Boltz fan-out: %d mutants across %d GPUs %s",
                          len(top), len(gpu_list), gpu_list)
            with ProcessPoolExecutor(max_workers=len(gpu_list), mp_context=mpctx) as ex:
                futs = [ex.submit(_run_mutant_chunk, g, ch) for g, ch in chunks]
                for fut in as_completed(futs):
                    for cand_id, mut_cx in fut.result():
                        predicted[cand_id] = mut_cx
        else:
            gpu = gpu_list[0] if gpu_list else None
            for p in payloads:
                cand_id, mut_cx = _mutant_predict_worker(p, gpu)
                predicted[cand_id] = mut_cx

        n_done = 0
        # Per-mutant predicted complex -> consumed by s10_md so REAL MD runs on
        # the actual mutant structure. In-memory map (Complex isn't JSON-persisted).
        mut_complexes = ctx.get("mutant_complexes", {}) or {}
        for cand in top:
            mut_cx = predicted[cand.candidate_id]
            mut_complexes[cand.candidate_id] = mut_cx
            delta = boltz_delta_features(mut_cx, wt, catalytic_positions=catalytic)
            cand.details["delta"] = delta
            # dry-run + backend=real returns a MOCK contract; label it honestly
            # ('dry-run') instead of 'real' so provenance matches stage meta.
            cand.details["boltz_delta_source"] = (
                "dry-run" if ctx.dry_run else backend.value  # dry-run | mock | real
            )
            feat = cand.details.setdefault("features", {})
            for dk in _DELTA_KEYS:
                v = delta.get(dk, 0.0)
                feat[dk] = v
                cand.scores[dk] = v
            n_done += 1

        ctx.put("mutant_complexes", mut_complexes)
        ctx.put("redock_candidates", candidates)
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
