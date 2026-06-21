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


# --------------------------------------------------------------------------- #
# GPU-batched mutant set (quality-neutral speedup, mirrors s06b's family
# ensemble). The per-mutant path above reloads the Boltz MODEL once per mutant
# (top-N mutants -> N model loads). Batched: one `boltz predict <chunk_dir>` per
# GPU = one model load per GPU, then a parallel per-mutant SCOPED parse. Two
# phases, exactly like s06b's _run_batched_ensemble:
#   1. GPU, batched  — LPT-partition the mutants across GPUs by fold cost; ONE
#                      Boltz process per GPU over that GPU's chunk of mutant YAMLs.
#   2. CPU, parallel — per-stem-SCOPED parse of each mutant's prediction into a
#                      finalized Complex (the GIL-bound part), fanned across procs.
# Real Boltz only — the mock predictor has no batch CLI, so the mock multi-GPU
# path keeps the per-mutant process pool (byte-identical mock outputs + seeds).
# The batch call-path (Boltz subprocess + numpy parse) is torch/xgboost-FREE, so
# fork on Linux is safe (no in-process CUDA, no libomp double-load segfault).
# --------------------------------------------------------------------------- #
def _run_mutant_batch_chunk(payload):
    """Phase 1 on ONE GPU: write every mutant YAML in this chunk into a shared
    chunk IN_DIR, then ONE ``boltz predict <in_dir>`` pinned to the GPU = ONE
    model load for the whole chunk. Reuses the SAME boltz-adapter batch
    primitives s06b's _run_batch_chunk uses. Returns
    (gpu, results_dir, [(cand_id, seq)]) so Phase 2 can scoped-parse each mutant
    from results_dir/predictions/<cand_id>_boltz_input/."""
    from evoliez.adapters.boltz import predict_batch, write_batch_input

    (gpu, muts_meta, ligand, cp_cfg, base_outdir, seed, extra_ligands,
     msa_path) = payload
    # IN_DIR and OUT_DIR MUST differ: Boltz rescans IN_DIR and errors if OUT_DIR
    # is nested inside it. Per-GPU names keep concurrent chunks isolated.
    chunk_in = Path(base_outdir) / f"_batch_in_gpu{gpu}"
    chunk_out = Path(base_outdir) / f"_batch_out_gpu{gpu}"
    any_msa_server = False
    for (cand_id, seq) in muts_meta:
        # Same shared WT MSA for every mutant (s08b passes one msa_path, not a
        # per-sequence map), so the Δ reflects the mutation, not a per-mutant
        # MSA difference. write_batch_input rewrites the a3m into chunk_in.
        any_msa_server |= write_batch_input(
            chunk_in, cand_id, seq, ligand, cp_cfg,
            msa_path=msa_path, extra_ligands=extra_ligands,
        )
    results_dir = predict_batch(
        chunk_in, chunk_out, cp_cfg, seed=seed, gpu_device=gpu,
        any_msa_server=any_msa_server,
    )
    return gpu, str(results_dir), [(cand_id, seq) for (cand_id, seq) in muts_meta]


def _parse_mutant_worker(payload):
    """Phase 2 (CPU, GIL-bound, one process per mutant): scoped-parse ONE mutant
    from its OWN predictions/<stem>/ subdir into a finalized Complex. Each mutant
    is parsed ONLY from its stem subdir (parse_prediction_dir scopes the
    confidence/plddt globs there), so a shared chunk output dir never
    cross-contaminates mutants. Returns (cand_id, Complex | None) — the FULL
    Complex (samples + metrics) exactly as predict_complex would build it, so the
    downstream ΔBoltz + s10 MD see identical data. None when the stem holds no
    prediction (honest fail — the caller does NOT fabricate a Δ for it)."""
    from evoliez.adapters.boltz import parse_prediction_dir

    (cand_id, seq, ligand, cp_cfg, results_dir) = payload
    stem_dir = Path(results_dir) / "predictions" / f"{cand_id}_boltz_input"
    cx = parse_prediction_dir(stem_dir, seq, ligand, cp_cfg.primary_method)
    return cand_id, cx


def _shared_msa_a3m(msa_path, outdir, log=None):
    """Return an MSA path safe to hand to EVERY batch chunk: an a3m that lives
    OUTSIDE the per-GPU chunk IN_DIRs so it is referenced BY PATH, never copied in.

    s03 writes an aligned FASTA MSA. Left as FASTA, ``write_batch_input`` ->
    ``_build_spec`` rewrites a stray ``<label>_msa.a3m`` INTO each chunk IN_DIR
    (boltz.py:235), and ``boltz predict <chunk_dir>`` then globs that .a3m as an
    input and ABORTS the whole chunk ("Unable to parse filetype .a3m"). Converting
    the FASTA to an a3m one level UP (next to the chunk dirs, not in them) makes
    ``_build_spec`` use it as-is (boltz.py:232 — no rewrite into in_dir).
    Already-a3m/csv MSAs (e.g. s06b's local rep MSAs) and ``None`` pass through
    untouched — which is exactly why s06b's batch never hit this."""
    if msa_path is None or Path(msa_path).suffix.lower() in (".a3m", ".csv"):
        return msa_path
    from evoliez.adapters.boltz import _to_a3m
    a3m = _to_a3m(Path(msa_path), Path(outdir) / "_wt_shared_msa.a3m")
    if a3m is None and log is not None:
        log.warning("s08b: WT MSA %s could not be rewritten to a3m; mutant Boltz "
                    "runs without an MSA (Δ may be confounded)", msa_path)
    return a3m


def _run_batched_mutants(payloads, gpu_list, ligand, cp_cfg, outdir, seed,
                         extra_ligands, msa_path, log):
    """GPU-batched mutant prediction: Phase 1 (one batched Boltz process per GPU)
    then Phase 2 (parallel per-mutant scoped parse). Returns
    {cand_id: Complex} — only mutants whose prediction parsed (a failed/skipped
    stem is omitted so the caller can fall back honestly, never fabricating a Δ).
    ``payloads`` reuses the per-mutant payload tuples for their (cand_id, seq)."""
    import multiprocessing as mp
    import sys
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from evoliez.stages.s06b_interaction_model import _lpt_partition

    # FORK on Linux: the batch call-path is torch/xgboost-FREE (Boltz subprocess
    # + numpy parse), so children never touch CUDA in-process and fork avoids the
    # spawn-time libomp double-load segfault. macOS -> spawn (local tests only).
    mpctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")

    # Convert the shared WT MSA to an a3m OUTSIDE the per-GPU chunk IN_DIRs ONCE
    # (see _shared_msa_a3m) and pass THAT a3m to every chunk, so a stray FASTA→a3m
    # rewrite never lands inside a chunk dir for `boltz predict <chunk_dir>` to glob.
    shared_msa = _shared_msa_a3m(msa_path, outdir, log)

    # payload layout: (cand_id, mut_seq, ligand, cp_cfg, outdir, backend, seed,
    # dry_run, msa_path, extra_ligands). LPT-balance mutants across GPUs by fold
    # cost (~O(seq_len^2)) — the WT-length-equal mutant seqs make this ~even, but
    # it matches s06b's balancing contract and the per-chunk Boltz seed draw.
    muts_meta_all = [(p[0], p[1]) for p in payloads]  # (cand_id, seq)
    mut_buckets = _lpt_partition(muts_meta_all, len(gpu_list),
                                 weight=lambda m: len(m[1]) ** 2)
    chunks = [
        (g, mut_buckets[gi], ligand, cp_cfg, str(outdir),
         seed, extra_ligands, shared_msa)
        for gi, g in enumerate(gpu_list)
    ]

    # --- Phase 1: one batched Boltz process per GPU (concurrent) ---
    log.info("batched mutant Boltz Phase 1: %d mutants over %d GPUs %s -> %d "
             "model load(s) total", len(muts_meta_all), len(gpu_list), gpu_list,
             len(gpu_list))
    parse_jobs = []  # (cand_id, seq, results_dir)
    with ProcessPoolExecutor(max_workers=len(gpu_list), mp_context=mpctx) as ex:
        futs = [ex.submit(_run_mutant_batch_chunk, ch) for ch in chunks]
        for fut in as_completed(futs):
            _gpu, results_dir, muts_in_chunk = fut.result()
            for (cand_id, seq) in muts_in_chunk:
                parse_jobs.append((cand_id, seq, results_dir))

    # --- Phase 2: parallel per-mutant scoped parse (GIL-bound) ---
    log.info("batched mutant Boltz Phase 2: parsing %d mutants (per-stem "
             "scoped, parallel)", len(parse_jobs))
    parse_payloads = [
        (cand_id, seq, ligand, cp_cfg, results_dir)
        for (cand_id, seq, results_dir) in parse_jobs
    ]
    # Cap parse workers to the launch thread budget (EVOLIEZ_NUM_THREADS), not
    # cpu_count(): a ~48-proc burst trips the shared-box core watchdog, and the
    # parse is I/O-bound. Mirrors s06b's Phase-2 worker cap.
    _cap = int(os.environ.get("EVOLIEZ_NUM_THREADS") or (os.cpu_count() or 2))
    n_workers = min(max(1, len(parse_payloads)), max(1, _cap))
    predicted = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mpctx) as ex:
        for cand_id, cx in ex.map(_parse_mutant_worker, parse_payloads):
            if cx is not None:
                predicted[cand_id] = cx
    return predicted


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
        # per-item GPU work as the s06b ensemble, so it reuses the SAME batched-
        # Boltz-per-GPU path (one `boltz predict <chunk_dir>` per GPU = one model
        # load per GPU, then a parallel per-mutant scoped parse) instead of a
        # serial loop pinned to ONE GPU. Real Boltz only; the mock multi-GPU path
        # keeps the per-mutant pool. The Δ computation is cheap CPU work, serial.
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
        if len(gpu_list) > 1 and not ctx.dry_run and backend.value == "real":
            # GPU-BATCHED (real Boltz): ONE `boltz predict <dir>` per GPU = one
            # model load per GPU instead of one PER MUTANT (top-N reloads -> #GPUs),
            # then a parallel per-stem-SCOPED parse. Quality-neutral, mirrors the
            # s06b family ensemble. The mock predictor has no batch CLI, so the
            # mock multi-GPU path below keeps the per-mutant pool (byte-identical).
            predicted = _run_batched_mutants(
                payloads, gpu_list, ligand, cp_cfg, outdir, ctx.config.seed,
                extra_ligands, msa_path, self.log)
        elif len(gpu_list) > 1 and not ctx.dry_run:
            import multiprocessing as mp
            import sys
            from concurrent.futures import ProcessPoolExecutor, as_completed
            chunks = [(g, payloads[i::len(gpu_list)]) for i, g in enumerate(gpu_list)]
            mpctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
            self.log.info("mutant Boltz fan-out: %d mutants across %d GPUs %s "
                          "(per-mutant process pool)", len(top), len(gpu_list),
                          gpu_list)
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
        n_failed = 0
        # Per-mutant predicted complex -> consumed by s10_md so REAL MD runs on
        # the actual mutant structure. In-memory map (Complex isn't JSON-persisted).
        mut_complexes = ctx.get("mutant_complexes", {}) or {}
        for cand in top:
            mut_cx = predicted.get(cand.candidate_id)
            if mut_cx is None:
                # The batched path omits any mutant whose Boltz stem failed/was
                # skipped. Honest fallback: leave this candidate's proxy Δ in
                # place (do NOT fabricate a real Δ from a missing prediction).
                n_failed += 1
                self.log.warning(
                    "mutant Boltz prediction missing for %s; keeping proxy Δ",
                    cand.candidate_id)
                continue
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
        if n_failed:
            ctx.persist_meta("n_mutant_boltz_failed", n_failed)
        ctx.persist_meta(
            "mutant_boltz_backend",
            backend.value if not ctx.dry_run else "dry-run",
        )
        self.log.info(
            "real ΔBoltz on top %d/%d candidates (backend=%s)%s; "
            "remainder keep proxy Δ",
            n_done, len(candidates), backend.value,
            f"; {n_failed} prediction(s) failed -> proxy Δ kept" if n_failed
            else "",
        )
