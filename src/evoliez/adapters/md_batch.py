"""GPU-batched s10 MD scheduler (ROADMAP_V2 Phase H1) — mirrors redock_batch.py.

s10 runs an OpenMM MD per candidate; serially that pins one GPU while the others idle. This
fans the EXPENSIVE step (run_md) across the configured GPU pool — one worker process per GPU,
each pinning ``CUDA_VISIBLE_DEVICES`` BEFORE importing openmm_engine (so it never inits CUDA on
the wrong device, the proven redock_batch pattern). The anchored build, the analysis, the pose
gate and the DB writes stay serial in the s10 main process (fast CPU work). Numerically
identical to the serial path per candidate — only the wall-clock drops (~Nx for N GPUs).

The mock backend is NOT routed here (the caller keeps the per-candidate path for mock /
single-GPU); only a real multi-GPU run batches.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# one MD unit: (candidate_id, mutant Complex, workdir, instability)
# optional 5th element: umbrella=(window_A, k_kcal) for the E4a PMF sweep (else unbiased)
MDTask = Tuple[str, object, str, float]


def _md_chunk_worker(payload):
    """Run ONE GPU's chunk of candidates through run_md (module-level + picklable). Pins the
    GPU before the heavy import; returns {candidate_id: MDResult}. A single candidate's failure
    is captured as a failed MDResult so it never kills the chunk."""
    gpu, tasks, mdcfg, ligand_cache_dir, extra_specs, catalytic, fail_loud, metal_requested = payload
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not tasks:
        return {}
    from evoliez.adapters.openmm_engine import MDResult, run_md  # AFTER the pin
    from evoliez.config import Backend

    lcd = Path(ligand_cache_dir) if ligand_cache_dir else None
    out: Dict[str, object] = {}
    for t in tasks:
        cid, mc, wd, inst = t[0], t[1], t[2], t[3]
        umbrella = t[4] if len(t) > 4 else None    # E4a PMF: (window_A, k) biases this window
        try:
            out[wd if umbrella is not None else cid] = run_md(
                mc, cid, mdcfg, Path(wd), instability=inst,
                catalytic_positions=catalytic, backend=Backend.real,
                dry_run=False, ligand_cache_dir=lcd, extra_ligands=extra_specs,
                fail_loud_on_cpu=fail_loud, metal_requested=metal_requested,
                umbrella=umbrella)
        except Exception as exc:  # noqa: BLE001 — one task must not kill the chunk
            out[wd if umbrella is not None else cid] = MDResult(
                candidate_id=cid, status="failed",
                protocol_level=getattr(mdcfg, "protocol_level", 0),
                solvent_mode=getattr(mdcfg, "solvent", "implicit"),
                simulation_time_ns=0.0, integration_failed=True,
                failure_reason=str(exc))
    return out


def run_md_batches(
    tasks: Sequence[MDTask],
    gpu_list: Sequence,
    mdcfg,
    *,
    ligand_cache_dir: Optional[str],
    extra_specs,
    catalytic: Sequence[int],
    fail_loud_on_cpu: bool = False,
    metal_requested: bool = False,
) -> Dict[str, object]:
    """Fan run_md across the GPU pool (one chunk per GPU, round-robin). Returns
    {candidate_id: MDResult}. The caller handles the <=1-GPU / mock serial fallback.

    ``metal_requested`` MUST be threaded here: the WT reference MD runs via a direct run_md()
    call (which passes it), but every candidate runs through this fan-out. Omitting it silently
    dropped Mg from ALL candidate MDs while WT kept it — confounding every WT-vs-mutant NAC
    comparison in a multi-GPU run (E3 diagnostic, 2026-07-05)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    gpus = [g for g in gpu_list if str(g).strip() != ""] or [None]
    chunks: List[list] = [[] for _ in gpus]
    for i, t in enumerate(tasks):
        chunks[i % len(gpus)].append(t)
    payloads = [(g, ch, mdcfg, ligand_cache_dir, extra_specs, list(catalytic),
                 fail_loud_on_cpu, metal_requested)
                for g, ch in zip(gpus, chunks) if ch]
    out: Dict[str, object] = {}
    # SPAWN (not fork): a forked worker inherits the parent's already-imported openmm_engine
    # AND its GPU context. On this box the WT-reference MD runs serially in the s10 parent
    # first, so a forked worker re-uses that (broken on the 12.4 driver) OpenCL/CUDA context
    # and silently falls back to the CPU platform (~200x slower, massive oversubscription).
    # spawn gives each worker a fresh interpreter so the `import ... AFTER the pin` actually
    # re-inits OpenMM on the pinned GPU. (redock_batch shells out to a subprocess so it was
    # immune; md_batch runs OpenMM in-process, so it needs spawn.)
    _ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, len(payloads)), mp_context=_ctx) as ex:
        futs = [ex.submit(_md_chunk_worker, p) for p in payloads]
        for f in as_completed(futs):
            out.update(f.result())
    return out
