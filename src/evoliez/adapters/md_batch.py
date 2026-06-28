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
MDTask = Tuple[str, object, str, float]


def _md_chunk_worker(payload):
    """Run ONE GPU's chunk of candidates through run_md (module-level + picklable). Pins the
    GPU before the heavy import; returns {candidate_id: MDResult}. A single candidate's failure
    is captured as a failed MDResult so it never kills the chunk."""
    gpu, tasks, mdcfg, ligand_cache_dir, extra_specs, catalytic = payload
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not tasks:
        return {}
    from evoliez.adapters.openmm_engine import MDResult, run_md  # AFTER the pin
    from evoliez.config import Backend

    lcd = Path(ligand_cache_dir) if ligand_cache_dir else None
    out: Dict[str, object] = {}
    for cid, mc, wd, inst in tasks:
        try:
            out[cid] = run_md(
                mc, cid, mdcfg, Path(wd), instability=inst,
                catalytic_positions=catalytic, backend=Backend.real,
                dry_run=False, ligand_cache_dir=lcd, extra_ligands=extra_specs)
        except Exception as exc:  # noqa: BLE001 — one candidate must not kill the chunk
            out[cid] = MDResult(
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
) -> Dict[str, object]:
    """Fan run_md across the GPU pool (one chunk per GPU, round-robin). Returns
    {candidate_id: MDResult}. The caller handles the <=1-GPU / mock serial fallback."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    gpus = [g for g in gpu_list if str(g).strip() != ""] or [None]
    chunks: List[list] = [[] for _ in gpus]
    for i, t in enumerate(tasks):
        chunks[i % len(gpus)].append(t)
    payloads = [(g, ch, mdcfg, ligand_cache_dir, extra_specs, list(catalytic))
                for g, ch in zip(gpus, chunks) if ch]
    out: Dict[str, object] = {}
    with ProcessPoolExecutor(max_workers=max(1, len(payloads))) as ex:
        futs = [ex.submit(_md_chunk_worker, p) for p in payloads]
        for f in as_completed(futs):
            out.update(f.result())
    return out
