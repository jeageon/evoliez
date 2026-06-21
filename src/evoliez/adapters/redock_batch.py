"""Batched per-candidate redocking schedulers (DiffDock GPU-batch + GNINA queue).

Stage 09's per-mutant redocking-consistency check runs a configured set of
dockers for EVERY surviving candidate. Done naively (the historical
per-candidate loop) the GPU dockers reload their model once per candidate:
~600 DiffDock candidates -> ~600 model loads, ~600 gnina launches.

This module gives s09 the SAME GPU-amortising skeleton s06b's per-rep docking
augmentation already uses, but returning RAW poses keyed by candidate (s06b
classifies inline; s09 needs the poses themselves):

* :func:`run_diffdock_batches` — LPT-partition the candidates across the pinned
  GPUs and run ONE batched ``diffdock.redock_batch`` per GPU (one model load per
  GPU), exactly the s06b ``_run_diffdock_batches`` schedule. Reuses the proven
  ``diffdock.redock_batch`` adapter primitive verbatim.
* :func:`run_gnina_queue` — a per-GPU worker queue (the s06b ``_run_gnina_queue``
  / ``_gnina_gpu_init`` pattern) where each worker pins one GPU and runs
  ``gnina.redock`` (the single REFERENCE-CONSISTENT pose s09 wants — identical to
  the per-candidate ``redock_with('gnina', ...)`` result), so the device a task
  runs on always matches a free worker.

NUMERICAL IDENTITY: each scheduler produces, per candidate, the SAME Pose the
per-candidate path would (same receptor + ligand + per-candidate seed); only the
number of model loads / process launches drops. The mock backend is deliberately
NOT routed here (its batch vs per-target ``base_instability`` differ — see
diffdock.redock_batch), so the caller keeps the per-candidate path for mock /
single-GPU and only batches on the real multi-GPU server.

Shared-server safety: GPU pinning is by ``CUDA_VISIBLE_DEVICES`` (the caller's
``gpu_list``); per-GPU gnina concurrency honours ``EVOLIEZ_GNINA_PER_GPU`` and
gnina's own ``EVOLIEZ_DOCK_CPU`` thread cap, so total cores stay under the box
watchdog.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evoliez.config import Backend, DockingConfig
from evoliez.types import LigandAtom, Pose, ProteinStructure

# One redock unit: (candidate_id, receptor_structure, reference_atoms, smiles,
# context_chains|None). ``context_chains`` is the co-modelled cofactor/substrate
# chains kept as fixed receptor context (gnina honours them; diffdock ignores
# them — same contract as ``s05_docking.redock_with``).
RedockTask = Tuple[str, ProteinStructure, Sequence[LigandAtom], str,
                   Optional[List[str]]]


def _lpt_partition(items, n_buckets, weight):
    """Longest-processing-time greedy partition of ``items`` into ``n_buckets``
    weight-balanced lists. Same balancing s06b uses to keep per-GPU makespan even
    (a stride ``items[i::n]`` ignores per-item cost and lets the GPU handed the
    long jobs straggle). Returns ``n_buckets`` lists (some may be empty)."""
    buckets: List[list] = [[] for _ in range(max(1, n_buckets))]
    loads = [0.0] * len(buckets)
    for it in sorted(items, key=weight, reverse=True):
        j = min(range(len(buckets)), key=lambda k: loads[k])
        buckets[j].append(it)
        loads[j] += float(weight(it))
    return buckets


# --------------------------------------------------------------------------- #
# DiffDock: GPU-batched (one model load per GPU).
# --------------------------------------------------------------------------- #
def _diffdock_chunk_worker(payload):
    """Run ONE GPU's chunk of candidates through ``diffdock.redock_batch`` (one
    model load). Module-level + picklable for a ProcessPool worker; the GPU is
    pinned via ``CUDA_VISIBLE_DEVICES`` before the heavy import so the child
    never initialises CUDA on the wrong device. Returns ``{cid: [Pose, ...]}``."""
    gpu, tasks, root, cfg, backend = payload
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not tasks:
        return {}
    from evoliez.adapters import diffdock

    batch_tasks = [(cid, struct, ref, smiles)
                   for (cid, struct, ref, smiles, _ctx) in tasks]
    out_root = Path(root) / f"dd_gpu{gpu if gpu is not None else 'single'}"
    return diffdock.redock_batch(
        batch_tasks, out_root, cfg, backend=backend, dry_run=False,
        gpu_device=gpu,
    )


def run_diffdock_batches(
    tasks: Sequence[RedockTask],
    gpu_list: Sequence[str],
    cfg: DockingConfig,
    *,
    backend: Backend,
    root: Path,
    log,
) -> Dict[str, List[Pose]]:
    """ALL DiffDock ranks (best first) per candidate, GPU-batched.

    LPT-balances the candidates across ``gpu_list`` by docking cost (~O(receptor
    residues x ligand atoms)) then runs ONE ``diffdock.redock_batch`` per GPU
    concurrently (one model load per GPU) — the same schedule as s06b's
    ``_run_diffdock_batches``. ``context_chains`` is intentionally dropped here:
    DiffDock is protein-only and ignores cofactor context (mirrors
    ``redock_with``). Returns ``{candidate_id: [Pose, ...]}``."""
    devices = list(gpu_list) or [None]
    buckets = _lpt_partition(
        tasks, len(devices),
        weight=lambda t: max(1, len(getattr(t[1], "residues", []) or []))
        * max(1, len(t[2] or [])))
    chunks = [(devices[i], ch, str(root), cfg, backend)
              for i, ch in enumerate(buckets) if ch]
    if not chunks:
        return {}
    if len(chunks) == 1:
        log.info("s09 diffdock batch: %d candidate(s), 1 model load",
                 len(tasks))
        return _diffdock_chunk_worker(chunks[0])

    import multiprocessing as mp
    import sys
    from concurrent.futures import ProcessPoolExecutor, as_completed

    mpctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
    result: Dict[str, List[Pose]] = {}
    log.info("s09 diffdock batch: %d candidates across %d GPU chunk(s) %s "
             "-> %d model load(s)", len(tasks), len(chunks), list(gpu_list),
             len(chunks))
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=mpctx) as ex:
        futs = [ex.submit(_diffdock_chunk_worker, ch) for ch in chunks]
        for fut in as_completed(futs):
            result.update(fut.result())
    return result


# --------------------------------------------------------------------------- #
# GNINA: per-GPU worker queue (one gnina at a time per GPU by default).
# --------------------------------------------------------------------------- #
def _gnina_gpu_init(gpu_queue):
    """ProcessPool worker initialiser: pin THIS worker to one GPU popped from the
    shared queue (so the device a task runs on matches the actually-free worker,
    not a static ti%N pin that could collide two tasks on one GPU). Mirrors
    s06b's ``_gnina_gpu_init``."""
    try:
        g = gpu_queue.get_nowait()
    except Exception:
        g = None
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(g)


def _gnina_task_worker(payload):
    """Run ONE candidate's gnina redock (single reference-consistent pose,
    identical to ``redock_with('gnina', ...)``). The GPU is already pinned by the
    worker initialiser (queue path) or passed explicitly (serial path). Returns
    ``(candidate_id, Pose)``; on any failure the exception propagates to the
    caller's future so it is never silently swallowed."""
    cid, struct, ref, smiles, context_chains, root, cfg, backend, gpu = payload
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from evoliez.adapters import gnina

    workdir = Path(root) / "gnina"
    pose = gnina.redock(
        cid, struct, ref, cfg, workdir, instability=0.05, backend=backend,
        dry_run=False, context_chains=context_chains, smiles=smiles,
    )
    return cid, pose


def run_gnina_queue(
    tasks: Sequence[RedockTask],
    gpu_list: Sequence[str],
    cfg: DockingConfig,
    *,
    backend: Backend,
    root: Path,
    log,
) -> Dict[str, Pose]:
    """One reference-consistent gnina Pose per candidate, GPU-queued.

    Up to ``EVOLIEZ_GNINA_PER_GPU`` (default 1) gnina per GPU; each worker pins
    one GPU via the init-time queue so the device matches a free worker — the
    exact s06b ``_run_gnina_queue`` pattern. Returns ``{candidate_id: Pose}``.
    The per-candidate gnina is identical to ``redock_with('gnina', ...)`` (same
    receptor + reference + context + per-candidate seed), so the poses are
    numerically identical to the per-candidate path; only the launch schedule
    differs."""
    devices = list(gpu_list)
    if len(devices) <= 1:
        gpu = devices[0] if devices else None
        out: Dict[str, Pose] = {}
        for (cid, struct, ref, smiles, ctx_chains) in tasks:
            cid, pose = _gnina_task_worker(
                (cid, struct, ref, smiles, ctx_chains, str(root), cfg,
                 backend, gpu))
            out[cid] = pose
        log.info("s09 gnina queue: %d candidate(s), serial on GPU %s",
                 len(tasks), gpu)
        return out

    import multiprocessing as mp
    import sys
    from concurrent.futures import ProcessPoolExecutor, as_completed

    mpctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
    per_gpu = max(1, int(os.environ.get("EVOLIEZ_GNINA_PER_GPU", "1") or "1"))
    n_workers = len(devices) * per_gpu
    gpu_queue = mpctx.Queue()
    for _ in range(per_gpu):
        for g in devices:
            gpu_queue.put(g)
    out = {}
    log.info("s09 gnina queue: %d candidates, %d worker(s) over %d GPUs %s "
             "(%d/GPU)", len(tasks), n_workers, len(devices), devices, per_gpu)
    payloads = [
        (cid, struct, ref, smiles, ctx_chains, str(root), cfg, backend, None)
        for (cid, struct, ref, smiles, ctx_chains) in tasks
    ]
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mpctx,
                             initializer=_gnina_gpu_init,
                             initargs=(gpu_queue,)) as ex:
        futs = {ex.submit(_gnina_task_worker, p): p[0] for p in payloads}
        for fut in as_completed(futs):
            cid, pose = fut.result()
            out[cid] = pose
    return out
