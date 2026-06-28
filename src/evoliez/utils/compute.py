"""Apply the ComputeConfig (ROADMAP_V2 §2b / Phase H1): GPU-first, CPU-bounded.

A single CPU-core budget drives every thread pool (BLAS/OMP) and every process-pool worker
cap, so no stage trips the shared-server watchdog (~48 cores for 10 min). The GPU pool is the
device set the GPU-bound stages (s04/s08b/s09/s06b/s10) fan across.
"""
from __future__ import annotations

import os
from typing import List, Optional

_THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def apply_cpu_budget(budget: int, *, per_proc: Optional[int] = None) -> int:
    """Set the BLAS/OMP thread-pool env vars from the core budget. Idempotent; never RAISES an
    already-lower OMP_NUM_THREADS (the launcher may pin a smaller value). Returns the per-process
    thread count chosen. `per_proc` overrides the default (min(4, budget) — 4 is watchdog-safe)."""
    budget = max(1, int(budget))
    n = per_proc if per_proc is not None else min(4, budget)
    cur = os.environ.get("OMP_NUM_THREADS")
    if cur and cur.isdigit():
        n = min(n, int(cur))
    n = max(1, int(n))
    for v in _THREAD_VARS:
        os.environ.setdefault(v, str(n))
    return n


def gpu_pool_list(pool: Optional[List[int]] = None) -> List[int]:
    """Resolve the GPU pool: the explicit config list if set, else parse
    CUDA_VISIBLE_DEVICES, else [] (CPU / single-device). Never assumes 'all 4 GPUs'."""
    if pool:
        return [int(g) for g in pool]
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [int(g) for g in env.split(",") if g.strip().lstrip("-").isdigit()]


def max_workers(budget: int, per_task_cores: int = 1) -> int:
    """Process-pool worker cap from the CPU budget so pools never oversubscribe the box."""
    return max(1, int(budget) // max(1, int(per_task_cores)))


def gpu_for_index(idx: int, pool: Optional[List[int]] = None) -> Optional[int]:
    """Round-robin a work item index onto the GPU pool (the per-candidate s10 fan-out). None if
    the pool is empty (caller keeps the inherited single-device / CPU behavior)."""
    gpus = gpu_pool_list(pool)
    if not gpus:
        return None
    return gpus[idx % len(gpus)]
