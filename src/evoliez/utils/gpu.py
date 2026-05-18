"""Shared-server GPU selection.

The target server has 4x RTX A6000 used by multiple people. We never assume a
free GPU: query ``nvidia-smi`` and pick the device with the most free VRAM and
lowest utilisation, then pin the process to it via ``CUDA_VISIBLE_DEVICES``.
On a laptop (no ``nvidia-smi``) every function degrades to a no-op.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from evoliez.logging_utils import get_logger
from evoliez.utils.subprocess_utils import run, which

log = get_logger("evoliez.gpu")


@dataclass
class GpuInfo:
    index: int
    name: str
    mem_total_mib: int
    mem_free_mib: int
    util_pct: int


def query_gpus() -> List[GpuInfo]:
    if which("nvidia-smi") is None:
        return []
    try:
        res = run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
        )
    except Exception as exc:  # pragma: no cover - server only
        log.warning("nvidia-smi query failed: %s", exc)
        return []
    gpus: List[GpuInfo] = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, total, free, util = parts
        gpus.append(
            GpuInfo(int(idx), name, int(total), int(free), int(util))
        )
    return gpus


def select_gpu(min_free_mib: int = 16000) -> Optional[int]:
    """Return the index of the least-busy GPU with enough free VRAM, or None."""
    gpus = query_gpus()
    if not gpus:
        return None
    candidates = [g for g in gpus if g.mem_free_mib >= min_free_mib]
    if not candidates:
        log.warning(
            "no GPU with >= %d MiB free; falling back to the most free one",
            min_free_mib,
        )
        candidates = gpus
    best = sorted(candidates, key=lambda g: (g.util_pct, -g.mem_free_mib))[0]
    log.info(
        "selected GPU %d (%s): %d MiB free, %d%% util",
        best.index,
        best.name,
        best.mem_free_mib,
        best.util_pct,
    )
    return best.index


def apply_gpu_selection(min_free_mib: int = 16000) -> Optional[int]:
    """Pin this process to a chosen GPU. Respects a pre-set
    ``CUDA_VISIBLE_DEVICES`` (e.g. when launched under a scheduler)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        log.info(
            "CUDA_VISIBLE_DEVICES already set to %s; not overriding",
            os.environ["CUDA_VISIBLE_DEVICES"],
        )
        return None
    idx = select_gpu(min_free_mib=min_free_mib)
    if idx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    return idx
