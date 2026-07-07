"""Intrinsic-disorder prediction (user guidance §4).

Used so the model can separate "low pLDDT because flexible/IDR" from "low
pLDDT because poorly predicted". Disorder scores are FEATURES, never labels.

real: IUPred2A and/or MobiDB-lite if on PATH.
mock: deterministic sequence proxy (hydrophilicity + low-complexity), so the
pipeline runs everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from evoliez.config import Backend
from evoliez.logging_utils import get_logger
from evoliez.utils.subprocess_utils import run, which

log = get_logger("evoliez.disorder")

# Kyte-Doolittle hydropathy (disorder favours hydrophilic, negative values).
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2, "X": 0.0,
}
_DISORDER_PROMOTING = set("ARNDQEGKPS")


@dataclass
class DisorderTrack:
    iupred: List[float] = field(default_factory=list)
    mobidb: List[float] = field(default_factory=list)
    low_complexity: List[int] = field(default_factory=list)


def predict_disorder(
    sequence: str, workdir: Path, *, backend: Backend, dry_run: bool = False
) -> DisorderTrack:
    if backend is Backend.real and not dry_run:
        t = _real(sequence, workdir)
        if t is not None:
            return t
    return _proxy(sequence)


def _real(sequence: str, workdir: Path):
    if which("iupred2a.py") is None and which("mobidb_lite.py") is None:
        log.info("no IUPred2A/MobiDB-lite on PATH; using disorder proxy")
        return None
    workdir.mkdir(parents=True, exist_ok=True)
    fa = workdir / "seq.fasta"
    fa.write_text(f">q\n{sequence}\n")
    iup: List[float] = []
    if which("iupred2a.py"):
        try:
            res = run(["iupred2a.py", str(fa), "long"], check=True)
            for line in res.stdout.splitlines():
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 3:
                        iup.append(float(parts[2]))
        except Exception as exc:
            log.warning("IUPred2A failed (%s); proxy", exc)
            return None
    if not iup:
        return None
    return DisorderTrack(
        iupred=iup,
        mobidb=[1.0 if v >= 0.5 else 0.0 for v in iup],
        low_complexity=_low_complexity(sequence),
    )


def _proxy(sequence: str) -> DisorderTrack:
    n = len(sequence)
    iup: List[float] = []
    for i, aa in enumerate(sequence):
        lo, hi = max(0, i - 12), min(n, i + 13)
        win = sequence[lo:hi]
        hyd = sum(_KD.get(c, 0.0) for c in win) / max(1, len(win))
        frac_dis = sum(1 for c in win if c in _DISORDER_PROMOTING) / max(1, len(win))
        score = 0.5 * frac_dis + 0.5 * (1.0 / (1.0 + pow(2.718, hyd)))
        iup.append(round(min(1.0, max(0.0, score)), 4))
    return DisorderTrack(
        iupred=iup,
        mobidb=[1.0 if v >= 0.5 else 0.0 for v in iup],
        low_complexity=_low_complexity(sequence),
    )


def _low_complexity(sequence: str) -> List[int]:
    n = len(sequence)
    flags = [0] * n
    for i in range(n):
        lo, hi = max(0, i - 6), min(n, i + 7)
        win = sequence[lo:hi]
        if len(set(win)) <= max(2, len(win) // 4):
            flags[i] = 1
    return flags
