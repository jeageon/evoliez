"""pLDDT-derived residue confidence features (user guidance).

pLDDT is a per-residue *coordinate confidence*, not a flexibility label. Low
pLDDT = "risky to use as a single fixed coordinate", which may be a flexible
loop OR just poorly predicted. These features feed the model as confidence /
uncertainty signals (never as supervised labels - see ml/labels.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import List, Sequence

from evoliez.types import ProteinStructure


def norm_plddt(v: float) -> float:
    """pLDDT may be 0-100 (AlphaFold/Boltz residue) or 0-1 (Boltz complex)."""
    return max(0.0, min(1.0, v / 100.0 if v > 1.5 else v))


def plddt_bin(v: float) -> int:
    """0 very-low/disorder-like, 1 low, 2 reliable backbone, 3 high."""
    p = v if v > 1.5 else v * 100.0
    if p >= 90:
        return 3
    if p >= 70:
        return 2
    if p >= 50:
        return 1
    return 0


@dataclass
class ResidueConfidence:
    index: int
    plddt_norm: float
    plddt_bin: int
    mean_w3: float
    mean_w7: float
    min_w7: float
    gradient: float
    low_seg_len: int  # length of the contiguous low-pLDDT run this residue is in
    frac_low_nearby: float


def _window(vals: List[float], i: int, w: int):
    lo, hi = max(0, i - w), min(len(vals), i + w + 1)
    return vals[lo:hi]


def residue_confidence(structure: ProteinStructure) -> List[ResidueConfidence]:
    res = structure.residues
    if not res:
        return []
    p = [norm_plddt(r.plddt) for r in res]
    low = [pi < 0.5 for pi in p]
    # contiguous low-pLDDT segment length each residue belongs to
    seg = [0] * len(p)
    i = 0
    while i < len(p):
        if low[i]:
            j = i
            while j < len(p) and low[j]:
                j += 1
            for k in range(i, j):
                seg[k] = j - i
            i = j
        else:
            i += 1

    out: List[ResidueConfidence] = []
    for idx, r in enumerate(res):
        w7 = _window(p, idx, 3)
        grad = (p[min(len(p) - 1, idx + 1)] - p[max(0, idx - 1)])
        near = _window(low, idx, 3)
        out.append(
            ResidueConfidence(
                index=r.index,
                plddt_norm=round(p[idx], 4),
                plddt_bin=plddt_bin(r.plddt),
                mean_w3=round(mean(_window(p, idx, 1)), 4),
                mean_w7=round(mean(w7), 4),
                min_w7=round(min(w7), 4),
                gradient=round(grad, 4),
                low_seg_len=seg[idx],
                frac_low_nearby=round(sum(near) / max(1, len(near)), 4),
            )
        )
    return out


def pocket_mean_plddt(
    confs: Sequence[ResidueConfidence], pocket_indices: Sequence[int]
) -> float:
    s = set(pocket_indices)
    vals = [c.plddt_norm for c in confs if c.index in s]
    return round(mean(vals), 4) if vals else 0.0
