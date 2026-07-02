"""Lead deconvolution helpers for v4 first-round libraries."""

from __future__ import annotations

from itertools import combinations
from typing import List


def split_mutations(mutation: str) -> List[str]:
    return [m.strip() for m in mutation.split(";") if m.strip()]


def deconvolution_set(mutation: str) -> List[str]:
    parts = split_mutations(mutation)
    out: List[str] = []
    for n in (1, 2):
        for combo in combinations(parts, n):
            out.append(";".join(combo))
    return out
