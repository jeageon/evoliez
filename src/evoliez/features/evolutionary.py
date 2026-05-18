"""Evolutionary features from an MSA (spec section 8.3).

Pure-numpy: conservation, Shannon entropy, amino-acid frequency, PSSM,
gap frequency, position-specific allowed residues, and the per-residue class
assignment used to build the design mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# BLOSUM62 background frequencies (Robinson & Robinson), order = AA above.
_BG = np.array(
    [
        0.078, 0.019, 0.053, 0.063, 0.039, 0.073, 0.023, 0.053, 0.059, 0.091,
        0.023, 0.043, 0.052, 0.042, 0.051, 0.068, 0.059, 0.066, 0.014, 0.032,
    ]
)


@dataclass
class PositionFeature:
    alignment_position: int
    target_position: Optional[int]
    conservation_score: float
    entropy: float
    gap_frequency: float
    amino_acid_frequencies: Dict[str, float] = field(default_factory=dict)
    pssm_vector: Dict[str, float] = field(default_factory=dict)
    allowed_aa: List[str] = field(default_factory=list)
    residue_class: Optional[int] = None


def target_column_map(msa: Sequence[tuple[str, str]]) -> Dict[int, int]:
    """Map alignment column -> 1-based target residue index (target = row 0)."""
    if not msa:
        return {}
    target_aln = msa[0][1]
    mapping: Dict[int, int] = {}
    res_i = 0
    for col, ch in enumerate(target_aln):
        if ch != "-":
            res_i += 1
            mapping[col] = res_i
    return mapping


def compute_position_features(
    msa: Sequence[tuple[str, str]],
) -> List[PositionFeature]:
    if not msa:
        return []
    seqs = [s for _, s in msa]
    ncol = len(seqs[0])
    nseq = len(seqs)
    colmap = target_column_map(msa)
    feats: List[PositionFeature] = []

    for col in range(ncol):
        counts = np.zeros(len(AA), dtype=float)
        gaps = 0
        for s in seqs:
            ch = s[col] if col < len(s) else "-"
            if ch in AA_INDEX:
                counts[AA_INDEX[ch]] += 1.0
            else:
                gaps += 1
        observed = counts.sum()
        gap_freq = gaps / nseq
        if observed > 0:
            freqs = counts / observed
        else:
            freqs = np.zeros(len(AA))

        nz = freqs[freqs > 0]
        entropy = float(-(nz * np.log2(nz)).sum()) if nz.size else 0.0
        max_entropy = math.log2(len(AA))
        conservation = 1.0 - (entropy / max_entropy if max_entropy else 0.0)

        # log-odds PSSM with pseudocounts vs background
        pseudo = (counts + _BG) / (observed + 1.0)
        pssm = np.log2(np.clip(pseudo, 1e-6, None) / _BG)

        allowed = [AA[i] for i in range(len(AA)) if freqs[i] >= 0.02]

        feats.append(
            PositionFeature(
                alignment_position=col,
                target_position=colmap.get(col),
                conservation_score=round(conservation, 4),
                entropy=round(entropy, 4),
                gap_frequency=round(gap_freq, 4),
                amino_acid_frequencies={
                    AA[i]: round(float(freqs[i]), 4)
                    for i in range(len(AA))
                    if freqs[i] > 0
                },
                pssm_vector={AA[i]: round(float(pssm[i]), 3) for i in range(len(AA))},
                allowed_aa=allowed,
            )
        )
    return feats


def assign_residue_classes(
    feats: List[PositionFeature],
    *,
    catalytic_positions: Set[int],
    fold_critical_threshold: float = 0.9,
    ligand_proximal: Optional[Set[int]] = None,
    second_shell: Optional[Set[int]] = None,
    low_conf_gap: float = 0.5,
) -> None:
    """Spec 8.3 classes (mutated in place onto each feature):

    1 catalytic fixed | 2 fold-critical conserved | 3 ligand-proximal variable
    4 second-shell active-site | 5 distant stability candidate | 6 low-confidence
    """
    ligand_proximal = ligand_proximal or set()
    second_shell = second_shell or set()
    for f in feats:
        tp = f.target_position
        if tp is None:
            f.residue_class = 6
            continue
        if f.gap_frequency >= low_conf_gap:
            f.residue_class = 6
        elif tp in catalytic_positions:
            f.residue_class = 1
        elif f.conservation_score >= fold_critical_threshold:
            f.residue_class = 2
        elif tp in ligand_proximal:
            f.residue_class = 3
        elif tp in second_shell:
            f.residue_class = 4
        else:
            f.residue_class = 5


def permissiveness(feat: PositionFeature, aa: str) -> float:
    """How evolutionarily acceptable substituting `aa` at this position is
    (0..1), from the family frequency tempered by conservation."""
    freq = feat.amino_acid_frequencies.get(aa, 0.0)
    return float(min(1.0, freq * 2.0) * (1.0 - 0.5 * feat.conservation_score) + 0.5 * freq)
