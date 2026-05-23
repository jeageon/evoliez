"""Per-ligand-atom interaction-distance descriptor (user-requested core).

For one (homolog/mutant) complex pose: for each ligand atom, the relative
distances to the nearest enzyme residue points within an interaction cutoff,
plus the interaction type. Aggregated into a **fixed-length, ligand-size- and
length-independent fingerprint** so poses from different homologs (different
sequence length, possibly different ligand atom count) are comparable and can
be stacked into one training matrix.

Reuses ``features/geometry`` (dist / classify_interaction / ligand_centroid).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from evoliez.features.geometry import classify_interaction, dist
from evoliez.types import LigandAtom, ProteinStructure

_ITYPES = ["hbond", "salt_bridge", "aromatic", "hydrophobic", "vdw", "none"]


def fingerprint_dim(k_nearest: int, n_bins: int = 8) -> int:
    # distance histogram (n_bins) + per-type counts (6) + summary (4) + k-shell
    return n_bins + len(_ITYPES) + 4 + k_nearest


def fingerprint_feature_labels(
    k_nearest: int, n_bins: int = 8, cutoff: float = 6.0,
) -> List[dict]:
    """Per-feature semantic labels for the complex fingerprint vector.

    Returns a list of length ``fingerprint_dim(...)``; each entry is
    ``{"label": str, "group": str, "index_in_group": int}``. The HTML
    report's fingerprint heatmap consumes this to label x-ticks
    ("0-0.75Å", "hbond", "mean", "k1") instead of the meaningless
    "feature_0", and to draw vertical separators between groups so a
    reviewer can read the four regions at a glance.

    Kept here (not in the figures package) so the labels stay in lockstep
    with the vector layout - any future change to ``complex_fingerprint``
    forces an update on both sides at once.
    """
    out: List[dict] = []
    bin_w = float(cutoff) / max(1, n_bins)
    for i in range(n_bins):
        lo = i * bin_w
        hi = (i + 1) * bin_w
        out.append({
            "label": f"{lo:.2g}-{hi:.2g}Å",
            "group": "dist_hist", "index_in_group": i,
        })
    for j, t in enumerate(_ITYPES):
        out.append({"label": t, "group": "itype", "index_in_group": j})
    for j, name in enumerate(("mean", "std", "min", "contact_frac")):
        out.append({"label": name, "group": "summary", "index_in_group": j})
    for j in range(k_nearest):
        out.append({"label": f"k{j + 1}", "group": "kshell", "index_in_group": j})
    return out


def complex_fingerprint(
    structure: ProteinStructure,
    ligand_atoms: Sequence[LigandAtom],
    *,
    cutoff: float = 6.0,
    k_nearest: int = 6,
    n_bins: int = 8,
) -> np.ndarray:
    """Ligand-atom interaction-distance fingerprint for a single pose."""
    res_pts = [
        (r.sidechain_centroid or r.ca, r.aa) for r in structure.residues
    ]
    if not ligand_atoms or not res_pts:
        return np.zeros(fingerprint_dim(k_nearest, n_bins), dtype=float)

    per_atom_min: List[float] = []
    per_atom_kshell = np.zeros(k_nearest, dtype=float)
    type_counts = {t: 0.0 for t in _ITYPES}
    n_atoms = len(ligand_atoms)
    n_with_contact = 0

    for atom in ligand_atoms:
        dists = []
        best_type = "none"
        best_d = 1e9
        for (pt, aa) in res_pts:
            d = dist(pt, atom.coord)
            if d <= cutoff:
                dists.append(d)
                if d < best_d:
                    best_d = d
                    best_type = classify_interaction(aa, atom, d)
        if dists:
            n_with_contact += 1
            dists.sort()
            per_atom_min.append(dists[0])
            for j in range(k_nearest):
                per_atom_kshell[j] += dists[min(j, len(dists) - 1)]
            type_counts[best_type if best_type in type_counts else "vdw"] += 1.0
        else:
            per_atom_min.append(cutoff)
            per_atom_kshell += cutoff
            type_counts["none"] += 1.0

    pam = np.asarray(per_atom_min, dtype=float)
    hist, _ = np.histogram(pam, bins=n_bins, range=(0.0, cutoff))
    hist = hist.astype(float) / max(1, n_atoms)
    kshell = per_atom_kshell / max(1, n_atoms)
    type_vec = np.array([type_counts[t] / max(1, n_atoms) for t in _ITYPES])
    summary = np.array(
        [
            float(pam.mean()),
            float(pam.std()),
            float(pam.min()),
            n_with_contact / max(1, n_atoms),
        ]
    )
    return np.concatenate([hist, type_vec, summary, kshell]).astype(float)


def describe(fp: np.ndarray, k_nearest: int = 6, n_bins: int = 8) -> dict:
    off = n_bins
    types = fp[off : off + len(_ITYPES)]
    summ = fp[off + len(_ITYPES) : off + len(_ITYPES) + 4]
    return {
        "contact_fraction": round(float(summ[3]), 3),
        "mean_min_dist": round(float(summ[0]), 3),
        "dominant_type": _ITYPES[int(np.argmax(types))] if types.size else "none",
    }
