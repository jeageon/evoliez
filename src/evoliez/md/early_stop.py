"""Adaptive MD early-stopping (ROADMAP_V3 D10 / V3-9).

Multi-lane sends MORE candidates into MD, so don't burn the full ns on a candidate that
has already failed. If, within an early window (~100-200 ps), the design ligand has
clearly left the pocket OR the catalytic contact network has collapsed irrecoverably,
stop the simulation and pin ``structural_viability``/``uncertainty`` to worst — an
HONEST early-failure verdict (recorded), never a silent skip.

Pure NumPy over the early frames + a small config; unit-testable on synthetic frames.
The s10 loop calls ``evaluate_early_stop`` after the early window and breaks if it says
so. This module makes the DECISION; it does not run MD.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EarlyStopConfig:
    enabled: bool = False
    check_after_ps: float = 200.0          # evaluate once the trajectory passes this
    # ligand escape: COM -> anchor distance beyond this (Å), and still moving away
    escape_distance_A: float = 12.0
    # catalytic network collapse: fraction of key contacts retained below this
    min_contact_fraction: float = 0.2
    min_early_frames: int = 5              # need at least this many frames to judge


@dataclass
class EarlyStopVerdict:
    stop: bool = False
    reason: str = ""
    ligand_anchor_distance: Optional[float] = None
    contact_fraction: Optional[float] = None

    def to_json(self) -> dict:
        return {"early_stopped": self.stop, "reason": self.reason,
                "ligand_anchor_distance": self.ligand_anchor_distance,
                "contact_fraction": self.contact_fraction}


def _com(frame: np.ndarray, idxs: Sequence[int]) -> np.ndarray:
    return np.asarray(frame, float)[list(idxs)].mean(axis=0)


def evaluate_early_stop(
    early_frames: Sequence[np.ndarray], *,
    ligand_atom_idxs: Sequence[int], anchor_atom_idx: int,
    catalytic_contacts: Optional[Sequence[Tuple[int, int, float]]] = None,
    cfg: Optional[EarlyStopConfig] = None,
) -> EarlyStopVerdict:
    """Decide whether to abort. ``catalytic_contacts`` = [(atomA, atomB, cutoff_A), ...]
    the key contacts whose collapse means the active site fell apart.

    Stop if EITHER:
      * the ligand COM is past ``escape_distance_A`` from the anchor in the last early
        frame AND moving away (escape, not a transient excursion); OR
      * the retained fraction of catalytic contacts in the last frame is below
        ``min_contact_fraction`` (network collapse).
    """
    cfg = cfg or EarlyStopConfig()
    frames = [np.asarray(f, float) for f in early_frames]
    if not cfg.enabled or len(frames) < cfg.min_early_frames or not ligand_atom_idxs:
        return EarlyStopVerdict(stop=False, reason="not_evaluated")

    # --- ligand escape -----------------------------------------------------------------
    anchor_last = frames[-1][anchor_atom_idx]
    d_last = float(np.linalg.norm(_com(frames[-1], ligand_atom_idxs) - anchor_last))
    d_prev = float(np.linalg.norm(
        _com(frames[-2], ligand_atom_idxs) - frames[-2][anchor_atom_idx]))
    if d_last > cfg.escape_distance_A and d_last >= d_prev:
        return EarlyStopVerdict(
            stop=True, reason="ligand_escaped_pocket", ligand_anchor_distance=round(d_last, 2))

    # --- catalytic network collapse ----------------------------------------------------
    frac = None
    if catalytic_contacts:
        ok = 0
        for a, b, cutoff in catalytic_contacts:
            if np.linalg.norm(frames[-1][a] - frames[-1][b]) <= cutoff:
                ok += 1
        frac = ok / len(catalytic_contacts)
        if frac < cfg.min_contact_fraction:
            return EarlyStopVerdict(
                stop=True, reason="catalytic_network_collapsed",
                ligand_anchor_distance=round(d_last, 2), contact_fraction=round(frac, 3))

    return EarlyStopVerdict(stop=False, reason="ok", ligand_anchor_distance=round(d_last, 2),
                            contact_fraction=(round(frac, 3) if frac is not None else None))


def early_stop_record(verdict: EarlyStopVerdict) -> dict:
    """The provenance an s10 candidate carries when early-stopped: worst structural
    viability + high uncertainty, with the honest reason (NOT a silent skip)."""
    return {
        "md_lite_status": "early_stopped",
        "early_stop": verdict.to_json(),
        "structural_viability_override": 0.0,
        "uncertainty_override": 1.0,
    }
