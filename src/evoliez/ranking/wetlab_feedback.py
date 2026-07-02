"""Wet-lab feedback / active-learning recalibration (ROADMAP_V3 D10 / V3-9).

Round 0 is computational evidence only. When wet-lab results arrive, this hook:

  1. keeps experimental labels SEPARATE from computational evidence (the ML data policy
     — Boltz/MD are never supervised labels; only assay endpoints are);
  2. recalibrates the reaction-geometry soft-kernel center/sigma to the geometry of the
     measured ACTIVE variants (so the kernel learns where activity actually lives);
  3. emits the claim-unlock decision (``wetlab_replicated``) ClaimGuard consumes — so
     activity/kcat claims unlock ONLY for replicated endpoints.

Pure / numeric (statistics); light env. It does NOT train a model — it recalibrates the
deterministic geometry kernel + gates the claims; the ML retrain (V3-7) is separate.
"""
from __future__ import annotations

import statistics as _stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class WetLabResult:
    variant_id: str
    active: bool                 # active_above_wt (a DERIVED experimental label)
    replicated: bool = False
    endpoint_value: Optional[float] = None   # e.g. initial rate; experimental only


@dataclass
class Recalibration:
    geometry_center: Optional[float] = None      # median geometry of measured actives
    geometry_sigma: Optional[float] = None        # spread of measured actives
    n_active: int = 0
    n_total: int = 0
    wetlab_replicated: bool = False               # -> ClaimGuard unlock
    notes: List[str] = field(default_factory=list)

    def claim_provenance_update(self) -> dict:
        """The fields to merge into ClaimProvenance once wet-lab data exists."""
        return {"wetlab_replicated": self.wetlab_replicated,
                "known_active_controls": self.n_active > 0}


def recalibrate_from_wetlab(
    results: Sequence[WetLabResult],
    geometry_by_variant: Optional[Dict[str, float]] = None,
    *, require_replicated: bool = True,
) -> Recalibration:
    """Recalibrate the geometry kernel + decide claim unlocking from wet-lab results.

    ``geometry_by_variant`` = the computational reaction-geometry value (e.g. mean
    donor-acceptor distance) per variant, used to learn where the ACTIVE variants sit.
    Activity claims unlock only if at least one active variant is REPLICATED
    (``require_replicated``)."""
    geometry_by_variant = geometry_by_variant or {}
    actives = [r for r in results if r.active]
    replicated_actives = [r for r in actives if r.replicated]

    active_geoms = [geometry_by_variant[r.variant_id]
                    for r in actives if r.variant_id in geometry_by_variant]
    center = _stats.median(active_geoms) if active_geoms else None
    sigma = (_stats.pstdev(active_geoms) if len(active_geoms) >= 2 else None)

    unlocked = bool(replicated_actives) if require_replicated else bool(actives)
    notes: List[str] = []
    if actives and not replicated_actives and require_replicated:
        notes.append("active variants found but none replicated -> activity claims stay locked")
    if active_geoms:
        notes.append(f"geometry kernel recentered on {len(active_geoms)} active variant(s)")
    else:
        notes.append("no geometry values for active variants -> kernel unchanged")

    return Recalibration(
        geometry_center=(round(center, 3) if center is not None else None),
        geometry_sigma=(round(sigma, 3) if sigma is not None else None),
        n_active=len(actives), n_total=len(results),
        wetlab_replicated=unlocked, notes=notes)
