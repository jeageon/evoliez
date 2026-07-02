"""Real-artifact preflight for v4.

Preflight is intentionally conservative: invalid real artifacts are reported as
invalid or skipped_not_validated, never converted into neutral pass records.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from pydantic import BaseModel, Field

from .artifacts import ArtifactProvenance
from .atom_map import ReactiveAtomMap


VALID = "valid"
INVALID = "invalid"
USABLE_WITH_WARNING = "usable_with_warning"
SKIPPED_NOT_VALIDATED = "skipped_not_validated"


class PreflightResult(BaseModel):
    artifact: ArtifactProvenance
    status: str
    reasons: List[str] = Field(default_factory=list)


def classify_artifact(
    artifact: ArtifactProvenance,
    atom_map: Optional[ReactiveAtomMap] = None,
    *,
    requires_pose_rmsd: bool = False,
) -> PreflightResult:
    reasons: List[str] = []
    if not artifact.is_full_atom:
        reasons.append("protein structure is not full-atom")
    if artifact.atom_count <= 0:
        reasons.append("atom_count missing")
    if not artifact.has_chain_numbering:
        reasons.append("chain/residue numbering map missing")
    if atom_map is None or not atom_map.validated:
        reasons.append("validated reactive atom map missing")
    if requires_pose_rmsd and not artifact.pose_rmsd_source:
        reasons.append("pose RMSD source missing")

    hard_fail = [r for r in reasons if "not full-atom" in r or "atom_count" in r]
    if hard_fail:
        return PreflightResult(artifact=artifact, status=INVALID, reasons=reasons)
    if reasons:
        if atom_map is None or not atom_map.validated:
            return PreflightResult(
                artifact=artifact, status=SKIPPED_NOT_VALIDATED, reasons=reasons
            )
        return PreflightResult(
            artifact=artifact, status=USABLE_WITH_WARNING, reasons=reasons
        )
    return PreflightResult(artifact=artifact, status=VALID, reasons=[])


def preflight_artifacts(
    artifacts: Iterable[ArtifactProvenance],
    atom_map: Optional[ReactiveAtomMap] = None,
    *,
    requires_pose_rmsd: bool = False,
) -> List[PreflightResult]:
    return [
        classify_artifact(a, atom_map=atom_map, requires_pose_rmsd=requires_pose_rmsd)
        for a in artifacts
    ]
