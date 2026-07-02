"""Stage s04x - V4 ReferenceEnsemble v0.

This optional stage is not inserted into the default s01-s11 pipeline. V4 tests
and server validation call it explicitly after the seed manifest is frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from evoliez.context import RunContext
from evoliez.experimental.seed_manifest import load_seed_manifest
from evoliez.guided.ensemble_builder import build_reference_ensemble_v0
from evoliez.stages.base import Stage


CTX_KEYS = (
    "reference_ensemble",
    "reference_conformers",
    "reference_distributions",
    "reference_confidence_card",
    "nadp_anchor_distributions",
    "formate_reactive_distributions",
    "catalytic_contact_distributions",
    "structure_validity_metrics",
    "ensemble_uncertainty",
)


class ReferenceEnsembleStage(Stage):
    name = "s04x_reference_ensemble"

    def _artifact_path(self, ctx: RunContext) -> Path:
        root = getattr(getattr(ctx, "paths", None), "root", None) or getattr(ctx, "root", ".")
        return Path(root) / "reports" / "provenance" / "v4_reference_ensemble.json"

    def _payload(self, ctx: RunContext) -> Dict[str, Any]:
        manifest = load_seed_manifest()
        ensemble = build_reference_ensemble_v0(manifest)
        conformers = [c.model_dump() for c in ensemble.conformers]
        distributions = {k: v.model_dump() for k, v in ensemble.distributions.items()}
        validity = {
            c.conformer_id: c.validity.model_dump()
            for c in ensemble.conformers
        }
        return {
            "reference_ensemble": ensemble.model_dump(),
            "reference_conformers": conformers,
            "reference_distributions": distributions,
            "reference_confidence_card": {
                "claim_level": ensemble.claim_level,
                "evidence_density": ensemble.uncertainty.evidence_density,
            },
            "nadp_anchor_distributions": {
                "contact": distributions["anchor_contact"],
            },
            "formate_reactive_distributions": {
                "distance_A": distributions["hydride_distance_A"],
                "angle_deg": distributions["hydride_angle_deg"],
            },
            "catalytic_contact_distributions": {
                "anchor_contact": distributions["anchor_contact"],
            },
            "structure_validity_metrics": validity,
            "ensemble_uncertainty": ensemble.uncertainty.model_dump(),
        }

    def _put_payload(self, ctx: RunContext, payload: Dict[str, Any]) -> None:
        for key in CTX_KEYS:
            ctx.put(key, payload[key])

    def run(self, ctx: RunContext) -> None:
        payload = self._payload(ctx)
        path = self._artifact_path(ctx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self._put_payload(ctx, payload)

    def load(self, ctx: RunContext) -> bool:
        path = self._artifact_path(ctx)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if any(key not in payload for key in CTX_KEYS):
            return False
        self._put_payload(ctx, payload)
        return True
