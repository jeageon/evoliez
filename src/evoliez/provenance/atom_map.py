"""Reactive atom-map provenance for v4 geometry claims."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field, model_validator


class ReactiveAtomMap(BaseModel):
    cofactor_atoms: Dict[str, int] = Field(default_factory=dict)
    substrate_atoms: Dict[str, int] = Field(default_factory=dict)
    catalytic_residue_atoms: Dict[str, str] = Field(default_factory=dict)
    source: str = "smarts+manual_review"
    validated: bool = False

    @model_validator(mode="after")
    def _check_validated_map(self) -> "ReactiveAtomMap":
        if self.validated:
            missing = []
            if not self.cofactor_atoms:
                missing.append("cofactor_atoms")
            if not self.substrate_atoms:
                missing.append("substrate_atoms")
            if not self.catalytic_residue_atoms:
                missing.append("catalytic_residue_atoms")
            if missing:
                raise ValueError("validated ReactiveAtomMap missing " + ", ".join(missing))
        return self

