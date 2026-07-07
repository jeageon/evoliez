"""Full-atom artifact provenance for v4 preflight."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


def sha256_file(path: Path) -> Optional[str]:
    if not path or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactProvenance(BaseModel):
    structure_path: str
    ligand_path: Optional[str] = None
    topology_path: Optional[str] = None
    source_stage: str
    backend: str
    hash: str = ""
    atom_count: int = 0
    is_full_atom: bool = False
    chain_map: Dict[str, str] = Field(default_factory=dict)
    residue_numbering_map: Dict[str, str] = Field(default_factory=dict)
    frame_count: Optional[int] = None
    timestep_fs: Optional[float] = None
    simulated_time_ns: Optional[float] = None
    pose_rmsd_source: Optional[str] = None

    @model_validator(mode="after")
    def _derive_hash(self) -> "ArtifactProvenance":
        if not self.hash and self.structure_path:
            digest = sha256_file(Path(self.structure_path))
            if digest:
                self.hash = "sha256:" + digest
        return self

    @property
    def has_chain_numbering(self) -> bool:
        return bool(self.chain_map) and bool(self.residue_numbering_map)

