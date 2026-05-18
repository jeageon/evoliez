"""Project directory layout (spec section 17.2).

A single :class:`ProjectPaths` owns every path the pipeline writes to, so
stages never hand-build paths and the tree is created exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    # top-level dirs
    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def homologs(self) -> Path:
        return self.root / "homologs"

    @property
    def msa(self) -> Path:
        return self.root / "msa"

    @property
    def structures(self) -> Path:
        return self.root / "structures"

    @property
    def complexes(self) -> Path:
        return self.root / "complexes"

    @property
    def docking(self) -> Path:
        return self.root / "docking"

    @property
    def interaction_graphs(self) -> Path:
        return self.root / "interaction_graphs"

    @property
    def mutations(self) -> Path:
        return self.root / "mutations"

    @property
    def validation(self) -> Path:
        return self.root / "validation"

    @property
    def md(self) -> Path:
        return self.root / "md"

    @property
    def ml_datasets(self) -> Path:
        return self.root / "ml_datasets"

    @property
    def graph_dataset(self) -> Path:
        return self.root / "datasets" / "graph_pt"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def db_path(self) -> Path:
        return self.root / "evoliez.sqlite"

    @property
    def state_path(self) -> Path:
        """Stage-completion checkpoint file (resume support)."""
        return self.root / "_state.json"

    def md_candidate(self, candidate_id: str) -> Path:
        return self.md / candidate_id

    def create_all(self) -> None:
        for p in (
            self.inputs,
            self.homologs,
            self.msa,
            self.structures,
            self.structures / "target",
            self.structures / "representatives",
            self.complexes,
            self.complexes / "boltz",
            self.complexes / "alternative_models",
            self.docking,
            self.interaction_graphs,
            self.mutations,
            self.validation,
            self.md,
            self.ml_datasets,
            self.graph_dataset,
            self.checkpoints,
            self.reports,
            self.figures,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)
