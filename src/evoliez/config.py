"""Configuration schema (spec section 19).

Pydantic models describing a full pipeline run. Loaded from YAML; unknown keys
are rejected so typos fail fast rather than silently doing nothing.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class Backend(str, Enum):
    """Adapter backend. ``mock`` runs everywhere; ``real`` shells out to the
    actual tools on the GPU server."""

    mock = "mock"
    real = "real"


class _Base(BaseModel):
    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
class LigandInput(_Base):
    id: str = "ligand_X"
    type: str = Field("smiles", description="smiles | sdf | mol2 | pdb | inchi | ccd")
    value: str = Field(..., description="SMILES string or path to a structure file")


class InputConfig(_Base):
    target_id: str = "target"
    target_fasta: Optional[str] = None
    target_sequence: Optional[str] = None
    ligand: LigandInput

    # Optional structural / biochemical context (spec 4.2).
    target_structure: Optional[str] = None
    catalytic_residues: List[str] = Field(default_factory=list)
    fixed_residues: List[str] = Field(default_factory=list)
    known_binding_site: List[str] = Field(default_factory=list)
    cofactor: Optional[str] = None
    target_ph: float = 7.4
    organism: Optional[str] = None
    ec_number: Optional[str] = None
    experimental_dataset: Optional[str] = None

    @model_validator(mode="after")
    def _need_sequence_source(self) -> "InputConfig":
        if not self.target_fasta and not self.target_sequence:
            raise ValueError("input requires either target_fasta or target_sequence")
        return self


# --------------------------------------------------------------------------- #
# Stage configs
# --------------------------------------------------------------------------- #
class HomologConfig(_Base):
    method: str = "mmseqs2"  # blastp | jackhmmer | mmseqs2
    database: Optional[str] = None
    max_sequences: int = 5000
    identity_min: float = 0.20
    identity_max: float = 0.95
    coverage_min: float = 0.70
    evalue_max: float = 1e-5
    length_ratio_min: float = 0.7
    length_ratio_max: float = 1.3
    cluster_identity: float = 0.90
    use_foldseek: bool = False


class MSAConfig(_Base):
    method: str = "mafft"  # mafft | mmseqs2
    compute_conservation: bool = True
    compute_pssm: bool = True
    compute_covariation: bool = False
    max_gap_frequency: float = 0.5
    remote_server: bool = False  # use a hosted MSA API instead of local DBs


class ComplexPredictionConfig(_Base):
    primary_method: str = "boltz2"  # boltz2 | boltz1
    use_msa_server: bool = True
    representative_homologs: int = 20
    use_templates: bool = True
    pocket_constraints: bool = True
    predict_affinity: bool = True


class DockingConfig(_Base):
    methods: List[str] = Field(default_factory=lambda: ["vina"])  # vina|gnina|diffdock
    poses_per_candidate: int = 10
    pose_rmsd_cluster: float = 2.0


class StabilityConfig(_Base):
    method: str = "foldx"  # foldx | rosetta | ml
    max_ddg_allowed: float = 2.5


class MDConfig(_Base):
    enabled: bool = True
    engine: str = "openmm"
    protocol_level: int = Field(1, ge=0, le=3)  # spec 15.2
    solvent: str = "implicit"  # implicit | explicit
    temperature_K: float = 300.0
    timestep_fs: float = 2.0
    minimize_steps: int = 5000
    equilibration_ps: float = 100.0
    restrained_md_ps: float = 500.0
    production_ns: float = 1.0
    replicas: int = 1
    top_candidates: int = 30


class ValidationConfig(_Base):
    redocking: DockingConfig = Field(default_factory=DockingConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    md: MDConfig = Field(default_factory=MDConfig)


class MutationGenConfig(_Base):
    methods: List[str] = Field(
        default_factory=lambda: ["chemistry_rules", "msa_sampler"]
    )  # ligandmpnn | msa_sampler | chemistry_rules
    max_candidates: int = 2000
    design_radius_angstrom: float = 8.0
    fix_catalytic_residues: bool = True
    fix_highly_conserved_residues: bool = True
    conservation_fix_threshold: float = 0.9
    ligandmpnn_samples: int = 32
    ligandmpnn_temperature: float = 0.1


class RerankConfig(_Base):
    model: str = "xgboost"  # xgboost | mlp | rules
    use_experimental_labels: bool = False
    top_for_redocking: int = 200
    top_for_md: int = 30


class OutputConfig(_Base):
    final_library_size: int = 96
    top_single_mutants: int = 50
    report_format: List[str] = Field(default_factory=lambda: ["markdown", "csv"])


class ScoreWeights(_Base):
    """Final multi-objective weights (spec section 16)."""

    ml_mutation: float = 1.0
    ligand_interaction_gain: float = 1.0
    msa_permissiveness: float = 1.0
    complex_confidence: float = 0.5
    redocking_consistency: float = 0.5
    key_contact_preservation: float = 1.0
    stability: float = 0.75
    md_lite: float = 1.0
    # penalties
    conservation_penalty: float = 1.0
    catalytic_geometry_penalty: float = 1.5
    ddg_penalty: float = 0.75
    clash_penalty: float = 1.0
    docking_uncertainty_penalty: float = 0.5
    md_instability_penalty: float = 1.0


class ProjectConfig(_Base):
    name: str = "evoliez_project"
    objective: str = "binding_enhancement"
    output_dir: str = "runs/evoliez_project"


class Config(_Base):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    input: InputConfig
    homologs: HomologConfig = Field(default_factory=HomologConfig)
    msa: MSAConfig = Field(default_factory=MSAConfig)
    complex_prediction: ComplexPredictionConfig = Field(
        default_factory=ComplexPredictionConfig
    )
    mutation_generation: MutationGenConfig = Field(default_factory=MutationGenConfig)
    reranking: RerankConfig = Field(default_factory=RerankConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    scoring: ScoreWeights = Field(default_factory=ScoreWeights)

    # Global default backend and optional per-stage overrides
    # (keys = stage name, e.g. {"s04_complex": "real"}).
    backend: Backend = Backend.mock
    backends: Dict[str, Backend] = Field(default_factory=dict)

    seed: int = 1234

    def backend_for(self, stage_name: str) -> Backend:
        return self.backends.get(stage_name, self.backend)


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a YAML config, apply optional dotted-key overrides, validate."""
    path = Path(path)
    raw: Dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(raw, dotted, value)
    return Config.model_validate(raw)


def _set_dotted(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
