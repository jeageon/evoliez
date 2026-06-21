"""Lightweight data containers shared across stages.

Deliberately framework-free (no torch / rdkit) so they exist on the laptop and
the server alike. Adapters fill them from real tool output or mock generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


@dataclass
class LigandAtom:
    id: str
    element: str
    coord: Vec3 = (0.0, 0.0, 0.0)
    formal_charge: int = 0
    partial_charge: float = 0.0
    hybridization: str = "sp3"
    aromatic: bool = False
    in_ring: bool = False
    is_donor: bool = False
    is_acceptor: bool = False
    is_hydrophobic: bool = False
    pharmacophore: str = "none"


@dataclass
class Ligand:
    id: str
    smiles: str
    atoms: List[LigandAtom] = field(default_factory=list)
    formal_charge: int = 0
    n_rotatable_bonds: int = 0
    source: str = "input"

    @property
    def n_heavy(self) -> int:
        return sum(1 for a in self.atoms if a.element != "H")


@dataclass
class Residue:
    index: int  # 1-based target numbering
    aa: str  # one-letter
    ca: Vec3 = (0.0, 0.0, 0.0)
    sidechain_centroid: Vec3 = (0.0, 0.0, 0.0)
    secondary_structure: str = "C"
    sasa: float = 0.0
    plddt: float = 0.0


@dataclass
class ProteinStructure:
    sequence: str
    residues: List[Residue] = field(default_factory=list)
    method: str = "unknown"
    confidence: float = 0.0
    pdb_path: Optional[str] = None


@dataclass
class BoltzSample:
    """One diffusion sample from Boltz. Structure backbone is shared at the
    Complex level; the ligand pose and confidence vary per sample."""

    idx: int
    ligand_atoms: List["LigandAtom"] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    residue_plddt: List[float] = field(default_factory=list)


@dataclass
class Complex:
    structure: ProteinStructure
    ligand: Ligand
    method: str = "unknown"
    confidence: float = 0.0
    affinity_score: Optional[float] = None
    path: Optional[str] = None
    # Rich Boltz confidence/affinity metrics (spec: features, NOT labels):
    # confidence_score, ptm, iptm, ligand_iptm, complex_plddt, complex_iplddt,
    # complex_pde, complex_ipde, affinity_pred_value, affinity_probability_binary,
    # affinity_pred_value1/2, ensemble_disagreement.
    metrics: Dict[str, float] = field(default_factory=dict)
    samples: List[BoltzSample] = field(default_factory=list)


@dataclass
class Pose:
    candidate_id: str
    method: str
    # ``score`` is the engine-native pose score (see ``score_type``). It is
    # ``None`` when the engine produced a pose but NO parseable score — e.g. a
    # DiffDock rank file with an absent/sentinel confidence token. ``None`` means
    # "genuinely unscored": downstream must skip it from score statistics rather
    # than inject a fabricated number that would pollute min/range (the old
    # ``-1000.0`` / ``0.0`` sentinels did exactly that).
    score: Optional[float]
    ligand_atoms: List[LigandAtom] = field(default_factory=list)
    rmsd_to_reference: Optional[float] = None
    cluster: int = 0
    # --- score provenance (paper-grade audit; defaults keep old construction) ---
    # ``rank`` is 1-based within the engine's own output ordering (gnina SDF mode
    # order / diffdock rankN). ``score_type`` names what ``score`` IS, e.g.
    # "minimizedAffinity" (gnina, lower=better) | "diffdock_confidence" (higher=
    # better). ``cnn_score``/``cnn_affinity`` are gnina's CNN tags (None for
    # diffdock/mock). ``engine_version`` / ``command_args`` record exactly how the
    # pose was produced so a reviewer can reconstruct the run. ``note`` carries a
    # short human-readable provenance remark (e.g. why ``score`` is None).
    rank: int = 1
    score_type: str = ""
    cnn_score: Optional[float] = None
    cnn_affinity: Optional[float] = None
    engine_version: str = ""
    command_args: str = ""
    note: str = ""


@dataclass
class Mutation:
    """A single point substitution, e.g. wt='A', position=153, mut='K'."""

    wt: str
    position: int
    mut: str

    def __str__(self) -> str:
        return f"{self.wt}{self.position}{self.mut}"

    @classmethod
    def parse(cls, token: str) -> "Mutation":
        token = token.strip()
        return cls(wt=token[0], position=int(token[1:-1]), mut=token[-1])


@dataclass
class Candidate:
    candidate_id: str
    mutations: List[Mutation]
    generator: str
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def mutation_str(self) -> str:
        return ";".join(str(m) for m in self.mutations)
