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
class Complex:
    structure: ProteinStructure
    ligand: Ligand
    method: str = "unknown"
    confidence: float = 0.0
    affinity_score: Optional[float] = None
    path: Optional[str] = None


@dataclass
class Pose:
    candidate_id: str
    method: str
    score: float
    ligand_atoms: List[LigandAtom] = field(default_factory=list)
    rmsd_to_reference: Optional[float] = None
    cluster: int = 0


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
