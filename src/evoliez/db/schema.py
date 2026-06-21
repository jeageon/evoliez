"""ORM tables mirroring spec section 17.1.

JSON-ish blobs are stored as TEXT (JSON) to keep the schema portable and the
file a single self-contained SQLite DB inside the run directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "project"
    project_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_name: Mapped[str] = mapped_column(String(256))
    enzyme_family: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    objective: Mapped[str] = mapped_column(String(128))
    ligand_id: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    sequences: Mapped[list["Sequence"]] = relationship(back_populates="project")
    candidates: Mapped[list["MutationCandidate"]] = relationship(
        back_populates="project"
    )


class Sequence(Base):
    __tablename__ = "sequence"
    sequence_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.project_id"))
    fasta: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(64))
    identity_to_target: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coverage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    annotation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    project: Mapped[Project] = relationship(back_populates="sequences")


class MSAPosition(Base):
    __tablename__ = "msa_position"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.project_id"))
    alignment_position: Mapped[int] = mapped_column(Integer)
    target_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    conservation_score: Mapped[float] = mapped_column(Float, default=0.0)
    entropy: Mapped[float] = mapped_column(Float, default=0.0)
    gap_frequency: Mapped[float] = mapped_column(Float, default=0.0)
    amino_acid_frequencies: Mapped[dict] = mapped_column(JSON, default=dict)
    pssm_vector: Mapped[dict] = mapped_column(JSON, default=dict)
    residue_class: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class Structure(Base):
    __tablename__ = "structure"
    structure_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.project_id"))
    sequence_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    method: Mapped[str] = mapped_column(String(64))
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pdb_path: Mapped[str] = mapped_column(Text)
    pocket_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class ComplexPrediction(Base):
    __tablename__ = "complex_prediction"
    complex_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.project_id"))
    structure_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ligand_id: Mapped[str] = mapped_column(String(128))
    method: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    affinity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_path: Mapped[str] = mapped_column(Text)


class DockingPose(Base):
    __tablename__ = "docking_pose"
    pose_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.project_id"))
    candidate_id: Mapped[str] = mapped_column(String(64))
    method: Mapped[str] = mapped_column(String(32))
    # Nullable: a pose can be produced with NO parseable engine score (e.g. a
    # DiffDock rank file with an absent/sentinel confidence). NULL = genuinely
    # unscored, never a fabricated sentinel.
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pose_cluster: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ligand_rmsd_to_reference: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )


class InteractionEdge(Base):
    __tablename__ = "interaction_edge"
    edge_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.project_id"))
    complex_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    target_position: Mapped[int] = mapped_column(Integer)
    ligand_atom_id: Mapped[str] = mapped_column(String(32))
    distance: Mapped[float] = mapped_column(Float)
    interaction_type: Mapped[str] = mapped_column(String(32))
    contact_probability: Mapped[float] = mapped_column(Float, default=0.0)


class MutationCandidate(Base):
    __tablename__ = "mutation_candidate"
    candidate_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.project_id"))
    mutations: Mapped[str] = mapped_column(String(256))  # e.g. "A153K;G88S"
    generator: Mapped[str] = mapped_column(String(64))
    ml_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stability_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    docking_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    md_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)

    project: Mapped[Project] = relationship(back_populates="candidates")


class MDSimulation(Base):
    __tablename__ = "md_simulation"
    md_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.project_id"))
    candidate_id: Mapped[str] = mapped_column(String(64))
    protocol_level: Mapped[int] = mapped_column(Integer)
    solvent_mode: Mapped[str] = mapped_column(String(16))
    simulation_time_ns: Mapped[float] = mapped_column(Float, default=0.0)
    replica_id: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    md_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trajectory_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    analysis_json: Mapped[dict] = mapped_column(JSON, default=dict)


def as_dict(obj: Any) -> dict:
    return {
        c.name: getattr(obj, c.name) for c in obj.__table__.columns  # type: ignore[attr-defined]
    }
