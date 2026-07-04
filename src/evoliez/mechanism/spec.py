"""MechanismSpec + ReactionState (ROADMAP_V3 D1).

The mechanism *envelope* over the generic geometry layer (``md.geometry_spec``). A
``GeometryTerm`` alone is FDH-agnostic, but the same "donor-acceptor distance" means
different things depending on redox/protonation/conformational state — so a mechanism
template binds geometry terms to a required ``ReactionState`` and a tolerance
calibration tier, and selects defaults by ``reaction.class``.

``reaction.class`` -> a registered TEMPLATE (default geometry terms + which
reaction-state fields are required). A per-target YAML overrides specifics. Required
fields that a template needs but the config omits ABORT at load (mirrors the v2
reference hard-gate discipline) — a wrong/missing reaction state silently shifts the
geometry score otherwise.

Pydantic v2 (matches config.py); ``extra='forbid'`` so typos fail fast.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .vocab import (
    G2, GEOMETRY_TIER_CLAIM_CEILING, GEOMETRY_TIERS,
)


class _Base(BaseModel):
    model_config = {"extra": "forbid"}


# --- geometry term (mirrors md.geometry_spec.GeometryTerm + protein-atom selectors) --
class GeometryTermSpec(_Base):
    """One functional-geometry primitive, mechanism-scoped. Atoms are selected either
    by SMARTS against the ligand molecules (``*_smarts``) OR by protein
    residue+atom-name (``*_residue``/``*_atom``) — the v3 protein-atom resolver
    (geometry_spec) handles both. A distance term uses a,b; an angle term a,b,c."""
    kind: str = Field(..., description="distance | angle")
    label: str = "term"
    # ligand selectors
    a_smarts: str = ""
    a_idx: int = 0
    b_smarts: str = ""
    b_idx: int = 0
    c_smarts: str = ""
    c_idx: int = 0
    # protein selectors (resname/seqid + atom name), used when the SMARTS is empty
    a_residue: Optional[str] = None
    a_atom: Optional[str] = None
    b_residue: Optional[str] = None
    b_atom: Optional[str] = None
    c_residue: Optional[str] = None
    c_atom: Optional[str] = None
    distance_min: float = 0.0
    distance_max: float = 3.5
    angle_min: float = 0.0
    angle_max: float = 180.0
    weight: float = 1.0

    @model_validator(mode="after")
    def _check_kind(self) -> "GeometryTermSpec":
        if self.kind not in ("distance", "angle"):
            raise ValueError(f"geometry term kind must be 'distance'|'angle', got {self.kind!r}")
        return self


# --- reaction state (the genuinely new D1 block) -------------------------------------
class ReactionState(_Base):
    """The biochemical state that makes a geometry meaningful. Template-required fields
    are enforced by MechanismSpec; an absent required field aborts at load."""
    pH: Optional[float] = None
    cofactor_redox_state: Optional[str] = None     # e.g. NADP+ / NADPH (REDOX cofactors)
    cofactor_state: Optional[str] = None           # e.g. ATP / ATP_Mg (NON-redox cofactors, e.g.
    #                                                adenylation): kept DISTINCT from
    #                                                cofactor_redox_state — the latter is the
    #                                                oxidation state of a redox cofactor; this is the
    #                                                identity/binding state of a non-redox cofactor.
    substrate_state: Optional[str] = None
    substrate_is_real: bool = True                 # real substrate vs analog/proxy
    protonation_model: Optional[str] = None
    metal_state: Optional[str] = None              # identity + coordination number
    tautomer_state: str = "default"
    covalent_intermediate: bool = False
    conformational_state: Optional[str] = None     # e.g. closed_ternary_complex
    required_waters: List[str] = Field(default_factory=list)


class HostContext(_Base):
    """OPTIONAL in-vivo / metabolic-pathway context (E. coli etc.). When enabled it
    adds a structural-viability sub-axis — NEVER a catalytic claim (ROADMAP_V3 D1)."""
    enabled: bool = False
    physiological_pH_profile: List[float] = Field(default_factory=list)  # range/profile
    metabolite_inhibition: List[str] = Field(default_factory=list)
    expression_toxicity_risk: str = "unknown"


class CatalyticResidueSpec(_Base):
    residue: str                       # resname+seqid, e.g. ARG290
    role: str
    required: bool = False


class GeometryKernel(_Base):
    type: str = "gaussian"
    center_source: str = "reference_ensemble_median"
    sigma_source: str = "reference_ensemble_iqr"


class GeometryCalibration(_Base):
    """Where the soft-kernel tolerances come from (ROADMAP_V3 D1) — reviewer-defensible.
    ``source_tier`` (G1..G5) CAPS the claim strength of any geometry evidence."""
    source_tier: str = G2
    distance_kernel: GeometryKernel = Field(default_factory=GeometryKernel)
    angle_kernel: GeometryKernel = Field(
        default_factory=lambda: GeometryKernel(type="von_mises_or_gaussian",
                                               sigma_source="mechanism_template_default"))
    known_active_controls_used: bool = False
    known_inactive_controls_used: bool = False
    claim_limit: str = "screening_only"

    @model_validator(mode="after")
    def _check_tier(self) -> "GeometryCalibration":
        if self.source_tier not in GEOMETRY_TIERS:
            raise ValueError(f"geometry source_tier must be one of {GEOMETRY_TIERS}, got {self.source_tier!r}")
        return self

    @property
    def claim_ceiling(self) -> str:
        return GEOMETRY_TIER_CLAIM_CEILING[self.source_tier]


class ReactionInfo(_Base):
    cls: str = Field(..., alias="class")           # template key; 'class' is a py keyword
    biological_objective: Dict[str, object] = Field(default_factory=dict)
    model_config = {"extra": "forbid", "populate_by_name": True}


class MechanismSpec(_Base):
    """The full mechanism envelope. ``reaction.class`` selects a template that supplies
    default geometry terms + required reaction-state fields; the config overrides."""
    schema_version: float = 1.0
    mechanism_spec_id: str = "mechanism_v1"
    reaction: ReactionInfo
    reaction_state: ReactionState = Field(default_factory=ReactionState)
    host_context: HostContext = Field(default_factory=HostContext)
    catalytic_residues: List[CatalyticResidueSpec] = Field(default_factory=list)
    geometry_terms: List[GeometryTermSpec] = Field(default_factory=list)
    geometry_calibration: GeometryCalibration = Field(default_factory=GeometryCalibration)

    @model_validator(mode="after")
    def _apply_template(self) -> "MechanismSpec":
        from .templates import get_template  # local import avoids cycle
        tmpl = get_template(self.reaction.cls)
        if tmpl is None:
            raise ValueError(
                f"unknown reaction.class {self.reaction.cls!r}; "
                f"register a template or use one of {list_template_keys()}")
        # 1) fill default geometry terms if the config gave none
        if not self.geometry_terms:
            self.geometry_terms = [GeometryTermSpec(**t) for t in tmpl["default_geometry_terms"]]
        # 2) HARD-GATE the required reaction-state fields (abort, never warn)
        missing = [f for f in tmpl["required_reaction_state"]
                   if getattr(self.reaction_state, f, None) in (None, "")]
        if missing:
            raise ValueError(
                f"reaction.class {self.reaction.cls!r} requires reaction_state fields "
                f"{missing} but they are unset — refusing to run (a missing reaction "
                f"state silently shifts the geometry score)")
        return self

    @property
    def claim_ceiling(self) -> str:
        """The geometry-calibration ceiling — the strongest claim this mechanism's
        geometry evidence can support before reference/control penalties."""
        return self.geometry_calibration.claim_ceiling

    def to_geometry_terms(self):
        """Convert to the runtime md.geometry_spec.GeometryTerm objects (lazy import so
        the schema stays usable in the light env without the md stack)."""
        from evoliez.md.geometry_spec import GeometryTerm
        out = []
        for t in self.geometry_terms:
            out.append(GeometryTerm(
                kind=t.kind, label=t.label,
                a_smarts=t.a_smarts, a_idx=t.a_idx,
                b_smarts=t.b_smarts, b_idx=t.b_idx,
                c_smarts=t.c_smarts, c_idx=t.c_idx,
                a_residue=t.a_residue, a_atom=t.a_atom,
                b_residue=t.b_residue, b_atom=t.b_atom,
                c_residue=t.c_residue, c_atom=t.c_atom,
                distance_min=t.distance_min, distance_max=t.distance_max,
                angle_min=t.angle_min, angle_max=t.angle_max, weight=t.weight))
        return out


def list_template_keys() -> List[str]:
    from .templates import TEMPLATES
    return sorted(TEMPLATES.keys())


def mechanism_from_reactive_geometry(
    rg, reaction_state: ReactionState, *,
    mechanism_spec_id: str = "legacy_hydride_transfer",
    geometry_calibration: Optional[GeometryCalibration] = None,
) -> MechanismSpec:
    """Lift a legacy ``config.ReactiveGeometryConfig`` (donor/acceptor SMARTS NAC) into a
    v3 ``MechanismSpec`` — back-compat so an existing FDH-style target gains the mechanism
    envelope by declaring its reaction state, with the SAME geometry it already used.

    The reaction state must be supplied explicitly (the hard gate is the point — a
    mechanism score is only meaningful with a defined redox/conformational state)."""
    terms = [
        GeometryTermSpec(
            kind="distance", label=f"{rg.label}_donor_acceptor",
            a_smarts=rg.donor_smarts, a_idx=rg.donor_idx,
            b_smarts=rg.acceptor_smarts, b_idx=rg.acceptor_idx,
            distance_min=0.0, distance_max=rg.distance_max),
        GeometryTermSpec(
            kind="angle", label=f"{rg.label}_axis",
            a_smarts=rg.donor_smarts, a_idx=rg.donor_idx,
            b_smarts=rg.donor_smarts, b_idx=rg.donor_idx,
            c_smarts=rg.acceptor_smarts, c_idx=rg.acceptor_idx,
            angle_min=rg.angle_min, angle_max=180.0),
    ]
    return MechanismSpec(
        mechanism_spec_id=mechanism_spec_id,
        reaction=ReactionInfo(**{"class": "hydride_transfer"}),
        reaction_state=reaction_state,
        geometry_terms=terms,
        geometry_calibration=geometry_calibration or GeometryCalibration(),
    )
