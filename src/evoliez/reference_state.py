"""ROADMAP_V2 Phase A — the functional REFERENCE STATE.

When `input.target_structure` is a curated/experimental PDB, use IT (not a computed Boltz
pose) as the WT reference complex the whole pipeline anchors on. A curated structure is only
admissible behind HARD GATES (§2a.3) — residue numbering ↔ sequence, ligand atom map, role↔
chain map, charge/protonation provenance — so a wrong mapping ABORTS instead of silently
producing garbage. Generic + multi-ligand (uses the per-chain parser).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class ReferenceGates:
    passed: bool
    checks: Dict[str, bool]
    reasons: List[str]
    source: str = "curated_pdb"

    def to_json(self) -> dict:
        return {"passed": self.passed, "reference_source": self.source,
                "checks": self.checks, "reasons": self.reasons}


def validate_reference(
    residues: Sequence,
    primary_ligand_atoms: Sequence,
    extra_ligand_atoms: Dict[str, List],
    *,
    target_sequence: Optional[str],
    catalytic_positions: Sequence[int],
    design_ligand_natoms: int,
    expected_extra_ligands: int,
    has_charge_provenance: bool,
) -> ReferenceGates:
    """Hard gates for a curated reference. ALL must pass; any failure -> caller aborts."""
    checks: Dict[str, bool] = {}
    reasons: List[str] = []

    checks["structure_parsed"] = bool(residues) and bool(primary_ligand_atoms)
    if not checks["structure_parsed"]:
        reasons.append("no protein residues and/or no primary ligand in the PDB")

    n, m = len(residues), len(target_sequence or "")
    checks["sequence_to_structure_mapping"] = n > 0 and (m == 0 or n >= 0.5 * m)
    if not checks["sequence_to_structure_mapping"]:
        reasons.append(f"residue count {n} implausible vs sequence length {m}")

    idxs = {getattr(r, "index", None) for r in residues}
    missing = [p for p in (catalytic_positions or []) if p not in idxs]
    checks["catalytic_residues_present"] = not missing
    if missing:
        reasons.append(f"catalytic positions absent (numbering mismatch): {missing}")

    if design_ligand_natoms:
        tol = max(2, int(0.3 * design_ligand_natoms))   # crystal H often absent
        ok = abs(len(primary_ligand_atoms) - design_ligand_natoms) <= tol
        checks["reference_atom_map_verified"] = ok
        if not ok:
            reasons.append(
                f"primary ligand atom count {len(primary_ligand_atoms)} != design "
                f"ligand {design_ligand_natoms} (tol {tol})")
    else:
        checks["reference_atom_map_verified"] = True

    checks["ligand_role_chain_map"] = len(extra_ligand_atoms) >= expected_extra_ligands
    if not checks["ligand_role_chain_map"]:
        reasons.append(
            f"expected >= {expected_extra_ligands} extra ligand chains, "
            f"found {len(extra_ligand_atoms)}")

    checks["charge_protonation_provenance"] = bool(has_charge_provenance)
    if not checks["charge_protonation_provenance"]:
        reasons.append("no charge/protonation provenance (charges_mol2 or net_charge) for the "
                       "design ligand — required for a paper-grade curated reference")

    passed = all(checks.values())
    return ReferenceGates(passed=passed, checks=checks, reasons=reasons)


def load_reference_complex(pdb_path: str, *, target_sequence: str, ligand,
                           extra_ligands: Sequence, catalytic_positions: Sequence[int]):
    """Parse a curated PDB into a (Complex, ReferenceGates). The Complex is returned even when
    the gates FAIL (so the caller can log the reasons); the caller must honor gates.passed."""
    from evoliez.adapters.boltz import _parse_structure_atoms
    from evoliez.types import Complex, Ligand, ProteinStructure

    residues, primary, extra = _parse_structure_atoms(Path(pdb_path))
    # adopt the target sequence onto the parsed residues (numbering preserved)
    for i, r in enumerate(residues):
        if i < len(target_sequence):
            r.aa = target_sequence[i]

    has_charge = bool(getattr(ligand, "charges_mol2", None)
                      or getattr(ligand, "formal_charge", None) is not None
                      or getattr(ligand, "net_charge", None) is not None)
    gates = validate_reference(
        residues, primary, extra,
        target_sequence=target_sequence,
        catalytic_positions=catalytic_positions,
        design_ligand_natoms=len(getattr(ligand, "atoms", []) or []),
        expected_extra_ligands=len(extra_ligands or []),
        has_charge_provenance=has_charge,
    )

    struct = ProteinStructure(sequence=target_sequence, residues=residues,
                              method="curated_pdb", pdb_path=str(pdb_path))
    cx = Complex(
        structure=struct,
        ligand=Ligand(id=ligand.id, smiles=getattr(ligand, "smiles", ""),
                      atoms=primary or list(getattr(ligand, "atoms", []) or []),
                      formal_charge=getattr(ligand, "formal_charge", None),
                      charges_mol2=getattr(ligand, "charges_mol2", None),
                      allow_am1bcc=getattr(ligand, "allow_am1bcc", True)),
        method="curated_pdb",
        path=str(pdb_path),
        extra_ligand_atoms=extra,
    )
    return cx, gates


# --------------------------------------------------------------------------------------
# ROADMAP_V3 D2/D6 — grade a reference (beyond pass/fail) + build the active-state ensemble
# --------------------------------------------------------------------------------------
def grade_reference(
    gates: "ReferenceGates", *, tier: str, source: str = "curated_pdb",
    substrate_is_real: bool = True, analog_identity: Optional[str] = None,
    sequence_identity_to_target: Optional[float] = None,
    resolution_A: Optional[float] = None, ligand_state: Optional[str] = None,
    known_active_controls_available: bool = False,
    known_inactive_controls_available: bool = False,
    active_closed_state_supported: str = "unknown",
):
    """Turn a binary ``ReferenceGates`` into a graded ``ReferenceConfidenceCard`` (D2).
    The hard gates still abort upstream; this only GRADES what survives, deriving a
    ``claim_strength`` that caps downstream report language (ClaimGuard, D3/V3-4)."""
    from evoliez.mechanism.cards import ReferenceConfidenceCard
    checks = gates.checks if gates else {}
    return ReferenceConfidenceCard(
        tier=tier, source=source,
        resolution_A=resolution_A,
        sequence_identity_to_target=sequence_identity_to_target,
        ligand_state=ligand_state,
        substrate_is_real=substrate_is_real,
        analog_identity=analog_identity,
        catalytic_residue_alignment_verified=bool(checks.get("catalytic_residues_present", False)),
        functional_atom_mapping_verified=bool(checks.get("reference_atom_map_verified", False)),
        protonation_state_assigned=bool(checks.get("charge_protonation_provenance", False)),
        redox_state_assigned=bool(checks.get("charge_protonation_provenance", False)),
        active_closed_state_supported=active_closed_state_supported,
        known_active_controls_available=known_active_controls_available,
        known_inactive_controls_available=known_inactive_controls_available,
    )


def build_reference_ensemble(
    ensemble_id: str, members: Sequence, *,
    mechanism_spec_id: str = "mechanism_v1",
    distance_measurements: Optional[Dict[str, Dict[str, float]]] = None,
    angle_measurements: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Build an ``ActiveStateReferenceEnsemble`` (D6) from member references + their
    per-term geometry measurements. ``*_measurements`` = {term_label: {reference_id:
    value}}. Computes each term's median/IQR distribution and a normalized geometry
    disagreement (coefficient of dispersion = IQR/|median|, averaged over terms), which
    the soft kernels (D4) center on and which lowers the geometry-axis confidence."""
    import statistics as _stats
    from evoliez.mechanism.cards import (
        ActiveStateReferenceEnsemble, EnsembleDisagreement, ReferenceMember,
        TermDistribution,
    )

    def _dist_for(meas):
        out, dispersions = {}, []
        for label, per_ref in (meas or {}).items():
            vals = [v for v in per_ref.values() if isinstance(v, (int, float))]
            if not vals:
                out[label] = TermDistribution(source_references=list(per_ref.keys()))
                continue
            med = _stats.median(vals)
            iqr = _iqr(vals)
            out[label] = TermDistribution(median=med, iqr=iqr,
                                          source_references=list(per_ref.keys()))
            if med not in (0, None) and len(vals) >= 2:
                dispersions.append(abs(iqr) / abs(med))
        return out, dispersions

    dist_terms, d1 = _dist_for(distance_measurements)
    angle_terms, d2 = _dist_for(angle_measurements)
    disp = (sum(d1 + d2) / len(d1 + d2)) if (d1 + d2) else 0.0

    mem_objs = []
    for m in members:
        if isinstance(m, ReferenceMember):
            mem_objs.append(m)
        else:  # accept a dict
            mem_objs.append(ReferenceMember(**m))

    return ActiveStateReferenceEnsemble(
        ensemble_id=ensemble_id, mechanism_spec_id=mechanism_spec_id,
        references=mem_objs, distance_terms=dist_terms, angle_terms=angle_terms,
        disagreement=EnsembleDisagreement(geometry_variance=round(disp, 4)),
    )


def _iqr(vals: Sequence[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n < 2:
        return 0.0

    def _pct(p):
        k = (n - 1) * p
        lo, hi = int(k), min(int(k) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return _pct(0.75) - _pct(0.25)
