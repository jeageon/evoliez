"""Multi-level ML dataset exporters (user spec section 7).

Builds the pose / residue / edge / mutation / variant tables and writes them
(CSV) under ``<run>/ml_datasets/`` with a ``roles.json`` declaring each
column's allowed role. Boltz-derived columns are tagged feature/weight/
weak-label/filter; experimental columns are the only supervised labels
(enforced via ml/labels).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

from evoliez.features.boltz_features import EnsembleContact
from evoliez.features.evolutionary import PositionFeature
from evoliez.ml.labels import COLUMN_ROLES, is_boltz_derived
from evoliez.types import Candidate, Complex


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def pose_rows(group_id: str, cx: Complex) -> List[dict]:
    """One row per Boltz diffusion sample (spec 7.1)."""
    rows = []
    for s in cx.samples:
        r = {"group_id": group_id, "sample_idx": s.idx}
        r.update({k: round(float(v), 5) for k, v in s.metrics.items()})
        rows.append(r)
    return rows


def residue_rows(
    cx: Complex, position_features: Sequence[PositionFeature]
) -> List[dict]:
    """One row per residue (spec 7.2)."""
    pf = {f.target_position: f for f in position_features
          if f.target_position is not None}
    rows = []
    for res in cx.structure.residues:
        f = pf.get(res.index)
        rows.append({
            "residue_index": res.index,
            "aa": res.aa,
            "conservation": f.conservation_score if f else 0.0,
            "entropy": f.entropy if f else 0.0,
            "gap_frequency": f.gap_frequency if f else 0.0,
            "plddt": round(res.plddt, 3),
            "sasa": round(res.sasa, 3),
            "residue_class": f.residue_class if f else None,
            # weak label (not experimental): designable vs not
            "weak_designable": int(bool(f and f.residue_class in (3, 4, 5))),
        })
    return rows


def edge_rows(
    contacts: Sequence[EnsembleContact], freq_threshold: float = 0.5
) -> List[dict]:
    """One row per (residue, ligand atom) - the core dataset (spec 7.3)."""
    rows = []
    for e in contacts:
        rows.append({
            "residue_index": e.residue_index,
            "ligand_atom_id": e.ligand_atom_id,
            "mean_distance": e.mean_distance,
            "contact_frequency": e.contact_frequency,
            "confidence_weighted_score": e.confidence_weighted_score,
            # weak label: high-confidence recurring contact (NOT experimental)
            "weak_contact": int(e.contact_frequency >= freq_threshold),
        })
    return rows


def mutation_rows(candidates: Sequence[Candidate]) -> List[dict]:
    """One row per mutation candidate (spec 7.4). Includes Boltz/delta
    features + a weak label; experimental label column left empty unless
    provided elsewhere."""
    rows = []
    for c in candidates:
        feat = c.details.get("features", {})
        row = {
            "candidate_id": c.candidate_id,
            "mutations": c.mutation_str,
            "generator": c.generator,
            "n_mutations": len(c.mutations),
        }
        for k, v in feat.items():
            row[k] = v
        for k in ("ml_score", "family_interaction_score", "ddg_fold",
                  "docking_score", "redocking_consistency",
                  "complex_confidence"):
            if k in c.scores:
                row[k] = c.scores[k]
        for k, v in c.details.get("delta", {}).items():
            row[k] = v
        # weak label only (no experiment): permissive vs not
        row["weak_permissive"] = int(
            feat.get("msa_permissiveness", 0.0) >= 0.5
        )
        row["experimental_label"] = ""  # filled only from experiment
        rows.append(row)
    return rows


def variant_rows(candidates: Sequence[Candidate]) -> List[dict]:
    """One row per (single/multi) variant (spec 7.5)."""
    rows = []
    for c in candidates:
        rows.append({
            "candidate_id": c.candidate_id,
            "mutations": c.mutation_str,
            "final_score": c.scores.get("final_score"),
            "ml_score": c.scores.get("ml_score"),
            "family_interaction_score": c.scores.get("family_interaction_score"),
            "md_lite_score": c.scores.get("md_lite_score"),
            "ddg_fold": c.scores.get("ddg_fold"),
            "experimental_label": "",
        })
    return rows


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def _column_role(col: str) -> str:
    c = col.lower()
    if c in ("experimental_label",):
        return "supervised_label (experiment only)"
    if c.startswith("weak_"):
        return "weak_label"
    if c.startswith("d_"):
        return "feature (delta)"
    if c in ("contact_frequency", "confidence_weighted_score"):
        return "feature/weak_label"
    if is_boltz_derived(c):
        return "feature/sample_weight/weak_label/filter"
    return "feature"


def write_datasets(out_dir: Path, tables: Dict[str, List[dict]]) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    roles: Dict[str, Dict[str, str]] = {}
    for name, rows in tables.items():
        path = out_dir / f"{name}.csv"
        cols: List[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        roles[name] = {c: _column_role(c) for c in cols}
        written.append(path)
    (out_dir / "roles.json").write_text(
        json.dumps(
            {
                "column_roles": roles,
                "policy": {k: sorted(v) for k, v in COLUMN_ROLES.items()},
                "principle": "Boltz outputs = feature/weight/weak-label/"
                "filter; experimental data = the only supervised label",
            },
            indent=2,
        )
    )
    return written
