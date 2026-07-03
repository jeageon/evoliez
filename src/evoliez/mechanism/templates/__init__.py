"""Mechanism template registry (ROADMAP_V3 D1).

A template maps a ``reaction.class`` to (a) its DEFAULT geometry terms and (b) the
``reaction_state`` fields it REQUIRES (hard-gated at load). The per-target config
overrides geometry specifics; the template makes a new enzyme of a known class a
config-only exercise rather than new code.

Each template is a plain dict so it stays importable in the light env (no rdkit/md):
    {
      "key": str,
      "required_reaction_state": [field, ...],   # MechanismSpec aborts if any is unset
      "default_geometry_terms": [ {GeometryTermSpec fields}, ... ],
      "notes": str,
    }
"""
from __future__ import annotations

from typing import Dict, Optional

# --- hydride transfer (FDH: formate -> nicotinamide C4) -------------------------------
HYDRIDE_TRANSFER = {
    "key": "hydride_transfer",
    "required_reaction_state": ["cofactor_redox_state", "conformational_state"],
    "default_geometry_terms": [
        {
            "kind": "distance", "label": "donor_acceptor",
            # donor = formate carboxylate C ; acceptor = nicotinamide C4
            "a_smarts": "[CX3H1](=O)[O-]", "a_idx": 0,
            "b_smarts": "[cH1]([cH0]C(=O)[NX3])[cH1]", "b_idx": 0,
            "distance_min": 0.0, "distance_max": 3.5, "weight": 1.0,
        },
        {
            "kind": "angle", "label": "hydride_axis",
            "a_smarts": "[CX3H1](=O)[O-]", "a_idx": 0,        # donor heavy
            "b_smarts": "[CX3H1](=O)[O-]", "b_idx": 0,        # transferring H proxy (same C, H attached)
            "c_smarts": "[cH1]([cH0]C(=O)[NX3])[cH1]", "c_idx": 0,  # acceptor
            "angle_min": 150.0, "angle_max": 180.0, "weight": 1.0,
        },
    ],
    "notes": "FDH hydride transfer; NADP+/NADPH redox + closed ternary state required.",
}

# --- nucleophilic acyl substitution (serine hydrolase / TEM-1 β-lactamase) ------------
# DRAFT (ROADMAP_V3 §8): catalytic Ser Oγ attacks the substrate carbonyl C. Protonation
# of the catalytic dyad/triad is state-defining, so protonation_model is required.
NUCLEOPHILIC_ACYL_SUBSTITUTION = {
    "key": "nucleophilic_acyl_substitution",
    "required_reaction_state": ["protonation_model", "conformational_state"],
    "default_geometry_terms": [
        {
            "kind": "distance", "label": "nucleophile_carbonyl",
            # protein nucleophile (catalytic Ser Oγ) -> substrate carbonyl C
            "a_residue": "SER", "a_atom": "OG",
            "b_smarts": "[CX3]=[OX1]", "b_idx": 0,
            "distance_min": 0.0, "distance_max": 3.5, "weight": 1.0,
        },
        {
            "kind": "angle", "label": "burgi_dunitz",
            "a_residue": "SER", "a_atom": "OG",
            "b_smarts": "[CX3]=[OX1]", "b_idx": 0,
            "c_smarts": "[CX3]=[OX1]", "c_idx": 1,            # carbonyl O
            "angle_min": 95.0, "angle_max": 115.0, "weight": 1.0,
        },
    ],
    "notes": "DRAFT — serine-hydrolase acyl-enzyme; needs acyl-enzyme proxy handling (V3-8).",
}

# --- placeholder classes (declared, geometry to be curated per target) ----------------
def _stub(key: str, required, notes: str) -> Dict:
    return {"key": key, "required_reaction_state": list(required),
            "default_geometry_terms": [], "notes": notes}


# --- glycosidic bond cleavage (retaining glycosidase; Koshland double-displacement) ---
# ROADMAP_V3 B7: the 3rd benchmark mechanism (FDH + TEM-1 + glycosidase) now carries real
# default geometry, so a non-FDH smoke contributes geometry evidence, not just a load test.
# Catalytic nucleophile carboxylate (Asp/Glu Oδ) attacks the anomeric carbon (C1, the ring
# carbon bearing the ring O and the exocyclic glycosidic O) in-line, anti to the leaving
# group. Per-target config overrides the residue numbers + substrate anomeric SMARTS.
GLYCOSIDIC_BOND_CLEAVAGE = {
    "key": "glycosidic_bond_cleavage",
    "required_reaction_state": ["protonation_model", "conformational_state"],
    "default_geometry_terms": [
        {
            "kind": "distance", "label": "nucleophile_anomeric_C",
            "a_residue": "ASP", "a_atom": "OD2",             # catalytic nucleophile carboxylate
            "b_smarts": "[CX4]([OX2])[OX2]", "b_idx": 0,     # anomeric C1 (ring-O + glycosidic-O)
            "distance_min": 0.0, "distance_max": 3.3, "weight": 1.0,
        },
        {
            "kind": "angle", "label": "anomeric_inline_attack",
            "a_residue": "ASP", "a_atom": "OD2",             # nucleophile
            "b_smarts": "[CX4]([OX2])[OX2]", "b_idx": 0,     # anomeric C1 (vertex)
            "c_smarts": "[CX4]([OX2])[OX2]", "c_idx": 1,     # a bonded O (ring/leaving) — anti axis
            "angle_min": 150.0, "angle_max": 180.0, "weight": 1.0,
        },
    ],
    "notes": ("Koshland double-displacement (retaining) default; nucleophile Asp/Glu Oδ "
              "-> anomeric C1, in-line attack anti to the leaving group. Per-target: set "
              "the real catalytic residue numbers + substrate anomeric SMARTS."),
}
PHOSPHORYL_TRANSFER = _stub(
    "phosphoryl_transfer", ["metal_state", "conformational_state"],
    "in-line attack on Pγ; Mg2+ coordination state-defining.")
METAL_COFACTOR_REDOX = _stub(
    "metal_cofactor_redox", ["metal_state", "cofactor_redox_state"],
    "metal/heme-assisted redox (P450 etc.); redox/intermediate state hard — stress test only.")
PROTON_TRANSFER_ISOMERIZATION = _stub(
    "proton_transfer_isomerization", ["protonation_model"],
    "general acid/base proton relay / isomerization.")


TEMPLATES: Dict[str, Dict] = {
    t["key"]: t for t in (
        HYDRIDE_TRANSFER,
        NUCLEOPHILIC_ACYL_SUBSTITUTION,
        GLYCOSIDIC_BOND_CLEAVAGE,
        PHOSPHORYL_TRANSFER,
        METAL_COFACTOR_REDOX,
        PROTON_TRANSFER_ISOMERIZATION,
    )
}


def get_template(key: str) -> Optional[Dict]:
    return TEMPLATES.get(key)
