"""Per-ligand-atom importance weighting (user §2).

Not every ligand atom matters equally for enzyme function. Reactive /
cofactor-specificity / charged atoms weigh more than a flexible hydrophobic
tail. These weights scale the contact loss and interaction-gain so the model
learns the geometry that matters for catalysis, not an unweighted average.
"""

from __future__ import annotations

from typing import Dict, Optional

from evoliez.features.mechanism import Mechanism
from evoliez.types import Ligand


def ligand_atom_importance(
    ligand: Ligand, mechanism: Optional[Mechanism] = None
) -> Dict[str, float]:
    reactive = set(mechanism.reactive_ligand_atoms) if mechanism else set()
    w: Dict[str, float] = {}
    for a in ligand.atoms:
        if a.id in reactive:
            v = 1.0  # reactive / catalysis-determining
        elif a.formal_charge != 0:
            v = 0.9  # phosphate / carboxylate - specificity / anchoring
        elif a.is_donor or a.is_acceptor:
            v = 0.7  # H-bond
        elif a.aromatic:
            v = 0.5  # ring recognition
        elif a.is_hydrophobic:
            v = 0.35  # packing
        else:
            v = 0.4
        w[a.id] = round(v, 3)
    return w
