"""Ligand atom-residue interaction graph (spec section 11).

Heterogeneous graph: residue nodes + ligand-atom nodes; residue-ligand contact
edges, residue-residue spatial edges, ligand bond edges. Used for mutation
priors, design-mask construction, and reranker features.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import networkx as nx

from evoliez.features.evolutionary import PositionFeature
from evoliez.features.geometry import (
    ResidueLigandContact,
    dist,
    ligand_centroid,
    residue_ligand_contacts,
)
from evoliez.types import Complex


def build_interaction_graph(
    cx: Complex,
    position_features: Sequence[PositionFeature],
    *,
    contact_cutoff: float = 5.0,
    residue_spatial_cutoff: float = 8.0,
    catalytic_positions: Optional[Sequence[int]] = None,
) -> nx.Graph:
    g = nx.Graph()
    catalytic_positions = set(catalytic_positions or [])
    pf_by_pos: Dict[int, PositionFeature] = {
        f.target_position: f for f in position_features if f.target_position is not None
    }
    lc = ligand_centroid(cx.ligand.atoms) if cx.ligand.atoms else (0.0, 0.0, 0.0)

    # residue nodes
    for res in cx.structure.residues:
        pf = pf_by_pos.get(res.index)
        ref = res.sidechain_centroid or res.ca
        g.add_node(
            f"R{res.index}",
            kind="residue",
            index=res.index,
            aa=res.aa,
            conservation=pf.conservation_score if pf else 0.0,
            entropy=pf.entropy if pf else 0.0,
            gap_frequency=pf.gap_frequency if pf else 0.0,
            residue_class=pf.residue_class if pf else None,
            secondary_structure=res.secondary_structure,
            sasa=res.sasa,
            plddt=res.plddt,
            dist_to_ligand=round(dist(ref, lc), 3),
            is_catalytic=res.index in catalytic_positions,
        )

    # ligand-atom nodes
    for atom in cx.ligand.atoms:
        g.add_node(
            f"L{atom.id}",
            kind="ligand_atom",
            element=atom.element,
            formal_charge=atom.formal_charge,
            partial_charge=atom.partial_charge,
            aromatic=atom.aromatic,
            is_donor=atom.is_donor,
            is_acceptor=atom.is_acceptor,
            is_hydrophobic=atom.is_hydrophobic,
            pharmacophore=atom.pharmacophore,
        )

    # residue-ligand contact edges
    contacts = residue_ligand_contacts(cx.structure, cx.ligand.atoms, contact_cutoff)
    for c in contacts:
        g.add_edge(
            f"R{c.residue_index}",
            f"L{c.ligand_atom_id}",
            etype="contact",
            distance=c.distance,
            interaction_type=c.interaction_type,
            contact_probability=c.contact_probability,
        )

    # residue-residue spatial edges
    residues = cx.structure.residues
    for i in range(len(residues)):
        ri = residues[i]
        ai = ri.sidechain_centroid or ri.ca
        for j in range(i + 1, len(residues)):
            rj = residues[j]
            aj = rj.sidechain_centroid or rj.ca
            d = dist(ai, aj)
            if d <= residue_spatial_cutoff:
                g.add_edge(
                    f"R{ri.index}", f"R{rj.index}", etype="spatial", distance=round(d, 3)
                )
    return g


def contact_summary(g: nx.Graph) -> List[ResidueLigandContact]:
    out: List[ResidueLigandContact] = []
    for u, v, data in g.edges(data=True):
        if data.get("etype") != "contact":
            continue
        rnode = u if g.nodes[u]["kind"] == "residue" else v
        lnode = v if rnode == u else u
        out.append(
            ResidueLigandContact(
                residue_index=g.nodes[rnode]["index"],
                residue_aa=g.nodes[rnode]["aa"],
                ligand_atom_id=lnode[1:],
                distance=data["distance"],
                interaction_type=data["interaction_type"],
                contact_probability=data["contact_probability"],
            )
        )
    return out


def residue_feature_table(g: nx.Graph) -> List[dict]:
    """Flat per-residue feature rows for the reranker (spec 13.2)."""
    rows: List[dict] = []
    for n, d in g.nodes(data=True):
        if d.get("kind") != "residue":
            continue
        contacts = [
            e
            for e in g.edges(n, data=True)
            if e[2].get("etype") == "contact"
        ]
        rows.append(
            {
                "residue_index": d["index"],
                "aa": d["aa"],
                "conservation": d["conservation"],
                "entropy": d["entropy"],
                "gap_frequency": d["gap_frequency"],
                "residue_class": d["residue_class"],
                "sasa": d["sasa"],
                "plddt": d["plddt"],
                "dist_to_ligand": d["dist_to_ligand"],
                "is_catalytic": d["is_catalytic"],
                "n_ligand_contacts": len(contacts),
                "min_contact_distance": round(
                    min((e[2]["distance"] for e in contacts), default=99.0), 3
                ),
            }
        )
    rows.sort(key=lambda r: r["residue_index"])
    return rows
