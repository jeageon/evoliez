"""Stage 06 - ligand atom-residue interaction graph + design mask
(spec sections 11 & 12.1)."""

from __future__ import annotations

import json

from evoliez.context import RunContext
from evoliez.db.schema import InteractionEdge
from evoliez.features.evolutionary import assign_residue_classes
from evoliez.features.geometry import (
    ligand_proximal_residues,
    residue_ligand_contacts,
)
from evoliez.features.graph import (
    build_interaction_graph,
    contact_summary,
    residue_feature_table,
)
from evoliez.stages.base import Stage


class InteractionGraphStage(Stage):
    name = "s06_graph"

    def run(self, ctx: RunContext) -> None:
        cx = ctx.require("wt_complex")
        feats = ctx.require("position_features")
        catalytic = set(ctx.require("catalytic_positions"))
        fixed = set(ctx.require("fixed_positions"))
        mgcfg = ctx.config.mutation_generation

        proximal = set(
            ligand_proximal_residues(
                cx.structure, cx.ligand.atoms, radius=mgcfg.design_radius_angstrom
            )
        )
        second_shell = set(
            ligand_proximal_residues(
                cx.structure, cx.ligand.atoms,
                radius=mgcfg.design_radius_angstrom + 4.0,
            )
        ) - proximal

        assign_residue_classes(
            feats,
            catalytic_positions=catalytic,
            fold_critical_threshold=mgcfg.conservation_fix_threshold,
            ligand_proximal=proximal,
            second_shell=second_shell,
        )

        g = build_interaction_graph(
            cx, feats, catalytic_positions=catalytic,
            contact_cutoff=5.0,
            residue_spatial_cutoff=8.0,
        )
        contacts = contact_summary(g)
        table = residue_feature_table(g)

        # design mask (spec 12.1): ligand-proximal, not fixed/catalytic,
        # not over-conserved if configured.
        cons_by_pos = {
            f.target_position: f.conservation_score
            for f in feats if f.target_position is not None
        }
        designable = []
        for pos in sorted(proximal):
            if mgcfg.fix_catalytic_residues and pos in catalytic:
                continue
            if pos in fixed:
                continue
            if (
                mgcfg.fix_highly_conserved_residues
                and cons_by_pos.get(pos, 0.0) >= mgcfg.conservation_fix_threshold
            ):
                continue
            designable.append(pos)

        ctx.put("interaction_graph", g)
        ctx.put("contacts", contacts)
        ctx.put("residue_table", table)
        ctx.put("designable_positions", designable)
        ctx.put("position_features", feats)  # now class-annotated
        ctx.persist_meta("n_contacts", len(contacts))
        ctx.persist_meta("n_designable", len(designable))
        ctx.persist_meta("designable_positions", designable)

        (ctx.paths.interaction_graphs / "graph_features.json").write_text(
            json.dumps(
                {
                    "residues": table,
                    "contacts": [vars(c) for c in contacts],
                    "designable_positions": designable,
                },
                indent=2,
            )
        )

        assert ctx.store is not None
        with ctx.store.session() as s:
            if not s.query(InteractionEdge).first():
                for c in contacts:
                    s.add(
                        InteractionEdge(
                            project_id=ctx.project_id,
                            target_position=c.residue_index,
                            ligand_atom_id=c.ligand_atom_id,
                            distance=c.distance,
                            interaction_type=c.interaction_type,
                            contact_probability=c.contact_probability,
                        )
                    )
        self.log.info(
            "graph: %d nodes, %d contacts, %d designable positions",
            g.number_of_nodes(), len(contacts), len(designable),
        )
        if not designable:
            self.log.warning(
                "no designable positions - widen design_radius_angstrom "
                "or relax conservation fixing"
            )
