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
    residues_near_positions,
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
        # v2 MULTI-LIGAND: include residues near EVERY co-modelled ligand (cofactor /
        # substrate / metal / ...), not only the primary design ligand, so the design mask
        # covers the full functional state. extra_ligand_atoms is populated by the predictor
        # (mock) / parser; empty for single-ligand inputs or the real parser fallback (then
        # the catalytic-neighborhood term below carries the active site).
        for _atoms in (getattr(cx, "extra_ligand_atoms", {}) or {}).values():
            if _atoms:
                proximal |= set(ligand_proximal_residues(
                    cx.structure, _atoms, radius=mgcfg.design_radius_angstrom))
        # v2 Phase B: the design mask follows the FUNCTIONAL STATE, not only the primary
        # ligand. Add the neighborhood of the catalytic residues (the reaction site where the
        # cofactor/substrate/metal act), so a generic enzyme's active site is designable even
        # when its functional partners are extra ligands the primary-ligand sphere misses.
        # For FDH the catalytic core sits by NADP so this barely changes the mask; for a
        # metalloenzyme/other reaction it captures the true active site. Catalytic residues
        # themselves stay protected below; only their NEIGHBORHOOD becomes designable.
        if catalytic:
            proximal |= set(residues_near_positions(
                cx.structure, catalytic, radius=mgcfg.design_radius_angstrom))
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

        # --- accuracy layers (user guidance) ------------------------------ #
        adv = ctx.config.advanced
        if adv.mechanism:
            from evoliez.features.mechanism import annotate

            mech = annotate(
                cx,
                catalytic_positions=catalytic,
                cofactor=ctx.config.input.cofactor,
                annotation_file=adv.mechanism_annotation_file,
            )
            ctx.put("mechanism", mech)
            ctx.persist_meta(
                "mechanism",
                {
                    "source": mech.source,
                    "reactive_ligand_atoms": mech.reactive_ligand_atoms,
                    "transfer_distance": mech.transfer_distance,
                    "ts_geometry_score": mech.ts_geometry_score,
                },
            )
        else:
            mech = None

        if adv.ligand_importance:
            from evoliez.features.ligand_importance import ligand_atom_importance

            ctx.put("ligand_importance",
                    ligand_atom_importance(cx.ligand, mech))

        if adv.interaction_fingerprint:
            from evoliez.adapters.plip import (
                fingerprint,
                type_counts,
            )

            ifp = fingerprint(
                cx.structure, cx.ligand.atoms,
                backend=ctx.config.backend_for(self.name),
            )
            ctx.put("ifp_contacts", ifp)
            ctx.persist_meta("ifp_type_counts", type_counts(ifp))

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
