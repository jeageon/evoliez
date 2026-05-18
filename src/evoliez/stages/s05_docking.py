"""Stage 05 - reference docking / redocking ensemble (spec section 10).

Establishes the WT reference pose and a docking-consistency baseline. Per-
mutant redocking happens in stage 09 (non-MD validation).
"""

from __future__ import annotations

from evoliez.adapters import diffdock, gnina, vina
from evoliez.context import RunContext
from evoliez.db.schema import DockingPose
from evoliez.stages.base import Stage


def redock_with(method: str, ctx: RunContext, cand_id, structure, ref_atoms,
                cfg, workdir, instability, smiles):
    backend = ctx.config.backend_for("s05_docking")
    if method == "gnina":
        return gnina.redock(cand_id, structure, ref_atoms, cfg, workdir,
                            instability=instability, backend=backend,
                            dry_run=ctx.dry_run)
    if method == "diffdock":
        return diffdock.redock(cand_id, structure, ref_atoms, cfg, workdir,
                               instability=instability, smiles=smiles,
                               backend=backend, dry_run=ctx.dry_run)
    return vina.redock(cand_id, structure, ref_atoms, cfg, workdir,
                       instability=instability, backend=backend,
                       dry_run=ctx.dry_run)


class DockingStage(Stage):
    name = "s05_docking"

    def run(self, ctx: RunContext) -> None:
        cx = ctx.require("wt_complex")
        cfg = ctx.config.validation.redocking
        ref_atoms = cx.ligand.atoms
        ctx.put("reference_atoms", ref_atoms)

        poses = []
        for method in cfg.methods:
            p = redock_with(
                method, ctx, "wt", cx.structure, ref_atoms, cfg,
                ctx.paths.docking / method, 0.05, cx.ligand.smiles,
            )
            poses.append(p)
        ctx.put("wt_reference_poses", poses)

        assert ctx.store is not None
        with ctx.store.session() as s:
            for p in poses:
                s.add(
                    DockingPose(
                        project_id=ctx.project_id,
                        candidate_id="wt",
                        method=p.method,
                        score=p.score,
                        confidence=1.0,
                        pose_cluster=p.cluster,
                        ligand_rmsd_to_reference=p.rmsd_to_reference or 0.0,
                    )
                )
        self.log.info(
            "WT reference docking: %s",
            ", ".join(f"{p.method}={p.score:.2f}" for p in poses),
        )
