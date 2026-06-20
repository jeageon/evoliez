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
                cfg, workdir, instability, smiles, *, stage_name="s05_docking",
                context_chains=None):
    # Backend of the CALLING stage, not always s05: s09 redocking must honour
    # `--stage-backend s09_nonmd=real` instead of silently using s05's backend
    # (audit P1 #8). s05's own call keeps the default. context_chains = the OTHER
    # co-modelled ligands' chains, kept as fixed receptor context (gnina/vina;
    # diffdock ignores it).
    backend = ctx.config.backend_for(stage_name)
    if method == "gnina":
        return gnina.redock(cand_id, structure, ref_atoms, cfg, workdir,
                            instability=instability, backend=backend,
                            dry_run=ctx.dry_run, context_chains=context_chains)
    if method == "diffdock":
        return diffdock.redock(cand_id, structure, ref_atoms, cfg, workdir,
                               instability=instability, smiles=smiles,
                               backend=backend, dry_run=ctx.dry_run,
                               context_chains=context_chains)
    return vina.redock(cand_id, structure, ref_atoms, cfg, workdir,
                       instability=instability, backend=backend,
                       dry_run=ctx.dry_run, context_chains=context_chains)


class DockingStage(Stage):
    name = "s05_docking"

    def run(self, ctx: RunContext) -> None:
        from evoliez.adapters.base import het_chains_in_pdb, parse_pdb_het_chain

        cx = ctx.require("wt_complex")
        cfg = ctx.config.validation.redocking
        extras = ctx.get("extra_ligands", []) or []
        # s09 (per-mutant redocking consistency) targets the DESIGN ligand only.
        ctx.put("reference_atoms", cx.ligand.atoms)

        # Dock EVERY input ligand, each with the OTHERS as fixed receptor context
        # (user choice). The s04 complex PDB carries each ligand at its placed
        # pose; map them to chains (design ligand first, then extras in order).
        pdb = getattr(cx.structure, "pdb_path", None)
        het = het_chains_in_pdb(pdb) if pdb else []
        ligs = [(cx.ligand.id, "wt", cx.ligand.smiles, cx.ligand.atoms,
                 het[0] if het else None)]
        for i, el in enumerate(extras):
            ch = het[i + 1] if i + 1 < len(het) else None
            placed = parse_pdb_het_chain(pdb, ch) if (pdb and ch) else el.atoms
            ligs.append((el.id, f"wt__{el.id}", el.smiles, placed, ch))
        all_chains = [c for (_, _, _, _, c) in ligs if c]

        triples = []  # (candidate_id, ligand_id, Pose)
        for lid, cand, smiles, ref_atoms, ch in ligs:
            if not ref_atoms:
                self.log.warning("no reference atoms for ligand %s; skip dock", lid)
                continue
            ctx_chains = [c for c in all_chains if c != ch] or None
            for method in cfg.methods:
                p = redock_with(
                    method, ctx, cand, cx.structure, ref_atoms, cfg,
                    ctx.paths.docking / method, 0.05, smiles,
                    context_chains=ctx_chains,
                )
                triples.append((cand, lid, p))
        ctx.put("wt_reference_poses", [p for _, _, p in triples])

        assert ctx.store is not None
        with ctx.store.session() as s:
            # idempotent (no load() on the original run): clear each candidate's
            # prior WT poses before inserting so a resume never duplicates rows.
            for cand in {c for c, _, _ in triples}:
                s.query(DockingPose).filter_by(
                    project_id=ctx.project_id, candidate_id=cand).delete()
            for cand, _lid, p in triples:
                s.add(DockingPose(
                    project_id=ctx.project_id, candidate_id=cand, method=p.method,
                    score=p.score, confidence=1.0, pose_cluster=p.cluster,
                    ligand_rmsd_to_reference=p.rmsd_to_reference or 0.0))
        self.log.info(
            "WT reference docking (%d ligand x %d method): %s",
            len(ligs), len(cfg.methods),
            ", ".join(f"{lid}:{p.method}={p.score:.2f}" for _, lid, p in triples),
        )
        self._write_report(ctx)

    def _write_report(self, ctx) -> None:
        """s05 reference-docking report: 3D pose overlay (reusing the s04 viewer
        patterns) + the DiffDock confidence landscape + cross-method agreement.
        Secondary — never fail the stage."""
        try:
            from datetime import datetime

            from evoliez.io.docking_report import (compute_docking_stats,
                                                   write_docking_report)
            cfg = ctx.config.validation.redocking
            li = ctx.config.input.ligand
            stats = compute_docking_stats(
                str(ctx.paths.root), str(ctx.paths.db_path), cfg.methods,
                ligand_smiles=li.value if li.type == "smiles" else None)
            if not stats["scores"]:
                return
            write_docking_report(
                ctx.paths.reports / "docking_report.html",
                target_id=ctx.config.input.target_id, stats=stats,
                ligand_name=ctx.config.input.ligand.id,
                generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                conditions=[
                    ("methods", ", ".join(cfg.methods)),
                    ("poses / method", cfg.poses_per_candidate),
                    ("reference", "s04 Boltz WT pose"),
                    ("backend", self.backend(ctx).value),
                ],
            )
            self.log.info("docking report: %s",
                          ctx.paths.reports / "docking_report.html")
        except Exception as exc:  # report is secondary — never fail the stage
            self.log.warning("docking report failed: %s", exc)

    def load(self, ctx: RunContext) -> bool:
        """Resume without re-docking. s05's only output consumed downstream from
        ctx is ``reference_atoms`` (s09) — which IS the WT ligand's atoms, so
        restore it from the already-in-ctx ``wt_complex`` (s04 repopulates it
        first on resume). The WT docking poses run() persists to the DB are never
        read back from ctx, so re-running gnina/diffdock on every --resume is
        pure waste — and would need the (separate, GPU) docking env just to reach
        a later stage. Reload instead."""
        cx = ctx.get("wt_complex")
        if cx is None:
            return False  # wt_complex not repopulated yet -> let s05 re-run
        ctx.put("reference_atoms", cx.ligand.atoms)
        self._write_report(ctx)
        return True
