"""Stage 06b - family interaction-geometry model (spec 9.2 + 13.3).

For clustered representative homologs: predict structure, dock the ligand as a
pose ensemble, extract per-ligand-atom interaction-distance fingerprints,
statistically select family-consensus poses (augmented training data), and
train a self-supervised consensus/outlier classifier. Runs before mutation
reranking; its score feeds candidate ranking.
"""

from __future__ import annotations

from typing import Dict, List

from evoliez.adapters.boltz import predict_complex
from evoliez.adapters.docking import dock_ensemble
from evoliez.context import RunContext
from evoliez.features.interaction_descriptor import (
    complex_fingerprint,
    describe,
    fingerprint_dim,
)
from evoliez.ml.interaction_model import InteractionModel
from evoliez.ml.pose_selection import PoseRecord, select_poses
from evoliez.stages.base import Stage


def _pick_representatives(homologs, n: int):
    """One representative per cluster (highest identity), topped up by
    identity-diverse extras until n."""
    by_cluster: Dict[int, list] = {}
    for h in homologs:
        by_cluster.setdefault(h.cluster_id, []).append(h)
    reps = []
    for cid in sorted(by_cluster):
        reps.append(max(by_cluster[cid], key=lambda h: h.identity))
    if len(reps) < n:
        chosen = {id(r) for r in reps}
        extra = sorted(
            (h for h in homologs if id(h) not in chosen),
            key=lambda h: h.identity, reverse=True,
        )
        reps.extend(extra[: n - len(reps)])
    return reps[:n]


class InteractionModelStage(Stage):
    name = "s06b_interaction"

    def run(self, ctx: RunContext) -> None:
        cfg = ctx.config.interaction_model
        if not cfg.enabled:
            self.log.info("interaction model disabled; skipping")
            ctx.put("interaction_model", None)
            return

        homologs = ctx.require("homologs")
        ligand = ctx.require("ligand")
        wt = ctx.require("wt_complex")
        backend = ctx.config.backend_for(self.name)
        dcfg = ctx.config.validation.redocking
        ref_atoms = wt.ligand.atoms

        reps = _pick_representatives(homologs, cfg.representative_homologs)
        self.log.info(
            "representatives=%d (of %d homologs), poses/homolog=%d",
            len(reps), len(homologs), cfg.poses_per_homolog,
        )

        records: List[PoseRecord] = []
        n_poses_total = 0
        for i, h in enumerate(reps):
            cx = predict_complex(
                f"hom_{i:03d}", h.sequence, ligand,
                ctx.config.complex_prediction,
                ctx.paths.structures / "representatives",
                backend=backend, dry_run=ctx.dry_run,
            )
            poses = dock_ensemble(
                cfg.docking_method, f"hom_{i:03d}", cx.structure,
                cx.ligand.atoms or ref_atoms, dcfg,
                ctx.paths.docking / "homolog_ensemble",
                n_poses=cfg.poses_per_homolog, smiles=ligand.smiles,
                backend=backend, dry_run=ctx.dry_run,
            )
            for pose in poses:
                fp = complex_fingerprint(
                    cx.structure, pose.ligand_atoms,
                    cutoff=cfg.contact_cutoff, k_nearest=cfg.k_nearest_residues,
                )
                records.append(
                    PoseRecord(
                        group_id=f"hom_{i:03d}",
                        fingerprint=fp,
                        msa_membership=1.0,
                        identity_to_target=float(h.identity),
                        pred_score=round(-pose.score + cx.confidence, 4),
                    )
                )
                n_poses_total += 1

        sel = select_poses(
            records,
            select_z=cfg.pose_select_mad_z,
            outlier_z=cfg.pose_outlier_mad_z,
            min_decoys_per_group=cfg.min_decoys_per_homolog,
            seed=ctx.config.seed,
        )

        model = InteractionModel(
            cutoff=cfg.contact_cutoff, k_nearest=cfg.k_nearest_residues
        )
        model.fit(sel)
        model.save(ctx.paths.interaction_graphs / "interaction_model.json")
        ctx.put("interaction_model", model)

        ctx.persist_meta(
            "interaction_model",
            {
                "representatives": len(reps),
                "poses_total": n_poses_total,
                "train_rows": int(sel.X.shape[0]),
                "n_consensus": sel.n_positive,
                "n_outlier": sel.n_outlier,
                "n_decoy": sel.n_decoy,
                "model_kind": model.kind,
                "fp_dim": fingerprint_dim(cfg.k_nearest_residues),
            },
        )
        self.log.info(
            "trained %s on %d rows (consensus=%d, outlier=%d, decoy=%d)",
            model.kind, sel.X.shape[0], sel.n_positive,
            sel.n_outlier, sel.n_decoy,
        )
        if sel.consensus.size:
            self.log.info(
                "family consensus interaction: %s",
                describe(sel.consensus, cfg.k_nearest_residues),
            )

    def load(self, ctx: RunContext) -> bool:
        p = ctx.paths.interaction_graphs / "interaction_model.json"
        if not p.exists():
            return False
        ctx.put("interaction_model", InteractionModel.load(p))
        return True
