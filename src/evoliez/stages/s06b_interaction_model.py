"""Stage 06b - family interaction-geometry model (spec 9.2 + 13.3).

For clustered representative homologs: predict the Boltz **diffusion-sample
ensemble**, extract per-ligand-atom interaction-distance fingerprints, use
ensemble **contact frequency** + Boltz pose reliability (as a SAMPLE WEIGHT,
never a label), statistically select family-consensus poses (augmented
training data), and train a self-supervised consensus/outlier classifier.
Exports the pose- and edge-level ML datasets.
"""

from __future__ import annotations

from typing import Dict, List

from evoliez.adapters.boltz import predict_complex
from evoliez.context import RunContext
from evoliez.features.boltz_features import ensemble_contacts, pose_consensus
from evoliez.features.interaction_descriptor import (
    complex_fingerprint,
    describe,
    fingerprint_dim,
)
from evoliez.logging_utils import get_logger
from evoliez.ml.datasets import edge_rows, pose_rows
from evoliez.ml.interaction_model import InteractionModel
from evoliez.ml.pose_selection import PoseRecord, select_poses
from evoliez.stages.base import Stage

_log = get_logger("evoliez.s06b_interaction")


def _export_fingerprint_matrix(out_dir, records, fp_dim: int) -> None:
    """Write the per-pose fingerprint matrix to ``ml_datasets/`` for HTML
    report consumption (section 5 - interaction fingerprint heatmap).

    Rows correspond to the **real Boltz poses** that fed into the
    InteractionModel training selection (one row per ``PoseRecord``), in the
    order they were collected. This is the data the heatmap renders: the
    synthetic decoy rows that ``select_poses`` mixes into ``sel.X`` for
    training are intentionally omitted - they are not real interaction
    fingerprints and would only distort the heatmap.

    Failure is logged at WARNING and swallowed so a CSV-write hiccup never
    breaks the pipeline.
    """
    import csv as _csv
    import json as _json
    from pathlib import Path as _Path

    # Pull the per-feature semantic labels from the descriptor module so
    # the heatmap renderer can show "0-0.75Å"/"hbond"/"mean"/"k1" instead
    # of "feature_N". Falling back to an empty list keeps this best-effort
    # for older fingerprints that don't match the standard dim.
    try:
        from evoliez.features.interaction_descriptor import (
            fingerprint_dim as _fp_dim,
            fingerprint_feature_labels as _fp_labels,
        )
        feature_labels = (
            _fp_labels(k_nearest=6, n_bins=8, cutoff=6.0)
            if int(fp_dim) == _fp_dim(k_nearest=6, n_bins=8)
            else []
        )
    except Exception:  # noqa: BLE001 - meta is best-effort
        feature_labels = []

    out_dir = _Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "fingerprint_matrix.csv"
        meta_path = out_dir / "fingerprint_matrix_meta.json"

        header = ["pose_id", "group_id"] + [f"feature_{i}" for i in range(fp_dim)]
        n_rows = 0
        with csv_path.open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(header)
            for idx, rec in enumerate(records):
                fp = rec.fingerprint
                if fp is None or len(fp) != fp_dim:
                    continue
                pose_id = f"{rec.group_id}__p{idx:04d}"
                w.writerow([pose_id, rec.group_id]
                           + [round(float(v), 6) for v in fp])
                n_rows += 1
        meta = {
            "row_kind": "pose",
            "row_headers": ["pose_id", "group_id"],
            "feature_dim": int(fp_dim),
            "n_rows": int(n_rows),
            "source": "s06b_interaction_model.select_poses input records",
            "note": (
                "Per-pose interaction-distance fingerprints (Boltz "
                "diffusion-sample ensemble across representative homologs). "
                "Synthetic training decoys are NOT included - this is the "
                "real-pose matrix used by the HTML report fingerprint heatmap."
            ),
            # New (P1f): semantic feature labels so the heatmap x-axis
            # reads as physical bins rather than opaque "feature_N".
            "feature_labels": feature_labels,
        }
        meta_path.write_text(_json.dumps(meta, indent=2))
    except Exception as exc:  # noqa: BLE001 - report-only artefact
        _log.warning("fingerprint_matrix export skipped: %s", exc)


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

        # Boltz diffusion-sample ensemble sized by poses_per_homolog.
        n_reps = cfg.representative_homologs
        n_samp = cfg.poses_per_homolog
        if ctx.dry_run:
            # dry-run only previews commands; real Boltz never runs, so do
            # NOT grind the synthetic mock over the full (e.g. 80x15)
            # ensemble - cap it so dry-run stays a fast command preview.
            n_reps, n_samp = min(n_reps, 2), min(n_samp, 2)
            self.log.info("[dry-run] capping s06b ensemble to %d reps x %d "
                          "samples (command-preview only)", n_reps, n_samp)
        cp_cfg = ctx.config.complex_prediction.model_copy(
            update={"diffusion_samples": n_samp}
        )
        reps = _pick_representatives(homologs, n_reps)
        self.log.info(
            "representatives=%d (of %d homologs), Boltz samples/homolog=%d",
            len(reps), len(homologs), n_samp,
        )

        records: List[PoseRecord] = []
        pose_table: List[dict] = []
        n_poses_total = 0
        for i, h in enumerate(reps):
            cx = predict_complex(
                f"hom_{i:03d}", h.sequence, ligand, cp_cfg,
                ctx.paths.structures / "representatives",
                backend=backend, dry_run=ctx.dry_run,
            )
            samples = cx.samples or []
            for s in samples:
                fp = complex_fingerprint(
                    cx.structure, s.ligand_atoms,
                    cutoff=cfg.contact_cutoff, k_nearest=cfg.k_nearest_residues,
                )
                records.append(
                    PoseRecord(
                        group_id=f"hom_{i:03d}",
                        fingerprint=fp,
                        msa_membership=1.0,
                        identity_to_target=float(h.identity),
                        # Boltz pose reliability -> SAMPLE WEIGHT (not a label)
                        pred_score=round(
                            s.metrics.get("confidence_score", cx.confidence), 4
                        ),
                    )
                )
                n_poses_total += 1
            pose_table.extend(pose_rows(f"hom_{i:03d}", cx))

        # Ensemble contact frequency (priority #1) + confidence-weighted edges,
        # from the WT Boltz ensemble -> edge-level dataset.
        econ = ensemble_contacts(
            wt.structure, wt.samples,
            cutoff=cfg.contact_cutoff,
            ligand_iptm=float(wt.metrics.get("ligand_iptm", 1.0)),
        )
        ctx.put("ensemble_contacts", econ)
        ctx.put("pose_dataset", pose_table)
        ctx.put("edge_dataset", edge_rows(econ))
        self.log.info(
            "WT pose consensus: %s | %d ensemble contacts (freq>=0.5: %d)",
            pose_consensus(wt.samples), len(econ),
            sum(1 for e in econ if e.contact_frequency >= 0.5),
        )

        sel_kw = dict(
            select_z=cfg.pose_select_mad_z,
            outlier_z=cfg.pose_outlier_mad_z,
            min_decoys_per_group=cfg.min_decoys_per_homolog,
            seed=ctx.config.seed,
            keep_alternative_band=cfg.keep_alternative_band,
            alternative_weight=cfg.alternative_weight,
            hard_decoys_per_group=cfg.hard_decoys_per_homolog,
        )
        sel = select_poses(records, **sel_kw)

        # subfamily-holdout validation: train without one homolog group,
        # check the model still ranks its held-out consensus poses above
        # decoys (guards against memorising / consensus circularity).
        holdout_auroc = self._subfamily_holdout(records, sel_kw, cfg)
        if holdout_auroc is not None:
            self.log.info("subfamily-holdout AUROC = %.3f", holdout_auroc)

        model = InteractionModel(
            cutoff=cfg.contact_cutoff, k_nearest=cfg.k_nearest_residues
        )
        model.fit(sel)
        model.save(ctx.paths.interaction_graphs / "interaction_model.json")
        ctx.put("interaction_model", model)

        # Per-pose fingerprint matrix CSV (additive, report-side artefact for
        # the HTML interaction-fingerprint heatmap). Failure-soft: a CSV-write
        # problem must not break model training above.
        _export_fingerprint_matrix(
            ctx.paths.ml_datasets,
            records,
            fingerprint_dim(cfg.k_nearest_residues),
        )

        ctx.persist_meta(
            "interaction_model",
            {
                "representatives": len(reps),
                "poses_total": n_poses_total,
                "train_rows": int(sel.X.shape[0]),
                "n_consensus": sel.n_positive,
                "n_alternative": sel.n_alternative,
                "n_outlier": sel.n_outlier,
                "n_decoy": sel.n_decoy,
                "n_hard_decoy": sel.n_hard_decoy,
                "subfamily_holdout_auroc": holdout_auroc,
                "model_kind": model.kind,
                "fp_dim": fingerprint_dim(cfg.k_nearest_residues),
            },
        )
        self.log.info(
            "trained %s on %d rows (consensus=%d, alt=%d, outlier=%d, "
            "decoy=%d, hard=%d)",
            model.kind, sel.X.shape[0], sel.n_positive, sel.n_alternative,
            sel.n_outlier, sel.n_decoy, sel.n_hard_decoy,
        )
        if sel.consensus.size:
            self.log.info(
                "family consensus interaction: %s",
                describe(sel.consensus, cfg.k_nearest_residues),
            )

    def _subfamily_holdout(self, records, sel_kw, cfg):
        """Hold out one homolog group; train on the rest; AUROC of the
        held-out consensus poses vs that group's decoys."""
        if not cfg.subfamily_holdout:
            return None
        groups = sorted({r.group_id for r in records})
        if len(groups) < 3:
            return None
        held = groups[-1]
        train_recs = [r for r in records if r.group_id != held]
        held_recs = [r for r in records if r.group_id == held]
        if not train_recs or not held_recs:
            return None
        sel = select_poses(train_recs, **sel_kw)
        m = InteractionModel(
            cutoff=cfg.contact_cutoff, k_nearest=cfg.k_nearest_residues
        ).fit(sel)
        held_sel = select_poses(held_recs, **{**sel_kw, "seed": sel_kw["seed"] + 1})
        if held_sel.X.shape[0] == 0 or len(set(held_sel.y.tolist())) < 2:
            return None
        from evoliez.ml.benchmark import _auroc

        scores = [m.score_vector(x) for x in held_sel.X]
        return _auroc(scores, [int(v) for v in held_sel.y.tolist()])

    def load(self, ctx: RunContext) -> bool:
        p = ctx.paths.interaction_graphs / "interaction_model.json"
        if not p.exists():
            return False
        ctx.put("interaction_model", InteractionModel.load(p))
        return True
