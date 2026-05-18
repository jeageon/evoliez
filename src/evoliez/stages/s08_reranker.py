"""Stage 08 - family-specific reranker (spec section 13).

Without experimental labels: a transparent weak-supervision heuristic over
evolutionary / interaction / structural features. With labels (config
``reranking.use_experimental_labels`` + ``input.experimental_dataset``):
trains XGBoost (falls back to the heuristic if xgboost is unavailable).
"""

from __future__ import annotations

import copy
import csv
from typing import Dict, List

from evoliez.context import RunContext
from evoliez.features.delta import boltz_delta_features
from evoliez.ml.labels import assert_supervised_label_allowed
from evoliez.stages.base import Stage
from evoliez.stages.s07_mutation_gen import _RULE_POOL, _ligand_role
from evoliez.types import Candidate

_BOLTZ_SCALED = (
    "confidence_score", "ptm", "iptm", "ligand_iptm",
    "complex_plddt", "complex_iplddt",
)


def _approx_mutant_complex(cx, cand):
    """Proxy mutant complex for WT-delta features WITHOUT a per-candidate
    Boltz run: WT geometry/metrics degraded by mutation disruptiveness
    (conservation x #mutations). A real per-mutant Boltz re-eval is the
    server-side enhancement; deltas remain FEATURES, never labels."""
    mc = copy.deepcopy(cx)
    by = {r.index: r for r in mc.structure.residues}
    seq = list(mc.structure.sequence)
    for m in cand.mutations:
        r = by.get(m.position)
        if r is not None:
            r.aa = m.mut
            if 0 < m.position <= len(seq):
                seq[m.position - 1] = m.mut
    mc.structure.sequence = "".join(seq)
    f = cand.details.get("features", {})
    inst = min(1.0, 0.5 * f.get("conservation", 0.5)
               + 0.15 * (len(cand.mutations) - 1))
    for k in _BOLTZ_SCALED:
        if k in mc.metrics:
            mc.metrics[k] = round(mc.metrics[k] * (1.0 - 0.30 * inst), 4)
    for k in ("complex_pde", "complex_ipde"):
        if k in mc.metrics:
            mc.metrics[k] = round(mc.metrics[k] * (1.0 + 0.40 * inst), 4)
    if "affinity_pred_value" in mc.metrics:
        mc.metrics["affinity_pred_value"] = round(
            mc.metrics["affinity_pred_value"] + 0.6 * inst, 4
        )
    return mc

_FEATURE_KEYS = [
    "msa_permissiveness",
    "interaction_gain",
    "conservation",
    "n_mutations",
    "buried_fraction",
    "dist_to_ligand",
    "family_interaction_score",
    "d_ligand_iptm",
    "d_complex_iplddt",
    "d_complex_ipde",
    "d_key_distance",
    "d_pocket_plddt",
    "specificity_divergence",
]


class RerankerStage(Stage):
    name = "s08_reranker"

    def run(self, ctx: RunContext) -> None:
        cx = ctx.require("wt_complex")
        feats = ctx.require("position_features")
        contacts = ctx.require("contacts")
        candidates: List[Candidate] = ctx.require("candidates")
        catalytic = ctx.get("catalytic_positions", [])
        rcfg = ctx.config.reranking

        res_by_pos = {r.index: r for r in cx.structure.residues}
        pf_by_pos = {
            f.target_position: f for f in feats if f.target_position is not None
        }
        atom_by_id = {a.id: a for a in cx.ligand.atoms}
        nearest: Dict[int, object] = {}
        for c in sorted(contacts, key=lambda c: c.distance):
            nearest.setdefault(c.residue_index, c)

        # family interaction-geometry model (stage s06b); None if disabled
        imodel = ctx.get("interaction_model")
        from evoliez.stages.s09_nonmd_validation import _mutant_complex

        # optional server-grade EvoLigand-GNN (torch + checkpoint required)
        gnn_scorer = None
        gcfg = ctx.config.gnn
        if gcfg.enabled:
            from pathlib import Path

            from evoliez.ml.gnn_scorer import EvoLigandGNNScorer

            gnn_scorer = EvoLigandGNNScorer.load(
                Path(ctx.config.project.output_dir) / gcfg.checkpoint
            )
            if gnn_scorer is None:
                self.log.info(
                    "gnn.enabled but no usable checkpoint/torch; "
                    "falling back to heuristic family model"
                )
        econ = ctx.get("ensemble_contacts", [])
        lig_imp = ctx.get("ligand_importance", {})

        for cand in candidates:
            feat = self._features(cand, res_by_pos, pf_by_pos, nearest, atom_by_id)

            if imodel is not None:
                mc = _mutant_complex(cx, cand)
                fam = imodel.score_complex(
                    mc.structure, mc.ligand.atoms,
                    msa_membership=0.0,  # designed mutant, not an MSA homolog
                    identity_to_target=1.0,
                    pred_score=round(-cand.scores.get("docking_score", -7.0)
                                     + cx.confidence, 4),
                )
            else:
                fam = 0.5  # neutral when the interaction model is disabled
            feat["family_interaction_score"] = fam

            # WT - mutant Boltz delta features (proxy; FEATURES, not labels)
            delta = boltz_delta_features(
                _approx_mutant_complex(cx, cand), cx,
                catalytic_positions=catalytic,
            )
            cand.details["delta"] = delta
            cand.details["boltz_delta_source"] = "proxy"  # s08b may upgrade
            for dk in ("d_ligand_iptm", "d_complex_iplddt", "d_complex_ipde",
                       "d_key_distance", "d_pocket_plddt"):
                feat[dk] = delta.get(dk, 0.0)

            if gnn_scorer is not None:
                gscore = gnn_scorer.score_complex(
                    _approx_mutant_complex(cx, cand), feats, econ,
                    catalytic_positions=catalytic,
                )
                cand.scores["gnn_score"] = gscore
                feat["gnn_score"] = gscore

            # ligand-atom importance weighting (user §2): scale interaction
            # gain by how catalytically important the contacted atom is.
            imps, specs = [], []
            for m in cand.mutations:
                c = nearest.get(m.position)
                if c and lig_imp:
                    imps.append(lig_imp.get(c.ligand_atom_id, 0.4))
                pf = pf_by_pos.get(m.position)
                if pf:
                    specs.append(pf.specificity_divergence)
            imp_w = sum(imps) / len(imps) if imps else 1.0
            feat["interaction_gain"] = round(
                feat["interaction_gain"] * (0.5 + imp_w), 4
            )
            feat["specificity_divergence"] = round(
                sum(specs) / len(specs), 4
            ) if specs else 0.0

            cand.details["features"] = feat
            cand.scores["family_interaction_score"] = fam
            cand.scores["specificity_divergence"] = feat[
                "specificity_divergence"
            ]
            cand.scores["msa_permissiveness"] = feat["msa_permissiveness"]
            cand.scores["interaction_gain"] = feat["interaction_gain"]
            cand.scores["conservation_penalty"] = round(
                max(0.0, feat["conservation"] - 0.5) * feat["n_mutations"], 4
            )

        labels = self._load_labels(ctx) if rcfg.use_experimental_labels else None
        if labels and rcfg.model == "xgboost":
            self._score_xgboost(candidates, labels)
        else:
            self._score_heuristic(candidates)

        candidates.sort(key=lambda c: -c.scores.get("ml_score", 0.0))
        top = candidates[: rcfg.top_for_redocking]
        ctx.put("candidates", candidates)
        ctx.put("redock_candidates", top)
        ctx.persist_meta("reranker_model", rcfg.model if labels else "heuristic")
        ctx.persist_meta("n_after_rerank", len(top))
        self.log.info(
            "reranked %d candidates (%s); %d advance to redocking",
            len(candidates),
            "xgboost" if labels and rcfg.model == "xgboost" else "heuristic",
            len(top),
        )

    # ------------------------------------------------------------------ #
    def _features(self, cand, res_by_pos, pf_by_pos, nearest, atom_by_id) -> dict:
        perms, cons, buried, dligand, gain = [], [], [], [], 0.0
        for m in cand.mutations:
            pf = pf_by_pos.get(m.position)
            r = res_by_pos.get(m.position)
            if pf:
                cons.append(pf.conservation_score)
            if r:
                buried.append(1.0 - r.sasa)
            c = nearest.get(m.position)
            if c:
                dligand.append(c.distance)
                atom = atom_by_id.get(c.ligand_atom_id)
                if atom is not None and m.mut in _RULE_POOL[_ligand_role(atom)]:
                    gain += max(0.0, 1.0 - c.distance / 5.0)
        n = len(cand.mutations)
        return {
            "msa_permissiveness": float(cand.details.get("msa_permissiveness", 0.0)),
            "interaction_gain": round(gain, 4),
            "conservation": round(sum(cons) / max(1, len(cons)), 4),
            "n_mutations": n,
            "buried_fraction": round(sum(buried) / max(1, len(buried)), 4),
            "dist_to_ligand": round(min(dligand) if dligand else 99.0, 3),
        }

    def _score_heuristic(self, candidates: List[Candidate]) -> None:
        for c in candidates:
            f = c.details["features"]
            score = (
                1.5 * f.get("family_interaction_score", 0.5)
                + 1.2 * f["interaction_gain"]
                + 0.9 * f["msa_permissiveness"]
                - 0.8 * max(0.0, f["conservation"] - 0.55)
                - 0.15 * (f["n_mutations"] - 1)
                - 0.05 * max(0.0, f["dist_to_ligand"] - 6.0)
            )
            c.scores["ml_score"] = round(float(score), 4)

    def _score_xgboost(self, candidates: List[Candidate], labels: Dict[str, float]) -> None:
        try:
            import numpy as np
            import xgboost as xgb
        except Exception:
            self.log.warning("xgboost unavailable; using heuristic reranker")
            return self._score_heuristic(candidates)
        train = [(c, labels[c.mutation_str]) for c in candidates
                 if c.mutation_str in labels]
        if len(train) < 8:
            self.log.warning("too few labelled examples (%d); heuristic", len(train))
            return self._score_heuristic(candidates)
        X = np.array([[c.details["features"][k] for k in _FEATURE_KEYS]
                      for c, _ in train], dtype=float)
        y = np.array([v for _, v in train], dtype=float)
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05
        )
        model.fit(X, y)
        allX = np.array(
            [[c.details["features"][k] for k in _FEATURE_KEYS] for c in candidates],
            dtype=float,
        )
        preds = model.predict(allX)
        for c, p in zip(candidates, preds):
            c.scores["ml_score"] = round(float(p), 4)

    def _load_labels(self, ctx: RunContext) -> Dict[str, float] | None:
        path = ctx.config.input.experimental_dataset
        if not path:
            return None
        labels: Dict[str, float] = {}
        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                label_col = next(
                    (c for c in (reader.fieldnames or [])
                     if c.lower() in ("activity", "relative_activity",
                                      "kcat", "km", "kcat_km",
                                      "thermostability")),
                    "activity",
                )
                # policy: a Boltz-derived column can never be the label
                assert_supervised_label_allowed(label_col)
                for row in reader:
                    mut = row.get("mutation") or row.get("mutations")
                    val = row.get(label_col)
                    if mut and val:
                        labels[mut.strip()] = float(val)
        except Exception as exc:
            self.log.warning("could not read experimental dataset: %s", exc)
            return None
        self.log.info("loaded %d experimental labels", len(labels))
        return labels
