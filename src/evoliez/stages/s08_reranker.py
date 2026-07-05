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
from evoliez.types import Candidate, Residue

_BOLTZ_SCALED = (
    "confidence_score", "ptm", "iptm", "ligand_iptm",
    "complex_plddt", "complex_iplddt",
)


def _approx_mutant_complex(cx, cand, *, conservation=None):
    """Proxy mutant complex for WT-delta + family-interaction features
    WITHOUT a per-candidate Boltz run AND without a full deepcopy.

    WT geometry is unchanged (coords identical) - only the mutated residue
    identities + the sequence change, and Boltz metrics are degraded by
    mutation disruptiveness (conservation x #mutations). Every downstream
    consumer (fingerprint / boltz_delta_features / gnn) only READS structure
    + ligand, so unchanged residues and the ligand are shared with WT by
    reference; only the few mutated Residues and the metrics dict are copied.
    This turns an O(471) deepcopy (x candidates) into O(#mutations). Deltas
    remain FEATURES, never labels."""
    muts = {m.position: m.mut for m in cand.mutations}
    seq = list(cx.structure.sequence)
    new_res = []
    for r in cx.structure.residues:
        mt = muts.get(r.index)
        if mt is not None:
            r = Residue(
                index=r.index, aa=mt, ca=r.ca,
                sidechain_centroid=r.sidechain_centroid,
                secondary_structure=r.secondary_structure,
                sasa=r.sasa, plddt=r.plddt,
            )
            if 0 < r.index <= len(seq):
                seq[r.index - 1] = mt
        new_res.append(r)
    st = copy.copy(cx.structure)        # shallow: don't deepcopy 471 residues
    st.residues = new_res
    st.sequence = "".join(seq)
    mc = copy.copy(cx)                  # shallow: ligand/samples shared (RO)
    mc.structure = st
    mc.metrics = dict(cx.metrics)       # copy so scaling can't touch WT
    # disruptiveness needs the REAL conservation. s08 builds this proxy before
    # cand.details['features'] is populated, so the caller passes conservation
    # explicitly; fall back to features only when it is already set (s11).
    cons = (conservation if conservation is not None
            else cand.details.get("features", {}).get("conservation", 0.5))
    inst = min(1.0, 0.5 * cons + 0.15 * (len(cand.mutations) - 1))
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
    # family_interaction_score is intentionally NOT a reranker feature: it has a
    # dedicated weighted contribution in ranking.score (avoids double-counting).
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
        # WT disorder track, computed ONCE on the shared backbone and reused for
        # every candidate - mirrors how s11 builds the training graphs, so the
        # inference disorder node features match training (else they are 0).
        gnn_disorder = None
        if gnn_scorer is not None and gcfg.use_disorder:
            from evoliez.adapters.disorder import predict_disorder
            gnn_disorder = predict_disorder(
                cx.structure.sequence, ctx.paths.root / "datasets",
                backend=ctx.config.backend_for("s06b_interaction"),
                dry_run=ctx.dry_run,
            )
        # GNN-fallback transparency (expert review #5): record what scored
        gnn_status = (
            "trained" if gnn_scorer is not None
            else ("heuristic_fallback" if gcfg.enabled else "disabled")
        )
        ctx.persist_meta("gnn_status", gnn_status)
        econ = ctx.get("ensemble_contacts", [])
        lig_imp = ctx.get("ligand_importance", {})

        for cand in candidates:
            feat = self._features(cand, res_by_pos, pf_by_pos, nearest, atom_by_id)

            # ONE lightweight approx-mutant build, reused for the family
            # interaction score, the WT-delta features and the GNN (was two
            # full deepcopies of the 471-residue complex per candidate).
            amc = _approx_mutant_complex(
                cx, cand, conservation=feat["conservation"]
            )
            if imodel is not None:
                # fingerprint-only: the model scores the mutant's interaction
                # geometry. (No msa_membership/pred_score scalars - those leaked
                # the label in training and were fed out-of-distribution here.)
                fam = imodel.score_complex(amc.structure, amc.ligand.atoms)
            else:
                fam = 0.5  # neutral when the interaction model is disabled
            feat["family_interaction_score"] = fam

            # WT - mutant Boltz delta features (proxy; FEATURES, not labels)
            delta = boltz_delta_features(
                amc, cx, catalytic_positions=catalytic,
            )
            cand.details["delta"] = delta
            cand.details["boltz_delta_source"] = "proxy"  # s08b may upgrade
            for dk in ("d_ligand_iptm", "d_complex_iplddt", "d_complex_ipde",
                       "d_key_distance", "d_pocket_plddt"):
                feat[dk] = delta.get(dk, 0.0)

            if gnn_scorer is not None:
                gscore = gnn_scorer.score_complex(
                    amc, feats, econ,
                    catalytic_positions=catalytic,
                    disorder=gnn_disorder,
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
        # ROADMAP_V3 B1 — ML is a PRIOR, not a hard filter. The scalar ml_score top-N cut
        # below is the true bottleneck: anything dropped HERE never gets a real structure
        # (s08b) or MD (s09), so the downstream multi-lane (s08b fold queue / s09 MD
        # shortlist) can only re-select from candidates that ALREADY survived this cut —
        # an ML false negative is unrecoverable. When selection_lanes is enabled, widen
        # the redock set to a UNION of lanes over the FULL candidate set so a low-ML /
        # diverse / mechanism-seed candidate reaches s08b. Only ml/diversity/low-ML-control
        # are feasible here (stability/geometry are s09 products, not yet computed) — same
        # feasible lanes as s08b's fold queue. Default off = the existing scalar cut.
        _sl = getattr(ctx.config, "selection_lanes", None)
        if _sl is not None and getattr(_sl, "enabled", False) and candidates:
            from evoliez.ranking.multi_lane import (
                LaneConfig, lane_counts, select_multi_lane)
            # The lanes must AUGMENT the scalar redock set, never SHRINK it: the ml_high
            # lane is the FULL top_for_redocking (the pre-lane baseline), and diversity +
            # low_ml_control ADD false-negative probes on top. (A small from_ml_high here
            # would collapse the whole downstream funnel — s08b/s09/final library all read
            # this set — to a few dozen candidates, starving the wet-lab plate.) The small
            # per-lane budgets in selection_lanes apply at the EXPENSIVE s08b/s09 stages,
            # not this cheap redock gate.
            _redock = LaneConfig(
                enabled=True,
                from_ml_high=max(rcfg.top_for_redocking, _sl.from_ml_high),
                from_stability_high=0, from_geometry_high=0,
                # stability/geometry are s09 products (inert here) -> repurpose their config
                # slots to size the s08-COMPUTABLE evolutionary (MSA permissiveness) + ligand
                # competence (interaction_gain) lanes, so a low-ML but evolutionarily-tolerated /
                # ligand-competent candidate reaches s08b. No new Config field (purge-safe).
                from_evolutionary_high=_sl.from_stability_high,
                from_ligand_high=_sl.from_geometry_high,
                from_diversity=_sl.from_diversity,
                low_ml_controls=_sl.low_ml_controls)
            top = select_multi_lane(candidates, _redock)
            self.log.info(
                "s08 redock queue via MULTI-LANE: %d candidate(s) %s "
                "(baseline top_for_redocking=%d + probes)",
                len(top), lane_counts(top), rcfg.top_for_redocking)
            # The low_ml_control lane is the explicit ML-false-negative safety net. It only
            # MATTERS when candidates are actually EXCLUDED from the union (len(top) <
            # len(candidates)): if the union already contains every candidate, an empty
            # low_ml_control is harmless (everything is redocked anyway). Fail loud only when
            # candidates were cut yet the low-ML probe produced none -- then the net vanished.
            if (_sl.low_ml_controls > 0
                    and lane_counts(top).get("low_ml_control", 0) == 0
                    and len(top) < len(candidates)):
                raise RuntimeError(
                    "s08 selection_lanes: low_ml_control lane is EMPTY while %d candidate(s) "
                    "were EXCLUDED from the redock union -- the ML false-negative safety net "
                    "vanished. Check lane sizing (low_ml_controls / from_ml_high)."
                    % (len(candidates) - len(top)))
            # Record the ACTUAL s08 lane->quota mapping (ROADMAP_V5 step 5): at s08 the
            # stability/geometry config slots are REPURPOSED to size the s08-computable
            # evolutionary / ligand-competence lanes (stability/geometry are s09 products). Made
            # explicit here so provenance is unambiguous about which config field sized which lane.
            ctx.persist_meta("s08_lane_quota_mapping", {
                "note": ("s08 repurposes from_stability_high -> evolutionary_high and "
                         "from_geometry_high -> ligand_competence_high because stability/geometry "
                         "are not yet computable at s08; a future PR splits these into dedicated "
                         "config fields."),
                "ml_high": _redock.from_ml_high,
                "evolutionary_high__from_stability_high": _redock.from_evolutionary_high,
                "ligand_competence_high__from_geometry_high": _redock.from_ligand_high,
                "diversity": _redock.from_diversity,
                "low_ml_control": _redock.low_ml_controls,
                "selected_by_lane": lane_counts(top),
            })
        else:
            top = candidates[: rcfg.top_for_redocking]
        # Persist per-candidate reranker scores + features (the s08 report reads this;
        # previously ctx.put in-memory only). The top candidates advance to
        # the s08b real-Boltz fold + s09 validation.
        import json as _json
        _prov = ctx.paths.reports / "provenance"
        _prov.mkdir(parents=True, exist_ok=True)
        _top_ids = {c.candidate_id for c in top}
        (_prov / "reranked_candidates.json").write_text(_json.dumps([{
            "rank": i + 1,
            "candidate_id": c.candidate_id,
            "mutation_string": ";".join(
                f"{m.wt}{m.position}{m.mut}" for m in c.mutations),
            "n_mutations": len(c.mutations),
            "generator": c.details.get("generator") or c.details.get("source"),
            "ml_score": c.scores.get("ml_score"),
            "family_interaction_score": c.scores.get("family_interaction_score"),
            "interaction_gain": c.scores.get("interaction_gain"),
            "specificity_divergence": c.scores.get("specificity_divergence"),
            "msa_permissiveness": c.scores.get("msa_permissiveness"),
            "conservation_penalty": c.scores.get("conservation_penalty"),
            "advanced_to_fold": c.candidate_id in _top_ids,
            "selection_lane": c.details.get("selection_lane"),
        } for i, c in enumerate(candidates)], indent=2, default=str))
        ctx.put("candidates", candidates)
        ctx.put("redock_candidates", top)
        ctx.persist_meta("reranker_model", rcfg.model if labels else "heuristic")
        ctx.persist_meta("n_after_rerank", len(top))
        try:  # auto-generate the s08 reranker HTML report (graceful on failure)
            from evoliez.io.rerank_report import write_rerank_report
            write_rerank_report(ctx.paths.root)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("s08 report generation skipped (%s)", exc)
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
            # family_interaction_score is NOT folded in here: it has its own
            # dedicated `family_interaction` contribution in ranking.score with
            # weight w.family_interaction. Including it both here (via ml_score
            # -> ml_mutation) and there double-counted it to ~2.5x the
            # documented weight and broke the additive decomposition.
            score = (
                1.2 * f["interaction_gain"]
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

    # supervised-target preference when the assay schema carries several label types
    _LABEL_TYPE_PREFERENCE = (
        "activity", "kinetics", "substrate_conversion", "product_formation",
        "stability", "specificity",
    )

    def _load_labels(self, ctx: RunContext) -> Dict[str, float] | None:
        path = ctx.config.input.experimental_dataset
        if not path:
            return None
        from pathlib import Path as _Path
        labels: Dict[str, float] = {}
        try:
            with open(path, newline="") as fh:
                cols = set(csv.DictReader(fh).fieldnames or [])

            # ROADMAP_V3 ML2 — PROVENANCE-based guard. A column NAME ("activity") is not
            # proof of an experimental measurement: a computed/predicted value dropped into
            # an `activity` column would pass the name-only guard and silently train an
            # "activity predictor". When the dataset is the long-format assay schema
            # (mutation,label_type,value,source), require source in {wetlab,literature}
            # via assay_label.supervised_targets — a computed_weak label can NEVER be a
            # supervised target — and never mix label types into one target.
            if {"label_type", "value", "source"} <= cols:
                from evoliez.ml.assay_label import (
                    load_assay_labels, supervised_targets)
                all_labels = load_assay_labels(_Path(path))
                chosen = next(
                    (lt for lt in self._LABEL_TYPE_PREFERENCE
                     if any(l.label_type == lt and l.is_supervised_source
                            for l in all_labels)), None)
                labels = supervised_targets(all_labels, label_type=chosen)
                dropped = sum(1 for l in all_labels if not l.is_supervised_source)
                if dropped:
                    self.log.warning(
                        "ML2: excluded %d computed_weak label(s) from supervised "
                        "training (a computed label is a feature/prior, never a target)",
                        dropped)
                if chosen:
                    self.log.info("supervised label_type=%r (provenance-gated)", chosen)
            else:
                # legacy wide format (one row per variant, a named readout column). Keep
                # the name-based Boltz-derived guard; when a `source` column is present,
                # honour it so a computed_weak row can't sneak in as a target.
                with open(path, newline="") as fh:
                    reader = csv.DictReader(fh)
                    has_source = "source" in (reader.fieldnames or [])
                    label_col = next(
                        (c for c in (reader.fieldnames or [])
                         if c.lower() in ("activity", "relative_activity",
                                          "kcat", "km", "kcat_km",
                                          "thermostability")),
                        "activity",
                    )
                    assert_supervised_label_allowed(label_col)
                    for row in reader:
                        if has_source and (row.get("source") or "").strip().lower() \
                                not in ("wetlab", "literature"):
                            continue  # provenance says not a supervised source
                        mut = row.get("mutation") or row.get("mutations")
                        val = row.get(label_col)
                        if mut and val:
                            labels[mut.strip()] = float(val)
        except Exception as exc:
            self.log.warning("could not read experimental dataset: %s", exc)
            return None
        self.log.info("loaded %d experimental labels", len(labels))
        return labels
