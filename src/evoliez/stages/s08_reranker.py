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
from evoliez.features.delta import WTDeltaCache, boltz_delta_features
from evoliez.ml.labels import assert_supervised_label_allowed
from evoliez.stages.base import Stage
from evoliez.stages.s07_mutation_gen import _RULE_POOL, _ligand_role
from evoliez.types import Candidate, Residue

_BOLTZ_SCALED = (
    "confidence_score", "ptm", "iptm", "ligand_iptm",
    "complex_plddt", "complex_iplddt",
)


def _approx_mutant_complex(cx, cand):
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
    # Cheap-run discovery: the conservation penalty (`- 0.8 * cons`)
    # was burying the Tishkov D222S/N/T/Q family because D222 is a
    # well-conserved NAD-binding-loop residue. But the user explicitly
    # marked D222 (and the other 16 NAD-binding residues) in
    # `known_binding_site` exactly BECAUSE those positions are
    # engineering-relevant. Adding `at_binding_site` lets the heuristic
    # and the xgboost path both lift binding-site mutations so they
    # actually reach the top-K cut and get real Boltz/MD evaluation.
    # 1.0 when the mutation's position is in ctx["known_binding_site"],
    # 0.0 otherwise. Multi-mutants: average across positions.
    "at_binding_site",
]


def _promote_binding_site_reservations(top, candidates, binding_site, n_reserve):
    """Guarantee `n_reserve` candidates per `known_binding_site` position
    survive the top_for_redocking cut, even if the heuristic score put them
    below. Cheap-run diagnosis: even with the +0.6 at_binding_site prior the
    Tishkov D222S/H/A/N/T/Q family was being cut by s08 (~conservation
    penalty), giving recall@10 = 0. This is the FLOOR (the prior was the
    lift); together they cover the recurring D222 miss.

    Logic: for each binding-site position with deficit, find the
    best-ranked tail candidate (assumed pre-sorted by ml_score desc)
    touching that position and swap it in for the lowest-ranked top entry
    that is NOT itself a binding-site reserver. Returns (new_top, n_promoted).
    No-op when n_reserve <= 0 or binding_site is empty.
    """
    if n_reserve <= 0 or not binding_site:
        return list(top), 0
    binding_site = {int(p) for p in binding_site}
    top = list(top)
    top_ids = {id(c) for c in top}
    in_top: Dict[int, int] = {p: 0 for p in binding_site}
    for c in top:
        for m in c.mutations:
            if m.position in binding_site:
                in_top[m.position] = in_top.get(m.position, 0) + 1
    promoted = 0
    for pos in sorted(binding_site):
        deficit = n_reserve - in_top.get(pos, 0)
        if deficit <= 0:
            continue
        for c in candidates:                              # pre-sorted desc
            if id(c) in top_ids:
                continue
            if any(m.position == pos for m in c.mutations):
                displ = next(
                    (x for x in sorted(
                        top, key=lambda x: x.scores.get("ml_score", 0.0))
                     if not any(m.position in binding_site
                                for m in x.mutations)),
                    None,
                )
                if displ is None:
                    break
                top.remove(displ)
                top_ids.discard(id(displ))
                top.append(c)
                top_ids.add(id(c))
                in_top[pos] = in_top.get(pos, 0) + 1
                promoted += 1
                deficit -= 1
                if deficit <= 0:
                    break
    if promoted:
        top.sort(key=lambda c: -c.scores.get("ml_score", 0.0))
    return top, promoted


class RerankerStage(Stage):
    name = "s08_reranker"

    def run(self, ctx: RunContext) -> None:
        cx = ctx.require("wt_complex")
        feats = ctx.require("position_features")
        contacts = ctx.require("contacts")
        candidates: List[Candidate] = ctx.require("candidates")
        catalytic = ctx.get("catalytic_positions", [])
        # Cheap-run discovery: surface user-declared binding-site
        # positions into the feature row so the reranker can lift them
        # past the conservation penalty (see _FEATURE_KEYS comment).
        binding_site = set(ctx.get("known_binding_site", []) or [])
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
        # GNN-fallback transparency (expert review #5): record what scored
        gnn_status = (
            "trained" if gnn_scorer is not None
            else ("heuristic_fallback" if gcfg.enabled else "disabled")
        )
        ctx.persist_meta("gnn_status", gnn_status)
        econ = ctx.get("ensemble_contacts", [])
        lig_imp = ctx.get("ligand_importance", {})

        # WT-side terms in boltz_delta_features (pocket pLDDT, contact
        # count, catalytic distances) are pure functions of the WT
        # complex - hoist them out of the candidate loop. Saves ~30
        # redundant pocket_plddt + contact-walk recomputations here, and
        # we hand the same cache to s08b too. Numerically identical to
        # the per-candidate path (same wt_cx, same cutoff).
        wt_delta_cache = WTDeltaCache.build(
            cx, catalytic_positions=catalytic
        )
        ctx.put("wt_delta_cache", wt_delta_cache)

        for cand in candidates:
            feat = self._features(
                cand, res_by_pos, pf_by_pos, nearest, atom_by_id,
                binding_site=binding_site,
            )

            # ONE lightweight approx-mutant build, reused for the family
            # interaction score, the WT-delta features and the GNN (was two
            # full deepcopies of the 471-residue complex per candidate).
            amc = _approx_mutant_complex(cx, cand)
            if imodel is not None:
                fam = imodel.score_complex(
                    amc.structure, amc.ligand.atoms,
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
                amc, cx, catalytic_positions=catalytic,
                wt_cache=wt_delta_cache,
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
        top = list(candidates[: rcfg.top_for_redocking])

        n_reserve = max(0, int(getattr(rcfg, "binding_site_reserved_per_position", 1)))
        top, promoted = _promote_binding_site_reservations(
            top, candidates, binding_site, n_reserve,
        )
        if promoted:
            self.log.info(
                "binding-site reserved-slots: promoted %d candidate(s) "
                "into top-%d (n_reserve=%d per position)",
                promoted, rcfg.top_for_redocking, n_reserve,
            )

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
    def _features(self, cand, res_by_pos, pf_by_pos, nearest, atom_by_id,
                  *, binding_site=None) -> dict:
        binding_site = binding_site or set()
        perms, cons, buried, dligand, gain = [], [], [], [], 0.0
        n_at_bs = 0
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
            if m.position in binding_site:
                n_at_bs += 1
        n = len(cand.mutations)
        return {
            "msa_permissiveness": float(cand.details.get("msa_permissiveness", 0.0)),
            "interaction_gain": round(gain, 4),
            "conservation": round(sum(cons) / max(1, len(cons)), 4),
            "n_mutations": n,
            "buried_fraction": round(sum(buried) / max(1, len(buried)), 4),
            "dist_to_ligand": round(min(dligand) if dligand else 99.0, 3),
            # Fraction of the candidate's mutated positions that the
            # user marked as known_binding_site. 1.0 for single-mutants
            # at a binding-site position, 0.0 otherwise; multi-mutants
            # get the mean. The heuristic adds +0.6 × this so the
            # conservation penalty (which is high for binding-loop
            # residues like PseFDH D222) doesn't bury the very mutations
            # the user is asking about.
            "at_binding_site": round(n_at_bs / max(1, n), 4),
        }

    def _score_heuristic(self, candidates: List[Candidate]) -> None:
        for c in candidates:
            f = c.details["features"]
            # Cheap-run PseFDH discovery: the conservation penalty
            # (`- 0.8 * (cons - 0.55)`) was pushing the Tishkov D222
            # family below the top_for_redocking cut because D222 is a
            # well-conserved NAD-binding-loop residue. Adding a +0.6
            # bonus for user-declared known_binding_site positions
            # neutralises that penalty for the residues the operator
            # explicitly asked the pipeline to mutate. Calibrated so a
            # binding-site mutation with cons=0.85 still scores higher
            # than a non-binding-site mutation at cons=0.55 with the
            # same other features. Universal — every enzyme card
            # declares known_binding_site.
            score = (
                1.5 * f.get("family_interaction_score", 0.5)
                + 1.2 * f["interaction_gain"]
                + 0.9 * f["msa_permissiveness"]
                + 0.6 * f.get("at_binding_site", 0.0)
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
        # P0.5: monotone constraints. Higher score must NOT be predicted when
        # a known-monotonic feature gets worse. Direction per feature:
        #   +1: higher value -> higher predicted score
        #    0: no constraint
        #   -1: higher value -> lower predicted score
        # Choices reflect domain knowledge:
        #   - interaction_gain, family_interaction_score, d_ligand_iptm,
        #     d_complex_iplddt, msa_permissiveness   -> +1 (more = better)
        #   - n_mutations, dist_to_ligand, conservation, d_complex_ipde,
        #     d_key_distance, specificity_divergence -> -1 (more = worse)
        monotone = {
            "msa_permissiveness":          +1,
            "interaction_gain":            +1,
            "conservation":                -1,
            "n_mutations":                 -1,
            "buried_fraction":              0,
            "dist_to_ligand":              -1,
            "family_interaction_score":    +1,
            "d_ligand_iptm":               +1,
            "d_complex_iplddt":            +1,
            "d_complex_ipde":              -1,
            "d_key_distance":              -1,
            "d_pocket_plddt":              +1,
            "specificity_divergence":      -1,
            # User-declared binding-site positions are engineering
            # targets, not residues to avoid. +1: higher → higher score
            # so the xgboost path agrees with the heuristic boost.
            "at_binding_site":             +1,
        }
        mono_tuple = tuple(monotone[k] for k in _FEATURE_KEYS)
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            monotone_constraints=str(mono_tuple),
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
