"""Stage 07 - mutation generation (spec section 12).

A multi-strategy generator, layered rather than single-tool:
  * chemistry rules over the CONTACT ENSEMBLE at each position (distance-weighted
    ligand-atom roles, not just the nearest atom),
  * GATED MSA sampling (drop gappy columns; require the family tolerates the
    residue; flag subfamily/ESM signal at generation, not only downstream),
  * LigandMPNN ligand-aware design, VALIDATED (designable-only, never
    catalytic/fixed, bounded edits per design),
  * a FuncLib-style MULTI-POINT active-site library built from the union of the
    single-substitution pools (2..N combinations on the most ligand-central
    positions, where active-site epistasis lives).
Per-generator QUOTAS stop the first generator from monopolising max_candidates.
Outputs deduplicated :class:`Candidate` objects with rationale metadata.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from evoliez.adapters.base import het_chains_in_pdb, parse_pdb_het_chain
from evoliez.context import RunContext
from evoliez.features.evolutionary import permissiveness
from evoliez.stages.base import Stage
from evoliez.types import Candidate, Mutation
from evoliez.utils.seeds import derive_seed
from evoliez.adapters.ligandmpnn import design_sequences

# spec 12.4 chemistry-aware pools keyed by a contacted ligand atom's role
_RULE_POOL = {
    "anion": "KRH",         # phosphate / carboxylate
    "cation": "DE",
    "aromatic": "FYWH",
    "hydrophobic": "LIVMF",
    "acceptor": "STNQYKR",  # ligand acceptor -> donor side chains
    "donor": "STNQDE",      # ligand donor -> acceptor side chains
}
# default per-generator share of max_candidates (LigandMPNN + multipoint favoured)
_DEFAULT_QUOTA = {
    "ligandmpnn": 0.40, "multipoint": 0.25, "chemistry_rules": 0.20,
    "msa_sampler": 0.15,
}

# --------------------------------------------------------------------------- #
# Safety-filter physicochemistry (GENERIC — residue identity only, no protein /
# ligand identity is assumed). One coarse class per residue; a substitution that
# JUMPS class (e.g. charge flip, polar<->hydrophobic, tiny<->bulky) near a
# catalytic residue is the "radical" 2-shell substitution the reviewer flags.
_AA_CLASS = {
    "D": "neg", "E": "neg",
    "K": "pos", "R": "pos", "H": "pos",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar",
    "Y": "polar", "C": "polar", "W": "polar",
    "A": "hydrophobic", "V": "hydrophobic", "L": "hydrophobic",
    "I": "hydrophobic", "M": "hydrophobic", "F": "hydrophobic",
    "G": "special", "P": "special",
}
# Coarse side-chain volume (Å^3, Zamyatnin) to detect a drastic size change.
_AA_VOLUME = {
    "G": 60.1, "A": 88.6, "S": 89.0, "C": 108.5, "D": 111.1, "P": 112.7,
    "N": 114.1, "T": 116.1, "E": 138.4, "V": 140.0, "Q": 143.8, "H": 153.2,
    "M": 162.9, "I": 166.7, "L": 166.7, "K": 168.6, "R": 173.4, "F": 189.9,
    "Y": 193.6, "W": 227.8,
}
_HELIX_BREAKERS = {"G", "P"}          # introduce-only structural-risk residues
_DISULFIDE_RISK = {"C"}              # free-cysteine / mispairing risk
# A substitution is "radical" when it flips class, or its volume changes by more
# than this many Å^3 (a large steric change in a packed active site).
_RADICAL_VOLUME_DELTA = 50.0
_CATALYTIC_SHELL_SEQ = 2             # |Δresidue index| <= this == 2nd shell
_CATALYTIC_SHELL_ANGSTROM = 6.0     # OR centroid within this of a catalytic res


def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _min_dist(point: Tuple[float, float, float], cloud) -> Optional[float]:
    """Min Euclidean distance from ``point`` to any coord in ``cloud`` (a list of
    LigandAtom or (x,y,z)). ``None`` when the cloud is empty (caller records a
    blank rather than a fabricated 0)."""
    best: Optional[float] = None
    for c in cloud:
        coord = getattr(c, "coord", c)
        d = _dist(point, coord)
        if best is None or d < best:
            best = d
    return best


def _is_radical_substitution(wt: str, mut: str) -> bool:
    """True iff the WT->mut substitution flips physicochemical class or makes a
    large side-chain-volume jump. Used only to grade substitutions NEAR a
    catalytic residue (the 2-shell risk)."""
    if _AA_CLASS.get(wt) != _AA_CLASS.get(mut):
        return True
    vol = abs(_AA_VOLUME.get(mut, 0.0) - _AA_VOLUME.get(wt, 0.0))
    return vol > _RADICAL_VOLUME_DELTA


def _ligand_role(atom) -> str:
    if atom.formal_charge < 0:
        return "anion"
    if atom.formal_charge > 0:
        return "cation"
    if atom.aromatic:
        return "aromatic"
    if atom.is_acceptor:
        return "acceptor"
    if atom.is_donor:
        return "donor"
    return "hydrophobic"


_Pool = List[Tuple[List[Mutation], dict]]


class MutationGenStage(Stage):
    name = "s07_mutation_gen"

    def run(self, ctx: RunContext) -> None:
        cx = ctx.require("wt_complex")
        feats = ctx.require("position_features")
        designable = list(ctx.require("designable_positions"))
        contacts = ctx.require("contacts")
        catalytic = set(ctx.get("catalytic_positions", []) or [])
        fixed = set(ctx.get("fixed_positions", []) or [])
        mgcfg = ctx.config.mutation_generation

        res = {r.index: r for r in cx.structure.residues}
        pf = {f.target_position: f for f in feats if f.target_position is not None}
        atom = {a.id: a for a in cx.ligand.atoms}
        by_pos_contacts: Dict[int, list] = {}
        for c in contacts:
            by_pos_contacts.setdefault(c.residue_index, []).append(c)
        nearest = {p: min(c.distance for c in cs)
                   for p, cs in by_pos_contacts.items()}

        # --- per-generator single-substitution pools (NO global cap here) --- #
        pools: Dict[str, _Pool] = {}
        if "chemistry_rules" in mgcfg.methods:
            pools["chemistry_rules"] = self._chemistry(
                designable, res, by_pos_contacts, atom)
        if "msa_sampler" in mgcfg.methods:
            pools["msa_sampler"] = self._msa(designable, res, pf, mgcfg)
        if "ligandmpnn" in mgcfg.methods:
            pools["ligandmpnn"] = self._ligandmpnn(
                ctx, cx, designable, catalytic, fixed, res, mgcfg)

        # allowed-AA pool per position from EVERY proposed substitution
        allowed: Dict[int, set] = {}
        for gen_pool in pools.values():
            for muts, _ in gen_pool:
                for m in muts:
                    allowed.setdefault(m.position, set()).add(m.mut)
        if mgcfg.multipoint:
            pools["multipoint"] = self._multipoint(allowed, res, nearest, mgcfg)

        # --- quota-bounded assembly + dedup + trim --------------------------- #
        candidates = self._assemble(pools, mgcfg)

        # CROSS-GENERATOR protected ASSERT (reviewer requirement): the generators
        # already exclude catalytic/fixed, so this is the independent check that
        # NO assembled candidate touches a protected position. Raises on any
        # violation (it must always pass).
        self._assert_no_protected(candidates, catalytic, fixed)

        # MSA permissiveness for downstream scoring
        for cand in candidates:
            perms = [permissiveness(pf.get(m.position), m.mut)
                     for m in cand.mutations if pf.get(m.position)]
            cand.details["msa_permissiveness"] = round(
                sum(perms) / max(1, len(perms)), 4)

        # --- per-candidate STRUCTURAL distances + SAFETY flags --------------- #
        # Coordinate sources are read GENERICALLY from the s04 complex in ctx:
        # residue centroids + design-ligand atoms from `cx`; each extra ligand's
        # placed pose from the WT-complex PDB chains (design ligand first, then
        # extras in input order — same chain assignment as s04/s05).
        res_coord = self._residue_coords(res)
        design_atoms = list(cx.ligand.atoms)
        extra_clouds = self._extra_ligand_clouds(ctx, cx)
        cat_coords = [res_coord[p] for p in sorted(catalytic) if p in res_coord]
        prov_feats = {
            c.candidate_id: self._candidate_safety_features(
                c, res, res_coord, design_atoms, extra_clouds, cat_coords,
                catalytic, fixed, mgcfg)
            for c in candidates}

        # Optionally DROP (vs only flag) risk-flagged candidates from the shipped
        # set. Default = flag-only (keep everything; downstream prunes). A drop
        # still records the full pool in the provenance table below.
        all_candidates = list(candidates)
        if mgcfg.risk_filter:
            candidates = [c for c in candidates
                          if not prov_feats[c.candidate_id]["risk_flag"]]
            n_dropped = len(all_candidates) - len(candidates)
            if n_dropped:
                self.log.info(
                    "risk_filter on: dropped %d risk-flagged candidate(s) from "
                    "the shipped set (still recorded in provenance)", n_dropped)
            if not candidates:
                raise RuntimeError(
                    "risk_filter dropped EVERY candidate - relax the filter or "
                    "review the risk flags in generated_candidates.csv")

        ctx.put("candidates", candidates)
        ctx.persist_meta("n_candidates_generated", len(candidates))
        # Durable per-candidate GENERATED-provenance table (paper methods): the
        # FULL pre-attrition pool with each candidate's generator + whatever
        # rationale features that generator recorded, NOW also carrying the
        # per-mutation structural distances + safety flags. Written here (not at
        # s11) because downstream stages prune the pool, so this is the only
        # place the complete generated set exists. The funnel report (s11) reads
        # it back. Records the full pre-filter pool so a dropped risky candidate
        # is still auditable.
        self._write_generated_provenance(ctx, all_candidates, prov_feats)
        # Reviewer-facing BUDGET TIERS for the s08-s10 cost (single / multipoint /
        # risky, top-N each by score). Separate files; ctx `candidates` stays the
        # full set so downstream pruning is unchanged.
        self._write_budget_tiers(ctx, all_candidates, prov_feats, mgcfg)
        counts: Dict[str, int] = {}
        for c in candidates:
            counts[c.generator] = counts.get(c.generator, 0) + 1
        self.log.info("generated %d candidates by generator: %s",
                      len(candidates), counts)
        if not candidates:
            raise RuntimeError(
                "no mutation candidates generated - check designable positions")

    # ------------------------------------------------------------------ #
    # Generator-rationale columns surfaced in the generated-provenance table.
    # Each candidate's ``details`` (and its nested ``features`` dict) is searched
    # for these keys; whatever a given generator recorded shows up, the rest are
    # left blank. Generic: no protein/ligand identity is assumed.
    _GEN_FEATURE_KEYS = (
        "msa_freq", "gap_freq", "conservation", "esm_permissive",
        "subfamily_specific", "msa_variable", "msa_permissiveness",
        "ligandmpnn_logp", "dropped_invalid", "multipoint_order",
        "ligand_roles", "n_contacts",
    )
    # Structural / safety columns added per candidate from the s04 complex. The
    # `distance_to_<extra_ligand_id>` columns are DYNAMIC (one per configured
    # extra ligand) and computed at write time; these are the static ones.
    _SAFETY_FEATURE_KEYS = (
        "distance_to_design_ligand", "distance_to_nearest_catalytic",
        "forbidden_check", "risk_flag", "disallowed_reason",
    )

    def _generated_rows(self, candidates, prov_feats=None) -> List[dict]:
        """One JSON-safe row per generated candidate: id, mutation string,
        generator, n_mutations + every recorded generator feature (blank if the
        generator did not record it) + the per-candidate structural distances +
        safety flags from ``prov_feats`` (keyed by candidate_id)."""
        prov_feats = prov_feats or {}
        rows: List[dict] = []
        for c in candidates:
            feats = c.details.get("features", {}) or {}
            row = {
                "candidate_id": c.candidate_id,
                "mutation_string": c.mutation_str,
                "generator": c.generator,
                "n_mutations": len(c.mutations),
            }
            for k in self._GEN_FEATURE_KEYS:
                if k in c.details:
                    v = c.details[k]
                elif k in feats:
                    v = feats[k]
                else:
                    v = None
                if isinstance(v, (list, tuple)):
                    v = ";".join(map(str, v))
                row[k] = v
            # structural distances + safety flags (dynamic extra-ligand columns
            # included verbatim from the precomputed feature dict)
            for k, v in (prov_feats.get(c.candidate_id, {}) or {}).items():
                if isinstance(v, (list, tuple)):
                    v = ";".join(map(str, v))
                row[k] = v
            rows.append(row)
        return rows

    @staticmethod
    def _safety_columns(rows) -> List[str]:
        """Ordered, DE-DUPLICATED list of the safety columns present across
        ``rows``. Stable layout: design-ligand distance (+per-pos), then each
        dynamic ``distance_to_<extra>`` distance (+per-pos, sorted by id), then
        nearest-catalytic distance (+per-pos), then the flag columns. The dynamic
        set is everything starting ``distance_to_`` MINUS the explicitly-placed
        design / catalytic columns, so none is emitted twice."""
        fixed = {
            "distance_to_design_ligand", "distance_to_design_ligand_per_pos",
            "distance_to_nearest_catalytic",
            "distance_to_nearest_catalytic_per_pos",
        }
        extra_ids = sorted({
            k[len("distance_to_"):] for r in rows for k in r
            if k.startswith("distance_to_") and k not in fixed
            and not k.endswith("_per_pos")})
        ordered: List[str] = ["distance_to_design_ligand"]
        if any("distance_to_design_ligand_per_pos" in r for r in rows):
            ordered.append("distance_to_design_ligand_per_pos")
        for lid in extra_ids:
            ordered.append(f"distance_to_{lid}")
            if any(f"distance_to_{lid}_per_pos" in r for r in rows):
                ordered.append(f"distance_to_{lid}_per_pos")
        ordered.append("distance_to_nearest_catalytic")
        if any("distance_to_nearest_catalytic_per_pos" in r for r in rows):
            ordered.append("distance_to_nearest_catalytic_per_pos")
        ordered += ["forbidden_check", "risk_flag", "disallowed_reason",
                    "buried_polar_note"]
        # final guard: dedupe while preserving order
        seen: set = set()
        return [c for c in ordered if not (c in seen or seen.add(c))]

    def _write_generated_provenance(self, ctx: RunContext, candidates,
                                    prov_feats=None) -> None:
        import csv as _csv
        import json as _json

        rows = self._generated_rows(candidates, prov_feats)
        out = ctx.paths.reports / "provenance"
        out.mkdir(parents=True, exist_ok=True)
        cols = (["candidate_id", "mutation_string", "generator", "n_mutations"]
                + list(self._GEN_FEATURE_KEYS)
                + self._safety_columns(rows))

        (out / "generated_candidates.json").write_text(
            _json.dumps(rows, indent=2, default=str))
        with (out / "generated_candidates.csv").open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})
        self.log.info("wrote generated provenance for %d candidates -> %s",
                      len(rows), out)

    # ------------------------------------------------------------------ #
    # cross-generator protected assert
    # ------------------------------------------------------------------ #
    def _assert_no_protected(self, candidates, catalytic, fixed) -> None:
        """Independent post-assembly verification (reviewer requirement): NO
        candidate may mutate a catalytic OR fixed position. The generators
        already exclude them, so this always passes — it's the cross-generator
        safety net. Raises RuntimeError listing the offenders on any violation."""
        protected = set(catalytic) | set(fixed)
        self.log.info(
            "protected positions (no candidate may mutate): catalytic=%s "
            "fixed=%s -> union of %d position(s)",
            sorted(catalytic), sorted(fixed), len(protected))
        offenders = []
        for c in candidates:
            bad = [m for m in c.mutations if m.position in protected]
            if bad:
                offenders.append(
                    f"{c.candidate_id}({c.mutation_str}):"
                    + ",".join(str(m) for m in bad))
        if offenders:
            raise RuntimeError(
                "PROTECTED-POSITION VIOLATION: %d candidate(s) mutate a "
                "catalytic/fixed position %s -> %s"
                % (len(offenders), sorted(protected), "; ".join(offenders[:10])))

    # ------------------------------------------------------------------ #
    # structural distances + safety flags (per candidate, from the s04 complex)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _residue_coords(res) -> Dict[int, Tuple[float, float, float]]:
        """{position: sidechain-centroid (fall back to CA)} for every residue in
        the complex. Generic — coords come straight off the s04 structure."""
        out: Dict[int, Tuple[float, float, float]] = {}
        for pos, r in res.items():
            sc = getattr(r, "sidechain_centroid", None)
            ca = getattr(r, "ca", None)
            # an all-zero centroid means it was never placed -> use CA instead
            coord = sc if (sc and any(sc)) else ca
            if coord and any(coord):
                out[pos] = tuple(coord)
        return out

    def _extra_ligand_clouds(self, ctx: RunContext, cx) -> "Dict[str, list]":
        """{extra_ligand_id: [placed LigandAtom...]} for each configured extra
        ligand, GENERIC. The placed poses live in the WT-complex PDB HETATM
        chains (design ligand first, then extras in input order — the SAME chain
        assignment s04/s05 use). Falls back to the parsed (un-placed) atoms only
        when the PDB has no chain for it; an empty list -> the distance column is
        blank rather than fabricated."""
        extras = list(ctx.get("extra_ligands", []) or [])
        clouds: Dict[str, list] = {}
        if not extras:
            return clouds
        pdb = getattr(cx.structure, "pdb_path", None) or getattr(cx, "path", None)
        het = het_chains_in_pdb(pdb) if pdb else []
        for i, el in enumerate(extras):
            ch = het[i + 1] if (i + 1) < len(het) else None
            placed = parse_pdb_het_chain(pdb, ch) if (pdb and ch) else []
            clouds[el.id] = placed or list(getattr(el, "atoms", []) or [])
        return clouds

    def _candidate_safety_features(
        self, cand, res, res_coord, design_atoms, extra_clouds, cat_coords,
        catalytic, fixed, mgcfg,
    ) -> dict:
        """Per-candidate structural distances + safety flags. For a multi-mutation
        candidate every distance is the MIN across its mutated positions (the
        closest approach) and the per-position values are kept as a ``;``-joined
        list column. Distances are blank (None) when the relevant coordinate set
        is unavailable (e.g. mock run with no placed extra-ligand pose)."""
        feats: dict = {}
        positions = [m.position for m in cand.mutations]
        coords = [res_coord[p] for p in positions if p in res_coord]

        def _agg(cloud):
            per = [_min_dist(c, cloud) for c in coords] if cloud else []
            per = [d for d in per if d is not None]
            return (round(min(per), 3) if per else None,
                    [round(d, 3) for d in per] if len(per) > 1 else None)

        d_design, d_design_pp = _agg(design_atoms)
        feats["distance_to_design_ligand"] = d_design
        if d_design_pp is not None:
            feats["distance_to_design_ligand_per_pos"] = d_design_pp
        for lid, cloud in extra_clouds.items():
            d, d_pp = _agg(cloud)
            feats[f"distance_to_{lid}"] = d
            if d_pp is not None:
                feats[f"distance_to_{lid}_per_pos"] = d_pp
        d_cat, d_cat_pp = _agg(cat_coords)
        feats["distance_to_nearest_catalytic"] = d_cat
        if d_cat_pp is not None:
            feats["distance_to_nearest_catalytic_per_pos"] = d_cat_pp

        # forbidden_check: confirm designable (none of the candidate's positions
        # is catalytic/fixed). Always 'ok' here (the assert already guaranteed
        # it); the column makes the guarantee explicit per row for the reviewer.
        protected = set(catalytic) | set(fixed)
        bad_pos = [p for p in positions if p in protected]
        feats["forbidden_check"] = (
            "ok" if not bad_pos
            else "VIOLATION:" + ",".join(map(str, bad_pos)))

        # risk flags (FLAG, do not silently drop): introduce Gly/Pro/Cys; OR a
        # radical physicochemical substitution in a catalytic residue's 2-shell
        # (sequence +-2 OR centroid within ~6 A of a catalytic residue).
        reasons: List[str] = []
        cat_set = set(catalytic)
        for m in cand.mutations:
            if m.mut in _HELIX_BREAKERS:
                reasons.append(f"introduces_{m.mut}@{m.position}")
            if m.mut in _DISULFIDE_RISK and m.wt not in _DISULFIDE_RISK:
                reasons.append(f"introduces_C@{m.position}")
            near_cat = self._near_catalytic(
                m.position, res_coord, cat_coords, cat_set)
            if near_cat and _is_radical_substitution(m.wt, m.mut):
                reasons.append(
                    f"radical_2shell_{m.wt}{m.position}{m.mut}({near_cat})")
        feats["risk_flag"] = bool(reasons)
        feats["disallowed_reason"] = ";".join(reasons)
        # buried-polar-unsatisfied heuristic: a TRUE unsatisfied-buried-polar call
        # needs the MUTANT side chain's H-bond partners + burial, which we do not
        # model at s07 (no mutant structure yet) -> deliberately skipped here and
        # left to s09/s11 (negative design: neg_buried_core_polar). Noted, not
        # silently omitted.
        feats["buried_polar_note"] = "deferred_to_s09_s11"
        return feats

    @staticmethod
    def _near_catalytic(pos, res_coord, cat_coords, cat_set) -> str:
        """Is ``pos`` in a catalytic residue's 2-shell? Returns a short tag
        ('seq'/'space'/'seq+space') or '' if not. Sequence shell = |Δindex| <=
        _CATALYTIC_SHELL_SEQ to any catalytic residue; spatial shell = centroid
        within _CATALYTIC_SHELL_ANGSTROM of any catalytic centroid."""
        by_seq = any(abs(pos - c) <= _CATALYTIC_SHELL_SEQ for c in cat_set)
        by_space = False
        pc = res_coord.get(pos)
        if pc is not None and cat_coords:
            md = _min_dist(pc, cat_coords)
            by_space = md is not None and md <= _CATALYTIC_SHELL_ANGSTROM
        if by_seq and by_space:
            return "seq+space"
        if by_seq:
            return "seq"
        if by_space:
            return "space"
        return ""

    # ------------------------------------------------------------------ #
    # reviewer-facing budget tiers (single / multipoint / risky)
    # ------------------------------------------------------------------ #
    def _write_budget_tiers(self, ctx: RunContext, candidates, prov_feats,
                            mgcfg) -> None:
        """Tier the FULL generated pool into single / multipoint / risky budget
        views for the s08-s10 cost, each ranked best-first by the existing
        per-candidate score (msa_permissiveness, then closeness to the design
        ligand) and capped to its configured top-N. Each tier is a SEPARATE
        provenance file; the combined file (and ctx `candidates`) is unchanged.

        risky = the risk-flagged candidates (single or multi); single/multipoint
        are the NON-risky ones split by mutation count, so a flagged candidate is
        surfaced in exactly the `risky` budget rather than hidden among clean
        ones."""
        import csv as _csv
        import json as _json

        def _score(c) -> float:
            # higher = preferred: family-permissive + close to the design ligand.
            perm = float(c.details.get("msa_permissiveness", 0.0) or 0.0)
            d = prov_feats.get(c.candidate_id, {}).get("distance_to_design_ligand")
            closeness = 0.0 if d is None else 1.0 / (1.0 + float(d))
            return perm + 0.1 * closeness

        risky = [c for c in candidates
                 if prov_feats.get(c.candidate_id, {}).get("risk_flag")]
        clean = [c for c in candidates
                 if not prov_feats.get(c.candidate_id, {}).get("risk_flag")]
        single = [c for c in clean if len(c.mutations) == 1]
        multipoint = [c for c in clean if len(c.mutations) >= 2]

        def _top(pool, n):
            return sorted(pool, key=_score, reverse=True)[:max(0, n)]

        tiers = {
            "single": _top(single, mgcfg.tier_single_top),
            "multipoint": _top(multipoint, mgcfg.tier_multipoint_top),
            "risky": _top(risky, mgcfg.tier_risky_top),
        }
        out = ctx.paths.reports / "provenance"
        out.mkdir(parents=True, exist_ok=True)
        summary = {}
        for name, pool in tiers.items():
            rows = self._generated_rows(pool, prov_feats)
            cols = (["candidate_id", "mutation_string", "generator",
                     "n_mutations"] + list(self._GEN_FEATURE_KEYS)
                    + self._safety_columns(rows))
            (out / f"generated_candidates_{name}.json").write_text(
                _json.dumps(rows, indent=2, default=str))
            with (out / f"generated_candidates_{name}.csv").open(
                    "w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(
                        {k: ("" if r.get(k) is None else r[k]) for k in cols})
            summary[name] = len(rows)
        self.log.info(
            "budget tiers (top single/multipoint/risky = %d/%d/%d): wrote %s",
            mgcfg.tier_single_top, mgcfg.tier_multipoint_top,
            mgcfg.tier_risky_top, summary)

    # ------------------------------------------------------------------ #
    def _chemistry(self, designable, res, by_pos_contacts, atom) -> _Pool:
        """Contact-ENSEMBLE rules: at each position aggregate ALL ligand contacts
        (distance-weighted role vote), not just the single nearest atom, and take
        the pools of the top roles."""
        out: _Pool = []
        for pos in designable:
            r = res.get(pos)
            cs = by_pos_contacts.get(pos)
            if r is None or not cs:
                continue
            vote: Dict[str, float] = {}
            for c in cs:
                a = atom.get(c.ligand_atom_id)
                if a is None:
                    continue
                role = _ligand_role(a)
                vote[role] = vote.get(role, 0.0) + 1.0 / max(0.5, c.distance)
            if not vote:
                continue
            top = sorted(vote, key=lambda k: -vote[k])[:2]   # top 2 roles
            poolset = "".join(dict.fromkeys("".join(_RULE_POOL[t] for t in top)))
            for aa in poolset:
                if aa != r.aa:
                    out.append(([Mutation(r.aa, pos, aa)],
                                {"ligand_roles": top, "n_contacts": len(cs)}))
        return out

    def _msa(self, designable, res, pf, mgcfg) -> _Pool:
        """GATED MSA sampling: skip gappy columns; require the family actually
        tolerates the residue; surface subfamily/ESM signal at GENERATION."""
        out: _Pool = []
        for pos in designable:
            r = res.get(pos)
            f = pf.get(pos)
            if r is None or f is None:
                continue
            if f.gap_frequency > mgcfg.msa_gap_max:      # poorly-aligned column
                continue
            ranked = sorted(
                f.allowed_aa,
                key=lambda aa: -f.amino_acid_frequencies.get(aa, 0.0))
            for aa in ranked[: mgcfg.msa_top_k]:
                freq = f.amino_acid_frequencies.get(aa, 0.0)
                if aa == r.aa or freq <= 0.0:            # not family-observed
                    continue
                out.append(([Mutation(r.aa, pos, aa)], {
                    "msa_freq": round(freq, 4),
                    "gap_freq": round(f.gap_frequency, 3),
                    "subfamily_specific": f.specificity_divergence > 0,
                    "esm_permissive": f.esm_variability > 0.5,
                    "msa_variable": f.conservation_score < 0.7,
                }))
        return out

    def _ligandmpnn(self, ctx, cx, designable, catalytic, fixed, res, mgcfg) -> _Pool:
        """LigandMPNN ligand-aware design, read as per-position marginals.

        LigandMPNN redesigns EVERY designable position in one pass, so each raw
        sample carries far more edits than a focused active-site candidate should
        — a per-design mutation cap discards them all. Instead we treat the
        sampled designs as an approximation of LigandMPNN's per-position
        substitution marginals: for each designable position (never
        catalytic/fixed, only where the WT matches) tally the residue it most
        often substitutes and emit a single-point WT->consensus candidate, ranked
        by the fraction of designs that support it (carried as ligandmpnn_logp).
        The strongest positions are also combined into one bounded multi-point
        "consensus active-site" candidate (<= ligandmpnn_max_mut_per_design).
        Per-generator quotas then trim to budget."""
        from evoliez.utils.seeds import derive_seed
        designs = design_sequences(
            cx, designable, mgcfg, ctx.paths.mutations / "ligandmpnn",
            backend=ctx.config.backend_for("s07_mutation_gen"),
            dry_run=ctx.dry_run,
            seed=derive_seed(ctx.config.seed, "ligandmpnn"))
        dset = set(designable)
        n = len(designs) or 1
        tally: Dict[int, Counter] = {}
        for muts, _logp in designs:
            for m in muts:
                if (m.position in dset and m.position not in catalytic
                        and m.position not in fixed
                        and res.get(m.position) is not None
                        and res[m.position].aa == m.wt):
                    tally.setdefault(m.position, Counter())[m.mut] += 1
        consensus = []   # (position, wt, mut, support_fraction)
        for pos, counts in tally.items():
            mut, cnt = counts.most_common(1)[0]
            consensus.append((pos, res[pos].aa, mut, cnt / n))
        consensus.sort(key=lambda t: (-t[3], t[0]))   # strongest support first
        out: _Pool = []
        for pos, wt, mut, frac in consensus:
            out.append(([Mutation(wt, pos, mut)],
                        {"ligandmpnn_logp": round(frac, 4), "dropped_invalid": 0}))
        top = consensus[:mgcfg.ligandmpnn_max_mut_per_design]
        if len(top) >= 2:
            out.append(([Mutation(wt, pos, mut) for pos, wt, mut, _f in top],
                        {"ligandmpnn_logp": round(top[0][3], 4),
                         "multipoint_order": len(top)}))
        return out

    def _multipoint(self, allowed, res, nearest, mgcfg) -> _Pool:
        """FuncLib-style multi-point library: sample 2..N-point combinations from
        the allowed-AA pool on the most ligand-CENTRAL designable positions
        (active-site epistasis). Bounded + deterministic, not exhaustive."""
        positions = [p for p in allowed if res.get(p) is not None]
        positions.sort(key=lambda p: nearest.get(p, 99.0))   # ligand-central first
        positions = positions[:12]
        cap = mgcfg.multipoint_max or max(1, int(0.3 * mgcfg.max_candidates))
        out: _Pool = []
        rng = derive_seed(0x07, "multipoint")
        seen: set = set()
        max_order = min(mgcfg.multipoint_order, len(positions))
        for order in range(2, max_order + 1):
            for combo in itertools.combinations(positions, order):
                if len(out) >= cap:
                    return out
                muts = []
                for p in combo:
                    opts = sorted(allowed[p])
                    rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
                    aa = opts[rng % len(opts)]
                    if res[p].aa != aa:
                        muts.append(Mutation(res[p].aa, p, aa))
                if len(muts) >= 2:
                    key = ";".join(str(m) for m in muts)
                    if key not in seen:
                        seen.add(key)
                        out.append((muts, {"multipoint_order": len(muts)}))
        return out

    def _assemble(self, pools, mgcfg) -> List[Candidate]:
        """Merge per-generator pools under per-generator QUOTAS (so the first
        generator can't fill the cap), dedup, trim to max_candidates."""
        active = [g for g in pools if pools[g]]
        quota = dict(mgcfg.generator_quota) or {
            g: _DEFAULT_QUOTA.get(g, 0.1) for g in active}
        tot = sum(quota.get(g, 0.0) for g in active) or 1.0
        cap = {g: max(1, int(mgcfg.max_candidates * quota.get(g, 0.0) / tot))
               for g in active}
        candidates: List[Candidate] = []
        seen: set = set()
        for g in active:
            n = 0
            for muts, details in pools[g]:
                if len(candidates) >= mgcfg.max_candidates or n >= cap[g]:
                    break
                muts = [m for m in muts if m.wt != m.mut]
                if not muts:
                    continue
                key = ";".join(str(m) for m in muts)
                if key in seen:
                    continue
                seen.add(key)
                n += 1
                candidates.append(Candidate(
                    candidate_id=f"mut_{len(candidates):05d}", mutations=muts,
                    generator=g, details=details))
        return candidates
