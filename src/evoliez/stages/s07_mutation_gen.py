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
from typing import Dict, List, Tuple

from evoliez.adapters.ligandmpnn import design_sequences
from evoliez.context import RunContext
from evoliez.features.evolutionary import permissiveness
from evoliez.stages.base import Stage
from evoliez.types import Candidate, Mutation
from evoliez.utils.seeds import derive_seed

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

        # MSA permissiveness for downstream scoring
        for cand in candidates:
            perms = [permissiveness(pf.get(m.position), m.mut)
                     for m in cand.mutations if pf.get(m.position)]
            cand.details["msa_permissiveness"] = round(
                sum(perms) / max(1, len(perms)), 4)

        ctx.put("candidates", candidates)
        ctx.persist_meta("n_candidates_generated", len(candidates))
        # Durable per-candidate GENERATED-provenance table (paper methods): the
        # FULL pre-attrition pool with each candidate's generator + whatever
        # rationale features that generator recorded. Written here (not at s11)
        # because downstream stages prune the pool, so this is the only place the
        # complete generated set exists. The funnel report (s11) reads it back.
        self._write_generated_provenance(ctx, candidates)
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

    def _generated_rows(self, candidates) -> List[dict]:
        """One JSON-safe row per generated candidate: id, mutation string,
        generator, n_mutations + every recorded generator feature (blank if the
        generator did not record it)."""
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
            rows.append(row)
        return rows

    def _write_generated_provenance(self, ctx: RunContext, candidates) -> None:
        import csv as _csv
        import json as _json

        rows = self._generated_rows(candidates)
        out = ctx.paths.reports / "provenance"
        out.mkdir(parents=True, exist_ok=True)
        cols = (["candidate_id", "mutation_string", "generator", "n_mutations"]
                + list(self._GEN_FEATURE_KEYS))

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
        """LigandMPNN designs, VALIDATED before they become candidates: keep only
        designable-position edits, never catalytic/fixed, only where the WT
        matches, and cap mutations per design (a focused active-site edit)."""
        designs = design_sequences(
            cx, designable, mgcfg, ctx.paths.mutations / "ligandmpnn",
            backend=ctx.config.backend_for("s07_mutation_gen"),
            dry_run=ctx.dry_run)
        dset = set(designable)
        out: _Pool = []
        for muts, logp in designs:
            kept = [m for m in muts
                    if m.position in dset and m.position not in catalytic
                    and m.position not in fixed and res.get(m.position) is not None
                    and res[m.position].aa == m.wt]
            if kept and len(kept) <= mgcfg.ligandmpnn_max_mut_per_design:
                out.append((kept, {"ligandmpnn_logp": logp,
                                   "dropped_invalid": len(muts) - len(kept)}))
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
