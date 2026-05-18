"""Stage 07 - mutation generation (spec section 12).

Three generators (config-selectable): chemistry-aware rules, MSA-guided
sampling, and LigandMPNN ligand-aware design. Outputs deduplicated
:class:`Candidate` objects with rationale metadata attached.
"""

from __future__ import annotations

from typing import Dict, List

from evoliez.adapters.ligandmpnn import design_sequences
from evoliez.context import RunContext
from evoliez.features.evolutionary import permissiveness
from evoliez.stages.base import Stage
from evoliez.types import Candidate, Mutation

# spec 12.4 chemistry-aware pools keyed by the nearest ligand atom's role
_RULE_POOL = {
    "anion": "KRH",        # phosphate / carboxylate
    "cation": "DE",
    "aromatic": "FYWH",
    "hydrophobic": "LIVMF",
    "acceptor": "STNQYKR",  # ligand acceptor -> donor side chains
    "donor": "STNQDE",      # ligand donor -> acceptor side chains
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


class MutationGenStage(Stage):
    name = "s07_mutation_gen"

    def run(self, ctx: RunContext) -> None:
        cx = ctx.require("wt_complex")
        feats = ctx.require("position_features")
        designable = ctx.require("designable_positions")
        contacts = ctx.require("contacts")
        mgcfg = ctx.config.mutation_generation

        by_pos_res = {r.index: r for r in cx.structure.residues}
        pf_by_pos = {
            f.target_position: f for f in feats if f.target_position is not None
        }
        atom_by_id = {a.id: a for a in cx.ligand.atoms}
        # nearest ligand contact per designable residue
        nearest: Dict[int, object] = {}
        for c in sorted(contacts, key=lambda c: c.distance):
            nearest.setdefault(c.residue_index, c)

        candidates: List[Candidate] = []
        seen: set[str] = set()

        def add(muts: List[Mutation], generator: str, details: dict) -> None:
            muts = [m for m in muts if m.wt != m.mut]
            if not muts:
                return
            key = ";".join(str(m) for m in muts)
            if key in seen or len(candidates) >= mgcfg.max_candidates:
                return
            seen.add(key)
            candidates.append(
                Candidate(
                    candidate_id=f"mut_{len(candidates):05d}",
                    mutations=muts,
                    generator=generator,
                    details=details,
                )
            )

        # ---- chemistry rules -------------------------------------------- #
        if "chemistry_rules" in mgcfg.methods:
            for pos in designable:
                r = by_pos_res.get(pos)
                c = nearest.get(pos)
                if r is None or c is None:
                    continue
                atom = atom_by_id.get(c.ligand_atom_id)
                if atom is None:
                    continue
                pool = _RULE_POOL[_ligand_role(atom)]
                for aa in pool:
                    if aa == r.aa:
                        continue
                    add(
                        [Mutation(r.aa, pos, aa)],
                        "chemistry_rules",
                        {
                            "near_ligand_atom": c.ligand_atom_id,
                            "ligand_role": _ligand_role(atom),
                        },
                    )

        # ---- MSA-guided sampling ---------------------------------------- #
        if "msa_sampler" in mgcfg.methods:
            for pos in designable:
                r = by_pos_res.get(pos)
                pf = pf_by_pos.get(pos)
                if r is None or pf is None:
                    continue
                ranked = sorted(
                    pf.allowed_aa,
                    key=lambda aa: -pf.amino_acid_frequencies.get(aa, 0.0),
                )
                for aa in ranked[:4]:
                    if aa == r.aa:
                        continue
                    add(
                        [Mutation(r.aa, pos, aa)],
                        "msa_sampler",
                        {
                            "msa_variable": pf.conservation_score < 0.7,
                            "family_observed": aa,
                            "msa_freq": pf.amino_acid_frequencies.get(aa, 0.0),
                        },
                    )

        # ---- LigandMPNN -------------------------------------------------- #
        if "ligandmpnn" in mgcfg.methods:
            designs = design_sequences(
                cx, designable, mgcfg, ctx.paths.mutations / "ligandmpnn",
                backend=ctx.config.backend_for("s07_mutation_gen"),
                dry_run=ctx.dry_run,
            )
            for muts, logp in designs:
                add(muts, "ligandmpnn", {"ligandmpnn_logp": logp})

        # attach MSA permissiveness for downstream scoring
        for cand in candidates:
            perms = []
            for m in cand.mutations:
                pf = pf_by_pos.get(m.position)
                perms.append(permissiveness(pf, m.mut) if pf else 0.0)
            cand.details["msa_permissiveness"] = round(
                sum(perms) / max(1, len(perms)), 4
            )

        ctx.put("candidates", candidates)
        ctx.persist_meta("n_candidates_generated", len(candidates))
        gen_counts: Dict[str, int] = {}
        for c in candidates:
            gen_counts[c.generator] = gen_counts.get(c.generator, 0) + 1
        self.log.info("generated %d candidates by generator: %s",
                      len(candidates), gen_counts)
        if not candidates:
            raise RuntimeError(
                "no mutation candidates generated - check designable positions"
            )
