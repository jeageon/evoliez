"""Stage 07 - mutation generation (spec section 12).

Three generators (config-selectable): chemistry-aware rules, MSA-guided
sampling, and LigandMPNN ligand-aware design. Outputs deduplicated
:class:`Candidate` objects with rationale metadata attached.
"""

from __future__ import annotations

from pathlib import Path
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

        # ---- binding-site exhaustive scan -------------------------------- #
        # Cheap-run discovery: chemistry_rules + msa_sampler with mock
        # homologs (which is what happens when the UniRef30 mmseqs DB isn't
        # available) failed to propose the documented D222S/N/T/Q/H/A
        # Tishkov-class NADP-switch family for PseFDH. Two reasons:
        #
        #   1. chemistry_rules selects substitutions based on the nearest
        #      ligand-atom role at each designable position. D222's nearest
        #      NADP+ atom is a phosphate oxygen (anion role) → pool=KRH;
        #      Tishkov's polar/amide swaps (S/N/T/Q) are NOT in that pool.
        #   2. msa_sampler ranks substitutions by frequency in the homolog
        #      MSA; mock homologs carry no real evolutionary signal at the
        #      cofactor-specificity loop, so the Tishkov AAs never make it
        #      to the top of the family-frequency ranking.
        #
        # The fix is to ALSO scan EVERY substitution at every residue the
        # config explicitly flagged as ``known_binding_site`` — these are
        # the residues the user already decided are mechanistically
        # important, so it's worth burning ~19 candidates each to make
        # sure the literature-known cofactor-switch / activity-switch
        # mutations are in the pool. Universal: every enzyme card declares
        # a known_binding_site, so this generator works the same way for
        # PseFDH / XR / TEM-1 / Bgl3 / P450 BM3.
        #
        # Respects ``fixed_positions`` (catalytic residues are never
        # mutated) and the ``max_candidates`` cap. AA order is roughly by
        # how often these substitutions appear as beneficial in published
        # literature for cofactor-switch / promiscuity work.
        if "binding_site_scan" in mgcfg.methods:
            binding_site_positions = list(
                ctx.get("known_binding_site", []) or []
            )
            fixed_positions = set(ctx.get("fixed_positions", []) or [])
            # Polar/amide/small first (Tishkov-class), then bulkier
            # changes; aromatic last because they rarely fit a cofactor
            # pocket without other accompanying mutations.
            _SCAN_AAS = "STNQHADEGRKVILMFCYWP"
            for pos in binding_site_positions:
                r = by_pos_res.get(pos)
                if r is None or pos in fixed_positions:
                    continue
                for aa in _SCAN_AAS:
                    if aa == r.aa:
                        continue
                    add(
                        [Mutation(r.aa, pos, aa)],
                        "binding_site_scan",
                        {
                            "known_binding_site": True,
                            "exhaustive_scan": True,
                        },
                    )

        # ---- LigandMPNN -------------------------------------------------- #
        # P0.4: real-backend default generator. Preflight before invoking
        # the adapter so a missing precondition is a CLEAR diagnostic instead
        # of a downstream crash:
        #   1. Real backend needs a full-atom structure (CA-only -> garbage
        #      ligand-aware design); skip ligandmpnn honestly if missing.
        #   2. Ligand must carry atoms (atom-index lock failed = downstream
        #      ID-keyed features unreliable).
        #   3. Catalytic / fixed positions must NOT appear in `designable` -
        #      LigandMPNN would otherwise propose mutations there.
        if "ligandmpnn" in mgcfg.methods:
            mpnn_backend = ctx.config.backend_for("s07_mutation_gen")
            skip_reason = None
            from evoliez.adapters.receptor_io import is_full_atom_pdb
            from evoliez.config import Backend

            pdb_path = getattr(cx.structure, "pdb_path", None)
            is_real = (mpnn_backend is Backend.real)
            if is_real and (
                not pdb_path or not is_full_atom_pdb(Path(pdb_path))
            ):
                skip_reason = ("real LigandMPNN requires a full-atom protein "
                               f"(got pdb_path={pdb_path!r}); skipping")
            elif not cx.ligand.atoms:
                skip_reason = ("ligand has no parsed atoms; LigandMPNN "
                               "cannot do ligand-aware design")
            else:
                fixed = set(ctx.get("fixed_positions", []) or [])
                bad = sorted(set(designable) & fixed)
                if bad:
                    skip_reason = (
                        f"designable positions overlap fixed_positions {bad[:8]}"
                        " - upstream filter regressed; refuse to let LigandMPNN"
                        " propose catalytic/fixed mutations"
                    )

            if skip_reason:
                self.log.warning("LigandMPNN preflight: %s", skip_reason)
                ctx.persist_meta("ligandmpnn_skipped_reason", skip_reason)
            else:
                designs = design_sequences(
                    cx, designable, mgcfg, ctx.paths.mutations / "ligandmpnn",
                    backend=mpnn_backend, dry_run=ctx.dry_run,
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
