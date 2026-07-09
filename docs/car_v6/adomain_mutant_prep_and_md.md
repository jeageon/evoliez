# CAR A-domain mutant prep + reproducible portfolio MD (V6-4 completion)

**Status:** server-verified (evo, 4×A6000), claim-safe
**Scope:** closes a reproducibility gap found during live V6 server verification.

## The gap (found in live verification)

The production pipeline emits **full-protein** anchored candidate structures
(`runs/srcar_3hp_full/md/mut_*/*_anchored.pdb`): ~1188 residues, ligand set
**ATP + 3-HP + NADP**, no explicit adenylation-site Mg²⁺, in a frame ~27 Å
displaced from the A-domain template. These **cannot** feed the `car_spec`
Amber builder, which models the **adenylation half-reaction** (A-domain,
**ATP + 3-HP + Mg²⁺**, ~720 res) and:

- graph-matches ATP/3-HP by SMILES (an all-`UNK`, H-bearing minimized blob fails), and
- **extracts Mg²⁺ from the input** (`_extract_metal_xyz`) — raises if absent.

The only `car_spec`-compatible input that existed was a hand-made `wt_reverted.pdb`
(WT only). The prior 3-job MD run depended on such ad-hoc inputs → **not
reproducible** from pipeline outputs (a V6-7 concern).

## The fix — `scripts/prep_car_adomain_mutant.py`

Derive each mutant input by **mutating the WT A-domain template in place**:

- the ATP + 3-HP + Mg²⁺ pose/frame is preserved **exactly** (HETATM copied verbatim);
- mutated residues are reduced to backbone (N/CA/C/O), resname relabelled →
  `tleap` rebuilds an idealized side chain, which the builder's minimization + 100 ps
  equilibration relaxes before production MD;
- WT residue identity at each site is **asserted** against the mutation string
  (fail-loud on wrong template/numbering).

Reproducible from `(WT template + mutation string)`. Verified: sites 407=GLY,
430=GLY, 433=SER, catalytic triad S268/T269/K273 — all consistent with `car_spec`.

> This is **template mutagenesis**, not functional-state anchoring. The resulting
> evidence is screening-level reaction-geometry access — the same claim ceiling as
> the WT run. Idealized side chains + single short replica are explicit limitations.

## Live deconvolution result (WT / G430R / lead), 0.2 ns explicit-solvent pmemd.cuda

| job | O→P min (Å) | O→P mean (Å) | in-line angle mean (°) | near-attack occ | Mg–Pα (Å) | energy drift | ligand ret. |
|-----|------|------|------|------|------|------|------|
| WT | 5.16 | 5.96 | 90.6 | 0.0 | 3.35 | 0.22 % | 1.0 |
| G430R | 10.27 | 11.41 | 122.8 | 0.0 | 3.33 | 0.06 % | 1.0 |
| LEAD (G430R;S433F;G407K) | 7.25 | 8.66 | **167.3** (occ 1.0) | 0.0 | 4.97 | 0.87 % | 1.0 |

All: `status=ok`, ligand retained, energy drift < 1 % (stable), claim ceiling
`screening-level reaction-geometry access evidence (NOT activity/kcat/activation-barrier)`.

### Reading (claim-safe)

- The two observables **disagree**, by design (roadmap V6-3): the **LEAD reaches a
  near-linear in-line attack angle (167° vs WT 90°, productive-angle occupancy 1.0)**
  — consistent with the G430R Arg-anchor hypothesis orienting 3-HP in-line to α-P —
  but **distance access is not improved** (mutants sit further out) and **near-attack
  occupancy is 0.0 for every variant** in unbiased 0.2 ns.
- No unbiased MD resolves the access **cost**; this is precisely the escalation
  trigger to the Amber PMF tier (V6-3). The distance signal and the angle signal must
  be reported separately.
- **Nothing here is an activity, kcat, or catalytic-superiority claim.** It is a
  screening-level, mechanism-probe geometry signal on hypothesis-grade candidates.

### Reproducibility

WT re-run (0.2 ns, GPU 0, fresh build) reproduced the prior WT (0.1 ns, prior
session, different GPU) to **0.02 Å** on O→P min (5.16 vs 5.18) and **0.02 Å** on
Mg–Pα (3.35 vs 3.33); near-attack occ 0.0 in both.

## Provenance

- inputs: `/mnt/data/jglee/v6_md_verify/inputs/mut_{G430R,LEAD}.pdb`
- runs: `/mnt/data/jglee/v6_md_verify/car/{wt,G430R,LEAD}/`
- records: `reports/provenance/amber_gpu_jobs_verify.jsonl`
