# V6-6 — Non-CAR, Non-Redox Real-Backend Proof (TEM-1 β-lactamase)

**Phase:** V6-6 · **Status:** ✅ PASS · **Config:** [`configs/tem1_acyl_amber.yaml`](../../configs/tem1_acyl_amber.yaml) · **Driver:** [`scripts/run_tem1_amber.py`](../../scripts/run_tem1_amber.py)
**Provenance:** `reports/provenance/tem1_amber_system_manifest.json`, `reports/provenance/tem1_amber_md.jsonl`

## Goal (Gate 3)

Prove V6 is **not a CAR-specific path**: run a genuinely different enzyme —
**TEM-1 β-lactamase** (class A serine hydrolase, UniProt P62593), a **non-redox,
non-metal** target — through the **same** V6 Amber framework, with **real backend
components** (not mock), resolving the **protein-nucleophile geometry** (catalytic
Ser68 Oγ attacking the β-lactam scissile carbonyl) from the topology.

## The new general capability: protein-nucleophile geometry

CAR/FDH have a **ligand** nucleophile; TEM-1's nucleophile is a **protein atom**
(Ser68 Oγ, the STFK-motif serine; Ambler Ser70). V6-6 makes this a general feature
of the mechanism-geometry resolver in the Amber builder:

- `donor_protein: "SER:OG:68"` → the O_nuc is resolved from the **prmtop topology**
  (ordinal + identity-validated, like the catalytic residues), mapped to a mask
  `:<resid>@OG`.
- the **acceptor** (scissile carbonyl C) + its Bürgi-Dunitz reference O come from the
  **design ligand** (benzylpenicillin).
- **β-lactam-specific SMARTS** `[CX3;r4](=[OX1])[NX3;r4]`: the scissile carbonyl is
  the one in the strained **4-membered ring** — the generic `[CX3](=O)[NX3]` would
  mis-select benzylpenicillin's acyclic side-chain amide (verified: it picks the
  wrong carbonyl).

The same `amber_builder` / `amber_scheduler` run this with **no CAR-specific code**
and **no metal / no co-substrate**.

## Real backend

| Component | Real backend used |
|---|---|
| Structure | **Boltz** Michaelis complex (TEM-1 + benzylpenicillin), MSA server, 5 diffusion samples |
| System build | **AmberTools** (pdb4amber / antechamber AM1-BCC / tleap), V6-1 builder |
| MD | **pmemd.cuda** explicit-solvent, V6-2 executor |
| Geometry | **cpptraj** from the real MD topology |

The Boltz complex placed benzylpenicillin correctly in the active site — **Ser68 Oγ
2.46 Å from the nearest substrate atom** (4.79 Å to the ligand centroid) — a
realistic Michaelis-like engagement of the catalytic serine.

## Result

**Amber system** (V6-1, no CAR code): 44,326 atoms, neutral, **no metal, no
co-substrate**. The protein-nucleophile reactive map (all identity-validated):

| Role | Mask | Amber # | Meaning |
|---|---|---|---|
| O_nuc | `:68@OG` | 1090 | catalytic **Ser68 Oγ** (from the topology) |
| acceptor (P_alpha) | `:LIG@C14` | 4456 | β-lactam scissile carbonyl C |
| O_leaving | `:LIG@O2` | 4457 | Bürgi-Dunitz carbonyl O |
| catalytic | `:68`,`:69`,`:71` | — | STFK Ser68 / Thr69 / Lys71 |

**Explicit-solvent MD** (V6-2 pmemd.cuda, 0.1 ns, 91.8 s): `status=ok`, stable
(ligand retention 1.0, backbone intact, energy drift **0.7 %**). Reaction geometry
from the real topology:

- **Ser68 Oγ → β-lactam carbonyl C**: mean **8.51 Å**, min 7.39 Å (250 frames).
- **Nuc–C=O angle** (Bürgi-Dunitz, read-only): mean **75.7°**.

**Analysis (screening-level, claim-safe):** the Boltz Michaelis complex started
near-attack (Ser68 Oγ 2.46 Å from the substrate), but unbiased explicit MD relaxes
the Oγ→carbonyl distance to ~8.5 Å over 0.1 ns — the *same* physics seen for CAR
(the near-attack geometry is not a stable minimum in unbiased classical MD). This is
**screening-level reaction-geometry evidence on WT TEM-1**, not an activity claim;
if a candidate campaign were run, the PMF tier (V6-3) would be the correct next
observable, exactly as for CAR. The point of V6-6 is proven regardless: the protein
nucleophile (Ser68 Oγ) is resolved from the real topology and reported through the
identical evidence path.

## Claim discipline

The report carries only screening-level geometry evidence — **no activity, kcat, or
catalytic claim** (there is no wet-lab data and no candidate ranking here; this is a
framework-generality proof on WT TEM-1). ClaimGuard-clean.

## Success conditions — checklist

- [x] A non-CAR, non-redox target runs with **real backend** components (Boltz + AmberTools + pmemd.cuda), not mock.
- [x] **Protein-nucleophile geometry** (Ser68 Oγ) resolved from the topology + reported as evidence.
- [x] The report is ClaimGuard-clean.
- [x] Candidate evidence + uncertainty reported without activity claims.
