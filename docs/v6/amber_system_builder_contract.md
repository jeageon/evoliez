# V6-1 — Amber System Builder Contract (server-verified)

**Phase:** V6-1 · **Status:** ✅ PASS · **Module:** [`src/evoliez/adapters/amber_builder.py`](../../src/evoliez/adapters/amber_builder.py)
**Driver:** [`scripts/build_amber_system.py`](../../scripts/build_amber_system.py) · **Runner:** [`scripts/run_amber_build_car.sh`](../../scripts/run_amber_build_car.sh)
**Manifest:** [`reports/provenance/amber_system_manifest.json`](../../reports/provenance/amber_system_manifest.json) · **Tests:** [`tests/test_amber_builder.py`](../../tests/test_amber_builder.py)

## Goal

Convert an EvoLiEZ mechanism complex (protein + design ligand + curated-charge
co-substrate(s) + catalytic metal + explicit solvent + ions) into an auditable
Amber `prmtop`/`inpcrd`, and map the reactive atoms (O_nuc, Pα, O_leaving, Mg,
catalytic residues) onto exact prmtop atom indices + cpptraj masks that the V6
MD/PMF tiers consume.

## The contract: `AmberBuildSpec`

The builder is **mechanism-generic** — nothing CAR-specific is hardcoded.
Everything comes from `AmberBuildSpec`:

| Field | Meaning |
|---|---|
| `ligands: [LigandBuildSpec]` | each: `id` (Amber resname), `role`, `smiles`, `net_charge`, `allow_am1bcc`, `charges_mol2` |
| `metal: MetalBuildSpec` | `enabled`, `ion`, `element`, `charge`, `ion_frcmod` (12-6-4) |
| `reactive: ReactiveBuildSpec` | `donor_smarts` / `acceptor_smarts` / `donor_protein`, `transfer_is_h`, `label` |
| `catalytic_residues: [str]` | e.g. `["S268","T269","K273"]` (one-letter + original PDB number) |
| `protein_ff`, `water`, `solvent`, `box_buffer_A` | FF/solvent choices |

FDH (hydride, no metal), TEM-1 (protein nucleophile via `donor_protein`) and CAR
(adenylation + Mg) all build through this one contract.

## Force-field / water / ion decision

For the catalytic **bridging Mg²⁺**, coordination geometry matters, so V6-1 uses
the **Li–Merz 12-6-4** divalent set (`frcmod.ions234lm_1264_tip3p`) — the C4
ion-induced-dipole term gives correct Mg²⁺–O distances (ROADMAP_V6 §14). There is
**no OPC 12-6-4 ion set** on the server, so the consistent, well-validated
combination is:

> **ff14SB protein · GAFF2 ligand · TIP3P water · Li–Merz 12-6-4 divalent · Joung–Cheatham monovalent (from `leaprc.water.tip3p`).**

The roadmap leaves the FF variant to the implementer (§12); this choice is
recorded in the manifest and driven by what is validated + available on the box.

## Ligand parameterization policy (fail-loud)

- **Design ligand** (`allow_am1bcc: true`): `antechamber` AM1-BCC (single-point,
  GAFF2) at the docked pose → `mol2` + `frcmod`.
- **Curated cofactor** (`allow_am1bcc: false`, `charges_mol2` set, or any
  high-risk cofactor per `md.charges.requires_fixed_charge_template`): the curated
  template (types + charges + atom order) is used, and the **pose heavy-atom
  coordinates are transplanted onto it by RDKit graph isomorphism**. The declared
  `net_charge` is asserted against the mol2 charge sum (`validate_mol2_net_charge`)
  — a mis-charged cofactor RAISES. There is **no on-the-fly AM1-BCC fallback** for
  a high-charge cofactor (the −4 ATP would hang `sqm`); this is the anti-s06 rule.

## Reactive-atom mapping (the correctness crux)

Reactive atoms are resolved with the **same SMARTS the NAC screen uses**
(`md.nac.identify_donor/acceptor/leaving`) against each ligand's reference graph,
then keyed to the ligand's mol2 atom names:

- **O_nuc** — `donor_smarts` on the design ligand → `:LIG@<name>`.
- **Pα** — `acceptor_smarts` on the cofactor → `:ATP@<name>`.
- **O_leaving** — the bridging O on Pα toward Pβ → `:ATP@<name>` (only when
  `transfer_is_h: false`, i.e. O→P attack).
- **Metal** — the ion residue → `:MG@MG`.

These use **residue-name masks**, which are numbering-independent.

**Catalytic residues** are the subtle case. `tleap` **renumbers residues 1..N**
in the prmtop, so a config residue number (original PDB frame) does not equal the
prmtop number when the structure has gaps. V6-1 maps them by **ordinal position**
in `protein_clean.pdb` (which preserves the original numbering) and **validates
the residue identity** (`S268` must actually be a SER); a mismatch is recorded as
`status: identity_mismatch`, never silently accepted. This bug was caught and
fixed during server validation (the naive number match returned THR at
"residue 268").

## Fail-loud guarantees

| Condition | Behavior |
|---|---|
| curated template missing (`mol2`/`.ref.sdf`) | `AmberBuildError(parameterization)` |
| curated mol2 net charge ≠ declared | `AmberBuildError(parameterization)` |
| pose graph-match incomplete | `AmberBuildError(parameterization)` |
| ligand not found in complex HETATM | `AmberBuildError(input)` |
| metal requested but absent | `AmberBuildError(input)` |
| tleap produced no prmtop | `AmberBuildError(tleap)` |
| a single-atom reactive mask selects ≠ 1 atom | `AmberBuildError(reactive_mapping)` |

## Reproducibility

`system_fingerprint` hashes (protein sequence, ordered ligand identities + net
charges + parameter sources, FF/water/ion choices, reactive spec). Two identical
builds of the CAR complex produced the **same fingerprint** `7f8e7306d4d9edd5`.

## Server validation — real CAR build

Built from `srcar_3hp_v5_e4a/md/mut_00000/mut_00000_anchored.pdb` (SrCAR A-domain
+ 3-HP + ATP + Mg):

| Metric | Value |
|---|---|
| System | 72 783 atoms (20 581 TIP3P waters, 31 Na⁺) |
| Net charge | −0.001 (neutralized) |
| Ligands | LIG(3-HP, −1, AM1-BCC) · ATP(−4, curated) |
| Metal | Mg²⁺ [12-6-4] `frcmod.ions234lm_1264_tip3p` |
| O_nuc | `:LIG@O2` → Amber #10958 |
| Pα | `:ATP@P1` → Amber #10982 |
| O_leaving | `:ATP@O5` → Amber #10985 |
| Mg | `:MG@MG` → Amber #10955 |
| Catalytic | SER268 / THR269 / LYS273 (`status: ok`) |

**Build integrity checks (cpptraj/parmed):**
- ATP internal P–O bonds physical: Pα–ester 1.60 Å, Pα–bridge 1.62 Å (not scrambled).
- **Transplant fidelity: built ATP(heavy)→catalytic = 10.09 Å, identical to the
  anchored input = 10.09 Å** (0.00 Å) — the pose is preserved exactly; the ATP–LIG
  distance is preserved to 0.05 Å.
- Reactive geometry (O_nuc→Pα = 7.4 Å, Mg 10 Å from O_nuc): correctly reflects the
  **raw anchored pose is not near-attack** — consistent with the E4a finding that
  the productive ~3 Å window is transient. Accessing it is V6-3's job (PMF), not
  the builder's.

## Success conditions — checklist

- [x] CAR A-domain + 3HP + ATP + Mg built as `prmtop`/`inpcrd`.
- [x] Reactive atoms mapped reproducibly: O_nuc, Pα, O_leaving, Mg, catalytic residues (by identity).
- [x] Charge, residue naming, atom count, parameter sources, ligand identities recorded (manifest).
- [x] Re-running the same config → same system fingerprint (`7f8e7306d4d9edd5`).
- [x] Ambiguous / missing parameters fail loudly (charge mismatch, missing template, ambiguous mask all raise).
