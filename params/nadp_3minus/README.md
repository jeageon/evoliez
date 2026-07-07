# NADP⁺ (−3) fixed-charge cofactor template

Fixed partial charges for the physiological **−3** NADP⁺ cofactor, used by the s10 MD
engine instead of on-the-fly AM1-BCC (which does not converge for this molecule — see
*Why* below). Consumed via `evoliez.md.charges` and the `LigandInput.charges_mol2`
config field.

| field | value |
|---|---|
| residue / role | NADP⁺ cofactor (`NADP_cofactor`) |
| net charge | **−3** |
| atom count | 73 (48 heavy + 25 H) |
| microspecies | oxidised NADP⁺ at pH 7: nicotinamide pyridinium (+1) + 4 deprotonated phosphate O⁻ → net −3 |
| charge method | AM1-BCC **single-point** (`antechamber -c bcc`, `sqm` `maxcyc=0`) at the Boltz bound pose |
| atom typing | GAFF2 (`-at gaff2`) |
| charge sum | −3.003 (BCC rounding; normalised to exactly −3 at transfer time) |

## Files
- `NAP.fixed.mol2` — AM1-BCC charges + GAFF2 atom types (the charge source).
- `_nadp_ref.sdf` — the reference graph (RDKit cannot read a GAFF2-typed mol2, so the
  graph is read from here; same atom order as the mol2, asserted element-by-element at
  load). Charges are mapped onto each pose's ligand by **graph isomorphism**, so the
  per-pose HETATM atom order does not matter.
- `../../scripts/_nadp_charges.sh` — regenerates `NAP.fixed.mol2` from `_nadp_ref.sdf`.

## Why a fixed template (not on-the-fly AM1-BCC)
The −3 NADP is tri-anionic and phosphate-rich. Under `antechamber`'s default gas-phase
geometry optimisation the `sqm` SCF crawls (the high negative charge density) and the
optimisation would distort the phosphate group anyway. Both the OpenMM
(`GAFFTemplateGenerator`) and Amber (`antechamber`) routes funnel through that same
`sqm` step, so neither could build the ligand and s10 silently skipped every NADP
candidate (`skipped_parameterization`). A single-point evaluation at the bound pose
(`maxcyc=0`) converges in ~1 min, is the right thing to do for a bound cofactor (no
gas-phase distortion), and is cached here so it never runs at MD time.

## Reproduce
```bash
bash scripts/_nadp_charges.sh        # needs AmberTools25 + the Boltz NADP _nadp_ref.sdf
# -> /tmp/nadp_amber/NAP.fixed.mol2 ; verify charge sum ≈ -3.0
```

## Known limitations / future work
- AM1-BCC, not RESP. AM1-BCC is the GAFF-standard small-molecule method (consistent with
  GAFF2 for the rest of the ligand); a published RESP/RESP2 NADP parameter set (e.g.
  Bryce/Manchester) is a future accuracy upgrade and can be dropped in as another
  `charges_mol2` with no code change.
- Charges derived at one bound pose; transferred to all poses by graph match (the BCC
  charges are conformer-insensitive at this level, so this is appropriate).
