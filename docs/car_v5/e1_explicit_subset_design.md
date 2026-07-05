# E1 — explicit-solvent subset design (reviewer Day 3)

## Why
E3 showed the anchored productive pose (3.20 Å / 180°) collapses in **implicit GBSA** — even the WT
(which had Mg) diffused; the charged 3-HP⁻/ATP⁴⁻/Mg²⁺ cluster is not stabilised without explicit
water + counter-ions. E1 adds a real explicit-solvent path and asks: **does explicit solvent retain
the productive O→Pα geometry that implicit lost?**

## Implementation (openmm_engine.py — real, no longer a stub)
- `_ligand_system_generator_explicit`: `amber14-all.xml` + `amber14/tip3p.xml` (no `implicit/obc2.xml`),
  `periodic_forcefield_kwargs={nonbondedMethod: PME}`, `nonperiodic_forcefield_kwargs={NoCutoff}` for
  the ligand-alone probe. GAFF ladder + shared cache unchanged.
- In `_run_real`: `actual_solvent = "explicit"` when `cfg.solvent=="explicit"` **and** the multi-ligand
  build is active (single-ligand explicit warns + falls back). After protein+ligands+**Mg** are in the
  modeller, `modeller.addSolvent(sg.forcefield, model="tip3p", padding=1.0 nm, neutralize=True,
  ionicStrength=0.15 M)` → periodic topology → `create_system` uses PME.
- **Invariants preserved**: addSolvent appends water/ions AFTER the solute, so every solute atom index
  (mol_blocks, NAC reactive indices, the inserted Mg) is unchanged; the co-substrate retention
  restraint + NAC analysis stay valid. **Mg is added before solvation as an Amber ion — never OpenFF.**
  **O→P angle stays read-only — no angle restraint** (would manufacture NAC). RBFE off.

## Subset (`configs/car_v5_e1_subset_manifest.csv`)
WT · G430R;S433F;G407K (prior static lead) · P438N (clean single) · G430H;P438L;A275I (scalar control) ·
T265S;G274A;A275V (binding/stability control). 5 systems (WT + 4). No lead/activity claim — the prior
focused ranking is withdrawn (confounded); these are re-evaluated Mg-consistent + explicit.

## Configs / run
- `car_srcar_3hp_v5_explicit_smoke.yaml` (WT+P438N, 20 ps) → `run_car_v5.sh explicit-smoke` — Day 4
  build gate (system builds? water/ions present? Mg+ATP+3HP retained? angle finite? no crash?).
- `car_srcar_3hp_v5_explicit_subset.yaml` (WT+4, 0.3 ns, replicas 1) → `run_car_v5.sh explicit-subset`
  — Day 5. Start short; escalate ns/replicas only if stable.

## Co-substrate restraint decision (avoid a silent confound)
E1 keeps `restrain_cosubstrate: true` (distance-only flat-bottom, r0 = 3.6 Å, k = 2 kcal/mol/Å²),
**identical to the implicit baseline**. This is deliberate, not an oversight:
- It makes E1 a **controlled implicit-vs-explicit contrast** — the ONLY changed variable is the
  solvent model, so a retention/geometry difference is attributable to solvent.
- The restraint is **flat-bottom → inactive within 3.6 Å**. In implicit the co-substrate escaped
  *past* 3.6 Å despite it (retention 0.52), so if explicit keeps the pose inside the well, that is
  the **solvent** holding it, not the restraint.
- It is **distance-only on the co-substrate — NOT an angle restraint** (the O→Pα angle stays fully
  read-only), so it cannot manufacture NAC. It is a declared *retention screen*, which the review
  explicitly permits.
- It restores a **valid WT baseline** (the focused WT diffused → no baseline); a retained WT gives
  a meaningful reference for candidate ΔNAC.

If E1 passes with the restraint, a **restraint-OFF confirmation** on the top candidate is the clean
follow-up (strongest claim: explicit solvent holds the pose with no tether at all).

## Gating & verdict
- **Do the build smoke first**; only run the subset if it builds clean.
- PASS: WT baseline finite · Mg bridge tracked · O→P distance/angle finite · candidate-level
  differences interpretable · ClaimGuard clean. CONDITIONAL: finite but no discrimination → active-state
  reference ensemble / QM-MM-lite. FAIL: build unstable / Mg-ATP-3HP mapping fails / angle NaN.
- **Not done before an interpretable E1 result**: EvidenceCard ranking flip, wet-lab plate, full-18
  explicit run, RBFE, any "validated lead" claim.

## Local-verification limit
openmm is **not** in the local `.venv-light`, so the explicit MD build cannot be smoke-tested locally
(only AST + logic review were done). The `explicit-smoke` server run is the first real verification.
