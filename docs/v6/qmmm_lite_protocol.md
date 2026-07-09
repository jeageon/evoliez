# V6-5 — QM/MM-Lite Protocol

**Phase:** V6-5 · **Module:** [`src/evoliez/amber/qmmm.py`](../../src/evoliez/amber/qmmm.py) · **Driver:** [`scripts/run_qmmm_lite.py`](../../scripts/run_qmmm_lite.py) · **Tests:** [`tests/test_amber_qmmm.py`](../../tests/test_amber_qmmm.py)

## Purpose

Escalate the **smallest necessary subset** — the reaction core — to an
electronic-structure-aware treatment, only when the classical force field cannot
answer the reaction-core question (ROADMAP_V6 §8). For CAR, V6-3 localized the
question to the **in-line attack angle** (never productive even when the umbrella
forces the near-attack distance); QM/MM-lite asks whether that geometric finding
survives an electronic-structure inspection.

This is **reaction-core plausibility evidence — not an activity, kcat, or barrier
claim.** It is single-frame (or few-frame) and semiempirical.

## QM region — chosen by the mechanism template

The QM region is the reactive substrate + cofactor + metal named by the mechanism,
not a hardcoded list:

| Mechanism | QM region | net charge |
|---|---|---|
| CAR adenylation | 3-HP nucleophile (`:LIG`) + ATP phosphates (`:ATP`) + bridging Mg (`:MG`) | −3 (−1 −4 +2) |
| FDH hydride | formate + nicotinamide ring | (from template) |
| TEM-1 acyl | Ser Oγ + scissile β-lactam carbonyl | (from template) |

Whole ligand/metal residues are taken, so the only QM/MM boundary is the
**non-covalent** ligand↔protein/solvent interface — **no link atoms** are needed.
The net charge is the sum of the residue formal charges (fail-loud if wrong: `sqm`
will not converge on a mis-charged region).

## Backend & protocol

- **Engine:** `sander` with `ifqnt=1` + the `&qmmm` namelist; `qm_theory='PM6'`
  (semiempirical `sqm`) for the *lite* tier. `quick` (ab-initio) is present on the
  server for a higher-cost escalation if PM6 is inconclusive.
- **Protocol:** a short QM/MM **minimization** of the reaction core from a
  **near-attack frame** (an umbrella window restart at ~3.1–3.2 Å O_nuc→Pα), with
  the **protein backbone restrained** (`@CA,C,N,O`, `restraint_wt=5`) so the core
  relaxes under QM while the fold is held. `qm_ewald=0` (cutoff QM/MM
  electrostatics) is the lite choice for a single-frame check.
- **Readout:** the QM/MM total energy + the **QM-relaxed reactive-core geometry**
  (O_nuc→Pα distance, in-line O_nuc–Pα–O_leaving angle, Mg–O_nuc / Mg–Pα) via
  cpptraj on the QM/MM-min restart.

## Interpretation rules (claim-safe)

- The QM/MM energy is **not** a barrier and **not** comparable across different
  systems as an activity ranking (different atom counts / environments).
- The meaningful, comparable observable is the **QM-relaxed in-line angle**: does
  the electronic-structure treatment reorganize the core toward a productive
  in-line approach, or does it keep the bent geometry the classical PMF found?
- If the QM-relaxed angle stays bent for WT **and** candidates, the classical
  finding **survives QM inspection** → the bottleneck is a genuine geometric
  feature, and a QM/MM *barrier* calculation is premature (the productive geometry
  is not reachable from the classical ensemble).
- **Decision rule:** if QM/MM-lite is inconclusive or merely confirms the classical
  geometry limit, escalate the **sampling** (2D PMF along distance + angle) or take
  the V6-4 mechanism-probe panel to wet-lab — do **not** report a QM/MM barrier or
  any activity interpretation. The concrete decision for CAR is in
  [`docs/car_v6/qmmm_lite_decision.md`](../car_v6/qmmm_lite_decision.md).

## Generality

Because the QM region, charge, and reactive masks come from the mechanism template
+ the V6-1 system manifest, the same `run_qmmm_lite` runs FDH or TEM-1 by pointing
at their manifest — no core edits.
