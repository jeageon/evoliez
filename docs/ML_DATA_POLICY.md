# ML data policy

> **Boltz score is NOT an experimental label.** Use Boltz outputs as
> **features / sample weights / weak labels / filtering criteria**. Supervised
> labels must come from experiment: activity, kcat, Km, kcat/Km,
> thermostability, substrate specificity, expression.

This is enforced in code by `evoliez/ml/labels.py`
(`assert_supervised_label_allowed`), called wherever a column becomes the
training target `y`. A Boltz-derived column can never silently become a label.

## Boltz outputs and how they are used

| Boltz output | Role here |
|---|---|
| `confidence_score`, `ptm`, `iptm`, `ligand_iptm` | feature; pose reliability **sample weight** (s06b) |
| `complex_plddt`, `complex_iplddt` | feature; local-confidence weight |
| `complex_pde`, `complex_ipde` | feature; inverse-PDE edge weight |
| per-token pLDDT / PAE / PDE (`*.npz`) | pocket / catalytic confidence features |
| `affinity_pred_value`, `affinity_probability_binary` | **auxiliary** feature / weak ranking signal / binder filter — never a kcat/Km label |
| diffusion-sample ensemble | **contact frequency** (priority #1), pose consensus, ligand-RMSD variance |

Confidence-weighted contact (user §3.2):

```
edge_weight = contact_score × ligand_iptm × local_pLDDT × 1/(1+PDE)
```

implemented in `features/boltz_features.confidence_weighted_contact`.

## Datasets produced (`<run>/ml_datasets/`, spec §7)

| File | Row = | Notes |
|---|---|---|
| `pose_level.csv` | one Boltz diffusion sample | confidence/affinity metrics |
| `residue_level.csv` | one residue | evolutionary + pLDDT; weak `designable` label |
| `edge_level.csv` | one (residue, ligand atom) | **contact_frequency**, conf-weighted; weak `contact` label — the core dataset |
| `mutation_level.csv` | one mutation candidate | Boltz/delta features; `experimental_label` blank until measured |
| `variant_level.csv` | one variant | aggregate scores; `experimental_label` blank |
| `roles.json` | — | per-column role; `experimental_label` is the only supervised target |

## WT–mutant delta features (spec §6)

`features/delta.boltz_delta_features` emits `d_ligand_iptm`,
`d_complex_iplddt`, `d_complex_ipde`, `d_affinity_pred_value`,
`d_pocket_plddt`, `d_key_distance`, `d_contact_count`, … — all **features**.
In mock / no-per-mutant-Boltz mode they come from a documented disruptiveness
proxy (`s08._approx_mutant_complex`); a real per-mutant Boltz re-evaluation is
the server-side enhancement.

## Feature priority (user §8)

1. ligand atom–residue **contact frequency** across Boltz samples
2. ligand interface confidence (`ligand_iptm`, `complex_iplddt`, `complex_ipde`)
3. MSA permissiveness (conservation / entropy / PSSM / mutant-AA frequency)
4. WT–mutant delta features
5. catalytic-geometry preservation

## Confidence-aware usage (pLDDT / PAE / PDE / disorder)

pLDDT is per-residue *coordinate confidence*, not a flexibility label
("low pLDDT = risky to use as a fixed coordinate", which may be a flexible
loop OR poor prediction). These are features / edge weights only:

- `features/confidence.py` — residue pLDDT bins, window means/min, gradient,
  low-pLDDT segment length, fraction-low-nearby.
- `adapters/disorder.py` — IUPred2A/MobiDB-lite (real) or sequence proxy:
  separates "low pLDDT because IDR" from "low pLDDT because poorly predicted".
- `features/boltz_features.edge_confidence` — `contact_freq × norm_pLDDT_i ×
  ligand_iptm × exp(-ipDE/10)`; used as a **message-passing edge weight** in
  the confidence-aware EGNN (low-confidence relative vectors down-weighted).
- Coordinate-noise augmentation: σ = `coord_noise_min + α·(1 − pLDDT)` during
  GNN training so the model never overfits uncertain coordinates.
- Low-pLDDT policy: residues are dropped only if **also** >10 Å from the
  ligand and non-catalytic; active-site / pocket low-pLDDT loops are kept,
  flagged uncertain, and down-weighted (functional flexible loops survive).
- Self-supervised tasks F (coordinate reliability) and G (flexible-pocket
  risk) use pLDDT/disorder-derived **pseudo-labels** — still not experimental.
- Ranking deltas after mutant re-eval: `d_complex_iplddt`, `d_complex_ipde`,
  `d_pocket_plddt` flag structurally risky candidates.

## Never use as a label

`confidence_score`, `ligand_iptm`, `affinity_pred_value`, a single Boltz pose
contact, or predicted affinity as kcat/Km. Convert single-pose contacts to
**ensemble contact frequency**; keep Boltz as feature / weight / weak label /
filter only.
