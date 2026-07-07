# Accuracy & paper-readiness layers

Added per the user guidance. All toggled by `advanced:` in config (default on);
each degrades gracefully and **never produces a supervised label**
(`ml/labels.py` blocks `neg_*`, `mechanism*`, `ifp*`, `subfamily*`,
`specificity*`, `acquisition*`, `uncertainty*` from being targets).

## 1. Mechanism layer (priority #1)
`features/mechanism.py` + `adapters/mcsa.py`. Catalytic / acid-base /
nucleophile / metal / cofactor residue roles, reactive ligand atom(s), key
transfer distance, transition-state-like geometry score. Source: a
user-supplied M-CSA-style JSON (`advanced.mechanism_annotation_file`) or a
config/chemistry heuristic. Annotated in `s06`; per-mutant geometry deviation
feeds negative design in `s09`.

## 2. Ligand-atom importance (user §2)
`features/ligand_importance.py`. Reactive / charged / H-bond / aromatic /
hydrophobic atoms get decreasing weight; `s08` scales `interaction_gain` by
the importance of the contacted atom so the model learns catalytically
relevant geometry, not an unweighted average.

## 3. Standardized interaction fingerprint (priority #3)
`adapters/plip.py`. 8 standard non-covalent types (PLIP if installed, else a
geometry-rule fingerprint). Stored on `ctx` + `ifp_type_counts` in meta.

## 4. Negative design (user §4)
`ranking/negative_design.py`. Penalties: catalytic-residue mutation,
conserved-motif disruption, buried-core polar, catalytic-geometry loss,
over-binding (slow product release), pose inversion. Objective is **maintain
fold + catalytic geometry + useful (not over-tight) binding**, not "maximise
binding". Weighted in `ranking/score.py` (`ScoreWeights.neg_*`).

## 5. Subfamily-aware MSA (priority #4)
`features/subfamily.py`. Per position: subfamily conservation +
specificity-divergence (variable globally but fixed within a
cofactor/substrate subfamily ⇒ specificity-switch candidate). Feeds `s08`
features and a ranking bonus.

## 6. Benchmark suite (priority #5)
`ml/benchmark.py` + `evoliez bench -c cfg --benchmark file.csv`. Metrics:
beneficial recall@k, deleterious avoidance, Spearman vs activity, AUROC.
Example: `examples/fdh/benchmark.csv`. Writes `reports/benchmark.json`.

## 7. Calibration + active learning (priority #6)
`ml/calibration.py` — per-candidate uncertainty + recommendation class
(strong / uncertain / reject) + ECE/reliability for the benchmark.
`ml/active_learning.py` — acquisition = improvement + β·uncertainty +
γ·diversity − δ·cost; the focused library is **diversity-clustered**
(position / chemistry / ligand-atom target / subfamily), not raw top-N.

## 8. Provenance (user §16)
`io/provenance.py` → `reports/provenance.json` (sequence/ligand/config
hashes, versions, seed, GNN ckpt, ranking-formula version) + per-candidate
`provenance_id`, so any score is reproducible.

## Positioning

```
EvoLigand-GNN
 → Mechanism-aware → Confidence-aware → Subfamily-aware
 → Active-learning enzyme-family engineering platform
```
