# FDH example data

- `target.fasta` — illustrative NAD-dependent formate dehydrogenase fragment
  (401 aa).
- `ligand.smi` — NADP+ cofactor SMILES.
- `benchmark.csv` — **illustrative, sequence-consistent** benchmark for
  `evoliez bench`. NOT experimental truth.

## benchmark.csv provenance

Every row's wild-type letter matches `target.fasta` (E85, F84, V155, R285),
and the beneficial/neutral rows are mutations the default mock pipeline
actually generates, so `evoliez bench -c configs/example_fdh_nadp.yaml
--benchmark examples/fdh/benchmark.csv` has non-zero overlap and produces
meaningful recall@k / AUROC / calibration. Deleterious rows mutate the
catalytic residues (V155, R285) — they should rank low / be protected.

For real evaluation, replace this with experimental data (DMS / kcat / Km)
or a ProteinGym / FLIP enzyme subset — see `docs/BENCHMARKS.md` and
`evoliez.ml.benchmark.load_external_benchmark`.
