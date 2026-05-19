# Benchmarking EvoLiEZ

`evoliez bench` is a pipeline + metric harness, not bundled experimental
truth. Layers, mirroring the literature:

## 1. Built-in (illustrative)
`examples/fdh/benchmark.csv` — sequence-consistent, overlaps the mock run.
Metrics: beneficial recall@k, deleterious avoidance, Spearman vs activity,
AUROC, catalytic-protection rate, binding-site enrichment, calibration
(reliability + ECE). `run_benchmark` reports `overlap`/`valid`/`warnings`;
the CLI fails on zero overlap (`--allow-no-overlap` to override).

## 2. Public mutation-fitness sets (user-supplied)
`evoliez.ml.benchmark.load_external_benchmark(path, fmt=...)` maps a
ProteinGym DMS CSV (`mutant`,`DMS_score`) or FLIP-style CSV into our rows;
continuous scores are bucketed by quantile. Multi-mutants `:` → `;`.
CLI: `evoliez bench --benchmark <file> --external-format proteingym`.
Download data from ProteinGym / FLIP yourself (not bundled).

- ProteinGym: https://proteingym.org
- FLIP: https://flip.protein.properties

## 3. Baselines
`--baselines` scores random / conservation-only / MSA-only /
interaction-only orderings next to the full model, so any gain is
attributable (expert review). `--ablation` disables each accuracy layer and
reports the AUROC delta.

## 4. Pose / structural validity
`evoliez.ml.pose_validity.pose_sanity` is a cheap geometric stand-in
(clash-free, ligand-in-pocket). Real physical validity needs **PoseBusters**
and PDBbind/Astex redocking sets (external package + data):

- PoseBusters: https://github.com/maabuu/posebusters
- PDBbind / Astex Diverse: standard redocking benchmarks

## 5. Tool-output contracts
`tests/fixtures/tool_outputs/` + `tests/test_tool_output_parsers.py` pin the
real Boltz/Vina/GNINA/DiffDock/MMseqs/BLAST/jackhmmer/FoldX/Rosetta parsers
against real output formats without the tools installed.

## Honest status
Mock orchestration, role policy, parser contracts and the metric harness are
covered. Real GPU-tool runtime and large public-benchmark campaigns require
the server + downloaded datasets and are run there, not in CI.
