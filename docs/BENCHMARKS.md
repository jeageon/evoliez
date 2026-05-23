# Benchmarking EvoLiEZ

`evoliez bench` is a pipeline + metric harness, not bundled experimental
truth. Layers, mirroring the literature:

## 1. Built-in (illustrative)
`examples/fdh/benchmark.csv` — sequence-consistent, overlaps the mock run.
Metrics: beneficial recall@k, deleterious avoidance, Spearman vs activity,
AUROC, catalytic-protection rate, binding-site enrichment, calibration
(reliability + ECE). `run_benchmark` reports `overlap`/`valid`/`warnings`;
the CLI fails on zero overlap (`--allow-no-overlap` to override).

### 1a. Literature-derived multi-enzyme benchmark suite

Five enzyme cards covering complementary chemistries; each card is a
self-contained `examples/<slug>/{benchmark.csv,README.md}` pair.
Every CSV row cites a published source (PMID/DOI/PMC) — no fabricated
labels. Contract tests (`tests/test_multi_enzyme_benchmarks.py`) pin
the cross-enzyme schema: ≥3 beneficial + ≥3 deleterious rows per card,
all rows in the `beneficial/neutral/deleterious` vocabulary, every
non-neutral mutation mentioned in the README so the citation chain is
auditable in one place.

| Card | Enzyme | UniProt | Why in the panel |
|---|---|---|---|
| [`pseudomonas_fdh/`](../examples/pseudomonas_fdh/)        | PseFDH       | P33160 | NAD→NADP cofactor switch (positive control; matches `configs/server_fdh_nadp.yaml`) |
| [`xylose_reductase/`](../examples/xylose_reductase/)      | XR (C. tenuis / P. stipitis) | O74237 / P31867 | NADPH→NADH cofactor switch (sibling direction to PseFDH) |
| [`tem1_betalactamase/`](../examples/tem1_betalactamase/)  | TEM-1        | P62593 | No-cofactor case; Stiffler 2015 DMS labels + clean ESBL beneficial gateways |
| [`beta_glucosidase_bgl3/`](../examples/beta_glucosidase_bgl3/) | Bgl3 (GH1) | P22073 (structural ref) | Largest published enzymatic-activity DMS (Romero 2015) |
| [`p450_bm3/`](../examples/p450_bm3/)                      | P450 BM3     | P14779 | Stress test: substrate promiscuity + heme + I-helix; recall@K expected lower |

**Suggested execution order (multi-enzyme validation):**

1. P0 fixes verified end-to-end on PseFDH (ranking gate, MD time-series).
2. For each new card, run `evoliez bench --allow-no-overlap` first to
   confirm the card loads and reports zero schema warnings.
3. Cheap run per enzyme (Boltz real, MD = mock or top-5 real MD) →
   check recall@10, recall@30, Reject/invalid top-leakage, MD
   failure/timeout rate. If those numbers look right, go to step 4.
4. Full production run, one enzyme at a time, with output diffed
   against the cheap-run snapshot.
5. Cross-enzyme summary (median recall@K, catalytic-avoidance rate)
   reported in `reports/multi_enzyme_summary.md`.

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
