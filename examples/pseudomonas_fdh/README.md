# PseFDH benchmark scaffold

This directory holds a **literature-derived benchmark scaffold** for
*Pseudomonas* sp. 101 formate dehydrogenase (PseFDH, UniProt **P33160**,
401 aa) — the gold-standard NAD→NADP cofactor-switching target picked in
[`docs/FDH_TARGET_CANDIDATES.md`](../../docs/FDH_TARGET_CANDIDATES.md).

This is **not** an experimental benchmark. It is a small scaffold seeded
from a single repo doc and the curated binding-site annotations already
shipped in [`configs/server_fdh_nadp.yaml`](../../configs/server_fdh_nadp.yaml).
The contract: every mutation in `benchmark.csv` MUST cite a source already
present in this repo. New mutations should be added with their citation
inline (or a pointer to a published DMS / kcat / Km table).

## Numbering convention — read this first

All positions use **UniProt P33160 1-based numbering** (the mature-protein
sequence in `target.fasta` once you fetch it via
`scripts/fetch_target_fasta.sh P33160`). Most older PseFDH papers (Tishkov,
Holmberg, Pollegioni, Bocanegra) use PDB-derived numbering for crystal
structures 1AQ1 / 2NAC, which can differ by a few residues from UniProt
(signal peptide vs. mature protein). Where the doc citation uses a
different number, the CSV row carries the UniProt-numbering position AND
the literature label is noted alongside in this README.

If you copy a row from a paper without checking numbering you will end up
benchmarking the wrong residue.

## Sources used for the current scaffold

| Mutation | Label | Source in this repo |
|---|---|---|
| `D222S` | beneficial | `configs/server_fdh_nadp.yaml` — D222 is the documented NADP-specificity-switch target (UniProt-numbered equivalent of literature D196S, Tishkov 2003) |
| `D222H` | beneficial | same docstring — Tishkov-class D→H variant (literature D196H) |
| `D222A` | beneficial | same docstring — Tishkov-class D→A variant (literature D196A) |
| `R285E` | deleterious | `configs/server_fdh_nadp.yaml` — R285 is annotated as catalytic; charge-flip kills activity |
| `H333A` | deleterious | `configs/server_fdh_nadp.yaml` — H333 is annotated as catalytic; ablation kills activity |

The three `D222X` rows are the **same Tishkov-class single-residue switch**
written in this repo's UniProt P33160 numbering. The two deleterious rows
exercise the catalytic-residue avoidance metric. That's the whole scaffold.

## Validate sequence consistency before using

Once you've fetched the FASTA (`scripts/fetch_target_fasta.sh P33160 >
examples/pseudomonas_fdh/target.fasta`):

```
evoliez bench -c configs/server_fdh_nadp.yaml \
  --benchmark examples/pseudomonas_fdh/benchmark.csv \
  --allow-no-overlap
```

`evoliez bench` runs `validate_benchmark`, which checks that every WT
letter (e.g., the `D` in `D222S`) actually matches the corresponding
position in `target.fasta`. A mismatch means the row is on the wrong
numbering scheme — fix the row (or the FASTA), do not silently override.

`--allow-no-overlap` is here only so the harness still produces a report
when the mock pipeline doesn't sample these specific positions. In a real
run you want the overlap >= 1.

## Extending this scaffold (please do)

The expected workflow is:

1. Find a published PseFDH mutation with a measured activity / Km / kcat
   ratio (Tishkov, Holmberg, Pollegioni, Bocanegra are the obvious starts;
   see `docs/FDH_TARGET_CANDIDATES.md` for the canonical citation list).
2. **Translate the literature position to UniProt P33160 numbering**
   using the fetched `target.fasta`. The most reliable check:
   `python -c "s=open('target.fasta').read().split('\\n',1)[1].replace('\\n','');
   print(s[pos-1])"` should return the WT letter from the paper.
3. Append a row to `benchmark.csv` (`mutation,label,activity`).
4. Append a row to the table above with the citation.
5. Re-run `evoliez bench` and confirm `validate_benchmark` reports zero
   WT-letter mismatches.

Do **not** add rows that aren't grounded in either a published source or
an annotation already in this repo's docs/configs. The illustrative
benchmark for the placeholder fragment lives in `examples/fdh/` — that
file documents which rows are demonstrative vs. literature-grounded.

## See also

- [`docs/FDH_TARGET_CANDIDATES.md`](../../docs/FDH_TARGET_CANDIDATES.md) — target selection rationale + canonical literature pointers.
- [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) — full benchmark harness, ProteinGym / FLIP import path, baseline + ablation flags.
- [`configs/server_fdh_nadp.yaml`](../../configs/server_fdh_nadp.yaml) — server-grade config; provides the annotated catalytic / binding-site residues this scaffold is keyed to.
- [`examples/fdh/`](../fdh/) — illustrative-fragment benchmark used by the mock CI run.
