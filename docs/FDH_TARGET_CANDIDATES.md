# Real FDH target candidates for first production run

After Step 1 (P0 line) production-validated end-to-end on illustrative
target (`>fdh_example NAD-dependent formate dehydrogenase (illustrative
fragment)`), the first REAL production run needs a verified target.
`evoliez doctor` currently BLOCKs the real backend on the illustrative
placeholder for this exact reason.

## Comparison

| Species | UniProt | Length | WT cofactor | Switch precedent | Why interesting |
|---|---|---|---|---|---|
| ***Pseudomonas* sp. 101** (PseFDH) | P33160 | 401 aa | NAD+ | Tishkov 2003 (D196H + others) — **gold standard** | Most-engineered FDH for NADP switch; rich literature; crystal structures (1AQ1, 2NAC) |
| ***Candida boidinii*** (CbFDH) | O13437 (UniProt) | 364 aa | NAD+ | Andreadeli 2008 (5-mutation NADP variant), Hoelsch 2013 | Commercial NAD regeneration enzyme; widely commercialized |
| ***Burkholderia stabilis*** (BsFDH) | A0A0E1WZG3 / Q70CV6 | 400 aa | NAD+ | Hoelsch 2013 (defined NADP switch) | More recent, cleaner switching mutations |
| ***Saccharomyces cerevisiae*** (ScFDH) | Q08911 | 376 aa | NAD+ | Less engineered | Eukaryotic, model organism, smaller engineering history |
| ***Mycobacterium vaccae* N10** (MvFDH) | P46154 | 401 aa | NAD+ | Tishkov group variants | Highly thermostable; second-best engineered |

## Recommendation

**For first real run: PseFDH (P33160).** Reasons:
1. Same length (401 aa) as our illustrative placeholder — config catalytic position numbering may transfer with minimal renumbering.
2. Decades of NAD→NADP engineering literature (Tishkov, Holmberg, Pollegioni) — easy to build retrospective benchmark.
3. Crystal structures available for Boltz template / pocket-constraint verification.
4. Standard catalytic / NAD-binding residues well-defined:
   - **Catalytic**: His332, Asn146 (or species-equivalents)
   - **NAD specificity determinants**: Asp195 (turn → NADP affinity loss), Tyr196
   - **Cofactor binding loop**: 195–198 region

**Known NADP-switching mutations** (good retrospective benchmark seed):
- D196S (the canonical single-mutation switch — Tishkov 2003)
- Y196H (additional polar contact)
- D196S/Y196H double (better NADP/NAD ratio)
- Plus the Holmberg/Bocanegra extended sets

## Critical caveats — what I CAN'T verify here

I have no web/UniProt access from this dev box. The accession numbers
and residue positions above are from training data and need to be
verified before going into the config:

- **UniProt FASTA fetch**: do this on the server with curl (one-line).
- **Residue numbering**: PDB 2NAC vs UniProt FASTA may differ by 1 (signal peptide vs not). Verify against the actual sequence before setting `catalytic_residues`.
- **NAD specificity residue position**: cited as 195/196 in much of the literature, but the canonical Tishkov D→S mutation is sometimes written as D196 (1-based PDB) vs D195 (mature protein). The literature 1-based number → fasta 1-based number alignment must be checked.

## Once you pick, the config change is a single commit

The handoff is:
1. You confirm species + accession.
2. I write a tiny helper: `scripts/fetch_target_fasta.sh <accession>` that curls UniProt → `examples/<species>/target.fasta`, validates length matches expectation, and prints the residue at each catalytic position so you can sanity-check numbering.
3. Update `configs/server_fdh_nadp.yaml`:
   - `target_fasta` → new fasta
   - `catalytic_residues` → verified (e.g., `["H332", "N146"]`)
   - `fixed_residues` → catalytic + cofactor-binding stable positions
   - `known_binding_site` → NAD/NADP pocket residues for Boltz steering
4. `evoliez doctor -c configs/server_fdh_nadp.yaml` — should now report ALL OK, no BLOCK.
5. First real production run: ~12–24 hours GPU wall.

Until step 4 passes I will not start the production run.
