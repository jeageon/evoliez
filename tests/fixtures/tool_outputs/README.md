# Real tool-output fixtures

Small samples of the **actual output formats** produced by the external tools
EvoLiEZ shells out to. They pin the `real`-backend parser contracts without
needing the tools/GPU installed (expert review: tool-validation datasets).

| File | Tool | Parser under test |
|---|---|---|
| `boltz/model_0.pdb` | Boltz (`--output_format pdb`) | `boltz._parse_real_structure` |
| `boltz/pred.cif` | Boltz (default mmCIF) | `boltz._parse_cif_atoms` |
| `boltz/confidence_model_0.json` | Boltz | `boltz._parse_real_samples` |
| `boltz/affinity.json` | Boltz-2 affinity | `boltz._parse_real_samples` |
| `boltz/plddt_model_0.npz` | Boltz | `boltz._load_plddt` |
| `vina/out.pdbqt` | AutoDock Vina | `vina._parse_vina` |
| `gnina/out.sdf` | GNINA | `gnina._parse_gnina` |
| `diffdock/rank1_confidence-0.42.sdf` | DiffDock | `diffdock._parse_diffdock` |
| `msa/blast.m8` | BLASTp outfmt 6 | `msa_tools._parse_blast_m8` |
| `msa/mmseqs.m8` | MMseqs2 easy-search | `msa_tools._parse_mmseqs_m8` |
| `msa/jackhmmer.sto` | jackhmmer `-A` | `msa_tools._parse_stockholm` |
| `foldx/Dif_x.fxout` | FoldX BuildModel | `foldx._parse_foldx` |
| `rosetta/x.ddg` | Rosetta cartesian_ddg | `rosetta._parse_rosetta` |

These are illustrative minimal samples, not real predictions.
