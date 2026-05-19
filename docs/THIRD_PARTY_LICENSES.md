# Third-party license audit

Why it matters: a dependency **imported in-process** can constrain the
license of *our* code; a dependency **invoked as a separate process** does
not (it only constrains bundling/redistribution and commercial use).

## A. In-process Python imports → constrain our code license

EvoLiEZ source imports only permissive libraries (verified: no
`import MDAnalysis / mdtraj / openbabel / pybel` anywhere in `src/`).

| Package | License |
|---|---|
| numpy, pandas, scipy, scikit-learn, networkx, rdkit | BSD-3 |
| biopython | Biopython License (BSD-style) |
| pydantic, SQLAlchemy, typer, rich, pyyaml | MIT |
| xgboost | Apache-2.0 |
| openmm (python pkg), openff-toolkit, openmmforcefields | MIT (OpenMM core MIT) |
| torch | BSD-3 |
| torch-geometric | MIT |

**Conclusion: the dependency closure is fully permissive → the license of
our own code is NOT constrained** (MIT / BSD / Apache-2.0 / proprietary all
viable).

Removed because they were never imported (would have added copyleft to the
closure): **MDAnalysis (GPL-2.0)**, **MDTraj (LGPL-2.1)**. MD metrics are
computed with numpy in `md/analysis.py`.

## B. External tools (subprocess only) → do NOT infect our code

Invoked via `utils.subprocess_utils.run`, never imported/linked/bundled.
They constrain only (1) redistributing a bundle that includes them and
(2) commercial use of the academic-only ones.

| Tool | License | Note |
|---|---|---|
| Boltz / Boltz-2 | MIT | permissive |
| LigandMPNN | MIT | permissive |
| DiffDock | MIT | permissive |
| AutoDock Vina | Apache-2.0 | permissive + patent grant |
| OpenMM | MIT | permissive |
| MAFFT | BSD-style (MAFFT license) | permissive |
| HMMER / jackhmmer | BSD-3 | permissive |
| BLAST+ | NCBI public domain (US Gov) | unrestricted |
| **GNINA** | **GPL-2.0** | copyleft — separate process OK; do NOT bundle into a non-GPL distribution |
| **MMseqs2** | **GPL-3.0** | as above |
| **Foldseek** | **GPL-3.0** | as above |
| **OpenBabel** (`obabel`) | **GPL-2.0** | as above |
| **PLIP** | **GPL-2.0** | as above |
| **FoldX** | **Proprietary, academic-only** | NO commercial use, NO redistribution. Optional (ML/mock fallback) |
| **Rosetta** | RosettaCommons (academic/non-commercial free; **commercial = paid**) | Optional (mock fallback) |
| **IUPred2A** | Academic-only, not OSS | Optional (sequence-proxy fallback) |
| MobiDB-lite | permissive (verify before bundling) | Optional |

## C. Practical conclusions

1. **Our code license is free to choose** — no in-process dep forces
   copyleft. Current choice: **Proprietary / All Rights Reserved**
   (unpublished research, private repo); Apache-2.0 is the planned option
   at publication.
2. **We never bundle external tools**; GPL ones (GNINA/MMseqs2/Foldseek/
   OpenBabel/PLIP) are user-installed subprocess deps → no GPL obligation on
   our code. Each adapter degrades to mock/alternative if the tool is absent.
3. **Commercial / redistribution caveat**: FoldX, Rosetta, IUPred2A are
   academic-only or separately licensed. Any commercial deployment or
   bundled redistribution must exclude them (all are optional with
   fallbacks) or obtain the appropriate license.
4. For eventual publication, **Apache-2.0** gives our code a patent
   grant/defense (relevant for enzyme-engineering IP); until then a private
   repo with no OSS grant is safest. **Confirm KAIST institutional IP /
   tech-transfer policy** before any public license — that is an
   institutional/legal decision, not a technical one.
