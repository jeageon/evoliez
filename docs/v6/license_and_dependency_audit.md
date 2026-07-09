# V6-7 — License & Dependency Audit

The V6 platform's dependencies, their licenses, and how they are used. Goal: a
defensible, redistributable core, with the one licensed component (Amber `pmemd`)
clearly isolated and covered by the user's academic license.

## Python packages

| Package | License | Use |
|---|---|---|
| numpy, scipy | BSD-3 | numerics, WHAM |
| pandas | BSD-3 | tabular I/O |
| pyyaml | MIT | config |
| pydantic (v2) | MIT | config schema, EvidenceCardV4 |
| SQLAlchemy | MIT | run DB |
| networkx | BSD-3 | interaction graphs |
| typer, rich | MIT | CLI |
| **RDKit** | BSD-3 | ligand graphs, SMARTS, reactive-atom resolution |
| biopython | BSD-3-like | sequence/structure I/O |
| scikit-learn | BSD-3 | ML priors |
| xgboost | Apache-2.0 | reranker |
| OpenMM | MIT | smoke/regression MD (non-production tier) |

All permissive (BSD/MIT/Apache) — the Python core is freely redistributable.

## Structure / simulation backends (server tools, not pip)

| Tool | License | Use in V6 | Notes |
|---|---|---|---|
| **AmberTools** (tleap, antechamber, parmchk2, cpptraj, sqm, MCPB.py, parmed, MMPBSA.py) | free (open) | V6-1 build, V6-2/3 analysis, V6-5 QM/MM prep | freely available |
| **Amber `pmemd` / `pmemd.cuda`** | **Amber academic license** (paid/registered) | V6-2/3 production GPU MD, PMF | ⚠️ the one licensed component; the server has a licensed build; NOT redistributed by this repo |
| **`sqm`** (semiempirical QM) | part of AmberTools (free) | V6-5 QM/MM-lite (PM6) | |
| **`quick`** (ab-initio QM) | GPL / free | V6-5 escalation option (present, not required) | |
| **Boltz** | MIT | V6-6 structure prediction (TEM-1 Michaelis complex) | weights cached in `$BOLTZ_CACHE` |
| gnina / DiffDock / Vina | Apache-2.0 / MIT / Apache-2.0 | docking (upstream stages) | |
| ThermoMPNN | open-source (MIT-style) | s09 ΔΔG (FoldX/Rosetta-free) | |

## License posture

- **Redistributable core**: all Python code + the OSS tools (RDKit, OpenMM, Boltz,
  AmberTools) are permissive. The repo ships no proprietary code.
- **Isolated licensed dependency**: only Amber `pmemd.cuda` requires the Amber
  academic license. It is invoked as an **external binary** (resolved via
  `EVOLIEZ_PMEMD_CUDA`), never vendored. A site without an Amber license can still
  run the entire build/analysis/QM-prep chain (AmberTools) and the OpenMM smoke tier;
  only the production GPU MD/PMF tiers need the licensed `pmemd`.
- **No license leakage into claims**: the licensing status does not affect claim
  safety — evidence tiers and ClaimGuard are independent of the backend.

## Action items (owners)

- Pin exact tool versions in the runtime profile (V6-0 already records tool paths +
  version strings; extend to hash the binaries for a rigorous, high-stakes run). Owner: V6-0.
- Confirm the site Amber license covers the `pmemd24` build in use. Owner: user.
