# Curated AMBER cofactor parameters (Bryce Lab / Manchester DB)

This directory holds `.lib` / `.frcmod` / `.mol2` files for the redox
cofactors NAD+ / NADH / NADP+ / NADPH, used by
`evoliez.adapters.openmm_engine._curated_param_system_generator` to bypass
GAFF/AM1-BCC on the routine MD setup path.

Why this exists: the server probe in
`scripts/server_test_p0_cofactor.sh` Step 4 confirmed - on the corrected
real NADP+ molecule (P3/O17/48-heavy, charge -3, not the previously
mis-shipped NAD+) - that **AM1-BCC hard-fails** on real NADP+:

```
antechamber: Fatal Error! Cannot properly run "sqm -O -i sqm.in -o sqm.out"
... CalledProcessError exit status 1 ...
```

Three OFF charge backends all refused: RDKit (no am1bcc), AmberTools
(sqm crash), Built-in (no am1bcc). Curated parameters are the standard
production answer for multiply-phosphorylated cofactors that GAFF/AM1-BCC
cannot type.

## Get the files

```bash
bash scripts/fetch_amber_cofactors.sh
```

The script reads `MANIFEST.yaml` here and downloads each `lib/frcmod/mol2`
into this directory. If the Manchester URLs change, edit `MANIFEST.yaml`
or pass a custom one:

```bash
bash scripts/fetch_amber_cofactors.sh --manifest /path/to/custom_manifest.yaml
```

Or just drop the files into this directory manually (e.g. if you already
have them from a previous project) - `_curated_param_system_generator`
only checks for existence.

Override the search location entirely with:

```bash
export EVOLIEZ_AMBER_PARAMS=/mnt/data2/shared/amber_cofactors
```

## What lands here

Per `MANIFEST.yaml`:

| residue | species  | files                               |
|---------|----------|-------------------------------------|
| `NAD`   | NAD+     | `NAD.lib`, `NAD.frcmod`, `NAD.mol2` |
| `NDH`   | NADH     | `NDH.lib`, `NDH.frcmod`, `NDH.mol2` |
| `NAP`   | NADP+    | `NAP.lib`, `NAP.frcmod`, `NAP.mol2` |
| `NDP`   | NADPH    | `NDP.lib`, `NDP.frcmod`, `NDP.mol2` |

These three-letter residue codes match the AMBER convention and the
Bryce Lab download names. Verify the headers of the `.lib` files
(`!entry.<RESIDUE>.unit.atoms`) match — if a download uses a different
internal residue name, update `CofactorSpec.amber_residue_name` in
`src/evoliez/features/cofactors.py` to match.

## License + citation

These parameters are distributed by the Manchester group for academic
use. Cite (at minimum):

> Walker, R.C.; de Souza, M.M.; Mercer, I.P.; Gould, I.R.; Klein, D.R.
> (2007) *J. Phys. Chem. B* 111, 4188-4204 — the implementation paper.
>
> Pavelites, J.J.; Gao, J.; Bash, P.A.; Mackerell, A.D. (1997) *J. Comput.
> Chem.* 18, 221-239 — the underlying parameter derivation.

See <http://amber.manchester.ac.uk/> for the current bibliography and
any updated parameter releases.

## .gitignore

`*.frcmod`, `*.lib`, `*.mol2` are git-ignored so the licensed files
never get accidentally pushed to a public mirror. Only `README.md`,
`MANIFEST.yaml`, and `.gitignore` are tracked.
