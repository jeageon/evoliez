# Server test plan — P0 cofactor guard + re-probe of the NADP "unparameterizable" diagnosis

Branch: `feat/p0-cofactor-guard` (parent: `feat/family-interaction-model`,
plus one cherry-picked commit `46f82a4` adding the cofactor declaration ↔
ligand-formula guard).

## TL;DR — run the bundled script

```bash
ssh <server>
cd ~/EvoLiEZ                            # or /mnt/data2/<you>/EvoLiEZ
git fetch origin && git checkout feat/p0-cofactor-guard && git pull --ff-only
conda activate evoliez                  # whichever env has openmm+openff+...
pip install -e . --quiet
bash scripts/server_test_p0_cofactor.sh
```

The script runs Steps 0–5 below, streams a structured log to
`runs/server_test/p0_cofactor_<ts>.log`, and prints a final **Summary**
block with exactly the lines to send back. Exit codes: `2` = MD stack
incomplete, `3` = guard sanity broken, `4` = resolver returned wrong
NADP+ formula, `0` = all expected outcomes (Step 4 VERDICT decides the
science direction). The Step-by-step recipe below documents what each
block does.

> **Convention**: every test branch ships a sibling
> `scripts/server_test_<topic>.sh`. The server only ever has to `git
> pull` + `bash` that one file; the per-branch script ages out with the
> branch.

## Why this test exists

Parent commit `4f14c82` ("Real MD: ligand-param probe → NADP-class
cofactor = neutral skip") concluded that **GAFF/AM1-BCC cannot
parameterize NADP**, based on a ligand-only `create_system` probe that
failed on the server. **That probe was run with the wrong molecule.** The
shipped configs declared `cofactor: NADP` but supplied an NAD+ SMILES
(`C21N7O14P2` / 44 heavy / -1, missing the 2′-phosphate; real NADP+ is
`C21N7O17P3` / 48 heavy / -3). So `_ligand_system_generator()` failed on
*NAD-like-but-mislabeled* chemistry, not on real NADP+.

The cherry-picked commit `46f82a4` adds a curated cofactor resolver +
`doctor` / `s01_input` guards that make this exact mismatch impossible
going forward. Every shipped config now sets `ligand.type=cofactor`,
`value: NADP`, `cofactor_redox: oxidized`, and the resolver yields the
real NADP+ SMILES at run time.

**This test re-asks the science question with the right molecule.**

## Step 0 — environment

```bash
ssh <server>
cd /mnt/data2/<you>/EvoLiEZ                 # NEVER root / or ~
git fetch origin
git checkout feat/p0-cofactor-guard
git pull --ff-only
# whichever conda env has openmm + openff + openmmforcefields + ambertools
conda activate <md-env>                      # e.g. .venv-md / evoligand-md
pip install -e . --quiet                     # pick up the new cofactors module
```

## Step 1 — P0 guard sanity (shipped configs are clean)

```bash
evoliez doctor -c configs/smoke.yaml
evoliez doctor -c configs/server_fdh_nadp.yaml
evoliez doctor -c configs/example_fdh_nadp.yaml
```

**Expected**: each report contains a line

```
ok      config:cofactor    matches NADP+
```

(plus the usual env / db / GPU checks). Any `block config:cofactor …
looks like NAD+` here means the cherry-pick didn't land — re-check
`git log --oneline -6`.

## Step 2 — P0 guard fail-fast (deliberate mismatch BLOCKs)

```bash
cp configs/smoke.yaml /tmp/smoke_bad.yaml
# put the OLD NAD+ SMILES back, under cofactor: NADP, on purpose:
python - <<'PY'
import re, pathlib
p = pathlib.Path("/tmp/smoke_bad.yaml")
s = p.read_text()
s = re.sub(r"type: cofactor\n\s+value: NADP",
           'type: smiles\n    value: "NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP'
           '([O-])(=O)OP([O-])(=O)OC[C@H]2O[C@@H](n3cnc4c3ncnc4N)[C@H](O)'
           '[C@@H]2O)[C@@H](O)[C@H]1O"', s, count=1)
p.write_text(s)
PY
evoliez doctor -c /tmp/smoke_bad.yaml | grep cofactor
```

**Expected** (the exact field bug, caught at preflight):

```
warn  config:cofactor  cofactor='NADP' (pH=7.4, redox=oxidized) expects NADP+
                       formula {'C':21,'N':7,'O':17,'P':3} (48 heavy) but the
                       parsed ligand is {'N':7,'C':21,'O':14,'P':2} (44 heavy);
                       looks like NAD+. Diff: O: expected 17 got 14, P: ...
```

(Same message becomes `BLOCK` if `backend: real` is added; smoke is `mock`.)

## Step 3 — resolver outputs the REAL NADP+ (formula sanity)

```bash
python - <<'PY'
from evoliez.config import load_config
from evoliez.features.cofactors import resolve_ligand_spec, formula_of
for path in ("configs/smoke.yaml", "configs/server_fdh_nadp.yaml"):
    cfg = load_config(path)
    eff = resolve_ligand_spec(cfg.input)
    print(path)
    print("  type :", eff.type)
    print("  smi  :", eff.value[:90] + "...")
    print("  form :", formula_of(eff.value), "n_heavy =",
          sum(formula_of(eff.value).values()))
PY
```

**Expected** for both configs:

```
  form : {'C': 21, 'N': 7, 'O': 17, 'P': 3} n_heavy = 48
```

If this still shows 44 / 2 P, the resolver wiring is off — stop and report.

## Step 4 — RE-PROBE the GAFF/espaloma ligand parameterization on REAL NADP+

This is the headline question. Parent's `_ligand_system_generator()` does
a ligand-only `create_system` per FF (`gaff-2.11`, then `espaloma-0.3.2`
if installed) and raises `_LigandParamUnsupported` if none works. Run it
against the RESOLVED NADP+ instead of the previous mis-shipped NAD+:

```bash
python - <<'PY'
from pathlib import Path
import tempfile
from evoliez.config import load_config
from evoliez.features.cofactors import resolve_cofactor
from evoliez.adapters.openmm_engine import (
    _ligand_offmol_at_pose, _ligand_system_generator,
    _LigandParamUnsupported,
)

# Synthesize a minimal PDB carrying NADP+ heavy atoms with CONECT so
# _ligand_offmol_at_pose() runs without a real Boltz output. Embed via
# RDKit, write HETATM + CONECT, then build the OpenFF mol + probe FFs.
from rdkit import Chem
from rdkit.Chem import AllChem

smi = resolve_cofactor("NADP", redox_state="oxidized").smiles
m = Chem.AddHs(Chem.MolFromSmiles(smi))
AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE)
AllChem.MMFFOptimizeMolecule(m)
pdb = Chem.MolToPDBBlock(m, flavor=4)            # HETATM + CONECT
work = Path(tempfile.mkdtemp(prefix="probe_"))
pdb_path = work / "nadp.pdb"
# rewrite ATOM->HETATM and force resname LIG so the parser is happy
out = []
for ln in pdb.splitlines():
    if ln.startswith(("ATOM  ", "HETATM")):
        ln = "HETATM" + ln[6:17] + "LIG" + ln[20:]
    out.append(ln)
pdb_path.write_text("\n".join(out) + "\n")
print(">> built NADP+ pose PDB at", pdb_path)

off = _ligand_offmol_at_pose(pdb_path, smi)
print(">> off molecule:", off.n_atoms, "atoms (incl. H)")

try:
    sg = _ligand_system_generator(off, work)
    print(">> PARAMETERIZED OK with", sg, "- the original 'NADP unparameterizable'")
    print("   diagnosis was a wrong-SMILES artifact. neutral-skip can be lifted.")
except _LigandParamUnsupported as exc:
    print(">> still unparameterizable on REAL NADP+:", exc)
    print("   genuine GAFF/espaloma limitation; curated-parameter path (tleap")
    print("   /prmtop) is now the next layer to wire in.")
PY
```

**Interpretation matrix:**

| Outcome | What it means | Next step |
|---|---|---|
| `PARAMETERIZED OK with gaff-2.11` | The original "NADP-unparameterizable" diagnosis was the wrong-SMILES artifact. GAFF handles real NADP+. | Remove the `_LigandParamUnsupported` neutral skip for NADP-class on parent's path (or scope it to only fire when the probe actually fails). Re-baseline `_run_real` to STATUS=ok for NADP. |
| `PARAMETERIZED OK with espaloma-0.3.2` | GAFF still struggles, but espaloma (already wired in parent) handles real NADP+. | Keep the probe; espaloma is now the production FF for NAD(P)-class. Make sure the server conda env has `espaloma` installed (parent's probe gates on it). |
| `still unparameterizable on REAL NADP+` | The diagnosis was directionally right even on the corrected molecule. | Stand up the curated-parameter / tleap → prmtop branch you proposed. Until then `skipped_parameterization` is the honest answer (which is what `4f14c82` already does). |

## Step 5 — end-to-end on a real WT Boltz PDB

```bash
# point at an existing real WT Boltz output (whatever the smoke produced
# previously, e.g. /mnt/data2/<you>/runs/.../complexes/wt_boltz_model_1.pdb)
WT_PDB=/mnt/data2/<you>/.../wt_boltz_model_1.pdb
python scripts/check_real_md.py "$WT_PDB" configs/smoke.yaml
```

**Expected reads:**
- `status     : ok` or `status : unstable` → real MD actually ran on real
  NADP+; analyse() produced metrics. The whole bring-up is unblocked.
- `status     : skipped_parameterization` → consistent with Step 4
  returning `still unparameterizable`. Means the script honestly didn't
  run MD; **NOT** a `VALIDATED`. Curated-parameter path is the next P0.
- `status     : failed` → genuine MD failure (not FF). Read
  `failure_reason`; this is now the next bug to chase.

The pipeline-level `bash scripts/server_smoke.sh md` (per-mutant) is
unchanged by this branch except that it now resolves NADP+ correctly; if
Step 5 returns `ok`/`unstable`, it should also start producing real MD
results.

## What to send back

For each of Steps 1–5, paste the relevant lines (the `cofactor:` doctor
row, the resolver formula print, the Step-4 verdict line, the Step-5
status/failure block). That uniquely determines which interpretation row
above applies and what the next code change is.
