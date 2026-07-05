# CAR V5 — PR reconciliation (reviewer §1, Day 1)

Merged branch of record: **`feat/server-hardening`** @ `28efed4`.

| PR | Scope | State | Action taken |
|----|-------|-------|--------------|
| #2 | server smoke/focused launcher + acceptance judge + runbook | **MERGED** (`2bbe1fd`) | none — diff vs base is **empty** (fully incorporated). Reviewer's "still open" was stale UI state. |
| #3 | structure-level Mg²⁺ OpenMM injection (`MDResult.metal_setup`) | **MERGED** (`514bc21`) | none — diff vs base is **empty**. |
| #4 | s08 `low_ml_control` guard: fail only when `len(top) < len(candidates)` | **MERGED** (`a351da8`) | merged after `test_v5_s08_lanes` 4/4. |
| #5 | s10 `metal_requested` threading fix + Boltz `ccd:MG` co-fold | **MERGED** (`28efed4`) | scope-patched (below), retargeted onto `feat/server-hardening`, verified diff = metal-work only, merged. |

## PR #5 metal-parser-drop scope decision (reviewer §1.3 / §5.1)

Reviewer's concern was a **broad element drop** (`_METAL_ION_ELEMENTS = {MG,MN,ZN,CA,FE,…}` dropping
every HETATM metal), which would strip a target's own heme Fe / catalytic Zn — a generality risk.

**Finding: the shipped code was already scoped; the reviewer read a STALE TEST.** The committed
parser keys the drop on a **caller-supplied CCD** (`fold_only_metal`) — the specific ion EvoLiEZ
injected into *this* Boltz spec — validated against a `_KNOWN_METAL_CCDS` allowlist. A target's own
metal is never dropped (`fold_only_metal` is `None` off the co-fold path). The old
`test_v5_fold_metal.py` imported a non-existent `_METAL_ION_ELEMENTS` (so it **ERRORed at
collection** — PR #5's "0 new failures" was wrong) and asserted the old unconditional broad drop,
which is what made the code *look* broad.

**Resolution (commit `4e26228`):**
- Rewrote the broken test against the real API.
- **Tightened** the drop to fire only for a true monatomic ion (`resname == element == injected CCD`)
  → a polyatomic ligand sharing a metal residue code can never lose an atom.
- Added the requested regression tests: drops the declared fold-only Mg; keeps a metal when NOT
  declared fold-only; keeps a heme Fe (resname `HEM` ≠ injected CCD); keeps a metal-centred organic
  ligand (resname `LIG` ≠ injected CCD).

Verdict per reviewer's rule: `diff core code → reconcile explicitly` (done); the "broad drop" is
**not present** in merged code. Suite: **9 pre-existing failures, 0 new** (baselined by stash-compare).
