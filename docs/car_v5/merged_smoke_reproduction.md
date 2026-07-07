# CAR V5 — merged-code smoke reproduction (reviewer §Day 2)

**Goal:** the reported smoke ran from rsync'd working files; confirm the verdict holds on the
**merged GitHub code** so result provenance is intact.

## Parity check
- Merged HEAD `feat/server-hardening` = `28efed4` (PR #4 + PR #5).
- The only code delta between the smoke's code (`5790ecc`) and merged HEAD is `boltz.py`
  (parser-scope commit `4e26228`) — **behavior-identical for CAR's monatomic Mg**
  (`resname == element == MG` still drops; nothing else changes).
- Server deploy `src/evoliez/adapters/boltz.py` sha256 == local merged sha256
  (`b30433a0…c9c2d`) → server runs merged code.
- `test_v5_fold_metal` 7/7 on the server under the merged file.

## Verdict re-judged on merged code
`scripts/check_car_v5_provenance.py runs/srcar_3hp_v5_smoke` → **CONDITIONAL_PASS** (see
`smoke_verdict_merged_code.json`). All Day-2 acceptance criteria hold:

| criterion | required | observed |
|-----------|----------|----------|
| verdict | PASS / CONDITIONAL_PASS | CONDITIONAL_PASS |
| `n_angle_nan` | 0 | 0 |
| `metal_status` | valid_metal_setup | valid_metal_setup |
| `openff_parameterized` | false | false |
| `n_angle_computed` | > 0 | 33 / 34 |
| `wt_angle_finite` | true | true |

`discriminates=false` is a 0.05 ns **sampling limitation**, not a failure — the focused 2 ns run
is the escalation. The focused run executes **entirely** on merged code (fresh from-scratch), so it
becomes the authoritative merged-code result; a byte-level re-run of the smoke is therefore not
separately needed.
