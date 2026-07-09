# CAR V7 deliverable — MANIFEST (canonical pointer)

**Target:** SrCAR A-domain → 3-HP adenylation (E5XP76, residues 1–720).
**Status:** computational triage only — **no activity / kcat / validated-lead claim** (ClaimGuard-enforced).

## ✅ USE THIS (canonical, 2026-07-09, protected two-layer panel)

| File | What it is |
|---|---|
| `README.md` | methodology, results, two-layer framing, round-1 objective |
| `v7_portfolio.csv` | **the experimental panel** (48 variants). First line is a `#`-caveat; each row has a `panel_layer` column |
| `v7_portfolio.html` | human-readable report (two layers, ClaimGuard-clean) |
| `v7_portfolio.json` | structured panel + bundle summary |
| `provenance/v7_tier_plan.json` | multi-fidelity allocator justification (why each expensive calc ran) |
| `provenance/md_candidates.json`, `provenance/validated_candidates.json` | s10 MD summary / s09 validation |

## ⚠️ Read before using the CSV

- **Layer A (statistical evidence-band candidates) is EMPTY.** No candidate reached a
  strong / significant / consensus band. This is a **calibration / mechanism-probe panel, not a
  computationally selected lead set.**
- **Layer B** carries the experimental variants: 28 mechanism-protected hypotheses (the V6 core
  lead `G430R;S433F;G407K` + full deconvolution, `P438*`/`G430*` pocket probes, alternative loop
  hypotheses), plus deconvolution / single-site / uncertainty probes and controls. Filter the CSV
  by `panel_layer` (`mechanism_protected` / `exploratory_probe` / `control`).
- **Round-1 objective = axis calibration**, not hit-finding. Use a separate A-domain adenylation
  assay + full-reduction assay. See README §6.

## Version history

- `archive/superseded_pre_protected/` — the earlier 2026-07-09 panel **before** the
  mechanism-protected lane was added (Layer B had no protected hypotheses; the V6 core lead was
  absent). Superseded — do not use.
