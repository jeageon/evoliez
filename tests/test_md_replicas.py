"""P0.6: final-tier replica loop + 'MD ran vs not run' explicit signal.

These are source-level + light-env checks for s10_md's new behaviour:
- top-25%-by-rank candidates get `final_tier_replicas` runs; the rest
  get `replicas`.
- per-replica RMSD series are stashed on the MDResult so the reporter
  can compute replica-spread confidence intervals.
- candidate.scores carry `md_did_run` distinguishing "MD passed" from
  "MD didn't run" (skipped_parameterization / skipped_no_full_atom /
  failed). The plan calls this out explicitly: "Final report
  distinguishes 'MD passed' from 'MD not run'."
"""

from __future__ import annotations

import inspect

from evoliez.stages import s10_md


def test_s10_runs_final_tier_replicas_for_top_quarter():
    src = inspect.getsource(s10_md.MDStage.run)
    # Replica logic: separate counts for final-tier vs rest, looped over
    # per candidate, aggregated into a primary result with per-replica
    # series.
    assert "final_tier_replicas" in src
    assert "n_final_tier" in src
    assert "replica_results" in src
    assert "ligand_rmsd_replicas" in src
    assert "pocket_rmsd_replicas" in src


def test_s10_persists_md_did_run_distinct_from_md_passed():
    src = inspect.getsource(s10_md.MDStage.run)
    # The MD-ran flag is its own column on both scores and details so the
    # report can render "Skipped (param)" / "Skipped (no full-atom)" /
    # "Failed" / "Passed" without re-deriving from status strings.
    assert "md_did_run" in src
    assert 'cand.scores["md_did_run"]' in src
    assert 'cand.details["md_did_run"]' in src


def test_s10_persists_replica_count_meta():
    src = inspect.getsource(s10_md.MDStage.run)
    assert "n_md_replicated" in src
    assert "n_md_final_tier" in src
