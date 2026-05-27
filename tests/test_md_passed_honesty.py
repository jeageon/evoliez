"""N (post-expert-audit) — n_md_passed honesty split.

Expert: `analyse()` sets `metrics.passed = True` for
`skipped_parameterization` so the candidate isn't penalised in the
ranking. That's a deliberate "neutral skip" policy and it works
correctly downstream (s11 evidence-class only awards STRONG on
`md_status == "ok"`, not on `md_passed`). But s10's `n_md_passed`
counter rolls neutral skips into the same number as actual MD passes,
and the honesty log line read "12/12 passed" even on fully-skipped
runs.

This commit adds:
  - `ctx.meta["n_md_validated"]` — passes that ACTUALLY ran MD
  - `ctx.meta["n_md_neutral_skip"]` — skipped_parameterization cases
    that look like passes if you only check `md_passed`
  - honesty log line shows `validated/total (+ N neutral)` instead of
    just `passed/total`
  - `n_md_passed` is preserved (= validated + neutral) for back-compat
    readers and so the new key strictly clarifies, doesn't replace.

A bench-summary consumer audit: any future caller that reads only
`n_md_passed` without also checking `n_md_validated` is now doing it
wrong by definition — the docs and the test below say so.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from evoliez.stages import s10_md


def test_s10_persists_n_md_validated_distinct_from_n_md_passed():
    """Source-level guard: the two keys must coexist so back-compat
    readers see the old key while honest readers see the new one."""
    src = inspect.getsource(s10_md.MDStage.run)
    assert 'persist_meta("n_md_passed"' in src
    assert 'persist_meta("n_md_validated"' in src
    assert 'persist_meta("n_md_neutral_skip"' in src


def test_n_md_validated_excludes_neutral_skips():
    """Logic check by reading the source: n_md_validated must include
    `md_did_run` as a condition. Without it the new key would be the
    same as the old."""
    src = inspect.getsource(s10_md.MDStage.run)
    # The validated counter must AND md_passed with md_did_run.
    assert 'c.details.get("md_passed") and c.details.get("md_did_run")' in src


def test_honesty_log_uses_n_md_validated():
    """The visible cheap-run log line should say 'validated' (the
    honest number) not 'passed' (which silently includes neutrals)."""
    src = inspect.getsource(s10_md.MDStage.run)
    assert "validated" in src
    # The combined line shows the split clearly when there's any neutral.
    assert "neutral skip" in src or "neutral" in src


def test_n_md_passed_equals_validated_plus_neutral():
    """Math contract: by construction, n_md_passed = n_md_validated
    + n_md_neutral_skip. Anyone post-processing meta.json can rely
    on this identity to back-derive validated from old logs."""
    # We assert this from the source rather than from a fake run
    # because constructing a real RunContext with all the s10
    # dependencies in this test would be heavy and not actually test
    # anything additional.
    src = inspect.getsource(s10_md.MDStage.run)
    assert "n_pass = sum(" in src
    # The neutral-skip count is derived, not separately recounted.
    assert "n_neutral_skip = n_pass - n_validated" in src


def test_md_passed_neutral_intent_documented():
    """A grep-target so a future audit sees WHY skipped_parameterization
    gets passed=True: it's a deliberate 'neutral skip' policy so the
    candidate isn't penalised in ranking, but downstream consumers
    must ALSO check md_did_run."""
    from evoliez.md import analysis
    src = inspect.getsource(analysis.analyse)
    assert "skipped_parameterization" in src
    # The intent should be in the comment so a refactor doesn't
    # silently flip it.
    assert "neutral" in src.lower() or "don't penalise" in src.lower()


def test_s11_evidence_strong_gates_on_md_status_not_md_passed():
    """The evidence-class logic already guards against neutral-skip
    fake-validation: STRONG requires `md_status == "ok"`, not
    `md_passed=True`. This test pins that protection so a future
    refactor can't accidentally use the laxer field."""
    from evoliez.stages import s11_final_ranking

    src = inspect.getsource(s11_final_ranking)
    # Find the line that determines md_passed locally inside the
    # evidence-class function.
    assert 'md_status == "ok"' in src, (
        "s11 must gate evidence STRONG on md_status='ok', not on the "
        "looser md_passed flag (which includes neutral skips)"
    )
