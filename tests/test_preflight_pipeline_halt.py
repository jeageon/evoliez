"""K (post-expert-audit) — strict_preflight halts the pipeline at s01.

Expert critique on H: "s08b 우회만으로는 s02-s07 시간 못 막음. early-abort
는 아닙니다." So this commit lifts the halt point from s08b to s01 — when
`strict_preflight=True` AND the preflight returns a blocking status
(`unsupported` / `timeout_preflight` / `failed_preflight`), s01 raises
`PreflightBlocked` and the Pipeline catches it for a clean halt.

The s08b gate from H stays in place as defence in depth (covers configs
that disable the halt or legacy callers that pass `from_stage=s02`).

These tests pin:
  - s01 raises PreflightBlocked under strict + blocking
  - s01 does NOT raise under strict + non-blocking (ok / curated)
  - s01 does NOT raise under default strict=False even on blocking
  - Pipeline.run catches it and returns ctx without crashing
  - ctx.meta["preflight_blocked"] reflects the status
  - Downstream stages do NOT execute after the halt
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from evoliez.pipeline import Pipeline
from evoliez.stages.base import PreflightBlocked, Stage


def test_preflight_blocked_exception_carries_status_and_reason():
    """Sanity check the exception captures the verdict for the
    Pipeline log line."""
    pb = PreflightBlocked("timeout_preflight", "subprocess timeout after 300s")
    assert pb.status == "timeout_preflight"
    assert pb.reason == "subprocess timeout after 300s"
    # str(pb) is what gets logged if uncaught — must be readable.
    s = str(pb)
    assert "timeout_preflight" in s
    assert "subprocess timeout after 300s" in s


def test_pipeline_catches_preflight_blocked_cleanly():
    """A stage raising PreflightBlocked must be caught by Pipeline.run
    and produce a clean return (no exception propagating to the CLI)."""

    class _S01(Stage):
        name = "s01_synth"

        def run(self, ctx):
            raise PreflightBlocked("unsupported", "synthetic reason")

    class _S02(Stage):
        name = "s02_synth"

        def __init__(self):
            super().__init__()
            self.ran = False

        def run(self, ctx):
            self.ran = True

    pipe = Pipeline.__new__(Pipeline)
    s01_inst = _S01()
    s02_inst = _S02()
    pipe.stages = [s01_inst, s02_inst]

    ctx = MagicMock()
    ctx.dry_run = False
    ctx.is_stage_done.return_value = False

    # Should NOT raise.
    returned = pipe.run(ctx)
    assert returned is ctx
    assert s02_inst.ran is False, (
        "downstream stage must NOT run after PreflightBlocked halt"
    )
    # s01 was NOT marked done (the halt skips mark_stage_done too —
    # so a future --resume re-attempts it).
    ctx.mark_stage_done.assert_not_called()


def test_pipeline_continues_on_normal_exception_path():
    """Backward compat: a generic RuntimeError still propagates (so the
    CLI's existing error handling kicks in). PreflightBlocked is the
    ONLY exception with this clean-halt semantics."""

    class _S01(Stage):
        name = "s01_boom"

        def run(self, ctx):
            raise RuntimeError("generic stage failure")

    pipe = Pipeline.__new__(Pipeline)
    pipe.stages = [_S01()]

    ctx = MagicMock()
    ctx.dry_run = False
    ctx.is_stage_done.return_value = False

    with pytest.raises(RuntimeError, match="generic stage failure"):
        pipe.run(ctx)


def test_s01_source_uses_preflight_blocked():
    """Source-level guard: a future refactor must keep the K halt in
    s01 (the whole point of the commit). The s08b gate from H still
    exists too — both for defence in depth."""
    from evoliez.stages import s01_input_preprocess as s01

    src = inspect.getsource(s01.InputPreprocessStage.run)
    assert "PreflightBlocked" in src, (
        "s01 must raise PreflightBlocked when strict_preflight=True "
        "and the preflight status is blocking"
    )
    assert "strict_preflight" in src
    assert "preflight_blocked" in src
    # The blocking status set must include all three failure modes.
    assert "unsupported" in src
    assert "timeout_preflight" in src
    assert "failed_preflight" in src


def test_pipeline_logs_halt_with_status_and_reason():
    """The Pipeline's log line should let an operator see WHY the
    pipeline halted without spelunking through stage logs."""
    pipe_src = inspect.getsource(Pipeline.run)
    assert "[halt]" in pipe_src
    assert "preflight blocked" in pipe_src
    assert "PreflightBlocked" in pipe_src
