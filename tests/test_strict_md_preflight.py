"""H (post-expert-audit) — strict_preflight gates s08b.

C added `md_preflight_status` to ctx.meta but no stage read it. So a
project whose ligand cannot be MD-parameterised (e.g. P450 BM3 with
heme, NADP+ on a server without curated AMBER files) still ran the
full per-mutant Boltz pass in s08b — burning hours of GPU time on
structures only MD would consume, and MD wasn't going to consume them.

When `MDConfig.strict_preflight = True`, s08b checks ctx.meta and
short-circuits if md_preflight_status is in the blocking set
{unsupported, timeout_preflight, failed_preflight}. Off by default so
existing non-strict benchmarks (which want Boltz / docking signals
even when MD will skip) still run.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from evoliez.config import MDConfig
from evoliez.stages import s08b_mutant_boltz


def _mk_ctx(*, md_preflight_status: str, strict: bool,
            md_candidates=None, mutant_boltz_enabled: bool = True):
    """Minimal duck-typed RunContext that satisfies the strict-preflight
    branch in s08b. We don't need a real pipeline — the gate runs before
    `wt_complex` / `target_sequence` are required, so a stub is enough."""
    ctx = MagicMock()
    ctx.config.reranking.mutant_boltz_enabled = mutant_boltz_enabled
    ctx.config.validation.md = MDConfig(strict_preflight=strict)
    ctx.config.backend_for.return_value = "mock"

    meta_store = {
        "md_preflight_status": md_preflight_status,
        "md_preflight_reason": "synthetic test reason",
    }
    persisted = {}

    def _meta(key, default=None):
        return meta_store.get(key, default)

    def _persist_meta(key, value):
        persisted[key] = value

    ctx.meta = _meta
    ctx.persist_meta = _persist_meta
    ctx._persisted = persisted

    def _get(key, default=None):
        if key == "md_candidates":
            return md_candidates if md_candidates is not None else []
        if key == "redock_candidates":
            return []
        if key == "mutant_complexes":
            return {}
        return default

    ctx.get = _get
    return ctx


def test_strict_preflight_unsupported_short_circuits_s08b():
    """Most important case: strict=True + unsupported → s08b returns
    without running Boltz."""
    fake_cands = [MagicMock(candidate_id="c1"), MagicMock(candidate_id="c2")]
    ctx = _mk_ctx(
        md_preflight_status="unsupported",
        strict=True,
        md_candidates=fake_cands,
    )
    stage = s08b_mutant_boltz.MutantBoltzStage()
    stage.run(ctx)
    # The honesty meta key is set so the report can show the skip.
    assert ctx._persisted.get("s08b_skipped_strict_preflight") == 1
    # wt_complex was NEVER required (would have crashed the MagicMock).
    # MagicMock.require would have been called; assert it wasn't.
    ctx.require.assert_not_called()


def test_strict_preflight_timeout_short_circuits_s08b():
    """timeout_preflight is also blocking — the ligand probe hung, so
    MD will hang too. Skip Boltz."""
    fake_cands = [MagicMock(candidate_id="c1")]
    ctx = _mk_ctx(
        md_preflight_status="timeout_preflight",
        strict=True,
        md_candidates=fake_cands,
    )
    stage = s08b_mutant_boltz.MutantBoltzStage()
    stage.run(ctx)
    assert ctx._persisted.get("s08b_skipped_strict_preflight") == 1


def test_strict_preflight_failed_short_circuits_s08b():
    fake_cands = [MagicMock(candidate_id="c1")]
    ctx = _mk_ctx(
        md_preflight_status="failed_preflight",
        strict=True,
        md_candidates=fake_cands,
    )
    stage = s08b_mutant_boltz.MutantBoltzStage()
    stage.run(ctx)
    assert ctx._persisted.get("s08b_skipped_strict_preflight") == 1


def test_strict_preflight_off_keeps_running_on_unsupported():
    """Default behaviour: strict_preflight=False → s08b proceeds even
    when MD will skip. We can't run the full Boltz here without real
    deps, but we can assert it tried to require wt_complex (i.e. it
    moved past the strict gate)."""
    fake_cands = [MagicMock(candidate_id="c1")]
    ctx = _mk_ctx(
        md_preflight_status="unsupported",
        strict=False,           # <-- the default
        md_candidates=fake_cands,
    )
    # Make `require` raise so the test catches the moment it's called
    # (proves the strict gate didn't short-circuit).
    ctx.require.side_effect = RuntimeError("require('wt_complex') reached")
    stage = s08b_mutant_boltz.MutantBoltzStage()
    try:
        stage.run(ctx)
    except RuntimeError as exc:
        assert "wt_complex" in str(exc), (
            f"unexpected exception path: {exc}"
        )
    # And we DID NOT mark s08b as skipped via the strict path.
    assert ctx._persisted.get("s08b_skipped_strict_preflight") is None


def test_strict_preflight_ok_runs_normally():
    """strict=True + status=ok → no short-circuit; the strict gate is
    only for blocking statuses."""
    fake_cands = [MagicMock(candidate_id="c1")]
    ctx = _mk_ctx(
        md_preflight_status="ok",
        strict=True,
        md_candidates=fake_cands,
    )
    ctx.require.side_effect = RuntimeError("require('wt_complex') reached")
    stage = s08b_mutant_boltz.MutantBoltzStage()
    try:
        stage.run(ctx)
    except RuntimeError as exc:
        assert "wt_complex" in str(exc)
    assert ctx._persisted.get("s08b_skipped_strict_preflight") is None


def test_strict_preflight_curated_runs_normally():
    """curated_available → MD will work via the tleap path; no reason
    to gate Boltz."""
    fake_cands = [MagicMock(candidate_id="c1")]
    ctx = _mk_ctx(
        md_preflight_status="curated_available",
        strict=True,
        md_candidates=fake_cands,
    )
    ctx.require.side_effect = RuntimeError("require('wt_complex') reached")
    stage = s08b_mutant_boltz.MutantBoltzStage()
    try:
        stage.run(ctx)
    except RuntimeError as exc:
        assert "wt_complex" in str(exc)
    assert ctx._persisted.get("s08b_skipped_strict_preflight") is None


def test_md_config_default_strict_preflight_is_false():
    """Default OFF so we don't surprise existing non-strict benchmarks
    (e.g. Bgl3, where Boltz signal alone is informative for some
    mutations even if MD can't run)."""
    assert MDConfig().strict_preflight is False


def test_md_config_strict_preflight_can_be_set():
    cfg = MDConfig(strict_preflight=True)
    assert cfg.strict_preflight is True


def test_s08b_source_has_strict_gate():
    """A future refactor must not silently delete this gate. The
    expert audit's point is that without this hook, C provides no
    actual wall-time saving."""
    src = inspect.getsource(s08b_mutant_boltz.MutantBoltzStage.run)
    assert "strict_preflight" in src, (
        "s08b must consult mdcfg.strict_preflight; otherwise C is "
        "metadata-only and the wall-time saving never materialises"
    )
    assert "md_preflight_status" in src
    assert "s08b_skipped_strict_preflight" in src
