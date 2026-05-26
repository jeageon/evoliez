"""Wave 5-B: s10_md observability + safety.

Three coordinated changes guard against the 22-hour / 22-timeout failure
mode we hit at production scale:

1. **failed_timeout is distinct from failed.** When the subprocess wall
   clock fires we get back ``MDResult(status='failed', failure_reason=
   'subprocess timeout after Ns')`` from ``openmm_subprocess``. The
   stage now upgrades that to ``status='failed_timeout'`` BEFORE
   stamping ``md_status`` on the candidate so bench-summary can count
   timeouts separately from genuine integration failures.

2. **Phase markers** with a fixed prefix on stdout: any tool greppable
   by ``r'^\\[evoliez-md-phase\\]'`` can build a timeline of which step
   stalled in seconds, instead of an hour of guesswork from OpenFF
   warnings.

3. **Preflight** runs a single 60-s ligand-parameterization probe at
   the top of ``MDStage.run``. A failure marks every candidate
   ``skipped_no_params`` and skips the per-candidate loop entirely —
   no more 22 timeouts in series on the same broken ligand.

The tests are deliberately import-light: heavy MD imports (openmm /
openff / rdkit) live behind ``_preflight_ligand_params`` and the
subprocess worker, both of which we monkey-patch here. CI runs in
``.venv-light`` and must not touch real MD code paths.
"""

from __future__ import annotations

import inspect
import re
import time
from typing import List

import pytest

from evoliez.adapters.openmm_engine import MDResult
from evoliez.stages import s10_md
from evoliez.stages.s10_md import MDStage, _PHASE_PREFIX, _phase
from evoliez.types import Candidate


# --------------------------------------------------------------------------- #
# 1. Phase marker format (the contract every downstream parser depends on)
# --------------------------------------------------------------------------- #
_PHASE_RE = re.compile(r"^\[evoliez-md-phase\] (\S+) (\d+\.\d{3})$")


def test_phase_marker_format(capsys):
    """`_phase(name)` MUST emit '[evoliez-md-phase] <name> <unix_ts>' on
    stdout. The prefix is the contract — any future log-parsing tool
    anchors a regex on it."""
    before = time.time()
    _phase("some_step_start")
    after = time.time()

    out = capsys.readouterr().out.strip().splitlines()
    assert out, "phase marker not emitted on stdout"
    assert len(out) == 1
    m = _PHASE_RE.match(out[0])
    assert m, f"phase marker doesn't match contract: {out[0]!r}"
    assert m.group(1) == "some_step_start"
    ts = float(m.group(2))
    # The timestamp is a real unix time captured between the two clocks.
    assert before - 1.0 <= ts <= after + 1.0


def test_phase_marker_prefix_exported_constant():
    """The prefix MUST be the exported _PHASE_PREFIX so worker + stage
    can't drift apart silently."""
    assert _PHASE_PREFIX == "[evoliez-md-phase]"


def test_worker_uses_same_phase_prefix():
    """openmm_subprocess_worker.py is the OTHER side of the phase
    contract: it emits markers from inside the subprocess. The literal
    string must match exactly so a single regex parses both streams."""
    from evoliez.adapters import openmm_subprocess_worker as w
    assert w._PHASE_PREFIX == _PHASE_PREFIX
    src = inspect.getsource(w)
    # Worker brackets the run_md call at minimum so a hung subprocess
    # shows "run_md_start" without a matching "run_md_done".
    assert "run_md_start" in src
    assert "run_md_done" in src
    assert "worker_start" in src
    assert "worker_done" in src


def test_s10_md_emits_phase_markers_around_subprocess_call():
    """Source guard: the stage wraps each subprocess call with
    candidate_start/_done markers so we can see WHICH candidate
    started but never finished."""
    src = inspect.getsource(MDStage.run)
    assert "_phase(" in src
    assert "candidate_start" in src
    assert "candidate_done" in src


# --------------------------------------------------------------------------- #
# 2. failed_timeout is its own status (source-level + behavioural guard)
# --------------------------------------------------------------------------- #


def test_failed_timeout_distinct_from_failed():
    """MDStage must distinguish wall-clock timeouts from generic failures
    by upgrading status to ``failed_timeout``. Source-guards both the
    upgrade site and the persisted-meta key so bench-summary can rely
    on the count."""
    src = inspect.getsource(s10_md)
    assert '"failed_timeout"' in src or "'failed_timeout'" in src
    # Bench-summary keys on this meta name. A typo here would silently
    # zero the timeout column.
    assert "n_md_failed_timeout" in src
    # The discriminator is the wrapper's failure_reason text.
    assert "subprocess timeout" in src.lower()


def test_timeout_result_gets_upgraded_to_failed_timeout():
    """Drive the upgrade logic with a synthetic MDResult mimicking what
    ``run_md_in_subprocess`` returns on timeout, and confirm the stage
    rewrites status to ``failed_timeout``."""
    cfg = _real_md_config()
    candidates = [_candidate("c0")]
    captured: List[MDResult] = []

    def _fake_run_md_in_subprocess(*_a, **_kw):
        # Same shape as openmm_subprocess._failed_result on timeout.
        r = MDResult(
            candidate_id="c0", status="failed",
            protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
            simulation_time_ns=0.0, integration_failed=True,
            failure_reason="subprocess timeout after 1800s",
            timestep_fs=float(cfg.timestep_fs),
        )
        captured.append(r)
        return r

    ctx = _build_ctx(candidates, cfg)
    _run_md_with_patched_subprocess(
        ctx, _fake_run_md_in_subprocess, preflight_returns=None,
    )

    assert ctx._meta["n_md_failed"] == 1
    assert ctx._meta["n_md_failed_timeout"] == 1
    assert candidates[0].scores["md_status"] == "failed_timeout"


def test_non_timeout_failure_stays_failed_not_failed_timeout():
    """A subprocess that crashes (rc=42, log tail in failure_reason but
    NO 'subprocess timeout' substring) must stay status='failed' — it's
    a real error, not a wall-clock event."""
    cfg = _real_md_config()
    candidates = [_candidate("c0")]

    def _fake_run_md_in_subprocess(*_a, **_kw):
        return MDResult(
            candidate_id="c0", status="failed",
            protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
            simulation_time_ns=0.0, integration_failed=True,
            failure_reason="subprocess crashed (rc=42) after 12.3s: ZeroDivisionError",
            timestep_fs=float(cfg.timestep_fs),
        )

    ctx = _build_ctx(candidates, cfg)
    _run_md_with_patched_subprocess(
        ctx, _fake_run_md_in_subprocess, preflight_returns=None,
    )

    assert ctx._meta["n_md_failed"] == 1
    assert ctx._meta["n_md_failed_timeout"] == 0
    assert candidates[0].scores["md_status"] == "failed"


# --------------------------------------------------------------------------- #
# 3. Preflight failure path (the 22-hour saver)
# --------------------------------------------------------------------------- #


def test_preflight_failure_skips_all_candidates():
    """When ``_preflight_ligand_params`` returns a reason, every
    candidate gets ``md_status='skipped_no_params'`` and the per-
    candidate loop is NEVER entered."""
    cfg = _real_md_config()
    candidates = [_candidate(f"c{i}") for i in range(12)]
    call_log: List[str] = []

    def _exploding_subprocess(*_a, **_kw):
        # Any entry into the per-candidate loop is a regression.
        call_log.append("subprocess_called")
        raise AssertionError("per-candidate loop must not run on preflight fail")

    ctx = _build_ctx(candidates, cfg)
    _run_md_with_patched_subprocess(
        ctx, _exploding_subprocess,
        preflight_returns=(
            "ligand_preflight_failed: _LigandParamUnsupported: "
            "sqm AM1-BCC hung on NADP (after 60.0s)"
        ),
    )

    assert call_log == [], "subprocess wrapper was called despite preflight fail"
    assert ctx._meta["n_md_skipped"] == len(candidates)
    assert ctx._meta["n_md_skipped_no_params"] == len(candidates)
    assert ctx._meta["n_md_real_ran"] == 0
    assert ctx._meta["n_md_failed"] == 0
    assert ctx._meta["n_md_failed_timeout"] == 0
    for c in candidates:
        assert c.scores["md_status"] == "skipped_no_params"
        assert c.scores["md_did_run"] == 0
        assert c.details["md_did_run"] is False
        assert c.details["md_passed"] is False
        # Honest reason persisted on the candidate for the report.
        assert "preflight" in str(c.details.get("md_failure_reasons", "")).lower()


def test_preflight_success_proceeds_normally():
    """When preflight returns None, the candidate loop runs normally
    and each candidate receives an MDResult-derived status."""
    cfg = _real_md_config()
    candidates = [_candidate("c0"), _candidate("c1")]
    n_called = 0

    def _ok_subprocess(_cx, candidate_id, _cfg, _workdir, **_kw):
        nonlocal n_called
        n_called += 1
        return MDResult(
            candidate_id=candidate_id, status="ok",
            protocol_level=cfg.protocol_level, solvent_mode=cfg.solvent,
            simulation_time_ns=1.0,
            minimized_pdb=None, trajectory_path=None,
            ligand_rmsd_series=[1.2, 1.3, 1.4],
            pocket_rmsd_series=[0.5, 0.6, 0.7],
        )

    ctx = _build_ctx(candidates, cfg)
    _run_md_with_patched_subprocess(
        ctx, _ok_subprocess, preflight_returns=None,
    )

    assert n_called >= len(candidates), (
        "every candidate should have been dispatched to the subprocess wrapper"
    )
    assert ctx._meta["n_md_real_ran"] == len(candidates)
    assert ctx._meta["n_md_failed_timeout"] == 0
    for c in candidates:
        assert c.scores["md_status"] == "ok"


def test_preflight_skips_for_mock_backend():
    """Mock backend doesn't have a parameterization step; preflight is
    a no-op (returns None) so the existing mock-pipeline tests keep
    working unchanged."""
    from evoliez.config import Backend
    cfg = _real_md_config()
    ctx = _build_ctx([_candidate("c0")], cfg, backend=Backend.mock)
    # Real preflight method, no monkeypatch: it should short-circuit
    # on the mock backend BEFORE touching any openmm/openff import.
    stage = MDStage()
    result = stage._preflight_ligand_params(ctx)
    assert result is None


def test_preflight_skips_for_dry_run():
    """Dry-run must not pay the preflight cost (and must not crash if
    the openmm/openff stack is absent locally)."""
    cfg = _real_md_config()
    ctx = _build_ctx([_candidate("c0")], cfg)
    ctx.dry_run = True
    stage = MDStage()
    assert stage._preflight_ligand_params(ctx) is None


def test_preflight_does_not_raise_on_inner_exception(monkeypatch):
    """Even if the openmm/openff probe blows up in an unexpected way,
    the preflight must return a reason string, not raise — a stage
    crash here would defeat the whole point of preflight (graceful
    degrade to skipped_no_params)."""
    from evoliez.config import Backend
    cfg = _real_md_config()
    ctx = _build_ctx([_candidate("c0")], cfg, backend=Backend.real)

    # Force the preflight body into the except branch by patching the
    # heavy import target. The preflight defers imports, so patching
    # the openmm_engine attribute is enough.
    from evoliez.adapters import openmm_engine

    def _boom(*_a, **_kw):
        raise RuntimeError("synthetic preflight failure")

    monkeypatch.setattr(
        openmm_engine, "_ligand_offmol_at_pose", _boom, raising=False,
    )
    monkeypatch.setattr(
        openmm_engine, "_ligand_system_generator", _boom, raising=False,
    )

    stage = MDStage()
    reason = stage._preflight_ligand_params(ctx)
    # Either the import inside preflight failed (missing openff in
    # .venv-light) or our _boom fired — both legitimately yield a
    # reason string, never a raise.
    assert reason is None or "ligand_preflight_failed" in reason


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _candidate(cid: str) -> Candidate:
    return Candidate(
        candidate_id=cid,
        mutations=[],
        generator="test",
        scores={"instability": 0.2},
        details={},
    )


def _real_md_config():
    from evoliez.config import MDConfig
    return MDConfig(
        enabled=True,
        protocol_level=0,
        solvent="implicit",
        replicas=1,
        final_tier_replicas=1,
        subprocess_isolation=False,  # mock backend short-circuits anyway
        subprocess_timeout_seconds=1800,
    )


def _build_ctx(candidates: List[Candidate], mdcfg, backend=None):
    """Build the smallest ctx surface MDStage.run touches.

    We use a SimpleNamespace-style stub rather than a real RunContext +
    SQLite store because:
      (a) the assertions are on candidate.scores / ctx._meta, not on
          the persisted MDSimulation rows;
      (b) a real RunContext requires load_config + Pipeline plumbing
          for wt_complex, which is way more setup than these unit
          tests warrant.
    """
    from contextlib import contextmanager
    from pathlib import Path
    import tempfile
    from types import SimpleNamespace

    from evoliez.config import Backend

    backend = backend or Backend.real
    workdir = Path(tempfile.mkdtemp(prefix="md_test_"))
    (workdir / "md").mkdir()

    paths = SimpleNamespace(
        md=workdir / "md",
        md_candidate=lambda cid: (workdir / "md" / cid),
    )
    # ensure each cand dir exists for analysis.json writes
    for c in candidates:
        (workdir / "md" / c.candidate_id).mkdir(parents=True, exist_ok=True)

    scoring = SimpleNamespace(md_lite=1.0)
    validation = SimpleNamespace(md=mdcfg)
    config = SimpleNamespace(
        validation=validation,
        scoring=scoring,
        backend_for=lambda _name: backend,
    )

    # Minimal Complex stub. wt_complex is required by run(); the
    # mock-backend candidate loop uses _mutant_complex(wt, cand) which
    # only reads structure.sequence / residues.
    from evoliez.adapters.base import synthetic_structure
    from evoliez.types import Complex, Ligand

    structure = synthetic_structure("ACDEFGHIKLMNPQRSTVWY", seed=7)
    ligand = Ligand(id="L", smiles="CCO", atoms=[])
    wt = Complex(structure=structure, ligand=ligand)

    artifacts = {
        "md_candidates": candidates,
        "wt_complex": wt,
        "catalytic_positions": [],
        "mutant_complexes": {},
    }
    meta = {}

    class _FakeSession:
        def query(self, *_a, **_kw): return self
        def filter(self, *_a, **_kw): return self
        def delete(self, *_a, **_kw): return 0
        def add(self, *_a, **_kw): return None

    @contextmanager
    def _session():
        yield _FakeSession()

    store = SimpleNamespace(session=_session)

    ctx = SimpleNamespace(
        config=config,
        paths=paths,
        store=store,
        project_id=1,
        dry_run=False,
        _artifacts=artifacts,
        _meta=meta,
        require=lambda k: artifacts[k],
        get=lambda k, default=None: artifacts.get(k, default),
        put=lambda k, v: artifacts.__setitem__(k, v),
        persist_meta=lambda k, v: meta.__setitem__(k, v),
    )
    return ctx


def _run_md_with_patched_subprocess(ctx, fake_subprocess, *, preflight_returns):
    """Run MDStage.run against the stub ctx with the subprocess wrapper
    and preflight both patched. Returns the stage instance for further
    poking."""
    stage = MDStage()
    # Patch the preflight on the instance so we don't need to import
    # openmm/openff in the test process.
    stage._preflight_ligand_params = lambda _ctx: preflight_returns  # type: ignore
    import evoliez.stages.s10_md as mod
    orig = mod.run_md_in_subprocess
    mod.run_md_in_subprocess = fake_subprocess
    try:
        stage.run(ctx)
    finally:
        mod.run_md_in_subprocess = orig
    return stage
