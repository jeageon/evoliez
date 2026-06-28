"""v2 Phase H1 — s10 run_md GPU-batch worker. The worker mechanics (per-task run_md + a
per-candidate fail-safe) are tested in-process with a monkeypatched run_md; the real GPU
pinning + the ProcessPool fan-out are server-verified (a real multi-GPU run)."""
from types import SimpleNamespace

from evoliez.adapters.openmm_engine import MDResult

_MDCFG = SimpleNamespace(protocol_level=0, solvent="implicit")


def test_worker_runs_each_task(monkeypatch):
    seen = []
    kw_seen = {}

    def fake(mc, cid, mdcfg, wd, **kw):
        seen.append(cid)
        kw_seen.update(kw)
        return MDResult(candidate_id=cid, status="ok", protocol_level=0,
                        solvent_mode="implicit", simulation_time_ns=0.1)

    monkeypatch.setattr("evoliez.adapters.openmm_engine.run_md", fake)
    from evoliez.adapters.md_batch import _md_chunk_worker
    tasks = [("c1", object(), "/tmp/c1", 0.3), ("c2", object(), "/tmp/c2", 0.3)]
    out = _md_chunk_worker((None, tasks, _MDCFG, None, [], [10], True))
    assert set(out) == {"c1", "c2"}
    assert out["c1"].status == "ok" and seen == ["c1", "c2"]
    # the fail_loud-on-CPU guard flag must flow through the worker to run_md
    assert kw_seen.get("fail_loud_on_cpu") is True


def test_worker_failsafe_per_candidate(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("md crashed")

    monkeypatch.setattr("evoliez.adapters.openmm_engine.run_md", boom)
    from evoliez.adapters.md_batch import _md_chunk_worker
    out = _md_chunk_worker((None, [("c1", object(), "/tmp/c1", 0.3)], _MDCFG,
                            None, [], [10], False))
    assert out["c1"].status == "failed" and out["c1"].integration_failed


def test_empty_chunk():
    from evoliez.adapters.md_batch import _md_chunk_worker
    assert _md_chunk_worker((0, [], _MDCFG, None, [], [], False)) == {}
