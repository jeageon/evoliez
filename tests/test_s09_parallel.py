"""s09 non-MD validation: bounded thread-pool fan-out + real-pose geometry.

Two behaviours added to ``s09_nonmd_validation``:

CHANGE 1  The per-candidate loop (FoldX/Rosetta stability + gnina/diffdock
          redocking — heavy work in subprocesses, so the orchestration is
          I/O-bound) is fanned out across a BOUNDED ThreadPoolExecutor. The
          result MUST be deterministic: results are collected for ALL
          candidates, then the ``kept`` filter + ``md_candidates`` selection are
          applied in candidate order, NOT completion order. So a threaded run
          and a forced-serial run produce byte-identical selections.

CHANGE 2 (audit #5)  The catalytic-geometry penalty is computed on the REAL
          s08b Boltz mutant pose (its backbone + its ligand pose) for the top-N
          that have one, instead of the inert WT-coords proxy (which yields
          penalty ~= 0). So the penalty actually reflects pose/backbone change.

The mock backend stubs every external tool, so this exercises the
orchestration + the kept/md selection logic, not the real binaries.
"""

from pathlib import Path

import pytest

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.stages import s09_nonmd_validation as s09
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


class _SerialPool:
    """Drop-in stand-in for ThreadPoolExecutor that runs ``map`` serially in the
    submitting thread. Used to prove the s09 result is independent of execution
    mode: the threaded fan-out must match this serial baseline exactly."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, it):
        return [fn(x) for x in it]


def _cfg(tmp_path: Path):
    # Enough candidates reach s09 (top_for_redocking) that the pool runs >1
    # worker, and s08b predicts real mutant complexes for the top-N so CHANGE 2
    # has a real pose to score.
    return load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "mutation_generation": {"methods": ["chemistry_rules", "msa_sampler"],
                                    "max_candidates": 60},
            "reranking": {"top_for_redocking": 8, "top_for_md": 4,
                          "mutant_boltz_enabled": True, "mutant_boltz_top_n": 4},
            "validation": {"md": {"top_candidates": 4}},
            "gnn": {"build_dataset": False},
        },
    )


def _run_to_s09(cfg):
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s09_nonmd")
    return ctx


def _selection(ctx):
    """The two ctx outputs s09 is responsible for, as comparable id-lists."""
    val = [c.candidate_id for c in (ctx.get("validated_candidates") or [])]
    md = [c.candidate_id for c in (ctx.get("md_candidates") or [])]
    return val, md


def test_every_candidate_is_scored(tmp_path):
    ctx = _run_to_s09(_cfg(tmp_path))
    cands = ctx.get("redock_candidates") or []
    assert len(cands) >= 2, "need >1 candidate to exercise the fan-out"
    # (a) EVERY candidate that entered s09 was processed: stability + redocking
    # scores are present, and each is tagged kept-or-rejected (mutually
    # exclusive). Nothing was skipped or double-counted by the pool.
    for c in cands:
        assert "ddg_fold" in c.scores
        assert "redocking_consistency" in c.scores
        assert "catalytic_geometry_penalty" in c.scores
        assert c.details.get("redock_structure_source") in ("mutant_boltz",
                                                             "wt_proxy")
    val_ids = {c.candidate_id for c in (ctx.get("validated_candidates") or [])}
    for c in cands:
        rejected = "nonmd_rejected" in c.details
        assert rejected != (c.candidate_id in val_ids)  # kept XOR rejected


def test_md_selection_matches_serial_run(tmp_path, monkeypatch):
    """The threaded fan-out must give the SAME kept/md selection as a serial run
    for the same seeded input — completion order must not leak into the result."""
    # threaded (real ThreadPoolExecutor)
    threaded = _selection(_run_to_s09(_cfg(tmp_path / "threaded")))

    # forced-serial: replace the pool with a serial shim, same seed/input.
    monkeypatch.setattr(s09, "ThreadPoolExecutor", _SerialPool)
    serial = _selection(_run_to_s09(_cfg(tmp_path / "serial")))

    assert threaded == serial, (
        f"threaded selection {threaded} != serial {serial}")
    # the selection is non-trivial (not empty / not everything) so the test
    # actually constrains ordering, and md is the head of validated.
    val, md = threaded
    assert val and md
    assert md == val[: len(md)]


def test_geometry_penalty_uses_real_mutant_pose(tmp_path):
    """CHANGE 2: candidates with a real s08b mutant complex get their catalytic
    penalty from that pose (tagged 'mutant_boltz'); the rest fall back to the WT
    proxy. The real-pose penalty is genuinely non-zero (the proxy's would be ~0
    because _mutant_complex only swaps residue identities, not coordinates)."""
    ctx = _run_to_s09(_cfg(tmp_path))
    cands = ctx.get("redock_candidates") or []
    by_src = {}
    for c in cands:
        by_src.setdefault(c.details.get("catalytic_geometry_source"), []).append(c)

    assert "mutant_boltz" in by_src, "top-N should score geometry on the real pose"
    assert "wt_proxy" in by_src, "remainder should fall back to the proxy"
    # the geometry source matches the redock source (both keyed on whether a real
    # mutant complex exists for that candidate_id).
    for c in cands:
        assert (c.details.get("catalytic_geometry_source")
                == c.details.get("redock_structure_source"))
        # CHANGE (audit P2 #6): the mechanism is annotated on the SAME structure
        # as the geometry penalty — the real s08b mutant pose when one exists, the
        # WT proxy otherwise (negative_design is on by default for this config).
        if "mechanism_source" in c.details:
            assert (c.details["mechanism_source"]
                    == c.details.get("redock_structure_source"))
    # the real mutant backbone+pose actually moves the active site -> at least one
    # real-pose candidate has a strictly positive penalty (the inert proxy could
    # not produce this).
    real_pens = [c.scores["catalytic_geometry_penalty"]
                 for c in by_src["mutant_boltz"]]
    assert any(p > 0.0 for p in real_pens), (
        f"real-pose geometry penalties all ~0: {real_pens}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
