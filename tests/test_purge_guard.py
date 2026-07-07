"""The fingerprint-change purge must NOT silently delete a non-empty run.

Regression for the data-loss incident: a single config-field edit changed
config_sha1, which auto-rmtree'd a finished multi-hour run. The guard now refuses
unless EVOLIEZ_ALLOW_PURGE=1 is set.
"""
from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext

ROOT = Path(__file__).resolve().parents[1]


def _ctx(tmp_path):
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {"project.output_dir": str(tmp_path / "run"),
         "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta")},
    )
    return RunContext(cfg, allow_small_disk=True).setup()


def test_purge_refused_on_nonempty_run(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLIEZ_ALLOW_PURGE", raising=False)
    ctx = _ctx(tmp_path)
    precious = ctx.paths.md / "candidate" / "trajectory.pdb"
    precious.parent.mkdir(parents=True, exist_ok=True)
    precious.write_text("ATOM   hours of MD\n")
    ctx._purge_stale_outputs()
    assert precious.exists(), "guard must NOT delete a non-empty run on fp change"
    assert ctx._purge_refused is True


def test_purge_allowed_with_explicit_optin(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLIEZ_ALLOW_PURGE", "1")
    ctx = _ctx(tmp_path)
    f = ctx.paths.md / "x.pdb"
    f.write_text("data\n")
    ctx._purge_stale_outputs()
    assert not f.exists(), "explicit EVOLIEZ_ALLOW_PURGE=1 should purge"
    assert ctx._purge_refused is False
