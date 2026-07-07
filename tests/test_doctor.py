from pathlib import Path

from evoliez.diagnostics import BLOCK, OK, collect

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_runs_without_tools():
    rep = collect()
    names = {c.name for c in rep.checks}
    assert "evoliez" in names and "python" in names
    assert any(n.startswith("py:") for n in names)
    assert any(n.startswith("tool:") for n in names)
    assert any(n.startswith("disk:") for n in names)
    # informational: every check has a valid status
    for c in rep.checks:
        assert c.status in {"ok", "warn", "missing", "block"}


def test_doctor_validates_example_config():
    rep = collect(str(ROOT / "configs" / "example_fdh_nadp.yaml"))
    cfg_checks = [c for c in rep.checks if c.name == "config"]
    assert cfg_checks and cfg_checks[0].status == OK


def test_doctor_flags_real_config_without_db(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "project: {output_dir: /tmp/x}\n"
        "input: {target_sequence: 'ACDEFGHIKLMNOP', ligand: {value: CCO}}\n"
        "backend: real\n"
        "msa: {remote_server: false}\n"
    )
    rep = collect(str(cfg))
    db = [c for c in rep.checks if c.name == "config:homolog_db"]
    assert db and db[0].status == BLOCK
    assert rep.n_block >= 1
