"""Static guards for server shell scripts (ultra-review #27). Bash scripts are
awkward to unit-test, so assert the structural invariant that broke: a smoke
script that uses cwd-relative config/script paths must first cd to the repo
root resolved from its own location."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_server_smoke_cds_to_repo_root():
    txt = (ROOT / "scripts" / "server_smoke.sh").read_text()
    assert "BASH_SOURCE" in txt, "server_smoke.sh must resolve its own dir"
    assert re.search(r'cd\s+"\$REPO_DIR"', txt), (
        "server_smoke.sh must cd to repo root before using relative paths"
    )


def test_setup_boltz_env_is_idempotent():
    """`conda create -p` aborts on an existing prefix under set -e; creation
    must be guarded so a re-run refreshes instead of failing."""
    txt = (ROOT / "scripts" / "setup_boltz_env.sh").read_text()
    assert re.search(r'if \[ ! -x "\$BOLTZ_ENV/bin/python" \]', txt), (
        "setup_boltz_env.sh must guard `create -p` against an existing prefix"
    )


def test_setup_server_env_gates_libmamba_on_plugin_presence():
    """The libmamba solver must be selected ONLY when the plugin is installed,
    not from `conda config --show solver` (which always exits 0)."""
    txt = (ROOT / "scripts" / "setup_server_env.sh").read_text()
    assert "conda config --show solver" not in txt, (
        "setup_server_env.sh must not gate libmamba on `conda config --show`"
    )
    assert "grep -q conda-libmamba-solver" in txt
