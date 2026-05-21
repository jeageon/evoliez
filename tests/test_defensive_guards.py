"""Production-stability audit: three defensive guards added after
the first interrupted production run (psefdh_nad_to_nadp, stopped
mid-s08b). All three protect against the same failure shape - a
process killed mid-write leaves a partially-written file that the
NEXT startup would have crashed on.

1. `_save_state` writes to a `.tmp` sibling then `os.replace`s. A
   Ctrl-C / OOM kill mid-write either leaves the OLD state intact
   or the NEW one intact - never a truncated json.

2. `_load_state` catches `JSONDecodeError`. If a state file IS
   somehow corrupted (e.g. from before the atomic-write fix), the
   pipeline starts fresh with a warning instead of crashing.

3. `_load_existing_real` swallows parse errors on the cache PDB
   and returns None - caller falls back to a fresh Boltz run rather
   than aborting the whole pipeline on a half-written file from a
   killed `boltz predict`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evoliez.adapters import boltz as boltz_adapter
from evoliez.adapters.boltz import _load_existing_real
from evoliez.config import ComplexPredictionConfig
from evoliez.types import Ligand


# --------------------------------------------------------------------------- #
# Guard 1+2: atomic state write + robust load
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[1]


def _make_ctx(tmp_path: Path):
    """Real RunContext over the example FDH config, redirected to
    tmp_path. We only exercise _save_state / _load_state here, so a
    fresh per-test ctx via .setup() is enough."""
    from evoliez.config import load_config
    from evoliez.context import RunContext

    cfg = load_config(
        _ROOT / "configs" / "example_fdh_nadp.yaml",
        {"project.output_dir": str(tmp_path / "run")},
    )
    return RunContext(cfg, allow_small_disk=True).setup()


def test_save_state_uses_atomic_replace(tmp_path, monkeypatch):
    """Source guard: _save_state must write through a tmp file and
    rename, NOT call Path.write_text directly. If someone reverts to
    write_text we lose crash safety - this fails them loudly."""
    ctx = _make_ctx(tmp_path)
    # Spy on os.replace; the atomic path goes through it exactly once
    # per _save_state call.
    import os as _os

    calls = {"replace": 0, "write_text": 0}
    real_replace = _os.replace

    def counting_replace(src, dst):
        calls["replace"] += 1
        return real_replace(src, dst)

    original_write_text = Path.write_text

    def counting_write_text(self, *a, **kw):
        if "state" in self.name:
            calls["write_text"] += 1
        return original_write_text(self, *a, **kw)

    monkeypatch.setattr(_os, "replace", counting_replace)
    monkeypatch.setattr(Path, "write_text", counting_write_text)

    ctx.persist_meta("smoke", "ok")

    assert calls["replace"] >= 1, (
        "_save_state must use os.replace for atomicity"
    )
    assert calls["write_text"] == 0, (
        "_save_state must NOT use Path.write_text on the state file "
        "directly - that's a non-atomic overwrite"
    )


def test_load_state_recovers_from_corrupt_json(tmp_path):
    """A truncated state file (e.g. from a pre-atomic-fix crash) must
    NOT crash the pipeline at startup. The user can ALWAYS recover by
    re-running; an unhandled JSONDecodeError leaves them stuck."""
    ctx = _make_ctx(tmp_path)
    sp = ctx.paths.state_path
    # Persist something legitimate first so the file has structure.
    ctx.persist_meta("phase", 1)
    # Now corrupt it (simulate Ctrl-C mid-write).
    sp.write_text('{"completed_stages": ["s01", "s0')  # truncated mid-json
    # Build a SECOND context against the same on-disk state. Old
    # behaviour: json.loads raises JSONDecodeError -> pipeline aborts.
    # New behaviour: warning + fresh state.
    ctx2 = _make_ctx(tmp_path)  # re-runs _load_state internally
    assert ctx2.meta("phase", default="missing") == "missing", (
        "corrupt state should be discarded (meta wiped), not propagated"
    )


def test_load_state_handles_empty_file(tmp_path):
    """Edge case: an empty file (open-but-no-write crash) - must
    behave like a missing state file, not raise."""
    ctx = _make_ctx(tmp_path)
    sp = ctx.paths.state_path
    sp.write_text("")
    ctx2 = _make_ctx(tmp_path)  # must not raise
    assert ctx2.meta("phase", default="missing") == "missing"


# --------------------------------------------------------------------------- #
# Guard 3: _load_existing_real swallows half-written-PDB parse errors
# --------------------------------------------------------------------------- #
def _write_partial_pdb(outdir: Path, label: str, body: str):
    root = outdir / f"boltz_results_{label}_boltz_input"
    pred = root / "predictions" / f"{label}_boltz_input"
    pred.mkdir(parents=True, exist_ok=True)
    pdb = pred / f"{label}_boltz_input_model_1.pdb"
    pdb.write_text(body)
    return root, pdb


def test_load_existing_real_returns_none_on_empty_pdb(tmp_path):
    """If a Boltz subprocess was killed BEFORE it wrote anything to
    model_1.pdb, the file is 0 bytes. The cache load must reject it
    (returning None) so the caller does a fresh prediction. Crashing
    or returning a zero-residue Complex would silently corrupt the
    downstream pipeline."""
    cfg = ComplexPredictionConfig(use_kernels=False)
    lig = Ligand(id="L", smiles="CCO")
    _write_partial_pdb(tmp_path, "wt", "")  # 0 bytes
    cx = _load_existing_real("wt", "ACDE", lig, cfg, tmp_path)
    assert cx is None, (
        "empty PDB should be treated as a cache miss, not a hit"
    )


def test_load_existing_real_returns_none_on_garbage_pdb(tmp_path):
    """If the killed Boltz wrote a few bytes of garbage (or the PDB is
    truncated mid-ATOM record), the parser will raise. We don't want
    the whole pipeline to abort - we want a cache miss so the caller
    re-predicts."""
    cfg = ComplexPredictionConfig(use_kernels=False)
    lig = Ligand(id="L", smiles="CCO")
    # 100 bytes of utter garbage; not a PDB.
    _write_partial_pdb(tmp_path, "mut_X", "!!! not a pdb file !!!\n" * 8)
    cx = _load_existing_real("mut_X", "ACDE", lig, cfg, tmp_path)
    assert cx is None, (
        "malformed PDB should be a cache miss; the pipeline must fall "
        "back to a fresh prediction instead of crashing"
    )


def test_load_existing_real_still_works_on_intact_pdb(tmp_path):
    """The defensive guard must not regress the happy path: a
    well-formed PDB still parses and yields a Complex."""
    cfg = ComplexPredictionConfig(use_kernels=False)
    lig = Ligand(id="L", smiles="CCO")
    # CA-only PDB - enough for _parse_real_structure to succeed.
    lines = []
    for i, aa in enumerate("ACDE", start=1):
        aa3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU"}[aa]
        lines.append(
            f"ATOM  {i:>5}  CA  {aa3} A{i:>4}    "
            f"{i*1.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00 50.00           C"
        )
    lines.append("END")
    _write_partial_pdb(tmp_path, "wt_ok", "\n".join(lines) + "\n")
    cx = _load_existing_real("wt_ok", "ACDE", lig, cfg, tmp_path)
    assert cx is not None
    assert "boltz_results_wt_ok_boltz_input" in str(cx.path)
