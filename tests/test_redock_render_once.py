"""RENDER-ONCE receptor reuse + diffdock version memo (perf, quality-neutral).

These cover the s06b perf fixes that must stay OUTPUT-identical:

* ``gnina.redock_all`` / ``diffdock.redock_all`` / ``diffdock.redock_batch``
  accept a pre-rendered ``receptor_pdb``; when supplied the adapter COPIES it to
  the per-candidate ``rec`` path and does NOT call ``full_atom_receptor_pdb``
  (the per-rep render runs once in the caller instead of once per engine/target).
* With ``receptor_pdb=None`` the render path is unchanged (still calls
  ``full_atom_receptor_pdb``).
* The DiffDock version string is memoized: the cold-interpreter ``tool_version``
  subprocess fires AT MOST once per process across many batch calls.

All real-backend; the GPU tools (`gnina`, `python -m inference`) are mocked, so
this runs on the CPU-only dev box.
"""

from __future__ import annotations

from pathlib import Path

from evoliez.adapters import diffdock, gnina
from evoliez.config import Backend, DockingConfig
from evoliez.types import LigandAtom, ProteinStructure


def _ref():
    return [
        LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="O1", element="O", coord=(1.2, 0.0, 0.0)),
        LigandAtom(id="N2", element="N", coord=(0.0, 1.3, 0.0)),
    ]


def _cfg():
    return DockingConfig(poses_per_candidate=2)


_REC_BYTES = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)

_GNINA_SDF = (
    "lig\n     RDKit          3D\n\n"
    "  3  2  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    1.2000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    0.0000    1.3000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0\n  1  3  1  0\nM  END\n"
    "> <minimizedAffinity>\n-9.42\n\n"
    "> <CNNscore>\n0.873\n\n"
    "> <CNNaffinity>\n6.21\n\n"
    "$$$$\n"
)

_DD_SDF = (
    "lig\n     RDKit          3D\n\n"
    "  3  2  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    1.2000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    0.0000    1.3000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0\n  1  3  1  0\nM  END\n$$$$\n"
)


# --------------------------------------------------------------------------- #
# gnina RENDER-ONCE
# --------------------------------------------------------------------------- #
def test_gnina_receptor_pdb_skips_render(tmp_path, monkeypatch):
    """A supplied receptor_pdb is copied to <cand>_rec.pdb; full_atom_receptor_pdb
    is NEVER called; gnina runs against that exact receptor."""
    pre = tmp_path / "prerendered_rec.pdb"
    pre.write_text(_REC_BYTES)
    workdir = tmp_path / "wd"

    calls = {"full_atom": 0}

    def _no_render(*a, **k):
        calls["full_atom"] += 1
        return True

    seen_rec = {}

    def _fake_run(cmd, *a, **k):
        # capture the receptor path passed to gnina (-r <rec>) and emit an SDF
        rec_idx = cmd.index("-r") + 1
        seen_rec["rec"] = cmd[rec_idx]
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_text(_GNINA_SDF)
        return None

    monkeypatch.setattr(gnina, "require", lambda *a, **k: None)
    monkeypatch.setattr(gnina, "apply_gpu_selection", lambda *a, **k: None)
    monkeypatch.setattr(gnina, "full_atom_receptor_pdb", _no_render)
    monkeypatch.setattr(gnina, "tool_version", lambda *a, **k: "gnina 1.0")
    monkeypatch.setattr(gnina, "run", _fake_run)

    poses = gnina.redock_all(
        "cand1", ProteinStructure(sequence="AG", residues=[]), _ref(), _cfg(),
        workdir, instability=0.05, backend=Backend.real, dry_run=False,
        receptor_pdb=pre,
    )

    assert calls["full_atom"] == 0                  # render skipped
    rec = workdir / "cand1_rec.pdb"
    assert rec.exists() and rec.read_text() == _REC_BYTES   # copied verbatim
    assert seen_rec["rec"] == str(rec)              # gnina pointed at the copy
    assert poses and poses[0].score == -9.42        # parsed the real SDF


def test_gnina_no_receptor_pdb_still_renders(tmp_path, monkeypatch):
    """Default (receptor_pdb=None) path is unchanged: full_atom_receptor_pdb IS
    called exactly as before."""
    workdir = tmp_path / "wd"
    calls = {"full_atom": 0}

    def _render(structure, out_path, keep_het_chains=None):
        calls["full_atom"] += 1
        Path(out_path).write_text(_REC_BYTES)
        return True

    def _fake_run(cmd, *a, **k):
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_text(_GNINA_SDF)
        return None

    monkeypatch.setattr(gnina, "require", lambda *a, **k: None)
    monkeypatch.setattr(gnina, "apply_gpu_selection", lambda *a, **k: None)
    monkeypatch.setattr(gnina, "full_atom_receptor_pdb", _render)
    monkeypatch.setattr(gnina, "tool_version", lambda *a, **k: "gnina 1.0")
    monkeypatch.setattr(gnina, "run", _fake_run)

    poses = gnina.redock_all(
        "cand1", ProteinStructure(sequence="AG", residues=[]), _ref(), _cfg(),
        workdir, instability=0.05, backend=Backend.real, dry_run=False,
    )
    assert calls["full_atom"] == 1
    assert poses and poses[0].score == -9.42


# --------------------------------------------------------------------------- #
# diffdock RENDER-ONCE (single-target + batch)
# --------------------------------------------------------------------------- #
def _wire_diffdock(monkeypatch, render_counter):
    """Mock diffdock's real path: count full_atom renders, write rank SDFs into
    each complex's out dir on run(), and stub require/gpu/tool_version."""

    def _render(structure, out_path, keep_het_chains=None):
        render_counter[0] += 1
        Path(out_path).write_text(_REC_BYTES)
        return True

    def _fake_run(cmd, *a, **k):
        # Single path parses ranks directly under --out_dir; batch parses under
        # --out_dir/<cid>. Mirror DiffDock: write a per-<cid> sub-dir for every
        # CSV row, and ALSO at the out_dir root when there is exactly one row
        # (the single-target layout).
        out_idx = cmd.index("--out_dir") + 1
        out_dir = Path(cmd[out_idx])
        csv_idx = cmd.index("--protein_ligand_csv") + 1
        rows = Path(cmd[csv_idx]).read_text().splitlines()[1:]
        for row in rows:
            cid = row.split(",")[0]
            d = out_dir / cid
            d.mkdir(parents=True, exist_ok=True)
            (d / "rank1_confidence-0.53.sdf").write_text(_DD_SDF)
        if len(rows) == 1:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "rank1_confidence-0.53.sdf").write_text(_DD_SDF)
        return None

    monkeypatch.setattr(diffdock, "require", lambda *a, **k: None)
    monkeypatch.setattr(diffdock, "apply_gpu_selection", lambda *a, **k: None)
    monkeypatch.setattr(diffdock, "full_atom_receptor_pdb", _render)
    monkeypatch.setattr(diffdock, "run", _fake_run)


def test_diffdock_single_receptor_pdb_skips_render(tmp_path, monkeypatch):
    pre = tmp_path / "pre_rec.pdb"
    pre.write_text(_REC_BYTES)
    rc = [0]
    _wire_diffdock(monkeypatch, rc)
    monkeypatch.setattr(diffdock, "_DIFFDOCK_VERSION", None, raising=False)
    monkeypatch.setattr(diffdock, "tool_version", lambda *a, **k: "diffdock 1.1")

    workdir = tmp_path / "wd"
    poses = diffdock.redock_all(
        "cand1", ProteinStructure(sequence="AG", residues=[]), _ref(), _cfg(),
        workdir, instability=0.05, smiles="CCO", backend=Backend.real,
        dry_run=False, receptor_pdb=pre,
    )
    assert rc[0] == 0                               # render skipped
    rec = workdir / "cand1_rec.pdb"
    assert rec.exists() and rec.read_text() == _REC_BYTES
    assert poses and poses[0].score == -0.53


def test_diffdock_batch_receptor_pdb_map_skips_render(tmp_path, monkeypatch):
    """A {cid: Path} map: cids present are copied (no render); cids absent render
    normally."""
    pre_a = tmp_path / "pre_a.pdb"
    pre_a.write_text(_REC_BYTES)
    rc = [0]
    _wire_diffdock(monkeypatch, rc)
    monkeypatch.setattr(diffdock, "_DIFFDOCK_VERSION", None, raising=False)
    monkeypatch.setattr(diffdock, "tool_version", lambda *a, **k: "diffdock 1.1")

    out_root = tmp_path / "out"
    struct = ProteinStructure(sequence="AG", residues=[])
    tasks = [
        ("A", struct, _ref(), "CCO"),
        ("B", struct, _ref(), "CCN"),   # no pre-rendered receptor -> renders
    ]
    res = diffdock.redock_batch(
        tasks, out_root, _cfg(), backend=Backend.real, dry_run=False,
        receptor_pdb={"A": pre_a},
    )
    # A used the supplied receptor verbatim; B was rendered (1 render total).
    assert rc[0] == 1
    assert (out_root / "A_rec.pdb").read_text() == _REC_BYTES
    assert (out_root / "B_rec.pdb").exists()
    assert res["A"] and res["A"][0].score == -0.53
    assert res["B"] and res["B"][0].score == -0.53


def test_diffdock_batch_no_map_renders_all(tmp_path, monkeypatch):
    rc = [0]
    _wire_diffdock(monkeypatch, rc)
    monkeypatch.setattr(diffdock, "_DIFFDOCK_VERSION", None, raising=False)
    monkeypatch.setattr(diffdock, "tool_version", lambda *a, **k: "diffdock 1.1")
    out_root = tmp_path / "out"
    struct = ProteinStructure(sequence="AG", residues=[])
    tasks = [("A", struct, _ref(), "CCO"), ("B", struct, _ref(), "CCN")]
    diffdock.redock_batch(tasks, out_root, _cfg(), backend=Backend.real,
                          dry_run=False)
    assert rc[0] == 2                               # both rendered as before


# --------------------------------------------------------------------------- #
# diffdock VERSION MEMO
# --------------------------------------------------------------------------- #
def test_diffdock_version_memoized_across_batches(tmp_path, monkeypatch):
    """The cold-interpreter version probe fires at most once per process even
    across multiple batch runs."""
    rc = [0]
    _wire_diffdock(monkeypatch, rc)
    monkeypatch.setattr(diffdock, "_DIFFDOCK_VERSION", None, raising=False)

    tv_calls = {"n": 0}

    def _tv(*a, **k):
        tv_calls["n"] += 1
        return "diffdock 1.1"

    monkeypatch.setattr(diffdock, "tool_version", _tv)

    struct = ProteinStructure(sequence="AG", residues=[])
    for i in range(3):
        diffdock.redock_batch(
            [(f"C{i}", struct, _ref(), "CCO")], tmp_path / f"out{i}",
            _cfg(), backend=Backend.real, dry_run=False,
        )
    assert tv_calls["n"] == 1                        # computed once, then cached


def test_diffdock_version_helper_memoizes(monkeypatch):
    """_diffdock_version() itself: underlying tool_version called once."""
    monkeypatch.setattr(diffdock, "_DIFFDOCK_VERSION", None, raising=False)
    n = {"v": 0}

    def _tv(*a, **k):
        n["v"] += 1
        return "diffdock 9.9"

    monkeypatch.setattr(diffdock, "tool_version", _tv)
    assert diffdock._diffdock_version() == "diffdock 9.9"
    assert diffdock._diffdock_version() == "diffdock 9.9"
    assert n["v"] == 1


# --------------------------------------------------------------------------- #
# _complex_out_dir scoped fallbacks
# --------------------------------------------------------------------------- #
def test_complex_out_dir_fast_path(tmp_path):
    d = tmp_path / "myc"
    d.mkdir()
    (d / "rank1_confidence-0.5.sdf").write_text(_DD_SDF)
    assert diffdock._complex_out_dir(tmp_path, "myc") == d


def test_complex_out_dir_scoped_fallback_one_level(tmp_path):
    """Nested-one-level layout: out_root/<something-cid>/<inner>/rank*.sdf is
    still found via the cid-scoped glob (not a full rglob)."""
    top = tmp_path / "job_myc_run"
    inner = top / "predictions"
    inner.mkdir(parents=True)
    (inner / "rank1_confidence-0.5.sdf").write_text(_DD_SDF)
    # a sibling complex's dir must NOT be returned
    other = tmp_path / "job_other_run"
    other.mkdir()
    (other / "rank1_confidence-9.9.sdf").write_text(_DD_SDF)
    assert diffdock._complex_out_dir(tmp_path, "myc") == inner


def test_complex_out_dir_none_when_absent(tmp_path):
    (tmp_path / "unrelated").mkdir()
    assert diffdock._complex_out_dir(tmp_path, "myc") is None
