"""Ultra-review P0/P1 adapter-honesty regressions.

Each test locks in a fix that stops an external-tool adapter from silently
laundering a failure / wrong-input into a plausible-looking result:

1. rosetta / foldx / ligandmpnn refuse a CA-only structure on the REAL
   backend (full side chains required) instead of feeding a stick figure to
   cartesian_ddg / FoldX / LigandMPNN.
2. vina / gnina / diffdock emit a *skipped* Pose (not a fabricated mock pose)
   when a REAL run produces no output (crash / OOM / killed).
3. diffdock writes its input CSV via csv.writer, so a comma in the SMILES no
   longer corrupts the columns.
4. remote_msa returns None when polling never reaches COMPLETE, so a
   partial/empty a3m is never treated as a valid MSA.
5. msa_tools logs loudly and tags the synthetic fallback when a REAL homolog
   search yields no output / 0 hits.

All subprocess invocations are mocked - no real binaries are required.
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest  # noqa: F401  (imported for parity with sibling test modules)

from evoliez.adapters import (
    diffdock, foldx, gnina, ligandmpnn, msa_tools, remote_msa, rosetta, vina,
)
from evoliez.config import (
    Backend, DockingConfig, HomologConfig, MutationGenConfig, StabilityConfig,
)
from evoliez.types import (
    Complex, Ligand, LigandAtom, Mutation, ProteinStructure, Residue,
)

CA_PDB = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00\n"
    "ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 80.00\n"
    "END\n"
)
FULL_PDB = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00\n"
    "ATOM      2  CA  ALA A   1       0.500   0.500   0.500  1.00 80.00\n"
    "ATOM      3  C   ALA A   1       1.000   1.000   1.000  1.00 80.00\n"
    "ATOM      4  O   ALA A   1       1.500   1.500   1.500  1.00 80.00\n"
    "ATOM      5  CB  ALA A   1       2.000   2.000   2.000  1.00 80.00\n"
    "END\n"
)


def _struct(pdb_path: Path | None) -> ProteinStructure:
    return ProteinStructure(
        sequence="AG",
        residues=[
            Residue(index=1, aa="A", ca=(0.0, 0.0, 0.0), sasa=0.2),
            Residue(index=2, aa="G", ca=(3.8, 0.0, 0.0), sasa=0.6),
        ],
        pdb_path=str(pdb_path) if pdb_path is not None else None,
    )


def _ca_struct(tmp_path: Path) -> ProteinStructure:
    p = tmp_path / "ca.pdb"
    p.write_text(CA_PDB)
    return _struct(p)


def _full_struct(tmp_path: Path) -> ProteinStructure:
    p = tmp_path / "full.pdb"
    p.write_text(FULL_PDB)
    return _struct(p)


def _ref_atoms():
    return [
        LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0)),
        LigandAtom(id="O1", element="O", coord=(1.2, 0.0, 0.0)),
    ]


# --------------------------------------------------------------------------- #
# Bug 1: rosetta / foldx refuse CA-only on the real backend
# --------------------------------------------------------------------------- #
def test_rosetta_real_skips_on_ca_only(tmp_path):
    s = _ca_struct(tmp_path)
    muts = [Mutation(wt="A", position=1, mut="V")]
    with patch.object(rosetta, "run") as mrun, \
            patch.object(rosetta, "require") as mreq:
        out = rosetta.estimate_stability(
            "c1", s, muts, tmp_path / "work",
            backend=Backend.real, dry_run=False,
        )
    # No binary required, no subprocess launched - the skip runs even on a
    # host without Rosetta installed.
    mrun.assert_not_called()
    mreq.assert_not_called()
    assert out["skipped"] == "no_full_atom_structure"
    assert out["ddg_fold"] == 0.0
    assert out["clash_score"] == 0.0


def test_foldx_real_skips_on_ca_only(tmp_path):
    s = _ca_struct(tmp_path)
    cfg = StabilityConfig(method="foldx")
    muts = [Mutation(wt="A", position=1, mut="V")]
    with patch.object(foldx, "run") as mrun, \
            patch.object(foldx, "require") as mreq:
        out = foldx.estimate_stability(
            "c1", s, muts, cfg, tmp_path / "work",
            backend=Backend.real, dry_run=False,
        )
    mrun.assert_not_called()
    mreq.assert_not_called()
    assert out["skipped"] == "no_full_atom_structure"
    assert out["ddg_fold"] == 0.0


def test_foldx_real_runs_when_full_atom(tmp_path):
    """Positive control: a full-atom structure is NOT skipped - it proceeds to
    write the (real) PDB and invoke FoldX (here mocked)."""
    s = _full_struct(tmp_path)
    cfg = StabilityConfig(method="foldx")
    muts = [Mutation(wt="A", position=1, mut="V")]
    work = tmp_path / "work"
    with patch.object(foldx, "run") as mrun, \
            patch.object(foldx, "require"):
        out = foldx.estimate_stability(
            "c1", s, muts, cfg, work, backend=Backend.real, dry_run=False,
        )
    mrun.assert_called_once()
    assert "skipped" not in out
    # The PDB handed to FoldX is the REAL full-atom file, not a CA re-write.
    written = (work / "c1.pdb").read_text()
    assert "CB" in written and " N  " in written


# --------------------------------------------------------------------------- #
# Bug 2: ligandmpnn refuses CA-only on the real backend
# --------------------------------------------------------------------------- #
def test_ligandmpnn_real_skips_on_ca_only(tmp_path):
    s = _ca_struct(tmp_path)
    cx = Complex(structure=s, ligand=Ligand(id="LIG", smiles="CCO", atoms=[
        LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0)),
    ]))
    cfg = MutationGenConfig()
    with patch.object(ligandmpnn, "run") as mrun, \
            patch.object(ligandmpnn, "require") as mreq, \
            patch.object(ligandmpnn, "_resolve_lmpnn_install") as minst:
        out = ligandmpnn.design_sequences(
            cx, [1, 2], cfg, tmp_path / "work",
            backend=Backend.real, dry_run=False,
        )
    # No designs produced, and we never resolved the install or ran anything:
    # the refuse short-circuits before any of that.
    assert out == []
    mrun.assert_not_called()
    mreq.assert_not_called()
    minst.assert_not_called()


# --------------------------------------------------------------------------- #
# Bug 3: vina / gnina / diffdock - real run, no output -> skipped, not mock
# --------------------------------------------------------------------------- #
def test_vina_real_no_output_is_skipped_not_mock(tmp_path):
    s = _full_struct(tmp_path)  # full-atom so we pass the receptor guard
    ref = _ref_atoms()
    cfg = DockingConfig()
    # run() is a no-op: the vina out.pdbqt is never created.
    with patch.object(vina, "run"), patch.object(vina, "require"):
        pose = vina.redock(
            "c1", s, ref, cfg, tmp_path / "work",
            instability=0.0, backend=Backend.real, dry_run=False,
        )
    assert pose.skipped == "real_tool_no_output"
    assert pose.pose_validity_status == "unknown"
    assert pose.score == 0.0


def test_gnina_real_no_output_is_skipped_not_mock(tmp_path):
    s = _full_struct(tmp_path)
    ref = _ref_atoms()
    cfg = DockingConfig()
    with patch.object(gnina, "run"), patch.object(gnina, "require"), \
            patch.object(gnina, "apply_gpu_selection"):
        pose = gnina.redock(
            "c1", s, ref, cfg, tmp_path / "work",
            instability=0.0, backend=Backend.real, dry_run=False,
        )
    assert pose.skipped == "real_tool_no_output"
    assert pose.pose_validity_status == "unknown"
    assert pose.score == 0.0


def test_diffdock_real_no_output_is_skipped_not_mock(tmp_path):
    s = _full_struct(tmp_path)
    ref = _ref_atoms()
    cfg = DockingConfig()
    install = tmp_path / "ddinstall"
    install.mkdir()
    (install / "inference.py").write_text("# stub\n")
    with patch.object(diffdock, "run"), patch.object(diffdock, "require"), \
            patch.object(diffdock, "apply_gpu_selection"), \
            patch.object(diffdock, "_resolve_diffdock_install",
                         return_value=install):
        pose = diffdock.redock(
            "c1", s, ref, cfg, tmp_path / "work",
            instability=0.0, smiles="CCO", backend=Backend.real, dry_run=False,
        )
    assert pose.skipped == "real_tool_no_output"
    assert pose.pose_validity_status == "unknown"
    assert pose.score == 0.0


def test_vina_dry_run_still_returns_mock(tmp_path):
    """Regression guard: dry_run must keep returning a mock pose (NOT skipped)
    so --dry-run previews still produce a usable pose object."""
    s = _full_struct(tmp_path)
    ref = _ref_atoms()
    cfg = DockingConfig()
    with patch.object(vina, "run"), patch.object(vina, "require"):
        pose = vina.redock(
            "c1", s, ref, cfg, tmp_path / "work",
            instability=0.2, backend=Backend.real, dry_run=True,
        )
    assert pose.skipped is None
    assert pose.method == "vina"


# --------------------------------------------------------------------------- #
# Bug 4: diffdock CSV injection - a comma in SMILES round-trips correctly
# --------------------------------------------------------------------------- #
def test_diffdock_csv_quotes_comma_smiles(tmp_path):
    # A SMILES with commas (e.g. atom-map/charge notations DiffDock may see)
    # AND a receptor path with a comma - both would corrupt a raw f-string CSV.
    full = tmp_path / "weird,dir" / "full.pdb"
    full.parent.mkdir(parents=True)
    full.write_text(FULL_PDB)
    s = _struct(full)
    smiles = "C[N+](C)(C),weird"
    ref = _ref_atoms()
    cfg = DockingConfig()
    install = tmp_path / "ddinstall"
    install.mkdir()
    (install / "inference.py").write_text("# stub\n")
    work = tmp_path / "work"
    with patch.object(diffdock, "run"), patch.object(diffdock, "require"), \
            patch.object(diffdock, "apply_gpu_selection"), \
            patch.object(diffdock, "_resolve_diffdock_install",
                         return_value=install):
        diffdock.redock(
            "c1", s, ref, cfg, work,
            instability=0.0, smiles=smiles, backend=Backend.real, dry_run=False,
        )
    csv_path = work / "c1_input.csv"
    with csv_path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "complex_name", "protein_path", "ligand_description", "protein_sequence"
    ]
    # The comma-bearing SMILES and path survive as single fields.
    assert rows[1][0] == "c1"
    assert rows[1][1] == str(full)
    assert rows[1][2] == smiles
    assert len(rows[1]) == 4


# --------------------------------------------------------------------------- #
# Bug 5: remote_msa polling that never completes returns None
# --------------------------------------------------------------------------- #
def test_remote_msa_returns_none_when_polling_never_completes(tmp_path):
    class _Resp:
        def __init__(self, payload: bytes):
            self._payload = payload

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    download_called = {"n": 0}

    def fake_urlopen(req, *args, **kwargs):
        url = getattr(req, "full_url", req)
        if isinstance(url, str) and "/result/download/" in url:
            download_called["n"] += 1
            return _Resp(b">q\nACDE\n")
        if isinstance(url, str) and "/ticket/msa" not in url and \
                "/ticket/" in url:
            return _Resp(b'{"status":"RUNNING"}')  # never COMPLETE
        return _Resp(b"ticket-123")  # submission

    with patch.object(urllib.request, "urlopen", fake_urlopen), \
            patch.object(remote_msa.time, "sleep", lambda *_: None):
        out = remote_msa.fetch_msa("ACDE", tmp_path)
    assert out is None
    # Crucial: we must NOT have downloaded a (partial) a3m after a timeout.
    assert download_called["n"] == 0


def test_remote_msa_completes_then_downloads(tmp_path):
    """Positive control: once status hits COMPLETE the a3m is downloaded."""
    class _Resp:
        def __init__(self, payload: bytes):
            self._payload = payload

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    state = {"polls": 0}

    def fake_urlopen(req, *args, **kwargs):
        url = getattr(req, "full_url", req)
        if isinstance(url, str) and "/result/download/" in url:
            return _Resp(b">query\nACDE\n>h1\nACDF\n")
        if isinstance(url, str) and "/ticket/msa" not in url and \
                "/ticket/" in url:
            state["polls"] += 1
            return _Resp(b'{"status":"COMPLETE"}')
        return _Resp(b"ticket-xyz")

    with patch.object(urllib.request, "urlopen", fake_urlopen), \
            patch.object(remote_msa.time, "sleep", lambda *_: None):
        out = remote_msa.fetch_msa("ACDE", tmp_path)
    assert out is not None
    assert out[0][0] == "query"


# --------------------------------------------------------------------------- #
# Bug 6: msa_tools real search empty -> loud + tagged fallback (not silent)
# --------------------------------------------------------------------------- #
def test_msa_real_no_output_logs_and_tags_fallback(tmp_path, caplog):
    cfg = HomologConfig(method="mmseqs2", database="/fake/db")
    # run() is a no-op so the hits.m8 output file is never created.
    with patch.object(msa_tools, "run"), patch.object(msa_tools, "require"), \
            caplog.at_level("ERROR", logger="evoliez.msa"):
        hits = msa_tools.search_homologs(
            "ACDEFGHIKLMNPQRSTVWY", cfg, tmp_path / "work",
            backend=Backend.real, dry_run=False,
        )
    # Synthetic fallback IS returned (pipeline keeps running) BUT it is loud
    # and detectably tagged - never passed off as real hits.
    assert hits, "expected synthetic fallback homologs"
    assert all(h.annotation == "synthetic_real_fallback" for h in hits)
    assert any("synthetic fallback" in r.message for r in caplog.records)


def test_msa_real_zero_hits_logs_and_tags_fallback(tmp_path, caplog):
    cfg = HomologConfig(method="mmseqs2", database="/fake/db")

    def fake_run(cmd, **kwargs):
        # Tool "succeeds" but writes an empty output file -> 0 parsed hits.
        # mmseqs easy-search <query> <db> <out> <tmp> -> out is argv index 4.
        out = Path(cmd[4])
        out.write_text("")
        from evoliez.utils.subprocess_utils import RunResult
        return RunResult(cmd=list(cmd), returncode=0, stdout="", stderr="")

    with patch.object(msa_tools, "run", fake_run), \
            patch.object(msa_tools, "require"), \
            caplog.at_level("ERROR", logger="evoliez.msa"):
        hits = msa_tools.search_homologs(
            "ACDEFGHIKLMNPQRSTVWY", cfg, tmp_path / "work",
            backend=Backend.real, dry_run=False,
        )
    assert hits
    assert all(h.annotation == "synthetic_real_fallback" for h in hits)
    assert any("0 hits" in r.message for r in caplog.records)


def test_msa_mock_backend_keeps_plain_synthetic_tag(tmp_path):
    """The deliberate mock backend keeps the bare 'synthetic' tag - only the
    real-search fallback gets the louder 'synthetic_real_fallback' marker."""
    cfg = HomologConfig()
    hits = msa_tools.search_homologs(
        "ACDEFGHIKLMNPQRSTVWY", cfg, tmp_path / "work",
        backend=Backend.mock, dry_run=False,
    )
    assert hits
    assert all(h.annotation == "synthetic" for h in hits)
