"""DiffDock rank-1 pose selection must pick rank 1, not rank 10.

Regression: `_rank1_pose_files` globbed `rank1*confidence*.sdf`, which also
matches rank10..19; `sorted()` then puts 'rank10' before 'rank1_' ('0' < '_'),
so the WORST pose was recorded (e.g. WT diffdock score -3.4 instead of -0.53).
"""

from __future__ import annotations

from evoliez.adapters.diffdock import _parse_diffdock, _rank1_pose_files


def _make(d):
    d.mkdir(parents=True, exist_ok=True)
    for n in [
        "rank1.sdf",                      # bare (no confidence)
        "rank1_confidence-0.53.sdf",      # the real best pose
        "rank2_confidence-0.55.sdf",
        "rank9_confidence-2.10.sdf",
        "rank10_confidence-3.40.sdf",     # the WORST — the old bug picked this
        "rank11_confidence-3.80.sdf",
    ]:
        (d / n).write_text("")


def test_rank1_selected_not_rank10(tmp_path):
    _make(tmp_path / "wt_dd_out" / "wt")
    files = _rank1_pose_files(tmp_path)
    assert files, "no rank-1 file found"
    assert files[0].name == "rank1_confidence-0.53.sdf"
    assert all("rank10" not in f.name and "rank11" not in f.name for f in files)


def test_parse_uses_rank1_confidence(tmp_path):
    _make(tmp_path / "wt_dd_out" / "wt")
    assert _parse_diffdock(tmp_path) == -0.53  # best pose, NOT -3.4


def test_bare_rank1_only(tmp_path):
    # only a bare rank1.sdf -> still selected (no confidence token -> 0.0 score)
    d = tmp_path / "out"
    d.mkdir(parents=True)
    (d / "rank1.sdf").write_text("")
    (d / "rank10.sdf").write_text("")
    files = _rank1_pose_files(tmp_path)
    assert [f.name for f in files] == ["rank1.sdf"]


def test_no_files(tmp_path):
    assert _rank1_pose_files(tmp_path) == []
    assert _parse_diffdock(tmp_path) == 0.0
