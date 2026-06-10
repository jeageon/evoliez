"""Guards against the 'failed measurement looks like a perfect result' theme
(ultra-review #12/#13/#14/#28). A failed/unverifiable measurement must NOT
default to the most-favourable value (which passes filters and inflates rank).
"""

from evoliez.adapters.diffdock import _parse_diffdock
from evoliez.adapters.foldx import _parse_foldx
from evoliez.adapters.rosetta import _parse_rosetta
from evoliez.stages.s09_nonmd_validation import _redock_metrics


# --- #12: unverifiable redock pose (rmsd None) must be neutral, not perfect --- #
def test_redock_metrics_unverified_pose_is_neutral():
    c, u, esc = _redock_metrics(None)
    assert c == 0.5            # NOT 1.0 (the previous best-possible default)
    assert u >= 0.5            # uncertainty surfaced
    assert esc is False        # unknown -> not asserted as escaped


def test_redock_metrics_verified_poses():
    c0, u0, esc0 = _redock_metrics(0.0)
    assert c0 == 1.0 and esc0 is False
    c5, u5, esc5 = _redock_metrics(5.0)
    assert c5 == 0.0 and esc5 is True       # rmsd 5 > 4.5 -> ligand escaped


# --- #14/#28: failed ddG parse must NOT read as 0.0 (best stability) --- #
def test_foldx_parse_returns_none_when_no_output(tmp_path):
    assert _parse_foldx(tmp_path) is None     # was 0.0 (max stability, passes filter)


def test_rosetta_parse_returns_none_when_no_output(tmp_path):
    assert _parse_rosetta(tmp_path) is None


# --- #13: DiffDock must not fabricate -7.0 when a bare rank1.sdf coexists --- #
def test_diffdock_prefers_confidence_file_over_bare_rank1(tmp_path):
    # a bare rank1.sdf sorts BEFORE rank1_confidence-*.sdf ('.' < '_'); the
    # score must come from the confidence-bearing file, not a fabricated -7.0.
    (tmp_path / "rank1.sdf").write_text("bare pose, no confidence token\n")
    (tmp_path / "rank1_confidence-0.42.sdf").write_text("real pose\n")
    assert _parse_diffdock(tmp_path) == -0.42
