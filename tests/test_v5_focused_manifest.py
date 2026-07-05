"""ROADMAP_V5 focused-run: EVOLIEZ_SEED_CANDIDATES_CSV seeds s07 with EXACTLY the reviewer-locked
manifest (no stochastic generation, no s08b fold-queue blow-up), fully reproducible. Env-gated so
it never adds a config field (config_sha1 purge-trap). WT/control rows are skipped; a wt/sequence
mismatch fails loud so a numbering error can't silently mutate the wrong residue."""
import tempfile
from pathlib import Path

import pytest

from evoliez.stages.s07_mutation_gen import MutationGenStage


class _R:
    def __init__(self, aa):
        self.aa = aa


# structure residues the manifest positions map onto (only the ones we test)
_RES = {430: _R("G"), 433: _R("S"), 407: _R("G"), 438: _R("P"), 264: _R("Y")}


def _csv(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    f.write(text)
    f.close()
    return f.name


def _stage():
    return MutationGenStage.__new__(MutationGenStage)   # no __init__ needed for the pure method


def test_manifest_seeds_exact_mutations_and_skips_wt():
    path = _csv(
        "mutation_string,category,rationale\n"
        "WT,control,parental baseline\n"
        "G430R;S433F;G407K,lead,integrated lead\n"
        "P438N,clean_single,cleanest single\n"
    )
    cands = _stage()._candidates_from_manifest(path, _RES)
    assert len(cands) == 2                                   # WT row skipped
    assert cands[0].candidate_id == "mut_00000"
    assert [(m.wt, m.position, m.mut) for m in cands[0].mutations] == [
        ("G", 430, "R"), ("S", 433, "F"), ("G", 407, "K")]
    assert cands[0].generator == "focused_manifest"
    assert cands[0].details["manifest_category"] == "lead"
    assert [(m.wt, m.position, m.mut) for m in cands[1].mutations] == [("P", 438, "N")]


def test_manifest_fails_loud_on_sequence_mismatch():
    # manifest says A430R but the structure has G at 430 -> must raise, not silently mutate
    path = _csv("mutation_string\nA430R\n")
    with pytest.raises(RuntimeError, match="numbering mismatch|disagree"):
        _stage()._candidates_from_manifest(path, _RES)


def test_manifest_empty_or_wt_only_raises():
    path = _csv("mutation_string,category\nWT,control\nparental,control\n")
    with pytest.raises(RuntimeError, match="0 mutation candidates"):
        _stage()._candidates_from_manifest(path, _RES)


def test_manifest_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        _stage()._candidates_from_manifest("/no/such/manifest.csv", _RES)


def test_manifest_unknown_position_is_accepted_when_structure_lacks_it():
    # a position not in res (res.get -> None) is NOT a mismatch (can't validate) -> accepted
    path = _csv("mutation_string\nK999A\n")
    cands = _stage()._candidates_from_manifest(path, _RES)
    assert len(cands) == 1
    assert (cands[0].mutations[0].wt, cands[0].mutations[0].position) == ("K", 999)
