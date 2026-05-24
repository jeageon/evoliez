"""s08_reranker: reserved-slots guarantee for known_binding_site.

Cheap-run diagnosis: even with the +0.6 at_binding_site prior, the
Tishkov D222S/H/A/N/T/Q family was still cut by `top_for_redocking`
(conservation penalty + low msa_permissiveness), giving recall@10 = 0
in the cheap_run despite binding_site_scan generating all of them.

These tests pin _promote_binding_site_reservations directly so they
don't need the heavy RunContext / Boltz stack.
"""

from __future__ import annotations

from evoliez.stages.s08_reranker import _promote_binding_site_reservations
from evoliez.types import Candidate, Mutation


def _cand(cid, pos, mut_aa="X", score=0.0, gen="test"):
    c = Candidate(
        candidate_id=cid,
        mutations=[Mutation(wt="A", position=pos, mut=mut_aa)],
        generator=gen,
    )
    c.scores["ml_score"] = float(score)
    return c


def test_no_op_when_n_reserve_zero():
    cands = sorted([_cand(f"c{i}", 100 + i, score=10 - i) for i in range(5)],
                   key=lambda c: -c.scores["ml_score"])
    top = cands[:3]
    new_top, n = _promote_binding_site_reservations(top, cands, {200}, 0)
    assert n == 0
    assert [c.candidate_id for c in new_top] == [c.candidate_id for c in top]


def test_no_op_when_binding_site_empty():
    cands = sorted([_cand(f"c{i}", 100 + i, score=10 - i) for i in range(5)],
                   key=lambda c: -c.scores["ml_score"])
    top = cands[:3]
    new_top, n = _promote_binding_site_reservations(top, cands, set(), 1)
    assert n == 0


def test_promotes_one_per_binding_site_position():
    """D222 case: position 222 (binding_site) has no candidate in top;
    the best tail candidate at 222 must be swapped in, displacing the
    lowest-scoring top entry that is NOT itself a binding-site reserver."""
    # top has 3 non-binding-site candidates (positions 100/101/102,
    # scores 10/9/8). The tail has D222S at score 5 and a non-bs at 7.
    cands = sorted([
        _cand("c100", 100, score=10),       # top
        _cand("c101", 101, score=9),        # top
        _cand("c102", 102, score=8),        # top (lowest in top, will be displaced)
        _cand("c103", 103, score=7),        # tail (non-binding-site)
        _cand("c222", 222, mut_aa="S", score=5),  # tail at binding-site 222
    ], key=lambda c: -c.scores["ml_score"])
    top = cands[:3]
    new_top, n = _promote_binding_site_reservations(top, cands, {222}, 1)
    assert n == 1
    ids = {c.candidate_id for c in new_top}
    assert "c222" in ids                                  # promoted
    assert "c102" not in ids                              # displaced (lowest top)
    assert len(new_top) == 3                              # cap preserved
    # remaining top should be sorted by score desc
    scores = [c.scores["ml_score"] for c in new_top]
    assert scores == sorted(scores, reverse=True)


def test_does_not_displace_an_existing_binding_site_entry():
    """If the lowest-ranked top entry is itself covering ANOTHER binding-
    site position, it must not be displaced (preserve coverage)."""
    cands = sorted([
        _cand("c100", 100, score=10),
        _cand("c101", 101, score=9),
        _cand("c148", 148, score=8),       # bs (148) - in top, do NOT displace
        _cand("c222", 222, mut_aa="S", score=5),
        _cand("c500", 500, score=4),       # tail non-bs
    ], key=lambda c: -c.scores["ml_score"])
    top = cands[:3]
    new_top, n = _promote_binding_site_reservations(top, cands, {148, 222}, 1)
    assert n == 1
    ids = {c.candidate_id for c in new_top}
    assert "c148" in ids                                  # protected
    assert "c222" in ids                                  # promoted
    # The displaced one must be the highest-scoring non-binding-site entry
    # eligible to drop (= c101). c100 stays (it's not eligible? actually
    # both c100 and c101 are non-bs; lowest-ranked non-bs is c101).
    assert "c101" not in ids


def test_no_promotion_when_quota_already_met():
    """If binding_site coverage already at n_reserve, no swaps happen."""
    cands = sorted([
        _cand("c222", 222, mut_aa="S", score=10),
        _cand("c222b", 222, mut_aa="N", score=9),
        _cand("c100", 100, score=8),
        _cand("c500", 500, score=4),
    ], key=lambda c: -c.scores["ml_score"])
    top = cands[:3]
    new_top, n = _promote_binding_site_reservations(top, cands, {222}, 1)
    assert n == 0                                         # already covered
    assert [c.candidate_id for c in new_top] == [c.candidate_id for c in top]


def test_gives_up_gracefully_when_no_displaceable_entries():
    """If every top entry already covers a binding-site position, we can't
    displace any without breaking another reservation -> stop, no error."""
    cands = sorted([
        _cand("c148", 148, score=10),
        _cand("c283", 283, score=9),
        _cand("c333", 333, score=8),
        _cand("c222", 222, mut_aa="S", score=5),
    ], key=lambda c: -c.scores["ml_score"])
    top = cands[:3]
    new_top, n = _promote_binding_site_reservations(top, cands,
                                                    {148, 222, 283, 333}, 1)
    assert n == 0                  # 222 wanted, but all top are reservers
    assert [c.candidate_id for c in new_top] == [c.candidate_id for c in top]
