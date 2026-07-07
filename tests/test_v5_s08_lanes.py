"""ROADMAP_V5 step 5 — s08-computable mechanism-plausibility lanes rescue low-ML but
evolutionarily-tolerated / ligand-competent candidates that a pure ml_score cut would drop."""
from types import SimpleNamespace

from evoliez.ranking.multi_lane import LaneConfig, lane_counts, select_multi_lane


def _c(cid, ml, msa=0.0, ig=0.0, pos=0):
    return SimpleNamespace(
        candidate_id=cid,
        scores={"ml_score": ml, "msa_permissiveness": msa, "interaction_gain": ig},
        mutations=[SimpleNamespace(position=pos)],
        details={},
    )


def test_evolutionary_lane_rescues_low_ml_high_msa():
    # a candidate with the LOWEST ml_score but the HIGHEST msa_permissiveness
    cands = [_c(f"m{i}", ml=1.0 - i * 0.1, pos=i) for i in range(5)]
    rescue = _c("rescue", ml=-5.0, msa=0.99, pos=99)   # ml_high would never pick this
    cands.append(rescue)
    # ml_high takes 2; evolutionary_high takes 1 (the high-MSA rescue)
    sel = select_multi_lane(cands, LaneConfig(
        enabled=True, from_ml_high=2, from_evolutionary_high=1,
        from_ligand_high=0, from_diversity=0, low_ml_controls=0))
    ids = {c.candidate_id for c in sel}
    assert "rescue" in ids, "high-MSA low-ML candidate must be rescued by evolutionary_high"
    assert rescue.details["selection_lane"] == "evolutionary_high"


def test_ligand_lane_rescues_low_ml_high_interaction_gain():
    cands = [_c(f"m{i}", ml=1.0 - i * 0.1, pos=i) for i in range(5)]
    rescue = _c("lig", ml=-5.0, ig=9.0, pos=99)
    cands.append(rescue)
    sel = select_multi_lane(cands, LaneConfig(
        enabled=True, from_ml_high=2, from_evolutionary_high=0,
        from_ligand_high=1, from_diversity=0, low_ml_controls=0))
    assert "lig" in {c.candidate_id for c in sel}


def test_small_pool_union_includes_all_no_spurious_low_ml_fail():
    # REGRESSION (smoke crash): 419-candidate pool + from_ml_high=400 -> ml_high takes 400, the
    # other lanes take the remaining 19, so the union includes ALL 419. low_ml_control is then
    # empty simply because nothing is left -- NOT a vanished safety net. The s08 guard keys on
    # len(top) < len(candidates) (candidates EXCLUDED), which is 0 here, so it must NOT fire.
    cands = [_c(f"m{i}", ml=float(i), pos=i % 7) for i in range(419)]
    sel = select_multi_lane(cands, LaneConfig(
        enabled=True, from_ml_high=400, from_evolutionary_high=12, from_ligand_high=12,
        from_diversity=8, low_ml_controls=8))
    ids = {c.candidate_id for c in sel}
    assert len(ids) == 419                                  # union includes every candidate
    assert 419 - len(ids) == 0                              # nothing excluded -> guard won't fire


def test_union_never_shrinks_below_ml_high_baseline():
    cands = [_c(f"m{i}", ml=float(i), pos=i) for i in range(20)]
    sel = select_multi_lane(cands, LaneConfig(
        enabled=True, from_ml_high=10, from_evolutionary_high=3,
        from_ligand_high=3, from_diversity=2, low_ml_controls=2))
    # ml_high alone would be 10; the union only ADDS (each candidate once)
    assert len(sel) >= 10
    assert len({c.candidate_id for c in sel}) == len(sel)  # no duplicates
    assert lane_counts(sel).get("ml_high", 0) == 10
