"""External benchmark loader + baselines + pose sanity (Bench-3)."""

from evoliez.ml.benchmark import (
    baseline_rankings,
    compare_baselines,
    load_external_benchmark,
)
from evoliez.ml.pose_validity import pose_sanity
from evoliez.types import Candidate, LigandAtom, Mutation, ProteinStructure, Residue


def test_load_proteingym_style_csv(tmp_path):
    f = tmp_path / "DMS.csv"
    f.write_text(
        "mutant,DMS_score\n"
        "A1K,2.5\nC2D,2.4\nE3F,0.1\nG4H,-1.0\nH5I:K6L,1.0\n"
    )
    rows = load_external_benchmark(f, beneficial_q=0.75, deleterious_q=0.25)
    by = {r["mutation"]: r for r in rows}
    assert by["A1K"]["label"] == "beneficial"     # top quantile
    assert by["G4H"]["label"] == "deleterious"    # bottom quantile
    assert "H5I;K6L" in by                         # ':' -> ';'


def _cand(cid, mut, **feat):
    c = Candidate(cid, [Mutation(mut[0], int(mut[1:-1]), mut[-1])], "g")
    c.scores["final_score"] = feat.pop("final", 0.0)
    c.details["features"] = feat
    return c


def test_baselines_distinct_and_scored():
    cands = [
        _cand("c1", "E85S", final=9.0, conservation=0.2,
              msa_permissiveness=0.9, interaction_gain=0.8),
        _cand("c2", "F84L", final=1.0, conservation=0.9,
              msa_permissiveness=0.1, interaction_gain=0.1),
    ]
    rk = baseline_rankings(cands)
    assert set(rk) == {"full_model", "random", "conservation_only",
                       "msa_only", "interaction_only"}
    assert rk["full_model"][0].candidate_id == "c1"      # by final_score
    assert rk["msa_only"][0].candidate_id == "c1"        # by permissiveness
    cmp = compare_baselines(
        cands, [{"mutation": "E85S", "label": "beneficial",
                 "activity": 2.0}], k=1)
    assert "full_model" in cmp and "auroc_beneficial" in cmp["full_model"]


def test_pose_sanity_flags_clash_and_escape():
    # Ligand sits offset from the protein backbone so the new receptor-clash
    # check (lig-atom <-> CA < 0.9 A) doesn't trip on a degenerate overlap.
    good = [LigandAtom(id=f"C{i}", element="C", coord=(i * 2.0, 2.0, 0))
            for i in range(4)]
    struct = ProteinStructure(
        sequence="AAAA",
        residues=[Residue(index=i + 1, aa="A", ca=(i, 0, 0)) for i in range(4)],
    )
    s = pose_sanity(good, struct)
    assert s["clash_free"] and s["valid"]
    assert s["receptor_clashes"] == 0
    assert s["status"] == "valid"

    clash = [LigandAtom(id="C0", element="C", coord=(0, 2.0, 0)),
             LigandAtom(id="C1", element="C", coord=(0.05, 2.0, 0)),
             LigandAtom(id="C2", element="C", coord=(0.06, 2.0, 0))]
    s2 = pose_sanity(clash, struct)
    assert s2["internal_clashes"] >= 1 and not s2["valid"]
    assert s2["status"] == "invalid"
    assert any("internal" in r for r in s2["reasons"])

    # New: protein-ligand collision is flagged (ligand atom on top of a CA).
    on_top = [LigandAtom(id="C0", element="C", coord=(0, 0, 0))] + [
        LigandAtom(id=f"C{i}", element="C", coord=(i * 3.0, 5.0, 0))
        for i in range(1, 4)
    ]
    s3 = pose_sanity(on_top, struct)
    assert s3["receptor_clashes"] >= 1 and not s3["valid"]
    assert any("protein-ligand" in r for r in s3["reasons"])
