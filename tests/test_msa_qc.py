"""v2 Phase G — MSA quality control (Neff via subfamily clusters, balance, source)."""
from evoliez.features.msa_qc import msa_qc


def test_neff_and_balance():
    clusters = [1] * 6 + [2] * 3 + [3]          # 10 seqs, 3 clusters (6/3/1)
    qc = msa_qc(10, cluster_ids=clusters)
    assert qc["depth"] == 10
    assert qc["neff_clusters"] == 3 and qc["n_subfamilies"] == 3
    assert qc["largest_subfamily_frac"] == 0.6   # 6/10
    assert qc["subfamily_balance"] == 0.3        # 3/10


def test_redundant_msa_flagged():
    qc = msa_qc(100, cluster_ids=[1] * 99 + [2])  # one dominant subfamily
    assert qc["largest_subfamily_frac"] == 0.99 and qc["neff_clusters"] == 2


def test_source_real_fraction():
    qc = msa_qc(10, source_breakdown={"mmseqs2": 7, "mock": 3})
    assert qc["real_fraction"] == 0.7            # mock excluded from "real"
    assert qc["sources"]["mmseqs2"] == 7


def test_depth_only():
    assert msa_qc(50) == {"depth": 50}
