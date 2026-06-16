"""The auto-generated s02 HTML homolog-analysis report.

Self-contained (Chart.js via CDN); every axis/bin/table/colour is derived from
the data so it generalises to any target protein.
"""

import json
import re

from evoliez.adapters.msa_tools import Homolog
from evoliez.io.homolog_report import build_homolog_report_html, write_homolog_report


def _homologs(annon_ids):
    out = []
    for i, (ann, ident, cl) in enumerate(annon_ids):
        out.append(Homolog(id=f"h{i}", sequence="ACDE", identity=ident,
                           coverage=1.0, annotation=ann, cluster_id=cl,
                           source="structure" if "struct" in ann or ann == "foldseek"
                           else "sequence"))
    return out


def _data_blob(html):
    m = re.search(r"var R=(\{.*?\});", html)
    return json.loads(m.group(1))


def test_report_structure_and_data(tmp_path):
    homs = _homologs([("colabfold", 0.22, 0), ("colabfold", 0.45, 0),
                      ("colabfold", 0.80, 1), ("foldseek", 0.18, 2),
                      ("foldseek", 0.25, 2), ("foldseek", 0.30, 3)])
    html = build_homolog_report_html(
        target_id="myprot", target_len=384, homologs=homs, msa_depth=512,
        conditions=[("backend", "real"), ("cluster_identity", "0.40")],
        generated="2026-06-16 12:00",
    )
    assert html.startswith("<!DOCTYPE html")
    for needle in ("idChart", "clChart", "Analysis conditions", "myprot",
                   "384 aa", "Chart.js", "cluster_identity", "colabfold", "foldseek"):
        assert needle in html, needle
    R = _data_blob(html)
    assert set(R["groups"]) == {"colabfold", "foldseek"}
    assert R["colors"]["colabfold"] and R["colors"]["foldseek"]
    # per-source identity histograms sum to the per-source counts
    assert sum(R["id_hist"]["colabfold"]) == 3
    assert sum(R["id_hist"]["foldseek"]) == 3
    assert sum(R["cluster_counts"]) == 4         # 4 distinct cluster_ids


def test_bins_auto_scale_to_data_range():
    # a NARROW identity range must yield finer bins than a wide one — axes adapt.
    wide = build_homolog_report_html(
        target_id="t", target_len=10,
        homologs=_homologs([("seq", 0.10, 0), ("seq", 0.95, 1)]),
        msa_depth=2, conditions=[], generated="x")
    narrow = build_homolog_report_html(
        target_id="t", target_len=10,
        homologs=_homologs([("seq", 0.40, 0), ("seq", 0.46, 1)]),
        msa_depth=2, conditions=[], generated="x")
    ws = _data_blob(wide)["id_labels"]
    ns = _data_blob(narrow)["id_labels"]
    assert ws and ns
    assert float(ws[0]) <= 0.10 and float(ws[-1]) >= 0.9   # spans the wide range
    assert float(ns[0]) <= 0.40 and float(ns[-1]) < 0.6    # zoomed to the narrow range


def test_write_creates_file(tmp_path):
    out = tmp_path / "reports" / "homolog_report.html"
    write_homolog_report(
        out, target_id="t", target_len=5,
        homologs=_homologs([("colabfold", 0.3, 0)]),
        msa_depth=None, conditions=[("backend", "mock")], generated="x")
    assert out.exists() and out.read_text().startswith("<!DOCTYPE html")
