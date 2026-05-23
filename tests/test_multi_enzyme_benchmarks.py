"""Contract tests for the 5-enzyme benchmark suite.

Expanded from the PseFDH-only scaffold to cover the four additional
enzymes recommended in the expert validation plan:

    examples/pseudomonas_fdh/           — FDH NAD→NADP cofactor switch (PseFDH)
    examples/xylose_reductase/          — AKR superfamily NADP→NAD switch (XR)
    examples/tem1_betalactamase/        — TEM-1 ESBL / catalytic dyad (no cofactor)
    examples/beta_glucosidase_bgl3/     — GH1 family DMS-derived labels
    examples/p450_bm3/                  — heme + I-helix + substrate gating (hard case)

Pins the cross-enzyme contract: every card loads via the shared
:func:`evoliez.ml.benchmark.load_benchmark` helper, every row uses the
``beneficial / neutral / deleterious`` label vocabulary, every card has
at least three beneficial and three deleterious rows (otherwise the
recall@K / catalytic-avoidance metrics have no signal), and the CSV is
in lockstep with its sibling ``README.md`` (each beneficial /
deleterious mutation is mentioned in the README so a reviewer can
chase the citation without leaving the example directory).

These tests do NOT validate WT-letter consistency against
``target.fasta`` because the FASTA files are fetched on the server via
``scripts/fetch_target_fasta.sh`` and not checked into the repo. The
PseFDH-specific WT-letter test lives in ``test_psefdh_benchmark.py``
and will gain siblings once the per-enzyme FASTA fetch scripts are
exercised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

from evoliez.ml.benchmark import load_benchmark

ROOT = Path(__file__).resolve().parents[1]
_VALID_LABELS = {"beneficial", "deleterious", "neutral"}

# (slug, display name, UniProt accession used by the README,
#  expected substrate hint shown for documentation).
ENZYMES: List[Dict[str, str]] = [
    {"slug": "pseudomonas_fdh",        "name": "PseFDH",        "uniprot": "P33160"},
    {"slug": "xylose_reductase",       "name": "XR (C. tenuis)", "uniprot": "O74237"},
    {"slug": "tem1_betalactamase",     "name": "TEM-1",         "uniprot": "P62593"},
    {"slug": "beta_glucosidase_bgl3",  "name": "Bgl3 (GH1)",    "uniprot": "P22073"},
    {"slug": "p450_bm3",               "name": "P450 BM3",      "uniprot": "P14779"},
]


def _card_dir(slug: str) -> Path:
    return ROOT / "examples" / slug


@pytest.mark.parametrize("entry", ENZYMES, ids=[e["slug"] for e in ENZYMES])
def test_card_has_benchmark_and_readme(entry):
    d = _card_dir(entry["slug"])
    assert d.is_dir(), f"missing {d}"
    assert (d / "benchmark.csv").exists(), f"{entry['slug']}: no benchmark.csv"
    assert (d / "README.md").exists(),    f"{entry['slug']}: no README.md"


@pytest.mark.parametrize("entry", ENZYMES, ids=[e["slug"] for e in ENZYMES])
def test_card_loads_via_load_benchmark(entry):
    csv_path = _card_dir(entry["slug"]) / "benchmark.csv"
    rows = load_benchmark(csv_path)
    assert rows, f"{entry['slug']}: empty after load_benchmark"
    for r in rows:
        assert r["mutation"], f"{entry['slug']}: empty mutation row"
        assert r["label"] in _VALID_LABELS, (
            f"{entry['slug']}: row {r['mutation']} has unsupported label "
            f"{r['label']!r}; must be one of {sorted(_VALID_LABELS)}"
        )
        # activity column (load_benchmark coerces to float or None).
        assert r["activity"] is None or isinstance(r["activity"], float)


@pytest.mark.parametrize("entry", ENZYMES, ids=[e["slug"] for e in ENZYMES])
def test_card_has_min_signal_for_figure3(entry):
    """Figure 3 (recall@K + per-class percentile) needs at least 3
    beneficial + 3 deleterious rows so the panels don't read as noise.
    """
    rows = load_benchmark(_card_dir(entry["slug"]) / "benchmark.csv")
    n_ben = sum(1 for r in rows if r["label"] == "beneficial")
    n_del = sum(1 for r in rows if r["label"] == "deleterious")
    assert n_ben >= 3, f"{entry['slug']}: need >=3 beneficial (got {n_ben})"
    assert n_del >= 3, f"{entry['slug']}: need >=3 deleterious (got {n_del})"


@pytest.mark.parametrize("entry", ENZYMES, ids=[e["slug"] for e in ENZYMES])
def test_card_readme_mentions_every_non_neutral_mutation(entry):
    """Sanity contract: every beneficial / deleterious row in the CSV
    must appear in the README so the citation chain is auditable
    without leaving the example directory. Neutral rows are exempt
    because some come from DMS bulk-fitness tables that get a single
    aggregate citation in the README.
    """
    d = _card_dir(entry["slug"])
    readme = (d / "README.md").read_text()
    rows = load_benchmark(d / "benchmark.csv")
    missing = [
        r["mutation"] for r in rows
        if r["label"] in ("beneficial", "deleterious")
        and r["mutation"] not in readme
    ]
    assert not missing, (
        f"{entry['slug']}: README doesn't mention non-neutral mutations: "
        f"{missing}"
    )


@pytest.mark.parametrize("entry", ENZYMES, ids=[e["slug"] for e in ENZYMES])
def test_card_readme_cites_uniprot(entry):
    """The README must name the UniProt accession so the
    ``scripts/fetch_target_fasta.sh`` command lands on the right
    sequence (the multi-enzyme harness depends on this)."""
    readme = (_card_dir(entry["slug"]) / "README.md").read_text()
    assert entry["uniprot"] in readme, (
        f"{entry['slug']}: README must cite UniProt {entry['uniprot']}"
    )
