"""Funnel-provenance reporting: the s07 generated-candidate table + the s11
attrition / per-generator / per-candidate evidence report.

Runs the mock pipeline end-to-end (to s11) and asserts the durable provenance
artifacts exist with the expected columns and that the attrition funnel is
internally consistent (generated >= kept >= md). Generic — nothing FDH-specific
is asserted, only structure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.io.provenance_report import build_funnel_report, write_funnel_report
from evoliez.pipeline import Pipeline
from evoliez.types import Candidate, Mutation
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path: Path):
    return load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "mutation_generation": {
                "methods": ["chemistry_rules", "msa_sampler", "ligandmpnn"],
                "max_candidates": 80,
                "ligandmpnn_samples": 8,
            },
            "reranking": {"top_for_redocking": 30, "top_for_md": 5,
                          "model": "xgboost", "use_experimental_labels": False},
            "validation": {"md": {"enabled": True, "protocol_level": 1,
                                   "top_candidates": 5}},
            "gnn": {"build_dataset": False},
        },
    )


def _csv_header(p: Path):
    with p.open() as fh:
        return next(csv.reader(fh))


def _csv_rows(p: Path):
    with p.open() as fh:
        return list(csv.DictReader(fh))


def test_generated_and_funnel_provenance_files(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)

    prov = ctx.paths.reports / "provenance"
    assert prov.is_dir(), "provenance dir not created"

    # --- (1) s07 GENERATED-candidate table ------------------------------- #
    gen_json = prov / "generated_candidates.json"
    gen_csv = prov / "generated_candidates.csv"
    assert gen_json.exists() and gen_csv.exists()
    gen_rows = json.loads(gen_json.read_text())
    assert gen_rows, "generated provenance is empty"
    # the FULL generated pool is captured, not just survivors
    assert len(gen_rows) == ctx.meta("n_candidates_generated")
    gcols = _csv_header(gen_csv)
    for col in ("candidate_id", "mutation_string", "generator", "n_mutations",
                "msa_freq", "gap_freq", "ligandmpnn_logp", "multipoint_order",
                "ligand_roles"):
        assert col in gcols, f"generated table missing column {col}"
    # at least one generator actually populated a feature column (not all blank)
    assert any(r.get("msa_freq") not in (None, "") for r in gen_rows) or \
        any(r.get("ligandmpnn_logp") not in (None, "") for r in gen_rows)

    # --- (2) s11 ATTRITION funnel ---------------------------------------- #
    attr_csv = prov / "attrition_funnel.csv"
    assert attr_csv.exists()
    assert set(_csv_header(attr_csv)) >= {
        "stage", "label", "count", "fraction_of_generated"}
    attr = {r["stage"]: r["count"] for r in _csv_rows(attr_csv)}
    assert set(attr) >= {"generated", "reranked", "boltz_reevaluated",
                         "validated", "selected_for_md", "ranked"}

    def _n(stage):
        return int(attr[stage]) if attr.get(stage) not in (None, "") else None

    n_gen = _n("generated")
    n_rerank = _n("reranked")
    n_val = _n("validated")
    n_md_sel = _n("selected_for_md")
    n_ranked = _n("ranked")
    assert n_gen and n_gen >= 1
    # monotone narrowing down the funnel: generated >= reranked >= ... >= md
    assert n_gen >= n_rerank >= n_val >= n_md_sel
    assert n_gen >= n_ranked >= n_md_sel
    # consistency with the live meta counts
    assert n_gen == ctx.meta("n_candidates_generated")
    assert n_md_sel == ctx.meta("n_for_md")

    # --- (3) per-generator survival -------------------------------------- #
    gs_csv = prov / "generator_survival.csv"
    assert gs_csv.exists()
    gs_cols = _csv_header(gs_csv)
    for col in ("generator", "n_generated", "n_in_ranked", "n_in_shortlist",
                "survival_fraction"):
        assert col in gs_cols
    gs_rows = _csv_rows(gs_csv)
    assert gs_rows
    # #in ranked never exceeds #generated for any generator
    for r in gs_rows:
        if r["n_generated"] not in (None, ""):
            assert int(r["n_in_ranked"]) <= int(r["n_generated"])
        assert int(r["n_in_shortlist"]) <= int(r["n_in_ranked"])
    # the per-generator generated counts reconstruct the total generated pool
    gen_total = sum(int(r["n_generated"]) for r in gs_rows
                    if r["n_generated"] not in (None, ""))
    if gen_total:
        assert gen_total == n_gen

    # --- (4) per-candidate evidence -------------------------------------- #
    cand_csv = prov / "candidate_provenance.csv"
    assert cand_csv.exists()
    ccols = _csv_header(cand_csv)
    for col in ("rank", "candidate_id", "mutations", "generator",
                "evidence_class", "survived_to", "boltz_delta_source", "md_ran"):
        assert col in ccols
    crows = _csv_rows(cand_csv)
    assert len(crows) == n_ranked
    # one evidence-class per ranked candidate, all from the known vocabulary
    allowed = {"real_boltz", "real_docking", "md_ran",
               "stability_unavailable", "proxy"}
    assert all(r["evidence_class"] in allowed for r in crows)
    # ranks are a 1..n permutation
    assert sorted(int(r["rank"]) for r in crows) == list(range(1, n_ranked + 1))
    # md_ran candidates count matches the candidates handed to MD
    md_ids = {c.candidate_id for c in ctx.get("md_candidates", [])}
    csv_md = {r["candidate_id"] for r in crows if r["md_ran"] in ("True", "true")}
    assert csv_md == (md_ids & {r["candidate_id"] for r in crows})

    # HTML view emitted and non-trivial
    html = prov / "funnel_provenance.html"
    assert html.exists() and "Attrition funnel" in html.read_text()

    # persisted summary survives in meta
    assert isinstance(ctx.meta("funnel_evidence_class_counts"), dict)


def test_build_funnel_report_evidence_classes_unit():
    """Unit-level: the evidence-class precedence + attrition wiring, with a
    hand-built candidate set and a stubbed meta() — no pipeline run."""
    def mk(cid, gen, **details):
        return Candidate(candidate_id=cid, mutations=[Mutation("A", 10, "K")],
                         generator=gen,
                         scores={"final_score": 1.0, "ml_score": 0.5},
                         details=details)

    ranked = [
        mk("c0", "ligandmpnn", boltz_delta_source="real", rank=1),
        mk("c1", "msa_sampler", redock={"vina": {"score": -7.0}}, rank=2),
        mk("c2", "chemistry_rules", stability_unavailable=True, rank=3),
        mk("c3", "multipoint", boltz_delta_source="proxy", rank=4),  # -> md via id
        mk("c4", "msa_sampler", boltz_delta_source="proxy", rank=5),  # -> proxy
    ]
    meta_store = {
        "n_candidates_generated": 50, "n_after_rerank": 20,
        "n_mutant_boltz_evaluated": 10, "n_after_nonmd": 8,
        "n_for_md": 3, "n_md_real_ran": 2, "n_md_passed": 1,
    }

    def meta(k, default=None):
        return meta_store.get(k, default)

    rep = build_funnel_report(ranked=ranked, meta=meta,
                              md_candidate_ids=["c3"], top_n=2)
    by_id = {r["candidate_id"]: r for r in rep["per_candidate"]}
    assert by_id["c0"]["evidence_class"] == "real_boltz"
    assert by_id["c1"]["evidence_class"] == "real_docking"
    assert by_id["c2"]["evidence_class"] == "stability_unavailable"
    assert by_id["c3"]["evidence_class"] == "md_ran"      # in md ids
    assert by_id["c4"]["evidence_class"] == "proxy"

    attr = {r["stage"]: r["count"] for r in rep["attrition"]}
    assert attr["generated"] == 50 and attr["ranked"] == 5
    assert attr["selected_for_md"] == 3
    # shortlist (top_n=2) restricts the per-generator shortlist count
    short = sum(r["n_in_shortlist"] for r in rep["per_generator"])
    assert short == 2
    assert rep["summary"]["evidence_class_counts"]["real_boltz"] == 1


def test_funnel_report_tolerates_missing_stages(tmp_path):
    """A partial run (only s07 meta present) must still produce a report with
    blanks for unrun stages rather than crashing or inventing zeros."""
    def mk(i):
        return Candidate(candidate_id=f"c{i}",
                         mutations=[Mutation("A", 10 + i, "K")],
                         generator="chemistry_rules",
                         scores={"final_score": 1.0 - 0.01 * i}, details={})

    ranked = [mk(i) for i in range(3)]
    meta_store = {"n_candidates_generated": 9}

    rep = build_funnel_report(ranked=ranked,
                              meta=lambda k, d=None: meta_store.get(k, d))
    attr = {r["stage"]: r["count"] for r in rep["attrition"]}
    assert attr["generated"] == 9 and attr["ranked"] == 3
    assert attr["reranked"] is None and attr["validated"] is None

    written = write_funnel_report(tmp_path, rep)
    names = {p.name for p in written}
    assert {"funnel_provenance.json", "attrition_funnel.csv",
            "generator_survival.csv", "candidate_provenance.csv",
            "funnel_provenance.html"} <= names
    # unrun stages render as empty cells, not 0
    attr_csv = tmp_path / "attrition_funnel.csv"
    rows = {r["stage"]: r["count"] for r in _csv_rows(attr_csv)}
    assert rows["reranked"] == "" and rows["generated"] == "9"
