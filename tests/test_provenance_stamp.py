"""Consistent PROVENANCE STAMP across every report (src/evoliez/io/_provenance).

Date alone is insufficient — same-day re-runs collide. The stamp adds run name +
config fingerprint + git + per-ligand RDKit NET FORMAL CHARGE (the safety
field: a mis-protonated ligand surfaces on every report). These tests pin the
helper API, the rendered line, RUN_INFO.json, and that the stamp lands in the
real report HTML — all GENERIC (net charge computed from whatever SMILES given).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.io._provenance import (
    provenance_fields,
    provenance_html,
    stamp,
    tag_json,
    write_run_info,
)

pytest.importorskip("rdkit")  # net-charge safety feature needs RDKit

ROOT = Path(__file__).resolve().parents[1]


def _ctx(tmp_path, **over):
    """A real RunContext on a charged ligand. Primary = acetate (net -1), one
    co-modelled extra = a quaternary-ammonium cation (net +1) — exercises both
    signs, nothing target-specific is hardcoded."""
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.name": "stamp_demo",
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "input.ligand": {"id": "acetate", "type": "smiles",
                             "value": "CC(=O)[O-]"},            # net -1
            "input.extra_ligands": [
                {"id": "cation", "type": "smiles",
                 "value": "C[N+](C)(C)C"}],                     # net +1
            "validation": {"redocking": {"methods": ["gnina", "diffdock"]}},
            **over,
        },
    )
    return RunContext(cfg, allow_small_disk=True).setup()


def test_fields_have_fingerprint_charge_and_backends(tmp_path):
    fields = provenance_fields(_ctx(tmp_path))
    assert fields["run_name"] == "stamp_demo"
    # config fingerprint: first 12 hex chars, matches the resume checkpoint key.
    assert len(fields["config_fingerprint"]) == 12
    assert "T" in fields["generated_at"]              # ISO seconds, not bare date
    # per-ligand NET FORMAL CHARGE computed from the SMILES (the safety feature).
    by_id = {l["id"]: l["net_formal_charge"] for l in fields["ligands"]}
    assert by_id["acetate"] == -1
    assert by_id["cation"] == +1
    # docking backends surfaced from the redocking config.
    assert "gnina" in fields["docking_backends"]


def test_html_contains_fingerprint_datetime_and_signed_charge(tmp_path):
    ctx = _ctx(tmp_path)
    fields = provenance_fields(ctx)
    html = provenance_html(fields)
    assert fields["config_fingerprint"] in html            # fingerprint rendered
    assert fields["generated_at"] in html                  # datetime rendered
    assert "run stamp_demo" in html
    assert "acetate (net -1)" in html                      # correct signed charge
    assert "cation (net +1)" in html
    assert "docking [gnina, diffdock]" in html


def test_fingerprint_matches_run_checkpoint(tmp_path):
    """The stamp's cfg fingerprint must equal what the resume checkpoint keys
    on (RunContext.run_fingerprint config_sha1, first 12), so a report's stamp
    can be cross-referenced to _state.json for that run."""
    ctx = _ctx(tmp_path)
    fields = provenance_fields(ctx)
    assert fields["config_fingerprint"] == ctx.run_fingerprint()["config_sha1"][:12]


def test_stamp_writes_run_info_json(tmp_path):
    ctx = _ctx(tmp_path)
    line = stamp(ctx)                                      # writes RUN_INFO.json
    import json
    info = json.loads((ctx.paths.root / "RUN_INFO.json").read_text())
    assert info["run_name"] == "stamp_demo"
    assert info["config_fingerprint"] in line
    assert {l["id"]: l["net_formal_charge"] for l in info["ligands"]}["acetate"] == -1


def test_wrong_protonation_surfaces_nonzero_net_charge(tmp_path):
    """Headline safety case: a ligand entered with the wrong protonation (here a
    neutral-acid SMILES carrying an extra proton -> net +1) shows a non-zero net
    charge on the stamp instead of slipping through silently."""
    ctx = _ctx(tmp_path, **{
        "input.ligand": {"id": "wrongprot", "type": "smiles",
                         "value": "CC(=O)[OH2+]"}})         # protonated -> +1
    html = provenance_html(provenance_fields(ctx))
    assert "wrongprot (net +1)" in html


def test_report_html_embeds_stamp(tmp_path):
    """End-to-end: the docking report HTML carries the stamp in its subtitle
    (no leftover %% tokens) — confirms the builder wiring, not just the helper."""
    from evoliez.io.docking_report import build_docking_report_html
    ctx = _ctx(tmp_path)
    prov = provenance_html(provenance_fields(ctx))
    stats = {
        "scores": {"gnina": {"score": -7.0}}, "gnina_props": {},
        "rmsd_pairs": [], "ligand_qc": {}, "landscape": [],
        "n_diffdock_poses": 0, "receptor": "", "poses": {},
        "reference_pdb": "", "ligand_smiles": "CC(=O)[O-]",
    }
    html = build_docking_report_html(
        target_id="myprot", stats=stats, conditions=[("methods", "gnina")],
        generated="2026-06-21 00:09", ligand_name="acetate", provenance=prov)
    assert "%%" not in html
    assert "acetate (net -1)" in html
    assert ctx.run_fingerprint()["config_sha1"][:12] in html


def test_standalone_shim_renders_without_full_context(tmp_path):
    """The standalone gen_s0*_report.py scripts reconstruct only cfg + run dir
    (a SimpleNamespace shim, no live RunContext / DB). The stamp must still
    render best-effort and RUN_INFO.json must still be written."""
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.name": "shim_demo",
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "input.ligand": {"id": "acetate", "type": "smiles",
                             "value": "CC(=O)[O-]"},
        },
    )
    rd = tmp_path / "rundir"
    rd.mkdir()
    shim = SimpleNamespace(config=cfg, root=rd)            # what the scripts build
    fields = provenance_fields(shim)
    write_run_info(rd, fields)
    html = provenance_html(fields)
    assert "run shim_demo" in html
    assert "acetate (net -1)" in html
    assert len(fields["config_fingerprint"]) == 12        # recomputed from cfg
    assert (rd / "RUN_INFO.json").exists()


def test_tag_json_adds_keys_non_breaking(tmp_path):
    """tag_json adds generated_at + run_fingerprint to a data JSON in place,
    preserving existing keys (non-breaking additive enrichment)."""
    import json
    ctx = _ctx(tmp_path)
    p = ctx.paths.interaction_graphs / "interaction_model.json"
    p.write_text(json.dumps({"kind": "logreg", "fp_dim": 42}))
    tag_json(p, ctx)
    data = json.loads(p.read_text())
    assert data["kind"] == "logreg" and data["fp_dim"] == 42     # preserved
    assert data["run_fingerprint"] == ctx.run_fingerprint()["config_sha1"][:12]
    assert "T" in data["generated_at"]


def test_html_omits_empty_fields():
    """Generic robustness: with no run_name / ligands / backends, the stamp is
    just the populated segments — never an empty leading/trailing ' · '."""
    line = provenance_html({"generated_at": "2026-06-21T00:09:11",
                            "config_fingerprint": "abc123def456"})
    assert line == "generated 2026-06-21T00:09:11 · cfg abc123def456"
