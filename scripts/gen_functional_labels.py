#!/usr/bin/env python3
"""Generate FUNCTIONAL-STATE labels (reference_like / alternative / displaced) for the
MD-free triage ML.

For EVERY s07 candidate of a FINISHED run — not just the handful that survive the funnel
to s10 — anchored-build the mutant on the reference complex, relax it with a CHEAP md_lite
protocol (NAC / RBFE / binding-ΔG OFF), then gate the relaxed cofactor/substrate pose vs
the WT reference. The gate verdict IS the expensive label the cheap-feature ML must learn
to predict.  This is the data the learnability probe showed we lack: the s10 survivors are
all reference_like (no negatives), so labels must be generated on the displaced candidates
that never reached s10.

SAFETY (this run touches a FINISHED production run):
  * read-only resume with the run's OWN config -> config_sha1 unchanged -> no purge.
    If setup reports the config no longer matches (ctx.invalidated), ABORT immediately;
    EVOLIEZ_ALLOW_PURGE is left unset so nothing is ever deleted.
  * the cheap MD overrides live only in memory; the persisted config / _state.json are
    never rewritten.
  * all MD scratch is written under <run-dir>/labels_md/ and the labels go to a NEW file
    reports/provenance/functional_labels.json -- existing fdh_v2 stage artifacts are
    never overwritten.
  * idempotent: candidate_ids already in the output file are skipped, and results flush
    after every chunk, so a kill/resume loses at most one chunk.

Usage (server):
  python scripts/gen_functional_labels.py \
      --config configs/target.local.v2.yaml \
      --run-dir runs/fdh_v2 --gpus 0,2,3 --limit 0 --production-ns 0.2
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def parse_mutations(mutation_string: str) -> List[Tuple[str, int, str]]:
    """'F84L', 'A12G;C30D', 'A12G+C30D' -> [(wt, pos, mut), ...].  Tuples are accepted by
    md.anchored_build._norm_mut, so no Mutation object is needed."""
    out: List[Tuple[str, int, str]] = []
    for tok in re.split(r"[;,+/ ]+", (mutation_string or "").strip()):
        m = _MUT_RE.match(tok)
        if m:
            out.append((m.group(1).upper(), int(m.group(2)), m.group(3).upper()))
    return out


def load_candidates(prov_dir: Path, limit: int, stratify: bool = True,
                    only_ids: "Optional[set]" = None) -> List[dict]:
    recs = json.load(open(prov_dir / "generated_candidates.json"))
    recs = recs if isinstance(recs, list) else recs.get("candidates", [])
    cands = []
    for r in recs:
        cid = r.get("candidate_id")
        muts = parse_mutations(r.get("mutation_string", ""))
        if cid and muts:               # skip WT / unparseable
            cands.append({"candidate_id": cid,
                          "mutation_string": r.get("mutation_string", ""),
                          "mutations": muts})
    if only_ids:                       # explicit curated set overrides limit/stratify
        return [c for c in cands if c["candidate_id"] in only_ids]
    if limit and limit > 0 and limit < len(cands):
        if stratify:
            # round-robin across the FIRST-mutation position so a capped run covers
            # all designable positions (first-N is biased to the earliest positions).
            from collections import OrderedDict
            buckets: "OrderedDict[int, list]" = OrderedDict()
            for c in cands:
                buckets.setdefault(c["mutations"][0][1], []).append(c)
            picked = []
            while len(picked) < limit and any(buckets.values()):
                for q in buckets.values():
                    if q:
                        picked.append(q.pop(0))
                        if len(picked) >= limit:
                            break
            cands = picked
        else:
            cands = cands[:limit]
    return cands


def cheap_mdcfg(mdcfg, production_ns: float, minimize_steps: int):
    """In-memory copy of the run's MDConfig with the EXPENSIVE tiers off and a short
    relaxation. Never persisted."""
    c = copy.deepcopy(mdcfg)
    for sub, flag in (("reactive_geometry", "enabled"), ("rbfe", "enabled"),
                      ("binding_dg", "enabled")):
        s = getattr(c, sub, None)
        if s is not None and hasattr(s, flag):
            setattr(s, flag, False)
    c.production_ns = production_ns
    c.restrained_md_ps = min(getattr(c, "restrained_md_ps", 500.0), 200.0)
    c.equilibration_ps = min(getattr(c, "equilibration_ps", 100.0), 50.0)
    c.minimize_steps = minimize_steps
    c.replicas = 1
    return c


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the run's ORIGINAL config (must still "
                    "match config_sha1, else this aborts without touching the run)")
    ap.add_argument("--run-dir", required=True, help="finished run dir, e.g. runs/fdh_v2")
    ap.add_argument("--gpus", default="", help="comma GPU ids for fan-out, e.g. 0,2,3 "
                    "(empty -> serial)")
    ap.add_argument("--limit", type=int, default=0, help="cap #candidates (0 = all). Use a "
                    "small value, e.g. 3, for a smoke run first.")
    ap.add_argument("--production-ns", type=float, default=0.2,
                    help="cheap md_lite production length (default 0.2 ns)")
    ap.add_argument("--minimize-steps", type=int, default=5000)
    ap.add_argument("--resume-to", default="s04_complex",
                    help="resume-load only up to here (s04 gives wt_complex; s01 gives "
                    "catalytic_positions/extra_ligands) — avoids re-running s05+ heavy tools")
    ap.add_argument("--chunk", type=int, default=0, help="candidates per MD batch "
                    "(0 -> max(4, 4*len(gpus)))")
    ap.add_argument("--no-stratify", action="store_true",
                    help="with --limit, take first-N instead of round-robin across positions")
    ap.add_argument("--candidate-ids", default="",
                    help="comma list or @file of candidate_ids to run EXACTLY (overrides "
                    "--limit/--stratify); for curated diagnostics")
    ap.add_argument("--out", default="", help="output json (default <run-dir>/reports/"
                    "provenance/functional_labels.json)")
    args = ap.parse_args(argv)

    from evoliez.config import Backend, load_config
    from evoliez.context import RunContext
    from evoliez.pipeline import Pipeline
    from evoliez.md.anchored_build import build_anchored_mutant_complex
    from evoliez.md.pose_gate import gate_from_pdb
    from evoliez.md.analysis import analyse
    from evoliez.adapters.openmm_engine import run_md
    from evoliez.adapters.md_batch import run_md_batches

    run_dir = Path(args.run_dir).resolve()
    prov = run_dir / "reports" / "provenance"
    out_path = Path(args.out) if args.out else prov / "functional_labels.json"
    labels_md = run_dir / "labels_md"
    labels_md.mkdir(parents=True, exist_ok=True)

    # ---- resume the finished run read-only with its OWN config (no purge) -------------
    cfg = load_config(args.config)
    ctx = RunContext(cfg)
    ctx.setup()
    if getattr(ctx, "invalidated", False):
        print("ABORT: config no longer matches this run's fingerprint (resume would "
              "invalidate it). Patch the config back or use the exact original.",
              file=sys.stderr)
        return 2
    backend = ctx.config.backend_for("s10_md")
    if backend is not Backend.real:
        print(f"ABORT: backend is {backend}, expected real MD.", file=sys.stderr)
        return 2
    Pipeline().run(ctx, resume=True, to_stage=args.resume_to)
    wt = ctx.require("wt_complex")
    catalytic = ctx.require("catalytic_positions")

    # extra ligands (e.g. formate) + ligand charge policy — mirror s10_md exactly
    _lig = ctx.config.input.ligand
    _cm = _lig.charges_mol2
    if _cm and not Path(_cm).is_absolute():
        _cm = str((Path.cwd() / _cm).resolve())

    def _apply_charge(cx):
        if cx is None or getattr(cx, "ligand", None) is None:
            return
        if cx.ligand.id == _lig.id:
            cx.ligand.charges_mol2 = _cm
            cx.ligand.allow_am1bcc = _lig.allow_am1bcc
            if _lig.net_charge is not None:
                cx.ligand.formal_charge = _lig.net_charge

    _apply_charge(wt)
    _extra = (ctx.config.input.extra_ligands or ctx.get("extra_ligands", []) or [])

    def _smiles(e):
        if getattr(e, "type", "smiles") not in ("smiles", None):
            return None
        return getattr(e, "value", None) or getattr(e, "smiles", None)
    extra_specs = [(getattr(e, "id", f"extra{i}"), _smiles(e)) for i, e in enumerate(_extra)]
    extra_specs = [(i, s) for i, s in extra_specs if s]
    lig_cache = labels_md / "_ligand_ff_cache"
    mdcfg = cheap_mdcfg(ctx.config.validation.md, args.production_ns, args.minimize_steps)
    _design_role = getattr(ctx.config.input.ligand, "role", None) or "cofactor"

    gpus = [g for g in args.gpus.split(",") if g.strip() != ""]
    chunk = args.chunk or max(4, 4 * max(1, len(gpus)))

    # ---- WT reference under the SAME cheap protocol (fair gate baseline) --------------
    wt_ref_pdb = getattr(wt.structure, "pdb_path", None)
    wt_wd = labels_md / "_wt_reference_cheap"
    if not (wt_wd / "_wt_reference_cheap_minimized.pdb").exists():
        print("[wt] running WT reference under cheap protocol ...", flush=True)
        wt_res = run_md(wt, "_wt_reference_cheap", mdcfg, wt_wd, instability=0.0,
                        catalytic_positions=catalytic, backend=backend,
                        ligand_cache_dir=lig_cache, extra_ligands=extra_specs,
                        fail_loud_on_cpu=True)
        if wt_res.minimized_pdb:
            wt_ref_pdb = wt_res.minimized_pdb
    else:
        wt_ref_pdb = str(wt_wd / "_wt_reference_cheap_minimized.pdb")
    if not wt_ref_pdb or not Path(wt_ref_pdb).exists():
        print("ABORT: no WT reference PDB for the pose gate.", file=sys.stderr)
        return 2

    # ---- candidates + idempotent resume ----------------------------------------------
    only_ids = None
    if args.candidate_ids:
        raw = args.candidate_ids
        if raw.startswith("@"):
            raw = Path(raw[1:]).read_text()
        only_ids = {x.strip() for x in re.split(r"[,\s]+", raw) if x.strip()}
    cands = load_candidates(prov, args.limit, stratify=not args.no_stratify,
                            only_ids=only_ids)
    done = {}
    if out_path.exists():
        done = {r["candidate_id"]: r for r in json.load(open(out_path))}
    todo = [c for c in cands if c["candidate_id"] not in done]
    print(f"[plan] {len(cands)} candidates, {len(done)} already labeled, {len(todo)} to do; "
          f"gpus={gpus or 'serial'} chunk={chunk} prod_ns={args.production_ns}", flush=True)

    def flush():
        out_path.write_text(json.dumps(list(done.values()), indent=2))

    def gate_record(cand, result) -> dict:
        rec = {"candidate_id": cand["candidate_id"],
               "mutation_string": cand["mutation_string"],
               "md_status": str(result.status),
               "md_failed": bool(getattr(result, "integration_failed", False)
                                 or str(result.status) in ("failed",)),
               "failure_reason": getattr(result, "failure_reason", None),
               "md_lite_score": None, "pose_gate": None}
        try:
            metrics = analyse(result, ctx.config.scoring)
            rec["md_lite_score"] = metrics.md_lite_score
        except Exception as e:        # noqa: BLE001
            rec["analyse_error"] = str(e)[:120]
        mpdb = result.minimized_pdb
        if mpdb and Path(mpdb).exists():
            try:
                pg = gate_from_pdb(wt_ref_pdb, mpdb, ligand_rank=0, role=_design_role,
                                   ligand_id="design_ligand")
                gj = {"design_ligand": pg.to_json()}
                co = gate_from_pdb(wt_ref_pdb, mpdb, ligand_rank=1, role="substrate",
                                   ligand_id="cosubstrate")
                if co.status != "skipped_no_correspondence":
                    gj["cosubstrate"] = co.to_json()
                rec["pose_gate"] = gj
            except Exception as e:    # noqa: BLE001
                rec["gate_error"] = str(e)[:120]
        return rec

    # ---- build (CPU) + run_md (GPU fan-out) + gate, per chunk -------------------------
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        tasks = []
        for c in batch:
            wd = labels_md / c["candidate_id"]
            wd.mkdir(parents=True, exist_ok=True)
            apdb = wd / f"{c['candidate_id']}_anchored.pdb"
            try:
                mc, ab = build_anchored_mutant_complex(wt, c["mutations"], apdb)
                if not ab.applied:
                    done[c["candidate_id"]] = {**c, "skipped": "no mutation applied",
                                               "anchored_skipped": ab.skipped}
                    continue
                _apply_charge(mc)
                tasks.append((c["candidate_id"], mc, str(wd),
                              0.3))  # instability prior
            except Exception as e:    # noqa: BLE001
                done[c["candidate_id"]] = {**c, "build_error": str(e)[:160]}
        if not tasks:
            flush()
            continue
        if len(gpus) > 1:
            results = run_md_batches(tasks, gpus, mdcfg, ligand_cache_dir=str(lig_cache),
                                     extra_specs=extra_specs, catalytic=catalytic,
                                     fail_loud_on_cpu=True)
        else:
            results = {}
            for cid, mc, wd, inst in tasks:
                results[cid] = run_md(mc, cid, mdcfg, Path(wd), instability=inst,
                                      catalytic_positions=catalytic, backend=backend,
                                      ligand_cache_dir=lig_cache, extra_ligands=extra_specs,
                                      fail_loud_on_cpu=True)
        cand_by_id = {c["candidate_id"]: c for c in batch}
        for cid, res in results.items():
            done[cid] = gate_record(cand_by_id[cid], res)
        flush()
        # progress + running label tally
        st = {}
        for r in done.values():
            if r.get("md_failed"):
                s = "md_failed"
            elif r.get("skipped"):
                s = "skipped"
            else:
                s = ((r.get("pose_gate") or {}).get("design_ligand") or {}).get(
                    "status", "no_gate")
            st[s] = st.get(s, 0) + 1
        print(f"[chunk {i//chunk + 1}] labeled={len(done)}/{len(cands)} verdicts={st}",
              flush=True)

    print(f"[done] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
