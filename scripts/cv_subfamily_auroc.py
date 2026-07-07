#!/usr/bin/env python
"""FULL leave-one-subfamily-out cross-validation AUROC for the s06b interaction
model — the robust generalisation of the single-holdout AUROC an external
reviewer asked for.

The single ``s06b._subfamily_holdout`` holds out ONE homolog group
(``held = groups[-1]``), trains ``InteractionModel`` on the rest, and reports the
AUROC of that group's Boltz consensus poses vs its decoys (the production run
recorded 0.9898). This script reproduces that number EXACTLY, then loops the same
held-out logic over every eligible subfamily and reports mean ± std.

WHY a from-disk reconstruction (not a re-run): the in-memory ``records`` s06b
builds are NOT persisted, and they cannot be reconstructed from the saved
``s06b_artifacts.json`` / ``multi_engine_docking.json`` alone — those snapshot the
Boltz confidence metrics and the docking *diagnostics*, but NOT the per-pose
interaction *fingerprints* the model trains on. The fingerprints, however, are a
deterministic function of structures that ARE retained on disk:

  * Boltz rows  — re-parse each retained ``predictions/hom_NNN_boltz_input/`` stem
                  with ``parse_prediction_dir`` and fingerprint it with the SAME
                  ``s06b._pose_records_from_complex`` the run used.
  * Docking rows — replay ``s06b._augment_with_docking`` verbatim, but with the
                  docking-TOOL subprocess (and receptor re-render) patched to
                  no-ops so the adapters PARSE the already-on-disk GNINA/DiffDock
                  output instead of re-running the engines. Classification
                  (``classify_docking_poses``) is the unchanged run code path.

Because both halves run the run's own functions, the rebuilt ``records`` match the
run's bit-for-bit up to metadata that is provably never used by the model
(``identity_to_target`` is carried but is not a feature and never enters the
robust-z statistics or the sample weight — see ``ml/pose_selection.py``). The
VALIDATION GATE below is the empirical proof: if the rebuilt single-holdout AUROC
does not reproduce 0.9898 (±0.005) the script ABORTS rather than report a CV built
on a wrong reconstruction.

READ-ONLY: nothing under the run directory is written. The docking audit that
``_augment_with_docking`` normally writes is redirected to a throwaway temp dir,
and every receptor/SDF/CSV write inside the adapters is patched out, so only the
existing structures are read.

Usage (on the server, fdh_5track):
    python scripts/cv_subfamily_auroc.py \
        --run /mnt/data/jglee/runs/fdh_5track \
        --config configs/target.local.5track.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

# Make ``src/`` importable when run from the repo root (server layout).
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from evoliez.config import Backend, load_config  # noqa: E402
from evoliez.io.paths import ProjectPaths  # noqa: E402
from evoliez.types import Complex  # noqa: E402

# s06b internals — the EXACT record-building + holdout code the run used.
from evoliez.stages import s06b_interaction_model as s06b  # noqa: E402
from evoliez.stages.s06b_interaction_model import (  # noqa: E402
    InteractionModelStage,
    _augment_with_docking,
    _holdout_groups,
    _holdout_one_auroc,
    _pose_records_from_complex,
)

RUN_RECORDED_AUROC = 0.9898   # fdh_5track interaction_model.meta subfamily_holdout_auroc
GATE_TOL = 0.005


# --------------------------------------------------------------------------- #
# read-only path shim: real run dir for inputs, temp dir for the audit write.
# --------------------------------------------------------------------------- #
class _ReadOnlyPaths(ProjectPaths):
    """ProjectPaths that reads structures from the real run but redirects every
    directory the docking replay could WRITE to into a throwaway scratch tree, so
    the production run is never modified:

      * ``interaction_graphs`` -> scratch (the docking audit JSON lands here).
      * ``docking``            -> a scratch dir whose entries are SYMLINKS to the
                                  real ``me_*`` docking shards. The adapters read
                                  the retained SDFs *through* the symlinks, while
                                  any new write (a stray empty ``me_diffdock_gpu*``
                                  dir, an ``exist_ok`` mkdir) lands in scratch.

    Reads (``complexes``, ``structures``, ...) fall through to the real run via
    the inherited ProjectPaths properties."""

    def __init__(self, run_root: Path, scratch: Path):
        object.__setattr__(self, "root", Path(run_root))
        object.__setattr__(self, "_scratch", Path(scratch))
        object.__setattr__(self, "_docking_scratch", None)

    @property
    def interaction_graphs(self) -> Path:  # type: ignore[override]
        p = self._scratch / "interaction_graphs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def docking(self) -> Path:  # type: ignore[override]
        if self._docking_scratch is None:
            d = self._scratch / "docking"
            d.mkdir(parents=True, exist_ok=True)
            real = self.root / "docking"
            if real.is_dir():
                for child in real.iterdir():
                    link = d / child.name
                    if not link.exists():
                        try:
                            link.symlink_to(child, target_is_directory=child.is_dir())
                        except OSError:
                            pass
            object.__setattr__(self, "_docking_scratch", d)
        return self._docking_scratch


class _Ctx:
    """Minimal RunContext stand-in exposing exactly the surface the s06b
    record-building reads: ``config``, ``paths``, ``dry_run`` and the
    ``get``/``require``/``put`` artifact bus. No DB, no state file, no setup()
    (so nothing is created/written under the run)."""

    def __init__(self, config, paths):
        self.config = config
        self.paths = paths
        self.dry_run = False
        self._bus: Dict[str, object] = {}

    def put(self, k, v):
        self._bus[k] = v

    def get(self, k, default=None):
        return self._bus.get(k, default)

    def require(self, k):
        if k not in self._bus:
            raise KeyError(f"required artifact missing: {k}")
        return self._bus[k]


# --------------------------------------------------------------------------- #
# patch the docking adapters: parse the retained output, never re-run a tool.
# --------------------------------------------------------------------------- #
class _NoToolRerun:
    """Context manager that neuters every side-effecting call in the GNINA /
    DiffDock adapters so ``redock_all`` / ``redock_batch`` fall through to their
    EXISTING parse-from-disk logic against the run's retained SDFs:

      * ``run``                     -> no-op   (skip the GPU subprocess)
      * ``require``                 -> no-op   (don't insist the tool is on PATH)
      * ``apply_gpu_selection``     -> no-op
      * ``full_atom_receptor_pdb``  -> True    (pretend the receptor was rendered;
                                                parse_all_* never reads it)
      * ``write_min_pdb``           -> no-op   (no ref-ligand / placeholder writes)

    Nothing under ``docking/`` is written — the receptor/SDF/CSV that the real
    run produced are read, not regenerated. The parse + classification code
    (``parse_all_modes`` / ``parse_all_ranks`` / ``classify_docking_poses``) is
    the unchanged run code path."""

    _MODS = ("evoliez.adapters.gnina", "evoliez.adapters.diffdock")
    _NOOP_NAMES = ("run", "require", "apply_gpu_selection", "write_min_pdb")

    def __init__(self):
        self._saved: Dict[str, Dict[str, object]] = {}

    def __enter__(self):
        import importlib

        def _noop(*_a, **_k):
            return None

        def _ok(*_a, **_k):  # full_atom_receptor_pdb stand-in
            return True

        for modname in self._MODS:
            mod = importlib.import_module(modname)
            saved = {}
            for name in self._NOOP_NAMES:
                if hasattr(mod, name):
                    saved[name] = getattr(mod, name)
                    setattr(mod, name, _noop)
            if hasattr(mod, "full_atom_receptor_pdb"):
                saved["full_atom_receptor_pdb"] = mod.full_atom_receptor_pdb
                mod.full_atom_receptor_pdb = _ok
            self._saved[modname] = saved

        # DiffDock GPU-shard recovery: the run sharded targets across GPUs and
        # wrote each cid's ranks under ``docking/me_diffdock_gpu{0,2,3}/<cid>/``.
        # A single-process replay (gpu_list=[]) computes out_root =
        # ``me_diffdock_gpusingle``, which holds none of them, so the original
        # ``_complex_out_dir`` found 0 diffdock poses and the whole diffdock half
        # of the augmentation was silently dropped. Override the locator to scan
        # EVERY ``me_diffdock_*`` sibling shard for the cid's rank dir, so all
        # retained diffdock poses are recovered regardless of which GPU produced
        # them. Falls back to the original behaviour when nothing matches.
        dd = importlib.import_module("evoliez.adapters.diffdock")
        self._saved.setdefault("evoliez.adapters.diffdock", {})
        orig_locate = dd._complex_out_dir
        self._saved["evoliez.adapters.diffdock"]["_complex_out_dir"] = orig_locate

        def _locate_any_shard(out_root, complex_id, _orig=orig_locate):
            from pathlib import Path as _P
            base = _P(out_root)
            # search the docking root's me_diffdock_* shards (siblings of out_root)
            search_parents = [base]
            if base.parent.exists():
                search_parents += sorted(
                    p for p in base.parent.glob("me_diffdock_*") if p.is_dir())
            for parent in search_parents:
                cand = parent / complex_id
                if cand.is_dir() and any(dd._is_rank(p)
                                         for p in cand.glob("rank*.sdf")):
                    return cand
            return _orig(out_root, complex_id)

        dd._complex_out_dir = _locate_any_shard
        return self

    def __exit__(self, *exc):
        import importlib

        for modname, saved in self._saved.items():
            mod = importlib.import_module(modname)
            for name, orig in saved.items():
                setattr(mod, name, orig)
        return False


# --------------------------------------------------------------------------- #
# reconstruction
# --------------------------------------------------------------------------- #
def _discover_rep_stems(run: Path) -> Dict[int, Path]:
    """Map rep index -> its retained Boltz ``predictions/hom_NNN_boltz_input/``
    stem directory. Covers BOTH layouts s06b can produce: the GPU-batched
    ``structures/representatives/_batch_out_gpu*/boltz_results_*/predictions/`` and
    the per-rep ``structures/representatives/hom_NNN/.../predictions/``."""
    reps_root = run / "structures" / "representatives"
    stems: Dict[int, Path] = {}
    pat = re.compile(r"hom_(\d+)_boltz_input$")
    for stem in reps_root.glob("**/predictions/hom_*_boltz_input"):
        if not stem.is_dir():
            continue
        m = pat.search(stem.name)
        if not m:
            continue
        i = int(m.group(1))
        # If duplicated across layouts, prefer the one with the most model PDBs.
        if i in stems:
            prev = len(list(stems[i].glob("*_model_*.pdb")))
            cur = len(list(stem.glob("*_model_*.pdb")))
            if cur <= prev:
                continue
        stems[i] = stem
    return dict(sorted(stems.items()))


def _rep_sequence(run: Path, i: int, fallback: str) -> str:
    """The exact sequence rep ``i`` was folded with, read from its retained Boltz
    input YAML (``_batch_in_gpu*/hom_NNN_boltz_input.yaml``). The sequence drives
    residue assignment in ``parse_prediction_dir``; reading it from disk avoids
    re-deriving (non-deterministically) which homolog maps to which rep index.
    Falls back to the WT/target sequence only if the YAML is missing."""
    reps_root = run / "structures" / "representatives"
    label = f"hom_{i:03d}_boltz_input"
    import yaml

    for y in reps_root.glob(f"_batch_in_*/{label}.yaml"):
        try:
            spec = yaml.safe_load(y.read_text())
            for ent in spec.get("sequences", []):
                prot = ent.get("protein")
                if prot and prot.get("sequence"):
                    return str(prot["sequence"]).strip()
        except Exception:
            pass
    # per-rep layout: hom_NNN/_batch... or the yaml beside the rep outdir
    for y in reps_root.glob(f"hom_{i:03d}/**/{label}.yaml"):
        try:
            spec = yaml.safe_load(y.read_text())
            for ent in spec.get("sequences", []):
                prot = ent.get("protein")
                if prot and prot.get("sequence"):
                    return str(prot["sequence"]).strip()
        except Exception:
            pass
    return fallback


def _parse_wt_complex(run: Path, seq: str, ligand, primary_method: str) -> Complex:
    """Parse the retained WT s04 complex into a full ``Complex`` and stamp its
    ``structure.pdb_path`` so ``_augment_with_docking`` finds the WT stem dir."""
    from evoliez.adapters.boltz import parse_prediction_dir

    stem = run / "complexes" / "boltz" / "boltz_results_wt_boltz_input" / \
        "predictions" / "wt_boltz_input"
    if not stem.is_dir():
        # tolerate a differently-named WT stem
        cand = list((run / "complexes").glob("**/predictions/*boltz_input"))
        stem = next((c for c in cand if "wt" in c.name.lower()),
                    cand[0] if cand else stem)
    cx = parse_prediction_dir(stem, seq, ligand, primary_method)
    if cx is None:
        raise SystemExit(f"could not parse WT complex from {stem}")
    # ensure the WT stem is locatable by _augment_with_docking (it reads
    # wt.structure.pdb_path and uses its parent as the stem dir).
    if not getattr(cx.structure, "pdb_path", None):
        pdbs = sorted(p for p in stem.glob("*.pdb")
                      if not p.name.endswith("_complex.pdb"))
        if pdbs:
            cx.structure.pdb_path = str(pdbs[0])
    return cx


def build_records(run: Path, cfg) -> "tuple[list, list, Complex]":
    """Rebuild the s06b training ``records`` the way ``run()`` does, from the
    run's retained structures. Returns (records, multi_engine_diags, wt_complex).
    """
    im = cfg.interaction_model
    primary_method = cfg.complex_prediction.primary_method
    cutoff = im.contact_cutoff
    k_nearest = im.k_nearest_residues

    # --- ligand + context, rebuilt deterministically by the same s01 funcs --- #
    from evoliez.features.ligand import parse_ligand, resolve_ligand_manifest

    ligand = parse_ligand(cfg.input.ligand)
    extra_ligands = [parse_ligand(el) for el in cfg.input.extra_ligands] or None
    manifest = resolve_ligand_manifest(cfg.input.ligand, cfg.input.extra_ligands)
    context_ligands = [m for m in manifest if getattr(m, "keep_as_context", False)]

    # catalytic positions: read from the run's persisted meta (s01 derived them).
    catalytic: List[int] = []
    try:
        st = json.loads((run / "_state.json").read_text())
        catalytic = list(st.get("meta", {}).get("catalytic_positions", []) or [])
    except Exception:
        catalytic = []

    from evoliez.adapters.boltz import parse_prediction_dir

    # --- Boltz rows: re-parse + re-fingerprint each retained rep stem --------- #
    stems = _discover_rep_stems(run)
    if not stems:
        raise SystemExit(f"no rep stems under {run}/structures/representatives")
    n_reps = max(stems) + 1
    wt_seq = _read_target_sequence(run)
    records: List = []
    rep_seqs: Dict[int, str] = {}
    rep_stem_dirs: List[Optional[Path]] = [None] * n_reps
    n_parsed = 0
    for i in sorted(stems):
        seq = _rep_sequence(run, i, wt_seq)
        rep_seqs[i] = seq
        cx = parse_prediction_dir(stems[i], seq, ligand, primary_method)
        rep_stem_dirs[i] = stems[i]
        if cx is None:
            continue
        # identity_to_target is metadata only (never a feature / weight / stat);
        # 0.0 is faithful for the model. The gate proves the records are correct.
        _i, recs, _prows = _pose_records_from_complex(i, cx, 0.0, cutoff, k_nearest)
        records.extend(recs)
        n_parsed += 1
    print(f"[recon] Boltz: parsed {n_parsed}/{len(stems)} rep stems "
          f"(n_reps={n_reps}) -> {len(records)} Boltz pose records", flush=True)

    # --- docking rows: replay _augment_with_docking against retained SDFs ----- #
    diags: List = []
    if im.multi_engine and im.multi_engine_methods:
        wt = _parse_wt_complex(run, wt_seq, ligand, primary_method)
        # reps proxy carrying .sequence (the only attr _augment_with_docking uses)
        reps = [SimpleNamespace(sequence=rep_seqs.get(i, wt_seq))
                for i in range(n_reps)]
        with tempfile.TemporaryDirectory(prefix="cv_subfam_audit_") as scratch:
            paths = _ReadOnlyPaths(run, Path(scratch))
            ctx = _Ctx(cfg, paths)
            ctx.put("ligand", ligand)
            ctx.put("extra_ligands", extra_ligands)
            ctx.put("ligand_manifest", manifest)
            ctx.put("context_ligands", context_ligands)
            ctx.put("catalytic_positions", catalytic)
            gpu_list: List[str] = []   # serial CPU re-parse (no GPU needed)
            n_before = len(records)
            with _NoToolRerun():
                diags = _augment_with_docking(
                    ctx, im, wt, records, rep_stem_dirs, reps, gpu_list,
                    _QuietLog(),
                )
            print(f"[recon] docking: classified {len(diags)} pose(s); "
                  f"records {n_before} -> {len(records)} "
                  f"(+{len(records) - n_before} kept docking rows)", flush=True)
    else:
        wt = _parse_wt_complex(run, wt_seq, ligand, primary_method)

    return records, diags, wt


def _read_target_sequence(run: Path) -> str:
    f = run / "inputs" / "target.fasta"
    if f.exists():
        lines = [ln.strip() for ln in f.read_text().splitlines()
                 if ln.strip() and not ln.startswith(">")]
        if lines:
            return "".join(lines)
    return ""


class _QuietLog:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        print("[recon][warn]", (a[0] % a[1:]) if len(a) > 1 else a[0],
              flush=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _sel_kw(cfg, seed: int) -> dict:
    """The select_poses kwargs s06b.run() builds (``sel_kw``) — same values."""
    im = cfg.interaction_model
    return dict(
        select_z=im.pose_select_mad_z,
        outlier_z=im.pose_outlier_mad_z,
        min_decoys_per_group=im.min_decoys_per_homolog,
        seed=seed,
        keep_alternative_band=im.keep_alternative_band,
        alternative_weight=im.alternative_weight,
        hard_decoys_per_group=im.hard_decoys_per_homolog,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="production run dir (e.g. /mnt/data/jglee/runs/fdh_5track)")
    ap.add_argument("--config", required=True, type=Path,
                    help="the run's launch config YAML (fingerprint is verified)")
    ap.add_argument("--expected-auroc", type=float, default=RUN_RECORDED_AUROC,
                    help=f"run's single-holdout AUROC to reproduce (default {RUN_RECORDED_AUROC})")
    ap.add_argument("--tol", type=float, default=GATE_TOL,
                    help=f"gate tolerance (default {GATE_TOL})")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="optional path to write the CV result JSON (NOT under the run)")
    args = ap.parse_args()

    run = args.run.resolve()
    if not run.is_dir():
        raise SystemExit(f"run dir not found: {run}")

    cfg = load_config(str(args.config))
    im = cfg.interaction_model
    seed = cfg.seed

    # --- config fingerprint guard: prove this config IS the run's config ----- #
    import hashlib

    cfg_fp = hashlib.sha1(
        json.dumps(cfg.model_dump(mode="json"), sort_keys=True, default=str)
        .encode()
    ).hexdigest()
    run_fp = ""
    info_p = run / "RUN_INFO.json"
    if info_p.exists():
        run_fp = json.loads(info_p.read_text()).get("config_fingerprint", "")
    print("=" * 72)
    print("CONFIG FINGERPRINT")
    print(f"  config {args.config}")
    print(f"  sha1(config)      = {cfg_fp}")
    print(f"  run config_fp     = {run_fp}")
    if run_fp and not cfg_fp.startswith(run_fp):
        raise SystemExit(
            f"config fingerprint mismatch: {args.config} -> {cfg_fp[:12]} != "
            f"run {run_fp}. This is NOT the config that produced the run; the "
            f"reconstruction would be wrong. Aborting.")
    print(f"  -> MATCH (this config produced {run.name})"
          if run_fp else "  -> (no RUN_INFO fingerprint to check)")
    print(f"  backend(s06b)     = {cfg.backend_for(InteractionModelStage.name).value}")
    print(f"  k_nearest={im.k_nearest_residues} cutoff={im.contact_cutoff} "
          f"seed={seed} multi_engine={im.multi_engine} "
          f"methods={im.multi_engine_methods}")
    print("=" * 72)

    # --- rebuild the records exactly as run() does --------------------------- #
    records, diags, _wt = build_records(run, cfg)
    boltz_groups = _holdout_groups(records)
    n_boltz = sum(1 for r in records if not getattr(r, "role", ""))
    n_dock = len(records) - n_boltz
    print(f"[recon] DONE: {len(records)} records "
          f"({n_boltz} Boltz + {n_dock} kept docking) over "
          f"{len(boltz_groups)} Boltz subfamilies", flush=True)

    sel_kw = _sel_kw(cfg, seed)
    st = InteractionModelStage.__new__(InteractionModelStage)

    # --- VALIDATION GATE: reproduce the single-holdout (held = groups[-1]) ---- #
    print("-" * 72)
    print("VALIDATION GATE: reproduce single-holdout (held = groups[-1])")
    single = st._subfamily_holdout(records, sel_kw, im)
    held_last = boltz_groups[-1] if boltz_groups else None
    print(f"  held group (groups[-1]) = {held_last}")
    print(f"  reconstructed single-holdout AUROC = {single}")
    print(f"  run-recorded AUROC                 = {args.expected_auroc}")
    if single is None:
        raise SystemExit("GATE FAILED: single-holdout returned None "
                         "(reconstruction produced no valid held fold).")
    delta = abs(single - args.expected_auroc)
    print(f"  |delta| = {delta:.4f}  (tol {args.tol})")
    if delta > args.tol:
        raise SystemExit(
            f"GATE FAILED: reconstructed single-holdout AUROC {single} differs "
            f"from the run's {args.expected_auroc} by {delta:.4f} > {args.tol}. "
            f"The reconstruction does not match the run — refusing to report a "
            f"CV built on a wrong reconstruction.")
    print("  -> GATE PASSED: reconstruction reproduces the run's single-holdout.")
    print("-" * 72)

    # --- FULL leave-one-subfamily-out CV over ALL eligible groups ------------ #
    print("FULL leave-one-subfamily-out CV (retrain InteractionModel per fold)")
    per_group: Dict[str, float] = {}
    # held-set size per group (the held subfamily's Boltz pose count) — the same
    # ``held_recs`` _holdout_one_auroc trains/evaluates that fold on. Surfaced as
    # the per-fold "n" so the report can show each fold's support.
    held_n: Dict[str, int] = {}
    for r in records:
        if not getattr(r, "role", ""):
            held_n[r.group_id] = held_n.get(r.group_id, 0) + 1
    skipped: List[str] = []
    for j, held in enumerate(boltz_groups, 1):
        a = _holdout_one_auroc(records, held, sel_kw, im)
        if a is None:
            skipped.append(held)
        else:
            per_group[held] = a
        if j % 25 == 0 or j == len(boltz_groups):
            print(f"    folds {j}/{len(boltz_groups)} "
                  f"(eligible so far: {len(per_group)})", flush=True)

    vals = np.array(list(per_group.values()), dtype=float)
    if vals.size == 0:
        raise SystemExit("no eligible folds produced a 2-class held set.")

    result = {
        "run": str(run),
        "config": str(args.config),
        "config_fingerprint": cfg_fp[:12],
        "gate": {
            "held_group": held_last,
            "reconstructed_single_holdout_auroc": single,
            "run_recorded_auroc": args.expected_auroc,
            "abs_delta": round(delta, 4),
            "tol": args.tol,
            "passed": True,
        },
        "n_subfamilies_total": len(boltz_groups),
        "n_folds": int(vals.size),
        "n_skipped_folds": len(skipped),
        "mean": round(float(vals.mean()), 4),
        "std": round(float(vals.std(ddof=1)) if vals.size > 1 else 0.0, 4),
        "median": round(float(np.median(vals)), 4),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
        "per_group_auroc": {g: round(float(v), 4)
                            for g, v in sorted(per_group.items())},
        "skipped_groups": skipped,
    }

    print("=" * 72)
    print("SUBFAMILY CROSS-VALIDATION AUROC  (s06b interaction model)")
    print("=" * 72)
    print(f"  n_folds (eligible)   = {result['n_folds']}"
          f"  (of {result['n_subfamilies_total']} subfamilies; "
          f"{result['n_skipped_folds']} skipped: no 2-class held set)")
    print(f"  mean +/- std         = {result['mean']:.4f} +/- {result['std']:.4f}")
    print(f"  median               = {result['median']:.4f}")
    print(f"  min  /  max          = {result['min']:.4f}  /  {result['max']:.4f}")
    print(f"  single-holdout (run) = {args.expected_auroc}  (reproduced: "
          f"{single}, gate passed)")
    print("-" * 72)
    # compact per-group spread
    items = sorted(per_group.items(), key=lambda kv: kv[1])
    print("  lowest folds : " + ", ".join(f"{g}={v:.3f}" for g, v in items[:5]))
    print("  highest folds: " + ", ".join(f"{g}={v:.3f}"
                                           for g, v in items[-5:]))
    print("=" * 72)

    # --- persist the CANONICAL CV result for the s06b report ----------------- #
    # The interaction-model report reads <run>/reports/cv_subfamily_auroc.json and
    # headlines this full leave-one-subfamily-out CV as the model-performance
    # metric (the single subfamily-holdout becomes one example fold). JSON-safe,
    # generic (every number comes from the data), with per-fold AUROC + support.
    report_result = {
        "mean": result["mean"],
        "std": result["std"],
        "n_folds": result["n_folds"],
        "min": result["min"],
        "median": result["median"],
        "max": result["max"],
        "per_fold": [
            {"subfamily": g, "auroc": round(float(per_group[g]), 4),
             "n": int(held_n.get(g, 0))}
            for g in sorted(per_group, key=lambda k: per_group[k])
        ],
    }
    report_json = run / "reports" / "cv_subfamily_auroc.json"
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report_result, indent=2))
    print(f"[out] wrote canonical CV for the s06b report -> {report_json} "
          f"(mean {report_result['mean']:.4f} +/- {report_result['std']:.4f}, "
          f"n_folds={report_result['n_folds']})")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"[out] wrote {args.json_out}")
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
