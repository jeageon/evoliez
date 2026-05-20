"""Protein-ligand complex prediction (spec section 9).

real: Boltz / Boltz-2 (https://github.com/jwohlwend/boltz).
mock: deterministic synthetic complex (structure + pocketed ligand + scores).

Outputs a diffusion-sample **ensemble** with rich confidence/affinity metrics.
Per the data policy these are FEATURES / sample-weights / weak-labels /
filters - never supervised labels (see ml/labels.py, docs/ML_DATA_POLICY.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import yaml

from evoliez.adapters.base import (
    place_ligand_in_pocket,
    synthetic_structure,
    write_min_pdb,
)
from evoliez.config import Backend, ComplexPredictionConfig
from evoliez.logging_utils import get_logger
from evoliez.types import BoltzSample, Complex, Ligand, LigandAtom
from evoliez.utils.gpu import apply_gpu_selection
from evoliez.utils.seeds import derive_seed
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.boltz")

# Confidence/affinity keys carried on Complex.metrics (Boltz output, §1/§5).
METRIC_KEYS = [
    "confidence_score", "ptm", "iptm", "ligand_iptm", "complex_plddt",
    "complex_iplddt", "complex_pde", "complex_ipde",
    "affinity_pred_value", "affinity_probability_binary",
    "affinity_pred_value1", "affinity_pred_value2", "ensemble_disagreement",
]


def predict_complex(
    label: str,
    sequence: str,
    ligand: Ligand,
    cfg: ComplexPredictionConfig,
    outdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
    msa_path: Optional[Path] = None,
    pocket_residues: Optional[List[int]] = None,
) -> Complex:
    outdir.mkdir(parents=True, exist_ok=True)
    if backend is Backend.real:
        return _predict_real(
            label, sequence, ligand, cfg, outdir,
            dry_run=dry_run, msa_path=msa_path,
            pocket_residues=pocket_residues,
        )
    return _predict_mock(label, sequence, ligand, cfg, outdir)


# --------------------------------------------------------------------------- #
# Metric synthesis (mock) / aggregation
# --------------------------------------------------------------------------- #
def _synth_metrics(seed: int, sample_idx: int, n: int) -> dict:
    """Deterministic, monotone-ish confidence: sample 0 best, later worse."""
    h = derive_seed(seed, "m", str(sample_idx))
    decay = sample_idx / max(1, n)
    base = 0.85 - 0.30 * decay - (h % 12) / 100.0
    base = max(0.30, min(0.95, base))
    iptm = max(0.2, base - 0.05 - (h % 8) / 100.0)
    lig_iptm = max(0.15, iptm - 0.04 - (h % 7) / 100.0)
    plddt = max(0.4, base + 0.03 - (h % 6) / 100.0)
    iplddt = max(0.35, plddt - 0.05 - (h % 5) / 100.0)
    pde = round(1.0 + 4.0 * (1.0 - base) + (h % 10) / 10.0, 3)  # Å, lower better
    ipde = round(pde + 0.5 + (h % 8) / 10.0, 3)
    aff = round(-1.0 - 4.0 * base - (h % 50) / 100.0, 3)  # log10(IC50)-like
    return {
        "confidence_score": round(0.8 * plddt + 0.2 * iptm, 4),
        "ptm": round(base, 4),
        "iptm": round(iptm, 4),
        "ligand_iptm": round(lig_iptm, 4),
        "complex_plddt": round(plddt, 4),
        "complex_iplddt": round(iplddt, 4),
        "complex_pde": pde,
        "complex_ipde": ipde,
        "affinity_pred_value": aff,
        "affinity_probability_binary": round(min(0.99, max(0.01, base)), 4),
        "affinity_pred_value1": round(aff - (h % 20) / 100.0, 3),
        "affinity_pred_value2": round(aff + (h % 20) / 100.0, 3),
    }


def _finalize(cx: Complex) -> Complex:
    if cx.samples:
        best = cx.samples[0]
        cx.metrics = dict(best.metrics)

        def _std(xs):
            if len(xs) < 2:
                return 0.0
            mu = sum(xs) / len(xs)
            return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5

        # Per-sample ensemble disagreement (P0.2). For real Boltz this is the
        # std across the diffusion-sample affinity / confidence; for mock it
        # collapses to the affinity_pred_value1 vs ..2 gap. Either way the
        # output is a single number the reranker / final report can use to
        # mark candidates as `Uncertain` when the prediction is internally
        # noisy.
        aff_per_sample = [
            s.metrics.get("affinity_pred_value")
            for s in cx.samples
            if s.metrics.get("affinity_pred_value") is not None
        ]
        conf_per_sample = [
            s.metrics.get("confidence_score")
            for s in cx.samples
            if s.metrics.get("confidence_score") is not None
        ]
        if aff_per_sample:
            cx.metrics["affinity_ensemble_std"] = round(_std(aff_per_sample), 4)
            cx.metrics["affinity_ensemble_mean"] = round(
                sum(aff_per_sample) / len(aff_per_sample), 4
            )
        if conf_per_sample:
            cx.metrics["confidence_ensemble_std"] = round(_std(conf_per_sample), 4)
        # Legacy single-pair gap (mock path provides v1/v2 stubs); keep it
        # for downstream code that already keys on it.
        v1 = cx.metrics.get("affinity_pred_value1")
        v2 = cx.metrics.get("affinity_pred_value2")
        if v1 is not None and v2 is not None:
            cx.metrics["ensemble_disagreement"] = round(abs(v1 - v2), 4)
        elif aff_per_sample and len(aff_per_sample) >= 2:
            # Real path with no v1/v2: define disagreement as the affinity std.
            cx.metrics["ensemble_disagreement"] = cx.metrics["affinity_ensemble_std"]

        cx.confidence = cx.metrics.get("confidence_score", cx.confidence)
        cx.affinity_score = cx.metrics.get("affinity_pred_value", cx.affinity_score)
    return cx


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #
def _predict_mock(
    label: str, sequence: str, ligand: Ligand, cfg: ComplexPredictionConfig,
    outdir: Path,
) -> Complex:
    seed = derive_seed(0xB01D, label, sequence[:32])
    struct = synthetic_structure(sequence, seed=seed, method="boltz-mock")
    n = max(1, cfg.diffusion_samples)

    # One shared consensus pocket pose; samples vary modestly around it so
    # recurring (residue, ligand-atom) contacts emerge -> meaningful
    # ensemble contact frequency (the priority-1 feature).
    base = place_ligand_in_pocket(struct, ligand, seed=seed)
    samples: List[BoltzSample] = []
    order = []
    for i in range(n):
        s = derive_seed(seed, "s", str(i))
        drift = 0.15 + 0.9 * (i / max(1, n))  # small wobble, grows with idx
        moved = []
        for k, a in enumerate(base):
            o = ((derive_seed(s, str(k)) % 200) / 100.0 - 1.0) * drift
            na = LigandAtom(**vars(a))
            na.coord = (a.coord[0] + o, a.coord[1] - o * 0.4, a.coord[2] + o * 0.3)
            moved.append(na)
        metrics = _synth_metrics(seed, i, n)
        rp = [round(metrics["complex_plddt"], 3)] * len(struct.residues)
        samples.append(BoltzSample(idx=i, ligand_atoms=moved, metrics=metrics,
                                   residue_plddt=rp))
        order.append((metrics["confidence_score"], i))

    order.sort(reverse=True)
    samples = [samples[i] for _, i in order]
    for j, smp in enumerate(samples):
        smp.idx = j

    best = samples[0]
    pdb = outdir / f"{label}_complex.pdb"
    write_min_pdb(pdb, struct, best.ligand_atoms)
    struct.pdb_path = str(pdb)
    cx = Complex(
        structure=struct,
        ligand=Ligand(
            id=ligand.id, smiles=ligand.smiles, atoms=best.ligand_atoms,
            formal_charge=ligand.formal_charge,
            n_rotatable_bonds=ligand.n_rotatable_bonds, source=ligand.source,
        ),
        method="boltz-mock",
        path=str(pdb),
        samples=samples,
    )
    return _finalize(cx)


# --------------------------------------------------------------------------- #
# Real
# --------------------------------------------------------------------------- #
def _to_a3m(src: Path, dst: Path) -> Optional[Path]:
    """Boltz-2 rejects FASTA MSAs ('only a3m or csv'); s03 writes aligned
    FASTA. A block alignment (equal-length, '-' gaps, no a3m insertion
    semantics) is valid a3m once uppercased, so rewrite it with an .a3m
    extension. Returns None if the source is missing/empty."""
    try:
        text = src.read_text()
    except OSError:
        return None
    recs: List[tuple[str, str]] = []
    hdr, buf = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if hdr is not None:
                recs.append((hdr, "".join(buf)))
            hdr, buf = line.rstrip(), []
        elif line.strip():
            buf.append(line.strip())
    if hdr is not None:
        recs.append((hdr, "".join(buf)))
    recs = [(h, s.upper().replace(".", "-").replace(" ", "-"))
            for h, s in recs if s.strip()]
    if not recs:
        return None
    dst.write_text("".join(f"{h}\n{s}\n" for h, s in recs))
    return dst


def _predict_real(
    label: str,
    sequence: str,
    ligand: Ligand,
    cfg: ComplexPredictionConfig,
    outdir: Path,
    *,
    dry_run: bool,
    msa_path: Optional[Path],
    pocket_residues: Optional[List[int]] = None,
) -> Complex:
    # dry-run previews the FULL command set (like Vina) without the tool
    # installed, writes the exact Boltz input YAML so the contract can be
    # eyeballed, and returns the same structured mock contract as a real run.
    if dry_run:
        log.info("[dry-run] boltz predict (diffusion_samples=%d) for %s",
                 cfg.diffusion_samples, label)
    else:
        require("boltz")
        apply_gpu_selection()
    spec = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence}},
            {"ligand": {"id": "B", "smiles": ligand.smiles}},
        ],
    }
    if cfg.predict_affinity:
        spec["properties"] = [{"affinity": {"binder": "B"}}]
    # P0.2: pocket steering contract. When the config asks for pocket
    # constraints AND we have residues to point at, emit Boltz's
    # `constraints` block so the diffusion model actively steers the
    # ligand toward those residues - instead of `pocket_constraints` being
    # a no-op metadata flag (the original behaviour). Format per Boltz-2
    # YAML schema: each pocket constraint binds a binder chain id to a
    # list of [chain, residue_index] contact residues.
    if cfg.pocket_constraints and pocket_residues:
        spec["constraints"] = [
            {
                "pocket": {
                    "binder": "B",
                    "contacts": [["A", int(i)] for i in pocket_residues],
                }
            }
        ]
    if msa_path is not None:
        mp = Path(msa_path)
        if mp.suffix.lower() in (".a3m", ".csv"):
            a3m: Optional[Path] = mp
        else:
            a3m = _to_a3m(mp, outdir / f"{label}_msa.a3m")
        if a3m is not None:
            spec["sequences"][0]["protein"]["msa"] = str(a3m)
            msa_path = a3m  # keep the --use_msa_server guard correct
        else:
            log.warning("MSA %s unusable; falling back to MSA server", mp)
            msa_path = None

    yml = outdir / f"{label}_boltz_input.yaml"
    yml.write_text(yaml.safe_dump(spec, sort_keys=False))

    cmd = [
        "boltz", "predict", str(yml), "--out_dir", str(outdir),
        "--diffusion_samples", str(max(1, cfg.diffusion_samples)),
        # Boltz defaults to mmCIF; force PDB so the structure parser works
        # (a CIF backstop parser also exists below).
        "--output_format", "pdb",
    ]
    if cfg.use_msa_server and msa_path is None:
        cmd.append("--use_msa_server")
    if not getattr(cfg, "use_kernels", False):
        # Boltz-2 hard-fails (ModuleNotFoundError cuequivariance_torch) if the
        # optimized kernels' optional dep is missing; --no_kernels uses the
        # pure-torch path. Stock `pip install boltz` has no cuequivariance.
        cmd.append("--no_kernels")
    run(cmd, dry_run=dry_run, timeout=None)

    if dry_run:
        return _predict_mock(label, sequence, ligand, cfg, outdir)

    # Boltz writes predictions ONLY under boltz_results_<label>_boltz_input/
    # (predictions/<label>_boltz_input/...). Scope discovery to THIS label's
    # directory only - when multiple Boltz calls share the same out_dir
    # (e.g., s08b runs per-mutant Boltz for several candidates into the
    # same `complexes/mutant_boltz/`), an unscoped glob picks the
    # alphabetically-first result and silently returns it for every
    # candidate. The sequence guard then fires for the OTHERS because
    # `_pdb_one_letter_seq` of the wrong PDB doesn't match their intended
    # mutant sequence - server-verified bug.
    label_results = outdir / f"boltz_results_{label}_boltz_input"
    roots = [label_results] if label_results.exists() else sorted(
        # Backwards-compat path: some Boltz versions/CLI flags emit
        # boltz_results_<label>/ without the _boltz_input suffix. Match
        # only THIS label so the cross-contamination above can't happen.
        outdir.glob(f"boltz_results_{label}*")
    )
    found: List[Path] = []
    for root in roots:
        found += sorted(root.rglob("*.cif")) + sorted(root.rglob("*.pdb"))
    found = [p for p in found if not p.name.endswith("_complex.pdb")]
    if not found:
        # Last-resort fallback: an unscoped glob, but emit a loud warning
        # so the cross-contamination case is visible if it ever recurs
        # with a fresh Boltz version using yet another naming convention.
        log.warning(
            "Boltz produced no scoped output for %s under "
            "boltz_results_%s_boltz_input/; falling back to wide glob "
            "(cross-mutant contamination possible)", label, label,
        )
        roots = sorted(outdir.glob("boltz_results_*"))
        for root in roots:
            found += sorted(root.rglob("*.cif")) + sorted(root.rglob("*.pdb"))
        found = [p for p in found if not p.name.endswith("_complex.pdb")]
    if not found:
        log.warning(
            "Boltz produced no prediction for %s "
            "(no boltz_results_*/.../*.cif|pdb); mock fallback", label,
        )
        return _predict_mock(label, sequence, ligand, cfg, outdir)

    structure_file = found[0]
    # Sample/affinity/plddt files are SCOPED to this candidate's boltz_results
    # directory; sharing the parent out_dir across mutants would otherwise
    # pull in the alphabetically-first mutant's metrics for every candidate.
    sample_scope = roots[0] if roots else outdir
    cx = _parse_real_structure(structure_file, sequence, ligand)
    cx.method = cfg.primary_method
    cx.path = str(structure_file)
    cx.samples = _parse_real_samples(sample_scope, cx.ligand.atoms)
    if not cx.samples:  # at least one sample from aggregate scores
        m = _parse_one_confidence(sample_scope) or {}
        cx.samples = [BoltzSample(idx=0, ligand_atoms=cx.ligand.atoms, metrics=m)]
    return _finalize(cx)


def _parse_real_samples(outdir: Path, lig_atoms) -> List[BoltzSample]:
    samples: List[BoltzSample] = []
    conf_files = sorted(outdir.rglob("confidence*model_*.json")) or sorted(
        outdir.rglob("confidence*.json")
    )
    aff = {}
    for jf in outdir.rglob("affinity*.json"):
        try:
            aff = json.loads(jf.read_text())
        except Exception:
            aff = {}
    for i, jf in enumerate(conf_files):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        m = {k: float(d[k]) for k in d if isinstance(d.get(k), (int, float))}
        for ak in ("affinity_pred_value", "affinity_probability_binary",
                   "affinity_pred_value1", "affinity_pred_value2"):
            if ak in aff:
                m[ak] = float(aff[ak])
        m.setdefault(
            "confidence_score",
            0.8 * m.get("complex_plddt", 0.0) + 0.2 * m.get("iptm", 0.0),
        )
        rp = _load_plddt(outdir, i)
        samples.append(BoltzSample(idx=i, ligand_atoms=lig_atoms, metrics=m,
                                   residue_plddt=rp))
    samples.sort(key=lambda s: -s.metrics.get("confidence_score", 0.0))
    for j, s in enumerate(samples):
        s.idx = j
    return samples


def _parse_one_confidence(outdir: Path) -> Optional[dict]:
    for jf in outdir.rglob("confidence*.json"):
        try:
            d = json.loads(jf.read_text())
            return {k: float(v) for k, v in d.items()
                    if isinstance(v, (int, float))}
        except Exception:
            pass
    return None


def _load_plddt(outdir: Path, idx: int) -> List[float]:
    try:
        import numpy as np

        npzs = sorted(outdir.rglob("plddt*.npz"))
        if idx < len(npzs):
            arr = np.load(npzs[idx])
            key = arr.files[0]
            return [float(x) for x in np.asarray(arr[key]).ravel()]
    except Exception:
        pass
    return []


def _parse_cif_atoms(path: Path):
    """Minimal mmCIF _atom_site loop parser (Boltz default output format).
    Returns (residues[CA], ligand_atoms[HETATM])."""
    from evoliez.types import LigandAtom, Residue

    lines = path.read_text().splitlines()
    cols: list[str] = []
    residues: list[Residue] = []
    lig: list[LigandAtom] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            hdr = []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                hdr.append(lines[j].strip())
                j += 1
            if hdr:
                cols = [h.split(".", 1)[1] for h in hdr]
                idx = {c: k for k, c in enumerate(cols)}
                need = ("group_PDB", "type_symbol", "label_atom_id",
                        "Cartn_x", "Cartn_y", "Cartn_z")
                if all(c in idx for c in need):
                    seqk = ("label_seq_id" if "label_seq_id" in idx
                            else "auth_seq_id")
                    while j < len(lines):
                        s = lines[j].strip()
                        if not s or s.startswith(("#", "loop_", "_")):
                            break
                        p = s.split()
                        if len(p) >= len(cols):
                            grp = p[idx["group_PDB"]]
                            x, y, z = (float(p[idx["Cartn_x"]]),
                                       float(p[idx["Cartn_y"]]),
                                       float(p[idx["Cartn_z"]]))
                            if grp == "ATOM" and p[idx["label_atom_id"]] == "CA":
                                try:
                                    ri = int(p[idx.get(seqk, -1)])
                                except (ValueError, KeyError):
                                    ri = len(residues) + 1
                                residues.append(
                                    Residue(index=ri, aa="X", ca=(x, y, z),
                                            sidechain_centroid=(x, y, z))
                                )
                            elif grp == "HETATM":
                                el = p[idx["type_symbol"]]
                                lig.append(LigandAtom(
                                    id=f"{el}{len(lig)}", element=el or "C",
                                    coord=(x, y, z)))
                        j += 1
                i = j
                continue
        i += 1
    return residues, lig


def _parse_real_structure(pdb: Path, sequence: str, ligand: Ligand) -> Complex:
    from evoliez.types import ProteinStructure, Residue

    residues: list[Residue] = []
    lig_atoms: list[LigandAtom] = []
    if pdb.suffix in (".cif", ".mmcif"):
        residues, lig_atoms = _parse_cif_atoms(pdb)
    else:
        for line in pdb.read_text().splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                idx = int(line[22:26])
                x, y, z = (float(line[30:38]), float(line[38:46]),
                           float(line[46:54]))
                residues.append(
                    Residue(index=idx, aa="X", ca=(x, y, z),
                            sidechain_centroid=(x, y, z))
                )
            elif line.startswith("HETATM"):
                x, y, z = (float(line[30:38]), float(line[38:46]),
                           float(line[46:54]))
                el = line[76:78].strip() or line[12:14].strip()
                lig_atoms.append(
                    LigandAtom(id=f"{el}{len(lig_atoms)}", element=el or "C",
                               coord=(x, y, z))
                )
    for i, r in enumerate(residues):
        if i < len(sequence):
            r.aa = sequence[i]
    if not lig_atoms:
        lig_atoms = ligand.atoms
    # atom-index lock: keep canonical ids/chemistry, adopt Boltz coordinates
    from evoliez.features.ligand import relabel_to_canonical

    lig_atoms, locked = relabel_to_canonical(lig_atoms, ligand.atoms)
    if not locked and ligand.atoms:
        # relabel_to_canonical already logged WHY (count mismatch or
        # chemistry-graph atom-order not verified); record the consequence and
        # tag the ligand source so id-keyed features can treat it as unreliable.
        log.warning(
            "Boltz ligand atom ids NOT verified for %s (canonical=%d, "
            "tool=%d); downstream id-keyed features may be inconsistent",
            getattr(ligand, "id", "?"), len(ligand.atoms), len(lig_atoms),
        )
    struct = ProteinStructure(
        sequence=sequence, residues=residues, method="boltz", pdb_path=str(pdb)
    )
    return Complex(structure=struct, ligand=Ligand(
        id=ligand.id, smiles=ligand.smiles, atoms=lig_atoms,
        formal_charge=ligand.formal_charge,
        source=ligand.source if locked else ligand.source + "|reindexed",
    ))
