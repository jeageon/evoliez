#!/usr/bin/env python
"""ThermoMPNN site-saturation ΔΔG on a WT pdb -> a JSON table keyed by AUTHOR
residue number + mutant AA, so the caller maps by PDB resnum (robust to any
numbering / the FDH transit-peptide offset) and verifies the wildtype. One model
embedding covers every position. CPU-only (the bundled weights were saved on CUDA
and ThermoMPNN's loaders don't pass map_location, so we force it here).

Run in the `thermompnn` conda env, from the ThermoMPNN repo dir.
Usage: thermompnn_ssm.py <wt.pdb> <chain> <out.json> <repo_dir>
Output: {"chain": "A", "by_resnum": {"<resnum>": {"wt": "A", "ddg": {"K": 1.23, ...}}}}
ΔΔG sign: positive = destabilizing (FoldX convention, drop-in).
"""
import json
import os
import sys

import torch

# Force CPU deserialization for every torch.load (PL checkpoint + the bundled
# ProteinMPNN encoder were saved on CUDA; this env is CPU-only torch).
_orig_load = torch.load
# Force CPU for EVERY load: PL's load_from_checkpoint passes map_location=None
# EXPLICITLY (so k.get(...,'cpu') would keep None and still cuda-deserialize); we
# override unconditionally since this env is CPU-only torch.
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})

wt_pdb, chain, out_json, repo = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.chdir(repo)
sys.path.insert(0, repo)
sys.path.insert(0, os.path.join(repo, "analysis"))

from omegaconf import OmegaConf  # noqa: E402
from protein_mpnn_utils import alt_parse_PDB  # noqa: E402
from datasets import Mutation  # noqa: E402
from thermompnn_benchmarking import get_trained_model  # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Config identical to analysis/custom_inference.py (merged onto the repo local.yaml).
cfg = OmegaConf.merge(
    OmegaConf.load(os.path.join(repo, "local.yaml")),
    OmegaConf.create({
        # override the authors' cluster paths (local.yaml ships /proj/kuhl_lab/...)
        "platform": {"thermompnn_dir": repo, "accel": "cpu"},
        "training": {"num_workers": 8, "learn_rate": 1e-3, "epochs": 100, "lr_schedule": True},
        "model": {"hidden_dims": [64, 32], "subtract_mut": True, "num_final_layers": 2,
                  "freeze_weights": True, "load_pretrained": True, "lightattn": True,
                  "lr_schedule": True},
    }),
)
model = get_trained_model(
    model_name=os.path.join(repo, "models/thermoMPNN_default.pt"),
    config=cfg, override_custom=True,
).eval().to("cpu")

mut_pdb = alt_parse_PDB(wt_pdb, chain)
seq = mut_pdb[0]["seq"]

# Map 0-based index into `seq` -> author residue number, from the pdb CA records.
# The candidate mutation positions are author resnums (the structure's residue
# index); building the map from the actual file makes it numbering-agnostic.
ca_resnums = []
with open(wt_pdb) as fh:
    for ln in fh:
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
            ca_resnums.append(int(ln[22:26]))
idx_to_resnum, ci = {}, 0
for i, aa in enumerate(seq):
    if aa == "-":
        continue
    if ci < len(ca_resnums):
        idx_to_resnum[i] = ca_resnums[ci]
        ci += 1

# Full SSM: every non-gap position x 19 substitutions (one embedding, many heads).
muts, meta = [], []
for i, aa in enumerate(seq):
    if aa == "-" or i not in idx_to_resnum or aa not in ALPHABET:
        continue
    for mt in ALPHABET:
        if mt == aa:
            continue
        muts.append(Mutation(position=i, wildtype=aa, mutation=mt, ddG=None,
                             pdb=mut_pdb[0]["name"]))
        meta.append((idx_to_resnum[i], aa, mt))

with torch.no_grad():
    preds, _ = model(mut_pdb, muts)

table = {}
for (rn, wt, mt), out in zip(meta, preds):
    if out is None:
        continue
    table.setdefault(str(rn), {"wt": wt, "ddg": {}})["ddg"][mt] = round(
        float(out["ddG"].cpu().item()), 4)

with open(out_json, "w") as fh:
    json.dump({"chain": chain, "by_resnum": table}, fh)
print(f"ThermoMPNN SSM: {len(table)} positions x ~19 AAs -> {out_json} (cpu)")
