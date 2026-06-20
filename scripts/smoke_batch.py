"""Server smoke: the GPU-batched Boltz chain on 2 tiny reps with REAL Boltz.
Verifies write_batch_input -> predict_batch (ONE model load) -> parse_prediction_dir
(per-stem scoped) end to end. ~2-3 min on one GPU."""
import shutil
from pathlib import Path

from evoliez.config import ComplexPredictionConfig
from evoliez.types import Ligand, LigandAtom
from evoliez.adapters.boltz import (
    write_batch_input, predict_batch, parse_prediction_dir)

ligand = Ligand(id="LIG", smiles="c1ccccc1",
                atoms=[LigandAtom(id="C%d" % i, element="C", coord=(float(i), 0.0, 0.0))
                       for i in range(6)])
cfg = ComplexPredictionConfig(diffusion_samples=1, use_msa_server=False)

base = Path("/tmp/batch_smoke")
shutil.rmtree(base, ignore_errors=True)
in_dir, out_dir = base / "in", base / "out"
seqs = {"hom_000": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        "hom_001": "GSHMKLVINGKTLKGEITVEGAKNAALPILF"}
for lab, seq in seqs.items():
    write_batch_input(in_dir, lab, seq, ligand, cfg, msa_path=None, extra_ligands=None)
print("yamls written:", sorted(p.name for p in in_dir.glob("*.yaml")))

results_dir = predict_batch(in_dir, out_dir, cfg, seed=42, gpu_device="0",
                            any_msa_server=False)
print("results_dir:", results_dir)

ok = True
for lab, seq in seqs.items():
    stem_dir = Path(results_dir) / "predictions" / ("%s_boltz_input" % lab)
    cx = parse_prediction_dir(stem_dir, seq, ligand, "boltz2")
    n = len(cx.samples) if cx else 0
    print("  %s: parsed=%s samples=%d residues=%d" % (
        lab, cx is not None, n, len(cx.structure.residues) if cx else 0))
    ok = ok and cx is not None and n >= 1
print("RESULT:", "PASS (one model load, both reps scoped-parsed)" if ok else "FAIL")
shutil.rmtree(base, ignore_errors=True)
