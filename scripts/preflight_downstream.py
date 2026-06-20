"""Preflight: run a TINY full mock pipeline (s01->s11) to verify the deployed
downstream stages + this session's fixes run error-free on the SERVER.

Exercises, with the mock backend (no real tools needed) and CUDA_VISIBLE_DEVICES
set so the parallel branches actually fire:
  * s06b + s08b fork ProcessPool fan-out (server uses fork, not the spawn the
    local tests use),
  * s09 bounded ThreadPool fan-out + multi-docker + real-pose catalytic,
  * s07 generated-provenance + s11 funnel/attrition report writing.

Throwaway output dir; does NOT touch any real run.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")   # mock ignores the GPU; this only trips the fan-out branch
import sys
import time
import traceback
from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

OUT = f"/mnt/data/jglee/preflight_{int(time.time())}"
cfg = load_config("configs/example_fdh_nadp.yaml", {
    "project.output_dir": OUT,
    "input.target_fasta": "examples/fdh/target.fasta",
    "backend": "mock",
    "mutation_generation": {"methods": ["chemistry_rules", "msa_sampler"],
                            "max_candidates": 14},
    "interaction_model": {"representative_homologs": 3, "poses_per_homolog": 3},
    "reranking": {"top_for_redocking": 6, "top_for_md": 2, "mutant_boltz_top_n": 3},
    "validation": {"md": {"top_candidates": 2}},
    "gnn": {"build_dataset": False},
})
seed_everything(cfg.seed)
print(f">> preflight out={OUT}  CVD={os.environ['CUDA_VISIBLE_DEVICES']}")
try:
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)
    print("=== PIPELINE OK (s01->s11, mock, fork fan-out exercised) ===")
    print("   ranked candidates:", len(ctx.get("ranked_candidates") or ctx.get("candidates") or []))
    print("   mutant_complexes:", len(ctx.get("mutant_complexes") or {}))
    rep = Path(OUT) / "reports"
    for f in ("provenance/generated_candidates.csv",
              "provenance/attrition_funnel.csv",
              "provenance/generator_survival.csv",
              "provenance/candidate_provenance.csv",
              "provenance/funnel_provenance.json"):
        print(f"   {f}: {'OK' if (rep / f).exists() else 'MISSING'}")
    print("RESULT: PASS")
except Exception:
    print("=== PIPELINE FAILED ===")
    traceback.print_exc()
    print("RESULT: FAIL")
    sys.exit(1)
