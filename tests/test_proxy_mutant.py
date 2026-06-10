"""Proxy-mutant disruptiveness must use the candidate's REAL conservation
(ultra-review #21). In s08 the proxy was built before cand.details['features']
existed, so conservation defaulted to 0.5 and every single-mutant got identical
disruptiveness -> byte-identical d_ligand_iptm / d_complex_* deltas (zero
discriminative power in the reranker)."""

from pathlib import Path

from evoliez.adapters.boltz import predict_complex
from evoliez.config import Backend, ComplexPredictionConfig, LigandInput
from evoliez.features.ligand import parse_ligand
from evoliez.stages.s08_reranker import _approx_mutant_complex
from evoliez.types import Candidate, Mutation


def _cx():
    lig = parse_ligand(LigandInput(id="L", type="smiles", value="CCO"))
    return predict_complex(
        "t", "ACDEFGHIKLMNPQRSTVWY" * 2, lig,
        ComplexPredictionConfig(diffusion_samples=2),
        Path("/tmp/ez_proxy"), backend=Backend.mock,
    )


def test_proxy_mutant_disruptiveness_tracks_conservation():
    cx = _cx()
    cand = Candidate(candidate_id="c", mutations=[Mutation("C", 2, "K")],
                     generator="g")
    lo = _approx_mutant_complex(cx, cand, conservation=0.1)
    hi = _approx_mutant_complex(cx, cand, conservation=0.9)
    # higher conservation -> more disruptive -> more-degraded Boltz metric
    assert lo.metrics["ligand_iptm"] != hi.metrics["ligand_iptm"]
    assert hi.metrics["ligand_iptm"] < lo.metrics["ligand_iptm"]


def test_proxy_mutant_falls_back_to_features_when_not_passed():
    cx = _cx()
    cand = Candidate(candidate_id="c", mutations=[Mutation("C", 2, "K")],
                     generator="g")
    cand.details["features"] = {"conservation": 0.9}
    via_features = _approx_mutant_complex(cx, cand)
    via_kwarg = _approx_mutant_complex(cx, cand, conservation=0.9)
    assert via_features.metrics["ligand_iptm"] == via_kwarg.metrics["ligand_iptm"]
