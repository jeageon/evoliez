"""Pipeline stages s01-s11 (spec sections 6-16)."""

from evoliez.stages.base import Stage
from evoliez.stages.s01_input_preprocess import InputPreprocessStage
from evoliez.stages.s02_homolog import HomologStage
from evoliez.stages.s03_msa import MSAStage
from evoliez.stages.s04_complex import ComplexPredictionStage
from evoliez.stages.s05_docking import DockingStage
from evoliez.stages.s06_interaction_graph import InteractionGraphStage
from evoliez.stages.s06b_interaction_model import InteractionModelStage
from evoliez.stages.s07_mutation_gen import MutationGenStage
from evoliez.stages.s08_reranker import RerankerStage
from evoliez.stages.s08b_mutant_boltz import MutantBoltzStage
from evoliez.stages.s09_nonmd_validation import NonMDValidationStage
from evoliez.stages.s10_md import MDStage
from evoliez.stages.s11_final_ranking import FinalRankingStage

ALL_STAGES = [
    InputPreprocessStage,
    HomologStage,
    MSAStage,
    ComplexPredictionStage,
    DockingStage,
    InteractionGraphStage,
    InteractionModelStage,
    MutationGenStage,
    RerankerStage,
    MutantBoltzStage,
    NonMDValidationStage,
    MDStage,
    FinalRankingStage,
]

__all__ = ["Stage", "ALL_STAGES"]
