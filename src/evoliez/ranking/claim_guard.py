"""ClaimGuard — the v3 keystone (ROADMAP_V3 D3).

Stops reports from over-claiming. Two layers:

  * V3-1b (this module's `lint_text`): a NEGATION-SAFE, multilingual (English / Korean /
    LaTeX) over-claim linter. A rendered report that asserts "activity improved" /
    "predicts kcat" / "stable functional complex" while no wet-lab evidence supports it
    is a TEST FAILURE, not a warning. "This workflow does NOT predict kcat" is allowed.
  * V3-4 (`evaluate` + template whitelist + schema-validated fail-safe): the full
    provenance-driven engine that decides which claim CATEGORIES are unlocked and caps
    the claim strength. Built on top of this linter.

Pure-stdlib (re); runs in the light env.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

# --- claim categories ----------------------------------------------------------------
ACTIVITY_IMPROVEMENT = "activity_improvement"
KINETIC_PARAMETER_PREDICTION = "kinetic_parameter_prediction"
SHORT_MD_OVERCLAIM = "short_md_interpretation"
INACTIVE_CLASSIFICATION = "inactive_classification"
LONG_TERM_STABILITY = "long_term_stability"
UNQUALIFIED_STRENGTH = "unqualified_strength"   # V5-5: catalysis-implying label words
CATALYTIC_VALIDATION = "catalytic_validation"   # V6: "validated lead", "experimentally active"
ACTIVATION_BARRIER = "activation_barrier"       # V6: "reduced activation barrier"

ALL_CATEGORIES = (
    ACTIVITY_IMPROVEMENT, KINETIC_PARAMETER_PREDICTION, SHORT_MD_OVERCLAIM,
    INACTIVE_CLASSIFICATION, LONG_TERM_STABILITY, UNQUALIFIED_STRENGTH,
    CATALYTIC_VALIDATION, ACTIVATION_BARRIER,
)

# category -> prohibited-claim regexes (lowercased English; Korean kept literal).
# These match the *over-claim intent*, not a single banned token, so "catalytically
# superior" is caught without the word kcat, and "does not predict kcat" is not.
_PATTERNS = {
    ACTIVITY_IMPROVEMENT: [
        r"activity (is |was |would be )?improved",
        r"improved activity",
        r"enhanced catalysis",
        r"catalytic enhancement (confirmed|demonstrated|shown)",
        r"catalytically superior",
        r"(variant|mutant)s? (is|are|will be) (more )?(catalytically )?(active|superior)",
        r"improves? (the )?activity",
        # catalytic-comparison synonyms that dodge the word "activity" (reviewer-found
        # false negatives): "more productive/reaction-competent/reactive than WT",
        # "beats WT reactivity", "geometrically more productive active site than WT".
        r"more (catalytically )?(productive|reaction[- ]competent|reactive)\b",
        r"beats? (the )?(wt|wild[- ]?type)",
        r"(productive|reactive)\b.{0,40}\b(than|over|vs\.?)\s+(the\s+)?(wt|wild[- ]?type)",
        r"활성\s*(이|을|의)?\s*(증가|향상|개선)",
        r"촉매\s*(효율|능력|활성)\s*(이|을)?\s*(증가|향상|개선)",
    ],
    KINETIC_PARAMETER_PREDICTION: [
        r"predicts? (the )?k_?cat",
        r"k_?cat\s*/\s*k_?m\s*(is |was )?(improved|predicted)",
        r"(improved|increased|higher) k_?cat",
        r"k_?cat (is |was )?(improved|increased)",
        r"predicts? catalytic efficiency",
        r"k_\{?cat\}?",                       # LaTeX k_{cat}
        r"kcat/km (improved|predicted)",
    ],
    SHORT_MD_OVERCLAIM: [
        r"short md (validates|confirms)",
        r"md (validates|confirms) (a )?functional (catalytic )?(state|complex)",
        r"confirms a functional catalytic state",
    ],
    LONG_TERM_STABILITY: [
        r"long-?term (ligand )?stability",
        r"stable functional complex",
        r"functional pose preservation",
    ],
    INACTIVE_CLASSIFICATION: [
        r"inactive (mutant|variant)",
        r"(mutant|variant) (is|are) inactive",
        r"비활성\s*(변이|돌연변이)",
    ],
    # V5-5: catalysis-implying / unqualified-strength LABEL words the pipeline must never
    # print without wet-lab activity. Bare "strong candidate" is forbidden but "strong
    # STRUCTURAL/BINDING candidate" passes (the qualifier sits BETWEEN the words, so the
    # \bstrong\s+candidate\b anchor never matches the qualified form — no lookbehind needed).
    # "catalytic lead" -> "top-ranked lead"; "paper-grade" -> "anchored-evaluated". This
    # category is NEVER unlocked in `evaluate` (it is phrasing, not an activity claim), so it
    # is always forbidden — the fix is to use the qualified alternative, not earn evidence.
    UNQUALIFIED_STRENGTH: [
        r"\bstrong\s+candidate\b",
        r"\bcatalytic\s+lead\b",
        r"paper[-\s]grade",
        r"\bconfirmed\s+productive\b",
    ],
    # V6 roadmap §13 forbidden-before-wet-lab claims the linter previously missed:
    # "catalytically validated", "validated lead", "experimentally active".
    CATALYTIC_VALIDATION: [
        r"catalytically validated",
        r"\bvalidated\s+(lead|hit|candidate|variant|mutant)\b",
        r"experimentally (active|validated|confirmed)",
        r"검증된\s*(리드|후보|변이)",
        r"실험적으로\s*(활성|검증)",
    ],
    # "reduced activation barrier" and paraphrases. The roadmap allows "near-attack
    # access COST" but NOT any activation-barrier lowering claim before wet-lab.
    ACTIVATION_BARRIER: [
        r"(reduce[sd]?|lower(s|ed)?|decrease[sd]?) (the )?activation (barrier|energy)",
        r"activation (barrier|energy) (is |was |being )?(reduced|lowered|decreased)",
        r"lower(s|ed)? (the )?(reaction )?barrier",
        r"활성화\s*(에너지|장벽)\s*(을|이|가)?\s*(낮|감소|저하)",
    ],
}

# negation cues. English cues must precede the claim in the sentence; Korean negation is
# post-verbal so a cue ANYWHERE in the sentence counts.
_EN_NEG = re.compile(
    r"\b(do(es)?\s+not|don't|doesn't|did\s+not|didn't|cannot|can't|"
    r"is\s+not|are\s+not|isn't|aren't|will\s+not|won't|never|no\s+(direct\s+)?)\b")
_KO_NEG = re.compile(r"(않|못\s|없|아니)")

_SENT_SPLIT = re.compile(r"[.!?\n;]+|(?<=다)\s+")


@dataclass
class ClaimViolation:
    category: str
    matched: str
    sentence: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.matched!r} in: {self.sentence.strip()[:120]!r}"


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def _is_negated(sentence_lc: str, match_start: int, sentence_raw: str) -> bool:
    # Korean: any negation cue in the sentence
    if _KO_NEG.search(sentence_raw):
        return True
    # English: a negation cue somewhere before the matched claim
    before = sentence_lc[:match_start]
    return _EN_NEG.search(before) is not None


BENCHMARK_CATEGORY = "benchmark_prohibited"


def lint_text(text: str, allow: Optional[Sequence[str]] = None,
              extra_forbidden: Optional[Sequence[str]] = None) -> List[ClaimViolation]:
    """Find prohibited over-claims in ``text``.

    ``allow`` = claim categories that ARE permitted (e.g. unlocked by replicated
    wet-lab evidence). Default = none allowed (the conservative current state: no
    wet-lab). ``extra_forbidden`` = literal benchmark-specific phrases (e.g. a
    fitness-labelled benchmark forbids "direct kcat prediction"); these are matched as
    substrings and are never in ``allow``. Negation-safe: a negated claim never counts.
    """
    allowed: Set[str] = set(allow or ())
    extra = [re.escape(p.lower()) for p in (extra_forbidden or [])]
    violations: List[ClaimViolation] = []
    for sent in _sentences(text):
        sent_lc = sent.lower()
        for category, patterns in _PATTERNS.items():
            if category in allowed:
                continue
            for pat in patterns:
                for m in re.finditer(pat, sent_lc):
                    if _is_negated(sent_lc, m.start(), sent):
                        continue
                    violations.append(ClaimViolation(category, m.group(0), sent))
                    break  # one violation per (category, sentence) is enough
        for pat in extra:                       # benchmark-specific phrases (never allowed)
            for m in re.finditer(pat, sent_lc):
                if _is_negated(sent_lc, m.start(), sent):
                    continue
                violations.append(ClaimViolation(BENCHMARK_CATEGORY, m.group(0), sent))
                break
    return violations


def assert_clean(text: str, allow: Optional[Sequence[str]] = None) -> None:
    """Raise if ``text`` contains a prohibited over-claim. Reports/tests call this so a
    forbidden phrase is a hard failure, never a silent leak."""
    v = lint_text(text, allow=allow)
    if v:
        raise AssertionError(
            "ClaimGuard: prohibited claim(s) in report:\n  " + "\n  ".join(str(x) for x in v))


# =====================================================================================
# V3-4 — the full provenance-driven engine (claim-category + template whitelist +
# schema-validated FAIL-SAFE). Built on top of the linter above.
# =====================================================================================
from pydantic import BaseModel, ValidationError  # noqa: E402

from evoliez.mechanism.vocab import (  # noqa: E402
    HYPOTHESIS_GRADE, STRONG_SCREENING, UNCALIBRATED, weakest_claim,
)


class ClaimProvenance(BaseModel):
    """Strict provenance the guard reasons over. EVERY field has a CONSERVATIVE default,
    so an absent field never unlocks a claim. ``extra='forbid'`` makes a typo'd key raise
    — which ``evaluate`` catches and converts to the most-conservative verdict (fail-safe:
    a guard must never fail OPEN)."""
    model_config = {"extra": "forbid"}

    reference_claim_strength: str = UNCALIBRATED
    geometry_claim_ceiling: str = UNCALIBRATED
    ensemble_claim_ceiling: Optional[str] = None
    wetlab_replicated: bool = False
    only_short_md: bool = True               # conservative: assume short MD unless told otherwise
    enhanced_sampling_or_qmmm: bool = False
    known_active_controls: bool = False
    known_inactive_controls: bool = False
    de_novo_pose_disagreement_high: bool = False
    wt_reaction_geometry_sparse: bool = False
    ligand_parameterization_uncertain: bool = False
    ml_label_source: str = "computational_surrogate"


@dataclass
class ClaimVerdict:
    claim_ceiling: str
    allowed_categories: Set[str]              # claim categories the report MAY assert
    flags: Set[str]                           # uncalibrated / diagnostic_only_geometry / ...
    fail_safe: bool = False                   # True if provenance was malformed -> defaulted

    def allow(self) -> List[str]:
        return sorted(self.allowed_categories)


# the most conservative verdict — the fail-safe floor
def _floor_verdict(fail_safe: bool = False) -> ClaimVerdict:
    return ClaimVerdict(claim_ceiling=UNCALIBRATED, allowed_categories=set(),
                        flags={"uncalibrated"}, fail_safe=fail_safe)


def evaluate(provenance) -> ClaimVerdict:
    """Decide the claim ceiling + which categories are unlocked, from run provenance.
    ``provenance`` may be a ``ClaimProvenance``, a dict, or None. Malformed/missing
    provenance fails SAFE to the floor verdict (never raises, never fails open)."""
    if provenance is None:
        return _floor_verdict(fail_safe=True)
    if not isinstance(provenance, ClaimProvenance):
        try:
            provenance = ClaimProvenance(**dict(provenance))
        except (ValidationError, TypeError, ValueError):
            return _floor_verdict(fail_safe=True)

    p = provenance
    # 1) claim ceiling = the WEAKEST of every contributing ceiling (fail-safe combinator)
    ceilings = [p.reference_claim_strength, p.geometry_claim_ceiling]
    if p.ensemble_claim_ceiling:
        ceilings.append(p.ensemble_claim_ceiling)
    ceiling = weakest_claim(*ceilings)

    flags: Set[str] = set()
    if not (p.known_active_controls or p.known_inactive_controls):
        ceiling = weakest_claim(ceiling, UNCALIBRATED)
        flags.add("uncalibrated")
    if p.wt_reaction_geometry_sparse:
        flags.add("diagnostic_only_geometry")
    if p.ligand_parameterization_uncertain:
        flags.add("md_confidence_downgrade")
    if p.de_novo_pose_disagreement_high:
        flags.add("pose_uncertainty_high")

    # 2) unlock claim categories ONLY when the evidence supports them
    allowed: Set[str] = set()
    if p.wetlab_replicated:
        allowed.add(ACTIVITY_IMPROVEMENT)
        allowed.add(KINETIC_PARAMETER_PREDICTION)
        allowed.add(INACTIVE_CLASSIFICATION)
        allowed.add(CATALYTIC_VALIDATION)   # "validated lead" is true ONLY with wet-lab
        allowed.add(ACTIVATION_BARRIER)     # barrier-lowering needs experimental kinetics
    # short-MD / long-term-stability claims need sampling beyond short MD
    if p.enhanced_sampling_or_qmmm or not p.only_short_md:
        allowed.add(SHORT_MD_OVERCLAIM)
        allowed.add(LONG_TERM_STABILITY)
    # a high de-novo pose disagreement forbids the "inactive" call regardless
    if p.de_novo_pose_disagreement_high:
        allowed.discard(INACTIVE_CLASSIFICATION)
    return ClaimVerdict(claim_ceiling=ceiling, allowed_categories=allowed, flags=flags)


# --- allowed-template whitelist (reports assemble ONLY from these per category) -------
ALLOWED_TEMPLATES = {
    ACTIVITY_IMPROVEMENT: {
        False: ["This variant is prioritized for experimental testing.",
                "This variant carries screening-level evidence for structural and "
                "ligand/cofactor competence."],
        True: ["This variant enriched experimentally active variants over the "
               "stability-only baseline."],
    },
    KINETIC_PARAMETER_PREDICTION: {
        False: ["The workflow does not directly predict kcat or kcat/KM."],
        True: ["Replicated assays quantified the kinetic parameters."],
    },
    SHORT_MD_OVERCLAIM: {
        False: ["Short MD supports local reference-pose accommodation."],
        True: ["Extended sampling supports retention of the functional state."],
    },
}


def allowed_templates(category: str, verdict: ClaimVerdict) -> List[str]:
    """The sentences a report MAY use for a category, given the verdict."""
    unlocked = category in verdict.allowed_categories
    return ALLOWED_TEMPLATES.get(category, {}).get(unlocked, [])


def lint_report(text: str, provenance) -> List[ClaimViolation]:
    """Lint a rendered report against its provenance (the V3-4 entry point): unlock only
    the categories the evidence supports, then run the negation-safe linter."""
    verdict = evaluate(provenance)
    return lint_text(text, allow=verdict.allow())


def assert_report_clean(text: str, provenance) -> None:
    v = lint_report(text, provenance)
    if v:
        raise AssertionError(
            "ClaimGuard: prohibited claim(s) for this provenance:\n  "
            + "\n  ".join(str(x) for x in v))
