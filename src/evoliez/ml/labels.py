"""Data-role policy (the central principle, enforced in code).

> Boltz score is NOT an experimental label. Use Boltz outputs as features /
> sample weights / weak labels / filtering criteria. Supervised labels must
> come from experiment (activity, kcat, Km, kcat/Km, thermostability,
> substrate specificity, expression).

`assert_supervised_label_allowed` is called wherever a column is about to be
used as the training target `y`, so a Boltz-derived column can never silently
become a label.
"""

from __future__ import annotations

from typing import Iterable

# Columns that may ONLY be experimental supervised targets.
SUPERVISED_LABEL_COLUMNS = {
    "activity", "relative_activity", "kcat", "km", "kcat_km", "kcat/km",
    "thermostability", "tm", "substrate_specificity", "expression",
    "solubility",
}

# Boltz-derived names that must NEVER be used as a supervised label.
BOLTZ_DERIVED_PREFIXES = (
    "confidence", "ptm", "iptm", "ligand_iptm", "complex_plddt",
    "complex_iplddt", "complex_pde", "complex_ipde", "affinity_pred",
    "affinity_probability", "pae", "pde", "plddt", "contact_frequency",
    "ensemble_disagreement", "d_",  # delta features
)

# Role each kind of column may play.
COLUMN_ROLES = {
    "boltz_confidence": {"feature", "sample_weight", "weak_label", "filter"},
    "boltz_affinity": {"feature", "weak_label", "filter"},
    "ensemble_contact_frequency": {"feature", "weak_label"},
    "delta": {"feature"},
    "msa_evolutionary": {"feature", "weak_label"},
    "experimental": {"supervised_label"},
}


class LabelPolicyError(RuntimeError):
    pass


def is_boltz_derived(column: str) -> bool:
    c = column.lower()
    return any(c.startswith(p) for p in BOLTZ_DERIVED_PREFIXES)


def assert_supervised_label_allowed(column: str) -> None:
    """Guard: raise if a Boltz-derived column is used as the training target."""
    c = column.lower().strip()
    if is_boltz_derived(c) and c not in SUPERVISED_LABEL_COLUMNS:
        raise LabelPolicyError(
            f"'{column}' is Boltz-derived and must NOT be a supervised label. "
            f"Use it as a feature / sample weight / weak label / filter. "
            f"Supervised labels must be experimental "
            f"({sorted(SUPERVISED_LABEL_COLUMNS)})."
        )


def filter_label_columns(columns: Iterable[str]) -> list[str]:
    """Keep only experiment-derived columns as candidate supervised labels."""
    out = []
    for c in columns:
        cl = c.lower().strip()
        if cl in SUPERVISED_LABEL_COLUMNS and not is_boltz_derived(cl):
            out.append(c)
    return out
