"""Enzyme-agnostic assay label schema + provenance guard (ML strategy §C)."""

from pathlib import Path

import pytest

from evoliez.ml.assay_label import (
    COMPUTED_WEAK,
    WETLAB,
    AssayLabel,
    LabelProvenanceError,
    assert_supervised_allowed,
    load_assay_labels,
    supervised_targets,
)

ROOT = Path(__file__).resolve().parents[1]


def _wet(mut, lt="activity", val=1.0, readout="rate", source=WETLAB):
    return AssayLabel(mutation=mut, label_type=lt, value=val, readout_name=readout, source=source)


def test_computed_weak_cannot_be_supervised_target():
    labs = [_wet("A1B"), AssayLabel(mutation="C2D", label_type="activity", value=0.5,
                                    source=COMPUTED_WEAK)]
    with pytest.raises(LabelProvenanceError):
        assert_supervised_allowed(labs)


def test_supervised_targets_filters_source_and_kind():
    labs = [
        _wet("A1B", "activity", 2.0, "kcatKM"),
        _wet("C2D", "expression", 0.9, "sol"),
        AssayLabel(mutation="E3F", label_type="activity", value=9.0, source=COMPUTED_WEAK),
    ]
    t = supervised_targets(labs, label_type="activity")
    assert t == {"A1B": 2.0}  # expression excluded (kind), computed_weak excluded (source)


def test_invalid_direction_and_source_rejected():
    with pytest.raises(ValueError):
        AssayLabel(mutation="A1B", label_type="activity", value=1.0, direction="bigger")
    with pytest.raises(ValueError):
        AssayLabel(mutation="A1B", label_type="activity", value=1.0, source="guess")


def test_is_mechanism_agnostic_no_fdh_tokens():
    text = (ROOT / "src/evoliez/ml/assay_label.py").read_text().lower()
    for token in ("nadp", "formate", "fdh", "nicotinamide"):
        assert token not in text


def test_load_long_format_csv(tmp_path):
    csv = tmp_path / "assay.csv"
    csv.write_text(
        "mutation,label_type,value,readout_name,source,direction\n"
        "I208T;R207K;R228P,activity,2.4,kcatKM,wetlab,maximize\n"
        "Q382R,expression,0.8,soluble,wetlab,maximize\n"
    )
    labs = load_assay_labels(csv)
    assert len(labs) == 2
    assert supervised_targets(labs, label_type="activity") == {"I208T;R207K;R228P": 2.4}
