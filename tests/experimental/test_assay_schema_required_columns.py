import pytest

from evoliez.experimental.assay_schema import REQUIRED_ASSAY_COLUMNS, validate_assay_columns


def test_assay_schema_required_columns():
    validate_assay_columns(list(REQUIRED_ASSAY_COLUMNS))
    with pytest.raises(ValueError):
        validate_assay_columns(["variant_id", "mutation"])
