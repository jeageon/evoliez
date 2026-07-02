import csv

from evoliez.experimental.library_plan import CSV_FIELDS, build_first_round_library, write_library_csv
from evoliez.experimental.seed_manifest import load_seed_manifest


def test_library_csv_schema(tmp_path):
    rows = build_first_round_library(load_seed_manifest(), size=20)
    path = write_library_csv(rows, tmp_path / "library.csv")
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == CSV_FIELDS
        first = next(reader)
    assert first["mutation"] == "WT"
    assert first["well"] == "A1"
    assert first["codon_design_note"]
