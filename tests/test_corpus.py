import csv
from pathlib import Path

import pytest

from judicial_ai.corpus.csv_repository import CsvCorpusRepository

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "synthetic_analysis.csv"


def load_with_value(tmp_path: Path, field: str, value: str) -> CsvCorpusRepository:
    with FIXTURE_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0][field] = value
    path = tmp_path / "invalid.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return CsvCorpusRepository(path)


@pytest.mark.parametrize(
    "field",
    ["special_order", "pretrial_agreement", "art_64", "art_69"],
)
@pytest.mark.parametrize("value", ["", "unknown", "null"])
def test_unknown_required_modes_are_not_false(
    tmp_path: Path, field: str, value: str
) -> None:
    repository = load_with_value(tmp_path, field, value)
    assert len(repository.records) == 9
    assert repository.skipped_records == 1


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "1e10000", "-1"])
def test_non_finite_or_negative_amount_is_rejected(
    tmp_path: Path, value: str
) -> None:
    repository = load_with_value(tmp_path, "amount_rub", value)
    assert len(repository.records) == 9
    assert repository.skipped_records == 1


@pytest.mark.parametrize("field", ["episodes", "mitigating_count"])
@pytest.mark.parametrize("value", ["inf", "1.5", "-1"])
def test_invalid_counters_are_rejected_without_overflow(
    tmp_path: Path, field: str, value: str
) -> None:
    repository = load_with_value(tmp_path, field, value)
    assert len(repository.records) == 9
    assert repository.skipped_records == 1


def test_out_of_range_and_unknown_optional_values_are_rejected(tmp_path: Path) -> None:
    for field, value in (("episodes", "1001"), ("fine_multiple", "maybe")):
        repository = load_with_value(tmp_path, field, value)
        assert len(repository.records) == 9
        assert repository.skipped_records == 1
