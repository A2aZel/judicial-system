import csv
import json
from pathlib import Path

import pytest

from judicial_ai.corpus.csv_repository import CsvCorpusRepository
from judicial_ai.corpus.processed_repository import ProcessedRepository
from judicial_ai.corpus.records import CaseRecord

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


def principal_outcome(record: CaseRecord) -> tuple[str | None, float | None, bool | None]:
    return record.principal_punishment, record.principal_value, record.principal_conditional


def test_synthetic_csv_declares_prison_and_keeps_the_additional_fine() -> None:
    record = CsvCorpusRepository(FIXTURE_PATH).records[2]
    assert principal_outcome(record) == ("imprisonment", 36, False)
    assert record.fine_rub == 300000


@pytest.mark.parametrize(
    ("extra", "expected"),
    [("", (None, None, None)), (",fine,500000,", ("fine", 500000, None))],
)
def test_csv_fine_requires_an_explicit_principal_label(
    tmp_path: Path, extra: str, expected: tuple[str | None, float | None, bool | None]
) -> None:
    header = (
        "convicted_290,parts_290,special_order,pretrial_agreement,art_64,art_69,art_73,"
        "prison_months,fine_rub,fine_multiple,ban_on_positions,confiscation"
    )
    if extra:
        header += ",principal_punishment,principal_value,principal_conditional"
    path = tmp_path / "cases.csv"
    path.write_text(header + "\nTrue,3,False,False,False,False,False,,500000,True,False,False" + extra, encoding="utf-8")
    record, = CsvCorpusRepository(path).records
    assert principal_outcome(record) == expected


CONVICTION = (
    "Признать подсудимого виновным в совершении преступления, "
    "предусмотренного ч. 3 ст. 290 УК РФ, и назначить ему наказание "
)


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        (
            "в виде лишения свободы сроком на 2 (два) года 6 месяцев с отбыванием "
            "наказания в исправительной колонии и со штрафом в размере 250 000 рублей.",
            ("imprisonment", 30, False),
        ),
        (
            "в виде лишения свободы на 8 месяцев. На основании ст. 73 УК РФ "
            "назначенное наказание считать условным с испытательным сроком 2 года.",
            ("imprisonment", 8, True),
        ),
        ("в виде лишения свободы сроком на 2 года.", ("imprisonment", 24, None)),
        (
            "в виде штрафа в размере 500 000 (пятьсот тысяч) рублей "
            "с лишением права занимать должности на 3 года.",
            ("fine", 500000, None),
        ),
        (
            "в виде штрафа в размере тридцатикратной суммы взятки, то есть 750 000 рублей.",
            ("fine", 750000, None),
        ),
        ("в виде лишения свободы на <данные изъяты> со штрафом 250 000 рублей.", (None, None, None)),
        ("в виде штрафа в размере тридцатикратной суммы взятки.", (None, None, None)),
        (
            "в виде штрафа в размере 500 000 рублей. На основании ст. 69 УК РФ "
            "по совокупности преступлений окончательно назначить 3 года лишения свободы.",
            (None, None, None),
        ),
        (
            "в виде штрафа в размере 500 000 рублей. " + CONVICTION + "в виде штрафа 500 000 рублей.",
            (None, None, None),
        ),
    ],
)
def test_processed_extracts_only_unambiguous_principal_outcome_from_disposition(
    tmp_path: Path, sentence: str, expected: tuple[str | None, float | None, bool | None]
) -> None:
    assert principal_outcome(load_processed(tmp_path, "П Р И Г О В О Р И Л:\n" + CONVICTION + sentence)) == expected


@pytest.mark.parametrize(
    "narrative",
    [
        CONVICTION + "в виде штрафа в размере 500 000 рублей.",
        "ПРИГОВОРИЛ: " + CONVICTION.replace("ч. 3", "ч. 5") + "в виде штрафа 500 000 рублей.",
        "ПРИГОВОРИЛ: " + CONVICTION.replace("подсудимого виновным", "подсудимых виновными")
        + "каждому в виде штрафа 500 000 рублей.",
        "ПРИГОВОРИЛ: " + CONVICTION.replace("преступления,", "преступлений,")
        + "в виде штрафа 500 000 рублей.",
        "ПРИГОВОРИЛ: " + CONVICTION + "в виде штрафа 500 000 рублей. "
        "Признать второго подсудимого невиновным и оправдать.",
    ],
)
def test_processed_abstains_if_the_verdict_does_not_confirm_a_single_matching_conviction(
    tmp_path: Path, narrative: str
) -> None:
    assert principal_outcome(load_processed(tmp_path, narrative)) == (None, None, None)


def test_processed_respects_other_articles_and_aggregate_flag(tmp_path: Path) -> None:
    narrative = "ПРИГОВОРИЛ: " + CONVICTION + "в виде штрафа 500 000 рублей."
    assert principal_outcome(load_processed(tmp_path, narrative, other_articles=["285"])) == (None, None, None)
    assert principal_outcome(load_processed(tmp_path, narrative, process={"art_69": True})) == (None, None, None)


def test_processed_uses_explicit_principal_punishment_without_narrative(tmp_path: Path) -> None:
    record = load_processed(
        tmp_path, None,
        punishment={"principal_punishment": "fine", "principal_value": 500000, "prison_months": 7},
    )
    assert principal_outcome(record) == ("fine", 500000, None)
    assert record.prison_months == 7


@pytest.mark.parametrize(
    "overrides",
    [
        {"article_parts": ["3", "5"]},
        {"article_parts": ["7"]},
        {"article_parts": ["12"]},
        {"other_articles": ["285"]},
        {"process": {"art_69": True}},
        {"convicted_290": False},
    ],
    ids=["multiple-parts", "invalid-part", "multi-digit-part", "other-article", "aggregate", "no-conviction"],
)
def test_processed_retains_records_but_ignores_ineligible_explicit_labels(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    record = load_processed(
        tmp_path, None,
        punishment={"principal_punishment": "fine", "principal_value": 500000, "prison_months": 7},
        **overrides,
    )
    assert principal_outcome(record) == (None, None, None)
    assert record.prison_months == 7


def test_csv_retains_records_but_ignores_an_explicit_aggregate_label(tmp_path: Path) -> None:
    repository = load_with_value(tmp_path, "art_69", "True")
    assert principal_outcome(repository.records[0]) == (None, None, None)
    assert repository.records[0].prison_months == 24


def test_processed_preserves_old_features_and_requires_explicit_recommendation_trust(tmp_path: Path) -> None:
    legacy = load_processed(tmp_path, None, facts={"amount_rub": 200000, "episodes": 1})
    trusted = load_processed(
        tmp_path, None, facts={"amount_rub": 200000, "episodes": 1, "recommendation_facts_trusted": True}
    )
    assert legacy.amount_rub == trusted.amount_rub == 200000
    assert legacy.episodes == trusted.episodes == 1
    assert legacy.recommendation_facts_trusted is False
    assert trusted.recommendation_facts_trusted is True


def test_processed_hash_matches_an_identical_body_with_a_different_signature(tmp_path: Path) -> None:
    body = "Суд решил уголовное дело.\nПРИГОВОРИЛ: " + CONVICTION + "в виде штрафа 500 000 рублей."
    first = load_processed(tmp_path, body + "\nСудья А.", verdict_group_id="first", source="synthetic")
    second = load_processed(
        tmp_path, body + "\nСудья А.Б.Примеров", verdict_group_id="second", source="synthetic"
    )
    third = load_processed(tmp_path, body.replace("500 000", "600 000"), source="synthetic")
    fourth = load_processed(tmp_path, body + "\nСудья А.", source="other")
    assert first.recommendation_group == "first"
    assert second.recommendation_group == "second"
    assert first.recommendation_text_hash is not None
    assert first.recommendation_text_hash == second.recommendation_text_hash
    assert first.recommendation_text_hash != third.recommendation_text_hash
    assert first.recommendation_text_hash != fourth.recommendation_text_hash


def load_processed(tmp_path: Path, narrative: str | None, **overrides: object) -> CaseRecord:
    payload = {
        "schema_version": 1,
        "case_id": "synthetic",
        "article_parts": ["3"],
        "process": {"art_73": False},
        "facts": {"narrative": narrative},
        "punishment": {"fine_rub": 500000, "conditional": False},
    }
    payload.update(overrides)
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    record, = ProcessedRepository(path).records
    return record
