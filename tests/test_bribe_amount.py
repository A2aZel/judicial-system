import json
from pathlib import Path

import pytest

from judicial_ai.corpus.processed_repository import ProcessedRepository
from judicial_ai.corpus.records import CaseRecord


def load_verdict(
    tmp_path: Path, findings: str, *, trusted: bool = False, legacy_amount: float = 900_000
) -> CaseRecord:
    narrative = (
        f"У С Т А Н О В И Л:\n{findings}\nП Р И Г О В О Р И Л:\n"
        "Признать подсудимого виновным в совершении преступления, "
        "предусмотренного ч. 3 ст. 290 УК РФ, и назначить ему наказание "
        "в виде штрафа в размере 900 000 рублей. "
        "Конфисковать предмет взятки в размере 80 000 рублей."
    )
    row = {
        "schema_version": 1,
        "case_id": "synthetic",
        "article_parts": ["3"],
        "process": {},
        "punishment": {},
        "facts": {
            "narrative": narrative,
            "amount_rub": legacy_amount,
            "recommendation_facts_trusted": trusted,
        },
    }
    path = tmp_path / "retrieval.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    record, = ProcessedRepository(path).records
    return record


@pytest.mark.parametrize(
    ("amount", "expected"),
    [("150 000", 150_000), ("1 500 000", 1_500_000), ("15000000", 15_000_000)],
)
def test_processed_scans_only_court_findings_for_the_bribe_amount(
    tmp_path: Path, amount: str, expected: float
) -> None:
    record = load_verdict(
        tmp_path,
        f"Подсудимый получил взятку в размере {amount} рублей. "
        "Суд обсудил назначение штрафа в размере 900 000 рублей.",
    )
    assert record.amount_rub == 900_000
    assert record.recommendation_amount_rub == expected
    assert record.recommendation_facts_trusted is False


@pytest.mark.parametrize("finding", ["Судом установлено", "В судебном заседании установлено"])
def test_processed_accepts_an_explicitly_established_amount_before_the_bribe_label(
    tmp_path: Path, finding: str
) -> None:
    record = load_verdict(
        tmp_path,
        f"Подсудимый вину не признал. {finding}, что подсудимый получил "
        "денежные средства в сумме 120 000 (сто двадцать тысяч) рублей в качестве взятки.",
    )
    assert record.recommendation_amount_rub == 120_000


@pytest.mark.parametrize(
    "findings",
    [
        "Подсудимый получил взятку в размере <сумма> рублей.",
        "Подсудимый получил взятку в размере рублей.",
        "Подсудимый получил взятку в размере 1 50 000 рублей.",
        "Подсудимый получил взятку в размере 50 000 или 60 000 рублей.",
        "Подсудимый не получил взятку в размере 50 000 рублей.",
        "Подсудимый отрицал, что получил взятку в размере 50 000 рублей.",
        "По версии следствия подсудимый получил взятку в размере 50 000 рублей.",
        "Согласно обвинительному заключению, подсудимый получил взятку в размере 50 000 рублей.",
        "Подсудимый потребовал взятку в размере 50 000 рублей.",
        "Подсудимый получил зарплату и предложил взятку в размере 50 000 рублей.",
        "Прокурор просил считать, что подсудимый получил взятку в размере 50 000 рублей.",
        "Свидетель показал, что подсудимый получил взятку в размере 50 000 рублей.",
        "Суд исключил из обвинения утверждение, что подсудимый получил взятку в размере 50 000 рублей.",
        "Подсудимый получил взятку в размере 50 000 рублей. Суд оправдал подсудимого в этой части обвинения.",
        "Согласно разъяснению Пленума ВС РФ, подсудимый получил взятку в размере 50 000 рублей.",
        "Подсудимый получил взятку в размере 50 000 рублей. "
        "Кроме того, он получил взятку в размере 60 000 рублей.",
        "Подсудимый получил взятку в размере 50 000 рублей. "
        "В другом эпизоде он получил взятку в размере 50 000 рублей.",
        "Подсудимый получил взятку в размере 50 000 рублей, а затем еще 60 000 рублей.",
        "Подсудимый получил денежные средства в сумме 50 000 рублей в качестве взятки или займа.",
        "Подсудимый потребовал взятку в виде денег в размере рублей. "
        "Подсудимый получил взятку в виде денег в размере 3 рублей, что является значительным размером.",
        "Подсудимый получил взятку в размере 50 000 рублей. "
        "Он получил денежные средства в сумме <сумма> рублей в качестве взятки.",
    ],
)
def test_processed_preserves_unknown_amount_when_findings_do_not_confirm_one_sum(
    tmp_path: Path, findings: str
) -> None:
    assert load_verdict(tmp_path, findings).recommendation_amount_rub is None


def test_processed_honors_a_trusted_explicit_amount(tmp_path: Path) -> None:
    record = load_verdict(tmp_path, "Размер взятки обезличен.", trusted=True, legacy_amount=150_000)
    assert record.recommendation_amount_rub == 150_000


def test_processed_accepts_a_court_restatement_of_the_same_fact(tmp_path: Path) -> None:
    record = load_verdict(
        tmp_path,
        "Подсудимый получил взятку в размере 50 000 рублей. Подсудимый вину не признал. "
        "Судом установлено, что подсудимый получил взятку в размере 50 000 рублей.",
    )
    assert record.recommendation_amount_rub == 50_000
