import csv
import json
from pathlib import Path

import pytest

from judicial_ai.corpus.csv_repository import CsvCorpusRepository
from judicial_ai.corpus.processed_repository import ProcessedRepository


def load_verdict(tmp_path: Path, reasoning: str, *, facts=None, **record_overrides):
    verdict = {
        "schema_version": 1,
        "case_id": "synthetic",
        "article_parts": ["3"],
        "process": {},
        "punishment": {},
        "facts": {
            "narrative": (
                f"УСТАНОВИЛ:\n{reasoning}\nП Р И Г О В О Р И Л:\n"
                "Признать подсудимого виновным в совершении преступления, "
                "предусмотренного ч. 3 ст. 290 УК РФ, и назначить ему наказание "
                "в виде штрафа в размере 900 000 рублей."
            ),
            **(facts or {}),
        },
        **record_overrides,
    }
    path = tmp_path / "retrieval.jsonl"
    path.write_text(json.dumps(verdict, ensure_ascii=False) + "\n", encoding="utf-8")
    record, = ProcessedRepository(path).records
    return record


@pytest.mark.parametrize(
    ("reasoning", "mitigating", "aggravating"),
    [
        (
            "В качестве обстоятельств, смягчающих наказание, суд признаёт явку с повинной и раскаяние.",
            {"явка с повинной", "раскаяние"},
            None,
        ),
        (
            "Суд в качестве обстоятельства, отягчающего наказание, учитывает совершение преступления группой лиц.",
            None,
            {"группа лиц"},
        ),
        (
            "Обстоятельством, отягчающим наказание, суд признает рецидив преступлений.",
            None,
            {"рецидив преступлений"},
        ),
        (
            "К обстоятельствам, смягчающим наказание, суд относит содействие расследованию и "
            "наличие малолетнего ребёнка, а обстоятельств, отягчающих наказание, судом не установлено.",
            {"содействие расследованию", "наличие малолетних детей"},
            set(),
        ),
        (
            "Смягчающими обстоятельствами суд признает состояние здоровья подсудимого "
            "и возмещение причинённого вреда. Отягчающих обстоятельств не установлено.",
            {"состояние здоровья", "возмещение вреда"},
            set(),
        ),
        (
            "Отягчающими обстоятельствами суд признает особо активную роль подсудимого "
            "и причинение тяжких последствий.",
            None,
            {"особо активная роль", "причинение тяжких последствий"},
        ),
    ],
)
def test_extracts_only_specific_court_findings(tmp_path, reasoning, mitigating, aggravating):
    record = load_verdict(tmp_path, reasoning)
    assert record.recommendation_mitigating == mitigating
    assert record.recommendation_aggravating == aggravating


@pytest.mark.parametrize(
    "reasoning",
    [
        "Подсудимый ранее судим. Прокурор просил признать рецидив преступлений.",
        "Защитник считает, что обстоятельством, отягчающим наказание, суд признает рецидив преступлений.",
        "В обвинительном заключении указано: «Обстоятельством, отягчающим наказание, "
        "суд признает рецидив преступлений».",
        "Если бы преступление было совершено группой лиц, отягчающим обстоятельством суд признает группу лиц.",
        "Обстоятельством, отягчающим наказание, суд не признает рецидив преступлений.",
        "Обстоятельством, смягчающим наказание, суд признает отсутствие тяжких последствий.",
        "Обстоятельством, смягчающим наказание, суд признает признание вины.",
        "Смягчающими обстоятельствами суд признает состояние здоровья родственников подсудимого.",
        "Смягчающими обстоятельствами суд признает состояние здоровья брата подсудимого.",
        "Смягчающими обстоятельствами суд признает состояние здоровья ФИО2, брата подсудимого.",
        "Смягчающими обстоятельствами суд признает состояние здоровья «ФИО2», брата подсудимого.",
        "В качестве обстоятельства, отягчающего наказание, суд признает предполагаемый рецидив преступлений.",
        "В качестве обстоятельств, отягчающих наказание, суд признаёт перечисленные обстоятельства, "
        "кроме рецидива преступлений.",
        "Обстоятельством, отягчающим наказание, суд признает ходатайство о признании "
        "рецидива преступлений необоснованным.",
        "Обстоятельством, отягчающим наказание, суд признает рецидив преступлений недоказанным.",
        "Как следует из ранее вынесенного приговора: обстоятельством, отягчающим наказание, "
        "суд признал рецидив преступлений.",
        "Сторона защиты оспорила довод о том, что обстоятельством, отягчающим наказание, "
        "суд признает рецидив преступлений.",
        "Отягчающим обстоятельством суд признает рецидив преступлений. "
        "Вывод о рецидиве преступлений подлежит исключению.",
        "Отягчающим обстоятельством суд признает рецидив преступлений. "
        "Отягчающих обстоятельств не установлено.",
    ],
)
def test_keeps_unverified_factors_unknown(tmp_path, reasoning):
    record = load_verdict(
        tmp_path,
        reasoning,
        facts={"mitigating": ["confession"], "aggravating": ["recidivism"]},
    )
    assert record.recommendation_mitigating is None
    assert record.recommendation_aggravating is None


def test_ignores_disposition_and_mixed_offences(tmp_path):
    marker_in_disposition = load_verdict(
        tmp_path,
        "",
        facts={"narrative": (
            "ПРИГОВОРИЛ: Признать подсудимого виновным в совершении преступления, "
            "предусмотренного ч. 3 ст. 290 УК РФ, и назначить ему наказание "
            "в виде штрафа в размере 900 000 рублей. "
            "Обстоятельством, отягчающим наказание, суд признает рецидив преступлений."
        )},
    )
    assert marker_in_disposition.principal_value == 900_000
    assert marker_in_disposition.recommendation_aggravating is None
    mixed = load_verdict(
        tmp_path,
        "Обстоятельством, отягчающим наказание, суд признает рецидив преступлений.",
        other_articles=["291"],
    )
    assert mixed.recommendation_aggravating is None


def test_extracts_only_the_recognized_list_item(tmp_path):
    record = load_verdict(
        tmp_path,
        "Обстоятельством, отягчающим наказание, суд признает рецидив преступлений, "
        "вопрос о совершении преступления группой лиц исследован отдельно.",
    )
    assert record.recommendation_aggravating == {"рецидив преступлений"}


def test_typed_facets_use_trusted_canonical_fields(tmp_path):
    record = load_verdict(
        tmp_path,
        "Обстоятельством, отягчающим наказание, суд признает рецидив преступлений.",
        facts={
            "recommendation_facts_trusted": True,
            "mitigating": ["  ЯВКА   С ПОВИННОЙ  ", "confession", "явка с повинной"],
            "aggravating": [],
        },
    )
    assert record.recommendation_mitigating == {"явка с повинной"}
    assert record.recommendation_aggravating == set()
    unknown = load_verdict(
        tmp_path,
        "",
        facts={"recommendation_facts_trusted": True, "mitigating": ["confession"]},
    )
    assert unknown.recommendation_mitigating is None
    assert unknown.recommendation_aggravating is None


def test_csv_optional_verified_factors_are_explicit_json_lists(tmp_path):
    fixture = Path(__file__).parents[1] / "fixtures" / "synthetic_analysis.csv"
    with fixture.open(encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source))
    verified = {**row, "mitigating_factors": '["  РАСКАЯНИЕ  "]', "aggravating_factors": "[]"}
    unknown = {**row, "mitigating_factors": '["confession"]', "aggravating_factors": ""}
    malformed = {**row, "mitigating_factors": '{"err":1}', "aggravating_factors": ""}
    path = tmp_path / "fixture.csv"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=tuple(verified))
        writer.writeheader()
        writer.writerows((verified, unknown, malformed))
    repository = CsvCorpusRepository(path)
    confirmed, missing = repository.records
    assert repository.skipped_records == 1
    assert confirmed.recommendation_mitigating == {"раскаяние"}
    assert confirmed.recommendation_aggravating == set()
    assert missing.recommendation_mitigating is None
    assert missing.recommendation_aggravating is None
