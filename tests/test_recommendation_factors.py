import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from judicial_ai.api.application import create_app
from judicial_ai.config import Settings


RECIDIVISM = "рецидив преступлений"
GROUP = "группа лиц"
REPENTANCE = "раскаяние"
REGIMES = dict.fromkeys(("special_order", "pretrial_agreement", "art_64", "art_69"), False)


def _case_row(index: int, kind: str, value: int, facts: dict) -> dict:
    return {
        "schema_version": 1,
        "case_id": f"example-{index}",
        "article_parts": ["3"],
        "process": REGIMES,
        "facts": facts,
        "punishment": {"principal_punishment": kind, "principal_value": value},
    }


@pytest.fixture
def factor_client(settings: Settings, tmp_path: Path):
    path = tmp_path / "retrieval_cases.jsonl"
    rows = [
        _case_row(
            index,
            kind,
            value,
            {
                "amount_rub": 150000,
                "aggravating": [factor],
                "recommendation_facts_trusted": True,
            },
        )
        for index, (factor, kind, value) in enumerate(
            [(RECIDIVISM, "imprisonment", term) for term in (24, 36, 48)]
            + [(GROUP, "fine", fine) for fine in (100000, 200000, 300000)]
        )
    ]
    rows.extend(
        (
            _case_row(6, "imprisonment", 120, {"amount_rub": 150000, "aggravating": [RECIDIVISM, GROUP]}),
            _case_row(
                7,
                "imprisonment",
                144,
                {"amount_rub": 150000, "aggravating": [], "recommendation_facts_trusted": True},
            ),
            _case_row(8, "imprisonment", 168, {"amount_rub": 150000, "recommendation_facts_trusted": True}),
        )
    )
    rows.extend(
        _case_row(
            index,
            "fine",
            fine,
            {"mitigating": [REPENTANCE], "recommendation_facts_trusted": True},
        )
        for index, fine in enumerate((400000, 500000, 600000), start=9)
    )
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    config = settings.model_copy(update={"processed_repository_path": path, "dense_enabled": False})
    with TestClient(create_app(config)) as client:
        yield client


def _analysis(client: TestClient, case: dict) -> dict:
    response = client.post("/api/v1/analyses", json={"case": {"article_part": "3", **REGIMES, **case}, "top_k": 1})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(
    ("mark", "kind", "median"),
    [(RECIDIVISM, "imprisonment", 36), (GROUP, "fine", 200000)],
)
def test_named_factor_changes_recommendation_with_same_amount_and_count(
    factor_client: TestClient, mark: str, kind: str, median: int
) -> None:
    result = _analysis(factor_client, {"amount_rub": 150000, "aggravating": [mark]})
    recommendation = result["recommendation"]
    assert recommendation["matching_cases"] == recommendation["usable_outcomes"] == 3
    assert recommendation["status"] == "available"
    assert (recommendation["kind"], recommendation["suggested_value"]) == (kind, median)
    assert any(mark in reason and "3 из 12" in reason for reason in recommendation["reasons"])
    assert result["punishment_summary"]["candidate_count"] == 12
    assert len(result["analogues"]) == 1


@pytest.mark.parametrize(
    ("selection", "second_factor"),
    [
        ({"aggravating": [RECIDIVISM, GROUP]}, GROUP),
        ({"aggravating": [RECIDIVISM], "mitigating": [REPENTANCE]}, REPENTANCE),
    ],
)
def test_all_selected_factors_must_have_joint_confirmation(
    factor_client: TestClient, selection: dict, second_factor: str
) -> None:
    result = _analysis(factor_client, selection)
    recommendation = result["recommendation"]
    assert recommendation["status"] == "unavailable"
    assert recommendation["alternatives"] == []
    assert recommendation["matching_cases"] == recommendation["usable_outcomes"] == 0
    reason = next(reason for reason in recommendation["reasons"] if "0 из 12" in reason)
    assert RECIDIVISM in reason and second_factor in reason


def test_named_factor_keeps_amount_gate(factor_client: TestClient) -> None:
    result = _analysis(factor_client, {"amount_rub": 1500000, "aggravating": [RECIDIVISM]})
    recommendation = result["recommendation"]
    assert recommendation["status"] == "unavailable"
    assert recommendation["matching_cases"] == recommendation["usable_outcomes"] == 0
    assert recommendation["alternatives"] == []
    assert any(RECIDIVISM in reason and "3 из 12" in reason for reason in recommendation["reasons"])
    assert any("1500000 руб." in reason and "0 из 3" in reason for reason in recommendation["reasons"])


def test_verified_factor_cohort_works_without_numeric_ranking(factor_client: TestClient) -> None:
    result = _analysis(factor_client, {"mitigating": [REPENTANCE]})
    recommendation = result["recommendation"]
    assert recommendation["matching_cases"] == recommendation["usable_outcomes"] == 3
    assert recommendation["status"] == "available"
    assert (recommendation["kind"], recommendation["suggested_value"]) == ("fine", 500000)
    assert any(REPENTANCE in reason and "3 из 12" in reason for reason in recommendation["reasons"])
    assert all("Признаки сравнения" not in reason for reason in recommendation["reasons"])


def test_confirmed_text_and_prepared_factors_enter_one_cohort(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "retrieval_cases.jsonl"
    rows = [
        _case_row(
            0, "imprisonment", 24,
            {"aggravating": [RECIDIVISM], "episodes": 1, "recommendation_facts_trusted": True},
        )
    ]
    rows.extend(
        _case_row(
            index, "imprisonment", term,
            {
                "episodes": 999,
                "narrative": (
                    f"Синтетический пример {index}.\n"
                    "УСТАНОВИЛ:\n"
                    "Обстоятельством, отягчающим наказание, суд признает рецидив преступлений.\n"
                    "ПРИГОВОРИЛ:\nПризнать подсудимого виновным по ч. 3 ст. 290 УК РФ."
                ),
            },
        )
        for index, term in enumerate((36, 48), start=1)
    )
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    config = settings.model_copy(update={"processed_repository_path": path, "dense_enabled": False})
    with TestClient(create_app(config)) as client:
        result = _analysis(client, {"episodes": 1, "aggravating": [RECIDIVISM]})
    recommendation = result["recommendation"]
    assert recommendation["matching_cases"] == recommendation["usable_outcomes"] == 3
    assert recommendation["status"] == "available"
    assert (recommendation["kind"], recommendation["suggested_value"]) == ("imprisonment", 36)
    assert all("Признаки сравнения" not in reason for reason in recommendation["reasons"])
