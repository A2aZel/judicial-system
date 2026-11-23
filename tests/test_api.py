import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from judicial_ai.api.application import create_app
from judicial_ai.config import Settings

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "synthetic_analysis.csv"


def write_outcome_corpus(
    path: Path,
    outcomes: list[tuple[str, str, str]],
    ranked: bool = True,
    term_execution: str = "",
) -> None:
    with FIXTURE_PATH.open(encoding="utf-8", newline="") as fixture:
        row = next(csv.DictReader(fixture))
    row.update(principal_punishment="", principal_value="", principal_conditional="")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        for kind, value, amount_rub in outcomes:
            writer.writerow(
                {
                    **row,
                    "amount_rub": amount_rub if ranked else "",
                    "episodes": "",
                    "mitigating_count": "",
                    "aggravating_count": "",
                    "principal_punishment": kind,
                    "principal_value": value,
                    "principal_conditional": term_execution if kind == "imprisonment" else "",
                    "prison_months": value if kind == "imprisonment" else "",
                    "fine_rub": value if kind == "fine" else "",
                }
            )


def test_openapi_health_and_models(client: TestClient) -> None:
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["corpus"]["records"] == 10
    assert "api_key" not in health.text.lower()

    models = client.get("/api/v1/models")
    assert models.status_code == 200
    assert models.json()["bert"]["repository"] == "deepvk/RuModernBERT-base"
    assert models.json()["bert"]["backbone_status"] == "lazy"
    assert models.json()["bert"]["risk_enabled"] is False
    assert models.json()["knn"]["algorithm"] == "deterministic_knn"

    openapi = client.get("/api/v1/openapi.json")
    assert openapi.status_code == 404
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/api/v1/analyses" in openapi.json()["paths"]
    assert client.get("/docs").status_code == 200


def test_analysis_contract_and_risk_unavailable(
    client: TestClient, valid_payload: dict
) -> None:
    response = client.post("/api/v1/analyses", json=valid_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["analysis_id"]
    assert len(body["analogues"]) == 3
    assert body["punishment_summary"]["statistics_scope"] == "cohort"
    assert body["punishment_summary"]["candidate_count"] == 5
    assert body["punishment_summary"]["returned_count"] == 3
    assert body["appeal_risk"]["status"] == "unavailable"
    assert body["appeal_risk"]["score"] is None
    assert body["model_info"]["knn"]["active_numeric_features"]
    assert body["model_info"]["knn"]["ranking_available"] is True
    assert valid_payload["case"]["narrative"] not in response.text

    allowed_analogue_fields = {
        "distance",
        "prison_months",
        "fine_rub",
        "fine_multiple",
        "ban_on_positions",
        "ban_years",
        "conditional",
        "confiscation",
    }
    for analogue in body["analogues"]:
        assert set(analogue) == allowed_analogue_fields
        assert analogue["distance"] is not None
        assert not {"id", "uid", "case_id", "name", "narrative"} & set(analogue)


def test_analysis_is_deterministic_except_analysis_id(
    client: TestClient, valid_payload: dict
) -> None:
    first = client.post("/api/v1/analyses", json=valid_payload).json()
    second = client.post("/api/v1/analyses", json=valid_payload).json()
    first.pop("analysis_id")
    second.pop("analysis_id")
    assert first == second


def test_facts_only_recommendation_ignores_draft_and_display_limit(
    client: TestClient, valid_payload: dict
) -> None:
    response = client.post(
        "/api/v1/analyses", json={"case": valid_payload["case"], "top_k": 1}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["analogues"]) == 1
    assert {check["code"]: check["status"] for check in body["legal_checks"]} == {
        "episode_regime_consistency": "passed",
        "statutory_sanction_limits": "not_checked",
    }
    recommendation = body["recommendation"]
    assert recommendation.pop("reasons")
    assert recommendation == {
        "status": "available",
        "data_source": "synthetic",
        "alternatives": [],
        "kind": "imprisonment",
        "execution": "unspecified",
        "suggested_value": 21,
        "range_min": 16.5,
        "range_max": 27,
        "unit": "months",
        "matching_cases": 4,
        "usable_outcomes": 4,
        "selected_outcomes": 4,
        "legal_status": "not_checked",
    }
    assert valid_payload["case"]["narrative"] not in response.text

    valid_payload["draft_punishment"] = {"conditional": True}
    legacy_response = client.post("/api/v1/analyses", json=valid_payload)
    assert legacy_response.status_code == 200, legacy_response.text
    legacy = legacy_response.json()
    legacy_checks = {check["code"]: check["status"] for check in legacy["legal_checks"]}
    assert legacy_checks["primary_punishment_present"] == "error"
    assert legacy_checks["conditional_requires_prison_term"] == "error"
    assert legacy["recommendation"] == response.json()["recommendation"]


def test_facts_only_recommendation_abstains_with_aggregate_sentencing(
    client: TestClient, valid_payload: dict
) -> None:
    valid_payload["case"]["art_69"] = True
    response = client.post("/api/v1/analyses", json={"case": valid_payload["case"]})
    assert response.status_code == 200, response.text
    body = response.json()
    legal_checks = {check["code"]: check["status"] for check in body["legal_checks"]}
    assert legal_checks["episode_regime_consistency"] == "warning"
    recommendation = body["recommendation"]
    assert recommendation["status"] == "unavailable"
    assert recommendation["matching_cases"] == 1
    assert recommendation["selected_outcomes"] == 0
    assert recommendation["suggested_value"] is None
    assert recommendation["reasons"]


def test_facts_only_recommendation_uses_twenty_closest_for_principal_fine(
    settings: Settings, valid_payload: dict, tmp_path: Path
) -> None:
    corpus = tmp_path / "outcomes.csv"
    write_outcome_corpus(
        corpus,
        [("imprisonment", "18", "450000")] * 20
        + [("fine", "100000", "150000")] * 10
        + [("fine", "200000", "150000")] * 10,
    )
    with TestClient(create_app(settings.model_copy(update={"corpus_path": corpus}))) as client:
        response = client.post("/api/v1/analyses", json={"case": valid_payload["case"]})
    assert response.status_code == 200, response.text
    recommendation = response.json()["recommendation"]
    assert recommendation.pop("reasons")
    assert recommendation == {
        "status": "available",
        "data_source": "case_corpus",
        "alternatives": [],
        "kind": "fine",
        "execution": None,
        "suggested_value": 150000,
        "range_min": 100000,
        "range_max": 200000,
        "unit": "rub",
        "matching_cases": 40,
        "usable_outcomes": 20,
        "selected_outcomes": 20,
        "legal_status": "not_checked",
    }


@pytest.mark.parametrize(
    ("term_execution", "expected"), [("True", "conditional"), ("False", "actual")]
)
def test_facts_only_recommendation_reports_unanimous_explicit_execution(
    settings: Settings, valid_payload: dict, tmp_path: Path, term_execution: str, expected: str
) -> None:
    corpus = tmp_path / "outcomes.csv"
    write_outcome_corpus(
        corpus,
        [
            ("imprisonment", "18", "120000"),
            ("imprisonment", "24", "150000"),
            ("imprisonment", "36", "180000"),
            ("fine", "100000", "150000"),
        ],
        term_execution=term_execution,
    )
    with TestClient(create_app(settings.model_copy(update={"corpus_path": corpus}))) as client:
        response = client.post("/api/v1/analyses", json={"case": valid_payload["case"]})
    assert response.status_code == 200, response.text
    recommendation = response.json()["recommendation"]
    assert recommendation["status"] == "available"
    assert recommendation["execution"] == expected
    assert recommendation["suggested_value"] == 24
    assert recommendation["selected_outcomes"] == 3


@pytest.mark.parametrize(
    ("ranked", "terms", "fines", "usable"),
    [(True, 3, 3, 6), (True, 2, 1, 3), (False, 3, 3, 0)],
    ids=["tied", "insufficient-majority", "unranked"],
)
def test_facts_only_recommendation_abstains_without_supported_majority(
    settings: Settings,
    valid_payload: dict,
    tmp_path: Path,
    ranked: bool,
    terms: int,
    fines: int,
    usable: int,
) -> None:
    corpus = tmp_path / "outcomes.csv"
    write_outcome_corpus(
        corpus,
        [("imprisonment", "18", "120000")] * terms
        + [("fine", "100000", "150000")] * fines,
        ranked=ranked,
    )
    with TestClient(create_app(settings.model_copy(update={"corpus_path": corpus}))) as client:
        response = client.post("/api/v1/analyses", json={"case": valid_payload["case"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_info"]["knn"]["ranking_available"] is ranked
    recommendation = body["recommendation"]
    assert recommendation["status"] == "unavailable"
    assert recommendation["matching_cases"] == (terms + fines if ranked else 0)
    assert recommendation["usable_outcomes"] == usable
    assert recommendation["selected_outcomes"] == 0
    assert recommendation["kind"] is None
    assert recommendation["range_min"] is None
    assert recommendation["range_max"] is None
    assert recommendation["reasons"]
    assert [estimate["kind"] for estimate in recommendation["alternatives"]] == (
        ["imprisonment", "fine"] if ranked and terms == fines else []
    )


def test_facts_only_legacy_verdicts_report_deduplicated_historical_tie(
    settings: Settings, valid_payload: dict, tmp_path: Path
) -> None:
    corpus = tmp_path / "retrieval_cases.jsonl"
    rows = []
    outcomes = [
        ("imprisonment", 12),
        ("imprisonment", 24),
        ("imprisonment", 36),
        ("imprisonment", 36),
        ("fine", 100000),
        ("fine", 200000),
        ("fine", 300000),
    ]
    for index, (kind, value) in enumerate(outcomes):
        rows.append({
            "schema_version": 1,
            "case_id": f"example-{index}",
            "article_parts": ["3"],
            "process": dict.fromkeys(("special_order", "pretrial_agreement", "art_64", "art_69"), False),
            "facts": {"amount_rub": 100000 * (index + 1), "narrative": f"Версия {min(index, 2) if index == 3 else index}"},
            "punishment": {
                "principal_punishment": kind,
                "principal_value": value,
                "principal_conditional": kind == "imprisonment" or None,
            },
        })
    corpus.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    settings = settings.model_copy(update={"processed_repository_path": corpus, "dense_enabled": False})
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/analyses", json={"case": valid_payload["case"]})
        valid_payload["case"].pop("amount_rub")
        no_amount_response = client.post("/api/v1/analyses", json={"case": valid_payload["case"]})
    assert response.status_code == 200, response.text
    by_amount = response.json()["recommendation"]
    assert by_amount["status"] == "unavailable"
    assert by_amount["matching_cases"] == by_amount["usable_outcomes"] == 0
    assert by_amount["alternatives"] == []
    assert any("150000 руб." in reason and "0 из 6" in reason for reason in by_amount["reasons"])

    assert no_amount_response.status_code == 200, no_amount_response.text
    recommendation = no_amount_response.json()["recommendation"]
    assert recommendation["data_source"] == "case_corpus"
    assert recommendation["status"] == "unavailable"
    assert recommendation["matching_cases"] == recommendation["usable_outcomes"] == 6
    assert recommendation["selected_outcomes"] == 0
    assert [
        (alternative["kind"], alternative["execution"], alternative["suggested_value"])
        for alternative in recommendation["alternatives"]
    ] == [("imprisonment", "conditional", 24), ("fine", None, 200000)]
    assert all("сумма взятки в рублях" not in reason for reason in recommendation["reasons"])
    assert "example-" not in response.text


def test_recommendation_uses_recovered_amount_without_legacy_ranking_facts(
    settings: Settings, valid_payload: dict, tmp_path: Path
) -> None:
    corpus = tmp_path / "retrieval_cases.jsonl"
    facts_by_case = [
        {
            "amount_rub": 1500000,
            "episodes": 999,
            "narrative": (
                f"УСТАНОВИЛ:\nПодсудимый получил взятку в размере {amount} рублей.\n"
                "ПРИГОВОРИЛ:\nПризнать подсудимого виновным по ч. 3 ст. 290 УК РФ."
            ),
        }
        for amount in (100000, 150000, 200000)
    ]
    facts_by_case.extend((
        {"amount_rub": 150000, "episodes": 1, "recommendation_facts_trusted": True},
        {"amount_rub": 150000, "episodes": 1},
    ))
    rows = [
        {
            "schema_version": 1,
            "case_id": f"example-{index}",
            "article_parts": ["3"],
            "process": dict.fromkeys(("special_order", "pretrial_agreement", "art_64", "art_69"), False),
            "facts": facts,
            "punishment": {"principal_punishment": "imprisonment", "principal_value": value},
        }
        for index, (facts, value) in enumerate(zip(facts_by_case, (12, 24, 36, 60, 120), strict=True))
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    settings = settings.model_copy(update={"processed_repository_path": corpus, "dense_enabled": False})
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/analyses", json={"case": valid_payload["case"]})
    assert response.status_code == 200, response.text
    recommendation = response.json()["recommendation"]
    assert recommendation["status"] == "available"
    assert recommendation["matching_cases"] == recommendation["usable_outcomes"] == 4
    assert recommendation["suggested_value"] == 30
    assert "Признаки сравнения — сумма взятки в рублях — 150000." in recommendation["reasons"]


@pytest.mark.parametrize(
    ("amount", "matching", "kind", "suggested"),
    [
        (0, 0, None, None),
        (150000, 3, "imprisonment", 24),
        (1500000, 3, "fine", 200000),
        (15000000, 0, None, None),
    ],
)
def test_recommendation_uses_only_nearby_known_amounts(
    settings: Settings,
    valid_payload: dict,
    tmp_path: Path,
    amount: int,
    matching: int,
    kind: str | None,
    suggested: int | None,
) -> None:
    corpus = tmp_path / "outcomes.csv"
    write_outcome_corpus(
        corpus,
        [
            ("imprisonment", "12", "50000"),
            ("imprisonment", "24", "150000"),
            ("imprisonment", "36", "450000"),
            ("fine", "100000", "500000"),
            ("fine", "200000", "1500000"),
            ("fine", "300000", "4500000"),
            ("fine", "5000000", ""),
        ],
    )
    valid_payload["case"]["amount_rub"] = amount
    with TestClient(create_app(settings.model_copy(update={"corpus_path": corpus}))) as client:
        response = client.post("/api/v1/analyses", json={"case": valid_payload["case"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["punishment_summary"]["candidate_count"] == 7
    recommendation = body["recommendation"]
    assert recommendation["matching_cases"] == recommendation["usable_outcomes"] == matching
    assert recommendation["status"] == ("available" if kind else "unavailable")
    assert recommendation["kind"] == kind
    assert recommendation["suggested_value"] == suggested
    assert any(
        f"{amount} руб." in reason and f"{matching} из 7" in reason
        for reason in recommendation["reasons"]
    )


def test_empty_analogues_are_not_hidden(client: TestClient, valid_payload: dict) -> None:
    valid_payload["case"]["article_part"] = "6"
    response = client.post("/api/v1/analyses", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["analogues"] == []
    assert body["punishment_summary"]["candidate_count"] == 0
    assert body["punishment_summary"]["prison_months"]["count"] == 0
    recommendation = body["recommendation"]
    assert recommendation["status"] == "unavailable"
    assert recommendation["matching_cases"] == 0
    assert recommendation["usable_outcomes"] == 0
    assert recommendation["reasons"]
    assert any("не найдены" in warning for warning in body["warnings"])


def test_validation_error_does_not_echo_input(client: TestClient, valid_payload: dict) -> None:
    valid_payload["case"]["article_part"] = "sensitive-input"
    valid_payload["case"]["narrative"] = "sensitive-narrative"
    response = client.post("/api/v1/analyses", json=valid_payload)
    assert response.status_code == 422
    assert "sensitive-input" not in response.text
    assert "sensitive-narrative" not in response.text
    assert response.json()["detail"] == "Request validation failed"


def test_bearer_auth_when_configured(valid_payload: dict) -> None:
    settings = Settings(
        app_env="test",
        api_key="unit-test-key",
        corpus_path=FIXTURE_PATH,
        _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/models").status_code == 401
        assert client.post("/api/v1/analyses", json=valid_payload).status_code == 401
        headers = {"Authorization": "Bearer unit-test-key"}
        assert client.get("/api/v1/models", headers=headers).status_code == 200
        assert client.post("/api/v1/analyses", json=valid_payload, headers=headers).status_code == 200
        assert client.get("/api/v1/health").status_code == 200


def test_non_ascii_bearer_is_401_not_500() -> None:
    settings = Settings(
        app_env="test", api_key="unit-test-key", corpus_path=FIXTURE_PATH, _env_file=None
    )
    with TestClient(create_app(settings)) as client:
        response = client.get(
            "/api/v1/models",
            headers={"Authorization": "Bearer токен".encode("utf-8")},
        )
        assert response.status_code == 401


def test_missing_corpus_degrades_without_500(valid_payload: dict, tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        corpus_path=tmp_path / "absent.csv",
        _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health").json()["status"] == "degraded"
        response = client.post("/api/v1/analyses", json=valid_payload)
        assert response.status_code == 200
        assert response.json()["analogues"] == []
        assert any("недоступен" in warning for warning in response.json()["warnings"])


def test_request_body_limit(valid_payload: dict) -> None:
    valid_payload["case"]["narrative"] = "x" * 4000
    settings = Settings(
        app_env="test",
        corpus_path=FIXTURE_PATH,
        max_request_bytes=1024,
        _env_file=None,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/analyses", json=valid_payload)
        assert response.status_code == 413


def test_chunked_request_body_limit_is_cumulative(valid_payload: dict) -> None:
    valid_payload["case"]["narrative"] = "x" * 4000
    settings = Settings(
        app_env="test",
        corpus_path=FIXTURE_PATH,
        max_request_bytes=1024,
        _env_file=None,
    )
    body = json.dumps(valid_payload).encode("utf-8")

    def chunks():
        for offset in range(0, len(body), 128):
            yield body[offset : offset + 128]

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/analyses",
            content=chunks(),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
