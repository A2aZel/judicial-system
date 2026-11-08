import json
from pathlib import Path

from fastapi.testclient import TestClient

from judicial_ai.api.application import create_app
from judicial_ai.config import Settings

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "synthetic_analysis.csv"


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


def test_empty_analogues_are_not_hidden(client: TestClient, valid_payload: dict) -> None:
    valid_payload["case"]["article_part"] = "6"
    response = client.post("/api/v1/analyses", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["analogues"] == []
    assert body["punishment_summary"]["candidate_count"] == 0
    assert body["punishment_summary"]["prison_months"]["count"] == 0
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
