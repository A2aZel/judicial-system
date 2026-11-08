from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from judicial_ai.api.application import create_app
from judicial_ai.config import Settings

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "synthetic_analysis.csv"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        api_key=None,
        corpus_path=FIXTURE_PATH,
        _env_file=None,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    return {
        "case": {
            "article_part": "3",
            "amount_rub": 150000,
            "episodes": 1,
            "special_order": False,
            "pretrial_agreement": False,
            "art_64": False,
            "art_69": False,
            "mitigating": ["синтетическое обстоятельство"],
            "aggravating": [],
            "narrative": "Полностью синтетический текст без реальных лиц и дел.",
        },
        "draft_punishment": {
            "prison_months": 24,
            "fine_rub": None,
            "fine_multiple": None,
            "ban_years": 2,
            "conditional": False,
            "confiscation": False,
        },
        "top_k": 3,
    }
