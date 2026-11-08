import pytest
from pydantic import ValidationError

from judicial_ai.config import (
    DEFAULT_BERT_MODEL_ID,
    DEFAULT_BERT_REVISION,
    Settings,
)


def test_default_encoder_is_pinned_and_risk_is_disabled() -> None:
    settings = Settings(app_env="test", _env_file=None)
    assert settings.bert_model_id == DEFAULT_BERT_MODEL_ID == "deepvk/RuModernBERT-base"
    assert settings.bert_revision == DEFAULT_BERT_REVISION
    assert len(settings.bert_revision) == 40
    assert settings.bert_risk_enabled is False
    assert settings.bert_risk_checkpoint_sha256 is None


def test_empty_optional_environment_values_are_normalized() -> None:
    settings = Settings(
        app_env="test",
        api_key="",
        bert_model_id="",
        bert_revision="",
        bert_risk_checkpoint="",
        bert_risk_checkpoint_sha256="",
        bert_cache_dir="",
        _env_file=None,
    )
    assert settings.api_key is None
    assert settings.bert_model_id is None
    assert settings.bert_revision is None
    assert settings.bert_risk_checkpoint is None
    assert settings.bert_risk_checkpoint_sha256 is None
    assert settings.bert_cache_dir is None


def test_production_requires_nonempty_api_key() -> None:
    with pytest.raises(ValidationError, match="JUDICIAL_AI_API_KEY"):
        Settings(app_env="production", api_key="", _env_file=None)
