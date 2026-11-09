from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BERT_MODEL_ID = "deepvk/RuModernBERT-base"
DEFAULT_BERT_REVISION = "8aee6f818db9070b635a549580dc64dcfa03e39a"
DEFAULT_DENSE_MODEL_ID = "deepvk/USER-bge-m3"
DEFAULT_DENSE_MODEL_REVISION = "0cc6cfe48e260fb0474c753087a69369e88709ae"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="JUDICIAL_AI_",
        env_file=".env",
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_key: SecretStr | None = None
    corpus_path: Path = Path("corpus/sudact/analysis.csv")
    processed_repository_path: Path = Path("corpus/processed/retrieval_cases.jsonl")
    dense_index_path: Path = Path("corpus/processed/dense-index.npz")
    dense_manifest_path: Path = Path("corpus/processed/model-card.json")
    dense_enabled: bool = True
    dense_model_id: str = DEFAULT_DENSE_MODEL_ID
    dense_model_revision: str = DEFAULT_DENSE_MODEL_REVISION
    dense_weight: float = Field(default=0.7, ge=0, le=1)
    dense_batch_size: int = Field(default=4, ge=1, le=32)
    dense_max_tokens: int = Field(default=1024, ge=64, le=8192)
    max_request_bytes: int = Field(default=65_536, ge=1024, le=1_048_576)
    default_top_k: int = Field(default=5, ge=1, le=50)
    bert_model_id: str | None = DEFAULT_BERT_MODEL_ID
    bert_revision: str | None = DEFAULT_BERT_REVISION
    bert_risk_enabled: bool = False
    bert_risk_checkpoint: Path | None = None
    bert_risk_checkpoint_sha256: str | None = None
    bert_cache_dir: Path | None = None

    @field_validator(
        "api_key",
        "bert_model_id",
        "bert_revision",
        "bert_risk_checkpoint",
        "bert_risk_checkpoint_sha256",
        "bert_cache_dir",
        mode="before",
    )
    @classmethod
    def empty_string_to_none(cls, value):
        return None if value == "" else value

    @model_validator(mode="after")
    def require_production_api_key(self) -> "Settings":
        if self.app_env == "production" and not self.api_key:
            raise ValueError("JUDICIAL_AI_API_KEY is required in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
