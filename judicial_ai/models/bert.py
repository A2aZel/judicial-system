import hashlib
import hmac
import importlib.metadata
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from judicial_ai.schemas.analysis import AppealRisk

logger = logging.getLogger(__name__)

IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class BertModelAdapter:
    """Lazy transformer encoder with deployment-approved binary risk scoring."""

    def __init__(
        self,
        model_id: str | None,
        revision: str | None = None,
        risk_checkpoint: Path | None = None,
        cache_dir: Path | None = None,
        risk_enabled: bool = False,
        approved_checkpoint_sha256: str | None = None,
    ):
        self.model_id = model_id or None
        self.revision = revision or None
        self.risk_checkpoint = risk_checkpoint
        self.cache_dir = cache_dir
        self.risk_enabled = risk_enabled
        self.approved_checkpoint_sha256 = approved_checkpoint_sha256
        self._tokenizer: Any = None
        self._backbone: Any = None
        self._head_weight: np.ndarray | None = None
        self._head_bias: float | None = None
        self._preprocessing: dict[str, Any] | None = None
        self._checkpoint_digest: str | None = None
        self._backbone_load_error = False
        self._inference_status = "not_run"
        self._risk_available = False

    def info(self) -> dict[str, object]:
        if not self.model_id:
            backbone_status = "not_configured"
        elif self._backbone_load_error:
            backbone_status = "error"
        elif self._backbone is not None:
            backbone_status = "loaded"
        else:
            backbone_status = "lazy"
        return {
            "repository": self.model_id,
            "revision": self.revision,
            "backbone_status": backbone_status,
            "inference_status": self._inference_status,
            "base_weights_usage": "representations_only",
            "risk_enabled": self.risk_enabled,
            "risk_checkpoint_configured": self.risk_checkpoint is not None,
            "risk_checkpoint_sha256": self._checkpoint_digest,
            "risk_status": "available" if self._risk_available else "unavailable",
            "transformers_version": self._version("transformers"),
            "sentence_transformers_version": self._version("sentence-transformers"),
            "torch_version": self._version("torch"),
        }

    def predict_appeal_risk(self, narrative: str | None) -> AppealRisk:
        self._risk_available = False
        model_name = self._qualified_name()
        if not self.risk_enabled:
            return self._unavailable(
                "Scoring апелляционного риска отключён конфигурацией; базовые веса не используются для вероятности.",
                model_name,
            )
        if not self.risk_checkpoint or not self.approved_checkpoint_sha256:
            return self._unavailable(
                "Не задан checkpoint и одобренный SHA256 из deployment config.",
                model_name,
            )
        if not narrative or not narrative.strip():
            return self._unavailable(
                "Для классификатора риска не предоставлен narrative.", model_name
            )
        try:
            self._load()
        except Exception as exc:
            self._backbone_load_error = True
            self._inference_status = "error"
            logger.warning("BERT risk loading unavailable: %s", type(exc).__name__)
            return self._unavailable(
                "Backbone или утверждённый checkpoint не прошёл загрузку и проверки deployment gates.",
                model_name,
            )

        assert self._tokenizer is not None
        assert self._backbone is not None
        assert self._head_weight is not None
        assert self._head_bias is not None
        assert self._preprocessing is not None
        try:
            import torch

            encoded = self._tokenizer(
                narrative,
                truncation=False,
                add_special_tokens=self._preprocessing["add_special_tokens"],
                return_tensors="pt",
            )
            token_count = int(encoded["input_ids"].shape[1])
            max_tokens = self._preprocessing["max_tokens"]
            if token_count > max_tokens:
                self._inference_status = "rejected"
                return self._unavailable(
                    f"Narrative содержит {token_count} токенов при лимите checkpoint {max_tokens}; усечение запрещено preprocessing contract.",
                    model_name,
                )
            device = next(self._backbone.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self._backbone(**encoded)
                hidden = output.last_hidden_state[:, 0]
                weight = torch.as_tensor(
                    self._head_weight, dtype=hidden.dtype, device=hidden.device
                )
                logit = (hidden @ weight) + self._head_bias
                score = float(torch.sigmoid(logit).cpu().item())
            if not math.isfinite(score):
                raise ValueError("non-finite risk score")
            self._backbone_load_error = False
            self._inference_status = "healthy"
            self._risk_available = True
            return AppealRisk(
                status="available",
                score=score,
                reasons=[
                    "Оценка утверждённого классификатора по тексту; это статистическая вероятность, не прогноз судебного решения."
                ],
                model=model_name + "+approved-risk-head",
            )
        except Exception as exc:
            self._inference_status = "error"
            logger.warning("BERT risk inference unavailable: %s", type(exc).__name__)
            return self._unavailable(
                "Inference утверждённого checkpoint завершился ошибкой; оценка не сформирована.",
                model_name,
            )

    def _load(self) -> None:
        if (
            self._backbone is not None
            and self._head_weight is not None
            and self._preprocessing is not None
        ):
            return
        if not self.model_id or not IMMUTABLE_REVISION_RE.fullmatch(self.revision or ""):
            raise ValueError("an immutable full backbone commit SHA is required")
        if not SHA256_RE.fullmatch(self.approved_checkpoint_sha256 or ""):
            raise ValueError("a full approved checkpoint SHA256 is required")
        actual_digest = self._file_sha256(self._weights_path())
        if not hmac.compare_digest(actual_digest, self.approved_checkpoint_sha256):
            raise ValueError("checkpoint digest does not match deployment approval")
        metadata = self._checkpoint_metadata()
        self._validate_checkpoint_metadata(metadata, actual_digest)
        preprocessing = metadata["preprocessing"]
        with np.load(self._weights_path(), allow_pickle=False) as checkpoint:
            weight = np.asarray(checkpoint["weight"], dtype=np.float32)
            bias = float(np.asarray(checkpoint["bias"]).item())
        if (
            weight.ndim != 1
            or not np.isfinite(weight).all()
            or not math.isfinite(bias)
            or not np.any(weight)
        ):
            raise ValueError("invalid or untrained zero risk checkpoint")
        from transformers import AutoModel, AutoTokenizer

        options: dict[str, Any] = {"revision": self.revision}
        if self.cache_dir:
            options["cache_dir"] = str(self.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **options)
        backbone = AutoModel.from_pretrained(self.model_id, **options)
        backbone.eval()
        hidden_size = int(backbone.config.hidden_size)
        if weight.shape != (hidden_size,):
            raise ValueError("risk checkpoint dimensions do not match backbone")
        self._tokenizer = tokenizer
        self._backbone = backbone
        self._head_weight = weight
        self._head_bias = bias
        self._preprocessing = preprocessing
        self._checkpoint_digest = actual_digest
        self._backbone_load_error = False

    def _validate_checkpoint_metadata(
        self, metadata: dict[str, Any], actual_digest: str
    ) -> None:
        if (
            metadata.get("weights_sha256") != actual_digest
            or metadata.get("architecture_version") != 1
            or metadata.get("base_model_id") != self.model_id
            or metadata.get("base_revision") != self.revision
            or metadata.get("task") != "binary_appeal_risk"
        ):
            raise ValueError("checkpoint metadata does not match the approved artifact")
        preprocessing = metadata.get("preprocessing")
        if not isinstance(preprocessing, dict):
            raise ValueError("checkpoint preprocessing contract is missing")
        max_tokens = preprocessing.get("max_tokens")
        if (
            preprocessing.get("tokenizer_model_id") != self.model_id
            or preprocessing.get("tokenizer_revision") != self.revision
            or preprocessing.get("pooling") != "cls"
            or preprocessing.get("overflow") != "reject"
            or preprocessing.get("normalization") != "none"
            or preprocessing.get("add_special_tokens") is not True
            or type(max_tokens) is not int
            or not 1 <= max_tokens <= 4096
        ):
            raise ValueError("unsupported checkpoint preprocessing contract")

    def _weights_path(self) -> Path:
        assert self.risk_checkpoint is not None
        if self.risk_checkpoint.is_dir():
            return self.risk_checkpoint / "risk_head.npz"
        return self.risk_checkpoint

    def _metadata_path(self) -> Path:
        assert self.risk_checkpoint is not None
        if self.risk_checkpoint.is_dir():
            return self.risk_checkpoint / "risk_head.json"
        return self.risk_checkpoint.with_suffix(".json")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        with self._metadata_path().open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError("invalid checkpoint metadata")
        return metadata

    @staticmethod
    def _file_sha256(path: Path) -> str:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    @staticmethod
    def _unavailable(reason: str, model: str | None) -> AppealRisk:
        return AppealRisk(status="unavailable", reasons=[reason], model=model)

    def _qualified_name(self) -> str | None:
        if not self.model_id:
            return None
        return self.model_id + (f"@{self.revision}" if self.revision else "")

    @staticmethod
    def _version(package: str) -> str | None:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None
