import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from judicial_ai.models.bert import BertModelAdapter

MODEL_ID = "deepvk/RuModernBERT-base"
REVISION = "8aee6f818db9070b635a549580dc64dcfa03e39a"


def make_checkpoint(tmp_path: Path, zero: bool = False) -> tuple[Path, str]:
    path = tmp_path / "risk_head.npz"
    np.savez(path, weight=np.array([0.0, 0.0] if zero else [1.0, 0.0]), bias=0.0)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "validated": True,
        "weights_sha256": digest,
        "architecture_version": 1,
        "base_model_id": MODEL_ID,
        "base_revision": REVISION,
        "task": "binary_appeal_risk",
        "preprocessing": {
            "tokenizer_model_id": MODEL_ID,
            "tokenizer_revision": REVISION,
            "pooling": "cls",
            "max_tokens": 4,
            "overflow": "reject",
            "normalization": "none",
            "add_special_tokens": True,
        },
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return path, digest


def test_base_backbone_never_produces_risk_score_when_disabled() -> None:
    adapter = BertModelAdapter(MODEL_ID, revision=REVISION)
    assert adapter.info()["backbone_status"] == "lazy"
    assert adapter.info()["risk_enabled"] is False
    result = adapter.predict_appeal_risk("Синтетический narrative")
    assert result.status == "unavailable"
    assert result.score is None
    assert adapter._backbone is None
    assert adapter._tokenizer is None


def test_unconfigured_model_is_reported_without_loading() -> None:
    adapter = BertModelAdapter(None)
    result = adapter.predict_appeal_risk("Текст")
    assert result.status == "unavailable"
    assert result.model is None
    assert adapter.info()["backbone_status"] == "not_configured"
    assert adapter.info()["base_weights_usage"] == "representations_only"


def test_zero_head_with_json_validated_but_without_approved_digest_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint, _ = make_checkpoint(tmp_path, zero=True)
    adapter = BertModelAdapter(
        MODEL_ID,
        REVISION,
        risk_checkpoint=checkpoint,
        risk_enabled=True,
    )
    result = adapter.predict_appeal_risk("Синтетический narrative")
    assert result.status == "unavailable"
    assert result.score is None
    assert adapter._backbone is None


def test_zero_head_is_rejected_before_backbone_download_even_with_matching_digest(
    tmp_path: Path,
) -> None:
    checkpoint, digest = make_checkpoint(tmp_path, zero=True)
    adapter = BertModelAdapter(
        MODEL_ID,
        REVISION,
        risk_checkpoint=checkpoint,
        risk_enabled=True,
        approved_checkpoint_sha256=digest,
    )
    result = adapter.predict_appeal_risk("Синтетический narrative")
    assert result.status == "unavailable"
    assert adapter._backbone is None
    assert adapter.info()["risk_status"] == "unavailable"


def test_checkpoint_digest_mismatch_is_rejected_before_loading(tmp_path: Path) -> None:
    checkpoint, _ = make_checkpoint(tmp_path)
    adapter = BertModelAdapter(
        MODEL_ID,
        REVISION,
        risk_checkpoint=checkpoint,
        risk_enabled=True,
        approved_checkpoint_sha256="0" * 64,
    )
    result = adapter.predict_appeal_risk("Синтетический narrative")
    assert result.status == "unavailable"
    assert adapter._backbone is None


def test_mutable_backbone_revision_is_rejected(tmp_path: Path) -> None:
    checkpoint, digest = make_checkpoint(tmp_path)
    adapter = BertModelAdapter(
        MODEL_ID,
        "main",
        risk_checkpoint=checkpoint,
        risk_enabled=True,
        approved_checkpoint_sha256=digest,
    )
    result = adapter.predict_appeal_risk("Синтетический narrative")
    assert result.status == "unavailable"
    assert adapter._backbone is None


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def __call__(self, text: str, **kwargs):
        self.kwargs = kwargs
        token_count = len(text.split()) + 2
        return {
            "input_ids": torch.ones((1, token_count), dtype=torch.long),
            "attention_mask": torch.ones((1, token_count), dtype=torch.long),
        }


class FakeBackbone:
    def __init__(self):
        self.fail = False
        self.calls = 0

    def parameters(self):
        return iter((torch.empty(0),))

    def __call__(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic inference failure")
        tokens = kwargs["input_ids"].shape[1]
        hidden = torch.zeros((1, tokens, 2))
        hidden[:, :, 0] = 1
        return SimpleNamespace(last_hidden_state=hidden)


def approved_adapter(monkeypatch: pytest.MonkeyPatch) -> tuple[BertModelAdapter, FakeBackbone, FakeTokenizer]:
    adapter = BertModelAdapter(
        MODEL_ID,
        REVISION,
        risk_checkpoint=Path("unused.npz"),
        risk_enabled=True,
        approved_checkpoint_sha256="a" * 64,
    )
    backbone = FakeBackbone()
    tokenizer = FakeTokenizer()

    def load() -> None:
        adapter._backbone = backbone
        adapter._tokenizer = tokenizer
        adapter._head_weight = np.array([1.0, 0.0], dtype=np.float32)
        adapter._head_bias = 0.0
        adapter._preprocessing = {
            "max_tokens": 4,
            "add_special_tokens": True,
        }
        adapter._backbone_load_error = False

    monkeypatch.setattr(adapter, "_load", load)
    return adapter, backbone, tokenizer


def test_health_recovers_and_error_does_not_leave_risk_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, backbone, _ = approved_adapter(monkeypatch)
    assert adapter.predict_appeal_risk("ok").status == "available"
    assert adapter.info()["risk_status"] == "available"
    assert adapter.info()["inference_status"] == "healthy"

    backbone.fail = True
    assert adapter.predict_appeal_risk("ok").status == "unavailable"
    assert adapter.info()["risk_status"] == "unavailable"
    assert adapter.info()["inference_status"] == "error"
    assert adapter.info()["backbone_status"] == "loaded"

    backbone.fail = False
    assert adapter.predict_appeal_risk("ok").status == "available"
    assert adapter.info()["risk_status"] == "available"
    assert adapter.info()["inference_status"] == "healthy"
    assert adapter.info()["backbone_status"] == "loaded"


def test_overlong_narrative_is_rejected_without_truncation_or_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, backbone, tokenizer = approved_adapter(monkeypatch)
    result = adapter.predict_appeal_risk("too many tokens here")
    assert result.status == "unavailable"
    assert "усечение запрещено" in result.reasons[0]
    assert tokenizer.kwargs["truncation"] is False
    assert backbone.calls == 0
    assert adapter.info()["inference_status"] == "rejected"
    assert adapter.info()["risk_status"] == "unavailable"
