import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import tools.build_dense_index as builder

from judicial_ai.analogues.dense import (
    DENSE_MODEL_ID,
    DENSE_MODEL_REVISION,
    DenseAnalogueFinder,
    DenseEncoder,
    DenseUnavailable,
    resolve_dense_index_paths,
    sliding_windows,
)
from judicial_ai.analogues.knn import KNNAnalogueFinder
from judicial_ai.api.application import create_app
from judicial_ai.case_card import CaseCard
from judicial_ai.config import Settings
from judicial_ai.corpus.processed_repository import ProcessedRepository
from judicial_ai.models.bert import BertModelAdapter
from judicial_ai.schemas.case import CaseInput
from judicial_ai.services.analysis import AnalysisService
from tools.build_dense_index import build_dense_index


class FakeEncoder:
    def __init__(self, revision: str = DENSE_MODEL_REVISION):
        self.model_id = DENSE_MODEL_ID
        self.revision = revision
        self.chunk_overlap = 64
        self.encoded_texts: list[str] = []
        self.fail = False
        self.preprocessing = {
            "max_tokens": 128,
            "chunk_overlap_tokens": 64,
            "pooling": "token_weighted_mean_of_normalized_chunk_embeddings",
            "normalize_embeddings": True,
            "truncate": False,
        }

    def configure(self, preprocessing: dict) -> None:
        self.preprocessing = preprocessing

    def info(self) -> dict[str, object]:
        return {"status": "loaded" if self.encoded_texts else "lazy"}

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.fail:
            raise DenseUnavailable("synthetic model failure")
        self.encoded_texts.extend(texts)
        vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma": [-1.0, 0.0],
            "delta": [0.8, 0.2],
        }
        return np.asarray([vectors.get(text, [0.5, 0.5]) for text in texts])


def synthetic_record(index: int, narrative: str | None) -> dict:
    parts = ["2"] if index == 2 else ["3"]
    return {
        "schema_version": 1,
        "case_id": f"synthetic-case-{index}",
        "source": "synthetic",
        "source_year": 2026,
        "article_parts": parts,
        "other_articles": [],
        "process": {
            "special_order": index == 3,
            "pretrial_agreement": False,
            "art_64": False,
            "art_69": False,
            "art_73": False,
        },
        "facts": {
            "amount_rub": None,
            "episodes": None,
            "mitigating": [],
            "aggravating": [],
            "narrative": narrative,
        },
        "punishment": {
            "prison_months": 12 + index * 12,
            "fine_rub": None,
            "fine_multiple": False,
            "ban_on_positions": True,
            "ban_years": 2,
            "conditional": False,
            "confiscation": False,
        },
        "appeal": {
            "full_pair": True,
            "outcome": "synthetic-label-must-not-leak",
            "reason_codes": ["synthetic"],
        },
        "quality": {"synthetic": True},
        "split": "train",
    }


def write_repository(path: Path, narratives: list[str | None]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(synthetic_record(index, narrative), ensure_ascii=False)
            for index, narrative in enumerate(narratives)
        )
        + "\n",
        encoding="utf-8",
    )


def build_fake_index(tmp_path: Path, encoder: FakeEncoder | None = None):
    repository_path = tmp_path / "retrieval_cases.jsonl"
    index_path = tmp_path / "dense-index.npz"
    manifest_path = tmp_path / "model-card.json"
    write_repository(repository_path, ["alpha", "beta", "alpha", "gamma", "gamma", None])
    encoder = encoder or FakeEncoder()
    manifest = build_dense_index(
        repository_path, index_path, manifest_path, encoder=encoder
    )
    repository = ProcessedRepository(repository_path)
    return repository, index_path, manifest_path, encoder, manifest


def case_card(narrative: str | None = "alpha") -> CaseCard:
    return CaseCard.from_input(
        CaseInput(
            article_part="3",
            special_order=False,
            pretrial_agreement=False,
            art_64=False,
            art_69=False,
            mitigating=[],
            aggravating=[],
            narrative=narrative,
        )
    )


def test_dense_ranking_hard_filters_and_cohort_statistics(tmp_path: Path) -> None:
    repository, index_path, manifest_path, encoder, _ = build_fake_index(tmp_path)
    finder = DenseAnalogueFinder(
        repository.cases, index_path, manifest_path, encoder, dense_weight=0.7
    )
    structured_finder = KNNAnalogueFinder(repository.records)
    structured = structured_finder.search(case_card(), 2)
    result = finder.search(case_card(), structured, structured_finder, 2)

    assert result.result.ranking_available is True
    assert result.result.candidate_count == 4
    assert len(result.result.neighbors) == 2
    assert result.result.neighbors[0].record.source_order == 0
    assert result.result.neighbors[1].record.source_order == 1
    assert all(neighbor.record.source_order not in {2, 3} for neighbor in result.result.neighbors)
    distances = [neighbor.distance for neighbor in result.result.neighbors]
    assert distances == sorted(distances)
    assert result.distance_metric == "dense_structured_combined"
    assert result.fallback_reason is None


def test_dense_enables_ranking_without_structured_numeric_features(tmp_path: Path) -> None:
    repository, index_path, manifest_path, encoder, _ = build_fake_index(tmp_path)
    cases = tuple(
        replace(
            case,
            record=replace(
                case.record,
                amount_rub=None,
                episodes=None,
                mitigating_count=None,
                aggravating_count=None,
            ),
        )
        for case in repository.cases
    )
    records = tuple(case.record for case in cases)
    structured_finder = KNNAnalogueFinder(records)
    structured = structured_finder.search(case_card(), 2)
    assert structured.ranking_available is False

    finder = DenseAnalogueFinder(cases, index_path, manifest_path, encoder)
    result = finder.search(case_card(), structured, structured_finder, 2)
    assert result.result.ranking_available is True
    assert result.distance_metric == "cosine_scaled"
    assert result.result.neighbors[0].record.source_order == 0
    assert all(neighbor.distance is not None for neighbor in result.result.neighbors)


def test_index_and_manifest_have_no_labels_or_identifiers(tmp_path: Path) -> None:
    _, index_path, manifest_path, _, manifest = build_fake_index(tmp_path)
    index_path, _ = resolve_dense_index_paths(index_path, manifest_path)
    with np.load(index_path, allow_pickle=False) as payload:
        assert set(payload.files) == {"embeddings", "text_hashes"}
        assert payload["embeddings"].shape == (3, 2)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "synthetic-case" not in manifest_text
    assert "alpha" not in manifest_text
    assert "synthetic-label-must-not-leak" not in manifest_text
    assert "punishment" not in manifest
    assert "appeal" not in manifest
    assert "outcome" not in manifest
    assert manifest["model_revision"] == DENSE_MODEL_REVISION


def test_resume_reuses_text_hashes_and_revision_change_rebuilds(tmp_path: Path) -> None:
    repository_path = tmp_path / "retrieval_cases.jsonl"
    index_path = tmp_path / "dense-index.npz"
    manifest_path = tmp_path / "model-card.json"
    write_repository(repository_path, ["alpha", "beta", "alpha"])

    first_encoder = FakeEncoder()
    first = build_dense_index(
        repository_path, index_path, manifest_path, encoder=first_encoder
    )
    assert first["encoded_texts"] == 2
    assert first_encoder.encoded_texts == ["alpha", "beta"]

    resumed_encoder = FakeEncoder()
    resumed = build_dense_index(
        repository_path, index_path, manifest_path, encoder=resumed_encoder
    )
    assert resumed["encoded_texts"] == 0
    assert resumed["reused_texts"] == 2
    assert resumed_encoder.encoded_texts == []

    write_repository(repository_path, ["alpha", "delta", "alpha"])
    changed_encoder = FakeEncoder()
    changed = build_dense_index(
        repository_path, index_path, manifest_path, encoder=changed_encoder
    )
    assert changed["encoded_texts"] == 1
    assert changed_encoder.encoded_texts == ["delta"]

    new_revision_encoder = FakeEncoder(revision="f" * 40)
    rebuilt = build_dense_index(
        repository_path, index_path, manifest_path, encoder=new_revision_encoder
    )
    assert rebuilt["model_revision"] == "f" * 40
    assert rebuilt["encoded_texts"] == 2


def test_missing_narrative_index_and_model_fall_back(tmp_path: Path) -> None:
    repository, index_path, manifest_path, encoder, _ = build_fake_index(tmp_path)
    finder = DenseAnalogueFinder(repository.cases, index_path, manifest_path, encoder)
    structured_finder = KNNAnalogueFinder(repository.records)

    narrative_missing = finder.search(
        case_card(None), structured_finder.search(case_card(None), 2), structured_finder, 2
    )
    assert narrative_missing.status == "fallback"
    assert narrative_missing.fallback_reason == "narrative_missing"

    missing_index_finder = DenseAnalogueFinder(
        repository.cases,
        tmp_path / "missing.npz",
        tmp_path / "missing.json",
        FakeEncoder(),
    )
    index_missing = missing_index_finder.search(
        case_card(), structured_finder.search(case_card(), 2), structured_finder, 2
    )
    assert index_missing.fallback_reason == "index_unavailable"

    encoder.fail = True
    model_missing = finder.search(
        case_card(), structured_finder.search(case_card(), 2), structured_finder, 2
    )
    assert model_missing.status == "fallback"
    assert model_missing.fallback_reason == "model_unavailable"
    assert "model_unavailable" in model_missing.result.warnings[-1]


def test_dense_api_has_no_sensitive_fields_and_reports_status(tmp_path: Path) -> None:
    repository, index_path, manifest_path, encoder, _ = build_fake_index(tmp_path)
    dense_finder = DenseAnalogueFinder(
        repository.cases, index_path, manifest_path, encoder, dense_weight=1.0
    )
    settings = Settings(app_env="test", _env_file=None)
    service = AnalysisService(
        settings,
        repository.records,
        BertModelAdapter(None),
        dense_finder=dense_finder,
        corpus_source="processed_jsonl",
    )
    payload = {
        "case": {
            "article_part": "3",
            "special_order": False,
            "pretrial_agreement": False,
            "art_64": False,
            "art_69": False,
            "mitigating": [],
            "aggravating": [],
            "narrative": "alpha",
        },
        "draft_punishment": {
            "prison_months": 24,
            "conditional": False,
            "confiscation": False,
        },
        "top_k": 2,
    }
    with TestClient(create_app(settings, service)) as client:
        response = client.post("/api/v1/analyses", json=payload)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model_info"]["dense"]["repository"] == DENSE_MODEL_ID
        assert body["model_info"]["dense"]["revision"] == DENSE_MODEL_REVISION
        assert body["model_info"]["dense"]["status"] == "ready"
        assert body["model_info"]["knn"]["ranking_available"] is True
        assert body["punishment_summary"]["candidate_count"] == 4
        assert body["punishment_summary"]["prison_months"]["count"] == 4
        assert all(analogue["distance"] is not None for analogue in body["analogues"])
        assert all(
            not {"case_id", "id", "uid", "narrative", "appeal", "outcome"}.intersection(analogue)
            for analogue in body["analogues"]
        )
        assert "synthetic-case" not in response.text
        assert "synthetic-label-must-not-leak" not in response.text


def test_sliding_windows_cover_all_tokens_without_truncation() -> None:
    token_ids = list(range(300))
    chunks = sliding_windows(token_ids, window=128, overlap=64)
    covered = {token_id for chunk in chunks for token_id in chunk}
    assert covered == set(token_ids)
    assert all(len(chunk) <= 128 for chunk in chunks)
    assert chunks[0][64:] == chunks[1][:64]



class FakeTokenizerForChunks:
    def __call__(self, text: str, **kwargs):
        special = 2 if kwargs.get("add_special_tokens") else 0
        return {"input_ids": list(range(len(text) + special))}

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return "x" * len(token_ids)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2


class FakeSentenceTransformerForChunks:
    def __init__(self):
        self.chunks: list[str] = []

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        self.chunks.extend(texts)
        return np.tile(np.array([1.0, 0.0]), (len(texts), 1))


def test_dense_encoder_chunks_and_pools_without_model_download() -> None:
    encoder = DenseEncoder(max_tokens=64, chunk_overlap=8)
    model = FakeSentenceTransformerForChunks()
    encoder._model = model
    encoder._tokenizer = FakeTokenizerForChunks()
    encoder._max_tokens = 64

    vectors = encoder.encode(["x" * 200])
    assert vectors.shape == (1, 2)
    assert np.allclose(vectors[0], [1.0, 0.0])
    assert len(model.chunks) > 1
    assert all(len(chunk) + 2 <= 64 for chunk in model.chunks)
    assert sum(len(chunk) for chunk in model.chunks) >= 200
    assert model.chunks[-1]
    assert encoder.info()["status"] == "loaded"



class FakeLoadableModel:
    def __init__(self):
        self.max_seq_length = 8192
        self.tokenizer = FakeTokenizerForChunks()
        self.config = SimpleNamespace(max_position_embeddings=8192)

    def eval(self) -> None:
        return None

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        return np.tile(np.array([1.0, 0.0]), (len(texts), 1))


def test_failed_compatibility_does_not_cache_model_for_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeLoadableModel()
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda *args, **kwargs: model
    )
    encoder = DenseEncoder(max_tokens=512)
    encoder.configure(
        {
            "max_tokens": 1024,
            "chunk_overlap_tokens": 64,
            "pooling": "token_weighted_mean_of_normalized_chunk_embeddings",
            "normalize_embeddings": True,
            "truncate": False,
        }
    )
    with pytest.raises(DenseUnavailable):
        encoder.encode(["short"])
    assert encoder._model is None

    encoder.requested_max_tokens = 1024
    vectors = encoder.encode(["short"])
    assert vectors.shape == (1, 2)
    assert encoder._model is model
    assert encoder.preprocessing["max_tokens"] == 1024


def test_invalid_manifest_roots_use_structured_fallback(tmp_path: Path) -> None:
    repository, index_path, manifest_path, _, _ = build_fake_index(tmp_path)
    for invalid in ("null", "[]", '"invalid"'):
        manifest_path.write_text(invalid, encoding="utf-8")
        finder = DenseAnalogueFinder(
            repository.cases, index_path, manifest_path, FakeEncoder()
        )
        assert finder.info()["index_status"] == "unavailable"
        structured_finder = KNNAnalogueFinder(repository.records)
        result = finder.search(
            case_card(), structured_finder.search(case_card(), 2), structured_finder, 2
        )
        assert result.status == "fallback"
        assert result.fallback_reason == "index_unavailable"


def test_preprocessing_change_rebuilds_instead_of_reusing_old_window(
    tmp_path: Path,
) -> None:
    repository_path = tmp_path / "retrieval_cases.jsonl"
    index_path = tmp_path / "dense-index.npz"
    manifest_path = tmp_path / "model-card.json"
    write_repository(repository_path, ["alpha", "beta", "delta"])
    first_encoder = FakeEncoder()
    first = build_dense_index(
        repository_path, index_path, manifest_path, encoder=first_encoder
    )
    assert first["preprocessing"]["max_tokens"] == 128

    changed_encoder = FakeEncoder()
    changed_encoder.preprocessing["max_tokens"] = 64
    changed = build_dense_index(
        repository_path, index_path, manifest_path, encoder=changed_encoder
    )
    assert changed["preprocessing"]["max_tokens"] == 64
    assert changed["encoded_texts"] == 3
    assert changed_encoder.encoded_texts == ["alpha", "beta", "delta"]

    resumed_encoder = FakeEncoder()
    resumed_encoder.preprocessing["max_tokens"] = 64
    resumed = build_dense_index(
        repository_path, index_path, manifest_path, encoder=resumed_encoder
    )
    assert resumed["encoded_texts"] == 0
    assert resumed_encoder.encoded_texts == []


def test_publication_failure_preserves_previous_generation_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_path = tmp_path / "retrieval_cases.jsonl"
    index_path = tmp_path / "dense-index.npz"
    manifest_path = tmp_path / "model-card.json"
    write_repository(repository_path, ["alpha", "beta"])
    build_dense_index(repository_path, index_path, manifest_path, encoder=FakeEncoder())
    previous_pointer = manifest_path.read_bytes()

    write_repository(repository_path, ["alpha", "beta", "delta"])
    original_write = builder._atomic_write_json

    def fail_pointer(path: Path, payload: dict) -> None:
        if path == manifest_path:
            raise RuntimeError("synthetic publication interruption")
        original_write(path, payload)

    monkeypatch.setattr(builder, "_atomic_write_json", fail_pointer)
    with pytest.raises(RuntimeError, match="publication interruption"):
        build_dense_index(repository_path, index_path, manifest_path, encoder=FakeEncoder())
    assert manifest_path.read_bytes() == previous_pointer
    previous_embeddings, _ = builder.load_dense_index(
        index_path, manifest_path, DENSE_MODEL_ID, DENSE_MODEL_REVISION
    )
    assert len(previous_embeddings) == 2

    monkeypatch.setattr(builder, "_atomic_write_json", original_write)
    resumed_encoder = FakeEncoder()
    resumed = build_dense_index(
        repository_path, index_path, manifest_path, encoder=resumed_encoder
    )
    assert resumed["encoded_texts"] == 0
    assert resumed["unique_texts"] == 3
    assert resumed_encoder.encoded_texts == []


class FailOnSecondBatchEncoder(FakeEncoder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if self.calls == 2:
            raise DenseUnavailable("synthetic last-batch failure")
        return super().encode(texts)


def test_interrupted_batch_encoding_resumes_from_checkpoint(tmp_path: Path) -> None:
    repository_path = tmp_path / "retrieval_cases.jsonl"
    index_path = tmp_path / "dense-index.npz"
    manifest_path = tmp_path / "model-card.json"
    write_repository(repository_path, ["alpha", "beta", "gamma", "delta"])
    failing = FailOnSecondBatchEncoder()
    with pytest.raises(DenseUnavailable, match="last-batch"):
        build_dense_index(
            repository_path,
            index_path,
            manifest_path,
            encoder=failing,
            checkpoint_batch_size=2,
        )
    checkpoints = list(tmp_path.glob("dense-index-work/*/checkpoint.npz"))
    assert len(checkpoints) == 1

    resumed_encoder = FakeEncoder()
    resumed = build_dense_index(
        repository_path,
        index_path,
        manifest_path,
        encoder=resumed_encoder,
        checkpoint_batch_size=2,
    )
    assert resumed["encoded_texts"] == 2
    assert resumed["reused_texts"] == 2
    assert resumed["unique_texts"] == 4
    assert resumed_encoder.encoded_texts == ["gamma", "delta"]


def test_zero_dense_weight_returns_exact_structured_result(tmp_path: Path) -> None:
    repository, index_path, manifest_path, encoder, _ = build_fake_index(tmp_path)
    cases = tuple(
        replace(
            case,
            record=replace(
                case.record,
                amount_rub=0 if case.record.source_order == 5 else 1000 * (case.record.source_order + 1),
            ),
        )
        for case in repository.cases
    )
    records = tuple(case.record for case in cases)
    query = CaseCard.from_input(
        CaseInput(
            article_part="3",
            amount_rub=0,
            special_order=False,
            pretrial_agreement=False,
            art_64=False,
            art_69=False,
            mitigating=[],
            aggravating=[],
            narrative="alpha",
        )
    )
    structured_finder = KNNAnalogueFinder(records)
    structured = structured_finder.search(query, 1)
    assert structured.neighbors[0].record.source_order == 5

    finder = DenseAnalogueFinder(cases, index_path, manifest_path, encoder, dense_weight=0)
    encoded_before = list(encoder.encoded_texts)
    result = finder.search(query, structured, structured_finder, 1)
    assert result.result is structured
    assert result.result.neighbors[0].record.source_order == 5
    assert result.status == "structured"
    assert encoder.encoded_texts == encoded_before



def test_encoder_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="device"):
        DenseEncoder(device="tpu")



def test_null_process_mode_is_preserved_and_excluded_from_cohort() -> None:
    payload = synthetic_record(0, "alpha")
    payload["process"]["art_64"] = None
    case = ProcessedRepository._parse(payload, 0)
    assert case.record.art_64 is None
    finder = KNNAnalogueFinder((case.record,))
    assert finder.search(case_card(), 5).neighbors == ()
