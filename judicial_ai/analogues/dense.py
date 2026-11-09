import hashlib
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from judicial_ai.analogues.knn import KNNAnalogueFinder, KNNResult, Neighbor
from judicial_ai.case_card import CaseCard
from judicial_ai.corpus.processed_repository import ProcessedCase

logger = logging.getLogger(__name__)

DENSE_MODEL_ID = "deepvk/USER-bge-m3"
DENSE_MODEL_REVISION = "0cc6cfe48e260fb0474c753087a69369e88709ae"
INDEX_PAYLOAD_KEYS = {"embeddings", "text_hashes"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class DenseUnavailable(RuntimeError):
    pass


class Encoder(Protocol):
    model_id: str
    revision: str
    chunk_overlap: int
    preprocessing: dict[str, Any]

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def configure(self, preprocessing: dict[str, Any]) -> None: ...

    def info(self) -> dict[str, object]: ...


def sliding_windows(
    token_ids: list[int], window: int, overlap: int
) -> list[list[int]]:
    if window < 1 or overlap < 0 or overlap >= window:
        raise ValueError("invalid token window")
    if not token_ids:
        return []
    step = window - overlap
    return [
        token_ids[start : start + window]
        for start in range(0, len(token_ids), step)
        if start < len(token_ids)
    ]


class DenseEncoder:
    def __init__(
        self,
        model_id: str = DENSE_MODEL_ID,
        revision: str = DENSE_MODEL_REVISION,
        cache_dir: Path | None = None,
        batch_size: int = 4,
        chunk_overlap: int = 64,
        max_tokens: int = 1024,
        device: str = "cpu",
    ):
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self.chunk_overlap = chunk_overlap
        self.requested_max_tokens = max_tokens
        self._model: Any = None
        self._tokenizer: Any = None
        self._max_tokens: int | None = None
        self._contract_max_tokens: int | None = None
        self._status = "lazy"
        self.preprocessing: dict[str, Any] = {}

    def configure(self, preprocessing: dict[str, Any]) -> None:
        max_tokens = preprocessing.get("max_tokens")
        overlap = preprocessing.get("chunk_overlap_tokens")
        if (
            type(max_tokens) is not int
            or max_tokens < 16
            or type(overlap) is not int
            or overlap != self.chunk_overlap
            or preprocessing.get("pooling")
            != "token_weighted_mean_of_normalized_chunk_embeddings"
            or preprocessing.get("normalize_embeddings") is not True
            or preprocessing.get("truncate") is not False
        ):
            raise DenseUnavailable("incompatible dense preprocessing contract")
        self._contract_max_tokens = max_tokens

    def info(self) -> dict[str, object]:
        return {"status": self._status}

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        try:
            self._load()
            embeddings = []
            for index, text in enumerate(texts, start=1):
                embeddings.append(self._encode_text(text))
                if index % 10 == 0 or index == len(texts):
                    logger.info("Dense encoding progress: %d/%d texts", index, len(texts))
            self._status = "loaded"
            return np.stack(embeddings).astype(np.float32)
        except Exception as exc:
            self._status = "error"
            raise DenseUnavailable("dense encoder is unavailable") from exc

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        options: dict[str, Any] = {}
        if self.cache_dir:
            options["cache_folder"] = str(self.cache_dir)
        model = SentenceTransformer(
            self.model_id,
            revision=self.revision,
            device=self.device,
            trust_remote_code=False,
            **options,
        )
        model.eval()
        tokenizer = model.tokenizer
        limits = [int(model.max_seq_length), self.requested_max_tokens]
        tokenizer_limit = int(getattr(tokenizer, "model_max_length", 0))
        if 0 < tokenizer_limit < 1_000_000:
            limits.append(tokenizer_limit)
        model_config = getattr(model, "config", None)
        position_limit = int(getattr(model_config, "max_position_embeddings", 0) or 0)
        if position_limit > 0:
            limits.append(position_limit)
        max_tokens = min(limits)
        if self._contract_max_tokens is not None and self._contract_max_tokens != max_tokens:
            raise DenseUnavailable("index and runtime token limits differ")
        preprocessing = {
            "max_tokens": self._contract_max_tokens or max_tokens,
            "chunk_overlap_tokens": self.chunk_overlap,
            "pooling": "token_weighted_mean_of_normalized_chunk_embeddings",
            "normalize_embeddings": True,
            "truncate": False,
        }
        self._model = model
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self.preprocessing = preprocessing

    def _encode_text(self, text: str) -> np.ndarray:
        if not text.strip():
            raise DenseUnavailable("empty narrative")
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        token_ids = list(encoded["input_ids"])
        if not token_ids:
            raise DenseUnavailable("narrative has no tokens")
        special_tokens = (
            int(self._tokenizer.num_special_tokens_to_add(pair=False))
            if hasattr(self._tokenizer, "num_special_tokens_to_add")
            else 2
        )
        max_tokens = self._contract_max_tokens or self._max_tokens
        if max_tokens is None:
            raise DenseUnavailable("token limit is unavailable")
        window = max_tokens - special_tokens - 8
        if window <= self.chunk_overlap:
            raise DenseUnavailable("token window is too small")
        overlap = min(self.chunk_overlap, window // 4)
        chunks = sliding_windows(token_ids, window, overlap)
        chunk_texts = [
            self._tokenizer.decode(
                chunk,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for chunk in chunks
        ]
        for chunk_text in chunk_texts:
            verified = self._tokenizer(
                chunk_text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )
            if len(verified["input_ids"]) > max_tokens:
                raise DenseUnavailable("decoded chunk exceeds token limit")
        vectors = self._model.encode(
            chunk_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise DenseUnavailable("invalid encoder output")
        weights = np.asarray([len(chunk) for chunk in chunks], dtype=np.float32)
        pooled = np.average(vectors, axis=0, weights=weights)
        norm = float(np.linalg.norm(pooled))
        if not math.isfinite(norm) or norm == 0 or not np.isfinite(vectors).all():
            raise DenseUnavailable("non-finite embedding")
        return pooled / norm


@dataclass(frozen=True, slots=True)
class DenseSearch:
    result: KNNResult
    status: str
    fallback_reason: str | None
    distance_metric: str | None


class DenseAnalogueFinder:
    def __init__(
        self,
        cases: tuple[ProcessedCase, ...],
        index_path: Path,
        manifest_path: Path,
        encoder: Encoder,
        dense_weight: float = 0.7,
    ):
        self._cases = {case.record.source_order: case for case in cases}
        self.encoder = encoder
        self.dense_weight = dense_weight
        self._embeddings: dict[str, np.ndarray] = {}
        self._index_status = "unavailable"
        self._status = "index_unavailable"
        self._fallback_reason: str | None = "index_unavailable"
        self._distance_metric: str | None = None
        try:
            self._embeddings, preprocessing = load_dense_index(
                index_path, manifest_path, encoder.model_id, encoder.revision
            )
            self.encoder.configure(preprocessing)
            self._index_status = "ready"
            self._status = "ready"
            self._fallback_reason = None
        except (OSError, ValueError, KeyError, TypeError, DenseUnavailable) as exc:
            logger.warning("Dense index unavailable: %s", type(exc).__name__)

    def info(self) -> dict[str, object]:
        return {
            "repository": self.encoder.model_id,
            "revision": self.encoder.revision,
            "status": self._status,
            "encoder_status": self.encoder.info()["status"],
            "index_status": self._index_status,
            "fallback_reason": self._fallback_reason,
            "distance_metric": self._distance_metric,
            "dense_weight": self.dense_weight,
        }

    def search(
        self,
        case: CaseCard,
        structured: KNNResult,
        structured_finder: KNNAnalogueFinder,
        top_k: int,
    ) -> DenseSearch:
        if self.dense_weight == 0:
            self._status = "structured"
            self._fallback_reason = None
            self._distance_metric = (
                "euclidean_robust_scaled" if structured.ranking_available else None
            )
            return DenseSearch(
                structured, "structured", None, self._distance_metric
            )
        if not structured.cohort:
            return self._fallback(structured, "empty_cohort")
        if not case.narrative or not case.narrative.strip():
            return self._fallback(structured, "narrative_missing")
        if self._index_status != "ready":
            return self._fallback(structured, "index_unavailable")

        eligible: list[tuple[Any, np.ndarray]] = []
        missing_embeddings = 0
        for record in structured.cohort:
            processed = self._cases.get(record.source_order)
            embedding = (
                self._embeddings.get(processed.text_sha256)
                if processed and processed.text_sha256
                else None
            )
            if embedding is None:
                missing_embeddings += 1
            else:
                eligible.append((record, embedding))
        if not eligible:
            return self._fallback(structured, "no_cohort_embeddings")
        try:
            query = self.encoder.encode([case.narrative])[0]
        except (DenseUnavailable, ValueError, IndexError) as exc:
            logger.warning("Dense query unavailable: %s", type(exc).__name__)
            return self._fallback(structured, "model_unavailable")

        if structured.ranking_available and len(structured.neighbors) == len(structured.cohort):
            all_structured = structured
        else:
            all_structured = structured_finder.search(case, len(structured.cohort))
        structured_distances = {
            neighbor.record.source_order: neighbor.distance
            for neighbor in all_structured.neighbors
        }
        neighbors: list[Neighbor] = []
        combined = all_structured.ranking_available
        for record, embedding in eligible:
            cosine = float(np.clip(1.0 - np.dot(query, embedding), 0.0, 2.0))
            structured_distance = structured_distances.get(record.source_order)
            if combined and structured_distance is not None:
                structured_component = 1.0 - math.exp(-structured_distance)
                distance = (
                    self.dense_weight * (cosine / 2.0)
                    + (1.0 - self.dense_weight) * structured_component
                )
            else:
                distance = cosine / 2.0
            neighbors.append(Neighbor(record, distance))
        neighbors.sort(key=lambda neighbor: (neighbor.distance, neighbor.record.source_order))
        warnings = list(structured.warnings if structured.ranking_available else ())
        if missing_embeddings:
            warnings.append(
                f"Для {missing_embeddings} дел cohort нет совместимого dense embedding; они не участвуют в ranking, но остаются в cohort-статистике."
            )
        result = KNNResult(
            tuple(neighbors[:top_k]),
            structured.cohort,
            len(structured.cohort),
            True,
            structured.active_numeric_features,
            structured.active_boolean_features,
            tuple(warnings),
        )
        self._status = "ready"
        self._fallback_reason = None
        self._distance_metric = (
            "dense_structured_combined"
            if combined and self.dense_weight < 1.0
            else "cosine_scaled"
        )
        return DenseSearch(result, "ready", None, self._distance_metric)

    def _fallback(self, structured: KNNResult, reason: str) -> DenseSearch:
        self._status = "fallback"
        self._fallback_reason = reason
        self._distance_metric = None
        warnings = list(structured.warnings)
        warnings.append(f"Dense ranking недоступен ({reason}); использован structured/unranked fallback.")
        return DenseSearch(
            KNNResult(
                structured.neighbors,
                structured.cohort,
                structured.candidate_count,
                structured.ranking_available,
                structured.active_numeric_features,
                structured.active_boolean_features,
                tuple(warnings),
            ),
            "fallback",
            reason,
            "euclidean_robust_scaled" if structured.ranking_available else None,
        )


def _read_manifest(path: Path) -> dict[str, Any]:
    import json

    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DenseUnavailable("invalid dense manifest JSON") from exc
    if not isinstance(manifest, dict):
        raise DenseUnavailable("dense manifest root must be an object")
    return manifest


def resolve_dense_index_paths(
    index_path: Path, manifest_path: Path
) -> tuple[Path, Path]:
    pointer = _read_manifest(manifest_path)
    if pointer.get("pointer") is None:
        return index_path, manifest_path
    if pointer.get("pointer") != "dense_index_generation_v1":
        raise DenseUnavailable("unsupported dense index pointer")
    root = manifest_path.parent.resolve()
    relative_index = pointer.get("index_path")
    relative_manifest = pointer.get("manifest_path")
    if not isinstance(relative_index, str) or not isinstance(relative_manifest, str):
        raise DenseUnavailable("invalid dense generation paths")
    resolved_index = (root / relative_index).resolve()
    resolved_manifest = (root / relative_manifest).resolve()
    try:
        resolved_index.relative_to(root)
        resolved_manifest.relative_to(root)
    except ValueError as exc:
        raise DenseUnavailable("dense generation path escapes repository") from exc
    return resolved_index, resolved_manifest


def load_dense_index(
    index_path: Path,
    manifest_path: Path,
    model_id: str,
    revision: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    index_path, manifest_path = resolve_dense_index_paths(index_path, manifest_path)
    manifest = _read_manifest(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("model_id") != model_id
        or manifest.get("model_revision") != revision
    ):
        raise DenseUnavailable("dense index model or schema mismatch")
    with index_path.open("rb") as handle:
        actual_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_digest != manifest.get("index_sha256"):
        raise DenseUnavailable("dense index digest mismatch")
    with np.load(index_path, allow_pickle=False) as payload:
        if set(payload.files) != INDEX_PAYLOAD_KEYS:
            raise DenseUnavailable("unsafe dense index payload")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        hashes = payload["text_hashes"]
    if (
        embeddings.ndim != 2
        or hashes.ndim != 1
        or embeddings.shape[0] != len(hashes)
        or embeddings.shape[0] == 0
        or not np.isfinite(embeddings).all()
        or manifest.get("vector_dimensions") != embeddings.shape[1]
    ):
        raise DenseUnavailable("invalid dense index dimensions")
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise DenseUnavailable("dense preprocessing manifest is missing")
    result: dict[str, np.ndarray] = {}
    for text_hash, embedding in zip(hashes, embeddings, strict=True):
        if not SHA256_RE.fullmatch(str(text_hash)) or str(text_hash) in result:
            raise DenseUnavailable("invalid or duplicate text hash")
        norm = float(np.linalg.norm(embedding))
        if not math.isfinite(norm) or norm == 0:
            raise DenseUnavailable("zero embedding")
        result[str(text_hash)] = embedding / norm
    return result, preprocessing
