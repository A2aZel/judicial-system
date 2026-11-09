import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judicial_ai.analogues.dense import (
    DENSE_MODEL_ID,
    DENSE_MODEL_REVISION,
    DenseEncoder,
    DenseUnavailable,
    Encoder,
    SHA256_RE,
    load_dense_index,
)
from judicial_ai.corpus.processed_repository import ProcessedRepository

logger = logging.getLogger(__name__)

POOLING = "token_weighted_mean_of_normalized_chunk_embeddings"


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _requested_preprocessing(encoder: Encoder) -> dict[str, Any]:
    configured = getattr(encoder, "preprocessing", {})
    max_tokens = getattr(encoder, "requested_max_tokens", configured.get("max_tokens"))
    return {
        "max_tokens": max_tokens,
        "chunk_overlap_tokens": encoder.chunk_overlap,
        "pooling": POOLING,
        "normalize_embeddings": True,
        "truncate": False,
    }


def _contract(encoder: Encoder, preprocessing: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": encoder.model_id,
        "model_revision": encoder.revision,
        "preprocessing": preprocessing,
    }


def _contract_digest(contract: dict[str, Any]) -> str:
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validated_rows(
    hashes: np.ndarray, embeddings: np.ndarray
) -> dict[str, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    hashes = np.asarray(hashes)
    if (
        embeddings.ndim != 2
        or hashes.ndim != 1
        or len(hashes) != embeddings.shape[0]
    ):
        raise ValueError("invalid embedding batch dimensions")
    result: dict[str, np.ndarray] = {}
    for text_hash, vector in zip(hashes, embeddings, strict=True):
        text_hash = str(text_hash)
        norm = float(np.linalg.norm(vector))
        if (
            not SHA256_RE.fullmatch(text_hash)
            or text_hash in result
            or not np.isfinite(vector).all()
            or not np.isfinite(norm)
            or norm == 0
        ):
            raise ValueError("invalid embedding batch")
        result[text_hash] = vector / norm
    return result


def _write_embeddings_npz(
    path: Path,
    embeddings: dict[str, np.ndarray],
    contract: dict[str, Any] | None = None,
) -> None:
    if not embeddings:
        raise ValueError("cannot write an empty embedding index")
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = list(embeddings)
    matrix = np.stack([embeddings[text_hash] for text_hash in hashes]).astype(np.float32)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        payload: dict[str, Any] = {
            "embeddings": matrix,
            "text_hashes": np.asarray(hashes, dtype=np.str_),
        }
        if contract is not None:
            payload["contract_json"] = np.asarray(
                json.dumps(contract, sort_keys=True, separators=(",", ":"))
            )
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_checkpoint(
    path: Path, contract: dict[str, Any]
) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"embeddings", "text_hashes", "contract_json"}:
            raise ValueError("invalid checkpoint payload")
        stored_contract = json.loads(str(payload["contract_json"].item()))
        if stored_contract != contract:
            raise ValueError("checkpoint contract mismatch")
        return _validated_rows(payload["text_hashes"], payload["embeddings"])


def build_dense_index(
    repository_path: Path,
    index_path: Path,
    manifest_path: Path,
    encoder: Encoder | None = None,
    batch_size: int = 4,
    checkpoint_batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    if checkpoint_batch_size < 1:
        raise ValueError("checkpoint_batch_size must be positive")
    repository = ProcessedRepository(repository_path)
    if not repository.records:
        raise ValueError("processed repository has no valid records")
    encoder = encoder or DenseEncoder(batch_size=batch_size, device=device)
    requested = _requested_preprocessing(encoder)
    contract = _contract(encoder, requested)
    encoder.configure(requested)
    documents: dict[str, str] = {}
    for case in repository.cases:
        if case.narrative and case.text_sha256:
            documents.setdefault(case.text_sha256, case.narrative)
    if not documents:
        raise ValueError("no narratives available for dense index")

    embeddings: dict[str, np.ndarray] = {}
    if manifest_path.exists():
        try:
            active_embeddings, active_preprocessing = load_dense_index(
                index_path,
                manifest_path,
                encoder.model_id,
                encoder.revision,
            )
            if active_preprocessing == requested:
                embeddings = active_embeddings
            else:
                logger.info("Dense preprocessing changed; active embeddings will be rebuilt")
        except (OSError, ValueError, KeyError, DenseUnavailable) as exc:
            logger.info("No compatible active dense index: %s", type(exc).__name__)

    root = manifest_path.parent
    checkpoint_path = (
        root
        / "dense-index-work"
        / _contract_digest(contract)
        / "checkpoint.npz"
    )
    try:
        checkpoint_embeddings = _load_checkpoint(checkpoint_path, contract)
        embeddings.update(checkpoint_embeddings)
        if checkpoint_embeddings:
            logger.info(
                "Dense checkpoint resumed: %d embeddings", len(checkpoint_embeddings)
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Dense checkpoint ignored: %s", type(exc).__name__)

    missing_hashes = [text_hash for text_hash in documents if text_hash not in embeddings]
    encoded_this_run = 0
    for offset in range(0, len(missing_hashes), checkpoint_batch_size):
        batch_hashes = missing_hashes[offset : offset + checkpoint_batch_size]
        vectors = encoder.encode([documents[text_hash] for text_hash in batch_hashes])
        if encoder.preprocessing != requested:
            raise ValueError("encoder preprocessing differs from requested contract")
        batch_embeddings = _validated_rows(
            np.asarray(batch_hashes, dtype=np.str_), np.asarray(vectors, dtype=np.float32)
        )
        embeddings.update(batch_embeddings)
        encoded_this_run += len(batch_hashes)
        _write_embeddings_npz(checkpoint_path, embeddings, contract)
        logger.info(
            "Dense checkpoint saved: %d/%d new embeddings",
            encoded_this_run,
            len(missing_hashes),
        )

    current_hashes = [text_hash for text_hash in documents if text_hash in embeddings]
    if len(current_hashes) != len(documents):
        raise ValueError("dense index is incomplete")
    current_embeddings = {text_hash: embeddings[text_hash] for text_hash in current_hashes}
    generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex
    generation_dir = root / "dense-index-generations" / generation
    generation_index = generation_dir / "dense-index.npz"
    generation_manifest = generation_dir / "model-card.json"
    _write_embeddings_npz(generation_index, current_embeddings)
    manifest = {
        "schema_version": 1,
        "generation": generation,
        "model_id": encoder.model_id,
        "model_revision": encoder.revision,
        "model_trust_remote_code": False,
        "device": getattr(encoder, "device", "cpu"),
        "repository_records": len(repository.cases),
        "source_records": len(repository.cases) + repository.skipped_records,
        "skipped_records": repository.skipped_records,
        "documents": sum(bool(case.text_sha256) for case in repository.cases),
        "unique_texts": len(current_hashes),
        "reused_texts": len(current_hashes) - encoded_this_run,
        "encoded_texts": encoded_this_run,
        "vector_dimensions": int(current_embeddings[current_hashes[0]].shape[0]),
        "distance": "cosine",
        "preprocessing": requested,
        "source_sha256": _file_sha256(repository_path),
        "index_sha256": _file_sha256(generation_index),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(generation_manifest, manifest)
    load_dense_index(
        generation_index,
        generation_manifest,
        encoder.model_id,
        encoder.revision,
    )
    pointer = {
        "schema_version": 1,
        "pointer": "dense_index_generation_v1",
        "generation": generation,
        "index_path": generation_index.relative_to(root).as_posix(),
        "manifest_path": generation_manifest.relative_to(root).as_posix(),
        "model_id": encoder.model_id,
        "model_revision": encoder.revision,
        "source_records": manifest["source_records"],
        "repository_records": manifest["repository_records"],
        "unique_texts": manifest["unique_texts"],
    }
    _atomic_write_json(manifest_path, pointer)
    checkpoint_path.unlink(missing_ok=True)
    logger.info(
        "Dense generation published: generation=%s records=%d unique_texts=%d encoded=%d reused=%d",
        generation,
        len(repository.cases),
        len(current_hashes),
        encoded_this_run,
        len(current_hashes) - encoded_this_run,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pinned dense retrieval index")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("corpus/processed/retrieval_cases.jsonl"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("corpus/processed/dense-index.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus/processed/model-card.json"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--checkpoint-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        parser.error("batch-size must be between 1 and 32")
    if not 64 <= args.max_tokens <= 8192:
        parser.error("max-tokens must be between 64 and 8192")
    if not 1 <= args.checkpoint_batch_size <= 128:
        parser.error("checkpoint-batch-size must be between 1 and 128")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    encoder = DenseEncoder(batch_size=args.batch_size, max_tokens=args.max_tokens, device=args.device)
    manifest = build_dense_index(
        args.repository,
        args.index,
        args.manifest,
        encoder=encoder,
        batch_size=args.batch_size,
        checkpoint_batch_size=args.checkpoint_batch_size,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "model_id": manifest["model_id"],
                "model_revision": manifest["model_revision"],
                "generation": manifest["generation"],
                "unique_texts": manifest["unique_texts"],
                "encoded_texts": manifest["encoded_texts"],
                "reused_texts": manifest["reused_texts"],
                "vector_dimensions": manifest["vector_dimensions"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
