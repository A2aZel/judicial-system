from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import pair_preparation as pairs

COLLECTION = ROOT / "corpus" / "regional" / "raw" / "collection-2026-11-10"
TARGET = COLLECTION / "unified-case-index.jsonl"
METADATA = COLLECTION / "metadata.jsonl"
LEDGER = COLLECTION / "deduplication-ledger.jsonl"
PAIR_STATUS = ROOT / "corpus" / "pairs" / "moscow" / "raw" / "pair-status.jsonl"
WORK = ROOT / "work" / "recovery-unified-case-index-2026-11-10"
RECOVERED = WORK / "recovered-new-748.jsonl"
AMBIGUOUS = WORK / "ambiguous-existing-42.jsonl"
REPORT = WORK / "recovery-report.json"
EXPECTED_ORIGINAL_SHA = "2f125827345583c2acdbc7ba9431ec486564ec01a07b45721e3298d1a9d6c4d2"
EXPECTED_NEW = 748
EXPECTED_OLD = 252


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def new_records() -> list[dict[str, Any]]:
    raw_paths = {}
    for path in (COLLECTION / "raw").iterdir():
        raw_paths.setdefault(sha256(path), path)
    text_paths = {}
    for path in (COLLECTION / "texts").iterdir():
        text_paths.setdefault(sha256(path), path)
    selected = [record for record in read_jsonl(LEDGER) if record["identity"].get("raw_sha256") in raw_paths]
    if len(selected) != EXPECTED_NEW or len(raw_paths) != EXPECTED_NEW:
        raise ValueError(f"Raw-backed selection mismatch: {len(selected)} records, {len(raw_paths)} hashes")
    rebuilt = []
    for record in selected:
        source_identity = record["identity"]
        raw_path = raw_paths[source_identity["raw_sha256"]]
        text_path = text_paths[source_identity["text_sha256"]]
        text = text_path.read_text(encoding="utf-8")
        case_key = next(key for key in source_identity["case_keys"] if key.startswith("first_case:"))
        court, number = case_key.removeprefix("first_case:").split("|", 1)
        decision_date = pairs.canonical_date(pairs.parse_russian_date(text[:8000])) or None
        identity = {
            "schema_version": 1,
            "provenance": record["source_metadata"],
            "source": record["source"],
            "text_path": text_path.relative_to(ROOT).as_posix(),
            "raw_path": raw_path.relative_to(ROOT).as_posix(),
            "text_sha256": source_identity["text_sha256"],
            "normalized_text_sha256": source_identity["normalized_text_sha256"],
            "raw_sha256": source_identity["raw_sha256"],
            "text_characters": len(text),
            "source_urls": [],
            "document_urls": [],
            "case_urls": [],
            "judicial_uids": source_identity.get("judicial_uids") or [],
            "unverified_judicial_uid_values": [],
            "case_keys": source_identity["case_keys"],
            "weak_case_keys": [f"first_number:{court}|{number}"],
            "first_court": court,
            "first_case_number": number,
            "first_decision_date": decision_date,
            "sentence_date_corroboration": "rebuilt_from_text_header" if decision_date else "unknown",
            "official_bound_sentence_date": None,
            "official_bound_sentence_date_corroboration": "unknown",
            "identity_status": source_identity["identity_status"],
            "identity_errors": source_identity.get("identity_errors") or [],
            "quality_errors": source_identity.get("quality_errors") or [],
            "document_type": source_identity["document_type"],
            "classification_basis": source_identity["classification_basis"],
            "article_290_evidence_basis": source_identity["article_290_evidence_basis"],
            "identity_link_verified": True,
            "identity_link_method": "deduplication_ledger_raw_sha256",
            "identity_link": {"method": "deduplication_ledger_raw_sha256", "source_metadata": record["source_metadata"]},
            "identity_link_evidence_sha256": source_identity["raw_sha256"],
            "identity_evidence_path": None,
            "full_relevant_merits_text": len(text) >= 1000,
            "fingerprint": [],
            "record_id": source_identity["record_id"],
            "decoded_text_sha256": source_identity["text_sha256"],
            "raw_path_status": "verified_existing_project_artifact",
            "rebuild": {
                "selection_status": "proven_by_raw_sha256",
                "original_metadata_status": "damaged_single_synthetic_line",
                "fingerprint_status": "not_recomputed",
            },
        }
        rebuilt.append({
            "schema_version": 1,
            "underlying_case_id": source_identity["record_id"],
            "case_keys": source_identity["case_keys"],
            "origin": "rebuilt_new_case",
            "identity": identity,
        })
    return rebuilt


def ambiguous_records() -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in read_jsonl(PAIR_STATUS):
        groups[record["verdict_group_id"]].append(record)
    representatives = [
        min(records, key=lambda record: (pairs.STATUS_RANK[record["pair_status"]], record["case_id"]))
        for records in groups.values()
    ]
    representatives.sort(key=lambda record: (pairs.STATUS_RANK[record["pair_status"]], record["verdict_group_id"]))
    if len(representatives) != EXPECTED_OLD + 42:
        raise ValueError(f"Unexpected old candidate count: {len(representatives)}")
    excluded = representatives[EXPECTED_OLD:]
    if len(excluded) != 42:
        raise ValueError(f"Expected 42 ambiguous candidates, got {len(excluded)}")
    return [
        {
            "case_id": record["case_id"],
            "verdict_group_id": record["verdict_group_id"],
            "source": record["source"],
            "pair_status": record["pair_status"],
            "identity_errors": record.get("identity_errors") or [],
            "artifacts": record["artifacts"],
            "selection_status": "ambiguous_not_included",
        }
        for record in excluded
    ]


def verify_new(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != EXPECTED_NEW:
        raise ValueError("New record count mismatch")
    ids = [record["underlying_case_id"] for record in records]
    normalized = []
    for record in records:
        identity = record["identity"]
        text_path = ROOT / identity["text_path"]
        if not text_path.is_file() or sha256(text_path) != identity["text_sha256"]:
            raise ValueError(f"Text SHA mismatch: {text_path}")
        raw_path = ROOT / identity["raw_path"] if identity.get("raw_path") else None
        if raw_path and sha256(raw_path) != identity["raw_sha256"]:
            raise ValueError(f"Raw SHA mismatch: {raw_path}")
        normalized.append(identity["normalized_text_sha256"])
    if len(set(ids)) != EXPECTED_NEW or len(set(normalized)) != EXPECTED_NEW:
        raise ValueError("New IDs or normalized texts are not unique")
    return {
        "records": len(records),
        "unique_underlying_case_ids": len(set(ids)),
        "unique_normalized_texts": len(set(normalized)),
        "source_counts": dict(sorted(collections.Counter(record["identity"]["source"] for record in records).items())),
        "document_type_counts": dict(sorted(collections.Counter(record["identity"]["document_type"] for record in records).items())),
        "origin_counts": dict(sorted(collections.Counter(record["origin"] for record in records).items())),
        "text_sha256_verified": len(records),
        "raw_sha256_verified": sum(bool(record["identity"].get("raw_path")) for record in records),
        "raw_path_nulls": sum(not record["identity"].get("raw_path") for record in records),
        "identity_complete_for_recovery": sum(
            all(record["identity"].get(field) for field in ("first_court", "first_case_number", "first_decision_date"))
            for record in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    records = new_records()
    checks = verify_new(records)
    checks["identity_incomplete_for_recovery"] = EXPECTED_NEW - checks["identity_complete_for_recovery"]
    if args.command == "build":
        write_jsonl(RECOVERED, records)
        write_jsonl(AMBIGUOUS, ambiguous_records())
    target_sha = sha256(TARGET)
    if target_sha != "d85f6390b446420d53de7c28b753c70a4e1a0ff5de73724598bf9ef21482c554":
        raise ValueError("Main target changed unexpectedly")
    report = {
        "schema_version": 1,
        "status": "proven_original_1000_not_reconstructible",
        "can_rebuild_proven_1000": False,
        "exact_original_sha256_restored": False,
        "expected_original_sha256": EXPECTED_ORIGINAL_SHA,
        "main_target_untouched": True,
        "main_target_sha256": target_sha,
        "proven_existing_records": 0,
        "recovered_new_records": EXPECTED_NEW,
        "ambiguous_existing_candidates": 42,
        "checks": checks,
        "recovered_new_path": str(RECOVERED.relative_to(ROOT)),
        "recovered_new_sha256": sha256(RECOVERED),
        "ambiguous_path": str(AMBIGUOUS.relative_to(ROOT)),
        "ambiguous_sha256": sha256(AMBIGUOUS),
        "ambiguous_pair_status_counts": dict(sorted(collections.Counter(record["pair_status"] for record in read_jsonl(AMBIGUOUS)).items())),
        "diagnostic_selection_rule": "The 42 records are a triage list selected by STATUS_RANK and verdict_group_id. This is not evidence of the original 252 selection and must not be used to claim a complete original 1000.",
        "underlying_case_id_policy": "Rebuilt records use record_id as underlying_case_id; original underlying_case_id values are not asserted.",
        "remaining_mismatches": [
            "The identities and exact subset of the original 252 existing cases cannot be proven from the remaining sources.",
            "Original underlying_case_id values for those 252 cases are unavailable.",
            "The original unified-case-index SHA256 cannot be reproduced.",
            f"{checks['identity_incomplete_for_recovery']} new records have no first_decision_date; null values are preserved instead of inferred dates.",
        ],
        "processed_dataset_or_other_raw_files_changed": False,
    }
    write_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
