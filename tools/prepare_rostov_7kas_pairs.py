from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.regional_courts import canonical_number, date_forms, numbers_in_text

DEFAULT_INDEX = ROOT / "corpus" / "regional" / "raw" / "collection-2026-11-10" / "unified-case-index.jsonl"
DEFAULT_METADATA = ROOT / "corpus" / "regional" / "raw" / "collection-2026-11-10" / "metadata.jsonl"
DEFAULT_OUTPUT = ROOT / "corpus" / "pairs" / "rostov-7kas"
TARGET_SOURCES = ("rostov", "7kas")
LATER_TYPES = {"appeal_against_verdict", "cassation_against_verdict", "cassation_on_merits_appeal"}
NEW_TRIAL_TYPES = {"appellate_verdict"}
OUTCOMES = ("unchanged", "changed", "cancelled", "remanded", "unknown")
PAIR_CONTRACT = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write(path, payload.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def normalize_court(value: Any) -> str:
    text = compact_text(str(value or "")).replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", text).strip()


def court_matches(text: str, court: Any) -> bool:
    ignored = {"суд", "суда", "город", "города", "г", "области", "областной", "края", "краевой"}
    normalized_court = normalize_court(court)
    stems = [token[:5] for token in normalized_court.split() if token not in ignored and not token.isdigit()]
    compact = normalize_court(text)
    if "районн" in normalized_court and "районн" in compact:
        stems = [stem for stem in stems if stem != "город"]
    return bool(stems) and all(stem in compact for stem in stems)


def header_decision_date_matches(header: str, court: Any, decision_date: Any) -> bool:
    court_tokens = [token for token in normalize_court(court).split() if token not in {"суд", "суда", "город", "города", "г"}]
    if not court_tokens:
        return False
    court_position = normalize_court(header).find(court_tokens[0])
    if court_position < 0:
        return False
    compact = compact_text(header)
    positions = [compact.find(compact_text(form)) for form in date_forms(decision_date)]
    return any(0 <= position < court_position for position in positions)


def parse_date(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            pass
    return None


def operative_text(text: str) -> str:
    normalized = compact_text(text)
    markers = list(re.finditer(r"п\s*о\s*с\s*т\s*а\s*н\s*о\s*в\s*и\s*л|о\s*п\s*р\s*е\s*д\s*е\s*л\s*и\s*л", normalized))
    return normalized[markers[-1].start():] if markers else ""


def extract_outcome(text: str) -> str:
    operative = operative_text(text)
    if not operative:
        return "unknown"
    outcomes = set()
    pattern = r"(?<![а-я])(приговор|судебные\s+решения)(?![а-я])(.{0,2200}?)(оставить\s+без\s+изменени(?:я|й)|изменить|отменить)"
    for match in re.finditer(pattern, operative, re.S):
        body = match.group(2)
        if re.search(r"постановлен|арест|мер[аы]\s+пресечения|восстанов|срок\w*\s+обжалован|обжалования\s+срока", body):
            continue
        action = re.sub(r"\s+", " ", match.group(3))
        outcomes.add("unchanged" if action.startswith("оставить") else "changed" if action == "изменить" else "cancelled")
    remand = (
        r"апелляционн\w*\s+(?:определени\w*|постановлени\w*)"
        r".{0,2200}?отменить.{0,900}?нов\w*\s+(?:судебн\w*\s+|апелляционн\w*\s+)?рассмотрени\w*"
    )
    if re.search(remand, operative, re.S):
        outcomes.add("remanded")
    if re.search(r"кассационн\w*\s+(?:жалоб\w*|представлени\w*).{0,2200}?оставить\s+без\s+удовлетворения", operative, re.S):
        outcomes.add("unchanged")
    for outcome in ("cancelled", "changed", "remanded", "unchanged"):
        if outcome in outcomes:
            return outcome
    return "unknown"


def first_case_keys(row: dict[str, Any]) -> set[str]:
    return {key for key in row["identity"].get("case_keys") or [] if key.startswith("first_case:")}


def document_role(row: dict[str, Any]) -> str:
    document_type = row["identity"].get("document_type")
    if document_type == "trial_verdict":
        return "sentence"
    if document_type in LATER_TYPES:
        return "appeal"
    return "other"


def is_new_trial_document(document: dict[str, Any]) -> bool:
    identity = document["row"]["identity"]
    if identity.get("document_type") in NEW_TRIAL_TYPES:
        return True
    if document["role"] != "appeal":
        return False
    intro = compact_text(document["text"][:6000])
    return "на апелляционный приговор" in intro and ("новый приговор" in intro or "нового приговора" in intro)


def fingerprint_sudact(root: Path = ROOT) -> int:
    docs_path = root / "corpus/sudact/docs.jsonl"
    fingerprints = {}
    for metadata in read_jsonl(docs_path):
        text_path = root / "corpus/sudact/docs" / (str(metadata["id"]) + ".txt")
        if not text_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8-sig")
        if len(text) == metadata.get("chars"):
            fingerprints[str(metadata["id"])] = sha256_file(text_path)
    atomic_json(root / "corpus/sudact/text-sha256.json", {"schema_version": 1, "algorithm": "sha256", "documents": fingerprints})
    return len(fingerprints)


def sudact_sentence_rows(root: Path, index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs_path = root / "corpus/sudact/docs.jsonl"
    hashes_path = root / "corpus/sudact/text-sha256.json"
    if not docs_path.exists() or not hashes_path.exists():
        return []
    documents = read_jsonl(docs_path)
    fingerprints = json.loads(hashes_path.read_text(encoding="utf-8")).get("documents") or {}
    evidence_sha = sha256_file(docs_path)
    targets = [row for row in index_rows if row["identity"].get("source") in TARGET_SOURCES and document_role(row) == "appeal"]
    result = []
    for target in targets:
        identity = target["identity"]
        expected = canonical_number(identity.get("first_case_number"))
        if not expected:
            continue
        matches = []
        for metadata in documents:
            title = metadata.get("title") or ""
            text_path = root / "corpus/sudact/docs" / (str(metadata.get("id")) + ".txt")
            expected_sha = fingerprints.get(str(metadata.get("id")))
            if (
                not compact_text(title).startswith("приговор")
                or expected not in numbers_in_text(title)
                or not re.match(r"https://sudact\.ru/regular/doc/", metadata.get("url") or "")
                or not text_path.is_file()
                or not expected_sha
                or sha256_file(text_path) != expected_sha
            ):
                continue
            text = text_path.read_text(encoding="utf-8-sig")
            header = text[:2500]
            squashed = re.sub(r"\s+", "", text.lower())
            if (
                len(text) != metadata.get("chars")
                or not court_matches(header, identity.get("first_court"))
                or expected not in numbers_in_text(header)
                or not header_decision_date_matches(header, identity.get("first_court"), identity.get("first_decision_date"))
                or "установил" not in squashed
                or "приговорил" not in squashed
            ):
                continue
            matches.append((metadata, text_path, text, expected_sha))
        if len(matches) != 1:
            continue
        metadata, text_path, text, expected_sha = matches[0]
        linked = dict(identity)
        linked.update({
            "record_id": "sudact:" + str(metadata["id"]),
            "source": "sudact",
            "document_type": "trial_verdict",
            "text_path": text_path.relative_to(root).as_posix(),
            "text_sha256": expected_sha,
            "normalized_text_sha256": expected_sha,
            "raw_path": None,
            "raw_sha256": None,
            "identity_evidence_path": docs_path.relative_to(root).as_posix(),
            "identity_link_evidence_sha256": evidence_sha,
            "identity_status": "proven",
            "identity_link_verified": True,
            "identity_link_method": "sudact_title_number_and_header_identity",
            "identity_errors": [],
            "quality_errors": [],
            "full_relevant_merits_text": True,
            "judicial_uids": [],
            "document_urls": [metadata["url"]],
            "text_characters": len(text),
            "artifact_origin": "sudact_text_only",
        })
        result.append({"underlying_case_id": "sudact:" + str(metadata["id"]), "identity": linked})
    return result


class DocumentStore:
    def __init__(self, root: Path, metadata_by_text_sha: dict[str, dict[str, Any]]):
        self.root = root
        self.metadata_by_text_sha = metadata_by_text_sha
        self.cache: dict[str, dict[str, Any]] = {}

    def get(self, row: dict[str, Any]) -> dict[str, Any]:
        identity = row["identity"]
        record_id = str(identity["record_id"])
        if record_id in self.cache:
            return self.cache[record_id]
        metadata = self.metadata_by_text_sha.get(identity.get("text_sha256"), {})
        text_path = self.root / identity["text_path"]
        raw_path = self.root / identity["raw_path"] if identity.get("raw_path") else None
        evidence_path = self.root / identity["identity_evidence_path"] if identity.get("identity_evidence_path") else None
        errors = []
        text = ""
        if not text_path.exists():
            errors.append("sentence_text_missing" if document_role(row) == "sentence" else "appeal_text_missing")
        else:
            if sha256_file(text_path) != identity.get("text_sha256"):
                errors.append("text_sha256_mismatch")
            else:
                text = text_path.read_text(encoding="utf-8-sig")
        text_only = identity.get("artifact_origin") == "sudact_text_only"
        if raw_path is None:
            if not text_only:
                errors.append("raw_artifact_missing")
        elif not raw_path.exists():
            errors.append("raw_artifact_missing")
        elif sha256_file(raw_path) != identity.get("raw_sha256"):
            errors.append("raw_sha256_mismatch")
        if evidence_path is None or not evidence_path.exists():
            errors.append("identity_evidence_missing")
        elif sha256_file(evidence_path) != identity.get("identity_link_evidence_sha256"):
            errors.append("identity_evidence_sha256_mismatch")
        if not identity.get("full_relevant_merits_text"):
            errors.append("full_text_not_confirmed")
        if identity.get("identity_status") != "proven" or not identity.get("identity_link_verified"):
            errors.append("identity_not_proven")
        errors.extend(identity.get("identity_errors") or [])
        errors.extend(identity.get("quality_errors") or [])
        document_type = identity.get("document_type")
        role = document_role(row)
        decision_date = metadata.get("decision_date") or identity.get("first_decision_date")
        own_number = metadata.get("case_number") or identity.get("first_case_number")
        own_court = metadata.get("court") or identity.get("first_court")
        artifact = {
            "record_id": record_id,
            "source": identity.get("source"),
            "document_type": document_type,
            "text_path": identity.get("text_path"),
            "text_sha256": identity.get("text_sha256"),
            "raw_path": identity.get("raw_path"),
            "raw_sha256": identity.get("raw_sha256"),
            "identity_evidence_path": identity.get("identity_evidence_path"),
            "identity_evidence_sha256": identity.get("identity_link_evidence_sha256"),
            "source_url": (identity.get("document_urls") or [None])[0],
            "court": own_court,
            "case_number": own_number,
            "decision_date": decision_date,
        }
        document = {
            "row": row,
            "metadata": metadata,
            "text": text,
            "artifact": artifact,
            "errors": list(dict.fromkeys(errors)),
            "role": role,
            "outcome": extract_outcome(text) if role == "appeal" and text else "unknown",
        }
        self.cache[record_id] = document
        document["errors"] = list(dict.fromkeys(document["errors"] + self._semantic_errors(document)))
        return document

    def _semantic_errors(self, document: dict[str, Any]) -> list[str]:
        row = document["row"]
        identity = row["identity"]
        text = document["text"]
        role = document["role"]
        if role == "other":
            return []
        errors = []
        if len(text) < 1000:
            return ["text_too_short"]
        compact = compact_text(text)
        squashed = re.sub(r"\s+", "", text.lower())
        if not court_matches(text, identity.get("first_court")):
            errors.append("first_court_mismatch")
        sentence_date = identity.get("first_decision_date")
        forms = date_forms(sentence_date)
        if not forms:
            errors.append("sentence_date_missing")
        elif not any(compact_text(form) in compact for form in forms):
            errors.append("sentence_date_mismatch")
        if role == "sentence":
            if "приговор" not in squashed or "приговорил" not in squashed:
                errors.append("sentence_type_not_confirmed")
            expected = canonical_number(identity.get("first_case_number"))
            if (not expected or expected not in numbers_in_text(text)) and "identity_evidence_sha256_mismatch" in document["errors"]:
                errors.append("sentence_number_mismatch")
        else:
            document_type = identity.get("document_type")
            instance_marker = "апелляцион" if document_type == "appeal_against_verdict" else "кассацион"
            merits_marker = "апелляцион" in squashed and "приговор" in squashed
            if document_type == "cassation_on_merits_appeal":
                type_confirmed = instance_marker in squashed and merits_marker
            else:
                type_confirmed = instance_marker in squashed and "наприговор" in squashed
            if not type_confirmed:
                errors.append("appeal_type_not_confirmed")
            if document["outcome"] == "unknown":
                errors.append("outcome_not_confirmed")
            appeal_date = document["artifact"].get("decision_date")
            appeal_forms = date_forms(appeal_date)
            if not appeal_forms:
                errors.append("appeal_date_missing")
            elif not any(compact_text(form) in compact for form in appeal_forms):
                errors.append("appeal_date_mismatch")
            if not document["artifact"].get("court") or not document["artifact"].get("case_number"):
                errors.append("appeal_identity_missing")
        return errors


def verify_pair(sentence: dict[str, Any], appeal: dict[str, Any]) -> list[str]:
    errors = list(sentence["errors"]) + list(appeal["errors"])
    sentence_row = sentence["row"]
    appeal_row = appeal["row"]
    if sentence["role"] != "sentence" or appeal["role"] != "appeal":
        errors.append("unsupported_pair_types")
    if is_new_trial_document(sentence) or is_new_trial_document(appeal):
        errors.append("new_trial_verdict_not_accepted")
    if not first_case_keys(sentence_row) & first_case_keys(appeal_row):
        errors.append("first_case_identity_mismatch")
    sentence_date = parse_date(sentence_row["identity"].get("first_decision_date"))
    appealed_date = parse_date(appeal_row["identity"].get("first_decision_date"))
    if not sentence_date or not appealed_date or sentence_date != appealed_date:
        errors.append("sentence_date_identity_mismatch")
    sentence_uids = set(sentence_row["identity"].get("judicial_uids") or [])
    appeal_uids = set(appeal_row["identity"].get("judicial_uids") or [])
    if sentence_uids and appeal_uids and sentence_uids.isdisjoint(appeal_uids):
        errors.append("judicial_uid_mismatch")
    appeal_date = parse_date(appeal["artifact"].get("decision_date"))
    if sentence_date and appeal_date and sentence_date > appeal_date:
        errors.append("sentence_after_appeal")
    return list(dict.fromkeys(errors))


def stable_id(prefix: str, parts: Iterable[Any]) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def verdict_group_id(row: dict[str, Any]) -> str:
    identity = row["identity"]
    return stable_id("vg_", (sorted(first_case_keys(row)), identity.get("first_decision_date"), "trial_verdict"))


def pair_id(sentence: dict[str, Any] | None, appeal: dict[str, Any] | None, source: dict[str, Any]) -> str:
    sentence_id = sentence["artifact"]["record_id"] if sentence else "missing"
    appeal_id = appeal["artifact"]["record_id"] if appeal else "missing"
    source_id = "" if sentence and appeal else source["artifact"]["record_id"]
    return stable_id("pair_", (sentence_id, appeal_id, source_id))


def event_key(document: dict[str, Any]) -> tuple[Any, ...]:
    row = document["row"]
    identity = row["identity"]
    normalized_sha = identity.get("normalized_text_sha256") or identity.get("text_sha256")
    if document["role"] == "appeal":
        uids = tuple(sorted(identity.get("judicial_uids") or []))
        own = ("uid", uids) if uids else (
            "court_number",
            normalize_court(document["artifact"].get("court")),
            canonical_number(document["artifact"].get("case_number")),
        )
        return own + (document["artifact"].get("decision_date"), identity.get("document_type"), normalized_sha)
    return (
        "sentence",
        tuple(sorted(first_case_keys(row))),
        identity.get("first_decision_date"),
        identity.get("document_type"),
        normalized_sha,
    )


def make_status(
    source: dict[str, Any],
    sentence: dict[str, Any] | None,
    appeal: dict[str, Any] | None,
    status: str,
    errors: Iterable[str] = (),
) -> dict[str, Any]:
    primary = appeal or sentence or source
    return {
        "schema_version": 1,
        "processing_contract": PAIR_CONTRACT,
        "pair_id": pair_id(sentence, appeal, source),
        "verdict_group_id": verdict_group_id(source["row"]),
        "source": source["artifact"]["source"],
        "source_record_id": source["artifact"]["record_id"],
        "event_identity": {
            "judicial_uids": primary["row"]["identity"].get("judicial_uids") or [],
            "court": primary["artifact"].get("court"),
            "case_number": primary["artifact"].get("case_number"),
            "decision_date": primary["artifact"].get("decision_date"),
            "document_type": primary["artifact"].get("document_type"),
        },
        "sentence_artifact": sentence["artifact"] if sentence else None,
        "appeal_artifact": appeal["artifact"] if appeal else None,
        "pair_status": status,
        "outcome": appeal["outcome"] if appeal else "unknown",
        "identity_errors": list(dict.fromkeys(errors)),
    }


def matching_rows(source_row: dict[str, Any], rows: list[dict[str, Any]], wanted_role: str) -> list[dict[str, Any]]:
    source_identity = source_row["identity"]
    return [
        row for row in rows
        if row["identity"].get("record_id") != source_identity.get("record_id")
        and document_role(row) == wanted_role
        and first_case_keys(row) & first_case_keys(source_row)
        and row["identity"].get("first_decision_date") == source_identity.get("first_decision_date")
    ]


def previous_confirmed(previous: dict[str, dict[str, Any]], current: dict[str, Any]) -> bool:
    old = previous.get(current["pair_id"])
    if (
        not old
        or old.get("pair_status") != "confirmed_pair"
        or old.get("processing_contract") != PAIR_CONTRACT
        or old.get("outcome") != current.get("outcome")
        or current["identity_errors"]
    ):
        return False
    for role in ("sentence_artifact", "appeal_artifact"):
        old_artifact = old.get(role) or {}
        new_artifact = current.get(role) or {}
        if old_artifact.get("text_sha256") != new_artifact.get("text_sha256") or old_artifact.get("raw_sha256") != new_artifact.get("raw_sha256"):
            return False
    return True


def prepare_pairs(
    root: Path = ROOT,
    index_path: Path = DEFAULT_INDEX,
    metadata_path: Path = DEFAULT_METADATA,
    output_dir: Path = DEFAULT_OUTPUT,
    extra_index_paths: Iterable[Path] = (),
    resume: bool = False,
) -> dict[str, Any]:
    index_rows = read_jsonl(index_path)
    repository_rows = index_rows + [row for path in extra_index_paths for row in read_jsonl(path)] + sudact_sentence_rows(root, index_rows)
    metadata_rows = read_jsonl(metadata_path)
    metadata_by_text_sha = {row.get("text_sha256"): row for row in metadata_rows if row.get("text_sha256")}
    store = DocumentStore(root, metadata_by_text_sha)
    target_rows = [row for row in index_rows if row["identity"].get("source") in TARGET_SOURCES]
    target_documents = [store.get(row) for row in target_rows]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for document in target_documents:
        groups[event_key(document)].append(document)
    duplicate_ids = set()
    for documents in groups.values():
        documents.sort(key=lambda item: item["artifact"]["record_id"])
        duplicate_ids.update(item["artifact"]["record_id"] for item in documents[1:])
    status_path = output_dir / "pair-status.jsonl"
    previous = {row["pair_id"]: row for row in read_jsonl(status_path)} if resume else {}
    statuses = []
    emitted_pairs = set()
    for source in sorted(target_documents, key=lambda item: (item["artifact"]["source"], item["artifact"]["record_id"])):
        source_id = source["artifact"]["record_id"]
        source_row = source["row"]
        if source_id in duplicate_ids:
            sentence = source if source["role"] == "sentence" else None
            appeal = source if source["role"] == "appeal" else None
            statuses.append(make_status(source, sentence, appeal, "duplicate", ["duplicate_source_event"]))
            write_jsonl(status_path, statuses)
            continue
        if is_new_trial_document(source):
            statuses.append(make_status(source, None, None, "rejected_new_trial_verdict", ["new_trial_verdict_not_accepted"]))
            write_jsonl(status_path, statuses)
            continue
        if source["role"] == "other":
            statuses.append(make_status(source, None, None, "identity_failed", ["unsupported_document_type"]))
            write_jsonl(status_path, statuses)
            continue
        wanted_role = "sentence" if source["role"] == "appeal" else "appeal"
        counterparts = [store.get(row) for row in matching_rows(source_row, repository_rows, wanted_role)]
        if not counterparts:
            sentence = source if source["role"] == "sentence" else None
            appeal = source if source["role"] == "appeal" else None
            status = "missing_sentence" if wanted_role == "sentence" else "missing_appeal"
            statuses.append(make_status(source, sentence, appeal, status, source["errors"]))
            write_jsonl(status_path, statuses)
            continue
        for counterpart in sorted(counterparts, key=lambda item: item["artifact"]["record_id"]):
            sentence = counterpart if wanted_role == "sentence" else source
            appeal = counterpart if wanted_role == "appeal" else source
            current_pair_id = pair_id(sentence, appeal, source)
            if current_pair_id in emitted_pairs:
                continue
            emitted_pairs.add(current_pair_id)
            errors = verify_pair(sentence, appeal)
            current = make_status(source, sentence, appeal, "identity_failed" if errors else "confirmed_pair", errors)
            if previous_confirmed(previous, current):
                current = previous[current["pair_id"]]
            statuses.append(current)
            write_jsonl(status_path, statuses)
    summary = build_summary(target_documents, duplicate_ids, statuses)
    atomic_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", summary)
    return summary


def build_summary(target_documents: list[dict[str, Any]], duplicate_ids: set[str], statuses: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = Counter(document["artifact"]["source"] for document in target_documents)
    status_counts = Counter((status["source"], status["pair_status"]) for status in statuses)
    identity_errors = Counter(error for status in statuses for error in status["identity_errors"] if error != "duplicate_source_event")
    outcomes_confirmed = Counter(status["outcome"] for status in statuses if status["pair_status"] == "confirmed_pair")
    outcomes_available = Counter(
        (document["artifact"]["source"], document["outcome"]) for document in target_documents
        if document["role"] == "appeal" and not is_new_trial_document(document)
        and document["artifact"]["record_id"] not in duplicate_ids
    )
    artifact_valid = Counter(
        document["artifact"]["source"] for document in target_documents
        if not any(
            error.endswith("sha256_mismatch") or error.endswith("artifact_missing")
            or error.endswith("text_missing") or error.endswith("evidence_missing")
            for error in document["errors"]
        )
    )
    by_source = {}
    for source in TARGET_SOURCES:
        by_source[source] = {
            "candidates": candidates[source],
            "unique_candidates": candidates[source] - sum(1 for document in target_documents if document["artifact"]["source"] == source and document["artifact"]["record_id"] in duplicate_ids),
            "sha_verified_artifacts": artifact_valid[source],
            "statuses": {status: status_counts[(source, status)] for status in sorted({key[1] for key in status_counts if key[0] == source})},
            "outcomes_available_later_acts": {outcome: outcomes_available[(source, outcome)] for outcome in OUTCOMES if outcomes_available[(source, outcome)]},
            "outcomes_confirmed_pairs": {outcome: sum(1 for status in statuses if status["source"] == source and status["pair_status"] == "confirmed_pair" and status["outcome"] == outcome) for outcome in OUTCOMES},
        }
        by_source[source]["outcomes_confirmed_pairs"] = {key: value for key, value in by_source[source]["outcomes_confirmed_pairs"].items() if value}
    return {
        "schema_version": 1,
        "candidates": sum(candidates.values()),
        "unique_candidates": len(target_documents) - len(duplicate_ids),
        "confirmed_pairs": sum(1 for status in statuses if status["pair_status"] == "confirmed_pair"),
        "text_only_provenance_pairs": sum(
            1 for status in statuses
            if status["pair_status"] == "confirmed_pair"
            and (status.get("sentence_artifact") or {}).get("source") == "sudact"
            and not (status.get("sentence_artifact") or {}).get("raw_path")
        ),
        "missing_sentence": sum(1 for status in statuses if status["pair_status"] == "missing_sentence"),
        "missing_appeal": sum(1 for status in statuses if status["pair_status"] == "missing_appeal"),
        "identity_failed": sum(1 for status in statuses if status["pair_status"] == "identity_failed"),
        "records_with_identity_errors": sum(1 for status in statuses if status["identity_errors"] and status["pair_status"] != "duplicate"),
        "duplicates": len(duplicate_ids),
        "rejected_new_trial_verdict": sum(1 for status in statuses if status["pair_status"] == "rejected_new_trial_verdict"),
        "identity_errors": dict(sorted(identity_errors.items())),
        "outcomes_confirmed_pairs": {outcome: outcomes_confirmed[outcome] for outcome in OUTCOMES if outcomes_confirmed[outcome]},
        "by_source": by_source,
        "processed_dataset_or_index_changed": False,
        "network_document_requests": 0,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Пары `rostov` и `7kas`",
        "",
        "Полная пара требует два независимых полных текста, действительные SHA и совпадающую идентичность. Цитата приговора в более позднем акте не засчитывается. Процессуальные акты и приговоры нового рассмотрения исключены.",
        "",
        "Новые запросы к карточкам и документам не выполнялись: официальные правила обоих источников закрывают соответствующие пути. Обработаны только локальные артефакты.",
        "",
        "## Агрегаты по источникам",
        "",
        "| Источник | Кандидаты | Уникальные события | SHA проверен | Confirmed | Missing sentence | Missing appeal | Identity failed | Duplicates | Новый приговор отклонён |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in TARGET_SOURCES:
        item = summary["by_source"][source]
        statuses = item["statuses"]
        lines.append(
            f"| {source} | {item['candidates']} | {item['unique_candidates']} | {item['sha_verified_artifacts']} | "
            f"{statuses.get('confirmed_pair', 0)} | {statuses.get('missing_sentence', 0)} | {statuses.get('missing_appeal', 0)} | "
            f"{statuses.get('identity_failed', 0)} | {statuses.get('duplicate', 0)} | {statuses.get('rejected_new_trial_verdict', 0)} |"
        )
    lines += [
        "",
        f"**Кандидаты: {summary['candidates']}; confirmed pairs: {summary['confirmed_pairs']}; duplicates: {summary['duplicates']}; записей с identity errors: {summary['records_with_identity_errors']}.**",
        "",
        f"Пар с полным текстом приговора из СудАкта без сохранённого исходного HTML: **{summary.get('text_only_provenance_pairs', 0)}**. SHA текста и метаданных проверен; это ограничение происхождения отмечено отдельно.",
        "",
        "## Identity errors",
        "",
        "| Код | Количество |",
        "|---|---:|",
    ]
    if summary["identity_errors"]:
        for error, count in summary["identity_errors"].items():
            lines.append(f"| {error} | {count} |")
    else:
        lines.append("| Нет | 0 |")
    for title, key in (("Outcome допустимых апелляционных/кассационных актов", "outcomes_available_later_acts"), ("Outcome подтверждённых пар", "outcomes_confirmed_pairs")):
        lines += ["", f"## {title}", "", "| Источник | Outcome | Количество |", "|---|---|---:|"]
        found = False
        for source in TARGET_SOURCES:
            outcomes = summary["by_source"][source][key]
            for outcome in OUTCOMES:
                if outcomes.get(outcome):
                    lines.append(f"| {source} | {outcome} | {outcomes[outcome]} |")
                    found = True
        if not found:
            lines.append("| Все | Нет | 0 |")
    lines += [
        "",
        "`remanded` означает отмену апелляционного акта с новым апелляционным рассмотрением без прямой отмены приговора. Исход извлекается только после маркера резолютивной части.",
        "",
        "Этот pipeline не меняет processed dataset/index. Включение подтверждённых пар выполняется отдельной пересборкой dataset с повторной проверкой SHA.",
        "",
    ]
    atomic_write(path, "\n".join(lines).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Строгая подготовка локальных пар rostov/7kas")
    parser.add_argument("command", choices=("run", "report", "fingerprint-sudact"), nargs="?", default="run")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extra-index", type=Path, action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "fingerprint-sudact":
        print(json.dumps({"fingerprinted_texts": fingerprint_sudact(args.root)}, ensure_ascii=False))
        return
    if args.command == "report":
        statuses = read_jsonl(args.output / "pair-status.jsonl")
        summary_path = args.output / "summary.json"
        if not summary_path.exists():
            raise SystemExit("Сначала выполните run: summary.json отсутствует")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        write_report(args.output / "report.md", summary)
        print(json.dumps({"statuses": len(statuses), "report": str(args.output / "report.md")}, ensure_ascii=False))
        return
    summary = prepare_pairs(args.root, args.index, args.metadata, args.output, args.extra_index, args.resume)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
