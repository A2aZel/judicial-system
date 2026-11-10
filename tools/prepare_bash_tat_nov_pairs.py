from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_training_dataset import read_json, read_jsonl, sha256_file
from tools.mosgorsud_open_data import (
    appealed_number_references,
    compact_text,
    document_court_match,
    extract_case_numbers,
    extract_text,
    number_identity_variants,
    parse_date,
)
from tools.regional_courts import (
    AccessBlocked,
    CaptchaError,
    Court,
    DownloadError,
    allowed_by_robots,
    canonical_number,
    date_forms,
    extract_outcome,
    parse_robots,
    safe_get,
)

INDEX = ROOT / "corpus" / "regional" / "raw" / "collection-2026-11-10" / "unified-case-index.jsonl"
METADATA = ROOT / "corpus" / "regional" / "raw" / "collection-2026-11-10" / "metadata.jsonl"
OUTPUT = ROOT / "corpus" / "pairs" / "bash-tat-nov"
SOURCES = ("bashkortostan", "tatarstan", "novosibirsk")
EXPECTED_COUNTS = {"bashkortostan": 129, "tatarstan": 99, "novosibirsk": 70}
COURTS = {
    "bashkortostan": Court("bashkortostan", "Верховный Суд Республики Башкортостан", "vs.bkr.sudrf.ru"),
    "tatarstan": Court("tatarstan", "Верховный Суд Республики Татарстан", "vs.tat.sudrf.ru"),
    "novosibirsk": Court("novosibirsk", "Новосибирский областной суд", "oblsud.nsk.sudrf.ru"),
}
OUTCOMES = ("unchanged", "changed", "cancelled", "unknown")
TERMINAL_ACCESS = {"robots_blocked"}


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def project_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Артефакт вне корня проекта") from exc
    return path


def artifact_from_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": identity.get("text_path") or identity.get("path"),
        "sha256": identity.get("text_sha256") or identity.get("sha256"),
        "raw_path": identity.get("raw_path"),
        "raw_sha256": identity.get("raw_sha256"),
        "text_characters": identity.get("text_characters"),
        "source": identity.get("source"),
        "record_id": identity.get("record_id"),
        "document_url": (identity.get("document_urls") or [identity.get("document_url")])[0],
    }


def validate_artifact(artifact: dict[str, Any] | None, root: Path) -> tuple[str | None, list[str]]:
    if not artifact:
        return None, ["artifact_missing"]
    path = project_path(root, artifact.get("path"))
    expected = artifact.get("sha256")
    if path is None or not path.is_file():
        return None, ["text_artifact_missing"]
    if not expected:
        return None, ["text_sha256_missing"]
    if sha256_file(path) != expected:
        return None, ["text_sha256_mismatch"]
    raw_path = project_path(root, artifact.get("raw_path"))
    if artifact.get("raw_path") and (raw_path is None or not raw_path.is_file()):
        return None, ["raw_artifact_missing"]
    if artifact.get("raw_path") and not artifact.get("raw_sha256"):
        return None, ["raw_sha256_missing"]
    if raw_path is not None and raw_path.is_file() and artifact.get("raw_sha256"):
        if sha256_file(raw_path) != artifact["raw_sha256"]:
            return None, ["raw_sha256_mismatch"]
    return path.read_text(encoding="utf-8"), []


def validate_identity_evidence(identity: dict[str, Any], root: Path) -> list[str]:
    path_value = identity.get("identity_evidence_path")
    expected = identity.get("identity_link_evidence_sha256")
    if not path_value:
        return [] if identity.get("identity_link_method") in {"official_open_data_uid_attachment", "direct_document_text"} else ["identity_evidence_missing"]
    path = project_path(root, path_value)
    if path is None or not path.is_file():
        return ["identity_evidence_missing"]
    if not expected or sha256_file(path) != expected:
        return ["identity_evidence_sha256_mismatch"]
    return []


def official_uids(identity: dict[str, Any]) -> set[str]:
    if not identity.get("identity_link_verified"):
        return set()
    return {str(value).upper() for value in identity.get("judicial_uids") or [] if value}


def verdict_group_id(record: dict[str, Any], sentence_date: Any = None) -> str:
    identity = record["identity"]
    uids = official_uids(identity)
    if len(uids) == 1:
        value = "uid:" + next(iter(uids))
    else:
        number = canonical_number(identity.get("first_case_number"))
        canonical = "|".join(map(str, number)) if number else "missing"
        date = sentence_date or identity.get("first_decision_date") or ""
        value = f"case:{identity.get('first_court') or ''}|{canonical}|{date}|trial_verdict"
    return "vg_" + digest(value)[:24]


def appeal_event(metadata: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "court": metadata.get("court") or "",
        "number": metadata.get("case_number") or "",
        "date": metadata.get("decision_date") or "",
        "type": identity.get("document_type") or metadata.get("judicial_act_kind") or "",
        "document_id": metadata.get("document_id") or identity.get("record_id"),
    }


def event_key(event: dict[str, Any]) -> tuple[str, str, str, str]:
    return event["court"].strip().lower(), event["number"].strip(), event["date"].strip(), event["type"].strip()


def appeal_event_id(source: str, event: dict[str, Any]) -> str:
    return "ae_" + digest(source + "|" + "|".join(event_key(event)))[:24]


def case_keys(record: dict[str, Any]) -> set[str]:
    return {key for key in record.get("case_keys") or [] if key.startswith("first_case:")}


def usable_record(record: dict[str, Any], document_type: str) -> bool:
    identity = record.get("identity") or {}
    return (
        identity.get("document_type") == document_type
        and identity.get("identity_status") == "proven"
        and identity.get("identity_link_verified") is True
        and identity.get("full_relevant_merits_text") is True
        and bool(case_keys(record))
    )


def pair_identity_metadata(record: dict[str, Any], metadata: dict[str, Any], sentence_record: dict[str, Any] | None = None) -> dict[str, Any]:
    identity = record["identity"]
    result = dict(metadata)
    result["lower_number"] = identity.get("first_case_number")
    result["lower_court"] = identity.get("first_court")
    result["sentence_date"] = identity.get("first_decision_date")
    result["identity_link_verified"] = identity.get("identity_link_verified")
    result["appeal_uids"] = sorted(official_uids(identity))
    if sentence_record:
        sentence_identity = sentence_record["identity"]
        result["sentence_date"] = result["sentence_date"] or sentence_identity.get("first_decision_date")
        result["sentence_uids"] = sorted(official_uids(sentence_identity))
    return result


def verify_pair(metadata: dict[str, Any], sentence_text: str, appeal_text: str) -> list[str]:
    if len(sentence_text) < 1000 or len(appeal_text) < 1000:
        return ["text_too_short"]
    errors: list[str] = []
    sentence_compact = compact_text(sentence_text)
    appeal_compact = compact_text(appeal_text)
    if "апелляционныйприговор" in sentence_compact or "кассационныйприговор" in sentence_compact:
        errors.append("new_trial_verdict_not_sentence")
    if "приговор" not in sentence_compact or "приговорил" not in sentence_compact:
        errors.append("sentence_type_not_confirmed")
    if "апелляционныйприговор" in appeal_compact or "кассационныйприговор" in appeal_compact:
        errors.append("new_trial_verdict_not_review_act")
    if (
        not any(marker in appeal_compact for marker in ("апелляцион", "кассацион"))
        or "наприговор" not in appeal_compact
        or extract_outcome(appeal_text) == "unknown"
    ):
        errors.append("appeal_type_not_confirmed")
    parsed_sentence_date = parse_date(metadata.get("sentence_date"))
    default_year = parsed_sentence_date.year if parsed_sentence_date else None
    expected = number_identity_variants(metadata.get("lower_number"), default_year)
    sentence_numbers = extract_case_numbers(sentence_text, default_year)
    lower_references = appealed_number_references(appeal_text, default_year)
    if not expected:
        errors.append("sentence_number_missing")
    elif expected.isdisjoint(sentence_numbers):
        errors.append("sentence_number_mismatch")
    elif lower_references and expected.isdisjoint(lower_references):
        errors.append("sentence_number_conflict")
    elif not lower_references and not metadata.get("identity_link_verified"):
        errors.append("appeal_number_mismatch")
    lower_court = metadata.get("lower_court")
    if not document_court_match(sentence_text, lower_court) or not document_court_match(appeal_text, lower_court):
        errors.append("lower_court_mismatch")
    sentence_date = metadata.get("sentence_date")
    forms = date_forms(sentence_date)
    if not forms:
        errors.append("sentence_date_missing")
    elif not all(any(compact_text(form) in compact for form in forms) for compact in (sentence_compact, appeal_compact)):
        errors.append("sentence_date_mismatch")
    appeal_forms = date_forms(metadata.get("decision_date"))
    if not appeal_forms:
        errors.append("appeal_date_missing")
    elif not any(compact_text(form) in appeal_compact for form in appeal_forms):
        errors.append("appeal_date_mismatch")
    sentence_uids = set(metadata.get("sentence_uids") or [])
    appeal_uids = set(metadata.get("appeal_uids") or [])
    if sentence_uids and appeal_uids and sentence_uids.isdisjoint(appeal_uids):
        errors.append("uid_mismatch")
    return list(dict.fromkeys(errors))


def check_url(url: str, source: str, rules: Iterable[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != COURTS[source].host:
        raise DownloadError("Разрешён только прямой HTTPS-документ официального суда")
    if "/search" in parsed.path.lower() or "name_op=r" in parsed.query.lower() or "name_op=sf" in parsed.query.lower():
        raise DownloadError("Поисковые запросы запрещены")
    if not allowed_by_robots(url, rules):
        raise AccessBlocked("Прямой документ запрещён robots.txt")


def explicit_artifact(
    metadata: dict[str, Any], role: str, source: str, event_id: str, output_dir: Path,
    access: dict[str, Any], pause: float, opener: Any = None, root: Path = ROOT,
) -> dict[str, Any] | None:
    saved = metadata.get(role + "_artifact")
    if isinstance(saved, dict) and saved.get("path"):
        return saved
    url = metadata.get(role + "_url")
    if not isinstance(url, str) or not url:
        return None
    state = access[source]
    if state.get("status") != "allowed":
        return None
    rules = state.get("rules") or []
    check_url(url, source, rules)
    key = digest(url)[:20]
    raw_path = output_dir / "artifacts" / f"{event_id}-{role}-{key}.bin"
    text_path = raw_path.with_suffix(".txt")
    if not raw_path.exists():
        body = safe_get(url, COURTS[source], rules, pause=pause, opener=opener)
        atomic_write(raw_path, body)
    if not text_path.exists():
        text = extract_text(raw_path)
        if not text.strip():
            raise DownloadError("Извлечён пустой текст")
        atomic_write(text_path, text.encode("utf-8"))
    raw_sha = sha256_file(raw_path)
    expected_raw = metadata.get(role + "_raw_sha256")
    if expected_raw and raw_sha != expected_raw:
        raise DownloadError("SHA256 прямого документа не совпал")
    return {
        "path": str(text_path.relative_to(root)),
        "sha256": sha256_file(text_path),
        "raw_path": str(raw_path.relative_to(root)),
        "raw_sha256": raw_sha,
        "text_characters": len(text_path.read_text(encoding="utf-8")),
        "source": source,
        "record_id": None,
        "document_url": url,
        "identity_link_method": "direct_document_text",
    }


def check_access(output_dir: Path, pause: float = 4, refresh: bool = False, opener: Any = None) -> dict[str, Any]:
    path = output_dir / "access.json"
    saved = read_json(path, {})
    if saved and not refresh:
        return saved
    result: dict[str, Any] = {}
    for source in SOURCES:
        court = COURTS[source]
        body = safe_get(court.robots_url, court, (), pause=pause, opener=opener)
        text = body.decode("utf-8", "replace")
        rules = parse_robots(text)
        direct_example = court.base + "/modules.php?name=sud_delo&name_op=doc&number=example"
        status = "allowed" if allowed_by_robots(direct_example, rules) else "robots_blocked"
        result[source] = {
            "status": status,
            "checked_at": utc_now(),
            "rules": rules,
            "robots_sha256": hashlib.sha256(body).hexdigest(),
        }
        atomic_write(output_dir / "access" / f"{source}-robots.txt", body)
        atomic_json(path, result)
    return result


def related_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        for key in case_keys(record):
            result[key].append(record)
    return result


def unique_artifacts(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, Any], dict[str, Any]] = {}
    for record in records:
        identity = record["identity"]
        unique[(identity.get("text_path"), identity.get("text_sha256"))] = record
    return list(unique.values())


def prepare_pairs(
    index_path: Path = INDEX, metadata_path: Path = METADATA, output_dir: Path = OUTPUT,
    root: Path = ROOT, pause: float = 4, refresh_access: bool = False,
    access: dict[str, Any] | None = None, opener: Any = None,
) -> dict[str, Any]:
    rows = read_jsonl(index_path)
    metadata_rows = read_jsonl(metadata_path)
    metadata_by_sha = {row.get("text_sha256"): row for row in metadata_rows if row.get("text_sha256")}
    target = [record for record in rows if record["identity"].get("source") in SOURCES]
    access = access if access is not None else check_access(output_dir, pause, refresh_access, opener)
    related = related_index(rows)
    pair_rows: list[dict[str, Any]] = []
    seen_events: dict[tuple[str, tuple[str, str, str, str]], dict[str, Any]] = {}
    local_links = 0

    for record in target:
        identity = record["identity"]
        source = identity["source"]
        metadata = metadata_by_sha.get(identity.get("text_sha256"), {})
        document_type = identity.get("document_type")
        event = appeal_event(metadata, identity)
        event_id = appeal_event_id(source, event)
        own_artifact = artifact_from_identity(identity)
        base = {
            "schema_version": 1,
            "source": source,
            "source_record_id": identity.get("record_id"),
            "underlying_case_id": record.get("underlying_case_id"),
            "verdict_group_id": verdict_group_id(record),
            "appeal_event_id": event_id,
            "appeal_event": event,
            "sentence": None,
            "appeal": None,
            "pair_status": None,
            "outcome": "unknown",
            "identity_errors": [],
            "access_status": access.get(source, {}).get("status", "unknown"),
        }
        if document_type == "appellate_verdict":
            base["pair_status"] = "rejected_new_trial"
            base["identity_errors"] = ["new_trial_verdict_not_review_act"]
            pair_rows.append(base)
            continue
        if document_type not in {"trial_verdict", "appeal_against_verdict"}:
            base["pair_status"] = "rejected_procedural"
            base["identity_errors"] = ["procedural_act_not_sentence_review"]
            pair_rows.append(base)
            continue
        duplicate = seen_events.get((source, event_key(event)))
        if duplicate is not None:
            base["pair_status"] = "duplicate_event" if duplicate["identity"].get("text_sha256") == identity.get("text_sha256") else "identity_failed"
            base["identity_errors"] = [] if base["pair_status"] == "duplicate_event" else ["event_identity_collision"]
            pair_rows.append(base)
            continue
        seen_events[(source, event_key(event))] = record

        own_text, artifact_errors = validate_artifact(own_artifact, root)
        evidence_errors = validate_identity_evidence(identity, root)
        if artifact_errors or evidence_errors:
            base["sentence" if document_type == "trial_verdict" else "appeal"] = own_artifact
            base["pair_status"] = "artifact_invalid"
            base["identity_errors"] = artifact_errors + evidence_errors
            pair_rows.append(base)
            continue
        if document_type == "appeal_against_verdict":
            base["appeal"] = own_artifact
            base["outcome"] = extract_outcome(own_text or "")
        else:
            base["sentence"] = own_artifact

        opposite_type = "trial_verdict" if document_type == "appeal_against_verdict" else "appeal_against_verdict"
        matches = unique_artifacts(
            candidate
            for key in case_keys(record)
            for candidate in related.get(key, [])
            if candidate is not record and usable_record(candidate, opposite_type)
        )
        if len(matches) > 1 and document_type == "trial_verdict":
            matches = matches
        elif len(matches) > 1:
            base["pair_status"] = "identity_failed"
            base["identity_errors"] = ["ambiguous_sentence"]
            pair_rows.append(base)
            continue
        if not matches:
            role = "sentence" if document_type == "appeal_against_verdict" else "appeal"
            try:
                downloaded = explicit_artifact(metadata, role, source, event_id, output_dir, access, pause, opener, root)
            except AccessBlocked:
                downloaded = None
            if downloaded is not None:
                matches = [{"identity": downloaded, "case_keys": record["case_keys"]}]
            else:
                base["pair_status"] = "missing_sentence" if role == "sentence" else "missing_appeal"
                pair_rows.append(base)
                continue
        local_links += 1
        sentence_record = record if document_type == "trial_verdict" else matches[0]
        appeal_records = [record] if document_type == "appeal_against_verdict" else matches
        for appeal_record in appeal_records:
            current = dict(base)
            if document_type == "trial_verdict":
                appeal_identity = appeal_record["identity"]
                appeal_metadata = metadata_by_sha.get(appeal_identity.get("text_sha256"), {})
                appeal_artifact = artifact_from_identity(appeal_identity)
                appeal_text, errors = validate_artifact(appeal_artifact, root)
                errors += validate_identity_evidence(appeal_identity, root)
                current["appeal"] = appeal_artifact
                current["appeal_event"] = appeal_event(appeal_metadata, appeal_identity)
                current["appeal_event_id"] = appeal_event_id(appeal_identity["source"], current["appeal_event"])
                current["outcome"] = extract_outcome(appeal_text or "")
                pair_metadata = pair_identity_metadata(appeal_record, appeal_metadata, record)
            else:
                sentence_artifact = artifact_from_identity(sentence_record["identity"])
                sentence_text, errors = validate_artifact(sentence_artifact, root)
                errors += validate_identity_evidence(sentence_record["identity"], root)
                current["sentence"] = sentence_artifact
                sentence_text = sentence_text
                appeal_text = own_text
                pair_metadata = pair_identity_metadata(record, metadata, sentence_record)
            if errors:
                current["pair_status"] = "artifact_invalid"
                current["identity_errors"] = errors
            else:
                identity_errors = verify_pair(pair_metadata, sentence_text or "", appeal_text or "")
                current["identity_errors"] = identity_errors
                current["pair_status"] = "confirmed_pair" if not identity_errors else "identity_failed"
                current["verdict_group_id"] = verdict_group_id(record, pair_metadata.get("sentence_date"))
            pair_rows.append(current)

    pair_rows.sort(key=lambda row: (SOURCES.index(row["source"]), row["source_record_id"] or "", row["appeal_event_id"]))
    write_jsonl(output_dir / "pair-status.jsonl", pair_rows)
    summary = summarize(pair_rows, len(target), access, local_links)
    atomic_json(output_dir / "summary.json", summary)
    write_report(summary, output_dir / "report.md")
    return summary


def summarize(rows: list[dict[str, Any]], candidates: int, access: dict[str, Any], local_links: int) -> dict[str, Any]:
    statuses = Counter(row["pair_status"] for row in rows)
    confirmed_outcomes = Counter(row["outcome"] for row in rows if row["pair_status"] == "confirmed_pair")
    appeal_outcomes = Counter(
        row["outcome"] for row in rows
        if row["appeal"] is not None and row["source_record_id"] == (row["appeal"] or {}).get("record_id")
        and row["pair_status"] not in {"rejected_new_trial", "rejected_procedural", "duplicate_event", "artifact_invalid"}
    )
    by_source: dict[str, Any] = {}
    for source in SOURCES:
        source_rows = [row for row in rows if row["source"] == source]
        source_statuses = Counter(row["pair_status"] for row in source_rows)
        by_source[source] = {
            "candidates": len({row["source_record_id"] for row in source_rows}),
            "pair_status_rows": len(source_rows),
            "statuses": dict(sorted(source_statuses.items())),
            "confirmed_outcomes": {outcome: sum(1 for row in source_rows if row["pair_status"] == "confirmed_pair" and row["outcome"] == outcome) for outcome in OUTCOMES},
            "access_status": access.get(source, {}).get("status", "unknown"),
        }
    return {
        "schema_version": 1,
        "candidates": candidates,
        "pair_status_rows": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "confirmed_pairs": statuses.get("confirmed_pair", 0),
        "missing_sentence": statuses.get("missing_sentence", 0),
        "missing_appeal": statuses.get("missing_appeal", 0),
        "identity_failures": statuses.get("identity_failed", 0),
        "duplicates": statuses.get("duplicate_event", 0),
        "rejected_new_trial": statuses.get("rejected_new_trial", 0),
        "rejected_procedural": statuses.get("rejected_procedural", 0),
        "artifact_invalid": statuses.get("artifact_invalid", 0),
        "local_second_text_links": local_links,
        "confirmed_outcomes": {outcome: confirmed_outcomes.get(outcome, 0) for outcome in OUTCOMES},
        "downloaded_appeal_outcomes": {outcome: appeal_outcomes.get(outcome, 0) for outcome in OUTCOMES},
        "by_source": by_source,
        "processed_dataset_included": False,
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Полные пары: Башкортостан, Татарстан, Новосибирск",
        "",
        "Полная пара засчитывается только после проверки двух полных текстов, их SHA256 и identity. Отчёт содержит только агрегаты.",
        "",
        "| Источник | Кандидаты | Полные пары | Нет приговора | Нет апелляции | Identity failures | Дубли события | Новый приговор | Процессуальные | Доступ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for source in SOURCES:
        item = summary["by_source"][source]
        statuses = item["statuses"]
        lines.append(
            f"| {source} | {item['candidates']} | {statuses.get('confirmed_pair', 0)} | {statuses.get('missing_sentence', 0)} | "
            f"{statuses.get('missing_appeal', 0)} | {statuses.get('identity_failed', 0)} | {statuses.get('duplicate_event', 0)} | "
            f"{statuses.get('rejected_new_trial', 0)} | {statuses.get('rejected_procedural', 0)} | {item['access_status']} |"
        )
    lines += [
        f"| **Итого** | **{summary['candidates']}** | **{summary['confirmed_pairs']}** | **{summary['missing_sentence']}** | **{summary['missing_appeal']}** | **{summary['identity_failures']}** | **{summary['duplicates']}** | **{summary['rejected_new_trial']}** | **{summary['rejected_procedural']}** | — |",
        "",
        f"Повреждённые или отсутствующие локальные артефакты: {summary['artifact_invalid']}. Локальных связей со вторым текстом: {summary['local_second_text_links']}.",
        "Статусы взаимно исключают друг друга. Дублируются только записи того же апелляционного события; разные даты, номера или типы актов по одному приговору не объединяются.",
        "",
        "## Outcome подтверждённых полных пар",
        "",
        "| Outcome | Количество |",
        "|---|---:|",
    ]
    for outcome in OUTCOMES:
        lines.append(f"| {outcome} | {summary['confirmed_outcomes'][outcome]} |")
    lines += ["", "## Outcome уже скачанных eligible апелляционных текстов", "", "| Outcome | Количество |", "|---|---:|"]
    for outcome in OUTCOMES:
        lines.append(f"| {outcome} | {summary['downloaded_appeal_outcomes'][outcome]} |")
    lines += [
        "",
        "Эти исходы не являются исходами подтверждённых пар. Outcome извлекается только после маркера резолютивной части; доводы и просьбы сторон не учитываются.",
        "",
        "При `robots_blocked` прямые карточки и документы не запрашивались. Отсутствующий текст означает отсутствие пригодного локального артефакта, а не отсутствие судебного акта.",
        "В processed dataset и index ничего не включалось; требуется проверка lead.",
        "",
    ]
    atomic_write(path, "\n".join(lines).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Полные пары по сохранённым региональным документам")
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pause", type=float, default=4)
    parser.add_argument("--refresh-access", action="store_true")
    parser.add_argument("--strict-counts", action="store_true")
    args = parser.parse_args()
    if args.pause < 4:
        parser.error("--pause должен быть не менее 4 секунд")
    if args.command == "report":
        summary = read_json(args.output / "summary.json", None)
        if summary is None:
            parser.error("Сначала выполните run")
        write_report(summary, args.output / "report.md")
        print(json.dumps({"report": str(args.output / "report.md")}, ensure_ascii=False))
        return
    try:
        summary = prepare_pairs(args.index, args.metadata, args.output, ROOT, args.pause, args.refresh_access)
    except CaptchaError:
        raise SystemExit("CAPTCHA: обработка остановлена немедленно")
    if args.strict_counts:
        actual = {source: summary["by_source"][source]["candidates"] for source in SOURCES}
        if actual != EXPECTED_COUNTS:
            raise SystemExit(f"Неожиданные количества кандидатов: {actual}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
