from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mosgorsud_open_data import extract_outcome, number_identity_variants, parse_date, verify_pair as verify_moscow_pair
MOSCOW_RAW = ROOT / "corpus" / "mosgorsud" / "raw"
EXPANSION_RAW = ROOT / "corpus" / "expansion" / "raw"
SUDACT_RAW = ROOT / "corpus" / "sudact"
REGIONAL_RAW = ROOT / "corpus" / "regional" / "raw"
REGIONAL_PAIR_STATUS = ROOT / "corpus" / "pairs" / "rostov-7kas" / "pair-status.jsonl"
PROCESSED = ROOT / "corpus" / "processed"
OUTCOMES = {"unchanged", "changed", "cancelled", "unknown"}
APPEAL_ONLY_ERRORS = {
    "appeal_type_not_confirmed", "appeal_number_mismatch", "appeal_court_mismatch",
    "appeal_date_missing", "appeal_date_mismatch", "new_trial", "sentence_number_conflict", "manual_review",
}
BUILD_CONTRACT = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write(path, payload.encode("utf-8"))


def read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def digest(value: str, prefix: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def text_path(raw: Path, document: dict[str, Any]) -> Path | None:
    path = raw / "texts" / document.get("text_path", Path(document.get("path", "")).with_suffix(".txt").name)
    return path if path.exists() else None


def document_integrity(raw: Path, document: dict[str, Any]) -> bool:
    binary = raw / "texts" / document.get("path", "")
    if not binary.exists() or sha256_file(binary) != document.get("sha256"):
        return False
    if document.get("bytes") is not None and document.get("bytes") != binary.stat().st_size:
        return False
    text = text_path(raw, document)
    if not text or document.get("text_source_sha256") != document.get("sha256"):
        return False
    return sha256_file(text) == document.get("text_sha256")


def copy_text(source: Path | None, target: Path) -> tuple[str | None, str | None]:
    if not source:
        return None, None
    data = source.read_bytes()
    atomic_write(target, data)
    return str(Path("raw") / "texts" / target.name), hashlib.sha256(data).hexdigest()


def article_parts(participants: Iterable[dict[str, Any]]) -> tuple[list[str], list[str]]:
    positive, other = set(), set()
    for participant in participants:
        for article in participant.get("codexArticles") or []:
            name = str(article.get("name") or "")
            for match in re.finditer(r"ст\s*\.?\s*(\d+(?:\.\d+)?)(?:\s*,?\s*ч\s*\.?\s*([\d|]+))?", name, re.I):
                number, part = match.group(1), match.group(2)
                if number == "290" and part:
                    positive.update(part.split("|"))
                elif number != "290":
                    other.add(number + (" ч." + part if part else ""))
    return sorted(positive), sorted(other)


def article_parts_from_text(text: str) -> tuple[list[str], list[str]]:
    names = re.findall(r"ст\s*\.?\s*\d+(?:\.\d+)?(?:\s*,?\s*ч\s*\.?\s*[\d|]+)?", text, re.I)
    return article_parts([{"codexArticles": [{"name": name} for name in names]}])


def sudact_articles(value: str) -> list[str]:
    return sorted({part.strip() for part in re.split(r"[|,;]", value or "") if re.match(r"^\d", part.strip())})


def text_flags(text: str) -> list[str]:
    flags = []
    if re.search(r"\bфио\b|<данные изъяты>", text, re.I):
        flags.append("names_masked")
    if re.search(r"<адрес>|\bадрес\b", text, re.I):
        flags.append("addresses_masked")
    if re.search(r"<дата>|\bдаты\b", text, re.I):
        flags.append("dates_masked")
    if re.search(r"<сумма>|\bсуммы\b", text, re.I):
        flags.append("amounts_masked")
    return flags


def amount(text: str) -> int | None:
    values = []
    for match in re.finditer(r"(\d[\d\s]{1,18})\s*(?:руб\.?|рублей|₽)", text, re.I):
        try:
            values.append(int(re.sub(r"\s", "", match.group(1))))
        except ValueError:
            pass
    return max(values) if values else None


def episodes(text: str) -> int | None:
    count = len(re.findall(r"\bэпизод\w*", text, re.I))
    return count or None


def circumstance_codes(text: str) -> tuple[list[str], list[str]]:
    mitigating = []
    aggravating = []
    for code, pattern in (
        ("confession", r"признан\w+\s+вин"),
        ("cooperation", r"сотрудничеств\w+|способствов\w+\s+раскрыти"),
        ("dependents", r"на\s+иждивении|несовершеннолетн\w+\s+дет"),
        ("health", r"состояни[ея]\s+здоровья|тяжел\w+\s+заболеван"),
        ("remorse", r"раскаяни[ея]|сожале"),
        ("compensation", r"возмещени[ея]\s+ущерба|компенсац"),
    ):
        if re.search(pattern, text, re.I):
            mitigating.append(code)
    for code, pattern in (
        ("organized_group", r"организованн\w+\s+групп"),
        ("large_amount", r"крупн\w+\s+размер"),
        ("special_large_amount", r"особо\s+крупн\w+\s+размер"),
        ("recidivism", r"рецидив|ранее\s+судим"),
        ("official_position", r"должностн\w+\s+положени"),
    ):
        if re.search(pattern, text, re.I):
            aggravating.append(code)
    return mitigating, aggravating


def prison_months(text: str) -> int | None:
    for context in re.finditer(r"лишени\w*\s+свободы|лишить\s+свободы", text, re.I):
        window = text[max(0, context.start() - 140):context.end() + 40]
        matches = list(re.finditer(r"(\d+)\s*(?:год|года|лет)(?:\s+(\d+)\s*месяц\w*)?", window, re.I))
        if matches:
            match = matches[-1]
            years = int(match.group(1))
            months = int(match.group(2) or 0)
            return years * 12 + months if years <= 50 else None
    return None


def fine_amount(text: str) -> int | None:
    match = re.search(r"штраф\w*[^.]{0,120}?(\d[\d\s]{2,18})\s*(?:руб\.?|рублей|₽)", text, re.I)
    if not match:
        return None
    try:
        return int(re.sub(r"\s", "", match.group(1)))
    except ValueError:
        return None


def ban_years(text: str) -> int | None:
    match = re.search(r"(?:запрет\w*|лишени\w+\s+права)[^.]{0,160}?(\d+)\s*(?:год|года|лет)", text, re.I)
    return int(match.group(1)) if match and int(match.group(1)) <= 20 else None


def process_flags(text: str) -> dict[str, bool | None]:
    if not text:
        return {key: None for key in ("special_order", "pretrial_agreement", "art_64", "art_69", "art_73")}
    return {
        "special_order": True if re.search(r"особ\w+\s+поряд\w+|40\.1|глава\s+40", text, re.I) else None,
        "pretrial_agreement": True if re.search(r"досудебн\w+\s+соглашени|ст\.?\s*62\b", text, re.I) else None,
        "art_64": True if re.search(r"ст\.?\s*64\b", text, re.I) else None,
        "art_69": True if re.search(r"ст\.?\s*69\b", text, re.I) else None,
        "art_73": True if re.search(r"ст\.?\s*73\b|условн\w+\s+осуждени", text, re.I) else None,
    }


def sudact_process(text: str, metadata: dict[str, str]) -> dict[str, bool | None]:
    flags = process_flags(text)
    for key in flags:
        if key in metadata:
            flags[key] = metadata[key] == "True"
    return flags


def extract_punishment(text: str, metadata: dict[str, str] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "prison_months": int(metadata["prison_months"]) if metadata.get("prison_months") else prison_months(text),
        "fine_rub": int(metadata["fine_rub"]) if metadata.get("fine_rub") else fine_amount(text),
        "fine_multiple": metadata.get("fine_multiple") == "True" if metadata.get("fine_multiple") else (bool(re.search(r"кратн\w+\s+(?:размер|сумм|штраф)", text, re.I)) if text else None),
        "ban_on_positions": metadata.get("ban_on_positions") == "True" if metadata.get("ban_on_positions") else (bool(re.search(r"запрет\w+\s+занимать\s+должности|лишени\w+\s+права\s+занимать\s+должности", text, re.I)) if text else None),
        "ban_years": ban_years(text) if text else None,
        "conditional": metadata.get("art_73") == "True" if metadata.get("art_73") else (bool(re.search(r"условн\w+\s+осуждени|ст\.?\s*73\b", text, re.I)) if text else None),
        "confiscation": metadata.get("confiscation") == "True" if metadata.get("confiscation") else (bool(re.search(r"конфискац", text, re.I)) if text else None),
    }


def reason_codes(text: str, outcome: str) -> list[str]:
    if outcome in {"unchanged", "unknown"}:
        return []
    codes = []
    for code, pattern in (
        ("too_harsh", r"чрезмерн\w+\s+суров|несправедлив\w+\s+суров"),
        ("too_lenient", r"чрезмерн\w+\s+мягк|несправедлив\w+\s+мягк"),
        ("misapplied_law", r"неправильн\w+\s+применени|нарушени\w+\s+закона"),
        ("mitigating", r"смягчающ\w+\s+обстоятельств"),
        ("additional_punishment", r"дополнительн\w+\s+наказани|дополнительн\w+\s+штраф"),
        ("confiscation", r"конфискац"),
        ("time_served", r"срок\s+отбытия|зачет\s+времени"),
        ("procedural_violation", r"существенн\w+\s+нарушени\w+\s+процессуального"),
    ):
        if re.search(pattern, text, re.I):
            codes.append(code)
    return codes


def identity_year(identity: dict[str, Any]) -> int | None:
    value = identity.get("year")
    if isinstance(value, int) and 1900 <= value <= 2100:
        return value
    for key in ("act_date", "dateReg"):
        text = str(identity.get(key) or "")
        match = re.match(r"(\d{4})-", text) or re.search(r"(\d{4})$", text)
        if match:
            year = int(match.group(1))
            return year if 1900 <= year <= 2100 else None
    return None


def moscow_records(raw: Path = MOSCOW_RAW) -> list[dict[str, Any]]:
    progress = read_json(raw / "pair-progress.json", {})
    records = []
    for candidate in read_jsonl(raw / "work" / "appeal-candidates.jsonl"):
        appeal_id = str(candidate["appeal_id"])
        state = progress.get(appeal_id, {})
        documents = state.get("documents", {})
        sentence_path = text_path(raw, documents.get("sentence", {})) if documents.get("sentence") else None
        appeal_path = text_path(raw, documents.get("appeal", {})) if documents.get("appeal") else None
        sentence_text = sentence_path.read_text(encoding="utf-8") if sentence_path else ""
        appeal_text = appeal_path.read_text(encoding="utf-8") if appeal_path else ""
        first = candidate.get("first") or {}
        group_key = str(first.get("_id") or candidate["identity"].get("uid") or appeal_id)
        if first.get("participants"):
            positive, other = article_parts(first["participants"])
        else:
            positive, other = article_parts_from_text(sentence_text)
        identity_errors = list(state.get("errors", []))
        retrieval_errors = [error for error in state.get("errors", []) if error not in APPEAL_ONLY_ERRORS]
        sentence_integrity = document_integrity(raw, documents["sentence"]) if documents.get("sentence") else False
        integrity = all(document_integrity(raw, documents[kind]) for kind in ("sentence", "appeal")) if all(kind in documents for kind in ("sentence", "appeal")) else False
        if not integrity:
            identity_errors.append("integrity_mismatch")
        if not sentence_integrity:
            retrieval_errors.append("integrity_mismatch")
        verification_errors = verify_moscow_pair(candidate, sentence_text, appeal_text) if integrity and sentence_text and appeal_text else []
        identity_errors.extend(verification_errors)
        retrieval_errors.extend(error for error in verification_errors if error not in APPEAL_ONLY_ERRORS)
        full_pair = state.get("status") == "confirmed_pair" and integrity and not identity_errors
        outcome = extract_outcome(appeal_text) if full_pair else "unknown"
        identity = candidate["identity"]
        records.append({
            "source": "mosgorsud",
            "source_key": appeal_id,
            "verdict_key": group_key,
            "source_year": identity_year(identity),
            "article_parts": positive,
            "other_articles": other,
            "process": process_flags(sentence_text),
            "sentence_text": sentence_text,
            "appeal_text": appeal_text,
            "appeal_participants": candidate.get("participants") or [],
            "identity_errors": identity_errors,
            "retrieval_errors": retrieval_errors,
            "first": first,
            "lower": (candidate.get("lower") or [{}])[0],
            "full_pair": full_pair,
            "outcome": outcome,
            "status": state.get("status", "not_started"),
        })
    return records


def sudact_records(raw: Path = SUDACT_RAW) -> list[dict[str, Any]]:
    analysis = read_csv(raw / "analysis.csv")
    appeals_analysis = read_csv(raw / "appeals_analysis.csv")
    records = []
    for metadata in read_jsonl(raw / "docs.jsonl"):
        row = analysis.get(metadata["id"], {})
        text_path = raw / "docs" / f"{metadata['id']}.txt"
        text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        positive = sudact_articles(row.get("parts_290", ""))
        other = sudact_articles(row.get("other_articles", ""))
        records.append({
            "source": "sudact",
            "source_key": metadata["id"],
            "verdict_key": metadata["id"],
            "source_year": int(metadata["period"][:4]),
            "article_parts": positive,
            "other_articles": other,
            "process": sudact_process(text, row),
            "sentence_text": text,
            "sentence_path": str(text_path) if text else None,
            "appeal_text": "",
            "first": row,
            "lower": {"courtName": metadata.get("court")},
            "full_pair": False,
            "outcome": "unknown",
            "status": "ok" if text else "missing_sentence_text",
            "metadata": row,
        })
    for metadata in read_jsonl(raw / "appeals.jsonl"):
        row = appeals_analysis.get(metadata["id"], {})
        text_path = raw / "appeals" / f"{metadata['id']}.txt"
        appeal_text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        outcome_map = {"без изменения": "unchanged", "изменён": "changed", "отменён": "cancelled"}
        records.append({
            "source": "sudact_appeal",
            "source_key": metadata["id"],
            "verdict_key": metadata["id"],
            "source_year": int(metadata["period"][:4]),
            "article_parts": sudact_articles(row.get("parts_290", "")),
            "other_articles": [],
            "process": process_flags(""),
            "sentence_text": "",
            "sentence_path": None,
            "appeal_text": appeal_text,
            "appeal_path": str(text_path) if appeal_text else None,
            "first": row,
            "lower": {"courtName": metadata.get("court")},
            "full_pair": False,
            "outcome": outcome_map.get(row.get("outcome"), "unknown"),
            "status": "unlinked_appeal",
            "metadata": row,
        })
    return records


def checked_pair_artifact(root: Path, artifact: dict[str, Any]) -> str | None:
    try:
        text_path = (root / artifact["text_path"]).resolve()
        text_path.relative_to(root.resolve())
        if sha256_file(text_path) != artifact.get("text_sha256"):
            return None
        for path_key, hash_key in (("raw_path", "raw_sha256"), ("identity_evidence_path", "identity_evidence_sha256")):
            if not artifact.get(path_key):
                continue
            path = (root / artifact[path_key]).resolve()
            path.relative_to(root.resolve())
            if sha256_file(path) != artifact.get(hash_key):
                return None
        return text_path.read_text(encoding="utf-8-sig")
    except (KeyError, OSError, ValueError, UnicodeError):
        return None


def regional_pair_records(status_path: Path = REGIONAL_PAIR_STATUS, root: Path = ROOT) -> list[dict[str, Any]]:
    analysis_path = SUDACT_RAW / "analysis.csv"
    analysis = read_csv(analysis_path) if analysis_path.exists() else {}
    records = []
    for row in read_jsonl(status_path):
        if row.get("pair_status") != "confirmed_pair" or row.get("identity_errors") or row.get("outcome") not in OUTCOMES - {"unknown"}:
            continue
        sentence = row.get("sentence_artifact") or {}
        appeal = row.get("appeal_artifact") or {}
        sentence_text = checked_pair_artifact(root, sentence)
        appeal_text = checked_pair_artifact(root, appeal)
        if not sentence_text or not appeal_text:
            continue
        metadata = analysis.get(str(sentence.get("record_id", "")).removeprefix("sudact:"), {})
        date = sentence.get("decision_date")
        parsed = parse_date(date)
        first = {
            "number": sentence.get("case_number"), "year": parsed.year if parsed else None,
            "courtName": sentence.get("court"), "dateFinal": date,
        }
        positive, other = (
            (sudact_articles(metadata.get("parts_290", "")), sudact_articles(metadata.get("other_articles", "")))
            if metadata else article_parts_from_text(sentence_text)
        )
        source = str(row["source"]) + ("+sudact" if sentence.get("source") == "sudact" else "")
        records.append({
            "source": source,
            "source_key": row["pair_id"],
            "verdict_key": row["verdict_group_id"],
            "source_year": identity_year({"act_date": appeal.get("decision_date")}),
            "article_parts": positive,
            "other_articles": other,
            "process": sudact_process(sentence_text, metadata),
            "sentence_text": sentence_text,
            "sentence_path": str((root / sentence["text_path"]).resolve()),
            "appeal_text": appeal_text,
            "appeal_path": str((root / appeal["text_path"]).resolve()),
            "appeal_participants": [],
            "identity_errors": [],
            "retrieval_errors": [],
            "provenance_warnings": ["sentence_raw_unavailable"] if not sentence.get("raw_path") else [],
            "first": first,
            "lower": {"number": sentence.get("case_number"), "year": parsed.year if parsed else None, "courtName": sentence.get("court")},
            "full_pair": True,
            "outcome": row["outcome"],
            "status": "confirmed_pair",
            "metadata": metadata,
        })
    return records


def normalized_verdict_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", normalized)


def normalized_identity_part(value: Any) -> str:
    return normalized_verdict_text(str(value or ""))


def canonical_case_number(value: Any, year: Any = None) -> str:
    variants = number_identity_variants(value, year)
    if not variants:
        return normalized_identity_part(value)
    prefix, sequence, numeric_year = sorted(variants)[0]
    return f"{'/'.join(str(part) for part in prefix)}-{sequence}/{numeric_year}"


def canonical_court_identity(value: Any) -> str:
    text = re.sub(r"\s*[-–—]?\s*уголовное(?:\s+дело)?\s*$", "", str(value or ""), flags=re.I)
    return normalized_identity_part(text)


def canonical_date_identity(value: Any) -> str:
    parsed = parse_date(value)
    return parsed.date().isoformat() if parsed else normalized_identity_part(value)


def canonical_verdict_identity(record: dict[str, Any]) -> str:
    lower = record.get("lower") or {}
    first = record.get("first") or {}
    year = lower.get("year") or first.get("year")
    return "|".join((
        canonical_court_identity(lower.get("courtName") or first.get("courtName")),
        canonical_case_number(lower.get("number") or first.get("number"), year),
        canonical_date_identity(first.get("dateFinal") or first.get("lastHearingDate")),
    ))


def make_case(record: dict[str, Any], processed: Path) -> dict[str, Any]:
    case_id = digest(f"{record['source']}|{record['verdict_key']}|{record['source_key']}", "case_")
    text_hash = hashlib.sha256(normalized_verdict_text(record["sentence_text"]).encode("utf-8")).hexdigest()
    identity = canonical_verdict_identity(record)
    verdict_group_id = digest(identity if all(identity.split("|")) else f"{text_hash}|{identity or record['verdict_key']}", "vg_")
    sentence_text = record["sentence_text"]
    appeal_text = record["appeal_text"]
    sentence_target = processed / "raw" / "texts" / f"{case_id}-sentence.txt"
    appeal_target = processed / "raw" / "texts" / f"{case_id}-appeal.txt"
    sentence_source = Path(record["sentence_path"]) if record.get("sentence_path") else None
    appeal_source = Path(record["appeal_path"]) if record.get("appeal_path") else None
    sentence_ref, sentence_sha = copy_text(sentence_source, sentence_target)
    appeal_ref, appeal_sha = copy_text(appeal_source, appeal_target)
    mitigating, aggravating = circumstance_codes(sentence_text)
    facts = {
        "amount_rub": amount(sentence_text) if sentence_text else None,
        "episodes": episodes(sentence_text) if sentence_text else None,
        "mitigating": mitigating,
        "aggravating": aggravating,
        "narrative": sentence_text,
    }
    expected = 2 if record["full_pair"] else 1
    available = bool(sentence_text) + (bool(appeal_text) if record["full_pair"] else 0)
    identity_errors = list(record.get("identity_errors") or [])
    quality = {
        "text_coverage": round(available / expected, 2) if expected else 0,
        "redaction_flags": text_flags(sentence_text or appeal_text),
        "parser_status": "identity_failed" if identity_errors else record["status"],
        "identity_errors": identity_errors,
        "retrieval_errors": list(record.get("retrieval_errors", identity_errors)),
        "provenance_warnings": list(record.get("provenance_warnings") or []),
    }
    return {
        "schema_version": 1,
        "case_id": case_id,
        "verdict_group_id": verdict_group_id,
        "source": record["source"],
        "source_year": record["source_year"],
        "article_parts": article_parts(record["first"]["participants"])[0] if record["first"].get("participants") else record["article_parts"],
        "other_articles": record["other_articles"],
        "process": record["process"],
        "facts": facts,
        "punishment": extract_punishment(sentence_text, record.get("metadata")),
        "appeal": {
            "full_pair": record["full_pair"],
            "outcome": record["outcome"],
            "reason_codes": reason_codes(appeal_text, record["outcome"]) if record["full_pair"] else [],
            "outcome_article_parts": article_parts(record.get("appeal_participants") or [])[0],
        },
        "quality": quality,
        "split": "unassigned",
        "artifacts": {"sentence": sentence_ref, "appeal": appeal_ref, "sentence_sha256": sentence_sha, "appeal_sha256": appeal_sha},
    }


def attach_paths(records: list[dict[str, Any]], raw: Path) -> None:
    for record in records:
        documents = read_json(raw / "pair-progress.json", {}).get(record["source_key"], {}).get("documents", {})
        if documents.get("sentence"):
            record["sentence_path"] = str(text_path(raw, documents["sentence"]))
        if documents.get("appeal"):
            record["appeal_path"] = str(text_path(raw, documents["appeal"]))


def retrieval_case(case: dict[str, Any]) -> dict[str, Any] | None:
    errors = case["quality"].get("retrieval_errors", case["quality"].get("identity_errors"))
    if not case["facts"]["narrative"] or errors:
        return None
    return {
        "schema_version": 1,
        "case_id": case["case_id"],
        "verdict_group_id": case["verdict_group_id"],
        "source": case["source"],
        "source_year": case["source_year"],
        "article_parts": case["article_parts"],
        "other_articles": case["other_articles"],
        "process": case["process"],
        "facts": case["facts"],
        "punishment": case["punishment"],
        "quality": case["quality"],
        "split": case["split"],
    }


def risk_case(case: dict[str, Any]) -> dict[str, Any] | None:
    if not case["appeal"]["full_pair"]:
        return None
    return {
        "schema_version": 1,
        "case_id": case["case_id"],
        "verdict_group_id": case["verdict_group_id"],
        "source": case["source"],
        "source_year": case["source_year"],
        "features": {
            "article_parts": case["article_parts"],
            "other_articles": case["other_articles"],
            "process": case["process"],
            "facts": case["facts"],
            "punishment": case["punishment"],
        },
        "outcome": case["appeal"]["outcome"],
        "split": case["split"],
    }


def article_case(case: dict[str, Any]) -> dict[str, Any] | None:
    errors = case["quality"].get("retrieval_errors", case["quality"].get("identity_errors"))
    if not case["facts"]["narrative"] or errors or not (case["article_parts"] or case["other_articles"]):
        return None
    return {
        "schema_version": 1,
        "example_id": case["case_id"],
        "verdict_group_id": case["verdict_group_id"],
        "source": case["source"],
        "source_year": case["source_year"],
        "text": case["facts"]["narrative"],
        "article_parts": case["article_parts"],
        "other_articles": case["other_articles"],
        "split": case["split"],
    }


def unify_verdict_text_groups(cases: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        text = case["facts"]["narrative"]
        if text:
            groups.setdefault(normalized_verdict_text(text), []).append(case)
    for duplicates in groups.values():
        group_id = min(case["verdict_group_id"] for case in duplicates)
        for case in duplicates:
            case["verdict_group_id"] = group_id


def deduplicate_retrieval(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for case in cases:
        row = retrieval_case(case)
        if row and row["verdict_group_id"] not in seen:
            seen.add(row["verdict_group_id"])
            result.append(row)
    return result


def assign_splits(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        group = row.get("verdict_group_id", "")
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
        row["split"] = "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def source_hashes() -> dict[str, str]:
    paths = [
        MOSCOW_RAW / "pair-progress.json",
        MOSCOW_RAW / "work" / "appeal-candidates.jsonl",
        EXPANSION_RAW / "pair-progress.json",
        EXPANSION_RAW / "work" / "appeal-candidates.jsonl",
        EXPANSION_RAW / "work" / "selection.json",
        SUDACT_RAW / "docs.jsonl",
        SUDACT_RAW / "text-sha256.json",
        SUDACT_RAW / "appeals.jsonl",
        SUDACT_RAW / "analysis.csv",
        SUDACT_RAW / "appeals_analysis.csv",
        REGIONAL_PAIR_STATUS,
    ]
    result = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths if path.exists()}
    for row in read_jsonl(REGIONAL_PAIR_STATUS):
        if row.get("pair_status") != "confirmed_pair":
            continue
        for artifact in (row.get("sentence_artifact"), row.get("appeal_artifact")):
            for key in ("text_path", "raw_path", "identity_evidence_path"):
                if artifact and artifact.get(key):
                    path = (ROOT / artifact[key]).resolve()
                    path.relative_to(ROOT.resolve())
                    result[str(path.relative_to(ROOT))] = sha256_file(path)
    for directory in (MOSCOW_RAW / "texts", EXPANSION_RAW / "texts", SUDACT_RAW / "docs", SUDACT_RAW / "appeals"):
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    result[str(path.relative_to(ROOT))] = sha256_file(path)
    return result


def output_hashes(processed: Path) -> dict[str, str]:
    names = (
        "cases.jsonl", "retrieval_cases.jsonl", "risk_examples.jsonl", "article_examples.jsonl",
        "manifest.json", "dataset-card.json", "model-card.json", "raw/id-map.jsonl",
    )
    return {name: sha256_file(processed / name) if (processed / name).exists() else "missing" for name in names}


def valid_resume(processed: Path, previous: dict[str, Any], inputs: dict[str, str], split: bool) -> bool:
    if previous.get("build_contract") != BUILD_CONTRACT:
        return False
    if previous.get("inputs") != inputs or previous.get("split") != split:
        return False
    manifest = previous.get("manifest") or {}
    if manifest.get("schema_version") != 1 or previous.get("output_hashes") != output_hashes(processed):
        return False
    counts = manifest.get("counts", {})
    cases = read_jsonl(processed / "cases.jsonl")
    retrieval = read_jsonl(processed / "retrieval_cases.jsonl")
    risks = read_jsonl(processed / "risk_examples.jsonl")
    articles = read_jsonl(processed / "article_examples.jsonl")
    if not cases or len(cases) != counts.get("cases") or len(read_jsonl(processed / "raw" / "id-map.jsonl")) != len(cases):
        return False
    if len(retrieval) != counts.get("retrieval_cases") or len(risks) != counts.get("risk_examples") or len(articles) != counts.get("article_examples"):
        return False
    if len({row["verdict_group_id"] for row in retrieval}) != len(retrieval):
        return False
    if any(row.get("schema_version") != 1 for row in cases + retrieval + risks + articles):
        return False
    if any("appeal" in row for row in retrieval):
        return False
    for case in cases:
        for kind in ("sentence", "appeal"):
            reference = case.get("artifacts", {}).get(kind)
            if not reference:
                continue
            path = processed / reference
            if not path.exists() or sha256_file(path) != case["artifacts"].get(kind + "_sha256"):
                return False
    return True


def build(processed: Path = PROCESSED, split: bool = False, force: bool = False) -> dict[str, Any]:
    state_path = processed / "raw" / "build-state.json"
    previous = read_json(state_path, {})
    inputs = source_hashes()
    if not force and valid_resume(processed, previous, inputs, split):
        return previous["manifest"]
    records = moscow_records()
    attach_paths(records, MOSCOW_RAW)
    expansion_records = moscow_records(EXPANSION_RAW)
    attach_paths(expansion_records, EXPANSION_RAW)
    records.extend(expansion_records)
    records.extend(sudact_records())
    records.extend(regional_pair_records())
    cases = [make_case(record, processed) for record in records]
    unify_verdict_text_groups(cases)
    retrieval = deduplicate_retrieval(cases)
    risks = [row for case in cases if (row := risk_case(case))]
    articles = [row for case in cases if (row := article_case(case))]
    if split:
        for rows in (cases, retrieval, risks, articles):
            assign_splits(rows)
    write_jsonl(processed / "cases.jsonl", cases)
    write_jsonl(processed / "retrieval_cases.jsonl", retrieval)
    write_jsonl(processed / "risk_examples.jsonl", risks)
    write_jsonl(processed / "article_examples.jsonl", articles)
    mapping = [{
        "case_id": case["case_id"], "verdict_group_id": case["verdict_group_id"],
        "source": case["source"], "source_key": record["source_key"], "verdict_key": record["verdict_key"],
    } for case, record in zip(cases, records)]
    write_jsonl(processed / "raw" / "id-map.jsonl", mapping)
    source_counts = Counter(case["source"] for case in cases)
    year_counts = Counter(str(case["source_year"]) for case in cases)
    outcome_counts = Counter(row["outcome"] for row in risks)
    part_counts = Counter(part for row in articles for part in row["article_parts"])
    quality_counts = Counter(case["quality"]["parser_status"] for case in cases)
    manifest = {
        "schema_version": 1,
        "build_contract": BUILD_CONTRACT,
        "created_at": utc_now(),
        "counts": {
            "cases": len(cases), "retrieval_cases": len(retrieval), "risk_examples": len(risks), "article_examples": len(articles),
            "by_source": dict(sorted(source_counts.items())),
            "by_year": dict(sorted(year_counts.items())),
            "risk_outcomes": {outcome: outcome_counts.get(outcome, 0) for outcome in sorted(OUTCOMES)},
            "article_parts": dict(sorted(part_counts.items())),
            "quality": dict(sorted(quality_counts.items())),
            "full_pairs": sum(case["appeal"]["full_pair"] for case in cases),
            "regional_text_only_pairs": sum("sentence_raw_unavailable" in case["quality"]["provenance_warnings"] for case in cases),
            "regional_historical_unavailable": 4,
        },
        "inputs": {
            key: value for key, value in inputs.items()
            if "/texts/" not in key.replace("\\", "/")
            and "/docs/" not in key.replace("\\", "/")
            and "/appeals/" not in key.replace("\\", "/")
            and not key.lower().endswith((".html", ".bin", ".txt"))
        },
        "leakage_rules": {
            "retrieval_excludes_appeal_text_and_labels": True,
            "risk_requires_confirmed_full_pair": True,
            "unknown_is_not_negative": True,
            "unconfirmed_moscow_candidates_excluded_from_risk": True,
        },
    }
    atomic_json(processed / "manifest.json", manifest)
    atomic_json(processed / "dataset-card.json", {
        "schema_version": 1,
        "status": "dataset_only",
        "intended_use": "Версионируемые тексты для kNN и исследование риска изменения приговора; не production-модель.",
        "leakage_rules": manifest["leakage_rules"],
        "limitations": [
            "Нет оценки качества модели или production readiness.",
            "Региональные исторические тексты отсутствуют в текущем raw и не включены.",
            "У двух региональных приговоров сохранены полный текст и метаданные СудАкта, но нет исходного HTML.",
            "Незаполненные признаки сохраняются как null/empty, без догадок.",
            "Split по умолчанию unassigned; групповое разбиение опционально.",
        ],
    })
    atomic_json(state_path, {"build_contract": BUILD_CONTRACT, "inputs": inputs, "manifest": manifest, "split": split, "output_hashes": output_hashes(processed)})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Версионируемый dataset без target leakage")
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = build(args.processed, args.split, args.force)
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
