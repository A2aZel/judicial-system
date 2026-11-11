from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.request
import urllib.robotparser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mosgorsud_open_data import new_trial_disposition

OUT = ROOT / "corpus" / "recovery-150" / "rostov-7kas"
STATUS = ROOT / "corpus" / "pairs" / "rostov-7kas" / "pair-status.jsonl"
INDEX = ROOT / "corpus" / "regional" / "raw" / "collection-2026-11-10" / "unified-case-index.jsonl"
METADATA = ROOT / "corpus" / "regional" / "raw" / "collection-2026-11-10" / "metadata.jsonl"
CANDIDATES = OUT / "candidates.jsonl"
SEARCH_LOG = OUT / "evidence" / "search-log.jsonl"
LEADS = OUT / "evidence" / "lead-log.jsonl"
PAIRS = OUT / "recovered-pairs.jsonl"
REJECTED = OUT / "rejected-pairs.jsonl"
MANIFEST = OUT / "sha256-manifest.jsonl"
RESUME = OUT / "resume.json"
SUMMARY = OUT / "summary.json"
REPORT = OUT / "report.md"
ALLOWED_OUTCOMES = {"changed", "unchanged", "cancelled"}
SEARCH_STATUSES = {"no_exact_direct_document", "lead_found", "rejected_lead", "search_error"}
CONTRACT = 2


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_date(value: str) -> str:
    year, month, day = value.split("-")
    return f"{day}.{month}.{year}"


def select_candidates() -> tuple[list[dict[str, Any]], dict[str, int]]:
    statuses = read_jsonl(STATUS)
    index = {record["identity"]["record_id"]: record for record in read_jsonl(INDEX)}
    metadata = {
        record.get("record_id") or record["identity"]["record_id"]: record
        for record in read_jsonl(METADATA)
    }
    missing = [
        record for record in statuses
        if record.get("pair_status") == "missing_sentence" and record.get("outcome") in ALLOWED_OUTCOMES
    ]
    candidates = []
    for status in sorted(missing, key=lambda item: item["source_record_id"]):
        record_id = status["source_record_id"]
        indexed = index.get(record_id)
        source = metadata.get(record_id)
        if not indexed or not source:
            continue
        identity = indexed["identity"]
        fields = ("first_court", "first_case_number", "first_decision_date")
        if not all(identity.get(field) for field in fields):
            continue
        if any(source.get(field) != identity.get(field) for field in fields):
            raise ValueError(f"Identity disagreement for {record_id}")
        candidates.append({
            "schema_version": 1,
            "processing_contract": CONTRACT,
            "pair_id": status["pair_id"],
            "verdict_group_id": status["verdict_group_id"],
            "source": status["source"],
            "source_record_id": record_id,
            "outcome": status["outcome"],
            "first_court": identity["first_court"],
            "first_case_number": identity["first_case_number"],
            "first_decision_date": identity["first_decision_date"],
            "exact_query": f'"{identity["first_case_number"]}" "{identity["first_court"]}" "{numeric_date(identity["first_decision_date"])}" приговор',
            "appeal_artifact": status["appeal_artifact"],
            "unified_identity": {
                "provenance": identity.get("provenance"),
                "identity_status": identity.get("identity_status"),
                "identity_link_verified": identity.get("identity_link_verified"),
                "identity_link_method": identity.get("identity_link_method"),
                "identity_evidence_path": identity.get("identity_evidence_path"),
                "identity_link_evidence_sha256": identity.get("identity_link_evidence_sha256"),
            },
            "metadata_provenance": source.get("integration_source_metadata"),
        })
    counts = {
        "status_records": len(statuses),
        "eligible_missing_sentence": len(missing),
        "identity_complete": len(candidates),
    }
    return candidates, counts


def normalized(value: str) -> str:
    return re.sub(r"[^0-9a-zа-я]+", " ", value.lower().replace("ё", "е")).strip()


def compact(value: str) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", value.lower().replace("ё", "е"))


def court_compact(value: str) -> str:
    without_city_word = re.sub(r"(?<![а-я])(?:г|города)(?![а-я])", " ", normalized(value))
    return compact(without_city_word)


def html_to_text(fragment: str) -> str:
    fragment = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", " ", fragment, flags=re.S | re.I)
    fragment = re.sub(r"<br\s*/?>|</p>|</div>|</center>|</h1>", "\n", fragment, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
    text = re.sub(r"[ \t \u00a0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def parse_sudact(page: str) -> dict[str, str]:
    title = re.search(r"<h1>(.*?)</h1>", page, re.S)
    court = re.search(r'<div class="b-justice">(.*?)</div>', page, re.S)
    start = page.find('<hr class="hr-h1">')
    end = page.find("<h3>Суд:</h3>", start)
    if not title or start < 0 or end < 0:
        raise ValueError("document layout not recognised")
    return {
        "title": html_to_text(title.group(1)),
        "court": html_to_text(court.group(1)) if court else "",
        "text": html_to_text(page[start:end]),
    }


def sentence_errors(text: str, candidate: dict[str, Any]) -> list[str]:
    if len(text) < 1000:
        return ["text_too_short"]
    header = text[:8000]
    established = re.search(r"установил", header, re.I)
    if established:
        header = header[:established.start()]
    header_compact = compact(header)
    whole_compact = compact(text)
    errors = []
    if "приговор" not in header_compact[:300]:
        errors.append("trial_verdict_header_missing")
    if "апелляционныйприговор" in header_compact or "кассационныйприговор" in header_compact:
        errors.append("new_trial_verdict")
    if not established:
        errors.append("motivation_missing")
    if "приговорил" not in whole_compact:
        errors.append("operative_missing")
    if compact(candidate["first_case_number"]) not in header_compact:
        errors.append("sentence_number_mismatch")
    court_position = court_compact(header).find(court_compact(candidate["first_court"]))
    if court_position < 0:
        errors.append("sentence_court_mismatch")
    year, month, day = candidate["first_decision_date"].split("-")
    months = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря")
    dates = (numeric_date(candidate["first_decision_date"]), f"{int(day)} {months[int(month) - 1]} {year}")
    date_positions = [header_compact.find(compact(date)) for date in dates]
    date_position = min((position for position in date_positions if position >= 0), default=-1)
    if date_position < 0:
        errors.append("sentence_date_mismatch")
    elif court_position >= 0 and date_position >= court_position:
        errors.append("sentence_date_not_before_court")
    return errors


def fetch_document(record_id: str, url: str, pause: float = 4.1) -> dict[str, Any]:
    if pause < 4:
        raise ValueError("pause must be at least four seconds")
    match = re.fullmatch(r"https://sudact\.ru/regular/doc/([A-Za-z0-9]+)/", url)
    if not match:
        raise ValueError("Only direct SudAct document URLs are allowed")
    candidates = {row["source_record_id"]: row for row in read_jsonl(CANDIDATES)}
    candidate = candidates[record_id]
    document_id = match.group(1)
    evidence_dir = OUT / "evidence" / "documents"
    for old_path in evidence_dir.glob("*.json"):
        old = json.loads(old_path.read_text(encoding="utf-8"))
        if old.get("source_url") == url:
            return {"status": "duplicate_document", "document_id": document_id, "evidence_path": old_path.relative_to(ROOT).as_posix(), "same_candidate": old.get("source_record_id") == record_id}
    robots_path = OUT / "evidence" / "hosts" / "sudact-robots.txt"
    rules = urllib.robotparser.RobotFileParser()
    rules.parse(robots_path.read_text(encoding="utf-8").splitlines())
    user_agent = "recover_rostov_150/1.0"
    if not rules.can_fetch(user_agent, url):
        raise ValueError("robots disallow")
    started = time.monotonic()
    time.sleep(pause)
    observed_pause = time.monotonic() - started
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}")
        raw = response.read()
        content_type = response.headers.get_content_type()
    lowered = raw.decode("utf-8", "replace").lower()
    if any(marker in lowered for marker in ("check_captcha", "проверяем что вы человек", "доступ запрещен", "access denied", "too many requests")):
        raise ValueError("captcha_or_denial")
    raw_sha = hashlib.sha256(raw).hexdigest()
    raw_path = OUT / "raw" / f"{raw_sha}.html"
    raw_path.write_bytes(raw)
    parsed = parse_sudact(raw.decode("utf-8", "replace"))
    text = parsed["text"]
    text_bytes = (text + "\n").encode("utf-8")
    text_sha = hashlib.sha256(text_bytes).hexdigest()
    text_path = OUT / "texts" / f"{text_sha}.txt"
    text_path.write_bytes(text_bytes)
    errors = sentence_errors(text, candidate)
    evidence_path = evidence_dir / f"{record_id}-{document_id}.json"
    evidence = {
        "schema_version": 1,
        "source_record_id": record_id,
        "source": "sudact",
        "document_id": document_id,
        "source_url": url,
        "robots_url": "https://sudact.ru/robots.txt",
        "robots_path": robots_path.relative_to(ROOT).as_posix(),
        "robots_sha256": sha256(robots_path),
        "robots_allows_document": True,
        "user_agent": user_agent,
        "minimum_pause_seconds": 4.0,
        "observed_pause_seconds": round(observed_pause, 3),
        "http_status": 200,
        "content_type": content_type,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": raw_path.relative_to(ROOT).as_posix(),
        "raw_sha256": raw_sha,
        "raw_size_bytes": len(raw),
        "text_path": text_path.relative_to(ROOT).as_posix(),
        "text_sha256": text_sha,
        "text_characters": len(text),
        "document_title": parsed["title"],
        "document_court": parsed["court"],
        "validation_errors": errors,
        "full_trial_verdict": not errors,
        "captcha_or_denial": False,
    }
    write_json(evidence_path, evidence)
    write_json(OUT / "evidence" / "access.json", {"last_document_get_at_utc": evidence["fetched_at_utc"], "last_document_id": document_id})
    return {"status": "validated_sentence" if not errors else "rejected_sentence", "document_id": document_id, "errors": errors, "evidence_path": evidence_path.relative_to(ROOT).as_posix(), "characters": len(text)}


def verify_pairs(candidates: list[dict[str, Any]]) -> None:
    by_id = {candidate["source_record_id"]: candidate for candidate in candidates}
    pairs = read_jsonl(PAIRS)
    if len(pairs) > 25:
        raise ValueError("Recovery target exceeded")
    seen = set()
    for pair in pairs:
        record_id = pair["source_record_id"]
        if record_id in seen:
            raise ValueError("Duplicate recovered candidate")
        seen.add(record_id)
        candidate = by_id[record_id]
        if pair.get("pair_status") != "confirmed_pair" or pair.get("identity_errors"):
            raise ValueError("Pair status is not confirmed")
        if pair.get("source") != candidate["source"] or pair.get("outcome") != candidate["outcome"]:
            raise ValueError("Source or outcome changed during recovery")
        if pair.get("appeal_artifact") != candidate["appeal_artifact"]:
            raise ValueError("Appeal artifact changed during recovery")
        sentence = pair["sentence_artifact"]
        if sentence.get("case_number") != candidate["first_case_number"] or sentence.get("court") != candidate["first_court"] or sentence.get("decision_date") != candidate["first_decision_date"]:
            raise ValueError("Sentence identity metadata changed during recovery")
        if not re.fullmatch(r"https://sudact\.ru/regular/doc/[A-Za-z0-9]+/", sentence.get("source_url", "")):
            raise ValueError("Sentence source is not a direct public document")
        for artifact in (sentence, pair["appeal_artifact"]):
            for kind in ("raw", "text"):
                path = ROOT / artifact[f"{kind}_path"]
                if not path.is_file() or sha256(path) != artifact[f"{kind}_sha256"]:
                    raise ValueError(f"Invalid {kind} SHA for {record_id}")
        evidence_path = ROOT / pair["recovery_evidence_path"]
        if sha256(evidence_path) != pair["recovery_evidence_sha256"]:
            raise ValueError("Recovery evidence SHA mismatch")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("source_record_id") != record_id or evidence.get("source_url") != sentence["source_url"]:
            raise ValueError("Recovery evidence belongs to another document")
        if evidence.get("raw_sha256") != sentence["raw_sha256"] or evidence.get("text_sha256") != sentence["text_sha256"]:
            raise ValueError("Recovery evidence artifact SHA mismatch")
        if not evidence.get("robots_allows_document") or evidence.get("captcha_or_denial") or float(evidence.get("observed_pause_seconds", 0)) < 4:
            raise ValueError("Access conditions are not proven")
        robots_path = ROOT / evidence["robots_path"]
        if sha256(robots_path) != evidence["robots_sha256"]:
            raise ValueError("Robots evidence SHA mismatch")
        if not evidence.get("full_trial_verdict") or evidence.get("validation_errors"):
            raise ValueError("Recovery evidence is not a full trial verdict")
        sentence_path = ROOT / sentence["text_path"]
        appeal_path = ROOT / pair["appeal_artifact"]["text_path"]
        if sha256(sentence_path) == sha256(appeal_path):
            raise ValueError("Sentence and appeal texts are not independent")
        if new_trial_disposition(appeal_path.read_text(encoding="utf-8")):
            raise ValueError("New trial is not a confirmed training pair")
        errors = sentence_errors(sentence_path.read_text(encoding="utf-8"), candidate)
        if errors:
            raise ValueError(f"Invalid sentence for {record_id}: {errors}")


def manifest_records(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    searched = {record["source_record_id"] for record in read_jsonl(SEARCH_LOG)}
    paths: dict[Path, str] = {STATUS: "input", INDEX: "input", METADATA: "input"}
    for candidate in candidates:
        if candidate["source_record_id"] not in searched:
            continue
        appeal = candidate["appeal_artifact"]
        for kind in ("raw", "text", "identity_evidence"):
            value = appeal.get(f"{kind}_path")
            if value:
                paths[ROOT / value] = f"appeal_{kind}"
    for path, role in ((CANDIDATES, "recovery"), (PAIRS, "recovery")):
        paths[path] = role
    for directory in ("raw", "texts", "evidence"):
        for path in (OUT / directory).glob("**/*"):
            if path.is_file():
                paths[path] = f"recovery_{directory}"
    for pair in read_jsonl(PAIRS):
        for artifact_name in ("sentence_artifact", "appeal_artifact"):
            artifact = pair[artifact_name]
            for kind in ("raw", "text"):
                paths[ROOT / artifact[f"{kind}_path"]] = f"{artifact_name}_{kind}"
    records = []
    for path, role in sorted(paths.items(), key=lambda item: str(item[0])):
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "role": role,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })
    return records


def finalize() -> dict[str, Any]:
    candidates = read_jsonl(CANDIDATES)
    searches = read_jsonl(SEARCH_LOG)
    pairs = read_jsonl(PAIRS)
    if any(record.get("status") not in SEARCH_STATUSES for record in searches):
        raise ValueError("Unknown search status")
    if len({(record["source_record_id"], record["query"]) for record in searches}) != len(searches):
        raise ValueError("Duplicate search evidence")
    verify_pairs(candidates)
    manifest = manifest_records(candidates)
    write_jsonl(MANIFEST, manifest)
    searched_ids = {record["source_record_id"] for record in searches}
    by_source = Counter(candidate["source"] for candidate in candidates if candidate["source_record_id"] in searched_ids)
    outcomes = Counter(pair["outcome"] for pair in pairs)
    confirmed_sources = Counter(pair["source"] for pair in pairs)
    document_evidence = list((OUT / "evidence" / "documents").glob("*.json"))
    robots_evidence = list((OUT / "evidence" / "hosts").glob("*-robots.txt"))
    rejected = read_jsonl(REJECTED)
    rejected_documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in document_evidence
        if json.loads(path.read_text(encoding="utf-8")).get("validation_errors")
    ]
    summary = {
        "schema_version": 1,
        "processing_contract": CONTRACT,
        "eligible_missing_sentence": sum(1 for record in read_jsonl(STATUS) if record.get("pair_status") == "missing_sentence" and record.get("outcome") in ALLOWED_OUTCOMES),
        "identity_complete_candidates": len(candidates),
        "searched_candidates": len(searched_ids),
        "web_search_requests": len(searches),
        "exact_web_searches": sum(record.get("query_kind") != "expanded_number_court_without_date" for record in searches),
        "expanded_web_searches": sum(record.get("query_kind") == "expanded_number_court_without_date" for record in searches),
        "search_errors": sum(record.get("status") == "search_error" for record in searches),
        "additional_leads_checked": len(read_jsonl(LEADS)),
        "searched_by_source": dict(sorted(by_source.items())),
        "robots_requests": len(robots_evidence),
        "direct_document_requests": len(document_evidence),
        "recovered_full_pairs": len(pairs),
        "sources_confirmed_pairs": dict(sorted(confirmed_sources.items())),
        "outcomes_confirmed_pairs": dict(sorted(outcomes.items())),
        "rejected_new_trial": sum(record.get("pair_status") == "rejected_new_trial" for record in rejected),
        "rejected_documents": len(rejected_documents),
        "manifest_entries": len(manifest),
        "processed_dataset_or_index_changed": False,
    }
    resume = {
        "schema_version": 1,
        "processing_contract": CONTRACT,
        "searched_record_ids": sorted(searched_ids),
        "pending_record_ids": sorted(candidate["source_record_id"] for candidate in candidates if candidate["source_record_id"] not in searched_ids),
        "confirmed_pair_record_ids": sorted(pair["source_record_id"] for pair in pairs),
        "rejected_record_ids": sorted(record["source_record_id"] for record in rejected),
        "rejected_document_record_ids": sorted(record["source_record_id"] for record in rejected_documents),
        "stopped_on_captcha_or_denial": False,
    }
    write_json(SUMMARY, summary)
    write_json(RESUME, resume)
    if pairs:
        verification_note = "Каждая подтверждённая пара имеет отдельные raw и полный текст приговора, совпадающие суд, номер и дату в заголовке текста до названия суда, мотивировку и резолютивную часть. Исход сохранён из существующей апелляции. Цитаты, публикации, процессуальные акты и новые приговоры не принимались."
    else:
        verification_note = "Подтверждённых пар нет. Найденные документы без совпадающей identity, неполные тексты, процессуальные акты, новые приговоры и направления на новое рассмотрение не принимались."
    lines = [
        "# Восстановление новых полных пар",
        "",
        "Проверялись только missing_sentence с исходами changed, unchanged и cancelled. Для поиска использовались суд, полный номер и дата из единого индекса; слабые и выведенные aliases не применялись.",
        "",
        "## Итог",
        "",
        f"- Допустимых missing_sentence: {summary['eligible_missing_sentence']}",
        f"- С полными реквизитами первой инстанции: {len(candidates)}",
        f"- Проверено кандидатов через WebSearch: {len(searched_ids)}",
        f"- Выполнено поисковых запросов: {summary['web_search_requests']} (точных: {summary['exact_web_searches']}, расширенных: {summary['expanded_web_searches']})",
        f"- Ошибок поискового сервиса: {summary['search_errors']}",
        f"- Дополнительно проверено прямых зацепок: {summary['additional_leads_checked']}",
        f"- Запросов robots.txt: {summary['robots_requests']}",
        f"- Прямых запросов к страницам документов: {summary['direct_document_requests']}",
        f"- **Новых полных пар: {len(pairs)}**",
        f"- Отклонено направлений на новое рассмотрение: {summary['rejected_new_trial']}",
        f"- Отклонено документов после полной проверки: {summary['rejected_documents']}",
        f"- Записей в независимом SHA256 manifest: {summary['manifest_entries']}",
        "",
        "## Метод и ограничения",
        "",
        verification_note,
        "Запрещённые robots пути, поисковые endpoints и CAPTCHA не запрашивались. Остальные кандидаты остаются в resume без подтверждения.",
        "",
        "## Outcome подтверждённых пар",
        "",
        "| Outcome | Количество |",
        "|---|---:|",
    ]
    for outcome in ("changed", "unchanged", "cancelled"):
        lines.append(f"| {outcome} | {outcomes.get(outcome, 0)} |")
    lines += [
        "",
        "SHA-манифест и постатусный журнал сохранены отдельно. Агрегированный отчёт не содержит реквизитов конкретных дел, персональных данных, URL или хешей.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(REPORT)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "finalize"))
    args = parser.parse_args()
    if args.command == "prepare":
        candidates, counts = select_candidates()
        write_jsonl(CANDIDATES, candidates)
        for path in (SEARCH_LOG, PAIRS):
            if not path.exists():
                write_jsonl(path, [])
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
        (OUT / "texts").mkdir(parents=True, exist_ok=True)
        print(json.dumps(counts, ensure_ascii=False))
    else:
        print(json.dumps(finalize(), ensure_ascii=False))


if __name__ == "__main__":
    main()
