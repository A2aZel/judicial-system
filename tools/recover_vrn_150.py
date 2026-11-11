from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "corpus" / "recovery-150" / "voronezh-krasnoyarsk"
STATUS_PATH = ROOT / "corpus" / "pairs" / "voronezh-krasnoyarsk" / "pair-status.jsonl"
INDEX_PATH = ROOT / "corpus" / "regional" / "raw" / "collection-2026-11-10" / "unified-case-index.jsonl"
INTEGRATION_PATHS = {
    source: ROOT / "corpus" / "regional" / "raw" / "collection-2026-11-09" / source / "integration-records-20261109T193752526102Z.jsonl"
    for source in ("voronezh", "krasnoyarsk")
}
ALLOWED_OUTCOMES = {"changed", "unchanged", "cancelled"}
USER_AGENT = "recover-vrn-150/1.0"
MONTHS = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write(path, data.encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip().casefold()


def normalized_court(value: str) -> str:
    value = compact(value).replace("ё", "е")
    value = re.sub(r"\bг(?:ород)?\.?\b", " ", value)
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def html_to_text(fragment: str) -> str:
    fragment = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", " ", fragment, flags=re.S | re.I)
    fragment = re.sub(r"<br\s*/?>|</p>|</div>|</center>|</h[1-6]>", "\n", fragment, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def parse_sudact(page: str) -> dict[str, str]:
    lowered = page.casefold()
    if "check_captcha" in lowered or "доступ ограничен" in lowered or ("captcha" in lowered and "<h1>" not in lowered):
        raise PermissionError("captcha_or_denial")
    title = re.search(r"<h1>(.*?)</h1>", page, re.S | re.I)
    court = re.search(r'<div class="b-justice">(.*?)</div>', page, re.S | re.I)
    start = page.find('<hr class="hr-h1">')
    end = page.find("<h3>Суд:</h3>", start)
    if not title or start < 0:
        raise ValueError("document_layout_not_recognised")
    return {
        "title": html_to_text(title.group(1)),
        "court": html_to_text(court.group(1)) if court else "",
        "text": html_to_text(page[start:end if end > 0 else None]),
    }


def date_variants(iso_date: str) -> list[str]:
    year, month, day = (int(part) for part in iso_date.split("-"))
    return [f"{day:02d}.{month:02d}.{year:04d}", f"{day} {MONTHS[month - 1]} {year}"]


def verdict_errors(candidate: dict[str, Any], document: dict[str, str]) -> list[str]:
    text = document["text"]
    folded = compact(text).replace("ё", "е")
    title = compact(document["title"]).replace("ё", "е")
    marker = re.search(r"\bустановил[аи]?\b", folded)
    operative = re.search(r"\bприговорил[аи]?\b", folded)
    errors = []
    if not title.startswith("приговор") or "апелляцион" in title:
        errors.append("not_trial_verdict_title")
    if not marker or not operative:
        errors.append("missing_verdict_sections")
        return sorted(set(errors))
    if marker.end() >= operative.start() and operative.start() < marker.start():
        errors.append("operative_before_facts")
    header = folded[:marker.start()]
    number = re.escape(compact(candidate["first_case_number"])).replace(r"\-", r"[-‐‑‒–—]")
    if not re.search(rf"(?<![0-9]){number}(?![0-9])", header):
        errors.append("case_number_not_in_header")
    court = normalized_court(candidate["first_court"])
    if court not in normalized_court(header + " " + document["court"]):
        errors.append("court_not_in_header")
    if not any(compact(value).replace("ё", "е") in header for value in date_variants(candidate["first_decision_date"])):
        errors.append("decision_date_not_in_header")
    if len(text) < 3000:
        errors.append("text_too_short")
    if candidate.get("appeal_decision_date") and candidate["first_decision_date"] >= candidate["appeal_decision_date"]:
        errors.append("not_earlier_than_appeal")
    if re.search(r"\bапелляционн(?:ый|ого|ая|ое)\s+приговор\b", header):
        errors.append("new_appellate_verdict")
    return sorted(set(errors))


def select_candidates(out: Path = OUT) -> dict[str, Any]:
    statuses = read_jsonl(STATUS_PATH)
    index = {row["underlying_case_id"]: row for row in read_jsonl(INDEX_PATH)}
    integrations = {}
    for source, path in INTEGRATION_PATHS.items():
        integrations[source] = {row["document_id"]: row for row in read_jsonl(path)}
    candidates = []
    progress = {}
    counts = Counter()
    for row in statuses:
        if row["pair_status"] != "missing_sentence" or row["outcome"] not in ALLOWED_OUTCOMES:
            continue
        counts["eligible_status_outcome"] += 1
        pair_id = row["pair_id"]
        identity = index.get(row["verdict_group_id"], {}).get("identity", {})
        integration = integrations[row["source"]].get(row["appeal_artifact"]["document_id"], {})
        document_type = identity.get("document_type") or integration.get("document_type")
        if document_type == "appellate_verdict":
            progress[pair_id] = {"status": "excluded_new_sentence", "reason": "appellate_verdict"}
            counts["excluded_new_sentence"] += 1
            continue
        if not all(identity.get(key) for key in ("first_court", "first_case_number", "first_decision_date")):
            progress[pair_id] = {"status": "insufficient_identity", "reason": "unified_index_missing_first_court_number_or_date"}
            counts["insufficient_identity"] += 1
            continue
        metadata = integration.get("metadata", {})
        candidate = {
            "pair_id": pair_id,
            "verdict_group_id": row["verdict_group_id"],
            "source": row["source"],
            "outcome": row["outcome"],
            "first_court": identity["first_court"],
            "first_case_number": identity["first_case_number"],
            "first_decision_date": identity["first_decision_date"],
            "appeal_decision_date": metadata.get("decision_date"),
            "appeal_artifact": row["appeal_artifact"],
            "integration_provenance": identity.get("provenance"),
        }
        if candidate["appeal_decision_date"]:
            day, month, year = (int(part) for part in candidate["appeal_decision_date"].split("."))
            candidate["appeal_decision_date"] = f"{year:04d}-{month:02d}-{day:02d}"
        candidates.append(candidate)
        progress[pair_id] = {"status": "pending", "attempts": []}
    previous_path = out / "progress.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        for pair_id, state in previous.items():
            if pair_id in progress and state.get("status") not in {"pending", "insufficient_identity", "excluded_new_sentence"}:
                progress[pair_id] = state
    write_jsonl(out / "candidates.jsonl", candidates)
    write_json(out / "progress.json", progress)
    counts["actionable"] = len(candidates)
    counts["unique_underlying_cases"] = len({row["verdict_group_id"] for row in candidates})
    return dict(counts)


def allowed_document_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "sudact.ru" or parsed.query or parsed.fragment:
        raise ValueError("url_not_allowed")
    match = re.fullmatch(r"/regular/doc/([A-Za-z0-9_-]+)/", parsed.path)
    if not match:
        raise ValueError("url_not_allowed")
    return match.group(1)


def wait_for_slot(out: Path, pause: float) -> None:
    if pause < 4:
        raise ValueError("pause_must_be_at_least_4_seconds")
    state_path = out / "http-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("blocked"):
        raise PermissionError("network_collection_stopped_after_captcha_or_denial")
    last = state.get("last_get_monotonic")
    if last is not None:
        time.sleep(max(0.0, pause - (time.monotonic() - float(last))))
    request = urllib.request.Request("https://sudact.ru/robots.txt", headers={"User-Agent": USER_AGENT})
    return request


def get(out: Path, url: str, pause: float) -> tuple[bytes, str]:
    state_path = out / "http-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("blocked"):
        raise PermissionError("network_collection_stopped_after_captcha_or_denial")
    last_wall = state.get("last_get_at_utc")
    if last_wall:
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_wall)).total_seconds()
        time.sleep(max(0.0, pause - elapsed))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status != 200:
                raise PermissionError(f"http_denial_{response.status}")
            data = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403, 429}:
            state["blocked"] = True
            state["block_reason"] = f"http_{exc.code}"
            write_json(state_path, state)
            raise PermissionError(f"http_denial_{exc.code}") from exc
        raise
    state["last_get_at_utc"] = utc_now()
    write_json(state_path, state)
    lowered = data[:200000].decode("utf-8", "replace").casefold()
    if "check_captcha" in lowered or "доступ ограничен" in lowered or "captcha" in lowered:
        state["blocked"] = True
        state["block_reason"] = "captcha_or_denial_content"
        write_json(state_path, state)
        raise PermissionError("captcha_or_denial")
    return data, content_type


def fetch_document(pair_id: str, url: str, out: Path = OUT, pause: float = 4.0) -> dict[str, Any]:
    document_id = allowed_document_url(url)
    if pause < 4:
        raise ValueError("pause_must_be_at_least_4_seconds")
    candidates = {row["pair_id"]: row for row in read_jsonl(out / "candidates.jsonl")}
    if pair_id not in candidates:
        raise ValueError("unknown_or_ineligible_pair")
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    confirmed = sum(state.get("status") == "confirmed" for state in progress.values())
    if confirmed >= 25 and progress[pair_id].get("status") != "confirmed":
        raise ValueError("recovery_limit_25_reached")
    candidate = candidates[pair_id]
    robots_path = out / "evidence" / "sudact-robots.txt"
    if not robots_path.exists():
        robots_raw, _ = get(out, "https://sudact.ru/robots.txt", pause)
        atomic_write(robots_path, robots_raw)
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(robots_path.read_text(encoding="utf-8", errors="replace").splitlines())
    if not parser.can_fetch(USER_AGENT, url):
        raise PermissionError("robots_disallow")
    raw, content_type = get(out, url, pause)
    page = raw.decode("utf-8", "replace")
    document = parse_sudact(page)
    text = document["title"] + "\n" + document["court"] + "\n" + document["text"] + "\n"
    errors = verdict_errors(candidate, document)
    stem = f"{pair_id}-{document_id}"
    raw_path = out / "raw" / f"{stem}.html"
    text_path = out / "text" / f"{stem}.txt"
    atomic_write(raw_path, raw)
    atomic_write(text_path, text.encode("utf-8"))
    evidence = {
        "schema_version": 1,
        "pair_id": pair_id,
        "document_id": document_id,
        "source_url": url,
        "fetched_at_utc": utc_now(),
        "content_type": content_type,
        "robots_path": str(robots_path.relative_to(out)),
        "robots_sha256": sha256(robots_path),
        "raw_path": str(raw_path.relative_to(out)),
        "raw_sha256": sha256(raw_path),
        "text_path": str(text_path.relative_to(out)),
        "text_sha256": sha256(text_path),
        "expected_identity": {key: candidate[key] for key in ("first_court", "first_case_number", "first_decision_date", "appeal_decision_date")},
        "parsed_title": document["title"],
        "parsed_court": document["court"],
        "validation_errors": errors,
    }
    evidence_path = out / "evidence" / f"{stem}.json"
    write_json(evidence_path, evidence)
    state = progress[pair_id]
    state.setdefault("attempts", []).append({"document_id": document_id, "evidence_path": str(evidence_path.relative_to(out)), "errors": errors})
    if errors:
        state["status"] = "rejected_document"
        state["reason"] = "strict_verdict_validation_failed"
    else:
        state["status"] = "confirmed"
        state["sentence_artifact"] = {
            "document_id": document_id,
            "source_url": url,
            "raw_path": str(raw_path.relative_to(out)),
            "raw_sha256": evidence["raw_sha256"],
            "text_path": str(text_path.relative_to(out)),
            "text_sha256": evidence["text_sha256"],
            "evidence_path": str(evidence_path.relative_to(out)),
        }
        state.pop("reason", None)
    write_json(out / "progress.json", progress)
    return {"pair_id": pair_id, "status": state["status"], "errors": errors, "text_characters": len(text)}


def finalize(out: Path = OUT) -> dict[str, Any]:
    candidates = {row["pair_id"]: row for row in read_jsonl(out / "candidates.jsonl")}
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8")) if (out / "progress.json").exists() else {}
    pairs = []
    for pair_id, state in sorted(progress.items()):
        if state.get("status") != "confirmed" or pair_id not in candidates:
            continue
        candidate = candidates[pair_id]
        artifact = state["sentence_artifact"]
        raw_path = out / artifact["raw_path"]
        text_path = out / artifact["text_path"]
        evidence_path = out / artifact["evidence_path"]
        if sha256(raw_path) != artifact["raw_sha256"] or sha256(text_path) != artifact["text_sha256"]:
            raise ValueError("confirmed_artifact_sha_mismatch")
        pairs.append({
            "schema_version": 1,
            "pair_id": pair_id,
            "verdict_group_id": candidate["verdict_group_id"],
            "source": candidate["source"],
            "outcome": candidate["outcome"],
            "first_court": candidate["first_court"],
            "first_case_number": candidate["first_case_number"],
            "first_decision_date": candidate["first_decision_date"],
            "sentence_artifact": artifact,
            "appeal_artifact": candidate["appeal_artifact"],
            "recovery": "new_direct_public_trial_verdict_strictly_validated",
        })
    if len(pairs) > 25:
        raise ValueError("more_than_25_confirmed_pairs")
    write_jsonl(out / "recovered-pairs.jsonl", pairs)
    manifest = []
    for directory, suffix in (("raw", ".html"), ("text", ".txt"), ("evidence", ".json")):
        for path in sorted((out / directory).glob("*" + suffix)):
            manifest.append({"path": str(path.relative_to(out)), "sha256": sha256(path), "bytes": path.stat().st_size})
    for name in ("candidates.jsonl", "recovered-pairs.jsonl", "progress.json"):
        path = out / name
        if path.exists():
            manifest.append({"path": name, "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest.append({"path": "report.md"})
    write_report(out, pairs, progress, manifest)
    for row in manifest:
        path = out / row["path"]
        row["sha256"] = sha256(path)
        row["bytes"] = path.stat().st_size
    write_jsonl(out / "sha256-manifest.jsonl", manifest)
    return {"confirmed_pairs": len(pairs), "manifest_entries": len(manifest)}


def write_report(out: Path, pairs: list[dict[str, Any]], progress: dict[str, Any], manifest: list[dict[str, Any]]) -> None:
    statuses = Counter(state.get("status", "unknown") for state in progress.values())
    sources = Counter(pair["source"] for pair in pairs)
    outcomes = Counter(pair["outcome"] for pair in pairs)
    lines = [
        "# Восстановление полных пар: Voronezh/Krasnoyarsk",
        "",
        "Отчёт содержит только агрегаты. Детальные идентификаторы, адреса и хеши остаются в локальных evidence и manifest.",
        "",
        "## Итог",
        "",
        f"- Новых строго подтверждённых полных пар: **{len(pairs)}** (лимит потока: 25).",
        f"- Исходный набор не изменялся; предполагаемый размер после отдельного импорта: **{77 + len(pairs)}** пар.",
        f"- Записей с допустимыми статусом и outcome: **{sum(statuses.values())}**.",
        f"- Кандидатов с полной исходной тройкой суд/номер/дата: **{len(read_jsonl(out / 'candidates.jsonl'))}**.",
        f"- Независимый SHA-256 manifest: **{len(manifest)}** файлов.",
        "",
        "## Подтверждённые пары по регионам",
        "",
        "| Регион | Пар |",
        "|---|---:|",
        f"| Voronezh | {sources.get('voronezh', 0)} |",
        f"| Krasnoyarsk | {sources.get('krasnoyarsk', 0)} |",
        "",
        "## Подтверждённые пары по outcome",
        "",
        "| Outcome | Пар |",
        "|---|---:|",
    ]
    for outcome in ("changed", "unchanged", "cancelled"):
        lines.append(f"| {outcome} | {outcomes.get(outcome, 0)} |")
    lines += ["", "## Состояние всех допустимых исходных записей", "", "| Состояние | Записей |", "|---|---:|"]
    for status, count in sorted(statuses.items()):
        lines.append(f"| {status} | {count} |")
    lines += [
        "",
        "## Критерии и ограничения",
        "",
        "- Принят только отдельный полный первичный приговор с совпадающими судом, номером и датой в заголовке, разделами `установил` и `приговорил`, а также датой раньше апелляции.",
        "- Цитаты в апелляции, процессуальные акты, новые апелляционные приговоры и записи без полной исходной тройки не принимались.",
        "- Загружались только разрешённые прямые публичные документы; запрещённые endpoints и региональные GAS не запрашивались.",
        "- Статус `no_direct_result` означает только отсутствие подходящего прямого документа в результатах точного веб-поиска, а не доказанное отсутствие документа.",
        "- Два дополнительных уточняющих запроса завершились ошибкой поискового backend; основные точные запросы по всем уникальным тройкам выполнены.",
        "- Возобновление обеспечивается `progress.json`; сырые ответы, извлечённые тексты и evidence сохраняются локально при наличии документа.",
        "- Нулевой результат допустим: отсутствие подтверждения не заменялось косвенным совпадением.",
        "",
    ]
    atomic_write(out / "report.md", "\n".join(lines).encode("utf-8"))


def mark_search(pair_id: str, status: str, query: str, out: Path = OUT) -> dict[str, Any]:
    if status not in {"no_direct_result", "rejected_search_result"}:
        raise ValueError("invalid_search_status")
    candidates = {row["pair_id"] for row in read_jsonl(out / "candidates.jsonl")}
    if pair_id not in candidates:
        raise ValueError("unknown_or_ineligible_pair")
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    if progress[pair_id].get("status") == "confirmed":
        raise ValueError("confirmed_pair_cannot_be_marked_unresolved")
    progress[pair_id].update({"status": status, "last_exact_query": query, "searched_at_utc": utc_now()})
    write_json(out / "progress.json", progress)
    return {"pair_id": pair_id, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select")
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--pair", required=True)
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--pause", type=float, default=4.0)
    mark = sub.add_parser("mark")
    mark.add_argument("--pair", required=True)
    mark.add_argument("--status", required=True)
    mark.add_argument("--query", required=True)
    sub.add_parser("finalize")
    args = parser.parse_args()
    try:
        if args.command == "select":
            result = select_candidates()
        elif args.command == "fetch":
            result = fetch_document(args.pair, args.url, pause=args.pause)
        elif args.command == "mark":
            result = mark_search(args.pair, args.status, args.query)
        else:
            result = finalize()
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, PermissionError, urllib.error.URLError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
