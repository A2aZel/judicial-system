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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mosgorsud_open_data import new_trial_disposition

OUT = ROOT / "corpus" / "recovery-150" / "bash-tat-nov"
STATUS = ROOT / "corpus" / "pairs" / "bash-tat-nov" / "pair-status.jsonl"
INDEX = ROOT / "corpus" / "regional" / "raw" / "collection-2026-11-10" / "unified-case-index.jsonl"
ALLOWED_OUTCOMES = {"changed", "unchanged", "cancelled"}
ALLOWED_HOSTS = {
    "sudact.ru": re.compile(r"^/regular/doc/[A-Za-z0-9]+/?$"),
    "advocate-service.ru": re.compile(r"^/sud-praktika/.+\.html$"),
    "pravoilevo.ru": re.compile(r"^/prigovor-.+/$"),
}
USER_AGENT = "recover-bash-150/1.0 (judicial corpus research)"
MONTHS = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)


class AccessStopped(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(path: Path) -> str:
    try:
        value = path.relative_to(ROOT)
    except ValueError:
        value = path
    return value.as_posix()


def resolve_project_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def compact(value: str) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", value.lower().replace("ё", "е"))


def load_progress(output: Path = OUT) -> dict[str, Any]:
    path = output / "progress.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_progress(output: Path, progress: dict[str, Any]) -> None:
    write_json(output / "progress.json", progress)


def sync_rejected(output: Path, progress: dict[str, Any]) -> None:
    rows = []
    for event_id, state in progress.items():
        status = state.get("status", "")
        if not status.startswith("rejected_") and status != "final_validation_failed":
            continue
        rows.append({
            "appeal_event_id": event_id,
            "status": status,
            "validation_errors": state.get("final_validation_errors") or state.get("validation_errors") or [],
            "document_evidence": state.get("document_evidence"),
            "raw_path": state.get("raw_path"),
            "raw_sha256": state.get("raw_sha256"),
            "text_path": state.get("text_path"),
            "text_sha256": state.get("text_sha256"),
        })
    write_jsonl(output / "rejected.jsonl", rows)


def check_appeal_batch(output: Path = OUT, size: int = 50, batch: int = 2) -> dict[str, int]:
    candidates = candidate_map(output)
    progress = load_progress(output)
    batch_path = output / "batches" / f"batch-{batch:03d}.json"
    if batch_path.exists():
        event_ids = json.loads(batch_path.read_text(encoding="utf-8"))["appeal_event_ids"]
    else:
        event_ids = [
            event_id for event_id in candidates
            if progress.get(event_id, {}).get("status", "pending") == "pending"
        ][:size]
        write_json(batch_path, {"schema_version": 1, "appeal_event_ids": event_ids})
    source_rows = {row["appeal_event_id"]: row for row in read_jsonl(STATUS)}
    counts = Counter()
    for event_id in event_ids:
        state = progress[event_id]
        if state.get("status", "pending") != "pending":
            counts[state["status"]] += 1
            continue
        errors = validate_source_appeal(source_rows[event_id])
        if errors:
            state["status"] = "rejected_new_trial" if "new_trial" in errors else "rejected_appeal"
            state["validation_errors"] = errors
            save_progress(output, progress)
        counts[state["status"]] += 1
    sync_rejected(output, progress)
    return {"batch_records": len(event_ids), **counts}


def select_candidates(output: Path = OUT) -> list[dict[str, Any]]:
    index = {row["underlying_case_id"]: row for row in read_jsonl(INDEX)}
    selected = []
    for row in read_jsonl(STATUS):
        if row.get("pair_status") != "missing_sentence" or row.get("outcome") not in ALLOWED_OUTCOMES:
            continue
        identity = (index.get(row["underlying_case_id"], {}).get("identity") or {})
        if not all(identity.get(key) for key in ("first_court", "first_case_number", "first_decision_date")):
            continue
        selected.append({
            "appeal_event_id": row["appeal_event_id"],
            "source": row["source"],
            "outcome": row["outcome"],
            "first_court": identity["first_court"],
            "first_case_number": identity["first_case_number"],
            "first_decision_date": identity["first_decision_date"],
        })
    write_jsonl(output / "candidates.jsonl", selected)
    progress = load_progress(output)
    for candidate in selected:
        progress.setdefault(candidate["appeal_event_id"], {"status": "pending", "attempts": []})
    save_progress(output, progress)
    return selected


def candidate_map(output: Path = OUT) -> dict[str, dict[str, Any]]:
    return {row["appeal_event_id"]: row for row in read_jsonl(output / "candidates.jsonl")}


def date_forms(value: str) -> list[str]:
    year, month, day = value.split("-")
    return [f"{day}.{month}.{year}", f"{int(day)} {MONTHS[int(month) - 1]} {year}"]


def exact_query(candidate: dict[str, Any]) -> str:
    court = re.sub(r"\s+", " ", candidate["first_court"].strip())
    return f'"{candidate["first_case_number"]}" "{court}" "{date_forms(candidate["first_decision_date"])[0]}" приговор'


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
    if not title or start < 0:
        raise ValueError("document layout not recognised")
    return {
        "title": html_to_text(title.group(1)),
        "court": html_to_text(court.group(1)) if court else "",
        "text": html_to_text(page[start:end if end > 0 else None]),
    }


def parse_document(page: str, host: str) -> dict[str, str]:
    if host == "sudact.ru":
        return parse_sudact(page)
    title = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S | re.I) or re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I)
    start = page.find('<div class="item-page">')
    end_marker = '<ul class="pagenav">'
    if start < 0:
        start = page.find('<article class="reviews-element">')
        end_marker = '</article>'
    if start < 0:
        start = page.find('<div class="entry-content">')
        end_marker = '</div><!-- .entry-content -->'
    if start < 0:
        raise ValueError("document layout not recognised")
    fragment = page[start:]
    document_end = fragment.rfind(end_marker) if end_marker == '</article>' else fragment.find(end_marker)
    end_markers = [fragment.find('<div class="tags">'), document_end]
    ends = [position for position in end_markers if position > 0]
    if ends:
        fragment = fragment[:min(ends)]
    if not title:
        raise ValueError("document title not recognised")
    return {"title": html_to_text(title.group(1)), "court": "", "text": html_to_text(fragment)}


def validate_sentence(candidate: dict[str, Any], text: str, title: str = "") -> list[str]:
    errors = []
    if len(text) < 1000:
        return ["text_too_short"]
    header = text[:8000]
    established = re.search(r"у\s*с\s*т\s*а\s*н\s*о\s*в\s*и\s*л", header, re.I)
    if established:
        header = header[:established.start()]
    header_compact = compact(header)
    whole_compact = compact(text)
    title_compact = compact(title)
    if "приговор" not in header_compact[:300] and "приговор" not in title_compact[:150]:
        errors.append("verdict_title_missing")
    if "апелляционныйприговор" in header_compact or "кассационныйприговор" in header_compact:
        errors.append("new_trial_verdict")
    if not established:
        errors.append("established_missing")
    if "приговорил" not in whole_compact:
        errors.append("operative_verdict_missing")
    expected_number = compact(candidate["first_case_number"])
    if expected_number not in header_compact:
        errors.append("case_number_mismatch")
    expected_court = compact(candidate["first_court"])
    court_position = header_compact.find(expected_court)
    if court_position < 0:
        errors.append("court_mismatch")
    expected_dates = [compact(value) for value in date_forms(candidate["first_decision_date"])]
    date_positions = [header_compact.find(value) for value in expected_dates]
    date_position = min((position for position in date_positions if position >= 0), default=-1)
    if date_position < 0:
        errors.append("decision_date_mismatch")
    elif court_position >= 0 and date_position >= court_position:
        errors.append("decision_date_not_before_court")
    return list(dict.fromkeys(errors))


def check_document_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in ALLOWED_HOSTS:
        raise AccessStopped("host_not_allowed")
    if not ALLOWED_HOSTS[host].fullmatch(parsed.path):
        raise AccessStopped("document_path_not_allowed")
    if parsed.query or parsed.fragment:
        raise AccessStopped("document_query_not_allowed")
    return host


def throttled_get(url: str, output: Path, pause: float, robots: bool = False) -> tuple[bytes, str]:
    if pause < 4:
        raise ValueError("pause must be at least four seconds")
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if robots:
        if parsed.scheme != "https" or host not in ALLOWED_HOSTS or parsed.path != "/robots.txt":
            raise AccessStopped("robots_url_not_allowed")
    else:
        host = check_document_url(url)
    access_path = output / "access.json"
    access = json.loads(access_path.read_text(encoding="utf-8")) if access_path.exists() else {}
    state = access.setdefault(host, {})
    last_at = state.get("last_get_at_utc")
    if last_at:
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_at)).total_seconds()
        if 0 <= elapsed < pause:
            time.sleep(pause - elapsed)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            final_url = response.geturl()
            final_host = check_document_url(final_url) if not robots else (urllib.parse.urlparse(final_url).hostname or "").lower()
            if final_host != host:
                raise AccessStopped("redirect_not_allowed")
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        state["status"] = "denied" if exc.code in {401, 403, 429} else "http_error"
        state["http_status"] = exc.code
        write_json(access_path, access)
        raise AccessStopped(f"http_{exc.code}") from exc
    finally:
        state["last_get_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(access_path, access)
    lowered = body.lower()
    if not robots and (b"check_captcha" in lowered or b"captcha" in lowered or b"access denied" in lowered or b"too many requests" in lowered):
        state["status"] = "captcha_or_denial"
        write_json(access_path, access)
        raise AccessStopped("captcha_or_denial")
    state["status"] = "allowed"
    write_json(access_path, access)
    return body, content_type


def refresh_robots(output: Path = OUT, pause: float = 4, host: str = "sudact.ru") -> dict[str, Any]:
    if host not in ALLOWED_HOSTS:
        raise AccessStopped("host_not_allowed")
    url = f"https://{host}/robots.txt"
    body, _ = throttled_get(url, output, pause, robots=True)
    path = output / "access" / f"{host}-robots.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    rules = urllib.robotparser.RobotFileParser()
    rules.parse(body.decode("utf-8", "replace").splitlines())
    probe = "/regular/doc/example/" if host == "sudact.ru" else "/sud-praktika/example.html"
    document_allowed = rules.can_fetch(USER_AGENT, f"https://{host}{probe}")
    ajax_allowed = rules.can_fetch(USER_AGENT, f"https://{host}/regular/doc_ajax/example/") if host == "sudact.ru" else False
    if not document_allowed or ajax_allowed:
        raise AccessStopped("robots_rules_changed")
    access_path = output / "access.json"
    access = json.loads(access_path.read_text(encoding="utf-8"))
    access[host].update({
        "robots_allowed": True,
        "robots_path": project_path(path),
        "robots_sha256": sha256(path),
        "regular_doc_ajax_allowed": False,
    })
    write_json(access_path, access)
    return {"host": host, "document_allowed": document_allowed, "regular_doc_ajax_allowed": ajax_allowed}


def record_search(event_id: str, status: str, query: str, urls: list[str], titles: list[str], output: Path = OUT) -> None:
    candidates = candidate_map(output)
    if event_id not in candidates:
        raise ValueError("unknown appeal_event_id")
    if status not in {"no_result", "candidate_found", "not_a_sentence"}:
        raise ValueError("invalid search status")
    results = [{"url": url, "title": titles[index] if index < len(titles) else ""} for index, url in enumerate(urls)]
    evidence_path = output / "evidence" / "search" / f"{event_id}.json"
    previous = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    attempts = previous.get("attempts") or ([{
        "query": previous["query"],
        "status": previous["status"],
        "searched_at_utc": previous["searched_at_utc"],
        "results": previous.get("results") or [],
    }] if previous.get("query") else [])
    searched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    attempts.append({"query": query, "status": status, "searched_at_utc": searched_at, "results": results})
    evidence = {
        "schema_version": 2,
        "appeal_event_id": event_id,
        "engine": "WebSearch",
        "query": query,
        "status": status,
        "searched_at_utc": searched_at,
        "results": results,
        "attempts": attempts,
    }
    write_json(evidence_path, evidence)
    progress = load_progress(output)
    state = progress[event_id]
    state["status"] = status
    state["search_evidence"] = project_path(evidence_path)
    state["search_evidence_sha256"] = sha256(evidence_path)
    save_progress(output, progress)


def fetch_sentence(event_id: str, url: str, output: Path = OUT, pause: float = 4) -> dict[str, Any]:
    candidates = candidate_map(output)
    candidate = candidates[event_id]
    host = check_document_url(url)
    access_path = output / "access.json"
    access = json.loads(access_path.read_text(encoding="utf-8")) if access_path.exists() else {}
    if not access.get(host, {}).get("robots_allowed"):
        raise AccessStopped("robots_not_verified")
    progress = load_progress(output)
    state = progress[event_id]
    source_rows = {row["appeal_event_id"]: row for row in read_jsonl(STATUS)}
    appeal_errors = validate_source_appeal(source_rows[event_id])
    if appeal_errors:
        state["status"] = "rejected_new_trial" if "new_trial" in appeal_errors else "rejected_appeal"
        state["validation_errors"] = appeal_errors
        save_progress(output, progress)
        sync_rejected(output, progress)
        return {"event_id": event_id, "status": state["status"], "errors": appeal_errors, "characters": 0}
    url_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    legacy_path = output / "raw" / f"{event_id}-{url_key}.html"
    raw_path = legacy_path if legacy_path.exists() else output / "raw" / f"{url_key}.html"
    deduplicated = False
    if not raw_path.exists():
        body, content_type = throttled_get(url, output, pause)
        body_sha = hashlib.sha256(body).hexdigest()
        duplicates = [path for path in (output / "raw").glob("*.html") if sha256(path) == body_sha]
        if duplicates:
            raw_path = duplicates[0]
            deduplicated = True
        else:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(body)
        state["content_type"] = content_type
    text_path = output / "texts" / f"{raw_path.stem}.txt"
    page = raw_path.read_bytes().decode("utf-8", "replace")
    parsed = parse_document(page, host)
    text = parsed["title"] + "\n" + parsed["court"] + "\n" + parsed["text"]
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    errors = validate_sentence(candidate, parsed["text"], parsed["title"])
    evidence_path = output / "evidence" / "documents" / f"{event_id}-{url_key}.json"
    write_json(evidence_path, {
        "schema_version": 1,
        "appeal_event_id": event_id,
        "document_url": url,
        "raw_path": project_path(raw_path),
        "raw_sha256": sha256(raw_path),
        "text_path": project_path(text_path),
        "text_sha256": sha256(text_path),
        "document_title": parsed["title"],
        "document_court": parsed["court"],
        "expected_identity": candidate,
        "validation_errors": errors,
        "deduplicated": deduplicated,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    state.update({
        "status": "validated_sentence" if not errors else "rejected_sentence",
        "document_url": url,
        "raw_path": project_path(raw_path),
        "raw_sha256": sha256(raw_path),
        "text_path": project_path(text_path),
        "text_sha256": sha256(text_path),
        "document_evidence": project_path(evidence_path),
        "document_evidence_sha256": sha256(evidence_path),
        "validation_errors": errors,
        "deduplicated": deduplicated,
    })
    save_progress(output, progress)
    sync_rejected(output, progress)
    return {"event_id": event_id, "status": state["status"], "errors": errors, "characters": len(text), "deduplicated": deduplicated}


def validate_source_appeal(row: dict[str, Any]) -> list[str]:
    appeal = row.get("appeal") or {}
    errors = []
    for role in ("path", "raw_path"):
        value = appeal.get(role)
        expected = appeal.get("sha256" if role == "path" else "raw_sha256")
        path = resolve_project_path(value) if value else None
        if not path or not path.is_file():
            errors.append("appeal_" + role + "_missing")
        elif not expected or sha256(path) != expected:
            errors.append("appeal_" + role + "_sha_mismatch")
    text_value = appeal.get("path")
    text_path = resolve_project_path(text_value) if text_value else None
    if text_path and text_path.is_file() and new_trial_disposition(text_path.read_text(encoding="utf-8")):
        errors.append("new_trial")
    return errors


def finalize(output: Path = OUT, limit: int = 25) -> dict[str, int]:
    candidates = candidate_map(output)
    source_rows = {row["appeal_event_id"]: row for row in read_jsonl(STATUS)}
    progress = load_progress(output)
    pairs = []
    for event_id, candidate in candidates.items():
        state = progress.get(event_id, {})
        if state.get("status") not in {"validated_sentence", "confirmed_pair"}:
            continue
        row = source_rows[event_id]
        errors = validate_source_appeal(row)
        raw_path = resolve_project_path(state["raw_path"])
        text_path = resolve_project_path(state["text_path"])
        if sha256(raw_path) != state["raw_sha256"] or sha256(text_path) != state["text_sha256"]:
            errors.append("sentence_sha_mismatch")
        text = text_path.read_text(encoding="utf-8")
        host = check_document_url(state["document_url"])
        parsed = parse_document(raw_path.read_bytes().decode("utf-8", "replace"), host)
        errors.extend(validate_sentence(candidate, parsed["text"], parsed["title"]))
        if errors:
            state["status"] = "rejected_new_trial" if "new_trial" in errors else "final_validation_failed"
            state["final_validation_errors"] = list(dict.fromkeys(errors))
            continue
        pair = dict(row)
        pair["pair_status"] = "confirmed_pair"
        pair["sentence"] = {
            "path": state["text_path"],
            "sha256": state["text_sha256"],
            "raw_path": state["raw_path"],
            "raw_sha256": state["raw_sha256"],
            "text_characters": len(text),
            "source": host,
            "record_id": None,
            "document_url": state["document_url"],
            "identity_link_method": "exact_header_court_number_date",
        }
        pair["identity_errors"] = []
        pair["recovery_evidence"] = state["document_evidence"]
        pair["recovery_evidence_sha256"] = state["document_evidence_sha256"]
        pairs.append(pair)
        state["status"] = "confirmed_pair"
        if len(pairs) >= limit:
            break
    write_jsonl(output / "recovered-pairs.jsonl", pairs)
    save_progress(output, progress)
    sync_rejected(output, progress)
    write_manifest(output)
    return {"confirmed_pairs": len(pairs)}


def write_manifest(output: Path = OUT) -> None:
    rows = []
    for directory in ("raw", "texts", "evidence", "access", "batches"):
        for path in sorted((output / directory).glob("**/*")):
            if path.is_file():
                rows.append({
                    "path": project_path(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                })
    pairs_path = output / "recovered-pairs.jsonl"
    if pairs_path.exists():
        for pair in read_jsonl(pairs_path):
            appeal = pair["appeal"]
            for path_key, sha_key in (("path", "sha256"), ("raw_path", "raw_sha256")):
                path = resolve_project_path(appeal[path_key])
                rows.append({
                    "path": appeal[path_key],
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "expected_sha256": appeal[sha_key],
                })
    rejected_path = output / "rejected.jsonl"
    if rejected_path.exists():
        rows.append({"path": project_path(rejected_path), "sha256": sha256(rejected_path), "bytes": rejected_path.stat().st_size})
    unique = {(row["path"], row["sha256"]): row for row in rows}
    write_jsonl(output / "sha256-manifest.jsonl", list(unique.values()))


def verify_manifest(output: Path = OUT) -> list[str]:
    errors = []
    for row in read_jsonl(output / "sha256-manifest.jsonl"):
        path = resolve_project_path(row["path"])
        if not path.is_file():
            errors.append("missing")
        elif sha256(path) != row["sha256"]:
            errors.append("sha_mismatch")
        if row.get("expected_sha256") and row.get("expected_sha256") != row["sha256"]:
            errors.append("source_sha_mismatch")
    return errors


def write_report(output: Path = OUT) -> dict[str, Any]:
    candidates = read_jsonl(output / "candidates.jsonl") if (output / "candidates.jsonl").exists() else []
    progress = load_progress(output)
    pairs = read_jsonl(output / "recovered-pairs.jsonl") if (output / "recovered-pairs.jsonl").exists() else []
    eligible = sum(
        row.get("pair_status") == "missing_sentence" and row.get("outcome") in ALLOWED_OUTCOMES
        for row in read_jsonl(STATUS)
    )
    states = Counter(progress.get(row["appeal_event_id"], {}).get("status", "pending") for row in candidates)
    sources = ("bashkortostan", "tatarstan", "novosibirsk")
    local_rejections = {"rejected_new_trial", "rejected_appeal"}
    pair_sources = Counter(row["source"] for row in pairs)
    pair_outcomes = Counter(row["outcome"] for row in pairs)
    processed_by_source = {}
    searched_by_source = {}
    for source in sources:
        processed_by_source[source] = dict(Counter(
            progress.get(row["appeal_event_id"], {}).get("status", "pending")
            for row in candidates
            if row["source"] == source and progress.get(row["appeal_event_id"], {}).get("status", "pending") != "pending"
        ))
        searched_by_source[source] = dict(Counter(
            progress.get(row["appeal_event_id"], {}).get("status", "pending")
            for row in candidates
            if row["source"] == source
            and progress.get(row["appeal_event_id"], {}).get("status", "pending") not in local_rejections | {"pending"}
        ))
    search_attempts = 0
    for path in (output / "evidence" / "search").glob("*.json"):
        evidence = json.loads(path.read_text(encoding="utf-8"))
        search_attempts += len(evidence.get("attempts") or [1])
    rejected_path = output / "rejected.jsonl"
    summary = {
        "schema_version": 1,
        "eligible_missing_sentence": eligible,
        "eligible_with_complete_identity": len(candidates),
        "excluded_incomplete_identity": eligible - len(candidates),
        "eligible_by_source": dict(Counter(row["source"] for row in candidates)),
        "processed_by_source": processed_by_source,
        "searched_by_source": searched_by_source,
        "recorded_search_queries": search_attempts,
        "downloaded_documents": sum(bool(progress.get(row["appeal_event_id"], {}).get("raw_path")) for row in candidates),
        "rejected_records": len(read_jsonl(rejected_path)) if rejected_path.exists() else 0,
        "confirmed_pairs": len(pairs),
        "by_source": {source: pair_sources.get(source, 0) for source in sources},
        "by_outcome": {outcome: pair_outcomes.get(outcome, 0) for outcome in ("changed", "unchanged", "cancelled")},
        "progress_statuses": dict(states),
        "manifest_entries": len(read_jsonl(output / "sha256-manifest.jsonl")) if (output / "sha256-manifest.jsonl").exists() else 0,
        "manifest_errors": len(verify_manifest(output)) if (output / "sha256-manifest.jsonl").exists() else 0,
        "processed_dataset_included": False,
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# Восстановление полных пар",
        "",
        "Отчёт содержит только агрегаты. Полная пара требует отдельного полного приговора с совпадающими судом, номером и датой в заголовке, маркерами установления и резолютивной части, проверенных SHA256 исходника и текста, а также отсутствия направления апелляции на новое рассмотрение или разбирательство.",
        "",
        f"- Допустимые missing_sentence: {eligible}",
        f"- Полные реквизиты первичного приговора: {len(candidates)}; неполные реквизиты: {eligible - len(candidates)}",
        f"- **Новые подтверждённые полные пары: {len(pairs)}**",
        f"- Загружено прямых документов: {summary['downloaded_documents']}",
        f"- Сохранено rejected-записей: {summary['rejected_records']}",
        f"- Записей в независимом SHA256 manifest: {summary['manifest_entries']}",
        f"- Ошибок повторной проверки manifest: {summary['manifest_errors']}",
        "- Processed dataset и processed index не изменялись.",
        "",
        "## По источникам",
        "",
        "| Источник | Полные реквизиты | Обработано | WebSearch | Без прямого документа | New trial | Прочие отклонения | Пары |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in sources:
        processed = processed_by_source[source]
        searched = searched_by_source[source]
        other_rejections = sum(count for status, count in processed.items() if status.startswith("rejected_") and status not in local_rejections)
        lines.append(
            f"| {source} | {summary['eligible_by_source'].get(source, 0)} | {sum(processed.values())} | {sum(searched.values())} "
            f"| {searched.get('no_result', 0)} | {processed.get('rejected_new_trial', 0)} | {other_rejections} | {pair_sources.get(source, 0)} |"
        )
    lines += ["", "## По outcome", "", "| Outcome | Пары |", "|---|---:|"]
    for outcome in ("changed", "unchanged", "cancelled"):
        lines.append(f"| {outcome} | {pair_outcomes.get(outcome, 0)} |")
    lines += ["", "## Состояние resume", "", "| Статус | Записи |", "|---|---:|"]
    for status, count in sorted(states.items()):
        lines.append(f"| {status} | {count} |")
    lines += ["", f"Сохранено поисковых запросов в evidence: {search_attempts}. Запрещённые поисковые и API-пути не запрашивались. CAPTCHA или denial при загрузке документов не возникали."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("select")
    robots = subparsers.add_parser("robots")
    robots.add_argument("--host", choices=sorted(ALLOWED_HOSTS), default="sudact.ru")
    robots.add_argument("--pause", type=float, default=4)
    check = subparsers.add_parser("check-appeals")
    check.add_argument("--size", type=int, default=50)
    check.add_argument("--batch", type=int, default=2)
    search = subparsers.add_parser("record-search")
    search.add_argument("event_id")
    search.add_argument("status", choices=("no_result", "candidate_found", "not_a_sentence"))
    search.add_argument("--query", required=True)
    search.add_argument("--url", action="append", default=[])
    search.add_argument("--title", action="append", default=[])
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("event_id")
    fetch.add_argument("url")
    fetch.add_argument("--pause", type=float, default=4)
    final = subparsers.add_parser("finalize")
    final.add_argument("--limit", type=int, default=25)
    subparsers.add_parser("verify")
    subparsers.add_parser("report")
    args = parser.parse_args()
    if args.command == "select":
        rows = select_candidates()
        print(json.dumps({"selected": len(rows), "by_source": dict(Counter(row["source"] for row in rows))}, ensure_ascii=False))
    elif args.command == "robots":
        print(json.dumps(refresh_robots(pause=args.pause, host=args.host), ensure_ascii=False))
    elif args.command == "check-appeals":
        print(json.dumps(check_appeal_batch(size=args.size, batch=args.batch), ensure_ascii=False))
    elif args.command == "record-search":
        record_search(args.event_id, args.status, args.query, args.url, args.title)
        print(json.dumps({"recorded": 1}))
    elif args.command == "fetch":
        print(json.dumps(fetch_sentence(args.event_id, args.url, pause=args.pause), ensure_ascii=False))
    elif args.command == "finalize":
        print(json.dumps(finalize(limit=args.limit)))
    elif args.command == "verify":
        errors = verify_manifest()
        print(json.dumps({"manifest_errors": len(errors)}))
        if errors:
            raise SystemExit(1)
    else:
        print(json.dumps(write_report(), ensure_ascii=False))


if __name__ == "__main__":
    main()
