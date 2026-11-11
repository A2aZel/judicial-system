from __future__ import annotations

import argparse
import codecs
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "corpus" / "mosgorsud" / "raw"
DEFAULT_REPORT = ROOT / "corpus" / "mosgorsud" / "report.md"
TARGET_YEARS = tuple(range(2022, 2026))
ALLOWED_HOSTS = {"data.mos-gorsud.ru", "mos-gorsud.ru"}
MONTHS = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    slug: str
    version: int
    modified: str
    filename: str

    @property
    def page_url(self) -> str:
        return f"https://data.mos-gorsud.ru/opendata/{self.slug}"

    @property
    def api_url(self) -> str:
        return f"https://data.mos-gorsud.ru/api/opendata/{self.slug}/v{self.version}/data/json"


APPEALS = DatasetSpec(
    "appeals",
    "1577117758402-casesgroupmgsappealcriminalquiik",
    287,
    "2026-11-08",
    "appeals-2026-11-08.json.zip",
)
FIRST_INSTANCE = DatasetSpec(
    "first",
    "1577116724824-casesgroupfirstcriminalluivn",
    287,
    "2026-11-08",
    "first-instance-2026-11-08.json.zip",
)
DATASETS = {item.key: item for item in (APPEALS, FIRST_INSTANCE)}


class DownloadError(RuntimeError):
    pass


class CaptchaError(DownloadError):
    pass


class UnsupportedFormat(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_url(url: str) -> None:
    parsed = urlparse(url)
    path = parsed.path.lower()
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise DownloadError(f"Недопустимый источник: {parsed.hostname}")
    if "/search" in path:
        raise DownloadError("Запросы к поисковым страницам запрещены")


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write(path, payload.encode("utf-8"))


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def valid_zip(path: Path, full_crc: bool = False) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if "data.json" not in names:
                return False
            if full_crc:
                return archive.testzip() is None
            with archive.open("data.json") as handle:
                return bool(handle.read(1))
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return False


def expected_download_size(response: Any, status: int, offset: int) -> int | None:
    headers = response.headers
    length = headers.get("Content-Length") if headers else None
    if length is not None:
        try:
            length_value = int(length)
        except ValueError as exc:
            raise DownloadError("Некорректный Content-Length") from exc
        if length_value < 0:
            raise DownloadError("Некорректный Content-Length")
    else:
        length_value = None
    if status == 200:
        return length_value
    content_range = headers.get("Content-Range", "") if headers else ""
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
    if not match:
        raise DownloadError("Некорректный или отсутствующий Content-Range")
    start, end, total = map(int, match.groups())
    if start != offset or end < start or total <= end:
        raise DownloadError("Content-Range не соответствует возобновлению")
    if length_value is not None and length_value != end - start + 1:
        raise DownloadError("Content-Length не соответствует Content-Range")
    return total


def resource_version(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    version = {}
    for key, header in (("etag", "ETag"), ("last_modified", "Last-Modified")):
        value = headers.get(header)
        if value:
            version[key] = value
    if version.get("etag", "").startswith("W/"):
        version.pop("etag", None)
    return version


def version_matches(version: dict[str, str], headers: Any) -> bool:
    if not version:
        return False
    current = resource_version(headers)
    return all(current.get(key) == value for key, value in version.items())


def http_error_has_captcha(error: HTTPError) -> bool:
    try:
        body = error.read(65536).lower()
    except (OSError, http.client.HTTPException):
        return False
    return b"captcha" in body or b"smartcaptcha" in body


def download_file(
    url: str,
    destination: Path,
    pause: float,
    timeout: float = 90,
    opener: Any = None,
    _restart: bool = False,
) -> tuple[str, int]:
    check_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    metadata_path = destination.with_name(destination.name + ".part.meta.json")
    saved = read_json(metadata_path, {})
    version = {
        key: value for key, value in saved.get("version", {}).items()
        if key in {"etag", "last_modified"}
    }
    has_partial_file = part.exists() and part.stat().st_size > 0
    offset = 0 if _restart else (part.stat().st_size if part.exists() else 0)
    if offset and not version:
        offset = 0
    headers = {"User-Agent": "judicial-open-data/1.0", "Accept": "application/zip,application/octet-stream,*/*"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if version.get("etag"):
            headers["If-Range"] = version["etag"]
        elif version.get("last_modified"):
            headers["If-Range"] = version["last_modified"]
    request = Request(url, headers=headers)
    if pause:
        time.sleep(pause)
    client = opener or build_opener(SafeRedirect())
    try:
        response = client.open(request, timeout=timeout)
    except HTTPError as exc:
        if http_error_has_captcha(exc):
            raise CaptchaError("Обнаружена капча в HTTP-ответе; загрузка остановлена") from exc
        if exc.code == 416 and has_partial_file and not _restart:
            content_range = exc.headers.get("Content-Range", "") if exc.headers else ""
            match = re.fullmatch(r"bytes \*/(\d+)", content_range)
            if match and int(match.group(1)) == part.stat().st_size and version_matches(version, exc.headers):
                return sha256_file(part), part.stat().st_size
            return download_file(url, destination, pause, timeout, opener, _restart=True)
        raise DownloadError(f"HTTP {exc.code}; загрузка остановлена") from exc
    except OSError as exc:
        raise DownloadError(f"Ошибка сети; загрузка остановлена: {exc}") from exc
    try:
        with response:
            status = getattr(response, "status", 200)
            if status not in (200, 206):
                raise DownloadError(f"HTTP {status}; загрузка остановлена")
            check_url(getattr(response, "url", url))
            if status == 206 and not version_matches(version, response.headers):
                response.close()
                if _restart:
                    raise DownloadError("Версия ресурса изменилась при повторной загрузке")
                return download_file(url, destination, pause, timeout, opener, _restart=True)
            expected_size = expected_download_size(response, status, offset)
            mode = "ab" if status == 206 and offset else "wb"
            response_version = resource_version(response.headers)
            if response_version:
                atomic_json(metadata_path, {"version": response_version})
            else:
                metadata_path.unlink(missing_ok=True)
            first = True
            with part.open(mode) as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    if first:
                        first = False
                        probe = chunk[:65536].lower()
                        if b"captcha" in probe or b"smartcaptcha" in probe:
                            raise CaptchaError("Обнаружена капча; загрузка остановлена")
                    handle.write(chunk)
        actual_size = part.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise DownloadError(f"Неполный документ: {actual_size} вместо {expected_size} байт")
    except DownloadError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise DownloadError(f"Ошибка чтения документа; загрузка остановлена: {exc}") from exc
    return sha256_file(part), part.stat().st_size


def download_dataset(spec: DatasetSpec, raw_dir: Path, pause: float = 4, opener: Any = None) -> dict[str, Any]:
    destination = raw_dir / spec.filename
    manifest_path = raw_dir / "datasets.json"
    manifests = read_json(manifest_path, {})
    previous = manifests.get(spec.key, {})
    if destination.exists() and valid_zip(destination):
        digest = sha256_file(destination)
        if not previous or previous.get("sha256") == digest:
            entry = {
                "key": spec.key,
                "dataset_page": spec.page_url,
                "api_url": spec.api_url,
                "dataset_modified": spec.modified,
                "filename": destination.name,
                "sha256": digest,
                "bytes": destination.stat().st_size,
                "verified_at": utc_now(),
            }
            manifests[spec.key] = entry
            atomic_json(manifest_path, manifests)
            return entry
    digest, size = download_file(spec.api_url, destination, pause, opener=opener)
    part = destination.with_name(destination.name + ".part")
    if not valid_zip(part, full_crc=True):
        raise DownloadError("Полученный файл не является корректным ZIP с data.json")
    os.replace(part, destination)
    entry = {
        "key": spec.key,
        "dataset_page": spec.page_url,
        "api_url": spec.api_url,
        "dataset_modified": spec.modified,
        "filename": destination.name,
        "sha256": digest,
        "bytes": size,
        "verified_at": utc_now(),
    }
    manifests[spec.key] = entry
    atomic_json(manifest_path, manifests)
    return entry


def iter_json_array(stream: Any, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    json_decoder = json.JSONDecoder()
    buffer = ""
    started = False
    closed = False
    while True:
        chunk = stream.read(chunk_size)
        buffer += decoder.decode(chunk, final=not chunk)
        if closed:
            if buffer.strip():
                raise ValueError("Данные после закрытия JSON-массива")
            if not chunk:
                return
            buffer = ""
            continue
        if not started:
            buffer = buffer.lstrip()
            if not buffer and chunk:
                continue
            if not buffer.startswith("["):
                raise ValueError("Ожидался корневой JSON-массив")
            buffer = buffer[1:]
            started = True
        position = 0
        while True:
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                buffer = buffer[position + 1:]
                closed = True
                break
            if position >= len(buffer):
                break
            try:
                item, end = json_decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                break
            if not isinstance(item, dict):
                raise ValueError("Элемент массива не является объектом")
            yield item
            position = end
        if closed:
            if buffer.strip():
                raise ValueError("Данные после закрытия JSON-массива")
            buffer = ""
        else:
            buffer = buffer[position:]
        if not chunk:
            if closed:
                return
            raise ValueError("Неполный JSON-массив")


def iter_zip_records(path: Path) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive, archive.open("data.json") as stream:
        yield from iter_json_array(stream)


def parse_date(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            pass
    return None


def date_year(value: Any) -> int | None:
    parsed = parse_date(value)
    return parsed.year if parsed else None


def number_year(record: dict[str, Any]) -> int | None:
    value = record.get("year")
    if isinstance(value, int) and 1900 <= value <= 2100:
        return value
    match = re.search(r"/((?:19|20)\d{2})(?:\D|$)", str(record.get("number", "")))
    return int(match.group(1)) if match else None


def act_date(record: dict[str, Any]) -> str | None:
    for key in ("dateFinal", "lastHearingDate"):
        if record.get(key):
            return record[key]
    dates = [item.get("date") for item in record.get("attachments", []) if item.get("date")]
    return max(dates) if dates else None


def record_years(record: dict[str, Any]) -> dict[str, Any]:
    act_years = {date_year(act_date(record))}
    act_years.update(date_year(item.get("date")) for item in record.get("attachments", []))
    return {
        "number": number_year(record),
        "registration": date_year(record.get("dateReg")),
        "act": sorted({year for year in act_years if year}),
    }


def in_target_period(record: dict[str, Any]) -> bool:
    years = record_years(record)
    values = {years["number"], years["registration"], *years["act"]}
    return any(year in TARGET_YEARS for year in values if year)


def article_290(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.search(r"(?iu)(?<!\d)ст\s*\.?\s*290(?!\d|\.\d)", value))


def has_article_290(record: dict[str, Any]) -> bool:
    for participant in record.get("participants") or []:
        for article in participant.get("codexArticles") or []:
            if article_290(article.get("name")):
                return True
    return False


def normalize_number(value: Any, year: Any = None) -> set[str]:
    text = str(value or "").upper().replace("№", " ")
    text = re.sub(r"\s+", "", text)
    years = str(year) if year else ""
    variants = {text} if text else set()
    for match in re.findall(r"\d+(?:/\d+)?-\d+(?:/\d{4})?", text):
        variants.add(match)
        if years and not re.search(r"/\d{4}$", match):
            variants.add(match + "/" + years)
    if years and text and not re.search(r"/\d{4}$", text):
        variants.add(text + "/" + years)
    for prefix, sequence, case_year in re.findall(r"(\d+)-(\d+)/(\d{2,4})(?:\D|$)", text):
        if len(case_year) == 2 and years:
            case_year = years[:2] + case_year
        if len(case_year) == 4:
            variants.add(f"{int(prefix)}-{int(sequence)}/{case_year}")
            variants.add(f"{int(prefix)}-{int(sequence)}/{case_year[2:]}")
    return variants


def normalize_court(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()
    return re.sub(r"[^0-9a-zа-я ]", "", text)


def court_matches(lower: dict[str, Any], first: dict[str, Any]) -> bool:
    lower_code = str(lower.get("courtCode") or "")
    first_code = str(first.get("courtCode") or "")
    if lower_code and first_code and lower_code == first_code:
        return True
    left = normalize_court(lower.get("courtName"))
    right = normalize_court(first.get("courtName"))
    return bool(left and right and (left == right or left in right or right in left))


def appellate_attachments(record: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in record.get("attachments") or []:
        name = normalize_court(item.get("displayName"))
        if item.get("link") and "апелляцион" in name and not re.search(r"жалоб|протокол|уведомлен", name):
            result.append(item)
    return result


def sentence_attachments(record: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in record.get("attachments") or []:
        name = normalize_court(item.get("displayName"))
        if item.get("link") and "приговор" in name and not re.search(r"исполнен|обжалован|вступлен", name):
            result.append(item)
    return result


def sentence_number(record: dict[str, Any]) -> bool:
    for lower in record.get("lower") or []:
        number = re.sub(r"\s+", "", str(lower.get("number") or ""))
        if number.startswith(("1-", "01-")):
            return True
    number = re.sub(r"\s+", "", str(record.get("number") or ""))
    return number.startswith(("1-", "01-"))


def confirmed_aliases(record: dict[str, Any]) -> list[dict[str, Any]]:
    aliases = []
    for item in record.get("upper") or []:
        if not isinstance(item, dict):
            continue
        item_date = parse_date(item.get("dateFinal"))
        record_date = parse_date(record.get("dateFinal"))
        item_type = normalize_court(item.get("documentType"))
        record_type = normalize_court(record.get("documentType"))
        same_date = bool(item_date and record_date and item_date == record_date)
        same_type = bool(item_type and record_type and item_type == record_type)
        if same_date and same_type and item.get("number"):
            aliases.append({"number": item.get("number"), "year": item.get("year"), "date": act_date(item), "documentType": item.get("documentType")})
    return aliases


def first_index(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        uid = str(record.get("uid") or "")
        if not uid:
            continue
        for variant in normalize_number(record.get("number"), record.get("year")):
            index.setdefault((uid, variant), []).append(record)
    return index


def link_first_instance(appeal: dict[str, Any], index: dict[tuple[str, str], list[dict[str, Any]]]) -> tuple[str, dict[str, Any] | None]:
    uid = str(appeal.get("uid") or "")
    if not uid:
        return "not_found", None
    matches: dict[str, dict[str, Any]] = {}
    for lower in appeal.get("lower") or []:
        for variant in normalize_number(lower.get("number")):
            for first in index.get((uid, variant), []):
                if court_matches(lower, first) and sentence_number(first):
                    matches[str(first.get("_id") or id(first))] = first
    if len(matches) != 1:
        return ("ambiguous", None) if matches else ("not_found", None)
    return "found", next(iter(matches.values()))


def slim_first(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "_id": record.get("_id"),
        "uid": record.get("uid"),
        "number": record.get("number"),
        "year": record.get("year"),
        "courtCode": record.get("courtCode"),
        "courtName": record.get("courtName"),
        "dateReg": record.get("dateReg"),
        "dateFinal": record.get("dateFinal"),
        "lastHearingDate": record.get("lastHearingDate"),
        "documentType": record.get("documentType"),
        "participants": record.get("participants") or [],
        "attachments": record.get("attachments") or [],
    }


def year_label(value: Any) -> str:
    return str(value) if value in TARGET_YEARS else "другой/нет"


def empty_status() -> dict[str, int]:
    return {
        "records": 0,
        "sentence_candidates": 0,
        "appeal_link": 0,
        "first_found": 0,
        "both_links": 0,
        "both_files_downloaded": 0,
        "both_texts_extracted": 0,
        "confirmed_pairs": 0,
    }


def document_artifact_errors(raw_dir: Path, document: dict[str, Any], require_text: bool = True) -> list[str]:
    errors = []
    binary = raw_dir / "texts" / document.get("path", "")
    if not binary.exists():
        return ["missing_binary"]
    if document.get("sha256") != sha256_file(binary):
        return ["binary_sha256_mismatch"]
    if document.get("bytes") is not None and document.get("bytes") != binary.stat().st_size:
        errors.append("binary_size_mismatch")
    if not require_text:
        return errors
    text_path = raw_dir / "texts" / document.get("text_path", binary.with_suffix(".txt").name)
    if document.get("text_source_sha256") != document.get("sha256"):
        errors.append("text_source_mismatch")
    elif not text_path.exists():
        errors.append("missing_text")
    elif document.get("text_sha256") != sha256_file(text_path):
        errors.append("text_sha256_mismatch")
    return errors


def refresh_artifact_flags(raw_dir: Path, state: dict[str, Any]) -> list[str]:
    documents = state.get("documents", {})
    file_errors = []
    text_errors = []
    for kind in ("sentence", "appeal"):
        document = documents.get(kind)
        if not document:
            file_errors.append(f"{kind}_missing_binary")
            text_errors.append(f"{kind}_missing_text")
            continue
        file_errors.extend(f"{kind}_{error}" for error in document_artifact_errors(raw_dir, document, require_text=False))
        text_errors.extend(f"{kind}_{error}" for error in document_artifact_errors(raw_dir, document, require_text=True))
    state["both_files_downloaded"] = not file_errors
    state["both_texts_extracted"] = not text_errors
    state.pop("both_downloaded", None)
    return text_errors


def revalidate_progress(raw_dir: Path) -> dict[str, Any]:
    progress_path = raw_dir / "pair-progress.json"
    progress = read_json(progress_path, {})
    candidates = {str(item["appeal_id"]): item for item in read_candidates(raw_dir / "work" / "appeal-candidates.jsonl")}
    changed = False
    for appeal_id, state in progress.items():
        previous = (state.get("status"), list(state.get("errors", [])), state.get("both_files_downloaded"), state.get("both_texts_extracted"))
        artifact_errors = refresh_artifact_flags(raw_dir, state)
        if artifact_errors:
            can_invalidate = state.get("documents") or state.get("status") not in {"verification_failed", "unsupported_format", "manual_review"}
            if state.get("status") in {"no_compatible_attachment", "unsupported_format"}:
                binary_errors = [
                    error
                    for document in state.get("documents", {}).values()
                    for error in document_artifact_errors(raw_dir, document, require_text=False)
                ]
                can_invalidate = bool(binary_errors)
            if can_invalidate and state.get("status") not in {"download_error", "captcha"}:
                state["status"] = "artifact_invalid"
                state["errors"] = artifact_errors
        elif state["both_texts_extracted"] and state.get("status") not in {"download_error", "captcha"}:
            candidate = candidates.get(appeal_id)
            if candidate:
                sentence_path = raw_dir / "texts" / state["documents"]["sentence"]["path"]
                appeal_path = raw_dir / "texts" / state["documents"]["appeal"]["path"]
                sentence_text = sentence_path.with_suffix(".txt").read_text(encoding="utf-8")
                appeal_text = appeal_path.with_suffix(".txt").read_text(encoding="utf-8")
                errors = verify_pair(candidate, sentence_text, appeal_text)
                state["status"] = "manual_review" if errors == ["manual_review"] else ("verification_failed" if errors else "confirmed_pair")
                state["errors"] = errors
                state["outcome"] = extract_outcome(appeal_text)
        current = (state.get("status"), list(state.get("errors", [])), state.get("both_files_downloaded"), state.get("both_texts_extracted"))
        if current != previous:
            state["verified_at"] = utc_now()
            changed = True
    if changed:
        atomic_json(progress_path, progress)
    return progress


def analyze(raw_dir: Path, report_path: Path) -> dict[str, Any]:
    first_path = raw_dir / FIRST_INSTANCE.filename
    appeals_path = raw_dir / APPEALS.filename
    if not first_path.exists() or not appeals_path.exists():
        raise FileNotFoundError("Сначала загрузите оба официальных набора")
    previous_analysis = read_json(raw_dir / "work" / "analysis.json", None)
    first_total = first_article = 0
    first_records = []
    for record in iter_zip_records(first_path):
        first_total += 1
        if has_article_290(record):
            first_article += 1
            first_records.append(record)
    index = first_index(first_records)
    progress = revalidate_progress(raw_dir)
    counts = empty_status()
    counts.update({"first_dataset_records": first_total, "first_article_290": first_article, "link_ambiguous": 0})
    by_year = {
        dimension: {label: empty_status() for label in (*map(str, TARGET_YEARS), "другой/нет")}
        for dimension in ("number", "registration", "act")
    }
    candidates_path = raw_dir / "work" / "appeal-candidates.jsonl"
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=candidates_path.name + ".", dir=candidates_path.parent)
    total = target = 0
    both_link_ids = []
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        for record in iter_zip_records(appeals_path):
            total += 1
            if not in_target_period(record) or not has_article_290(record):
                continue
            target += 1
            counts["records"] += 1
            years = record_years(record)
            dimensions = [year_label(years["number"]), year_label(years["registration"])]
            dimensions.append(year_label(years["act"][0]) if len(years["act"]) == 1 else "другой/нет")
            for dimension, label in zip(("number", "registration", "act"), dimensions):
                by_year[dimension][label]["records"] += 1
            is_candidate = sentence_number(record)
            attachments = appellate_attachments(record)
            link_status, first = link_first_instance(record, index) if is_candidate else ("not_applicable", None)
            first_links = sentence_attachments(first) if first else []
            pair = progress.get(str(record.get("_id")), {})
            both_files = bool(pair.get("both_files_downloaded"))
            both_texts = bool(pair.get("both_texts_extracted"))
            confirmed = pair.get("status") == "confirmed_pair"
            candidate = {
                "appeal_id": record.get("_id"),
                "identity": {
                    "uid": record.get("uid"), "number": record.get("number"), "year": record.get("year"),
                    "courtCode": record.get("courtCode"), "courtName": record.get("courtName"),
                    "dateReg": record.get("dateReg"), "act_date": act_date(record),
                    "documentType": record.get("documentType"), "years": years,
                },
                "aliases": confirmed_aliases(record),
                "lower": record.get("lower") or [],
                "participants": record.get("participants") or [],
                "appeal_attachments": attachments,
                "first_status": link_status,
                "first": slim_first(first) if first else None,
                "first_attachments": first_links,
                "years": dimensions,
            }
            output.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            if not is_candidate:
                continue
            counts["sentence_candidates"] += 1
            if attachments:
                counts["appeal_link"] += 1
            if link_status == "found":
                counts["first_found"] += 1
            elif link_status == "ambiguous":
                counts["link_ambiguous"] += 1
            if attachments and first_links:
                counts["both_links"] += 1
                both_link_ids.append(str(record.get("_id")))
            if both_files:
                counts["both_files_downloaded"] += 1
            if both_texts:
                counts["both_texts_extracted"] += 1
            if confirmed:
                counts["confirmed_pairs"] += 1
            for dimension, label in zip(("number", "registration", "act"), dimensions):
                entry = by_year[dimension][label]
                entry["sentence_candidates"] += 1
                entry["appeal_link"] += bool(attachments)
                entry["first_found"] += link_status == "found"
                entry["both_links"] += bool(attachments and first_links)
                entry["both_files_downloaded"] += both_files
                entry["both_texts_extracted"] += both_texts
                entry["confirmed_pairs"] += confirmed
    os.replace(temp_name, candidates_path)
    terminal_statuses = {"confirmed_pair", "verification_failed", "unsupported_format", "download_error", "captcha", "no_compatible_attachment", "manual_review"}
    resume = {
        "not_started": sum(appeal_id not in progress for appeal_id in both_link_ids),
        "remaining": sum(
            appeal_id not in progress or progress[appeal_id].get("status") not in terminal_statuses
            for appeal_id in both_link_ids
        ),
    }
    outcomes_confirmed: dict[str, int] = {}
    outcomes_other: dict[str, int] = {}
    for state in progress.values():
        outcome = state.get("outcome", "unknown")
        outcome_target = outcomes_confirmed if state.get("status") == "confirmed_pair" else outcomes_other
        outcome_target[outcome] = outcome_target.get(outcome, 0) + 1
    result = {
        "generated_at": utc_now(),
        "target_years": list(TARGET_YEARS),
        "appeals_dataset_records": total,
        "target_article_records": target,
        "counts": counts,
        "by_year": by_year,
        "pair_statuses": status_summary(progress),
        "resume": resume,
        "diagnostics": {
            "sentence_bad": sum("sentence_number_mismatch" in state.get("errors", []) for state in progress.values()),
            "sentence_number_conflict": sum("sentence_number_conflict" in state.get("errors", []) for state in progress.values()),
            "manual_review": sum(state.get("status") == "manual_review" for state in progress.values()),
        },
        "outcomes_confirmed": outcomes_confirmed,
        "outcomes_other": outcomes_other,
        "sources": [source_dict(APPEALS), source_dict(FIRST_INSTANCE)],
    }
    if previous_analysis and previous_analysis.get("counts", {}).get("confirmed_pairs") != counts["confirmed_pairs"]:
        result["review_history"] = {
            "previous_confirmed_pairs": previous_analysis.get("counts", {}).get("confirmed_pairs", 0),
            "previous_no_compatible": previous_analysis.get("pair_statuses", {}).get("no_compatible_attachment", 0),
        }
    atomic_json(raw_dir / "work" / "analysis.json", result)
    write_report(result, report_path)
    return result


def source_dict(spec: DatasetSpec) -> dict[str, Any]:
    return {"key": spec.key, "page": spec.page_url, "api": spec.api_url, "modified": spec.modified}


def status_summary(progress: dict[str, Any]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in progress.values():
        status = item.get("status", "unknown")
        summary[status] = summary.get(status, 0) + 1
    return summary


def write_report(analysis: dict[str, Any], path: Path) -> None:
    counts = analysis["counts"]
    lines = [
        "# Московский корпус по ст. 290 УК РФ: текущие агрегаты",
        "",
        f"Сформировано: {analysis['generated_at']}. Период отбора: 2022–2025.",
        "Отчёт содержит только агрегаты; номера, УИД, участники, URL карточек/актов и их хеши остаются в raw.",
        "Год номера, поступления и акта показан отдельно; строки разных таблиц не суммируются.",
        "",
        "## Источники",
        "",
    ]
    for source in analysis["sources"]:
        lines.append(f"- {source['key']}: {source['page']}; API {source['api']}; дата набора {source['modified']}.")
    lines += [
        "- Условия портала: https://data.mos-gorsud.ru/terms-of-use — копирование/адаптация с указанием URL и даты обновления.",
        "",
        "## Общая воронка",
        "",
        "Записи — после фильтра периода и структурированной ст. 290. Кандидат на приговор ещё не является подтверждённой парой.",
        "",
        "| Статус | Количество |",
        "|---|---:|",
        f"| Записи после фильтра | {counts['records']} |",
        f"| Кандидаты на приговор | {counts['sentence_candidates']} |",
        f"| Есть ссылка на апелляционный акт | {counts['appeal_link']} |",
        f"| Найдена первая инстанция | {counts['first_found']} |",
        f"| Есть обе ссылки | {counts['both_links']} |",
        f"| Скачаны оба исходных файла | {counts['both_files_downloaded']} |",
        f"| Извлечены оба текста | {counts['both_texts_extracted']} |",
        f"| Подтверждённые полные пары | {counts['confirmed_pairs']} |",
        f"| Неоднозначные связки | {counts['link_ambiguous']} |",
        "",
        f"Всего записей в апелляционном наборе: {analysis['appeals_dataset_records']}. В наборе первой инстанции: {counts['first_dataset_records']}; со ст. 290 без годового фильтра: {counts['first_article_290']}.",
        f"Из кандидатов с обеими ссылками ещё не начато: {analysis['resume']['not_started']}. Осталось для resume с учётом незавершённых и повреждённых: {analysis['resume']['remaining']}.",
        "",
    ]
    titles = {"number": "Год номера апелляционного дела", "registration": "Год поступления апелляции", "act": "Год апелляционного акта"}
    for dimension, title in titles.items():
        lines += [
            f"## {title}", "",
            "| Год | Записи | Кандидаты | Апелляционная ссылка | Найдена первая | Обе ссылки | Оба файла | Оба текста | Полные пары |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, entry in analysis["by_year"][dimension].items():
            lines.append(
                f"| {label} | {entry['records']} | {entry['sentence_candidates']} | {entry['appeal_link']} | "
                f"{entry['first_found']} | {entry['both_links']} | {entry['both_files_downloaded']} | {entry['both_texts_extracted']} | {entry['confirmed_pairs']} |"
            )
        lines.append("")
    if analysis.get("review_history"):
        history = analysis["review_history"]
        lines += [
            "## Изменение относительно сохранённого отчёта",
            "",
            f"Предыдущий сохранённый анализ содержал {history['previous_confirmed_pairs']} подтверждённых пар; сейчас — {counts['confirmed_pairs']}. "
            f"Текущих `no_compatible_attachment`: {analysis['pair_statuses'].get('no_compatible_attachment', 0)}.",
            "",
        ]
    for title, key in (("Исход подтверждённых пар", "outcomes_confirmed"), ("Исход других проверенных/неподтверждённых записей", "outcomes_other")):
        lines += [f"## {title}", "", "| Outcome | Количество |", "|---|---:|"]
        for outcome in ("unchanged", "changed", "cancelled", "unknown"):
            lines.append(f"| {outcome} | {analysis[key].get(outcome, 0)} |")
        lines.append("")
    lines += ["Outcome извлекается только при наличии маркера резолютивной части и не используется как доказательство идентичности.", ""]
    lines += ["## Диагностика номера первой инстанции", "", "| Категория | Количество |", "|---|---:|"]
    for key in ("sentence_bad", "sentence_number_conflict", "manual_review"):
        lines.append(f"| {key} | {analysis['diagnostics'].get(key, 0)} |")
    lines += ["", "`sentence_bad` считается только по самому приговору; отсутствие его номера в апелляции само по себе не отклоняется.", ""]
    lines += ["## Статусы скачивания и проверки текстов", "", "| Статус | Количество |", "|---|---:|"]
    if analysis["pair_statuses"]:
        for status, count in sorted(analysis["pair_statuses"].items()):
            lines.append(f"| {status} | {count} |")
    else:
        lines.append("| Скачивание ещё не запускалось | 0 |")
    lines += [
        "",
        "Полная пара засчитывается только после фактического получения двух файлов, извлечения текста и проверки типа и идентичности. Ссылка сама по себе полноту не подтверждает.",
        "`artifact_invalid` возвращается в resume; `no_compatible_attachment` и `unsupported_format` терминальны и не считаются неначатыми. Сетевые ошибки повторяются только с `--retry-errors`.",
        "Повторный запуск не повторяет валидные ZIP и уже скачанные документы; текущий прогресс может быть неполным.",
        "",
    ]
    atomic_write(path, "\n".join(lines).encode("utf-8"))


class TextHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def extract_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"%PDF"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise UnsupportedFormat("PDF: не установлен pypdf") from exc
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        executable = shutil.which("antiword")
        if not executable:
            raise UnsupportedFormat("DOC/OLE: antiword недоступен")
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / "document.doc"
            temporary.write_bytes(data)
            result = subprocess.run(
                [executable, "-m", "UTF-8.txt", str(temporary)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
        if result.returncode:
            raise UnsupportedFormat("DOC/OLE: antiword не смог прочитать файл")
        return result.stdout.decode("utf-8", "replace")
    if data.startswith(b"PK"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                xml = archive.read("word/document.xml")
            root = ElementTree.fromstring(xml)
            return " ".join(root.itertext())
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
            raise UnsupportedFormat("ZIP/DOCX: неподдерживаемая структура") from exc
    probe = data[:4096].lstrip().lower()
    if probe.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"<html" in probe:
        parser = TextHTMLParser()
        parser.feed(data.decode("utf-8", "replace"))
        return re.sub(r"\s+", " ", " ".join(parser.parts))
    if b"\x00" not in probe:
        for encoding in ("utf-8", "cp1251"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                pass
    raise UnsupportedFormat("Неизвестный или неподдерживаемый формат")


def compact_text(value: str) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", value.lower().replace("ё", "е"))


def canonical_number(prefix: str, sequence: str, year: str, default_year: Any = None) -> tuple[tuple[int, ...], int, int]:
    numeric_year = int(year)
    if len(year) == 2:
        century = int(default_year) // 100 * 100 if default_year else 2000
        numeric_year += century
    return tuple(int(part) for part in prefix.split("/")), int(sequence), numeric_year


def number_identity_variants(value: Any, year: Any = None) -> set[tuple[tuple[int, ...], int, int]]:
    text = re.sub(r"\s+", "", str(value or "").replace("\\", "/"))
    return {
        canonical_number(match.group(1), match.group(2), match.group(3), year)
        for match in re.finditer(r"(?<![\d/])(\d+(?:/\d+)?)-(\d+)/(\d{2,4})(?![\d/])", text)
    }


def extract_case_numbers(text: str, default_year: Any = None) -> set[tuple[tuple[int, ...], int, int]]:
    return number_identity_variants(text, default_year)


def document_number_match(text: str, value: Any, year: Any = None) -> bool:
    expected = number_identity_variants(value, year)
    return bool(expected and expected & extract_case_numbers(text, year))


def appealed_number_references(appeal_text: str, default_year: Any = None) -> set[tuple[tuple[int, ...], int, int]]:
    normalized = re.sub(r"\s+", " ", appeal_text.lower())
    number = r"\d+(?:/\d+)?\s*-\s*\d+\s*/\s*\d{2,4}"
    patterns = [
        (normalized, rf"(?:на\s+приговор|обжалуем[а-я]+\s+приговор).{{0,300}}?(?:дел[аоу]|№)\s*({number})"),
    ]
    operative = operative_text(appeal_text)
    if operative is not None:
        patterns.append((operative, rf"(?<![а-я])приговор(?![а-я]).{{0,300}}?(?:дел[аоу]|№)\s*({number})"))
    return {
        canonical
        for text, pattern in patterns
        for match in re.finditer(pattern, text, re.S)
        for canonical in number_identity_variants(match.group(1), default_year)
    }


def date_forms(value: Any) -> set[str]:
    parsed = parse_date(value)
    if not parsed:
        return set()
    return {
        parsed.strftime("%d.%m.%Y"),
        f"{parsed.day}.{parsed.month:02d}.{parsed.year}",
        f"{parsed.day} {MONTHS[parsed.month - 1]} {parsed.year}",
    }


def court_stems(value: Any) -> list[str]:
    return [
        token[:5] if len(token) > 5 else token
        for token in normalize_court(value).split()
        if token not in {"города", "москвы", "город", "области", "суд"}
        and not token.isdigit()
    ]


def document_court_match(text: str, court_name: Any) -> bool:
    compact = compact_text(text)
    stems = court_stems(court_name)
    return bool(stems) and all(stem in compact for stem in stems)


def operative_text(appeal_text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", appeal_text.lower())
    marker_pattern = (
        r"п\s*о\s*с\s*т\s*а\s*н\s*о\s*в\s*и\s*л|"
        r"о\s*п\s*р\s*е\s*д\s*е\s*л\s*и\s*л"
    )
    markers = list(re.finditer(marker_pattern, normalized))
    return normalized[markers[-1].start():] if markers else None


def new_trial_disposition(appeal_text: str) -> bool:
    operative = operative_text(appeal_text) or ""
    hearing = r"на\s+нов\w*\s+(?:судебн\w*\s+)?(?:рассмотрен\w*|разбирательств\w*)"
    operative = re.sub(r"без\s+направления\s+[^.]{0,100}" + hearing, "", operative)
    return bool(re.search(r"направ\w*\s+[^.]{0,100}" + hearing + "|" + hearing, operative))


def ocr_word(value: str) -> str:
    return r"\s*".join(re.escape(character) for character in value)


def extract_outcome(appeal_text: str) -> str:
    operative = operative_text(appeal_text)
    if operative is None:
        return "unknown"
    outcomes = set()
    sentence = ocr_word("приговор")
    unchanged = ocr_word("оставить") + r"\s+" + ocr_word("без") + r"\s+" + ocr_word("изменения")
    pattern = rf"(?<![а-я]){sentence}(?![а-я])(.{{0,350}}?)({unchanged}|{ocr_word('изменить')}|{ocr_word('отменить')})"
    for match in re.finditer(pattern, operative, re.S):
        body = match.group(1)
        if re.search(r"постановлен|определен|арест|мер[аы]\s*пресечения|восстанов|срок\w*\s+обжалован|обжалования\s+срока", body):
            continue
        outcomes.add(match.group(2).replace(" ", ""))
    if "отменить" in outcomes:
        return "cancelled"
    if "изменить" in outcomes:
        return "changed"
    if outcomes == {"оставитьбезизменения"} and re.search(r"возвращени\w*.{0,100}без\s+рассмотрен\w*|без\s+рассмотрен\w*.{0,100}возвращени\w*", operative, re.S):
        return "unknown"
    if "оставитьбезизменения" in outcomes:
        return "unchanged"
    return "unknown"


def has_sentence_operative(appeal_text: str) -> bool:
    return extract_outcome(appeal_text) != "unknown"


def verify_pair(candidate: dict[str, Any], sentence_text: str, appeal_text: str) -> list[str]:
    errors = []
    if len(sentence_text) < 1000 or len(appeal_text) < 1000:
        return ["text_too_short"]
    sentence_compact = compact_text(sentence_text)
    appeal_compact = compact_text(appeal_text)
    if "приговор" not in sentence_compact or "приговорил" not in sentence_compact:
        errors.append("sentence_type_not_confirmed")
    if "установил" not in sentence_compact:
        errors.append("sentence_full_text_not_confirmed")
    intro_compact = compact_text(appeal_text[:6000])
    procedural_intro = re.search(r"напостановление", intro_compact)
    if "апелляцион" not in appeal_compact or not has_sentence_operative(appeal_text) or ("наприговор" not in appeal_compact and procedural_intro):
        errors.append("appeal_type_not_confirmed")
    if new_trial_disposition(appeal_text):
        errors.append("new_trial")
    first = candidate["first"]
    expected_lower_numbers = number_identity_variants(first.get("number"), first.get("year"))
    if not expected_lower_numbers or not expected_lower_numbers & extract_case_numbers(sentence_text, first.get("year")):
        errors.append("sentence_number_mismatch")
    lower_references = appealed_number_references(appeal_text, first.get("year")) & expected_lower_numbers
    all_lower_references = appealed_number_references(appeal_text, first.get("year"))
    if all_lower_references and not lower_references:
        if len(all_lower_references) == 1:
            errors.append("sentence_number_conflict")
        else:
            errors.append("manual_review")
    identity = candidate["identity"]
    appeal_number_ok = document_number_match(appeal_text, identity.get("number"), identity.get("year"))
    if not appeal_number_ok:
        identity_date = parse_date(identity.get("act_date"))
        identity_type = normalize_court(identity.get("documentType"))
        for alias in candidate.get("aliases") or []:
            alias_date = parse_date(alias.get("date"))
            alias_type = normalize_court(alias.get("documentType"))
            if not identity_date or not alias_date or alias_date != identity_date:
                continue
            if not identity_type or not alias_type or alias_type != identity_type:
                continue
            if document_number_match(appeal_text, alias.get("number"), alias.get("year")):
                appeal_number_ok = True
                break
    if not appeal_number_ok:
        errors.append("appeal_number_mismatch")
    lower_court = (candidate.get("lower") or [{}])[0].get("courtName")
    if not document_court_match(sentence_text, lower_court) or not document_court_match(appeal_text, lower_court):
        errors.append("lower_court_mismatch")
    appeal_court = identity.get("courtName") or "Московский городской суд"
    if not document_court_match(appeal_text, appeal_court):
        errors.append("appeal_court_mismatch")
    date_options = {first.get("dateFinal"), first.get("lastHearingDate")}
    valid_sentence_dates = {value for value in date_options if parse_date(value)}
    if not valid_sentence_dates:
        errors.append("sentence_date_missing")
    else:
        shared_date = any(
            any(compact_text(form) in sentence_compact for form in date_forms(value))
            and any(compact_text(form) in appeal_compact for form in date_forms(value))
            for value in valid_sentence_dates
        )
        if not shared_date:
            errors.append("sentence_date_mismatch")
    if not parse_date(identity.get("act_date")):
        errors.append("appeal_date_missing")
    else:
        appeal_date_forms = date_forms(identity.get("act_date"))
        if not any(compact_text(form) in appeal_compact for form in appeal_date_forms):
            errors.append("appeal_date_mismatch")
    return errors


def read_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def attachment_combinations(candidate: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    combinations = []
    for sentence in candidate.get("first_attachments") or []:
        for appeal in candidate.get("appeal_attachments") or []:
            sentence_date = parse_date(sentence.get("date") or (candidate.get("first") or {}).get("dateFinal"))
            appeal_date = parse_date(appeal.get("date") or candidate["identity"].get("act_date"))
            if sentence_date and appeal_date and sentence_date > appeal_date:
                continue
            combinations.append((sentence, appeal))
    return combinations


def ensure_document(
    raw_dir: Path,
    appeal_id: str,
    kind: str,
    attachment: dict[str, Any],
    current: dict[str, Any],
    progress_path: Path,
    progress: dict[str, Any],
    pause: float,
) -> tuple[dict[str, Any], str]:
    url = attachment.get("link") or attachment.get("url")
    if not url:
        raise DownloadError("У документа отсутствует URL")
    check_url(url)
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    document = current["documents"].get(kind, {})
    binary = (
        raw_dir / "texts" / document["path"]
        if document.get("url") == url and document.get("path")
        else raw_dir / "texts" / f"{appeal_id}-{kind}-{key}.bin"
    )
    binary.parent.mkdir(parents=True, exist_ok=True)
    valid_existing = (
        document.get("url") == url and binary.exists()
        and document.get("sha256") == sha256_file(binary)
        and (document.get("bytes") is None or document.get("bytes") == binary.stat().st_size)
    )
    if not valid_existing:
        digest, size = download_file(url, binary, pause)
        part = binary.with_name(binary.name + ".part")
        os.replace(part, binary)
        document = {"url": url, "sha256": digest, "bytes": size, "path": binary.name}
        current["documents"][kind] = document
        atomic_json(progress_path, progress)
    text_path = raw_dir / "texts" / document.get("text_path", binary.with_suffix(".txt").name)
    text_valid = (
        document.get("text_source_sha256") == document.get("sha256")
        and text_path.exists()
        and document.get("text_sha256") == sha256_file(text_path)
    )
    if not text_valid:
        text = extract_text(binary)
        if not text.strip():
            raise UnsupportedFormat("Извлечён пустой текст")
        atomic_write(text_path, text.encode("utf-8"))
        document["text_path"] = text_path.name
        document["text_source_sha256"] = document["sha256"]
        document["text_sha256"] = sha256_file(text_path)
        current["documents"][kind] = document
        atomic_json(progress_path, progress)
    return document, text_path.read_text(encoding="utf-8")


def download_texts(raw_dir: Path, report_path: Path, pause: float = 4, limit: int = 5, retry_errors: bool = False) -> dict[str, Any]:
    candidates = read_candidates(raw_dir / "work" / "appeal-candidates.jsonl")
    if not candidates:
        raise FileNotFoundError("Сначала выполните analyze и оцените число кандидатов")
    progress_path = raw_dir / "pair-progress.json"
    progress = read_json(progress_path, {})
    attempted = 0
    consecutive_errors = 0
    stopped_reason = None
    terminal = {"confirmed_pair", "unsupported_format", "verification_failed", "no_compatible_attachment", "manual_review"}
    if not retry_errors:
        terminal.update({"download_error", "captcha"})
    for candidate in candidates:
        appeal_id = str(candidate["appeal_id"])
        current = progress.get(appeal_id, {})
        if current.get("status") in terminal:
            continue
        if not candidate["appeal_attachments"] or not candidate["first_attachments"]:
            continue
        if limit and attempted >= limit:
            break
        attempted += 1
        current["status"] = "processing"
        current.setdefault("documents", {})
        current.setdefault("attempts", [])
        progress[appeal_id] = current
        atomic_json(progress_path, progress)
        combinations = attachment_combinations(candidate)
        if not combinations:
            current["status"] = "no_compatible_attachment"
            current["errors"] = ["no_compatible_attachment_combination"]
            atomic_json(progress_path, progress)
            continue
        failed_errors = []
        unsupported_errors = []
        try:
            for sentence_attachment, appeal_attachment in combinations:
                try:
                    _, sentence_text = ensure_document(raw_dir, appeal_id, "sentence", sentence_attachment, current, progress_path, progress, pause)
                    _, appeal_text = ensure_document(raw_dir, appeal_id, "appeal", appeal_attachment, current, progress_path, progress, pause)
                    current["both_files_downloaded"] = True
                    current["both_texts_extracted"] = True
                    errors = verify_pair(candidate, sentence_text, appeal_text)
                    outcome = extract_outcome(appeal_text)
                    current["outcome"] = outcome
                    current["attempts"].append({
                        "sentence_url": sentence_attachment["link"], "appeal_url": appeal_attachment["link"],
                        "errors": errors, "outcome": outcome,
                    })
                    current["errors"] = errors
                    current["verified_at"] = utc_now()
                    if not errors:
                        current["status"] = "confirmed_pair"
                        atomic_json(progress_path, progress)
                        break
                    failed_errors = errors
                    atomic_json(progress_path, progress)
                except UnsupportedFormat as exc:
                    unsupported_errors.append(str(exc))
                    current["attempts"].append({
                        "sentence_url": sentence_attachment["link"], "appeal_url": appeal_attachment["link"],
                        "errors": [str(exc)], "outcome": "unknown",
                    })
                    refresh_artifact_flags(raw_dir, current)
                    atomic_json(progress_path, progress)
            else:
                if failed_errors:
                    current["status"] = "manual_review" if failed_errors == ["manual_review"] else "verification_failed"
                    current["errors"] = failed_errors
                else:
                    current["status"] = "unsupported_format"
                    current["errors"] = unsupported_errors or ["no_compatible_attachment_combination"]
                refresh_artifact_flags(raw_dir, current)
                atomic_json(progress_path, progress)
        except DownloadError as exc:
            current["status"] = "captcha" if isinstance(exc, CaptchaError) else "download_error"
            current["errors"] = [str(exc)]
            refresh_artifact_flags(raw_dir, current)
            atomic_json(progress_path, progress)
            if isinstance(exc, CaptchaError):
                raise
            if "HTTP 404" in str(exc):
                continue
            consecutive_errors += 1
            if consecutive_errors >= 3:
                stopped_reason = "network_error_series"
                break
        else:
            consecutive_errors = 0
    analysis_path = raw_dir / "work" / "analysis.json"
    analysis = read_json(analysis_path, None)
    if analysis:
        analyze(raw_dir, report_path)
    return {"attempted": attempted, "statuses": status_summary(progress), "stopped_reason": stopped_reason}


def main() -> None:
    parser = argparse.ArgumentParser(description="Официальные открытые данные Мосгорсуда без /search")
    parser.add_argument("command", choices=("download", "analyze", "download-texts", "report", "run"))
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="first")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--pause", type=float, default=4)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        result = download_dataset(DATASETS[args.dataset], args.raw_dir, args.pause)
        print(json.dumps({key: result[key] for key in ("key", "filename", "bytes", "sha256")}, ensure_ascii=False))
    elif args.command == "analyze":
        result = analyze(args.raw_dir, args.report)
        print(json.dumps(result["counts"], ensure_ascii=False))
    elif args.command == "download-texts":
        result = download_texts(args.raw_dir, args.report, args.pause, args.limit, args.retry_errors)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "report":
        analysis = read_json(args.raw_dir / "work" / "analysis.json", None)
        if not analysis:
            raise FileNotFoundError("Сначала выполните analyze")
        analysis = analyze(args.raw_dir, args.report)
        print(json.dumps(analysis["counts"], ensure_ascii=False))
    else:
        download_dataset(DATASETS["first"], args.raw_dir, args.pause)
        analysis = analyze(args.raw_dir, args.report)
        print(json.dumps(analysis["counts"], ensure_ascii=False))
        if args.limit:
            result = download_texts(args.raw_dir, args.report, args.pause, args.limit, args.retry_errors)
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
