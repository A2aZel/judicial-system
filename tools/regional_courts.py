from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "corpus" / "regional" / "raw"
REPORT = ROOT / "corpus" / "regional" / "report.md"
YEARS = (2022, 2023, 2024, 2025)
MONTHS = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря")


@dataclass(frozen=True)
class Court:
    key: str
    name: str
    host: str

    @property
    def base(self) -> str:
        return f"https://{self.host}"

    @property
    def robots_url(self) -> str:
        return self.base + "/robots.txt"

    @property
    def form_url(self) -> str:
        return self.base + "/modules.php?name=sud_delo&srv_num=1&name_op=sf&nc=1&delo_id=4&new=4"


KRK = Court("krk", "Красноярский краевой суд", "kraevoy.krk.sudrf.ru")
VRN = Court("vrn", "Воронежский областной суд", "oblsud.vrn.sudrf.ru")
COURTS = {court.key: court for court in (KRK, VRN)}


class AccessBlocked(RuntimeError):
    pass


class CaptchaError(RuntimeError):
    pass


class DownloadError(RuntimeError):
    pass


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


def read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_robots(text: str) -> list[str]:
    rules = []
    active = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith("user-agent:"):
            active = line.split(":", 1)[1].strip() == "*"
        elif active and line.lower().startswith("disallow:"):
            rules.append(line.split(":", 1)[1].strip())
    return rules


def allowed_by_robots(url: str, rules: Iterable[str]) -> bool:
    parsed = urlparse(url)
    target = (parsed.path + ("?" + parsed.query if parsed.query else "")).lower()
    for rule in rules:
        pattern = re.escape(rule.lower()).replace(r"\*", ".*")
        if re.match(pattern, target):
            return False
    return True


def required_paths_allowed(court: Court, rules: Iterable[str]) -> bool:
    paths = (
        court.form_url,
        court.base + "/modules.php?name=sud_delo&srv_num=1&name_op=r&delo_id=4&new=4",
        court.base + "/modules.php?name=sud_delo&srv_num=1&name_op=case&case_id=example",
        court.base + "/modules.php?name=sud_delo&srv_num=1&name_op=doc&number=example",
    )
    return all(allowed_by_robots(url, rules) for url in paths)


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise DownloadError("Небезопасный redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_error_captcha(error: HTTPError) -> bool:
    try:
        body = error.read(65536).lower()
    except OSError:
        return False
    return b"captcha" in body or b"smartcaptcha" in body


def safe_get(url: str, court: Court, rules: Iterable[str], pause: float = 4, opener: Any = None) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != court.host:
        raise DownloadError("Разрешены только страницы официального суда")
    if not allowed_by_robots(url, rules):
        raise AccessBlocked("Путь запрещён robots.txt")
    if pause:
        time.sleep(pause)
    request = Request(url, headers={"User-Agent": "regional-courts-research/1.0"})
    client = opener or build_opener(SafeRedirect())
    try:
        response = client.open(request, timeout=60)
    except HTTPError as exc:
        if http_error_captcha(exc):
            raise CaptchaError("CAPTCHA в HTTP-ответе") from exc
        raise DownloadError(f"HTTP {exc.code}") from exc
    except OSError as exc:
        raise DownloadError(f"Сетевая ошибка: {exc}") from exc
    try:
        with response:
            expected = response.headers.get("Content-Length")
            expected_size = int(expected) if expected is not None else None
            body = bytearray()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                if not body and (b"captcha" in chunk[:65536].lower() or b"smartcaptcha" in chunk[:65536].lower()):
                    raise CaptchaError("CAPTCHA на странице")
                body.extend(chunk)
        if expected_size is not None and len(body) != expected_size:
            raise DownloadError("Неполный ответ")
        return bytes(body)
    except DownloadError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise DownloadError(f"Ошибка чтения: {exc}") from exc


def check_robots(court: Court, raw_dir: Path = RAW, pause: float = 4, opener: Any = None) -> dict[str, Any]:
    body = safe_get(court.robots_url, court, (), pause, opener)
    text = body.decode("utf-8", "replace")
    rules = parse_robots(text)
    allowed = required_paths_allowed(court, rules)
    result = {
        "court": court.key,
        "allowed": allowed,
        "status": "allowed" if allowed else "robots_blocked",
        "checked_at": utc_now(),
        "rules": rules,
        "robots_sha256": hashlib.sha256(body).hexdigest(),
    }
    court_dir = raw_dir / court.key
    atomic_write(court_dir / "robots.txt", body)
    atomic_json(court_dir / "access.json", result)
    if not allowed:
        raise AccessBlocked("Картотека запрещена robots.txt")
    return result


class ListingParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.rows: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.link: dict[str, str] | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "tr":
            self.current = {"text": [], "case_url": None, "document_urls": []}
        elif tag == "a" and self.current is not None:
            href = attributes.get("href", "")
            self.link = {"href": urljoin(self.base_url, href), "text": []}

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"].append(data)
        if self.link is not None:
            self.link["text"].append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.link is not None and self.current is not None:
            if "name_op=case" in self.link["href"]:
                self.current["case_url"] = self.link["href"]
            elif "name_op=doc" in self.link["href"]:
                self.current["document_urls"].append(self.link["href"])
            self.link = None
        elif tag == "tr" and self.current is not None:
            text = re.sub(r"\s+", " ", " ".join(self.current["text"])).strip()
            if self.current["case_url"]:
                self.rows.append({**self.current, "text": text})
            self.current = None


def parse_listing(html: str, base_url: str) -> list[dict[str, Any]]:
    parser = ListingParser(base_url)
    parser.feed(html)
    return parser.rows


def total_found(html: str) -> int | None:
    match = re.search(r"Всего по запросу найдено\s*—\s*(\d+)", html)
    return int(match.group(1)) if match else None


def parse_date(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            pass
    return None


def date_forms(value: Any) -> set[str]:
    parsed = parse_date(value)
    if not parsed:
        return set()
    return {parsed.strftime("%d.%m.%Y"), f"{parsed.day} {MONTHS[parsed.month - 1]} {parsed.year}"}


def canonical_number(value: Any) -> tuple[str, int, int] | None:
    text = re.sub(r"\s+", "", str(value or "").upper().replace("К", "K"))
    match = re.fullmatch(r"(\d+(?:/\d+)?)-(\d+)/(\d{2,4})", text)
    if not match:
        return None
    prefix, sequence, year = match.groups()
    numeric_year = int(year)
    if len(year) == 2:
        numeric_year += 2000
    return prefix, int(sequence), numeric_year


def numbers_in_text(text: str) -> set[tuple[str, int, int]]:
    result = set()
    normalized = re.sub(r"\s*([/-])\s*", r"\1", text)
    for match in re.finditer(r"(?<![\d/])(\d+(?:/\d+)?)-(\d+)/(\d{2,4})(?![\d/])", normalized):
        canonical = canonical_number(match.group(0))
        if canonical:
            result.add(canonical)
    return result


def sentence_candidate(card_text: str) -> bool:
    text = re.sub(r"\s+", " ", card_text.lower())
    if any(marker in text for marker in ("меры пресечения", "содержания под стражей", "в порядке исполнения приговора", "ареста на имущество")):
        return False
    return "приговор" in text and any(
        marker in text for marker in ("обвинительный приговор", "вынесен приговор", "апелляционное определение", "приговор изменен", "приговор отменен", "приговор оставлен")
    )


def event_identity(court_key: str, case_url: str, card_text: str) -> dict[str, Any]:
    query = parse_qs(urlparse(case_url).query)
    uid = re.search(r"\d{2}[A-Z]{2}\d{4}-\d{2}-\d{4}-\d{6}-\d{2}", card_text)
    date = re.search(r"Дата рассмотрения\s*([0-9.]+)", card_text)
    return {
        "court": court_key,
        "production_id": query.get("case_id", [""])[0],
        "act_date": date.group(1) if date else None,
        "act_type": "Апелляционное определение" if "АПЕЛЛЯЦИОННОЕ ОПРЕДЕЛЕНИЕ" in card_text else "Апелляционное постановление",
        "uid": uid.group(0) if uid else None,
    }


def event_key(identity: dict[str, Any]) -> tuple[str, str, str, str]:
    return identity["court"], identity["production_id"], identity.get("act_date") or "", identity["act_type"]


def participant_tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-ZА-Я][a-zа-я]{2,}", text))


def link_first(appeal: dict[str, Any], first_records: Iterable[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    matches = []
    for first in first_records:
        if appeal.get("uid") != first.get("uid"):
            continue
        if appeal.get("lower_court") != first.get("court"):
            continue
        if canonical_number(appeal.get("lower_number")) != canonical_number(first.get("number")):
            continue
        matches.append(first)
    if len(matches) == 1:
        return "found", matches[0]
    return ("ambiguous", None) if matches else ("not_found", None)


def operative_text(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text.lower())
    pattern = r"п\s*о\s*с\s*т\s*а\s*н\s*о\s*в\s*и\s*л|о\s*п\s*р\s*е\s*д\s*е\s*л\s*и\s*л"
    markers = list(re.finditer(pattern, normalized))
    return normalized[markers[-1].start():] if markers else None


def extract_outcome(text: str) -> str:
    operative = operative_text(text)
    if not operative:
        return "unknown"
    outcomes = set()
    for match in re.finditer(r"(?<![а-я])приговор(?![а-я])(.{0,300}?)(оставить\s+без\s+изменения|изменить|отменить)", operative, re.S):
        if re.search(r"постановлен|определен|арест|мер[аы]\s*пресечения|восстанов|срок\w*\s+обжалован", match.group(1)):
            continue
        outcomes.add(match.group(2).replace(" ", ""))
    if "отменить" in outcomes:
        return "cancelled"
    if "изменить" in outcomes:
        return "changed"
    if "оставитьбезизменения" in outcomes:
        return "unchanged"
    return "unknown"


def verify_pair(metadata: dict[str, Any], sentence_text: str, appeal_text: str) -> list[str]:
    errors = []
    if len(sentence_text) < 1000 or len(appeal_text) < 1000:
        return ["text_too_short"]
    expected = canonical_number(metadata.get("lower_number"))
    if not expected or expected not in numbers_in_text(sentence_text):
        errors.append("sentence_bad")
    if "приговор" not in sentence_text.lower() or "приговорил" not in sentence_text.lower():
        errors.append("sentence_type")
    if "апелляцион" not in appeal_text.lower() or extract_outcome(appeal_text) == "unknown":
        errors.append("appeal_type")
    date = metadata.get("sentence_date")
    forms = date_forms(date)
    if not forms:
        errors.append("sentence_date_missing")
    elif not all(any(form in text for form in forms) for text in (sentence_text, appeal_text)):
        errors.append("sentence_date_mismatch")
    if metadata.get("act_date"):
        appeal_forms = date_forms(metadata["act_date"])
        if not any(form in appeal_text for form in appeal_forms):
            errors.append("appeal_date_mismatch")
    if metadata.get("participants"):
        appeal_tokens = participant_tokens(appeal_text)
        if not set(metadata["participants"]).issubset(appeal_tokens):
            errors.append("participants_mismatch")
    return errors


def status_for_errors(errors: list[str]) -> str:
    if not errors:
        return "confirmed_pair"
    if errors == ["manual_review"]:
        return "manual_review"
    return "verification_failed"


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.data = read_json(path, {"candidates": {}, "errors": 0})

    def save(self) -> None:
        atomic_json(self.path, self.data)

    def add(self, identity: dict[str, Any]) -> None:
        key = "|".join(event_key(identity))
        self.data["candidates"][key] = {"status": "pending", "identity": identity}
        self.save()

    def finish(self, identity: dict[str, Any], status: str, outcome: str = "unknown") -> None:
        key = "|".join(event_key(identity))
        self.data["candidates"][key].update({"status": status, "outcome": outcome})
        self.save()

    def pending(self) -> list[dict[str, Any]]:
        terminal = {"confirmed_pair", "verification_failed", "unsupported_format", "manual_review", "no_compatible_attachment", "download_error", "captcha"}
        return [item for item in self.data["candidates"].values() if item["status"] not in terminal]


def search_url(court: Court, year: int, page: int = 1) -> str:
    return (
        court.base + "/modules.php?name=sud_delo&srv_num=1&name_op=r&delo_id=4&case_type=0&new=4"
        "&delo_table=u2_case"
        f"&u2_case__ENTRY_DATE1D=01.01.{year}&u2_case__ENTRY_DATE2D=31.12.{year}"
        "&U2_DEFENDANT__LAW_ARTICLESS=290"
        + (f"&page={page}" if page > 1 else "")
    )


def collect_court(court: Court, rules: list[str], raw_dir: Path = RAW, pause: float = 4, opener: Any = None) -> dict[str, Any]:
    checkpoint = Checkpoint(raw_dir / court.key / "checkpoint.json")
    errors = 0
    for year in YEARS:
        page = 1
        while True:
            try:
                html = safe_get(search_url(court, year, page), court, rules, pause, opener).decode("cp1251", "replace")
            except DownloadError:
                errors += 1
                checkpoint.data["errors"] = errors
                checkpoint.save()
                if errors >= 3:
                    return {"status": "network_error_series", "court": court.key}
                break
            rows = parse_listing(html, court.base)
            if not rows:
                break
            for row in rows:
                if not row["case_url"]:
                    continue
                try:
                    card = safe_get(row["case_url"], court, rules, pause, opener).decode("cp1251", "replace")
                except DownloadError:
                    errors += 1
                    checkpoint.data["errors"] = errors
                    checkpoint.save()
                    if errors >= 3:
                        return {"status": "network_error_series", "court": court.key}
                    continue
                if sentence_candidate(card):
                    checkpoint.add(event_identity(court.key, row["case_url"], card))
            if total_found(html) is None or page * 25 >= total_found(html):
                break
            page += 1
    return {"status": "collected", "court": court.key, "tracked": len(checkpoint.data["candidates"])}


def empty_counts() -> dict[str, Any]:
    return {
        "records": 0, "candidates": 0, "appeal_link": 0, "first_found": 0,
        "both_links": 0, "both_texts": 0, "confirmed_pairs": 0,
        "outcomes": {"unchanged": 0, "changed": 0, "cancelled": 0, "unknown": 0},
        "remaining": None,
    }


def write_report(access: dict[str, dict[str, Any]], path: Path = REPORT, historical: dict[str, int] | None = None) -> None:
    lines = [
        "# Региональные пары по ст. 290 УК РФ",
        "",
        "Период: 2022–2025 по дате поступления апелляции. Публичны только агрегаты и статусы доступа; прямые ссылки на дела, номера, УИД, ФИО и хеши документов не публикуются.",
        "",
        "## Доступ и фактический охват",
        "",
        "| Суд | Доступ | Записи | Кандидаты | Ссылка на апелляцию | Найдена первая | Обе ссылки | Оба текста | Confirmed | Remaining |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for court in (KRK, VRN):
        state = access.get(court.key, {"status": "not_started"})
        counts = state.get("counts", empty_counts())
        remaining = counts["remaining"] if counts["remaining"] is not None else "не определён"
        lines.append(
            f"| {court.name} | {state['status']} | {counts['records']} | {counts['candidates']} | {counts['appeal_link']} | "
            f"{counts['first_found']} | {counts['both_links']} | {counts['both_texts']} | {counts['confirmed_pairs']} | {remaining} |"
        )
    lines += ["", "## Outcome", "", "| Суд | unchanged | changed | cancelled | unknown |", "|---|---:|---:|---:|---:|"]
    for court in (KRK, VRN):
        outcomes = access.get(court.key, {}).get("counts", empty_counts())["outcomes"]
        lines.append(f"| {court.name} | {outcomes['unchanged']} | {outcomes['changed']} | {outcomes['cancelled']} | {outcomes['unknown']} |")
    lines += ["", "## Охват по годам поступления", "", "| Суд | Год | Записи | Кандидаты | Confirmed | Remaining |", "|---|---:|---:|---:|---:|---:|"]
    for court in (KRK, VRN):
        for year in YEARS:
            counts = access.get(court.key, {}).get("years", {}).get(str(year), empty_counts())
            remaining = counts["remaining"] if counts["remaining"] is not None else "не определён"
            lines.append(f"| {court.name} | {year} | {counts['records']} | {counts['candidates']} | {counts['confirmed_pairs']} | {remaining} |")
    lines += [
        "",
        "## Ограничение robots.txt",
        "",
        "На дату проверки оба robots.txt запрещают `/*name=sud_delo`, включая необходимые картотеки, карточки и документы. Поэтому фактический сбор не запускался и объёмы не выдумываются. Нулевые значения означают отсутствие собранных записей, а не отсутствие дел.",
        "",
    ]
    if historical:
        lines += [
            "## Исторический малый пилот до этой проверки",
            "",
            "Ранее, в рамках отдельного ручного пилота, было подтверждено по 2 полные пары в каждом суде. Это не результат текущего автоматического сбора и не оценка годовой полноты.",
            "",
        ]
    atomic_write(path, "\n".join(lines).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Региональные официальные пары с проверкой robots")
    parser.add_argument("command", choices=("robots", "run", "report"))
    parser.add_argument("--pause", type=float, default=4)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    access = {}
    for court in COURTS.values():
        access_file = args.raw_dir / court.key / "access.json"
        if args.command == "report" and access_file.exists():
            access[court.key] = read_json(access_file, {})
            continue
        try:
            access[court.key] = check_robots(court, args.raw_dir, args.pause)
        except AccessBlocked:
            access[court.key] = {"status": "robots_blocked", "allowed": False}
        except (DownloadError, CaptchaError) as exc:
            access[court.key] = {"status": type(exc).__name__, "allowed": False}
        if args.command == "run" and access[court.key]["allowed"]:
            rules = read_json(access_file, {}).get("rules", [])
            access[court.key]["collection"] = collect_court(court, rules, args.raw_dir, args.pause)
    write_report(access, args.report, historical={"krk": 2, "vrn": 2})
    print(json.dumps({key: value["status"] for key, value in access.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
